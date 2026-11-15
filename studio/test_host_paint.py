"""Native handoff, privacy and restart tests. Synthetic transport fixtures, no external call."""

import hashlib
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from PIL import Image, PngImagePlugin

from engine import host_paint as paint
from engine.pipeline import HOST_PIPELINE
from engine.triposg_cpu import digest
from studio.configure import configure
from studio.core import AwaitingHostImage, JobCore
from studio.server import Server


class NativeHostTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = self.root / "store"
        configure(self.store, "triposg-guided-host", allow_send_to_openai=True)
        self.token = (self.store / paint.TOKEN_FILE).read_text()
        self.png = io.BytesIO()
        info = PngImagePlugin.PngInfo()
        info.add_text("comment", "private metadata")
        Image.new("RGB", (160, 80), "green").save(self.png, "PNG", pnginfo=info)
        self.png = self.png.getvalue()
        self.start()
        self.addCleanup(self.stop)

    def run_fixture(self, jid):
        folder = self.store / jid
        work = folder / "output/local-guided-work"
        guide = work / "guides/guides.png"
        guide.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (160, 80), "gray").save(guide)
        layout = {"photo_sha": digest(folder / "input.image"), "model_sha": "1"*64, "guide_sha": digest(guide)}
        req = paint.prepare(work, folder / "input.image", guide, layout)
        if not paint.result(work, req):
            raise AwaitingHostImage()

    def start(self):
        self.core = JobCore(self.root, directory=self.store, runner=self.run_fixture)
        self.core._validate_image = lambda data: None
        self.server = Server(("127.0.0.1", 0), self.core)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.core.close()

    def send(self, method, path, headers=None, body=b""):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=6)
        conn.request(method, path, body, {"X-Studio-Client":"1", **(headers or {})})
        response = conn.getresponse()
        status, data = response.status, response.read()
        conn.close()
        return status, data

    def new_job(self, key):
        code, body = self.send("POST", "/api/jobs", {"Idempotency-Key": key, "X-Studio-Pipeline": HOST_PIPELINE}, self.png)
        self.assertEqual(code, 201)
        jid = json.loads(body)["job"]["id"]
        path = "/api/jobs/" + jid
        code, events = self.send("GET", path + "/events", {"X-Studio-Yield-Host": "1"})
        self.assertEqual(code, 200)
        self.assertIn(b"awaiting_host_image", events)
        return path

    def test_authenticated_once_across_restart_and_worker_release(self):
        path = self.new_job("first")
        self.new_job("next")  # queue can advance while the first job has no returned image
        auth = {"X-Studio-Host-Token": self.token}
        for bad in ({}, {"X-Studio-Host-Token": "wrong"}, {**auth, "Origin": "https://foreign.example"}):
            self.assertEqual(self.send("POST", path + "/host/claim", bad)[0], 403)
        code, body = self.send("POST", path + "/host/claim", auth)
        self.assertEqual(code, 200)
        claim = json.loads(body)
        self.assertTrue(claim["generate_once"])
        self.assertEqual(claim["prompt_sha256"], paint.PROMPT_SHA)
        self.assertEqual([digest(Path(p)) for p in claim["referenced_image_paths"]], claim["uploaded_png_sha256"])
        with Image.open(claim["referenced_image_paths"][1]) as image:
            self.assertFalse(image.info)
        self.stop()
        self.start()
        repeat = json.loads(self.send("POST", path + "/host/claim", auth)[1])
        self.assertFalse(repeat["generate_once"])
        self.assertNotIn("referenced_image_paths", repeat)
        self.assertEqual(repeat["claim_id"], claim["claim_id"])
        headers = {**auth, "Content-Type":"image/png", "X-Studio-Claim":claim["claim_id"],
                   "X-Studio-Tool-Call": "fixture_call_1", "X-Studio-Request-Sha256": claim["request_sha256"],
                   "X-Studio-Paint-Sha256": hashlib.sha256(self.png).hexdigest()}
        submit = lambda h, b: self.send("POST", path + "/host/submit", h, b)[0]
        for field, wrong in (("X-Studio-Claim", "a"*43), ("X-Studio-Request-Sha256", "0"*64),
                             ("X-Studio-Paint-Sha256", "0"*64), ("X-Studio-Tool-Call", "../bad")):
            self.assertEqual(submit({**headers, field:wrong}, self.png), 400)
        self.assertEqual(submit({**headers,"X-Studio-Paint-Sha256": hashlib.sha256(b"bad").hexdigest()}, b"bad"), 400)
        self.assertEqual(submit(headers, self.png), 200)
        self.assertIn(b"needs_review", self.send("GET", path + "/events")[1])
        # Restart may mark a committed result's queued worker as interrupted.
        with self.core.lock:
            self.core._update(path.rsplit("/", 1)[1], "interrupted", "interrupted")
        self.assertEqual(submit(headers, self.png), 200)
        self.assertIn(b"needs_review", self.send("GET", path + "/events")[1])
        self.assertEqual(json.loads(self.send("GET", path)[1])["job"]["status"], "needs_review")
        self.assertEqual(submit({**headers,"X-Studio-Tool-Call":"different"}, self.png), 400)
        self.assertNotIn(self.token.encode(), self.send("GET", path)[1])
        self.assertEqual(self.send("GET", path + "/artifacts/host-claim.json")[0], 404)

    def test_cancel_and_revocation(self):
        path = self.new_job("cancel")
        self.send("POST", path + "/cancel")
        self.assertEqual(self.send("POST", path + "/host/claim", {"X-Studio-Host-Token":self.token})[0], 409)
        configure(self.store, "triposg")
        self.assertEqual(self.send("POST", path + "/host/claim", {"X-Studio-Host-Token":self.token})[0], 403)

    def test_a_brief_is_bound_before_claim_and_cannot_replace_a_claimed_request(self):
        work = self.root / "brief-job"
        guide = work / "guides/guides.png"
        guide.parent.mkdir(parents=True)
        guide.write_bytes(self.png)
        source = self.root / "source.jpg"
        with Image.open(io.BytesIO(self.png)) as image:
            image.convert("RGB").save(source, format="JPEG")
        layout = {"photo_sha":digest(source), "model_sha":"3"*64, "guide_sha":digest(guide), "visual_brief":{"fixture":1}}
        actual = paint.PROMPT + "\nVerified appearance constraints."
        with patch("engine.guided_provider.appearance_prompt", create=True, return_value=(actual, "b"*64)) as build:
            request = paint.prepare(work, source, guide, layout)
            build.assert_called_once_with(paint.PROMPT, layout, layout["photo_sha"])
        claim = paint.claim(work, "fixture")
        self.assertEqual(claim["prompt"], actual)
        self.assertEqual(claim["visual_brief_sha256"], "b"*64)
        self.assertEqual(request["prompt_sha256"], hashlib.sha256(actual.encode()).hexdigest())
        with patch("engine.guided_provider.appearance_prompt", create=True, side_effect=AssertionError("must use sealed prompt")):
            self.assertEqual(paint.prepare(work, source, guide, layout), request)
            with self.assertRaises(ValueError):
                paint.prepare(work, source, guide, {**layout, "visual_brief": {"fixture": 2}})
            with self.assertRaises(ValueError):
                paint.prepare(work, source, guide, {**layout, "camera": {"rear": [0, 0]}})
        self.assertFalse(paint.claim(work, "fixture")["generate_once"])
        paint.submit(work, claim["claim_id"], request["input_sha256"],
                     hashlib.sha256(self.png).hexdigest(), "fixture", self.png)
        self.assertEqual(paint.result(work, request)["provenance"]["prompt_sha256"], request["prompt_sha256"])
        (work / "host-input/prompt.txt").write_text("changed")
        with self.assertRaises(ValueError):
            paint.claim(work, "fixture")
        with self.assertRaises(ValueError):
            paint.submit(work, claim["claim_id"], request["input_sha256"],
                         hashlib.sha256(self.png).hexdigest(), "fixture", self.png)

    def test_old_brief_claim_survives_new_template_and_never_redispatches(self):
        from engine.guided_provider import legacy_appearance_prompt
        from engine.triposg_cpu import atomic_json

        for extended in (False, True):
            with self.subTest(extended=extended):
                files = self.root / str(extended)
                work = files / "output/local-guided-work"
                folder, guides = work / "host-input", work / "guides"
                folder.mkdir(parents=True)
                guides.mkdir()
                model = work / "base/model.glb"
                model.parent.mkdir()
                model.write_bytes(b"fixture native mesh")
                source, guide = files / "input.image", guides / "guides.png"
                for path in [source, guide, folder / "photo.png"]:
                    path.write_bytes(self.png)
                sha = digest(source)
                brief = dict(method="photo-appearance-qwen3-vl-2b-v1", source_sha256=sha,
                             visible_palette=["blue","white"], optical_finish_candidate="glossy",
                             layout_candidate="focal_and_repeated", unseen_surface_observed=False,
                             observed_face_cues={"focal":{"family":"abstract","location":"center"},
                                                 "repeated":{"family":"botanical","location":"upper_border"}})
                if extended:
                    brief.update(visible_structures=[], visible_surround={"color":"ivory","pigment_layout":"uniform"})
                layout = {"photo_sha":sha, "model_sha":digest(model), "guide_sha":digest(guide), "visual_brief":brief}
                atomic_json(guides / "request-guide.json", layout)
                old, brief_sha = legacy_appearance_prompt(paint.LEGACY_PROMPT, layout, sha)
                # Frozen output SHA of the builder at 41bc035, before structural sentences.
                old_sha = "dec0ed788d704f2a921844a2ceee4ca3749121dafdc0b1eebd328a07d135bde2"
                self.assertEqual(hashlib.sha256(old.encode()).hexdigest(), old_sha)
                record = dict(photo_sha256=sha, geometry_sha256=layout["model_sha"], guide_sha256=layout["guide_sha"],
                              prompt_sha256=old_sha, visual_brief_sha256=brief_sha, uploaded_png_sha256=[sha, sha])
                record["input_sha256"] = hashlib.sha256(json.dumps(
                    [sha, sha, layout["model_sha"], sha, old_sha, brief_sha]).encode()).hexdigest()
                atomic_json(folder / "request.json", record)
                (folder / "prompt.txt").write_bytes(old.encode())
                atomic_json(work / "host-claim.json", {"claim_id":"x"*43,"request_sha256":record["input_sha256"]})
                with patch("engine.guided_provider.appearance_prompt", side_effect=AssertionError("must not call current recipe")):
                    self.assertEqual(paint.prepare(work, source, guide, layout),record)
                    self.assertFalse(paint.claim(work,"fixture")["generate_once"])
                    paint.submit(work, "x"*43, record["input_sha256"], sha, "historical_fixture", self.png)
                    self.assertEqual(paint.result(work,record)["provenance"]["sheet_sha256"],sha)
                atomic_json(guides / "request-guide.json", {**layout,"visual_brief":{**brief,"visible_palette":["blue"]}})
                with self.assertRaises((ValueError, RuntimeError)):
                    paint.claim(work, "fixture")
                atomic_json(guides / "request-guide.json", layout)
                model.write_bytes(b"different model")
                with self.assertRaises(ValueError):
                    paint.submit(work, "x"*43, record["input_sha256"], sha, "historical_fixture", self.png)
