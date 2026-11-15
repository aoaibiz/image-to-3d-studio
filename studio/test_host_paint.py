"""Native handoff, privacy and restart tests. Synthetic transport fixtures, no external call."""

import hashlib
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
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
        self.assertEqual(submit(headers, self.png), 200)
        self.assertEqual(submit({**headers,"X-Studio-Tool-Call":"different"}, self.png), 400)
        self.assertNotIn(self.token.encode(), self.send("GET", path)[1])
        self.assertEqual(self.send("GET", path + "/artifacts/host-claim.json")[0], 404)

    def test_cancel_and_revocation(self):
        path = self.new_job("cancel")
        self.send("POST", path + "/cancel")
        self.assertEqual(self.send("POST", path + "/host/claim", {"X-Studio-Host-Token":self.token})[0], 409)
        configure(self.store, "triposg")
        self.assertEqual(self.send("POST", path + "/host/claim", {"X-Studio-Host-Token":self.token})[0], 403)
