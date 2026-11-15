"""Route a genuine diagnostic bundle from a synthetic textured GLB through the local service."""
import hashlib
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from engine import sf3d_observed_job as observed, sf3d_shape_job as native
from engine.sf3d_provider import native_glb
from quality.test_sf3d_provider import fixture_glb, png
from studio.configure import configure
from studio.core import JobCore, JobError
from studio.server import Server


class HoldSurfaceTest(unittest.TestCase):
    def test_job_event_evidence_download_and_refusal_to_redispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = root / "jobs"
            configure(store, "sf3d-observed", allow_official_space=True,
                      allow_send_to_openai=True, chatgpt_photo_analysis=True)
            photo = png()
            sha = hashlib.sha256(photo).hexdigest()

            def fixture(job_id):
                out = store / job_id / "output"
                base = out / "sf3d-observed-work/native"
                base.mkdir(parents=True)
                mesh = out / "test-fixture.glb"
                mesh.write_bytes(fixture_glb())
                receipt = {"source_sha256": sha, "foreground_sha256": sha,
                           "artifact": native_glb(mesh), "settings": {}}
                native.finalize(mesh, out.parent / "input.image", {"selection": {}, "source_sha256": sha}, {}, receipt, base)
                observed._save_hold(out, base, {"pipeline": observed.PIPELINE, "input_sha256": sha}, "registration")
                core._command(job_id, [sys.executable, "-c", 'print(\'{"event":"studio_error","code":"sf3d_observed_review"}\');raise SystemExit(1)'], "starting")

            core = JobCore(root, directory=store, runner=fixture)
            server = Server(("127.0.0.1", 0), core)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()

            def get(url):
                conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=15)
                try:
                    conn.request("GET", url)
                    response = conn.getresponse()
                    return response.status, response.read()
                finally:
                    conn.close()

            try:
                with patch.object(core, "_validate_image"):
                    job, _ = core.create(photo, "hold-case", observed.PIPELINE)
                jid = job["id"]
                api = "/api/jobs/" + jid
                status, events = get(api + "/events")
                self.assertEqual(status, 200)
                self.assertIn(b'"review_hold"', events)
                detail = json.loads(get(api)[1])["job"]
                self.assertEqual(detail["status"], "review_hold")
                self.assertEqual(set(detail["artifacts"]), {"review-native.glb", "review-hold.json"})
                bundle = json.loads(get(api + "/artifacts/review-hold.json")[1])
                self.assertEqual(bundle, detail["hold"])
                self.assertEqual(bundle["reason"], observed.HOLD_REASONS["source_fit"])
                preview = get(api + "/artifacts/review-native.glb")[1]
                self.assertEqual(hashlib.sha256(preview).hexdigest(), bundle["preview_sha256"])
                for name in ["model.glb", "input-processed.png", "input.image", "settings.json", "generation.json", "review-native.png"]:
                    self.assertEqual(get(api + "/artifacts/" + name)[0], 404)
                with self.assertRaises(JobError):
                    core.resume(jid)
                self.assertFalse(core.create(photo, "hold-case", observed.PIPELINE)[1])
                marker = store / jid / "output/review-hold.json"
                marker.write_text(json.dumps(dict(bundle, private_note="never relay")))
                self.assertEqual(get(api + "/artifacts/review-hold.json")[0], 404)
                self.assertNotIn(b"never relay", get(api)[1])
                marker.write_text(json.dumps(bundle))
                (marker.parent / "review-native.glb").write_bytes(b"")
                self.assertEqual(get(api + "/artifacts/review-native.glb")[0], 404)
                self.assertEqual(json.loads(get(api)[1])["job"]["artifacts"], [])
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
                core.close()
