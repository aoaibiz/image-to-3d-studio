import http.client
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import threading
import unittest
from pathlib import Path
from studio.core import JobCore
from studio.server import Server
from engine.pipeline import PIPELINE, SHAPE_PIPELINE


class HttpTests(unittest.TestCase):
    def test_host_origin_header_and_sse(self):
        with tempfile.TemporaryDirectory() as tmp:
            core = JobCore(Path(tmp), runner=lambda job: None)
            core._validate_image = lambda data: None
            server = Server(("127.0.0.1", 0), core)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            port = server.server_address[1]

            def request(method, path, headers=None, body=None):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request(method, path, body=body, headers=headers or {})
                response = conn.getresponse()
                data = response.read()
                conn.close()
                return response.status, data

            try:
                code, body = request("GET", "/api/engine")
                self.assertEqual((code, json.loads(body)), (200, {"pipeline": PIPELINE, "local": True}))
                code, _ = request("POST", "/api/jobs", {
                    "X-Studio-Client": "1", "X-Studio-Pipeline": SHAPE_PIPELINE, "Idempotency-Key": "stale",
                }, b"x")
                self.assertEqual(code, 409)
                self.assertEqual(core.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0)
                self.assertEqual(
                    request("GET", "/api/doctor", {"Host": "evil.example"})[0], 403
                )
                self.assertEqual(
                    request("GET", "/api/doctor", {"Origin": "https://evil.example"})[
                        0
                    ],
                    403,
                )
                states = ("setup_or_verification_required", "ready_to_try")
                responses = [
                    SimpleNamespace(stdout=json.dumps({"local_cpu": {"preflight": s}}))
                    for s in states
                ]
                with patch("studio.server.subprocess.run", side_effect=responses) as probe:
                    for expected in (*states, "ready_to_try"):
                        code, body = request("GET", "/api/doctor")
                        self.assertEqual(code, 200)
                        self.assertIn(expected, body.decode("utf8"))
                    self.assertEqual(probe.call_count, 2)
                self.assertEqual(request("POST", "/api/jobs", body=b"x")[0], 403)
                code, data = request(
                    "POST",
                    "/api/jobs",
                    {"X-Studio-Client": "1", "X-Studio-Pipeline": PIPELINE, "Idempotency-Key": "a"},
                    b"x",
                )
                self.assertEqual(code, 201)
                job = json.loads(data)["job"]["id"]
                code, events = request("GET", "/api/jobs/" + job + "/events")
                self.assertEqual(code, 200)
                self.assertIn(b"needs_review", events)
                self.assertIn(b"id:", events)
                self.assertEqual(
                    request(
                        "POST",
                        "/api/jobs/" + job + "/review",
                        {"X-Studio-Client": "1"},
                        b"not json",
                    )[0],
                    400,
                )
                self.assertEqual(
                    request(
                        "POST",
                        "/api/jobs/" + job + "/inspect",
                        {"X-Studio-Client": "1"},
                        b"{}",
                    )[0],
                    400,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
                core.close()

    def test_missing_optional_runtime_reports_selected_shape_environment(self):
        from studio.configure import configure
        from studio.server import shape_doctor
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configure(root / "jobs/studio", "triposg")
            core = JobCore(root, runner=lambda _: None)
            try:
                with patch("engine.local_shape_job.runtime_path", return_value=root / "missing"):
                    result = shape_doctor(core)
                self.assertEqual(result["pipeline"], SHAPE_PIPELINE)
                self.assertFalse(result["ok"])
                self.assertFalse(result["checks"]["shape_python"])
                self.assertFalse(result["checks"]["model"])
                self.assertEqual(result["diagnostics"]["local_cpu"]["engine"], "TripoSG")
            finally:
                core.close()


if __name__ == "__main__":
    unittest.main()
