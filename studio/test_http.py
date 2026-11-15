import http.client
import tempfile
import threading
import unittest
from pathlib import Path
from studio.core import JobCore
from studio.server import Server


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
                self.assertEqual(
                    request("GET", "/api/doctor", {"Host": "evil.example"})[0], 403
                )
                self.assertEqual(
                    request("GET", "/api/doctor", {"Origin": "https://evil.example"})[
                        0
                    ],
                    403,
                )
                self.assertEqual(request("POST", "/api/jobs", body=b"x")[0], 403)
                code, data = request(
                    "POST",
                    "/api/jobs",
                    {"X-Studio-Client": "1", "Idempotency-Key": "a"},
                    b"x",
                )
                self.assertEqual(code, 201)
                import json

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


if __name__ == "__main__":
    unittest.main()
