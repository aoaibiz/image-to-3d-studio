"""Two loopback ports must not be able to recover the same job store."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from studio.core import JobCore
from studio.server import Server


ROOT = Path(__file__).resolve().parent.parent


class StoreLockTests(unittest.TestCase):
    def test_same_store_cannot_be_opened_by_another_server_port(self):
        with tempfile.TemporaryDirectory() as temporary:
            with Server(("127.0.0.1", 0), None) as first:
                first.core = JobCore(directory=Path(temporary))
                try:
                    with first.core.lock:
                        first.core.db.execute(
                            "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)",
                            (
                                "existing",
                                "locktest",
                                "hash",
                                "running",
                                "test",
                                None,
                                "now",
                                "now",
                                "{}",
                            ),
                        )
                        first.core.db.commit()
                    attempt = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "studio.server",
                            "--port",
                            "0",
                            "--jobs-dir",
                            temporary,
                        ],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    self.assertNotEqual(attempt.returncode, 0)
                    self.assertIn("Job store is already in use", attempt.stderr)
                    with first.core.lock:
                        row = first.core.db.execute(
                            "SELECT status, stage FROM jobs WHERE id='existing'"
                        ).fetchone()
                    self.assertEqual(tuple(row), ("running", "test"))
                finally:
                    first.core.close()


if __name__ == "__main__":
    unittest.main()
