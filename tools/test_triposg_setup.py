import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
import triposg_setup as setup


class Response(io.BytesIO):
    def __init__(self, data, interval):
        super().__init__(data)
        self.status = 206
        self.headers = {"Content-Range": interval}


class RuntimeSetupTests(unittest.TestCase):
    def test_resume_boundaries_and_final_digest(self):
        data = b"official-test-content"
        with tempfile.TemporaryDirectory() as d:
            dst = Path(d) / "model"
            dst.with_suffix(".partial").write_bytes(data[:8])

            def download(request, timeout):
                self.assertEqual(request.get_header("Range"), f"bytes=8-{len(data)-1}")
                return Response(data[8:], f"bytes 8-{len(data)-1}/{len(data)}")

            with patch.object(setup.urllib.request, "urlopen", side_effect=download):
                setup.acquire(dst, (len(data), hashlib.sha256(data).hexdigest()), "https://example.invalid/x")
            self.assertEqual(dst.read_bytes(), data)
            with patch.object(setup.urllib.request, "urlopen", side_effect=AssertionError):
                setup.acquire(dst, (len(data), hashlib.sha256(data).hexdigest()), "", offline=True)

    def test_unconfirmed_range_cannot_overwrite_partial(self):
        with tempfile.TemporaryDirectory() as d:
            part = Path(d) / "model.partial"
            part.write_bytes(b"abc")
            with patch.object(setup.urllib.request, "urlopen", return_value=Response(b"f", "bytes 5-5/6")):
                with self.assertRaisesRegex(ValueError, "byte range"):
                    setup.fetch_range("https://example.invalid/data", part, 6)
            self.assertEqual(part.read_bytes(), b"abc")

    def test_cache_corruption_never_becomes_an_installed_file(self):
        with tempfile.TemporaryDirectory() as d:
            cache, output = Path(d) / "cache", Path(d) / "installed"
            cache.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                setup.acquire(output, (5, hashlib.sha256(b"right").hexdigest()), "", cache, offline=True)
            self.assertFalse(output.exists())
            self.assertEqual(cache.read_bytes(), b"wrong")
            self.assertFalse(output.with_suffix(".partial").exists())

    def test_rerun_handles_generated_bytecode_and_rejects_source_drift(self):
        with tempfile.TemporaryDirectory() as d:
            archive, target = Path(d) / "upstream.zip", Path(d) / "source"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr(f"TripoSG-{setup.CODE}/triposg/sample.py", "value = 1\n")
            first = setup.source_files(archive, target)
            pyc = target / "triposg/__pycache__/sample.cpython-311.pyc"
            pyc.parent.mkdir()
            pyc.write_bytes(b"old bytecode")
            self.assertEqual(setup.source_files(archive, target), first)
            self.assertFalse(pyc.exists())
            (target / "triposg/sample.py").write_bytes(b"value = 2\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                setup.source_files(archive, target)


if __name__ == "__main__":
    unittest.main()
