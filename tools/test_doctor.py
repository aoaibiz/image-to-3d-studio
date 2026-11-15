import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import doctor


class ForegroundReadinessTests(unittest.TestCase):
    def test_cold_probe_verifies_session_without_running_rembg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for module in ("PIL", "xatlas", "trimesh", "skimage"):
                (root / f"{module}.py").touch()
            (root / "torch.py").write_text("__version__ = 'tested'")
            (root / "onnxruntime.py").write_text(
                "def get_available_providers(): return ['CPUExecutionProvider']"
            )
            sessions = root / "rembg/sessions"
            sessions.mkdir(parents=True)
            (sessions.parent / "__init__.py").write_text(
                "raise RuntimeError('cold probe must not compile pymatting')"
            )
            for installed in (False, True):
                if installed:
                    (sessions / "u2net.py").touch()
                probe = subprocess.run(
                    [sys.executable, "-E", "-S", "-c", doctor.RUNTIME_PROBE],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if installed:
                    self.assertEqual(probe.returncode, 0, probe.stderr)
                    self.assertEqual(probe.stdout.strip(), "tested")
                else:
                    self.assertIn("rembg U2Net session is missing", probe.stderr)

    def test_photo_weights_do_not_gate_transparent_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / ".runtime"
            python = "venv/Scripts/python.exe" if os.name == "nt" else "venv/bin/python"
            for name in [
                python,
                "model/model.ckpt",
                "model/config.yaml",
                "model/dino-config.json",
                "TripoSR/tsr/system.py",
            ]:
                path = runtime / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"verified")
            segmenter = runtime / "u2net/u2net.onnx"
            segmenter.parent.mkdir()
            expected = hashlib.sha256(b"verified").hexdigest()
            with (
                patch.object(doctor, "ROOT", root),
                patch.object(doctor, "WEIGHT_SHA256", expected),
                patch.object(doctor, "U2NET_SHA256", expected),
                patch.object(doctor, "gpu_inventory", return_value=None),
                patch.object(doctor.subprocess, "run") as probe,
            ):
                probe.return_value.returncode = 0
                probe.return_value.stdout = "2.2.2"
                for contents, status in [
                    (None, "missing"),
                    (b"broken", "mismatch"),
                    (b"verified", "verified"),
                ]:
                    with self.subTest(status=status):
                        if contents is not None:
                            segmenter.write_bytes(contents)
                        result = doctor.diagnose(check_hash=True)["local_cpu"]
                        self.assertEqual(
                            result["foreground_weight_verification"], status
                        )
                        self.assertEqual(
                            result["input_readiness"],
                            {"transparent": True, "opaque_photo": status == "verified"},
                        )
                        self.assertEqual(
                            result["preflight"] == "ready_to_try", status == "verified"
                        )
                        self.assertEqual(
                            str(segmenter.relative_to(root)) in result["files_missing"],
                            status == "missing",
                        )
                unverified = doctor.diagnose(check_hash=False)["local_cpu"]
                self.assertEqual(
                    unverified["foreground_weight_verification"], "unchecked"
                )
                self.assertFalse(unverified["input_readiness"]["opaque_photo"])


if __name__ == "__main__":
    unittest.main()
