"""Public-only Pixal routing, sealed artifact export and no-resubmit behavior; no network."""

import base64
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from engine import pixal_provider, pixal_shape_job as job
from engine.pipeline import PIXAL_PIPELINE
from quality.test_sf3d_provider import fixture_glb, png
from studio.configure import configure, read_pipeline
from studio.core import JobCore, PipelineConflict, _shape_notice
from studio.server import space_engine_info


class PixalRouteTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "photo.png"
        self.source.write_bytes(png())
        self.sha = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.out = self.root / "output"
        self.setup = self.root / "engine.json"

    def enable(self):
        configure(self.root, "pixal3d-space", allow_public_photo_to_pixal_space=True)

    def test_per_image_scope_required_and_stale_config_rejected(self):
        with self.assertRaises(ValueError):
            configure(self.root, "pixal3d-space")
        self.enable()
        self.assertTrue(space_engine_info(PIXAL_PIPELINE)["public_cc0_only"])
        core = JobCore(self.root, directory=self.root, runner=lambda _: None)
        self.addCleanup(core.close)
        with mock.patch.object(core, "_validate_image"):
            for pipeline, digest in [(None, self.sha), (PIXAL_PIPELINE, None), (PIXAL_PIPELINE, "f" * 64)]:
                with self.assertRaises(PipelineConflict):
                    core.create(png(), "one", pipeline, digest)
            entry, new = core.create(png(), "one", PIXAL_PIPELINE, self.sha)
            self.assertTrue(new)
            self.assertEqual(entry["config"]["public_cc0_sha256"], self.sha)
            self.assertFalse(core.create(png(), "one", PIXAL_PIPELINE, self.sha)[1])
            configure(self.root, "triposr")
            with self.assertRaises(PipelineConflict):
                core.create(png(), "two", PIXAL_PIPELINE, self.sha)
        self.enable()
        grant = json.loads(self.setup.read_bytes())
        grant["agreement"]["max_generation_requests"] = True
        self.setup.write_text(json.dumps(grant))
        with self.assertRaises(ValueError):
            read_pipeline(self.root)

    def test_job_invokes_exact_module_and_digest(self):
        import threading
        self.enable()
        core = JobCore(self.root, directory=self.root)
        self.addCleanup(core.close)
        reached = threading.Event()
        def stop(job_id, command, stage):
            self.assertIn("engine.pixal_shape_job", command)
            self.assertEqual(command[command.index("--public-cc0-sha256") + 1], self.sha)
            self.assertNotIn("--resolution", command)
            reached.set()
            raise RuntimeError("offline routing check")
        with mock.patch.object(core, "_validate_image"), mock.patch.object(core, "_command", side_effect=stop):
            core.create(png(), "run", PIXAL_PIPELINE, self.sha)
            self.assertTrue(reached.wait(3))

    def test_native_export_keeps_raw_geometry_and_pbr_texture_bindings(self):
        self.enable()
        work = self.out / "pixal-work" / "provider"
        work.mkdir(parents=True)
        native = fixture_glb()
        old_n = struct.unpack_from("<I", native, 12)[0]
        doc = json.loads(native[20:20 + old_n])
        doc["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": 0}
        chunk = json.dumps(doc).encode()
        chunk += b" " * (-len(chunk) % 4)
        binary = native[20 + old_n:]
        native = struct.pack("<4sIIII", b"glTF", 2, 20 + len(chunk) + len(binary), len(chunk), 0x4e4f534a) + chunk + binary
        (work / "model.glb").write_bytes(native)
        (work / "input.png").write_bytes(png())
        material = pixal_provider.inspect_pbr(work / "model.glb")
        receipt = {"state": "completed", "provider_space": pixal_provider.SPACE,
                   "revision": pixal_provider.REVISION, "quality": "needs_review", "source_sha256": self.sha,
                   "upload_sha256": self.sha, "sha256": material["sha256"], "material": material,
                   "artifacts": {"input.png": self.sha, "model.glb": material["sha256"]}}
        receipt.update(origin=pixal_provider.ORIGIN, resolution=1024, quality_setting="balanced",
                       asset_id="b" * 12, event_id="a" * 32, seed=pixal_provider.SEED, mock=False,
                       share_to_community=False, download_url=pixal_provider.ORIGIN + "/community-assets/" + "b"*12 + "/model.glb")
        record = work / "provider.json"
        record.write_text(json.dumps(receipt))
        result = job.finalize(work, self.out, self.sha, receipt)
        self.assertTrue(JobCore._artifacts_valid(self.out, result))
        for key, wrong in [("revision", "other"), ("asset_id", "c" * 12), ("seed", 8),
                           ("event_id", "bad"), ("share_to_community", True)]:
            changed = {**receipt, key: wrong}
            record.write_text(json.dumps(changed))
            self.assertFalse(JobCore._artifacts_valid(self.out, result))
            with self.assertRaises(ValueError):
                job.finalize(work, self.out, self.sha, changed)
        record.write_text(json.dumps(receipt))
        master = json.loads((self.out / "editable.gltf").read_bytes())
        self.assertEqual(master["materials"], doc["materials"])
        self.assertEqual(base64.b64decode(master["buffers"][0]["uri"].split(",")[1]), binary[8:])
        self.assertEqual((self.out / "model.glb").read_bytes(), native)
        with mock.patch.object(pixal_provider, "generate", side_effect=AssertionError("unexpected send")):
            with self.assertRaises(job.ConsentError):
                job.run(self.source, self.out, setup=self.setup, public_cc0_sha256="f" * 64)
            self.assertEqual(job.run(self.source, self.out, setup=self.setup, public_cc0_sha256=self.sha), result)
            (self.out / "model.glb").write_bytes(b"damaged")
            self.assertFalse(job.valid(self.out, result))
            with self.assertRaises((OSError, ValueError)):
                job.run(self.source, self.out, setup=self.setup, public_cc0_sha256=self.sha)

    def test_quota_failure_has_safe_numbers_only_and_zero_reset_is_unknown(self):
        for wait, seconds in [("0:00:00", None), ("2:01:04", 7264), ("unknown", None)]:
            message = f"You have exceeded your ZeroGPU quota (120s requested vs. 0s left). Try again in {wait}. PRIVATE"
            old = {"error": "Pixal3D event ended in error: " + json.dumps({"error": message})}
            failure = pixal_provider._saved_failure(old)
            self.assertIsInstance(failure, pixal_provider.QuotaExceeded)
            self.assertEqual(failure.details["retry_after_seconds"], seconds)
            state = {"failure_kind": "anonymous_quota", "quota": failure.details}
            self.assertEqual(pixal_provider._saved_failure(state).details, failure.details)
            text = _shape_notice(json.dumps({"event": "studio_error", "code": "pixal_quota", **failure.details}))[1]
            self.assertIn("残りは0秒", text)
            self.assertIn("必要な120秒", text)
            self.assertIn("回復時刻は未確認" if seconds is None else "7264秒後", text)
            self.assertNotIn("PRIVATE", text)
        self.assertNotIsInstance(pixal_provider._remote_failure("unrelated PRIVATE"), pixal_provider.QuotaExceeded)
        self.assertNotIn("PRIVATE", str(pixal_provider._remote_failure("unrelated PRIVATE")))
