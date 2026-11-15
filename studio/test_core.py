import tempfile
import threading
import time
import unittest
from pathlib import Path
from studio.core import JobCore, JobError
from engine.pipeline import PIPELINE


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.started = threading.Event()
        self.release = threading.Event()

        def run(job):
            self.started.set()
            self.release.wait(3)

        self.core = JobCore(self.root, runner=run)
        self.core._validate_image = lambda data: None

    def tearDown(self):
        self.release.set()
        self.core.close()
        self.tmp.cleanup()

    def test_idempotency_cancel_resume_and_single_worker(self):
        job, created = self.core.create(b"image", "same")
        self.assertTrue(created)
        self.assertTrue(self.started.wait(2))
        again, created = self.core.create(b"image", "same")
        self.assertFalse(created)
        self.assertEqual(job["id"], again["id"])
        with self.assertRaises(JobError):
            self.core.create(b"different", "same")
        other, _ = self.core.create(b"image2", "other")
        self.assertEqual(self.core.get(other["id"])["status"], "queued")
        self.assertEqual(self.core.cancel(job["id"])["status"], "cancelled")
        with self.assertRaises(JobError):
            self.core.resume(job["id"])
        self.release.set()
        for _ in range(100):
            if self.core.active != job["id"]:
                break
            time.sleep(0.01)
        self.assertEqual(self.core.resume(job["id"])["status"], "queued")

    def test_restart_interrupts_queued(self):
        first, _ = self.core.create(b"one", "one")
        self.assertTrue(self.started.wait(2))
        second, _ = self.core.create(b"two", "two")
        self.release.set()
        self.core.close()
        self.core = JobCore(self.root, runner=lambda job: None)
        self.core._validate_image = lambda data: None
        self.assertEqual(self.core.get(second["id"])["status"], "interrupted")
        self.assertTrue(self.core.events(second["id"], 0))

    def test_failure(self):
        self.core.runner = lambda job: (_ for _ in ()).throw(ValueError("failed"))
        job, _ = self.core.create(b"one", "one")
        for _ in range(100):
            if self.core.get(job["id"])["status"] == "failed":
                break
            time.sleep(0.01)
        self.assertEqual(self.core.get(job["id"])["status"], "failed")
        self.assertIn("再開", self.core.get(job["id"])["error"])

    def test_legacy_job_does_not_switch_generation_pipeline(self):
        import json

        job, _ = self.core.create(b"old", "old")
        self.assertTrue(self.started.wait(2))
        with self.core.lock:
            self.core.db.execute(
                "UPDATE jobs SET config=? WHERE id=?",
                (json.dumps({"resolution": 96, "texture_size": 1024}), job["id"]),
            )
            self.core.db.commit()
        self.core._command = lambda *args: self.fail("legacy job started new inference")
        with self.assertRaisesRegex(JobError, "旧方式"):
            self.core._run_pipeline(job["id"])

    def test_retry_reuses_hash_matched_inference_and_hides_partial_output(self):
        import hashlib, json

        job, _ = self.core.create(b"one", "one")
        self.assertTrue(self.started.wait(2))
        folder = self.core.directory / job["id"] / "output"
        folder.mkdir()
        glb = b"fake-glb"
        digest = hashlib.sha256(glb).hexdigest()
        (folder / "model.glb").write_bytes(glb)
        artifact_hashes = {"model.glb": digest}
        for name in (
            "editable.gltf",
            "editable.obj",
            "editable.ply",
            "texture.png",
            "material_12.png",
        ):
            (folder / name).write_bytes(b"artifact")
            artifact_hashes[name] = hashlib.sha256(b"artifact").hexdigest()
        (folder / "settings.json").write_text(
            json.dumps(
                {
                    "input_sha256": job["input_sha256"],
                    "glb_sha256": digest,
                    "marching_cubes_resolution": 96,
                    "texture_size": 1024,
                    "auto_finish_complete": True,
                    "pipeline": PIPELINE,
                    "artifact_sha256": artifact_hashes,
                }
            )
        )
        self.assertEqual(self.core.get(job["id"])["artifacts"], [])
        stages = []

        def command(job_id, command, stage):
            stages.append(stage)
            if stage == "inspection":
                return json.dumps({"sha256": digest, "format": {"status": "pass"}})
            if stage == "rendering":
                (folder / "offline-preview.html").write_text("preview")
                (folder / "view.png").write_bytes(b"view")
                (folder / "manifest.json").write_text(
                    json.dumps(
                        {
                            "sha256": digest,
                            "offlinePreviewVerified": True,
                            "files": ["view.png"],
                        }
                    )
                )
                return ""
            self.fail("inference was repeated")

        self.core._command = command
        self.core._run_pipeline(job["id"])
        self.assertEqual(stages, ["inspection", "rendering"])
        (folder / "editable.gltf").write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "no finish checkpoint"):
            self.core._run_pipeline(job["id"])
        checkpoint = folder / "raw-export" / "finish-checkpoint.json"
        checkpoint.parent.mkdir()
        checkpoint.write_text("saved checkpoint")
        settings = {
            "input_sha256": job["input_sha256"],
            "glb_sha256": digest,
            "marching_cubes_resolution": 96,
            "texture_size": 1024,
            "auto_finish_complete": True,
            "pipeline": PIPELINE,
            "artifact_sha256": artifact_hashes,
        }

        def repair(job_id, command_args, stage):
            if stage == "starting":
                stages.append(stage)
                self.assertTrue(checkpoint.is_file())
                (folder / "model.glb").write_bytes(glb)
                for name in (
                    "editable.gltf",
                    "editable.obj",
                    "editable.ply",
                    "texture.png",
                    "material_12.png",
                ):
                    (folder / name).write_bytes(b"artifact")
                (folder / "settings.json").write_text(json.dumps(settings))
                return ""
            return command(job_id, command_args, stage)

        self.core._command = repair
        self.core._run_pipeline(job["id"])
        self.assertEqual(
            stages, ["inspection", "rendering", "starting", "inspection", "rendering"]
        )
        self.assertEqual((folder / "editable.gltf").read_bytes(), b"artifact")
        with self.core.lock:
            self.core._update(job["id"], "needs_review", "needs_review")
        self.assertIn("material_12.png", self.core.get(job["id"])["artifacts"])

    def test_review_and_semantic_inspection(self):
        import hashlib, json
        from unittest.mock import patch

        job, _ = self.core.create(b"one", "one")
        self.assertTrue(self.started.wait(2))
        self.release.set()
        for _ in range(100):
            if self.core.active is None:
                break
            time.sleep(0.01)
        out = self.core.directory / job["id"] / "output"
        out.mkdir()
        glb = b"glb"
        digest = hashlib.sha256(glb).hexdigest()
        (out / "model.glb").write_bytes(glb)
        (out / "settings.json").write_text(
            json.dumps({"input_sha256": job["input_sha256"], "glb_sha256": digest})
        )
        inspection = {
            "sha256": digest,
            "format": {"status": "pass"},
            "geometry": {"status": "review"},
        }
        (out / "inspection.json").write_text(json.dumps(inspection))
        files = [f"{digest}_front_{mode}.png" for mode in ("material", "clay")]
        for name in files:
            (out / name).write_bytes(b"image")
        (out / "manifest.json").write_text(
            json.dumps({"sha256": digest, "files": files})
        )
        review = {
            "sha256": digest,
            "decision": "pass",
            "reference_fidelity": "good",
            "major_part_volume": "good",
            "attachment_material": "good",
            "inspected_views": ["material", "clay"],
        }
        with self.assertRaises(JobError):
            self.core.review(job["id"], dict(review, sha256="0" * 64))
        with self.assertRaises(JobError):
            self.core.review(job["id"], dict(review, attachment_material=""))
        inspection["format"]["status"] = "unknown"
        (out / "inspection.json").write_text(json.dumps(inspection))
        with self.assertRaises(JobError):
            self.core.review(job["id"], review)
        inspection["format"]["status"] = "pass"
        inspection["geometry"]["status"] = "fail"
        (out / "inspection.json").write_text(json.dumps(inspection))
        with self.assertRaises(JobError):
            self.core.review(job["id"], review)
        inspection["geometry"]["status"] = "review"
        (out / "inspection.json").write_text(json.dumps(inspection))
        self.assertEqual(
            self.core.review(job["id"], review)["review"]["decision"], "pass"
        )
        region = {
            "bounds": [[0, 1], [0, 1], [0, 1]],
            "levels": [0.2, 0.5, 0.8],
            "min_depth_ratio": 0.1,
        }
        profile = {
            "subject": "cat",
            "up_axis": 1,
            "depth_axis": 2,
            "regions": {"head": region, "body": region},
        }
        with self.assertRaises(JobError):
            self.core.inspect(job["id"], {"profile": dict(profile, subject="car")})
        with patch(
            "quality.inspection.inspect",
            return_value={"sha256": digest, "geometry": {"status": "fail"}},
        ):
            result = self.core.inspect(job["id"], {"profile": profile})
        self.assertIsNone(result["review"])
        self.assertIn("semantic-inspection.json", result["artifacts"])
        with self.assertRaises(JobError):
            self.core.review(job["id"], review)
        self.assertEqual(
            self.core.review(job["id"], dict(review, decision="fail"))["review"][
                "decision"
            ],
            "fail",
        )
        with self.core.lock:
            self.core._update(job["id"], "failed", "failed")
        with self.assertRaises(JobError):
            self.core.review(job["id"], review)


if __name__ == "__main__":
    unittest.main()
