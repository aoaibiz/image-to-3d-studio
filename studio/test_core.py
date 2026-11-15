import tempfile
import threading
import time
import unittest
from pathlib import Path
from studio.core import JobCore, JobError, PipelineConflict
from studio.configure import configure
from engine.pipeline import PIPELINE, SHAPE_PIPELINE


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

    def test_shape_selection_survives_restart_and_old_requests_remain_idempotent(self):
        old, _ = self.core.create(b"one", "old-key", PIPELINE)
        self.assertTrue(self.started.wait(2))
        self.release.set()
        self.core.close()
        configure(self.root / "jobs/studio", "triposg")
        self.core = JobCore(self.root, runner=lambda _: None)
        self.core._validate_image = lambda _: None
        with self.assertRaises(PipelineConflict):
            self.core.create(b"one", "stale", PIPELINE)
        with self.assertRaises(PipelineConflict):
            self.core.create(b"one", "old-key", SHAPE_PIPELINE)
        saved, created = self.core.create(b"one", "old-key")
        self.assertFalse(created)
        self.assertEqual(saved["id"], old["id"])
        self.assertEqual(saved["config"]["pipeline"], PIPELINE)
        shape, _ = self.core.create(b"one", "shape-key", SHAPE_PIPELINE)
        self.assertEqual(shape["config"]["resolution"], 256)
        self.assertEqual(shape["config"]["pipeline"], SHAPE_PIPELINE)
        self.core.close()
        configure(self.root / "jobs/studio", "triposr")
        self.core = JobCore(self.root, runner=lambda _: None)
        self.assertEqual(self.core.get(shape["id"])["config"]["pipeline"], SHAPE_PIPELINE)
        def check_command(job_id, argv, stage):
            self.assertEqual(argv[1:4], ["-u", "-m", "engine.local_shape_job"])
            self.assertEqual(argv[-4:], ["--resolution", "256", "--texture-size", "1024"])
            self.assertEqual(stage, "starting")
            raise JobError("test-stop")
        self.core._command = check_command
        with self.assertRaisesRegex(JobError, "test-stop"):
            self.core._run_pipeline(shape["id"])

    def test_real_child_progress_and_sanitized_shape_error(self):
        import sys
        job, _ = self.core.create(b"stages", "stages")
        self.assertTrue(self.started.wait(2))
        child = """import json
print("segmenting source")
print(json.dumps({"event":"progress","stage":"denoise","step":7,"total":50}))
print(json.dumps({"event":"progress","stage":"denoise","step":88,"total":50}))
print(json.dumps({"event":"resumed","stage":"surface"}))
print(json.dumps({"event":"studio_error","code":"shape_appearance","detail":"secret-detail"}))
raise SystemExit(1)
"""
        with self.assertRaisesRegex(JobError, "着色") as failure:
            self.core._command(job["id"], [sys.executable, "-c", child], "starting")
        self.assertNotIn("secret-detail", str(failure.exception))
        stages = [e["payload"]["stage"] for e in self.core.events(job["id"], 0)]
        self.assertIn("shape step 7/50", stages)
        self.assertIn("resuming shape", stages)
        self.assertNotIn("shape step 88/50", stages)

    def test_finished_shape_requires_hash_bound_history_and_master(self):
        import hashlib, json
        self.core.pipeline = SHAPE_PIPELINE
        job, _ = self.core.create(b"original", "model-preview", SHAPE_PIPELINE)
        self.assertTrue(self.started.wait(2))
        out = self.core.directory / job["id"] / "output"
        out.mkdir()
        sha = lambda data: hashlib.sha256(data).hexdigest()
        hashes = {}
        def add(name, content):
            (out / name).write_bytes(content)
            hashes[name] = sha(content)
        for name in ("model.glb", "editable.obj", "editable.gltf", "editable.ply", "texture.png",
                     "material.mtl", "material_0.png", "shape-master.glb", "input-processed.png"):
            add(name, name.encode())
        source = sha(b"original")
        generation = {"engine":{"input_sha256":hashes["input-processed.png"]},
                      "shape_result":{"sha256":hashes["shape-master.glb"]}, "foreground":{"source_sha256":source}}
        appearance = {"output_sha256":hashes["model.glb"], "shape_sha256":hashes["shape-master.glb"],
                      "foreground_sha256":hashes["input-processed.png"],"original_sha256":source}
        add("generation.json", json.dumps(generation).encode())
        add("appearance.json", json.dumps(appearance).encode())
        settings = {"pipeline":SHAPE_PIPELINE,"input_sha256":source,"glb_sha256":hashes["model.glb"],"artifact_sha256":hashes,
                    "marching_cubes_resolution":256,"texture_size":1024,"auto_finish_complete":True}
        self.assertTrue(self.core._artifacts_valid(out, settings))
        (out / "settings.json").write_text(json.dumps(settings))
        def verify(job_id, cmd, stage):
            digest = hashes["model.glb"]
            if stage == "inspection":
                return json.dumps({"sha256":digest, "format":{"status":"pass"}})
            if stage == "rendering":
                (out / "offline-preview.html").write_text("fixture")
                (out / "view.png").write_bytes(b"fixture")
                (out / "manifest.json").write_text(json.dumps({"sha256":digest, "files":["view.png"], "offlinePreviewVerified":True}))
                return ""
            if stage == "appearance":
                self.assertEqual(cmd[-3:], [str(out / "input-processed.png"), str(out), digest])
                return json.dumps({"sha256":digest, "status":"needs_review"})
            self.fail("Shape generation repeated.")
        self.core._command = verify
        self.core._run_pipeline(job["id"])
        inspect = json.loads((out / "inspection.json").read_text("utf8"))
        self.assertEqual(inspect["image_comparison"]["sha256"], hashes["model.glb"])
        self.assertEqual(inspect["source_registration"]["shape_sha256"], hashes["shape-master.glb"])
        for name in ("shape-master.glb", "appearance.json", "generation.json", "input-processed.png"):
            with self.subTest(name=name):
                self.assertFalse(self.core._artifacts_valid(out, dict(settings, artifact_sha256={k:v for k,v in hashes.items() if k != name})))
        appearance["output_sha256"] = "0" * 64
        add("appearance.json", json.dumps(appearance).encode())
        self.assertFalse(self.core._artifacts_valid(out, settings))
        with self.assertRaisesRegex(JobError, "バックアップ"):
            self.core._run_pipeline(job["id"])
        self.assertTrue((out / "model.glb").exists())

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
