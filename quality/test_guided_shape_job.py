"""Job durability/consent tests with synthetic files; no provider or inference calls."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from engine import guided_shape_job as job


class GuidedJobTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "source.png"
        self.source.write_bytes(b"photo")
        self.out = self.root / "out"
        self.setup = self.root / "engine.json"
        self.agreement = {"recipient": "OpenAI", "personal_plan": True}
        job.sg.atomic_json(
            self.setup, {"pipeline": job.GUIDED_PIPELINE, "agreement": self.agreement}
        )
        self.paint = types.ModuleType("engine.guided_provider")
        self.paint.AGREEMENT = self.agreement
        self.paint.generate_guided_paint = Mock(side_effect=self.service)
        self.texture = types.ModuleType("engine.guided_texture")
        self.texture.prepare_guides = Mock(side_effect=self.guides)
        self.texture.apply_guides = Mock(side_effect=self.project)
        modules = patch.dict(
            sys.modules,
            {
                "engine.guided_provider": self.paint,
                "engine.guided_texture": self.texture,
            },
        )
        modules.start()
        self.addCleanup(modules.stop)
        local = patch.object(job.local_shape_job, "run", side_effect=self.base)
        self.local = local.start()
        self.addCleanup(local.stop)
        self.submissions = 0

    def base(self, source, out, resolution, texture_size):
        out.mkdir(parents=True, exist_ok=True)
        for name in job.BASE_FILES:
            (out / name).write_bytes(("base " + name).encode())
        h = job.sg.digest
        job.sg.atomic_json(
            out / "appearance.json",
            {
                "original_sha256": h(source),
                "output_sha256": h(out / "model.glb"),
                "foreground_sha256": h(out / "input-processed.png"),
                "shape_sha256": h(out / "shape-master.glb"),
            },
        )
        job.sg.atomic_json(
            out / "generation.json",
            {
                "camera": {"camera_frame": {"image_up": [0, 1, 0]}},
                "engine": {"input_sha256": h(out / "input-processed.png")},
                "shape_result": {"sha256": h(out / "shape-master.glb")},
                "foreground": {"source_sha256": h(source)},
                "external_generation_calls": 0,
            },
        )
        job.sg.atomic_json(
            out / "settings.json",
            {
                "pipeline": job.SHAPE_PIPELINE,
                "auto_finish_complete": True,
                "input_sha256": h(source),
                "glb_sha256": h(out / "model.glb"),
                "marching_cubes_resolution": resolution,
                "texture_size": texture_size,
                "artifact_sha256": {f: h(out / f) for f in job.BASE_FILES},
            },
        )

    def guides(self, model, out):
        out.mkdir()
        (out / "guides.png").write_bytes(b"real guides would go here")
        job.sg.atomic_json(
            out / "guide.json",
            {
                "model_sha256": job.sg.digest(model),
                "guide_sha256": job.sg.digest(out / "guides.png"),
                "views": {"rear": [130, 12], "opposite": [-148, 10]},
                "size": 768,
            },
        )

    def service(self, photo, guide, manifest, work, *, setup):
        d = work / "guided-paint"
        if not d.exists():
            d.mkdir()
            (d / "sheet.png").write_bytes(b"fake sheet")
            self.submissions += 1
        layout = job.sg.read_json(manifest)
        return {
            "sheet": d / "sheet.png",
            "provenance": {
                "state": "completed",
                "photo_sha256": job.sg.digest(photo),
                "geometry_sha256": layout["model_sha"],
                "guide_sha256": job.sg.digest(guide),
                "sheet_sha256": job.sg.digest(d / "sheet.png"),
                "thread_id": "verified-in-fixture",
            },
        }

    def project(self, model, foreground, camera, guide, sheet, out):
        out.mkdir()
        for name in job.EDITABLE:
            (out / name).write_bytes(("finished " + name).encode())
        job.sg.atomic_json(
            out / "guided-paint.json",
            {
                "input_sha256": job.sg.digest(model),
                "output_sha256": job.sg.digest(out / "model.glb"),
                "foreground_sha256": job.sg.digest(foreground),
                "inferred_sha256": job.sg.digest(sheet),
                "guide_sha256": job.sg.digest(guide / "guides.png"),
                "locked_bytes_changed": 0,
                "modified_surface_texels": 4,
                "appearance": "needs_review",
            },
        )

    def invoke(self):
        return job.run(self.source, self.out, 128, 512, setup=self.setup)

    def test_atomic_completion_preserves_all_final_masters_and_valid_resume(self):
        result = self.invoke()
        self.assertEqual(result["visual_quality"], "needs_review")
        self.assertEqual(set(result["artifact_sha256"]), job.GUIDED_FILES)
        for n in job.EDITABLE:
            self.assertEqual((self.out / n).read_bytes(), ("finished " + n).encode())
        self.assertEqual(
            job.sg.digest(self.out / "shape-master.glb"),
            result["artifact_sha256"]["shape-master.glb"],
        )
        self.assertEqual(self.invoke(), result)
        self.assertEqual(self.local.call_count, 1)
        self.assertEqual(self.paint.generate_guided_paint.call_count, 1)
        (self.out / "inferred-paint.png").write_bytes(b"unexpected change")
        with self.assertRaises(job.StageError) as err:
            self.invoke()
        self.assertEqual(err.exception.code, "guided_checkpoint")
        self.assertEqual(self.submissions, 1)

    def test_projection_failure_preserves_base_and_provider_receipt(self):
        self.texture.apply_guides.side_effect = ValueError(
            "panel lacks reliable support"
        )
        with self.assertRaises(job.StageError) as err:
            self.invoke()
        self.assertEqual(err.exception.code, "guided_projection")
        self.assertFalse((self.out / "settings.json").exists())
        self.assertTrue((self.out / "local-guided-work/base/model.glb").exists())
        self.assertEqual(
            job.sg.read_json(self.out / "local-guided-work/state.json")["status"],
            "failed",
        )
        self.texture.apply_guides.side_effect = self.project
        self.invoke()
        self.assertEqual(self.submissions, 1)
        self.assertEqual(self.local.call_count, 1)

    def test_revocation_and_corrupt_cached_base_prevent_resubmission(self):
        self.paint.generate_guided_paint.side_effect = RuntimeError(
            "accepted result unknown"
        )
        with self.assertRaises(job.StageError) as err:
            self.invoke()
        self.assertEqual(err.exception.code, "guided_provider")
        self.setup.unlink()
        with self.assertRaises(job.StageError) as err:
            self.invoke()
        self.assertEqual(err.exception.code, "guided_consent")
        job.sg.atomic_json(
            self.setup, {"pipeline": job.GUIDED_PIPELINE, "agreement": self.agreement}
        )
        (self.out / "local-guided-work/base/editable.gltf").write_bytes(b"tamper")
        with self.assertRaises(job.StageError) as err:
            self.invoke()
        self.assertEqual(err.exception.code, "guided_checkpoint")
        self.assertEqual(self.paint.generate_guided_paint.call_count, 1)

    def test_flat_agreement_and_wrong_pipeline_are_not_consent(self):
        for config in (
            self.agreement,
            {"agreement": self.agreement, "pipeline": job.SHAPE_PIPELINE},
        ):
            job.sg.atomic_json(self.setup, config)
            with self.assertRaises(job.StageError) as err:
                self.invoke()
            self.assertEqual(err.exception.code, "guided_consent")
        self.local.assert_not_called()
        self.paint.generate_guided_paint.assert_not_called()


if __name__ == "__main__":
    unittest.main()
