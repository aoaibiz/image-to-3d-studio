"""Adapter and recovery contracts; no test double here provides generation-quality evidence."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from engine import local_shape_job as job


class LocalShapeJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "photo.png"
        im = Image.new("RGBA", (80, 80))
        ImageDraw.Draw(im).ellipse((10, 10, 70, 70), fill=(45, 120, 61, 255))
        im.save(self.source)

    def test_alpha_cache_is_bound_to_input_and_mask(self):
        work = self.base / "work"
        work.mkdir()
        key = job.sg.digest(self.source)
        path, record = job.foreground(self.source, work, key)
        self.assertEqual(record["method"], "source alpha")
        self.assertEqual(job.foreground(self.source, work, key)[1], record)
        with self.assertRaises(ValueError):
            job.foreground(self.source, work, "different")
        path.write_bytes(b"damaged")
        with self.assertRaises(ValueError):
            job.foreground(self.source, work, key)

    def test_source_border_contact_survives_padding(self):
        work = self.base / "clipped"
        work.mkdir()
        im = Image.new("RGBA", (80, 80))
        ImageDraw.Draw(im).rectangle((0, 20, 40, 70), fill=(55, 71, 90, 255))
        im.save(self.source)
        _, stamp = job.foreground(self.source, work, job.sg.digest(self.source))
        self.assertTrue(stamp["source_border_contact"])

    def test_faint_background_does_not_shrink_conditioning_object(self):
        work = self.base / "halo"
        work.mkdir()
        im = Image.new("RGBA", (700, 700), (110, 111, 112, 1))
        ImageDraw.Draw(im).rectangle((300, 280, 400, 420), fill=(55, 71, 90, 255))
        im.save(self.source)
        path, _ = job.foreground(self.source, work, job.sg.digest(self.source))
        with Image.open(path) as crop:
            self.assertLess(crop.height, 165)
            self.assertEqual(crop.getpixel((0, 0))[3], 0)
            self.assertIn(1, crop.getchannel("A").getdata())
        conditioned = job.sg.prepare_input(path)
        bounds = (
            conditioned.convert("L").point(lambda x: 255 if x < 130 else 0).getbbox()
        )
        self.assertGreater((bounds[3] - bounds[1]) / conditioned.height, 0.70)

    def test_setup_failure_before_segmentation(self):
        with (
            patch.object(job, "runtime_path", return_value=self.base / "missing"),
            self.assertRaises(job.StageError) as err,
        ):
            job.run(self.source, self.base / "out", 128, 256)
        self.assertEqual(err.exception.code, "shape_setup")
        self.assertFalse((self.base / "out" / "settings.json").exists())

    def test_projection_failure_preserves_raw_and_never_commits_output(self):
        def fixture(image, work, output, runtime, **kwargs):
            work.mkdir()
            output.write_bytes(b"fixture only")
            job.sg.atomic_json(
                work / "identity.json",
                {"input_sha256": job.sg.digest(image), "job_key": "fixture"},
            )
            field = work / "field-256.npy"
            field.write_bytes(b"fixture field")
            job.sg.seal(field, "fixture")
            return {"sha256": job.sg.digest(output), "cells": kwargs["cells"]}

        out = self.base / "out"
        with (
            patch.object(job.sg, "runtime_python"),
            patch.object(job.sg, "generate", side_effect=fixture),
            patch(
                "engine.shape_camera.register_camera",
                side_effect=ValueError("no alignment"),
            ),
            self.assertRaises(job.StageError) as err,
        ):
            job.run(self.source, out, 256, 512)
        self.assertEqual(err.exception.code, "shape_appearance")
        self.assertTrue((out / "local-shape-work/triposg/raw-shape.glb").exists())
        self.assertFalse((out / "settings.json").exists())


if __name__ == "__main__":
    unittest.main()
