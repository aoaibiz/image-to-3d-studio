"""Guard pose selection and the unclassified neural fallback."""

import tempfile
import unittest
from pathlib import Path

import trimesh
from PIL import Image, ImageDraw

from engine.auto_finish import finish_generated, prepare_source
from engine.pipeline import LEGACY_PIPELINE
from unittest.mock import patch


class SelectionTests(unittest.TestCase):
    def test_legacy_jobs_do_not_acquire_a_new_dessert_finish(self):
        image = Image.new("RGB", (200, 300), "white")
        with patch(
            "engine.dessert_measure.measure_dessert", side_effect=AssertionError
        ):
            _, selection = prepare_source(image, LEGACY_PIPELINE)
        self.assertNotIn("dessert", selection)

    def grid(self, colors):
        image = Image.new("RGB", (260, 200), "white")
        draw = ImageDraw.Draw(image)
        for i, color in enumerate(colors):
            x, y = 10 + (i % 3) * 86, 12 + (i // 3) * 95
            draw.ellipse((x, y, x + 56, y + 62), fill=color)
        return image

    def test_duplicate_pose_selection_is_relative(self):
        for size in ((260, 200), (520, 400)):
            with self.subTest(size=size):
                source = self.grid(["#9a5530"] * 6).resize(size)
                selected, selection = prepare_source(source)
                self.assertEqual(selection["detected_count"], 6)
                left, top, right, bottom = selection["selected_bbox"]
                self.assertLess(left, size[0] / 2)
                self.assertGreater(right, size[0] / 2)
                self.assertLess(bottom, size[1] / 2)
                self.assertEqual(selected.size, (right - left, bottom - top))
                self.assertEqual(selected.getextrema()[3], (0, 255))

    def test_unrelated_subjects_and_single_object_stay_whole(self):
        for colors in (["red", "green", "blue", "brown", "black", "gold"], ["black"]):
            source = self.grid(colors)
            result, selection = prepare_source(source)
            self.assertEqual(selection["method"], "single image")
            self.assertEqual(result.size, source.size)

    def test_unsupported_shape_preserves_exact_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            path = out / "model.glb"
            path.write_bytes(trimesh.creation.icosphere().export(file_type="glb"))
            original = path.read_bytes()
            foreground = out / "foreground.png"
            Image.new("RGBA", (90, 90), "orange").save(foreground)
            info = finish_generated(out, foreground, {})
            self.assertEqual(info["finish_profile"], "inferred-mesh")
            self.assertEqual(path.read_bytes(), original)
            self.assertTrue((out / "editable.gltf").is_file())

    def test_same_colored_different_outlines_are_not_poses(self):
        image = Image.new("RGB", (250, 220), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((12, 12, 95, 90), fill="black")
        draw.rectangle((145, 12, 222, 90), fill="black")
        draw.polygon([(15, 197), (55, 120), (95, 197)], fill="black")
        draw.polygon([(137, 155), (176, 115), (220, 155), (176, 198)], fill="black")
        _, selection = prepare_source(image)
        self.assertEqual(selection["method"], "single image")
