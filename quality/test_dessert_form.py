import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from engine.dessert_form import add_dessert_glass
from engine.finish_export import export_finished


def glass_spec(colors, rim=0.24):
    return {
        "image_size": [240, 320],
        "rim": [0.5, 0.35, rim],
        "foot": [0.5, 0.91, 0.105],
        "glass_profile": [[0, 0.105], [0.5, 0.14], [1, rim]],
        "berries": [],
        "layers": [
            {"bottom": i / len(colors), "top": (i + 1) / len(colors), "rgb": rgb}
            for i, rgb in enumerate(colors)
        ],
    }


class DessertFormTests(unittest.TestCase):
    def test_measured_palette_and_shape_change(self):
        data = glass_spec([[92, 40, 32], [175, 197, 88], [245, 227, 181]])
        first = trimesh.Scene()
        add_dessert_glass(first, data)
        new_data = glass_spec([[89, 65, 97], [231, 159, 184]], rim=0.32)
        second = trimesh.Scene()
        add_dessert_glass(second, new_data)
        self.assertGreater(
            second.geometry["dessert_glass"].extents[0],
            first.geometry["dessert_glass"].extents[0] * 1.2,
        )
        atlas = np.asarray(
            first.geometry["spooned_filling"].visual.material.baseColorTexture
        )
        changed = np.asarray(
            second.geometry["spooned_filling"].visual.material.baseColorTexture
        )
        np.testing.assert_allclose(
            np.median(atlas[310:330], axis=(0, 1)), data["layers"][1]["rgb"], atol=4
        )
        self.assertGreater(np.abs(changed.astype(int) - atlas).mean(), 20)
        np.testing.assert_array_equal(atlas[:, 0], atlas[:, -1])
        # A spooned interface has finite pigment blending and changes height around the vessel.
        red = atlas[..., 0]
        transition = ((red > 191) & (red < 230)).sum(axis=0)
        self.assertGreater(np.percentile(transition, 50), 7)
        self.assertGreater(np.ptp(np.argmax((red > 191)[::-1], axis=0)), 13)

    def test_clear_foot_requires_supported_lower_ellipse(self):
        spec = glass_spec([[230, 203, 174], [148, 181, 104]])
        plain = Image.new("RGB", (240, 320), (87, 76, 61))
        ImageDraw.Draw(plain).line((34, 295, 225, 284), fill=(251, 248, 230), width=3)
        scene = trimesh.Scene()
        metadata = add_dessert_glass(scene, spec, plain)
        self.assertIsNone(metadata["foot_disc_half_width"])
        self.assertNotIn("clear_glass_foot", scene.geometry)
        ImageDraw.Draw(plain).ellipse(
            (72, 253, 168, 309), outline=(247, 241, 222), width=3
        )
        model = trimesh.Scene()
        detected = add_dessert_glass(model, spec, plain)
        self.assertAlmostEqual(detected["foot_disc_half_width"], 48 / 240, delta=0.025)
        self.assertGreater(
            model.geometry["clear_glass_foot"].extents[0],
            model.geometry["glass_base"].extents[0] * 1.5,
        )

    def test_observed_local_fruit_required_for_front_patch(self):
        spec = glass_spec([[222, 213, 183], [184, 205, 123], [246, 225, 175]])
        img = Image.new("RGB", (240, 320), (240, 218, 176))
        front = trimesh.Scene()
        plain = add_dessert_glass(front, spec, img)
        self.assertEqual([], plain["inclusions"])
        ImageDraw.Draw(img).ellipse((119, 158, 145, 180), fill=(157, 47, 59))
        with_fruit = trimesh.Scene()
        metadata = add_dessert_glass(with_fruit, spec, img)
        fruit = [p for p in metadata["inclusions"] if p["kind"] == "fruit"]
        self.assertEqual(1, len(fruit))
        self.assertTrue(0.3 < fruit[0]["height"] < 0.9)
        first = np.asarray(
            front.geometry["spooned_filling"].visual.material.baseColorTexture
        )
        second = np.asarray(
            with_fruit.geometry["spooned_filling"].visual.material.baseColorTexture
        )
        self.assertGreater(
            np.count_nonzero(np.abs(first.astype(int) - second).max(axis=2) > 30), 500
        )

    def test_export_preserves_independent_layer_finish_and_glass_transparency(self):
        scene = trimesh.Scene()
        spec = glass_spec([[149, 68, 55], [190, 204, 119], [250, 235, 195]])
        spec["collar"] = {"half_width": 0.33, "rgb": [209, 164, 72]}
        add_dessert_glass(scene, spec)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            export_finished(scene, out)
            model = trimesh.load(out / "model.glb", force="scene", process=False)
            self.assertIn("cereal_collar", model.geometry)
            self.assertEqual(
                model.geometry["dessert_glass"].visual.material.alphaMode, "BLEND"
            )
            self.assertTrue(
                np.isfinite(model.geometry["spooned_filling"].vertex_normals).all()
            )
            self.assertTrue((out / "editable.gltf").is_file())
