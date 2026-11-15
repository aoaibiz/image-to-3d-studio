"""Regressions for topping placement and the measured cream/fruit finish."""

import unittest

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from engine.dessert_toppings import add_dessert_toppings, _fruit_landmarks


class DessertToppingTests(unittest.TestCase):
    def fixture(self, piped):
        image = Image.new("RGB", (220, 260), (70, 43, 32))
        draw = ImageDraw.Draw(image)
        cream = (235, 209, 166) if piped else (235, 238, 225)
        draw.polygon(
            [
                (55, 172),
                (64, 131),
                (89, 76),
                (104, 65),
                (120, 81),
                (139, 130),
                (161, 171),
            ],
            fill=cream,
        )
        if piped:
            for y in range(78, 170, 8):
                half = min(46, 8 + (y - 76) * 0.47)
                draw.line(
                    (104 - half, y, 104 + half, y + 2), fill=(170, 133, 94), width=2
                )
            for x, y in [
                (94, 96),
                (101, 91),
                (119, 103),
                (106, 107),
                (96, 110),
                (111, 112),
            ]:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(114, 53, 48))
        draw.polygon(
            [(165, 74), (177, 80), (172, 91), (163, 104), (151, 86), (152, 78)],
            fill=(199, 30, 42),
        )
        draw.polygon(
            [(154, 74), (161, 76), (170, 70), (168, 78), (175, 75), (165, 81)],
            fill=(76, 110, 57),
        )
        m = {
            "cream_bbox": [54 / 220, 64 / 260, 162 / 220, 174 / 260],
            "rim": [0.50, 0.68, 0.23],
            "foot": [0.50, 0.96, 0.095],
            "cream_rgb": cream,
            "berries": [
                {
                    "kind": "red",
                    "center": [164 / 220, 87 / 260],
                    "radii": [14 / 220, 18 / 260],
                    "rgb": [199, 30, 42],
                }
            ],
        }
        return image, m

    def test_single_whipped_peak_and_berry_crown(self):
        im, m = self.fixture(False)
        top, leaf = _fruit_landmarks(np.asanyarray(im), m["berries"][0])
        self.assertTrue(top)
        self.assertIsNotNone(leaf)
        sc = trimesh.Scene()
        info = add_dessert_toppings(sc, m, im)
        self.assertEqual(info["piped_courses"], 0)
        self.assertIn("fluted_cream_1", sc.geometry)
        self.assertIn("berry_0_leaf_crown", sc.geometry)
        self.assertEqual(info["inferred_rear_berries"], 0)
        self.assertGreater(sc.geometry["berry_0"].bounds[0, 1], 1.3)

    def test_stripes_and_palette_are_read_from_input(self):
        image, m = self.fixture(True)
        sc = trimesh.Scene()
        info = add_dessert_toppings(sc, m, image)
        self.assertGreaterEqual(info["piped_courses"], 3)
        self.assertIn("piped_cream_1", sc.geometry)
        self.assertFalse(info["observed_decor"])
        pigment = np.asarray(
            sc.geometry["piped_cream_1"].visual.material.baseColorTexture
        )
        np.testing.assert_allclose(
            np.median(pigment.reshape(-1, 3), axis=0), m["cream_rgb"], atol=2
        )
        self.assertGreater(sc.geometry["piped_cream_1"].extents.min(), 0.5)


if __name__ == "__main__":
    unittest.main()
