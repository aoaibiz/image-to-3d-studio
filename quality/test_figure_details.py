import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from figure_details import detected_lines, add_detected_details


class DetailsTest(unittest.TestCase):
    def setUp(self):
        self.mapping = {
            "landmarks_px": {"head_center": [100, 80]},
            "detection": {"head_width_pixels": 100},
            "projection": {"center_px": [100, 100], "pixels_per_unit": 200},
        }

    def test_blank_does_not_add_marks(self):
        image = Image.new("RGBA", (200, 200), "white")
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.icosphere(), geom_name="head")
        self.assertEqual(detected_lines(image, self.mapping), [])
        self.assertEqual(
            len(add_detected_details(scene, image, self.mapping).geometry), 1
        )

    def test_requires_evidence_on_both_sides(self):
        image = Image.new("RGBA", (200, 200), "white")
        d = ImageDraw.Draw(image)
        d.line((45, 90, 55, 90), fill="black", width=2)
        self.assertEqual(detected_lines(image, self.mapping), [])

    def test_bilateral_lines_and_surface(self):
        image = Image.new("RGBA", (200, 200), "white")
        d = ImageDraw.Draw(image)
        d.line((45, 90, 55, 90), fill="black", width=2)
        d.line((145, 90, 155, 90), fill="black", width=2)
        self.assertTrue(detected_lines(image, self.mapping))
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
        scene = trimesh.Scene()
        scene.add_geometry(mesh, geom_name="head")
        self.assertGreater(
            len(add_detected_details(scene, image, self.mapping).geometry), 1
        )
        self.assertEqual(len(scene.geometry), 1)
        finished = add_detected_details(scene, image, self.mapping)
        for name, part in finished.geometry.items():
            if name.startswith("detected_whisker_"):
                self.assertTrue(part.is_watertight)
                self.assertTrue(part.is_winding_consistent)

    def test_invisible_black_pixels_do_not_create_whiskers(self):
        image = Image.new("RGBA", (200, 200), (255, 255, 255, 255))
        d = ImageDraw.Draw(image)
        d.line((45, 90, 55, 90), fill=(0, 0, 0, 0), width=2)
        d.line((145, 90, 155, 90), fill=(0, 0, 0, 0), width=2)
        self.assertEqual(detected_lines(image, self.mapping), [])


if __name__ == "__main__":
    unittest.main()
