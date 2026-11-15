import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from engine.vehicle_shape import build_vehicle

ROOT = Path(__file__).resolve().parents[1]


class VehicleShapeTests(unittest.TestCase):
    def test_product_and_stretched_image_follow_proportions(self):
        image = Image.open(ROOT / "demo" / "product-cutout.png").convert("RGBA")
        normal = build_vehicle(image)
        stretched = build_vehicle(image.resize((1229, 1024)))
        self.assertIsNotNone(normal)
        self.assertIsNotNone(stretched)
        a, ma = normal
        b, mb = stretched
        self.assertGreater(a.extents[1] - b.extents[1], 0.1)
        self.assertNotEqual(ma["silhouette_bbox"][3], mb["silhouette_bbox"][3])
        self.assertEqual(len([n for n in a.geometry if n.startswith("tire_")]), 4)
        self.assertGreaterEqual(
            len([n for n in a.geometry if n.startswith("window_")]), 2
        )
        self.assertEqual(ma["front_back"].split()[0], "unknown")
        self.assertIsNotNone(a.geometry["body"].visual.material.baseColorTexture)

    def test_ground_clearance_trim_and_inferred_end_glass(self):
        image = Image.open(ROOT / "demo" / "product-cutout.png").convert("RGBA")
        scene, meta = build_vehicle(image)
        self.assertEqual(meta["cabin_trim_source"], "sampled_yellow_trim")
        self.assertEqual(len(meta["window_polygons"]), 2)
        self.assertGreater(meta["underbody_clearance"], 0.02)
        wheel_bottom = min(
            m.bounds[0, 1] for n, m in scene.geometry.items() if n.startswith("tire_")
        )
        self.assertGreater(scene.geometry["body"].bounds[0, 1] - wheel_bottom, 0.02)
        for side in (-1, 1):
            pane = scene.geometry[f"inferred_end_glass_{side}"]
            self.assertEqual(len(pane.faces), 2)
            self.assertGreater(
                float(np.dot(pane.face_normals[0], pane.face_normals[1])), 0.999
            )
            color = np.asarray(pane.visual.material.baseColorTexture)[0, 0]
            self.assertGreater(int(color[2]), int(color[0]))

    def test_visible_lamps_follow_pixels(self):
        image = Image.open(ROOT / "demo" / "product-cutout.png").convert("RGBA")
        scene, meta = build_vehicle(image)
        self.assertEqual(len(meta["visible_lamps_detected"]), 2)
        self.assertIn("visible_lamp_0", scene.geometry)
        ImageDraw.Draw(image).rectangle((790, 480, 880, 530), fill=(45, 150, 140, 255))
        without, info = build_vehicle(image)
        self.assertEqual(info["visible_lamps_detected"], [])
        self.assertNotIn("visible_lamp_0", without.geometry)

    def test_nonvehicle_shapes_are_rejected(self):
        self.assertIsNone(
            build_vehicle(Image.new("RGBA", (300, 300), (255, 255, 255, 255)))
        )
        image = Image.new("RGBA", (400, 300))
        draw = ImageDraw.Draw(image)
        draw.ellipse((110, 20, 290, 260), fill=(200, 120, 60, 255))
        self.assertIsNone(build_vehicle(image))
        draw.ellipse((115, 190, 150, 225), fill=(20, 20, 20, 255))
        draw.ellipse((250, 190, 285, 225), fill=(20, 20, 20, 255))
        self.assertIsNone(build_vehicle(image))

    def test_no_reference_pixels_projected_onto_far_side(self):
        image = Image.open(ROOT / "demo" / "product-cutout.png").convert("RGBA")
        scene, meta = build_vehicle(image)
        self.assertIn("no side image projection", meta["inferred"])
        material = scene.geometry["body"].visual.material
        patch = np.asarray(material.baseColorTexture)
        self.assertEqual(patch.shape, (8, 8, 3))
        self.assertEqual(material.metallicFactor, 0)


if __name__ == "__main__":
    unittest.main()
