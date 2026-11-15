import unittest
import json
from PIL import Image, ImageDraw
from engine.figure_shape import _measure, build_figure


def figure(head_left=50, head_right=250, tail_x=280):
    im = Image.new("RGBA", (300, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon(
        [(76, 112), (70, 30), (120, 75), (180, 75), (230, 30), (224, 112)],
        fill=(220, 200, 180, 255),
    )
    d.ellipse((head_left, 75, head_right, 210), fill=(220, 200, 180, 255))
    d.ellipse((65, 175, 235, 340), fill=(220, 200, 180, 255))
    d.rounded_rectangle((70, 300, 125, 390), 15, fill=(220, 200, 180, 255))
    d.rounded_rectangle((175, 300, 230, 390), 15, fill=(220, 200, 180, 255))
    d.line([(220, 285), (260, 265), (tail_x, 180)], fill=(220, 200, 180, 255), width=24)
    d.arc((75, 170, 150, 250), 0, 200, fill="black", width=3)
    d.arc((150, 170, 225, 250), 340, 180, fill="black", width=3)
    return im


class FigureShapeTests(unittest.TestCase):
    def test_unsupported_opaque_and_rectangle(self):
        self.assertIsNone(build_figure(Image.new("RGB", (100, 100), "white")))
        im = Image.new("RGBA", (300, 400))
        ImageDraw.Draw(im).rectangle((60, 20, 240, 390), fill="white")
        self.assertIsNone(build_figure(im))

    def test_ear_foot_tail_detected_and_changes_with_subject(self):
        a = _measure(figure())
        b = _measure(figure(60, 240, 260))
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertEqual(len(a["ears"]), 2)
        self.assertEqual(len(a["feet"]), 2)
        self.assertGreater(len(a["tail"]), 5)
        self.assertNotEqual(
            a["head_right"] - a["head_left"], b["head_right"] - b["head_left"]
        )
        self.assertNotEqual(max(p[0] for p in a["tail"]), max(p[0] for p in b["tail"]))

    def test_geometry_and_projection(self):
        result = build_figure(figure())
        self.assertIsNotNone(result)
        scene, evidence = result
        self.assertEqual(
            set(scene.geometry), {"cast_body", "head", "ear_left", "ear_right"}
        )
        json.dumps(evidence)
        self.assertGreater(evidence["detection"]["neck_skull_axis_overlap"], 0.01)
        self.assertGreater(evidence["detection"]["neck_torso_axis_overlap"], 0.01)
        p = evidence["projection"]
        self.assertGreater(p["pixels_per_unit"], 0)
        self.assertEqual(p["image_size"], [300, 400])
        self.assertTrue(scene.geometry["cast_body"].is_watertight)
        self.assertGreater(
            scene.geometry["cast_body"].bounds[1, 1],
            scene.geometry["head"].bounds[0, 1] + 0.02,
        )
        self.assertEqual(
            len(scene.geometry["cast_body"].split(only_watertight=False)), 1
        )


if __name__ == "__main__":
    unittest.main()
