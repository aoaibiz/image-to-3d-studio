import unittest
from quality.inspection import inspect_geometry
from quality.measured_regions import measured_profile


def box(x0, x1, y0, y1, z0, z1):
    v = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (3, 2, 6),
        (3, 6, 7),
        (0, 3, 7),
        (0, 7, 4),
        (1, 5, 6),
        (1, 6, 2),
    ]
    return [tuple(v[i] for i in face) for face in faces]


class MeasuredRegionsTests(unittest.TestCase):
    def setUp(self):
        self.settings = {
            "finish_profile": "standing-figure",
            "finish_measurements": {
                "projection": {
                    "axes": "+X right, +Y up, +Z front",
                    "center_px": [150, 183.5],
                    "pixels_per_unit": 313,
                },
                "landmarks_px": {
                    "head_center": [150, 102],
                    "feet": [[126, 324], [199, 324]],
                    "arm_edges": [
                        [[102, 176.5], [130, 176.5]],
                        [[192, 176.5], [169, 176.5]],
                    ],
                },
                "detection": {"head_width_pixels": 126, "body_width_pixels": 130},
            },
        }
        self.profile = measured_profile(self.settings)

    def test_thin_head_and_torso_with_tail(self):
        tail = box(0.30, 0.40, -0.3, 0.0, -0.09, 0.09)
        body = box(-0.17, 0.17, -0.3, -0.04, -0.12, 0.12)
        head = box(-0.18, 0.18, 0.17, 0.35, -0.12, 0.12)
        thin_head = box(-0.18, 0.18, 0.17, 0.35, -0.005, 0.005)
        thin_body = box(-0.17, 0.17, -0.3, -0.04, -0.005, 0.005)
        self.assertNotEqual(
            inspect_geometry(body + head + tail, self.profile)["status"], "fail"
        )
        self.assertEqual(
            inspect_geometry(body + thin_head + tail, self.profile)["regions"]["head"][
                "status"
            ],
            "fail",
        )
        self.assertEqual(
            inspect_geometry(thin_body + head + tail, self.profile)["regions"]["body"][
                "status"
            ],
            "fail",
        )

    def test_unavailable_for_other_subject_or_invalid_landmarks(self):
        self.assertIsNone(measured_profile({"finish_profile": "wheeled-vehicle"}))
        self.settings["finish_measurements"]["projection"]["pixels_per_unit"] = 0
        self.assertIsNone(measured_profile(self.settings))


if __name__ == "__main__":
    unittest.main()
