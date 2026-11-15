import math, sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from inspection import inspect_geometry


def tube(width, center=0, curve=False):
    tris = []
    n = 24
    levels = [0, 0.5, 1, 1.5, 2]
    rings = [
        [
            (
                math.cos(i * 2 * math.pi / n) * 0.5,
                y,
                center
                + math.sin(i * 2 * math.pi / n)
                * width
                * (0.8 + 0.2 * math.sin(y * math.pi / 2) if curve else 1),
            )
            for i in range(n)
        ]
        for y in levels
    ]
    for r in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            a, b, c, d = rings[r][i], rings[r][j], rings[r + 1][i], rings[r + 1][j]
            tris.extend([(a, b, c), (b, d, c)])
    return tris


def profile():
    return {
        "subject": "cat",
        "up_axis": 1,
        "depth_axis": 2,
        "regions": {
            "body": {
                "bounds": [[-1, 1], [0, 1], [-0.6, 0.6]],
                "levels": [0.2, 0.5, 0.8],
                "min_depth_ratio": 0.2,
            },
            "head": {
                "bounds": [[-1, 1], [1, 2], [-0.6, 0.6]],
                "levels": [1.2, 1.5, 1.8],
                "min_depth_ratio": 0.2,
            },
        },
    }


class TestGeometry(unittest.TestCase):
    def test_flat_plane(self):
        self.assertEqual(inspect_geometry(tube(0), profile())["status"], "fail")

    def test_thin_torso_with_large_tail(self):
        body = tube(0.015)
        tail = tube(0.4, center=2)
        self.assertEqual(inspect_geometry(body + tail, profile())["status"], "fail")

    def test_curved_thick_form(self):
        self.assertEqual(
            inspect_geometry(tube(0.4, curve=True), profile())["status"], "pass"
        )

    def test_legitimate_thin_item_has_no_global_threshold(self):
        self.assertEqual(
            inspect_geometry(
                tube(0.01), {"subject": "sheet", "up_axis": 1, "depth_axis": 2}
            )["status"],
            "review",
        )

    def test_snug_depth_bounds_do_not_hide_flatness(self):
        p = profile()
        for region in p["regions"].values():
            region["bounds"][2] = [-0.02, 0.02]
        self.assertEqual(inspect_geometry(tube(0.015), p)["status"], "fail")

    def test_tail_inside_bounds_cannot_hide_flat_torso(self):
        p = profile()
        for region in p["regions"].values():
            region["bounds"][2] = [-0.6, 3]
        # A surface strip attached to the torso stretches deep on one side only.
        tail = [
            ((0.45, 0, 0), (0.5, 0, 2.5), (0.45, 2, 0)),
            ((0.5, 0, 2.5), (0.5, 2, 2.5), (0.45, 2, 0)),
        ]
        self.assertNotEqual(inspect_geometry(tube(0.015) + tail, p)["status"], "pass")

    def test_two_disconnected_plates_do_not_pass(self):
        plates = []
        for z in (-0.4, 0.4):
            plates.extend(
                [
                    ((-0.5, 0, z), (0.5, 0, z), (-0.5, 2, z)),
                    ((0.5, 0, z), (0.5, 2, z), (-0.5, 2, z)),
                ]
            )
        self.assertEqual(inspect_geometry(plates, profile())["status"], "review")

    def test_uneven_tessellation_does_not_change_result(self):
        base = tube(0.4)
        refined = []
        for a, b, c in base[:20]:
            ab = tuple((a[i] + b[i]) / 2 for i in range(3))
            bc = tuple((b[i] + c[i]) / 2 for i in range(3))
            ca = tuple((c[i] + a[i]) / 2 for i in range(3))
            refined.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
        self.assertEqual(
            inspect_geometry(refined + base[20:], profile())["status"], "pass"
        )

    def test_no_orientation_or_segmentation_needs_review(self):
        self.assertEqual(inspect_geometry(tube(0.4), None)["status"], "review")

    def test_missing_section_needs_review(self):
        p = profile()
        p["regions"]["head"]["levels"] = [1.2, 1.5, 1.8]
        self.assertEqual(
            inspect_geometry([t for t in tube(0.4) if max(v[1] for v in t) <= 1], p)[
                "status"
            ],
            "review",
        )


if __name__ == "__main__":
    unittest.main()
