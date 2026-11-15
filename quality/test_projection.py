import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from reference_texture import visibility, raster


class ProjectionTests(unittest.TestCase):
    def test_front_visible_back_and_outside_blocked(self):
        zbuf = np.ones((3, 3))
        mask = np.ones((3, 3), dtype=bool)
        mask[0, 0] = False
        result = visibility(
            np.array([1.0, 0.4, 1.0, 1.0]),
            np.array([1, 1, 0, -1]),
            np.array([1, 1, 0, 1]),
            zbuf,
            mask,
            0.01,
        )
        np.testing.assert_array_equal(result, [True, False, False, False])

    def test_triangle_raster_has_barycentric_weights(self):
        y, x, weights = raster(np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]), (3, 3))
        self.assertIn((0, 0), set(zip(y, x)))
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
