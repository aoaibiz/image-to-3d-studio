import io
import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from figure_paint import _wrap_head, paint_figure


class FigurePaintTests(unittest.TestCase):
    def fixture(self, ink):
        image = Image.new("RGBA", (120, 120), (255, 0, 255, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse((10, 10, 110, 110), fill=(230, 232, 226, 255))
        draw.ellipse((48, 42, 72, 63), fill=(*ink, 255))
        scene = trimesh.Scene()
        head = trimesh.creation.icosphere(subdivisions=3, radius=0.50)
        scene.add_geometry(head, geom_name="head")
        mapping = {
            "projection": {
                "center_px": [60, 60],
                "pixels_per_unit": 100,
                "image_size": [120, 120],
            },
            "landmarks_px": {"head_center": [60, 60], "tail_skeleton": []},
            "detection": {"head_width_pixels": 100, "tail_side": 1},
        }
        return scene, image, mapping

    def test_round_trip_front_feature_is_from_input_and_rear_stays_plain(self):
        for ink in ((12, 110, 230), (238, 65, 20)):
            scene, image, mapping = self.fixture(ink)
            result = paint_figure(scene, image, mapping)
            reloaded = trimesh.load(
                io.BytesIO(result.export(file_type="glb")),
                file_type="glb",
                force="scene",
                process=False,
            )
            head = next(iter(reloaded.geometry.values()))
            atlas = np.asarray(head.visual.material.baseColorTexture)
            scale = atlas.shape[0] / image.height
            np.testing.assert_allclose(
                atlas[int(52 * scale), int(60 * scale), :3], ink, atol=5
            )
            np.testing.assert_allclose(
                atlas[int(52 * scale), int(180 * scale), :3], [230, 232, 226], atol=5
            )
            self.assertEqual(len(head.faces), len(scene.geometry["head"].faces))
            np.testing.assert_allclose(
                head.bounds, scene.geometry["head"].bounds, atol=1e-6
            )
            self.assertFalse(
                any(
                    r > 240 and b > 240 and g < 5
                    for r, g, b in mapping["paint"]["source_palette_srgb"]
                )
            )
            # Both UV islands actually appear on the intended faces.
            front = head.face_normals[:, 2] > 0.3
            back = head.face_normals[:, 2] < -0.3
            uv = head.visual.uv[head.faces][:, :, 0]
            self.assertTrue(np.all(uv[front] < 0.5))
            self.assertTrue(np.all(uv[back] > 0.5))

    def test_projection_must_match_reference(self):
        scene, image, mapping = self.fixture((30, 75, 120))
        with self.assertRaisesRegex(ValueError, "sizes differ"):
            paint_figure(scene, image.resize((80, 80)), mapping)

    def test_offset_forehead_stripe_continues_over_the_top(self):
        scene, image, mapping = self.fixture((25, 80, 220))
        front = np.full((120, 120, 3), [230, 232, 226], np.uint8)
        front[:71, :53] = [210, 96, 55]
        front[:71, 64:] = [22, 28, 32]
        front[53:58, 33:37] = [245, 246, 246]
        panel = np.concatenate((front, np.zeros_like(front)), axis=1)
        result, _ = _wrap_head(
            panel,
            scene.geometry["head"],
            image,
            mapping["projection"],
            np.array([230, 232, 226]),
        )
        # At the projected crown rim z=0, the same colour must be found on
        # both hemispheres, including either side of the off-centre stripe.
        for x in range(23, 97):
            y = round(60 - 50 * np.sqrt(1 - ((x - 60) / 50) ** 2))
            if y < 23:
                np.testing.assert_array_equal(result[y, x], result[y, x + 120])
        np.testing.assert_array_equal(result[55, 35], front[55, 35])


if __name__ == "__main__":
    unittest.main()
