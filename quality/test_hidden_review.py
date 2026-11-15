"""Source binding, projection vetoes, and negative controls for hidden-paint review."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from engine import hidden_review as hr


def fixture():
    image = np.full((hr.ROWS, hr.COLS, 3), [173, 92, 36], dtype=np.uint8)
    image[32:69, 34:64] = [18, 24, 27]
    image[82:116, 60:82] = [23, 30, 30]
    image[144:177, 38:76] = [16, 21, 24]
    mask = np.ones(image.shape[:2], bool)
    mask[:, :12] = False
    mask[:, -12:] = False
    return image, mask


class HiddenReviewTests(unittest.TestCase):
    def test_congruent_face_on_unseen_material_requires_hold(self):
        photo, mask = fixture()
        score = hr._panel_score(photo, mask, photo[:, ::-1].copy(), mask)
        self.assertGreater(score["dice"], 0.98)
        self.assertGreater(score["layout_margin"], 0.15)
        scores = [score, score]
        self.assertEqual(hr._diagnosis(scores, None)[:2], ("needs_review", True))
        structure = {"kind": "screen_or_controls", "location": "center"}
        brief = {"visible_structures": [structure], "observed_face_cues": {}}
        self.assertEqual(hr._diagnosis(scores, brief)[:2], ("rejected", True))
        brief["observed_face_cues"]["repeated"] = {
            "family": "botanical",
            "location": "across_visible_face",
        }
        self.assertEqual(hr._diagnosis(scores, brief)[:2], ("needs_review", False))

    def test_plain_reverse_and_front_visible_patches_do_not_trigger(self):
        photo, mask = fixture()
        wrong = photo[:, ::-1].copy()
        plain = np.full_like(wrong, [164, 83, 29])
        empty = np.zeros_like(mask)
        for target, unseen in ((plain, mask), (wrong, empty)):
            score = hr._panel_score(photo, mask, target, unseen)
            self.assertEqual(
                hr._diagnosis([score, score], None)[:2], ("needs_review", False)
            )

    def test_gray_lines_or_long_uniform_stripe_do_not_trigger(self):
        photo, mask = fixture()
        gray = np.full_like(photo, 146)
        gray[35:178, 44:61] = 30
        stripe = np.full_like(photo, [173, 92, 36])
        stripe[:, 40:70] = [19, 22, 20]
        for image in (gray, stripe):
            score = hr._panel_score(image, mask, image[:, ::-1].copy(), mask)
            self.assertEqual(
                hr._diagnosis([score, score], None)[:2], ("needs_review", False)
            )

    def test_restart_is_bound_to_photograph_shape_matte_sheet_camera_and_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                root / name
                for name in ("model.glb", "original.jpg", "matte.png", "paint.png")
            ]
            for f in paths:
                f.write_bytes(f.name.encode())
            out = root / "review"

            def inspect(brief=None, camera=None):
                return hr.review_hidden_paint(
                    *paths, brief, out, camera_frame=camera or {"frame": 1}
                )

            photo, mask = fixture()
            score = hr._panel_score(photo, mask, photo[:, ::-1].copy(), mask)
            with patch.object(
                hr, "_score_images", return_value={k: score for k in hr.gt.ANGLES}
            ) as rendered:
                result = inspect()
                self.assertEqual(inspect(), result)
                for path in paths:
                    stable = path.read_bytes()
                    path.write_bytes(stable + b" different")
                    with self.assertRaisesRegex(ValueError, "different inputs"):
                        inspect()
                    path.write_bytes(stable)
                with self.assertRaisesRegex(ValueError, "different inputs"):
                    inspect(camera={"frame": 2})
                with self.assertRaisesRegex(ValueError, "observation fields"):
                    inspect(brief={"arbitrary": "untrusted"})
                marker = out / "hidden-review.json"
                tampered = json.loads(marker.read_bytes())
                tampered["hold_projection"] = False
                marker.write_text(json.dumps(tampered), encoding="utf8")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    inspect()
                rendered.assert_called_once()


if __name__ == "__main__":
    unittest.main()
