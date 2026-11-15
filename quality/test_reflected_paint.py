import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

from quality.reflected_paint import assess_reflected_paint


class ReflectedPaintTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.model = self.folder / "model.glb"
        self.model.write_bytes(trimesh.creation.box().export(file_type="glb"))
        self.digest = hashlib.sha256(self.model.read_bytes()).hexdigest()
        yy, xx = np.ogrid[:180, :180]
        self.body = ((xx - 89.5) / 72) ** 2 + ((yy - 89.5) / 76) ** 2 < 1
        self.clay = self.image((210, 205, 201))
        self.front = self.image((155, 112, 71))
        rng = np.random.default_rng(234)
        for x, y in rng.integers(42, 139, size=(25, 2)):
            cv2.circle(self.front, (int(x), int(y)), 3, (25, 48, 62), -1)

    def image(self, color):
        rgb = np.full((180, 180, 3), (236, 235, 230), dtype=np.uint8)
        rgb[self.body] = color
        return rgb

    def assess(self, observed, opposite, clay=None):
        names = []
        for view, paint in (("front", observed), ("back", opposite)):
            neutral = self.clay if clay is None else clay[view]
            for mode, rgb in (("clay", neutral), ("material", paint)):
                name = f"{self.digest}_{view}_{mode}.png"
                Image.fromarray(rgb).save(self.folder / name)
                names.append(name)
        manifest = {
            "sha256": self.digest,
            "files": names,
            "synthetic": False,
            "front": "+Z",
            "projection": "orthographic",
        }
        (self.folder / "manifest.json").write_text(json.dumps(manifest))
        return assess_reflected_paint(self.model, self.folder)

    def test_reflected_asymmetry_with_shift_scale_and_dimming(self):
        matrix = cv2.getRotationMatrix2D((89.5, 89.5), 0, 1.035)
        matrix[:, 2] += (3, -1)
        opposite = cv2.warpAffine(self.front[:, ::-1], matrix, (180, 180))
        opposite = (opposite * 0.73).astype(np.uint8)
        result = self.assess(self.front, opposite)
        self.assertEqual(result["status"], "review_hold")
        self.assertTrue(result["manual_review_required"])
        self.assertEqual(len(result["render_sha256"]), 4)
        self.assertEqual(result["glb_sha256"], self.digest)

    def test_different_back_does_not_raise_a_hold(self):
        result = self.assess(self.front, self.image((112, 62, 21)))
        self.assertEqual(result["status"], "no_evidence")
        self.assertTrue(result["manual_review_required"])

    def test_plain_stripes_periodic_tiles_and_bilateral_symmetry_abstain(self):
        plain = self.image((108, 78, 58))
        stripes = plain.copy()
        stripes[(np.indices(self.body.shape)[0] % 16 < 6) & self.body] = (180, 123, 61)
        tiled = plain.copy()
        for y in range(14, 180, 18):
            for x in range(19, 180, 18):
                cv2.rectangle(tiled, (x, y), (x + 4, y + 8), (50, 29, 75), -1)
        bilateral = self.front.copy()
        bilateral[:, 90:] = bilateral[:, :90][:, ::-1]
        for name, values in (
            ("plain", plain),
            ("stripes", stripes),
            ("repeat", tiled),
            ("bilateral", bilateral),
        ):
            with self.subTest(case=name):
                result = self.assess(values, values[:, ::-1])
                self.assertEqual(result["status"], "inconclusive")

    def test_neutral_relief_is_not_classified_as_reflected_paint(self):
        gray = cv2.cvtColor(self.front, cv2.COLOR_RGB2GRAY)
        neutral = np.stack([gray, gray, gray], axis=-1)
        neutral[~self.body] = (236, 235, 230)
        result = self.assess(
            neutral, neutral[:, ::-1], clay={"front": neutral, "back": neutral[:, ::-1]}
        )
        self.assertEqual(result["reason"], "low_contrast_or_plain_paint")

    def test_foreign_manifest_is_not_evidence_and_missing_files_do_not_pass(self):
        self.assess(self.front, self.front[:, ::-1])
        self.model.write_bytes(
            trimesh.creation.icosphere(subdivisions=0).export(file_type="glb")
        )
        result = assess_reflected_paint(self.model, self.folder)
        self.assertEqual(result["reason"], "unbound_or_noncanonical_evidence")
        (self.folder / "manifest.json").unlink()
        self.assertEqual(
            assess_reflected_paint(self.model, self.folder)["status"], "inconclusive"
        )
