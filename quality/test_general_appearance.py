import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from quality.general_appearance import assess_general


class AppearanceTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.folder = Path(self.work.name)
        model = self.folder / "model.glb"
        model.write_bytes(b"synthetic model identity for image-space tests")
        self.sha = hashlib.sha256(model.read_bytes()).hexdigest()
        self.source = self.folder / "foreground.png"
        reference = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(reference)
        draw.rectangle((56, 30, 140, 169), fill=(210, 45, 34, 255))
        draw.rectangle((20, 55, 56, 78), fill=(210, 45, 34, 255))
        reference.save(self.source)
        names = []
        for kind in ["front_material", "front_clay", "left_clay", "right_clay"]:
            im = Image.new("RGB", (200, 200), (195, 200, 203))
            front = kind.startswith("front")
            draw = ImageDraw.Draw(im)
            draw.rectangle(
                (56 if front else 98, 30, 140 if front else 102, 169),
                fill=(24, 159, 64) if kind == "front_material" else (242, 244, 244),
            )
            name = self.sha + "_" + kind + ".png"
            im.save(self.folder / name)
            names.append(name)
        self.manifest = {
            "sha256": self.sha,
            "files": names,
            "front": "+Z",
            "projection": "orthographic",
        }
        (self.folder / "manifest.json").write_text(json.dumps(self.manifest))

    def assess(self):
        return assess_general(self.source, self.folder, self.sha)

    def test_flat_sides_missing_region_and_different_colours(self):
        result = self.assess()
        self.assertEqual(result["status"], "review")
        self.assertIn(
            "narrow_side_sections_check_intended_subject_depth", result["flags"]
        )
        self.assertIn(
            "substantial_input_region_unmatched_check_for_missing_parts_or_pose_difference",
            result["flags"],
        )
        self.assertLess(result["front_chroma_histogram_intersection"], 0.01)
        self.assertIn(
            "low_chroma_overlap_check_materials_and_lighting", result["flags"]
        )
        self.assertTrue((self.folder / result["contact_sheet"]).is_file())
        self.assertEqual(len(result["render_sha256"]), 4)

    def test_missing_or_wrong_evidence_stays_unknown(self):
        (self.folder / "manifest.json").write_text(
            json.dumps({**self.manifest, "sha256": "f" * 64})
        )
        self.assertEqual(self.assess()["status"], "unknown")
        (self.folder / "manifest.json").write_text(json.dumps(self.manifest))
        (self.folder / self.manifest["files"][-1]).unlink()
        self.assertEqual(self.assess()["status"], "unknown")

    def test_unsegmented_source_is_unknown(self):
        Image.new("RGB", (200, 200), "white").save(self.source)
        result = self.assess()
        self.assertEqual(result["status"], "unknown")
        self.assertIn("alpha", result["reason"])

    def test_edge_on_disappearance_is_unknown_but_keeps_comparison(self):
        path = self.folder / (self.sha + "_right_clay.png")
        Image.new("RGB", (200, 200), (195, 200, 203)).save(path)
        result = self.assess()
        self.assertEqual(result["status"], "unknown")
        self.assertIn(
            "side_silhouette_not_resolved_check_edge_on_view", result["flags"]
        )
        self.assertTrue((self.folder / result["contact_sheet"]).is_file())

    def test_ideal_match_never_grants_appearance_pass(self):
        reference = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        ImageDraw.Draw(reference).rectangle((56, 30, 140, 169), fill=(24, 159, 64, 180))
        reference.save(self.source)
        result = self.assess()
        self.assertEqual(result["status"], "review")
        self.assertEqual(result["front_outline"]["iou"], 1)
        self.assertEqual(result["front_chroma_histogram_intersection"], 1)
        self.assertIn(
            "partially_transparent_input_outline_is_approximate", result["flags"]
        )


if __name__ == "__main__":
    unittest.main()
