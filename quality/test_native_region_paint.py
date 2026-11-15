"""Native material inputs, asymmetric-face holds and hole-aware regional admission."""

from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import trimesh

from engine import native_region_paint as paint
from engine import codex_photo_observations as obs
from engine.visual_brief import json_sha


def observation():
    return {
        "method": obs.METHOD_V2,
        "source_sha256": "1" * 64,
        "visible_palette": ["red", "gold"],
        "optical_finish_candidate": "glossy",
        "layout_candidate": "repeated",
        "observed_face_cues": {
            "repeated": {"family": "botanical", "location": "across_visible_face"}
        },
        "visible_structures": [],
        "visible_surround": {"color": "red", "pigment_layout": "patterned"},
        "unseen_surface_observed": False,
        "function_context": {
            "observed_cues": [{"kind": "boundary_seam", "location": "side_border"}],
            "interpretation": {
                "visible_face_role": "cover_or_outer_shell",
                "possible_purpose": "enclose_or_protect",
                "reverse_candidates": [
                    "distinct_working_surface",
                    "display_or_interface",
                ],
                "confidence": "low",
            },
            "reverse_observed": False,
        },
    }


class NativePaintTests(unittest.TestCase):
    def test_preserves_geometry_and_decoded_embedded_native_texture(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        uv = np.clip(mesh.vertices[:, :2] * 0.4 + 0.5, 0, 1)
        albedo = Image.new("RGB", (256, 256), (64, 42, 17))
        normals = Image.new("RGB", (256, 256), (128, 128, 255))
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=albedo,
            normalTexture=normals,
            roughnessFactor=0.3,
            metallicFactor=0.5,
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "candidate.glb"
            p.write_bytes(
                trimesh.exchange.gltf.export_glb(
                    trimesh.Scene(mesh), include_normals=True
                )
            )
            doc, blob, rgb, loaded = paint._native(p)
            self.assertEqual(len(loaded.faces), len(mesh.faces))
            self.assertEqual(rgb.shape, (256, 256, 3))
            self.assertEqual(doc["materials"][0]["normalTexture"]["index"], 1)
            changed = deepcopy(doc)
            changed["images"][0]["uri"] = "https://invalid.example/absent.jpg"
            serialized = json.dumps(changed).encode()
            serialized += b" " * ((-len(serialized)) % 4)
            header = struct.pack(
                "<4sIIII",
                b"glTF",
                2,
                28 + len(serialized) + len(blob),
                len(serialized),
                0x4E4F534A,
            )
            p.write_bytes(
                header + serialized + struct.pack("<II", len(blob), 0x004E4942) + blob
            )
            with patch.object(
                trimesh, "load", side_effect=AssertionError("unbounded decoder invoked")
            ):
                with self.assertRaisesRegex(ValueError, "embedded"):
                    paint._native(p)

    def test_shifted_opening_in_solid_body_and_chair_like_frame_are_refused(self):
        y, x = np.ogrid[:384, :384]
        mask = (x - 192) ** 2 + (y - 192) ** 2 < 150**2
        expected = mask & ((x - 156) ** 2 + (y - 190) ** 2 > 28**2)
        shifted = mask & ((x - 230) ** 2 + (y - 205) ** 2 > 28**2)
        rgb = np.where(shifted[:, :, None], [112, 63, 29], 255).astype("uint8")
        with self.assertRaisesRegex(ValueError, "opening"):
            paint.align(rgb, expected)
        frame = np.zeros_like(mask)
        frame[60:320, 90:99] = True
        frame[60:320, 270:279] = True
        frame[60:69, 90:279] = True
        with self.assertRaisesRegex(ValueError, "thick|dominant"):
            paint.body(frame)

    def test_two_valid_solid_surfaces_keep_uncertain_edges_unpainted(self):
        y, x = np.ogrid[:384, :384]
        solid = (x - 192) ** 2 + (y - 192) ** 2 < 120**2
        rgb = np.where(
            np.roll(solid, 8, axis=1)[:, :, None], [155, 82, 41], 255
        ).astype("uint8")
        _, weight, metrics = paint.align(rgb, solid)
        self.assertGreater(metrics["solid_iou"], 0.99)
        self.assertLess(metrics["retained_pixel_fraction"], 1)
        self.assertFalse(weight[~solid].any())
        self.assertEqual(weight[192, 192], 1)

    def test_semantic_veto_and_possible_other_function_win_over_repeated_ornament(self):
        brief = observation()
        sha = brief["source_sha256"]
        h = json_sha(brief)
        layout = {"visual_brief": brief}
        receipt = {"visual_brief_sha256": h}
        review = {
            "status": "needs_review",
            "hold_projection": False,
            "inputs": {"brief_sha256": h},
        }
        with self.assertRaisesRegex(ValueError, "distinct function"):
            paint._admit_observation(layout, receipt, review, sha)
        with self.assertRaisesRegex(ValueError, "orientation"):
            paint._admit_observation(
                layout, receipt, {**review, "hold_projection": True}, sha
            )
        with self.assertRaisesRegex(ValueError, "identit"):
            paint._admit_observation(
                layout, {"visual_brief_sha256": "0" * 64}, review, sha
            )
        prior = deepcopy(brief)
        prior["method"] = obs.METHOD
        del prior["function_context"]
        h = json_sha(prior)
        review["inputs"]["brief_sha256"] = h
        accepted = paint._admit_observation(
            {"visual_brief": prior}, {"visual_brief_sha256": h}, review, sha
        )
        self.assertEqual(accepted["method"], obs.METHOD)
        self.assertFalse(accepted["unseen_surface_observed"])

    def test_seam_cannot_paint_outside_support_or_drop_required_core(self):
        a = np.full((280, 280, 3), 150, np.uint8)
        b = a.copy()
        b[:, :, 1] = 70
        q = np.zeros((280, 280), np.float32)
        q[40:240, 40:240] = 1
        seam = paint.cut(a, b, q)
        self.assertEqual(float(seam[140, 140]), 1.0)
        self.assertEqual(float(seam[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
