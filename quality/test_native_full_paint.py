"""Synthetic controls exercise strict native full-view projection and receipt bindings."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import trimesh

from engine import guided_texture as gt
from engine import native_full_paint as paint
from engine import native_region_paint as native
from engine.codex_photo_observations import METHOD
from engine.triposg_cpu import digest
from engine.visual_brief import json_sha


def fixture(dirpath):
    body = trimesh.creation.icosphere(subdivisions=1)
    tri = body.faces
    cells = np.arange(len(tri))
    grid = int(np.ceil(len(cells) ** 0.5))
    offset = np.column_stack((cells % grid, cells // grid))
    uv = (offset[:, None] + [[0.08, 0.08], [0.92, 0.08], [0.45, 0.92]]) / grid
    mesh = trimesh.Trimesh(
        vertices=body.vertices[tri].reshape(-1, 3),
        vertex_normals=body.vertex_normals[tri].reshape(-1, 3),
        faces=np.arange(tri.size).reshape(-1, 3),
        process=False,
    )
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.new("RGB", (256, 256), (70, 92, 128)),
        metallicFactor=0.25,
        roughnessFactor=0.35,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv.reshape(-1, 2), material=mat)
    original, common, photo, matte = [
        dirpath / s for s in ("native.glb", "common.glb", "input.png", "foreground.png")
    ]

    def export(path):
        path.write_bytes(
            trimesh.exchange.gltf.export_glb(trimesh.Scene(mesh), include_normals=True)
        )

    export(common)
    mat.normalTexture = Image.new("RGB", (256, 256), (128, 128, 255))
    export(original)
    guide = dirpath / "guides"
    manifest = gt.prepare_guides(common, guide)
    frame = dict(
        zip(
            ("image_right", "image_up", "camera_direction_from_origin"),
            np.eye(3).tolist(),
        )
    )
    reg = {
        "mesh_file_sha256": digest(original),
        "camera_frame": frame,
        "upright_transform": np.eye(4).tolist(),
    }
    front = gt._view(mesh, (0, 0))[3] >= 0
    rgba = np.zeros((*front.shape, 4), np.uint8)
    rgba[front] = [70, 92, 128, 255]
    Image.fromarray(rgba).save(matte)
    Image.fromarray(rgba[:, :, :3]).save(photo)
    reg["foreground_file_sha256"] = digest(matte)
    sheet = Image.new("RGB", (2 * gt.SIZE, gt.SIZE), "white")
    y, x = np.indices((gt.SIZE, gt.SIZE))
    for i, angle in enumerate(gt.ANGLES.values()):
        valid = gt._view(mesh, angle)[3] >= 0
        rgb = np.full((*front.shape, 3), 255, np.uint8)
        rgb[valid] = [140, 104, 75]
        pigment = valid & (
            ((x - gt.SIZE * 0.48) ** 2 + (y - gt.SIZE * 0.55) ** 2)
            < (gt.SIZE * 0.09) ** 2
        )
        rgb[pigment] = [76, 80, 65]
        sheet.paste(Image.fromarray(rgb), (i * gt.SIZE, 0))
    sheet_path = dirpath / "sheet.png"
    sheet.save(sheet_path)
    brief = {
        "method": METHOD,
        "source_sha256": digest(photo),
        "visible_palette": ["blue"],
        "optical_finish_candidate": "glossy",
        "layout_candidate": "plain_or_uncertain",
        "observed_face_cues": {},
        "visible_structures": [],
        "visible_surround": {"color": "blue", "pigment_layout": "uniform"},
        "unseen_surface_observed": False,
    }
    layout = {
        "model_sha": digest(common),
        "photo_sha": digest(photo),
        "guide_sha": manifest["guide_sha256"],
        "camera": manifest["views"],
        "visual_brief": brief,
    }
    receipt = {
        "thread_id": "fixture",
        "state": "completed",
        "geometry_sha256": digest(common),
        "photo_sha256": digest(photo),
        "guide_sha256": layout["guide_sha"],
        "sheet_sha256": digest(sheet_path),
        "visual_brief_sha256": json_sha(brief),
    }
    evidence = {
        "model_sha256": digest(common),
        "source_sha256": digest(photo),
        "foreground_sha256": digest(matte),
        "brief_sha256": json_sha(brief),
        "sheet_sha256": digest(sheet_path),
        "camera_sha256": json_sha(frame),
        "angles": manifest["views"],
    }
    orientation = {
        "status": "needs_review",
        "hold_projection": False,
        "inputs": evidence,
    }
    orientation["report_sha256"] = json_sha(orientation)
    return [
        original,
        common,
        photo,
        matte,
        reg,
        guide,
        sheet_path,
        layout,
        receipt,
        orientation,
        dirpath / "output",
    ]


class NativeFullTests(unittest.TestCase):
    def test_thick_non_repeated_subject_and_native_material(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            diagnostic = paint.preflight_native_paint(args[0], args[4])
            self.assertTrue(diagnostic["full_sheet_eligible"])
            self.assertFalse(diagnostic["region_depth_eligible"])
            result = paint.paint_native_full(*args)
            self.assertEqual(result["appearance"], "needs_review")
            self.assertFalse(result["source_rear_observed"])
            self.assertGreater(result["changed_uv"], 100)
            self.assertGreater(result["two_view_chroma_uv"], 0)
            self.assertEqual(
                (result["locked_changed"], result["unsupported_surface_changed"]),
                (0, 0),
            )
            self.assertEqual(result["output_glb"], digest(args[-1] / "model.glb"))
            self.assertEqual(
                result, json.loads((args[-1] / "full-paint.json").read_text())
            )
            old, blob, _, _ = native._native(args[0])
            new, other, _, _ = native._native(args[-1] / "model.glb")
            self.assertEqual(old["materials"], new["materials"])
            self.assertEqual(old["accessors"], new["accessors"])
            for i, view in enumerate(old["bufferViews"]):
                if i != old["images"][0]["bufferView"]:
                    self.assertEqual(
                        native._view_bytes(blob, view),
                        native._view_bytes(other, new["bufferViews"][i]),
                    )

    def test_tamper_is_not_a_fallback_and_precedes_semantic_veto(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            orientation = args[-2]
            orientation["hold_projection"] = True
            orientation["report_sha256"] = json_sha(
                {k: v for k, v in orientation.items() if k != "report_sha256"}
            )
            with self.assertRaises(paint.NativeFullReviewHold):
                paint.paint_native_full(*args)
            args[-3]["sheet_sha256"] = "0" * 64
            with self.assertRaises(ValueError) as caught:
                paint.paint_native_full(*args)
            self.assertIs(type(caught.exception), ValueError)
            self.assertFalse(args[-1].exists())

    def test_reflected_functional_or_unscreenable_face_keeps_native(self):
        different = {
            "function_context": {
                "interpretation": {"reverse_candidates": ["display_or_interface"]}
            }
        }
        for status in ("inconclusive", "review_hold"):
            with self.assertRaises(paint.NativeFullReviewHold):
                paint._require_copy_evidence({"status": status}, different)
        self.assertTrue(
            paint._require_copy_evidence({"status": "no_evidence"}, different)
        )
        self.assertFalse(paint._require_copy_evidence({"status": "inconclusive"}, {}))

    def test_second_displaced_hole_refuses_both_views(self):
        from unittest.mock import patch

        n = gt.SIZE
        y, x = np.indices((n, n))
        solid = (x - n // 2) ** 2 + (y - n // 2) ** 2 < (n // 3) ** 2

        def circle(cx):
            return (x - cx) ** 2 + (y - n // 2) ** 2 < (n // 15) ** 2

        right = solid & ~circle(n // 2)
        wrong = solid & ~circle(3 * n // 5)

        def panel(m):
            return Image.fromarray(
                np.where(m[:, :, None], [135, 89, 55], 255).astype("uint8")
            )

        sheet = Image.new("RGB", (2 * n, n))
        sheet.paste(panel(right), (0, 0))
        sheet.paste(panel(wrong), (n, 0))
        view = (
            np.eye(3),
            np.zeros(3),
            1,
            np.where(right, 0, -1),
            None,
            np.zeros_like(x),
        )
        with patch.object(gt, "_view", return_value=view):
            with self.assertRaisesRegex(
                paint.NativeFullRegistrationError, "opposite.*opening"
            ):
                paint._field(None, sheet, np.array([[0, 0, 0]]), np.array([[0, 0, 1]]))

    def test_reopened_material_reflected_source_layout_is_a_hold(self):
        mesh = trimesh.creation.box(extents=(1, 1, 0.4))
        # Reuse the exact same spatial layout on the two planes as the negative control.
        tex = Image.new("RGB", (256, 256), "tan")
        from PIL import ImageDraw

        draw = ImageDraw.Draw(tex)
        random = np.random.default_rng(37)
        for x, y, width, height in random.integers(
            [4, 4, 3, 4], [240, 240, 20, 22], (46, 4)
        ):
            draw.ellipse((x, y, x + width, y + height), fill=(45, 53, 84))
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=mesh.vertices[:, :2] + 0.5,
            material=trimesh.visual.material.PBRMaterial(baseColorTexture=tex),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicated.glb"
            path.write_bytes(
                trimesh.exchange.gltf.export_glb(
                    trimesh.Scene(mesh), include_normals=True
                )
            )
            _, _, pixels, reopened = native._native(path)
            result = paint._copy_check(reopened, pixels)
            self.assertEqual(result["status"], "review_hold")
            self.assertGreater(result["reflected"], 0.95)
