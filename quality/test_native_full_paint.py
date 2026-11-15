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


def fixture(dirpath, sheet_base=(70, 92, 128)):
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
        rgb[valid] = sheet_base
        pigment = valid & (
            ((x - gt.SIZE * 0.48) ** 2 + (y - gt.SIZE * 0.55) ** 2)
            < (gt.SIZE * 0.09) ** 2
        )
        rgb[pigment] = [63, 77, 106]
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
    def test_new_visible_color_break_on_matched_silhouettes_holds_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp), sheet_base=(140, 104, 75))
            with self.assertRaisesRegex(paint.NativeFullReviewHold, "塗装"):
                paint.paint_native_full(*args)
            self.assertFalse((args[-1] / "model.glb").exists())

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
            self.assertEqual(
                sum(result["single_view_uv"].values()), result["supported_uv"]
            )
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

    def test_symmetric_focal_face_is_distinct_from_plain_symmetric_material(self):
        from PIL import ImageDraw

        mesh = trimesh.creation.box(extents=(1, 1, 0.4))
        mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.vertices[:, :2] + 0.5)
        tex = Image.new("RGB", (256, 256), "tan")
        d = ImageDraw.Draw(tex)
        for x in (83, 172):
            d.ellipse((x - 12, 85, x + 12, 112), fill="#202020")
            d.ellipse((x - 9, 88, x - 4, 94), fill="white")
        d.polygon([(115, 130), (140, 130), (127, 142)], fill="#202020")
        d.arc((98, 126, 128, 153), 0, 125, fill="black", width=3)
        d.arc((126, 126, 156, 153), 55, 180, fill="black", width=3)
        for y in (130, 140, 150):
            d.line((30, y - 8, 82, y + 1), fill="black", width=3)
            d.line((225, y - 8, 173, y + 1), fill="black", width=3)
        focal = {
            "observed_face_cues": {
                "focal": {"family": "character", "location": "center"}
            }
        }
        self.assertEqual(
            paint._copy_check(mesh, np.asarray(tex), focal)["status"], "review_hold"
        )
        flat = np.full((256, 256, 3), 135, np.uint8)
        self.assertEqual(paint._copy_check(mesh, flat, {})["status"], "inconclusive")

    def test_supported_paint_replaces_prior_ink_instead_of_multiplying_it(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp))
            _, _, _, mesh = native._native(args[0])
            y, x = np.indices((256, 256))
            base = np.where(
                (((x // 6 + y // 4) % 2) == 0)[:, :, None], [154, 37, 56], [39, 76, 154]
            ).astype(np.uint8)
            mesh.visual.material.baseColorTexture = Image.fromarray(base)
            mask = gt._matte(args[3], gt.SIZE)[:, :, 3] >= 150

            def single_view(_mesh, _sheet, points, _normals):
                field = np.tile([95, 140, 195], (len(points), 1))
                return (
                    [field, 255 - field],
                    [np.ones(len(field)), np.zeros(len(field))],
                    {},
                )

            with patch.object(paint, "_field", side_effect=single_view):
                result, record = paint._project(
                    mesh, base, mask, args[4]["camera_frame"], None
                )
            _, _, _, _, yy, xx = native.atlas(mesh)
            changes = (result[yy, xx] != base[yy, xx]).any(axis=1)
            self.assertGreater(changes.sum(), 500)
            self.assertTrue((result[yy[changes], xx[changes]] == [95, 140, 195]).all())
            self.assertGreaterEqual(len(record["paint_join_screen"]["angles_with_new_edges"]), 2)
            self.assertEqual(record["locked_changed"], 0)
            self.assertEqual(record["unsupported_surface_changed"], 0)

    def test_roll_and_uniform_scale_require_independent_interior_landmarks(self):
        import cv2

        n = gt.SIZE
        y, x = np.indices((n, n))
        v = (y - 0.15 * n) / (0.7 * n)
        width = n * (0.13 + 0.065 * np.sin(1.45 * np.pi * v) ** 2 + 0.032 * v)
        mask = (abs(x - 0.5 * n) < width) & (v > 0) & (v < 1)
        depth = (
            0.8
            + 0.11 * np.tanh((v - 0.28) / 0.008)
            - 0.22 * np.tanh((v - 0.67) / 0.008)
        )
        z = np.where(mask, depth, -np.inf)
        rgb = np.full((n, n, 3), 254, np.uint8)
        ink = np.where((v > 0.28) & (v < 0.67), 125, 80)
        for c, offset in enumerate([24, 40, 72]):
            rgb[:, :, c][mask] = ink[mask] + offset
        M = cv2.getRotationMatrix2D((n / 2, n / 2), 3.1, 1.15)

        def camera(image):
            warped = cv2.warpAffine(image, M, (n, n), borderValue=(254, 254, 254))
            return Image.fromarray(warped)

        with self.assertRaises(ValueError):
            gt._register_panel(camera(rgb), mask)
        _, _, info = paint._calibrate_panel(camera(rgb), mask, z)
        self.assertGreater(info["result_iou"], 0.97)
        self.assertLess(abs(info["roll_degrees"] + 3.1), 0.7)
        rgb[mask] = [125, 135, 120]
        with self.assertRaisesRegex(ValueError, "landmarks"):
            paint._calibrate_panel(camera(rgb), mask, z)
