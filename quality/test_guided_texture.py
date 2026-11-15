"""Geometry-guided detail must preserve source-facing color and native mesh data."""

import base64
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw
import trimesh

from engine import guided_texture as guided


def _glb_parts(path):
    data = path.read_bytes()
    (size,) = struct.unpack_from("<I", data, 12)
    doc = json.loads(data[20 : 20 + size])
    binary = data[28 + size :]
    views = [
        binary[v.get("byteOffset", 0) : v.get("byteOffset", 0) + v["byteLength"]]
        for v in doc["bufferViews"]
    ]
    return doc, views, binary


def _sample_model(path):
    # One chart per triangle, separated by several pixels on all sides.
    base = trimesh.creation.icosphere(subdivisions=1)
    corners = base.faces.reshape(-1)
    v = base.vertices[corners]
    indices = np.arange(len(corners)).reshape(-1, 3)
    uv = []
    for face_id in range(len(indices)):
        left, bottom = (face_id % 10) / 10, (face_id // 10) / 10
        uv.extend(
            [
                (left + 0.01, bottom + 0.01),
                (left + 0.09, bottom + 0.01),
                (left + 0.05, bottom + 0.09),
            ]
        )
    atlas = Image.new("RGB", (256, 256), (80, 173, 155))
    mesh = trimesh.Trimesh(
        vertices=v * 1.5,
        faces=indices,
        vertex_normals=v,
        process=False,
        visual=trimesh.visual.TextureVisuals(
            uv=uv,
            material=trimesh.visual.material.PBRMaterial(
                name="Test material",
                baseColorTexture=atlas,
                roughnessFactor=0.55,
                metallicFactor=0,
            ),
        ),
    )
    path.write_bytes(
        trimesh.exchange.gltf.export_glb(trimesh.Scene(mesh), include_normals=True)
    )
    return mesh


class GuidedTextureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name)
        cls.model = cls.base / "source.glb"
        mesh = _sample_model(cls.model)
        cls.input_sha = hashlib.sha256(cls.model.read_bytes()).hexdigest()
        cls.guides = cls.base / "guides"
        cls.manifest = guided.prepare_guides(cls.model, cls.guides)
        cls.sheet = cls.base / "inferred.png"
        with Image.open(cls.guides / "guides.png") as src:
            gray = np.asarray(src)[:, :, 0]
        yy, xx = np.indices(gray.shape)
        image = np.full((*gray.shape, 3), 255, np.uint8)
        pigment = np.where((xx + yy) % 32 < 10, 70, 155).astype(np.uint8)
        image[gray < 250] = np.stack((pigment, pigment // 2, pigment), axis=2)[
            gray < 250
        ]
        Image.fromarray(image).save(cls.sheet)
        cls.foreground = cls.base / "foreground.png"
        alpha = guided._view(mesh, (0, 0))[3] >= 0
        cutout = Image.new("RGBA", (guided.SIZE, guided.SIZE), (80, 173, 155, 0))
        cutout.putalpha(Image.fromarray(alpha.astype(np.uint8) * 255))
        cutout.save(cls.foreground)
        cls.frame = {
            "image_right": [1, 0, 0],
            "image_up": [0, 1, 0],
            "camera_direction_from_origin": [0, 0, 1],
            "pinhole": {
                "inverse_distance": 0.05,
                "magnification": 1,
                "shift_fraction": [0, 0],
            },
        }

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_complete_export_preserves_exact_geometry_visible_pigment_and_masters(self):
        out = self.base / "complete"
        result = guided.apply_guides(
            self.model, self.foreground, self.frame, self.guides, self.sheet, out
        )
        self.assertEqual(result["input_sha256"], self.input_sha)
        self.assertEqual(
            hashlib.sha256(self.model.read_bytes()).hexdigest(), self.input_sha
        )
        self.assertGreater(result["modified_surface_texels"], 20)
        self.assertEqual(result["appearance"], "needs_review")
        self.assertEqual(result["locked_bytes_changed"], 0)
        self.assertEqual(json.loads((out / "guided-paint.json").read_text()), result)
        src, src_views, _ = _glb_parts(self.model)
        dst, dst_views, bin_chunk = _glb_parts(out / "model.glb")
        self.assertEqual(src["meshes"], dst["meshes"])
        self.assertEqual(src["accessors"], dst["accessors"])
        self.assertEqual(src["nodes"], dst["nodes"])
        self.assertEqual(src["materials"], dst["materials"])
        image_id = dst["images"][0]["bufferView"]
        for i, chunk in enumerate(src_views):
            if i != image_id:
                self.assertEqual(dst_views[i], chunk)
        self.assertEqual(dst_views[image_id], (out / "texture.png").read_bytes())
        master = json.loads((out / "editable.gltf").read_text())
        data = base64.b64decode(master["buffers"][0].pop("uri").split(",")[1])
        self.assertEqual(data, bin_chunk[: len(data)])
        self.assertEqual(master, dst)
        for name in ("editable.ply", "editable.obj", "material.mtl", "material_0.png"):
            self.assertGreater((out / name).stat().st_size, 0)
        self.assertEqual(
            (out / "texture.png").read_bytes(), (out / "material_0.png").read_bytes()
        )
        # Independent probe near the UV barycentre of all front-facing triangles.
        m, _, _, _ = guided._model(self.model)
        xy = m.visual.uv[m.faces[m.triangles_center[:, 2] > 0.25]].mean(axis=1)
        xx, yy = np.rint(xy * [256, -256] + [-0.5, 255.5]).astype(int).T
        pixels = np.array(Image.open(io.BytesIO(dst_views[image_id])))
        self.assertTrue(np.all(pixels[yy, xx] == [80, 173, 155]))

    def test_reject_second_wrong_silhouette_without_partial_export(self):
        with Image.open(self.sheet) as source:
            bad = source.copy()
        draw = ImageDraw.Draw(bad)
        draw.rectangle((768, 0, 1535, 767), fill="white")
        draw.rectangle((845, 220, 1450, 550), fill="purple")
        f = self.base / "wrong.png"
        bad.save(f)
        with self.assertRaisesRegex(ValueError, "silhouette"):
            guided.apply_guides(
                self.model,
                self.foreground,
                self.frame,
                self.guides,
                f,
                self.base / "rejected",
            )
        self.assertFalse((self.base / "rejected").exists())

    def test_opening_loss_or_invention_rejected_even_at_high_global_iou(self):
        mask = np.zeros((768, 768), bool)
        mask[80:685, 115:650] = True
        holed = mask.copy()
        holed[220:255, 220:260] = False
        self.assertGreater(guided._iou(mask, holed), 0.99)
        for first, second in ((mask, holed), (holed, mask)):
            panel = Image.fromarray(
                np.where(second[:, :, None], [170, 80, 90], [255, 255, 255]).astype(
                    np.uint8
                )
            )
            with self.assertRaisesRegex(ValueError, "opening agreement"):
                guided._register_panel(panel, first)

    def test_disconnected_piece_survives_registration(self):
        mask = np.zeros((768, 768), bool)
        mask[60:450, 90:560] = True
        mask[560:675, 320:535] = True
        panel = Image.fromarray(
            np.where(mask[:, :, None], [150, 85, 90], [255, 255, 255]).astype(np.uint8)
        )
        _, border, report = guided._register_panel(panel, mask)
        self.assertEqual(report["result_iou"], 1)
        self.assertGreater(border[615, 400], 10)

    def test_digest_bound_guide_and_no_existing_output_overwrite(self):
        with self.assertRaisesRegex(ValueError, "empty dedicated"):
            guided.prepare_guides(self.model, self.guides)
        swap = self.base / "tampered"
        swap.mkdir()
        source = Image.open(self.guides / "guides.png")
        source.putpixel((100, 100), (128, 200, 66))
        source.save(swap / "guides.png")
        manifest = dict(
            self.manifest,
            guide_sha256=hashlib.sha256((swap / "guides.png").read_bytes()).hexdigest(),
        )
        (swap / "guide.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "guide pixels"):
            guided.apply_guides(
                self.model,
                self.foreground,
                self.frame,
                swap,
                self.sheet,
                self.base / "tampered-out",
            )
        self.assertFalse((self.base / "tampered-out").exists())

    def test_resource_input_limits_and_invalid_frame(self):
        with self.assertRaisesRegex(ValueError, "bounded"):
            guided._single(self.model, 5)
        with self.assertRaisesRegex(ValueError, "pixel count"):
            guided._image(self.sheet.read_bytes(), 32)
        invalid = dict(self.frame, image_right=[-1, 0, 0])
        with self.assertRaisesRegex(ValueError, "camera frame"):
            guided.apply_guides(
                self.model,
                self.foreground,
                invalid,
                self.guides,
                self.sheet,
                self.base / "invalid",
            )


if __name__ == "__main__":
    unittest.main()
