import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from engine import shape_appearance as paint


class ShapeAppearanceTests(unittest.TestCase):
    def test_pinhole_depth_is_affine_in_screen_space_even_off_origin(self):
        mesh = trimesh.creation.box(extents=[2, 3, 1])
        mesh.apply_translation([7, -11, 4])
        basis = np.eye(3)
        params = np.array([0.68, 1.03, -0.02, 0.015])
        mask = np.full((96, 96), False)
        mask[10:89, 12:82] = True
        projected, scale, offset, *_ = paint._fit(mesh.vertices, basis, mask, params)
        center, inv = paint._lens_coordinates(mesh.vertices, basis, params)
        ids = [0, 2, 7]
        bary = np.array([0.24, 0.35, 0.41])
        original = bary @ mesh.vertices[ids]
        sample = paint.project_pinhole(original[None, :], center, inv)[0]
        w = 1 - (mesh.vertices[ids, 2] - center[2]) * inv
        screen_bary = bary * w / (bary @ w)
        interpolated = screen_bary @ projected[ids]
        np.testing.assert_allclose(
            interpolated[:2], sample[:2] * [scale, -scale] + offset
        )
        self.assertAlmostEqual(interpolated[2], sample[2])

    def test_native_sdf_only_changes_lighting_normals(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
            mesh.vertex_normals = paint._norm(mesh.vertex_normals + [0.15, -0.08, 0])
            raw = d / "raw.glb"
            raw.write_bytes(
                trimesh.exchange.gltf.export_glb(mesh.scene(), include_normals=True)
            )
            axis = np.linspace(-1.005, 1.005, 33, dtype=np.float32)
            q = np.meshgrid(axis, axis, axis, indexing="ij", sparse=True)
            sdf = d / "field.npy"
            np.save(sdf, np.sqrt(sum(c * c for c in q)) - 0.7)
            source = d / "photo.png"
            matte = Image.new("RGBA", (160, 160))
            ImageDraw.Draw(matte).ellipse((14, 14, 145, 145), fill=(188, 93, 32, 255))
            ImageDraw.Draw(matte).ellipse((68, 44, 81, 58), fill=(41, 32, 26, 255))
            matte.save(source)
            old = paint.paint_shape(
                raw, source, source, d / "usual", texture_size=256, camera_hint="+Z"
            )
            new = paint.paint_shape(
                raw,
                source,
                source,
                d / "polished",
                texture_size=256,
                camera_hint="+Z",
                sdf_field=sdf,
            )
            self.assertNotIn("sdf_shading", old)
            self.assertEqual(new["sdf_shading"]["field_sha256"], paint._sha(sdf))
            self.assertEqual(new["sdf_shading"]["raw_shape_sha256"], paint._sha(raw))
            self.assertEqual(
                (d / "polished/shape-master.glb").read_bytes(), raw.read_bytes()
            )
            self.assertEqual(
                (d / "polished/texture.png").read_bytes(),
                (d / "usual/texture.png").read_bytes(),
            )
            a = paint._load(d / "usual/model.glb")[0]
            b = paint._load(d / "polished/model.glb")[0]
            np.testing.assert_equal(a.triangles, b.triangles)
            self.assertGreater(
                np.linalg.norm(
                    a.vertex_normals[a.faces] - b.vertex_normals[b.faces], axis=2
                ).max(),
                0.035,
            )

    def test_touching_materials_and_entirely_hidden_part(self):
        parts = []
        for x, z in ((-0.5, 0), (0.5, 0), (-0.5, -1)):
            box = trimesh.creation.box(extents=[1, 1, 0.6])
            box.apply_translation([x, 0, z])
            parts.append(box)
        mesh = trimesh.util.concatenate(parts)
        labels = np.repeat(np.arange(3), [len(m.faces) for m in parts])
        proj = mesh.vertices.copy()
        proj[:, :2] = proj[:, :2] * [52, -52] + 62
        fid, _, _ = paint._raster(proj, mesh.faces, 128)
        mask = fid >= 0
        rgba = np.zeros((128, 128, 4), np.uint8)
        rgba[mask, :3] = np.array([[192, 74, 35], [30, 94, 193], [0, 255, 0]])[
            labels[fid[mask]]
        ]
        rgba[mask, 3] = 255
        # Deliberately supplied as one mesh: coincident contact edges must not
        # make separately indexed box shells into one material component.
        result, info = paint._surface_material(
            mesh,
            np.zeros(len(mesh.faces), int),
            np.eye(3),
            fid,
            rgba,
            paint.ndimage.distance_transform_edt(mask),
        )
        rgb = paint._srgb(result).mean(axis=1)
        back = mesh.face_normals[:, 2] < -0.9
        self.assertTrue(np.all(rgb[(labels == 0) & back, 0] > 175))
        self.assertTrue(np.all(rgb[(labels == 1) & back, 2] > 175))
        self.assertTrue(np.allclose(rgb[labels == 2], 160, atol=1))
        self.assertEqual(info["components"], 3)
        self.assertEqual(info["components_with_photo_seeds"], 2)
        self.assertEqual(info["faces_without_component_evidence"], len(parts[2].faces))

    def test_side_by_side_materials_on_one_connected_surface(self):
        mesh = (
            trimesh.creation.box(extents=[2, 1, 0.28])
            .subdivide()
            .subdivide()
            .subdivide()
        )
        proj = mesh.vertices.copy()
        proj[:, :2] = proj[:, :2] * [52, -52] + 62
        ids, _, _ = paint._raster(proj, mesh.faces, 128)
        mask = ids >= 0
        rgba = np.zeros((128, 128, 4), np.uint8)
        rgba[mask, 3] = 255
        rgba[mask, :3] = [37, 151, 71]
        rgba[:, 63:, :3] = [194, 97, 22]
        result, info = paint._surface_material(
            mesh,
            np.zeros(len(mesh.faces), int),
            np.eye(3),
            ids,
            rgba,
            paint.ndimage.distance_transform_edt(mask),
        )
        back = mesh.face_normals[:, 2] < -0.9
        rgb = paint._srgb(result).mean(axis=1)
        left = rgb[back & (mesh.triangles_center[:, 0] < -0.45)]
        right = rgb[back & (mesh.triangles_center[:, 0] > 0.45)]
        self.assertTrue((left[:, 1] > left[:, 0]).mean() > 0.9)
        self.assertTrue((right[:, 0] > right[:, 1]).mean() > 0.9)
        self.assertEqual(info["components"], 1)

    def test_zbuffer_keeps_occluded_face_and_empty_pixels_out(self):
        vertices = np.array(
            [[1, 1, 2], [7, 1, 2], [1, 7, 2], [1, 1, -2], [7, 1, -2], [1, 7, -2]], float
        )
        ids, _, depth = paint._raster(vertices, np.array([[0, 1, 2], [3, 4, 5]]), 10)
        self.assertEqual(ids[2, 2], 0)
        self.assertEqual(depth[2, 2], 2)
        self.assertEqual(ids[8, 8], -1)

    def test_explicit_hard_normals_survive_uv_and_instance_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            shape = directory / "hard-shaded.glb"
            box = trimesh.creation.box([1, 1, 0.8])
            box.unmerge_vertices()
            # Authored directions deliberately differ from normals that a
            # mesh library can recover just from the disconnected triangles.
            authored = paint._norm(box.vertex_normals + [0.19, 0.09, 0.02])
            box.vertex_normals = authored
            node = np.diag([1.0, 1.0, 0.72, 1.0])
            scene = trimesh.Scene()
            scene.add_geometry(box, transform=node)
            shape.write_bytes(
                trimesh.exchange.gltf.export_glb(scene, include_normals=True)
            )
            with self.assertRaisesRegex(ValueError, "native untransformed"):
                paint._load(shape, require_native=True)
            matte = Image.new("RGBA", (128, 128))
            ImageDraw.Draw(matte).rectangle((14, 14, 113, 113), fill=(31, 84, 120, 255))
            matte.save(directory / "matte.png")
            rotation = trimesh.transformations.rotation_matrix(0.47, [1, 1, 0])
            rotation[:3, 3] = [0.2, 0.3, -0.2]
            paint.paint_shape(
                shape,
                directory / "matte.png",
                directory / "matte.png",
                directory / "out",
                texture_size=256,
                camera_hint="+Z",
                display_from_raw=rotation,
            )
            # Read geometry straight from the output GLB. Comparing two
            # outputs of _load would mask the same normal loss on both sides.
            result = next(
                iter(
                    trimesh.load(
                        directory / "out/model.glb", force="scene", process=False
                    ).geometry.values()
                )
            )
            combined = rotation @ node
            points = trimesh.transform_points(box.vertices, combined)[box.faces]
            normals = paint._norm(authored @ np.linalg.inv(combined[:3, :3]))[box.faces]
            dist, tid = paint.cKDTree(points.mean(axis=1)).query(
                result.triangles_center
            )
            self.assertLess(dist.max(), 1e-6)
            order = np.linalg.norm(
                result.triangles[:, :, None] - points[tid][:, None, :], axis=3
            ).argmin(2)
            np.testing.assert_allclose(
                result.vertex_normals[result.faces],
                normals[tid[:, None], order],
                atol=1e-6,
            )

    def test_hole_source_front_and_inferred_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            shape = d / "shape.glb"
            ring = trimesh.creation.annulus(0.38, 1, 0.45, sections=64)
            ring.export(shape)
            shape_hash = paint._sha(shape)
            rgba = Image.new("RGBA", (192, 192), (0, 0, 255, 0))
            draw = ImageDraw.Draw(rgba)
            draw.ellipse((16, 16, 176, 176), fill=(30, 180, 55, 255))
            draw.rectangle((34, 67, 54, 122), fill=(245, 30, 20, 255))
            draw.ellipse((66, 66, 126, 126), fill=(0, 0, 255, 0))
            rgba.save(d / "matte.png")
            source = Image.new("RGB", rgba.size, "white")
            source.paste(rgba, mask=rgba.getchannel("A"))
            source.save(d / "photo.png")
            out = d / "result"
            stats = paint.paint_shape(
                shape,
                d / "photo.png",
                d / "matte.png",
                out,
                texture_size=256,
                camera_hint="+Z",
            )
            self.assertEqual(stats["source_faces"], stats["exported_faces"])
            self.assertEqual(stats["shape_sha256"], shape_hash)
            self.assertEqual(paint._sha(shape), shape_hash)
            self.assertEqual(paint._sha(out / "shape-master.glb"), shape_hash)
            mesh = paint._load(out / "model.glb")[0]
            welded = mesh.copy()
            welded.merge_vertices(merge_tex=True, merge_norm=True)
            self.assertEqual(welded.euler_number, ring.euler_number)
            self.assertTrue(np.allclose(welded.bounds, ring.bounds))
            tex = np.array(mesh.visual.material.baseColorTexture.convert("RGB"))
            uv = mesh.visual.uv[mesh.faces].mean(axis=1)
            rgb = paint._sample(
                tex, np.column_stack((uv[:, 0] * 256 - 0.5, (1 - uv[:, 1]) * 256 - 0.5))
            )
            red = (rgb[:, 0] > 150) & (rgb[:, 0] > rgb[:, 1] * 1.5)
            self.assertTrue(np.any(red & (mesh.face_normals[:, 2] > 0.8)))
            self.assertFalse(np.any(red & (mesh.face_normals[:, 2] < -0.8)))
            self.assertFalse(np.any((rgb[:, 2] > 220) & (rgb[:, 0] < 25)))
            reflected = {
                "image_right": [-1, 0, 0],
                "image_up": [0, 1, 0],
                "camera_direction_from_origin": [0, 0, 1],
            }
            with self.assertRaisesRegex(ValueError, "reflection"):
                paint._registration(
                    ring.vertices, ring.faces, np.asarray(rgba), None, "+Y", reflected
                )
            self.assertEqual(stats["appearance"], "needs_review")
            rigid = trimesh.transformations.rotation_matrix(0.63, [0, 1, 0])
            rigid[:3, 3] = [1, -2, 0.4]
            oriented = d / "oriented"
            moved = paint.paint_shape(
                shape,
                d / "photo.png",
                d / "matte.png",
                oriented,
                texture_size=256,
                camera_hint="+Z",
                display_from_raw=rigid,
            )
            np.testing.assert_allclose(moved["display_from_raw"], rigid)
            self.assertEqual(paint._sha(oriented / "shape-master.glb"), shape_hash)
            rotated = paint._load(oriented / "model.glb")[0]
            rotated.apply_transform(np.linalg.inv(rigid))
            tree = paint.cKDTree(ring.triangles_center)
            dist, i = tree.query(rotated.triangles_center)
            self.assertLess(dist.max(), 1e-6)
            np.testing.assert_allclose(
                rotated.face_normals, ring.face_normals[i], atol=1e-5
            )
            _, original_face = paint.cKDTree(mesh.triangles_center).query(
                rotated.triangles_center
            )
            np.testing.assert_allclose(
                rotated.vertex_normals[rotated.faces].mean(axis=1),
                mesh.vertex_normals[mesh.faces[original_face]].mean(axis=1),
                atol=1e-5,
            )
            for invalid in (np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1])):
                with self.assertRaisesRegex(ValueError, "rigid transform"):
                    paint._display_matrix(invalid)
            for name in (
                "editable.obj",
                "editable.gltf",
                "editable.ply",
                "texture.png",
                "appearance.json",
            ):
                self.assertTrue((out / name).is_file())
            with self.assertRaisesRegex(ValueError, "empty dedicated"):
                paint.paint_shape(shape, d / "photo.png", d / "matte.png", out)
            Image.new("RGB", (192, 192), "magenta").save(d / "unrelated.png")
            with self.assertRaisesRegex(ValueError, "original photograph"):
                paint.paint_shape(
                    shape,
                    d / "unrelated.png",
                    d / "matte.png",
                    d / "wrong",
                    texture_size=256,
                )


if __name__ == "__main__":
    unittest.main()
