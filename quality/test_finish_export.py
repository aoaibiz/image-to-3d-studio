import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from finish_export import embed_master, export_finished


class ExportTest(unittest.TestCase):
    def test_obj_and_ply_preserve_corner_shading_for_mixed_instances(self):
        scene = trimesh.Scene()
        expected = {}
        for i in range(2):
            mesh = trimesh.Trimesh(
                [[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]], [[0, 1, 2]], process=False
            )
            provided = np.array([[0.3, 0, 1], [0, 0.4, 1], [-0.2, 0.4, 1]])
            provided /= np.linalg.norm(provided, axis=1, keepdims=True)
            mesh.vertex_normals = provided
            if not i:
                mesh.visual = trimesh.visual.TextureVisuals(
                    uv=mesh.vertices[:, :2],
                    material=trimesh.visual.material.PBRMaterial(
                        baseColorTexture=Image.new("RGB", (2, 2), "red")
                    ),
                )
            else:
                mesh.visual.vertex_colors = [60, 120, 180, 255]
            matrix = trimesh.transformations.rotation_matrix(0.44, [0, 1, 0]) @ np.diag(
                [1.1, 0.85, 1.3, 1]
            )
            matrix[0, 3] = i * 3.0
            scene.add_geometry(mesh, transform=matrix)
            positions = trimesh.transform_points(mesh.vertices, matrix)
            normals = provided @ np.linalg.inv(matrix[:3, :3])
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            for pos, normal in zip(positions, normals):
                expected[tuple(np.round(pos, 5))] = normal
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            export_finished(scene, folder)
            text = (folder / "editable.obj").read_text().splitlines()
            verts = [
                list(map(float, line.split()[1:4]))
                for line in text
                if line.startswith("v ")
            ]
            normals = np.array(
                [
                    list(map(float, line.split()[1:]))
                    for line in text
                    if line.startswith("vn ")
                ]
            )
            faces = [line.split()[1:] for line in text if line.startswith("f ")]
            self.assertTrue(any("//" in face[0] for face in faces))
            self.assertTrue(any("//" not in face[0] for face in faces))
            for face in faces:
                for corner in face:
                    v, _, n = corner.split("/")
                    np.testing.assert_allclose(
                        normals[int(n) - 1],
                        expected[tuple(np.round(verts[int(v) - 1], 5))],
                        atol=1e-6,
                    )
            ply_bytes = (folder / "editable.ply").read_bytes()
            header = ply_bytes.split(b"end_header", 1)[0]
            for axis in (b"nx", b"ny", b"nz"):
                self.assertIn(b"property float " + axis, header)
            ply = trimesh.load(folder / "editable.ply", process=False)
            for pos, normal in zip(ply.vertices, ply.vertex_normals):
                np.testing.assert_allclose(
                    normal, expected[tuple(np.round(pos, 5))], atol=1e-6
                )

    def test_round_trip_multimaterial_and_colors(self):
        scene = trimesh.Scene()
        for i, color in enumerate(("red", "blue")):
            mesh = trimesh.creation.box()
            mesh.apply_translation((i * 2, 0, 0))
            mesh.visual = trimesh.visual.texture.TextureVisuals(
                uv=np.tile([0.25, 0.75], (len(mesh.vertices), 1)),
                material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=Image.new("RGB", (2, 2), color)
                ),
            )
            scene.add_geometry(mesh, geom_name=f"textured{i}")
        mesh = trimesh.creation.box()
        mesh.apply_translation((4, 0, 0))
        mesh.visual.vertex_colors = [12, 99, 20, 255]
        scene.add_geometry(mesh, geom_name="vertex")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            export_finished(scene, out)
            for f in (
                "model.glb",
                "editable.gltf",
                "editable.obj",
                "editable.ply",
                "material.mtl",
                "material_0.png",
                "material_1.png",
                "texture.png",
            ):
                self.assertTrue((out / f).is_file(), f)
            for f in ("model.glb", "editable.gltf", "editable.obj"):
                result = trimesh.load(out / f, force="scene", process=False)
                self.assertEqual(
                    sum(len(m.faces) for m in result.geometry.values()), 36, f
                )
                self.assertEqual(len(result.geometry), 3, f)
            restored = trimesh.load(out / "editable.gltf", force="scene", process=False)
            textures = [
                g.visual
                for g in restored.geometry.values()
                if g.visual.kind == "texture"
            ]
            self.assertEqual(len(textures), 2)
            self.assertTrue(all(np.allclose(t.uv, [0.25, 0.75]) for t in textures))
            obj = (out / "editable.obj").read_text()
            mtl = (out / "material.mtl").read_text()
            self.assertIn("vt 0.25 0.75", obj)
            self.assertIn("map_Kd material_1.png", mtl)
            self.assertEqual(
                Image.open(out / "material_0.png").convert("RGB").getpixel((0, 0)),
                (255, 0, 0),
            )
            self.assertEqual(
                Image.open(out / "material_1.png").convert("RGB").getpixel((0, 0)),
                (0, 0, 255),
            )
            ply = trimesh.load(out / "editable.ply", process=False)
            self.assertEqual(len(ply.faces), 36)
            self.assertTrue(
                any(
                    np.array_equal(c[:3], [12, 99, 20])
                    for c in ply.visual.vertex_colors
                )
            )
            embedded = json.loads((out / "editable.gltf").read_text())
            self.assertTrue(embedded["buffers"][0]["uri"].startswith("data:"))

    def test_factor_only_glass_without_uv(self):
        mesh = trimesh.creation.box()
        mesh.visual = trimesh.visual.texture.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                name="glass",
                baseColorFactor=[128, 255, 255, 64],
                alphaMode="BLEND",
                roughnessFactor=0.12,
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            export_finished(mesh, folder)
            self.assertIsNone(mesh.visual.uv)
            doc = json.loads((folder / "editable.gltf").read_text())
            index = doc["meshes"][0]["primitives"][0]["material"]
            self.assertEqual(doc["materials"][index]["alphaMode"], "BLEND")
            self.assertAlmostEqual(
                doc["materials"][index]["pbrMetallicRoughness"]["baseColorFactor"][3],
                64 / 255,
            )
            self.assertIn("d 0.250980392", (folder / "material.mtl").read_text())
            with Image.open(folder / "material_0.png") as image:
                self.assertEqual(image.getpixel((0, 0)), (188, 255, 255))
            ply = trimesh.load(folder / "editable.ply", process=False)
            np.testing.assert_equal(ply.visual.vertex_colors[0], [188, 255, 255, 64])

    def test_opacity_map_and_linear_color_factor(self):
        image = Image.new("RGBA", (2, 1))
        image.putdata([(188, 100, 200, 255), (188, 100, 200, 0)])
        for mode, opacity in (
            ("OPAQUE", [255, 255]),
            ("BLEND", [128, 0]),
            ("MASK", [255, 0]),
        ):
            mesh = trimesh.creation.box()
            uv = np.resize([[0.25, 0.0], [1.75, 0.0]], (len(mesh.vertices), 2))
            mesh.visual = trimesh.visual.texture.TextureVisuals(
                uv=uv,
                material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=image,
                    baseColorFactor=[128, 255, 255, 128],
                    alphaMode=mode,
                    alphaCutoff=0.4,
                ),
            )
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                export_finished(mesh, folder)
                mtl = (folder / "material.mtl").read_text()
                ply = trimesh.load(folder / "editable.ply", process=False)
                self.assertEqual(
                    sorted(set(ply.visual.vertex_colors[:, 3])), sorted(set(opacity))
                )
                np.testing.assert_equal(
                    ply.visual.vertex_colors[0, :3], [138, 100, 200]
                )
                if mode == "OPAQUE":
                    self.assertNotIn("map_d ", mtl)
                else:
                    self.assertIn("map_d material_1.png", mtl)
                    with Image.open(folder / "material_1.png") as alpha:
                        self.assertEqual(list(alpha.getdata()), opacity)

    def test_tangent_frame_preserves_handedness_and_normal_map(self):
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2]], process=False)
        mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=vertices[:, :2].copy(),
            material=trimesh.visual.material.PBRMaterial(
                normalTexture=Image.new("RGB", (2, 2), (128, 128, 255)),
                metallicRoughnessTexture=Image.new("RGB", (2, 2), (0, 155, 0)),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            export_finished(mesh, folder)
            doc = json.loads((folder / "editable.gltf").read_text())
            prim = doc["meshes"][0]["primitives"][0]
            mat = doc["materials"][prim["material"]]
            self.assertIn("normalTexture", mat)
            self.assertIn("metallicRoughnessTexture", mat["pbrMetallicRoughness"])
            acc = doc["accessors"][prim["attributes"]["TANGENT"]]
            view = doc["bufferViews"][acc["bufferView"]]
            data = base64.b64decode(doc["buffers"][0]["uri"].split(",", 1)[1])
            frames = np.frombuffer(
                data,
                dtype="<f4",
                count=acc["count"] * 4,
                offset=view.get("byteOffset", 0) + acc.get("byteOffset", 0),
            ).reshape(-1, 4)
            np.testing.assert_allclose(frames, np.tile([1, 0, 0, -1], (3, 1)))
            self.assertNotIn("TANGENT", mesh.vertex_attributes)

    def test_reject_bad_master(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.glb"
            path.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                embed_master(path, Path(tmp) / "out.gltf")


if __name__ == "__main__":
    unittest.main()
