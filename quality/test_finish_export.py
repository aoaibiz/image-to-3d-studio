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

    def test_reject_bad_master(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.glb"
            path.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                embed_master(path, Path(tmp) / "out.gltf")


if __name__ == "__main__":
    unittest.main()
