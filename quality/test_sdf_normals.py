import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from skimage.measure import marching_cubes
import trimesh

from engine.sdf_normals import sdf_vertex_normals, shade_glb


def samples(n):
    v = np.linspace(-1.005, 1.005, n, dtype="float32")
    return np.meshgrid(v, v, v, indexing="ij", sparse=True)


def surface(field):
    v, f, _, _ = marching_cubes(field, 0, spacing=(2.01 / (len(field) - 1),) * 3)
    mesh = trimesh.Trimesh(v - 1.005, f, process=True)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def glb(path):
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20 : 20 + n]), data[28 + n :]


class SdfNormalsTests(unittest.TestCase):
    def test_sphere_normal_error_improves_and_works_with_reversed_sdf(self):
        x, y, z = samples(45)
        field = np.sqrt(x * x + y * y + z * z) - 0.72
        mesh = surface(field)
        before = mesh.vertex_normals.copy()
        exact = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
        for grid in (field, -field):
            normals, info = sdf_vertex_normals(mesh, grid)
            self.assertGreater(info["adjusted"] / len(normals), 0.9)
            self.assertLess(
                np.mean(np.linalg.norm(normals - exact, axis=1)),
                np.mean(np.linalg.norm(before - exact, axis=1)) * 0.5,
            )
            self.assertTrue(np.allclose(np.linalg.norm(normals, axis=1), 1))
        np.testing.assert_array_equal(mesh.vertex_normals, before)

    def test_sharp_box_vertices_leave_existing_normals_untouched(self):
        axes = samples(65)
        field = np.maximum(np.maximum(abs(axes[0]), abs(axes[1])), abs(axes[2])) - 0.5
        mesh = trimesh.creation.box(extents=(1, 1, 1))
        normals, info = sdf_vertex_normals(mesh, field)
        self.assertEqual(info["hard_edge_or_boundary"], len(mesh.vertices))
        np.testing.assert_allclose(normals, mesh.vertex_normals, atol=1e-7)

    def test_thin_rod_keeps_normals_across_its_opposite_surfaces(self):
        x, y, z = samples(65)
        field = np.maximum(np.sqrt(x * x + z * z) - 0.038, abs(y) - 0.7)
        mesh = surface(field)
        normals, info = sdf_vertex_normals(mesh, field)
        mid = abs(mesh.vertices[:, 1]) < 0.55
        self.assertGreater(info["thin_or_ambiguous_crossing"], mid.sum())
        np.testing.assert_allclose(normals[mid], mesh.vertex_normals[mid], atol=1e-6)

    def test_insufficient_gradient_falls_back_to_incoming_normals(self):
        x, y, z = samples(32)
        mesh = surface(np.sqrt(x * x + y * y + z * z) - 0.6)
        normals, info = sdf_vertex_normals(
            mesh, np.zeros((32, 32, 32), dtype="float32")
        )
        self.assertEqual(info["adjusted"], 0)
        self.assertEqual(info["weak_gradient"], len(mesh.vertices))
        np.testing.assert_allclose(normals, mesh.vertex_normals, atol=1e-6)

    def test_glb_appends_only_normals_and_preserves_master_and_winding(self):
        x, y, z = samples(28)
        field = np.sqrt(x * x + y * y + z * z) - 0.6
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder)
            np.save(d / "field.npy", field)
            source = d / "master.glb"
            target = d / "light.glb"
            encoded = surface(field).export(file_type="glb")
            source.write_bytes(encoded)
            shade_glb(source, d / "field.npy", target)
            raw, a = glb(source)
            new, b = glb(target)
            self.assertEqual(source.read_bytes(), encoded)
            self.assertEqual(a, b[: len(a)])
            old_prim = raw["meshes"][0]["primitives"][0]
            new_prim = new["meshes"][0]["primitives"][0]
            self.assertEqual(
                old_prim["attributes"]["POSITION"], new_prim["attributes"]["POSITION"]
            )
            self.assertEqual(old_prim["indices"], new_prim["indices"])
            self.assertEqual(raw["nodes"], new["nodes"])
            self.assertEqual(raw["accessors"], new["accessors"][:-1])
            with self.assertRaises(FileExistsError):
                shade_glb(source, d / "field.npy", target)
