import unittest

import numpy as np
import trimesh

from engine.surface_fairing import fair_surface


def noisy_tube():
    angle, z = np.meshgrid(np.arange(96) * 2 * np.pi / 96, np.linspace(-0.8, 0.8, 32))
    radius = 1 + 0.009 * np.sin(27 * angle + 7 * z) * np.sin(38 * z)
    points = np.stack(
        (radius * np.cos(angle), radius * np.sin(angle), z), axis=-1
    ).reshape(-1, 3)
    idx = np.arange((len(z) - 1) * 96)
    following = idx // 96 * 96 + (idx + 1) % 96
    faces = np.vstack(
        (
            np.column_stack((idx, following, idx + 96)),
            np.column_stack((following, following + 96, idx + 96)),
        )
    )
    return trimesh.Trimesh(points, faces, process=False)


class SurfaceFairingTests(unittest.TestCase):
    def test_curved_wall_noise_reduced_without_changing_open_rims(self):
        mesh = noisy_tube()
        prior = mesh.vertices.copy()
        vertices, normals, info = fair_surface(mesh)
        self.assertGreater(info["moved_unique_vertices"], 100)
        n = np.asarray(mesh.faces)

        def error(v):
            return np.mean(np.abs(np.linalg.norm(v[:, :2], axis=1) - 1))

        movable = np.linalg.norm(vertices - prior, axis=1) > 1e-9
        self.assertLess(error(vertices[movable]), 0.6 * error(prior[movable]))
        self.assertLess(error(vertices), error(prior))
        np.testing.assert_array_equal(prior, mesh.vertices)
        np.testing.assert_array_equal(vertices[:96], prior[:96])
        np.testing.assert_array_equal(vertices[-96:], prior[-96:])
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)
        self.assertTrue(
            (
                np.einsum(
                    "ij,ij->i",
                    np.cross(
                        vertices[n[:, 1]] - vertices[n[:, 0]],
                        vertices[n[:, 2]] - vertices[n[:, 0]],
                    ),
                    mesh.face_normals,
                )
                > 0
            ).all()
        )

    def test_weld_only_for_fit_indexing_and_features_survive(self):
        mesh = noisy_tube()
        # Duplicate every chart vertex. The output must have exactly the
        # original indexing, with both copies following one physical point.
        faces = mesh.faces.ravel()
        charts = trimesh.Trimesh(
            mesh.vertices[faces], np.arange(len(faces)).reshape(-1, 3), process=False
        )
        result = fair_surface(charts)
        self.assertIsNotNone(result)
        v, _, evidence = result
        order = np.argsort(faces)
        same = np.diff(faces[order]) == 0
        np.testing.assert_allclose(np.diff(v[order], axis=0)[same], 0, atol=1e-12)
        self.assertEqual(evidence["vertex_components"], 1)

    def test_real_crease_and_separate_parts_are_preserved(self):
        x, y = np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-0.7, 0.7, 24))
        # The sloped ridge is not a global bounding extremum; preserving the
        # crest itself requires the explicit dihedral guard.
        z = 0.82 * np.abs(x) + 0.18 * y + 0.008 * np.sin(31 * y) * np.sin(19 * x)
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        ids = (np.arange(23)[:, None] * 41 + np.arange(40)).ravel()
        faces = np.vstack(
            (
                np.column_stack((ids, ids + 1, ids + 41)),
                np.column_stack((ids + 1, ids + 42, ids + 41)),
            )
        )
        fold = trimesh.Trimesh(points, faces, process=False)
        other = fold.copy()
        other.apply_translation([2.6, 0, 0.13])
        assembly = trimesh.util.concatenate([fold, other])
        moved, _, report = fair_surface(assembly)
        self.assertEqual(report["vertex_components"], 2)
        crest = np.flatnonzero(x.ravel() == 0)
        all_crests = np.r_[crest, crest + len(points)]
        np.testing.assert_array_equal(moved[all_crests], assembly.vertices[all_crests])

    def test_coarse_shapes_are_not_smoothed(self):
        self.assertIsNone(fair_surface(trimesh.creation.box()))
