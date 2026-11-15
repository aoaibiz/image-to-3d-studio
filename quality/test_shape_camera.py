import unittest

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import trimesh

from engine import shape_camera as camera
from engine.shape_camera import register_camera


def openwork():
    boxes = []
    for size, center in (
        ((0.23, 2.0, 0.34), (-0.75, 0.0, 0.0)),
        ((0.23, 1.82, 0.40), (0.75, -0.09, 0.0)),
        ((1.65, 0.23, 0.42), (0.0, 0.78, 0.0)),
        ((1.70, 0.30, 0.52), (0.0, -0.81, 0.0)),
        ((0.38, 0.50, 1.30), (0.66, 0.38, 0.43)),
        ((0.25, 0.65, 0.24), (-0.62, 1.12, -0.18)),
        ((0.55, 0.38, 0.51), (-0.42, -0.60, -0.42)),
    ):
        box = trimesh.creation.box(extents=size)
        box.apply_translation(center)
        boxes.append(box)
    return trimesh.util.concatenate(boxes)


def reference(mesh, view, inverse_distance=0):
    # Independent supersampled polygon painter: preserve actual enclosed holes.
    v = mesh.vertices @ view.T
    delta = (mesh.vertices - mesh.bounds.mean(0)) @ view.T
    xy = delta[:, :2] / (1 - delta[:, 2:3] * inverse_distance / max(mesh.extents))
    xy *= [1, -1]
    xy = (xy - (xy.min(0) + xy.max(0)) / 2) * (400 / np.ptp(xy, axis=0).max()) + 240
    img = Image.new("RGBA", (480, 480))
    draw = ImageDraw.Draw(img)
    directions = mesh.face_normals @ view.T
    for f in np.argsort(v[mesh.faces, 2].mean(1)):
        shade = int(128 + directions[f] @ np.array([-35, 50, 22]))
        draw.polygon(
            [tuple(x) for x in xy[mesh.faces[f]]], fill=(shade, shade, shade, 255)
        )
    return img.resize((240, 240), Image.Resampling.LANCZOS)


class CameraTests(unittest.TestCase):
    def test_signed_up_openwork_and_proper_display_transform(self):
        mesh = openwork()
        view = Rotation.from_euler("yx", [25, -11], degrees=True).as_matrix()
        source = reference(mesh, view)
        raw = Rotation.from_euler("zy", [92, 29], degrees=True).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = raw
        mesh.apply_transform(transform)
        raw_vertices = mesh.vertices.copy()
        got = register_camera(mesh, source)
        np.testing.assert_array_equal(raw_vertices, mesh.vertices)
        frame = np.array(list(got["camera_frame"].values()))
        np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(frame), 1.0)
        np.testing.assert_allclose(np.array(got["upright_transform"])[:3, :3], frame)
        self.assertGreater(got["silhouette_iou"], 0.9)
        self.assertLess(got["filled_source_hole_fraction"], 0.1)
        # Both camera direction and image up must recover the signed orientation.
        expected = view @ raw.T
        self.assertGreater(np.min(np.diag(frame @ expected.T)), 0.94)
        self.assertFalse(got["hidden_sides_observed"])

    def test_ambiguous_cube_has_no_verified_pose(self):
        cube = trimesh.creation.box()
        got = register_camera(cube, reference(cube, np.eye(3)))
        self.assertEqual(got["status"], "ambiguous")
        self.assertFalse(got["photographic_pose_verified"])
        self.assertIsNotNone(got["alternate_rotation_degrees"])

    def test_general_pinhole_refines_single_photo_with_enclosed_holes(self):
        mesh = openwork()
        params = (2, 32, 17, 4)
        truth = camera._basis(*params)
        source = reference(mesh, truth, inverse_distance=0.72)
        vertices = (mesh.vertices - mesh.bounds.mean(0)) / mesh.extents.max()
        ref = camera._reference(camera._matte(source)[0], 112)
        seed = (2, 29, 21, 2)
        basis = camera._basis(*seed)
        initial = camera._score(
            np.isfinite(camera._depth(vertices, mesh.faces, basis, ref)), ref
        )[1]
        fit = camera._refine_lens(vertices, mesh.faces, ref, [(0, seed, basis)])[0]
        self.assertGreater(fit[3], initial + 0.05)
        self.assertGreater(fit[3], 0.85)
        self.assertGreater(fit[6][0], 0.1)
        self.assertFalse(np.any(np.abs(fit[6][2:]) > 0.055))
        # A fit is evidence for a view only. Even egregious lens parameters
        # from a manually supplied frame must not create a projection.
        with self.assertRaisesRegex(ValueError, "projection"):
            camera.lens_parameters(
                dict(
                    pinhole=dict(
                        inverse_distance=3, magnification=1, shift_fraction=[0, 0]
                    )
                )
            )

    def test_unsegmented_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "foreground"):
            register_camera(openwork(), Image.new("RGB", (80, 80), "white"))


if __name__ == "__main__":
    unittest.main()
