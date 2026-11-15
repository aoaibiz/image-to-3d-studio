"""Independent analytic surfaces and one-off markings exercise inferred pigment transport."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from engine import codex_photo_observations as obs
from engine import continuity_paint as paint
from engine import guided_texture as gt
from engine import native_region_paint as native
from engine.triposg_cpu import atomic_json, digest

REPEAT = {
    "optical_finish_candidate": "matte",
    "layout_candidate": "repeated",
    "observed_face_cues": {"repeated": {"family": "geometric"}},
}
PLAIN = {**REPEAT, "observed_face_cues": {}, "layout_candidate": "plain_or_uncertain"}


def samples(form):
    yy, xx = np.indices((128, 512))
    y, t = (yy.ravel() + 0.5) / 128, (xx.ravel() + 0.5) / 512
    a = t * 2 * np.pi
    sn, cs = np.sin(a), np.cos(a)
    if form == "box":
        # Map each uniformly spaced sample to the four flat faces, with corner
        # at one eighth of the perimeter and front centre at zero.
        face = (np.floor(4 * t + 0.5) % 4).astype(int)
        local = (4 * t + 0.5) % 1 - 0.5
        dirs = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        vec = dirs[face] + 2 * local[:, None] * np.column_stack(
            (dirs[face, 1], -dirs[face, 0])
        )
        sn, cs = vec.T
        nx, nz = dirs[face].T
    else:
        r = 1 + 0.18 * np.cos(y * np.pi)
        if form == "bent":
            # Translated transverse rings, taper and an off-centre upper body.
            sn = r * sn + 0.32 * y**2
        else:
            sn = r * sn
        cs = r * cs
        nx, nz = np.sin(a), np.cos(a)
    xyz = np.column_stack((sn, 2 * y, cs))
    normals = np.column_stack((nx, np.zeros_like(nx), nz))
    phase = 2 * np.pi * 12 * t + 0.62 * np.sin(2 * np.pi * y)
    # Two colours, bands and secondary texture in independently specified arc coordinates.
    freq = np.cos(phase) + 0.30 * np.cos(2 * phase) * np.sin(5 * np.pi * y)
    expected = np.rint(
        np.column_stack((118 + 51 * freq, 140 - 26 * freq, 171 + 39 * freq))
    ).astype(np.uint8)
    seen = np.cos(a) > 0.26
    locked = np.cos(a) > -0.10
    native = expected.copy()
    native[~locked] = [125, 135, 143]
    return xyz, normals, native, expected, seen, locked


class Continuity(unittest.TestCase):
    def test_curved_and_bent_repetition_reaches_real_rear_without_changing_front(self):
        for form in ("curved", "bent"):
            with self.subTest(form=form):
                p, n, old, true, seen, lock = samples(form)
                found = paint._transport(p, n, old, true, seen, lock, REPEAT)
                self.assertIsNotNone(found)
                result, support, rep = found
                self.assertTrue(np.array_equal(result[lock], old[lock]))
                self.assertTrue(np.array_equal(result[~support], old[~support]))
                self.assertGreater((support & ~lock).sum(), (~lock).sum() * 0.9)
                self.assertLess(
                    np.abs(result[support].astype(float) - true[support]).mean(), 3
                )
                self.assertGreater(rep["independent_donor_agreement"], 0.92)

    def test_flat_faces_and_distinctive_wordmark(self):
        p, n, old, true, seen, lock = samples("box")
        # A frontal orthographic observation sees a single flat face in full.
        seen &= n[:, 2] > 0.5
        actual = paint._transport(p, n, old, true, seen, lock, REPEAT)
        self.assertIsNotNone(actual)
        result, support, _ = actual
        self.assertLess(np.abs(result[support].astype(float) - true[support]).mean(), 4)
        self.assertTrue(np.array_equal(result[lock], old[lock]))
        marked = {
            **REPEAT,
            "observed_face_cues": {
                "focal": {"family": "lettering_or_logo"},
                "repeated": {"family": "geometric"},
            },
        }
        self.assertIsNone(paint._transport(p, n, old, true, seen, lock, marked))

    def test_supported_plain_substrate_and_semantic_face_control(self):
        p, n, old, true, seen, lock = samples("bent")
        true[:] = [174, 151, 125]
        old[:] = true
        old[~lock] -= 18
        result, support, rep = paint._transport(p, n, old, true, seen, lock, PLAIN)
        self.assertTrue(np.array_equal(result[support], true[support]))
        self.assertEqual(rep["surface_mode"], "inferred_uniform_substrate")
        mark = (~lock) & (p[:, 1] > 0.7) & (p[:, 1] < 1.15) & (p[:, 0] > 0.1)
        old[mark] = [25, 45, 55]
        self.assertIsNone(paint._transport(p, n, old, true, seen, lock, PLAIN))
        for family in ("face_or_emblem", "lettering_or_logo", "figurative_scene"):
            obs = {**REPEAT, "observed_face_cues": {"focal": {"family": family}}}
            self.assertFalse(paint.eligible(obs))

    def test_unlabelled_nonperiodic_source_and_alternate_face_abstain(self):
        p, n, old, true, seen, lock = samples("curved")
        rng = np.random.default_rng(417)
        photo = rng.integers(45, 212, true.shape, dtype=np.uint8)
        self.assertIsNone(paint._transport(p, n, old, photo, seen, lock, REPEAT))
        ambiguous = {
            **REPEAT,
            "function_context": {
                "interpretation": {"reverse_candidates": ["distinct_working_surface"]}
            },
        }
        self.assertFalse(paint.eligible(ambiguous))


def pbr_fixture(case):
    h, w = 48, 128
    yy, xx = np.indices((h + 1, w + 1))
    theta = xx / w * 2 * np.pi
    elev = yy / h * 2 - 1
    rad = 0.61 + 0.065 * np.sin(2 * elev)
    vertices = np.column_stack(
        ((np.sin(theta) * rad).ravel(), elev.ravel(), (np.cos(theta) * rad).ravel())
    )
    norms = np.column_stack(
        (
            np.sin(theta).ravel(),
            (-0.13 * np.cos(2 * elev)).ravel(),
            np.cos(theta).ravel(),
        )
    )
    uvs = np.column_stack(
        ((0.01 + 0.98 * xx / w).ravel(), (0.05 + 0.90 * yy / h).ravel())
    )
    tri = []
    for y in range(h):
        for x in range(w):
            k = y * (w + 1) + x
            tri.extend([[k, k + 1, k + w + 1], [k + 1, k + w + 2, k + w + 1]])
    mesh = trimesh.Trimesh(
        vertices=vertices, vertex_normals=norms, faces=tri, process=False
    )
    iy, ix = np.indices((512, 512))
    u = (ix / 511 - 0.01) / 0.98
    v = ((1 - iy / 511) - 0.05) / 0.90
    phase = 2 * np.pi * 12 * u + 0.62 * np.sin(2 * np.pi * v)
    f = np.cos(phase) + 0.30 * np.cos(2 * phase) * np.sin(5 * np.pi * v)
    true = np.rint(
        np.stack((118 + 51 * f, 140 - 26 * f, 171 + 39 * f), axis=-1)
    ).astype(np.uint8)
    tex = true.copy()
    tex[np.cos(2 * np.pi * u) < -0.15] = [125, 135, 143]
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(tex),
        roughnessFactor=0.5,
        metallicFactor=0.1,
        normalTexture=Image.new("RGB", (512, 512), (128, 128, 255)),
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=mat)
    nativefile = case / "provider-native.glb"
    nativefile.write_bytes(
        trimesh.exchange.gltf.export_glb(trimesh.Scene(mesh), include_normals=True)
    )
    _, _, _, ids, b, _ = gt._view(mesh, (0, 0))
    valid = ids >= 0
    fc = mesh.faces[ids[valid]]
    bary = b[valid]
    weights = np.column_stack((1 - bary.sum(1), bary))
    uv = np.einsum("ij,ijk->ik", weights, mesh.visual.uv[fc])
    px, py = np.rint(uv * [511, -511] + [0, 511]).astype(int).T
    rgba = np.full((*valid.shape, 4), 255, np.uint8)
    rgba[:, :, 3] = valid * 255
    rgba[:, :, :3][valid] = true[py, px]
    source, fg = case / "photo.png", case / "foreground.png"
    Image.fromarray(rgba[:, :, :3]).save(source)
    Image.fromarray(rgba).save(fg)
    frame = dict(
        zip(
            ("image_right", "image_up", "camera_direction_from_origin"),
            np.eye(3).tolist(),
        )
    )
    reg = {
        "mesh_file_sha256": digest(nativefile),
        "camera_frame": frame,
        "upright_transform": np.eye(4).tolist(),
        "foreground_file_sha256": digest(fg),
    }
    brief = {
        "method": obs.METHOD,
        "source_sha256": digest(source),
        "visible_palette": ["blue"],
        "optical_finish_candidate": "glossy",
        "layout_candidate": "repeated",
        "observed_face_cues": {
            "repeated": {"family": "geometric", "location": "across_visible_face"}
        },
        "visible_structures": [],
        "visible_surround": {"color": "blue", "pigment_layout": "patterned"},
        "unseen_surface_observed": False,
    }
    atomic_json(case / "registration.json", reg)
    atomic_json(case / "observation.json", brief)
    Image.fromarray(true).save(case / "expected.png")
    return nativefile, source, fg, reg, brief


class NativeSurface(unittest.TestCase):
    def test_bound_actual_glb_round_trip_and_associated_camera(self):
        with tempfile.TemporaryDirectory() as folder:
            case = Path(folder)
            native_file, photo, fg, reg, brief = pbr_fixture(case)
            destination = case / "output"
            report = paint.paint_continuity(
                native_file, photo, fg, reg, brief, destination
            )
            self.assertGreater(report["changed_uv"], 60_000)
            self.assertEqual(report["output_glb"], digest(destination / "model.glb"))
            self.assertTrue(
                report[
                    "normals_uv_nodes_material_and_all_other_buffers_bitwise_identical"
                ]
            )
            self.assertEqual(report["locked_changed"], 0)
            _, _, final, _ = native._native(destination / "model.glb")
            _, _, old, _ = native._native(native_file)
            changed = (final != old).any(axis=-1)
            expected = np.array(Image.open(case / "expected.png"))
            # Exclude empty covered-mesh padding from the analytic colour comparison.
            roi = changed.copy()
            roi[:35] = roi[-35:] = False
            roi[:, :9] = roi[:, -9:] = False
            err = abs(final[roi].astype(float) - expected[roi])
            self.assertLess(float(err.mean()), 3)
            with self.assertRaisesRegex(ValueError, "registration"):
                paint.paint_continuity(
                    native_file,
                    photo,
                    fg,
                    {**reg, "mesh_file_sha256": "0" * 64},
                    brief,
                    case / "bad",
                )
            with self.assertRaises(ValueError):
                paint.paint_continuity(
                    native_file,
                    photo,
                    fg,
                    reg,
                    {**brief, "source_sha256": "0" * 64},
                    case / "bad",
                )
            self.assertFalse((case / "bad").exists())
