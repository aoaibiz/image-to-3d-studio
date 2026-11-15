"""Shape hypotheses must preserve sealed evidence and hold on inadequate rim cues."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw
import trimesh
from trimesh.visual.material import PBRMaterial

from engine import inferred_cavity as cavity


OBSERVATION = {
    "function_context": {
        "interpretation": {"possible_purpose": "contain_or_pass", "confidence": "low"},
        "observed_cues": [{"kind": "rim_or_frame", "location": "upper_border"}],
    },
    "structures": [],
}


def vessel(height, radius):
    profile = np.array(
        [
            [0, 0],
            [0.25, 0],
            [0.45, 0.035],
            [0.65, 0.15],
            [0.8, 0.38],
            [0.45, 0.64],
            [0.27, 0.76],
            [0.24, 0.90],
            [0.28, 0.95],
            [0.38, 0.983],
            [0.4, 0.99],
            [0.4, 1],
            *[[0.4 * q, 1] for q in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]],
        ]
    ) * [radius, height]
    mesh = trimesh.creation.revolve(profile, sections=64)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return mesh


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InferredCavityTest(unittest.TestCase):
    def test_geometry_variation_and_floor_orientation(self):
        for height, radius in [(2.0, 1.0), (1.5, 0.82)]:
            with self.subTest(height=height, radius=radius):
                mesh = vessel(height, radius)
                mask, vertices, faces, metrics = cavity.make(mesh)
                self.assertGreater(mask.sum(), 18)
                self.assertTrue(metrics["connected_watertight"])
                self.assertEqual(metrics["euler_before"], metrics["euler_after"])
                inner = trimesh.Trimesh(vertices, faces, process=False)
                floor = inner.triangles[:, :, 1].max(1) <= vertices[:, 1].min() + 1e-9
                self.assertTrue((inner.face_normals[floor, 1] > 0.99999).all())
                self.assertGreater(metrics["minimum_design_wall"], 0)
                self.assertLessEqual(metrics["depth"], height * 0.15 + 1e-8)

    def test_denies_solid_boxes_cylinder_and_sculpture(self):
        cylinder = trimesh.creation.cylinder(radius=0.4, height=1.8, sections=48)
        cylinder.apply_transform(
            trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        )
        for mesh in [
            trimesh.creation.box([1, 0.15, 0.6]),
            trimesh.creation.box([0.5, 1, 0.5]),
            cylinder,
            trimesh.creation.icosphere(subdivisions=2),
        ]:
            with (
                self.subTest(bounds=mesh.bounds.tolist()),
                self.assertRaises(cavity.Hold),
            ):
                cavity.make(mesh)

    def test_denies_closures_missing_evidence_and_decoration(self):
        self.assertTrue(cavity.eligible(OBSERVATION))
        for kind in [
            "boundary_seam",
            "hinge_or_pivot",
            "latch_or_fastener",
            "control_or_interface",
            "attachment_point",
        ]:
            obs = copy.deepcopy(OBSERVATION)
            obs["function_context"]["observed_cues"].append(
                {"kind": kind, "location": "upper_border"}
            )
            self.assertFalse(cavity.eligible(obs), kind)
        for confidence in ["insufficient", None, "unrecognized"]:
            observation = copy.deepcopy(OBSERVATION)
            observation["function_context"]["interpretation"]["confidence"] = confidence
            self.assertFalse(cavity.eligible(observation))
        self.assertFalse(cavity.eligible({}))
        obs = copy.deepcopy(OBSERVATION)
        obs["function_context"]["interpretation"]["possible_purpose"] = "unknown"
        self.assertFalse(cavity.eligible(obs))
        self.assertTrue(
            cavity.cap_might_carry_graphics(
                [[0.8, 0.7, 0.6]] * 85 + [[0.05, 0.05, 0.2]] * 15
            )
        )
        self.assertFalse(cavity.cap_might_carry_graphics([[0.8, 0.7, 0.6]] * 100))

    def test_editor_round_trip_seals_and_visible_cap_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            src = path / "source.png"
            fg = path / "foreground.png"
            model = path / "native.glb"
            image = Image.new("RGBA", (200, 390), (0, 0, 0, 0))
            ImageDraw.Draw(image).polygon(
                [
                    (58, 13),
                    (142, 13),
                    (126, 44),
                    (126, 95),
                    (178, 220),
                    (140, 360),
                    (58, 360),
                    (22, 220),
                    (74, 95),
                    (74, 44),
                ],
                fill=(200, 185, 165, 255),
            )
            image.save(fg)
            image.convert("RGB").save(src)
            native = vessel(2.0, 1.0)
            native.visual = trimesh.visual.TextureVisuals(
                uv=np.zeros((len(native.vertices), 2)),
                material=PBRMaterial(
                    baseColorTexture=Image.new("RGB", (8, 8), (208, 190, 164)),
                    roughnessFactor=0.5,
                ),
            )
            native.export(model)
            sealed = dict(
                source_sha256=digest(src),
                foreground_sha256=digest(fg),
                display_model_sha256=digest(model),
                observer_output_sha256="1" * 64,
                observation_canonical_sha256=cavity.canonical_observation_sha(
                    OBSERVATION
                ),
                geometry_master_sha256=digest(model),
                display_rotation_xyzw=[0, 1, 0, 0],
            )
            reg = dict(
                shape_hypothesis_inputs=sealed,
                mesh_file_sha256=digest(model),
                foreground_file_sha256=digest(fg),
                camera_frame=dict(
                    camera_direction_from_origin=[0, 0, -1], image_up=[0, 1, 0]
                ),
                foreground_touches_source_border=False,
                silhouette_iou=0.95,
            )
            output = path / "suggested.glb"
            args = dict(
                source=src,
                foreground=fg,
                display_model=model,
                observation=OBSERVATION,
                registration=reg,
                output=output,
            )
            result = cavity.editor(**args)
            self.assertEqual(result["decision"], "alternative", result)
            self.assertEqual(result["evidence"]["opening"], "plausible")
            self.assertTrue(output.exists())
            self.assertFalse(result["evidence"]["painted_cap"] is True)
            self.assertEqual(sealed["display_model_sha256"], digest(model))
            previous = digest(output)
            cavity.editor(**args)
            self.assertEqual(digest(output), previous)
            reg["camera_frame"]["camera_direction_from_origin"] = [0, 0.6, -0.8]
            args["output"] = path / "no.glb"
            self.assertEqual(cavity.editor(**args)["decision"], "unchanged")
            self.assertFalse(args["output"].exists())
            reg["shape_hypothesis_inputs"]["source_sha256"] = "0" * 64
            args["observation"] = {}
            with self.assertRaises(cavity.IntegrityError):
                cavity.editor(**args)


if __name__ == "__main__":
    unittest.main()
