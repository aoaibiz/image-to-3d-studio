"""Class-independent refinement must keep parts, sharp evidence and true thin objects."""

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from engine.general_finish import build_general, _clean_texture, _pad_atlas
from engine.finish_export import export_finished


def textured(mesh, bitmap=None):
    xy = mesh.vertices[:, :2]
    uv = (xy - xy.min(0)) / np.maximum(np.ptp(xy, axis=0), 1e-8)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (64, 64), (245, 200, 160))
            if bitmap is None
            else bitmap,
            metallicFactor=0.37,
            roughnessFactor=0.28,
        ),
    )
    return mesh


def photo(mesh):
    im = Image.new("RGBA", (256, 256))
    xy = mesh.vertices[:, :2] * [1, -1]
    xy = (xy - (xy.min(0) + xy.max(0)) / 2) * (216 / np.ptp(xy, axis=0).max()) + 128
    draw = ImageDraw.Draw(im)
    for tri in xy[mesh.faces]:
        draw.polygon([tuple(p) for p in tri], fill=(245, 200, 160, 255))
    return im


class GeneralFinishTests(unittest.TestCase):
    def test_reliable_separated_surfaces_retain_topology_cavity_and_appearance(self):
        # A ring with a real hole and a smaller disconnected part has neither
        # a face nor vehicle landmarks. It exercises thin walls and open space.
        ring = trimesh.creation.torus(
            major_radius=0.78, minor_radius=0.23, major_sections=50, minor_sections=24
        )
        bead = trimesh.creation.icosphere(subdivisions=2, radius=0.13)
        bead.apply_translation([1.03, 0.73, 0.28])
        mesh = textured(trimesh.util.concatenate([ring, bead]))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate.glb"
            mesh.export(path)
            scene, info = build_general(path, photo(mesh))
            self.assertIsNotNone(scene, info)
            m = next(iter(scene.geometry.values()))
            self.assertEqual(len(m.faces), len(mesh.faces))
            self.assertTrue(np.array_equal(m.faces, mesh.faces))
            np.testing.assert_allclose(m.extents, mesh.extents, rtol=0.001)
            self.assertGreaterEqual(min(info["outline_self_iou_front_side_top"]), 0.995)
            self.assertGreater(info["geometry"]["moved_unique_vertices"], 0)
            self.assertEqual(len(m.split(only_watertight=False)), 2)
            self.assertAlmostEqual(m.visual.material.metallicFactor, 0.37)
            self.assertAlmostEqual(m.visual.material.roughnessFactor, 0.28)
            export_finished(scene, Path(td) / "out")
            self.assertTrue((Path(td) / "out/editable.gltf").is_file())

    def test_rejected_first_candidate_does_not_block_safe_geometry(self):
        mesh = textured(trimesh.creation.icosphere(subdivisions=3))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate.glb"
            mesh.export(path)

            def bad_candidate(surface):
                # Correct topology and dimension ratios alone cannot justify
                # moving the inferred object away from its observed outline.
                v = np.array(surface.vertices) + [0.25, 0, 0]
                return (
                    v,
                    surface.vertex_normals,
                    {"moved_unique_vertices": len(v), "method": "test-unsafe"},
                )

            with patch("engine.general_finish.fair_surface", side_effect=bad_candidate):
                scene, info = build_general(path, photo(mesh))
            self.assertIsNotNone(scene, info)
            self.assertNotEqual(info["geometry"].get("method"), "test-unsafe")
            np.testing.assert_allclose(scene.bounds, mesh.bounds, atol=0.001)

    def test_thin_object_does_not_acquire_depth(self):
        m = textured(trimesh.creation.box(extents=(1, 0.9, 0.003)))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "thin.glb"
            m.export(path)
            result, info = build_general(path, photo(m))
            self.assertIsNone(result, info)
            self.assertTrue(
                np.allclose(
                    trimesh.load(path, force="mesh").extents, [1, 0.9, 0.003], atol=1e-6
                )
            )

    def test_cleanup_keeps_photo_supported_marks_and_multi_texel_edges(self):
        rgb = np.full((64, 64, 3), [245, 200, 160], dtype=np.uint8)
        rgb[14, 18] = [24, 42, 220]  # Single unsupported blue contamination.
        rgb[31, 27] = [148, 25, 22]  # Isolated red, but present in photograph.
        rgb[20:24, 35:44] = [2, 3, 4]  # Dark multi-texel hardware/mark.
        m = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            faces=[[0, 1, 2], [0, 2, 3]],
            process=False,
        )
        m = textured(m, Image.fromarray(rgb))
        ref = np.full((20, 20, 3), [245, 200, 160], dtype=np.uint8)
        ref[7, 7] = [148, 25, 22]
        image, count = _clean_texture(m, ref, np.ones(ref.shape[:2], bool))
        self.assertEqual(count, 1)
        fixed = np.asarray(image)
        np.testing.assert_array_equal(fixed[14, 18], [245, 200, 160])
        np.testing.assert_array_equal(fixed[31, 27], rgb[31, 27])
        np.testing.assert_array_equal(fixed[20:24, 35:44], rgb[20:24, 35:44])

    def test_chart_padding_repairs_stale_texels_outside_the_chart_only(self):
        m = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            faces=[[0, 1, 2], [0, 2, 3]],
            process=False,
        )
        image = np.zeros((17, 17, 3), dtype=np.uint8)
        image[4:13, 4:13] = [100, 130, 160]
        image[7:10, 7:10] = [23, 7, 92]
        m = textured(m, Image.fromarray(image))
        m.visual.uv = 0.25 + m.visual.uv * 0.5
        result, count = _pad_atlas(m, m.visual.material.baseColorTexture)
        self.assertGreater(count, 0)
        fixed = np.asarray(result)
        np.testing.assert_array_equal(fixed[3, 8], image[4, 8])
        np.testing.assert_array_equal(fixed[4:13, 4:13], image[4:13, 4:13])
        np.testing.assert_array_equal(fixed[0, 0], image[0, 0])
        self.assertEqual(_pad_atlas(m, result)[1], 0)

    def test_missing_mask_misaligned_mask_and_invalid_input_fall_back(self):
        m = textured(trimesh.creation.icosphere(subdivisions=2))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "baseline.glb"
            m.export(path)
            for image in (
                Image.new("RGBA", (66, 66), "red"),
                Image.new("RGBA", (66, 66)),
            ):
                self.assertIsNone(build_general(path, image)[0])
            skinny = Image.new("RGBA", (140, 140))
            ImageDraw.Draw(skinny).rectangle((60, 12, 73, 122), fill="red")
            self.assertIsNone(build_general(path, skinny)[0])
            path.write_bytes(b"glTFxxx")
            self.assertIsNone(build_general(path, photo(m))[0])
