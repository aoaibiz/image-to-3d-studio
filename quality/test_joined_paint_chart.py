"""Single-strip transfer and admission controls. All provider tests are offline."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from PIL import Image

from engine import joined_paint_chart as chart
from engine import joined_chart_provider as remote
from engine import guided_provider
from engine.visual_brief import json_sha


def cylinder_surface(fold=False):
    # Sampling duplicates per chart cell let the measured-ring classifier test the true geometry.
    v, a = np.meshgrid(np.arange(128 * 2), np.arange(384 * 2), indexing="ij")
    theta = (a.ravel() + 0.4) * 2 * np.pi / (384 * 2)
    h = (v.ravel() + 0.4) / (128 * 2)
    radius = 0.32 + 0.02 * np.cos(2 * np.pi * h)
    points = np.column_stack((radius * np.sin(theta), h, radius * np.cos(theta)))
    normals = np.column_stack((np.sin(theta), np.zeros_like(h), np.cos(theta)))
    if fold:
        points[::2, [0]] *= 0.40
        points[::2, [2]] *= 0.40
        normals[::2] *= -1
    xx = np.mod(theta / (2 * np.pi) * 384, 384)
    yy = h * 128
    observed = np.cos(theta) > 0.43
    source = np.stack(
        (
            130 + 68 * np.sin(0.22 * xx + 0.15 * yy),
            110 + 50 * np.cos(0.31 * yy + 0.05 * xx),
            90 + 60 * np.sin(0.17 * xx - 0.43 * yy),
        ),
        axis=1,
    ).astype("uint8")
    shape = v.shape
    profile = np.array([[0.0, 0.0], [0.34, 0.0], [0.32, 0.5], [0.34, 1.0], [0.0, 1.0]])
    mesh = trimesh.creation.revolve(profile, sections=64)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return dict(
        mesh=mesh,
        points=points,
        normals=normals,
        source_rgb=source,
        observed=observed,
        y=v.ravel(),
        x=a.ravel(),
        old=np.full((*shape, 3), 115, np.uint8),
        locked=observed.reshape(shape).copy(),
        observation={"observed_face_cues": {}},
    )


class JoinedChartTests(unittest.TestCase):
    def test_one_coordinate_and_protected_uv(self):
        s = cylinder_surface()
        c = chart._build(s)
        lo, hi = c["photo"].shape[:2]
        y, x = np.mgrid[:lo, :hi]
        pattern = np.stack(
            (
                100 + 60 * np.sin(x * 0.23 + y * 0.18),
                80 + 65 * np.cos(x * 0.19 - y * 0.24),
                115 + 60 * np.sin(y * 0.21),
            ),
            axis=2,
        )
        rgb = np.where(c["anchor"][:, :, None], c["photo"], pattern)
        result = Image.fromarray(cv2.resize(rgb[::-1].astype("uint8"), chart.SIZE))
        pixels, stats = chart._transfer(s, c, result)
        self.assertGreater(stats["changed_uv"], 500)
        np.testing.assert_array_equal(pixels[s["locked"]], s["old"][s["locked"]])
        self.assertEqual(stats["unsupported_surface_changed"], 0)
        with self.assertRaises(chart.ChartUnavailable):
            chart._transfer(s, c, Image.fromarray(c["guide"]))
        with self.assertRaises(chart.ChartUnavailable):
            chart._transfer(s, c, Image.fromarray(np.asarray(result)[:, ::-1]))

    def test_natural_completion_never_masks_an_incompatible_immutable_band(self):
        s = cylinder_surface()
        c = chart._build(s)
        h, w = c["anchor"].shape
        y, x = np.mgrid[:h, :w]
        pattern = np.stack(
            (
                70 + 60 * np.sin(x * 0.14),
                150 + 45 * np.cos(y * 0.22),
                210 - 25 * np.sin(x * 0.23),
            ),
            axis=2,
        )
        rgb = np.where(c["anchor"][:, :, None], c["photo"], pattern)
        im = Image.fromarray(cv2.resize(rgb[::-1].astype("uint8"), chart.SIZE))
        locked = c["usable"] & (c["az"] > 148) & (c["az"] < 193)
        s["locked"][s["y"][locked], s["x"][locked]] = True
        s["old"][s["y"][locked], s["x"][locked]] = [255, 0, 255]
        with self.assertRaisesRegex(chart.ChartUnavailable, "protected native band"):
            chart._transfer(s, c, im)

    def test_confirmation_requires_two_margins_and_measured_geometry(self):
        s = cylinder_surface()
        inner = s["mesh"].copy()
        inner.vertices[:, [0, 2]] *= 0.97
        self.assertFalse(
            chart._single_loops(
                trimesh.util.concatenate([s["mesh"], inner]), 0, 1, 12
            ).any()
        )
        s = cylinder_surface(True)
        with self.assertRaises(chart.ChartUnavailable):
            chart._build(s)
        s = cylinder_surface()
        s["observed"] &= s["points"][:, 0] > 0
        with self.assertRaises(chart.ChartUnavailable):
            chart._build(s)

    def test_inference_cannot_repeat_a_distinct_front_patch(self):
        rng = np.random.default_rng(120)
        p = rng.integers(0, 255, (17, 17, 3), dtype=np.uint8)
        output = rng.integers(0, 255, (62, 106, 3), dtype=np.uint8)
        hidden = np.ones(output.shape[:2], bool)
        chart._check_unique_patches(output, hidden, [p])
        output[13:30, 75:92] = p[:, ::-1]
        with self.assertRaises(chart.ChartUnavailable):
            chart._check_unique_patches(output, hidden, [p])
        hidden[13:30, 75:92] = False
        chart._check_unique_patches(output, hidden, [p])

    def test_shared_slot_replay_and_competing_recipe_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            guide = root / "guide.png"
            manifest = root / "chart.json"
            job = root / "job"
            config = root / "setup.json"
            Image.new("RGB", (64, 64), "teal").save(source)
            Image.new("RGB", chart.SIZE, "gray").save(guide)
            from studio.configure import agreement

            pipeline = remote.SF3D_OBSERVED_PIPELINE
            current = agreement(pipeline)
            current = {
                **current,
                "schema": 2,
                "appearance": {
                    **current["appearance"],
                    "paint_guide_recipes": [
                        "two-rendered-geometry-views",
                        "source-anchored-surface-chart",
                    ],
                },
            }
            config.write_text(json.dumps({"pipeline": pipeline, "agreement": current}))
            observation = {"method": next(iter(remote.OBSERVATION_METHODS))}
            m = dict(
                method=chart.METHOD,
                photo_sha=remote._sha(source.read_bytes()),
                guide_sha=remote._sha(guide.read_bytes()),
                registration_sha="1" * 64,
                model_sha="2" * 64,
                foreground_sha="3" * 64,
                visual_brief=observation,
                visual_brief_sha256=json_sha(observation),
            )
            m["chart_id"] = json_sha(m)
            manifest.write_text(json.dumps(m))

            def call():
                return remote.generate_chart(
                    source,
                    guide,
                    manifest,
                    job,
                    native_model=source,
                    foreground=source,
                    registration={},
                    observation=observation,
                    setup=config,
                )

            with (
                patch("studio.configure.agreement", return_value=current),
                patch.object(remote, "confirm_chart"),
                patch.object(
                    remote, "validate_photo_observations", return_value=observation
                ),
                patch.object(
                    guided_provider,
                    "preflight",
                    return_value={"available": True, "cli_version": "fixture"},
                ),
                patch.object(
                    guided_provider,
                    "_run_image_once",
                    side_effect=RuntimeError("fixture interrupted"),
                ) as submit,
            ):
                with self.assertRaisesRegex(RuntimeError, "fixture interrupted"):
                    call()
                work = job / "guided-paint"
                receip = work / "receipt.json"
                r = json.loads(receip.read_text())
                self.assertEqual(r["attempts"], 1)
                with self.assertRaises(guided_provider.TurnaroundProviderError):
                    call()
                self.assertEqual(submit.call_count, 1)
                r.update(
                    state="completed",
                    returncode=0,
                    thread_id="c186320d-8b12-46dd-9733-ba11b60ba772",
                    sheet_size=list(chart.SIZE),
                    generated_sha256=m["guide_sha"],
                    sheet_sha256=m["guide_sha"],
                )
                (work / "sheet.png").write_bytes(guide.read_bytes())
                receip.write_text(json.dumps(r))
                self.assertEqual(call()["sheet"], work / "sheet.png")
                remote.verify_completed(r, m, source, guide, guide.read_bytes())
                with self.assertRaises(guided_provider.TurnaroundProviderError):
                    remote.verify_completed(r, m, source, guide, b"changed")
                r["directions"] = ["rear", "opposite"]
                receip.write_text(json.dumps(r))
                with self.assertRaises(guided_provider.TurnaroundProviderError):
                    call()
                self.assertEqual(submit.call_count, 1)
                config.write_text("{}")
                with self.assertRaises(guided_provider.TurnaroundProviderError):
                    call()


if __name__ == "__main__":
    unittest.main()

