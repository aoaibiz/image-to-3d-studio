"""Local tests. The opt-in archived trial never calls a remote provider."""

import contextlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine import sf3d_observed_job as job
from engine import triposg_cpu as files


def observed(*, cues=None, structures=None, reverse="continuous_substrate"):
    return {
        "observed_face_cues": cues or {},
        "visible_surround": {"pigment_layout": "uniform"},
        "visible_structures": structures or [],
        "source_sha256": "d" * 64,
        "function_context": {
            "observed_cues": [],
            "interpretation": {"reverse_candidates": [reverse]},
        },
    }


class DecisionTests(unittest.TestCase):
    def test_chart_consent_is_frozen_per_job_and_paint_receipt_owns_one_recipe(self):
        from engine.guided_provider import TurnaroundProviderError
        from engine.joined_chart_provider import chart_consent
        from studio.configure import agreement, legacy_sf3d_observed_agreement

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup = root / "engine.json"
            old, new = root / "old", root / "new"
            old.mkdir()
            new.mkdir()
            key = {"pipeline": job.PIPELINE, "input_sha256": "3" * 64}
            obs = observed(cues={"repeated": {"family": "geometric"}})
            files.atomic_json(
                setup,
                {
                    "pipeline": job.PIPELINE,
                    "agreement": legacy_sf3d_observed_agreement(),
                },
            )
            job._consent(setup)
            self.assertFalse(job._chart_policy(setup, old, key, True))
            with self.assertRaises(TurnaroundProviderError):
                chart_consent(setup)
            files.atomic_json(
                setup, {"pipeline": job.PIPELINE, "agreement": agreement(job.PIPELINE)}
            )
            chart_consent(setup)
            self.assertFalse(job._chart_policy(setup, old, key, False))
            self.assertTrue(job._chart_policy(setup, new, key, True))
            self.assertTrue(job._chart_route(new, False, True, obs))
            (new / "guided-paint").mkdir()
            path = new / "guided-paint/receipt.json"
            files.atomic_json(
                path, {"state": "submitted", "directions": ["rear", "opposite"]}
            )
            self.assertFalse(job._chart_route(new, True, True, obs))
            files.atomic_json(
                path,
                {
                    "directions": ["joined_chart"],
                    "job_recipe": "joined-intrinsic-strip-v1",
                },
            )
            with self.assertRaises(job.StageError):
                job._chart_route(new, True, True, obs)
            (new / "chart-guide").mkdir()
            files.atomic_json(new / "chart-guide/chart.json", {})
            self.assertTrue(job._chart_route(new, True, True, obs))
            with self.assertRaises(job.StageError):
                job._chart_route(new, True, False, obs)

    def test_local_finish_never_replaces_an_existing_or_lost_paint_receipt(self):
        source = observed(cues={"repeated": {"family": "geometric"}})
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            self.assertFalse(job._existing_paint_attempt(work))
            files.atomic_json(work / "paint-attempt.json", {"state": "local_preflight"})
            self.assertFalse(job._existing_paint_attempt(work))
            files.atomic_json(work / "paint-attempt.json", {"state": "pending"})
            self.assertTrue(job._existing_paint_attempt(work))
            self.assertIsNone(job._local_paint_method(source, "guided_region", True))
            (work / "paint-attempt.json").unlink()
            (work / "appearance/guided-paint").mkdir(parents=True)
            with self.assertRaises(job.StageError) as pending:
                job._existing_paint_attempt(work)
            self.assertEqual(pending.exception.code, "sf3d_pending")
            files.atomic_json(
                work / "paint-attempt.json",
                {"stage": "paint", "state": "local_preflight"},
            )
            (work / "appearance/guided-paint/run.lock").touch()
            self.assertFalse(job._existing_paint_attempt(work))
            self.assertFalse((work / "appearance/guided-paint").exists())
            (work / "appearance/guided-paint").mkdir()
            (work / "appearance/guided-paint/unexpected").touch()
            with self.assertRaises(job.StageError):
                job._existing_paint_attempt(work)
            files.atomic_json(
                work / "appearance/guided-paint/receipt.json", {"state": "submitted"}
            )
            self.assertTrue(job._existing_paint_attempt(work))
            self.assertIsNone(job._local_paint_method(source, "inferred_paint", False))

    def test_inconclusive_local_measurement_leaves_no_model_or_external_attempt(self):
        from quality.test_sf3d_provider import fixture_glb, png

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            base = out / "sf3d-observed-work/native"
            base.mkdir(parents=True)
            (base / "input-processed.png").write_bytes(png())
            (base / "provider-native.glb").write_bytes(fixture_glb())

            def no_period(native, source, matte, reg, brief, stage):
                (stage / "partial").write_bytes(b"unusable local fit")

            result = job._native_finish(
                out,
                base / "input-processed.png",
                base,
                {},
                {},
                {},
                {},
                continue_surface=no_period,
            )
            self.assertIsNone(result)
            self.assertFalse((out / "model.glb").exists())
            self.assertFalse(job._existing_paint_attempt(base.parent))
            self.assertEqual(list(base.parent.iterdir()), [base])

    def test_native_or_guided_requires_bound_observation(self):
        self.assertEqual(job._appearance_mode(observed(), False), "native_material")
        self.assertEqual(
            job._appearance_mode(
                observed(cues={"repeated": {"family": "geometric"}}), False
            ),
            "guided_region",
        )
        self.assertEqual(job._appearance_mode(observed(), True), "guided_region")
        carved = observed(
            structures=[{"kind": "recessed_opening"}, {"kind": "raised_relief"}],
            reverse="unknown",
        )
        self.assertEqual(job._appearance_mode(carved, False), "native_material")
        with self.assertRaises(job.NeedsReview):
            job._appearance_mode(
                observed(
                    structures=[{"kind": "screen_or_controls"}], reverse="unknown"
                ),
                False,
            )
        focal = observed(cues={"focal": {"family": "geometric"}})
        with self.assertRaises(job.NeedsReview):
            job._appearance_mode(focal, False)
        self.assertEqual(job._appearance_mode(focal, True), "guided_region")
        with self.assertRaises(job.NeedsReview):
            job._appearance_mode(observed(reverse="distinct_working_surface"), False)
        context = observed(reverse="unknown")
        context["function_context"]["observed_cues"] = [
            {"kind": "hinge_or_pivot", "location": "upper_border"}
        ]
        with self.assertRaises(job.NeedsReview):
            job._appearance_mode(context, False)

    def test_full_requires_two_registered_views_and_no_copy_veto(self):
        from engine import native_full_paint as full
        from engine import native_region_paint as region

        focal = observed(cues={"focal": {"family": "geometric"}})
        self.assertEqual(
            job._appearance_mode(focal, False, full_supported=True), "inferred_paint"
        )
        self.assertEqual(
            job._appearance_mode(
                observed(reverse="distinct_working_surface"), False, full_supported=True
            ),
            "inferred_paint",
        )
        repeated = observed(cues={"repeated": {"family": "geometric"}})
        args = [None] * 11
        args[7] = {"visual_brief": repeated}
        with (
            patch.object(full, "paint_native_full") as accepted,
            patch.object(region, "paint_native_region") as regional,
        ):
            self.assertEqual(
                job._synthesize(*args)[:2], ("guided_full", "full-paint.json")
            )
            accepted.assert_called_once()
            regional.assert_not_called()
        with (
            patch.object(
                full,
                "paint_native_full",
                side_effect=full.NativeFullRegistrationError("outline"),
            ),
            patch.object(
                full,
                "preflight_native_paint",
                return_value={"region_depth_eligible": True},
            ),
            patch.object(region, "paint_native_region") as regional,
        ):
            self.assertEqual(
                job._synthesize(*args), ("guided_region", "region.json", "outline")
            )
            regional.assert_called_once()
            args[7] = {"visual_brief": focal}
            with self.assertRaises(job.NeedsReview):
                job._synthesize(*args)
            regional.assert_called_once()
        for error in (
            ValueError("wrong source"),
            full.NativeFullReviewHold("copy veto"),
        ):
            with (
                patch.object(full, "paint_native_full", side_effect=error),
                patch.object(region, "paint_native_region") as regional,
                self.assertRaises(ValueError),
            ):
                job._synthesize(*args)
            regional.assert_not_called()

    def test_only_typed_regional_fit_error_keeps_native_for_review(self):
        from engine import native_full_paint as full
        from engine import native_region_paint as region

        args = [None] * 11
        args[7] = {"visual_brief": observed(cues={"repeated": {"family": "geometric"}})}
        with (
            patch.object(
                full,
                "paint_native_full",
                side_effect=full.NativeFullRegistrationError("outline"),
            ),
            patch.object(
                full,
                "preflight_native_paint",
                return_value={"region_depth_eligible": True},
            ),
        ):
            with (
                patch.object(
                    region,
                    "paint_native_region",
                    side_effect=region.NativeRegionQualityHold("no supported region"),
                ),
                self.assertRaises(job.NeedsReview),
            ):
                job._synthesize(*args)
            error = ValueError("invalid source receipt")
            with (
                patch.object(region, "paint_native_region", side_effect=error),
                self.assertRaises(ValueError) as failure,
            ):
                job._synthesize(*args)
            self.assertIs(failure.exception, error)

    def test_uncertain_receipt_or_wrong_paint_does_not_auto_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "provider/receipt.json"
            for idx in range(2):
                with (
                    self.assertRaisesRegex(ValueError, "local preflight"),
                    job._request(root, "paint", receipt),
                ):
                    raise ValueError("local preflight")
            receipt.parent.mkdir()
            with (
                self.assertRaisesRegex(RuntimeError, "lost acknowledgement"),
                job._request(root, "paint", receipt),
            ):
                files.atomic_json(receipt, {"state": "submitted_or_ambiguous"})
                raise RuntimeError("lost acknowledgement")
            with job._request(root, "paint", receipt):
                pass
            receipt.unlink()
            with (
                self.assertRaises(job.StageError),
                job._request(root, "paint", receipt),
            ):
                self.fail("reposted without receipt")
            (root / "finished-native.glb").write_bytes(b"orphan")
            with self.assertRaises(job.StageError):
                job._guard_downstream(root, root / "work")
            sheet = root / "sheet.png"
            sheet.write_bytes(b"test")
            with self.assertRaises(ValueError):
                job._verify_proposal(
                    {"thread_id": None},
                    sheet,
                    {
                        "model_sha": "a" * 64,
                        "photo_sha": "b" * 64,
                        "guide_sha": "c" * 64,
                        "camera": {},
                    },
                    observed(),
                )
            self.assertFalse(job.valid(root, {"input_sha256": "a" * 64}))

    def test_lost_provider_receipt_with_only_downstream_evidence(self):
        for name in (
            "inspection.json",
            "manifest.json",
            "offline-preview.html",
            "abcdef_front_material.png",
            "sf3d-observed-work/appearance/photo-observer/receipt.json",
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                p = out / name
                p.parent.mkdir(exist_ok=True, parents=True)
                p.touch()
                with self.assertRaises(job.StageError):
                    job._guard_downstream(out, out / "sf3d-observed-work")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            work = out / "sf3d-observed-work"
            for p in (
                work / "identity.json",
                work / "native/settings.json",
                work / "appearance/photo-observer/receipt.json",
                work / "orientation/hidden-review.json",
            ):
                p.parent.mkdir(parents=True, exist_ok=True)
                files.atomic_json(p, {})
            with self.assertRaises(job.StageError):
                job._guard_downstream(out, work)

    def test_unpainted_mesh_measured_against_monochrome_photo_and_colour_mismatch(self):
        import numpy as np
        import trimesh
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            mesh = trimesh.creation.icosphere(subdivisions=2)
            mesh.vertex_normals = mesh.vertex_normals.copy()
            uv = np.full((len(mesh.vertices), 2), 0.5)
            colour = (155, 112, 72)
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=uv,
                material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=Image.new("RGB", (256, 256), colour),
                    metallicFactor=0.0,
                    roughnessFactor=0.7,
                ),
            )
            (base / "model.glb").write_bytes(
                trimesh.exchange.gltf.export_glb(
                    trimesh.Scene(mesh), include_normals=True
                )
            )
            alpha = np.zeros((256, 256), bool)
            alpha[12:-12, 12:-12] = True
            rgb, _, ids = job._render_native(
                mesh, np.array(mesh.visual.material.baseColorTexture), np.eye(3), alpha
            )
            matte = Image.fromarray(np.dstack((rgb, (ids >= 0).astype("uint8") * 255)))
            matte.save(base / "input-processed.png")
            reg = {
                "camera_frame": {
                    "image_right": [-1.0, 0.0, 0.0],
                    "image_up": [0.0, 1.0, 0.0],
                    "camera_direction_from_origin": [0.0, 0.0, -1.0],
                }
            }
            job._native_inspection(base, reg, observed(), base)
            report = files.read_json(base / "native-material.json")
            self.assertGreater(report["silhouette_iou"], 0.9)
            self.assertFalse(report["hold_projection"])
            self.assertEqual(report["model_sha256"], files.digest(base / "model.glb"))
            self.assertEqual(report["view_order"][0], "registered front")
            self.assertEqual(report["visual_quality"], "needs_review")
            self.assertEqual(len(report["central_depth_span_fractions"]), 3)
            wrong = Image.new("RGBA", matte.size, (33, 8, 250, 255))
            wrong.putalpha(matte.getchannel("A"))
            wrong.save(base / "input-processed.png")
            with self.assertRaises(job.NeedsReview):
                job._native_inspection(base, reg, observed(), base)


@unittest.skipUnless(
    os.getenv("STUDIO_SF3D_OBSERVED_REPLAY"),
    "Private three-provider archives are optional.",
)
class ArchivedIntegration(unittest.TestCase):
    def test_sealed_watch_recovery_no_network_or_process_launch(self):
        from engine import inferred_cavity, sf3d_provider, sf3d_shape_job
        from quality.test_sf3d_provider import Replay
        from studio.configure import agreement

        self.assertTrue(callable(inferred_cavity.editor))
        jobs = Path(os.environ["STUDIO_SF3D_OBSERVED_REPLAY"])
        raw = jobs / "sf3d-official-20261116/fresh"
        guided = jobs / "auto-finish-20261116/fresh-guided"
        src = guided.parent / "fresh/input.jpg"
        observer = jobs / "auto-local-20261116/vision-observation-fresh"
        source_sha = "9e5bf4c14ae95b419b0d5e0259d571da25dd7dcf85b72a75dbb5dca4fcacbe7d"
        native_sha = "0f9722e8a1adf270eabe4cdf97f97cb07b4fd4a4339024cfc31ba22697831db4"
        historical = files.read_json(raw / "receipt.json")
        self.assertEqual(files.digest(src), source_sha)
        self.assertEqual(files.digest(raw / "model.glb"), native_sha)
        with contextlib.ExitStack() as stack:
            retained = os.getenv("STUDIO_SF3D_REPLAY_OUTPUT")
            if retained:
                out = Path(retained)
                out.mkdir(exist_ok=False)
            else:
                out = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            work = out / "sf3d-observed-work"
            base = work / "native"
            full = base / "sf3d-work/provider"
            matte = raw / "foreground.png"
            # Reuse original event IDs/status and genuine native bytes, with reconstructed
            # transport envelopes; the shipping client strips unseen/faint pixel RGB before upload.
            api = Replay((raw / "model.glb").read_bytes(), historical)
            api.work = full
            receipt = sf3d_provider._generate(
                matte, source_sha, full, True, None, None, api
            )
            sf3d_shape_job.finalize(
                full / "model.glb",
                matte,
                {"source_sha256": source_sha, "selection": {"method": "single image"}},
                {"method": "archived original foreground"},
                receipt,
                base,
            )
            files.atomic_json(
                work / "identity.json",
                {"pipeline": job.PIPELINE, "input_sha256": source_sha},
            )
            setup = work / "setup.json"
            files.atomic_json(
                setup, {"pipeline": job.PIPELINE, "agreement": agreement(job.PIPELINE)}
            )
            shutil.copyfile(guided / "registration.json", work / "registration.json")
            shutil.copytree(guided / "base", work / "proxy")
            account = work / "appearance"
            shutil.copytree(observer, account / "photo-observer")
            shutil.copytree(guided / "guided-paint", account / "guided-paint")
            # No generated proposal may be substituted, and no user account may be contacted.
            for name in ("requests.sessions.Session.request", "subprocess.Popen"):
                stack.enter_context(
                    patch(
                        name,
                        side_effect=AssertionError(
                            "offline replay contacted an external service"
                        ),
                    )
                )
            # Exercise the no-support path against this real source and native GLB,
            # replacing only the local fitted region mask; archived provider bytes stay intact.
            import numpy as np

            from engine import native_region_paint

            align = native_region_paint.align

            def no_solid_region(proposal, projected):
                return align(proposal, np.zeros_like(projected))

            with (
                patch.object(native_region_paint, "align", side_effect=no_solid_region),
                self.assertRaises(job.StageError) as held,
            ):
                job.run(src, out, setup=setup)
            self.assertEqual(held.exception.code, "sf3d_observed_review")
            self.assertIn("no supported region", str(held.exception))
            self.assertFalse((out / "settings.json").exists())
            preview = files.read_json(out / "review-hold.json")
            self.assertEqual(preview["reason_code"], "native_projection")
            self.assertEqual(preview["reason"], job.HOLD_REASONS["native_projection"])
            self.assertEqual(preview["input_sha256"], source_sha)
            self.assertEqual(preview["shape_sha256"], native_sha)
            self.assertEqual(
                preview["preview_sha256"], files.digest(out / "review-native.glb")
            )
            self.assertEqual(
                preview["preview_sha256"], files.digest(base / "model.glb")
            )
            self.assertEqual(
                files.read_json(work / "state.json")["status"], "needs_review"
            )
            result = job.run(src, out, setup=setup)
            self.assertFalse((out / "review-hold.json").exists())
            self.assertTrue(job.valid(out, result))
            self.assertEqual(result["appearance_mode"], "guided_region")
            self.assertEqual(result["shape_hypothesis"]["decision"], "unchanged")
            self.assertEqual(result["artifact_sha256"]["shape-master.glb"], native_sha)
            self.assertEqual(
                files.digest(out / "inferred-paint.png"),
                files.digest(guided / "guided-paint/sheet.png"),
            )
            self.assertEqual(
                files.digest(out / "finished-native.glb"),
                files.digest(guided / "region/native-model/model.glb"),
            )
            region = files.read_json(out / "region.json")
            self.assertEqual(region["locked_changed"], 0)
            self.assertEqual(region["unsupported_surface_changed"], 0)
            self.assertGreater(region["changed_uv"], 0)
            self.assertEqual(job.run(src, out, setup=setup), result)
            self.assertEqual(api.joins, [4, 5])


class OptionalShapeTests(unittest.TestCase):
    def _stage(self, root):
        from engine.finish_export import embed_master
        from engine.sf3d_provider import native_glb
        from quality.test_sf3d_provider import fixture_glb, png

        source = root / "original.png"
        source.write_bytes(png())
        stage = root / "stage"
        stage.mkdir()
        out = root / "out"
        response = (
            out / "sf3d-observed-work/appearance/photo-observer/last-message.json"
        )
        response.parent.mkdir(parents=True)
        files.atomic_json(response, {"test": "fixture"})
        for name in ("model.glb", "shape-master.glb"):
            (stage / name).write_bytes(fixture_glb())
        for name in ("input-processed.png", "native-inspection.png"):
            (stage / name).write_bytes(png())
        embed_master(stage / "model.glb", stage / "editable.gltf")
        asset = native_glb(stage / "model.glb")
        src = files.digest(source)
        camera = {"display_rotation_xyzw": [0, 1, 0, 0]}
        observer = {
            "state": "completed",
            "source_sha256": src,
            "unseen_surface_observed": False,
            "output_sha256": files.digest(response),
            "request_id": "fixture",
        }
        files.atomic_json(
            response.parent.parent / "observation-seal.json",
            {"receipt": observer, "visual_brief": {"source_sha256": src}},
        )
        files.atomic_json(
            stage / "generation.json",
            {
                "source_sha256": src,
                "external_image_generations": 0,
                "photo_observation": observer,
                "visual_brief": {"source_sha256": src},
                "camera": camera,
                "registration": {
                    "mesh_file_sha256": asset["sha256"],
                    "foreground_file_sha256": files.digest(
                        stage / "input-processed.png"
                    ),
                },
                "display_glb_sha256": asset["sha256"],
                "shape": {
                    "camera": camera,
                    "native_sha256": asset["sha256"],
                    "output_sha256": asset["sha256"],
                },
            },
        )
        files.atomic_json(
            stage / "native-material.json",
            {
                "source_sha256": src,
                "model_sha256": asset["sha256"],
                "inspection_sha256": files.digest(stage / "native-inspection.png"),
                "hold_projection": False,
            },
        )
        meta = {
            "pipeline": job.PIPELINE,
            "appearance_mode": "native_material",
            "input_sha256": src,
            "visual_quality": "needs_review",
            "auto_finish_complete": True,
            "triangles": 1,
            "embedded_images": 1,
            "glb_sha256": asset["sha256"],
            "camera": camera,
            "artifact_sha256": {f.name: files.digest(f) for f in stage.iterdir()},
        }
        files.atomic_json(stage / "settings.json", meta)
        return stage, out, source, response, meta

    def test_no_evidence_preserves_display_and_previous_paint_master(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage, out, source, raw, old = self._stage(Path(tmp))

            def no_op(**args):
                self.assertEqual(
                    args["registration"]["shape_hypothesis_inputs"][
                        "observer_output_sha256"
                    ],
                    files.digest(raw),
                )
                self.assertFalse(args["output"].exists())
                return {
                    "decision": "unchanged",
                    "method": "test",
                    "evidence": {},
                    "reason": "ambiguous closure",
                }

            result = job._finish_hypothesis(
                stage, out, source, job.ShapeEditor(no_op, "1" * 64)
            )
            self.assertTrue(job.valid(stage, result))
            self.assertEqual(result["glb_sha256"], old["glb_sha256"])
            self.assertEqual(
                result["artifact_sha256"]["editable.gltf"],
                old["artifact_sha256"]["editable.gltf"],
            )
            report = files.read_json(stage / "shape-hypothesis.json")
            self.assertFalse(report["dimensions_observed"])
            self.assertEqual(report["decision"], "unchanged")
            self.assertFalse((stage / "shape-alternative.glb").exists())

    def test_visible_closure_veto_preserves_source_paint(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage, out, src, _, old = self._stage(Path(tmp))
            info = {"sha256": old["glb_sha256"], "faces": 1}

            def proposes_despite_closure(**args):
                shutil.copyfile(args["display_model"], args["output"])
                return {
                    "method": "fixture",
                    "decision": "alternative",
                    "reason": "proposed by test adapter",
                    "inferred_dimensions": {"depth": 0.2},
                    "evidence": {
                        "opening": "plausible",
                        "visible_closure": True,
                        "negative_controls_passed": True,
                        "geometry_supported": True,
                        "source_registered": True,
                    },
                }

            with patch.object(job, "_shape_candidate", return_value=info):
                result = job._finish_hypothesis(
                    stage, out, src, job.ShapeEditor(proposes_despite_closure, "1" * 64)
                )
            self.assertEqual(result["shape_hypothesis"]["decision"], "unchanged")
            self.assertEqual(result["glb_sha256"], old["glb_sha256"])
            self.assertFalse((stage / "shape-alternative.glb").exists())

    def test_damaged_source_or_observation_stops_the_editor(self):
        for key in ("source", "observer"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                stage, out, src, raw, _ = self._stage(Path(tmp))
                (src if key == "source" else raw).write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    job._finish_hypothesis(
                        stage,
                        out,
                        src,
                        job.ShapeEditor(
                            lambda **_: self.fail("unbound editor"), "1" * 64
                        ),
                    )
                self.assertFalse((stage / "shape-hypothesis.json").exists())


class AnalyticSurfaceIntegration(unittest.TestCase):
    def test_generated_surface_observation_uses_no_paint_request(self):
        from engine import sf3d_provider, sf3d_shape_job
        from quality.test_continuity_paint import pbr_fixture

        fixture = Path(self.enterContext(tempfile.TemporaryDirectory()))
        pbr_fixture(fixture)
        source = fixture / "photo.png"
        matte = fixture / "foreground.png"
        native = fixture / "provider-native.glb"
        obs = files.read_json(fixture / "observation.json")
        src = files.digest(source)
        receipt = {
            "artifact": sf3d_provider.native_glb(native),
            "source_sha256": src,
            "foreground_sha256": files.digest(matte),
            "settings": {
                "fixture": "analytically generated geometry and source pattern"
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            base = out / "sf3d-observed-work/native"
            original = sf3d_shape_job.finalize(
                native,
                matte,
                {"source_sha256": src, "selection": "synthetic test"},
                {"method": "synthetic test"},
                receipt,
                base,
            )
            method = job._local_paint_method(
                obs, job._appearance_mode(obs, False, full_supported=True), False
            )
            self.assertIsNotNone(method)
            with patch(
                "requests.sessions.Session.request",
                side_effect=AssertionError("remote request"),
            ):
                settings = job._native_finish(
                    out,
                    source,
                    base,
                    original,
                    files.read_json(fixture / "registration.json"),
                    obs,
                    {"method": "synthetic test", "source_sha256": src},
                    continue_surface=method,
                )
            self.assertTrue(job.valid(out, settings))
            self.assertEqual(settings["appearance_mode"], "local_continuity")
            self.assertEqual(
                files.digest(out / "shape-master.glb"), files.digest(native)
            )
            self.assertEqual(
                files.digest(out / "continuity-native.glb"),
                files.read_json(out / "continuity-paint.json")["output_glb"],
            )
            self.assertEqual(settings["texture_dimensions"], [512, 512])
            self.assertEqual(
                files.read_json(out / "generation.json")["external_image_generations"],
                0,
            )
            self.assertFalse(job._existing_paint_attempt(base.parent))


@unittest.skipUnless(
    os.getenv("STUDIO_SF3D_SHAPE_REPLAY"),
    "Private sealed painted-geometry trial is optional.",
)
class ArchivedShapeIntegration(unittest.TestCase):
    def test_chart_transfer_uses_real_mesh_and_generated_test_receipt(self):
        """The hidden sheet is an earlier texture projection, not a new model response."""
        import io

        from engine import guided_provider, joined_chart_provider
        from studio.configure import agreement

        root = Path(os.environ["STUDIO_SF3D_SHAPE_REPLAY"])
        archive = (
            root
            / "run/store/b7bc98b1-c8d3-4ccd-8e40-7700ad1abbcb/output/sf3d-observed-work"
        )
        sheet = root / "joined-chart/offline-paint-transfer.png"
        source = root / "input.jpg"
        reg = files.read_json(archive / "registration.json")
        seal = files.read_json(archive / "appearance/observation-seal.json")
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            work = out / "sf3d-observed-work"
            base = work / "native"
            shutil.copytree(archive / "native", base)
            account = work / "appearance"
            account.mkdir()
            setup = work / "test-setup.json"
            files.atomic_json(
                setup, {"pipeline": job.PIPELINE, "agreement": agreement(job.PIPELINE)}
            )

            def synthetic_transport(photo, guide, manifest, account_dir, **kwargs):
                joined_chart_provider.chart_consent(kwargs["setup"], kwargs["pipeline"])
                record = joined_chart_provider._identity(
                    files.read_json(manifest),
                    photo.read_bytes(),
                    guide.read_bytes(),
                    io.BytesIO(),
                    job.PIPELINE,
                )
                record.update(
                    state="completed",
                    attempts=1,
                    returncode=0,
                    thread_id="00000000-0000-4000-8000-000000000001",
                    sheet_sha256=files.digest(sheet),
                    generated_sha256=files.digest(sheet),
                    sheet_size=guided_provider._sheet(sheet.read_bytes()),
                )
                (account_dir / "guided-paint").mkdir()
                files.atomic_json(account_dir / "guided-paint/receipt.json", record)
                return {"sheet": sheet, "provenance": record}

            args = (
                out,
                source,
                base,
                files.read_json(base / "settings.json"),
                reg,
                seal["visual_brief"],
                seal["receipt"],
                None,
            )
            with (
                patch.object(
                    joined_chart_provider,
                    "generate_chart",
                    side_effect=synthetic_transport,
                ) as fake,
                patch(
                    "requests.sessions.Session.request",
                    side_effect=AssertionError("remote request"),
                ),
            ):
                meta = job._chart_finish(args, setup, account, lambda stage: None)
                self.assertEqual(fake.call_count, 1)
            self.assertTrue(job.valid(out, meta))
            self.assertEqual(meta["appearance_mode"], "guided_chart")
            self.assertEqual(
                files.digest(out / "shape-master.glb"),
                files.digest(base / "provider-native.glb"),
            )
            self.assertEqual(
                files.digest(out / "inferred-paint.png"), files.digest(sheet)
            )
            self.assertEqual(
                files.read_json(out / "chart-paint.json")["locked_changed"], 0
            )

    def test_real_composer_emits_an_inferred_alternative_with_zero_reposts(self):
        import base64
        import json
        import struct

        # Scientific libraries may run local capability probes during first import.
        # Load the shipped plugin before prohibiting all subprocesses throughout the job.
        from engine import inferred_cavity

        self.assertTrue(callable(inferred_cavity.editor))
        root = Path(os.environ["STUDIO_SF3D_SHAPE_REPLAY"])
        source = root / "input.jpg"
        work = (
            root
            / "run/store/b7bc98b1-c8d3-4ccd-8e40-7700ad1abbcb/output/sf3d-observed-work"
        )
        frozen = root / "run/evidence/native-fit-review/coherent"
        setup = root / "run/store/engine.json"
        src = files.digest(source)
        self.assertEqual(
            src, "f1b9008ad284105fdc3f5fd7bb27b3af7872f1e3a1c2b0f5ea87d4614984c879"
        )
        with contextlib.ExitStack() as stack:
            retained = os.getenv("STUDIO_SF3D_SHAPE_REPLAY_OUTPUT")
            if retained:
                out = Path(retained)
                out.mkdir(exist_ok=False)
            else:
                out = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            shutil.copytree(
                work,
                out / "sf3d-observed-work",
                ignore=shutil.ignore_patterns("run.lock"),
            )
            for method in ("requests.sessions.Session.request", "subprocess.Popen"):
                stack.enter_context(
                    patch(
                        method,
                        side_effect=AssertionError("Unpermitted external submission"),
                    )
                )
            meta = job.run(source, out, setup=setup)
            self.assertTrue(job.valid(out, meta))
            self.assertEqual(meta["shape_hypothesis"]["decision"], "alternative")
            report = files.read_json(out / "shape-hypothesis.json")
            self.assertEqual(report["source_sha256"], src)
            self.assertEqual(meta["glb_sha256"], files.digest(frozen / "model.glb"))
            self.assertEqual(
                report["candidate"]["sha256"],
                files.digest(out / "shape-alternative.glb"),
            )
            self.assertFalse(report["dimensions_observed"])
            self.assertGreater(report["candidate"]["new_faces"], 0)
            self.assertEqual(
                report["painted_before"]["faces"]
                - report["candidate"]["removed_faces"]
                + report["candidate"]["new_faces"],
                report["candidate"]["faces"],
            )
            raw = (out / "shape-alternative.glb").read_bytes()
            size = struct.unpack_from("<I", raw, 12)[0]
            editable = json.loads((out / "shape-alternative.gltf").read_bytes())[
                "buffers"
            ][0]
            self.assertEqual(
                base64.b64decode(editable["uri"].split(",", 1)[1]),
                raw[28 + size : 28 + size + editable["byteLength"]],
            )
            self.assertEqual(job.run(source, out, setup=setup), meta)
