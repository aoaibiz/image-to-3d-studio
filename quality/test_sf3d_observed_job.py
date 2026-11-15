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

    def test_uncertain_receipt_or_wrong_paint_does_not_auto_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "provider/receipt.json"
            for idx in range(2):
                with self.assertRaisesRegex(ValueError, "local preflight"):
                    with job._request(root, "paint", receipt):
                        raise ValueError("local preflight")
            receipt.parent.mkdir()
            with self.assertRaisesRegex(RuntimeError, "lost acknowledgement"):
                with job._request(root, "paint", receipt):
                    files.atomic_json(receipt, {"state": "submitted_or_ambiguous"})
                    raise RuntimeError("lost acknowledgement")
            with job._request(root, "paint", receipt):
                pass
            receipt.unlink()
            with self.assertRaises(job.StageError):
                with job._request(root, "paint", receipt):
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
        from engine import sf3d_provider, sf3d_shape_job
        from quality.test_sf3d_provider import Replay
        from studio.configure import agreement

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
            result = job.run(src, out, setup=setup)
            self.assertTrue(job.valid(out, result))
            self.assertEqual(result["appearance_mode"], "guided_region")
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
