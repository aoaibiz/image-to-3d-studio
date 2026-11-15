import hashlib
import io
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from engine import guided_provider as provider


class GuidedProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / "personal-codex"
        self.config = self.root / "engine.json"
        self.envelope = {
            "pipeline": "triposg-codex-guided-v1",
            "agreement": provider.AGREEMENT,
        }
        self.config.write_text(json.dumps(self.envelope), encoding="utf8")
        self.photo = self.root / "private-original-name.jpg"
        self.guide = self.root / "layout.png"
        self.manifest = self.root / "guide.json"
        Image.new("RGB", (120, 110), "green").save(self.photo)
        Image.new("RGB", (1536, 768), "gray").save(self.guide)
        self.desc = {
            "photo_sha": hashlib.sha256(self.photo.read_bytes()).hexdigest(),
            "guide_sha": hashlib.sha256(self.guide.read_bytes()).hexdigest(),
            "model_sha": "a" * 64,
            "camera": {
                "rear": {"angles": [130, 12]},
                "opposite": {"angles": [-148, 10]},
            },
        }
        self.manifest.write_text(json.dumps(self.desc), encoding="utf8")
        self.job = self.root / "job"
        self.thread = str(uuid.uuid4())
        context = patch.dict(
            os.environ,
            {
                "CODEX_HOME": str(self.home),
                "OPENAI_API_KEY": "fixture",
                "CODEX_API_KEY": "fixture",
                "OPENAI_BASE_URL": "https://invalid.example",
            },
        )
        context.start()
        self.addCleanup(context.stop)

    def generate(self):
        return provider.generate_guided_paint(
            self.photo, self.guide, self.manifest, self.job, setup=self.config
        )

    def cli(self, *, create_output, output_size=(128, 64)):
        def start(cmd, **kwargs):
            work = Path(kwargs["cwd"])
            self.assertEqual(
                sorted(p.name for p in work.iterdir()),
                ["reference.png"],
            )
            self.assertEqual(
                [cmd[i + 1] for i, t in enumerate(cmd) if t == "--image"],
                ["reference.png"],
            )
            self.assertNotIn(str(self.photo), cmd)
            self.assertIn('forced_login_method="chatgpt"', cmd)
            self.assertIn('model_provider="openai"', cmd)
            self.assertIn("mcp_servers={}", cmd)
            for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL"):
                self.assertNotIn(key, kwargs["env"])
            submitted = json.loads(
                (self.job / "guided-paint" / "receipt.json").read_text(encoding="utf8")
            )
            self.assertEqual(submitted["state"], "submitted")
            self.assertEqual(submitted["attempts"], 1)
            if create_output:
                target = self.home / "generated_images" / self.thread
                target.mkdir(parents=True)
                Image.new("RGB", output_size, "blue").save(target / "tool_1.png")
            events = [
                {"type": "thread.started", "thread_id": self.thread},
                {"type": "turn.completed"},
            ]
            return Mock(
                stdout=io.BytesIO(
                    b"".join(json.dumps(e).encode() + b"\n" for e in events)
                ),
                wait=lambda: 0,
                poll=lambda: 0,
            )

        return start

    def test_personal_login_gate_refuses_api_key_before_diagnostic_feature_check(self):
        def shell(command, **kwargs):
            text = (
                "codex-cli 1.0"
                if command[-1] == "--version"
                else "Logged in using an API key"
            )
            return Mock(returncode=0, stdout=text, stderr="")

        with (
            patch.object(provider, "_which", return_value="codex"),
            patch.object(provider.subprocess, "run", side_effect=shell) as call,
        ):
            result = provider.preflight()
        self.assertFalse(result["available"])
        self.assertFalse(result["api_key_route"])
        self.assertEqual(call.call_count, 2)

    def test_only_exact_consent_envelope_can_submit(self):
        with patch.object(provider.subprocess, "Popen") as start:
            for invalid in (
                provider.AGREEMENT,
                {"pipeline": "triposg", "agreement": provider.AGREEMENT},
                {},
            ):
                self.config.write_text(json.dumps(invalid))
                with self.assertRaises(provider.TurnaroundProviderError):
                    self.generate()
            start.assert_not_called()

    def test_consent_is_checked_again_immediately_before_submission(self):
        save = provider._record

        def revoke(work, receipt):
            save(work, receipt)
            self.config.write_text('{"pipeline":"triposg"}')

        with (
            patch.object(
                provider,
                "preflight",
                return_value={"available": True, "cli_version": "test"},
            ),
            patch.object(provider, "_record", side_effect=revoke),
            patch.object(provider.subprocess, "Popen") as start,
            self.assertRaises(provider.TurnaroundProviderError),
        ):
            self.generate()
        start.assert_not_called()
        receipt = json.loads(
            (self.job / "guided-paint" / "receipt.json").read_text(encoding="utf8")
        )
        self.assertEqual(
            (receipt["state"], receipt["attempts"]), ("blocked_before_submission", 0)
        )

    def test_no_traffic_on_mismatched_layout_or_non_object_manifest(self):
        with patch.object(provider.subprocess, "Popen") as start:
            for value in (dict(self.desc, guide_sha="0" * 64), []):
                self.manifest.write_text(json.dumps(value))
                with self.assertRaises(provider.TurnaroundProviderError):
                    self.generate()
            start.assert_not_called()

    def test_one_reference_board_and_verified_artifact_on_reentry(self):
        with (
            patch.object(
                provider,
                "preflight",
                return_value={"available": True, "cli_version": "test"},
            ),
            patch.object(
                provider.subprocess, "Popen", side_effect=self.cli(create_output=True)
            ) as start,
        ):
            result = self.generate()
            self.assertEqual(self.generate(), result)
            self.assertEqual(start.call_count, 1)
            result["sheet"].write_bytes(b"modified")
            with self.assertRaises(provider.TurnaroundProviderError):
                self.generate()
        self.assertEqual(result["provenance"]["appearance_quality"], "needs_review")
        self.assertEqual(len(result["provenance"]["uploaded_png_sha256"]), 1)

    def brief(self):
        return {
            "method": "photo-appearance-qwen3-vl-2b-v1",
            "source_sha256": self.desc["photo_sha"],
            "visible_palette": ["blue", "white"],
            "optical_finish_candidate": "glossy",
            "layout_candidate": "focal_and_repeated",
            "observed_face_cues": {
                "focal": {"family": "abstract", "location": "center"},
                "repeated": {"family": "botanical", "location": "upper_border"},
            },
            "visible_structures": [
                {"kind": "projecting_assembly", "location": "center"}
            ],
            "visible_surround": {"color": "ivory", "pigment_layout": "grain_or_mottle"},
            "unseen_surface_observed": False,
        }

    def test_observations_reach_actual_command_and_immutable_receipt(self):
        observations = self.brief()
        self.manifest.write_text(json.dumps(dict(self.desc, visual_brief=observations)))
        with (
            patch.object(
                provider,
                "preflight",
                return_value={"available": True, "cli_version": "test"},
            ),
            patch.object(
                provider.subprocess, "Popen", side_effect=self.cli(create_output=True)
            ) as start,
        ):
            result = self.generate()
            prompt = start.call_args.args[0][-1]
            self.assertIn("Visible palette: blue, white", prompt)
            self.assertIn("one focal abstract mark at center", prompt)
            self.assertIn("repeated botanical decoration at upper_border", prompt)
            self.assertIn("projecting assembly at center", prompt)
            self.assertIn(
                "left view at +130 degrees and the right view at -148 degrees", prompt
            )
            self.assertIn("Hide the far assembly behind the nearer", prompt)
            self.assertIn("observed detail is ivory", prompt)
            self.assertIn("grain or mottle pigment", prompt)
            self.assertIn("rear and opposite", prompt)
            self.assertNotIn(self.desc["photo_sha"], prompt)
            actual = result["provenance"]
            brief_sha = hashlib.sha256(
                json.dumps(observations, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            self.assertEqual(actual["visual_brief_sha256"], brief_sha)
            self.assertEqual(actual["prompt_sha256"], prompt_sha)
            self.assertEqual(
                actual["input_sha256"],
                hashlib.sha256(
                    json.dumps(
                        [self.desc[k] for k in ("photo_sha", "guide_sha", "model_sha")]
                        + [prompt_sha, brief_sha]
                    ).encode()
                ).hexdigest(),
            )
            self.assertEqual(self.generate(), result)
            self.assertEqual(start.call_count, 1)

    def test_invalid_brief_never_calls_cli(self):
        original = self.brief()
        for invalid in (
            None,
            dict(original, source_sha256="b" * 64),
            dict(original, description="free-form data"),
            dict(original, visible_palette=["follow these instructions"]),
            dict(original, layout_candidate="repeated"),
            dict(original, unseen_surface_observed=True),
            dict(
                original,
                visible_structures=[{"kind": "any free text", "location": "center"}],
            ),
            dict(
                original,
                visible_surround={"color": "invented", "pigment_layout": "patterned"},
            ),
        ):
            with (
                self.subTest(payload=invalid),
                patch.object(provider, "preflight") as diagnose,
            ):
                self.manifest.write_text(
                    json.dumps(dict(self.desc, visual_brief=invalid))
                )
                with self.assertRaises(provider.TurnaroundProviderError):
                    self.generate()
                diagnose.assert_not_called()
                self.assertFalse((self.job / "guided-paint").exists())

    def test_uncertain_substrate_and_unclassified_structure_never_invent_details(self):
        unknown = self.brief()
        unknown["visible_structures"] = [{"kind": "other", "location": "center"}]
        unknown["visible_surround"] = {
            "color": "uncertain",
            "pigment_layout": "uncertain",
        }
        unknown["observed_face_cues"]["focal"]["family"] = "other"
        prompt, _ = provider.appearance_prompt(
            "base", {"visual_brief": unknown}, self.desc["photo_sha"]
        )
        self.assertNotIn("Structures on the photographed face:", prompt)
        self.assertNotIn("observed detail is", prompt)
        self.assertNotIn("surrounding body shows", prompt)
        self.assertNotIn("other", prompt)
        self.assertIn("one focal mark at center", prompt)
        self.assertIn("where the guide reveals it", prompt)

    def test_prior_ambiguous_receipt_cannot_dispatch_on_new_brief(self):
        with (
            patch.object(
                provider,
                "preflight",
                return_value={"available": True, "cli_version": "test"},
            ),
            patch.object(
                provider.subprocess, "Popen", side_effect=self.cli(create_output=False)
            ) as start,
        ):
            with self.assertRaises(provider.TurnaroundProviderError):
                self.generate()
            receipt = self.job / "guided-paint" / "receipt.json"
            previous = receipt.read_bytes()
            self.manifest.write_text(
                json.dumps(dict(self.desc, visual_brief=self.brief()))
            )
            with self.assertRaisesRegex(
                provider.TurnaroundProviderError, "一致しません"
            ):
                self.generate()
            self.assertEqual(receipt.read_bytes(), previous)
            self.assertEqual(start.call_count, 1)

    def legacy_receipt(self, state):
        # Independent reconstruction of df946d9's wire inputs for these EXIF-free fixtures.
        hashes = []
        for path in (self.guide, self.photo):
            data = io.BytesIO()
            with Image.open(path) as image:
                image.convert("RGBA").save(data, "PNG")
            hashes.append(hashlib.sha256(data.getvalue()).hexdigest())
        prompt_sha = "7336850e18be2f50e20ddfdc24817afb5ed75573c9b111789905976b77dcc896"
        identity = [self.desc[k] for k in ("photo_sha", "guide_sha", "model_sha")] + [
            prompt_sha
        ]
        receipt = {
            "state": state,
            "thread_id": self.thread,
            "returncode": 0,
            "provider": provider.PROVIDER,
            "recipient": "OpenAI",
            "api_key_route": False,
            "input_sha256": hashlib.sha256(json.dumps(identity).encode()).hexdigest(),
            "prompt_sha256": prompt_sha,
            "photo_sha256": self.desc["photo_sha"],
            "geometry_sha256": self.desc["model_sha"],
            "guide_sha256": self.desc["guide_sha"],
            "cameras": self.desc["camera"],
            "uploaded_attachment_order": ["geometry", "photograph"],
            "uploaded_png_sha256": hashes,
        }
        directory = self.job / "guided-paint"
        directory.mkdir(parents=True)
        sheet = directory / "sheet.png"
        Image.new("RGB", (128, 64), "white").save(sheet, "PNG")
        receipt["sheet_sha256"] = hashlib.sha256(sheet.read_bytes()).hexdigest()
        path = directory / "receipt.json"
        path.write_text(json.dumps(receipt), encoding="utf8")
        return path, receipt, sheet

    def test_verified_two_attachment_completion_is_replayed_without_new_traffic(self):
        path, receipt, sheet = self.legacy_receipt("completed")
        before = path.read_bytes()
        with patch.object(provider, "preflight") as diagnose:
            result = self.generate()
            self.assertEqual(result, {"sheet": sheet, "provenance": receipt})
            self.assertEqual(path.read_bytes(), before)
            diagnose.assert_not_called()

    def test_historical_brief_board_is_recovered_only_when_all_bound_inputs_match(self):
        path, old, sheet = self.legacy_receipt("copy_pending")
        brief = self.brief()
        del brief["visible_structures"], brief["visible_surround"]
        self.manifest.write_text(json.dumps(dict(self.desc, visual_brief=brief)))
        # Produced by running the prompt builder at commit 906ad8b with this brief.
        prompt_sha = "2a988751b84ced69449dc0be26db84196421034aede8b488ac78eeca8d457a72"
        brief_sha = hashlib.sha256(
            json.dumps(brief, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        stream = io.BytesIO()
        board = provider._reference_board(
            self.photo.read_bytes(), self.guide.read_bytes(), stream
        )
        uploaded = hashlib.sha256(stream.getvalue()).hexdigest()
        identity = [self.desc[k] for k in ("photo_sha", "guide_sha", "model_sha")] + [
            prompt_sha,
            brief_sha,
        ]
        old.update(
            input_sha256=hashlib.sha256(json.dumps(identity).encode()).hexdigest(),
            prompt_sha256=prompt_sha,
            visual_brief_sha256=brief_sha,
            uploaded_attachment_order=["reference_board"],
            reference_board=board,
            uploaded_png_sha256=[uploaded],
        )
        target = self.home / "generated_images" / self.thread / "earlier.png"
        target.parent.mkdir(parents=True)
        sheet.rename(target)
        with patch.object(provider, "preflight") as diagnose:
            for mutation in (
                {"uploaded_png_sha256": ["9" * 64]},
                {"visual_brief_sha256": "9" * 64},
                {"reference_board": {}},
                {"prompt_sha256": "9" * 64},
            ):
                path.write_text(json.dumps(dict(old, **mutation)), encoding="utf8")
                with self.assertRaisesRegex(
                    provider.TurnaroundProviderError, "一致しません"
                ):
                    self.generate()
                self.assertFalse(sheet.exists())
            path.write_text(json.dumps(old), encoding="utf8")
            result = self.generate()
            self.assertEqual(result["sheet"].read_bytes(), target.read_bytes())
            self.assertEqual(result["provenance"]["prompt_sha256"], prompt_sha)
            self.assertEqual(self.generate(), result)
            diagnose.assert_not_called()

    def test_verified_two_attachment_copy_pending_recovers_thread_artifact_only(self):
        _, old, sheet = self.legacy_receipt("copy_pending")
        target = self.home / "generated_images" / self.thread / "historical.png"
        target.parent.mkdir(parents=True)
        sheet.rename(target)
        with patch.object(provider, "preflight") as diagnose:
            result = self.generate()
            self.assertEqual(result["sheet"].read_bytes(), target.read_bytes())
            self.assertEqual(result["provenance"]["state"], "completed")
            self.assertEqual(
                result["provenance"]["prompt_sha256"], old["prompt_sha256"]
            )
            self.assertEqual(
                result["provenance"]["uploaded_png_sha256"], old["uploaded_png_sha256"]
            )
            diagnose.assert_not_called()

    def test_unknown_or_ambiguous_historical_request_cannot_trigger_submission(self):
        path, old, _ = self.legacy_receipt("unconfirmed")
        with patch.object(provider, "preflight") as diagnose:
            for change in (
                {},
                {
                    "state": "completed",
                    "uploaded_png_sha256": list(reversed(old["uploaded_png_sha256"])),
                },
                {
                    "state": "completed",
                    "uploaded_attachment_order": ["photograph", "geometry"],
                },
                {"state": "completed", "input_sha256": "9" * 64},
                {"state": "completed", "prompt_sha256": "9" * 64},
                {"state": "completed", "cameras": {}},
            ):
                with self.subTest(change=change):
                    path.write_text(json.dumps(dict(old, **change)), encoding="utf8")
                    snapshot = path.read_bytes()
                    with self.assertRaises(provider.TurnaroundProviderError):
                        self.generate()
                    self.assertEqual(path.read_bytes(), snapshot)
            diagnose.assert_not_called()

    def test_board_preserves_actual_normalized_photo_and_render_pixels(self):
        board = self.root / "board.png"
        evidence = provider._reference_board(
            self.photo.read_bytes(), self.guide.read_bytes(), board
        )
        with Image.open(board) as image:
            self.assertFalse(image.getexif())
            for source, name in ((self.photo, "photograph"), (self.guide, "geometry")):
                x, y, w, h = evidence["panels"][name]["xywh"]
                with Image.open(source) as original:
                    self.assertEqual((w, h), original.size)
                    self.assertEqual(
                        image.crop((x, y, x + w, y + h)).tobytes(),
                        original.convert("RGB").tobytes(),
                    )

    def test_tool_failure_and_missing_artifact_are_distinct_and_no_raw_text_persists(
        self,
    ):
        receipt = {}
        event = {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": "I do not have access to an image_generation tool in this session. PRIVATE-DUMMY",
            },
        }
        provider._notice(event, receipt)
        self.assertEqual(receipt["provider_hint"], "unavailable")
        self.assertNotIn("PRIVATE-DUMMY", json.dumps(receipt))

    def test_completed_turn_without_image_never_submits_again(self):
        with (
            patch.object(
                provider,
                "preflight",
                return_value={"available": True, "cli_version": "test"},
            ),
            patch.object(
                provider.subprocess, "Popen", side_effect=self.cli(create_output=False)
            ) as start,
        ):
            for _ in range(2):
                with self.assertRaisesRegex(
                    provider.TurnaroundProviderError, "生成 PNG"
                ):
                    self.generate()
            self.assertEqual(start.call_count, 1)
        receipt = json.loads(
            (self.job / "guided-paint" / "receipt.json").read_text(encoding="utf8")
        )
        self.assertEqual(receipt["thread_id"], self.thread)
        self.assertEqual(receipt["state"], "copy_pending")
        self.assertEqual(receipt["provider_hint"], "no_image_unconfirmed")

    def test_wrong_layout_saved_as_unapproved_without_retry(self):
        with (
            patch.object(
                provider,
                "preflight",
                return_value={"available": True, "cli_version": "test"},
            ),
            patch.object(
                provider.subprocess,
                "Popen",
                side_effect=self.cli(create_output=True, output_size=(64, 128)),
            ) as start,
        ):
            for _ in range(2):
                with self.assertRaisesRegex(
                    provider.TurnaroundProviderError, "寸法基準"
                ):
                    self.generate()
            self.assertEqual(start.call_count, 1)
        work = self.job / "guided-paint"
        receipt = json.loads((work / "receipt.json").read_text(encoding="utf8"))
        self.assertEqual(receipt["state"], "rejected_layout")
        self.assertFalse((work / "sheet.png").exists())
        self.assertEqual(
            receipt["generated_sha256"],
            hashlib.sha256((work / "unusable-output.png").read_bytes()).hexdigest(),
        )
