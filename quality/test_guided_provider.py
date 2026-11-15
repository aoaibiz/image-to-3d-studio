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
        self.envelope = {"pipeline": "triposg-codex-guided-v1", "agreement": provider.AGREEMENT}
        self.config.write_text(json.dumps(self.envelope), encoding="utf8")
        self.photo = self.root / "private-original-name.jpg"
        self.guide = self.root / "layout.png"
        self.manifest = self.root / "guide.json"
        Image.new("RGB", (120, 110), "green").save(self.photo)
        Image.new("RGB", (1536, 768), "gray").save(self.guide)
        self.desc = {
            "photo_sha": hashlib.sha256(self.photo.read_bytes()).hexdigest(),
            "guide_sha": hashlib.sha256(self.guide.read_bytes()).hexdigest(),
            "model_sha": "a"*64,
            "camera": {"rear": {"angles": [130,12]}, "opposite": {"angles": [-148,10]}},
        }
        self.manifest.write_text(json.dumps(self.desc),encoding="utf8")
        self.job = self.root / "job"
        self.thread = str(uuid.uuid4())
        context = patch.dict(os.environ, {
            "CODEX_HOME": str(self.home),
            "OPENAI_API_KEY": "fixture",
            "CODEX_API_KEY": "fixture",
            "OPENAI_BASE_URL": "https://invalid.example",
        })
        context.start()
        self.addCleanup(context.stop)

    def generate(self):
        return provider.generate_guided_paint(self.photo, self.guide, self.manifest, self.job, setup=self.config)

    def cli(self, *, create_output):
        def start(cmd, **kwargs):
            work = Path(kwargs["cwd"])
            self.assertEqual(sorted(p.name for p in work.iterdir()), ["appearance.png", "geometry.png"])
            self.assertEqual([cmd[i+1] for i,t in enumerate(cmd) if t=="--image"], ["geometry.png","appearance.png"])
            self.assertNotIn(str(self.photo), cmd)
            self.assertIn('forced_login_method="chatgpt"',cmd)
            self.assertIn('model_provider="openai"',cmd)
            self.assertIn('mcp_servers={}',cmd)
            for key in ("OPENAI_API_KEY","CODEX_API_KEY","OPENAI_BASE_URL"):
                self.assertNotIn(key,kwargs["env"])
            submitted = json.loads((self.job/"guided-paint"/"receipt.json").read_text(encoding="utf8"))
            self.assertEqual(submitted["state"], "submitted")
            self.assertEqual(submitted["attempts"], 1)
            if create_output:
                target = self.home/"generated_images"/self.thread
                target.mkdir(parents=True)
                Image.new("RGB",(128,64),"blue").save(target/"tool_1.png")
            events=[{"type":"thread.started","thread_id":self.thread},{"type":"turn.completed"}]
            return Mock(stdout=io.BytesIO(b"".join(json.dumps(e).encode()+b"\n" for e in events)),wait=lambda:0,poll=lambda:0)
        return start

    def test_personal_login_gate_refuses_api_key_before_diagnostic_feature_check(self):
        def shell(command, **kwargs):
            text="codex-cli 1.0" if command[-1] == "--version" else "Logged in using an API key"
            return Mock(returncode=0,stdout=text,stderr="")
        with patch.object(provider,"_which",return_value="codex"), patch.object(provider.subprocess,"run",side_effect=shell) as call:
            result=provider.preflight()
        self.assertFalse(result["available"])
        self.assertFalse(result["api_key_route"])
        self.assertEqual(call.call_count,2)

    def test_only_exact_consent_envelope_can_submit(self):
        with patch.object(provider.subprocess,"Popen") as start:
            for invalid in (provider.AGREEMENT,{"pipeline":"triposg","agreement":provider.AGREEMENT},{}):
                self.config.write_text(json.dumps(invalid))
                with self.assertRaises(provider.TurnaroundProviderError):
                    self.generate()
            start.assert_not_called()

    def test_consent_is_checked_again_immediately_before_submission(self):
        save=provider._record
        def revoke(work, receipt):
            save(work,receipt)
            self.config.write_text('{"pipeline":"triposg"}')
        with patch.object(provider,"preflight",return_value={"available":True,"cli_version":"test"}), patch.object(provider,"_record",side_effect=revoke), patch.object(provider.subprocess,"Popen") as start:
            with self.assertRaises(provider.TurnaroundProviderError):
                self.generate()
        start.assert_not_called()
        receipt=json.loads((self.job/"guided-paint"/"receipt.json").read_text(encoding="utf8"))
        self.assertEqual((receipt["state"],receipt["attempts"]),("blocked_before_submission",0))

    def test_no_traffic_on_mismatched_layout_or_non_object_manifest(self):
        with patch.object(provider.subprocess,"Popen") as start:
            for value in (dict(self.desc,guide_sha="0"*64),[]):
                self.manifest.write_text(json.dumps(value))
                with self.assertRaises(provider.TurnaroundProviderError):
                    self.generate()
            start.assert_not_called()

    def test_one_submission_two_images_and_verified_artifact_on_reentry(self):
        with patch.object(provider,"preflight",return_value={"available":True,"cli_version":"test"}), patch.object(provider.subprocess,"Popen",side_effect=self.cli(create_output=True)) as start:
            result=self.generate()
            self.assertEqual(self.generate(),result)
            self.assertEqual(start.call_count,1)
            result["sheet"].write_bytes(b"modified")
            with self.assertRaises(provider.TurnaroundProviderError):
                self.generate()
        self.assertEqual(result["provenance"]["appearance_quality"],"needs_review")
        self.assertEqual(len(result["provenance"]["uploaded_png_sha256"]),2)

    def test_completed_turn_without_image_never_submits_again(self):
        with patch.object(provider,"preflight",return_value={"available":True,"cli_version":"test"}), patch.object(provider.subprocess,"Popen",side_effect=self.cli(create_output=False)) as start:
            for _ in range(2):
                with self.assertRaisesRegex(provider.TurnaroundProviderError,"生成 PNG"):
                    self.generate()
            self.assertEqual(start.call_count,1)
        receipt=json.loads((self.job/"guided-paint"/"receipt.json").read_text(encoding="utf8"))
        self.assertEqual(receipt["thread_id"],self.thread)
        self.assertEqual(receipt["state"],"copy_pending")
