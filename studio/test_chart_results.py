import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from studio.chart_results import CHART_PAINT_FILES, chart_files, publish_chart_hold, read_chart_hold


class ChartResultsTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.out = Path(temp.name) / "output"
        self.out.mkdir()
        self.source = hashlib.sha256(b"test source").hexdigest()
        (self.out.parent / "input.image").write_bytes(b"test source")
        (self.out / "model.glb").write_bytes(b"test transport bytes")
        self.sha = hashlib.sha256((self.out / "model.glb").read_bytes()).hexdigest()
        self.report = {"pipeline": "triposg-codex-surface-chart-v1", "source_sha256": self.source,
                       "shape_sha256":"d"*64, "base_sha256":self.sha, "output_sha256":self.sha}
        self.settings = {"artifact_sha256": dict.fromkeys(
            {"shape-master.glb","generation.json","appearance.json","chart-result.json",*CHART_PAINT_FILES}, self.sha)}
        (self.out / "settings.json").write_text(json.dumps(self.settings))
        checked = patch("studio.chart_results._checked", return_value=self.report)
        self.validate = checked.start()
        self.addCleanup(checked.stop)

    def test_mode_specific_allowlist(self):
        for decision, reason, calls in (("local_continuity","measured_continuity",0),
                                        ("inferred_chart","one_inferred_chart",1)):
            self.report.update(decision=decision, reason=reason, paint_attempts=calls)
            names, summary = chart_files(self.out, self.settings, self.source)
            self.assertIn("chart-result.json", names)
            self.assertEqual(set(names) & CHART_PAINT_FILES, CHART_PAINT_FILES if calls else set())
            self.assertEqual(summary["paint_attempts"], calls)
        self.report["paint_attempts"] = 0
        self.assertIsNone(chart_files(self.out, self.settings, self.source))
        self.validate.side_effect = ValueError("unverified engine output")
        self.assertIsNone(chart_files(self.out, self.settings, self.source))

    def test_diagnostic_public_fields_and_content_binding(self):
        self.report.update(decision="diagnostic_source_model", reason="provider_unconfirmed", paint_attempts=1,
                           private_note="internal only")
        self.assertIsNone(chart_files(self.out, self.settings, self.source))
        publish_chart_hold(self.out, self.settings, self.source)
        hold, names = read_chart_hold(self.out, self.source)
        self.assertEqual(names, ["review-hold.json","review-native.glb"])
        self.assertEqual(hold["reason_code"], "provider_unconfirmed")
        self.assertNotIn("private_note", hold)
        self.assertEqual(hold["preview_sha256"], self.sha)
        (self.out/"review-native.glb").write_bytes(b"corrupt")
        self.assertIsNone(read_chart_hold(self.out, self.source))
        self.report["output_sha256"] = "f"*64
        with self.assertRaises(ValueError):
            publish_chart_hold(self.out, self.settings, self.source)
        self.report["output_sha256"] = self.sha
        self.report["reason"] = "internal path or error"
        with self.assertRaises(ValueError):
            publish_chart_hold(self.out, self.settings, self.source)
