"""Studio exposes an alternate shape only when the engine accepts the sealed result."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from studio.core import HYPOTHESIS_GLB, HYPOTHESIS_MASTER, HYPOTHESIS_REPORT, _hypothesis_files


class AlternateFilesTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.out = Path(temp.name)
        self.source = "a" * 64
        self.primary = "b" * 64
        self.other = "c" * 64
        self.report = {"decision": "alternative", "source_sha256": self.source,
                       "dimensions_observed": False, "visual_quality": "needs_review",
                       "candidate": {"sha256": self.other}, "painted_before": {"sha256": self.primary},
                       "delivered": {"sha256": self.primary}}
        blob = json.dumps(self.report).encode()
        (self.out / HYPOTHESIS_REPORT).write_bytes(blob)
        self.sealed = {HYPOTHESIS_REPORT: hashlib.sha256(blob).hexdigest(), HYPOTHESIS_GLB: self.other,
                       HYPOTHESIS_MASTER: "d" * 64, "model.glb": self.primary}
        self.selection = {"decision": "alternative", "report": HYPOTHESIS_REPORT,
                          "alternative_model": HYPOTHESIS_GLB, "alternative_editable": HYPOTHESIS_MASTER}
        self.settings = {"input_sha256": self.source, "glb_sha256": self.primary,
                         "artifact_sha256": self.sealed, "shape_hypothesis": self.selection}

    def test_known_names_only_and_core_validation_is_required(self):
        validate = Mock(return_value=True)
        names = _hypothesis_files(self.out, self.settings, self.source, validate)
        self.assertEqual(set(names), {HYPOTHESIS_GLB, HYPOTHESIS_MASTER, HYPOTHESIS_REPORT})
        validate.assert_called_once_with(self.out, self.settings)
        self.assertFalse(_hypothesis_files(self.out, self.settings, self.source, lambda *_: False))
        self.sealed["pre-hypothesis.glb"] = self.primary
        self.assertFalse(_hypothesis_files(self.out, self.settings, self.source, validate))

    def test_legacy_and_invalid_source_or_filename_do_not_activate(self):
        forbidden = Mock(side_effect=AssertionError("Unexpected validation"))
        self.assertFalse(_hypothesis_files(self.out, {}, self.source, forbidden))
        self.assertFalse(_hypothesis_files(self.out, self.settings, "0" * 64, forbidden))
        self.selection["alternative_model"] = "../input.image"
        self.assertFalse(_hypothesis_files(self.out, self.settings, self.source, forbidden))

    def test_measured_dimensions_and_mismatched_master_do_not_activate(self):
        for overrides in [{"dimensions_observed": True}, {"candidate": {"sha256": self.primary}},
                          {"delivered": {"sha256": self.other}}, {"source_sha256": self.primary}]:
            (self.out / HYPOTHESIS_REPORT).write_text(json.dumps(self.report | overrides))
            self.assertFalse(_hypothesis_files(self.out, self.settings, self.source, lambda *_: True))
