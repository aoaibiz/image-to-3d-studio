import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from studio.core import CONTINUITY_FILES, SF3D_OBSERVED_PIPELINE, _local_continuity


class LocalContinuityTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)
        self.source = "a" * 64
        self.settings = {"appearance_mode": "local_continuity", "pipeline": SF3D_OBSERVED_PIPELINE,
                         "input_sha256": self.source, "artifact_sha256": dict.fromkeys(CONTINUITY_FILES, "b" * 64)}
        self.counts = {"external_shape_generations": 1, "external_photo_observations": 1,
                       "external_image_generations": 0}
        self.history = self.out / "generation.json"
        self.history.write_text(json.dumps(self.counts))

    def test_source_bound_sealed_output_and_zero_image_generations(self):
        validate = Mock(return_value=True)
        self.assertTrue(_local_continuity(self.out, self.settings, self.source, validate))
        validate.assert_called_once_with(self.out, self.settings)
        self.assertFalse(_local_continuity(self.out, self.settings, self.source, lambda *_: False))
        for invalid in [dict(external_image_generations=1), dict(external_image_generations=False),
                        dict(external_photo_observations=0), dict(external_shape_generations=None)]:
            with self.subTest(invalid=invalid):
                self.history.write_text(json.dumps(self.counts | invalid))
                self.assertFalse(_local_continuity(self.out, self.settings, self.source, validate))
        self.history.unlink()
        self.assertFalse(_local_continuity(self.out, self.settings, self.source, validate))

    def test_unrelated_and_unsealed_jobs_do_not_activate(self):
        validate = Mock(side_effect=AssertionError("should not verify an unrelated output"))
        for invalid in [dict(appearance_mode="native_material"), dict(appearance_mode="guided_full"),
                        dict(pipeline="other"), dict(input_sha256="c"*64), dict(artifact_sha256={})]:
            self.assertFalse(_local_continuity(self.out, self.settings | invalid, self.source, validate))
