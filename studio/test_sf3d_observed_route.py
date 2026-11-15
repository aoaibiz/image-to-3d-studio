"""No remote inference in these tests. Exercise the same job store and API used by clients."""
import http.client
import itertools
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.guided_provider import TurnaroundProviderError, _require_consent
from engine.pipeline import PIPELINE, SF3D_OBSERVED_PIPELINE, SF3D_PIPELINE
from studio.configure import configure, read_pipeline
from studio.core import JobCore, PipelineConflict
from studio.server import Server


class CombinedRoutingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = self.root / "studio"
        self.flags = dict(allow_official_space=True, chatgpt_photo_analysis=True,
                          allow_send_to_openai=True)

    def test_each_grant_is_required_and_release_to_single_provider_revokes_reuse(self):
        for grants in itertools.product([False, True], repeat=3):
            if all(grants):
                continue
            with self.subTest(grants=grants), self.assertRaises(ValueError):
                configure(self.store, "sf3d-observed", **dict(zip(self.flags, grants)))
            self.assertEqual(read_pipeline(self.store), PIPELINE)
        configure(self.store, "sf3d-observed", **self.flags)
        self.assertEqual(read_pipeline(self.store), SF3D_OBSERVED_PIPELINE)
        setup = self.store / "engine.json"
        _require_consent(setup, SF3D_OBSERVED_PIPELINE)
        grant = json.loads(setup.read_text())
        grant["agreement"]["max_shape_inferences_per_job"] = True
        setup.write_text(json.dumps(grant))
        with self.assertRaises(ValueError):
            read_pipeline(self.store)
        with self.assertRaises(TurnaroundProviderError):
            _require_consent(setup, SF3D_OBSERVED_PIPELINE)
        for engine, flags in [(SF3D_PIPELINE, {"allow_official_space": True}),
                              ("triposg-observed", {"allow_send_to_openai": True, "chatgpt_photo_analysis": True}),
                              ("triposr", {})]:
            configure(self.store, engine, **flags)
            with self.assertRaises(TurnaroundProviderError):
                _require_consent(setup, SF3D_OBSERVED_PIPELINE)

    def test_real_store_uses_combined_entry_once_and_pins_job_to_engine(self):
        configure(self.store, "sf3d-observed", **self.flags)
        core = JobCore(self.root, directory=self.store)
        self.addCleanup(core.close)
        started = threading.Event()
        invokes = []

        def stop_before_external_work(job_id, command, stage):
            invokes.append((job_id, command, stage))
            started.set()
            raise RuntimeError("transport test stops before generation")

        with (patch.object(core, "_command", side_effect=stop_before_external_work),
              patch.object(core, "_validate_image")):
            job, created = core.create(b"local test", "one", SF3D_OBSERVED_PIPELINE)
            self.assertTrue(created)
            self.assertTrue(started.wait(3))
            self.assertNotIn("resolution", job["config"])
            self.assertIn("engine.sf3d_observed_job", invokes[0][1])
            self.assertNotIn("--resolution", invokes[0][1])
            self.assertIn(str(self.store / "engine.json"), invokes[0][1])
            again, created = core.create(b"local test", "one", SF3D_OBSERVED_PIPELINE)
            self.assertFalse(created)
            self.assertEqual(again["id"], job["id"])
            self.assertEqual(len(invokes), 1)
            with self.assertRaises(PipelineConflict):
                core.create(b"local test", "one", SF3D_PIPELINE)
            configure(self.store, "triposr")
            with self.assertRaises(PipelineConflict):
                core.create(b"local test", "fresh", SF3D_OBSERVED_PIPELINE)
            self.assertFalse(core.create(b"local test", "one", SF3D_OBSERVED_PIPELINE)[1])

    def test_doctor_and_engine_disclose_both_recipients_without_local_shape_weights(self):
        configure(self.store, "sf3d-observed", **self.flags)
        core = JobCore(self.root, directory=self.store, runner=lambda _: None)
        self.addCleanup(core.close)
        server = Server(("127.0.0.1", 0), core)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            def get(path):
                conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                try:
                    conn.request("GET", path)
                    response = conn.getresponse()
                    self.assertEqual(response.status, 200)
                    return json.loads(response.read())
                finally:
                    conn.close()

            engine = get("/api/engine")
            self.assertFalse(engine["local"])
            self.assertEqual(engine["recipients"], ["Stability AI / Hugging Face", "OpenAI ChatGPT"])
            with (patch("studio.server.shape_doctor", side_effect=AssertionError("Unexpected TripoSG")),
                  patch("engine.guided_provider.preflight", return_value={"available": True}),
                  patch("engine.codex_photo_observations.preflight", return_value={"available": False, "reason": "要ログイン"})):
                report = get("/api/doctor")
            self.assertFalse(report["ok"])
            self.assertEqual(report["recipients"], engine["recipients"])
            self.assertNotIn("local_visual_analysis", report["diagnostics"])
            self.assertTrue(report["diagnostics"]["guided_paint"]["available"])
            self.assertFalse(report["diagnostics"]["photo_observer"]["available"])
            self.assertNotIn("shape_python", report["checks"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
