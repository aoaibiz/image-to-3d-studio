"""Verify consent and the normal Studio job entry before any external work."""

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.pipeline import OBSERVED_PIPELINE, SG_CHART_PIPELINE
from studio.configure import agreement, configure, read_pipeline, valid_agreement
from studio.core import JobCore
from studio.server import Server


class ChartRoute(unittest.TestCase):
    def test_http_engine_reports_local_shape_and_limits_the_chart(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = root / "jobs"
            configure(
                store,
                "triposg-chart",
                chatgpt_photo_analysis=True,
                allow_send_to_openai=True,
            )
            core = JobCore(root, directory=store, runner=lambda _: None)
            server = Server(("127.0.0.1", 0), core)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                conn = http.client.HTTPConnection(
                    "127.0.0.1", server.server_port, timeout=3
                )
                conn.request("GET", "/api/engine")
                res = conn.getresponse()
                data = json.loads(res.read())
                conn.close()
                self.assertEqual(res.status, 200)
                self.assertEqual(data["shape"], "local TripoSG")
                self.assertEqual(data["pipeline"], SG_CHART_PIPELINE)
                self.assertEqual(data["recipient"], "OpenAI ChatGPT")
                self.assertEqual(data["max_image_generations_per_job"], 1)
                self.assertEqual(
                    data["paint_guide_recipes"], ["source-anchored-surface-chart"]
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
                core.close()

    def test_separate_account_consent_and_pinned_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = root / "jobs"
            with self.assertRaises(ValueError):
                configure(store, "triposg-chart", chatgpt_photo_analysis=True)
            configure(
                store,
                "triposg-chart",
                chatgpt_photo_analysis=True,
                allow_send_to_openai=True,
            )
            self.assertEqual(read_pipeline(store), SG_CHART_PIPELINE)
            doc = json.loads((store / "engine.json").read_bytes())
            self.assertEqual(
                doc["agreement"]["paint_guide_recipes"],
                ["source-anchored-surface-chart"],
            )
            self.assertEqual(doc["agreement"]["max_image_generations_per_job"], 1)
            self.assertFalse(
                valid_agreement(
                    {
                        "pipeline": SG_CHART_PIPELINE,
                        "agreement": agreement(OBSERVED_PIPELINE),
                    },
                    SG_CHART_PIPELINE,
                )
            )
            core = JobCore(root, directory=store)
            started = threading.Event()
            calls = []

            def capture(job_id, command, stage):
                calls.append(command)
                started.set()
                raise RuntimeError("fixture ends before engine launch")

            try:
                with (
                    patch.object(core, "_command", side_effect=capture),
                    patch.object(core, "_validate_image"),
                ):
                    _, created = core.create(b"test bytes", "chart", SG_CHART_PIPELINE)
                    self.assertTrue(created)
                    self.assertTrue(started.wait(5))
                    self.assertIn("engine.guided_shape_job", calls[0])
                    self.assertEqual(
                        calls[0][calls[0].index("--pipeline") + 1], SG_CHART_PIPELINE
                    )
                    self.assertEqual(
                        calls[0][calls[0].index("--setup") + 1],
                        str(store / "engine.json"),
                    )
            finally:
                core.close()
