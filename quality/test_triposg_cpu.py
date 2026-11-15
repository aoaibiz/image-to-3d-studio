"""Regression checks for local TripoSG staging; none substitute for an inference trial."""

import json
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest import mock
import subprocess
import sys

import numpy as np
from PIL import Image

from engine import triposg_cpu as engine


class TripoSGStateTests(unittest.TestCase):
    def test_parent_watch_allows_native_import_then_observes_pipe_eof(self):
        script = (
            "import sys,threading,time\n"
            "from engine.triposg_cpu import watch_supervisor\n"
            "threading.Thread(target=watch_supervisor,args=(sys.stdin.fileno(),),daemon=True).start()\n"
            "import numpy\n"
            "print('native-ready',flush=True)\n"
            "time.sleep(25)"
        )
        child = subprocess.Popen([sys.executable, "-u", "-c", script], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, text=True)
        observed = queue.Queue()
        reader = threading.Thread(target=lambda: observed.put(child.stdout.readline()), daemon=True)
        reader.start()
        try:
            self.assertEqual(observed.get(timeout=15).strip(), "native-ready")
            child.stdin.close()
            self.assertEqual(child.wait(timeout=4), 130)
        finally:
            if not child.stdin.closed:
                child.stdin.close()
            if child.poll() is None:
                child.terminate()
            child.wait()
            reader.join(timeout=2)
            child.stdout.close()

    def test_resume_uses_last_intact_checkpoint_for_the_same_job(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            older, newer = work / "flow-a.safetensors", work / "flow-b.safetensors"
            older.write_bytes(b"first")
            newer.write_bytes(b"second")
            engine.seal(older, "job-a", step=1)
            engine.seal(newer, "job-a", step=2)
            self.assertEqual(engine.latest_step(work, "job-a"), (2, newer))
            self.assertEqual(engine.latest_step(work, "job-b"), (0, None))
            newer.write_bytes(b"partial write")
            self.assertEqual(engine.latest_step(work, "job-a"), (1, older))
            engine.receipt_path(newer).write_text("[]")
            self.assertEqual(engine.latest_step(work, "job-a"), (1, older))

    def test_same_work_directory_has_one_writer(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            with engine.writer_lock(work), self.assertRaises(RuntimeError):
                with engine.writer_lock(work):
                    self.fail("second owner acquired the lock")
            with engine.writer_lock(work):
                pass

    def test_cancel_file_stops_running_child_and_records_interrupt(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            signal = work / "cancel"
            original_popen = subprocess.Popen

            def waiting_child(_, **kwargs):
                return original_popen([sys.executable, "-c", "import time; time.sleep(20)"], **kwargs)

            timer = threading.Timer(0.1, signal.touch)
            with mock.patch.object(engine.subprocess, "Popen", side_effect=waiting_child):
                timer.start()
                try:
                    with self.assertRaises(InterruptedError):
                        engine.supervise_stage("denoise", work, work, Path(sys.executable), signal,
                                               {"threads": 1}, "abc")
                finally:
                    timer.join()
            record = json.loads((work / "timings.jsonl").read_text())
            self.assertEqual(record["outcome"], "cancelled")
            self.assertLess(record["seconds"], 8)

    def test_rgba_hole_is_white_and_foreground_is_centered(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "shape.png"
            data = np.zeros((39, 52, 4), np.uint8)
            data[7:29, 19:43] = (25, 60, 140, 255)
            data[14:22, 26:36, 3] = 0
            Image.fromarray(data).save(path)
            rgb = np.asarray(engine.prepare_input(path))
            self.assertEqual(rgb.shape[0], rgb.shape[1])
            self.assertTrue(np.all(rgb[0] == 255))
            self.assertTrue(np.all(rgb[rgb.shape[0] // 2, rgb.shape[1] // 2] == 255))
            self.assertEqual(int((rgb[:, :, 2] == 140).sum()), 24 * 22 - 8 * 10)

    def test_subdivision_preserves_parent_vertices_and_linear_field(self):
        grid = np.indices((4, 4, 4), dtype=np.float32)
        parent = 2 * grid[0] - grid[1] + grid[2]
        refined = engine.subdivide(parent)
        expected = np.indices((7, 7, 7), dtype=np.float32) * 0.5
        np.testing.assert_array_equal(refined[::2, ::2, ::2], parent)
        np.testing.assert_allclose(refined, 2 * expected[0] - expected[1] + expected[2])


if __name__ == "__main__":
    unittest.main()
