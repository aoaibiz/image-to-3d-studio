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
        work = Path(self.enterContext(tempfile.TemporaryDirectory()))
        script = (
            "import sys,threading,time\n"
            "from pathlib import Path\n"
            "sys.path.insert(0,sys.argv[1])\n"
            "from triposg_cpu import watch_supervisor,record_stage\n"
            "start=time.monotonic()\n"
            "def lost(): record_stage(Path(sys.argv[2]),'denoise','job',start,'interrupted',130,reason='parent_lost')\n"
            "threading.Thread(target=watch_supervisor,args=(sys.stdin.fileno(),lost),daemon=True).start()\n"
            "import numpy\n"
            "print('native-ready',flush=True)\n"
            "time.sleep(25)"
        )
        child = subprocess.Popen([sys.executable, "-u", "-c", script, str(Path(engine.__file__).parent), str(work)],
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, text=True)
        observed = queue.Queue()
        reader = threading.Thread(target=lambda: observed.put(child.stdout.readline()), daemon=True)
        reader.start()
        try:
            self.assertEqual(observed.get(timeout=15).strip(), "native-ready")
            child.stdin.close()
            self.assertEqual(child.wait(timeout=4), 130)
            logged = json.loads((work / "timings.jsonl").read_text())
            self.assertEqual((logged["reason"], logged["exit_code"]), ("parent_lost", 130))
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

    def test_worker_relays_live_progress_and_records_only_typed_ram_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            signal = work / "heard"
            original_popen = subprocess.Popen
            script = (
                "import sys,json,time,pathlib\n"
                "sys.stdin.readline()\n"
                "print('private debug text',flush=True)\n"
                "print(json.dumps(dict(event='progress',stage='denoise',step=36,total=50)),flush=True)\n"
                "end=time.monotonic()+4\n"
                "while not pathlib.Path(sys.argv[1]).exists() and time.monotonic()<end: time.sleep(.02)\n"
                "if not pathlib.Path(sys.argv[1]).exists(): sys.exit(8)\n"
                "print(json.dumps(dict(event='resource_shortfall',stage='denoise',"
                "available_gib=1.52,required_gib=1.75,private='secret')),flush=True)\n"
                "sys.exit(75)\n"
            )
            def launch(_, **kwargs):
                return original_popen([sys.executable, "-u", "-c", script, str(signal)], **kwargs)
            def received(event, **payload):
                if event == "progress":
                    self.assertEqual(payload, dict(stage="denoise", step=36, total=50))
                    signal.touch()
            with (mock.patch.object(engine.subprocess, "Popen", side_effect=launch),
                  mock.patch.object(engine, "emit", side_effect=received)):
                with self.assertRaises(engine.RamShortfall) as caught:
                    engine.supervise_stage("denoise", work, work, Path(sys.executable), None, {"threads": 1}, "key")
            self.assertEqual(caught.exception.available_gib, 1.52)
            history = (work / "timings.jsonl").read_text()
            self.assertNotIn("secret", history)
            self.assertNotIn("private", history)
            record = json.loads(history)
            self.assertEqual((record["stage"], record["reason"], record["exit_code"], record["required_gib"]),
                             ("denoise", "resource_shortfall", 75, 1.75))

    def test_missing_native_dependency_is_recorded_without_raw_exception_or_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            original_popen = subprocess.Popen
            script = (
                "import sys\nsys.path.insert(0,sys.argv.pop(1))\nimport triposg_cpu as engine\n"
                "def missing(*args): raise ImportError('private absolute path or secret')\n"
                "engine.stage_worker=missing\nengine.main()"
            )
            def launch(command, **kwargs):
                return original_popen([sys.executable, "-c", script, str(Path(engine.__file__).parent),
                                       *command[2:]], **kwargs)
            with (mock.patch.object(engine.subprocess, "Popen", side_effect=launch) as calls,
                  mock.patch.object(engine, "emit"), mock.patch.object(engine, "wait_for_memory") as wait):
                with self.assertRaisesRegex(RuntimeError, "dependency_unavailable"):
                    engine.run_stage("encode", work, work, Path(sys.executable), None, {"threads": 1}, "key")
                calls.assert_called_once()
                wait.assert_not_called()
            history = (work / "timings.jsonl").read_text()
            self.assertNotIn("secret", history)
            self.assertEqual(json.loads(history)["reason"], "dependency_unavailable")

    def test_low_ram_resumes_same_sealed_step_once_and_never_retries_other_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            args = ("denoise", work, work, Path(sys.executable), None, {"threads": 4}, "job-x")
            path = work / "flow-b.safetensors"
            def simulate(*actual):
                self.assertEqual(actual, args)
                if engine.latest_step(work, actual[-1])[0] == 0:
                    path.write_bytes(b"sealed step 36 fixture")
                    engine.seal(path, actual[-1], step=36)
                    raise engine.RamShortfall(1.72, 1.75)
                self.assertEqual(engine.latest_step(work, actual[-1]), (36, path))
            with (mock.patch.object(engine, "supervise_stage", side_effect=simulate) as stage,
                  mock.patch.object(engine, "available_ram_gib", side_effect=[5.0, 6.6, 6.49, 6.7, 6.8]),
                  mock.patch.object(engine.time, "sleep"), mock.patch.object(engine, "emit") as event):
                engine.run_stage(*args)
                self.assertEqual(stage.call_count, 2)
                event.assert_called_once_with("waiting_for_memory", stage="denoise", available_gib=1.72,
                                              required_gib=6.5, max_wait_seconds=600)
            with (mock.patch.object(engine, "supervise_stage", side_effect=RuntimeError("worker_error")) as stage,
                  mock.patch.object(engine, "wait_for_memory") as wait):
                with self.assertRaisesRegex(RuntimeError, "worker_error"):
                    engine.run_stage(*args)
                stage.assert_called_once()
                wait.assert_not_called()
            with (mock.patch.object(engine, "supervise_stage", side_effect=engine.RamShortfall(1.7, 1.75)) as stage,
                  mock.patch.object(engine, "wait_for_memory") as wait):
                with self.assertRaises(engine.RamShortfall):
                    engine.run_stage(*args)
                self.assertEqual(stage.call_count, 2)
                wait.assert_called_once()

    def test_memory_wait_timeout_or_cancel_never_relaunches(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            cancel = work / "cancel"
            for stop in ("timeout", "cancel"):
                if stop == "cancel":
                    cancel.touch()
                with (mock.patch.object(engine, "supervise_stage", side_effect=engine.RamShortfall(.7, 6)) as stage,
                      mock.patch.object(engine, "MEMORY_WAIT_SECONDS", 0),
                      mock.patch.object(engine, "available_ram_gib", return_value=2), mock.patch.object(engine, "emit")):
                    with self.assertRaises(MemoryError if stop == "timeout" else InterruptedError):
                        engine.run_stage("encode", work, work, Path(sys.executable), cancel, {"threads": 1}, "job-x")
                    stage.assert_called_once()
            records = [json.loads(line) for line in (work / "timings.jsonl").read_text().splitlines()]
            self.assertEqual([r["reason"] for r in records], ["memory_wait_timeout", "cancelled"])

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

    def test_faint_alpha_noise_does_not_shrink_subject_or_erase_soft_edges(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "figure.png"
            img = np.zeros((700, 700, 4), np.uint8)
            img[:, :, :3] = 80
            img[280:420, 300:400] = (200, 10, 12, 255)
            img[280:420, 299] = (200, 10, 12, 70)
            extents = []
            for alpha_floor in (0, 1):
                img[:, :, 3] = np.maximum(img[:, :, 3], alpha_floor)
                Image.fromarray(img).save(source)
                rgb = np.asarray(engine.prepare_input(source))
                red = rgb[:, :, 1] < 30
                y, x = np.nonzero(red)
                extents.append((rgb.shape, int(np.ptp(y)), int(np.ptp(x)), int(x.min())))
                self.assertLess(int(rgb[y.min() + 5, x.min() - 1, 1]), 220)
                self.assertGreater(int(rgb[y.min() + 5, x.min() - 1, 1]), 150)
            self.assertEqual(extents[0], extents[1])
            self.assertGreater(extents[0][1] / extents[0][0][0], .80)

    def test_extreme_aspect_ratio_is_downsampled_before_square_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "strip.png"
            image = Image.new("RGBA", (100_000, 20), (0, 0, 0, 0))
            image.paste((255, 30, 20, 255), (10, 5, 99_990, 15))
            image.save(source)
            create = Image.new

            def checked_canvas(mode, size, *args, **kwargs):
                self.assertLessEqual(max(size), 4096, "oversized canvas requested")
                return create(mode, size, *args, **kwargs)

            with mock.patch.object(Image, "new", side_effect=checked_canvas):
                real = engine.prepare_input(source)
            self.assertEqual(real.size, (768, 768))

    def test_generate_repairs_only_surface_once_if_selected_field_is_damaged(self):
        # Stage doubles exercise the real scheduler/receipts, not model inference.
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder).resolve()
            source = work / "tiny.png"
            Image.new("RGBA", (9, 9), (40, 90, 160, 255)).save(source)
            raw = work / "raw-shape.glb"
            field = work / "field-256.npy"
            lower = work / "field-128.npy"
            shape = work / "shape.safetensors"
            def run():
                return engine.generate(source, work, raw, work)

            def produce(stage, runtime, directory, interpreter, cancel_file, settings, job):
                if stage in ("encode", "denoise"):
                    path = directory / ("condition.safetensors" if stage == "encode" else "shape.safetensors")
                    path.write_bytes(stage.encode())
                    engine.seal(path, job)
                    return
                if not lower.exists():
                    np.save(lower, np.zeros((3, 3, 3), dtype=np.float32))
                    engine.seal(lower, job, cells=128)
                self.assertIsNotNone(engine.valid_artifact(lower, job, cells=128))
                self.assertIsNotNone(engine.valid_artifact(shape, job))
                np.save(field, np.ones((5, 5, 5), dtype=np.float32))
                engine.seal(field, job, cells=settings["cells"])
                raw.write_bytes(b"scheduler fixture")
                engine.seal(raw, job, cells=settings["cells"])

            with (mock.patch.object(engine, "verify_runtime"), mock.patch.object(engine, "runtime_python"),
                  mock.patch.object(engine, "installed_versions", return_value={}), mock.patch.object(engine, "emit"),
                  mock.patch.object(engine, "supervise_stage", side_effect=produce) as simulate):
                original = run()
                self.assertEqual([call.args[0] for call in simulate.call_args_list], ["encode", "denoise", "surface"])
                frozen = [(p.read_bytes(), p.stat().st_mtime_ns) for p in (shape, lower)]
                simulate.reset_mock()
                self.assertEqual(run(), original)
                simulate.assert_not_called()

                for mode in ("bytes", "missing", "other-job", "other-resolution"):
                    with self.subTest(mode=mode):
                        if mode == "bytes":
                            field.write_bytes(b"damaged")
                        elif mode == "missing":
                            engine.receipt_path(field).unlink()
                        elif mode == "other-job":
                            engine.seal(field, "another job", cells=256)
                        else:
                            engine.seal(field, original["job_key"], cells=128)
                        self.assertEqual(run(), original)
                        self.assertEqual(run(), original)
                        self.assertEqual([call.args[0] for call in simulate.call_args_list], ["surface"])
                        self.assertEqual([(p.read_bytes(), p.stat().st_mtime_ns) for p in (shape, lower)], frozen)
                        simulate.reset_mock()

                field.unlink()
                simulate.side_effect = None
                with self.assertRaisesRegex(RuntimeError, "surface"):
                    run()
                simulate.assert_called_once()

                simulate.reset_mock()
                original_identity = (work / "identity.json").read_bytes()
                other_runner = work / "other-version.py"
                other_runner.write_bytes(b"# different build")
                with mock.patch.object(engine, "__file__", str(other_runner)):
                    with self.assertRaisesRegex(ValueError, json.loads(original_identity)["runner_sha256"]):
                        run()
                simulate.assert_not_called()
                self.assertEqual((work / "identity.json").read_bytes(), original_identity)


if __name__ == "__main__":
    unittest.main()
