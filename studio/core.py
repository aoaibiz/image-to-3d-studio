import hashlib
import json
import math
import os
from pathlib import Path
from .process_group import ProcessGroup
import queue
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
TERMINAL = {"needs_review", "failed", "cancelled", "interrupted"}
MAX_INPUT = 20 * 1024 * 1024
PYTHON = ROOT / ".runtime" / "venv" / "Scripts" / "python.exe"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class JobError(Exception):
    pass


class JobCore:
    def __init__(self, root=ROOT, runner=None):
        self.root = Path(root)
        self.directory = self.root / "jobs" / "studio"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(
            self.directory / "jobs.sqlite3", check_same_thread=False
        )
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.process = None
        self.group = ProcessGroup()
        self.active = None
        self.checking = set()
        self.stopping = False
        self.runner = runner or self._run_pipeline
        with self.lock:
            self.db.executescript(
                """CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, idem TEXT UNIQUE NOT NULL, input_sha256 TEXT NOT NULL, status TEXT NOT NULL, stage TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, config TEXT NOT NULL); CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, type TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL);"""
            )
            for row in self.db.execute(
                "SELECT id FROM jobs WHERE status IN ('running','queued')"
            ).fetchall():
                self._update(
                    row["id"],
                    "interrupted",
                    "interrupted",
                    "前回の処理が中断されました。再開ボタンでやり直してください。",
                )
            self.db.commit()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _event(self, job_id, kind, payload):
        self.db.execute(
            "INSERT INTO events(job_id,type,created_at,payload) VALUES (?,?,?,?)",
            (job_id, kind, now(), json.dumps(payload, ensure_ascii=False)),
        )
        self.db.commit()
        self.changed.notify_all()

    def _update(self, job_id, status, stage, error=None):
        self.db.execute(
            "UPDATE jobs SET status=?, stage=?, error=?, updated_at=? WHERE id=?",
            (status, stage, error, now(), job_id),
        )
        self._event(job_id, "state", {"status": status, "stage": stage, "error": error})

    def get(self, job_id):
        with self.lock:
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                raise KeyError(job_id)
            job = dict(row)
            job.pop("idem")
            job["config"] = json.loads(job["config"])
            folder = self.directory / job_id / "output"
            names = [
                "model.glb",
                "editable.obj",
                "editable.mtl",
                "editable.ply",
                "material.mtl",
                "material_0.png",
                "texture.png",
                "settings.json",
                "inspection.json",
                "semantic-inspection.json",
                "review.json",
                "offline-preview.html",
                "manifest.json",
            ]
            if folder.exists():
                names += [
                    p.name
                    for p in folder.glob("*.png")
                    if re.fullmatch(
                        r"[0-9a-f]{64}_[a-z-]+_(?:material|clay)\.png", p.name
                    )
                ]
            job["artifacts"] = [
                name
                for name in names
                if job["status"] == "needs_review" and (folder / name).is_file()
            ]
            job["review"] = None
            if "review.json" in job["artifacts"]:
                try:
                    review = json.loads(
                        (folder / "review.json").read_text(encoding="utf-8")
                    )
                    if (
                        review.get("sha256")
                        == hashlib.sha256(
                            (folder / "model.glb").read_bytes()
                        ).hexdigest()
                    ):
                        job["review"] = review
                    else:
                        job["artifacts"].remove("review.json")
                except (OSError, ValueError):
                    job["artifacts"].remove("review.json")
            return job

    def _review_inputs(self, job_id):
        job = self.get(job_id)
        if job["status"] != "needs_review":
            raise JobError("生成と検査が終わってから評価してください。")
        out = self.directory / job_id / "output"
        try:
            digest = hashlib.sha256((out / "model.glb").read_bytes()).hexdigest()
            inspection = json.loads(
                (out / "inspection.json").read_text(encoding="utf-8")
            )
            settings = json.loads((out / "settings.json").read_text(encoding="utf-8"))
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            source_digest = hashlib.sha256(
                (out.parent / "input.image").read_bytes()
            ).hexdigest()
            if (
                inspection["sha256"] != digest
                or settings["glb_sha256"] != digest
                or manifest["sha256"] != digest
                or source_digest != job["input_sha256"]
                or settings["input_sha256"] != source_digest
            ):
                raise ValueError("hash mismatch")
            files = manifest["files"]
            for mode in ("material", "clay"):
                if not any(
                    isinstance(name, str)
                    and re.fullmatch(digest + r"_[a-z-]+_" + mode + r"\.png", name)
                    and (out / name).is_file()
                    for name in files
                ):
                    raise ValueError("review images missing")
            return job, out, digest, inspection, settings
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise JobError(
                "成果物の照合に失敗しました。検証をやり直してください。"
            ) from exc

    @staticmethod
    def _atomic_json(path, value):
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temp, path)

    def review(self, job_id, data):
        if not isinstance(data, dict) or set(data) != {
            "sha256",
            "decision",
            "reference_fidelity",
            "major_part_volume",
            "attachment_material",
            "inspected_views",
        }:
            raise JobError("評価項目と表示の確認をすべて入力してください。")
        if data["decision"] not in ("pass", "fail", "review") or data[
            "inspected_views"
        ] != ["material", "clay"]:
            raise JobError("判定と素材・粘土表示の確認を指定してください。")
        if any(
            not isinstance(data[k], str) or not data[k].strip() or len(data[k]) > 4000
            for k in ("reference_fidelity", "major_part_volume", "attachment_material")
        ):
            raise JobError("各評価欄に4000文字以内で内容を入力してください。")
        with self.lock:
            if job_id in self.checking:
                raise JobError("形状検査が終了してから評価してください。")
            job, out, digest, inspection, settings = self._review_inputs(job_id)
            if data["sha256"] != digest:
                raise JobError(
                    "評価対象が現在のモデルと異なります。表示を確認し直してください。"
                )
            geometry = inspection.get("geometry", {}).get("status")
            semantic = out / "semantic-inspection.json"
            if semantic.exists():
                try:
                    result = json.loads(semantic.read_text(encoding="utf-8"))
                    if result["sha256"] != digest:
                        raise ValueError("hash mismatch")
                    if result["geometry"]["status"] == "fail":
                        geometry = "fail"
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    raise JobError(
                        "形状検査の照合に失敗しました。検査をやり直してください。"
                    ) from exc
            if data["decision"] == "pass" and (
                inspection.get("format", {}).get("status") != "pass"
                or geometry == "fail"
            ):
                raise JobError(
                    "形式検査が合格し、形状検査が不合格でない場合に合格を選べます。"
                )
            report = dict(
                data,
                created_at=now(),
                input_sha256=job["input_sha256"],
                settings=settings,
            )
            self._atomic_json(out / "review.json", report)
            self._event(job_id, "review", report)
            return self.get(job_id)

    @staticmethod
    def _validate_profile(profile):
        def finite(value):
            try:
                return (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                )
            except OverflowError:
                return False

        if (
            not isinstance(profile, dict)
            or set(profile) != {"subject", "up_axis", "depth_axis", "regions"}
            or profile["subject"] != "cat"
            or type(profile["up_axis"]) is not int
            or type(profile["depth_axis"]) is not int
            or profile["up_axis"] not in (0, 1, 2)
            or profile["depth_axis"] not in (0, 1, 2)
            or profile["up_axis"] == profile["depth_axis"]
        ):
            raise JobError("猫の対象と異なる上下・奥行き軸を指定してください。")
        regions = profile["regions"]
        if not isinstance(regions, dict) or set(regions) != {"head", "body"}:
            raise JobError("頭と胴の領域を両方指定してください。")
        for region in regions.values():
            if not isinstance(region, dict) or set(region) != {
                "bounds",
                "levels",
                "min_depth_ratio",
            }:
                raise JobError("領域の範囲・断面・しきい値を指定してください。")
            bounds, levels, ratio = (
                region["bounds"],
                region["levels"],
                region["min_depth_ratio"],
            )
            if (
                not isinstance(bounds, list)
                or len(bounds) != 3
                or any(
                    not isinstance(b, list)
                    or len(b) != 2
                    or not all(finite(v) for v in b)
                    or b[0] >= b[1]
                    for b in bounds
                )
                or not isinstance(levels, list)
                or not 3 <= len(levels) <= 8
                or not all(
                    finite(v)
                    and bounds[profile["up_axis"]][0]
                    < v
                    < bounds[profile["up_axis"]][1]
                    for v in levels
                )
                or not finite(ratio)
                or not 0 < ratio <= 1
            ):
                raise JobError(
                    "有限の範囲、3〜8個の範囲内断面、0より大きく1以下の比率を指定してください。"
                )

    def inspect(self, job_id, data):
        if not isinstance(data, dict) or set(data) != {"profile"}:
            raise JobError("形状検査の profile を指定してください。")
        profile = data["profile"]
        self._validate_profile(profile)
        with self.lock:
            if self.checking or self.active:
                raise JobError("別の処理が終了してから形状検査を実行してください。")
            _, out, digest, _, _ = self._review_inputs(job_id)
            self.checking.add(job_id)
        try:
            from quality.inspection import inspect

            try:
                result = inspect(out / "model.glb", profile=profile)
            except Exception as exc:
                raise JobError(
                    "形状検査に失敗しました。範囲とモデルを確認し、再度お試しください。"
                ) from exc
            with self.lock:
                self._review_inputs(job_id)
                if result.get("sha256") != digest:
                    raise JobError(
                        "検査中にモデルが変わりました。検査し直してください。"
                    )
                result["profile"] = profile
                result["created_at"] = now()
                self._atomic_json(out / "semantic-inspection.json", result)
                # New geometry evidence invalidates any earlier human decision.
                (out / "review.json").unlink(missing_ok=True)
                self._event(
                    job_id,
                    "inspection",
                    {"sha256": digest, "geometry": result["geometry"]},
                )
                return self.get(job_id)
        finally:
            with self.changed:
                self.checking.discard(job_id)
                self.changed.notify_all()

    def create(self, data, idem):
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", idem or ""):
            raise JobError(
                "Idempotency-Key を指定してください。英数字など128文字以内です。"
            )
        if not data or len(data) > MAX_INPUT:
            raise JobError("画像は20MB以下で選び直してください。")
        digest = hashlib.sha256(data).hexdigest()
        with self.lock:
            row = self.db.execute(
                "SELECT id,input_sha256 FROM jobs WHERE idem=?", (idem,)
            ).fetchone()
            if row:
                if row["input_sha256"] != digest:
                    raise JobError(
                        "同じキーに異なる画像は使えません。新しいキーで送信してください。"
                    )
                return self.get(row["id"]), False
            self._validate_image(data)
            job_id = str(uuid.uuid4())
            folder = self.directory / job_id
            folder.mkdir()
            (folder / "input.image").write_bytes(data)
            stamp = now()
            self.db.execute(
                "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    idem,
                    digest,
                    "queued",
                    "queued",
                    None,
                    stamp,
                    stamp,
                    json.dumps({"resolution": 96, "texture_size": 1024}),
                ),
            )
            self._event(
                job_id, "state", {"status": "queued", "stage": "queued", "error": None}
            )
            return self.get(job_id), True

    def _validate_image(self, data):
        # Pillow resides in the isolated runtime, not necessarily the service interpreter.
        script = "import sys,io,warnings; from PIL import Image; Image.MAX_IMAGE_PIXELS=40000000; warnings.simplefilter('error'); im=Image.open(io.BytesIO(sys.stdin.buffer.read())); assert im.format in ('PNG','JPEG','WEBP'); assert im.width<=10000 and im.height<=10000 and im.width*im.height<=40000000; im.verify()"
        try:
            result = subprocess.run(
                [
                    str(self.root / ".runtime" / "venv" / "Scripts" / "python.exe"),
                    "-c",
                    script,
                ],
                input=data,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise JobError(
                "画像検証環境を起動できません。セットアップを確認してください。"
            ) from exc
        if result.returncode:
            raise JobError(
                "PNG・JPEG・WebP画像（最大40メガピクセル）を選び直してください。"
            )

    def cancel(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job["status"] in ("queued", "running"):
                self._update(
                    job_id,
                    "cancelled",
                    "cancelled",
                    "処理を中止しました。必要なら再開してください。",
                )
                if self.active == job_id and self.process:
                    self._kill(self.process)
            return self.get(job_id)

    def resume(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job["status"] not in ("interrupted", "failed", "cancelled"):
                raise JobError("中断・失敗・中止した処理だけ再開できます。")
            if self.active == job_id:
                raise JobError("中止処理が終わるまで待ってから再開してください。")
            self._update(job_id, "queued", "queued")
            return self.get(job_id)

    def events(self, job_id, after):
        self.get(job_id)
        rows = self.db.execute(
            "SELECT * FROM events WHERE job_id=? AND id>? ORDER BY id", (job_id, after)
        ).fetchall()
        return [dict(row) | {"payload": json.loads(row["payload"])} for row in rows]

    def _worker(self):
        while True:
            with self.changed:
                while not self.stopping:
                    if self.checking:
                        self.changed.wait()
                        continue
                    row = self.db.execute(
                        "SELECT id FROM jobs WHERE status='queued' ORDER BY created_at,id LIMIT 1"
                    ).fetchone()
                    if row:
                        break
                    self.changed.wait()
                if self.stopping:
                    return
                job_id = row["id"]
                self.active = job_id
                self._update(job_id, "running", "starting")
            try:
                self.runner(job_id)
                with self.lock:
                    if self.get(job_id)["status"] == "running":
                        self._update(job_id, "needs_review", "needs_review")
            except Exception as exc:
                with self.lock:
                    if self.get(job_id)["status"] == "running":
                        stage = self.get(job_id)["stage"]
                        messages = {
                            "inspection": "モデルの検査に失敗しました。検査ツールと空き容量を確認して再開してください。",
                            "rendering": "プレビューの作成に失敗しました。ブラウザーと Node.js の準備を確認して再開してください。",
                        }
                        message = messages.get(
                            stage,
                            "3D生成に失敗しました。実行環境と空き容量を確認して再開してください。",
                        )
                        with (self.directory / job_id / "diagnostic.log").open(
                            "a", encoding="utf-8"
                        ) as log:
                            log.write(
                                f"{now()} stage={stage} error_type={type(exc).__name__}\n"
                            )
                        self._update(job_id, "failed", "failed", message)
            finally:
                with self.changed:
                    self.active = None
                    self.process = None
                    self.changed.notify_all()

    def _kill(self, proc):
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True
            )
        else:
            import signal

            os.killpg(proc.pid, signal.SIGKILL)

    def _command(self, job_id, command, stage):
        with self.lock:
            if self.get(job_id)["status"] != "running":
                raise JobError("cancelled")
            self._update(job_id, "running", stage)
            proc = subprocess.Popen(
                command,
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=(os.name != "nt"),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt"
                else 0,
            )
            try:
                self.group.assign(proc)
            except Exception:
                self._kill(proc)
                proc.wait()
                raise
            self.process = proc
        lines = []
        for line in proc.stdout:
            lines.append(line)
            marker = line.strip()
            if marker in (
                "initializing model",
                "preprocessing",
                "running model",
                "extracting mesh",
                "baking texture",
            ):
                with self.lock:
                    if self.get(job_id)["status"] == "running":
                        self._update(job_id, "running", marker)
        code = proc.wait()
        with self.lock:
            self.process = None
            if self.get(job_id)["status"] != "running":
                raise JobError("cancelled")
        if code:
            raise RuntimeError("command failed")
        return "".join(lines)

    def _run_pipeline(self, job_id):
        folder = self.directory / job_id
        out = folder / "output"
        out.mkdir(exist_ok=True)
        # A rerun creates new verification evidence; an earlier opinion must not be reused.
        (out / "review.json").unlink(missing_ok=True)
        (out / "semantic-inspection.json").unlink(missing_ok=True)
        py = str(self.root / ".runtime" / "venv" / "Scripts" / "python.exe")
        glb = out / "model.glb"
        settings_path = out / "settings.json"
        config = self.get(job_id)["config"]
        if (
            hashlib.sha256((folder / "input.image").read_bytes()).hexdigest()
            != self.get(job_id)["input_sha256"]
        ):
            raise RuntimeError("input hash mismatch")
        valid_inference = False
        try:
            previous = json.loads(settings_path.read_text(encoding="utf-8"))
            valid_inference = (
                previous["input_sha256"] == self.get(job_id)["input_sha256"]
                and previous["glb_sha256"]
                == hashlib.sha256(glb.read_bytes()).hexdigest()
                and previous["marching_cubes_resolution"] == config["resolution"]
                and previous["texture_size"] == config["texture_size"]
            )
        except (OSError, ValueError, KeyError):
            pass
        if not valid_inference:
            # Only clear files inside this job's own output directory.
            for p in out.iterdir():
                if p.is_file():
                    p.unlink()
            self._command(
                job_id,
                [
                    py,
                    str(self.root / "engine" / "infer.py"),
                    str(folder / "input.image"),
                    str(out),
                    "--resolution",
                    str(config["resolution"]),
                    "--texture-size",
                    str(config["texture_size"]),
                ],
                "starting",
            )
        digest = hashlib.sha256(glb.read_bytes()).hexdigest()
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if (
            settings["glb_sha256"] != digest
            or settings["input_sha256"] != self.get(job_id)["input_sha256"]
            or settings["marching_cubes_resolution"] != config["resolution"]
            or settings["texture_size"] != config["texture_size"]
        ):
            raise RuntimeError("hash mismatch")
        validator = self.root / ".runtime" / "validator" / "gltf_validator.exe"
        cmd = [py, str(self.root / "quality" / "inspection.py"), str(glb)]
        if validator.is_file():
            cmd.extend(["--validator", str(validator)])
        inspection = json.loads(self._command(job_id, cmd, "inspection"))
        if inspection["sha256"] != digest or inspection["format"]["status"] == "fail":
            raise RuntimeError("inspection failed")
        (out / "inspection.json").write_text(
            json.dumps(inspection, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._command(
            job_id,
            ["node", str(self.root / "viewer" / "proof.cjs"), str(glb), str(out)],
            "rendering",
        )
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["sha256"] != digest
            or hashlib.sha256(glb.read_bytes()).hexdigest() != digest
            or not manifest.get("offlinePreviewVerified")
            or not (out / "offline-preview.html").is_file()
            or not manifest.get("files")
            or not all((out / name).is_file() for name in manifest["files"])
        ):
            raise RuntimeError("proof hash mismatch")

    def close(self):
        with self.changed:
            self.stopping = True
            if self.active and self.get(self.active)["status"] == "running":
                self._update(
                    self.active,
                    "interrupted",
                    "interrupted",
                    "処理が中断されました。再開ボタンでやり直してください。",
                )
            if self.process:
                self._kill(self.process)
            self.changed.notify_all()
        self.worker.join(timeout=15)
        self.group.close()
        self.db.close()
