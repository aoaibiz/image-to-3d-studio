import hashlib
import json
import os
from pathlib import Path
from .process_group import ProcessGroup
from .configure import read_pipeline
from engine.pipeline import (GUIDED_PIPELINES, HOST_PIPELINE, LEGACY_PIPELINE, OBSERVED_PIPELINE,
                             PIPELINE, PIXAL_PIPELINE, SPACE_PIPELINES, SF3D_OBSERVED_PIPELINE, SF3D_PIPELINE,
                             SHAPE_PIPELINE, SHAPE_PIPELINES)
import queue
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
TERMINAL = {"needs_review", "review_hold", "failed", "cancelled", "interrupted"}
MAX_INPUT = 20 * 1024 * 1024
PYTHON = ROOT / ".runtime" / "venv" / "Scripts" / "python.exe"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class JobError(Exception):
    pass


class PipelineConflict(JobError):
    pass


class AwaitingHostImage(Exception):
    pass


class ReviewHold(Exception):
    pass


SHAPE_OUTPUTS = {"shape-master.glb", "appearance.json", "generation.json", "input-processed.png"}
GUIDED_OUTPUTS = {"guided-paint.json", "guided-views.png", "guided-layout.json", "inferred-paint.png"}
HYPOTHESIS_REPORT = "shape-hypothesis.json"
HYPOTHESIS_GLB = "shape-alternative.glb"
HYPOTHESIS_MASTER = "shape-alternative.gltf"
HYPOTHESIS_FILES = {HYPOTHESIS_REPORT, HYPOTHESIS_GLB, HYPOTHESIS_MASTER, "pre-hypothesis.glb"}
PUBLIC_SHAPE_METHODS = {"source-rim-inferred-cavity-v1"}
CONTINUITY_FILES = ("continuity-paint.json", "continuity-native.glb")


def _local_continuity(out, settings, source_sha, validate):
    if settings.get("appearance_mode") != "local_continuity" or settings.get("pipeline") != SF3D_OBSERVED_PIPELINE:
        return False
    try:
        sealed = settings["artifact_sha256"]
        if (settings["input_sha256"] != source_sha or not all(name in sealed for name in CONTINUITY_FILES)
                or any((out / name).resolve() != out.resolve() / name for name in (*CONTINUITY_FILES, "generation.json"))
                or not validate(out, settings)):
            return False
        generation = json.loads((out / "generation.json").read_bytes())
        counts = {"external_image_generations": 0, "external_photo_observations": 1, "external_shape_generations": 1}
        return all(type(generation.get(key)) is int and generation[key] == count for key, count in counts.items())
    except (OSError, KeyError, ValueError, TypeError, AttributeError):
        return False


def _hypothesis_files(out, settings, source_sha, validate):
    selection = settings.get("shape_hypothesis")
    if not isinstance(selection, dict) or selection.get("report") != HYPOTHESIS_REPORT:
        return {}
    try:
        sealed = settings["artifact_sha256"]
        decision = selection["decision"]
        names = {HYPOTHESIS_REPORT}
        if decision == "alternative":
            if selection.get("alternative_model") != HYPOTHESIS_GLB or selection.get("alternative_editable") != HYPOTHESIS_MASTER:
                return {}
            names.update((HYPOTHESIS_GLB, HYPOTHESIS_MASTER))
        elif decision == "applied":
            if selection.get("paint_master") != "pre-hypothesis.glb":
                return {}
            names.add("pre-hypothesis.glb")
        elif decision != "unchanged":
            return {}
        if (names != HYPOTHESIS_FILES.intersection(sealed) or settings["input_sha256"] != source_sha
                or any((out / name).resolve() != out.resolve() / name for name in names)
                or not validate(out, settings)):
            return {}
        report = json.loads((out / HYPOTHESIS_REPORT).read_bytes())
        if (type(report["version"]) is not int or report["version"] != 1 or report["decision"] != decision
                or report["method"] not in PUBLIC_SHAPE_METHODS
                or report["source_sha256"] != source_sha or not re.fullmatch("[0-9a-f]{64}", report["implementation_sha256"])):
            return {}
        if decision == "alternative" and (
                report["dimensions_observed"] is not False or report["visual_quality"] != "needs_review"
                or report["candidate"]["sha256"] != sealed[HYPOTHESIS_GLB]
                or report["painted_before"]["sha256"] != settings["glb_sha256"]
                or report["delivered"]["sha256"] != settings["glb_sha256"]
                or sealed[HYPOTHESIS_GLB] == settings["glb_sha256"]):
            return {}
        return {name: sealed[name] for name in sorted(names)}
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        return {}
SHAPE_ERRORS = {
    "sf3d_observed_review": "材質または塗装の向きを確認できず、仕上げを保留しました。同じジョブに保存した検品記録を確認してください。",
    "sf3d_observed_error": "生成を中断しました。SF3Dの利用枠、Codexのログインと元の受付を確認し、同じジョブを再開してください。",
    "pixal_consent": "この公開CC0写真の送信同意と、Pixal3D用の初回設定を確認してください。",
    "pixal_pending": "個人Spaceの応答待ちです。元のジョブを再開して受付済みの結果を確認してください。",
    "pixal_error": "Pixal3Dの利用枠、元の受付と材質付きGLBの検査結果を確認してください。生成の自動再送はありません。",
    "sf3d_consent": "公式Spaceへの送信と匿名利用枠の使用設定を確認し、同じジョブを再開してください。",
    "sf3d_pending": "公式Spaceの受付は保存されています。同じジョブを再開して結果を確認してください。",
    "sf3d_error": "SF3Dの前景切り抜き・公式Spaceの利用枠・保存した成果物を確認してください。受け付けた生成は再送しません。",
    "guided_chatgpt_observation": "元写真のChatGPT解析を確認できません。ログイン・利用枠と保存した受付を確認し、同じジョブを再開してください。",
    "guided_observation": "写真の模様を読み取れませんでした。ローカルの画像解析モデルと空きメモリを確認して同じジョブを再開してください。",
    "guided_consent": "OpenAIへの送信設定を確認してください。ローカルの形状はジョブに保存されています。",
    "guided_checkpoint": "保存した形状または塗装の記録を照合できません。ジョブのバックアップを確認してください。",
    "guided_guides": "立体から背面の確認画像を作成できませんでした。保存した形状を確認してください。",
    "guided_provider": "背面の推定画像を取得できませんでした。Codexのログイン・利用枠を確認してください。再開時は受付済みの結果を確認します。",
    "guided_orientation": "推定した背面の向きを確認できませんでした。元の立体と検品画像をジョブに保存しています。",
    "guided_projection": "推定画像を立体に重ねられませんでした。元のローカルモデルはジョブに保存されています。",
    "guided_export": "背面を仕上げたGLBの書き出しを確認できません。同じジョブを再開してください。",
    "shape_setup": "TripoSG のローカル環境を確認し、オプションのセットアップ後に再開してください。",
    "shape_memory": "空きメモリが足りず、計算を一時停止しました。ほかの作業が終わったら同じジョブの「再開」を押してください。",
    "shape_foreground": "画像の切り抜きに失敗しました。背景と被写体、U2Net の設定を確認してください。",
    "shape_version": "このジョブは更新前のエンジンで開始されました。受付時のインストールで同じジョブを再開するか、現在の版で新しいジョブを開始してください。",
    "shape_inference": "形状生成に失敗しました。TripoSG のモデル、RAM と空き容量を確認して同じジョブを再開してください。",
    "shape_registration": "生成した形状を写真の輪郭に重ねられず、着色を停止しました。切り抜きを確認してください。別の画像で試す場合は新しいジョブを作成してください。",
    "shape_appearance": "形状の着色に失敗しました。保存した形状と入力画像を確認して再開してください。",
}


def _shape_notice(line):
    """Map the worker protocol to fixed status/error text. Never relay raw details."""
    try:
        event = json.loads(line)
    except ValueError:
        return None, None
    if not isinstance(event, dict):
        return None, None
    kind, stage = event.get("event"), event.get("stage")
    if kind == "studio_error" and isinstance(event.get("code"), str):
        return None, SHAPE_ERRORS.get(event["code"])
    if kind == "waiting_for_memory" and stage in ("encode", "denoise", "surface"):
        free, needed = event.get("available_gib"), event.get("required_gib")
        if all(type(value) in (int, float) and 0 <= value < 100000 for value in (free, needed)):
            return f"waiting for memory {free:.2f}/{needed:.2f}", None
    if kind == "resumed" and stage in ("encode", "denoise", "surface"):
        return "resuming shape", None
    if kind == "stage" and stage in ("verify", "encode", "denoise", "surface"):
        stages = {"verify": "initializing model", "encode": "preprocessing",
                  "denoise": "generating shape", "surface": "extracting mesh"}
        return stages[stage], None
    if kind == "progress" and stage == "denoise":
        step, total = event.get("step"), event.get("total")
        if type(step) is int and type(total) is int and 0 < step <= total <= 10000:
            return f"shape step {step}/{total}", None
    return None, None


def _lock_store(directory):
    """Hold one OS lock per job store for the lifetime of its worker."""
    handle = open(directory / ".store.lock", "a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except OSError as exc:
        handle.close()
        raise RuntimeError("Job store is already in use") from exc


def _unlock_store(handle):
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


class JobCore:
    def __init__(self, root=ROOT, runner=None, *, directory=None):
        self.root = Path(root)
        self.directory = (
            Path(directory).resolve()
            if directory is not None
            else self.root / "jobs" / "studio"
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pipeline = read_pipeline(self.directory)
        self.store_lock = _lock_store(self.directory)
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
                "editable.gltf",
                "editable.obj",
                "editable.mtl",
                "editable.ply",
                "material.mtl",
                "material_0.png",
                "texture.png",
                "input-processed.png",
                "settings.json",
                "inspection.json",
                "automatic-profile.json",
                "semantic-inspection.json",
                "review.json",
                "offline-preview.html",
                "manifest.json",
            ]
            if job["config"].get("pipeline") in GUIDED_PIPELINES:
                names.extend(sorted(GUIDED_OUTPUTS))
            if job["config"].get("pipeline") in SHAPE_PIPELINES:
                names.extend(["shape-master.glb", "appearance.json", "generation.json"])
            if job["config"].get("pipeline") in (SF3D_PIPELINE, PIXAL_PIPELINE):
                names.extend(["provider-native.glb", "generation.json"])
            if job["config"].get("pipeline") == SF3D_OBSERVED_PIPELINE and job["status"] == "needs_review":
                candidates = {"shape-master.glb", "generation.json", "finished-native.glb",
                              "guided-views.png", "guided-layout.json", "inferred-paint.png",
                              "hidden-review.json", "region.json", "full-paint.json", "native-inspection.png", "native-material.json"}
                try:
                    settings = json.loads((folder / "settings.json").read_bytes())
                    sealed = settings["artifact_sha256"]
                    if isinstance(sealed, dict):
                        names.extend(sorted(candidates.intersection(sealed)))
                        extra = _hypothesis_files(folder, settings, job["input_sha256"], self._artifacts_valid)
                        if extra:
                            names.extend(extra)
                            job["shape_hypothesis_files"] = extra
                        # Both checks use the same engine verifier on these exact settings.
                        if _local_continuity(folder, settings, job["input_sha256"],
                                             lambda out, cfg: bool(extra) or self._artifacts_valid(out, cfg)):
                            names.extend(CONTINUITY_FILES)
                            job["local_continuity"] = {"external_image_generations": 0, "report": "continuity-paint.json"}
                except (KeyError, OSError, TypeError, ValueError):
                    pass
            if folder.exists():
                names.extend(
                    p.name
                    for p in folder.glob("*_appearance-comparison.jpg")
                    if re.fullmatch(r"[0-9a-f]{64}_appearance-comparison\.jpg", p.name)
                )
                names += [
                    p.name
                    for p in folder.glob("*.png")
                    if re.fullmatch(
                        r"[0-9a-f]{64}_[a-z-]+_(?:material|clay)\.png", p.name
                    )
                ]
                names += [
                    p.name
                    for p in folder.glob("material_*.png")
                    if re.fullmatch(r"material_[0-9]+\.png", p.name)
                    and p.name not in names
                ]
            job["artifacts"] = [
                name
                for name in names
                if job["status"] == "needs_review" and (folder / name).is_file()
            ]
            job["review"] = None
            if job["status"] == "review_hold" and job["config"].get("pipeline") == SF3D_OBSERVED_PIPELINE:
                from .review_hold import read_hold
                diagnostic = read_hold(folder, job["input_sha256"])
                if diagnostic:
                    job["hold"], job["artifacts"] = diagnostic
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
        from quality.inspection import validate_profile

        try:
            validate_profile(profile)
        except ValueError as exc:
            raise JobError(str(exc)) from exc

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

    def create(self, data, idem, expected_pipeline=None, public_cc0_sha=None):
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", idem or ""):
            raise JobError(
                "Idempotency-Key を指定してください。英数字など128文字以内です。"
            )
        if not data or len(data) > MAX_INPUT:
            raise JobError("画像は20MB以下で選び直してください。")
        if expected_pipeline is not None and expected_pipeline != self.pipeline:
            raise PipelineConflict("サービスの生成方式が変わりました。使用する方式を確認してください。")
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
                job = self.get(row["id"])
                if expected_pipeline is not None and job["config"].get("pipeline") != expected_pipeline:
                    raise PipelineConflict("このキーは別の生成方式のジョブで使用済みです。")
                return job, False
            if self.pipeline == PIXAL_PIPELINE:
                if expected_pipeline != PIXAL_PIPELINE or public_cc0_sha != digest:
                    raise PipelineConflict("このCC0公開写真を個人Spaceへ送る同意を、写真ごとに指定してください。")
            if self.pipeline in (SF3D_OBSERVED_PIPELINE, PIXAL_PIPELINE):
                try:
                    current = read_pipeline(self.directory)
                except (OSError, ValueError) as exc:
                    raise PipelineConflict("送信設定を読み取れません。初回設定を確認してStudioを再起動してください。") from exc
                if current != self.pipeline:
                    raise PipelineConflict("送信設定が変わりました。Studioを再起動して現在の方式を確認してください。")
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
                    json.dumps(
                        {
                            **({"public_cc0_sha256": digest} if self.pipeline == PIXAL_PIPELINE else {}),
                            **({} if self.pipeline in SPACE_PIPELINES else {
                                "resolution": 256 if self.pipeline in SHAPE_PIPELINES else 96, "texture_size": 1024,
                            }),
                            "auto_finish": True,
                            "pipeline": self.pipeline,
                        }
                    ),
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
            if job["status"] in ("queued", "running", "awaiting_host_image"):
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

    def host_transfer(self, job_id, token, submission=None):
        from engine import host_paint
        with self.lock:
            if not host_paint.authorize(self.directory, token):
                raise PermissionError("ホストの送信設定を確認してください。")
            job = self.get(job_id)
            status = job["status"]
            if self.pipeline != HOST_PIPELINE or job["config"].get("pipeline") != HOST_PIPELINE:
                raise PipelineConflict("このジョブはホスト画像生成用に受付されていません。")
            work = self.directory / job_id / "output/local-guided-work"
            try:
                if submission is None:
                    if status != "awaiting_host_image":
                        raise PipelineConflict("このジョブは画像を要求していません。")
                    return host_paint.claim(work, job_id)
                if status != "awaiting_host_image" and (
                        status not in {"queued", "running", "needs_review", "failed", "interrupted"} or
                        not (work / "host-result/receipt.json").is_file()):
                    raise PipelineConflict("受付状態を確認してから同じジョブを続行してください。")
                host_paint.submit(work, **submission, validate=self._validate_host_sheet)
                if status in {"awaiting_host_image", "interrupted"}:
                    self._update(job_id, "queued", "queued")
                return {"job": self.get(job_id)}
            except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
                raise JobError("ホストの受付、PNG、生成IDとハッシュを確認してください。") from exc

    def _validate_host_sheet(self, data):
        # The server itself can run on a stdlib-only Python.
        try:
            from engine.guided_provider import _sheet
        except ModuleNotFoundError as exc:
            if exc.name != "PIL":
                raise
            check = subprocess.run(
                [str(self.root / ".runtime/venv/Scripts/python.exe"), "-c",
                 "import sys; from engine.guided_provider import _sheet; _sheet(sys.stdin.buffer.read())"],
                cwd=self.root, input=data, capture_output=True, timeout=30,
            )
            if check.returncode:
                raise ValueError("ホスト画像のPNGを検証できません。")
        else:
            _sheet(data)

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
            except AwaitingHostImage:
                with self.lock:
                    if self.get(job_id)["status"] == "running":
                        self._update(job_id, "awaiting_host_image", "awaiting_host_image")
            except ReviewHold:
                with self.lock:
                    if self.get(job_id)["status"] == "running":
                        self._update(job_id, "review_hold", "review_hold", "推定塗装保留・未合格")
            except Exception as exc:
                with self.lock:
                    if self.get(job_id)["status"] == "running":
                        stage = self.get(job_id)["stage"]
                        messages = {
                            "inspection": "モデルの検査に失敗しました。検査ツールと空き容量を確認して再開してください。",
                            "rendering": "プレビューの作成に失敗しました。ブラウザーと Node.js の準備を確認して再開してください。",
                            "finishing model": "自動仕上げに失敗しました。空き容量と仕上げ環境を確認して再開してください。",
                        }
                        message = messages.get(
                            stage,
                            "3D生成に失敗しました。実行環境と空き容量を確認して再開してください。",
                        )
                        if isinstance(exc, JobError):
                            message = str(exc)
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
        failure = None
        for line in proc.stdout:
            if stage != "starting":
                lines.append(line)
            marker = line.strip()
            notice = None
            if stage == "starting" and marker.startswith("{"):
                notice, problem = _shape_notice(marker)
                failure = problem or failure
            if notice or marker in (
                "initializing model",
                "preprocessing",
                "running model",
                "extracting mesh",
                "baking texture",
                "selecting source",
                "resuming finish",
                "finishing model",
                "pixal connecting", "pixal uploaded", "pixal generation_waiting",
                "pixal downloading", "pixal complete", "pixal exporting",
                "sf3d connecting", "sf3d uploaded", "sf3d preprocess_waiting", "sf3d preprocessed",
                "sf3d generation_waiting", "sf3d downloading", "sf3d complete", "sf3d exporting",
                "sf3d observed native", "sf3d observed native_inspection", "sf3d observed observation", "sf3d observed registration",
                "sf3d observed guides", "sf3d observed paint", "sf3d observed orientation",
                "sf3d observed projection", "sf3d observed export",
                "segmenting source",
                "generating shape",
                "resuming shape",
                "painting model",
                "analyzing surface markings",
                "observing photo with ChatGPT",
                "preparing shape guides",
                "requesting inferred paint",
                "checking hidden surfaces",
                "applying inferred paint",
                "exporting guided model",
            ):
                with self.lock:
                    if self.get(job_id)["status"] == "running":
                        self._update(job_id, "running", notice or marker)
        proc.stdout.close()
        code = proc.wait()
        with self.lock:
            self.process = None
            if self.get(job_id)["status"] != "running":
                raise JobError("cancelled")
        if code:
            if code == 72 and stage == "starting" and self.get(job_id)["config"].get("pipeline") == HOST_PIPELINE:
                raise AwaitingHostImage()
            if failure:
                if (failure == SHAPE_ERRORS["sf3d_observed_review"]
                        and self.get(job_id)["config"].get("pipeline") == SF3D_OBSERVED_PIPELINE):
                    raise ReviewHold()
                raise JobError(failure)
            raise RuntimeError("command failed")
        return "".join(lines)

    @staticmethod
    def _artifacts_valid(out, settings):
        if settings.get("pipeline") == PIXAL_PIPELINE:
            from engine.pixal_shape_job import valid
            return valid(out, settings)
        if settings.get("pipeline") == SF3D_OBSERVED_PIPELINE:
            from engine.sf3d_observed_job import valid
            return valid(out, settings)
        if settings.get("pipeline") == SF3D_PIPELINE:
            from engine.sf3d_shape_job import valid
            return valid(out, settings)
        artifacts = settings.get("artifact_sha256")
        required = {
            "model.glb",
            "editable.gltf",
            "editable.obj",
            "editable.ply",
            "texture.png",
        }
        shape = settings.get("pipeline") in SHAPE_PIPELINES
        guided = settings.get("pipeline") in GUIDED_PIPELINES
        extra = SHAPE_OUTPUTS | GUIDED_OUTPUTS if guided else SHAPE_OUTPUTS
        if guided:
            required.update(GUIDED_OUTPUTS)
        if shape:
            required.update(SHAPE_OUTPUTS, {"material.mtl", "material_0.png"})
        if not isinstance(artifacts, dict) or not required.issubset(artifacts):
            return False
        present_materials = {
            p.name
            for p in out.glob("material_*.png")
            if re.fullmatch(r"material_[0-9]+\.png", p.name) and p.is_file()
        }
        if not present_materials.issubset(artifacts):
            return False
        try:
            for name, expected in artifacts.items():
                if (
                    not isinstance(name, str)
                    or not (shape and name in extra or re.fullmatch(
                        r"(?:model\.glb|editable\.gltf|editable\.obj|editable\.ply|texture\.png|material\.mtl|material_[0-9]+\.png)",
                        name,
                    ))
                    or not isinstance(expected, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", expected)
                ):
                    return False
                if hashlib.sha256((out / name).read_bytes()).hexdigest() != expected:
                    return False
            if shape:
                appearance = json.loads((out / "appearance.json").read_text(encoding="utf-8"))
                generation = json.loads((out / "generation.json").read_text(encoding="utf-8"))
                if not (
                    appearance["output_sha256"] == artifacts["model.glb"] == settings["glb_sha256"]
                    and appearance["shape_sha256"] == artifacts["shape-master.glb"] == generation["shape_result"]["sha256"]
                    and appearance["foreground_sha256"] == artifacts["input-processed.png"] == generation["engine"]["input_sha256"]
                    and appearance["original_sha256"] == generation["foreground"]["source_sha256"] == settings["input_sha256"]
                ):
                    return False
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

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
        remote_space = config.get("pipeline") in SPACE_PIPELINES
        pixal = config.get("pipeline") == PIXAL_PIPELINE
        shape = config.get("pipeline") in SHAPE_PIPELINES
        guided = config.get("pipeline") in GUIDED_PIPELINES
        if (
            hashlib.sha256((folder / "input.image").read_bytes()).hexdigest()
            != self.get(job_id)["input_sha256"]
        ):
            raise RuntimeError("input hash mismatch")
        valid_inference = False
        previous = None
        try:
            previous = json.loads(settings_path.read_text(encoding="utf-8"))
            valid_inference = (
                previous["input_sha256"] == self.get(job_id)["input_sha256"]
                and previous["glb_sha256"]
                == hashlib.sha256(glb.read_bytes()).hexdigest()
                and (remote_space or (previous["marching_cubes_resolution"] == config["resolution"]
                              and previous["texture_size"] == config["texture_size"]))
                and (
                    not config.get("auto_finish")
                    or previous.get("auto_finish_complete") is True
                )
                and (
                    not config.get("pipeline")
                    or previous.get("pipeline") == config["pipeline"]
                )
                and (
                    not config.get("auto_finish")
                    or self._artifacts_valid(out, previous)
                )
            )
        except (OSError, ValueError, TypeError, KeyError):
            pass
        if not valid_inference:
            if not config.get("auto_finish"):
                raise JobError(
                    "このジョブは旧方式で作られており、元の仕上がりを復元できません。新しいジョブを作成してください。"
                )
            if (
                config.get("auto_finish")
                and isinstance(previous, dict)
                and previous.get("auto_finish_complete") is True
                and not shape and not remote_space
                and not (out / "raw-export" / "finish-checkpoint.json").is_file()
            ):
                raise RuntimeError(
                    "finished artifacts are damaged and no finish checkpoint exists"
                )
            if shape and isinstance(previous, dict) and previous.get("auto_finish_complete") is True:
                from engine.triposg_cpu import read_json, valid_artifact

                base = out / "local-guided-work" / "base" if guided else out
                work = base / "local-shape-work" / "triposg"
                try:
                    key = read_json(work / "identity.json")["job_key"]
                    checkpoint = valid_artifact(work / "raw-shape.glb", key)
                except (OSError, ValueError, KeyError, TypeError):
                    checkpoint = None
                if not checkpoint:
                    raise JobError("成果物と形状の再開用データを照合できません。ジョブのバックアップを確認してください。")
            # Only clear files inside this job's own output directory.
            if not shape and not remote_space:
                for p in out.iterdir():
                    if p.is_file():
                        p.unlink()
            if pixal:
                if config.get("public_cc0_sha256") != self.get(job_id)["input_sha256"]:
                    raise JobError(SHAPE_ERRORS["pixal_consent"])
                command = [py, "-u", "-m", "engine.pixal_shape_job", "--setup", str(self.directory / "engine.json"),
                           "--public-cc0-sha256", config["public_cc0_sha256"]]
            elif remote_space:
                module = "engine.sf3d_observed_job" if config["pipeline"] == SF3D_OBSERVED_PIPELINE else "engine.sf3d_shape_job"
                command = [py, "-u", "-m", module, "--setup", str(self.directory / "engine.json")]
            elif guided:
                command = [py, "-u", "-m", "engine.guided_shape_job", "--setup", str(self.directory / "engine.json")]
                if config.get("pipeline") in (HOST_PIPELINE, OBSERVED_PIPELINE):
                    command.extend(["--pipeline", config["pipeline"]])
            elif shape:
                command = [py, "-u", "-m", "engine.local_shape_job"]
            else:
                command = [py, str(self.root / "engine" / "infer.py")]
            command.extend([str(folder / "input.image"), str(out)])
            if not remote_space:
                command.extend(["--resolution", str(config["resolution"]), "--texture-size", str(config["texture_size"])])
            if not shape and not remote_space:
                command.extend(["--pipeline", config.get("pipeline", LEGACY_PIPELINE)])
            self._command(job_id, command, "starting")
        digest = hashlib.sha256(glb.read_bytes()).hexdigest()
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if (
            settings["glb_sha256"] != digest
            or settings["input_sha256"] != self.get(job_id)["input_sha256"]
            or (not remote_space and (settings["marching_cubes_resolution"] != config["resolution"]
                              or settings["texture_size"] != config["texture_size"]))
            or (
                config.get("auto_finish")
                and settings.get("auto_finish_complete") is not True
            )
            or (
                config.get("pipeline")
                and settings.get("pipeline") != config["pipeline"]
            )
        ):
            raise RuntimeError("hash mismatch")
        if config.get("auto_finish"):
            if not self._artifacts_valid(out, settings):
                raise RuntimeError("artifact hash mismatch")
        validator = self.root / ".runtime" / "validator" / "gltf_validator.exe"
        cmd = [py, str(self.root / "quality" / "inspection.py"), str(glb)]
        from quality.measured_regions import measured_profile

        profile = measured_profile(settings)
        profile_path = out / "automatic-profile.json"
        if profile is not None:
            self._atomic_json(profile_path, profile)
            cmd.extend(["--profile", str(profile_path)])
        else:
            profile_path.unlink(missing_ok=True)
        if validator.is_file():
            cmd.extend(["--validator", str(validator)])
        inspection = json.loads(self._command(job_id, cmd, "inspection"))
        inspection["profile_source"] = (
            "measured figure landmarks"
            if profile
            else "unavailable; geometry requires review"
        )
        if profile:
            inspection["profile"] = profile
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
        if config.get("pipeline") in (PIPELINE, *SPACE_PIPELINES, *SHAPE_PIPELINES):
            source = None
            raw = settings.get("raw_export", "")
            if remote_space:
                source = out / "input-processed.png"
                inspection["source_registration"] = {**settings["camera"], "status": "needs_review"}
            elif shape:
                source = out / "input-processed.png"
                appearance = json.loads((out / "appearance.json").read_text(encoding="utf-8"))
                inspection["source_registration"] = {
                    "geometry_master": "shape-master.glb",
                    "shape_sha256": appearance["shape_sha256"],
                    "source_agreement": appearance.get("source_agreement"),
                    "alignment": appearance.get("registration"),
                    "status": "needs_review",
                }
            elif re.fullmatch(r"raw-export/[0-9a-f]{32}", raw):
                source = out / raw / "foreground.png"
            if source is not None:
                report = json.loads(
                    self._command(
                        job_id,
                        [
                            py,
                            str(self.root / "quality" / "general_appearance.py"),
                            str(source),
                            str(out),
                            digest,
                            *(["--observed-view", "front"] if shape or (remote_space and not pixal) else []),
                        ],
                        "appearance",
                    )
                )
                inspection["image_comparison"] = report
            else:
                inspection["image_comparison"] = {
                    "status": "unknown",
                    "reason": "No measured foreground.",
                }
            self._atomic_json(out / "inspection.json", inspection)

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
        if self.worker.is_alive():
            raise RuntimeError("Job worker did not stop; store lock retained")
        self.group.close()
        self.db.close()
        _unlock_store(self.store_lock)
