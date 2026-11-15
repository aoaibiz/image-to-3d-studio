"""Loopback-only HTTP transport for the studio job core."""

import argparse
import hashlib
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import urlsplit, parse_qs
from .core import JobCore, JobError, PipelineConflict, MAX_INPUT, ROOT, TERMINAL
from engine.pipeline import (GUIDED_PIPELINE, GUIDED_PIPELINES, HOST_PIPELINE, OBSERVATION_METHODS,
                             OBSERVED_PIPELINE, PIXAL_PIPELINE, PIXAL_RECIPIENT, SPACE_PIPELINES, SF3D_OBSERVED_PIPELINE, SG_CHART_PIPELINE,
                             SHAPE_PIPELINE, SHAPE_PIPELINES)

STATIC = {
    "/": "viewer/index.html",
    "/viewer.js": "viewer/viewer.js",
    "/studio-ui.js": "viewer/studio-ui.js",
    "/studio.css": "viewer/studio.css",
}
for name in (
    "build/three.module.js",
    "build/three.core.js",
    "examples/jsm/loaders/GLTFLoader.js",
    "examples/jsm/controls/OrbitControls.js",
    "examples/jsm/environments/RoomEnvironment.js",
    "examples/jsm/utils/BufferGeometryUtils.js",
):
    STATIC["/node_modules/three/" + name] = "viewer/node_modules/three/" + name


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address, core):
        super().__init__(address, Handler)
        self.core = core


def shape_doctor(core):
    from engine.local_shape_job import runtime_path
    from engine.triposg_cpu import verify_runtime
    from engine.triposg_setup import PROBE, WEIGHTS
    from tools.doctor import RUNTIME_PROBE, U2NET_SHA256, local_memory, verify_weight

    checks = {}
    status = {"engine": "TripoSG", "preflight": "setup_or_verification_required",
              "weight_verification": "unchecked", "torch_probe": "not_checked"}
    result = {"ok": False, "pipeline": SHAPE_PIPELINE, "local": True, "checks": checks,
              "diagnostics": {"ram": local_memory(), "disk_free_bytes": shutil.disk_usage(core.root).free,
                              "local_cpu": status},
              "message": "通常のセットアップと engine/setup-triposg.ps1 の実行結果を確認してください。"}
    try:
        runtime = runtime_path()
    except ValueError:
        result["message"] = "STUDIO_TRIPOSG_RUNTIME には絶対パスを指定してください。"
        return result
    base = core.root / ".runtime"
    python = base / "venv/Scripts/python.exe"
    shape_python = runtime / "venv/Scripts/python.exe"
    model_dir = runtime / "weights"
    foreground = base / "u2net/u2net.onnx"
    source = runtime / "source/triposg"
    resources = [model_dir / name for name in WEIGHTS]
    checks.update(python=python.is_file(), shape_python=shape_python.is_file(),
                  shape_source=(source / "models/transformers/triposg_transformer.py").is_file(),
                  model=all(p.is_file() for p in resources), foreground_model=foreground.is_file(),
                  validator=(base / "validator/gltf_validator.exe").is_file(), node=bool(shutil.which("node")),
                  viewer=(core.root / "viewer/node_modules/three/build/three.module.js").is_file())
    if not all(checks.values()):
        return result
    # Cache successful multi-GB verification while the local files are unchanged.
    files = [python, shape_python, foreground, *resources, *sorted(source.rglob("*.py"))]
    try:
        signature = tuple((str(p), s.st_size, s.st_mtime_ns) for p in files for s in (p.stat(),))
    except OSError:
        return result
    with core.changed:
        if getattr(core, "shape_doctor_signature", None) == signature:
            result["ok"] = True
            result["diagnostics"]["local_cpu"] = core.shape_doctor_status.copy()
            result["message"] = "ローカル TripoSG の重み、実行環境、着色ツールを確認しました。"
            return result
        if core.active or core.checking:
            status["preflight"] = "not_checked"
            result["message"] = "処理中のため診断を保留しました。終了後にもう一度確認してください。"
            return result
        core.checking.add("doctor")
    try:
        verify_runtime(runtime)
        status["weight_verification"] = "verified"
        status["foreground_weight_verification"] = verify_weight(foreground, U2NET_SHA256, True)
        if status["foreground_weight_verification"] != "verified":
            return result
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HUB_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="1")
        for cmd in ([str(python), "-I", "-B", "-c", RUNTIME_PROBE],
                    [str(shape_python), "-I", "-B", "-c", PROBE, str(runtime / "source")]):
            subprocess.run(cmd, env=env, capture_output=True, check=True, timeout=70)
        status.update(torch_probe="verified", preflight="ready_to_try")
        result.update(ok=True, message="ローカル TripoSG の重み、実行環境、着色ツールを確認しました。")
        with core.changed:
            core.shape_doctor_signature = signature
            core.shape_doctor_status = status.copy()
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    finally:
        with core.changed:
            core.checking.discard("doctor")
            core.changed.notify_all()
    return result


def space_engine_info(pipeline):
    if pipeline == PIXAL_PIPELINE:
        return {"pipeline": pipeline, "local": False, "recipient": PIXAL_RECIPIENT, "public_cc0_only": True}

    info = {"pipeline": pipeline, "local": False}
    if pipeline == SF3D_OBSERVED_PIPELINE:
        return {**info, "recipients": ["Stability AI / Hugging Face", "OpenAI ChatGPT"]}
    return {**info, "recipient": "Stability AI / Hugging Face"}


def add_codex_checks(result, *, observation):
    from engine.guided_provider import preflight
    paint = preflight()
    result["diagnostics"]["guided_paint"] = paint
    if not paint.get("available"):
        result.update(ok=False, message=paint.get("reason") or "Codexの画像生成環境を確認してください。")
    if observation:
        from engine.codex_photo_observations import preflight as observer_preflight
        photo = {**observer_preflight(), "method": OBSERVATION_METHODS[-1],
                 "recoverable_methods": list(OBSERVATION_METHODS)}
        result["diagnostics"]["photo_observer"] = photo
        if not photo.get("available"):
            result.update(ok=False, message=photo.get("reason") or "元写真を解析するCodexの環境を確認してください。")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def authorized(self, write=False):
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "").lower()
        origin = self.headers.get("Origin")
        if host not in hosts or (
            origin and origin.lower() not in {f"http://{h}" for h in hosts}
        ):
            self.close_connection = True
            self.send_json(403, {"error": "ローカル画面からアクセスしてください。"})
            return False
        if write and self.headers.get("X-Studio-Client") != "1":
            self.close_connection = True
            self.send_json(403, {"error": "ローカル画面から操作してください。"})
            return False
        return True

    def send_json(self, status, value):
        self.send_bytes(
            status,
            json.dumps(value, ensure_ascii=False).encode(),
            "application/json; charset=utf-8",
        )

    def send_bytes(self, status, data, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self' blob: data:; base-uri 'none'; form-action 'none'",
        )
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.authorized():
            return
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/favicon.ico":
            self.send_bytes(204, b"", "image/x-icon")
            return
        if path == "/api/engine":
            pipeline = self.server.core.pipeline
            if pipeline in SPACE_PIPELINES:
                self.send_json(200, space_engine_info(pipeline))
                return
            self.send_json(200, {"pipeline": pipeline, "local": pipeline not in GUIDED_PIPELINES,
                                 **({"recipient": "OpenAI ChatGPT"} if pipeline in GUIDED_PIPELINES else {})})
            return
        if path == "/api/doctor":
            if self.server.core.pipeline in SPACE_PIPELINES:
                checks = {name: (ROOT / path).is_file() for name, path in {
                    "python": ".runtime/venv/Scripts/python.exe", "foreground_model": ".runtime/u2net/u2net.onnx",
                    "validator": ".runtime/validator/gltf_validator.exe", "viewer": "viewer/node_modules/three/build/three.module.js",
                }.items()}
                if self.server.core.pipeline == PIXAL_PIPELINE:
                    checks.pop("foreground_model")
                checks["node"] = bool(shutil.which("node"))
                result = {**space_engine_info(self.server.core.pipeline), "ok": all(checks.values()), "checks": checks,
                          "diagnostics": {"remote_availability": "未確認。実行時に公式Spaceの混雑・匿名利用枠を確認します。"}}
                if self.server.core.pipeline == PIXAL_PIPELINE:
                    result["diagnostics"]["remote_availability"] = "個人Spaceの稼働状況は送信前に確認します。匿名GPUの残り時間は未確認です。"
                if self.server.core.pipeline == SF3D_OBSERVED_PIPELINE:
                    add_codex_checks(result, observation=True)
                self.send_json(200, result)
                return
            if self.server.core.pipeline in SHAPE_PIPELINES:
                result = shape_doctor(self.server.core)
                if self.server.core.pipeline in (GUIDED_PIPELINE, HOST_PIPELINE):
                    from engine.guided_shape_job import vision_runtime_path
                    from engine.visual_brief_setup import ASSETS, EXECUTABLE, PROJECTOR, WEIGHTS
                    try:
                        vision = vision_runtime_path()
                        present = (vision / "bin" / EXECUTABLE).is_file() and all(
                            (vision / name).stat().st_size == ASSETS[name][0]
                            for name in (WEIGHTS, PROJECTOR)
                        )
                    except (OSError, ValueError):
                        present = False
                    result["diagnostics"]["local_visual_analysis"] = {
                        "model": "Qwen3-VL-2B-Instruct", "files_present": present,
                        "integrity": "verified before inference",
                    }
                    if not present:
                        result.update(ok=False, message="ローカル画像解析の初期設定を実行してください。")
                if self.server.core.pipeline == HOST_PIPELINE:
                    result.update(pipeline=HOST_PIPELINE, local=False, recipient="OpenAI ChatGPT")
                    result["diagnostics"]["guided_paint"] = {"provider": "Codex host / OpenAI ImageGen",
                                                             "host_image_tool": "unverified until a connected host returns a generated image"}
                if self.server.core.pipeline in (GUIDED_PIPELINE, OBSERVED_PIPELINE, SG_CHART_PIPELINE):
                    result.update(pipeline=self.server.core.pipeline, local=False, recipient="OpenAI ChatGPT")
                    add_codex_checks(result, observation=self.server.core.pipeline in (OBSERVED_PIPELINE, SG_CHART_PIPELINE))
                self.send_json(200, result)
                return
            checks = {
                "python": (ROOT / ".runtime/venv/Scripts/python.exe").is_file(),
                "model": (ROOT / ".runtime/model/model.ckpt").is_file(),
                "foreground_model": (ROOT / ".runtime/u2net/u2net.onnx").is_file(),
                "validator": (ROOT / ".runtime/validator/gltf_validator.exe").is_file(),
                "node": bool(shutil.which("node")),
                "viewer": (
                    ROOT / "viewer/node_modules/three/build/three.module.js"
                ).is_file(),
            }
            core = self.server.core
            weight_key = []
            for relative in (
                "model/model.ckpt",
                "model/config.yaml",
                "model/dino-config.json",
                "u2net/u2net.onnx",
            ):
                try:
                    stat = (ROOT / ".runtime" / relative).stat()
                    weight_key.append((stat.st_size, stat.st_mtime_ns))
                except OSError:
                    weight_key.append(None)
            with core.changed:
                busy = bool(core.active or core.checking)
                if not busy and (
                    not getattr(core, "doctor_cache", None)
                    or getattr(core, "doctor_weight_key", None) != weight_key
                ):
                    core.checking.add("doctor")
                    should_probe = True
                else:
                    should_probe = False
                cached = getattr(core, "doctor_cache", None)
                if getattr(core, "doctor_weight_key", None) != weight_key:
                    cached = None
            if should_probe:
                try:
                    probe = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "tools" / "doctor.py"),
                            "--verify-weights",
                        ],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        timeout=70,
                        check=True,
                    )
                    cached = json.loads(probe.stdout)
                    # A repaired installation must be diagnosable in the same session.
                    ready = cached.get("local_cpu", {}).get("preflight") == "ready_to_try"
                    with core.changed:
                        core.doctor_cache = cached if ready else None
                        core.doctor_weight_key = weight_key
                except (OSError, ValueError, subprocess.SubprocessError):
                    cached = None
                finally:
                    with core.changed:
                        core.checking.discard("doctor")
                        core.changed.notify_all()
            if cached is None:
                from tools.doctor import local_memory, gpu_inventory
                import platform

                cached = {
                    "os": platform.platform(),
                    "architecture": platform.machine(),
                    "ram": local_memory(),
                    "graphics": gpu_inventory(),
                    "disk_free_bytes": shutil.disk_usage(ROOT).free,
                    "local_cpu": {
                        "engine": "TripoSR",
                        "weight_verification": "unchecked",
                        "torch_probe": "deferred",
                        "preflight": "not_checked",
                    },
                    "trellis2": {"implemented": False},
                    "tripo_cloud": {"implemented": False},
                }
            self.send_json(
                200,
                {
                    "ok": all(checks.values())
                    and cached.get("local_cpu", {}).get("preflight") == "ready_to_try",
                    "checks": checks,
                    "pipeline": core.pipeline,
                    "local": True,
                    "diagnostics": cached,
                    "message": "準備状況を確認しました。未準備の項目はセットアップしてください。",
                },
            )
            return
        if path in STATIC:
            file = ROOT / STATIC[path]
            if file.is_file():
                mime = mimetypes.guess_type(str(file))[0] or "application/octet-stream"
                data = file.read_bytes()
                if path == "/":
                    data = data.replace(
                        b"</head>", b'<link rel="stylesheet" href="/studio.css"></head>'
                    )
                    data = data.replace(
                        b"</body>",
                        b'<script src="/studio-ui.js" defer></script></body>',
                    )
                self.send_bytes(
                    200,
                    data,
                    mime
                    + (
                        "; charset=utf-8"
                        if mime.startswith("text/") or mime == "application/javascript"
                        else ""
                    ),
                )
            else:
                self.send_json(404, {"error": "ファイルがありません。"})
            return
        match = re.fullmatch(
            r"/api/jobs/([0-9a-f-]{36})(?:/(events|artifacts/([^/]+)))?", path
        )
        if not match:
            self.send_json(404, {"error": "見つかりません。"})
            return
        job_id, action, filename = match.groups()
        try:
            job = self.server.core.get(job_id)
            if action == "events":
                value = (
                    self.headers.get("Last-Event-ID")
                    or parse_qs(parsed.query).get("after", ["0"])[0]
                )
                if not value.isdigit():
                    self.send_json(400, {"error": "イベント位置を確認してください。"})
                    return
                self.sse(job_id, int(value), stop_for_host=self.headers.get("X-Studio-Yield-Host") == "1")
                return
            if filename:
                if filename not in job["artifacts"]:
                    self.send_json(404, {"error": "成果物がありません。"})
                    return
                file = self.server.core.directory / job_id / "output" / filename
                if filename in job.get("shape_hypothesis_files", {}):
                    data = file.read_bytes()
                    if hashlib.sha256(data).hexdigest() != job["shape_hypothesis_files"][filename]:
                        self.send_json(404, {"error": "形状の検品記録とファイルが一致しません。"})
                        return
                    self.send_bytes(200, data, mimetypes.guess_type(filename)[0] or "application/octet-stream")
                    return
                if job["status"] == "review_hold":
                    hold = job["hold"]
                    if filename == "review-hold.json":
                        self.send_json(200, hold)
                        return
                    expected = hold["preview_sha256"] if filename == "review-native.glb" else hold["image_sha256"]
                    with file.open("rb") as stream:
                        data = stream.read((49 << 20) + 1)
                    if len(data) > (49 << 20) or hashlib.sha256(data).hexdigest() != expected:
                        self.send_json(404, {"error": "検品用データの照合に失敗しました。"})
                        return
                    self.send_bytes(200, data, "model/gltf-binary" if filename.endswith(".glb") else "image/png")
                    return
                self.send_bytes(
                    200,
                    file.read_bytes(),
                    mimetypes.guess_type(filename)[0] or "application/octet-stream",
                )
                return
            self.send_json(200, {"job": job})
        except KeyError:
            self.send_json(404, {"error": "ジョブがありません。"})

    def sse(self, job_id, cursor, *, stop_for_host=False):
        stop = TERMINAL | ({"awaiting_host_image"} if stop_for_host else set())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            while True:
                with self.server.core.changed:
                    events = self.server.core.events(job_id, cursor)
                    status = self.server.core.get(job_id)["status"]
                    if not events and status not in stop:
                        self.server.core.changed.wait(timeout=15)
                        events = self.server.core.events(job_id, cursor)
                        status = self.server.core.get(job_id)["status"]
                for event in events:
                    cursor = event["id"]
                    self.wfile.write(
                        (
                            "id: "
                            + str(cursor)
                            + "\ndata: "
                            + json.dumps(event, ensure_ascii=False)
                            + "\n\n"
                        ).encode()
                    )
                if not events:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
                if status in stop:
                    return
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def do_POST(self):
        if not self.authorized(write=True):
            return
        path = urlsplit(self.path).path
        host_action = re.fullmatch(r"/api/jobs/([0-9a-f-]{36})/host/(claim|submit)", path)
        host_token = self.headers.get("X-Studio-Host-Token")
        if host_action:
            from engine.host_paint import authorize
            if not authorize(self.server.core.directory, host_token):
                self.close_connection = True
                self.send_json(403, {"error": "この端末のホスト送信設定を確認してください。"})
                return
        length = self.headers.get("Content-Length", "")
        if not length.isdigit() or int(length) > MAX_INPUT:
            self.close_connection = True
            self.send_json(413, {"error": "画像は20MB以下で選び直してください。"})
            return
        if path != "/api/jobs" and not (host_action and host_action[2] == "submit") and int(length) > 65536:
            self.close_connection = True
            self.send_json(413, {"error": "入力は64KB以下にしてください。"})
            return
        data = self.rfile.read(int(length))
        try:
            if host_action:
                submission = None
                if host_action[2] == "submit":
                    if self.headers.get("Content-Type") != "image/png":
                        raise JobError("画像生成ツールのPNGを指定してください。")
                    submission = dict(claim_id=self.headers.get("X-Studio-Claim"),
                                      request_sha=self.headers.get("X-Studio-Request-Sha256"),
                                      sheet_sha=self.headers.get("X-Studio-Paint-Sha256"),
                                      tool_call=self.headers.get("X-Studio-Tool-Call"), png=data)
                elif data:
                    raise JobError("受付に追加の入力は必要ありません。")
                self.send_json(200, self.server.core.host_transfer(host_action[1], host_token, submission))
                return
            if path == "/api/jobs":
                job, created = self.server.core.create(
                    data, self.headers.get("Idempotency-Key"), self.headers.get("X-Studio-Pipeline"),
                    self.headers.get("X-Studio-CC0-Source")
                )
                self.send_json(201 if created else 200, {"job": job})
                return
            match = re.fullmatch(
                r"/api/jobs/([0-9a-f-]{36})/(cancel|resume|review|inspect)", path
            )
            if match:
                action = match.group(2)
                if action in ("review", "inspect"):
                    try:
                        value = json.loads(data)
                    except (ValueError, UnicodeError):
                        raise JobError("JSON形式の入力を確認してください。")
                    job = getattr(self.server.core, action)(match.group(1), value)
                else:
                    job = getattr(self.server.core, action)(match.group(1))
                self.send_json(200, {"job": job})
                return
            self.send_json(404, {"error": "見つかりません。"})
        except KeyError:
            self.send_json(404, {"error": "ジョブがありません。"})
        except PipelineConflict as exc:
            self.send_json(409, {"error": str(exc)})
        except PermissionError as exc:
            self.send_json(403, {"error": str(exc)})
        except JobError as exc:
            self.send_json(400, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=47631)
    parser.add_argument("--jobs-dir", type=Path)
    args = parser.parse_args()
    # Bind first: the listening socket is the exclusive owner lock, before recovery touches SQLite.
    server = Server(("127.0.0.1", args.port), None)
    try:
        server.core = JobCore(directory=args.jobs_dir)
        print(f"http://127.0.0.1:{server.server_address[1]}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if server.core:
            server.core.close()


if __name__ == "__main__":
    main()
