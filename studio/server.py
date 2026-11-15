"""Loopback-only HTTP transport for the studio job core."""

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import urlsplit, parse_qs
from .core import JobCore, JobError, MAX_INPUT, ROOT, TERMINAL

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
        if path == "/api/doctor":
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
                self.sse(job_id, int(value))
                return
            if filename:
                if filename not in job["artifacts"]:
                    self.send_json(404, {"error": "成果物がありません。"})
                    return
                file = self.server.core.directory / job_id / "output" / filename
                self.send_bytes(
                    200,
                    file.read_bytes(),
                    mimetypes.guess_type(filename)[0] or "application/octet-stream",
                )
                return
            self.send_json(200, {"job": job})
        except KeyError:
            self.send_json(404, {"error": "ジョブがありません。"})

    def sse(self, job_id, cursor):
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
                    if not events and status not in TERMINAL:
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
                if status in TERMINAL:
                    return
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def do_POST(self):
        if not self.authorized(write=True):
            return
        path = urlsplit(self.path).path
        length = self.headers.get("Content-Length", "")
        if not length.isdigit() or int(length) > MAX_INPUT:
            self.close_connection = True
            self.send_json(413, {"error": "画像は20MB以下で選び直してください。"})
            return
        if path != "/api/jobs" and int(length) > 65536:
            self.close_connection = True
            self.send_json(413, {"error": "入力は64KB以下にしてください。"})
            return
        data = self.rfile.read(int(length))
        try:
            if path == "/api/jobs":
                job, created = self.server.core.create(
                    data, self.headers.get("Idempotency-Key")
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
