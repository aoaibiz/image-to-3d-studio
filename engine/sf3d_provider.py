"""Opt-in client for the official SF3D anonymous Gradio Space; no model weights."""

import hashlib
import io
import json
import os
import re
import struct
import time
import uuid
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests
from PIL import Image

from engine.triposg_cpu import digest, read_json, writer_lock

ORIGIN = "https://stabilityai-stable-fast-3d.hf.space"
APP = "stabilityai/stable-fast-3d"
MIB = 1024**2
MAX_GLB = 48 * MIB
SETTINGS = {
    "foreground_ratio": 0.85,
    "remesh": "None",
    "vertex_count": -1,
    "texture_size": 1024,
}
EVENT = re.compile(r"[a-fA-F0-9]{32}")
GRADIO_FILE = re.compile(
    r"/tmp/gradio/[a-zA-Z0-9_-]{8,128}/[a-zA-Z0-9_-]{1,140}\.(png|glb)"
)
TEMP_GLB = re.compile(r"/tmp/tmp[a-zA-Z0-9_-]{5,80}\.glb")


class Pending(RuntimeError):
    """Delivery or remote progress is uncertain. Resume with this same work directory."""


class RemoteFailure(RuntimeError):
    """A terminal provider error. This job will not be resubmitted."""


def checked_path(value, suffix):
    if not isinstance(value, str) or not value.endswith(suffix):
        raise ValueError("SF3D returned an unexpected file type.")
    if not GRADIO_FILE.fullmatch(value) and not (
        suffix == ".glb" and TEMP_GLB.fullmatch(value)
    ):
        raise ValueError("SF3D returned an unexpected remote path.")
    return value


def output_path(data):
    if not isinstance(data, list) or len(data) != 6:
        raise ValueError("Unexpected SF3D completion layout.")
    result = data[4]
    if isinstance(result, dict):
        result = result.get("value", result)
    if isinstance(result, str):
        return checked_path(result, ".glb")
    if not isinstance(result, dict):
        raise TypeError("Missing SF3D output file.")
    path = checked_path(result.get("path"), ".glb")
    if result.get("url") is not None:
        url = urlsplit(result["url"])
        if (
            url.scheme != "https"
            or url.netloc != urlsplit(ORIGIN).netloc
            or url.query
            or url.fragment
            or url.path != "/file=" + path
        ):
            raise ValueError("Unexpected SF3D file origin or location.")
    return path


def upload_png(raw):
    if len(raw) > 32 * MIB:
        raise ValueError("Use a foreground PNG below 32 MiB.")
    with Image.open(io.BytesIO(raw)) as im:
        if (
            im.format != "PNG"
            or im.mode != "RGBA"
            or im.n_frames != 1
            or im.width * im.height > 40_000_000
        ):
            raise ValueError("A single RGBA foreground PNG of up to 40 MP is required.")
        if min(im.size) < 8:
            raise ValueError("Foreground too small.")

        # A matte can retain an entire original photo in invisible RGB. Remove it
        # before resampling, then clear any faint edge samples produced by the filter.
        def clear_background(frame):
            mask = frame.getchannel("A").point([0] * 48 + [255] * 208)
            return Image.composite(frame, Image.new("RGBA", frame.size), mask)

        im = clear_background(im)
        im.thumbnail((1152, 1152))
        im = clear_background(im)
        alpha = im.getchannel("A")
        lo, hi = alpha.getextrema()
        if lo > 32 or hi < 150:
            raise ValueError(
                "Source needs an opaque subject and a transparent background."
            )
        # Copy only actual pixels; EXIF, ICC profiles, names and text do not leave the host.
        clean = Image.frombytes("RGBA", im.size, im.tobytes())
        out = io.BytesIO()
        clean.save(out, "PNG")
        return out.getvalue()


def bounded(response, limit):
    length = response.headers.get("Content-Length")
    if length is not None and not 0 <= int(length) <= limit:
        raise ValueError("SF3D response exceeds the size limit.")
    total = 0
    for chunk in response.iter_content(16384):
        total += len(chunk)
        if total > limit:
            raise ValueError("SF3D response exceeds the size limit.")
        yield chunk


class Space:
    """Anonymous fixed-origin transport. No implicit Hugging Face credentials or redirects."""

    def __init__(self):
        self.http = requests.Session()
        self.http.trust_env = False

    def close(self):
        self.http.close()

    def request(self, method, path, **kwargs):
        response = self.http.request(
            method,
            ORIGIN + path,
            allow_redirects=False,
            stream=True,
            timeout=(12, 60),
            **kwargs,
        )
        if response.status_code != 200:
            status = response.status_code
            response.close()
            raise requests.HTTPError(f"SF3D Space HTTP {status}")
        return response

    def json(self, method, path, limit=65536, **kwargs):
        with self.request(method, path, **kwargs) as res:
            return json.loads(b"".join(bounded(res, limit)))

    def configuration(self):
        config = self.json("GET", "/config", 2 * MIB)
        if config.get("protocol") != "sse_v3" or config.get("version") != "4.41.0":
            raise ValueError("Unsupported SF3D Space/Gradio API version.")
        result = {"gradio_version": config["version"], "protocol": config["protocol"]}
        for key, count in (("requires_bg_remove", 2), ("run_button", 7)):
            matches = [d for d in config["dependencies"] if d.get("api_name") == key]
            if (
                len(matches) != 1
                or len(matches[0]["inputs"]) != count
                or len(matches[0]["outputs"]) != 6
                or type(matches[0]["id"]) is not int
            ):
                raise ValueError("Unsupported SF3D Space function signature.")
            result[key] = matches[0]["id"]
        # Public metadata only; no photo, filename, token or session identifier is sent here.
        with self.http.get(
            "https://huggingface.co/api/spaces/" + APP,
            allow_redirects=False,
            stream=True,
            timeout=(12, 30),
        ) as res:
            if res.status_code != 200:
                raise RuntimeError("Cannot verify the official SF3D Space metadata.")
            info = json.loads(b"".join(bounded(res, 2 * MIB)))
        if info.get("id") != APP or not re.fullmatch(
            r"[a-f0-9]{40}", str(info.get("sha"))
        ):
            raise ValueError("Unexpected SF3D Space identity.")
        result["space_head_revision"] = info["sha"]
        result["runtime_revision"] = info.get("runtime", {}).get("sha")
        return result

    def upload(self, png):
        payload = self.json(
            "POST", "/upload", files={"files": ("foreground.png", png, "image/png")}
        )
        if not isinstance(payload, list) or len(payload) != 1:
            raise ValueError("Unexpected SF3D upload response.")
        return checked_path(payload[0], ".png")

    def join(self, session, fn, data):
        payload = self.json(
            "POST",
            "/queue/join",
            json={
                "data": data,
                "event_data": None,
                "fn_index": fn,
                "trigger_id": None,
                "session_hash": session,
            },
        )
        event_id = payload.get("event_id")
        if not isinstance(event_id, str) or not EVENT.fullmatch(event_id):
            raise ValueError("No valid SF3D queue event ID.")
        return event_id

    def events(self, session):
        deadline = time.monotonic() + 540
        with self.request(
            "GET",
            "/queue/data",
            params={"session_hash": session},
            headers={"Accept": "text/event-stream"},
        ) as res:
            if (
                res.headers.get("Content-Type", "").split(";")[0].strip()
                != "text/event-stream"
            ):
                raise ValueError("Expected an SF3D event stream.")
            buffer = b""
            for chunk in bounded(res, 8 * MIB):
                if time.monotonic() >= deadline:
                    raise TimeoutError("SF3D event deadline elapsed.")
                lines = (buffer + chunk).split(b"\n")
                buffer = lines.pop()
                if len(buffer) > 512 * 1024:
                    raise ValueError("SF3D event line too long.")
                for line in lines:
                    if line.startswith(b"data:"):
                        yield json.loads(line[5:])

    def download(self, path, dest):
        with (
            self.request(
                "GET", "/file=" + quote(checked_path(path, ".glb"), safe="/")
            ) as res,
            dest.open("wb") as f,
        ):
            for chunk in bounded(res, MAX_GLB):
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())


def native_glb(path):
    """Check the expected sealed static indexed textured GLB, before downstream Khronos QA."""
    size = path.stat().st_size
    if not 40 <= size <= MAX_GLB:
        raise ValueError("Invalid SF3D file size.")
    with path.open("rb") as f:
        magic, version, total, n, kind = struct.unpack("<4sIIII", f.read(20))
        if (
            (magic, version, total, kind) != (b"glTF", 2, size, 0x4E4F534A)
            or n % 4
            or n > min(2 * MIB, size - 28)
        ):
            raise ValueError("Invalid SF3D GLB header.")
        doc = json.loads(f.read(n))
        length, kind = struct.unpack("<II", f.read(8))
        if kind != 0x004E4942 or length % 4 or length != size - 28 - n:
            raise ValueError("Unexpected binary buffer.")
        binary = f.read(length)
    buffers = doc.get("buffers", [])
    if (
        doc["asset"]["version"] != "2.0"
        or len(buffers) != 1
        or "uri" in buffers[0]
        or not 0 <= length - buffers[0]["byteLength"] <= 3
    ):
        raise ValueError("SF3D must embed its complete mesh buffer.")
    if any(
        doc.get(k)
        for k in ("extensionsUsed", "extensionsRequired", "skins", "animations")
    ):
        raise ValueError("Unsupported SF3D extension or animation.")

    def ref(items, index):
        if type(index) is not int or not 0 <= index < len(items):
            raise ValueError("Invalid SF3D index.")
        return items[index]

    views = doc["bufferViews"]
    for v in views:
        start, count = v.get("byteOffset", 0), v["byteLength"]
        if (
            v.get("buffer") != 0
            or min(start, count) < 0
            or start + count > buffers[0]["byteLength"]
        ):
            raise ValueError("Invalid SF3D buffer range.")
    if not 1 <= len(doc["images"]) <= 8:
        raise ValueError("Expected embedded SF3D textures.")
    for img in doc["images"]:
        if "uri" in img or img.get("mimeType") not in ("image/jpeg", "image/png"):
            raise ValueError("Unexpected SF3D texture type or URI.")
        v = ref(views, img["bufferView"])
        start = v.get("byteOffset", 0)
        encoded = binary[start : start + v["byteLength"]]
        with Image.open(io.BytesIO(encoded)) as tex:
            if (
                tex.width * tex.height > 4_194_304
                or Image.MIME.get(tex.format) != img["mimeType"]
            ):
                raise ValueError("Unexpected SF3D texture size or format.")
            tex.verify()
        with Image.open(io.BytesIO(encoded)) as tex:
            tex.load()
    for a in doc["accessors"]:
        v = ref(views, a["bufferView"])
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        elem = width * {5121: 1, 5123: 2, 5125: 4, 5126: 4}[a["componentType"]]
        offset = a.get("byteOffset", 0)
        count = a["count"]
        stride = v.get("byteStride", elem)
        if (
            "sparse" in a
            or type(count) is not int
            or not 0 < count <= 6_000_000
            or offset < 0
            or stride < elem
            or offset + (count - 1) * stride + elem > v["byteLength"]
        ):
            raise ValueError("Invalid SF3D mesh accessor.")
    # A scene must really instantiate its mesh. The two sampled native exports are static scenes.
    nodes = doc["nodes"]
    roots = ref(doc["scenes"], doc.get("scene", 0))["nodes"]
    stack = list(roots)
    reached = set()
    instantiated = set()
    while stack:
        index = stack.pop()
        node = ref(nodes, index)
        if index in reached:
            raise ValueError("Malformed SF3D scene hierarchy.")
        reached.add(index)
        stack.extend(node.get("children", []))
        if "mesh" in node:
            ref(doc["meshes"], node["mesh"])
            instantiated.add(node["mesh"])
    if not instantiated or instantiated != set(range(len(doc["meshes"]))):
        raise ValueError("Missing or unreachable SF3D geometry.")
    for tx in doc["textures"]:
        ref(doc["images"], tx["source"])
    import numpy as np

    def array(a):
        v = views[a["bufferView"]]
        dt = np.dtype(
            {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[a["componentType"]]
        )
        w = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        return np.ndarray(
            (a["count"], w),
            dt,
            buffer=binary,
            offset=v.get("byteOffset", 0) + a.get("byteOffset", 0),
            strides=(v.get("byteStride", dt.itemsize * w), dt.itemsize),
        )

    total_faces = 0
    for mesh in doc["meshes"]:
        for prim in mesh["primitives"]:
            if prim.get("mode", 4) != 4 or "targets" in prim:
                raise ValueError("Expected static SF3D triangles.")
            pos, uv, normal = [
                ref(doc["accessors"], prim["attributes"][k])
                for k in ("POSITION", "TEXCOORD_0", "NORMAL")
            ]
            indices = ref(doc["accessors"], prim["indices"])
            if (
                (pos["type"], uv["type"], normal["type"], indices["type"])
                != ("VEC3", "VEC2", "VEC3", "SCALAR")
                or indices["count"] % 3
                or len({a["count"] for a in (pos, uv, normal)}) != 1
            ):
                raise ValueError("Invalid indexed textured geometry.")
            if any(
                a["componentType"] != 5126 or not np.isfinite(array(a)).all()
                for a in (pos, uv, normal)
            ):
                raise ValueError("Invalid SF3D positions, normals or UV values.")
            if (
                indices["componentType"] not in (5121, 5123, 5125)
                or int(array(indices).max()) >= pos["count"]
            ):
                raise ValueError("SF3D triangles reference missing vertices.")
            mat = ref(doc["materials"], prim["material"])
            texture = ref(
                doc["textures"],
                mat["pbrMetallicRoughness"]["baseColorTexture"]["index"],
            )
            ref(doc["images"], texture["source"])
            total_faces += indices["count"] // 3
    if not 0 < total_faces <= 2_000_000:
        raise ValueError("Invalid SF3D triangle count.")
    return {
        "faces": total_faces,
        "embedded_images": len(doc["images"]),
        "bytes": size,
        "sha256": digest(path),
    }


def _save(work, record):
    tmp = work / "provider.json.part"
    with tmp.open("w", encoding="utf8") as f:
        json.dump(record, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, work / "provider.json")
    if os.name == "posix":
        fd = os.open(work, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _stage(work, r, stage, progress, **values):
    r.update(values, stage=stage)
    _save(work, r)
    if progress:
        progress(stage)


def _abort(cancelled):
    if cancelled and cancelled():
        raise InterruptedError(
            "SF3Dの待機を中断しました。本人が再開すると同じ受付に復帰します。"
        )


def _before_post(work, r, stage, progress, cancelled, **values):
    _abort(cancelled)
    previous = r["stage"]
    _stage(work, r, stage, progress, **values)
    try:
        _abort(cancelled)
    except InterruptedError:
        # Known not sent; the next explicit resume can still perform this step.
        _stage(work, r, previous, None)
        raise


def _queue(work, r, api, step, fn, data, progress, cancelled):
    event_key = step + "_event_id"
    waiting = step + "_waiting"
    if r["stage"] != waiting:
        # Committed BEFORE the POST. A crash or lost response must never cause a second join.
        _before_post(work, r, waiting, progress, cancelled, **{event_key: None})
        try:
            event = api.join(r["session_hash"], fn, data)
        except (requests.RequestException, OSError, ValueError, RuntimeError) as exc:
            raise Pending(
                "SF3Dの受付結果が不明です。同じジョブを再開してイベントを受信してください。"
            ) from exc
        r[event_key] = event
        _save(work, r)
    try:
        for message in api.events(r["session_hash"]):
            _abort(cancelled)
            if not isinstance(message, dict):
                raise TypeError("Unexpected event payload.")
            event = message.get("event_id")
            if not isinstance(event, str) or not EVENT.fullmatch(event):
                continue
            if not r[event_key]:
                if step == "generation" and event == r.get("preprocess_event_id"):
                    continue
                r[event_key] = event
                _save(work, r)
            if event != r[event_key]:
                continue
            typ = message.get("msg")
            if typ in ("process_starts", "estimation") and r.get("queue_event") != typ:
                r["queue_event"] = typ
                _save(work, r)
                if progress:
                    progress(step + "_" + typ)
            if typ in ("unexpected_error", "process_completed"):
                if typ == "unexpected_error" or message.get("success") is not True:
                    _stage(work, r, "failed", progress, failure="remote_error")
                    raise RemoteFailure(
                        "SF3Dの処理が失敗しました。提供元の稼働状況と匿名利用枠を確認してください。"
                    )
                output = message.get("output")
                if not isinstance(output, dict) or not isinstance(
                    output.get("data"), list
                ):
                    _stage(work, r, "failed", progress, failure="invalid_completion")
                    raise RemoteFailure("SF3Dの完了イベントに成果物情報がありません。")
                return output["data"]
    except (requests.RequestException, OSError, TypeError, ValueError) as exc:
        if isinstance(exc, InterruptedError):
            raise
        raise Pending(
            "SF3Dのイベント受信が途切れました。同じジョブを再開してください。"
        ) from exc
    raise Pending(
        "SF3Dから完了通知を受け取れませんでした。同じジョブを再開してください。"
    )


def generate(
    foreground,
    source_sha256,
    work,
    *,
    allow_official_space=False,
    progress=None,
    cancelled=None,
):
    """Produce work/model.glb from one source-bound matte and return non-session metadata."""
    api = Space()
    try:
        return _generate(
            foreground,
            source_sha256,
            work,
            allow_official_space,
            progress,
            cancelled,
            api,
        )
    finally:
        api.close()


def _generate(foreground, source_sha256, work, allow, progress, cancelled, api):
    if not isinstance(source_sha256, str) or not re.fullmatch(
        r"[a-f0-9]{64}", source_sha256
    ):
        raise ValueError("元画像のSHA-256が必要です。")
    foreground, work = Path(foreground), Path(work)
    with foreground.open("rb") as stream:
        raw = stream.read(32 * MIB + 1)
    png = upload_png(raw)
    identity = {
        "source_sha256": source_sha256,
        "foreground_sha256": hashlib.sha256(raw).hexdigest(),
        "upload_sha256": hashlib.sha256(png).hexdigest(),
    }
    work.mkdir(parents=True, exist_ok=True)
    with writer_lock(work):
        file = work / "provider.json"
        model = work / "model.glb"
        r = read_json(file) if file.exists() else None
        if r:
            if (
                r.get("provider") != APP
                or r.get("recipient") != ORIGIN
                or r.get("settings") != SETTINGS
                or not EVENT.fullmatch(str(r.get("session_hash")))
            ):
                raise ValueError("SF3Dの受付設定を照合できません。")
            if any(r.get(k) != v for k, v in identity.items()):
                raise ValueError("既存のSF3Dジョブと入力画像が一致しません。")
            if r["stage"] == "complete":
                if digest(model) != r["artifact"]["sha256"]:
                    raise ValueError("SF3D GLBのハッシュが一致しません。")
                return _public(r)
        if allow is not True:
            raise PermissionError(
                "初期設定でStability AIの公式Hugging Face Spaceへの画像送信と本人の匿名ZeroGPU利用枠の使用を許可してください。"
            )
        _abort(cancelled)
        if not r:
            if model.exists():
                raise ValueError(
                    "Use a dedicated SF3D job directory; an unowned model exists."
                )
            r = {
                **identity,
                "provider": APP,
                "recipient": ORIGIN,
                "session_hash": uuid.uuid4().hex,
                "auth": "anonymous",
                "settings": SETTINGS,
                "created_at": time.time(),
                "stage": "prepared",
            }
            _save(work, r)
        if r["stage"] in ("failed", "upload_unknown"):
            raise RemoteFailure(
                "このSF3D受付は再送しません。失敗または未確認の受付履歴を確認してください。"
            )
        if r["stage"] == "prepared":
            config = api.configuration()
            _before_post(
                work, r, "upload_unknown", progress, cancelled, configuration=config
            )
            remote = api.upload(png)
            _stage(work, r, "uploaded", progress, upload_path=remote)
        fd = {
            "path": checked_path(r["upload_path"], ".png"),
            "orig_name": "foreground.png",
            "mime_type": "image/png",
            "meta": {"_type": "gradio.FileData"},
        }
        cfg = r["configuration"]
        if r["stage"] in ("uploaded", "preprocess_waiting"):
            answer = _queue(
                work,
                r,
                api,
                "preprocess",
                cfg["requires_bg_remove"],
                [fd, 0.85],
                progress,
                cancelled,
            )
            if (
                not isinstance(answer, list)
                or len(answer) != 6
                or not isinstance(answer[0], dict)
                or answer[0].get("value") != "Run"
            ):
                _stage(work, r, "failed", progress, failure="invalid_preprocess")
                raise RemoteFailure("透過前景を提供元が受理しませんでした。")
            _stage(work, r, "preprocessed", progress)
        if r["stage"] in ("preprocessed", "generation_waiting"):
            answer = _queue(
                work,
                r,
                api,
                "generation",
                cfg["run_button"],
                ["Run", fd, None, 0.85, "None", -1, 1024],
                progress,
                cancelled,
            )
            try:
                remote = output_path(answer)
            except (KeyError, TypeError, ValueError) as exc:
                _stage(work, r, "failed", progress, failure="invalid_result")
                raise RemoteFailure(
                    "SF3Dが予期しない形式の生成結果を返しました。"
                ) from exc
            _stage(work, r, "downloading", progress, result_path=remote)
        if r["stage"] != "downloading":
            raise ValueError("Unknown SF3D receipt state.")
        _abort(cancelled)
        tmp = work / "model.download"
        try:
            api.download(r["result_path"], tmp)
            artifact = native_glb(tmp)
            os.replace(tmp, model)
        finally:
            tmp.unlink(missing_ok=True)
        _stage(
            work, r, "complete", progress, artifact=artifact, finished_at=time.time()
        )
        return _public(r)


def _public(r):
    return {
        key: r[key]
        for key in (
            "provider",
            "recipient",
            "auth",
            "source_sha256",
            "foreground_sha256",
            "upload_sha256",
            "configuration",
            "preprocess_event_id",
            "generation_event_id",
            "settings",
            "artifact",
            "created_at",
            "finished_at",
        )
    }
