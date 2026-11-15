"""Anonymous opt-in trial of one pinned public Pixal3D Space, restricted to public CC0 inputs."""
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps

from engine.triposg_cpu import atomic_json, digest, read_json, writer_lock
from quality.inspection import load_glb

SPACE = "victor/pixal3d-studio"
REVISION = "ef1cb838893269417b3eaa81dd8fc635f65a5180"
ORIGIN = "https://victor-pixal3d-studio.hf.space"
HUB = "https://huggingface.co/api/spaces/" + SPACE
BUCKET = "https://huggingface.co/buckets/victor/pixal3d-community/resolve/assets/"
SEED = 1234
MAX_GLB = 96 * 1024**2
FIELDS = ["image", "prompt", "seed", "resolution", "quality", "manual_fov", "share_to_community"]
CALL = ORIGIN + "/gradio_api/call/generate_fast"


class Pending(RuntimeError):
    """Resume the same receipt. Never repeat a POST after uncertain acceptance."""


class RemoteFailure(RuntimeError):
    """Terminal failure of the one permitted remote generation."""


class QuotaExceeded(RemoteFailure):
    """The anonymous provider declined this event before it could run."""

    def __init__(self, requested, available, retry_after=None):
        super().__init__("Anonymous ZeroGPU allowance exhausted.")
        self.details = {"requested_seconds": requested, "available_seconds": available,
                        "retry_after_seconds": retry_after}


def _remote_failure(value):
    # Accept only the known provider error signature. Opaque remote text stays private.
    if isinstance(value, str):
        value = value.removeprefix("Pixal3D event ended in error:").strip()
        if value.startswith("{"):
            try:
                value = json.loads(value)
            except ValueError:
                value = None
    if isinstance(value, dict):
        value = value.get("error")
    if isinstance(value, str):
        match = re.match(r"^You have exceeded your ZeroGPU quota \((\d{1,6})s requested vs\. (\d{1,6})s left\)\.", value)
        if match:
            requested, available = map(int, match.groups())
            delay = re.match(r" Try again in (\d{1,3}):([0-5]\d):([0-5]\d)\.", value[match.end():])
            seconds = None
            if delay:
                h, m, s = map(int, delay.groups())
                seconds = h * 3600 + m * 60 + s or None
            if 0 <= available < requested <= 3600:
                return QuotaExceeded(requested, available, seconds)
    return RemoteFailure("Pixal3D ended this event without a model.")


def _saved_failure(state):
    if state.get("failure_kind") == "anonymous_quota":
        details = state.get("quota", {})
        needed, free, delay = (details.get(k) for k in ("requested_seconds", "available_seconds", "retry_after_seconds"))
        if (type(needed) is int and type(free) is int and 0 <= free < needed <= 3600
                and (delay is None or type(delay) is int and 0 < delay < 3600000)):
            return QuotaExceeded(needed, free, delay)
    return _remote_failure(state.get("error"))


def _session():
    s = requests.Session()
    s.trust_env = False  # Do not attach local account credentials or proxy authorization.
    return s


def _json(session, method, url, **kwargs):
    with session.request(method, url, timeout=(10, 35), allow_redirects=False, stream=True, **kwargs) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Pixal3D returned HTTP {r.status_code}.")
        data = bytearray()
        for chunk in r.iter_content(65536):
            data.extend(chunk)
            if len(data) > 2 * 1024**2:
                raise ValueError("Pixal3D metadata exceeds its size limit.")
        return json.loads(data)


def preflight():
    """Read the published revision, running configuration and typed operation. No GPU work."""
    with _session() as session:
        hub = _json(session, "GET", HUB)
        health = _json(session, "GET", ORIGIN + "/health")
        api = _json(session, "GET", ORIGIN + "/gradio_api/info")
    rt = hub["runtime"]
    if (hub.get("id") != SPACE or hub.get("sha") != REVISION or rt.get("sha") != REVISION
            or rt.get("stage") != "RUNNING" or hub.get("private") is not False or hub.get("gated") is not False
            or health.get("backend") != SPACE or health.get("mock") is not False
            or health.get("fast_backend") != "local" or health.get("fast_final") is not True
            or health.get("ok") is not True):
        raise ValueError("The running Pixal3D deployment differs from the reviewed revision.")
    endpoint = api["named_endpoints"]["/generate_fast"]
    if [p["parameter_name"] for p in endpoint["parameters"]] != FIELDS or len(endpoint["returns"]) != 1:
        raise ValueError("Unexpected Pixal3D operation schema.")
    return {"provider_space": SPACE, "revision": REVISION, "origin": ORIGIN,
            "reservation_seconds": 120, "anonymous_remaining_seconds": None,
            "fast_backend": "local", "mock": False, "share_to_community": False}


def _source_png(raw):
    with Image.open(io.BytesIO(raw)) as image:
        if (image.format not in {"JPEG", "PNG", "WEBP"} or image.width * image.height > 24_000_000
                or getattr(image, "n_frames", 1) != 1):
            raise ValueError("Provide one public CC0 JPEG, PNG or WebP image under 24 megapixels.")
        oriented = ImageOps.exif_transpose(image)
        mode = "RGBA" if "A" in oriented.getbands() or "transparency" in oriented.info else "RGB"
        clean = oriented.convert(mode)
        clean.info.clear()
        clean.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        clean.save(buffer, format="PNG")
    return buffer.getvalue()


def _asset(result):
    if not isinstance(result, list) or len(result) != 1:
        raise ValueError("Unexpected Pixal3D event payload.")
    record = json.loads(result[0]) if isinstance(result[0], str) else result[0]
    if record.get("ok") is False:
        raise _remote_failure(record)
    a = record["asset"]
    ident = a["id"]
    if not isinstance(ident, str) or not re.fullmatch(r"[a-f0-9]{12}", ident):
        raise ValueError("Unexpected remote asset identifier.")
    name = f"{ident}/model.glb"
    local = "/community-assets/" + name
    if (record.get("ok") is not True or record.get("mode") != "fast_final" or a.get("shared") is not False
            or a.get("source_space") != SPACE or a.get("model_filename") != "model.glb"
            or a.get("model_url") not in {local, ORIGIN + local, BUCKET + name}
            or a.get("input_url") not in {local.replace("model.glb", "input.png"),
                ORIGIN + local.replace("model.glb", "input.png"), BUCKET + ident + "/input.png"}
            or a.get("status") != "complete" or a.get("seed") != SEED
            or a.get("resolution") != 1024 or a.get("quality") != "balanced"):
        raise ValueError("The submitted settings and public model reference do not match.")
    # The same persistent storage is mounted on the Space. Avoid arbitrary URLs or redirects.
    return ident, ORIGIN + local


def _wait(session, event):
    deadline = time.monotonic() + 600
    with session.get(CALL + "/" + event, headers={"Accept": "text/event-stream"}, stream=True,
                     timeout=(10, 75), allow_redirects=False) as r:
        if r.status_code != 200 or "text/event-stream" not in r.headers.get("content-type", ""):
            raise Pending("The saved Pixal3D event cannot currently be opened.")
        kind = ""
        size = 0
        for line in r.iter_lines():
            size += len(line)
            if size > 2 * 1024**2 or time.monotonic() > deadline:
                raise Pending("Reconnect to the saved Pixal3D event.")
            if line.startswith(b"event:"):
                kind = line[6:].strip().decode("ascii")
            elif line.startswith(b"data:") and kind == "error":
                raise _remote_failure(line[5:4101].decode("utf-8", "replace"))
            elif line.startswith(b"data:") and kind == "complete":
                return _asset(json.loads(line[5:]))
    raise Pending("The event stream ended before completion. Keep the existing job.")


def inspect_pbr(path):
    """Check the delivered mesh and both embedded PBR channels on the same bytes."""
    path = Path(path)
    if path.is_symlink() or not 0 < path.stat().st_size <= MAX_GLB:
        raise ValueError("Invalid local Pixal3D artifact.")
    raw, doc, binary = load_glb(path)
    if (len(doc["buffers"]) != 1 or "uri" in doc["buffers"][0] or not binary
            or len(binary) - doc["buffers"][0]["byteLength"] not in range(4)):
        raise ValueError("Expected one embedded model buffer.")
    sizes = []
    for image in doc["images"]:
        if "uri" in image:
            raise ValueError("External image in Pixal3D result.")
        view = doc["bufferViews"][image["bufferView"]]
        start, length = view.get("byteOffset", 0), view["byteLength"]
        if view.get("buffer", 0) or not 0 <= start < start + length <= len(binary):
            raise ValueError("Invalid embedded image.")
        with Image.open(io.BytesIO(binary[start:start + length])) as texture:
            if (texture.format not in {"PNG", "JPEG", "WEBP"} or min(texture.size) <= 1
                    or max(texture.size) > 4096):
                raise ValueError("Unexpected embedded material texture.")
            texture.verify()
            sizes.append(list(texture.size))
    faces = 0
    for mesh in doc["meshes"]:
        for primitive in mesh["primitives"]:
            pbr = doc["materials"][primitive["material"]]["pbrMetallicRoughness"]
            idx = doc["accessors"][primitive["indices"]]
            attrs = primitive["attributes"]
            if (primitive.get("mode", 4) != 4 or idx.get("type") != "SCALAR"
                    or idx["count"] <= 0 or idx["count"] % 3 or not {"POSITION", "TEXCOORD_0"} <= attrs.keys()):
                raise ValueError("Expected a triangle mesh with texture coordinates.")
            for slot in ["baseColorTexture", "metallicRoughnessTexture"]:
                tex = doc["textures"][pbr[slot]["index"]]
                source = tex.get("extensions", {}).get("EXT_texture_webp", {}).get("source", tex.get("source"))
                if type(source) is not int or not 0 <= source < len(sizes):
                    raise ValueError("Missing embedded PBR channel: " + slot)
            faces += idx["count"] // 3
    if not faces or not sizes:
        raise ValueError("The GLB lacks textured triangle geometry.")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "triangles": faces,
            "texture_sizes": sizes, "channels": ["baseColor", "metallicRoughness"]}


def _download(session, url, dest, limit=MAX_GLB):
    with session.get(url, allow_redirects=False, stream=True, timeout=(10, 45)) as r, dest.open("wb") as f:
        if r.status_code != 200:
            raise Pending(f"Saved Pixal3D asset returned HTTP {r.status_code}.")
        count = 0
        for chunk in r.iter_content(262144):
            count += len(chunk)
            if count > limit:
                raise ValueError("Pixal3D asset exceeds the size limit.")
            f.write(chunk)


def _intent(work, name, photo):
    # Prevent replay even if the latest journal was rolled back after a power cut.
    try:
        with (work / (name + ".intent")).open("x", encoding="ascii") as f:
            f.write(photo)
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError as exc:
        raise Pending("Prior " + name + " intent exists. Do not submit again.") from exc


def generate(source, work, *, allow_public_cc0=False, expected_source_sha256=None, progress=None):
    """One irreversible submission per work directory. Recorded inputs are publicly URL-addressable."""
    source, work = Path(source), Path(work)
    if not source.is_file() or source.stat().st_size > 32 * 1024**2:
        raise ValueError("Invalid source photo.")
    raw = source.read_bytes()
    photo = hashlib.sha256(raw).hexdigest()
    if expected_source_sha256 is not None and photo != expected_source_sha256:
        raise ValueError("The approved source digest differs from the supplied photo.")
    work.mkdir(parents=True, exist_ok=True)
    with writer_lock(work), _session() as session:
        statepath, png, final = work / "provider.json", work / "input.png", work / "model.glb"
        def save(stage, **updates):
            state.update(updates, state=stage)
            atomic_json(statepath, state)
            if progress:
                progress(stage)
        if statepath.exists():
            state = read_json(statepath)
            if state["source_sha256"] != photo or state.get("revision") != REVISION or digest(png) != state["upload_sha256"]:
                raise ValueError("The existing job has different or damaged inputs.")
            if state["state"] == "completed":
                if not final.is_file() or final.stat().st_size > MAX_GLB or digest(final) != state["sha256"]:
                    raise ValueError("The saved Pixal3D GLB is missing or has changed.")
                return state
        else:
            if any(f.name != "run.lock" for f in work.iterdir()):
                raise Pending("The original Pixal3D receipt is missing. Do not resubmit.")
            if allow_public_cc0 is not True:
                raise PermissionError("Only a verified public CC0 image may be sent to this community Space.")
            attestation = preflight()
            png.write_bytes(_source_png(raw))
            state = dict(attestation, source_sha256=photo, upload_sha256=digest(png), started_at=time.time(),
                         seed=SEED, resolution=1024, quality_setting="balanced", quality="needs_review")
            save("prepared")
        if state["state"] == "failed":
            raise _saved_failure(state)
        if state["state"] in {"upload_unknown", "submission_unknown"}:
            raise Pending("An earlier POST has no confirmed result. Do not submit again.")
        if state["state"] in {"prepared", "uploaded"}:
            if allow_public_cc0 is not True:
                raise PermissionError("Sending the public CC0 image requires explicit approval.")
            preflight()
        try:
            if state["state"] == "prepared":
                _intent(work, "upload", photo)
                save("upload_unknown")
                with png.open("rb") as f:
                    uploaded = _json(session, "POST", ORIGIN + "/gradio_api/upload",
                                     files={"files": ("input.png", f, "image/png")})
                if not (isinstance(uploaded, list) and len(uploaded) == 1 and isinstance(uploaded[0], str)
                        and re.fullmatch(r"/tmp/gradio/[a-f0-9]{32,128}/input\.png", uploaded[0])):
                    raise ValueError("Unrecognized upload response; do not send the image again.")
                save("uploaded", uploaded_path=uploaded[0])
            if state["state"] == "uploaded":
                remote = state["uploaded_path"]
                if not re.fullmatch(r"/tmp/gradio/[a-f0-9]{32,128}/input\.png", remote):
                    raise ValueError("Invalid saved remote path.")
                spec = {"path": remote, "orig_name": "input.png", "mime_type": "image/png",
                        "meta": {"_type": "gradio.FileData"}}
                _intent(work, "submission", photo)
                save("submission_unknown")
                queued = _json(session, "POST", CALL, json={"data": [spec, "", SEED, 1024, "balanced", -1.0, False]})
                event = queued.get("event_id")
                if not isinstance(event, str) or not re.fullmatch(r"[a-f0-9]{32}", event):
                    raise Pending("No durable Pixal3D event identifier. Do not submit again.")
                save("accepted", event_id=event)
            if state["state"] == "accepted":
                event = state["event_id"]
                if not re.fullmatch(r"[a-f0-9]{32}", event):
                    raise ValueError("The saved Pixal3D event is invalid.")
                ident, url = _wait(session, event)
                save("downloading", asset_id=ident, download_url=url)
            if state["state"] != "downloading":
                raise ValueError("Unknown Pixal3D checkpoint.")
            ident = state["asset_id"]
            if (not re.fullmatch(r"[a-f0-9]{12}", ident) or
                    state["download_url"] != ORIGIN + "/community-assets/" + ident + "/model.glb"):
                raise ValueError("Invalid saved asset location.")
            echo = work / "input-returned.png.part"
            _download(session, ORIGIN + "/community-assets/" + ident + "/input.png", echo, png.stat().st_size)
            if digest(echo) != state["upload_sha256"]:
                raise ValueError("The returned model belongs to a different uploaded image.")
            echo.unlink()
            part = work / "model.glb.part"
            _download(session, state["download_url"], part)
            mat = inspect_pbr(part)
            os.replace(part, final)
            save("completed", sha256=mat["sha256"], material=mat,
                 artifacts={"input.png": state["upload_sha256"], "model.glb": mat["sha256"]}, finished_at=time.time())
            return state
        except RemoteFailure as exc:
            if isinstance(exc, QuotaExceeded):
                save("failed", failure_kind="anonymous_quota", quota=exc.details)
            else:
                save("failed", failure_kind="remote_failure", error="Provider event failed.")
            raise
        except (requests.RequestException, RuntimeError, ValueError, KeyError, TypeError) as exc:
            if state["state"] in {"upload_unknown","submission_unknown","accepted"}:
                raise Pending(str(exc)) from exc
            raise
