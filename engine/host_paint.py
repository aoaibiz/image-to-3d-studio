"""Local, durable handoff to a user's native image generation tool. No network client."""

import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import tempfile

from engine.pipeline import HOST_PIPELINE
from engine.triposg_cpu import atomic_json, digest, read_json

PROMPT = """The first attached image is a two-panel geometry and camera constraint. The second is the only real observation of the object. Produce a single two-panel image of the SAME object in the exact two poses, pixel positions, scale, crop and silhouettes of attachment ONE. Keep all its depth ordering, openings, self-occlusion, attachments and empty white space fixed. Internally identify the observed object's material, base colors, surface finish, motifs, decoration density, construction, and which details are directional or belong only on the photographed face. Apply these visual findings to the two grey meshes. Infer plausible continuations on unobserved surfaces: carry the same style and scale around the three-dimensional form, preserve material boundaries, use only motifs actually supported by the photograph, avoid mirroring the front or duplicating any front-specific elements. Do not copy the photographic perspective. Render flat, neutral, evenly lit color without baked shadow, specular highlights or surrounding environment, for reprojecting color onto the mesh. Preserve a pure white background inside openings and outside the exact two silhouettes. No text, labels, frame, arrow, camera change or extra subjects. Return a wide, even 2:1 panel sheet."""
PROMPT_SHA = hashlib.sha256(PROMPT.encode()).hexdigest()
PROVIDER = "Codex host / OpenAI ImageGen"
AGREEMENT = {"schema": 1, "provider": PROVIDER, "recipient": "OpenAI ChatGPT",
             "send": ["normalized source photo", "two-view render derived from current GLB"],
             "max_submissions_per_job": 1, "personal_subscription": True}
TOKEN_FILE = "host-access.token"
TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")
SHA = re.compile(r"[a-f0-9]{64}")
CALL = re.compile(r"[A-Za-z0-9_-]{1,160}")


def authorize(directory, token):
    try:
        config = read_json(directory / "engine.json")
        expected = (directory / TOKEN_FILE).read_text("ascii").strip()
    except (OSError, ValueError):
        return False
    return (json.dumps(config, sort_keys=True) == json.dumps({"pipeline": HOST_PIPELINE, "agreement": AGREEMENT}, sort_keys=True)
            and bool(TOKEN.fullmatch(expected)) and isinstance(token, str)
            and bool(TOKEN.fullmatch(token)) and hmac.compare_digest(expected, token))


def setup_token(directory):
    path = directory / TOKEN_FILE
    if not path.exists():
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "w", encoding="ascii") as stream:
                stream.write(secrets.token_urlsafe(32))
    if not TOKEN.fullmatch(path.read_text("ascii").strip()):
        raise ValueError("ホスト用トークンを読み込めません。専用ジョブフォルダーを確認してください。")


def prepare(work, source, guide, layout):
    from PIL import Image, ImageOps

    if digest(source) != layout["photo_sha"] or digest(guide) != layout["guide_sha"]:
        raise ValueError("元写真またはガイドのハッシュが生成履歴と一致しません。")
    folder = work / "host-input"
    if not folder.exists():
        with tempfile.TemporaryDirectory(dir=work) as tmp:
            stage = Path(tmp) / "input"
            stage.mkdir()
            with Image.open(source) as photo:
                if (photo.format not in {"PNG", "WEBP", "JPEG"} or photo.n_frames != 1
                        or photo.width * photo.height > 40_000_000):
                    raise ValueError("40メガピクセル以下の静止画像を指定してください。")
                normalized = ImageOps.exif_transpose(photo).convert("RGBA")
                normalized.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                normalized.info.clear()
                normalized.save(stage / "photo.png", format="PNG")
            photo_sha = digest(stage / "photo.png")
            identity = [layout["photo_sha"], photo_sha, layout["model_sha"], layout["guide_sha"], PROMPT_SHA]
            atomic_json(stage / "request.json", {
                "input_sha256": hashlib.sha256(json.dumps(identity).encode()).hexdigest(),
                "photo_sha256": layout["photo_sha"], "geometry_sha256": layout["model_sha"],
                "guide_sha256": layout["guide_sha"], "prompt_sha256": PROMPT_SHA,
                "uploaded_png_sha256": [layout["guide_sha"], photo_sha],
            })
            os.replace(stage, folder)
    request = read_json(folder / "request.json")
    if (request["photo_sha256"] != layout["photo_sha"] or request["prompt_sha256"] != PROMPT_SHA
            or request["geometry_sha256"] != layout["model_sha"]
            or request["guide_sha256"] != layout["guide_sha"]
            or request["uploaded_png_sha256"] != [digest(guide), digest(folder / "photo.png")]):
        raise ValueError("保存したホスト用画像のハッシュが一致しません。")
    return request


def claim(work, job_id):
    request = read_json(work / "host-input/request.json")
    files = [work / "guides/guides.png", work / "host-input/photo.png"]
    if request["prompt_sha256"] != PROMPT_SHA or [digest(p) for p in files] != request["uploaded_png_sha256"]:
        raise ValueError("写真・ガイド・画像生成指示が受付時の記録と一致しません。")
    path = work / "host-claim.json"
    first = not path.exists()
    if first:
        atomic_json(path, {"claim_id": secrets.token_urlsafe(32), "request_sha256": request["input_sha256"]})
    saved = read_json(path)
    if saved["request_sha256"] != request["input_sha256"]:
        raise ValueError("保存した受付と画像が一致しません。")
    reply = {"job_id": job_id, **saved, "generate_once": first}
    if first:
        reply.update(prompt=PROMPT, prompt_sha256=PROMPT_SHA,
                     referenced_image_paths=[str(path.resolve()) for path in files],
                     uploaded_png_sha256=request["uploaded_png_sha256"], recipient=AGREEMENT["recipient"])
    return reply


def submit(work, claim_id, request_sha, sheet_sha, tool_call, png, *, validate=None):
    # Serialized by the single store owner. Publishing the directory commits both files.
    if not (isinstance(claim_id, str) and TOKEN.fullmatch(claim_id)
            and SHA.fullmatch(request_sha or "") and SHA.fullmatch(sheet_sha or "")
            and CALL.fullmatch(tool_call or "")):
        raise ValueError("受付番号、生成IDとSHA-256を確認してください。")
    saved = read_json(work / "host-claim.json")
    request = read_json(work / "host-input/request.json")
    if (not hmac.compare_digest(claim_id, saved["claim_id"])
            or request_sha != saved["request_sha256"] or request_sha != request["input_sha256"]
            or hashlib.sha256(png).hexdigest() != sheet_sha):
        raise ValueError("受付番号または画像のハッシュが一致しません。")
    out = work / "host-result"
    if out.exists():
        record = read_json(out / "receipt.json")
        if (record["sheet_sha256"], record["host_tool_call_id"]) != (sheet_sha, tool_call) or digest(out / "sheet.png") != sheet_sha:
            raise ValueError("このジョブには別の結果が登録されています。再送を停止しました。")
        return
    if validate is None:
        from engine.guided_provider import _sheet as validate
    validate(png)
    record = {**request, "provider": PROVIDER, "recipient": AGREEMENT["recipient"], "state": "completed",
              "host_tool_call_id": tool_call, "delivery_mode": "host_attested",
              "sheet_sha256": sheet_sha, "original_photograph_count": 1, "guide_count": 2}
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        stage = Path(tmp) / "result"
        stage.mkdir()
        (stage / "sheet.png").write_bytes(png)
        atomic_json(stage / "receipt.json", record)
        os.replace(stage, out)


def result(work, request):
    folder = work / "host-result"
    if not folder.exists():
        return None
    record = read_json(folder / "receipt.json")
    if (any(record.get(key) != value for key, value in request.items())
            or digest(folder / "sheet.png") != record.get("sheet_sha256")
            or not CALL.fullmatch(record.get("host_tool_call_id", ""))):
        raise ValueError("ホストから受け取った画像の記録が一致しません。")
    return {"sheet": folder / "sheet.png", "provenance": record}
