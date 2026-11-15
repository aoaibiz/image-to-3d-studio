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
# Pre-versioned jobs did not always save prompt.txt. Retain their exact instruction.
LEGACY_PROMPT = PROMPT
LEGACY_PROMPT_SHA = "ffd49af58cd0efbdd663d73274556e6f6655ef984028859ff7914961e04b3211"
PROVIDER = "Codex host / OpenAI ImageGen"
AGREEMENT = {"schema": 1, "provider": PROVIDER, "recipient": "OpenAI ChatGPT",
             "send": ["normalized source photo", "two-view render derived from current GLB"],
             "max_submissions_per_job": 1, "personal_subscription": True}
TOKEN_FILE = "host-access.token"
TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")
SHA = re.compile(r"[a-f0-9]{64}")
CALL = re.compile(r"[A-Za-z0-9_-]{1,160}")


def _identity(request):
    guide, photo = request["uploaded_png_sha256"]
    if guide != request["guide_sha256"]:
        raise ValueError("送信用ガイドの記録が一致しません。")
    hashes = [request["photo_sha256"], photo, request["geometry_sha256"], guide, request["prompt_sha256"]]
    if "visual_brief_sha256" in request:
        hashes.append(request["visual_brief_sha256"])
    if "version" in request:
        if type(request["version"]) is not int or request["version"] != 2:
            raise ValueError("ホスト画像依頼の保存形式を確認してください。")
        hashes.append(request["manifest_sha256"])
    if not all(isinstance(value, str) and SHA.fullmatch(value) for value in hashes):
        raise ValueError("画像生成の入力ハッシュが無効です。")
    identity = ["host-paint-v2", *hashes] if "version" in request else hashes
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def _request(work):
    record = read_json(work / "host-input/request.json")
    if not isinstance(record, dict):
        raise ValueError("ホスト画像依頼の保存形式が無効です。")
    keys = {"photo_sha256", "geometry_sha256", "guide_sha256", "prompt_sha256", "uploaded_png_sha256", "input_sha256"}
    if "version" in record:
        keys.update({"version", "manifest_sha256"})
    if "visual_brief_sha256" in record:
        keys.add("visual_brief_sha256")
    if (set(record) != keys or type(record["uploaded_png_sha256"]) is not list
            or len(record["uploaded_png_sha256"]) != 2 or record.get("input_sha256") != _identity(record)):
        raise ValueError("ホストに引き渡す依頼の識別子が一致しません。")
    return record


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _legacy_prompt(request, layout, prompt):
    if "visual_brief_sha256" not in request:
        valid = "visual_brief" not in layout and request["prompt_sha256"] == LEGACY_PROMPT_SHA
    else:
        # Unversioned briefs (41bc035) need their original vocabulary and recipe.
        # This verifier is only reachable for an existing, sealed host-input directory.
        from engine.guided_provider import appearance_prompt, legacy_appearance_prompt
        if not isinstance(layout.get("visual_brief"), dict):
            raise ValueError("受付時の外観分析が見つかりません。")
        expected, brief_sha = legacy_appearance_prompt(LEGACY_PROMPT, layout, request["photo_sha256"])
        if expected != prompt and "visible_structures" in layout["visual_brief"]:
            # A short-lived unversioned writer used the structural recipe.
            expected, brief_sha = appearance_prompt(LEGACY_PROMPT, layout, request["photo_sha256"])
        valid = brief_sha == request["visual_brief_sha256"] and expected == prompt
    if not valid:
        raise ValueError("旧方式で保存した画像生成依頼の指示と外観分析が一致しません。")


def _verify(work, request, layout=None):
    folder = work / "host-input"
    if (folder / "prompt.txt").is_file():
        prompt_path = folder / "prompt.txt"
        if prompt_path.stat().st_size > 32768:
            raise ValueError("保存した画像生成指示の長さが無効です。")
        prompt = prompt_path.read_bytes().decode("utf-8")
    elif "version" not in request and "visual_brief_sha256" not in request:
        prompt = LEGACY_PROMPT
    else:
        raise ValueError("受付時の画像生成指示が見つかりません。")
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != request["prompt_sha256"]:
        raise ValueError("保存した画像生成指示が破損しています。")

    guide_manifest = work / "guides/request-guide.json"
    source_file, model_file = work.parent.parent / "input.image", work / "base/model.glb"
    source_bound, model_bound = source_file.is_file(), model_file.is_file()
    checks_path = guide_manifest.parent / "checksums.json"
    has_model_record = (work / "base/settings.json").is_file()
    if checks_path.is_file() or has_model_record:
        checks = read_json(checks_path)
        files = {"request-guide.json", "guides.png", "guide.json"}
        if set(checks) != files or any(digest(checks_path.parent / name) != checks[name] for name in files):
            raise ValueError("受付時のガイドの画像・カメラまたは記録が破損しています。")
    if guide_manifest.is_file():
        current = read_json(guide_manifest)
        if layout is not None and _canonical(current) != _canonical(layout):
            raise ValueError("現在のカメラ・ガイドの記録が一致しません。")
        layout = current
    if "version" in request:
        binding = read_json(folder / "manifest.json")
        if (hashlib.sha256(_canonical(binding)).hexdigest() != request["manifest_sha256"]
                or set(binding) != {"layout", "job_source", "base_model", "guide_manifest"}
                or any(type(binding[k]) is not bool for k in ("job_source", "base_model", "guide_manifest"))
                or binding["guide_manifest"] and not guide_manifest.is_file()
                or layout is not None and _canonical(layout) != _canonical(binding["layout"])):
            raise ValueError("受付時と現在のカメラ・元写真・外観分析が一致しません。")
        source_bound, model_bound = binding["job_source"], binding["base_model"]
        layout = binding["layout"]
    else:
        if layout is None and "visual_brief_sha256" in request:
            raise ValueError("元写真に照合した旧ガイドが見つかりません。")
        _legacy_prompt(request, layout or {}, prompt)
    if layout is not None and (
            [layout["photo_sha"], layout["model_sha"], layout["guide_sha"]]
            != [request["photo_sha256"], request["geometry_sha256"], request["guide_sha256"]]):
        raise ValueError("写真・形状・ガイドの組み合わせが受付時と一致しません。")
    files = [work / "guides/guides.png", folder / "photo.png"]
    if [digest(p) for p in files] != request["uploaded_png_sha256"]:
        raise ValueError("送信用の写真またはガイドが変更されています。")
    if ((source_bound and digest(source_file) != request["photo_sha256"])
            or ((model_bound or has_model_record) and digest(model_file) != request["geometry_sha256"])):
        raise ValueError("受付時の元写真・形状と実際の保存ファイルが一致しません。")
    return prompt, files


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
        if (work / "host-claim.json").exists() or (work / "host-result").exists():
            raise ValueError("送信済みの依頼の保存データが欠けています。同じジョブを再送しません。")
        prompt, brief_sha = PROMPT, None
        if "visual_brief" in layout:
            from engine.guided_provider import appearance_prompt
            prompt, brief_sha = appearance_prompt(PROMPT, layout, layout["photo_sha"])
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory(dir=work) as tmp:
            stage = Path(tmp) / "input"
            stage.mkdir()
            with Image.open(source) as photo:
                if (photo.format not in {"PNG", "WEBP", "JPEG"} or getattr(photo, "n_frames", 1) != 1
                        or photo.width * photo.height > 40_000_000):
                    raise ValueError("40メガピクセル以下の静止画像を指定してください。")
                normalized = ImageOps.exif_transpose(photo).convert("RGBA")
                normalized.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                normalized.info.clear()
                normalized.save(stage / "photo.png", format="PNG")
            photo_sha = digest(stage / "photo.png")
            (stage / "prompt.txt").write_bytes(prompt.encode("utf-8"))
            binding = {"layout": layout, "job_source": source.resolve() == (work.parent.parent / "input.image").resolve(),
                       "base_model": (work / "base/model.glb").is_file(),
                       "guide_manifest": (work / "guides/request-guide.json").is_file()}
            atomic_json(stage / "manifest.json", binding)
            request = {
                "version": 2, "manifest_sha256": hashlib.sha256(_canonical(binding)).hexdigest(),
                "photo_sha256": layout["photo_sha"], "geometry_sha256": layout["model_sha"],
                "guide_sha256": layout["guide_sha"], "prompt_sha256": prompt_sha,
                **({"visual_brief_sha256": brief_sha} if brief_sha is not None else {}),
                "uploaded_png_sha256": [layout["guide_sha"], photo_sha],
            }
            request["input_sha256"] = _identity(request)
            atomic_json(stage / "request.json", request)
            os.replace(stage, folder)
    request = _request(work)
    _verify(work, request, layout)
    return request


def claim(work, job_id):
    request = _request(work)
    prompt, files = _verify(work, request)
    path = work / "host-claim.json"
    first = not path.exists()
    if first:
        if (work / "host-result").exists() or path.with_suffix(".json.part").exists():
            raise ValueError("画像生成の受付状態が不明です。同じジョブを再送しません。")
        atomic_json(path, {"claim_id": secrets.token_urlsafe(32), "request_sha256": request["input_sha256"]})
    saved = read_json(path)
    if saved["request_sha256"] != request["input_sha256"]:
        raise ValueError("保存した受付と画像が一致しません。")
    reply = {"job_id": job_id, **saved, "generate_once": first}
    if first:
        reply.update(prompt=prompt, prompt_sha256=request["prompt_sha256"],
                     referenced_image_paths=[str(path.resolve()) for path in files],
                     uploaded_png_sha256=request["uploaded_png_sha256"], recipient=AGREEMENT["recipient"])
        if "visual_brief_sha256" in request:
            reply["visual_brief_sha256"] = request["visual_brief_sha256"]
    return reply


def submit(work, claim_id, request_sha, sheet_sha, tool_call, png, *, validate=None):
    # Serialized by the single store owner. Publishing the directory commits both files.
    if not (isinstance(claim_id, str) and TOKEN.fullmatch(claim_id)
            and SHA.fullmatch(request_sha or "") and SHA.fullmatch(sheet_sha or "")
            and CALL.fullmatch(tool_call or "")):
        raise ValueError("受付番号、生成IDとSHA-256を確認してください。")
    saved = read_json(work / "host-claim.json")
    request = _request(work)
    _verify(work, request)
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
