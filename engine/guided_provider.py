"""Single-submission geometry-guided appearance using the local operator's Codex CLI."""

import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from PIL import Image, ImageOps

# The reference board preserves both normalized inputs at their own pixel scale.
PROMPT = """Use the built-in image generation tool exactly once to render one image now. The attached reference board has two sections. The UPPER strip shows exactly two neutral 3D geometry views side by side. The LOWER section contains the one source photograph. Return ONLY the UPPER strip as a wide 2:1 image with its two views in the original order, now painted. Copy their camera, silhouette, silhouette position/size, holes, occlusion, pose and component boundaries exactly. Do not include the lower photograph in the result. Derive all colors, material boundaries, palette, finish, and recurring decorative style from that photograph. Continue evidenced material or decorative structure naturally and consistently around the unseen sides. Hidden paint is inferred; avoid mirroring unique marks or inventing text, branding or extra parts. Use neutral diffuse photographic light with no cast shadows, a plain white background and no captions. Return the generated image, not a description or instructions. If the tool cannot generate, report its actual failure in one sentence. No other tools, no code, no retries."""
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
PROVIDER = "OpenAI Codex CLI / ChatGPT image_generation"
AGREEMENT = {
    "schema": 1,
    "provider": PROVIDER,
    "recipient": "OpenAI ChatGPT",
    "send": ["normalized source photo", "two-view render derived from current GLB"],
    "personal_subscription": True,
    "api_key_billing": False,
    "subscription_credits": "ChatGPT may use available credits after included limits; account billing controls apply",
    "max_submissions_per_job": 1,
}
_LIMIT = 64 * 1024 * 1024
_TURN_TIMEOUT = 20 * 60
_AUTH_MESSAGE = (
    "Codex CLI で ChatGPT にログインしてください。API キー方式は利用できません。"
)
_AMBIGUOUS = "送信済みの依頼の完了を確認できません。同じジョブは再送しません。受付記録と Codex の生成結果を確認してください。"
_NO_SHEET = "Codex は終了しましたが、この受付番号に生成 PNG がありません。再接続は保存済み結果だけを確認します。"
_FAILURES = {
    "usage_limit": (
        r"usage limit|quota exceeded|rate limit|利用上限|利用枠.*(?:超|到達)",
        "Codex の利用枠またはレート上限に達しました。",
    ),
    "connection": (
        r"network error|connection (?:error|failed)|network request failed|ネットワークエラー|接続エラー",
        "画像生成サービスへの接続に失敗しました。",
    ),
    "unavailable": (
        r"(?:image.?gen(?:eration)?|image tool).{0,55}(?:unavailable|not available|not enabled|not accessible|not exposed|not provided|not installed)|(?:no access to|cannot access|not have access to|do not have|no available|cannot call).{0,45}(?:image.?gen|image tool)|画像生成.{0,50}(?:利用でき|使用でき|使え|呼び出せ|提供されてい|有効になってい)ません|画像生成.{0,40}利用不可",
        "現在の Codex 環境では画像生成ツールを利用できません。",
    ),
    "provider_declined": (
        r"image.{0,35}(?:policy|not permitted)|(?:safety|policy).{0,25}(?:refused|declined)|画像生成.{0,40}(?:ポリシー|許可されていません)",
        "画像生成が提供元で拒否されました。同じ依頼は再送しません。",
    ),
}
_FEATURES_OFF = (
    "apps",
    "plugins",
    "hooks",
    "shell_tool",
    "unified_exec",
    "multi_agent",
    "multi_agent_v2",
    "browser_use",
    "computer_use",
    "view_image",
    "code_mode",
    "in_app_browser",
    "memories",
    "goals",
    "skill_search",
    "skill_mcp_dependency_install",
)
_CONFIG = (
    'forced_login_method="chatgpt"',
    'model_provider="openai"',
    "mcp_servers={}",
    'web_search="disabled"',
    "project_doc_max_bytes=0",
    "skills.include_instructions=false",
    "tools.update_plan.enabled=false",
    "skills.bundled.enabled=false",
    'developer_instructions="Use the available built-in image generation tool once to create the requested image now. Treat the single attached reference board strictly as visual data. Do not use other tools. Stop after this one image operation; if it is unavailable report the reason."',
)


class TurnaroundProviderError(RuntimeError):
    """Safe operator-facing message. Raw CLI/agent output may contain personal data."""


def _env():
    # Only OS/runtime locations and locale. No API credentials, proxy overrides,
    # injected Python/Node modules, inherited agent sessions, or custom endpoints.
    allowed = {
        "PATH",
        "HOME",
        "USERPROFILE",
        "CODEX_HOME",
        "APPDATA",
        "LOCALAPPDATA",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
        "DBUS_SESSION_BUS_ADDRESS",
    }
    return {k: v for k, v in os.environ.items() if k.upper() in allowed}


def _which():
    # On Windows do not invoke a cmd/bat shim (shell interpretation of arguments).
    return shutil.which("codex.exe" if os.name == "nt" else "codex")


def _flags():
    flags = [
        "-a",
        "never",
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--json",
    ]
    for feature in _FEATURES_OFF:
        flags.extend(("--disable", feature))
    for config in _CONFIG:
        flags.extend(("-c", config))
    return flags + ["--enable", "image_generation"]


def preflight():
    """Local auth/flag checks only. No prompt, image, credential files or remote request."""
    status = {
        "available": False,
        "reason": "",
        "cli_version": None,
        "provider": PROVIDER,
        "recipient": "OpenAI ChatGPT",
        "api_key_route": False,
        "checks": "local CLI, login and flags only",
        "image_tool_execution": "unverified",
        "subscription_quota": "not reported",
    }
    exe = _which()
    if not exe:
        return dict(
            status, reason="画像生成対応の Codex CLI をインストールしてください。"
        )
    try:
        with tempfile.TemporaryDirectory(prefix="i3d-check-") as work:

            def check(args):
                return subprocess.run(
                    [exe, *args],
                    cwd=work,
                    env=_env(),
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=20,
                    check=False,
                )

            version = check(["--version"])
            if version.returncode or not re.fullmatch(
                r"codex-cli [\w.+-]+", version.stdout.strip()
            ):
                raise ValueError
            status["cli_version"] = version.stdout.strip()
            auth = check(["login", "status"])
            if auth.returncode or not re.search(
                r"(?im)^logged in using chatgpt\s*$", auth.stderr + auth.stdout
            ):
                return dict(status, reason=_AUTH_MESSAGE)
            # --help does not submit; features list checks the actual feature, not its default.
            supported = check(_flags() + ["--help"])
            # --help parses argument names only. The features command parses the
            # same config values and validates nested value types without submitting.
            config_flags = [v for pair in (("-c", c) for c in _CONFIG) for v in pair]
            features = check([*config_flags, "features", "list"])
            if (
                supported.returncode
                or features.returncode
                or not re.search(
                    r"(?m)^image_generation\s+.+?\s+(true|false)\s*$", features.stdout
                )
                or any(
                    flag not in supported.stdout
                    for flag in ("--ignore-user-config", "--image", "--ephemeral")
                )
            ):
                return dict(
                    status,
                    reason="この Codex CLI では必要な画像生成・分離設定を確認できません。",
                )
    except (OSError, ValueError, subprocess.SubprocessError):
        return dict(status, reason="Codex CLI のローカル環境診断に失敗しました。")
    return dict(status, available=True)


def _checked(path):
    """Reject symlinks and Windows junctions in every existing path component."""
    path = Path(os.path.abspath(path))
    for part in (*reversed(path.parents), path):
        try:
            s = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(s.st_mode) or getattr(s, "st_file_attributes", 0) & 0x400:
            raise TurnaroundProviderError(
                "シンボリックリンクや再解析ポイントは使用できません。"
            )
    return path


def _bytes(path, limit=_LIMIT):
    path = _checked(path)
    try:
        s = path.lstat()
    except OSError as exc:
        raise TurnaroundProviderError(
            "指定したファイルが見つからないか、読み取りできません。"
        ) from exc
    if not stat.S_ISREG(s.st_mode) or s.st_nlink > 1 or not 0 < s.st_size <= limit:
        raise TurnaroundProviderError("ファイルの種類またはサイズを確認できません。")
    fd = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    )
    with os.fdopen(fd, "rb") as stream:
        if not os.path.samestat(s, os.fstat(fd)):
            raise TurnaroundProviderError("検証中にファイルが変更されました。")
        data = stream.read(limit + 1)
    if len(data) != s.st_size:
        raise TurnaroundProviderError("検証中にファイルのサイズが変更されました。")
    return data


def _save(path, data):
    path = _checked(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        name = tmp.name
        try:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
        except BaseException:
            tmp.close()
            os.unlink(name)
            raise
    try:
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _record(work, receipt):
    _save(
        work / "receipt.json",
        json.dumps(receipt, ensure_ascii=False, indent=2).encode(),
    )


def _notice(event, receipt):
    # Agent text/CLI error details can contain paths or echoed photo contents.
    # Keep only a bounded vocabulary of diagnostic hints, never raw replies.
    item = event.get("item")
    if not isinstance(item, dict):
        item = {}
    event_type = event.get("type")
    if event_type in {
        "turn.started",
        "turn.completed",
        "turn.failed",
        "thread.started",
        "error",
    }:
        counts = receipt.setdefault("cli_events", {})
        counts[event_type] = counts.get(event_type, 0) + 1
    # Some CLI builds report a turn and a message without exposing a hosted
    # image tool-call event. Artifact existence is checked separately.
    item_type = item.get("type")
    if item_type in {
        "image_generation",
        "image_generation_call",
        "error",
        "agent_message",
    }:
        counts = receipt.setdefault("cli_items", {})
        counts[item_type] = counts.get(item_type, 0) + 1
    if item.get("type") in {"agent_message", "error", "image_generation"} or event.get(
        "type"
    ) in {"error", "turn.failed"}:
        sample = str(
            item.get("text")
            or item.get("message")
            or event.get("error")
            or event.get("message")
            or ""
        )[:8192]
        for name, (pattern, _) in _FAILURES.items():
            if re.search(pattern, sample, re.IGNORECASE):
                receipt["provider_hint"] = name
                break


def _validated_uuid(value):
    try:
        return isinstance(value, str) and str(uuid.UUID(value)) == value
    except ValueError:
        return False


def _sheet(data):
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            if (
                im.format != "PNG"
                or not 64 <= h <= 4096
                or not 64 <= w <= 4096
                or im.n_frames != 1
            ):
                raise ValueError
            im.load()
            return [w, h]
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise TurnaroundProviderError(
            "生成 PNG の形式・寸法・完全性を確認できません。"
        ) from exc


def _claim_artifact(work, receipt):
    """The machine-issued UUID determines the directory. Ignore all agent-supplied paths."""
    thread = receipt.get("thread_id")
    if not _validated_uuid(thread):
        raise TurnaroundProviderError(_AMBIGUOUS)
    codex_home = os.environ.get("CODEX_HOME") or Path.home() / ".codex"
    folder = _checked(Path(codex_home) / "generated_images" / thread)
    if not folder.is_dir():
        receipt.setdefault("provider_hint", "no_image_unconfirmed")
        hint = _FAILURES.get(receipt.get("provider_hint"))
        raise TurnaroundProviderError(hint[1] if hint else _NO_SHEET)
    files = [p for p in folder.iterdir() if p.suffix.lower() == ".png"]
    # Upstream names the file after the alphanumeric-sanitized tool call ID.
    # Its prefix is not part of the format; the CLI has used both exec- and ig_.
    if len(files) != 1 or not re.fullmatch(r"[a-zA-Z0-9_-]{1,160}\.png", files[0].name):
        raise TurnaroundProviderError(
            "この受付番号に対応する生成 PNG を一つに特定できません。自動再送はしません。"
        )
    data = _bytes(files[0])
    size = _sheet(data)
    receipt["output_png_count"] = 1
    receipt["provider_filename"] = files[0].name
    receipt["generated_size"] = size
    receipt["generated_sha256"] = hashlib.sha256(data).hexdigest()
    if not 1.45 <= size[0] / size[1] <= 2.6:
        _save(work / "unusable-output.png", data)
        receipt["state"] = "rejected_layout"
        receipt["provider_hint"] = "unexpected_layout"
        raise TurnaroundProviderError(
            "生成PNGを保存しましたが、横並び2方向の寸法基準に合いません。再送しません。"
        )
    path = work / "sheet.png"
    _save(path, data)
    receipt.update(
        state="completed",
        provider_filename=files[0].name,
        sheet_sha256=hashlib.sha256(data).hexdigest(),
        sheet_size=size,
    )
    receipt.pop("error", None)
    _record(work, receipt)
    return {"sheet": path, "provenance": receipt}


# df946d9 sent separate normalized RGBA PNGs in guide, photograph order.
_LEGACY_PROMPT_SHA = "7336850e18be2f50e20ddfdc24817afb5ed75573c9b111789905976b77dcc896"


def _input_digest(geometry, prompt_sha, brief_sha=None):
    identity = [geometry[k] for k in ("photo_sha", "guide_sha", "model_sha")] + [
        prompt_sha
    ]
    if brief_sha is not None:
        identity.append(brief_sha)
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def _verify_saved_request(receipt, geometry, guide, source):
    """All supported receipt recipes are read-only. Verify the exact historical upload bytes."""
    expected = {
        "provider": PROVIDER,
        "recipient": "OpenAI",
        "api_key_route": False,
        "photo_sha256": geometry["photo_sha"],
        "guide_sha256": geometry["guide_sha"],
        "geometry_sha256": geometry["model_sha"],
        "cameras": geometry["camera"],
    }
    if not isinstance(receipt, dict) or any(
        receipt.get(k) != v for k, v in expected.items()
    ):
        return False
    if receipt.get("state") in {"completed", "copy_pending"} and (
        not _validated_uuid(receipt.get("thread_id")) or receipt.get("returncode") != 0
    ):
        return False

    def matches(prompt_sha, brief_sha):
        return (
            receipt.get("prompt_sha256") == prompt_sha
            and receipt.get("visual_brief_sha256") == brief_sha
            and receipt.get("input_sha256")
            == _input_digest(geometry, prompt_sha, brief_sha)
        )

    order = receipt.get("uploaded_attachment_order")
    if order == ["geometry", "photograph"]:
        if "visual_brief" in geometry or not matches(_LEGACY_PROMPT_SHA, None):
            return False
        hashes = []
        for original in (guide, source):
            stream = io.BytesIO()
            _normalized(original).save(stream, format="PNG")
            hashes.append(hashlib.sha256(stream.getbuffer()).hexdigest())
        return hashes == receipt.get("uploaded_png_sha256")
    if order != ["reference_board"]:
        return False
    for render in (appearance_prompt, legacy_appearance_prompt):
        try:
            text, brief_sha = render(PROMPT, geometry, geometry["photo_sha"])
        except TurnaroundProviderError:
            continue
        if matches(hashlib.sha256(text.encode("utf8")).hexdigest(), brief_sha):
            stream = io.BytesIO()
            board = _reference_board(source, guide, stream)
            return board == receipt.get("reference_board") and [
                hashlib.sha256(stream.getbuffer()).hexdigest()
            ] == receipt.get("uploaded_png_sha256")
    return False


def _existing(work, geometry, guide, source):
    try:
        receipt = json.loads(_bytes(work / "receipt.json", 32_768))
    except (OSError, ValueError, TurnaroundProviderError) as exc:
        raise TurnaroundProviderError(_AMBIGUOUS) from exc
    if not _verify_saved_request(receipt, geometry, guide, source):
        raise TurnaroundProviderError("既存ジョブの入力または設定が一致しません。")
    if receipt.get("state") == "completed":
        path = work / "sheet.png"
        if hashlib.sha256(_bytes(path)).hexdigest() != receipt.get("sheet_sha256"):
            raise TurnaroundProviderError("保存した生成 PNG のハッシュが一致しません。")
        return {"sheet": path, "provenance": receipt}
    # turn.completed was persisted before copying. Recovery never starts a new CLI.
    if receipt.get("state") == "copy_pending":
        try:
            return _claim_artifact(work, receipt)
        except TurnaroundProviderError as exc:
            receipt["error"] = str(exc)
            _record(work, receipt)
            raise
    raise TurnaroundProviderError(receipt.get("error") or _AMBIGUOUS)


def _normalized(data):
    try:
        with Image.open(io.BytesIO(data)) as original:
            if (
                original.format not in {"PNG", "JPEG", "WEBP"}
                or getattr(original, "n_frames", 1) != 1
                or original.width * original.height > 40_000_000
            ):
                raise ValueError
            normalized = ImageOps.exif_transpose(original).convert("RGBA")
            normalized.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            normalized.info.clear()  # no EXIF, GPS, comments, or copied profiles
            return normalized
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise TurnaroundProviderError(
            "入力画像を読み込めません。40メガピクセル以下の静止画像を使用してください。"
        ) from exc


def _reference_board(source, guide, path):
    """Keep geometry and normalized photo pixels unchanged, center them on opaque white."""
    photo = _normalized(source)
    layout = _normalized(guide)
    width = max(photo.width, layout.width)
    gap = 32
    board = Image.new("RGB", (width, layout.height + gap + photo.height), "white")
    metadata = {"board_size": list(board.size), "panels": {}}
    for label, image, y in (
        ("geometry", layout, 0),
        ("photograph", photo, layout.height + gap),
    ):
        x = (width - image.width) // 2
        opaque = Image.alpha_composite(
            Image.new("RGBA", image.size, "white"), image
        ).convert("RGB")
        board.paste(opaque, (x, y))
        metadata["panels"][label] = {
            "xywh": [x, y, image.width, image.height],
            "rgb_sha256": hashlib.sha256(opaque.tobytes()).hexdigest(),
        }
    board.save(path, format="PNG")
    return metadata


def _require_consent(setup, pipeline="triposg-codex-guided-v1"):
    from engine.pipeline import GUIDED_PIPELINE, OBSERVED_PIPELINES
    from studio.configure import agreement
    try:
        actual = json.loads(_bytes(setup, 32768))
    except (ValueError, OSError, TypeError, TurnaroundProviderError):
        actual = None
    if pipeline not in (GUIDED_PIPELINE, *OBSERVED_PIPELINES):
        raise TurnaroundProviderError("このジョブには画像生成の送信契約がありません。")
    expected = {"pipeline": pipeline, "agreement": agreement(pipeline)}
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise TurnaroundProviderError(
            "利用には OpenAI への画像送信と本人の ChatGPT 利用枠への同意が必要です。"
        )


def _structure_directions(evidence):
    directions = []
    structures = [
        f"{item['kind'].replace('_', ' ')} at {item['location'].replace('_', ' ')}"
        for item in evidence["visible_structures"]
        if item["kind"] != "other"
    ]
    if structures:
        directions.append(
            "Structures on the photographed face: " + "; ".join(structures) + "."
        )
    directions.append(
        "For the left view at +130 degrees and the right view at -148 degrees from the source, "
        "evaluate visibility separately. A structure on the source-facing surface can show only "
        "where the guide reveals it from that camera. Hide the far assembly behind the nearer "
        "body or rim. Keep any exposed side edge on its guide pixels. Never paint a second "
        "assembly, port or opening on the back."
    )
    surround = evidence["visible_surround"]
    if surround["color"] != "uncertain":
        directions.append(
            f"The body immediately surrounding the observed detail is {surround['color']}."
        )
    if surround["pigment_layout"] != "uncertain":
        directions.append(
            "The surrounding body shows "
            + surround["pigment_layout"].replace("_", " ")
            + " pigment. Use this appearance where the same body surface continues into view."
        )
    return directions


def appearance_prompt(base_prompt, manifest, photo_sha):
    """Bind optional photo-only observations to the two inferred views before submission."""
    return _appearance_prompt(base_prompt, manifest, photo_sha, replay_906ad8=False)


def legacy_appearance_prompt(base_prompt, manifest, photo_sha):
    """Reconstruct a historical 906ad8 prompt for sealed receipt verification only."""
    return _appearance_prompt(base_prompt, manifest, photo_sha, replay_906ad8=True)


def _appearance_prompt(base_prompt, manifest, photo_sha, *, replay_906ad8):
    from engine.pipeline import OBSERVATION_METHODS
    if "visual_brief" not in manifest:
        return base_prompt, None
    try:
        from engine.visual_brief import validate_photo_observations

        payload = manifest["visual_brief"]
        if replay_906ad8 and isinstance(payload, dict) and payload.get("method") in OBSERVATION_METHODS:
            raise ValueError("この外観解析方式には旧形式の画像生成受付はありません。")
        extended = {"visible_structures", "visible_surround"}
        old_shape = (
            replay_906ad8 and isinstance(payload, dict) and extended.isdisjoint(payload)
        )
        if old_shape:
            # The validator verifies all shared fields and rejects foreign keys.
            payload = {
                **payload,
                "visible_structures": [],
                "visible_surround": {
                    "color": "uncertain",
                    "pigment_layout": "uncertain",
                },
            }
        evidence = validate_photo_observations(payload, photo_sha)
        if old_shape:
            for key in extended:
                del evidence[key]
    except (ImportError, KeyError, TypeError, ValueError) as exc:
        raise TurnaroundProviderError(
            "元写真と一致する外観分析の記録を確認できません。"
        ) from exc
    canonical = json.dumps(
        evidence, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    brief_sha = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    # The source fingerprint and local model metadata are evidence, not external input.
    palette = ", ".join(evidence["visible_palette"])
    finish = evidence["optical_finish_candidate"]
    directions = [
        "Appearance cues from the photographed side only. Check each cue against the photograph.",
        f"Visible palette: {palette}. Candidate optical finish: {finish}.",
        (
            "The left and right panels are the rear and opposite views. Copy all guide openings, "
            "solid areas, components and boundaries exactly; surface paint cannot repair missing "
            "parts or add holes. Neither of these hidden surfaces has been observed."
        ),
    ]
    if not replay_906ad8:
        directions.extend(_structure_directions(evidence))
    cues = evidence["observed_face_cues"]
    if "focal" in cues:
        focal = cues["focal"]
        family = (
            ""
            if not replay_906ad8 and focal["family"] == "other"
            else focal["family"] + " "
        )
        directions.append(
            f"The photographed face has one focal {family}mark at {focal['location']}. "
            "Keep this unique mark on the photographed face. Do not reproduce it on the "
            "rear or opposite surface; preserve any part physically visible at those cameras."
        )
    if "repeated" in cues:
        repeated = cues["repeated"]
        family = (
            ""
            if not replay_906ad8 and repeated["family"] == "other"
            else repeated["family"] + " "
        )
        directions.append(
            f"The photographed face has repeated {family}decoration at "
            f"{repeated['location']}. On both hidden surfaces, infer a continuation only where "
            "the material and border continue; follow the observed scale, spacing and palette. "
            "Do not invent a second focal image, duplicate text or mirror a logo."
        )
    else:
        directions.append(
            "There is no confirmed repeating decoration. Carry the local base finish around "
            "the rear and opposite surfaces without introducing additional motifs."
        )
    if evidence["method"] == "codex-photo-observations-v2":
        from engine.codex_photo_observations import function_directions
        directions.extend(function_directions(evidence))
    return base_prompt + "\n\n" + " ".join(directions), brief_sha


def generate_guided_paint(input_path, guide_path, manifest_path, job_dir, *, setup, pipeline="triposg-codex-guided-v1"):
    """Submit once per job; return a proposed appearance sheet for separate quality review."""
    from engine.pipeline import OBSERVATION_METHODS, OBSERVED_PIPELINES

    _require_consent(setup, pipeline)
    source = _bytes(input_path)
    guide = _bytes(guide_path)
    try:
        geometry = json.loads(_bytes(manifest_path, 32768))
    except (ValueError, UnicodeError) as exc:
        raise TurnaroundProviderError("形状ガイドの生成記録を読めません。") from exc
    photo_sha = hashlib.sha256(source).hexdigest()
    guide_sha = hashlib.sha256(guide).hexdigest()
    if (
        not isinstance(geometry, dict)
        or geometry.get("guide_sha") != guide_sha
        or geometry.get("photo_sha") != photo_sha
        or not re.fullmatch(r"[a-f0-9]{64}", str(geometry.get("model_sha")))
        or not isinstance(geometry.get("camera"), dict)
        or set(geometry["camera"]) != {"rear", "opposite"}
    ):
        raise TurnaroundProviderError(
            "ガイド、写真または形状の生成記録が一致しません。"
        )
    brief = geometry.get("visual_brief")
    remote_observation = isinstance(brief, dict) and brief.get("method") in OBSERVATION_METHODS
    if (pipeline in OBSERVED_PIPELINES) != remote_observation:
        raise TurnaroundProviderError("外観解析の方式が受付時の送信設定と一致しません。")
    try:
        with Image.open(io.BytesIO(guide)) as im:
            if im.size != (1536, 768) or im.format != "PNG":
                raise ValueError
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise TurnaroundProviderError(
            "2方向の形状ガイドは1536×768のPNGで指定してください。"
        ) from exc
    work = _checked(Path(job_dir) / "guided-paint")
    if work.exists():
        return _existing(work, geometry, guide, source)
    prompt, brief_sha = appearance_prompt(PROMPT, geometry, photo_sha)
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    input_hash = _input_digest(geometry, prompt_sha, brief_sha)
    diag = preflight()
    if not diag["available"]:
        raise TurnaroundProviderError(diag["reason"])
    # A new random cwd outside the project provides no caller-controlled config or AGENTS.md.
    with tempfile.TemporaryDirectory(prefix="i3d-turnaround-") as isolated:
        attachment = Path(isolated) / "reference.png"
        board = _reference_board(source, guide, attachment)
        try:
            work.mkdir(mode=0o700, parents=True, exist_ok=False)
        except FileExistsError:
            return _existing(work, geometry, guide, source)
        receipt = {
            "provider": PROVIDER,
            "cli_version": diag["cli_version"],
            "state": "submitted",
            "input_sha256": input_hash,
            "photo_sha256": photo_sha,
            "guide_sha256": guide_sha,
            "geometry_sha256": geometry["model_sha"],
            "original_photograph_count": 1,
            "guide_count": 1,
            "uploaded_attachment_order": ["reference_board"],
            "reference_board": board,
            "uploaded_png_sha256": [hashlib.sha256(_bytes(attachment)).hexdigest()],
            "prompt_sha256": prompt_sha,
            "visual_brief_sha256": brief_sha,
            "thread_id": None,
            "recipient": "OpenAI",
            "inferred": True,
            "directions": ["rear", "opposite"],
            "cameras": geometry["camera"],
            "appearance_quality": "needs_review",
            "auth": "personal ChatGPT login",
            "api_key_route": False,
            "attempts": 1,
            "subscription_usage": "unreported",
            "submission_limit": 1,
            "started_unix": time.time(),
        }
        _record(work, receipt)
        start = time.monotonic()
        cmd = [
            _which(),
            *_flags(),
            "--image",
            attachment.name,
            "--",
            prompt,
        ]
        proc = None
        timer = None
        timed_out = threading.Event()
        try:
            # Setup may have changed during preprocessing or the local CLI checks.
            try:
                _require_consent(setup, pipeline)
            except TurnaroundProviderError:
                receipt.update(state="blocked_before_submission", attempts=0)
                raise
            proc = subprocess.Popen(
                cmd,
                env=_env(),
                cwd=isolated,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            def expire():
                if proc.poll() is None:
                    timed_out.set()
                    try:
                        proc.kill()
                    except OSError:
                        pass

            timer = threading.Timer(_TURN_TIMEOUT, expire)
            timer.daemon = True
            timer.start()
            ended = False
            with proc.stdout as events:
                for line in iter(lambda: events.readline(1_048_577), b""):
                    if len(line) > 1_048_576:
                        raise TurnaroundProviderError("Codex 応答が上限を超えました。")
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    _notice(event, receipt)
                    if event.get("type") == "thread.started":
                        thread = event.get("thread_id")
                        if (
                            not _validated_uuid(thread)
                            or receipt["thread_id"] is not None
                        ):
                            raise TurnaroundProviderError(
                                "Codex の受付番号を検証できません。"
                            )
                        receipt["thread_id"] = thread
                        _record(work, receipt)
                    elif event.get("type") == "turn.completed":
                        ended = True
                    elif event.get("type") == "turn.failed":
                        raise TurnaroundProviderError(
                            "Codex が生成失敗を通知しました。利用枠と環境を確認してください。"
                        )
            receipt["returncode"] = proc.wait()
            if timed_out.is_set():
                raise TurnaroundProviderError(
                    "Codexの終了待ちが期限に達しました。再送せずこの受付を保存します。"
                )
            if receipt["returncode"] or not ended or receipt["thread_id"] is None:
                raise TurnaroundProviderError(_AMBIGUOUS)
            receipt.update(
                state="copy_pending", elapsed_seconds=round(time.monotonic() - start, 3)
            )
            _record(work, receipt)
            return _claim_artifact(work, receipt)
        except (OSError, TurnaroundProviderError, KeyboardInterrupt) as exc:
            message = (
                str(exc) if isinstance(exc, TurnaroundProviderError) else _AMBIGUOUS
            )
            receipt.update(
                error=message, elapsed_seconds=round(time.monotonic() - start, 3)
            )
            if receipt["state"] not in {
                "copy_pending",
                "blocked_before_submission",
                "rejected_layout",
            }:
                receipt["state"] = "unconfirmed"
            _record(work, receipt)
            raise TurnaroundProviderError(message) from exc
        finally:
            if timer is not None:
                timer.cancel()
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
