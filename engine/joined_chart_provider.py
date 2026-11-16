"""Admit exactly one consented ChatGPT image edit for a joined intrinsic chart."""

import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import time

from engine import guided_provider as provider
from engine.joined_paint_chart import METHOD, SIZE, confirm_chart
from engine.pipeline import (
    SF3D_OBSERVED_PIPELINE,
    SG_CHART_PIPELINE,
    OBSERVATION_METHODS,
)
from engine.visual_brief import json_sha, validate_photo_observations

PROMPT = """Use the built-in image generation tool once. The upper wide rectangle is one continuous UV texture strip, unwrapped from the actual curved surface of the object. It is a single continuous painting canvas, not a 3D view, not two views, and not a montage. The lower photograph shows the real observed side. Return only the upper wide rectangle, filling the central gray-clay area with an inferred continuous diffuse colour painting that suits the same object. The left and right coloured margins are fixed measured samples from the photograph. Keep their precise positions, sizes, colours, broad lines and fine details unchanged. Extend compatible lines across each input boundary, matching their thickness, scale and flow. Preserve the existing registration, top and bottom, and keep object-up unchanged. No caption, photo, label, extra frame, cast shadow, highlight, perspective recamera, new structural feature, mirrored frontal image, repeated unique central emblem, lettering, or branding. The object's unseen decoration is inferred: continue surrounding palette and ornamental vocabulary naturally, reserving each distinctive photographed feature for its original face. If the tool cannot produce the requested image, report its failure; do not retry, use code, or use another tool."""


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _identity(manifest, source, guide, target, pipeline):
    if pipeline not in (SF3D_OBSERVED_PIPELINE, SG_CHART_PIPELINE):
        raise provider.TurnaroundProviderError(
            "この塗装方式には専用の写真＋ガイド送信設定が必要です。"
        )
    rest = dict(manifest)
    cid = rest.pop("chart_id", None)
    if (
        cid != json_sha(rest)
        or rest.get("method") != METHOD
        or rest.get("photo_sha") != _sha(source)
        or rest.get("guide_sha") != _sha(guide)
        or provider._sheet(guide) != list(SIZE)
        or not all(
            re.fullmatch(r"[a-f0-9]{64}", str(rest.get(k)))
            for k in ("registration_sha", "foreground_sha", "model_sha")
        )
    ):
        raise provider.TurnaroundProviderError(
            "展開図、元写真または受付設定の識別子が一致しません。"
        )
    brief = validate_photo_observations(rest["visual_brief"], rest["photo_sha"])
    if brief.get("method") not in OBSERVATION_METHODS or json_sha(brief) != rest.get(
        "visual_brief_sha256"
    ):
        raise provider.TurnaroundProviderError(
            "元写真に対応する外観観察を確認できません。"
        )
    board = provider._reference_board(source, guide, target)
    payload = (
        target.getvalue() if hasattr(target, "getvalue") else provider._bytes(target)
    )
    return dict(
        provider=provider.PROVIDER,
        recipient="OpenAI",
        api_key_route=False,
        method=METHOD,
        job_recipe=METHOD,
        auth="personal ChatGPT login",
        pipeline=pipeline,
        chart_id=cid,
        input_sha256=json_sha(
            [METHOD, cid, _sha(PROMPT.encode()), _sha(payload), pipeline]
        ),
        photo_sha256=rest["photo_sha"],
        guide_sha256=rest["guide_sha"],
        geometry_sha256=rest["model_sha"],
        foreground_sha256=rest["foreground_sha"],
        registration_sha256=rest["registration_sha"],
        prompt_sha256=_sha(PROMPT.encode()),
        visual_brief_sha256=json_sha(brief),
        original_photograph_count=1,
        guide_count=1,
        inferred=True,
        directions=["joined_chart"],
        uploaded_attachment_order=["reference_board"],
        uploaded_png_sha256=[_sha(payload)],
        reference_board=board,
        submission_limit=1,
    )


def _matches(receipt, identity):
    if not isinstance(receipt, dict) or any(
        receipt.get(k) != v for k, v in identity.items()
    ):
        raise provider.TurnaroundProviderError(
            "既存の画像生成受付が別の設定です。受付済みの方式だけで再開できます。"
        )


def verify_completed(receipt, manifest, source_path, guide_path, payload):
    identity = _identity(
        manifest,
        provider._bytes(source_path),
        provider._bytes(guide_path),
        io.BytesIO(),
        receipt.get("pipeline"),
    )
    _matches(receipt, identity)
    if (
        receipt.get("state") != "completed"
        or receipt.get("attempts") != 1
        or receipt.get("returncode") != 0
        or not provider._validated_uuid(receipt.get("thread_id"))
        or receipt.get("sheet_sha256") != _sha(payload)
        or receipt.get("generated_sha256") != _sha(payload)
        or receipt.get("sheet_size") != provider._sheet(payload)
    ):
        raise provider.TurnaroundProviderError(
            "展開図の完了受付と生成ファイルが一致しません。"
        )


def _recover(work, identity):
    try:
        receipt = json.loads(provider._bytes(work / "receipt.json", 32768))
    except (OSError, ValueError, provider.TurnaroundProviderError) as exc:
        raise provider.TurnaroundProviderError(provider._AMBIGUOUS) from exc
    _matches(receipt, identity)
    if (
        receipt.get("attempts") != 1
        or receipt.get("returncode") != 0
        or not provider._validated_uuid(receipt.get("thread_id"))
    ):
        raise provider.TurnaroundProviderError(
            "受付結果を確定できません。自動で再送しません。"
        )
    if receipt["state"] == "copy_pending":
        return provider._claim_artifact(work, receipt)
    if receipt["state"] != "completed":
        raise provider.TurnaroundProviderError(
            "受付が完了していません。自動で再送しません。"
        )
    path = work / "sheet.png"
    blob = provider._bytes(path)
    if (
        _sha(blob) != receipt.get("sheet_sha256")
        or receipt.get("generated_sha256") != _sha(blob)
        or receipt.get("sheet_size") != provider._sheet(blob)
    ):
        raise provider.TurnaroundProviderError(
            "保存した展開図のハッシュまたは寸法が不一致です。"
        )
    return {"sheet": path, "provenance": receipt}


def chart_consent(setup, pipeline=SF3D_OBSERVED_PIPELINE):
    provider._require_chart_consent(setup, pipeline)


def generate_chart(
    input_path,
    guide_path,
    manifest_path,
    job_dir,
    *,
    native_model,
    foreground,
    registration,
    observation,
    setup,
    pipeline=SF3D_OBSERVED_PIPELINE,
    before_submission=None,
):
    """Shares the physical guided-paint reservation with the two-view recipe."""
    chart_consent(setup, pipeline)
    manifest = json.loads(provider._bytes(manifest_path, 65536))
    source = provider._bytes(input_path)
    guide = provider._bytes(guide_path)
    work = provider._checked(Path(job_dir) / "guided-paint")
    with tempfile.TemporaryDirectory(prefix="i3d-continuous-paint-") as isolated:
        attachment = Path(isolated) / "reference.png"
        identity = _identity(manifest, source, guide, attachment, pipeline)
        if work.exists():
            prior = json.loads(provider._bytes(work / "receipt.json", 32768))
            _matches(prior, identity)
            if (pipeline == SG_CHART_PIPELINE
                    and before_submission is not None
                    and prior.get("state") == "blocked_before_submission"
                    and prior.get("attempts") == 0 and prior.get("thread_id") is None
                    and not (work.parent / "paint-intent.json").exists()
                    and not (work / "sheet.png").exists()):
                # No external process was started; preserve the old journal and claim afresh.
                archive = Path(tempfile.mkdtemp(prefix="paint-no-send-", dir=work.parent))
                work.rename(archive / "journal")
            else:
                return _recover(work, identity)
        confirm_chart(
            source_photo=input_path,
            native_model=native_model,
            foreground=foreground,
            registration=registration,
            observation=observation,
            guide_bytes=guide,
            manifest=manifest,
        )
        diag = provider.preflight()
        if not diag["available"]:
            raise provider.TurnaroundProviderError(diag["reason"])
        try:
            work.mkdir(mode=0o700, parents=True, exist_ok=False)
        except FileExistsError:
            return _recover(work, identity)
        receipt = dict(
            identity,
            cli_version=diag["cli_version"],
            state="submitted",
            thread_id=None,
            appearance_quality="needs_review",
            attempts=1,
            subscription_usage="unreported",
            started_unix=time.time(),
        )
        provider._record(work, receipt)
        def confirm_setup(config, selected):
            chart_consent(config, selected)
            if before_submission is not None:
                before_submission()

        return provider._run_image_once(
            work,
            receipt,
            attachment,
            PROMPT,
            setup,
            pipeline,
            consent_check=confirm_setup,
        )
