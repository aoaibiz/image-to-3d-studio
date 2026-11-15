"""Bind downloadable results and redacted diagnostics to the verified TripoSG chart job."""
import json
import os
import shutil
import tempfile
from pathlib import Path

from .review_hold import _digest, _file

CHART_REASONS = {
    "registration_or_chart_unsupported": "写真と立体を照合しましたが、模様をつなげる範囲を確認できませんでした。",
    "proposed_paint_rejected": "推定した塗装が元写真または立体の形と合わず、採用を保留しました。",
    "provider_unavailable": "塗装用の画像生成を利用できませんでした。",
    "provider_unconfirmed": "塗装用の画像生成の受付結果が未確定のため、自動再送を停止しました。",
}
CHART_PAINT_FILES = {"chart-guide.png", "inferred-paint.png", "chart-paint.json"}


def _checked(out, settings, source_sha):
    from engine.guided_shape_job import valid_chart
    from engine.pipeline import SG_CHART_PIPELINE
    if (settings["pipeline"] != SG_CHART_PIPELINE or settings["input_sha256"] != source_sha
            or not valid_chart(out, settings)):
        raise ValueError("Unverified chart output")
    report = json.loads(_file(out, "chart-result.json", 256 << 10).read_bytes())
    if (report["pipeline"] != SG_CHART_PIPELINE or report["source_sha256"] != source_sha
            or report["decision"] != settings["chart_decision"] or report["output_sha256"] != settings["glb_sha256"]
            or type(report["paint_attempts"]) is not int or report["paint_attempts"] not in (0, 1)):
        raise ValueError("Chart report disagrees with the job")
    return report


def chart_files(out, settings, source_sha):
    try:
        report = _checked(out, settings, source_sha)
        decision, count = report["decision"], report["paint_attempts"]
        if (decision, report["reason"], count) not in {
                ("local_continuity", "measured_continuity", 0), ("inferred_chart", "one_inferred_chart", 1)}:
            return None
        names = {"shape-master.glb", "appearance.json", "generation.json", "chart-result.json"}
        if decision == "inferred_chart":
            names.update(CHART_PAINT_FILES)
        else:
            names.add("continuity-paint.json")
        sealed = settings["artifact_sha256"]
        if not names <= sealed.keys() or any((out / n).resolve() != out.resolve() / n for n in names):
            return None
        return sorted(names), {"decision": decision, "paint_attempts": count, "report": "chart-result.json"}
    except (ValueError, TypeError, KeyError, OSError):
        return None


def _hold(out, settings, source_sha):
    report = _checked(out, settings, source_sha)
    if (report["decision"] != "diagnostic_source_model" or report["reason"] not in CHART_REASONS
            or report["base_sha256"] != report["output_sha256"]
            or _digest(_file(out.parent, "input.image", 20 << 20)) != source_sha):
        raise ValueError("Unverified diagnostic")
    return {"version": 1, "pipeline": report["pipeline"], "input_sha256": source_sha,
            "shape_sha256": report["shape_sha256"], "preview_sha256": report["output_sha256"],
            "stage": "paint", "reason_code": report["reason"], "reason": CHART_REASONS[report["reason"]]}


def publish_chart_hold(out, settings, source_sha):
    """Install a digest-checked diagnostic preview and publish its marker last."""
    hold = _hold(out, settings, source_sha)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        root = Path(tmp)
        preview = root / "review-native.glb"
        shutil.copyfile(_file(out, "model.glb", 49 << 20), preview)
        if _digest(preview) != hold["preview_sha256"]:
            raise ValueError("Diagnostic GLB changed")
        (root / "review-hold.json").write_text(json.dumps(hold, ensure_ascii=False), encoding="utf-8")
        for name in ("review-native.glb", "review-hold.json"):
            os.replace(root / name, out / name)


def read_chart_hold(out, source_sha):
    try:
        settings = json.loads(_file(out, "settings.json", 1 << 20).read_bytes())
        expected = _hold(out, settings, source_sha)
        stored = json.loads(_file(out, "review-hold.json", 4096).read_bytes())
        if stored != expected or _digest(_file(out, "review-native.glb", 49 << 20)) != expected["preview_sha256"]:
            return None
        return expected, ["review-hold.json", "review-native.glb"]
    except (OSError, KeyError, ValueError, TypeError):
        return None
