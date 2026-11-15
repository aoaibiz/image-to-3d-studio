"""Run a single explicitly authorized public CC0 image through the Pixal3D community demo."""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from engine import triposg_cpu as files
from engine.pixal_identity import matches_completed
from engine.pipeline import PIXAL_AGREEMENT, PIXAL_PIPELINE

ARTIFACTS = {"model.glb", "provider-native.glb", "editable.gltf", "input-processed.png", "generation.json"}


class ConsentError(ValueError):
    pass


def consent(setup, source_sha, granted):
    from studio.configure import read_pipeline

    if (setup.name != "engine.json" or read_pipeline(setup.parent) != PIXAL_PIPELINE
            or not source_sha or source_sha != granted):
        raise ConsentError("このCC0公開写真の個別同意と送信先を確認してください。")


def valid(out, settings, *, receipt_path=None):
    try:
        out = Path(out)
        seals = settings["artifact_sha256"]
        if settings["pipeline"] != PIXAL_PIPELINE or set(seals) != ARTIFACTS:
            return False
        if any((out / name).resolve() != out.resolve() / name or files.digest(out / name) != sha
               for name, sha in seals.items()):
            return False
        history = files.read_json(out / "generation.json")
        receipt = history["provider_receipt"]
        original_receipt = receipt_path or out / "pixal-work" / "provider" / "provider.json"
        if original_receipt.is_symlink() or receipt != files.read_json(original_receipt):
            return False
        return (history["agreement"] == PIXAL_AGREEMENT
                and matches_completed(receipt)
                and receipt["quality"] == settings["visual_quality"] == "needs_review"
                and receipt["source_sha256"] == history["public_cc0_sha256"] == settings["input_sha256"]
                and receipt["sha256"] == receipt["material"]["sha256"] == settings["glb_sha256"]
                == seals["model.glb"] == seals["provider-native.glb"] == receipt["artifacts"]["model.glb"]
                and receipt["upload_sha256"] == seals["input-processed.png"] == receipt["artifacts"]["input.png"]
                and receipt["material"]["triangles"] == settings["triangles"]
                and history["camera"] == settings["camera"])
    except (OSError, KeyError, TypeError, ValueError):
        return False


def finalize(work, out, key, receipt):
    from engine.finish_export import embed_master
    from engine.pixal_provider import inspect_pbr

    native, upload = work / "model.glb", work / "input.png"
    material = inspect_pbr(native)
    if (material != receipt["material"] or material["sha256"] != receipt["sha256"]
            or receipt["source_sha256"] != key or receipt["upload_sha256"] != files.digest(upload)
            or receipt["quality"] != "needs_review"):
        raise ValueError("Pixal3Dの元画像、送信画像、材質付きGLBの受付記録が一致しません。")
    camera = {"source_view": "unmeasured", "display_axis": "+Z", "basis": "unaltered native scene"}
    with tempfile.TemporaryDirectory(dir=out) as temp:
        stage = Path(temp)
        shutil.copyfile(native, stage / "provider-native.glb")
        shutil.copyfile(native, stage / "model.glb")
        embed_master(stage / "model.glb", stage / "editable.gltf")
        shutil.copyfile(upload, stage / "input-processed.png")
        files.atomic_json(stage / "generation.json", {
            "engine": "Pixal3D", "provider_receipt": receipt, "agreement": PIXAL_AGREEMENT,
            "public_cc0_sha256": key, "camera": camera,
        })
        seals = {name: files.digest(stage / name) for name in ARTIFACTS}
        settings = {
            "pipeline": PIXAL_PIPELINE, "engine": "Pixal3D", "auto_finish_complete": True,
            "input_sha256": key, "glb_sha256": material["sha256"], "artifact_sha256": seals,
            "geometry_master": "provider-native.glb", "editable_master": "editable.gltf",
            "generation_history": "generation.json", "visual_quality": "needs_review",
            "camera": camera, "triangles": material["triangles"], "pbr_channels": material["channels"],
        }
        if not valid(stage, settings, receipt_path=work / "provider.json"):
            raise ValueError("Pixal3D成果物のハッシュを照合できません。")
        (out / "settings.json").unlink(missing_ok=True)
        for name in ARTIFACTS:
            os.replace(stage / name, out / name)
        files.atomic_json(out / "settings.json", settings)
        return settings


def run(source, out, *, setup, public_cc0_sha256):
    from engine import pixal_provider

    source, out, setup = (Path(p).resolve() for p in (source, out, setup))
    key = files.digest(source)
    consent(setup, key, public_cc0_sha256)
    work = out / "pixal-work"
    provider_work = work / "provider"
    work.mkdir(parents=True, exist_ok=True)
    with files.writer_lock(work):
        marker = out / "settings.json"
        if marker.is_file():
            settings = files.read_json(marker)
            if settings.get("input_sha256") == key and valid(out, settings):
                return settings
        ticket = work / "attempt.json"
        identity = {"pipeline": PIXAL_PIPELINE, "public_cc0_sha256": key}
        if ticket.exists() or marker.exists() or any((out / name).exists() for name in ARTIFACTS):
            if files.read_json(ticket) != identity or not (provider_work / "provider.json").is_file():
                raise ValueError("以前の受付を確認できないため、再送を停止しました。")

        def stage(name):
            if name in {"upload_unknown", "submission_unknown"}:
                consent(setup, key, public_cc0_sha256)
                files.atomic_json(ticket, identity)
            markers = {"uploaded": "uploaded", "accepted": "generation_waiting",
                       "downloading": "downloading", "completed": "complete"}
            if name in markers:
                print("pixal " + markers[name], flush=True)

        print("pixal connecting", flush=True)
        receipt = pixal_provider.generate(source, provider_work, allow_public_cc0=True, progress=stage, expected_source_sha256=key)
        if key != files.digest(source):
            raise ValueError("生成中に入力ファイルが変わりました。")
        print("pixal exporting", flush=True)
        return finalize(provider_work, out, key, receipt)


if __name__ == "__main__":
    from engine import pixal_provider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--public-cc0-sha256", required=True)
    args = parser.parse_args()
    try:
        run(args.source, args.out, setup=args.setup, public_cc0_sha256=args.public_cc0_sha256)
    except pixal_provider.QuotaExceeded as exc:
        files.emit("studio_error", code="pixal_quota", **exc.details)
        raise SystemExit(1)
    except (KeyError, RuntimeError, ValueError, OSError) as exc:
        code = "pixal_consent" if isinstance(exc, ConsentError) else "pixal_pending" if isinstance(exc, pixal_provider.Pending) else "pixal_error"
        files.emit("studio_error", code=code)
        raise SystemExit(1)
