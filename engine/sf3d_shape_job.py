"""Finalize an original Stable Fast 3D textured asset without re-baking its PBR material."""

import argparse
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path

from engine import triposg_cpu as files
from engine.pipeline import SF3D_AGREEMENT, SF3D_PIPELINE

AGREEMENT = SF3D_AGREEMENT

ARTIFACTS = {
    "model.glb",
    "provider-native.glb",
    "editable.gltf",
    "input-processed.png",
    "generation.json",
}


class ConsentError(InterruptedError):
    pass


def consent(setup, *, pipeline=SF3D_PIPELINE):
    try:
        config = files.read_json(setup)
    except (OSError, ValueError):
        config = None
    from studio.configure import ENGINES, valid_agreement

    if (
        pipeline not in (SF3D_PIPELINE, "sf3d-codex-observed-v1")
        or pipeline not in ENGINES.values()
        or not valid_agreement(config, pipeline)
    ):
        raise ConsentError(
            "公式Spaceへの前景画像送信と匿名利用枠の使用を、初回設定で許可してください。"
        )


def _matte(source, work, key):
    from PIL import Image
    from engine.auto_finish import prepare_source
    from engine.local_shape_job import foreground

    receipt = work / "selection.json"
    if receipt.exists():
        selection = files.read_json(receipt)
    else:
        with Image.open(source) as im:
            if im.width * im.height > 40_000_000 or getattr(im, "n_frames", 1) != 1:
                raise ValueError("40メガピクセル以下の静止画像を1枚選んでください。")
            selected, data = prepare_source(im, pipeline=SF3D_PIPELINE)
        temp = work / "selected.tmp"
        selected.save(temp, format="PNG")
        os.replace(temp, work / "selected.png")
        selection = {
            "source_sha256": key,
            "selected_sha256": files.digest(work / "selected.png"),
            "selection": data,
        }
        files.atomic_json(receipt, selection)
    if (
        selection["source_sha256"] != key
        or files.digest(work / "selected.png") != selection["selected_sha256"]
    ):
        raise ValueError("保存した被写体と元の写真が一致しません。")
    matte, metadata = foreground(
        work / "selected.png", work, selection["selected_sha256"]
    )
    return matte, selection, metadata


def _oriented_bytes(native):
    """Rotate the scene roots as rigid objects; copy the original binary chunk verbatim."""
    raw = Path(native).read_bytes()
    n, kind = struct.unpack_from("<II", raw, 12)
    if kind != 0x4E4F534A:
        raise ValueError("Missing glTF scene document.")
    doc = json.loads(raw[20 : 20 + n])
    nodes = doc["nodes"]
    for scene in doc["scenes"]:
        roots = scene.get("nodes", [])
        if roots:
            scene["nodes"] = [len(nodes)]
            nodes.append(
                {
                    "name": "SF3D source view",
                    "rotation": [0, 1, 0, 0],
                    "children": roots,
                }
            )
    data = json.dumps(doc, ensure_ascii=True, separators=(",", ":")).encode()
    data += b" " * ((-len(data)) % 4)
    bin_chunk = raw[20 + n :]
    return (
        struct.pack(
            "<4sIIII", b"glTF", 2, 20 + len(data) + len(bin_chunk), len(data), kind
        )
        + data
        + bin_chunk
    )


def oriented_glb(native, target):
    target.write_bytes(_oriented_bytes(native))


def oriented_digest(native):
    import hashlib
    return hashlib.sha256(_oriented_bytes(native)).hexdigest()


def finalize(native, matte, selection, matte_meta, receipt, out):
    from engine import sf3d_provider as provider
    from engine.finish_export import embed_master

    actual = provider.native_glb(native)
    if (
        actual["sha256"] != receipt["artifact"]["sha256"]
        or actual["faces"] != receipt["artifact"]["faces"]
        or files.digest(matte) != receipt["foreground_sha256"]
        or selection["source_sha256"] != receipt["source_sha256"]
    ):
        raise ValueError("公式GLB・元画像・前景の生成記録を照合できません。")
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        stage = Path(tmp)
        shutil.copyfile(native, stage / "provider-native.glb")
        oriented_glb(native, stage / "model.glb")
        result = provider.native_glb(stage / "model.glb")
        if (
            result["faces"] != actual["faces"]
            or result["embedded_images"] != actual["embedded_images"]
        ):
            raise ValueError("表示用の座標変換で形状または画像が変化しました。")
        embed_master(stage / "model.glb", stage / "editable.gltf")
        shutil.copyfile(matte, stage / "input-processed.png")
        camera = {
            "provider_source_axis": "-Z",
            "source_view": "front",
            "display_source_axis": "+Z",
            "display_rotation_xyzw": [0, 1, 0, 0],
            "basis": "SF3D conditioned-view convention; not a measured physical camera",
        }
        files.atomic_json(
            stage / "generation.json",
            {
                "engine": "Stable Fast 3D",
                "source_selection": selection,
                "matte": matte_meta,
                "provider_receipt": receipt,
                "camera": camera,
                "output_sha256": result["sha256"],
                "native_sha256": actual["sha256"],
                "visual_quality": "needs_review",
            },
        )
        hashes = {p.name: files.digest(p) for p in stage.iterdir()}
        settings = {
            "pipeline": SF3D_PIPELINE,
            "engine": "Stable Fast 3D",
            "auto_finish_complete": True,
            "input_sha256": receipt["source_sha256"],
            "glb_sha256": result["sha256"],
            "artifact_sha256": hashes,
            "source_selection": selection["selection"],
            "geometry_master": "provider-native.glb",
            "editable_master": "editable.gltf",
            "generation_history": "generation.json",
            "provider_parameters": receipt["settings"],
            "camera": camera,
            "triangles": actual["faces"],
            "embedded_images": actual["embedded_images"],
            "visual_quality": "needs_review",
        }
        (out / "settings.json").unlink(missing_ok=True)
        for name in sorted(hashes):
            os.replace(stage / name, out / name)
        files.atomic_json(out / "settings.json", settings)
        return settings


def valid(out, settings):
    try:
        hashes = settings["artifact_sha256"]
        if (
            set(hashes) != ARTIFACTS
            or settings["pipeline"] != SF3D_PIPELINE
            or not all(files.digest(out / n) == h for n, h in hashes.items())
        ):
            return False
        hist = files.read_json(out / "generation.json")
        receipt = hist["provider_receipt"]
        return (
            hist["output_sha256"] == settings["glb_sha256"] == hashes["model.glb"]
            and hist["native_sha256"]
            == hashes["provider-native.glb"]
            == receipt["artifact"]["sha256"]
            and receipt["source_sha256"]
            == settings["input_sha256"]
            == hist["source_selection"]["source_sha256"]
            and hashes["input-processed.png"] == receipt["foreground_sha256"]
            and settings["camera"] == hist["camera"]
            and settings["camera"]["source_view"] == "front"
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def run(source, out, *, setup, progress=None, cancelled=None, pipeline=SF3D_PIPELINE):
    from engine import sf3d_provider as provider

    source, out, setup = map(lambda p: Path(p).resolve(), (source, out, setup))
    work = out / "sf3d-work"
    work.mkdir(parents=True, exist_ok=True)
    key = files.digest(source)
    with files.writer_lock(work):
        marker = out / "settings.json"
        if marker.exists():
            settings = files.read_json(marker)
            if settings.get("input_sha256") == key and valid(out, settings):
                return settings
        if (
            marker.exists()
            or (out / "generation.json").exists()
            or (out / "provider-native.glb").exists()
        ) and not (work / "provider" / "provider.json").is_file():
            raise ValueError("元の受付記録が失われました。再送を避けるため停止します。")
        consent(setup, pipeline=pipeline)
        print("segmenting source", flush=True)
        matte, selection, metadata = _matte(source, work, key)
        consent(setup, pipeline=pipeline)
        if files.digest(source) != key:
            raise ValueError("処理中に元画像が変更されました。")

        def changed(stage):
            if progress:
                progress(stage)
            if stage in (
                "uploaded",
                "preprocess_waiting",
                "preprocessed",
                "generation_waiting",
                "downloading",
                "complete",
            ):
                print("sf3d " + stage, flush=True)

        def stopped():
            consent(setup, pipeline=pipeline)
            return bool(cancelled and cancelled())

        print("sf3d connecting", flush=True)
        receipt = provider.generate(
            matte,
            key,
            work / "provider",
            allow_official_space=True,
            progress=changed,
            cancelled=stopped,
        )
        print("sf3d exporting", flush=True)
        return finalize(
            work / "provider" / "model.glb", matte, selection, metadata, receipt, out
        )


if __name__ == "__main__":
    from engine import sf3d_provider as provider

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--setup", type=Path, required=True)
    a = p.parse_args()
    try:
        run(a.image, a.output, setup=a.setup)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        code = (
            "sf3d_pending"
            if isinstance(exc, provider.Pending)
            else "sf3d_consent"
            if isinstance(exc, ConsentError)
            else "sf3d_error"
        )
        files.emit("studio_error", code=code, detail=str(exc))
        raise SystemExit(1)
