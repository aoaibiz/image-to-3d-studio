"""One photograph to a textured local TripoSG mesh. No category dispatch or network inference."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from engine import triposg_cpu as sg
from engine.pipeline import SHAPE_PIPELINE
from tools.doctor import U2NET_SHA256

ROOT = Path(__file__).resolve().parents[1]


def runtime_path():
    configured = os.environ.get("STUDIO_TRIPOSG_RUNTIME")
    if not configured:
        return ROOT / ".runtime" / "triposg"
    path = Path(configured)
    if not path.is_absolute():
        raise ValueError("STUDIO_TRIPOSG_RUNTIME must be an absolute path.")
    return path


class StageError(RuntimeError):
    def __init__(self, code, reason):
        super().__init__(reason)
        self.code = code


def foreground(source, work, source_hash):
    from PIL import Image, ImageOps

    png = work / "foreground.png"
    record = work / "foreground.json"
    if record.exists():
        stamp = sg.read_json(record)
        if stamp.get("source_sha256") != source_hash or sg.digest(png) != stamp.get(
            "foreground_sha256"
        ):
            raise ValueError("Saved foreground differs from its input or checksum.")
        return png, stamp
    with Image.open(source) as f:
        if f.width * f.height > 40_000_000:
            raise ValueError("Input exceeds 40 megapixels.")
        image = ImageOps.exif_transpose(f).convert("RGBA")
    image.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
    alpha = image.getchannel("A")
    transparent = alpha.histogram()
    # Ignore isolated transparent pixels in otherwise opaque camera images.
    if sum(transparent[:33]) >= max(4, image.width * image.height // 500):
        method = "source alpha"
    else:
        home = ROOT / ".runtime" / "u2net"
        if sg.digest(home / "u2net.onnx") != U2NET_SHA256:
            raise ValueError("U2Net checksum mismatch.")
        os.environ["U2NET_HOME"] = str(home)
        os.environ["OMP_NUM_THREADS"] = "1"
        import rembg

        session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
        alpha = rembg.remove(
            image.convert("RGB"), session=session, only_mask=True
        ).convert("L")
        image.putalpha(alpha)
        method = "local U2Net alpha with original RGB"
    visible = alpha.point(lambda x: 255 if x >= 150 else 0).getbbox()
    if alpha.getextrema()[0] > 32 or visible is None:
        raise ValueError("No usable foreground silhouette.")
    # Discard nearly invisible exterior residue when framing the network input.
    # Keep the photographed soft edge and translucent interior inside the crop.
    left, top, right, bottom = visible
    margin = max(2, round(0.025 * max(right - left, bottom - top)))
    window = (
        max(0, left - margin),
        max(0, top - margin),
        min(image.width, right + margin),
        min(image.height, bottom + margin),
    )
    cutout = image.crop(window)
    framed = Image.new("RGBA", (cutout.width + 4, cutout.height + 4))
    framed.paste(cutout, (2, 2))
    tmp = png.with_suffix(".part")
    framed.save(tmp, format="PNG")
    os.replace(tmp, png)
    stamp = {
        "source_sha256": source_hash,
        "foreground_sha256": sg.digest(png),
        "method": method,
        "source_window_after_thumbnail": list(window),
        "visible_alpha_threshold": 150,
        "source_border_contact": left == 0
        or top == 0
        or right == image.width
        or bottom == image.height,
    }
    if method != "source alpha":
        stamp["segmentation_weights_sha256"] = U2NET_SHA256
    sg.atomic_json(record, stamp)
    return png, stamp


def run(source, out, resolution, texture_size):
    source, out = Path(source).resolve(), Path(out).resolve()
    start = time.monotonic()
    runtime = runtime_path()
    try:
        sg.runtime_python(runtime)
    except FileNotFoundError as exc:
        raise StageError("shape_setup", str(exc)) from exc
    out.mkdir(parents=True, exist_ok=True)
    work = out / "local-shape-work"
    work.mkdir(exist_ok=True)
    source_hash = sg.digest(source)
    with sg.writer_lock(work):
        print("segmenting source", flush=True)
        try:
            png, stamp = foreground(source, work, source_hash)
        except (OSError, RuntimeError, ValueError) as exc:
            raise StageError("shape_foreground", str(exc)) from exc
        print("generating shape", flush=True)
        provider = work / "triposg"
        raw = provider / "raw-shape.glb"
        try:
            receipt = sg.generate(png, provider, raw, runtime, cells=resolution)
        except subprocess.SubprocessError as exc:
            raise StageError("shape_setup", str(exc)) from exc
        except (OSError, RuntimeError, ValueError, MemoryError) as exc:
            raise StageError("shape_inference", str(exc)) from exc
        identity = sg.read_json(provider / "identity.json")
        if (
            receipt["sha256"] != sg.digest(raw)
            or identity["input_sha256"] != stamp["foreground_sha256"]
        ):
            raise StageError("shape_inference", "Raw inference provenance mismatch.")
        field = provider / f"field-{resolution}.npy"
        if not sg.valid_artifact(field, identity["job_key"]):
            raise StageError(
                "shape_inference", "Signed distance field failed its provider checksum."
            )
        print("painting model", flush=True)
        from engine.shape_appearance import paint_shape
        from engine.shape_camera import register_camera

        with tempfile.TemporaryDirectory(dir=work) as temporary:
            stage = Path(temporary) / "deliverables"
            try:
                camera = register_camera(raw, png)
                camera["foreground_touches_source_border"] = stamp.get(
                    "source_border_contact"
                )
                sg.atomic_json(work / "source-camera.json", camera)
                if camera["status"] == "poor-fit":
                    raise StageError(
                        "shape_registration",
                        f"Camera registration failed (IoU {camera['silhouette_iou']:.3f}).",
                    )
                appearance = paint_shape(
                    raw,
                    source,
                    png,
                    stage,
                    texture_size=texture_size,
                    sdf_field=field,
                    camera_frame=camera["camera_frame"],
                    display_from_raw=camera["upright_transform"],
                )
            except StageError:
                raise
            except (OSError, RuntimeError, ValueError, MemoryError) as exc:
                raise StageError("shape_appearance", str(exc)) from exc
            if (
                appearance["original_sha256"] != source_hash
                or appearance["shape_sha256"] != receipt["sha256"]
            ):
                raise StageError("shape_appearance", "Appearance provenance mismatch.")
            shutil.copyfile(png, stage / "input-processed.png")
            timings = [
                json.loads(line)
                for line in (provider / "timings.jsonl")
                .read_text(encoding="utf8")
                .splitlines()
            ]
            sg.atomic_json(
                stage / "generation.json",
                {
                    "engine": identity,
                    "shape_result": receipt,
                    "foreground": stamp,
                    "camera": camera,
                    "stage_history": timings,
                    "conditioning_photographs": 1,
                    "elapsed_for_this_invocation_seconds": round(
                        time.monotonic() - start, 2
                    ),
                    "external_generation_calls": 0,
                },
            )
            hashes = {p.name: sg.digest(p) for p in stage.iterdir() if p.is_file()}
            settings = {
                "pipeline": SHAPE_PIPELINE,
                "auto_finish_complete": True,
                "input_sha256": source_hash,
                "glb_sha256": hashes["model.glb"],
                "artifact_sha256": hashes,
                "marching_cubes_resolution": receipt["cells"],
                "texture_size": texture_size,
                "engine": "TripoSG",
                "source_revision": sg.SOURCE_REVISION,
                "weights_revision": sg.WEIGHTS_REVISION,
                "generation_history": "generation.json",
                "appearance_history": "appearance.json",
                "geometry_master": "shape-master.glb",
                "visual_quality": "needs_review",
            }
            # A crash leaves the provider checkpoint intact. The final manifest is the commit marker.
            (out / "settings.json").unlink(missing_ok=True)
            for file in stage.iterdir():
                os.replace(file, out / file.name)
            sg.atomic_json(out / "settings.json", settings)
    return settings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--resolution", type=int, choices=(128, 256), default=256)
    p.add_argument(
        "--texture-size", type=int, choices=(256, 512, 1024, 2048), default=1024
    )
    args = p.parse_args()
    try:
        run(args.image, args.output, args.resolution, args.texture_size)
    except StageError as exc:
        sg.emit("studio_error", code=exc.code, detail=str(exc))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
