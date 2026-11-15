"""Verify the three public diagnostics of a withheld SF3D finish."""
import hashlib
import json
import re
import struct

from engine.pipeline import SF3D_OBSERVED_PIPELINE


def _file(root, name, limit):
    path = root / name
    if path.resolve() != root.resolve() / name or not 0 < path.stat().st_size <= limit:
        raise ValueError("Invalid diagnostic file")
    return path


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_hold(out, source_sha):
    from engine.sf3d_observed_job import HOLD_REASONS

    try:
        hold = json.loads(_file(out, "review-hold.json", 4096).read_bytes())
        required = {"version", "pipeline", "input_sha256", "shape_sha256", "preview_sha256",
                    "stage", "reason_code", "reason"}
        if (not isinstance(hold, dict) or not required <= hold.keys() <= required | {"image_sha256"}
                or type(hold["version"]) is not int or hold["version"] != 1
                or hold["pipeline"] != SF3D_OBSERVED_PIPELINE or hold["input_sha256"] != source_sha
                or hold["stage"] not in {"native", "observation", "registration", "native_inspection", "guides", "paint",
                                         "orientation", "projection", "export"}
                or hold["reason"] != HOLD_REASONS[hold["reason_code"]]):
            return None
        base = out / "sf3d-observed-work" / "native"
        if base.resolve() != out.resolve() / "sf3d-observed-work" / "native":
            return None
        history = json.loads(_file(base, "generation.json", 2 << 20).read_bytes())
        preview = _file(out, "review-native.glb", 49 << 20)
        with preview.open("rb") as stream:
            magic, version, length = struct.unpack("<4sII", stream.read(12))
        if ((magic, version, length) != (b"glTF", 2, preview.stat().st_size)
                or not re.fullmatch(r"[0-9a-f]{64}", source_sha)
                or source_sha != _digest(_file(out.parent, "input.image", 20 << 20))
                or source_sha != history["source_selection"]["source_sha256"]
                or source_sha != history["provider_receipt"]["source_sha256"]
                or hold["shape_sha256"] != history["provider_receipt"]["artifact"]["sha256"]
                or hold["shape_sha256"] != history["native_sha256"]
                or hold["shape_sha256"] != _digest(_file(base, "provider-native.glb", 48 << 20))
                or hold["preview_sha256"] != history["output_sha256"]
                or hold["preview_sha256"] != _digest(preview)):
            return None
        artifacts = ["review-hold.json", "review-native.glb"]
        if hold.get("image_sha256") is not None:
            picture = _file(out, "review-native.png", 16 << 20)
            if _digest(picture) != hold["image_sha256"]:
                return None
            artifacts.append("review-native.png")
        return hold, artifacts
    except (OSError, ValueError, TypeError, KeyError, struct.error):
        return None
