"""Prepare camera evidence using the fast endpoint's expected square black RGB input."""

import hashlib
import io
import os
from pathlib import Path

from PIL import Image

from engine.local_shape_job import foreground
from engine.triposg_cpu import digest


def prepare(normalized_png, work):
    """Preserve the local measured alpha and composite it only for the model input."""
    work = Path(work)
    work.mkdir(exist_ok=True)
    source = work / "normalized.png"
    key = hashlib.sha256(normalized_png).hexdigest()
    if source.exists():
        if digest(source) != key:
            raise ValueError("Prepared input and source image hashes differ.")
    else:
        part = source.with_suffix(".part")
        part.write_bytes(normalized_png)
        os.replace(part, source)
    matte, metadata = foreground(source, work, key)
    with Image.open(matte) as decoded:
        img = decoded.convert("RGBA")
    strong = img.getchannel("A").point(lambda a: 255 if a > 204 else 0)
    bounds = strong.getbbox()
    if not bounds:
        raise ValueError("No opaque foreground could be measured.")
    x0, y0, x1, y1 = bounds
    side = max(x1 - x0, y1 - y0)
    if side < 4:
        raise ValueError("The measured object is too small.")
    side = round(side * 1.1)
    left, top = (x0 + x1 - side) // 2, (y0 + y1 - side) // 2
    framed = img.crop((left, top, left + side, top + side))
    framed.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    rgb = Image.new("RGB", framed.size)
    rgb.paste(framed, mask=framed.getchannel("A"))
    buffer = io.BytesIO()
    rgb.save(buffer, "PNG")
    return buffer.getvalue(), {
        "method": "pixal-local-foreground-v1", "normalized_sha256": key,
        "foreground_sha256": metadata["foreground_sha256"], "alpha_method": metadata["method"],
        "segmentation_weights_sha256": metadata.get("segmentation_weights_sha256"),
        "square_after_foreground_framing": [left, top, side], "upload_size": list(framed.size),
        "foreground_threshold": 204, "background_rgb": [0, 0, 0],
    }
