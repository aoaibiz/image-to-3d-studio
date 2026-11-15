"""Bind the review sheets to the actual input, model, and captured views.

Run after the browser proof. This command makes no visual quality judgment.
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


VIEWS = (
    "front",
    "right",
    "back",
    "left",
    "top",
    "bottom",
    "front-right",
    "front-left",
    "back-right",
    "back-left",
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_sheets(job, reference, evidence):
    job_id = job.parent.name if job.name == "output" else job.name
    model = job / "model.glb"
    model_hash = digest(model)
    report = json.loads((evidence / "manifest.json").read_text(encoding="utf-8-sig"))
    settings = json.loads((job / "settings.json").read_text(encoding="utf-8-sig"))
    if (
        report.get("synthetic") is not False
        or report["sha256"] != model_hash
        or settings["glb_sha256"] != model_hash
        or settings["input_sha256"] != digest(reference)
    ):
        raise ValueError(
            "Input, inference result and browser evidence must refer to the same job."
        )
    entries = []
    sheets = []
    for mode in ("material", "clay"):
        sheet = Image.new("RGB", (1200, 1112), "#f8faf9")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=16)
        draw.text(
            (18, 12),
            f"{job_id}  /  {mode}  /  GLB {model_hash[:16]}",
            fill="#162921",
            font=font,
        )
        for i, view in enumerate(VIEWS):
            filename = f"{model_hash}_{view}_{mode}.png"
            if filename not in report["files"]:
                raise ValueError(f"Missing browser evidence for {view} {mode}")
            path = evidence / filename
            with Image.open(path) as frame:
                frame.load()
                thumbnail = frame.convert("RGB")
                backdrop = Image.new("RGB", thumbnail.size, thumbnail.getpixel((0, 0)))
                bounds = ImageChops.difference(thumbnail, backdrop).getbbox()
                if bounds:
                    left, top, right, bottom = bounds
                    gutter = 10
                    crop = (
                        max(0, left - gutter),
                        max(0, top - gutter),
                        min(frame.width, right + gutter),
                        min(frame.height, bottom + gutter),
                    )
                    thumbnail = thumbnail.crop(crop)
                thumbnail.thumbnail((570, 178), Image.Resampling.LANCZOS)
            x, y = 18 + (i % 2) * 594, 48 + (i // 2) * 212
            sheet.paste(thumbnail, (x + (570 - thumbnail.width) // 2, y))
            draw.text((x, y + 180), view, fill="#162921", font=font)
            entries.append(
                {"file": filename, "view": view, "mode": mode, "sha256": digest(path)}
            )
        output = evidence / f"sheet-{mode}.jpg"
        sheet.save(output, quality=92)
        sheets.append({"file": output.name, "sha256": digest(output)})
    result = {
        "job_id": job_id,
        "input_sha256": digest(reference),
        "glb_sha256": model_hash,
        "settings_sha256": digest(job / "settings.json"),
        "views": entries,
        "sheets": sheets,
        "appearance": "pending_review",
    }
    (evidence / "evidence.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("views", type=Path)
    args = parser.parse_args()
    make_sheets(args.job, args.reference, args.views)
