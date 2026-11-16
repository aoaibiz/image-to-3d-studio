"""Visual evidence for an unclassified subject. Measurements never approve appearance."""

from hashlib import sha256 as digest
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def _digest(path):
    with path.open("rb") as stream:
        h = digest()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _small(path, mode):
    with Image.open(path) as opened:
        size = opened.size
        image = ImageOps.exif_transpose(opened)
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        return np.array(image.convert(mode)), size


def _clay_mask(rgb):
    # All inspection views have a solid background. Do not assume its colour.
    corners = np.stack((rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1])).astype(int)
    if np.ptp(corners, axis=0).max() > 8:
        raise ValueError("render background is not uniform at the corners")
    # The fixed PNG background is lossless. A high threshold would cut holes
    # into matte grey parts whose shading crosses the background brightness.
    mask = np.abs(rgb.astype(int) - np.median(corners, axis=0)).max(axis=2) > 3
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError("render silhouette reaches image boundary")
    return mask


def _bbox(mask):
    rows, cols = np.nonzero(mask)
    return int(cols.min()), int(rows.min()), int(cols.max() + 1), int(rows.max() + 1)


def _normalise(mask):
    left, top, right, bottom = _bbox(mask)
    scale = 140 / max(right - left, bottom - top)
    width, height = (
        max(1, round((right - left) * scale)),
        max(1, round((bottom - top) * scale)),
    )
    fit = cv2.resize(
        mask[top:bottom, left:right].astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.zeros((180, 180), np.uint8)
    x, y = (180 - width) // 2, (180 - height) // 2
    canvas[y : y + height, x : x + width] = fit
    return canvas.astype(bool)


def _compare_outlines(reference, front):
    source = _normalise(reference)
    model = _normalise(front)
    best = 0
    aligned = model
    alignment = (1, 0, 0)
    # Small translation/scale tolerance; no flipping, deformation or rotation.
    for scale in (0.94, 1.0, 1.06):
        matrix = cv2.getRotationMatrix2D((89.5, 89.5), 0, scale)
        scaled = cv2.warpAffine(
            model.astype(np.uint8), matrix, (180, 180), flags=cv2.INTER_NEAREST
        ).astype(bool)
        for dx in (-6, 0, 6):
            for dy in (-6, 0, 6):
                candidate = np.roll(scaled, (dy, dx), axis=(0, 1))
                overlap = (source & candidate).sum() / (source | candidate).sum()
                if overlap > best:
                    best, aligned, alignment = (
                        float(overlap),
                        candidate,
                        (scale, dx, dy),
                    )
    # Ignore one-pixel resampling discrepancies before identifying unmatched regions.
    fringe = cv2.dilate(aligned.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(
        bool
    )
    missing = source & ~fringe
    _, _, stats, _ = cv2.connectedComponentsWithStats(missing.astype(np.uint8))
    largest = max((int(s[cv2.CC_STAT_AREA]) for s in stats[1:]), default=0)
    largest_share = largest / int(source.sum())
    overlay = np.full((*source.shape, 3), (247, 247, 245), np.uint8)
    overlay[source & aligned] = (103, 112, 104)
    overlay[source & ~aligned] = (194, 55, 112)
    overlay[aligned & ~source] = (45, 126, 178)
    return {
        "iou": round(best, 4),
        "unmatched_input_fraction": round(float(missing.sum() / source.sum()), 4),
        "largest_unmatched_input_component_fraction": round(largest_share, 4),
        "fit_scale": alignment[0],
        "fit_offset_px_in_180_square": list(alignment[1:]),
    }, overlay


def _palette(rgb, mask):
    # Chroma distribution excludes the lightness channel; illumination can still affect it.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[mask]
    bins = lab[:, 1:].astype(int) // 16
    counts = np.bincount(bins[:, 0] * 16 + bins[:, 1], minlength=256)
    return counts / counts.sum()


def _sections(mask, front_bounds):
    # Use the front view's physical heights: the cameras share the same ortho scale.
    _, y0, _, y1 = front_bounds
    edges = np.linspace(y0, y1, 6).astype(int)
    return [float(np.median(mask[a:b].sum(axis=1))) for a, b in zip(edges, edges[1:])]


def _cropped(rgb, mask):
    if not mask.any():
        return rgb
    x0, y0, x1, y1 = _bbox(mask)
    margin = max(2, round(0.05 * max(x1 - x0, y1 - y0)))
    return rgb[max(0, y0 - margin) : y1 + margin, max(0, x0 - margin) : x1 + margin]


def _sheet(path, tiles):
    sheet = Image.new("RGB", (5 * 220, 250), "#efeeeb")
    labels = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(tiles):
        img = Image.fromarray(im)
        if img.mode == "RGBA":
            matte = Image.new("RGB", img.size, "#efeeeb")
            matte.paste(img, mask=img.getchannel("A"))
            img = matte
        img = ImageOps.contain(img, (216, 206), Image.Resampling.LANCZOS)
        x = i * 220
        sheet.paste(img, (x + (220 - img.width) // 2, (210 - img.height) // 2))
        labels.text((x + 6, 211), label, fill="#1a2722")
    labels.text(
        (6, 231),
        "Outline overlay: magenta = input only, blue = model only, grey = overlap",
        fill="#1a2722",
    )
    with NamedTemporaryFile(dir=path.parent, suffix=".jpg", delete=False) as temp:
        tmp = Path(temp.name)
    try:
        sheet.save(tmp, "JPEG", quality=90)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def assess_general(source, output, sha256):
    """Compare a real masked source with hash-associated Studio renders in output."""
    folder = Path(output)
    report = {"status": "unknown", "sha256": sha256, "flags": []}
    try:
        if not isinstance(sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", sha256):
            raise ValueError("invalid artifact digest")
        if _digest(folder / "model.glb") != sha256:
            raise ValueError("model digest mismatch")
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("sha256") != sha256
            or manifest.get("projection") != "orthographic"
            or manifest.get("front") != "+Z"
        ):
            raise ValueError("unmatched render manifest or unsupported camera")
        names = ("front_material", "front_clay", "left_clay", "right_clay")
        images, hashes, sizes = {}, {}, []
        for key in names:
            name = f"{sha256}_{key}.png"
            if (
                not isinstance(manifest.get("files"), list)
                or name not in manifest["files"]
            ):
                raise ValueError("missing render in manifest: " + key)
            images[key], size = _small(folder / name, "RGB")
            sizes.append(size)
            hashes[key] = _digest(folder / name)
        if len(set(sizes)) != 1:
            raise ValueError("inconsistent render dimensions")
        reference, _ = _small(Path(source), "RGBA")
        source_mask = reference[:, :, 3] >= 128
        if source_mask.all() or source_mask.sum() < 20:
            raise ValueError("reference foreground alpha is missing or insufficient")
        if (
            source_mask[0].any()
            or source_mask[-1].any()
            or source_mask[:, 0].any()
            or source_mask[:, -1].any()
        ):
            report["flags"].append("foreground_touches_input_edge_possible_crop")
        alpha = reference[:, :, 3]
        if np.count_nonzero((alpha > 8) & (alpha < 247)) > 0.08 * source_mask.sum():
            report["flags"].append("partially_transparent_input_outline_is_approximate")
        masks = {
            angle: _clay_mask(images[angle + "_clay"])
            for angle in ("front", "left", "right")
        }
        front = masks["front"]
        if front.sum() < 20:
            raise ValueError("no measurable frontal clay silhouette")
        bounds = _bbox(front)
        if bounds[3] - bounds[1] < 10:
            raise ValueError("front silhouette is too short for height sections")
        outlines, overlay = _compare_outlines(source_mask, front)
        color = np.minimum(
            _palette(reference[:, :, :3], source_mask),
            _palette(images["front_material"], front),
        ).sum()
        report.update(
            input_sha256=_digest(Path(source)),
            render_sha256=hashes,
            front_outline=outlines,
            front_chroma_histogram_intersection=round(float(color), 4),
        )
        if color < 0.20:
            report["flags"].append("low_chroma_overlap_check_materials_and_lighting")
        if outlines["largest_unmatched_input_component_fraction"] >= 0.05:
            report["flags"].append(
                "substantial_input_region_unmatched_check_for_missing_parts_or_pose_difference"
            )
        front_widths = _sections(front, bounds)
        sides = {}
        for angle, mask in masks.items():
            widths = _sections(mask, bounds)
            ratios = [
                round(y / x, 3) if x >= 3 else None
                for x, y in zip(front_widths, widths)
            ]
            sides[angle] = {
                "occupied_fraction_of_frame": round(float(mask.mean()), 5),
                "area_relative_to_front": round(float(mask.sum() / front.sum()), 4),
                "occupied_width_relative_to_front_top_to_bottom": ratios,
            }
        report["orthographic_silhouettes"] = sides
        if not masks["left"].any() or not masks["right"].any():
            report["flags"].append("side_silhouette_not_resolved_check_edge_on_view")
        elif all(
            sum(
                r is not None and r < 0.15
                for r in sides[a]["occupied_width_relative_to_front_top_to_bottom"]
            )
            >= 3
            for a in ("left", "right")
        ):
            report["flags"].append("narrow_side_sections_check_intended_subject_depth")
        name = f"{sha256}_appearance-comparison.jpg"
        _sheet(
            folder / name,
            [
                ("Segmented input (inference)", _cropped(reference, source_mask)),
                ("Front material", _cropped(images["front_material"], front)),
                ("Aligned outlines", overlay),
                ("Left clay", _cropped(images["left_clay"], masks["left"])),
                ("Right clay", _cropped(images["right_clay"], masks["right"])),
            ],
        )
        report["contact_sheet"] = name
        report["status"] = (
            "review" if masks["left"].any() and masks["right"].any() else "unknown"
        )
        report["limitations"] = (
            "Image-space evidence only. Lighting, shadows, segmentation and camera angle "
            "can change the metrics. A narrow side may be correct. Compare the actual views."
        )
    except (
        FileNotFoundError,
        ValueError,
        TypeError,
        KeyError,
        Image.UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as exc:
        report["reason"] = str(exc)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("foreground", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("sha256")
    parser.add_argument("--observed-view", choices=("front", "back"))
    args = parser.parse_args()
    try:
        result = assess_general(args.foreground, args.output, args.sha256)
        if args.observed_view:
            from reflected_paint import assess_reflected_paint

            result["opposite_paint"] = assess_reflected_paint(
                args.output / "model.glb", args.output, args.observed_view, args.foreground
            )
    except OSError as error:
        parser.exit(1, f"Unable to read or write appearance evidence: {error}\n")
    print(json.dumps(result, ensure_ascii=True))
