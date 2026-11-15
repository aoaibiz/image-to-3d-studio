"""Screen a final GLB's opposite material renders for reflected distinctive marking.

A positive is a request for human inspection, including when the object really
is decorated on both sides. A negative never certifies unseen appearance.
"""

from hashlib import sha256
from io import BytesIO
import json
import struct
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from quality.general_appearance import _bbox, _clay_mask, _digest


SIZE = 144
CENTER = (SIZE - 1) / 2
OPPOSITE = {"front": "back", "back": "front"}


def _read_frame(path, hashes, mode):
    data = path.read_bytes()
    hashes[path.name] = sha256(data).hexdigest()
    with Image.open(BytesIO(data)) as image:
        if image.width * image.height > 24_000_000:
            raise ValueError("image exceeds inspection budget")
        if mode == "RGBA" and "A" not in image.getbands():
            raise ValueError("source_matte requires an alpha channel")
        return np.asarray(image.convert(mode))


def _fit_pixels(pixels, box, interpolation):
    x0, y0, x1, y1 = box
    return cv2.resize(pixels[y0:y1, x0:x1], (SIZE, SIZE), interpolation=interpolation)


def _features(rgb, mask, clay=None):
    if rgb.shape[:2] != mask.shape or mask.sum() < 256:
        raise ValueError("missing matching material and foreground mask")
    box = _bbox(mask)
    if min(box[2] - box[0], box[3] - box[1]) < 24:
        raise ValueError("insufficient subject sampling")
    inside = _fit_pixels(mask.astype(np.uint8), box, cv2.INTER_NEAREST)
    color = _fit_pixels(rgb.astype(np.float32), box, cv2.INTER_AREA)
    # Normalized convolution prevents a hole/background colour from becoming a motif.
    den = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), 4)
    smooth = cv2.GaussianBlur(color * inside[:, :, None], (0, 0), 4)
    detail = color - smooth / np.maximum(den[:, :, None], 1e-6)
    dist = cv2.distanceTransform(np.pad(inside, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    keep = dist > 6
    if keep.sum() < 300:
        raise ValueError("insufficient interior texels")
    if clay is not None:
        # Form/lighting relief can itself be distinctive on a plain object.
        # Discount the local grey relief shared with the neutral geometry pass.
        clay_px = _fit_pixels(clay.astype(np.float32), box, cv2.INTER_AREA)
        gray = cv2.cvtColor(clay_px, cv2.COLOR_RGB2GRAY)
        smooth_gray = cv2.GaussianBlur(gray * inside, (0, 0), 4) / np.maximum(den, 1e-6)
        relief = gray - smooth_gray
        predictor = relief[keep]
        slope = (detail[keep] * predictor[:, None]).sum(axis=0) / max(
            1, predictor @ predictor
        )
        detail -= relief[:, :, None] * np.clip(slope, -2, 2)
    limit = max(2.0, float(np.quantile(abs(detail[keep]), 0.98)))
    return np.clip(detail, -limit, limit), keep


def _correlation(a, b, mask):
    if mask.sum() < 160:
        return 0.0
    u, v = a[mask], b[mask]
    u, v = u - u.mean(axis=0), v - v.mean(axis=0)
    norm = float(np.sqrt((u * u).sum() * (v * v).sum()))
    return 0.0 if norm < 1 else float((u * v).sum() / norm)


def _align(first, second, mirrored):
    a, ma = first
    b, mb = second
    if mirrored:
        b, mb = b[:, ::-1], mb[:, ::-1]
    best = (-1.0, None, None, None)
    required = max(300, 0.65 * min(ma.sum(), mb.sum()))
    for scale in (0.96, 1.0, 1.04):
        mat = cv2.getRotationMatrix2D((CENTER, CENTER), 0, scale)
        scaled = cv2.warpAffine(b, mat, (SIZE, SIZE))
        valid = cv2.warpAffine(
            mb.astype(np.uint8), mat, (SIZE, SIZE), flags=cv2.INTER_NEAREST
        ).astype(bool)
        for dx in (-5, -2, 0, 2, 5):
            for dy in (-5, -2, 0, 2, 5):
                moved = np.roll(scaled, (dy, dx), axis=(0, 1))
                mask = ma & np.roll(valid, (dy, dx), axis=(0, 1))
                if dx > 0:
                    mask[:, :dx] = False
                elif dx < 0:
                    mask[:, dx:] = False
                if dy > 0:
                    mask[:dy] = False
                elif dy < 0:
                    mask[dy:] = False
                if mask.sum() < required:
                    continue
                score = _correlation(a, moved, mask)
                if score > best[0]:
                    best = (score, (dx, dy, scale), moved, mask)
    return best


def _signal(first):
    a, mask = first
    rms = float(np.sqrt(np.mean(a[mask] ** 2)))
    symmetric = _align(first, first, True)[0]
    dx = cv2.Sobel(a, cv2.CV_32F, 1, 0)[mask].ravel()
    dy = cv2.Sobel(a, cv2.CV_32F, 0, 1)[mask].ravel()
    eig = np.linalg.eigvalsh(np.array([[dx @ dx, dx @ dy], [dx @ dy, dy @ dy]]))
    orientation = float(eig[0] / max(1, eig.sum()))
    repeated = 0.0
    # Compare across appreciable offsets; avoid declaring neighbouring pixels a repeat.
    for axis in (0, 1):
        for shift in range(12, 43):
            valid = mask & np.roll(mask, shift, axis=axis)
            if axis == 0:
                valid[:shift] = False
            else:
                valid[:, :shift] = False
            if valid.sum() >= mask.sum() * 0.5:
                repeated = max(
                    repeated, _correlation(a, np.roll(a, shift, axis=axis), valid)
                )
    return {
        "rms_rgb": rms,
        "self_reflection": symmetric,
        "self_translation": repeated,
        "gradient_minor_fraction": orientation,
    }


def _supported_quadrants(a, b, valid):
    scores = []
    for y in (0, SIZE // 2):
        for x in (0, SIZE // 2):
            roi = np.s_[y : y + SIZE // 2, x : x + SIZE // 2]
            mask = valid[roi]
            if mask.sum() >= 160 and np.sqrt(np.mean(a[roi][mask] ** 2)) >= 3:
                scores.append(_correlation(a[roi], b[roi], mask))
    return scores


def assess_reflected_paint(
    model_path, views_dir, observed_view="front", source_matte=None
):
    """Read-only result. observed_view identifies the registered photographic side.

    An optional foreground RGBA photo provides a separate source-agreement
    measurement. It should be a crop in the same pose as observed_view.
    """
    model, folder = Path(model_path), Path(views_dir)
    result = {
        "status": "inconclusive",
        "manual_review_required": True,
        "observed_view": observed_view,
        "render_sha256": {},
    }
    if not isinstance(observed_view, str) or observed_view not in OPPOSITE:
        return dict(result, reason="unknown_observed_camera")
    try:
        digest = result["glb_sha256"] = _digest(model)
        with model.open("rb") as stream:
            header = stream.read(12)
        if len(header) != 12 or struct.unpack("<4sII", header) != (
            b"glTF",
            2,
            model.stat().st_size,
        ):
            return dict(result, reason="invalid_glb_header")
        raw = (folder / "manifest.json").read_bytes()
        result["manifest_sha256"] = sha256(raw).hexdigest()
        manifest = json.loads(raw)
        if (
            not isinstance(manifest, dict)
            or not isinstance(manifest.get("files"), list)
            or manifest.get("sha256") != digest
            or manifest.get("synthetic") is not False
            or manifest.get("front") != "+Z"
            or manifest.get("projection") != "orthographic"
        ):
            return dict(result, reason="unbound_or_noncanonical_evidence")
        frames = []
        for view in (observed_view, OPPOSITE[observed_view]):
            render = {}
            for mode in ("material", "clay"):
                name = f"{digest}_{view}_{mode}.png"
                if name not in manifest.get("files", []):
                    return dict(result, reason="missing_hash_bound_render")
                render[mode] = _read_frame(
                    folder / name, result["render_sha256"], "RGB"
                )
            frame = _features(
                render["material"], _clay_mask(render["clay"]), render["clay"]
            )
            frames.append(frame)
        first, second = frames
        signal = result["observed_detail"] = _signal(first)
        if source_matte is not None:
            hashes = {}
            rgba = _read_frame(Path(source_matte), hashes, "RGBA")
            photo = _features(rgba[:, :, :3], rgba[:, :, 3] >= 160)
            result["source_sha256"] = next(iter(hashes.values()))
            result["photo_to_observed_detail_agreement"] = _align(first, photo, False)[
                0
            ]
        if signal["rms_rgb"] < 3:
            return dict(result, reason="low_contrast_or_plain_paint")
        if (
            signal["self_reflection"] >= 0.75
            or signal["self_translation"] >= 0.78
            or signal["gradient_minor_fraction"] < 0.055
        ):
            return dict(result, reason="symmetric_periodic_or_directional_markings")
        direct = _align(first, second, False)[0]
        reflected, transform, aligned, interior = _align(first, second, True)
        if aligned is None:
            return dict(result, reason="insufficient_common_interior")
        quadrants = _supported_quadrants(first[0], aligned, interior)
        result.update(
            reflected_correlation=reflected,
            unreflected_correlation=direct,
            alignment_xy_scale=transform,
            matching_quadrant_correlations=quadrants,
        )
        if (
            reflected >= 0.5
            and reflected - direct >= 0.17
            and sum(value >= 0.42 for value in quadrants) >= 2
        ):
            return dict(
                result,
                status="review_hold",
                reason="distinctive_opposite_paint_matches_reflection",
            )
        return dict(
            result, status="no_evidence", reason="no_distinctive_reflection_detected"
        )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        cv2.error,
        Image.DecompressionBombError,
    ):
        return dict(result, reason="invalid_or_incomplete_inputs")
