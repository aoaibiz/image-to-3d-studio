"""Measure berry-topped, layered desserts from visible colour and tapered glass edges.

Landmarks are normalized to the original oriented image: x and horizontal radii
divide by image width, y and vertical radii by height. Hidden surfaces and the
identity of an ingredient cannot be established by this detector.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def _blobs(rgb, hsv, alpha, dark=False):
    h, s, v = cv2.split(hsv)
    if dark:
        mask = (v < 77) & (s < 154)
    else:
        mask = ((h < 8) | (h > 170)) & (s > 144) & (v > 54)
    mask &= alpha
    closed = cv2.morphologyEx(
        mask.astype("uint8"), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    height, width = mask.shape
    blobs = []
    for label in range(1, count):
        x, y, w, hg, area = (int(t) for t in stats[label])
        if (
            area < max(16, 0.0008 * width * height)
            or area > 0.075 * width * height
            or w < 0.4 * hg
            or w > 1.65 * hg
            or min(x, y) < 1
            or x + w >= width
            or y + hg >= height
            or area / (w * hg) < 0.43
        ):
            continue
        component = labels[y : y + hg, x : x + w] == label
        contour = cv2.findContours(
            component.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0][0]
        if area < 0.70 * cv2.contourArea(cv2.convexHull(contour)):
            continue
        pixels = rgb[y : y + hg, x : x + w][component & mask[y : y + hg, x : x + w]]
        if not len(pixels):
            continue
        blobs.append(
            (
                x + w / 2,
                y + hg / 2,
                w / 2,
                hg / 2,
                np.median(pixels, axis=0).astype(int).tolist(),
                area,
            )
        )
    return sorted(blobs, key=lambda b: -b[-1])


def _pair(red, dark):
    candidates = []
    for i, a in enumerate(red[:16]):
        for b in red[i + 1 : 16]:
            scale = np.sqrt(a[2] * b[2]) * 2
            gap = abs(a[0] - b[0]) / scale
            if not (
                1.5 < gap < 5.2
                and 0.46 < b[2] / a[2] < 1.6
                and abs(a[1] - b[1]) < max(a[3], b[3]) * 1.05
            ):
                continue
            left, right = sorted((a, b), key=lambda e: e[0])
            cy = (a[1] + b[1]) / 2
            nearby = [
                d
                for d in dark
                if left[0] - 1.25 * scale < d[0] < right[0] + scale
                and abs(d[1] - cy) < scale * 0.92
                and 0.18 * scale < d[2] < 0.68 * scale
            ]
            if not any(left[0] < d[0] < right[0] for d in nearby):
                continue
            candidates.append((a[-1] + b[-1], left, right, nearby))
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c[0])
    # Overlapping fruit pairs may describe the same arrangement. Two separate
    # equally sized arrangements are ambiguous rather than a default selection.
    best = candidates[0]
    for other in candidates[1:]:
        if other[0] > 0.77 * best[0] and (
            abs(other[1][1] - best[1][1]) > 2 * best[1][3]
            or abs(other[1][0] - best[1][0]) > best[2][0] - best[1][0]
        ):
            return False
    return best[1:]


def _glass_edges(rgb, alpha, cx0, rim0, radius0, target_height):
    height, width = alpha.shape
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype("float32")
    blurred = cv2.GaussianBlur(lab, (5, 5), 1.0)
    gx = np.linalg.norm(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3), axis=2)
    gy = np.linalg.norm(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3), axis=2)
    # Restrict orientation: an oblique table line alone is poor evidence for
    # two upright glass walls. A cutout boundary counts as direct evidence.
    edge = np.minimum(gx, 120) / (28 + np.minimum(gy, 95))
    if not alpha.all():
        edge += 2.5 * np.abs(cv2.Sobel(alpha.astype("float32"), cv2.CV_32F, 1, 0))
    edge = cv2.GaussianBlur(edge, (5, 3), 0.9)
    best = None
    t = np.linspace(0.06, 0.94, 31)
    for rim in np.linspace(rim0 - 0.12 * radius0, rim0 + 0.12 * radius0, 3):
        for depth in np.linspace(0.78, 1.28, 7) * target_height:
            if rim < 0 or rim + depth >= height - 2:
                continue
            rows = np.rint(rim + t * depth).astype(int)
            for foot_ratio in (0.25, 0.40, 0.55):
                profile = foot_ratio + (1 - foot_ratio) * (1 - t) ** 1.4
                # Score a family of tapered outlines against the actual edges.
                centers = np.linspace(cx0 - 0.24 * radius0, cx0 + 0.24 * radius0, 11)[
                    :, None, None
                ]
                radii = radius0 * np.linspace(0.76, 1.20, 10)[None, :, None]
                left_x = np.rint(centers - radii * profile).astype(int)
                right_x = np.rint(centers + radii * profile).astype(int)
                if left_x.min() < 0 or right_x.max() >= width:
                    continue
                left = edge[rows, left_x]
                right = edge[rows, right_x]
                support = np.minimum(
                    left.mean(axis=2), right.mean(axis=2)
                ) + 0.25 * np.minimum(left, right).mean(axis=2)
                # Broad blank background with superimposed table edges does
                # not supply a textured layered filling at this position.
                support -= 0.12 * np.abs(centers[:, :, 0] - cx0) / radius0
                i, j = np.unravel_index(np.argmax(support), support.shape)
                score = float(support[i, j])
                if best is None or score > best[0]:
                    best = (
                        score,
                        float(centers[i, 0, 0]),
                        float(rim),
                        float(depth),
                        float(radii[0, j, 0]),
                        foot_ratio,
                    )
    if best is None or best[0] < 0.31:
        return None
    _, cx, rim, depth, rr, ratio = best
    # A thick glass heel often creates a final transverse highlight/shadow below
    # the last coloured band. Snap to a coherent edge across the centre.
    foot = int(round(rim + depth))
    inner = max(2, round(0.48 * rr * ratio))
    start = max(int(rim + depth * 0.93), 2)
    stop = min(height - 3, int(foot + 0.40 * depth))
    luma = blurred[:, :, 0]
    xx = int(round(cx))
    if start < stop:
        delta = (
            luma[start - 2 : stop - 2, xx - inner : xx + inner + 1]
            - luma[start + 2 : stop + 2, xx - inner : xx + inner + 1]
        )
        strength = np.quantile(delta, 0.32, axis=1)
        step = int(np.argmax(strength))
        if strength[step] > 18:
            depth = start + step - rim
    return cx, rim, depth, rr, ratio


def _cream(rgb, hsv, alpha, group, cx, span, single):
    h, w = alpha.shape
    centers = [b[1] for b in group]
    middle = float(np.median(centers))
    scale = np.median([b[3] for b in group]) * 2
    yl = max(0, int(middle - (1.22 if single else 0.94) * span))
    yh = min(h, int(middle + (0.79 * scale if single else 0.05 * span)))
    xl = max(0, int(cx - 0.41 * span))
    xh = min(w, int(cx + 0.41 * span))
    if xh - xl < 8 or yh - yl < 12:
        return None
    area = rgb[yl:yh, xl:xh].astype(int)
    # Restrict to highlight flesh and exclude deeply stained fruits/crumble.
    sat = hsv[yl:yh, xl:xh, 1]
    bright = (
        (area[:, :, 0] > 168)
        & (area[:, :, 1] > 116)
        & (area[:, :, 2] > 70)
        & (sat < 145)
        & alpha[yl:yh, xl:xh]
    )
    xx = np.arange(xl, xh)
    core = abs(xx - cx) < 0.20 * span
    support = gaussian_filter1d(bright[:, core].mean(axis=1), max(1, 0.018 * span))
    if support.max() < 0.24:
        return None
    strong = support > max(0.20, 0.28 * support.max())
    runs = np.split(
        np.flatnonzero(strong), np.flatnonzero(np.diff(np.flatnonzero(strong)) > 1) + 1
    )
    candidates = [
        r for r in runs if len(r) > 0.15 * span and r[-1] > len(support) * 0.40
    ]
    if not candidates:
        return None
    run = max(candidates, key=len)
    top = yl + int(run[0])
    bot = min(
        h - 1, int(max(yl + run[-1], middle + (0.27 if single else 0.47) * scale))
    )
    if bot - top < 0.18 * span:
        return None
    yy = np.arange(yl, yh)[:, None]
    pixels = area[bright & (yy >= top) & core[None, :]]
    if len(pixels) < 20:
        return None
    color = np.quantile(pixels, 0.72, axis=0)
    return [xl, top, xh, bot], color


def _singleton(rgb, hsv, alpha, red):
    # The lone strawberry may lean outside the rim. Find the contiguous bright
    # cream adjoining that fruit, then measure the narrowing vessel below it.
    candidates = []
    for berry in red[:6]:
        x, y, rx, ry, _, _ = berry
        if y > 0.53 * len(rgb) or ry < 0.029 * len(rgb):
            continue
        yy, xx = np.indices(alpha.shape)
        white = (hsv[:, :, 2] > 167) & (hsv[:, :, 1] < 110) & alpha
        local = (abs(xx - x) < 3.3 * rx) & (yy > y - 1.35 * ry) & (yy < y + 2.3 * ry)
        cs = cv2.findContours(
            cv2.morphologyEx(
                (white & local).astype("uint8"),
                cv2.MORPH_CLOSE,
                np.ones((5, 5), np.uint8),
            ),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )[0]
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(c)
        if not (
            0.85 * rx < bw < 4.5 * rx
            and bh > ry * 0.68
            and cv2.contourArea(c) > 0.55 * rx * ry
        ):
            continue
        cx = bx + bw * 0.5
        if abs(cx - x) > 2.6 * rx:
            continue
        # Locate the widest coherent cream row above the taper, rather than
        # assuming the bottom of a connected white region is the rim.
        cream_rows = white[by : by + bh, bx : bx + bw]
        spans = gaussian_filter1d(
            cream_rows.sum(axis=1).astype(float), max(1, 0.035 * bh)
        )
        allowed = (by + np.arange(bh) >= y) & (by + np.arange(bh) < y + 1.6 * ry)
        if not allowed.any():
            continue
        widest = np.flatnonzero(allowed & (spans > 0.95 * spans[allowed].max()))
        if not len(widest):
            continue
        rim_y = by + float(np.median(widest))
        # A glass with a conspicuous stem or base may extend below the edible
        # material. Its stem geometry remains an estimate from one view.
        proposal = ([berry], [], cx, rim_y, bw / 2, bw * 2.0, bw * 1.48)
        if candidates:
            first = candidates[0]
            if berry[-1] > 0.77 * first[0][0][-1] and (
                abs(cx - first[2]) > bw or abs(rim_y - first[3]) > bw
            ):
                return None
        candidates.append(proposal)
    return candidates[0] if candidates else None


def _analyze_layers(rgb, glass, warm_reference):
    cx, rim, depth, rr, ratio = glass
    f = np.linspace(0.045, 0.98, 128)
    y = np.rint(rim + (1 - f) * depth).astype(int)
    radius = rr * (ratio + (1 - ratio) * f**1.4)
    x = np.rint(cx + radius[:, None] * np.linspace(-0.49, 0.49, 29)).astype(int)
    profile = np.median(rgb[y[:, None], x], axis=1)
    profile = gaussian_filter1d(profile, 1.1, axis=0)
    lab = cv2.cvtColor(profile[None, :, :].astype("uint8"), cv2.COLOR_RGB2LAB)[
        0
    ].astype(float)
    grad = np.linalg.norm(lab[2:] - lab[:-2], axis=1)
    peaks = find_peaks(grad, distance=6, prominence=2.4, height=4.5)[0] + 1
    if len(peaks) < 2:
        return None
    if len(peaks) > 10:
        peaks = np.sort(peaks[np.argsort(grad[peaks - 1])[-10:]])
    ends = [0] + [int(p) for p in peaks if 4 < p < len(f) - 4] + [len(f)]
    if len(ends) < 4:
        return None
    layers = []
    # Neutralize illumination from a bright diffuse cream sample. Limit gain:
    # added brightness should not erase the original dark chocolate/berry hues.
    ref = np.maximum(warm_reference, 1)
    gains = np.clip(np.array([244.0, 232.0, 211.0]) / ref, 0.94, 1.75)
    if ref[0] > 1.25 * ref[2]:
        # Recover some light lost in the glass using its own diffuse, light
        # samples. Mixing this illuminant with the cream highlight protects
        # scenes whose lowest layer is naturally caramel rather than white.
        light = profile[
            np.sum(profile, axis=1) >= np.quantile(np.sum(profile, axis=1), 0.79)
        ]
        local = np.quantile(light, 0.77, axis=0)
        under_glass = np.clip([242, 231, 205] / np.maximum(local, 1), 1.0, 2.55)
        gains = 0.40 * gains + 0.60 * under_glass
    for start, end in zip(ends, ends[1:]):
        pix = profile[start:end]
        raw = np.median(pix, axis=0)
        color = np.clip(raw * gains, 0, 255)
        # The material is seen through glass; compensate the photographed dark
        # exposure locally but retain relative hue, especially blue/red layers.
        color = 255 * (color / 255) ** 0.70
        hh, ss, vv = cv2.cvtColor(np.uint8([[raw]]), cv2.COLOR_RGB2HSV)[0, 0]
        if (
            ref[0] > 1.25 * ref[2]
            and 16.5 <= hh <= 35
            and ss > 161
            and raw[1] > 0.652 * raw[0]
        ):
            # Amber ambient light shifts a saturated yellow-green band toward
            # olive. Estimate a nut-green albedo only for this observed pigment.
            color = np.clip(color * [0.96, 1.16, 0.96] + [2, 7, 13], 0, 255)
        host_from_sides = False
        mid = (start + end - 1) // 2
        if f[mid] > 0.45 and end - start >= 8:
            # A cut berry interrupts the mousse locally. Turning its dark
            # centre into a 360-degree filling band would create a false stripe.
            # Recover the continuous host only if the same rows show a much
            # brighter, less saturated ingredient against the lateral walls.
            row_ids = np.arange(start, end)
            side_x = np.rint(
                cx
                + radius[row_ids, None]
                * np.r_[np.linspace(-0.83, -0.42, 14), np.linspace(0.42, 0.83, 14)]
            ).astype(int)
            sides = rgb[y[row_ids, None], side_x].reshape(-1, 3).astype(float)
            light = sides[
                np.sum(sides, axis=1) >= np.quantile(np.sum(sides, axis=1), 0.79)
            ]
            side_rgb = np.median(light, axis=0)
            if (
                side_rgb[1] - raw[1] > 38
                and side_rgb[2] - raw[2] > 29
                and raw[1] < 0.68 * raw[0]
            ):
                diffuse = np.clip(side_rgb * gains, 0, 255)
                color = 255 * (diffuse / 255) ** 0.58
                if ref[0] > 1.25 * ref[2]:
                    color = 0.66 * color + 0.34 * np.array([244, 228, 194])
                host_from_sides = True
        layers.append(
            {
                "bottom": 0.0 if start == 0 else float(f[start]),
                "top": 1.0 if end == len(f) else float(f[end]),
                "rgb": np.rint(color).astype(int).tolist(),
                "raw_rgb": np.rint(raw).astype(int).tolist(),
                "host_sample": "lateral mousse" if host_from_sides else "center",
            }
        )
    return layers


def _repeated_subject(rgb, bbox):
    # Similar-size competing cups may be missed at different threshold edges.
    # Compare the actual topping and vessel against other nonoverlapping regions.
    # This is an in-image ambiguity check, never a stored template lookup.
    x0, y0, x1, y1 = np.rint(bbox).astype(int)
    h, w = rgb.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 12 or y1 - y0 < 24:
        return True
    if w < 1.65 * (x1 - x0) and h < 1.65 * (y1 - y0):
        return False
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    target = gray[y0:y1, x0:x1]
    if target.std() < 9:
        return False
    scores = cv2.matchTemplate(gray, target, cv2.TM_CCOEFF_NORMED)
    yy, xx = np.indices(scores.shape)
    disjoint = (np.abs(xx - x0) > 0.85 * (x1 - x0)) | (
        np.abs(yy - y0) > 0.85 * (y1 - y0)
    )
    return bool(np.any(scores[disjoint] > 0.89))


def measure_dessert(image):
    """Return image-measured normalized landmarks, or None if evidence is weak."""
    if not isinstance(image, Image.Image):
        return None
    oriented = ImageOps.exif_transpose(image) if image.getexif().get(274, 1) != 1 else image
    if min(oriented.size) < 48:
        return None
    scale = min(1.0, 640 / max(oriented.size))
    size = tuple(max(1, round(n * scale)) for n in oriented.size)
    thumb = oriented.resize(size, Image.Resampling.LANCZOS) if size != oriented.size else oriented
    data = np.asarray(thumb.convert("RGBA"))
    rgb = np.ascontiguousarray(data[:, :, :3])
    alpha = data[:, :, 3] > 128
    h, w = alpha.shape
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = _blobs(rgb, hsv, alpha)
    if not red:
        return None
    dark = _blobs(rgb, hsv, alpha, True)
    pair = _pair(red, dark)
    if pair is False:
        return None
    if pair:
        left, right, blues = pair
        reds = [left, right]
        fruits = reds + blues
        x0 = min(b[0] - b[2] for b in fruits)
        x1 = max(b[0] + b[2] for b in fruits)
        span = x1 - x0
        cx = (x0 + x1) / 2
        rim0 = max(b[1] + b[3] for b in fruits) + 0.125 * span
        rr0 = 0.275 * span
        depth0 = 0.77 * span
    else:
        evidence = _singleton(rgb, hsv, alpha, red)
        if not evidence:
            return None
        reds, blues, cx, rim0, rr0, depth0, span = evidence
    cream = _cream(rgb, hsv, alpha, reds, cx, span, not pair)
    if not cream:
        return None
    cream_box, color = cream
    if not pair:
        cream_box[3] = int(np.clip(rim0, cream_box[1] + 1, h - 1))
    glass = _glass_edges(rgb, alpha, cx, rim0, rr0, depth0)
    if glass is None and pair:
        # Some toppings sit just inside a wide rim; others (a flake collar)
        # overhang a narrower vessel. Demand paired edge support for either.
        glass = _glass_edges(rgb, alpha, cx, rim0, rr0 * 1.63, depth0 * 1.64)
    if glass is None:
        return None
    layers = _analyze_layers(rgb, glass, color)
    if not layers:
        return None
    gc, gy, gd, gr, ratio = glass
    if min(gr, gd) < 10 or gd < gr * 1.45:
        return None
    collar = None
    if pair:
        row = int(np.clip((max(b[1] + b[3] for b in reds + blues) + gy) / 2, 1, h - 2))
        strip = rgb[
            row - 1 : row + 2,
            max(0, int(cx - 0.45 * span)) : min(w, int(cx + 0.45 * span)),
        ]
        collar = {
            "cx": float(cx / w),
            "y": float(row / h),
            "half_width": float(0.55 * span / w),
            "rgb": np.clip(
                np.quantile(strip.reshape(-1, 3), 0.72, axis=0)
                * np.clip([250, 239, 214] / np.maximum(color, 1), 0.95, 1.65),
                0,
                255,
            )
            .astype(int)
            .tolist(),
        }
    bbox = [
        min(cream_box[0], gc - gr, min(b[0] - b[2] for b in reds + blues)),
        min(cream_box[1], min(b[1] - b[3] for b in reds + blues)),
        max(cream_box[2], gc + gr, max(b[0] + b[2] for b in reds + blues)),
        gy + gd,
    ]
    if _repeated_subject(rgb, bbox):
        return None
    unit = np.asarray([w, h, w, h])
    result = {
        "bbox": (np.clip(np.asarray(bbox) / unit, 0, 1)).tolist(),
        "rim": [gc / w, gy / h, gr / w],
        "foot": [gc / w, (gy + gd) / h, gr * ratio / w],
        "cream_bbox": (np.array(cream_box) / unit).tolist(),
        "image_size": list(oriented.size),
        "cream_rgb": np.rint(color).astype(int).tolist(),
        "berries": [
            {
                "kind": kind,
                "center": [b[0] / w, b[1] / h],
                "radii": [b[2] / w, b[3] / h],
                "rgb": b[4],
            }
            for kind, group in [("red", reds), ("dark", blues)]
            for b in group
        ],
        "layers": layers,
        "glass_profile": [
            [float(t), float(gr * (ratio + (1 - ratio) * t**1.4) / w)]
            for t in np.linspace(0, 1, 11)
        ],
        "garnish": [],
        "measurement_method": "Visible berry and cream landmarks, paired glass edges, robust row-wise pigment; hidden surfaces inferred.",
    }
    if collar is not None:
        result["collar"] = collar
    return result
