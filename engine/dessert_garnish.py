"""Reconstruct bounded, upright garnishes attached to the measured dessert."""

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter
from skimage.measure import marching_cubes


def _candidate(mask, roi, peak, diameter, kind):
    # Ignore regions cut by the search window. A long dark background edge
    # can otherwise look like chocolate when the photo contains a table scene.
    x0, y0, x1, y1 = roi
    region = mask[y0:y1, x0:x1].astype("uint8")
    if region.size == 0:
        return None
    size = max(1, round(0.008 * diameter))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * size + 1, 2 * size + 1))
    region = cv2.morphologyEx(region, cv2.MORPH_OPEN, kernel)
    contours = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    found = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (
            min(x, y) < 1
            or x + w >= region.shape[1] - 1
            or y + h >= region.shape[0] - 1
        ):
            continue
        area = cv2.contourArea(contour)
        aspect = w / h
        if kind == "dark":
            if not (
                0.22 * diameter < h < 1.08 * diameter
                and 0.12 < aspect < 0.85
                and 0.20 < area / (w * h) < 0.94
            ):
                continue
            height_from_peak = (y0 + y + h - peak[1]) / diameter
            if not (-0.045 < height_from_peak < 0.35):
                continue
            if abs(x0 + x + w / 2 - peak[0]) > 0.43 * diameter:
                continue
        else:
            # A card has a polygonal silhouette and a contrasting border.
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if not (
                0.18 * diameter < h < 0.61 * diameter
                and 0.37 < aspect < 1.8
                and hull_area > 0
                and 0.12 < area / hull_area < 0.75
            ):
                continue
            vertices = cv2.approxPolyDP(hull, 0.025 * cv2.arcLength(hull, True), True)
            if not 3 <= len(vertices) <= 8:
                continue
        found.append((area, contour))
    if not found:
        return None
    contour = max(found, key=lambda item: item[0])[1]
    contour = contour + np.asarray([x0, y0], dtype=contour.dtype)
    return contour


def _component_mask(mask, contour, kind):
    x, y, w, h = cv2.boundingRect(contour)
    outline = np.zeros((h, w), dtype="uint8")
    contour = contour - [x, y]
    cv2.drawContours(
        outline,
        [
            cv2.convexHull(contour.astype("int32"))
            if kind == "card"
            else contour.astype("int32")
        ],
        0,
        1,
        cv2.FILLED,
    )
    observed = outline.astype(bool) & mask[y : y + h, x : x + w]
    if kind == "card":
        return outline.astype(bool), observed
    # Keep openings through the chocolate. Small isolated highlights inside
    # a dark, closed region are paint and should stay on its front face.
    body = observed.astype("uint8")
    closed = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((3, 3), "uint8"))
    enclosed = binary_fill_holes(closed) & ~closed.astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(enclosed.astype("uint8"))
    limit = max(12, 0.045 * max(w, h) ** 2)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < limit:
            closed[labels == k] = 1
    # Contours filled in 2D can enclose unconnected specks. Only retain
    # the largest solid island before creating the equivalent beveled candy.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    if len(stats) > 1:
        closed = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.asarray(closed, bool), observed


def _plate(scene, kind, rgb, mask, contour, center, rim, scale, depth):
    x, y, w, h = cv2.boundingRect(contour)
    solid, observed = _component_mask(mask, contour, kind)
    if not observed.any() or not solid.any():
        return False
    pixels = rgb[y : y + h, x : x + w]
    color = np.quantile(pixels[observed], 0.56, axis=0)
    pigment = np.clip(color * (1.13 if kind == "dark" else 1.04), 0, 255)
    atlas = np.zeros((h, w * 2, 3), dtype="uint8")
    atlas[:] = pigment
    if kind == "dark":
        light = solid & ~observed & (pixels.mean(axis=2) > color.mean() + 30)
        atlas[:, :w][light] = np.clip(
            color * 0.25 + np.array([205, 194, 155]) * 0.75, 0, 255
        )
    else:
        distance = cv2.distanceTransform(
            np.pad(solid.astype("uint8"), 1), cv2.DIST_L2, 5
        )[1:-1, 1:-1]
        paper = distance > max(2, min(w, h) * 0.092)
        # Segment only small print strokes in the connected interior; no raw
        # source pixels or background patches are used in the model.
        atlas[:, :w][paper] = [244, 226, 192]
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        local = cv2.morphologyEx(
            gray,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        stroke = (local > 12) & (distance > max(3, min(w, h) * 0.16))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(stroke.astype("uint8"))
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if 3 <= area <= max(5, solid.sum() * 0.05):
                atlas[:, :w][labels == i] = [158, 116, 78]
    # March the signed distance of a rounded solid. An edible flat garnish
    # has a thin curved edge but the rest of the dessert is fully volumetric.
    binary = np.pad(solid, 4)
    sdf = distance_transform_edt(binary) - distance_transform_edt(~binary)
    if kind == "dark":
        sdf = gaussian_filter(sdf, 0.72)
    half = max(1.45, min(w, h) * (0.049 if kind == "dark" else 0.023))
    n = int(np.ceil(half)) + 3
    z = np.arange(-n, n + 1)
    field = np.minimum(sdf[:, :, None], half - np.abs(z)[None, None, :])
    vertices, faces, _, _ = marching_cubes(
        field.astype("float32"), level=0.23, step_size=1
    )
    row, col, side = vertices.T
    row = row - 4
    col = col - 4
    wx = (x + col - center) / scale
    wy = 1 + (rim - y - row) / scale
    wz = depth + (side - n) / scale
    # A modest angle makes both the folded thickness and silhouette visible.
    wz -= 0.085 * (wx - np.mean(wx))
    mesh = trimesh.Trimesh(
        vertices=np.column_stack((wx, wy, wz)), faces=faces, process=False
    )
    if mesh.volume < 0:
        mesh.invert()
    front = mesh.face_normals[:, 2] > 0.28
    ids, inverse = np.unique(
        mesh.faces + (~front)[:, None] * len(vertices), return_inverse=True
    )
    source = ids % len(vertices)
    u = (
        np.clip(col[source] / max(1, w - 1), 0.006, 0.994) + (ids // len(vertices))
    ) * 0.5
    v = 1 - np.clip(row[source] / max(1, h - 1), 0.006, 0.994)
    name = "raised_" + kind + "_decoration"
    material = trimesh.visual.material.PBRMaterial(
        name=name,
        baseColorTexture=Image.fromarray(atlas),
        metallicFactor=0.0,
        roughnessFactor=0.50 if kind == "dark" else 0.67,
    )
    part = trimesh.Trimesh(
        vertices=mesh.vertices[source],
        faces=inverse.reshape(-1, 3),
        vertex_normals=mesh.vertex_normals[source],
        process=False,
        visual=trimesh.visual.TextureVisuals(
            uv=np.column_stack((u, v)), material=material
        ),
    )
    scene.add_geometry(part, node_name=name, geom_name=name)
    return True


def add_measured_decor(scene, measurements, image, cream_radius):
    analysis = image.copy()
    analysis.thumbnail((720, 720), Image.Resampling.LANCZOS)
    rgb = np.asarray(analysis.convert("RGB"))
    h, w = rgb.shape[:2]
    valid = np.asarray(analysis.convert("RGBA"))[:, :, 3] > 160
    left, top, right, base = np.array(measurements["cream_bbox"]) * [w, h, w, h]
    diameter = right - left
    if diameter < 25:
        return []
    cream_x = (left + right) * 0.5
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = np.moveaxis(hsv, -1, 0)

    def bounds(x0, y0, x1, y1):
        return (
            max(0, int(x0)),
            max(0, int(y0)),
            min(w, int(np.ceil(x1))),
            min(h, int(np.ceil(y1))),
        )

    brown_limit = np.clip(max(measurements["cream_rgb"]) * 0.34, 58, 105)
    dark = (hue >= 4) & (hue <= 23) & (sat > 103) & (val < brown_limit) & valid
    rim_x, rim_y, _ = measurements["rim"]
    scale = (measurements["foot"][1] - rim_y) * h
    search = bounds(
        cream_x - 0.58 * diameter,
        top - 1.10 * diameter,
        cream_x + 0.58 * diameter,
        top + 0.42 * diameter,
    )
    # Test nearby exposure cutoffs: below the chocolate midtone only a stem
    # may be visible; above it an adjacent dark fruit can join the contour.
    candidates = []
    for threshold in brown_limit * np.array([0.92, 0.97, 1.0, 1.025, 1.06, 1.12]):
        mask = (hue >= 4) & (hue <= 23) & (sat > 103) & (val < threshold) & valid
        contour = _candidate(mask, search, (cream_x, top), diameter, "dark")
        if contour is not None:
            candidates.append((contour, mask))
    dark_contour = None
    if candidates:
        tallest = max(cv2.boundingRect(c)[3] for c, _ in candidates)
        full = [
            (c, mask)
            for c, mask in candidates
            if cv2.boundingRect(c)[3] >= 0.92 * tallest
        ]
        dark_contour, dark = min(full, key=lambda item: cv2.boundingRect(item[0])[2])
    parts = []
    if dark_contour is not None:
        # Include genuine midtone portions next to the anchored dark core.
        # A narrow search band prevents an adjoining background fruit from
        # merging with it when the photograph has warm cafe lighting.
        core = np.zeros_like(dark, dtype="uint8")
        cv2.drawContours(core, [dark_contour], 0, 1, cv2.FILLED)
        margin = max(2, round(diameter * 0.025))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)
        )
        neighborhood = cv2.dilate(core, kernel) > 0
        loose = (
            (hue >= 4) & (hue <= 23) & (sat > 85) & (val < brown_limit * 1.53) & valid
        )
        dark = neighborhood & loose
        filled = cv2.morphologyEx(
            dark.astype("uint8"), cv2.MORPH_CLOSE, np.ones((3, 3), "uint8")
        )
        contours = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
            0
        ]
        if contours:
            dark_contour = max(contours, key=cv2.contourArea)
    if dark_contour is not None and _plate(
        scene,
        "dark",
        rgb,
        dark,
        dark_contour,
        rim_x * w,
        rim_y * h,
        scale,
        -0.14 * cream_radius,
    ):
        parts.append("dark raised garnish")
    # Border colors include stamps/berries too, so a dark interior or a solid
    # single-color blob cannot pass the paper-card test.
    card = ((hue < 7) | (hue > 158)) & (sat > 65) & (val > 133) & valid
    search = bounds(
        left - 0.18 * diameter,
        top - 0.21 * diameter,
        right + 0.30 * diameter,
        top + 0.64 * diameter,
    )
    c = _candidate(card, search, (cream_x, top), diameter, "card")
    if c is not None:
        x, y, cw, ch = cv2.boundingRect(c)
        solid, pigment = _component_mask(card, c, "card")
        content = solid & ~pigment
        visible = content & valid[y : y + ch, x : x + cw]
        if (
            visible.sum() > 0.8 * content.sum()
            and visible.any()
            and np.median(val[y : y + ch, x : x + cw][visible]) > 151
        ):
            if _plate(
                scene,
                "card",
                rgb,
                card,
                c,
                rim_x * w,
                rim_y * h,
                scale,
                0.57 * cream_radius,
            ):
                parts.append("bordered pastry card")
    return parts
