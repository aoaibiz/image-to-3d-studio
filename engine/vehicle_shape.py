"""Conservative side/three-quarter toy-car scaffold from explicit RGBA pixels.

This is a geometric heuristic, not a semantic classifier. It returns None when
its narrow visual evidence is missing, and cannot determine hidden surfaces.
"""

from __future__ import annotations

import cv2
import numpy as np
import trimesh
from PIL import Image
from skimage.measure import marching_cubes

MAX_SIDE = 512


def _features(image):
    if not isinstance(image, Image.Image):
        return None
    rgba = image.convert("RGBA")
    rgba.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    data = np.asarray(rgba)
    alpha = data[:, :, 3] > 160
    if alpha.sum() < 100 or alpha.mean() > 0.94:
        return None
    ys, xs = np.where(alpha)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    w, h = x1 - x0 + 1, y1 - y0 + 1
    if w < 80 or h < 50 or not 1.25 <= w / h <= 3.6:
        return None
    gray = cv2.cvtColor(data[:, :, :3], cv2.COLOR_RGB2GRAY)
    pixels = gray[alpha]
    threshold = int(np.clip(np.percentile(pixels, 35) * 0.72, 32, 100))
    binary = ((gray < threshold) & alpha).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    wheels, windows = [], []
    for contour in contours:
        if len(contour) < 5:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        area = abs(cv2.contourArea(contour))
        cx, cy = x + bw / 2, y + bh / 2
        if (
            y0 + 0.58 * h < cy < y1
            and 0.035 * w < bw < 0.13 * w
            and 0.06 * h < bh < 0.24 * h
            and 0.50 < bw / bh < 1.35
            and area / (bw * bh) > 0.5
        ):
            wheels.append((cx, cy, bw, bh, area))
        if (
            y0 < cy < y0 + 0.49 * h
            and 0.10 * w < bw < 0.30 * w
            and 0.15 * h < bh < 0.45 * h
            and area / (bw * bh) > 0.38
        ):
            windows.append((x, y, bw, bh, area, contour))
    pairs = []
    for i, a in enumerate(wheels):
        for b in wheels[i + 1 :]:
            a, b = sorted((a, b))
            if (
                0.27 * w < b[0] - a[0] < 0.70 * w
                and abs(a[1] - b[1]) < 0.075 * h
                and 0.65 < a[2] / b[2] < 1.55
                and 0.65 < a[3] / b[3] < 1.55
            ):
                pairs.append((a, b))
    if not pairs or len(windows) < 2:
        return None
    window_centers = sorted(x + bw / 2 for x, _, bw, _, _, _ in windows)
    if window_centers[-1] - window_centers[0] < 0.12 * w:
        return None
    left, right = max(pairs, key=lambda pair: pair[1][0] - pair[0][0])
    # Contour nesting may produce two outlines for one window; use their union.
    wx0 = min(a[0] for a in windows)
    wx1 = max(a[0] + a[2] for a in windows)
    wy0 = min(a[1] for a in windows)
    wy1 = max(a[1] + a[3] for a in windows)
    if not (
        x0 + 0.1 * w < wx0 < wx1 < x1 - 0.08 * w and 0.18 * w < wx1 - wx0 < 0.72 * w
    ):
        return None
    split = (wx0 + wx1) / 2
    groups = [
        [a for a in windows if (a[0] + a[2] / 2 < split) == side]
        for side in (True, False)
    ]
    if any(not group for group in groups):
        return None
    window_boxes = [
        (
            min(a[0] for a in g),
            min(a[1] for a in g),
            max(a[0] + a[2] for a in g),
            max(a[1] + a[3] for a in g),
        )
        for g in groups
    ]
    polygons = []
    for group in groups:
        contour = max(group, key=lambda a: a[4])[5]
        hull = cv2.convexHull(contour)
        poly = cv2.approxPolyDP(hull, 0.025 * cv2.arcLength(hull, True), True).reshape(
            -1, 2
        )
        if len(poly) < 3:
            return None
        polygons.append(poly.tolist())
    return (
        data,
        (x0, y0, w, h),
        (left, right),
        (wx0, wy0, wx1, wy1),
        window_boxes,
        polygons,
    )


def _rounded_box(extents, radius):
    half = np.asarray(extents) / 2
    step = min(extents) / 28
    axes = [np.arange(-v - step, v + 2 * step, step) for v in half]
    x, y, z = np.meshgrid(*axes, indexing="ij", sparse=True)
    q = [np.abs(c) - (v - radius) for c, v in zip((x, y, z), half)]
    d = (
        np.sqrt(sum(np.maximum(v, 0) ** 2 for v in q))
        + np.minimum(np.maximum(np.maximum(q[0], q[1]), q[2]), 0)
        - radius
    )
    verts, faces, _, _ = marching_cubes(
        d.astype(np.float32), 0, spacing=(step,) * 3, allow_degenerate=False
    )
    verts += np.array([a[0] for a in axes])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh, multibody=True)
    return mesh


def _material(mesh, color, texture=None):
    if texture is None:
        texture = Image.new("RGB", (8, 8), tuple(int(x) for x in color[:3]))
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture,
        baseColorFactor=[255] * 4,
        metallicFactor=0,
        roughnessFactor=0.65,
    )
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.full((len(mesh.vertices), 2), 0.5), material=mat
    )
    return mesh


def _sample_body_patch(pixels, bbox):
    x0, y0, w, h = bbox
    region = pixels[
        int(y0 + 0.48 * h) : int(y0 + 0.74 * h), int(x0 + 0.1 * w) : int(x0 + 0.9 * w)
    ]
    rgb = region[:, :, :3].astype(int)
    alpha = region[:, :, 3] > 200
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    eligible = alpha & (rgb.mean(axis=2) > 65) & (chroma > 28)
    if not np.any(eligible):
        eligible = alpha & (rgb.mean(axis=2) > 65)
    if not np.any(eligible):
        return None
    target = np.median(rgb[eligible], axis=0)
    delta = np.linalg.norm(rgb - target, axis=2)
    delta[~eligible] = 1e6
    y, x = np.unravel_index(np.argmin(delta), delta.shape)
    # Actual 8x8 input pixels form a visible PBR texture, rather than painting
    # source wheels or windows onto unrelated geometry.
    yy = int(np.clip(y, 4, region.shape[0] - 5))
    xx = int(np.clip(x, 4, region.shape[1] - 5))
    patch = Image.fromarray(region[yy - 4 : yy + 4, xx - 4 : xx + 4, :3].copy(), "RGB")
    return target.astype("uint8"), patch


def _window_trim(pixels, windows, fallback):
    x0, y0, x1, y1 = windows
    pad = max(4, int((x1 - x0) * 0.06))
    region = pixels[
        max(0, y0 - pad) : min(pixels.shape[0], y1 + pad),
        max(0, x0 - pad) : min(pixels.shape[1], x1 + pad),
    ]
    hsv = cv2.cvtColor(region[:, :, :3], cv2.COLOR_RGB2HSV)
    select = (
        (hsv[:, :, 0] >= 15)
        & (hsv[:, :, 0] <= 40)
        & (hsv[:, :, 1] >= 90)
        & (hsv[:, :, 2] >= 140)
        & (region[:, :, 3] > 200)
    )
    if int(select.sum()) < max(12, int(select.size * 0.003)):
        return np.asarray(fallback, dtype=np.uint8), "body_color_fallback"
    return np.median(region[:, :, :3][select], axis=0).astype(
        np.uint8
    ), "sampled_yellow_trim"


def _lights(pixels, bbox):
    x0, y0, w, h = bbox
    rgb = pixels[:, :, :3].astype(np.int16)
    bright = (
        (rgb[:, :, 0] > 180)
        & (rgb[:, :, 1] > 150)
        & (rgb[:, :, 2] > 90)
        & (pixels[:, :, 3] > 200)
    )
    contours, _ = cv2.findContours(
        (bright * 255).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    found = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if (
            x0 + 0.80 * w < x < x0 + 0.98 * w
            and y0 + 0.38 * h < y < y0 + 0.65 * h
            and 0.012 * w < bw < 0.09 * w
            and 0.025 * h < bh < 0.14 * h
            and area > 8
            and area / (bw * bh) > 0.40
        ):
            found.append((x, y, bw, bh))
    found.sort()
    for i, a in enumerate(found):
        for b in found[i + 1 :]:
            if (
                0.025 * w < b[0] - a[0] < 0.15 * w
                and abs((a[1] + a[3] / 2) - (b[1] + b[3] / 2)) < 0.06 * h
            ):
                return [a, b]
    return []


def build_vehicle(rgba: Image.Image):
    result = _features(rgba)
    if result is None:
        return None
    pixels, bbox, wheels, windows, window_boxes, window_polygons = result
    sampled = _sample_body_patch(pixels, bbox)
    if sampled is None:
        return None
    body_color, body_texture = sampled
    x0, y0, w, h = bbox
    scale = 2 / w
    wx0, wy0, wx1, wy1 = windows

    def xpos(px):
        return (px - (x0 + w / 2)) * scale

    def ypos(py):
        return (y0 + h - py) * scale

    # Side proportions and positions are measured, but depth and far side are guesses.
    wheel_y = float(np.mean([a[1] for a in wheels]))
    # The lowest silhouette includes wheels; retain visible clearance above it.
    body_bottom = min(y0 + h - 0.055 * h, wheel_y + 0.55 * (y0 + h - wheel_y))
    body_top = wy1 + 0.02 * h
    if body_bottom <= body_top:
        return None
    depth = 0.68
    bottom = ypos(body_bottom)
    top = ypos(body_top)
    body = _rounded_box([1.94, top - bottom, depth], min(0.14, (top - bottom) * 0.22))
    body.vertices[:, 0] *= 1 - 0.10 * np.abs(body.vertices[:, 1]) / max(
        (top - bottom) / 2, 1e-5
    )
    body.apply_translation([0, (top + bottom) / 2, 0])
    _material(body, body_color, body_texture)
    # Extruded trapezoid, from measured window extents; the image supplies its visible color.
    xlo = xpos(wx0 - 0.045 * w)
    xhi = xpos(wx1 + 0.045 * w)
    roof = ypos(wy0 - 0.04 * h)
    base = top - 0.02
    rooflo = xpos(wx0 + 0.03 * w)
    roofhi = xpos(wx1 - 0.03 * w)
    poly = np.array(
        [[xlo, base], [xhi, base], [roofhi, roof], [rooflo, rooflo * 0 + roof]]
    )
    verts = np.array([[x, y, z] for z in (-0.29, 0.29) for x, y in poly])
    faces = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]
    cabin = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    trimesh.repair.fix_normals(cabin)
    trim_color, trim_source = _window_trim(pixels, windows, body_color)
    _material(cabin, trim_color)
    scene = trimesh.Scene()
    scene.add_geometry(body, node_name="body", geom_name="body")
    scene.add_geometry(cabin, node_name="cabin", geom_name="cabin")
    # Contours from both visible windows become glazing geometry, not image decals.
    for i, ((ax, ay, bx, by), polygon) in enumerate(zip(window_boxes, window_polygons)):
        sample = pixels[int((ay + by) / 2), int((ax + bx) / 2), :3]
        xy = np.array(
            [
                [xpos(p[0]), np.clip(ypos(p[1]), base + 0.01, roof - 0.01)]
                for p in polygon
            ]
        )
        for side in (-1, 1):
            vv = np.array([[x, y, side * 0.296] for x, y in xy])
            ff = np.array([[0, j, j + 1] for j in range(1, len(vv) - 1)])
            pane = trimesh.Trimesh(vertices=vv, faces=ff, process=False)
            _material(pane, sample)
            pane.visual.material.doubleSided = True
            scene.add_geometry(
                pane, node_name=f"window_{i}_{side}", geom_name=f"window_{i}_{side}"
            )
    # Both end windows are inferred; the plane follows each sloped cabin end.
    # A margin leaves visible pillars and roof thickness. Offset avoids z-fighting.
    sax, say, sbx, sby = window_boxes[0]
    sample = pixels[int((say + sby) / 2), int((sax + sbx) / 2), :3]
    for side, low_x, high_x in ((-1, xlo, rooflo), (1, xhi, roofhi)):
        t0, t1 = 0.14, 0.84
        points = []
        for t, z in ((t0, -0.21), (t0, 0.21), (t1, 0.21), (t1, -0.21)):
            x = low_x + (high_x - low_x) * t + side * 0.002
            points.append([x, base + (roof - base) * t, z])
        pane = trimesh.Trimesh(
            vertices=np.array(points), faces=[[0, 1, 2], [0, 2, 3]], process=False
        )
        _material(pane, sample)
        pane.visual.material.doubleSided = True
        scene.add_geometry(
            pane,
            node_name=f"inferred_end_glass_{side}",
            geom_name=f"inferred_end_glass_{side}",
        )
    light_boxes = _lights(pixels, bbox)
    if light_boxes:
        # The visible endpoint is +X. Depth positions and opposite face remain inferred.
        for i, (lx, ly, lw, lh) in enumerate(light_boxes):
            z = (i - 0.5) * 0.34
            y = np.clip(ypos(ly + lh / 2), bottom + 0.045, top - 0.045)
            lamp = trimesh.creation.box(
                extents=[0.025, max(0.04, lh * scale), max(0.06, lw * scale)]
            )
            vertices = body.vertices
            nearby = vertices[
                (np.abs(vertices[:, 1] - y) < 0.025)
                & (np.abs(vertices[:, 2] - z) < 0.045)
            ]
            surface_x = (
                float(nearby[:, 0].max())
                if len(nearby)
                else float(body.bounds[1, 0] - 0.07)
            )
            lamp.apply_translation([surface_x + 0.004, y, z])
            color = np.median(
                pixels[ly : ly + lh, lx : lx + lw, :3].reshape(-1, 3), axis=0
            )
            _material(lamp, color)
            scene.add_geometry(
                lamp, node_name=f"visible_lamp_{i}", geom_name=f"visible_lamp_{i}"
            )
    tire_rgb = np.array([26, 52, 54, 255], dtype=np.uint8)
    for i, (cx, cy, bw, bh, _) in enumerate(wheels):
        radius = float(np.clip((y0 + h - cy) * scale * 0.95, 0.12, 0.28))
        wy = ypos(cy)
        # Sample the ring outside the dark hub; this preserves the image's hub hue.
        px = int(np.clip(cx + bw * 0.60, 0, pixels.shape[1] - 1))
        py = int(np.clip(cy, 0, pixels.shape[0] - 1))
        hub = np.r_[pixels[py, px, :3], 255].astype(np.uint8)
        for side in (-1, 1):
            z = side * (depth / 2 + 0.012)
            tire = trimesh.creation.cylinder(radius=radius, height=0.105, sections=48)
            tire.apply_translation([xpos(cx), wy, z])
            _material(tire, tire_rgb)
            hubmesh = trimesh.creation.cylinder(
                radius=radius * 0.47, height=0.112, sections=40
            )
            hubmesh.apply_translation([xpos(cx), wy, z + side * 0.004])
            _material(hubmesh, hub)
            scene.add_geometry(
                tire, node_name=f"tire_{i}_{side}", geom_name=f"tire_{i}_{side}"
            )
            scene.add_geometry(
                hubmesh, node_name=f"hub_{i}_{side}", geom_name=f"hub_{i}_{side}"
            )
    metadata = {
        "method": "RGBA contour heuristic; 2 aligned lower dark ellipses and upper window region",
        "input_size": list(rgba.size),
        "analysis_size": [int(pixels.shape[1]), int(pixels.shape[0])],
        "silhouette_bbox": list(bbox),
        "wheel_candidates": [list(map(float, a[:4])) for a in wheels],
        "window_bbox": list(windows),
        "window_polygons": window_polygons,
        "cabin_trim_color_rgb": trim_color.tolist(),
        "cabin_trim_source": trim_source,
        "end_glass": "inferred, inset within sloped end panels; not observed geometry",
        "underbody_clearance": float(
            bottom
            - min(
                ypos(cy) - float(np.clip((y0 + h - cy) * scale * 0.95, 0.12, 0.28))
                for _, cy, _, _, _ in wheels
            )
        ),
        "visible_lamps_detected": [list(map(int, b)) for b in light_boxes],
        "visible_lamp_note": "features on image right placed on +X endpoint; positions in depth are estimated",
        "units": "relative; +Y up, +Z observed side, +X image right",
        "depth": depth,
        "front_back": "unknown from single image; +X is image right, not established front",
        "inferred": "far side, top, depth, and occluded geometry; no side image projection",
        "warning": "heuristic cannot prove vehicle identity or exact vehicle type",
    }
    return scene, metadata
