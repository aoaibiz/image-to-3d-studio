"""Conservative image-evidenced face strokes, attached to the front head mesh."""

import cv2
import numpy as np
import trimesh
from PIL import Image


def detected_lines(rgba, mapping):
    if not isinstance(rgba, Image.Image):
        return []
    pixels = np.asarray(rgba.convert("RGBA"))
    rgb = pixels[:, :, :3]
    h, w = rgb.shape[:2]
    if max(h, w) > 2048:
        return []
    try:
        cx, cy = map(float, mapping["landmarks_px"]["head_center"])
        width = float(mapping["detection"]["head_width_pixels"])
        if width < 30 or not (0 <= cx < w and 0 <= cy < h):
            return []
    except (KeyError, TypeError, ValueError):
        return []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Isolate short near-horizontal marks in the lower lateral face; broad
    # patches and long sloping outlines are not interpreted as whiskers.
    mask = ((gray < 85) & (pixels[:, :, 3] >= 128)).astype("uint8") * 255
    yy, xx = np.indices(gray.shape)
    dx = np.abs(xx - cx)
    mask[
        (dx < width * 0.39)
        | (dx > width * 0.65)
        | (yy < cy + width * 0.05)
        | (yy > cy + width * 0.34)
    ] = 0
    lines = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 180,
        threshold=max(5, int(width * 0.045)),
        minLineLength=max(5, int(width * 0.055)),
        maxLineGap=2,
    )
    candidates = []
    if lines is None:
        return []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        length = np.hypot(dx, dy)
        if abs(dx) < width * 0.055 or length > width * 0.24 or abs(dy) > abs(dx) * 0.55:
            continue
        side = -1 if (x1 + x2) / 2 < cx else 1
        # Require a genuinely dark detected tip outside the measured head width.
        if max((x1 - cx) * side, (x2 - cx) * side) < width * 0.51:
            continue
        candidates.append(
            (side, float((y1 + y2) / 2), [int(x1), int(y1), int(x2), int(y2)])
        )
    # Merge Hough duplicates, requiring evidence on both sides before adding any.
    result = []
    for side in (-1, 1):
        group = sorted((a for a in candidates if a[0] == side), key=lambda a: a[1])
        kept = []
        for item in group:
            if not kept or item[1] - kept[-1][1] > max(2, width * 0.025):
                kept.append(item)
        result.extend(kept)
    if not all(any(x[0] == side for x in result) for side in (-1, 1)):
        return []
    return [item[2] for item in result]


def _front_z(mesh, x, y):
    triangles = np.asarray(mesh.vertices)[mesh.faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (
        a[:, 1] - c[:, 1]
    )
    valid = np.abs(den) > 1e-12
    den = np.where(valid, den, 1)
    u = (
        (b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])
    ) / den
    v = (
        (c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])
    ) / den
    mask = valid & (u >= 0) & (v >= 0) & (u + v <= 1)
    if not mask.any():
        return None
    return float(np.max((u * a[:, 2] + v * b[:, 2] + (1 - u - v) * c[:, 2])[mask]))


def add_detected_details(scene, rgba, mapping):
    """Return a copied scene; unsupported landmarks or missing surface yield no additions."""
    result = scene.copy()
    if "head" not in scene.geometry:
        return result
    try:
        projection = mapping["projection"]
        center = projection["center_px"]
        scale = float(projection["pixels_per_unit"])
        if scale <= 0:
            return result
        head = scene.geometry["head"]
        cx = float(mapping["landmarks_px"]["head_center"][0])
        width = float(mapping["detection"]["head_width_pixels"])
        for index, line in enumerate(detected_lines(rgba, mapping)):
            ends = np.array([[line[0], line[1]], [line[2], line[3]]], dtype=float)
            if abs(ends[0, 0] - cx) > abs(ends[1, 0] - cx):
                ends = ends[::-1]
            direction = ends[1] - ends[0]
            if np.linalg.norm(direction) == 0:
                continue
            # Extend inward along the observed stroke only to find a seated root.
            direction = direction / np.linalg.norm(direction)
            root = None
            for n in np.linspace(0, width * 0.12, 20):
                pix = ends[0] - direction * n
                x = (pix[0] - center[0]) / scale
                y = (center[1] - pix[1]) / scale
                z = _front_z(head, x, y)
                if z is not None:
                    root = np.array([x, y, z - 0.001])
                    break
            if root is None:
                continue
            tip = np.array(
                [
                    (ends[1, 0] - center[0]) / scale,
                    (center[1] - ends[1, 1]) / scale,
                    root[2],
                ]
            )
            axis = tip - root
            length = np.linalg.norm(axis)
            if length < 1e-6:
                continue
            axis /= length
            normal = np.cross(axis, [0, 0, 1])
            norm = np.linalg.norm(normal)
            if norm < 1e-8:
                continue
            normal /= norm
            other = np.cross(axis, normal)
            vertices = []
            faces = []
            segments = 8
            rings = 8
            for j in range(rings):
                t = j / (rings - 1)
                radius = 0.0018 * (1 - 0.8 * t)
                point = root * (1 - t) + tip * t
                for k in range(segments):
                    theta = 2 * np.pi * k / segments
                    vertices.append(
                        point
                        + radius * (np.cos(theta) * normal + np.sin(theta) * other)
                    )
            for j in range(rings - 1):
                for k in range(segments):
                    a = j * segments + k
                    b = j * segments + (k + 1) % segments
                    faces.extend(
                        [[a, b, b + segments], [a, b + segments, a + segments]]
                    )
            root_center = len(vertices)
            vertices.extend([root, tip])
            tip_center = root_center + 1
            last = (rings - 1) * segments
            for k in range(segments):
                nxt = (k + 1) % segments
                faces.extend(
                    [[root_center, nxt, k], [tip_center, last + k, last + nxt]]
                )
            tube = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            tube.visual.vertex_colors = [30, 26, 24, 255]
            name = f"detected_whisker_{index}"
            result.add_geometry(tube, geom_name=name, node_name=name)
    except (KeyError, TypeError, ValueError, IndexError):
        return scene.copy()
    return result
