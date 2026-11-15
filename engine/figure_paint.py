"""Infer flat vinyl paint from an aligned, front-facing character cutout."""

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


def _labels_and_palette(rgb, mask, height):
    # Suppress textile grain before estimating the ink colours. Hue is circular,
    # so red on opposite sides of zero maps to a single pigment.
    filtered = cv2.bilateralFilter(cv2.medianBlur(rgb, 3), 5, 37, 3)
    hsv = cv2.cvtColor(filtered, cv2.COLOR_RGB2HSV)
    hue, saturation, value = hsv.transpose(2, 0, 1)
    chromatic = (saturation > 62) & (value > 70)
    neutral = mask & ~chromatic
    colors = []
    labels = np.zeros(mask.shape, np.int32)
    lum = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
    if neutral.any():
        levels = lum[neutral]
        threshold, _ = cv2.threshold(
            levels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        low = lum <= threshold
        groups = (
            [low, ~low]
            if np.ptp(np.percentile(levels, [2, 98])) > 53
            else [np.ones_like(low)]
        )
        for group in groups:
            observed = neutral & group
            if np.any(observed):
                labels[group] = len(colors)
                colors.append(np.median(filtered[observed], axis=0))

    selected = mask & chromatic
    if selected.any():
        histogram = np.bincount(hue[selected], minlength=180)
        bins = np.arange(180)

        def signed_dist(a, b):
            return (a - b + 90) % 180 - 90

        centres = []
        unassigned = histogram.copy()
        for _ in range(10):
            counts = sum(np.roll(unassigned, k) for k in range(-5, 6))
            peak = int(np.argmax(counts))
            if counts[peak] < max(4, 0.0015 * mask.sum()):
                break
            cluster = abs(signed_dist(bins, peak)) < 14
            centre = peak + np.average(
                signed_dist(bins[cluster], peak), weights=unassigned[cluster]
            )
            centres.append(centre % 180)
            unassigned[cluster] = 0
        if centres:
            choices = np.argmin(
                abs(signed_dist(hue[..., None].astype(float), np.array(centres))),
                axis=-1,
            )
            for i in range(len(centres)):
                group = selected & (choices == i)
                labels[group] = len(colors)
                colors.append(np.median(filtered[group], axis=0))
    if not colors:
        colors = [np.median(rgb[mask], axis=0)]
    # Replace disconnected specks with the nearest surviving patch. Facial
    # details larger than a few source pixels remain intact.
    valid = mask.copy()
    for i in range(len(colors)):
        count, regions, stats, _ = cv2.connectedComponentsWithStats(
            ((labels == i) & mask).astype("uint8")
        )
        if count > 1:
            tiny = (
                np.flatnonzero(
                    stats[1:, cv2.CC_STAT_AREA] < max(2, height**2 * 0.000025)
                )
                + 1
            )
            valid[np.isin(regions, tiny)] = False
    if valid.any():
        closest = distance_transform_edt(
            ~valid, return_distances=False, return_indices=True
        )
        labels = labels[tuple(closest)]
    return labels, np.uint8(colors)


def _sample_panels(rgba, mapping):
    projection = mapping["projection"]
    image = rgba.convert("RGBA")
    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    pixels = np.array(image)
    mask = pixels[:, :, 3] >= 128
    if not mask.any():
        raise ValueError("figure cutout has no visible pixels")
    scale = np.array(image.size) / np.array(rgba.size)
    landmarks = mapping["landmarks_px"]
    head_x, head_y = np.array(landmarks["head_center"]) * scale
    height = projection["pixels_per_unit"] * scale[1]
    head_width = mapping["detection"]["head_width_pixels"] * scale[0]
    distance = distance_transform_edt(mask)
    interior = distance > max(1.5, 0.010 * height)
    if not interior.any():
        raise ValueError("figure cutout has no colour inside its outline")
    labels, palette = _labels_and_palette(pixels[:, :, :3], interior, height)
    yy, xx = np.indices(mask.shape)
    body_sample = (
        interior
        & (yy > head_y + height * 0.20)
        & (np.abs(xx - head_x) < head_width * 0.44)
    )
    base = int(
        np.bincount(labels[body_sample if body_sample.any() else interior]).argmax()
    )
    # The camera sees the arms and toes as dark outline strokes in the source.
    # Thin strokes below the head belong to the illustrated contours. Preserve
    # broad pigment, even when the stroke and pigment share the same hue.
    body_labels = np.full_like(labels, base)
    radius = max(1, round(height * 0.009))
    disk = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    for i in range(len(palette)):
        if i != base:
            region = cv2.morphologyEx(
                ((labels == i) & interior).astype("uint8"), cv2.MORPH_OPEN, disk
            )
            _, components, stats, _ = cv2.connectedComponentsWithStats(region)
            keep = (
                np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] > max(5, height**2 * 0.0016))
                + 1
            )
            for component in keep:
                patch = components == component
                thickness = cv2.distanceTransform(
                    patch.astype("uint8"), cv2.DIST_L2, 3
                ).max()
                if (
                    np.max(distance[patch]) >= height * 0.028
                    and thickness >= height * 0.018
                ):
                    body_labels[patch] = i
    # Carry broad flank pigment around the sides, keeping central symbols and
    # facial features off the back. Tail rings follow their measured centreline.
    rear_body = np.full_like(labels, base)
    rear_head = np.full_like(labels, base)
    for side in (-1, 1):
        lateral = (xx - head_x) * side
        # Only side-touching patches may continue onto a flank. Exclude the
        # base of the tail, where several projected colours overlap.
        if side == mapping["detection"]["tail_side"]:
            continue
        flank = (lateral > 0.35 * head_width) & (yy > head_y + 0.16 * height)
        for pigment in range(len(palette)):
            if pigment == base:
                continue
            _, components, stats, _ = cv2.connectedComponentsWithStats(
                ((body_labels == pigment) & flank & interior).astype("uint8")
            )
            for component, stat in enumerate(stats[1:], 1):
                patch = components == component
                if (
                    stat[cv2.CC_STAT_AREA] > height**2 * 0.001
                    and np.min(distance[patch]) < height * 0.022
                ):
                    cy = np.mean(yy[patch])
                    axis_y = max(height * 0.025, stat[cv2.CC_STAT_HEIGHT] * 0.56)
                    oval = (
                        (lateral - 0.50 * head_width) / (0.15 * head_width)
                    ) ** 2 + ((yy - cy) / axis_y) ** 2
                    rear_body[oval < 1] = pigment
    tail_panel = np.full_like(labels, base)
    if landmarks.get("tail_skeleton"):
        path = np.array(landmarks["tail_skeleton"]) * scale
        tail_side = mapping["detection"]["tail_side"]
        query = ((xx - head_x) * tail_side > head_width * 0.49) & (
            yy > head_y + height * 0.12
        )
        py, px = np.where(query)
        if len(py):
            dist, index = cKDTree(path).query(np.column_stack([px, py]))
            close = dist < height * 0.105
            centre = np.rint(path[index[close]]).astype(int)
            ring = labels[
                centre[:, 1].clip(0, image.height - 1),
                centre[:, 0].clip(0, image.width - 1),
            ]
            tail_panel[py[close], px[close]] = ring
            body_labels[py[close], px[close]] = base
    return (
        {
            "head": np.concatenate([palette[labels], palette[rear_head]], axis=1),
            "body": np.concatenate(
                [palette[body_labels], palette[rear_body], palette[tail_panel]], axis=1
            ),
        },
        palette,
        base,
    )


def _wrap_head(panel, mesh, rgba, projection, base_color):
    """Continue the measured lateral paint around the actual ellipsoid rim."""
    h, w = panel.shape[:2]
    w //= 2
    scale = np.array([w, h]) / rgba.size
    lo, hi = mesh.bounds
    center = (
        (lo + hi)[:2] * 0.5 * [1, -1] * projection["pixels_per_unit"]
        + projection["center_px"]
    ) * scale
    rx, ry = (hi - lo)[:2] * 0.5 * projection["pixels_per_unit"] * scale
    yy, xx = np.indices((h, w))
    mid_x, mid_y = center

    def width_at(rows):
        return rx * np.sqrt(np.clip(1 - ((rows - mid_y) / ry) ** 2, 0, 1))

    side = np.sign(xx - mid_x)
    fraction = np.clip(abs(xx - mid_x) / np.maximum(1, width_at(yy)), 0, 1)
    front = panel[:, :w].copy()
    original = front.copy()
    back = np.full_like(front, base_color)
    # On the ellipsoid this is the square of the normalised out-of-plane
    # distance. Both UV islands evaluate to zero at their common silhouette.
    depth_squared = np.maximum(
        0, 1 - ((xx - mid_x) / rx) ** 2 - ((yy - mid_y) / ry) ** 2
    )
    elevation = (mid_y - yy) / ry
    crown_weight = np.clip((elevation - 0.54) / 0.24, 0, 1)
    crown_weight = crown_weight * crown_weight * (3 - 2 * crown_weight)
    crown = crown_weight > 0
    front[crown] = base_color
    pigments = {}
    for sign in (-1, 1):
        offset = (xx - mid_x) * sign / rx
        observed = (
            (offset > 0.45)
            & (offset < 0.78)
            & (yy > mid_y - ry * 0.65)
            & (yy < mid_y - ry * 0.20)
        )
        if not observed.any():
            continue
        colors, counts = np.unique(original[observed], return_counts=True, axis=0)
        pigment = colors[np.argmax(counts)]
        pigments[sign] = pigment
        if np.array_equal(pigment, base_color):
            continue
        source_area = np.all(original == pigment, axis=2).astype("uint8")
        area = source_area
        r = max(1, round(ry * 0.055))
        area = cv2.morphologyEx(
            area,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1)),
        )
        ends = []
        for x in np.flatnonzero((offset[0] > 0.60) & (offset[0] < 0.86)):
            hit = np.flatnonzero(
                area[:, x]
                & (yy[:, x] > mid_y - ry * 0.40)
                & (yy[:, x] < mid_y + ry * 0.66)
            )
            if len(hit):
                ends.append(hit[-1])
        if not ends:
            continue
        level = float(np.median(ends))
        # Front and back meet at a common fitted cheek boundary. Texture at
        # that glancing angle is poorly sampled and magnified by projection.
        # Replace the rim on both sides with the same continuous paint edge.
        limit = level - ry * 0.12 * np.clip((fraction - 0.72) / 0.28, 0, 1)
        rim = (side == sign) & (fraction > 0.80) & (yy > mid_y - ry * 0.68)
        front[rim] = base_color
        front[rim & (yy < limit)] = pigment
        # Estimate the actual pigment boundary on the forehead. Use absolute
        # horizontal coordinates so the stripe has the same width at the pole
        # when approached from the source-facing or inferred hemisphere.
        boundaries = []
        for row in np.flatnonzero((elevation[:, 0] > 0.45) & (elevation[:, 0] < 0.72)):
            columns = np.flatnonzero(
                (offset[row] > -0.32) & (offset[row] < 0.65) & (area[row] > 0)
            )
            if len(columns) > rx * 0.12:
                boundaries.append(np.percentile((columns - mid_x) * sign, 3))
        edge = float(np.median(boundaries)) if boundaries else rx * 0.12
        signed = sign * (xx - mid_x) - edge
        observed_signed = cv2.distanceTransform(
            source_area, cv2.DIST_L2, 3
        ) - cv2.distanceTransform(1 - source_area, cv2.DIST_L2, 3)
        front[
            crown & ((1 - crown_weight) * observed_signed + crown_weight * signed > 0)
        ] = pigment
        # A single curved boundary continues over the pole. The rear patch
        # draws inwards only as its depth increases, with zero slope at the
        # seam. Fine eye details remain on the observed face.
        nape_limit = limit - ry * 1.02 * depth_squared
        side_margin = signed - rx * 0.34 * depth_squared
        bottom_margin = nape_limit - yy
        radius = np.maximum(0.01, rx * 0.72 * depth_squared)
        corner = np.maximum(radius - abs(side_margin - bottom_margin), 0)
        rear_margin = np.minimum(side_margin, bottom_margin) - corner**2 / (4 * radius)
        back[rear_margin > 0] = pigment
    return np.concatenate([front, back], axis=1), pigments


def paint_figure(scene, rgba, mapping):
    """Paint an upright figure from build_figure's orthographic coordinate map.

    Only UV seam vertices are duplicated, preserving smooth normals and geometry.
    The atlas contains separate source-facing and inferred rear panels.
    """
    projection = mapping["projection"]
    if list(rgba.size) != list(projection["image_size"]):
        raise ValueError("figure projection and paint reference sizes differ")
    panels, palette, base = _sample_panels(rgba, mapping)
    result = scene.copy()
    pigments = {}
    if "head" in scene.geometry:
        head_panel, pigments = _wrap_head(
            panels["head"], scene.geometry["head"], rgba, projection, palette[base]
        )
    else:
        head_panel = panels["head"]
    for name, mesh in scene.geometry.items():
        part = "head" if name == "head" or name.startswith("ear") else "body"
        atlas = head_panel if name == "head" else panels[part]
        if name.startswith("ear"):
            atlas = atlas.copy()
            side = -1 if name.endswith("left") else 1
            shell_color = pigments.get(side, palette[base])
            half = atlas.shape[1] // 2
            yy, xx = np.indices((atlas.shape[0], half))
            ext = (
                (
                    mesh.bounds[:, :2] * [1, -1] * projection["pixels_per_unit"]
                    + projection["center_px"]
                )
                * np.array([half, atlas.shape[0]])
                / rgba.size
            )
            xmin, ymin = ext.min(axis=0)
            xmax, ymax = ext.max(axis=0)
            t = (yy - ymin) / max(1, ymax - ymin)
            u = (xx - (xmin + xmax) * 0.5) / max(1, (xmax - xmin) * 0.5)
            central = (abs(u) < 0.35) & (t > 0.23) & (t < 0.67)
            atlas[:, half:] = shell_color
            if central.any():
                inner, counts = np.unique(
                    atlas[:, :half][central], axis=0, return_counts=True
                )
                inner_color = inner[np.argmax(counts)]
                atlas[:, :half] = shell_color
                progress = np.clip((t - 0.15) / 0.67, 0, 1)
                width = (
                    0.69 * np.sin(np.pi * progress) ** 0.70 * (0.55 + 0.45 * progress)
                )
                atlas[:, :half][(abs(u) < width) & (progress > 0) & (progress < 1)] = (
                    inner_color
                )
        height, width = atlas.shape[:2]
        panel_count = 3 if part == "body" else 2
        # In a frontal orthographic projection the depth sign of each normal
        # identifies the two visible sides of the shell, including curved limbs.
        rear = (mesh.face_normals[:, 2] < 0).astype(np.int64)
        if part == "body" and mapping["landmarks_px"].get("tail_skeleton"):
            points_xy = (
                mesh.triangles_center[:, :2] * [1, -1] * projection["pixels_per_unit"]
                + projection["center_px"]
            )
            path = np.asarray(mapping["landmarks_px"]["tail_skeleton"])
            side = mapping["detection"]["tail_side"]
            head_center = mapping["landmarks_px"]["head_center"]
            lateral = (points_xy[:, 0] - head_center[0]) * side
            body_radius = mapping["detection"]["body_width_pixels"] / 2
            distance, _ = cKDTree(path).query(points_xy)
            radius = mapping["detection"]["head_width_pixels"] * 0.09
            rear[(distance < radius * 1.15) & (lateral > body_radius * 1.15)] = 2
        keys = mesh.faces + len(mesh.vertices) * rear[:, None]
        ids, inverse = np.unique(keys.ravel(), return_inverse=True)
        indices = ids % len(mesh.vertices)
        verts = mesh.vertices[indices]
        points = (
            verts[:, :2] * [1, -1] * projection["pixels_per_unit"]
            + projection["center_px"]
        )
        coords = (points + 0.5) / np.array(rgba.size)
        coords = np.clip(
            coords, [1 / width, 0.5 / height], [1 - 1 / width, 1 - 0.5 / height]
        )
        uv = np.column_stack(
            ((coords[:, 0] + ids // len(mesh.vertices)) / panel_count, 1 - coords[:, 1])
        )
        material = trimesh.visual.material.PBRMaterial(
            name=f"figure_{part}_paint",
            baseColorTexture=Image.fromarray(
                cv2.resize(
                    cv2.GaussianBlur(atlas, (3, 3), 0.65),
                    None,
                    fx=2,
                    fy=2,
                    interpolation=cv2.INTER_CUBIC,
                )
            ),
            metallicFactor=0.0,
            roughnessFactor=0.43,
        )
        painted = trimesh.Trimesh(
            vertices=verts,
            faces=inverse.reshape(-1, 3),
            vertex_normals=mesh.vertex_normals[indices],
            process=False,
        )
        painted.visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
        result.geometry[name] = painted
    mapping["paint"] = {
        "source_palette_srgb": palette.tolist(),
        "inferred_base_srgb": palette[base].tolist(),
        "rear": "inferred base, lateral pigment and measured tail rings; no rear reference",
    }
    return result
