"""Build cream and fruit from the measured foreground. +Y is up, +Z faces the camera."""

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from engine.dessert_garnish import add_measured_decor


def _mat(name, rgb, roughness):
    tile = np.asarray(rgb).clip(0, 255).astype("uint8")
    if tile.ndim == 1:
        tile = np.broadcast_to(tile, (2, 2, 3)).copy()
    return trimesh.visual.material.PBRMaterial(
        name=name,
        baseColorTexture=Image.fromarray(tile),
        metallicFactor=0,
        roughnessFactor=roughness,
    )


def _add(scene, name, mesh, material, uv=None):
    if uv is None:
        uv = np.zeros((len(mesh.vertices), 2))
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    scene.add_geometry(mesh, node_name=name, geom_name=name)


def _cream_profile(crop, rgb):
    """Sample the side envelope and the number of visible piping courses."""
    height, width = crop.shape[:2]
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(float)
    reference = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    distance = np.linalg.norm(lab[:, :, 1:] - reference[1:], axis=2)
    cream = (distance < 19) & (np.abs(lab[:, :, 0] - reference[0]) < 57)
    levels = np.linspace(0.05, 0.95, 14)
    fractions = []
    for level in levels:
        row = int(level * (height - 1))
        band = cream[max(0, row - height // 72) : min(height, row + height // 72 + 1)]
        xs = np.where(np.mean(band, axis=0) > 0.50)[0]
        xs = xs[(xs > 0.015 * width) & (xs < 0.985 * width)]
        fractions.append(
            (np.quantile(xs, 0.95) - np.quantile(xs, 0.05)) / width
            if len(xs) > 4
            else np.nan
        )
    fractions = np.asarray(fractions)
    valid = np.isfinite(fractions) & (fractions > 0.08)
    guide = np.interp(levels, [0, 0.18, 0.45, 0.72, 1], [0.25, 0.68, 0.92, 0.96, 0.73])
    if np.count_nonzero(valid) >= 5:
        widths = gaussian_filter1d(
            np.interp(levels, levels[valid], fractions[valid]), 0.95
        )
        guide = 0.72 * widths / max(0.3, widths.max()) + 0.28 * guide
    guide = np.clip(guide, 0.18, 1)
    columns = np.linspace(0.29, 0.70, 9)
    counts = []
    for col in columns:
        k = int(col * (width - 1))
        signal = gaussian_filter1d(lab[:, k, 0], max(0.6, height * 0.004))
        troughs = find_peaks(-signal, distance=max(3, height * 0.028), prominence=6)[0]
        counts.append(
            sum(bool(cream[min(height - 1, t + height // 60), k]) for t in troughs)
        )
    courses = int(np.median(counts))
    return levels, guide, min(10, max(3, round(courses * 0.76))) if courses >= 4 else 0


def _tube(points, widths, sides=12):
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(1e-9, np.linalg.norm(tangent, axis=1, keepdims=True))
    outward = np.cross(tangent, [0, 1, 0])
    outward /= np.maximum(1e-9, np.linalg.norm(outward, axis=1, keepdims=True))
    vertical = np.cross(tangent, outward)
    t = np.linspace(0, 2 * np.pi, sides + 1)
    normals = (
        np.cos(t)[None, :, None] * outward[:, None, :]
        + np.sin(t)[None, :, None] * vertical[:, None, :]
    )
    vertices = (
        points[:, None, :]
        + widths[:, None, None] * (1 + 0.031 * np.cos(5 * t))[None, :, None] * normals
    )
    idx = np.arange(len(points) * (sides + 1)).reshape(-1, sides + 1)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    faces = np.vstack(
        (
            np.stack((a, b, c), axis=-1).reshape(-1, 3),
            np.stack((b, d, c), axis=-1).reshape(-1, 3),
        )
    )
    uv = np.column_stack(
        (
            np.repeat(np.linspace(0, len(points) / 50, len(points)), sides + 1),
            np.tile(t / (2 * np.pi), len(points)),
        )
    )
    return trimesh.Trimesh(
        vertices=vertices.reshape(-1, 3),
        faces=faces,
        vertex_normals=normals.reshape(-1, 3),
        process=False,
    ), uv


def _whipped_peaks(scene, crop, rgb, center, radius, bottom, height, material):
    """Measure the leading peak and model a fluted nozzle cross-section."""
    rows, cols = crop.shape[:2]
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(float)
    target = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    good = (np.linalg.norm(lab[:, :, 1:] - target[1:], axis=2) < 16) & (
        np.abs(lab[:, :, 0] - target[0]) < 45
    )
    # Reject isolated highlights beyond the cream silhouette.
    good = cv2.morphologyEx(
        good.astype("uint8"), cv2.MORPH_OPEN, np.ones((3, 3), "uint8")
    ).astype(bool)
    valid = good.any(axis=0)
    top = np.where(valid, good.argmax(axis=0), rows)
    top = gaussian_filter1d(top.astype(float), max(0.75, cols * 0.022))
    middle = np.arange(max(1, cols // 9), max(2, cols - cols // 9))
    primary = middle[np.argmin(top[middle])] / cols
    peaks, properties = find_peaks(
        -top, prominence=max(2, rows * 0.08), distance=max(2, cols // 4)
    )
    locations = [primary]
    for px in sorted(peaks, key=lambda i: top[i]):
        x = px / cols
        if (
            0.12 < x < 0.88
            and min(abs(x - a) for a in locations) > 0.27
            and top[px] < rows * 0.67
        ):
            locations.append(x)
        if len(locations) >= 3:
            break
    if len(locations) == 1 and cols > 1.35 * rows:
        # The near shoulder hides the far creases in a frontal view.
        locations.append(0.73 if primary < 0.55 else 0.27)
    a = np.linspace(0, 2 * np.pi, 73)
    q = np.linspace(0, 1, 53)
    t, angle = np.meshgrid(q, a, indexing="ij")
    for i, location in enumerate(locations):
        h = (
            height
            if i == 0
            else height
            * np.clip(1 - top[int(location * (cols - 1))] / rows, 0.48, 0.73)
        )
        r = radius * (0.65 if len(locations) > 1 else 0.91) * (1 if i == 0 else 0.70)
        lobes = (
            1
            + 0.155 * np.cos(7 * angle + 3.2 * t)
            + 0.034 * np.cos(14 * angle + 6.4 * t)
        )
        shape = (1 - t) ** 0.68 * (0.83 + 0.27 * np.sin(np.pi * t))
        offset = (location - 0.5) * 2 * radius
        zcenter = -0.13 * radius if i == 0 else 0.37 * radius
        # The straight nozzle is lifted off at an angle, leaving a narrow tip.
        vx = center + offset + r * shape * lobes * np.cos(angle) + 0.14 * r * t**3
        vz = zcenter + r * shape * lobes * np.sin(angle)
        vy = bottom - 0.01 * height + h * 1.01 * t
        idx = np.arange(t.size).reshape(t.shape)
        u, v, w, k = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
        f = np.vstack(
            (
                np.stack((u, w, v), 2).reshape(-1, 3),
                np.stack((v, w, k), 2).reshape(-1, 3),
            )
        )
        verts = np.stack((vx, vy, vz), axis=-1).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=verts, faces=f, process=False)
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        if mesh.volume < 0:
            mesh.invert()
        uv = np.column_stack(
            (
                np.arctan2(
                    mesh.vertices[:, 0] - center - offset, mesh.vertices[:, 2] - zcenter
                )
                / (2 * np.pi),
                (mesh.vertices[:, 1] - bottom) / max(height, 0.001),
            )
        )
        _add(scene, "fluted_cream_" + str(i + 1), mesh, material, uv)


def _add_mound(scene, crop, rgb, center, radius, bottom, height):
    levels, guide, courses = _cream_profile(crop, rgb)
    grain = gaussian_filter(
        np.random.default_rng(604).normal(size=(128, 256)), 0.7, mode="wrap"
    )
    reflectance = np.asarray(rgb) * (0.94 if np.mean(rgb) > 230 else 1.0)
    tile = reflectance + grain[:, :, None] * [6, 7, 7]
    material = _mat("Cream pigment", tile, 0.58)

    def profile(q):
        return radius * np.interp(
            1 - q, np.r_[0, levels, 1], np.r_[0.045, guide, guide[-1]]
        )

    if courses:
        # The cafe background may share the color of cream near the peak.
        # A piped mound closes to a rounded tip; retain its measured shoulders.
        cap = np.interp(
            levels,
            [0, 0.10, 0.25, 0.48, 0.68, 0.9, 1],
            [0.09, 0.42, 0.70, 0.94, 1, 0.95, 0.80],
        )
        guide = np.minimum(guide, cap + 0.07)
    rings = np.linspace(0, 1, 44)
    core = trimesh.creation.revolve(
        np.vstack(
            (
                [0, bottom],
                np.column_stack(
                    (profile(rings) * (0.91 if courses else 1), bottom + height * rings)
                ),
                [0, bottom + height],
            )
        ),
        sections=72,
    )
    core.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    core.apply_translation([center, 0, 0])
    x, y, z = core.vertices.T
    _add(
        scene,
        "cream_mound",
        core,
        material,
        np.column_stack(
            (np.arctan2(x - center, z) / (2 * np.pi), (y - bottom) / height)
        ),
    )
    if courses:
        q = np.linspace(0.015, 0.99, courses * 78)
        t = q * courses * 2 * np.pi
        strand_width = min(radius * 0.087, height / (courses * 4.1))
        for i, d in enumerate((-1, 0, 1, 1.68)):
            turn = t + 0.08 * np.sin(1.84 * t) + 0.13 * np.cos(0.48 * t)
            tilt = 0.68 * np.sin(0.67 * t) + 0.32 * np.cos(1.19 * t)
            r = profile(q) + strand_width * (
                0.48 * np.cos(2.6 * t + i * 0.2)
                + 0.32 * np.cos(0.75 * t)
                + d * np.sin(tilt)
            )
            y = (
                bottom
                + q * height
                + strand_width
                * (
                    0.61 * np.sin(2.17 * t)
                    + 0.36 * np.cos(3.08 * t)
                    + d * 1.55 * np.cos(tilt)
                )
            )
            # Tilt and local folds in the moving nozzle lay adjacent strands
            # softly across one another, especially around the upper courses.
            fold = np.maximum(0, np.sin(0.78 * t + 1.3 * i)) ** 4
            y += strand_width * (
                0.82 * np.sin(1.8 * t + 0.3 * i) + 0.95 * fold * np.sin(2.3 * t + i)
            )
            r += strand_width * (
                0.44 * np.sin(1.7 * t + 0.52 * i) + 0.52 * fold * np.cos(2 * t + i)
            )
            if i == 3:
                r -= 0.95 * strand_width
            r *= np.minimum(1, (1 - q) / 0.04)
            x = center + np.sin(turn) * r
            z = np.cos(turn) * r * 0.98
            width = (
                strand_width
                * (1 + 0.095 * np.cos(2.29 * t + i))
                * (0.25 + 0.75 * np.minimum(1, (1 - q) / 0.025))
            )
            mesh, uv = _tube(np.column_stack((x, y, z)), width)
            _add(scene, "piped_cream_" + str(i + 1), mesh, material, uv)
    else:
        # The low core connects individual flutes at the glass rim; avoid a
        # smooth lathed lid showing through between their tips.
        core.vertices[:, 1] = (
            bottom - height * 0.025 + (core.vertices[:, 1] - bottom) * 0.09
        )
        core.vertices[:, 0] = center + (core.vertices[:, 0] - center) * 0.85
        core.vertices[:, 2] *= 0.85
        _whipped_peaks(scene, crop, rgb, center, radius, bottom, height, material)
    return courses, profile


def _scatter_crumbs(scene, crop, rgb, center, radius, bottom, height, profile):
    # Only dark flecks surrounded by the cream pigment count. A background
    # stripe or a broad shadow must not create a layer of sprinkles.
    h, w = crop.shape[:2]
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    lum = lab[:, :, 0]
    filter_width = max(5, round(min(w, h) * 0.042) | 1)
    contrast = cv2.morphologyEx(
        lum,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_width, filter_width)),
    )
    mask = (contrast > 30).astype("uint8")
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    rng = np.random.default_rng(738)
    grains = []
    for k in range(1, count):
        _, _, bw, bh, area = stats[k]
        x, y = centroids[k]
        fraction = 1 - y / h
        if not (
            0.10 * w < x < 0.90 * w
            and 0.035 * h < y < 0.64 * h
            and 3 <= area <= w * h * 0.0018
            and max(bw, bh) < min(w, h) * 0.07
        ):
            continue
        rad = profile(fraction)
        world_x = (x / w - 0.5) * radius * 2
        if abs(world_x) > rad * 0.82:
            continue
        offset = max(2, round(max(bw, bh) * 1.3))
        ring = np.array(
            [
                crop[
                    int(np.clip(y + dy * offset, 0, h - 1)),
                    int(np.clip(x + dx * offset, 0, w - 1)),
                ]
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ]
        )
        if min(np.linalg.norm(ring.astype(float) - rgb, axis=1)) > 65:
            continue
        x0, y0, bw, bh, _ = stats[k]
        patch = crop[y0 : y0 + bh, x0 : x0 + bw].reshape(-1, 3)
        brightness = patch @ [0.3, 0.59, 0.11]
        color = np.median(patch[brightness <= np.quantile(brightness, 0.30)], axis=0)
        if color.mean() > np.mean(rgb) - 20:
            continue
        size = np.clip(
            np.sqrt(area) * radius / w * 1.78, 0.008 * radius, 0.060 * radius
        )
        grains.append((center + world_x, bottom + height * fraction, size, color))
        if len(grains) >= 160:
            break
    if len(grains) < 3:
        return 0

    # Locate the real surface of each cord. A radial approximation would
    # bury most crumbs below the front of the bundle precisely at the crest.
    body = [
        mesh
        for name, mesh in scene.geometry.items()
        if name == "cream_mound" or name.startswith("piped_cream_")
    ]
    centers = np.concatenate([mesh.triangles_center for mesh in body])
    normals = np.concatenate([mesh.face_normals for mesh in body])
    front = normals[:, 2] > 0.24
    near = centers[front]
    if not len(near):
        return 0
    tree = cKDTree(near[:, :2])
    groups = [[], [], []]
    pigments = [[], [], []]

    def scatter(point, size, color):
        pearl = (
            color[0] > 98 and color[2] > 0.43 * color[0] and color[1] < 0.91 * color[0]
        )
        index = 2 if pearl else int(color.mean() > 90)
        grain = trimesh.creation.icosphere(subdivisions=1 if pearl else 0)
        scale = (
            [1, 0.93, 1]
            if pearl
            else [rng.uniform(0.8, 1.5), rng.uniform(0.58, 1.0), 0.87]
        )
        grain.apply_scale(np.array(scale) * size)
        grain.apply_translation(point)
        groups[index].append(grain)
        pigments[index].append(np.clip(color * 1.10 + (8 if pearl else 1), 0, 255))

    for x, y, size, color in grains:
        nearby = tree.query_ball_point((x, y), radius * 0.048)
        if not nearby:
            gap, idx = tree.query((x, y))
            if gap > radius * 0.10:
                continue
            nearby = [idx]
        candidates = near[nearby]
        point = candidates[np.argmax(candidates[:, 2])].copy()
        point[2] += size * 0.42
        scatter(point, size, color)
    # Where the photo resolves a cluster, carry that same measured pigment
    # over the visible top of the curl. Depth visibility excludes buried cords.
    if len(grains) >= 9:
        crest = centers[
            (normals[:, 1] > 0.43) & (centers[:, 1] > bottom + 0.65 * height)
        ]
        if len(crest):
            cell = radius * 0.036
            ij = np.floor(
                (crest[:, [0, 2]] - crest[:, [0, 2]].min(axis=0)) / cell
            ).astype(int)
            heights = np.full(tuple(ij.max(axis=0) + 1), -np.inf)
            np.maximum.at(heights, (ij[:, 0], ij[:, 1]), crest[:, 1])
            crest = crest[crest[:, 1] >= heights[ij[:, 0], ij[:, 1]] - cell]
            for _ in range(min(310, len(grains) * 3)):
                top_grains = [
                    g
                    for g in grains
                    if g[1] > bottom + height * 0.56 and g[3].mean() < np.mean(rgb) - 45
                ]
                pool = top_grains if top_grains else grains
                x, y, size, color = pool[int(rng.integers(len(pool)))]
                weight = np.exp(
                    -0.5 * ((crest[:, 0] - x) / (radius * 0.26)) ** 2
                    - 0.5 * ((crest[:, 1] - y) / (height * 0.19)) ** 2
                    - 0.5 * ((crest[:, 2] - radius * 0.17) / (radius * 0.36)) ** 2
                )
                weight /= max(weight.sum(), 1e-12)
                if weight.sum() < 0.99:
                    continue
                point = crest[rng.choice(len(crest), p=weight)].copy()
                point[1] += size * 0.52
                scatter(point, size * rng.uniform(0.69, 1.14), color)
    for index, group in enumerate(groups):
        if group:
            _add(
                scene,
                f"cream_flecks_{index}",
                trimesh.util.concatenate(group),
                _mat(
                    "Sugar pearl" if index == 2 else "Crumb",
                    np.median(pigments[index], axis=0),
                    0.28 if index == 2 else 0.70,
                ),
            )
    return sum(map(len, groups))


def _fruit_landmarks(photo, part):
    h, w = photo.shape[:2]
    cx, cy = np.array(part["center"]) * [w, h]
    rx, ry = np.array(part["radii"]) * [w, h]

    def red_span(row):
        band = photo[
            max(0, int(row - ry * 0.08)) : min(h, int(row + ry * 0.08) + 1),
            max(0, int(cx - rx)) : min(w, int(cx + rx) + 1),
        ].astype(float)
        if not band.size:
            return 0
        g_ratio = min(0.54, part["rgb"][1] / max(1, part["rgb"][0]) + 0.22)
        red = (band[:, :, 0] * g_ratio > band[:, :, 1]) & (
            band[:, :, 0] > 1.42 * band[:, :, 2]
        )
        return np.count_nonzero(np.mean(red, axis=0) > 0.5)

    broad_top = red_span(cy - 0.48 * ry) > red_span(cy + 0.48 * ry) * 1.12
    crown = photo[
        max(0, int(cy - ry * 1.25)) : min(h, int(cy - ry * 0.67) + 1),
        max(0, int(cx - 0.85 * rx)) : min(w, int(cx + 0.85 * rx) + 1),
    ]
    if not crown.size:
        return broad_top, None
    pixels = crown.reshape(-1, 3).astype(float)
    green = pixels[
        (pixels[:, 1] > pixels[:, 0] * 1.06) & (pixels[:, 1] > pixels[:, 2] * 1.24)
    ]
    leaf = (
        np.median(green, axis=0) if len(green) > max(4, len(pixels) * 0.022) else None
    )
    # The crown supplies a second orientation cue when leaves hide the upper shoulder.
    return broad_top or leaf is not None, leaf


def _leaves(scene, key, pos, rx, ry, color):
    leaves = []
    for k in range(6):
        ang = 2 * np.pi * k / 6
        direction = np.array([np.cos(ang), 0, np.sin(ang)])
        side = np.array([-np.sin(ang), 0, np.cos(ang)])
        center = pos + [0, ry * 0.96, 0]
        mid = center + direction * rx * 0.26 + [0, ry * 0.09, 0]
        tip = (
            center
            + direction * rx * (0.44 + 0.08 * np.sin(k))
            + [0, ry * (0.18 + 0.14 * np.cos(k)), 0]
        )
        verts = np.array(
            [
                center,
                mid + 0.105 * rx * side,
                mid + [0, 0.045 * ry, 0],
                mid - 0.105 * rx * side,
                tip,
            ]
        )
        leaves.append(
            trimesh.Trimesh(
                vertices=verts,
                faces=[[0, 1, 2], [0, 2, 3], [1, 4, 2], [3, 2, 4]],
                process=False,
            )
        )
    mat = _mat("Green calyx", np.clip(color * 1.12, 0, 255), 0.61)
    mat.doubleSided = True
    _add(scene, key + "_leaf_crown", trimesh.util.concatenate(leaves), mat)


def _berry(
    scene, key, pos, rx, ry, rgb, seeded, dark, broad_top=False, leaf_color=None
):
    # A geodesic skin has no repeated pole vertices. Place berry seeds and
    # the blossom-end calyx in the local surface frame before posing the fruit.
    skin = trimesh.creation.icosphere(subdivisions=4 if seeded else 3)
    unit = skin.vertices.copy()
    x, y, z = unit.T
    a = np.arctan2(z, x)
    slope = 0.27 if broad_top else -0.19
    taper = (
        (1 + slope * y) * (1 + 0.012 * np.cos(3 * a + 2 * y))
        if seeded
        else np.ones_like(y)
    )
    local = unit * [rx, ry, rx]
    local[:, [0, 2]] *= taper[:, None]
    skin.vertices = local + pos
    v = 0.5 + 0.5 * y
    u = 0.5 + a / (2 * np.pi)
    yy, xx = np.mgrid[1:0:128j, 0:1:192j]
    noise = np.cos(6 * np.pi * xx + 5 * yy) * np.sin(4 * np.pi * xx - 5 * yy)
    if dark:
        color = np.asarray(rgb) + (noise[..., None] + 1) * [2, 3, 5]
    else:
        color = (
            np.asarray(rgb)
            + (1 - yy[..., None]) * [10, 16, -3]
            + noise[..., None] * [6, 3, 2]
        )
    _add(
        scene,
        key,
        skin,
        _mat("Berry skin", color, 0.50 if dark else 0.39),
        np.column_stack((u, v)),
    )
    if seeded:
        grains = []
        rng = np.random.default_rng(len(key) * 31)
        for ring in range(1, 12):
            lat = (ring / 12 - 0.5) * np.pi
            count = max(4, round(19 * np.cos(lat)))
            for i in range(count):
                angle = 2 * np.pi * (
                    i + (0.5 if ring % 2 else 0)
                ) / count + rng.uniform(-0.045, 0.045)
                sy = np.sin(lat + rng.uniform(-0.012, 0.012))
                r = (
                    rx
                    * np.sqrt(1 - sy * sy)
                    * (1 + slope * sy)
                    * (1 + 0.012 * np.cos(3 * angle + 2 * sy))
                )
                grain = trimesh.creation.icosphere(subdivisions=0)
                grain.apply_scale([rx * 0.018, ry * 0.028, rx * 0.012])
                grain.apply_transform(
                    trimesh.transformations.rotation_matrix(
                        np.pi / 2 - angle, [0, 1, 0]
                    )
                )
                grain.apply_translation(
                    pos + [r * np.cos(angle), ry * sy, r * np.sin(angle)]
                )
                grains.append(grain)
        _add(
            scene,
            key + "_seeds",
            trimesh.util.concatenate(grains),
            _mat("Seed", (200, 142, 55), 0.52),
        )
        if leaf_color is not None:
            _leaves(scene, key, pos, rx, ry, leaf_color)
    elif dark:
        t = np.linspace(0, 2 * np.pi, 51)
        rr = rx * (0.16 + 0.046 * np.cos(5 * t))
        mesh, uv = _tube(
            np.column_stack(
                (
                    pos[0] + rr * np.sin(t),
                    pos[1] + ry * 0.27 + rr * np.cos(t),
                    pos[2] + rx * 0.97 + 0.011 * rx * np.cos(5 * t),
                )
            ),
            np.full(len(t), 0.025 * rx),
            8,
        )
        _add(
            scene,
            key + "_calyx",
            mesh,
            _mat("Berry calyx", np.asarray(rgb) * 0.62, 0.63),
            uv,
        )


def _berry_pigment(image, part, dark):
    """Estimate albedo from the illuminated side rather than baked cafe shadows."""
    h, w = image.shape[:2]
    cx, cy = np.asarray(part["center"]) * [w, h]
    rx, ry = np.maximum(2, np.asarray(part["radii"]) * [w, h])
    x0, x1 = max(0, int(cx - rx * 0.70)), min(w, int(cx + rx * 0.70) + 1)
    y0, y1 = max(0, int(cy - ry * 0.70)), min(h, int(cy + ry * 0.70) + 1)
    cut = image[y0:y1, x0:x1]
    reference = np.asarray(part["rgb"], float)
    y, x = np.ogrid[y0:y1, x0:x1]
    interior = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 < 0.50
    if dark:
        matches = (cut.max(axis=2) < 108) & (cut.min(axis=2) > 3)
    else:
        f = cut.astype(float)
        matches = (f[:, :, 0] > 1.65 * f[:, :, 1]) & (f[:, :, 0] > 1.7 * f[:, :, 2])
    samples = cut[interior & matches]
    if len(samples) >= 12:
        luminance = samples @ [0.30, 0.59, 0.11]
        lo, hi = np.quantile(luminance, [0.65, 0.95])
        lit = samples[(luminance >= lo) & (luminance <= hi)]
        reference = 0.25 * reference + 0.75 * np.median(lit, axis=0)
    if dark and max(reference) - min(reference) < 24:
        # A subtle dusty bloom is visible even on the brightest side of dark fruit.
        reference = 0.64 * reference + 0.36 * np.mean(reference) + [0, 2, 8]
    return np.clip(reference * 1.16 + ([1, 2, 3] if dark else [7, 3, 2]), 0, 255)


def add_dessert_toppings(scene, measurements, image):
    """Add editable toppings. The ring depth and unobserved side are inferred."""
    if image.getexif().get(274, 1) != 1:
        image = ImageOps.exif_transpose(image)
    # All landmarks are fractions. Filtering the original 40 MP frame
    # would expand float arrays without adding a useful piping course.
    ratio = min(1.0, 900 / max(image.size))
    if ratio < 1:
        image = image.resize(
            tuple(max(1, round(length * ratio)) for length in image.size),
            Image.Resampling.LANCZOS,
            reducing_gap=3,
        )
    rgb_image = image.convert("RGB")
    width, height = image.size
    rim_x, rim_y, rim_width = measurements["rim"]
    scale = (measurements["foot"][1] - rim_y) * height
    left, top, right, base = measurements["cream_bbox"]
    crop = rgb_image.crop(
        (
            round(left * width),
            round(top * height),
            round(right * width),
            round(base * height),
        )
    )
    color = np.array(measurements["cream_rgb"], dtype=float)
    center = ((left + right) * 0.5 - rim_x) * width / scale
    radius = (right - left) * width / (2 * scale)
    bottom = 1.0
    mound_height = (rim_y - top) * height / scale
    courses, profile = _add_mound(
        scene, np.asarray(crop), color, center, radius, bottom, mound_height
    )
    crumbs = _scatter_crumbs(
        scene, np.asarray(crop), color, center, radius, bottom, mound_height, profile
    )
    photo = np.asarray(rgb_image)
    observed = []
    ring = max(radius * 1.03, rim_width * width / scale)
    for index, part in enumerate(measurements["berries"]):
        cx, cy = part["center"]
        rx, ry = part["radii"]
        rx = rx * width / scale
        ry = ry * height / scale
        x = (cx - rim_x) * width / scale
        dark = part["kind"] == "dark"
        z = np.sqrt(max((ring * 0.30) ** 2, ring**2 - (x - center) ** 2))
        # A higher placement in the photograph suggests the far side of the
        # wreath, not a fruit hovering above the supporting rim.
        elevation = (rim_y - cy) * height / scale - ry
        z *= np.clip(1 - 0.65 * max(0, elevation) / max(radius, 1e-6), 0.60, 1)
        pos = np.array([x, 1 + ry, z])
        pigment = _berry_pigment(photo, part, dark)
        broad_top, leaf_color = (
            _fruit_landmarks(photo, part) if not dark else (False, None)
        )
        seeded = not dark and (ry >= rx * 1.07 or leaf_color is not None)
        if len(measurements["berries"]) == 1:
            pos[1] = max(1 + ry, 1 + (rim_y - cy) * height / scale)
            pos[2] = ring * 0.12
        _berry(
            scene,
            "berry_" + str(index),
            pos,
            rx,
            ry,
            pigment,
            seeded,
            dark,
            broad_top,
            leaf_color,
        )
        observed.append((x, rx, ry, pigment, seeded, dark, broad_top, leaf_color))
    # The visible fruit ring generally continues at the far lip. Infer only
    # when multiple observed fruits support a ring rather than an isolated garnish.
    rear_count = min(3, len(observed) - 1) if len(observed) > 2 else 0
    for i in range(rear_count):
        _, rx, ry, pigment, seeded, dark, broad_top, leaf_color = observed[
            (i * len(observed)) // rear_count
        ]
        angle = np.pi * (1 + (i + 1) / (rear_count + 1))
        pos = np.array(
            [center + ring * np.cos(angle), 1 + ry * 0.94, ring * np.sin(angle)]
        )
        _berry(
            scene,
            "inferred_rear_berry_" + str(i),
            pos,
            rx * 0.92,
            ry * 0.94,
            pigment,
            seeded,
            dark,
            broad_top,
            leaf_color,
        )
    decor = add_measured_decor(scene, measurements, image, radius)
    return {
        "piped_courses": courses,
        "measured_cream_flecks": crumbs,
        "visible_berries": len(observed),
        "inferred_rear_berries": rear_count,
        "observed_decor": decor,
    }
