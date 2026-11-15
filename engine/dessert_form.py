"""Construct a tapered dessert glass and a continuous, spoon-folded filling from image measurements."""

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, map_coordinates


def _smooth(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def _angle_delta(angle, center):
    return (angle - center + np.pi) % (2 * np.pi) - np.pi


def _mix(paint, pigment, coverage):
    return paint + (np.asarray(pigment, dtype=np.float32) - paint) * coverage[..., None]


def _interior_pieces(image, measured):
    """Measure localized red and biscuit-colored islands within the tapered glass.

    Work in a rectified interior strip so perspective and the narrowing wall do
    not bias component area. Only ellipse parameters and color samples survive.
    """
    if image is None:
        return []
    if tuple(measured["image_size"]) != image.size:
        raise ValueError("interior measurements refer to a different source image")
    scale = min(1.0, 1600 / max(image.size))
    width, height = (max(1, round(n * scale)) for n in image.size)
    sample = image.resize((width, height), Image.Resampling.LANCZOS) if scale < 1 else image
    photo = np.asarray(sample.convert("RGB"))
    rim, foot = measured["rim"], measured["foot"]
    profile = np.asarray(measured["glass_profile"])
    across, level = np.meshgrid(
        np.linspace(-0.91, 0.91, 176), np.linspace(0.98, 0.04, 192)
    )
    half = np.interp(level, profile[:, 0], profile[:, 1])
    cx = foot[0] + level * (rim[0] - foot[0])
    cy = foot[1] + level * (rim[1] - foot[1])
    strip = cv2.remap(
        photo,
        ((cx + across * half) * width).astype("float32"),
        (cy * height).astype("float32"),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    strip = cv2.GaussianBlur(strip, (5, 5), 1.3)
    lab = cv2.cvtColor(strip, cv2.COLOR_RGB2LAB).astype("float32")
    # Compare against the lighter mousse present in the same part of the glass.
    pale = np.quantile(lab[:, :, 0], 0.72, axis=1)[:, None]
    red = lab[:, :, 1] - np.quantile(lab[:, :, 1], 0.30, axis=1)[:, None]
    yellow = lab[:, :, 2] - np.quantile(lab[:, :, 2], 0.30, axis=1)[:, None]
    rgb = strip.astype("float32")
    r, g, b = rgb.transpose(2, 0, 1)
    dimmer = pale - lab[:, :, 0]
    fruit = (red > 6) & (r > g * 1.14) & (r > b * 1.24) & (dimmer > 7)
    biscuit = (yellow > 5) & (g > b * 1.17) & (red < 14) & (dimmer > 7) & ~fruit
    pieces = []
    for kind, mask in [("fruit", fruit), ("biscuit", biscuit)]:
        mask = cv2.morphologyEx(
            mask.astype("uint8"), cv2.MORPH_OPEN, np.ones((3, 3), "uint8")
        )
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        candidates = sorted(
            range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True
        )
        for idx in candidates[:4]:
            _, _, w, h, area = stats[idx]
            if not (
                17 <= w <= 131
                and 13 <= h <= 64
                and area >= 250
                and area >= 0.35 * w * h
            ):
                continue
            ix, iy = np.rint(centers[idx]).astype(int)
            if not 0.15 < level[iy, ix] < 0.94:
                continue
            left = max(0, ix - w // 2)
            right = min(across.shape[1] - 1, ix + w // 2)
            angular_radius = (
                np.arcsin(across[iy, right]) - np.arcsin(across[iy, left])
            ) / 2
            observed = np.median(rgb[labels == idx], axis=0)
            # Darkening behind glass is not itself the paint color: use the
            # hue of this fruit, with the detected topping pigments as a prior.
            if kind == "fruit":
                red_fruit = [
                    p["rgb"] for p in measured.get("berries", []) if p["kind"] == "red"
                ]
                if red_fruit:
                    observed = 0.55 * observed + 0.45 * np.median(red_fruit, axis=0)
            else:
                observed += 0.15 * (255 - observed)
            calibrated_layers = [
                layer for layer in measured["layers"] if "raw_rgb" in layer
            ]
            if calibrated_layers:
                # Use a global white-balance estimate rather than the single
                # band's palette: a host mousse can be much lighter than its fruit.
                gain = np.median(
                    [
                        (np.asarray(layer["rgb"]) + 12)
                        / (np.asarray(layer["raw_rgb"]) + 12)
                        for layer in calibrated_layers
                    ],
                    axis=0,
                )
                observed = (observed + 12) * np.clip(
                    gain, 0.85, [1.34, 1.58, 1.95]
                ) - 12
            if kind == "biscuit":
                # Nut cream under warm lighting can look like toast until its
                # white balance is restored. Do not paint olive scraps over it.
                collar = measured.get("collar")
                if observed[0] < observed[1] + 17:
                    continue
                if collar:
                    observed = 0.65 * observed + 0.35 * np.asarray(collar["rgb"])
            pieces.append(
                dict(
                    kind=kind,
                    area=int(area),
                    angle=float(np.arcsin(across[iy, ix])),
                    height=float(level[iy, ix]),
                    half_angle=float(angular_radius),
                    half_height=float(0.47 * h / level.shape[0]),
                    rgb=np.clip(observed, 0, 255).round().astype(int).tolist(),
                )
            )
    # Sparse front evidence should stay sparse inside the mousse. A speck of
    # shadow cannot become a second whole berry or a freckle on every side.
    largest = {}
    for item in pieces:
        largest[item["kind"]] = max(largest.get(item["kind"], 0), item["area"])
    return [p for p in pieces if p["area"] >= 0.49 * largest[p["kind"]]]


def _piece_pigment(paint, angle, level, piece, tilt):
    x = _angle_delta(angle, piece["angle"]) / piece["half_angle"]
    y = (level - piece["height"]) / piece["half_height"] - tilt * x
    x = x / (1 + 0.16 * np.clip(y, -1, 1)) - 0.11 * np.sin(2.8 * y + tilt)
    distance = np.hypot(x, y) + 0.07 * np.sin(3 * y + 2 * x)
    grain = np.sin(8 * x - 3 * y) * np.cos(5 * y + x)
    if piece["kind"] == "fruit":
        core = np.exp(-(((x - 0.11 * y) / 0.28) ** 2) - ((y - 0.18) / 0.65) ** 2)
        fibers = np.maximum(0, np.cos(29 * np.arctan2(x, y - 0.8) + 5 * distance)) ** 6
        shading = (
            31 * core
            + 11 * fibers * np.exp(-(((distance - 0.55) / 0.35) ** 2))
            + 5 * grain
        )
        # Only the uncovered portion meets the glass. The edge beneath the
        # spoonful loses contrast gradually, not as a closed outline.
        coverage = _smooth((1 - distance) / 0.28) * _smooth(
            (y + 0.54 * x + 0.82) / 0.95
        )
    else:
        shading = 13 * grain - 12 * _smooth((distance - 0.67) / 0.34)
        coverage = _smooth((1 - distance + 0.10 * grain) / 0.23)
    pigment = np.asarray(piece["rgb"]) + shading[..., None] * [1, 0.82, 0.62]
    return _mix(paint, pigment, 0.88 * coverage)


def _filling_texture(layers, pieces):
    # The whole wall samples a single periodic field. The same pigment flows
    # through both sides of every interface; caps cannot form shiny white rims.
    angle, level = np.meshgrid(
        np.linspace(-np.pi, np.pi, 768, dtype="float32"),
        np.linspace(1, 0, 640, dtype="float32"),
    )
    rng = np.random.default_rng(43)
    grid = gaussian_filter(rng.normal(size=(18, 39)), 0.55, mode="wrap")
    noise = map_coordinates(
        grid,
        [level * 18, (angle + np.pi) * 39 / (2 * np.pi)],
        mode="grid-wrap",
        order=3,
    ).astype("float32")
    paint = (
        np.broadcast_to(layers[0]["rgb"], (*angle.shape, 3)).astype("float32").copy()
    )
    for index in range(1, len(layers)):
        lower, upper = layers[index - 1], layers[index]
        boundary = 0.5 * (lower["top"] + upper["bottom"])
        thickness = min(lower["top"] - lower["bottom"], upper["top"] - upper["bottom"])
        amplitude = np.clip(thickness * 0.38, 0.006, 0.049)
        phase = index * 2.399963
        # Rotate the local coordinates of two spoon marks. Unlike a repeated
        # sine band, this can leave an overhanging tongue of one cream in another.
        a, y = angle.copy(), level.copy()
        for offset, spin in [(0, 1.75), (2.71, -1.55)]:
            dx = _angle_delta(a, phase + offset) / 0.53
            dy = (y - boundary) / (amplitude * 1.7)
            bend = spin * np.exp(-0.88 * (dx * dx + dy * dy))
            c, s = np.cos(bend), np.sin(bend)
            a += 0.53 * ((c - 1) * dx - s * dy)
            y += amplitude * 1.7 * (s * dx + (c - 1) * dy)
        weave = (
            0.48 * np.sin(2 * a + phase)
            + 0.31 * np.cos(3 * a - 0.6 * phase)
            + 0.24 * np.sin(a - phase)
            + 0.19 * noise
        )
        softness = amplitude * (
            0.38 + 0.77 * (0.5 + 0.5 * np.sin(a + 0.76 * phase)) ** 2
        )
        fraction = _smooth(0.5 + (y - boundary - amplitude * weave) / (2 * softness))
        paint = _mix(paint, upper["rgb"], fraction)
    for index, piece in enumerate(pieces):
        paint = _piece_pigment(paint, angle, level, piece, 0.21 * np.cos(index * 1.7))
        # Backside is inferred. Vary burial, orientation, and size so a front
        # fragment is not mirrored round the glass as a second photograph.
        if index < 2:
            hidden = {
                **piece,
                "angle": piece["angle"] + 2.74 + 0.27 * index,
                "half_height": 0.74 * piece["half_height"],
                "height": piece["height"] + 0.17 * piece["half_height"],
            }
            paint = _piece_pigment(paint, angle, level, hidden, -0.46)
    paint += (noise * 2.8)[..., None]
    paint[:, -1] = paint[:, 0]
    return Image.fromarray(np.clip(paint, 0, 255).round().astype("uint8"))


def _revolve(points, material):
    mesh = trimesh.creation.revolve(np.asarray(points), sections=96)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    if mesh.volume < 0:
        mesh.invert()
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)), material=material
    )
    return mesh


def _filling_shell(radius, texture, wall):
    phi = np.linspace(-np.pi, np.pi, 145)
    height = np.linspace(0.047, 0.997, 50)
    a, y = np.meshgrid(phi, height)
    r = radius(y) - wall * 1.16
    vertices = np.column_stack(
        ((r * np.sin(a)).ravel(), y.ravel(), (r * np.cos(a)).ravel())
    )
    normals = np.column_stack(
        (np.sin(a).ravel(), -radius.derivative()(y).ravel(), np.cos(a).ravel())
    )
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    uv = np.column_stack((((a + np.pi) / (2 * np.pi)).ravel(), y.ravel()))
    n = len(phi)
    ix = (np.arange(len(height) - 1)[:, None] * n + np.arange(n - 1)).ravel()
    faces = np.vstack(
        (
            np.column_stack((ix, ix + 1, ix + n)),
            np.column_stack((ix + 1, ix + n + 1, ix + n)),
        )
    )
    for ring, sign in [(0, -1), (len(height) - 1, 1)]:
        rim = ring * n + np.arange(n - 1)
        offset = len(vertices)
        center = offset + len(rim)
        vertices = np.vstack((vertices, vertices[rim], [0, height[ring], 0]))
        normals = np.vstack((normals, np.tile([0, sign, 0], (n, 1))))
        uv = np.vstack((uv, uv[rim], [0.5, height[ring]]))
        tri = np.column_stack(
            (
                np.full(n - 1, center),
                offset + np.arange(n - 1),
                offset + (np.arange(n - 1) + 1) % (n - 1),
            )
        )
        faces = np.vstack((faces, tri if sign > 0 else tri[:, ::-1]))
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, vertex_normals=normals, process=False
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.PBRMaterial(
            name="blended cream and fruit",
            baseColorTexture=texture,
            metallicFactor=0,
            roughnessFactor=0.47,
        ),
    )
    return mesh


def _disc_radius(image, measured):
    """Look for a coherent reflected lip below the bowl; reject a lone table edge."""
    if image is None:
        return None
    preview = image.convert("L")
    preview.thumbnail((768, 768))
    w, h = preview.size
    gray = cv2.GaussianBlur(np.asarray(preview, dtype="float32"), (0, 0), 0.8)
    cx, fy, half_width = measured["foot"]
    foot = half_width * w
    glass_height = (fy - measured["rim"][1]) * h
    angle = np.linspace(-0.32, np.pi + 0.32, 91, dtype="float32")
    offsets = np.linspace(-0.16, 0.06, 15) * glass_height
    shifts, heights, ratios = np.meshgrid(
        np.linspace(-0.22, 0.22, 5) * foot,
        offsets,
        np.linspace(0.24, 1.02, 18),
        indexing="ij",
    )
    shifts, heights, ratios = [v.reshape(-1, 1) for v in (shifts, heights, ratios)]
    nx = np.broadcast_to(np.cos(angle), (len(shifts), len(angle)))
    ny = np.sin(angle) / ratios
    length = np.hypot(nx, ny)
    nx, ny = nx / length, ny / length
    best_score, best_radius = 0.0, None
    for radius in np.arange(
        foot * 1.30, min(measured["rim"][2] * w * 1.18, foot * 2.9), 1.3
    ):
        x = cx * w + shifts + radius * np.cos(angle)
        y = fy * h + heights + radius * ratios * np.sin(angle)

        def sample(offset):
            col = np.rint(x + offset * nx).astype(int).clip(0, w - 1)
            row = np.rint(y + offset * ny).astype(int).clip(0, h - 1)
            return gray[row, col]

        ridge = sample(0) - 0.5 * (sample(-2.8) + sample(2.8))
        clipped = np.clip(ridge, -13, 72)
        thirds = [
            segment.mean(axis=1) for segment in np.array_split(clipped, 3, axis=1)
        ]
        score = 0.55 * clipped.mean(axis=1) + 0.45 * np.min(thirds, axis=0)
        supported = ((ridge > 3).mean(axis=1) >= 0.72) & (y.max(axis=1) < h - 2)
        score = np.where(supported, score, 0)
        confidence = float(score.max())
        if confidence > max(8.0, best_score):
            best_score, best_radius = confidence, float(radius / w)
    return best_radius


def _flakes(collar, scale, rim_radius):
    outer = max(rim_radius, collar["half_width"] * scale)
    rng = np.random.default_rng(6819)
    group = []
    for row in range(4):
        rc = outer * (0.89 - 0.15 * row)
        for i in range(31 - row * 4):
            angle = (
                2 * np.pi * (i + row * 0.49 + rng.uniform(-0.17, 0.17)) / (31 - row * 4)
            )
            aa = np.linspace(0, 2 * np.pi, 14, endpoint=False)
            long = outer * rng.uniform(0.098, 0.153)
            x = np.cos(aa) * long * rng.uniform(0.90, 1.10, len(aa))
            z = np.sin(aa) * long * rng.uniform(0.62, 0.85)
            y = 0.009 * (np.cos(2 * aa + angle) + 1) - 0.17 * x
            top = np.vstack(([0, -0.003, 0], np.column_stack((x, y, z))))
            pts = np.vstack((top, top - [0, 0.004, 0]))
            n = len(aa)
            k = 1 + np.arange(n)
            nxt = 1 + np.arange(1, n + 1) % n
            faces = np.vstack(
                (
                    np.column_stack((np.zeros(n, int), nxt, k)),
                    np.column_stack((np.full(n, n + 1), k + n + 1, nxt + n + 1)),
                    np.column_stack((k, nxt, nxt + n + 1)),
                    np.column_stack((k, nxt + n + 1, k + n + 1)),
                )
            )
            mesh = trimesh.Trimesh(pts, faces, process=False)
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(-angle, [0, 1, 0])
            )
            mesh.apply_translation(
                [rc * np.cos(angle), 1 + 0.006 * row, rc * np.sin(angle)]
            )
            group.append(mesh)
    mesh = trimesh.util.concatenate(group)
    pigment = tuple(int(v) for v in collar["rgb"])
    mat = trimesh.visual.material.PBRMaterial(
        name="measured flake collar",
        baseColorTexture=Image.new("RGB", (2, 2), pigment),
        metallicFactor=0,
        roughnessFactor=0.62,
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)), material=mat
    )
    return mesh


def add_dessert_glass(scene, measured, image=None):
    """Add editable vessel and filling, measured in glass heights; return provenance."""
    width, height = measured["image_size"]
    scale = width / ((measured["foot"][1] - measured["rim"][1]) * height)
    profile = np.asarray(measured["glass_profile"], dtype=float)
    radii = profile[:, 1] * scale
    # Photographic endpoints can be on a highlight rather than at the extreme.
    # Hold the nearest measured radius constant over the unmeasured last pixels.
    y = np.unique(np.r_[0.0, profile[:, 0], 1.0])
    radius = PchipInterpolator(y, np.interp(y, profile[:, 0], radii))
    wall = min(0.008, float(radii.min()) * 0.055)
    glass = trimesh.visual.material.PBRMaterial(
        name="clear glass",
        baseColorTexture=Image.new("RGBA", (2, 2), (236, 247, 249, 12)),
        alphaMode="BLEND",
        doubleSided=True,
        metallicFactor=0,
        roughnessFactor=0.12,
    )
    wall_y = np.linspace(0.028, 1, 75)
    contour = [(radius(v), v) for v in wall_y] + [
        (radius(v) - wall, v) for v in wall_y[::-1]
    ]
    scene.add_geometry(
        _revolve(contour + contour[:1], glass),
        geom_name="dessert_glass",
        node_name="dessert_glass",
    )
    foot = float(radius(0.028))
    bottom = [
        (0, 0),
        (0.85 * foot, 0),
        (foot, 0.008),
        (foot, 0.028),
        (foot - wall, 0.045),
        (0, 0.045),
    ]
    scene.add_geometry(
        _revolve(bottom, glass), geom_name="glass_base", node_name="glass_base"
    )
    # A rounded lip and a thick heel catch light even without environment
    # maps in a basic GLB viewer. Alpha is part of the texture for OBJ export.
    edge = trimesh.visual.material.PBRMaterial(
        name="glass lip",
        baseColorTexture=Image.new("RGBA", (2, 2), (224, 239, 244, 92)),
        alphaMode="BLEND",
        metallicFactor=0,
        roughnessFactor=0.16,
    )
    for name, level, rr in [
        ("glass_lip", 0.997, float(radius(0.997))),
        ("glass_heel", 0.020, foot),
    ]:
        t = np.linspace(0, 2 * np.pi, 10)
        edge_profile = np.column_stack(
            (rr + wall * 0.48 * np.cos(t), level + wall * 0.48 * np.sin(t))
        )
        scene.add_geometry(_revolve(edge_profile, edge), geom_name=name, node_name=name)
    disc_width = _disc_radius(image, measured)
    if disc_width is not None:
        disc = disc_width * scale
        plate = [
            (0, 0),
            (0.90 * disc, 0),
            (0.985 * disc, 0.002),
            (disc, 0.007),
            (0.995 * disc, 0.011),
            (0.965 * disc, 0.015),
            (0.76 * disc, 0.016),
            (0, 0.016),
        ]
        scene.add_geometry(
            _revolve(plate, glass),
            geom_name="clear_glass_foot",
            node_name="clear_glass_foot",
        )
        t = np.linspace(0, 2 * np.pi, 12)
        lip = np.column_stack(
            (disc + wall * 0.34 * np.cos(t), 0.008 + wall * 0.31 * np.sin(t))
        )
        scene.add_geometry(
            _revolve(lip, edge),
            geom_name="foot_reflection",
            node_name="foot_reflection",
        )
    pieces = _interior_pieces(image, measured)
    texture = _filling_texture(measured["layers"], pieces)
    scene.add_geometry(
        _filling_shell(radius, texture, wall),
        geom_name="spooned_filling",
        node_name="spooned_filling",
    )
    if measured.get("collar"):
        scene.add_geometry(
            _flakes(measured["collar"], scale, float(radius(1))),
            geom_name="cereal_collar",
            node_name="cereal_collar",
        )
    return {
        "profile": profile.tolist(),
        "layers": measured["layers"],
        "foot_disc_half_width": disc_width,
        "inclusions": pieces,
        "unobserved_surfaces": "Rear pigment, spoon marks and circular sections inferred.",
    }
