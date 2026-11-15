"""Conservative image-conditioned reconstruction for front-facing standing figures.

Silhouette does not prove hidden depth, anatomy, or colour. Return None when
landmarks cannot be supported; callers must not present a preview as a scan.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d, label, sobel
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.measure import marching_cubes
from skimage.morphology import skeletonize
import trimesh


def _runs(row):
    x = np.flatnonzero(row)
    return [v for v in np.split(x, np.flatnonzero(np.diff(x) > 1) + 1) if len(v)]


def _measure(rgba):
    if rgba.mode != "RGBA" or rgba.getextrema()[3][0] == 255:
        return None
    original = rgba.size
    im = rgba.copy()
    im.thumbnail((512, 512), Image.Resampling.LANCZOS)
    arr = np.asarray(im)
    mask = arr[:, :, 3] >= 128
    labels, count = label(mask)
    if not count:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    mask = labels == np.argmax(sizes)
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    h = y1 - y0
    if h < 80 or (x1 - x0) < h * 0.25 or (x1 - x0) > h * 1.5:
        return None

    def at(frac):
        return int(round(y0 + h * frac))

    # Head centre derives from the head's silhouette, not the tail-inclusive bbox.
    head_y = at(0.24)
    head_runs = sorted(_runs(mask[head_y]), key=len, reverse=True)
    if not head_runs or len(head_runs[0]) < 0.18 * h:
        return None
    head_run = head_runs[0]
    left, right = int(head_run[0]), int(head_run[-1])
    mid = (left + right) / 2
    hw = (right - left) / 2
    # Topmost contour in each head column identifies a pair of distinct ears.
    top = []
    for x in range(left, right + 1):
        hit = np.flatnonzero(mask[y0 : at(0.26), x])
        top.append(y0 + hit[0] if len(hit) else at(0.26))
    top = gaussian_filter1d(np.asarray(top, dtype=float), max(1, h * 0.006))
    peaks, _ = find_peaks(
        -top, prominence=max(2, h * 0.012), distance=max(3, int(hw * 0.55))
    )
    peaks = [int(p + left) for p in peaks if abs((p + left) - mid) > hw * 0.2]
    pairs = [
        (a, b)
        for a in peaks
        for b in peaks
        if a < mid < b and abs((a + b) / 2 - mid) < hw * 0.3
    ]
    if not pairs:
        return None
    ear_l, ear_r = max(pairs, key=lambda p: p[1] - p[0])
    # Split toes/feet: a real background gap below the torso is mandatory.
    foot_y = at(0.95)
    feet = [
        r
        for r in _runs(mask[foot_y])
        if len(r) > h * 0.035 and r[-1] > mid - hw * 1.2 and r[0] < mid + hw * 1.2
    ]
    if len(feet) != 2 or feet[0][-1] >= mid or feet[1][0] <= mid:
        return None
    # Internal intensity gradients near the upper torso are evidence of folded arms.
    rgb = arr[:, :, :3].astype(float)
    gray = rgb.mean(axis=2)
    grad = np.hypot(sobel(gray, axis=0), sobel(gray, axis=1))
    yy, xx = np.indices(mask.shape)
    region = (yy > at(0.38)) & (yy < at(0.58)) & (np.abs(xx - mid) < hw * 0.9) & mask
    strong = (
        grad > max(50, np.quantile(grad[region], 0.80))
        if region.any()
        else np.zeros_like(mask)
    )
    left_edges = int(np.count_nonzero(strong & region & (xx < mid - hw * 0.12)))
    right_edges = int(np.count_nonzero(strong & region & (xx > mid + hw * 0.12)))
    if min(left_edges, right_edges) < max(10, int(h * 0.10)):
        return None
    arm_marks = []
    for side in (-1, 1):
        edge_y, edge_x = np.where(strong & region & ((xx - mid) * side > hw * 0.12))
        # Robust inner and outer quartiles localise the visible folded arm.
        hand_x = float(np.quantile(edge_x, 0.75 if side < 0 else 0.25))
        outer_x = float(np.quantile(edge_x, 0.20 if side < 0 else 0.80))
        hand_y = float(np.median(edge_y))
        arm_marks.append([[outer_x, hand_y], [hand_x, hand_y]])
    # Locate dark fold edges near the centre at the measured arm height.
    # Keep their measured evidence; the physical wrist gap remains explicit.
    dark = gray < min(120, np.quantile(gray[region], 0.25))
    for side, mark in zip((-1, 1), arm_marks):
        near = (
            region
            & dark
            & (np.abs(yy - mark[1][1]) < h * 0.055)
            & ((xx - mid) * side > h * 0.008)
            & ((xx - mid) * side < hw * 0.5)
        )
        near_y, near_x = np.where(near)
        if len(near_x) >= max(8, int(h * 0.025)):
            distance = float(np.quantile(np.abs(near_x - mid), 0.12))
            mark[1][0] = float(mid + side * distance)
            mark[1][1] = float(np.median(near_y))
    # Align approximately symmetric paired arm contours; retain measured asymmetry
    # if the two sides differ substantially.
    arm_symmetry = abs(arm_marks[0][1][1] - arm_marks[1][1][1]) < h * 0.06
    if arm_symmetry:
        mean_y = (arm_marks[0][1][1] + arm_marks[1][1][1]) / 2
        for mark in arm_marks:
            mark[0][1] = mean_y
            mark[1][1] = mean_y
    # The tail is the connected silhouette beyond a robust central torso envelope.
    tail_region = (
        mask & (yy > at(0.42)) & (yy < at(0.88)) & (np.abs(xx - mid) > hw * 1.05)
    )
    lab, n = label(tail_region)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    if not n or counts.max() < h * 0.05:
        return None
    component = lab == np.argmax(counts)
    skeleton = skeletonize(component)
    sy, sx = np.where(skeleton)
    if len(sx) < h * 0.06:
        return None
    side = 1 if np.mean(sx) > mid else -1
    if np.mean((sx - mid) * side) < hw * 0.9:
        return None
    # Trace the farthest geodesic branch from the near-body skeleton endpoint.
    coords = list(zip(sy.tolist(), sx.tolist()))
    lookup = {p: i for i, p in enumerate(coords)}
    rows = []
    cols = []
    cost = []
    for i, (y, x) in enumerate(coords):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dx or dy):
                    continue
                j = lookup.get((y + dy, x + dx))
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    cost.append(float(np.hypot(dx, dy)))
    graph = csr_matrix((cost, (rows, cols)), shape=(len(coords), len(coords)))
    root = int(np.argmin((sx - mid) * side))
    dist, previous = dijkstra(graph, indices=root, return_predecessors=True)
    finite = np.flatnonzero(np.isfinite(dist))
    end = int(finite[np.argmax(dist[finite])])
    path_ids = []
    while end >= 0:
        path_ids.append(end)
        if end == root:
            break
        end = int(previous[end])
    if not path_ids or path_ids[-1] != root:
        return None
    path_ids = path_ids[::-1]
    pick = np.linspace(0, len(path_ids) - 1, min(24, len(path_ids))).astype(int)
    path = np.array([[sx[path_ids[i]], sy[path_ids[i]]] for i in pick])
    # Reject a proximal skeleton segment that hugs the torso edge vertically
    # before the observed outward bend; keep the tail's lateral exit.
    early_end = max(2, len(path) // 2)
    early = path[:early_end]
    low = int(np.argmax(early[:, 1]))
    if low > 0 and early[low, 1] - early[0, 1] > h * 0.08:
        exit_candidates = np.flatnonzero(
            early[: low + 1, 1] >= early[low, 1] - h * 0.025
        )
        if len(exit_candidates):
            path = path[int(exit_candidates[0]) :]
    if len(path) < 4:
        return None
    # Sample head and body silhouette sizes with tail excluded.
    body_y = at(0.70)
    bruns = _runs(mask[body_y])
    center_runs = [r for r in bruns if r[0] <= mid <= r[-1]]
    if not center_runs:
        return None
    bodyrun = center_runs[0]
    body_left = max(int(bodyrun[0]), int(mid - hw * 1.20))
    body_right = min(int(bodyrun[-1]), int(mid + hw * 1.20))
    if body_right - body_left < h * 0.2:
        return None
    # First central transparent gap that persists across several rows is the
    # visible crotch, rather than a fixed fraction of figure height.
    gap_center = float((feet[0].mean() + feet[1].mean()) / 2)
    margin = max(3, int(h * 0.045))
    gap_counts = []
    for y in range(at(0.70), at(0.98)):
        row = mask[
            y,
            max(0, int(gap_center) - margin) : min(
                mask.shape[1], int(gap_center) + margin
            ),
        ]
        gap_counts.append(np.count_nonzero(~row) >= max(3, int(h * 0.01)))
    persistence = max(3, int(h * 0.012))
    crotch = None
    for i in range(len(gap_counts) - persistence + 1):
        if all(gap_counts[i : i + persistence]):
            crotch = at(0.70) + i
            break
    if crotch is None:
        return None
    feet_bottom = []
    for run in feet:
        xmin = max(0, int(run[0] - len(run) * 0.25))
        xmax = min(mask.shape[1], int(run[-1] + len(run) * 0.25) + 1)
        fy, _ = np.where(mask[foot_y:, xmin:xmax])
        feet_bottom.append(int(foot_y + (fy.max() if len(fy) else 0)))
    return dict(
        mask=mask,
        arr=arr,
        original=original,
        shrink=original[0] / im.width,
        y0=int(y0),
        y1=int(y1),
        height=int(h),
        mid=float(mid),
        head_left=left,
        head_right=right,
        head_y=head_y,
        ears=[(ear_l, int(top[ear_l - left])), (ear_r, int(top[ear_r - left]))],
        feet=[(float(r.mean()), bottom) for r, bottom in zip(feet, feet_bottom)],
        foot_widths=[len(r) for r in feet],
        crotch_y=int(crotch),
        body_left=body_left,
        body_right=body_right,
        body_y=body_y,
        tail=path.tolist(),
        tail_side=side,
        arm_edges=[left_edges, right_edges],
        arm_marks=arm_marks,
        arm_symmetry=arm_symmetry,
    )


def build_figure(rgba: Image.Image):
    """Return (scene, evidence), or None if a supported silhouette is not detected."""
    measured = _measure(rgba)
    if measured is None:
        return None
    m = measured
    h = m["height"]
    mid = m["mid"]
    center_y = (m["y0"] + m["y1"]) / 2

    def xy(p):
        return np.array([(p[0] - mid) / h, (center_y - p[1]) / h])

    head_xy = xy((mid, m["head_y"]))
    head_rx = (m["head_right"] - m["head_left"]) / (2 * h)
    body_xy = xy(((m["body_left"] + m["body_right"]) / 2, m["body_y"]))
    body_rx = (m["body_right"] - m["body_left"]) / (2 * h)
    # Depth is a disclosed prior scaled to measured width, not an observed dimension.
    pitch = 0.009
    axes = [
        np.arange(-0.75, 0.76, pitch, dtype=np.float32),
        np.arange(-0.57, 0.58, pitch, dtype=np.float32),
        np.arange(-0.36, 0.38, pitch, dtype=np.float32),
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij", sparse=True)

    def ell(c, r):
        return (
            np.sqrt(
                ((x - c[0]) / r[0]) ** 2
                + ((y - c[1]) / r[1]) ** 2
                + ((z - c[2]) / r[2]) ** 2
            )
            - 1
        ) * min(r)

    def blend(a, b, k):
        t = np.maximum(k - np.abs(a - b), 0) / k
        return np.minimum(a, b) - t * t * k * 0.25

    # A smooth superellipse keeps measured flank width through the lower torso
    # and places its bottom at the first persistent observed foot gap.
    top_y = xy((mid, m["y0"] + h * 0.36))[1]
    bottom_y = xy((mid, m["crotch_y"]))[1]
    center_body = [body_xy[0], (top_y + bottom_y) / 2, 0]
    br = [body_rx, max(0.08, (top_y - bottom_y) / 2), body_rx * 0.78]
    qx = (x - center_body[0]) / br[0]
    qy = (y - center_body[1]) / br[1]
    qz = z / br[2]
    field = (np.sqrt(qx * qx + qy**4 + qz * qz) - 1) * min(br)
    # A concealed neck joins the visible torso and penetrates the skull; this
    # explicitly avoids a floating head when their ellipsoid tips only touch.
    skull_bottom = head_xy[1] - 0.14
    torso_top = center_body[1] + br[1]
    neck_y = (skull_bottom + torso_top) / 2
    neck_radius_y = max(0.055, abs(skull_bottom - torso_top) / 2 + 0.035)
    skull_overlap = neck_y + neck_radius_y - skull_bottom
    torso_overlap = torso_top - (neck_y - neck_radius_y)
    if min(skull_overlap, torso_overlap) < 0.01:
        return None
    neck = ell(
        [head_xy[0], neck_y, 0.01], [head_rx * 0.52, neck_radius_y, head_rx * 0.52]
    )
    field = blend(field, neck, 0.02)
    feet = [xy(p) for p in m["feet"]]
    for f, measured_width in zip(feet, m["foot_widths"]):
        # Foot centre is raised above the observed bottom so its sole remains
        # close to the detected input silhouette; the ankle joins higher up.
        foot_center = np.array([f[0], f[1] + 0.046])
        attach = np.array(
            [
                center_body[0] + (f[0] - center_body[0]) * 0.75,
                center_body[1] - br[1] * 0.87,
            ]
        )
        foot_width = max(measured_width / h * 0.55, head_rx * 0.20)
        for t in np.linspace(0, 1, 6):
            # Ankle bends gently back into the torso while the toe projects forward.
            q = foot_center * (1 - t) + attach * t
            q[0] += np.sin(np.pi * t) * (center_body[0] - f[0]) * 0.08
            field = blend(
                field,
                ell(
                    [q[0], q[1], 0.055 - 0.035 * t],
                    [foot_width, 0.047 if t < 0.5 else 0.05, 0.090 - 0.022 * t],
                ),
                0.018,
            )
    # Cubic curved arm sweep follows the visible edge landmarks, with a tapered
    # wrist and a wider mitten; local min avoids accumulating an inflated seam.
    for outer, hand in m["arm_marks"]:
        shoulder = xy(outer)
        hand = xy(hand)
        # Preserve separate mittens even when drawn strokes touch in 2D.
        side = -1 if hand[0] < 0 else 1
        hand[0] = side * max(abs(hand[0]), head_rx * 0.20)
        midpoint = (shoulder + hand) / 2
        front = br[2]
        path = np.array(
            [
                [shoulder[0], shoulder[1] + head_rx * 0.18, front * 0.76],
                [shoulder[0], shoulder[1] - head_rx * 0.12, front * 0.95],
                [midpoint[0], midpoint[1] - head_rx * 0.15, front * 1.05],
                [hand[0], hand[1], front * 1.12],
            ]
        )
        from scipy.interpolate import CubicSpline

        spline = CubicSpline(
            np.linspace(0, 1, len(path)), path, axis=0, bc_type="natural"
        )
        arm_field = None
        for t in np.linspace(0, 1, 16):
            c = spline(t)
            radius = head_rx * np.interp(
                t, [0, 0.35, 0.72, 1], [0.25, 0.28, 0.17, 0.17]
            )
            d = ell(c, [radius, radius * 0.9, radius])
            arm_field = d if arm_field is None else np.minimum(arm_field, d)
        field = blend(field, arm_field, 0.022)
    tail = np.array([xy(p) for p in m["tail"]])
    side = m["tail_side"]
    # Smooth the measured skeleton before sweeping a circular section; take a
    # union of samples once so repeated smooth-min does not swell the tail.
    root = np.array([center_body[0] + side * body_rx * 0.85, tail[0, 1]])
    path = np.vstack((root, tail))
    path = gaussian_filter1d(path, sigma=1, axis=0, mode="nearest")
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if length[-1] <= 1e-6:
        return None
    samples = np.linspace(0, length[-1], max(30, int(length[-1] / 0.008)))
    path = np.column_stack([np.interp(samples, length, path[:, i]) for i in range(2)])
    tail_field = None
    for point in path:
        d = ell([point[0], point[1], -0.015], [head_rx * 0.18] * 3)
        tail_field = d if tail_field is None else np.minimum(tail_field, d)
    field = blend(field, tail_field, 0.012)
    vertices, faces, _, _ = marching_cubes(
        field.astype("float32"), level=0, spacing=(pitch,) * 3
    )
    vertices += np.array([a[0] for a in axes])
    body = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if body.volume < 0:
        body.invert()
    scene = trimesh.Scene()
    scene.add_geometry(body, node_name="cast_body", geom_name="cast_body")
    # Head and ears are separate inferred pieces, kept editable for painting.
    head = trimesh.creation.icosphere(subdivisions=4, radius=1)
    head.apply_scale([head_rx, 0.14, head_rx * 0.76])
    head.apply_translation([head_xy[0], head_xy[1], 0.01])
    # Gently raise the front for continuous cheeks and a central muzzle.
    hv = np.asarray(head.vertices).copy()
    dx = hv[:, 0] - head_xy[0]
    dy = hv[:, 1] - head_xy[1]
    front = np.clip((hv[:, 2] - 0.01) / (head_rx * 0.55), 0, 1)
    cheek = np.exp(
        -(((dx - head_rx * 0.30) / (head_rx * 0.35)) ** 2) - ((dy + 0.055) / 0.05) ** 2
    )
    cheek += np.exp(
        -(((dx + head_rx * 0.30) / (head_rx * 0.35)) ** 2) - ((dy + 0.055) / 0.05) ** 2
    )
    muzzle = np.exp(-((dx / (head_rx * 0.42)) ** 4) - ((dy + 0.065) / 0.04) ** 2)
    hv[:, 2] += front * head_rx * (0.035 * cheek + 0.06 * muzzle)
    head.vertices = hv
    scene.add_geometry(head, node_name="head", geom_name="head")
    for p, name in zip(m["ears"], ("ear_left", "ear_right")):
        pt = xy(p)
        base = head_xy[1] + 0.035
        height = max(0.03, pt[1] - base)
        # Rounded, tapered ellipsoid: the base is seated into the skull and
        # the finite-curvature tip follows the observed topmost ear landmark.
        ear = trimesh.creation.icosphere(subdivisions=4, radius=1)
        ev = np.asarray(ear.vertices).copy()
        level = (ev[:, 1] + 1) / 2
        taper = 1 - 0.38 * level
        ev[:, 0] = pt[0] + ev[:, 0] * head_rx * 0.30 * taper
        ev[:, 2] = 0.01 + ev[:, 2] * head_rx * 0.18 * taper
        ev[:, 1] = base + level * height
        local_width = np.maximum(
            head_rx * 0.30 * taper * np.sqrt(np.maximum(1 - (2 * level - 1) ** 2, 0)),
            1e-6,
        )
        central = np.clip(1 - np.abs(ev[:, 0] - pt[0]) / local_width, 0, 1)
        inset = central * np.sin(np.pi * level) ** 2 * (ev[:, 2] > 0.01)
        ev[:, 2] -= head_rx * 0.035 * inset
        ear.vertices = ev
        scene.add_geometry(ear, node_name=name, geom_name=name)
    shrink = m["shrink"]
    evidence = {
        "projection": {
            "center_px": [mid * shrink, center_y * shrink],
            "pixels_per_unit": h * shrink,
            "image_size": list(m["original"]),
            "axes": "+X right, +Y up, +Z front",
            "equation": "x=(pixel_x-center_x)/pixels_per_unit; y=(center_y-pixel_y)/pixels_per_unit",
        },
        "landmarks_px": {
            "head_center": [mid * shrink, m["head_y"] * shrink],
            "ears": [[x * shrink, y * shrink] for x, y in m["ears"]],
            "feet": [[x * shrink, y * shrink] for x, y in m["feet"]],
            "crotch_y": m["crotch_y"] * shrink,
            "arm_edges": (np.array(m["arm_marks"]) * shrink).tolist(),
            "tail_skeleton": (np.array(m["tail"]) * shrink).tolist(),
            "tail_start": (np.array(m["tail"][0]) * shrink).tolist(),
        },
        "detection": {
            "arm_internal_gradient_counts": m["arm_edges"],
            "symmetric_arm_alignment": bool(m["arm_symmetry"]),
            "neck_skull_axis_overlap": float(skull_overlap),
            "neck_torso_axis_overlap": float(torso_overlap),
            "tail_side": m["tail_side"],
            "head_width_pixels": (m["head_right"] - m["head_left"]) * shrink,
            "body_width_pixels": (m["body_right"] - m["body_left"]) * shrink,
        },
        "scope": "front-facing standing figure with two ears, split feet, folded-arm internal edges and a visible side tail",
        "limitations": [
            "depth, back and exact arm anatomy inferred",
            "not a scan; human appearance review required",
        ],
    }
    return scene, evidence
