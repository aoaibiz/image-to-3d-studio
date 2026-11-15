"""Estimate a proper orthographic source frame for an untextured triangle mesh.

The pinned TripoSG inference script exports the decoder XYZ coordinates without
camera metadata. A fitted view describes this one image only; gravity, lighting,
the physical camera, and hidden surfaces are not measured.
"""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
import trimesh


def _basis(up, azimuth, elevation, roll):
    vertical = np.eye(3)[abs(up) - 1] * np.sign(up)
    forward = np.eye(3)[abs(up) % 3]
    lateral = np.cross(vertical, forward)
    a, e, r = np.deg2rad([azimuth, elevation, roll])
    direction = (forward * np.cos(a) + lateral * np.sin(a)) * np.cos(
        e
    ) + vertical * np.sin(e)
    right = lateral * np.cos(a) - forward * np.sin(a)
    top = np.cross(direction, right)
    return np.array(
        [
            right * np.cos(r) + top * np.sin(r),
            top * np.cos(r) - right * np.sin(r),
            direction,
        ]
    )


def _matte(source):
    def read(image):
        if image.width * image.height > 40_000_000:
            raise ValueError(
                "Foreground exceeds 40 megapixels. Supply a downscaled RGBA copy for registration."
            )
        image = ImageOps.exif_transpose(image)
        if "A" not in image.getbands():
            raise ValueError(
                "A segmented RGBA foreground with transparent background is required."
            )
        rgba = image.convert("RGBA")
        alpha = rgba.getchannel("A")
        if alpha.getextrema()[0] > 32:
            raise ValueError("Foreground has no transparent background.")
        box = alpha.point(lambda x: 255 if x >= 150 else 0).getbbox()
        if (
            not box
            or min(box[2] - box[0], box[3] - box[1]) < 3
            or max(box[2] - box[0], box[3] - box[1]) < 24
        ):
            raise ValueError(
                "Foreground has insufficient visible detail at the source resolution."
            )
        clipped = (
            box[0] == 0 or box[1] == 0 or box[2] == rgba.width or box[3] == rgba.height
        )
        return rgba.crop(box), clipped

    if isinstance(source, Image.Image):
        return read(source)
    with Image.open(source) as im:
        return read(im)


def _reference(cutout, n):
    size = tuple(max(1, round(s * (n - 8) / max(cutout.size))) for s in cutout.size)
    panel = Image.new("RGBA", (n, n))
    panel.paste(
        cutout.resize(size, Image.Resampling.LANCZOS),
        ((n - size[0]) // 2, (n - size[1]) // 2),
    )
    pixels = np.asarray(panel)
    mask = pixels[:, :, 3] >= 150
    ys, xs = np.where(mask)
    if not len(xs) or min(np.ptp(xs), np.ptp(ys)) < 1:
        raise ValueError("Foreground is too narrow at registration resolution.")
    span = np.array([np.ptp(xs), np.ptp(ys)])
    mid = np.array([xs.min(), ys.min()]) + span / 2
    holes = ndimage.binary_fill_holes(mask) & ~mask
    holes = holes if holes.sum() >= 5 else None
    return pixels[..., :3], mask, holes, span, mid


def _project(points, basis, ref):
    # Explicit small-column sums keep BLAS from allocating a worker team per view.
    xyz = sum(points[:, j, None] * basis[:, j] for j in range(3))
    lo = xyz[:, :2].min(0)
    hi = xyz[:, :2].max(0)
    scale = min(ref[3] / np.maximum(hi - lo, 1e-10))
    xyz[:, :2] = (xyz[:, :2] - (hi + lo) / 2) * [scale, -scale] + ref[4]
    return xyz


def _score(mask, ref):
    observed, holes = ref[1:3]
    iou = np.count_nonzero(mask & observed) / np.count_nonzero(mask | observed)
    blocked = float(mask[holes].mean()) if holes is not None else 0.0
    return iou - 0.08 * blocked, iou, blocked


def _point_mask(points, basis, ref):
    n = ref[1].shape[0]
    xy = _project(points, basis, ref)[:, :2]
    xy = np.clip(np.rint(xy).astype(int), 0, n - 1)
    hit = np.zeros((n, n), dtype=bool)
    hit[xy[:, 1], xy[:, 0]] = True
    return ndimage.binary_closing(hit, structure=np.ones((3, 3), bool))


def _depth(vertices, faces, basis, ref):
    """Exact orthographic closest-surface depth, with bounded raster working sets."""
    n = ref[1].shape[0]
    projected = _project(vertices, basis, ref)
    depth = np.full(n * n, -np.inf, dtype=np.float32)
    for start in range(0, len(faces), 4096):
        tri = projected[faces[start : start + 4096]]
        lo = np.maximum(np.ceil(tri[:, :, :2].min(1)).astype(int), 0)
        hi = np.minimum(np.floor(tri[:, :, :2].max(1)).astype(int), n - 1)
        edges = tri[:, 1:] - tri[:, :1]
        det = edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]
        keep = (hi >= lo).all(axis=1) & (abs(det) > 1e-10)
        if not keep.any():
            continue
        base, edges, det, lo, hi = (
            tri[keep, 0],
            edges[keep],
            det[keep],
            lo[keep],
            hi[keep],
        )
        widths, heights = (hi - lo + 1).T
        cumulative = np.r_[0, np.cumsum(widths * heights)]
        first = 0
        while first < len(base):
            last = (
                np.searchsorted(cumulative, cumulative[first] + 65536, side="right") - 1
            )
            last = max(first + 1, last)
            ids = np.repeat(
                np.arange(first, last), np.diff(cumulative[first : last + 1])
            )
            offset = np.arange(cumulative[first], cumulative[last]) - cumulative[ids]
            x = lo[ids, 0] + offset % widths[ids]
            y = lo[ids, 1] + offset // widths[ids]
            dx, dy = x - base[ids, 0], y - base[ids, 1]
            u = (dx * edges[ids, 1, 1] - dy * edges[ids, 1, 0]) / det[ids]
            v = (dy * edges[ids, 0, 0] - dx * edges[ids, 0, 1]) / det[ids]
            inside = (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1.000001)
            z = base[ids, 2] + u * edges[ids, 0, 2] + v * edges[ids, 1, 2]
            np.maximum.at(depth, y[inside] * n + x[inside], z[inside])
            first = last
    return depth.reshape((n, n))


def _edge_support(depth, ref):
    # Depth discontinuities often coincide with an occlusion in the photograph.
    # Printed patterns, shadows, missing model geometry, and uniform paint make
    # this unreliable on its own. It has at most 0.025 weight in the ranking.
    interior = ndimage.binary_erosion(np.isfinite(depth) & ref[1], iterations=3)
    if interior.sum() < 64:
        return None
    visible = np.where(np.isfinite(depth), depth, 0.0)
    relief = ndimage.maximum_filter(visible, size=3) - ndimage.minimum_filter(
        visible, size=3
    )
    cut = max(0.035, np.percentile(relief[interior], 90))
    boundary = interior & (relief >= cut)
    if boundary.sum() < 10:
        return None
    rgb = ndimage.gaussian_filter(ref[0].astype(float) / 255, sigma=(0.9, 0.9, 0))
    dy, dx = np.gradient(rgb, axis=(0, 1))
    gradient = np.linalg.norm(np.sqrt(dx**2 + dy**2), axis=2)
    threshold = max(0.04, np.percentile(gradient[interior], 80))
    photo_edge = (gradient >= threshold) & interior
    if photo_edge.sum() < 10:
        return None
    near = ndimage.maximum_filter(photo_edge, size=3)
    return float(near[boundary].mean() - near[interior].mean())


def _distance(a, b):
    cosine = (np.sum(a * b) - 1) * 0.5
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def _distinct(ranked, count, degrees):
    result = []
    for item in sorted(ranked, key=lambda item: -item[0]):
        if all(_distance(item[2], other[2]) >= degrees for other in result):
            result.append(item)
            if len(result) == count:
                break
    return result


def _file_digest(value):
    if isinstance(value, (str, Path)):
        with Path(value).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    return None


def _frame(basis):
    return dict(
        zip(("image_right", "image_up", "camera_direction_from_origin"), basis.tolist())
    )


def register_camera(mesh, foreground):
    """Fit a raw GLB/Trimesh to a true RGBA foreground; return JSON-ready evidence."""
    mesh_digest = _file_digest(mesh)
    if isinstance(mesh, (str, Path)):
        mesh = trimesh.load(mesh, force="mesh", process=False)
    if (
        not isinstance(mesh, trimesh.Trimesh)
        or len(mesh.faces) == 0
        or len(mesh.vertices) < 3
    ):
        raise ValueError("Source registration requires a nonempty triangle mesh.")
    if len(mesh.faces) > 2_000_000:
        raise ValueError(
            "Registration exceeds its working-memory limit (2 million faces)."
        )
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    lo, hi = verts.min(0), verts.max(0)
    span = np.max(hi - lo)
    if (
        not np.isfinite(span)
        or span < 1e-12
        or not np.isfinite(mesh.area)
        or mesh.area < 1e-15
    ):
        raise ValueError("Source mesh has invalid or degenerate geometry.")
    center = lo + (hi - lo) / 2
    verts = (verts - center) / span
    cutout, touched = _matte(foreground)
    low = _reference(cutout, 56)
    high = _reference(cutout, 112)
    points, _ = trimesh.sample.sample_surface(mesh, 12000, seed=7021)
    points = (points - center) / span
    trials = []
    seen = set()

    def rough(parameters):
        b = _basis(*parameters)
        key = tuple(np.rint(b.ravel() * 1e5).astype(int))
        if key not in seen:
            seen.add(key)
            score = _score(_point_mask(points, b, low), low)[0]
            trials.append((score, parameters, b))

    for up in (2, 3, 1, -2, -3, -1):
        for yaw in range(0, 360, 30):
            for pitch in (0, -30, 30, -60, 60):
                for roll in (0, -30, 30):
                    rough((up, yaw, pitch, roll))
    for step, separation in ((10, 26), (4, 17)):
        for _, (up, yaw, pitch, roll), _ in _distinct(trials, 6, separation):
            for dy in (-step, 0, step):
                for dp in (-step, 0, step):
                    for dr in (-step, 0, step):
                        rough((up, yaw + dy, pitch + dp, roll + dr))
    finalists = _distinct(trials, 10, 8)
    # Also inspect the strongest substantially different view, even when many
    # near-equivalent perturbations of the leading view occupy the shortlist.
    for additional in _distinct(trials, 6, 50):
        if all(_distance(additional[2], candidate[2]) > 1 for candidate in finalists):
            finalists.append(additional)
    exact = []
    for _, args, b in finalists:
        depth = _depth(verts, mesh.faces, b, high)
        shape_score, iou, blocked = _score(np.isfinite(depth), high)
        support = _edge_support(depth, high)
        exact.append(
            (shape_score + 0.025 * (support or 0), args, b, iou, blocked, support)
        )
    ranked = sorted(exact, key=lambda item: -item[0])
    score, parameters, b, iou, blocked, support = ranked[0]
    alternative = next((r for r in ranked[1:] if _distance(b, r[2]) >= 40), None)
    gap = float(score - alternative[0]) if alternative is not None else None
    ambiguous = gap is None or gap < 0.025
    transform = np.eye(4)
    transform[:3, :3] = b
    return {
        "method": "right-handed orthographic fit to foreground alpha and visible depth boundaries",
        "mesh_file_sha256": mesh_digest,
        "foreground_file_sha256": _file_digest(foreground),
        "camera_frame": _frame(b),
        "upright_transform": transform.tolist(),
        "search_parameters": dict(
            zip(("signed_axis", "azimuth_deg", "elevation_deg", "roll_deg"), parameters)
        ),
        "sample_seed": 7021,
        "surface_samples": len(points),
        "transform_scope": "raw coordinates to source-image up and front; physical gravity is unknown",
        "status": "poor-fit"
        if iou < 0.72
        else ("ambiguous" if ambiguous else "candidate"),
        "requires_visual_review": True,
        "photographic_pose_verified": False,
        "hidden_sides_observed": False,
        "silhouette_iou": round(float(iou), 5),
        "filled_source_hole_fraction": round(float(blocked), 5),
        "visible_depth_edge_support": None if support is None else round(support, 5),
        "alternate_score_gap": None if gap is None else round(gap, 5),
        "alternate_rotation_degrees": None
        if alternative is None
        else round(float(_distance(b, alternative[2])), 2),
        "alternate_silhouette_iou": None
        if alternative is None
        else round(float(alternative[3]), 5),
        "alternate_camera_frame": None
        if alternative is None
        else _frame(alternative[2]),
        "foreground_touches_source_border": bool(touched),
        "coarse_views": len(trials),
        "exact_views": len(exact),
        "raster_resolution": high[1].shape[0],
    }
