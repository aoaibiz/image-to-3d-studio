"""Transfer two geometry-registered inferred views onto an existing native UV atlas."""

import io
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order

from engine import guided_texture as gt
from engine import native_region_paint as native
from engine.codex_photo_observations import (
    METHOD,
    METHOD_V2,
    validate_provider_observations as validate_codex,
)
from engine.finish_export import embed_master, export_finished
from engine.shape_appearance import _display_matrix
from engine.triposg_cpu import digest
from engine.visual_brief import (
    json_sha,
    validate_provider_observations as validate_local,
)
from quality import paint_join, reflected_paint as copy_test


class NativeFullRegistrationError(ValueError):
    """A valid bound proposal cannot register both silhouettes and openings."""


class NativeFullReviewHold(ValueError):
    """Keep the native trial for review; do not try another appearance projection."""


def _registered(mesh, registration):
    matrix = _display_matrix(registration["upright_transform"])
    if matrix is None:
        raise ValueError("Missing native registration transform")
    frame = registration["camera_frame"]
    axes = [
        frame[k] for k in ("image_right", "image_up", "camera_direction_from_origin")
    ]
    if not np.allclose(matrix[:3, :3], axes, atol=1e-6, rtol=0) or not np.allclose(
        matrix[:3, 3], 0
    ):
        raise ValueError("Source camera and registered transform differ")
    result = mesh.copy()
    result.apply_transform(matrix)
    return result


def preflight_native_paint(native_model, registration):
    """Check native format and depth before requesting hidden views; no appearance verdict."""
    path = Path(native_model)
    if (
        registration.get("mesh_file_sha256") != gt._sha(gt._single(path, 48 * gt.MIB))
        or registration.get("status") == "poor-fit"
    ):
        raise ValueError("Registration does not match this native GLB")
    _, _, _, raw = native._native(path)
    extent = _registered(raw, registration).extents
    if not np.isfinite(extent).all() or min(extent[:2]) <= 1e-9:
        raise ValueError("Degenerate registered geometry")
    depth = float(extent[2] / min(extent[:2]))
    return {
        "full_sheet_eligible": True,
        "region_depth_eligible": depth <= 0.65,
        "depth_over_lesser_frontal_span": depth,
    }


def _bound(paths, registration, guide, layout, receipt, orientation):
    shapes = {
        "source": 32,
        "native": 48,
        "model": 48,
        "matte": 32,
        "guide": 8,
        "sheet": 32,
    }
    ids = {k: gt._sha(gt._single(p, shapes[k] * gt.MIB)) for k, p in paths.items()}
    angles = {k: list(v) for k, v in gt.ANGLES.items()}
    if guide != {
        "version": gt.VERSION,
        "model_sha256": ids["model"],
        "guide_sha256": ids["guide"],
        "views": angles,
        "size": gt.SIZE,
    } or (
        layout.get("photo_sha"),
        layout.get("model_sha"),
        layout.get("guide_sha"),
        layout.get("camera"),
    ) != (ids["source"], ids["model"], ids["guide"], angles):
        raise ValueError("Wrong photograph, registered geometry or camera guides")
    if (
        registration.get("mesh_file_sha256"),
        registration.get("foreground_file_sha256"),
    ) != (ids["native"], ids["matte"]) or registration.get("status") == "poor-fit":
        raise ValueError("Wrong native mesh or photograph registration")
    external_id = receipt.get("thread_id") or receipt.get("host_tool_call_id")
    if (
        not external_id
        or receipt.get("state") != "completed"
        or (
            receipt.get("photo_sha256"),
            receipt.get("geometry_sha256"),
            receipt.get("guide_sha256"),
            receipt.get("sheet_sha256"),
        )
        != (ids["source"], ids["model"], ids["guide"], ids["sheet"])
    ):
        raise ValueError("Missing completed, bound generation receipt")
    brief = layout["visual_brief"]
    validator = (
        validate_codex if brief.get("method") in (METHOD, METHOD_V2) else validate_local
    )
    obs = validator(brief, ids["source"])
    evidence = orientation["inputs"]
    if (
        receipt.get("visual_brief_sha256") != json_sha(obs)
        or any(
            evidence.get(k) != v
            for k, v in {
                "brief_sha256": json_sha(obs),
                "source_sha256": ids["source"],
                "model_sha256": ids["model"],
                "foreground_sha256": ids["matte"],
                "sheet_sha256": ids["sheet"],
                "camera_sha256": json_sha(registration["camera_frame"]),
                "angles": angles,
            }.items()
        )
        or orientation.get("report_sha256")
        != json_sha({k: v for k, v in orientation.items() if k != "report_sha256"})
    ):
        raise ValueError(
            "Source, observation or camera differs from the hidden-view review"
        )
    if (
        orientation.get("hold_projection") is not False
        or orientation.get("status") != "needs_review"
    ):
        raise NativeFullReviewHold(
            "Unresolved pose or duplicated source face in hidden-view review"
        )
    return ids, obs, external_id


def _same_shape(mesh, common):
    if len(mesh.faces) != len(common.faces):
        raise ValueError("Native and registered triangle counts differ")
    tol = mesh.extents.max() * 2e-5
    dist, ix = cKDTree(common.triangles_center).query(mesh.triangles_center, workers=1)
    if dist.max() > tol or len(np.unique(ix)) != len(ix):
        raise ValueError("Native and registered triangle positions differ")
    # UV seam splits may reorder vertices. Bound temporary arrays for dense meshes.
    for start in range(0, len(ix), 8192):
        sl = slice(start, start + 8192)
        corners = np.linalg.norm(
            mesh.triangles[sl, :, None] - common.triangles[ix[sl], None], axis=3
        )
        if max(corners.min(axis=2).max(), corners.min(axis=1).max()) > tol:
            raise ValueError("Different native and registered triangle corners")


def _panel_matte(rgb):
    """Estimate material against a nearly uniform neutral background, without the mesh."""
    edges = np.concatenate(
        [a.reshape(-1, 3) for a in (rgb[:8], rgb[-8:], rgb[:, :8], rgb[:, -8:])]
    )
    low, high = np.quantile(edges, [0.01, 0.99], axis=0)
    if min(low) < 244 or max(high - low) > 5:
        raise ValueError("Framing recovery requires a uniform light background")
    diff = np.maximum(low - rgb, rgb - high).max(axis=2)
    weak = diff > 1
    strong = diff > 17
    if strong.sum() < 300:
        raise ValueError("Insufficient independent contrast to calibrate framing")
    classes = np.where(weak, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    classes[strong] = cv2.GC_FGD
    classes[[0, -1], :] = cv2.GC_BGD
    classes[:, [0, -1]] = cv2.GC_BGD
    # Deep background-coloured interiors remain available as genuine openings.
    cavities = ndimage.binary_fill_holes(weak) & ~weak
    classes[ndimage.distance_transform_edt(cavities) >= 3] = cv2.GC_BGD
    cv2.setRNGSeed(1)
    cv2.grabCut(
        rgb,
        classes,
        None,
        np.zeros((1, 65)),
        np.zeros((1, 65)),
        2,
        cv2.GC_INIT_WITH_MASK,
    )
    mask = (classes == cv2.GC_FGD) | (classes == cv2.GC_PR_FGD)
    labels, k = ndimage.label(mask)
    keep = np.bincount(labels.ravel(), minlength=k + 1) >= 8
    keep[0] = False
    return keep[labels]


def _similarity(mask, target):
    """Fit only shared 2-D camera framing: translation, isotropic scale and roll."""

    def box(m):
        y, x = np.nonzero(m)
        if not len(x) or min(np.ptp(x), np.ptp(y)) < 16:
            raise ValueError("Missing calibrated panel silhouette")
        span = np.array([np.ptp(x), np.ptp(y)])
        return np.array([x.min(), y.min()]) + span / 2, span

    center, span = box(mask)
    aim, extent = box(target)
    ratio = extent / span
    if (
        np.any(center - span / 2 < 3)
        or np.any(center + span / 2 > gt.SIZE - 4)
        or np.max(abs(center - aim)) > 0.12 * gt.SIZE
        or min(ratio) < 0.72
        or max(ratio) > 1.34
        or max(ratio) / min(ratio) > 1.12
    ):
        raise ValueError("Panel size, aspect, crop or displacement is incompatible")
    scale = np.dot(span, extent) / np.dot(span, span)
    side = gt.SIZE // 2

    def down(m):
        return cv2.resize(
            m.astype(np.float32), (side, side), interpolation=cv2.INTER_AREA
        )

    fixed, moving = down(target), down(mask)

    def matrix(p):
        roll, x, y, ds = p
        result = cv2.getRotationMatrix2D(tuple(center), roll, scale * ds)
        result[:, 2] += aim - center + [x, y]
        return result

    def objective(p):
        m = matrix(p)
        m[:, 2] *= side / gt.SIZE
        test = cv2.warpAffine(moving, m, (side, side))
        return -np.minimum(test, fixed).sum() / np.maximum(test, fixed).sum()

    t = minimize(
        objective,
        [0, 0, 0, 1],
        method="Powell",
        bounds=[
            (-5, 5),
            (-0.02 * gt.SIZE, 0.02 * gt.SIZE),
            (-0.02 * gt.SIZE, 0.02 * gt.SIZE),
            (0.985, 1.015),
        ],
        options={"maxiter": 8, "xtol": 0.005},
    ).x
    return matrix(t), float(t[0])


def _interior_landmarks(depth, target, rgb, aligned):
    """Compare cross-body depth breaks against independent, coherent colour transitions."""
    rows = np.flatnonzero(target.any(axis=1))
    n = len(rows)
    geometry = np.zeros((n, 5))
    appearance = np.zeros_like(geometry)
    intensity = rgb.mean(axis=2)
    for i, y in enumerate(rows):
        x = np.flatnonzero(target[y] & aligned[y])
        if len(x) < 12:
            continue
        cuts = np.quantile(x, [0.18, 0.30, 0.44, 0.56, 0.70, 0.82]).astype(int)
        for j, (left, right) in enumerate(zip(cuts, cuts[1:])):
            part = x[(x >= left) & (x <= right)]
            if len(part):
                geometry[i, j] = np.median(depth[y, part])
                appearance[i, j] = np.median(intensity[y, part])
    slope = np.gradient(ndimage.gaussian_filter1d(geometry, 2, axis=0), axis=0)
    curve = np.median(abs(np.gradient(slope, axis=0)), axis=1)
    photo = np.median(
        abs(np.gradient(ndimage.gaussian_filter1d(appearance, 1.3, axis=0), axis=0)),
        axis=1,
    )
    curve[:6] = curve[-6:] = 0
    photo[:6] = photo[-6:] = 0
    geo = find_peaks(
        curve,
        distance=max(1, int(0.06 * n)),
        prominence=max(1e-7, np.quantile(curve, 0.90)),
    )[0]
    light = find_peaks(photo, distance=5, prominence=max(1, np.quantile(photo, 0.68)))[
        0
    ]
    geo = geo[(geo > 0.025 * n) & (geo < 0.975 * n)]
    if len(geo) < 2 or not len(light):
        raise ValueError("Insufficient distinct interior geometry landmarks")
    strongest = geo[np.argsort(curve[geo])[-6:]]
    distances = np.min(abs(strongest[:, None] - light[None, :]), axis=1) / n
    matched = strongest[distances <= 0.02]
    if (
        len(matched) < 2
        or np.ptp(matched) < 0.25 * n
        or (distances <= 0.02).mean() < 0.75
    ):
        raise ValueError("Independent interior geometry landmarks disagree")
    return {
        "native_height_fractions": (strongest / n).tolist(),
        "nearest_paint_edge_fraction": distances.tolist(),
    }


def _calibrate_panel(panel, obj, depth):
    # Expanding the framing range must never repair a genuine aperture mismatch.
    if np.any(ndimage.binary_fill_holes(obj) & ~obj):
        raise ValueError(
            "Native openings require the original strict panel registration"
        )
    rgb = np.asarray(panel.resize((gt.SIZE, gt.SIZE), Image.Resampling.LANCZOS))
    gen = _panel_matte(rgb)
    matrix, roll = _similarity(gen, obj)
    size = (gt.SIZE, gt.SIZE)
    aligned = (
        cv2.warpAffine(gen.astype(np.uint8), matrix, size, flags=cv2.INTER_NEAREST) > 0
    )
    painted = cv2.warpAffine(rgb, matrix, size, borderValue=(255, 255, 255))
    score = gt._iou(obj, aligned)
    holes = gt._hole_agreement(obj, aligned)
    if score < 0.94 or holes < 0.65:
        raise ValueError(
            f"Calibrated silhouette {score:.3f}, opening agreement {holes:.3f}"
        )
    bands = []
    for a, b in [(obj, aligned), (obj.T, aligned.T)]:
        bins = np.array_split(np.flatnonzero(a.any(axis=1)), 8)
        bands.extend(gt._iou(a[rows], b[rows]) for rows in bins)

    def errors(a, b):
        edge = a ^ ndimage.binary_erosion(a)
        other = b ^ ndimage.binary_erosion(b)
        return np.quantile(ndimage.distance_transform_edt(~edge)[other], 0.95)

    residual = max(errors(obj, aligned), errors(aligned, obj))
    span = max(np.ptp(axis) for axis in np.nonzero(obj))
    if min(bands) < 0.82 or residual > 0.018 * span:
        raise ValueError("Local calibrated outline or silhouette landmark mismatch")
    landmarks = _interior_landmarks(depth, obj, painted, aligned)
    nearest = ndimage.distance_transform_edt(
        ~aligned, return_distances=False, return_indices=True
    )
    color = painted[tuple(nearest)].astype(np.float32)
    low = cv2.GaussianBlur(color, (0, 0), 0.032 * span)
    return (
        np.clip((color + 8) / (low + 8), 0.36, 1.35),
        ndimage.distance_transform_edt(obj & aligned),
        {
            "method": "bounded_similarity",
            "raw_iou": gt._iou(obj, gen),
            "result_iou": score,
            "opening_agreement": holes,
            "roll_degrees": roll,
            "fit": matrix.tolist(),
            "band_iou_min": min(bands),
            "outline_residual_px95": float(residual),
            "depth_landmarks": landmarks,
        },
    )


def _field(mesh, sheet, points, normals):
    """Sample independent RGB pigment fields and geometric support on native texels."""
    fields, weights = [], []
    stats = {}
    for i, (name, angle) in enumerate(gt.ANGLES.items()):
        b, center, scale, ids, _, depth = gt._view(mesh, angle)
        panel = sheet.crop(
            (i * sheet.width // 2, 0, (i + 1) * sheet.width // 2, sheet.height)
        )
        if not np.any(ids >= 0):
            raise ValueError("Empty registered projection")
        try:
            detail, border, stats[name] = gt._register_panel(panel, ids >= 0)
        except ValueError as err:
            try:
                detail, border, stats[name] = _calibrate_panel(panel, ids >= 0, depth)
            except ValueError as fit_error:
                raise NativeFullRegistrationError(
                    f"{name}: {err}; {fit_error}"
                ) from fit_error
        q = points @ b.T
        xy = (q[:, :2] - center[:2]) * [scale, -scale] + gt.SIZE / 2
        x, y = np.rint(xy).astype(int).clip(0, gt.SIZE - 1).T
        visible = (ids[y, x] >= 0) & (abs(depth[y, x] - q[:, 2]) < 1.7 / scale)
        margin = np.clip((border[y, x] - 3) / 8, 0, 1)
        w = visible * margin * np.clip((normals @ b[2] - 0.18) / 0.65, 0, 1)
        aligned = cv2.warpAffine(
            np.asarray(panel.resize((gt.SIZE, gt.SIZE), Image.Resampling.LANCZOS)),
            np.array(stats[name]["fit"]),
            (gt.SIZE, gt.SIZE),
            borderValue=(255, 255, 255),
        )
        # Remove only a weak directional studio component. Gloss and roughness
        # remain native PBR data; a single view cannot separate all illumination.
        lamp = gt._norm(np.array([[-0.4, 0.66, 1.0]]))[0] @ b
        irradiance = 0.85 + 0.15 * np.clip(normals @ lamp, 0, 1)
        sample = gt._sample(aligned, xy)
        fields.append(
            np.clip(
                (sample + 14.025) * irradiance[:, None] ** (-1 / 2.4) - 14.025, 0, 255
            )
        )
        weights.append(w)
        stats[name]["effective_texels"] = int((w > 0.5).sum())
    return fields, weights, stats


def _cut_faces(cost_a, cost_b, edges, seam_cost):
    """Choose a single material label on each 3D triangle, including across UV seams."""
    n = len(cost_a)
    vertices = np.arange(n)
    start, end = n, n + 1
    a, b = edges.T
    rows = np.concatenate((np.full(n, start), vertices, a, b))
    cols = np.concatenate((vertices, np.full(n, end), b, a))
    values = np.rint(
        64 * np.concatenate((cost_b, cost_a, seam_cost, seam_cost))
    ).astype(np.int32)
    graph = coo_matrix((values, (rows, cols)), shape=(n + 2, n + 2)).tocsr()
    residual = graph - maximum_flow(graph, start, end, method="dinic").flow
    residual.eliminate_zeros()
    chosen = np.zeros(n + 2, bool)
    chosen[
        breadth_first_order(residual, start, directed=True, return_predecessors=False)
    ] = True
    return chosen[:n]


def _choose_surface(mesh, face, fields, weights, original, locked):
    # Face adjacency respects occluding body parts even when they project onto
    # the same point. Rejoin identical world positions split at atlas seams.
    _, index = np.unique(mesh.vertices, axis=0, return_inverse=True)
    edge = trimesh.graph.face_adjacency(faces=index[mesh.faces])
    n = len(mesh.faces)
    counts = np.bincount(face, minlength=n)

    def average(value):
        return np.bincount(face, weights=value, minlength=n) / np.maximum(counts, 1)

    wa, wb = [np.where(locked, 0, w) for w in weights]
    qa, qb = average(wa), average(wb)
    disagreement = average(np.max(abs(fields[0] - fields[1]), axis=1)) / 30
    # Preserve a camera's coherent motif until a physically supported, cheap seam.
    first = _cut_faces(
        3 * (1 - qa) ** 2 + 80 * ((qa < 0.14) & (qb > 0.3)),
        3 * (1 - qb) ** 2 + 80 * ((qb < 0.14) & (qa > 0.3)),
        edge,
        2 + disagreement[edge].mean(axis=1),
    )
    one = first[face]
    chosen = np.where(one[:, None], fields[0], fields[1])
    support = np.where(one, wa, wb)
    quality = average(support)
    delta = average(np.max(abs(chosen - original), axis=1)) / 30
    use = _cut_faces(
        12 * (1 - quality) ** 2 + 100 * (quality < 0.03),
        12 * quality**2 + 100 * (quality > 0.74),
        edge,
        2 + delta[edge].mean(axis=1),
    )
    apply = use[face] & (support > 0.16) & ~locked
    return chosen, apply, one


def _copy_check(mesh, texture, observation=None):
    """Render unlit native albedo in the source-registered frame on the exact mesh."""
    pair = []
    try:
        for angle in ((0, 0), (180, 0)):
            _, _, _, ids, bary, _ = gt._view(mesh, angle)
            valid = ids >= 0
            b = bary[valid]
            uv = np.einsum(
                "ij,ijk->ik",
                np.column_stack((1 - b.sum(1), b)),
                mesh.visual.uv[mesh.faces[ids[valid]]],
            )
            xy = uv * [texture.shape[1], -texture.shape[0]] + [
                -0.5,
                texture.shape[0] - 0.5,
            ]
            rgb = np.zeros((*ids.shape, 3), dtype=np.uint8)
            rgb[valid] = gt._sample(texture, xy)
            pair.append(copy_test._features(rgb, valid))
    except ValueError as err:
        return {"status": "inconclusive", "reason": str(err)}
    first, second = pair
    signal = copy_test._signal(first)
    focal = "focal" in (observation or {}).get("observed_face_cues", {})
    if (
        signal["rms_rgb"] < 3
        or (signal["self_reflection"] >= 0.75 and not focal)
        or signal["self_translation"] >= 0.78
        or signal["gradient_minor_fraction"] < 0.055
    ):
        return {
            "status": "inconclusive",
            "reason": "plain_or_symmetric",
            "signal": signal,
        }
    scores = []
    for mirrored in (False, True):
        score, _, aligned, mask = copy_test._align(first, second, mirrored)
        if aligned is None:
            return {"status": "inconclusive", "reason": "insufficient_common_interior"}
        quadrants = copy_test._supported_quadrants(first[0], aligned, mask)
        scores.append((score, sum(value >= 0.42 for value in quadrants)))
    hold = any(
        score >= 0.5 and score - scores[1 - i][0] >= 0.17 and n >= 2
        for i, (score, n) in enumerate(scores)
    )
    ambiguous_match = all(score >= 0.5 and n >= 2 for score, n in scores)
    return {
        "status": "review_hold" if hold or ambiguous_match else "no_evidence",
        "direct": scores[0][0],
        "reflected": scores[1][0],
        "signal": signal,
    }


def _project(mesh, albedo, matte, camera, sheet):
    before, ids, points, normals, y, x = native.atlas(mesh)
    if not np.array_equal(albedo, before) or not len(y):
        raise ValueError("Empty or mismatched native atlas")
    lock, fit = gt._source_lock(mesh, matte, camera, ids, points)
    # The rasterizer chooses the last covering face. Reversing the order exposes
    # every multiply-addressed UV sample, even when the other face is occluded.
    reversed_ids = gt._raster_bounded(
        gt._uv_pixels(mesh, len(ids)),
        np.ascontiguousarray(mesh.faces[::-1], dtype=np.int32),
        len(ids),
    )[0]
    overlap = (
        (ids >= 0) & (reversed_ids >= 0) & (ids != len(mesh.faces) - 1 - reversed_ids)
    )
    protected = lock | ndimage.binary_dilation(overlap, iterations=1)
    fields, weights, stats = _field(mesh, sheet, points, normals)
    chosen, support, first = _choose_surface(
        mesh, ids[y, x], fields, weights, albedo[y, x].astype(float), protected[y, x]
    )
    if support.sum() < max(32, round(len(y) * 0.005)):
        raise NativeFullReviewHold("No significant registered hidden surface")
    join = paint_join.inspect(mesh, ids, y, x, albedo, fields, weights, support, first)
    updated = albedo.copy()
    updated[y[support], x[support]] = np.rint(chosen[support])
    delta = updated.astype("int16") - albedo.astype("int16")
    changed = np.any(delta[y, x] != 0, axis=1)
    if changed.sum() < 32:
        raise NativeFullReviewHold("No measurable hidden appearance change")
    nearest = ndimage.distance_transform_edt(
        ids < 0, return_distances=False, return_indices=True
    )
    pad = (ids < 0) & ~protected
    updated[pad] = np.clip(
        albedo[pad].astype("int16") + delta[nearest[0][pad], nearest[1][pad]], 0, 255
    )
    if np.any(updated[protected] != albedo[protected]) or np.any(changed & ~support):
        raise ValueError("Native source lock or unsupported UV changed")
    return updated, {
        "pair_registration": stats,
        "paint_join_screen": join,
        "source_camera_iou": fit,
        "visible_front_locked": int(lock.sum()),
        "shared_uv_protected": int(overlap.sum()),
        "locked_changed": 0,
        "unsupported_surface_changed": 0,
        "changed_uv": int(changed.sum()),
        "supported_uv": int(support.sum()),
        "total_uv": len(x),
        "single_view_uv": {
            name: int((support & part).sum())
            for name, part in zip(gt.ANGLES, (first, ~first))
        },
    }


def _save(doc, blob, raw, updated, out):
    raw.visual.material.baseColorTexture = Image.fromarray(updated)
    export_finished(raw, out)
    Image.fromarray(updated).save(out / "texture.png")
    doc["images"][0]["mimeType"] = "image/png"
    other = gt._seal_glb(
        doc, blob, (out / "texture.png").read_bytes(), out / "model.glb"
    )
    embed_master(out / "model.glb", out / "editable.gltf")
    reread, binary, actual, loaded = native._native(out / "model.glb")
    if (
        not np.array_equal(updated, actual)
        or any(doc[k] != reread[k] for k in doc if k not in {"buffers", "bufferViews"})
        or other
        != [
            gt._sha(native._view_bytes(binary, view))
            for i, view in enumerate(reread["bufferViews"])
            if i != reread["images"][0]["bufferView"]
        ]
    ):
        raise ValueError("Reopened native texture, mesh or material differs")
    return loaded, actual, other


def _require_copy_evidence(copy, observation):
    candidates = (
        observation.get("function_context", {})
        .get("interpretation", {})
        .get("reverse_candidates", [])
    )
    distinct = bool(
        set(candidates)
        & {"distinct_working_surface", "display_or_interface", "opening_or_cavity"}
    )
    if copy["status"] == "review_hold" or distinct and copy["status"] != "no_evidence":
        raise NativeFullReviewHold(
            "Opposite side copied the observed face or its distinct function cannot be screened"
        )
    return distinct


def paint_native_full(
    native_model,
    registered_model,
    source_photo,
    foreground,
    registration,
    guide_dir,
    inferred_sheet,
    layout,
    receipt,
    orientation,
    output_dir,
):
    """Transfer both complete views. Return a hash-bound candidate requiring visual review.

    Only NativeFullRegistrationError allows trying a separately gated regional interpretation.
    Input drift, semantic vetoes, lock failures and significant opposite paint copies are hard holds.
    """
    folder = Path(guide_dir)
    paths = dict(
        zip(
            ("native", "model", "source", "matte", "guide", "sheet"),
            map(
                Path,
                (
                    native_model,
                    registered_model,
                    source_photo,
                    foreground,
                    folder / "guides.png",
                    inferred_sheet,
                ),
            ),
        )
    )
    guide_bytes = gt._single(folder / "guide.json", 65536)
    ids, obs, generation = _bound(
        paths, registration, json.loads(guide_bytes), layout, receipt, orientation
    )
    common, *_ = gt._model(paths["model"])
    doc, blob, old, raw = native._native(paths["native"])
    mesh = _registered(raw, registration)
    _same_shape(mesh, common)
    guide_image = gt._image(gt._single(paths["guide"], 8 * gt.MIB), 2 * gt.SIZE**2)
    if not np.array_equal(np.asarray(guide_image), gt._guide_sheet(common)):
        raise ValueError("Guide pixels differ from the registered geometry")
    sheet = gt._image(gt._single(paths["sheet"], 32 * gt.MIB), 12_000_000)
    if (
        sheet.mode != "RGB"
        or min(sheet.size) < 256
        or abs(sheet.width / sheet.height - 2) >= 0.08
    ):
        raise ValueError("Expected an opaque two-panel sheet")
    fg = gt._single(paths["matte"], 32 * gt.MIB)
    with Image.open(io.BytesIO(fg)) as image:
        if (
            image.mode != "RGBA"
            or image.format != "PNG"
            or image.width * image.height > 40000000
            or image.n_frames != 1
        ):
            raise ValueError("Expected a bounded segmented foreground")
    mask = gt._matte(io.BytesIO(fg), gt.SIZE)[:, :, 3] >= 150
    updated, info = _project(mesh, old, mask, registration["camera_frame"], sheet)
    angles = info["paint_join_screen"]["angles_with_new_edges"]
    if len(angles) >= 2:
        raise NativeFullReviewHold(
            "塗装を切り替えた位置に元の模様にない境界が複数の角度で現れました。"
            " 確認角度 " + ", ".join(angles)
        )
    with gt._stage(Path(output_dir)) as stage:
        reopened, actual, other = _save(doc, blob, raw, updated, stage)
        copy = _copy_check(_registered(reopened, registration), actual, obs)
        distinct = _require_copy_evidence(copy, obs)
        if any(
            digest(path) != ids[k] for k, path in paths.items()
        ) or guide_bytes != gt._single(folder / "guide.json", 65536):
            raise ValueError("Input changed during the transfer")
        info.update(
            method="native-full-guided-v2",
            appearance="needs_review",
            source_image_sha256=ids["source"],
            source_glb=ids["native"],
            registered_sha=ids["model"],
            output_glb=digest(stage / "model.glb"),
            guide_sha=ids["guide"],
            sheet_sha=ids["sheet"],
            visual_brief_sha256=json_sha(obs),
            hidden_layout_sha256=orientation["report_sha256"],
            observation_method=obs["method"],
            generation_id=generation,
            source_rear_observed=False,
            opposite_function_possible=distinct,
            copy_screen=copy,
            non_base_view_sha256=other,
            normals_uv_nodes_material_and_all_other_buffers_bitwise_identical=True,
            interpretation="Inferred hidden colour and detail, measured only against the registered guides.",
        )
        gt._json(stage / "full-paint.json", info)
    return info
