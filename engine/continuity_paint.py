"""Conservative source-derived pigment transport around closed measured mesh cross sections."""

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage, spatial

from engine import guided_texture as gt
from engine import native_full_paint as full
from engine import native_region_paint as native
from engine.shape_camera import lens_parameters, project_pinhole
from engine.triposg_cpu import digest
from engine.visual_brief import json_sha, validate_photo_observations

ROWS = 128
COLS = 384


class ContinuityNoFit(ValueError):
    """Valid input with insufficient evidence or unsupported local representation."""


def _chart(points, normals):
    """Construct arclength coordinates on measured, star-shaped transverse rings.

    Radial turns with multiple layers or tangential folds have no single-valued chart.
    Their rows (and open/missing sectors) are excluded from the transport.
    """
    height = points[:, 1]
    span = np.ptp(height)
    if span <= 0:
        raise ValueError("Invalid surface height")
    v = (height - height.min()) / span * ROWS
    row = np.clip(v.astype(int), 0, ROWS - 1)
    planar = points[:, [0, 2]]
    az = np.zeros(len(points))
    ok = np.zeros(len(points), bool)
    nb = 48
    ref = np.arange(257)
    theta = ref[:-1] / 256 * (2 * np.pi)
    rays = np.column_stack((np.sin(theta), np.cos(theta)))
    idx = np.argsort(row)
    for r, subset in enumerate(
        np.split(idx, np.searchsorted(row[idx], np.arange(1, ROWS)))
    ):
        if len(subset) < 3 * nb:
            continue
        plane = planar[subset]
        low, high = np.quantile(plane, [0.015, 0.985], axis=0)
        if np.min(high - low) <= 0.006 * span:
            continue
        vec = plane - (high + low) / 2
        dist = np.linalg.norm(vec, axis=1)
        unit = vec / np.maximum(dist[:, None], 1e-8)
        incidence = (normals[subset][:, [0, 2]] * unit).sum(axis=1)
        angle = np.mod(np.arctan2(vec[:, 0], vec[:, 1]) / (2 * np.pi), 1) * nb
        sector = angle.astype(int)
        radii = np.zeros(nb)
        good = incidence > 0.4
        for i in range(nb):
            all_in_sector = sector == i
            samples = dist[all_in_sector & good]
            if len(samples) < 2:
                break
            r10, r50, r90 = np.quantile(samples, [0.1, 0.5, 0.9])
            if r90 - r10 > 0.15 * r50 or good[all_in_sector].mean() < 0.6:
                break
            radii[i] = r50
        if radii.min() <= 0:
            continue
        # Trace the actual convex cross section. A quarter circle is longer
        # than a face of a box with the same angular span around its midpoint.
        try:
            hull = spatial.ConvexHull(vec)
        except spatial.QhullError:
            continue
        plane_normal, intercept = hull.equations[:, :2], hull.equations[:, 2]
        if intercept.max() >= 0:
            continue
        radius = 1 / np.max((rays @ plane_normal.T) / -intercept, axis=1)
        ring = radius[:, None] * rays
        seg = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)
        if seg.min() <= 0 or np.max(seg) > 5 * np.min(seg):
            continue
        arc = np.r_[0, np.cumsum(seg)] / seg.sum()
        t = angle * len(rays) / nb
        local_radius = np.interp(t, ref, np.r_[radius, radius[0]])
        if (
            np.abs(dist[good] - local_radius[good]) > 0.10 * local_radius[good]
        ).mean() > 0.05:
            continue
        az[subset] = np.mod(np.interp(t, ref, arc), 1) * COLS
        ok[subset] = good & (np.abs(dist - local_radius) <= 0.10 * local_radius)
    return v, az, ok


def _grid(v, az, valid, values):
    row = np.clip(v.astype(int), 0, ROWS - 1)
    col = az.astype(int) % COLS
    keys = (row * COLS + col)[valid]
    size = ROWS * COLS
    count = np.bincount(keys, minlength=size).reshape(ROWS, COLS)
    rgb = np.stack(
        [
            np.bincount(keys, weights=channel[valid], minlength=size).reshape(
                ROWS, COLS
            )
            / np.maximum(count, 1)
            for channel in values.T
        ],
        axis=-1,
    )
    # Resolve sub-pixel raster gaps only. A 1-pixel fringe never extends the
    # observed field by a whole ornament.
    nearest = ndimage.distance_transform_edt(count == 0, return_indices=True)
    return rgb[tuple(nearest[1])], nearest[0] < 1.5


def _periods(rgb, coverage):
    """Look for agreeing motifs across a genuinely observed arclength interval.

    The centre band must support enough paired data to compare the same motif at
    two positions, and locally distinctive structures with conflicting appearance
    are left alone.
    """
    weight = ndimage.gaussian_filter1d(coverage.astype(float), 6, axis=1, mode="wrap")
    low = ndimage.gaussian_filter1d(rgb * coverage[:, :, None], 6, axis=1, mode="wrap")
    detail = rgb - low / np.maximum(weight[:, :, None], 1e-8)
    # Glazes of different hues can have identical luminance. Estimate the
    # dominant varying colour axis on measured samples, before searching phases.
    samples = detail[coverage]
    if len(samples) < 3:
        return [(n, np.full(ROWS, -1.0), np.full(ROWS, 255.0)) for n in range(3, 34)]
    _, axes = np.linalg.eigh(samples.T @ samples)
    high = detail @ axes[:, -1]
    gray = rgb @ axes[:, -1]
    cov = ndimage.binary_erosion(coverage, structure=np.ones((1, 5)))
    results = []
    for repeats in range(3, 34):
        shift = round(COLS / repeats)
        a = cov & np.roll(cov, shift, axis=1)
        b = np.roll(high, shift, axis=1)

        def summation(z):
            return (
                ndimage.uniform_filter1d(
                    np.sum(z, axis=1).astype(float), 9, mode="nearest"
                )
                * 9
            )

        count = summation(a)
        aa, bb, ab = (
            summation(a * high**2),
            summation(a * b**2),
            summation(a * b * high),
        )
        corr = ab / np.maximum(np.sqrt(aa * bb), 1e-9)
        err = summation(a * abs(gray - np.roll(gray, shift, axis=1))) / np.maximum(
            count, 1
        )
        qualified = (count >= 9 * max(13, shift * 1.05)) & (
            np.sqrt((aa + bb) / np.maximum(2 * count, 1)) > 4
        )
        results.append((repeats, np.where(qualified, corr, -1), err))
    return results


def _read_rgb(data):
    with Image.open(io.BytesIO(data)) as opened:
        if (
            opened.width * opened.height > 18_000_000
            or getattr(opened, "n_frames", 1) != 1
        ):
            raise ContinuityNoFit("Original photograph exceeds continuity input limit")
        return np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))


def _source_transform(photo, rgba, raw_matte):
    """Map the fitted matte pixels to actual input-photo pixels; verify colour and placement."""
    mask = cv2.erode((rgba[:, :, 3] >= 220).astype(np.uint8), np.ones((5, 5), np.uint8))
    if mask.sum() < 300:
        raise ContinuityNoFit("No opaque source region")
    if raw_matte.shape[:2] == photo.shape[:2]:
        solid = raw_matte[:, :, 3] >= 245
        if solid.sum() >= 300 and np.array_equal(photo[solid], raw_matte[solid, :3]):
            y, x = np.where(raw_matte[:, :, 3] >= 150)
            w, h = x.max() - x.min() + 1, y.max() - y.min() + 1
            factor = (len(rgba) - 12) / max(w, h)
            nw, nh = max(1, round(w * factor)), max(1, round(h * factor))
            left, top = (len(rgba) - nw) // 2, (len(rgba) - nh) // 2
            sx, sy = w / nw, h / nh
            affine = np.array(
                [
                    [sx, 0, x.min() + (0.5 - left) * sx - 0.5],
                    [0, sy, y.min() + (0.5 - top) * sy - 0.5],
                ]
            )
            return affine, {
                "method": "identical original-coordinate opaque pixels",
                "opaque_samples": int(solid.sum()),
                "matte_to_original": affine.tolist(),
            }
    sh, sw = photo.shape[:2]
    scale = min(1.0, 1200 / max(sh, sw))
    iw, ih = max(1, round(sw * scale)), max(1, round(sh * scale))
    small = cv2.resize(photo, (iw, ih), interpolation=cv2.INTER_AREA)
    detector = cv2.SIFT_create(nfeatures=2200)
    key1, d1 = detector.detectAndCompute(
        cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY), mask
    )
    key2, d2 = detector.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), None)
    if d1 is None or d2 is None or len(d2) < 2:
        raise ContinuityNoFit("Insufficient source image landmarks")
    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    chosen = [a for a, b in pairs if a.distance < 0.73 * b.distance]
    if len(chosen) < 8:
        raise ContinuityNoFit("Insufficient matched source image landmarks")
    src = np.float32([key1[a.queryIdx].pt for a in chosen])
    ref = np.float32([key2[a.trainIdx].pt for a in chosen])
    matrix, inliers = cv2.estimateAffinePartial2D(
        src, ref, method=cv2.RANSAC, ransacReprojThreshold=2.5
    )
    if matrix is None or inliers.sum() < 8 or np.linalg.det(matrix[:, :2]) <= 0:
        raise ContinuityNoFit("Unresolved photo-to-matte transform")
    affine = matrix.copy()
    affine *= np.array([sw / iw, sh / ih])[:, None]
    affine[:, 2] += 0.5 * np.array([sw / iw, sh / ih]) - 0.5
    y, x = np.nonzero(mask)
    # Verify more than just sparse keypoints; a repeated logo can match the wrong copy.
    sample = np.column_stack((x, y)) @ affine[:, :2].T + affine[:, 2]
    ix, iy = np.rint(sample).astype(int).T
    valid = (ix >= 0) & (ix < sw) & (iy >= 0) & (iy < sh)
    if valid.mean() < 0.98:
        raise ContinuityNoFit("Source placement is outside the photograph")
    delta = np.max(
        np.abs(
            photo[iy[valid], ix[valid]].astype(float) - rgba[y[valid], x[valid], :3]
        ),
        axis=1,
    )
    if np.median(delta) > 15 or (delta < 32).mean() < 0.75:
        raise ContinuityNoFit("Foreground RGB does not register to original photograph")
    return affine, {
        "method": "colour-verified original-photo landmark fit",
        "matched_inliers": int(inliers.sum()),
        "median_rgb_residual": float(np.median(delta)),
        "fraction_below_32": float((delta < 32).mean()),
        "matte_to_original": affine.tolist(),
    }


def _source_projection(mesh, rgba, frame):
    basis = np.asarray(
        [frame[k] for k in ("image_right", "image_up", "camera_direction_from_origin")],
        float,
    )
    if (
        basis.shape != (3, 3)
        or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-5)
        or np.linalg.det(basis) < 0.999
    ):
        raise ValueError("Invalid source camera axes")
    raw = mesh.vertices @ basis
    lens = lens_parameters(frame)
    projected, scale, offset, _, _ = gt._fit(raw, basis, rgba[:, :, 3] >= 150, lens)
    ids, _, depth = gt._raster_bounded(
        projected, np.asarray(mesh.faces, np.int32), len(rgba)
    )
    agreement = gt._iou(rgba[:, :, 3] >= 150, ids >= 0)
    if agreement < 0.72:
        raise ContinuityNoFit("Source silhouette no longer fits this camera")
    # Distance and view directions always refer to the calibrated source camera.
    center = (raw.max(0) + raw.min(0)) * 0.5 @ basis.T
    focal = 0 if lens is None else lens[0] / np.ptp(raw, axis=0).max()

    def project(points):
        q = project_pinhole(points, center, focal) if focal else points
        direction = (
            gt._norm(center + [0, 0, 1 / focal] - points)
            if focal
            else np.array([0.0, 0.0, 1.0])
        )
        return q[:, :2] * [scale, -scale] + offset, q[:, 2], direction

    return ids, depth, scale, project, agreement


def eligible(observation):
    """Semantic veto is separate from numerical evidence for a transferable material."""
    if not isinstance(observation, dict):
        return False
    context = observation.get("function_context", {}).get("interpretation", {})
    if set(context.get("reverse_candidates", [])) & {
        "distinct_working_surface",
        "display_or_interface",
        "opening_or_cavity",
    }:
        return False
    cues = observation.get("observed_face_cues", {})
    if any(
        cue.get("family") in {"face_or_emblem", "figurative_scene", "lettering_or_logo"}
        for cue in cues.values()
    ):
        return False
    if observation.get("optical_finish_candidate") not in {"matte", "satin", "glossy"}:
        return False
    return "repeated" in cues or (
        not cues and observation.get("layout_candidate") == "plain_or_uncertain"
    )


def _axis_subset(ok, rows, selected):
    """A new horizontal cut through a pattern is not a valid end to the material."""
    counts = np.bincount(rows[ok], minlength=ROWS)
    active = counts >= 64
    qualified = (selected > 0) & active
    if not qualified.any() or np.sum(counts[qualified]) < 0.85 * np.sum(counts):
        return np.zeros_like(ok)
    return ok & qualified[rows]


def _plain_prior(old, measured, direct, ok):
    # A single-colour return uses the observed native PBR pigment. Discernible
    # tonal detail on either the original or the destination is a veto.
    direct = direct & ok
    if direct.sum() < 100:
        return None
    color = np.median(old[direct], axis=0)
    for field, mask, limit in ((measured, direct, 9), (old, direct, 8)):
        median = np.median(field[mask], axis=0)
        err = np.max(np.abs(field[mask].astype(float) - median), axis=1)
        if np.quantile(err, 0.995) > limit:
            return None
    return color


def _transport(points, normals, native_rgb, photo_rgb, observed, locked, observation):
    """Return updates in covered texel order, plus the exact inferred support."""
    if not eligible(observation):
        return None
    v, az, ok = _chart(points, normals)
    rows = np.clip(v.astype(int), 0, ROWS - 1)
    if (ok & observed).sum() < 250 or (ok & ~locked).sum() < 100:
        return None
    if not observation.get("observed_face_cues"):
        color = _plain_prior(native_rgb, photo_rgb, observed, ok)
        if color is None:
            return None
        hidden = ok & ~locked
        far = native_rgb[hidden].astype(float)
        # Preserve an already coloured far-side mark, discontinuous material or
        # deliberate high contrast detail. Only correct a smooth low-chroma drift.
        far_median = np.median(far, axis=0)
        if np.quantile(np.max(abs(far - far_median), axis=1), 0.99) > 9:
            return None
        displacement = color - far_median
        if np.max(abs(displacement)) > 40 or np.ptp(displacement) > 12:
            return None
        updated = native_rgb.copy()
        updated[hidden] = np.clip(np.rint(far + displacement), 0, 255)
        return updated, hidden, {"surface_mode": "inferred_uniform_substrate"}
    observed = observed & ok & locked
    rgb, cov = _grid(v, az, observed, photo_rgb)
    tex, _ = _grid(v, az, observed, native_rgb)
    periods = _periods(rgb, cov)
    source_scores = np.stack([score for _, score, _ in periods])
    source_errors = np.stack([err for _, _, err in periods])
    native_scores = np.stack([score for _, score, _ in _periods(tex, cov)])
    # Distinct harmonics are legitimate for a multi-scale ornament. Prefer the
    # largest demonstrably repeating unit; never declare a half-period from colour alone.
    candidate = (source_scores > 0.72) & (native_scores > 0.69) & (source_errors < 18)
    best = np.max(source_scores, axis=0)
    accepted = candidate & (source_scores >= best[None, :] - 0.085)
    periods_per_row = np.where(
        np.any(accepted, axis=0), 3 + np.argmax(accepted, axis=0), 0
    )
    # Changes of period split a pattern. Require a single stable unit over an
    # entire connected material before applying it round an unseen circumference.
    counts = np.bincount(periods_per_row, minlength=34)
    repeat = int(1 + np.argmax(counts[1:]))
    selected = periods_per_row == repeat
    supported = _axis_subset(ok, rows, selected)
    if supported.sum() < 250:
        return None
    # Independent captured photograph copies vote at the same height and
    # periodic arclength; outlying pigment cannot be carried to the far surface.
    donor = np.flatnonzero(observed & supported)
    target = np.flatnonzero(supported & ~locked)
    if not len(donor) or not len(target):
        return None
    positions = np.column_stack((v, az))
    tree = spatial.cKDTree(positions[donor])
    step = COLS / repeat
    errors, donor_ids = [], []
    for turn in range(repeat):
        q = positions[target].copy()
        q[:, 1] = (q[:, 1] + turn * step) % COLS
        distance, index = tree.query(q, workers=1)
        donor_ids.append(donor[index])
        errors.append(distance)
    ds, indices = np.stack(errors), np.stack(donor_ids)
    indices_sorted = np.argsort(ds, axis=0)
    cols = np.arange(len(target))
    first, second = indices_sorted[0], indices_sorted[1]
    ia, ib = indices[first, cols], indices[second, cols]
    # Compare independent copies including chroma, not only 1-D frequency.
    agree = (
        (ds[second, cols] < 2.5)
        & (np.max(abs(photo_rgb[ia].astype(float) - photo_rgb[ib]), axis=1) <= 28)
        & (np.max(abs(native_rgb[ia].astype(float) - native_rgb[ib]), axis=1) <= 28)
    )
    if agree.mean() < 0.92:
        return None
    # A local conflict on a carried contour cannot be left as speckled holes.
    # Abort the entire material if conflicts form a significant connected patch.
    _, conflict = _grid(v, az, np.isin(np.arange(len(v)), target[~agree]), photo_rgb)
    components, num = ndimage.label(conflict)
    if num and np.bincount(components.ravel())[1:].max() > 0.018 * ROWS * COLS:
        return None
    updated = native_rgb.copy()
    # Select one actual source-side PBR texel, preserving texture contrast.
    # Use the closest central donor where two measured copies agree. Any
    # disagreement is retained only when its native colour already agrees.
    may_keep = ~agree & (
        np.max(abs(native_rgb[target].astype(float) - native_rgb[ia]), axis=1) <= 9
    )
    if (~(agree | may_keep)).mean() > 0.003:
        return None
    target, ia = target[agree], ia[agree]
    updated[target] = native_rgb[ia]
    support = np.zeros_like(ok)
    support[target] = True
    return (
        updated,
        support,
        {
            "surface_mode": "inferred_measured_repetition",
            "measured_period_count": repeat,
            "measured_rows": int(selected.sum()),
            "independent_donor_agreement": float(agree.mean()),
            "mean_source_correlation": float(
                np.mean(source_scores[repeat - 3, selected])
            ),
        },
    )


def prepare_surface(native_model, source_photo, foreground, registration, observation):
    """Bind measured camera rays to true input RGB and immutable native texel IDs."""
    paths = dict(
        native=Path(native_model),
        source=Path(source_photo),
        foreground=Path(foreground),
    )
    sizes = dict(native=48, source=32, foreground=32)
    if any(not p.is_file() or p.is_symlink() for p in paths.values()):
        raise ValueError("Expected a regular native, photo and foreground")
    hashes = {k: digest(p) for k, p in paths.items()}
    if (
        registration.get("mesh_file_sha256") != hashes["native"]
        or registration.get("foreground_file_sha256") != hashes["foreground"]
    ):
        raise ValueError("Surface registration differs from native mesh or foreground")
    canonical = validate_photo_observations(observation, hashes["source"])
    if json_sha(canonical) != json_sha(observation):
        raise ValueError("Observation differs from its normalized source binding")
    if registration.get("status") == "poor-fit":
        raise ContinuityNoFit("Poor fitted camera")
    if any(p.stat().st_size > sizes[k] * gt.MIB for k, p in paths.items()):
        raise ContinuityNoFit("Surface binding exceeds the local input budget")
    data = {
        k: gt._single(p, sizes[k] * gt.MIB) for k, p in paths.items() if k != "native"
    }
    photo = _read_rgb(data["source"])
    with Image.open(io.BytesIO(data["foreground"])) as im:
        if im.width * im.height > 18_000_000:
            raise ContinuityNoFit("Foreground exceeds pixel limit")
        source_matte = np.array(ImageOps.exif_transpose(im).convert("RGBA"))
    matte = gt._matte(io.BytesIO(data["foreground"]), gt.SIZE)
    affine, _ = _source_transform(photo, matte, source_matte)
    try:
        doc, blob, old, raw = native._native(paths["native"])
    except ValueError as exc:
        if str(exc) in {
            "Unsupported native scene transforms",
            "Unsupported native hierarchy",
            "Unsupported native material binding",
            "Unsupported texture format or dimensions",
            "Only one opaque base colour and an optional native normal map are supported",
            "Native normals, triangles and UVs are required",
        }:
            raise ContinuityNoFit(str(exc)) from exc
        raise
    mesh = full._registered(raw, registration)
    before, ids, points, normals, y, x = native.atlas(mesh)
    if not len(y) or not np.array_equal(old, before):
        raise ValueError("Native atlas differs from source material")
    frame = registration["camera_frame"]
    src_ids, depth, scale, project, _ = _source_projection(mesh, matte, frame)
    xy, z, ray = project(points)
    uv = np.rint(xy @ affine[:, :2].T + affine[:, 2]).astype(int)
    inside_photo = np.all((uv >= 0) & (uv < [photo.shape[1], photo.shape[0]]), axis=1)
    uv = np.clip(uv, 0, [photo.shape[1] - 1, photo.shape[0] - 1])
    pixel = np.rint((uv - affine[:, 2]) @ np.linalg.inv(affine[:, :2]).T).astype(int)
    inside_matte = np.all((pixel >= 0) & (pixel < gt.SIZE), axis=1)
    i, j = np.clip(pixel, 0, gt.SIZE - 1).T
    observed = (
        inside_photo
        & inside_matte
        & (src_ids[j, i] == ids[y, x])
        & (np.abs(depth[j, i] - z) < 1.7 / scale)
        & (np.sum(normals * ray, axis=1) > 0.22)
        & (matte[j, i, 3] >= 242)
        & (ndimage.distance_transform_edt(matte[:, :, 3] >= 200)[j, i] > 2.5)
    )
    lock, _ = gt._source_lock(mesh, matte[:, :, 3] >= 150, frame, ids, points)
    reverse = gt._raster_bounded(
        gt._uv_pixels(mesh, len(ids)),
        np.ascontiguousarray(mesh.faces[::-1], np.int32),
        len(ids),
    )[0]
    overlap = (ids >= 0) & (reverse >= 0) & (ids != len(mesh.faces) - 1 - reverse)
    lock |= ndimage.binary_dilation(overlap)
    if any(
        gt._sha(gt._single(p, sizes[k] * gt.MIB)) != hashes[k] for k, p in paths.items()
    ):
        raise ValueError("A bound continuity input changed")
    return dict(
        mesh=mesh,
        raw=raw,
        doc=doc,
        blob=blob,
        old=old,
        ids=ids,
        points=points,
        normals=normals,
        y=y,
        x=x,
        locked=lock,
        source_rgb=photo[uv[:, 1], uv[:, 0]],
        observed=observed,
        hashes=hashes,
        observation=canonical,
    )


def paint_continuity(
    native_model, source_photo, foreground, registration, observation, output_dir
):
    """Generate a local inferred candidate only for numerically supported surfaces."""
    try:
        s = prepare_surface(
            native_model, source_photo, foreground, registration, observation
        )
    except ContinuityNoFit:
        return None
    y, x, old = s["y"], s["x"], s["old"]
    found = _transport(
        s["points"],
        s["normals"],
        old[y, x],
        s["source_rgb"],
        s["observed"],
        s["locked"][y, x],
        s["observation"],
    )
    if found is None:
        return None
    values, support, evidence = found
    changed = np.any(old[y, x] != values, axis=1)
    if changed.sum() < max(32, round(len(y) * 0.005)):
        return None
    if np.any(changed & (s["locked"][y, x] | ~support)):
        raise ValueError("Attempted changes to locked or unsupported surface")
    updated = old.copy()
    updated[y, x] = values
    # Retain source islands and material buffers. Copy the neighbouring covered
    # texel's delta into empty mipmap padding, keeping every covered texel intact.
    empty = s["ids"] < 0
    near = ndimage.distance_transform_edt(
        empty, return_distances=False, return_indices=True
    )
    pad = empty & ~s["locked"]
    delta = updated.astype(np.int16) - old
    updated[pad] = np.clip(
        old[pad].astype(np.int16) + delta[near[0][pad], near[1][pad]], 0, 255
    )
    with gt._stage(Path(output_dir)) as stage:
        reopened, pixels, retained = full._save(
            s["doc"], s["blob"], s["raw"], updated, stage
        )
        if not np.array_equal(pixels[s["locked"]], old[s["locked"]]):
            raise ValueError("Source or shared UV material changed on export")
        report = dict(
            **evidence,
            method="source-surface-continuity-v1",
            appearance="needs_review",
            source_image_sha256=s["hashes"]["source"],
            source_glb=s["hashes"]["native"],
            foreground_sha256=s["hashes"]["foreground"],
            output_glb=gt._sha((stage / "model.glb").read_bytes()),
            visual_brief_sha256=json_sha(s["observation"]),
            source_rear_observed=False,
            normals_uv_nodes_material_and_all_other_buffers_bitwise_identical=True,
            non_base_view_sha256=retained,
            locked_changed=0,
            unsupported_surface_changed=0,
            changed_uv=int(changed.sum()),
            observed_texels=int(s["observed"].sum()),
        )
        gt._json(stage / "continuity-paint.json", report)
    return report
