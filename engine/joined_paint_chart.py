"""A source-anchored unwrapped strip for a single hidden pigment hypothesis.

Only an outer, nearly star-shaped connected band of a single native surface
participates. All angles refer to the registered geometry. No per-view colour
fields, camera-facing decals or external submissions are made by this module.
"""

import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from engine import guided_texture as gt
from engine import native_full_paint as full
from engine.visual_brief import json_sha

METHOD = "joined-intrinsic-strip-v1"
SIZE = (1024, 512)


class ChartUnavailable(ValueError):
    """No appropriate continuous sheet can be inferred conservatively."""


def _png(rgb):
    stream = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def _plane(points):
    return np.ascontiguousarray(points, dtype=np.uint8)


def _longest(mask):
    labels, n = ndimage.label(mask)
    if not n:
        raise ChartUnavailable("No connected measured height range")
    ids = np.flatnonzero(labels == (np.argmax(np.bincount(labels)[1:]) + 1))
    return int(ids[0]), int(ids[-1] + 1)


def _single_loops(mesh, height_start, height_span, bins):
    """Reject nested or disjoint/coincident shells even when their radii are similar."""
    triangles = np.asarray(mesh.triangles)
    # The GLB may split vertices at UV seams. A section endpoint belongs to an
    # original geometric edge, so identify it by its two positions, not by the
    # separate floating-point interpolation performed on neighbouring triangles.
    _, welded = np.unique(mesh.vertices, axis=0, return_inverse=True)
    face_vertices = welded[mesh.faces]
    low = triangles[:, :, 1].min(1)
    high = triangles[:, :, 1].max(1)
    closed = np.zeros(bins, bool)
    directions = np.c_[
        np.sin(np.arange(48) * 2 * np.pi / 48 + 0.009),
        np.cos(np.arange(48) * 2 * np.pi / 48 + 0.009),
    ]
    cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    eps = max(1e-9, height_span * 2e-6)
    for k in range(bins):
        y = height_start + (k + 0.417) * height_span / bins
        crosses = (low < y) & (high > y)
        tri = triangles[crosses]
        faces = face_vertices[crosses]
        edge_keys = []
        if len(tri) < 8:
            continue
        ends = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            va, vb = tri[:, a], tri[:, b]
            hit = (va[:, 1] < y) != (vb[:, 1] < y)
            ids = np.flatnonzero(hit)
            edge_keys.append(np.sort(faces[ids][:, [a, b]], axis=1))
            t = (y - va[ids, 1]) / (vb[ids, 1] - va[ids, 1])
            ends.append(
                (
                    ids,
                    va[ids][:, [0, 2]]
                    + t[:, None] * (vb[ids][:, [0, 2]] - va[ids][:, [0, 2]]),
                )
            )
        ids = np.concatenate([i for i, _ in ends])
        points = np.concatenate([p for _, p in ends])
        if len(points) != 2 * len(tri):
            continue
        segments = points[np.argsort(ids)].reshape(-1, 2, 2)
        span = np.ptp(points, axis=0)
        if min(span) < eps:
            continue
        # Endpoint valence and angular rays jointly exclude a second boundary,
        # folded-over public mesh, or disjoint appendage at the selected height.
        vertices, ix, counts = np.unique(
            np.concatenate(edge_keys)[np.argsort(ids)],
            axis=0,
            return_counts=True,
            return_inverse=True,
        )
        if np.count_nonzero(counts != 2) > 0:
            continue
        origin = (points.min(0) + points.max(0)) * 0.5
        a = segments[:, 0] - origin
        e = segments[:, 1] - segments[:, 0]
        den = cross(directions[:, None, :], e[None, :, :])
        ok = np.abs(den) > eps * 1e-6
        safe = np.where(ok, den, 1.0)
        t = cross(a, e)[None, :] / safe
        u = cross(a[None, :, :], directions[:, None, :]) / safe
        hits = (ok & (t > 0) & (u >= 0) & (u < 1)).sum(1)
        if np.all(hits == 1):
            # An off-axis small component could sit between angular sample rays.
            # Require exactly one connected polygon around the section.
            adjacency = [[] for _ in vertices]
            for left, right in ix.reshape(-1, 2):
                adjacency[left].append(right)
                adjacency[right].append(left)
            seen = {0}
            todo = [0]
            while todo:
                for neighbor in adjacency[todo.pop()]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        todo.append(neighbor)
            closed[k] = len(seen) == len(vertices)
    return closed


def _build(surface):
    from engine.continuity_paint import ROWS, COLS, _chart, _grid

    v, az, usable = _chart(surface["points"], surface["normals"])
    photo, measured = _grid(v, az, usable & surface["observed"], surface["source_rgb"])
    normals, covered = _grid(v, az, usable, surface["normals"])
    rows = np.clip(v.astype(int), 0, ROWS - 1)
    left = np.full(ROWS, -1)
    right = np.full(ROWS, COLS)
    valid = np.bincount(rows[usable], minlength=ROWS) >= 144
    for k in range(ROWS):
        # Require an extended, genuinely observed part of each side; one isolated
        # ray of the photograph cannot anchor an invented motif.
        a = np.flatnonzero(ndimage.binary_opening(measured[k, : COLS // 2], np.ones(8)))
        b = (
            np.flatnonzero(ndimage.binary_opening(measured[k, COLS // 2 :], np.ones(8)))
            + COLS // 2
        )
        if len(a) and len(b):
            left[k], right[k] = a[-1], b[0]
    valid &= (
        (left >= COLS * 0.13)
        & (left <= COLS * 0.36)
        & (right >= COLS * 0.64)
        & (right <= COLS * 0.87)
    )
    valid &= covered[:, COLS // 8 : -COLS // 8].mean(1) >= 0.985
    height = surface["points"][:, 1]
    valid &= _single_loops(
        surface["mesh"], float(height.min()), float(np.ptp(height)), ROWS
    )
    lo, hi = _longest(valid)
    if hi - lo < ROWS * 0.24:
        raise ChartUnavailable(
            "Insufficient continuous ring coverage or two-sided photo overlap"
        )
    # Use one coordinate system across the complete band; never stretch
    # individual rows to align independently with an inferred picture.
    start = max(0, int(left[lo:hi].min()) - 19)
    stop = min(COLS, int(right[lo:hi].max()) + 20)
    hidden = ~surface["locked"][surface["y"], surface["x"]] & ~surface["observed"]
    supported = usable & hidden & (v >= lo) & (v < hi) & (az >= start) & (az < stop)
    if supported.sum() < 128 or supported.sum() < 0.15 * max(1, hidden.sum()):
        raise ChartUnavailable(
            "Too little eligible native hidden shell for a complete strip"
        )
    selection = np.s_[lo:hi, start:stop]
    anchor = measured[selection] & covered[selection]
    # Avoid scheduling a costly request when the fixed margins contain no
    # positional feature at all (plain/repeating finish belongs to local routes).
    target = photo[selection].astype(np.float32)
    distinct = []
    for side in (slice(0, 23), slice(-23, None)):
        obs = anchor[:, side]
        gx = np.linalg.norm(np.diff(target[:, side], axis=1), axis=2)
        gy = np.linalg.norm(np.diff(target[:, side], axis=0), axis=2)
        edges = np.r_[gx[obs[:, :-1] & obs[:, 1:]], gy[obs[:-1] & obs[1:]]]
        distinct.append(float(np.mean(edges > 16)) if len(edges) else 0.0)
        if obs.mean() < 0.70 or distinct[-1] < 0.032:
            raise ChartUnavailable(
                "Not enough source detail to verify both sheet margins"
            )
    # Clay relief comes from the actual outward surface normals in the local
    # circumferential basis, without superimposing guessed decoration.
    theta = np.arange(COLS) * 2 * np.pi / COLS
    tangent = np.stack((np.cos(theta), np.zeros(COLS), -np.sin(theta)), axis=1)
    tangent_normal = (normals * tangent[None, :, :]).sum(2)
    relief = np.clip(186 + 37 * normals[:, :, 1] - 35 * tangent_normal, 108, 235)
    grey = np.repeat(relief[selection][:, :, None], 3, axis=2)
    guide = np.where(anchor[:, :, None], target, grey)
    guide[~covered[selection]] = 249
    # Geometric vertical increases upwards, so source-up is the first image row.
    coarse = _plane(np.rint(guide[::-1]))
    image = cv2.resize(coarse, SIZE, interpolation=cv2.INTER_LINEAR)
    evidence = dict(
        method=METHOD,
        grid_size=[COLS, ROWS],
        rectangle=[start, lo, stop, hi],
        y_axis="up",
        image_top="highest",
        source_azimuth=0,
        back_azimuth=COLS // 2,
        editable_hidden_only=True,
        detected_anchor_edge_fractions=distinct,
        guide_size=list(SIZE),
        chart_shape=list(anchor.shape),
        anchor_mask_sha256=hashlib.sha256(np.packbits(anchor).tobytes()).hexdigest(),
        coverage_sha256=hashlib.sha256(
            np.packbits(covered[selection]).tobytes()
        ).hexdigest(),
    )
    return dict(
        v=v,
        az=az,
        usable=usable,
        row=rows,
        geometry=covered[selection],
        anchor=anchor,
        coarse=coarse,
        photo=target,
        guide=image,
        rectangle=(start, lo, stop, hi),
        definition=evidence,
        supported_hidden_fraction=float(supported.sum() / max(1, hidden.sum())),
        unique_patches=_unique_source_patches(photo, measured, surface["observation"]),
    )


def _prepare(native_model, source_photo, foreground, registration, observation):
    from engine.continuity_paint import prepare_surface, ContinuityNoFit

    try:
        surface = prepare_surface(
            native_model, source_photo, foreground, registration, observation
        )
    except ContinuityNoFit as exc:
        raise ChartUnavailable(str(exc)) from exc
    chart = _build(surface)
    data = _png(chart["guide"])
    manifest = dict(
        method=METHOD,
        model_sha=surface["hashes"]["native"],
        photo_sha=surface["hashes"]["source"],
        foreground_sha=surface["hashes"]["foreground"],
        registration_sha=json_sha(registration),
        guide_sha=gt._sha(data),
        visual_brief=surface["observation"],
        visual_brief_sha256=json_sha(surface["observation"]),
        chart=chart["definition"],
    )
    manifest["chart_id"] = json_sha(manifest)
    return surface, chart, data, manifest


def confirm_chart(
    *,
    source_photo,
    native_model,
    foreground,
    registration,
    observation,
    guide_bytes,
    manifest,
):
    # Do not accept even a self-consistent forged guide+manifest pair as an upload.
    _, _, data, expected = _prepare(
        native_model, source_photo, foreground, registration, observation
    )
    if data != guide_bytes or expected != manifest:
        raise ValueError(
            "Chart inputs or derived source margins changed before submission"
        )


def prepare_chart(
    *,
    native_model,
    source_photo,
    foreground,
    registration,
    observation,
    output_dir,
    minimum_hidden_fraction=0.0,
):
    _, chart, data, manifest = _prepare(
        native_model, source_photo, foreground, registration, observation
    )
    if chart["supported_hidden_fraction"] < minimum_hidden_fraction:
        raise ChartUnavailable(
            "The chart covers too little writable hidden native material"
        )
    with gt._stage(Path(output_dir)) as stage:
        (stage / "guide.png").write_bytes(data)
        gt._json(stage / "chart.json", manifest)
        Image.fromarray(
            _plane((chart["geometry"] & ~chart["anchor"])[::-1]) * 255
        ).save(stage / "editable-mask.png")
    return manifest


def _unique_source_patches(photo, measured, observation):
    if not observation.get("observed_face_cues", {}).get("focal"):
        return []
    # Put camera azimuth zero in the middle of one continuous observed strip.
    full = np.ascontiguousarray(
        np.roll(photo, photo.shape[1] // 2, axis=1), dtype=np.uint8
    )
    known = np.roll(measured, measured.shape[1] // 2, axis=1).astype(np.float32)
    h, w = full.shape[:2]
    patches = []
    size = 17
    cover = cv2.matchTemplate(
        known, np.ones((size, size), np.float32), cv2.TM_CCORR
    ) / (size * size)
    for y in range(9, h - size, 8):
        for x in range(w // 2 - 30, w // 2 + 25, 8):
            if cover[y, x] < 0.98:
                continue
            patch = full[y : y + size, x : x + size]
            energy = cv2.cornerMinEigenVal(
                cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY), 3
            ).mean()
            if energy < 0.0004:
                continue
            selfscore = cv2.matchTemplate(full, patch, cv2.TM_CCOEFF_NORMED)
            selfscore[cover < 0.98] = -1
            selfscore[
                max(0, y - size) : y + size + 1, max(0, x - size) : x + size + 1
            ] = -1
            if selfscore.max() < 0.76:
                patches.append((energy, y, x, patch.copy()))
    chosen = []
    for energy, y, x, patch in sorted(patches, key=lambda t: -t[0]):
        if all(np.hypot(y - q, x - r) > size for q, r, _ in chosen):
            chosen.append((y, x, patch))
            if len(chosen) == 6:
                break
    return [patch for _, _, patch in chosen]


def _check_unique_patches(result, hidden, patches):
    strongest = 0.0
    for p in patches:
        h, w = p.shape[:2]
        coverage = cv2.matchTemplate(
            hidden.astype(np.float32), np.ones((h, w), np.float32), cv2.TM_CCORR
        ) / (h * w)
        for template in [p, p[:, ::-1]]:
            score = cv2.matchTemplate(
                np.ascontiguousarray(result, dtype=np.uint8),
                np.ascontiguousarray(template),
                cv2.TM_CCOEFF_NORMED,
            )
            score[coverage < 0.98] = -1
            strongest = max(strongest, float(score.max()))
    if strongest > 0.93:
        raise ChartUnavailable(
            "A distinctive photographed patch reappears on the unseen surface"
        )
    return round(strongest, 4)


def _verify_anchors(chart, result):
    # Compare the returned sheet against invariants before allowing any change
    # to the model. No inferred translation, shear or independent side registration.
    ref = chart["photo"].astype(float)
    mask = chart["anchor"]
    scores = []
    for side in (slice(0, 23), slice(-23, None)):
        src = ref[:, side]
        out = result[:, side].astype(float)
        valid = mask[:, side]
        error = np.linalg.norm((src - out) / 255, axis=2) / np.sqrt(3)
        for part in np.array_split(np.arange(len(valid)), 3):
            use = valid[part]
            if use.sum() < 40:
                raise ChartUnavailable("Insufficient immutable margin")
            a = src[part][use].ravel()
            b = out[part][use].ravel()
            correlation = (
                float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 4 else 0.0
            )
            mean = float(error[part][use].mean())
            if not np.isfinite(correlation) or correlation < 0.78 or mean > 0.095:
                raise ChartUnavailable(
                    "Returned left/right margins do not match measured source pixels"
                )
            scores.append([round(correlation, 4), round(mean, 4)])
    return scores


def _transfer(surface, chart, sheet):
    if (
        sheet.mode != "RGB"
        or not 256 <= sheet.width <= 4096
        or not 128 <= sheet.height <= 4096
        or abs(sheet.width / sheet.height - 2) > 0.09
    ):
        raise ChartUnavailable(
            "The returned image is not the prescribed single wide strip"
        )
    start, lo, stop, hi = chart["rectangle"]
    result = cv2.resize(
        np.asarray(sheet), (stop - start, hi - lo), interpolation=cv2.INTER_AREA
    )[::-1].copy()
    scores = _verify_anchors(chart, result)
    interior = chart["geometry"] & ~ndimage.binary_dilation(
        chart["anchor"], iterations=4
    )
    prior = cv2.resize(
        chart["guide"], (stop - start, hi - lo), interpolation=cv2.INTER_AREA
    )[::-1]
    if (
        interior.sum() < 160
        or np.mean(np.max(np.abs(result.astype(float) - prior), axis=2)[interior] > 12)
        < 0.24
    ):
        raise ChartUnavailable("The returned chart did not complete the hidden paint")
    repeated = surface["observation"].get("observed_face_cues", {}).get("repeated", {})
    if repeated.get("location") in {"across_visible_face", "separate_visible_areas"}:
        gradient = np.linalg.norm(np.diff(result.astype(float), axis=1), axis=2)
        interior_edge = interior[:, 1:] & interior[:, :-1]
        if np.mean(gradient[interior_edge] > 12) < 0.035:
            raise ChartUnavailable(
                "An apparently blank fill cannot establish the observed surface decoration"
            )
    strongest = _check_unique_patches(result, interior, chart["unique_patches"])
    y, x = surface["y"], surface["x"]
    eligible = (
        chart["usable"]
        & ~surface["locked"][y, x]
        & ~surface["observed"]
        & (chart["v"] >= lo)
        & (chart["v"] < hi)
        & (chart["az"] >= start)
        & (chart["az"] < stop)
    )
    coords = np.stack((chart["v"] - lo - 0.5, chart["az"] - start - 0.5))
    # A centimetre of surface coordinates has one colour regardless of the
    # number of original UV islands or cameras that see it.
    values = np.stack(
        [
            ndimage.map_coordinates(
                result[:, :, i].astype(np.float32), coords, order=1, mode="nearest"
            )
            for i in range(3)
        ],
        axis=-1,
    )
    edit = chart["geometry"] & ~chart["anchor"]
    dist = ndimage.distance_transform_edt(np.pad(edit, 1))[1:-1, 1:-1]
    margin = ndimage.map_coordinates(dist, coords, order=1, mode="constant")
    locked_in_strip = (
        chart["usable"] & surface["locked"][y, x] & ~surface["observed"] & (margin >= 3)
    )
    native = surface["old"][y, x].astype(float)
    # Do not hide a collision behind the immutability promise: a large retained
    # band differing sharply from the proposed continuation would remain visible.
    large_delta = np.max(np.abs(native - values), axis=1) > 56
    fixed_count = int(locked_in_strip.sum())
    collision = int(np.count_nonzero(locked_in_strip & large_delta))
    if collision >= 40 and collision > max(0.18 * fixed_count, 0.012 * eligible.sum()):
        raise ChartUnavailable("Proposed paint conflicts with a protected native band")
    blend = np.where(eligible, np.clip((margin - 0.55) / 2.5, 0.0, 1.0), 0.0)[:, None]
    old = surface["old"]
    updated = old.copy()
    updated[y, x] = np.rint(old[y, x] * (1 - blend) + values * blend).astype(np.uint8)
    diff = np.any(updated != old, axis=2)
    if (
        diff.sum() < 64
        or np.any(diff[surface["locked"]])
        or np.any(diff[y, x][~eligible])
    ):
        raise ChartUnavailable(
            "Insufficient eligible hidden surface or protected UV write"
        )
    return updated, dict(
        changed_uv=int(diff.sum()),
        eligible_uv=int(eligible.sum()),
        supported_hidden_fraction=chart["supported_hidden_fraction"],
        locked_changed=0,
        unsupported_surface_changed=0,
        source_observed_changed=0,
        anchor_scores=scores,
        protected_hidden_samples=fixed_count,
        divergent_protected_samples=collision,
        tested_unique_source_patches=len(chart["unique_patches"]),
        strongest_unique_source_patch_score=strongest,
    )


def apply_chart(
    *,
    native_model,
    source_photo,
    foreground,
    registration,
    observation,
    guide_dir,
    sheet_path,
    receipt,
    output_dir,
):
    from engine.joined_chart_provider import verify_completed

    s, chart, data, manifest = _prepare(
        native_model, source_photo, foreground, registration, observation
    )
    directory = Path(guide_dir)
    if (
        gt._single(directory / "guide.png", 8 * gt.MIB) != data
        or json.loads(gt._single(directory / "chart.json", 65536)) != manifest
    ):
        raise ValueError(
            "Stored chart is not reproducible from the sealed source and geometry"
        )
    payload = gt._single(sheet_path, 32 * gt.MIB)
    verify_completed(receipt, manifest, source_photo, directory / "guide.png", payload)
    updated, info = _transfer(s, chart, gt._image(payload, 12_000_000))
    with gt._stage(Path(output_dir)) as stage:
        reopened, pixels, other = full._save(
            s["doc"], s["blob"], s["raw"], updated, stage
        )
        if not np.array_equal(pixels, updated):
            raise ValueError("Reopened chart differs from native export")
        copy = full._copy_check(
            full._registered(reopened, registration), pixels, s["observation"]
        )
        full._require_copy_evidence(copy, s["observation"])
        info.update(
            method=METHOD,
            appearance="needs_review",
            source_glb=s["hashes"]["native"],
            source_image_sha256=s["hashes"]["source"],
            foreground_sha256=s["hashes"]["foreground"],
            output_glb=gt._sha((stage / "model.glb").read_bytes()),
            visual_brief_sha256=manifest["visual_brief_sha256"],
            chart_id=manifest["chart_id"],
            guide_sha=manifest["guide_sha"],
            sheet_sha=gt._sha(payload),
            generation_id=receipt["thread_id"],
            copy_screen=copy,
            source_rear_observed=False,
            non_base_view_sha256=other,
            normals_uv_nodes_material_and_all_other_buffers_bitwise_identical=True,
        )
        gt._json(stage / "chart-paint.json", info)
    return info
