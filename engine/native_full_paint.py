"""Transfer two geometry-registered inferred views onto an existing native UV atlas."""

import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

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
from quality import reflected_paint as copy_test


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


def _field(mesh, sheet, points, normals):
    """Transfer local detail per view. Transfer broad chroma only when both views agree."""
    fields, weights, hues = [], [], []
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
            raise NativeFullRegistrationError(f"{name}: {err}") from err
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
        valid = (border > 0).astype(np.float32)
        sigma = 0.032 * max(np.ptp(axis) for axis in np.nonzero(valid))
        coverage = cv2.GaussianBlur(valid, (0, 0), sigma)
        low = cv2.GaussianBlur(aligned * valid[:, :, None], (0, 0), sigma)
        low /= np.maximum(coverage[:, :, None], 1e-6)
        hues.append(gt._sample(low, xy))
        fields.append(gt._sample(detail, xy))
        weights.append(w)
        stats[name]["effective_texels"] = int((w > 0.5).sum())
    wa, wb = weights
    primary = (wa >= wb)[:, None]
    detail = np.where(primary, fields[0], fields[1])
    hue = np.where(primary, hues[0], hues[1])
    strength = np.clip((np.maximum(wa, wb) - 0.1) / 0.68, 0, 1)
    # No mixture of competing layouts. Keep the native pigment at disputed seams.
    conflict = (np.max(abs(fields[0] - fields[1]), axis=1) > 0.32) & (
        np.minimum(wa, wb) > 0.35
    )
    strength[conflict] = 0

    def chroma(rgb):
        return rgb - (rgb @ np.array([0.2126, 0.7152, 0.0722]))[:, None]

    chroma_error = np.max(abs(chroma(hues[0]) - chroma(hues[1])), axis=1)
    broad = np.clip((np.minimum(wa, wb) - 0.45) / 0.35, 0, 1) * np.clip(
        (32 - chroma_error) / 20, 0, 1
    )
    return detail, chroma(hue), strength, broad, stats


def _copy_check(mesh, texture):
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
    if (
        signal["rms_rgb"] < 3
        or signal["self_reflection"] >= 0.75
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
    detail, proposed_chroma, strength, broad, stats = _field(
        mesh, sheet, points, normals
    )
    # Feather into native albedo at the measured front, including UV seam charts.
    strength *= np.clip(
        (ndimage.distance_transform_edt(~protected)[y, x] - 1) / 5, 0, 1
    )
    support = strength > 0
    if support.sum() < max(32, round(len(y) * 0.005)):
        raise NativeFullReviewHold("No significant registered hidden surface")
    base = albedo[y, x].astype(float)
    base_chroma = base - (base @ np.array([0.2126, 0.7152, 0.0722]))[:, None]
    # The guide can carry studio lighting. Retain native luminance and BRDF.
    target = (base + broad[:, None] * (proposed_chroma - base_chroma)) * detail
    updated = albedo.copy()
    updated[y, x] = np.clip(np.rint(base + strength[:, None] * (target - base)), 0, 255)
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
        "source_camera_iou": fit,
        "visible_front_locked": int(lock.sum()),
        "shared_uv_protected": int(overlap.sum()),
        "locked_changed": 0,
        "unsupported_surface_changed": 0,
        "changed_uv": int(changed.sum()),
        "supported_uv": int(support.sum()),
        "total_uv": len(x),
        "two_view_chroma_uv": int(np.sum(support & (broad > 0))),
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
    with gt._stage(Path(output_dir)) as stage:
        reopened, actual, other = _save(doc, blob, raw, updated, stage)
        copy = _copy_check(_registered(reopened, registration), actual)
        distinct = _require_copy_evidence(copy, obs)
        if any(
            digest(path) != ids[k] for k, path in paths.items()
        ) or guide_bytes != gt._single(folder / "guide.json", 65536):
            raise ValueError("Input changed during the transfer")
        info.update(
            method="native-full-guided-v1",
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
