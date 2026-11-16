"""Conservative, local veto for a photographed face reappearing on occluded surfaces."""

import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from engine import guided_texture as gt
from engine.triposg_cpu import atomic_json, writer_lock
from engine.visual_brief import json_sha, validate_photo_observations


VERSION = "hidden-front-layout-v1"
ROWS, COLS = 192, 128
# Compare colour marks inside the material, away from the object's outer contour.
FRONT_HARDWARE = {
    "projecting_assembly",
    "recessed_opening",
    "parallel_wires",
    "fasteners",
    "screen_or_controls",
}


def _canonical(pixels, mask):
    """Normalize row widths for foreshortening. Keep gaps and narrow parts out of the comparison."""
    y = np.nonzero(mask)[0]
    if not len(y) or np.ptp(y) < 12:
        raise ValueError("Insufficient foreground for the orientation check.")
    rows = np.linspace(y.min(), y.max(), ROWS).astype(int)
    left = mask.argmax(1)[rows]
    right = mask.shape[1] - 1 - mask[:, ::-1].argmax(1)[rows]
    widths = right - left
    u = np.linspace(0, 1, COLS)
    cx = np.float32(left[:, None] + widths[:, None] * u)
    cy = np.float32(np.broadcast_to(rows[:, None], cx.shape))
    rgb = cv2.remap(pixels, cx, cy, cv2.INTER_LINEAR)
    inside = cv2.remap(mask.astype(np.uint8), cx, cy, cv2.INTER_NEAREST).astype(bool)
    inside &= widths[:, None] > 0.4 * max(widths)
    inside[:, :12] = False
    inside[:, -12:] = False
    return rgb, inside


def _contrast(rgb, interior):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    for y in range(len(lab)):
        if interior[y].any():
            lab[y] -= np.median(lab[y, interior[y]], axis=0)
    return lab


def _translate(array, offset, axis):
    shifted = np.roll(array, offset, axis=axis)
    if offset:
        index = [slice(None)] * array.ndim
        index[axis] = slice(None, offset) if offset > 0 else slice(offset, None)
        shifted[tuple(index)] = 0
    return shifted


def _overlap(front, target, valid, sign):
    # Require both a perceptible luminance change and a chromatic shift relative to
    # the nearby substrate. A dark silhouette, lighting gradient or gray line is insufficient.
    a = (
        valid
        & (sign * front[:, :, 0] > 25)
        & (np.linalg.norm(front[:, :, 1:], axis=2) > 18)
    )
    b = (
        valid
        & (sign * target[:, :, 0] > 25)
        & (np.linalg.norm(target[:, :, 1:], axis=2) > 18)
    )
    shared = a & b & (np.sum(front[:, :, 1:] * target[:, :, 1:], axis=2) > 0)
    return 2 * shared.sum() / max(1, a.sum() + b.sum()), shared


def _panel_score(front, front_mask, target, target_mask):
    a = _contrast(front, front_mask)[:, ::-1]
    b = _contrast(target, target_mask)
    best = {
        "dice": 0.0,
        "hidden_coverage": 0.0,
        "layout_margin": 0.0,
        "row_bands": [],
        "sign": 0,
    }
    for offset in range(-24, 25, 4):
        candidate = _translate(a, offset, 1)
        valid = target_mask & _translate(front_mask[:, ::-1], offset, 1)
        if valid.sum() < ROWS * COLS * 0.12:
            continue
        for sign in (-1, 1):
            score, shared = _overlap(candidate, b, valid, sign)
            if score <= best["dice"]:
                continue
            # Contrast that forms a uniform stripe should match equally well when
            # displaced vertically. Require a spatially distinctive arrangement.
            decoys = [
                _overlap(
                    _translate(candidate, dy, 0),
                    b,
                    valid & _translate(valid, dy, 0),
                    sign,
                )[0]
                for dy in (-34, 34)
            ]
            counts = shared.reshape(8, ROWS // 8, COLS).sum(axis=(1, 2))
            best = {
                "dice": float(score),
                "hidden_coverage": float(shared.sum() / valid.sum()),
                "layout_margin": float(score - max(decoys)),
                "row_bands": np.where(counts >= 25)[0].tolist(),
                "sign": sign,
            }
    return best


def _diagnosis(scores, brief):
    """Return an appearance-review status, never an automatic visual pass."""
    if len(scores) != 2:
        raise ValueError("Both inferred camera directions must be reviewed.")
    implicated = (
        all(
            p["dice"] >= 0.55
            and p["hidden_coverage"] >= 0.05
            and p["layout_margin"] >= 0.15
            and len(p["row_bands"]) >= 3
            for p in scores
        )
        and scores[0]["sign"] == scores[1]["sign"]
    )
    if not implicated:
        return "needs_review", False, "no_corroborated_front_layout"
    if brief and "repeated" in brief["observed_face_cues"]:
        return "needs_review", False, "repeated_decoration_requires_visual_review"
    # Restrict vetoes to photographed, localized functional structures. Cast or
    # sculpted details and colour alone do not establish that the reverse is wrong.
    places = {
        "center": {2, 3, 4, 5},
        "upper_field": {0, 1, 2, 3},
        "lower_field": {4, 5, 6, 7},
        "across_visible_face": set(range(8)),
        "separate_visible_areas": set(range(8)),
    }
    if brief and any(
        cue["kind"] in FRONT_HARDWARE
        and cue["location"] in places
        and all(set(p["row_bands"]) & places[cue["location"]] for p in scores)
        for cue in brief["visible_structures"]
    ):
        return "rejected", True, "camera_occluded_front_hardware_layout"
    return "needs_review", True, "camera_occluded_repeated_front_layout"


def _score_images(model, foreground, generated, camera_frame, expected_sha):
    mesh, _, _, actual_sha = gt._model(model)
    if actual_sha != expected_sha:
        raise ValueError("The reviewed model changed after admission.")
    # The base mesh has already been rotated to the source camera. Enlarge the
    # set of front-visible faces to spare protruding parts seen near an edge.
    seen = np.zeros(len(mesh.faces), bool)
    for angle in ((0, 0), (10, 0), (-10, 0), (0, 10), (0, -10)):
        ids = gt._view(mesh, angle)[3]
        seen[ids[ids >= 0]] = True
    with Image.open(io.BytesIO(foreground)) as image:
        if (
            image.format != "PNG"
            or image.mode != "RGBA"
            or image.width * image.height > 40_000_000
            or getattr(image, "n_frames", 1) != 1
        ):
            raise ValueError("Expected the segmented input photograph.")
    photo = gt._matte(io.BytesIO(foreground), gt.SIZE)
    photo_mask = photo[:, :, 3] >= 150
    basis = np.array(
        [
            camera_frame[k]
            for k in ("image_right", "image_up", "camera_direction_from_origin")
        ],
        float,
    )
    if (
        basis.shape != (3, 3)
        or not np.isfinite(basis).all()
        or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-5)
        or np.linalg.det(basis) < 0.999
    ):
        raise ValueError("Invalid source camera for orientation check.")
    projected, *_ = gt._fit(
        mesh.vertices @ basis, basis, photo_mask, gt.lens_parameters(camera_frame)
    )
    ids = gt._raster_bounded(projected, np.asarray(mesh.faces, np.int32), gt.SIZE)[0]
    if gt._iou(photo_mask, ids >= 0) < 0.72:
        raise ValueError(
            "Foreground or fitted source camera does not match the reviewed mesh."
        )
    seen[ids[ids >= 0]] = True
    source, source_mask = _canonical(photo[:, :, :3], photo_mask)
    sheet = gt._image(generated, 12_000_000)
    if min(sheet.size) < 256 or abs(sheet.width / sheet.height - 2) >= 0.08:
        raise ValueError("Expected two adjacent square inferred views.")
    if sheet.mode == "RGBA":
        white = Image.new("RGB", sheet.size, "white")
        white.paste(sheet, mask=sheet.getchannel("A"))
        sheet = white
    else:
        sheet = sheet.convert("RGB")
    scores = {}
    for i, (name, angle) in enumerate(gt.ANGLES.items()):
        box = (i * sheet.width // 2, 0, (i + 1) * sheet.width // 2, sheet.height)
        panel = np.asarray(
            sheet.crop(box).resize((gt.SIZE, gt.SIZE), Image.Resampling.LANCZOS)
        )
        panel_mask = panel.min(axis=2) < 223
        ids = gt._view(mesh, angle)[3]
        solid = ids >= 0
        occluded = solid & ~seen[ids] & (mesh.face_normals[ids, 2] < -0.2)
        hidden, _ = _canonical(occluded.astype(np.uint8) * 255, solid)
        target, inner = _canonical(panel, panel_mask)
        scores[name] = _panel_score(
            source, source_mask, target, inner & (hidden >= 240)
        )
    return scores


def review_hidden_paint(
    model,
    source_photo,
    foreground,
    inferred_sheet,
    visual_brief,
    output_dir,
    *,
    camera_frame,
):
    """Check a proposed sheet before projection. A true hold_projection must block automatic use."""
    model = Path(model)
    source_sha = hashlib.sha256(gt._single(source_photo, 32 * gt.MIB)).hexdigest()
    brief = None if visual_brief is None else validate_photo_observations(visual_brief, source_sha)
    matte = gt._single(foreground, 32 * gt.MIB)
    generated = gt._single(inferred_sheet, 32 * gt.MIB)
    inputs = {
        "version": VERSION,
        "implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": source_sha,
        "model_sha256": hashlib.sha256(gt._single(model, 48 * gt.MIB)).hexdigest(),
        "foreground_sha256": hashlib.sha256(matte).hexdigest(),
        "sheet_sha256": hashlib.sha256(generated).hexdigest(),
        "brief_sha256": json_sha(brief),
        "camera_sha256": json_sha(camera_frame),
        "angles": {k: list(v) for k, v in gt.ANGLES.items()},
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "hidden-review.json"
    with writer_lock(out):
        if marker.exists():
            report = json.loads(gt._single(marker, 16384))
            if report.pop("report_sha256", None) != json_sha(report):
                raise ValueError("Stored orientation report checksum mismatch.")
            if report.get("inputs") != inputs:
                raise ValueError(
                    "Stored orientation review belongs to different inputs."
                )
            scores = report["panels"]
            if set(scores) != set(gt.ANGLES):
                raise ValueError("Stored orientation review lacks camera directions.")
            expected = _diagnosis(list(scores.values()), brief)
            if (
                report["status"],
                report["hold_projection"],
                report["reason"],
            ) != expected:
                raise ValueError("Stored orientation finding is inconsistent.")
            report["report_sha256"] = json_sha(report)
            return report
        scores = _score_images(
            model, matte, generated, camera_frame, inputs["model_sha256"]
        )
        status, hold, reason = _diagnosis(list(scores.values()), brief)
        report = {
            "inputs": inputs,
            "panels": scores,
            "status": status,
            "hold_projection": hold,
            "reason": reason,
            "back_observed": False,
        }
        report["report_sha256"] = json_sha(report)
        atomic_json(marker, report)
    return report
