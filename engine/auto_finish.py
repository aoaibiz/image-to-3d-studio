"""Select a pose on a light contact sheet and apply supported image-measured finishes."""

from pathlib import Path
import os
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageOps


def _sheet_components(image):
    preview = image.copy()
    preview.thumbnail((1200, 1200))
    rgba = np.asarray(preview)
    rgb = rgba[:, :, :3].astype(np.int16)
    alpha = rgba[:, :, 3]
    rim = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    backdrop = np.median(rim, axis=0)
    if np.any(alpha < 128):
        mask = alpha >= 128
    elif backdrop.min() >= 225 and np.quantile(np.abs(rim - backdrop), 0.9) < 12:
        mask = np.max(np.abs(rgb - backdrop), axis=2) > 38
    else:
        return []
    height, width = mask.shape
    radius = max(1, round(min(height, width) * 0.003))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    mask = cv2.morphologyEx(mask.astype("uint8"), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= width * height * 0.012
    ]
    boxes = [cv2.boundingRect(contour) for contour in large]
    if not 4 <= len(boxes) <= 32:
        return []
    areas = [w * h for x, y, w, h in boxes]
    if max(areas) > 3 * min(areas):
        return []
    # A grid of unrelated products must not be cropped as repeated poses.
    profiles = []
    for x, y, w, h in boxes:
        pixels = rgba[y : y + h, x : x + w, :3][mask[y : y + h, x : x + w].astype(bool)]
        ids = pixels.astype(np.uint16) // 48
        bins = np.bincount(ids[:, 0] * 36 + ids[:, 1] * 6 + ids[:, 2], minlength=216)
        profiles.append(bins / max(bins.sum(), 1))
    median = np.median(profiles, axis=0)
    if any(np.abs(profile - median).sum() > 0.65 for profile in profiles):
        return []
    rows = []
    for box in sorted(boxes, key=lambda b: b[1] + b[3] / 2):
        cy = box[1] + box[3] / 2
        if not rows or abs(
            cy - np.mean([b[1] + b[3] / 2 for b in rows[-1]])
        ) > 0.42 * np.median([b[3] for b in boxes]):
            rows.append([])
        rows[-1].append(box)
    if len(rows) < 2 or any(len(row) < 2 for row in rows):
        return []
    # Only auto-select a subject if its outline recurs elsewhere on the sheet.
    preferred = min(rows[0], key=lambda b: abs(b[0] + b[2] / 2 - width / 2))
    silhouettes = []
    for contour, (x, y, w, h) in zip(large, boxes):
        silhouette = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(silhouette, [contour], -1, 255, cv2.FILLED, offset=(-x, -y))
        silhouettes.append(
            cv2.resize(silhouette, (48, 48), interpolation=cv2.INTER_AREA) > 127
        )
    selected = silhouettes[boxes.index(preferred)]
    if not any(
        np.count_nonzero(selected & other) >= 0.90 * np.count_nonzero(selected | other)
        for box, other in zip(boxes, silhouettes)
        if box != preferred
    ):
        return []
    sx, sy = image.width / width, image.height / height
    result = []
    for row in rows:
        row.sort()
        if any(left[0] + left[2] > right[0] for left, right in zip(row, row[1:])):
            return []
        result.append(
            [
                tuple(
                    round(v * s) for v, s in zip((x, y, x + w, y + h), (sx, sy, sx, sy))
                )
                for x, y, w, h in row
            ]
        )
    return result


def prepare_source(pil_input):
    image = ImageOps.exif_transpose(pil_input).convert("RGBA")
    rows = _sheet_components(image)
    info = {
        "input_size": list(image.size),
        "method": "single image",
        "selected_bbox": [0, 0, image.width, image.height],
    }
    if not rows:
        return image, info
    center = image.width / 2
    box = min(rows[0], key=lambda b: abs((b[0] + b[2]) / 2 - center))
    margin = max(2, round(min(box[2] - box[0], box[3] - box[1]) * 0.035))
    left, top, right, bottom = box
    bounds = [
        max(0, left - margin),
        max(0, top - margin),
        min(image.width, right + margin),
        min(image.height, bottom + margin),
    ]
    crop = image.crop(bounds)
    if crop.getextrema()[3][0] == 255:
        arr = np.asarray(crop).copy()
        ink = (arr[:, :, :3].min(axis=2) < 218).astype("uint8")
        closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            silhouette = np.zeros_like(ink)
            cv2.drawContours(
                silhouette, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED
            )
            arr[:, :, 3] = silhouette
            crop = Image.fromarray(arr)
    info.update(
        method="contact sheet, centered pose in first row",
        selected_bbox=bounds,
        detected_rows=rows,
        detected_count=sum(map(len, rows)),
    )
    return crop, info


def finish_generated(out: Path, foreground: Path, source_info: dict):
    from engine.figure_details import add_detected_details
    from engine.figure_shape import build_figure
    from engine.figure_paint import paint_figure
    from engine.vehicle_shape import build_vehicle
    from engine.finish_export import embed_master, export_finished

    with Image.open(foreground) as source:
        rgba = source.convert("RGBA")
    figure = build_figure(rgba)
    if figure is not None:
        shape, measurements = figure
        scene = paint_figure(shape, rgba, measurements)
        scene = add_detected_details(scene, rgba, measurements)
        measurements["detected_whisker_count"] = sum(
            name.startswith("detected_whisker_") for name in scene.geometry
        )
        profile = "standing-figure"
    else:
        vehicle = build_vehicle(rgba)
        if vehicle is None:
            embed_master(out / "model.glb", out / "editable.gltf")
            return {
                "finish_profile": "inferred-mesh",
                "finish_measurements": {},
                "finish_reason": "No supported set of landmarks detected; review the inferred mesh.",
            }
        scene, measurements = vehicle
        profile = "wheeled-vehicle"
    with tempfile.TemporaryDirectory(dir=out, prefix="measured-") as tmp:
        stage = Path(tmp)
        export_finished(scene, stage)
        for artifact in stage.iterdir():
            os.replace(artifact, out / artifact.name)
    return {
        "finish_profile": profile,
        "finish_measurements": measurements,
        "finish_geometry_source": "Analytic shape from observed image landmarks with inferred hidden surfaces; the neural baseline is retained separately.",
        "finished_faces": sum(len(m.faces) for m in scene.geometry.values()),
    }
