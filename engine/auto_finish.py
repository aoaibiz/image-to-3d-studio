"""Select a pose on a light contact sheet and apply supported image-measured finishes."""

from pathlib import Path
import os
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageOps

from engine.pipeline import PIPELINE


def _dessert_foreground(image, measurements):
    """Isolate the measured subject for the neural baseline without including the table."""
    if max(image.size) > 1280:
        image = image.copy()
        image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    width, height = image.size
    rim_x, rim_y, rim_radius = measurements["rim"]
    foot_x, foot_y, _ = measurements["foot"]
    left, top, right, bottom = measurements["cream_bbox"]
    silhouette = np.zeros((height, width), dtype=np.uint8)

    def polygon(points):
        px = np.rint(np.asarray(points) * [width, height]).astype(np.int32)
        cv2.fillPoly(silhouette, [px], 255)

    bowl = []
    for h, radius in measurements["glass_profile"]:
        center = foot_x + h * (rim_x - foot_x)
        y = foot_y + h * (rim_y - foot_y)
        bowl.append(((center - radius, y), (center + radius, y)))
    polygon([p[0] for p in bowl] + [p[1] for p in reversed(bowl)])
    # The mound narrows near the top. Berries extend beyond its sides.
    mx, rx = (left + right) / 2, (right - left) / 2
    polygon(
        [
            (mx - 0.24 * rx, top),
            (mx + 0.24 * rx, top),
            (right, top + 0.74 * (bottom - top)),
            (right, bottom),
            (rim_x + rim_radius, rim_y),
            (rim_x - rim_radius, rim_y),
            (left, bottom),
            (left, top + 0.74 * (bottom - top)),
        ]
    )
    for berry in measurements["berries"]:
        center = tuple(
            np.rint(np.asarray(berry["center"]) * [width, height]).astype(int)
        )
        axes = tuple(
            np.maximum(1, np.rint(np.asarray(berry["radii"]) * [width, height])).astype(
                int
            )
        )
        cv2.ellipse(silhouette, center, axes, 0, 0, 360, 255, -1)
    arr = np.asarray(image).copy()
    arr[:, :, 3] = np.minimum(arr[:, :, 3], silhouette)
    return Image.fromarray(arr)


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


def prepare_source(pil_input, pipeline=PIPELINE):
    image = ImageOps.exif_transpose(pil_input).convert("RGBA")
    rows = _sheet_components(image)
    info = {
        "input_size": list(image.size),
        "method": "single image",
        "selected_bbox": [0, 0, image.width, image.height],
    }
    if not rows:
        if pipeline == PIPELINE:
            from engine.dessert_measure import measure_dessert

            measured = measure_dessert(image)
            if measured is not None:
                info.update(
                    method="measured layered dessert in original image",
                    selected_bbox=[
                        round(v * image.size[i % 2])
                        for i, v in enumerate(measured["bbox"])
                    ],
                    dessert=measured,
                )
                return _dessert_foreground(image, measured), info
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


def finish_generated(
    out: Path, foreground: Path, source_info: dict, original=None, pipeline=PIPELINE
):
    from engine.figure_details import add_detected_details
    from engine.figure_shape import build_figure
    from engine.figure_paint import paint_figure
    from engine.vehicle_shape import build_vehicle
    from engine.finish_export import embed_master, export_finished

    with Image.open(foreground) as source:
        rgba = source.convert("RGBA")
    dessert = source_info.get("dessert")
    if dessert is not None:
        import trimesh
        from engine.dessert_form import add_dessert_glass
        from engine.dessert_toppings import add_dessert_toppings

        with Image.open(original) as im:
            observed = ImageOps.exif_transpose(im).convert("RGBA")
        if list(observed.size) != list(dessert["image_size"]):
            raise ValueError("Dessert measurements do not match the source image")
        scene = trimesh.Scene(metadata={"imageTo3dProfile": "layered-dessert"})
        measurements = dict(dessert)
        measurements["vessel"] = add_dessert_glass(scene, dessert, observed)
        measurements["toppings"] = add_dessert_toppings(scene, dessert, observed)
        profile = "layered-dessert"
    elif (figure := build_figure(rgba)) is not None:
        shape, measurements = figure
        scene = paint_figure(shape, rgba, measurements)
        scene = add_detected_details(scene, rgba, measurements)
        measurements["detected_whisker_count"] = sum(
            name.startswith("detected_whisker_") for name in scene.geometry
        )
        profile = "standing-figure"
    elif (vehicle := build_vehicle(rgba)) is not None:
        scene, measurements = vehicle
        profile = "wheeled-vehicle"
    else:
        scene, measurements = None, {}
        if pipeline == PIPELINE:
            from engine.general_finish import build_general

            scene, measurements = build_general(out / "model.glb", rgba)
        if scene is None:
            embed_master(out / "model.glb", out / "editable.gltf")
            return {
                "finish_profile": "inferred-mesh",
                "finish_measurements": measurements,
                "finish_reason": "Retained the inferred mesh; review its shape and appearance.",
            }
        profile = "image-guided-general"
    with tempfile.TemporaryDirectory(dir=out, prefix="measured-") as tmp:
        stage = Path(tmp)
        export_finished(scene, stage)
        for artifact in stage.iterdir():
            os.replace(artifact, out / artifact.name)
    return {
        "finish_profile": profile,
        "finish_measurements": measurements,
        "finish_geometry_source": (
            "Image-conditioned TripoSR mesh refined with silhouette and feature preservation."
            if profile == "image-guided-general"
            else "Analytic shape from observed image landmarks with inferred hidden surfaces; the neural baseline is retained separately."
        ),
        "finished_faces": sum(len(m.faces) for m in scene.geometry.values()),
    }
