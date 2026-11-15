"""Conservative, offline front-only reference projection. Requires RGBA reference."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import (
    affine_transform,
    binary_dilation,
    distance_transform_edt,
    map_coordinates,
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def raster(tri, shape):
    h, w = shape
    lo = np.maximum(np.floor(tri.min(0)).astype(int), 0)
    hi = np.minimum(np.ceil(tri.max(0)).astype(int), [w - 1, h - 1])
    if np.any(hi < lo):
        return None
    x, y = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
    den = (tri[1, 1] - tri[2, 1]) * (tri[0, 0] - tri[2, 0]) + (
        tri[2, 0] - tri[1, 0]
    ) * (tri[0, 1] - tri[2, 1])
    if abs(den) < 1e-10:
        return None
    a = (
        (tri[1, 1] - tri[2, 1]) * (x - tri[2, 0])
        + (tri[2, 0] - tri[1, 0]) * (y - tri[2, 1])
    ) / den
    b = (
        (tri[2, 1] - tri[0, 1]) * (x - tri[2, 0])
        + (tri[0, 0] - tri[2, 0]) * (y - tri[2, 1])
    ) / den
    c = 1 - a - b
    inside = (a >= -1e-7) & (b >= -1e-7) & (c >= -1e-7)
    return y[inside], x[inside], np.stack([a[inside], b[inside], c[inside]], axis=1)


def visibility(z, x, y, zbuf, mask, tolerance):
    """Return pixels that are inside the matte and on the closest +Z surface."""
    h, w = mask.shape
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    return inside & mask[y, x] & (z >= zbuf[y, x] - tolerance)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("glb")
    p.add_argument("reference")
    p.add_argument("out")
    p.add_argument("--min-iou", type=float, default=0.7)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("output must be empty")
    ref = Image.open(a.reference)
    if ref.mode != "RGBA" or ref.getextrema()[3][0] == 255:
        raise ValueError("reference must have alpha foreground")
    n = 512
    ref.thumbnail((n, n), Image.Resampling.LANCZOS)
    canvas_ref = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    canvas_ref.paste(ref, ((n - ref.width) // 2, (n - ref.height) // 2))
    rgba = np.asarray(canvas_ref)
    mask = rgba[:, :, 3] >= 128
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("reference has no visible foreground")
    scene = trimesh.load(a.glb, force="scene", process=False)
    if len(scene.geometry) != 1:
        raise ValueError("expected a single mesh")
    mesh = next(iter(scene.geometry.values()))
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    uv = np.asarray(mesh.visual.uv)
    if len(verts) == 0 or len(faces) == 0:
        raise ValueError("empty mesh")
    # Orthographic +Z view, +X image right, +Y image up, uniform scale.
    vxy = verts[:, :2].copy()
    vxy[:, 1] *= -1
    b0 = vxy.min(0)
    b1 = vxy.max(0)
    target = np.array([xs.min(), ys.min()])
    extent = np.array([xs.max() - xs.min(), ys.max() - ys.min()])
    scale = min(extent / (b1 - b0))
    offset = target + (extent - (b1 - b0) * scale) / 2 - b0 * scale
    proj = vxy * scale + offset
    canvas = Image.new("1", (n, n))
    draw = ImageDraw.Draw(canvas)
    for f in faces:
        draw.polygon([tuple(x) for x in proj[f]], fill=1)
    base = np.asarray(canvas, dtype=bool)
    best = (-1, None)
    # Estimate a small, uniform camera adjustment by silhouette IoU; no rotation or mirror.
    for factor in (0.9, 0.95, 1.0, 1.05, 1.1):
        for dy in (-12, -6, 0, 6, 12):
            for dx in (-12, -6, 0, 6, 12):
                shift = np.array([dy, dx])
                center = np.array([n / 2, n / 2])
                offset_nd = center - (center + shift) / factor
                trial = affine_transform(
                    base.astype(np.uint8),
                    np.eye(2) / factor,
                    offset=offset_nd,
                    output_shape=(n, n),
                    order=0,
                ).astype(bool)
                union = np.count_nonzero(trial | mask)
                iou = np.count_nonzero(trial & mask) / union if union else 0
                if iou > best[0]:
                    best = (iou, (factor, dx, dy))
    iou, (factor, dx, dy) = best
    if iou < a.min_iou:
        raise ValueError(f"silhouette IoU {iou:.4f} below threshold {a.min_iou}")
    proj = (proj - n / 2) * factor + n / 2 + [dx, dy]
    zbuf = np.full((n, n), -np.inf, dtype=np.float32)
    for f in faces:
        r = raster(proj[f], (n, n))
        if r is None:
            continue
        y, x, w = r
        z = w @ verts[f, 2]
        np.maximum.at(zbuf, (y, x), z)
    image = mesh.visual.material.baseColorTexture
    if image is None:
        raise ValueError("GLB lacks baseColor texture")
    atlas = np.array(image.convert("RGB")).copy()
    h, ww = atlas.shape[:2]
    result = atlas.astype(np.float32)
    distance = distance_transform_edt(mask)
    normals = np.asarray(mesh.vertex_normals)
    changed = np.zeros((h, ww), bool)
    coverage = np.zeros((h, ww), bool)
    uvxy = np.column_stack([uv[:, 0] * (ww - 1), (1 - uv[:, 1]) * (h - 1)])
    tolerance = np.ptp(verts[:, 2]) * 0.01
    for f in faces:
        r = raster(uvxy[f], (h, ww))
        if r is None:
            continue
        y, x, w = r
        coverage[y, x] = True
        pos = w @ verts[f]
        xy = w @ proj[f]
        xi = np.rint(xy[:, 0]).astype(int)
        yi = np.rint(xy[:, 1]).astype(int)
        valid = visibility(pos[:, 2], xi, yi, zbuf, mask, tolerance)
        xi = np.clip(xi, 0, n - 1)
        yi = np.clip(yi, 0, n - 1)
        norm = w @ normals[f]
        norm /= np.maximum(np.linalg.norm(norm, axis=1, keepdims=True), 1e-10)
        weight = (
            np.clip((norm[:, 2] - 0.1) / 0.5, 0, 1)
            * np.clip(distance[yi, xi] / 8, 0, 1)
            * valid
        )
        if not np.any(weight):
            continue
        coords = np.stack([xy[:, 1], xy[:, 0]])
        rgb = np.stack(
            [
                map_coordinates(
                    rgba[:, :, k].astype(float), coords, order=1, mode="nearest"
                )
                for k in range(3)
            ],
            axis=1,
        )
        result[y, x] = result[y, x] * (1 - weight[:, None]) + rgb * weight[:, None]
        changed[y[weight > 0], x[weight > 0]] = True
    if not changed.any():
        raise ValueError("projection did not touch any visible texels")
    # Supply filter padding only outside all UV faces, never overwrite inferred back texels.
    dist, indices = distance_transform_edt(~changed, return_indices=True)
    padding = (~binary_dilation(coverage, iterations=2)) & (dist <= 3)
    result[padding] = result[tuple(indices[:, padding])]
    image = Image.fromarray(np.uint8(np.clip(result, 0, 255)))
    image.save(out / "texture.png")
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=image,
        baseColorFactor=[255] * 4,
        metallicFactor=0,
        roughnessFactor=0.52,
    )
    patched = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=normals,
        visual=trimesh.visual.texture.TextureVisuals(uv=uv, material=material),
        process=False,
    )
    patched.export(out / "model.glb", include_normals=True)
    patched.export(out / "editable.obj", include_normals=True)
    meta = {
        "source_glb_sha256": sha(a.glb),
        "reference_sha256": sha(a.reference),
        "output_glb_sha256": sha(out / "model.glb"),
        "iou": iou,
        "min_iou": a.min_iou,
        "camera": "+Z orthographic, +X right, +Y up",
        "camera_scale": float(scale * factor),
        "registration": {"uniform_factor": factor, "dx": dx, "dy": dy},
        "z_tolerance": float(tolerance),
        "atlas_texels_updated": int(changed.sum()),
        "scope": "visible front only, normal and alpha-border feather; back remains inferred",
        "appearance": "pending human review",
    }
    (out / "settings.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
