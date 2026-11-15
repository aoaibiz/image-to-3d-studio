"""Repair normals and PBR material using the original mesh and baked atlas."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    args = parser.parse_args()
    job = Path(args.job).resolve()
    old = job / "original-export"
    old.mkdir(exist_ok=True)
    for name in (
        "model.glb",
        "editable.obj",
        "material.mtl",
        "material_0.png",
        "settings.json",
    ):
        path = job / name
        if path.exists() and not (old / name).exists():
            shutil.copy2(path, old / name)
    source = trimesh.load(job / "editable.ply", process=False)
    scene = trimesh.load(job / "model.glb", force="scene", process=False)
    if len(scene.geometry) != 1:
        raise ValueError("expected one mesh in GLB")
    m = next(iter(scene.geometry.values()))
    distances, indices = cKDTree(source.vertices).query(m.vertices)
    if float(distances.max()) > 1e-6:
        raise ValueError(f"UV vertices do not match source mesh: {distances.max()}")
    image = Image.open(job / "texture.png").convert("RGB")
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=image,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=0.52,
    )
    mesh = trimesh.Trimesh(
        vertices=m.vertices,
        faces=m.faces,
        vertex_normals=source.vertex_normals[indices],
        visual=trimesh.visual.texture.TextureVisuals(uv=m.visual.uv, material=material),
        process=False,
    )
    mesh.export(job / "model.glb", include_normals=True)
    mesh.export(job / "editable.obj", include_normals=True)
    meta_path = job / "settings.json"
    meta = json.loads(meta_path.read_text(encoding="utf8"))
    meta["original_glb_sha256"] = sha(old / "model.glb")
    meta["glb_sha256"] = sha(job / "model.glb")
    meta["export_repair"] = {
        "smooth_normals": True,
        "metallic_factor": 0.0,
        "roughness_factor": 0.52,
        "base_color_factor": [1, 1, 1, 1],
        "max_vertex_distance": float(distances.max()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf8")
    print(json.dumps(meta["export_repair"], indent=2))
    print("glb_sha256", meta["glb_sha256"])


if __name__ == "__main__":
    main()
