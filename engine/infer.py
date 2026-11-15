import argparse
import hashlib
import json
import os
import sys
import time
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import trimesh
import xatlas

ROOT = Path(__file__).resolve().parent.parent
os.environ["U2NET_HOME"] = str(ROOT / ".runtime" / "u2net")
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(1, str(ROOT / ".runtime" / "TripoSR"))
from tsr.system import TSR  # noqa: E402
import tsr.models.tokenizers.image as dino_image  # noqa: E402

dino_image.hf_hub_download = lambda **kwargs: str(
    ROOT / ".runtime" / "model" / "dino-config.json"
)
from tsr.utils import remove_background, resize_foreground  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def new_attempt(out):
    raw_dir = out / "raw-export" / uuid.uuid4().hex
    raw_dir.mkdir(parents=True, exist_ok=False)
    return raw_dir


def preprocess(inp, out, no_remove_bg, foreground_path):
    im = Image.open(inp).convert("RGBA")
    if im.getextrema()[3][0] < 255:
        im = resize_foreground(im, 0.85)
    elif not no_remove_bg:
        import rembg

        im = resize_foreground(remove_background(im, rembg.new_session()), 0.85)
    if im.getextrema()[3][0] < 255:
        im.save(foreground_path)
    im = np.asarray(im).astype(np.float32) / 255
    im = Image.fromarray(
        ((im[:, :, :3] * im[:, :, 3:4] + (1 - im[:, :, 3:4]) * 0.5) * 255).astype(
            "uint8"
        )
    )
    im.save(out / "input-processed.png")
    return im


def bake_and_export(mesh, native_mesh, model, codes, out, texture_size):
    # UV atlas then evaluate model color at texel positions, rasterized in software.
    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    options = xatlas.PackOptions()
    options.padding = 4
    options.resolution = texture_size
    options.bilinear = True
    atlas.generate(pack_options=options)
    vmapping, indices, uvs = atlas[0]
    verts = mesh.vertices[vmapping]
    native_verts = native_mesh.vertices[vmapping]
    n = texture_size
    pos = np.zeros((n, n, 3), np.float32)
    mask = np.zeros((n, n), bool)
    for tri in indices:
        uv = uvs[tri].copy()
        uv[:, 1] = 1 - uv[:, 1]
        xy = uv * (n - 1)
        lo = np.maximum(np.floor(xy.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.ceil(xy.max(axis=0)).astype(int), n - 1)
        if np.any(hi < lo):
            continue
        x, y = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        den = (xy[1, 1] - xy[2, 1]) * (xy[0, 0] - xy[2, 0]) + (xy[2, 0] - xy[1, 0]) * (
            xy[0, 1] - xy[2, 1]
        )
        if abs(den) < 1e-9:
            continue
        w0 = (
            (xy[1, 1] - xy[2, 1]) * (x - xy[2, 0])
            + (xy[2, 0] - xy[1, 0]) * (y - xy[2, 1])
        ) / den
        w1 = (
            (xy[2, 1] - xy[0, 1]) * (x - xy[2, 0])
            + (xy[0, 0] - xy[2, 0]) * (y - xy[2, 1])
        ) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= -0.0001) & (w1 >= -0.0001) & (w2 >= -0.0001)
        yy = y[inside]
        xx = x[inside]
        pos[yy, xx] = (
            w0[inside, None] * native_verts[tri[0]]
            + w1[inside, None] * native_verts[tri[1]]
            + w2[inside, None] * native_verts[tri[2]]
        )
        mask[yy, xx] = True
    print("texture texels:", int(mask.sum()), flush=True)
    if not mask.any():
        raise RuntimeError("empty texture atlas")
    tex = np.zeros((n, n, 3), dtype=np.uint8)
    with torch.no_grad():
        colors = (
            model.renderer.query_triplane(
                model.decoder, torch.from_numpy(pos[mask]), codes[0]
            )["color"]
            .cpu()
            .numpy()
        )
    tex[mask] = np.clip(colors * 255, 0, 255).astype(np.uint8)
    from scipy.ndimage import distance_transform_edt

    _, ix = distance_transform_edt(~mask, return_indices=True)
    tex = tex[tuple(ix)]
    image = Image.fromarray(tex)
    image.save(out / "texture.png")
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=image,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=0.52,
    )
    visual = trimesh.visual.texture.TextureVisuals(uv=uvs, material=material)
    textured = trimesh.Trimesh(
        vertices=verts,
        faces=indices,
        vertex_normals=mesh.vertex_normals[vmapping],
        visual=visual,
        process=False,
    )
    textured.export(out / "model.glb", include_normals=True)
    textured.export(out / "editable.obj", include_normals=True)
    return len(verts), len(indices)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("output")
    p.add_argument("--resolution", type=int, default=64, choices=[64, 96, 128])
    p.add_argument("--texture-size", type=int, default=512, choices=[256, 512, 1024])
    p.add_argument("--no-remove-bg", action="store_true")
    a = p.parse_args()
    inp = Path(a.image).resolve()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    raw_dir = new_attempt(out)
    foreground = raw_dir / "foreground.png"
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model_dir = ROOT / ".runtime" / "model"
    print("initializing model", flush=True)
    model = TSR.from_pretrained(
        str(model_dir), config_name="config.yaml", weight_name="model.ckpt"
    )
    model.renderer.set_chunk_size(2048)
    model.eval()
    print("preprocessing", flush=True)
    im = preprocess(inp, out, a.no_remove_bg, foreground)
    print("running model", flush=True)
    with torch.no_grad():
        codes = model([im], device="cpu")
        print("extracting mesh", flush=True)
        mesh = model.extract_mesh(codes, True, resolution=a.resolution)[0]
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()
    print(
        "mesh:",
        len(mesh.vertices),
        len(mesh.faces),
        "signed volume:",
        mesh.volume,
        flush=True,
    )
    # Keep inference in native coordinates. Rotate the export consistently: (x,y,z)->(y,z,x).
    native_mesh = mesh.copy()
    mesh.vertices = mesh.vertices[:, [1, 2, 0]]
    mesh.export(out / "editable.ply")
    print("baking texture", flush=True)
    vertices, faces = bake_and_export(
        mesh, native_mesh, model, codes, out, a.texture_size
    )
    for name in (
        "model.glb",
        "editable.obj",
        "material.mtl",
        "material_0.png",
        "texture.png",
    ):
        src = out / name
        if src.exists():
            shutil.copy2(src, raw_dir / name)
    raw_sha = sha(raw_dir / "model.glb")
    refinement = {
        "status": "skipped",
        "reason": "no foreground alpha",
        "source_glb_sha256": raw_sha,
    }
    refinement_start = time.time()
    if foreground.exists():
        with tempfile.TemporaryDirectory(prefix="projection-", dir=out) as temp:
            candidate = Path(temp) / "candidate"
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "engine" / "reference_texture.py"),
                        str(raw_dir / "model.glb"),
                        str(foreground),
                        str(candidate),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                if proc.returncode:
                    refinement = {
                        "status": "skipped",
                        "reason": proc.stderr[-1000:] or proc.stdout[-1000:],
                        "source_glb_sha256": raw_sha,
                    }
                else:
                    refinement = json.loads(
                        (candidate / "settings.json").read_text(encoding="utf8")
                    )
                    if refinement["source_glb_sha256"] != raw_sha or refinement[
                        "output_glb_sha256"
                    ] != sha(candidate / "model.glb"):
                        raise ValueError("projection hash mismatch")
                    for name in (
                        "model.glb",
                        "editable.obj",
                        "material.mtl",
                        "material_0.png",
                        "texture.png",
                    ):
                        if (candidate / name).exists():
                            shutil.copy2(candidate / name, out / name)
                    refinement["status"] = "used"
            except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
                for name in (
                    "model.glb",
                    "editable.obj",
                    "material.mtl",
                    "material_0.png",
                    "texture.png",
                ):
                    if (raw_dir / name).exists():
                        shutil.copy2(raw_dir / name, out / name)
                refinement = {
                    "status": "skipped",
                    "reason": str(exc),
                    "source_glb_sha256": raw_sha,
                }
    refinement["elapsed_seconds"] = round(time.time() - refinement_start, 2)
    meta = {
        "model": "stabilityai/TripoSR",
        "source_commit": "107cefdc244c39106fa830359024f6a2f1c78871",
        "device": "cpu",
        "export_material": {
            "smooth_normals": True,
            "metallic_factor": 0.0,
            "roughness_factor": 0.52,
            "base_color_factor": [1, 1, 1, 1],
        },
        "marching_cubes_resolution": a.resolution,
        "texture_size": a.texture_size,
        "input_sha256": sha(inp),
        "reference_refinement": refinement,
        "raw_glb_sha256": raw_sha,
        "raw_export": str(raw_dir.relative_to(out)).replace("\\", "/"),
        "appearance": "pending human review",
        "model_sha256": sha(model_dir / "model.ckpt"),
        "glb_sha256": sha(out / "model.glb"),
        "vertices": vertices,
        "faces": faces,
        "elapsed_seconds": round(time.time() - start, 2),
        "coordinates": "native (x,y,z) to glTF (y,z,x); native +Z up and +X camera become glTF +Y up and +Z camera; no scale",
        "background_removal": not a.no_remove_bg,
    }
    (out / "settings.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
