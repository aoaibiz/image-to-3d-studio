"""Export finished scenes; GLB and self-contained glTF are authoritative."""

import base64
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def embed_master(glb: Path, target: Path):
    """Convert a GLB to self-contained glTF without reconstructing its scene."""
    raw = Path(glb).read_bytes()
    if len(raw) < 20:
        raise ValueError("truncated GLB")
    magic, version, length = struct.unpack_from("<4sII", raw)
    if magic != b"glTF" or version != 2 or length != len(raw):
        raise ValueError("invalid GLB header")
    chunks = []
    offset = 12
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise ValueError("truncated GLB chunk")
        size, kind = struct.unpack_from("<II", raw, offset)
        offset += 8
        if offset + size > len(raw):
            raise ValueError("truncated GLB chunk")
        chunks.append((kind, raw[offset : offset + size]))
        offset += size
    if (
        not chunks
        or chunks[0][0] != 0x4E4F534A
        or sum(k == 0x4E4F534A for k, _ in chunks) != 1
        or sum(k == 0x004E4942 for k, _ in chunks) > 1
    ):
        raise ValueError("unsupported GLB chunks")
    if any(k not in (0x4E4F534A, 0x004E4942) for k, _ in chunks):
        raise ValueError("unknown GLB chunk")
    doc = json.loads(chunks[0][1].decode("utf-8").rstrip(" \0"))
    buffers = doc.get("buffers", [])
    if len(buffers) > 1 or (buffers and "uri" in buffers[0]):
        raise ValueError("external buffers unsupported")
    bins = [v for k, v in chunks if k == 0x004E4942]
    if buffers:
        if not bins or len(bins[0]) < buffers[0]["byteLength"]:
            raise ValueError("missing GLB buffer")
        buffers[0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(
            bins[0][: buffers[0]["byteLength"]]
        ).decode("ascii")
    elif bins:
        raise ValueError("unexpected GLB buffer")
    if any(
        "uri" in image and not image["uri"].startswith("data:")
        for image in doc.get("images", [])
    ):
        raise ValueError("external images unsupported")
    Path(target).write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")


def export_finished(scene, stage: Path):
    """Write a scene's final artifacts into an existing or new staging directory.

    GLB/glTF carry full material detail. OBJ/PLY have format-specific limits;
    PLY stores vertex colors, approximating textured surfaces at vertices.
    """
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)
    if not isinstance(scene, trimesh.Scene) or not scene.geometry:
        raise ValueError("nonempty scene required")
    (stage / "model.glb").write_bytes(
        trimesh.exchange.gltf.export_glb(scene, include_normals=True)
    )
    embed_master(stage / "model.glb", stage / "editable.gltf")
    # Scene.dump applies instance transforms before exporting OBJ and PLY.
    meshes = scene.dump(concatenate=False)
    colored = []
    for mesh in meshes:
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("non-mesh scene geometry unsupported")
        color_mesh = mesh.copy()
        if mesh.visual.kind == "texture":
            if mesh.visual.uv is None or len(mesh.visual.uv) != len(mesh.vertices):
                raise ValueError("missing UV coordinates")
            color_mesh.visual = mesh.visual.to_color()
        colored.append(color_mesh)
    # Write one consistent vertex format for textured and colored meshes. OBJ's
    # widely supported vertex RGB extension carries untextured car parts.
    obj = ["mtllib material.mtl"]
    mtl = []
    vertex_offset = 0
    uv_offset = 0
    for i, mesh in enumerate(meshes):
        visual = mesh.visual
        obj.append(f"o part_{i}")
        if visual.kind == "texture":
            image = getattr(visual.material, "baseColorTexture", None)
            if image is None:
                image = getattr(visual.material, "image", None)
            if image is None:
                raise ValueError("texture material without image unsupported")
            factor = getattr(visual.material, "baseColorFactor", None)
            factor = (
                np.asarray(
                    factor if factor is not None else [255, 255, 255, 255],
                    dtype=np.float64,
                )
                / 255
            )
            if len(factor) != 4 or factor[3] != 1:
                raise ValueError("transparent texture factor unsupported")
            rgba = np.asarray(image.convert("RGBA")).copy()
            rgb = rgba[:, :, :3].astype(np.float64) / 255
            linear = (
                np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
                * factor[:3]
            )
            rgb = np.where(
                linear <= 0.0031308,
                linear * 12.92,
                1.055 * np.power(linear, 1 / 2.4) - 0.055,
            )
            rgba[:, :, :3] = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
            Image.fromarray(rgba).save(stage / f"material_{i}.png")
            mtl.extend(
                [f"newmtl material_{i}", "Kd 1 1 1", f"map_Kd material_{i}.png", ""]
            )
            obj.append(f"usemtl material_{i}")
            colors = np.full((len(mesh.vertices), 3), 255, dtype=np.uint8)
        elif visual.kind in ("vertex", None):
            colors = (
                np.asarray(visual.vertex_colors)[:, :3]
                if visual.kind == "vertex"
                else np.full((len(mesh.vertices), 3), 255, dtype=np.uint8)
            )
            mtl.extend([f"newmtl material_{i}", "Kd 1 1 1", ""])
            obj.append(f"usemtl material_{i}")
        else:
            raise ValueError(
                "face colors and unsupported visuals cannot be exported faithfully"
            )
        for v, c in zip(mesh.vertices, colors):
            obj.append(
                "v "
                + " ".join(f"{x:.9g}" for x in v)
                + " "
                + " ".join(f"{x / 255:.9g}" for x in c)
            )
        if visual.kind == "texture":
            for uv in visual.uv:
                obj.append("vt " + " ".join(f"{x:.9g}" for x in uv))
            for face in mesh.faces:
                obj.append(
                    "f "
                    + " ".join(
                        f"{vertex_offset + int(k) + 1}/{uv_offset + int(k) + 1}"
                        for k in face
                    )
                )
            uv_offset += len(visual.uv)
        else:
            for face in mesh.faces:
                obj.append(
                    "f " + " ".join(str(vertex_offset + int(k) + 1) for k in face)
                )
        vertex_offset += len(mesh.vertices)
    (stage / "editable.obj").write_text("\n".join(obj) + "\n", encoding="utf-8")
    (stage / "material.mtl").write_text("\n".join(mtl), encoding="utf-8")
    # Keep stable texture entry points for consumers; never replace actual OBJ references.
    pngs = sorted(stage.glob("material_*.png"))
    if pngs:
        if not (stage / "material_0.png").exists():
            shutil.copyfile(pngs[0], stage / "material_0.png")
        shutil.copyfile(stage / "material_0.png", stage / "texture.png")
    else:
        Image.new("RGB", (1, 1), "white").save(stage / "material_0.png")
        shutil.copyfile(stage / "material_0.png", stage / "texture.png")
        if not (stage / "material.mtl").exists():
            (stage / "material.mtl").write_text(
                "# Vertex-color-only scene; see OBJ vertex colors.\n", encoding="utf-8"
            )
    combined = trimesh.util.concatenate(colored)
    combined.export(stage / "editable.ply", file_type="ply")
    return {"mesh_count": len(meshes), "ply_texture": "vertex-sampled approximation"}
