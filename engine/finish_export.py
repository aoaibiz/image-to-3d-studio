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


def _normal_map_tangents(mesh):
    """Area-weighted tangent frames in glTF UV coordinates (V is flipped on export)."""
    positions = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    if not np.all(np.isfinite(norm)) or np.any(norm < 1e-12):
        raise ValueError("normal-mapped vertices need nonzero finite normals")
    normals = normals / norm
    uv = mesh.visual.uv.copy()
    uv[:, 1] = 1 - uv[:, 1]
    a, b, c = mesh.faces.T
    edge1, edge2 = positions[b] - positions[a], positions[c] - positions[a]
    uv1, uv2 = uv[b] - uv[a], uv[c] - uv[a]
    det = uv1[:, 0] * uv2[:, 1] - uv1[:, 1] * uv2[:, 0]
    weight = np.divide(
        np.linalg.norm(np.cross(edge1, edge2), axis=1),
        det,
        out=np.zeros_like(det),
        where=np.abs(det) > 1e-12,
    )[:, None]
    per_face_u = (edge1 * uv2[:, 1:2] - edge2 * uv1[:, 1:2]) * weight
    per_face_v = (edge2 * uv1[:, 0:1] - edge1 * uv2[:, 0:1]) * weight
    tangent, bitangent = np.zeros_like(positions), np.zeros_like(positions)
    for corners in mesh.faces.T:
        np.add.at(tangent, corners, per_face_u)
        np.add.at(bitangent, corners, per_face_v)
    tangent -= np.sum(tangent * normals, axis=1, keepdims=True) * normals
    weak = np.linalg.norm(tangent, axis=1) < 1e-12
    axes = np.eye(3)[np.argmin(np.abs(normals[weak]), axis=1)]
    tangent[weak] = np.cross(normals[weak], axes)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-15)
    handed = np.where(np.sum(np.cross(normals, tangent) * bitangent, axis=1) < 0, -1, 1)
    return np.column_stack([tangent, handed]).astype(np.float32)


def _gltf_scene(scene):
    geometry = scene.geometry.copy()
    for name, mesh in scene.geometry.items():
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("non-mesh scene geometry unsupported")
        visual = mesh.visual
        if visual.kind != "texture":
            continue
        mat = visual.material
        has_normal = getattr(mat, "normalTexture", None) is not None
        if visual.uv is not None and len(visual.uv) != len(mesh.vertices):
            raise ValueError("invalid UV coordinate count")
        if visual.uv is None and any(
            getattr(mat, prop, None) is not None
            for prop in (
                "image",
                "baseColorTexture",
                "metallicRoughnessTexture",
                "normalTexture",
                "emissiveTexture",
                "occlusionTexture",
            )
        ):
            raise ValueError("missing UV coordinates for a material texture")
        if visual.uv is None or has_normal:
            # Trimesh 4.0.5 only binds a material when its UVs are present.
            # Constant factors can use any UV, even if the author supplied none.
            patched = mesh.copy(include_cache=True)
            patched.vertex_attributes.update(mesh.vertex_attributes)
            if visual.uv is None:
                patched.visual.uv = np.zeros((len(mesh.vertices), 2))
            if has_normal:
                patched.vertex_attributes["TANGENT"] = _normal_map_tangents(patched)
            geometry[name] = patched
    return trimesh.Scene(
        geometry,
        graph=scene.graph.copy(),
        metadata=scene.metadata,
        camera=scene.camera.copy() if scene.has_camera else None,
    )


def _standard_tangents(doc):
    # Trimesh encodes user vertex attributes with an underscore.
    for mesh in doc["meshes"]:
        for primitive in mesh["primitives"]:
            attrs = primitive["attributes"]
            if "_TANGENT" in attrs:
                attrs["TANGENT"] = attrs.pop("_TANGENT")


def _base_color(material):
    """Bake the linear glTF factor into sRGB texels; alpha is always linear."""
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(material, "image", None)
    if image is None:
        image = Image.new("RGBA", (1, 1), "white")
    rgba = np.array(image.convert("RGBA"))
    factor = getattr(material, "baseColorFactor", None)
    factor = np.ones(4) if factor is None else np.asarray(factor, dtype=float) / 255
    if not np.all(factor[:3] == 1):
        srgb = rgba[:, :, :3].astype(np.float32) / 255
        linear = np.where(
            srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4
        )
        linear *= factor[:3]
        srgb = np.where(
            linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055
        )
        rgba[:, :, :3] = np.rint(srgb * 255).clip(0, 255).astype(np.uint8)
    alpha = rgba[:, :, 3].astype(float) / 255 * factor[3]
    mode = getattr(material, "alphaMode", None) or "OPAQUE"
    if mode == "MASK":
        cutoff = getattr(material, "alphaCutoff", None)
        alpha = alpha >= (0.5 if cutoff is None else cutoff)
    elif mode == "OPAQUE":
        alpha = np.ones_like(alpha)
    rgba[:, :, 3] = np.rint(alpha * 255).astype(np.uint8)
    return Image.fromarray(rgba)


def world_meshes(scene):
    """Flatten instances while retaining imported corner shading through chart splits."""
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        source = scene.geometry[name]
        if not isinstance(source, trimesh.Trimesh):
            raise ValueError("non-mesh scene geometry unsupported")
        # Trimesh stores imported NORMAL in the mesh cache. Scene.dump() and
        # ordinary copies discard it, causing lighting seams on UV boundaries.
        source_normals = np.asarray(source.vertex_normals)
        instance = source.copy()
        instance.apply_transform(transform)
        linear = np.asarray(transform)[:3, :3]
        try:
            normals = source_normals @ np.linalg.inv(linear)
        except np.linalg.LinAlgError:
            # A collapsed instance has no inverse normal matrix. Its flattened
            # triangles can only supply geometric normals where area remains.
            normals = instance.vertex_normals
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        instance.vertex_normals = normals / np.maximum(length, 1e-15)
        instance.metadata.update(name=name, node=node)
        meshes.append(instance)
    return meshes


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
    scene = _gltf_scene(scene)
    (stage / "model.glb").write_bytes(
        trimesh.exchange.gltf.export_glb(
            scene, include_normals=True, tree_postprocessor=_standard_tangents
        )
    )
    embed_master(stage / "model.glb", stage / "editable.gltf")
    meshes = world_meshes(scene)
    colored = []
    material_maps = {}
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
            # Instances and layers can share one high-resolution contact texture.
            source_material = scene.geometry[mesh.metadata["name"]].visual.material
            key = id(source_material)
            if key not in material_maps:
                image = _base_color(source_material)
                color_map = f"material_{i}.png"
                image.convert("RGB").save(stage / color_map)
                alpha = image.getchannel("A")
                low, high = alpha.getextrema()
                maps = ["Kd 1 1 1", f"map_Kd {color_map}"]
                if low != high:
                    opacity_map = f"material_{len(meshes) + i}.png"
                    alpha.save(stage / opacity_map)
                    maps.append(f"map_d {opacity_map}")
                elif low < 255:
                    maps.append(f"d {low / 255:.9g}")
                material_maps[key] = image, maps
            image, maps = material_maps[key]
            mtl.extend([f"newmtl material_{i}", *maps, ""])
            obj.append(f"usemtl material_{i}")
            colors = np.full((len(mesh.vertices), 3), 255, dtype=np.uint8)
            # Match the default glTF repeat sampler, including UVs outside [0,1].
            uv = np.asarray(visual.uv)
            x = np.floor(uv[:, 0] * image.width).astype(np.int64) % image.width
            y = np.floor((1 - uv[:, 1]) * image.height).astype(np.int64) % image.height
            color_mesh = mesh.copy()
            color_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=color_mesh,
                vertex_colors=np.asarray(image)[y, x],
            )
            colored.append(color_mesh)
        elif visual.kind in ("vertex", None):
            colored.append(mesh)
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
        for normal in mesh.vertex_normals:
            obj.append("vn " + " ".join(f"{value:.9g}" for value in normal))
        if visual.kind == "texture":
            for uv in visual.uv:
                obj.append("vt " + " ".join(f"{x:.9g}" for x in uv))
            for face in mesh.faces:
                obj.append(
                    "f "
                    + " ".join(
                        f"{vertex_offset + int(k) + 1}/{uv_offset + int(k) + 1}/{vertex_offset + int(k) + 1}"
                        for k in face
                    )
                )
            uv_offset += len(visual.uv)
        else:
            for face in mesh.faces:
                obj.append(
                    "f "
                    + " ".join(
                        f"{vertex_offset + int(k) + 1}//{vertex_offset + int(k) + 1}"
                        for k in face
                    )
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
    combined.vertex_normals = np.vstack([mesh.vertex_normals for mesh in meshes])
    combined.export(stage / "editable.ply", file_type="ply", vertex_normal=True)
    return {"mesh_count": len(meshes), "ply_texture": "vertex-sampled approximation"}
