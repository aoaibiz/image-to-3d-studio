"""Experimental shading-only polish for raw indexed TripoSG isosurfaces."""

import json
import struct
from pathlib import Path

import numpy as np
from scipy import ndimage
import trimesh


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-20)


def _sample(grid, at):
    return ndimage.map_coordinates(grid, at.T, order=1, mode="nearest", prefilter=False)


def sdf_vertex_normals(mesh, field, *, bounds=(-1.005, 1.005), sigma=1.5):
    """Estimate lighting normals; geometry, winding and scene remain untouched.

    Vertices and the uniformly sampled scalar grid must share an XYZ frame.
    The median agreement with the incoming normals determines the sign of the
    learned scalar field. Low-confidence vertices keep their original normal.
    """
    field = np.asarray(field, dtype=np.float32)
    if field.ndim != 3 or len(set(field.shape)) != 1 or not 8 <= len(field) <= 257:
        raise ValueError("Expected a cubic scalar grid of 8 to 257 samples per axis.")
    if not np.isfinite(field).all() or not 0 < sigma <= 2 or not bounds[0] < bounds[1]:
        raise ValueError("Invalid SDF, grid bounds or Gaussian scale.")
    v = np.asarray(mesh.vertices)
    original = _unit(np.asarray(mesh.vertex_normals, dtype=np.float64))
    if (
        not len(v)
        or not len(mesh.faces)
        or not np.isfinite(v).all()
        or not np.isfinite(original).all()
    ):
        raise ValueError("A finite mesh with surface normals is required.")
    pitch = (bounds[1] - bounds[0]) / (len(field) - 1)
    grid = (v - bounds[0]) / pitch
    if np.any((grid < -1e-3) | (grid > len(field) - 1 + 1e-3)):
        raise ValueError("Mesh is outside the SDF coordinate frame.")
    smooth = ndimage.gaussian_filter(field, sigma=sigma, mode="nearest")
    gradient = np.empty(v.shape, dtype=np.float64)
    for axis in range(3):
        offset = np.eye(3)[axis]
        gradient[:, axis] = (
            _sample(smooth, grid + offset) - _sample(smooth, grid - offset)
        ) * 0.5
    strength = np.linalg.norm(gradient, axis=1)
    neural = _unit(gradient)
    alignment = np.einsum("ij,ij->i", neural, original)
    reliable = strength > max(float(np.median(strength)) * 0.03, 1e-12)
    sign = 1 if not np.any(reliable) or np.median(alignment[reliable]) >= 0 else -1
    neural *= sign
    alignment *= sign
    # A second crossing a few voxels into the solid indicates a thin shell,
    # pole or narrow gap. A normal averaged across it would be unreliable.
    reach = 2 * sigma + 0.75
    inside = sign * _sample(field, grid - reach * original)
    outside = sign * _sample(field, grid + reach * original)
    thin = (inside >= 0) | (outside <= 0)
    # Multiple incident planes meeting sharply represent a physical crease.
    # Raw TripoSG meshes share vertex indices; also protect boundary edges.
    sharp = np.zeros(len(v), dtype=bool)
    if len(mesh.face_adjacency_edges):
        edge_index = mesh.face_adjacency_edges[
            mesh.face_adjacency_angles >= np.deg2rad(52)
        ]
        sharp[edge_index.ravel()] = True
    boundary_edges = (
        np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique)) != 2
    )
    sharp[mesh.edges_unique[boundary_edges].ravel()] = True
    zero_error = np.abs(_sample(field, grid))
    near_surface = zero_error <= 1.2 * strength + 1e-10
    if np.mean(near_surface) < 0.85:
        raise ValueError("Mesh and SDF zero surface disagree.")
    weight = np.clip((alignment - 0.66) / 0.23, 0, 1)
    weight *= ~thin & ~sharp & reliable & near_surface
    weight = weight**2 * (3 - 2 * weight)
    result = _unit(original + weight[:, None] * (neural - original))
    return result.astype("float32"), {
        "method": "trilinear gradient of isotropically filtered signed-distance samples",
        "sigma_voxels": sigma,
        "grid_cells": len(field) - 1,
        "pitch": pitch,
        "sign": sign,
        "vertices": len(v),
        "faces": len(mesh.faces),
        "adjusted": int(np.count_nonzero(weight > 0)),
        "hard_edge_or_boundary": int(sharp.sum()),
        "thin_or_ambiguous_crossing": int(thin.sum()),
        "weak_gradient": int((~reliable).sum()),
    }


def shade_glb(source, field_path, destination):
    """Write a new GLB, appending NORMAL only; preserve original buffer bytes.

    Accept only the simple, identity-frame, single primitive produced by the
    raw shape extractor. A general transformed/textured scene is a different
    coordinate contract and must not be paired blindly with this scalar grid.
    """
    source, destination = Path(source), Path(destination)
    data = source.read_bytes()
    magic, ver, size, n, typ = struct.unpack_from("<4sIIII", data)
    if (magic, ver, size, typ) != (b"glTF", 2, len(data), 0x4E4F534A):
        raise ValueError("Invalid GLB header.")
    doc = json.loads(data[20 : 20 + n])
    bsize, btyp = struct.unpack_from("<II", data, 20 + n)
    binary = data[28 + n :]
    if btyp != 0x004E4942 or bsize != len(binary) or len(doc["buffers"]) != 1:
        raise ValueError("Expected one embedded buffer.")
    if (
        len(doc["meshes"]) != 1
        or len(doc["meshes"][0]["primitives"]) != 1
        or any(
            any(
                key in node
                for key in ("rotation", "scale", "translation", "matrix", "skin")
            )
            for node in doc["nodes"]
        )
    ):
        raise ValueError("Raw, untransformed single-primitive mesh required.")
    prim = doc["meshes"][0]["primitives"][0]
    if prim.get("mode", 4) != 4 or "targets" in prim:
        raise ValueError("Raw triangle mesh required.")

    def accessor(index, expected):
        a = doc["accessors"][index]
        view = doc["bufferViews"][a["bufferView"]]
        dtype = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[a["componentType"]]
        components = 3 if expected == "VEC3" else 1
        if (
            a["type"] != expected
            or (expected == "VEC3" and a["componentType"] != 5126)
            or view.get("buffer", 0) != 0
            or "byteStride" in view
            or "sparse" in a
        ):
            raise ValueError("Unexpected raw mesh accessor layout.")
        return np.frombuffer(
            binary,
            dtype,
            offset=view.get("byteOffset", 0) + a.get("byteOffset", 0),
            count=components * a["count"],
        ).reshape(-1, components)

    attr = prim["attributes"]
    vertices = accessor(attr["POSITION"], "VEC3")
    faces = accessor(prim["indices"], "SCALAR").reshape(-1, 3)
    given = accessor(attr["NORMAL"], "VEC3") if "NORMAL" in attr else None
    mesh = trimesh.Trimesh(vertices, faces, vertex_normals=given, process=False)
    normals, metrics = sdf_vertex_normals(
        mesh, np.load(field_path, mmap_mode="r", allow_pickle=False)
    )
    extra = normals.astype("<f4", copy=False).tobytes()
    offset = len(binary)
    if offset % 4:
        raise ValueError("Unaligned BIN chunk.")
    attr["NORMAL"] = len(doc["accessors"])
    doc["accessors"].append(
        {
            "bufferView": len(doc["bufferViews"]),
            "componentType": 5126,
            "type": "VEC3",
            "count": len(vertices),
        }
    )
    doc["bufferViews"].append(
        {"buffer": 0, "byteOffset": offset, "byteLength": len(extra), "target": 34962}
    )
    doc["buffers"][0]["byteLength"] = offset + len(extra)
    j = json.dumps(doc, separators=(",", ":")).encode()
    j += b" " * (-len(j) % 4)
    total = 28 + len(j) + offset + len(extra)
    result = (
        struct.pack("<4sIIII", b"glTF", 2, total, len(j), typ)
        + j
        + struct.pack("<II", offset + len(extra), btyp)
        + binary
        + extra
    )
    with destination.open("xb") as output:
        output.write(result)
    return metrics
