"""Bake observed colour onto an existing shape; hidden material remains inferred."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
import xatlas
from numba import njit
from PIL import Image, ImageOps
from scipy import ndimage
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import cg, spsolve
from scipy.spatial import cKDTree

from engine.finish_export import export_finished, world_meshes
from engine.shape_camera import lens_parameters, project_pinhole


def _sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _load(path, *, require_native=False):
    scene = trimesh.load(path, force="scene", process=False)
    if require_native:
        nodes = list(scene.graph.nodes_geometry)
        if (
            len(nodes) != 1
            or len(scene.geometry) != 1
            or not np.allclose(scene.graph[nodes[0]][0], np.eye(4), rtol=0, atol=1e-9)
        ):
            raise ValueError("SDF shading requires one native untransformed shape.")
    meshes = world_meshes(scene)
    for mesh in meshes:
        lengths = np.linalg.norm(mesh.vertex_normals, axis=1)
        if not np.isfinite(lengths).all() or np.any(lengths < 1e-12):
            raise ValueError(
                "Input shape contains invalid or undefined vertex normals."
            )
    if not meshes:
        raise ValueError("A nonempty triangle scene is required.")
    return meshes


def _matte(path, n):
    with Image.open(path) as f:
        rgba = ImageOps.exif_transpose(f).convert("RGBA")
    alpha = rgba.getchannel("A")
    box = alpha.point(lambda a: 255 if a >= 150 else 0).getbbox()
    if alpha.getextrema()[0] > 32 or box is None:
        raise ValueError(
            "A separately segmented, nonempty RGBA foreground is required."
        )
    rgba = rgba.crop(box)
    scale = (n - 12) / max(rgba.size)
    rgba = rgba.resize(
        tuple(max(1, round(s * scale)) for s in rgba.size), Image.Resampling.LANCZOS
    )
    canvas = Image.new("RGBA", (n, n))
    canvas.paste(rgba, ((n - rgba.width) // 2, (n - rgba.height) // 2))
    return np.array(canvas)


def _source_agreement(path, matte):
    """Verify that the separately supplied matte still contains this photograph."""
    with Image.open(path) as photo:
        image = ImageOps.exif_transpose(photo).convert("RGB")
        image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    image = np.asarray(image)
    valid = np.uint8(matte[:, :, 3] > 220) * 255
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8))
    detector = cv2.SIFT_create(nfeatures=2000)
    a, da = detector.detectAndCompute(
        cv2.cvtColor(matte[:, :, :3], cv2.COLOR_RGB2GRAY), valid
    )
    b, db = detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
    if da is not None and db is not None and len(db) > 1:
        pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
        selected = [
            first for first, second in pairs if first.distance < 0.73 * second.distance
        ]
        if len(selected) >= 7:
            src = np.float32([a[m.queryIdx].pt for m in selected])
            dst = np.float32([b[m.trainIdx].pt for m in selected])
            matrix, inliers = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5
            )
            if matrix is not None and np.count_nonzero(inliers) >= 7:
                warped = cv2.warpAffine(
                    image,
                    matrix,
                    matte.shape[:2][::-1],
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                )
                diff = np.max(np.abs(warped.astype(float) - matte[:, :, :3]), axis=2)[
                    valid > 0
                ]
                if np.median(diff) <= 15 and (diff < 32).mean() >= 0.78:
                    return {
                        "method": "registered local features and opaque RGB residual",
                        "inliers": int(np.count_nonzero(inliers)),
                        "median_max_channel_error": round(float(np.median(diff)), 2),
                    }
    # A flat-colour logo can have few distinctive keypoints. Match the visible
    # colour layout at thumbnail scales; background contributes no pixels.
    original = Image.fromarray(image)
    original.thumbnail((192, 192), Image.Resampling.LANCZOS)
    original = np.array(original)
    y, x = np.where(valid > 0)
    if not len(x):
        raise ValueError("No opaque foreground for source comparison.")
    panel = matte[y.min() : y.max() + 1, x.min() : x.max() + 1]
    best = float("inf")
    for factor in np.geomspace(0.12, 1, 48):
        scale = factor * min(
            original.shape[1] / panel.shape[1], original.shape[0] / panel.shape[0]
        )
        size = tuple(max(1, int(value * scale)) for value in panel.shape[:2][::-1])
        if min(size) < 8:
            continue
        small = cv2.resize(panel, size, interpolation=cv2.INTER_AREA)
        solid = cv2.erode(np.uint8(small[:, :, 3] > 240), np.ones((3, 3), np.uint8))
        if solid.sum() < 32:
            continue
        err = cv2.matchTemplate(original, small[:, :, :3], cv2.TM_SQDIFF, mask=solid)
        best = min(best, float(np.sqrt(max(0, np.min(err)) / (3 * solid.sum()))))
    if best > 15:
        raise ValueError(
            "The supplied foreground colours could not be verified against the original photograph."
        )
    return {
        "method": "masked multiscale RGB template comparison",
        "rms_error": round(best, 2),
    }


@njit
def _raster(projected, triangles, n):
    """Closest triangle and two barycentric coordinates at pixel centres."""
    depth = np.full((n, n), -np.inf, np.float32)
    face = np.full((n, n), -1, np.int32)
    weights = np.zeros((n, n, 2), np.float32)
    for i in range(len(triangles)):
        a, b, c = projected[triangles[i]]
        abx, aby = b[0] - a[0], b[1] - a[1]
        acx, acy = c[0] - a[0], c[1] - a[1]
        area = abx * acy - aby * acx
        if abs(area) < 1e-10:
            continue
        x0 = max(0, int(np.ceil(min(a[0], b[0], c[0]))))
        x1 = min(n - 1, int(np.floor(max(a[0], b[0], c[0]))))
        y0 = max(0, int(np.ceil(min(a[1], b[1], c[1]))))
        y1 = min(n - 1, int(np.floor(max(a[1], b[1], c[1]))))
        for row in range(y0, y1 + 1):
            for col in range(x0, x1 + 1):
                dx, dy = col - a[0], row - a[1]
                wb = (dx * acy - dy * acx) / area
                wc = (abx * dy - aby * dx) / area
                if min(wb, wc, 1 - wb - wc) >= -1e-6:
                    z = a[2] + wb * (b[2] - a[2]) + wc * (c[2] - a[2])
                    if z >= depth[row, col]:
                        depth[row, col] = z
                        face[row, col] = i
                        weights[row, col, 0] = wb
                        weights[row, col, 1] = wc
    return face, weights, depth


def _norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-10)


def _display_matrix(value):
    if value is None:
        return None
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(
            matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6, rtol=0
        )
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError(
            "display_from_raw must be a proper rigid transform without scale, shear or reflection."
        )
    return matrix


def _basis(yaw, pitch, up):
    a, b = np.radians([yaw, pitch])
    right = [np.cos(a), 0, -np.sin(a)]
    outward = [np.sin(a) * np.cos(b), np.sin(b), np.cos(a) * np.cos(b)]
    rotation = np.array([right, np.cross(outward, right), outward])
    if up == "+Z":
        rotation = rotation @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    return rotation


def _lens_coordinates(verts, basis, lens):
    if lens is None:
        return None
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    return (lo + hi) * 0.5 @ basis.T, lens[0] / np.max(hi - lo)


def _fit(verts, basis, mask, lens=None):
    xyz = verts @ basis.T
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    y, x = np.nonzero(mask)
    target_mid = np.array([x.min() + x.max(), y.min() + y.max()]) * 0.5
    span = np.array([np.ptp(x), np.ptp(y)])
    projected = xyz.copy()
    xlo, xhi = lo[:2], hi[:2]
    if lens is not None:
        projected = project_pinhole(xyz, *_lens_coordinates(verts, basis, lens))
        xlo, xhi = projected[:, :2].min(0), projected[:, :2].max(0)
        target_mid += max(span) * lens[2:]
    scale = np.min(span / np.maximum(xhi - xlo, 1e-9))
    if lens is not None:
        scale *= lens[1]
    offset = target_mid - ((xlo + xhi) * 0.5 * [1, -1]) * scale
    projected[:, :2] = projected[:, :2] * [scale, -scale] + offset
    return projected, scale, offset, (lo + hi) / 2, max(hi[1] - lo[1], 1e-9)


def _registration(verts, faces, matte, hint, up, frame=None):
    low = (
        np.asarray(Image.fromarray(matte).resize((96, 96), Image.Resampling.BOX))[
            :, :, 3
        ]
        >= 150
    )
    if frame is not None:
        if hint:
            raise ValueError("Select either a camera frame or an axis hint.")
        basis = np.array(
            [
                frame[key]
                for key in ("image_right", "image_up", "camera_direction_from_origin")
            ],
            dtype=float,
        )
        if (
            basis.shape != (3, 3)
            or not np.isfinite(basis).all()
            or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-4)
            or np.linalg.det(basis) < 0.999
        ):
            raise ValueError(
                "Camera frame must be a proper orthonormal basis without reflection."
            )
        lens = lens_parameters(frame)
        projected, *_ = _fit(verts, basis, low, lens)
        mask = _raster(projected, faces, 96)[0] >= 0
        score = np.count_nonzero(mask & low) / max(1, np.count_nonzero(mask | low))
        if score < 0.72:
            raise ValueError(
                f"Supplied camera frame does not register (IoU {score:.3f})."
            )
        return basis, {
            "method": "supplied source-camera frame verified against observed silhouette",
            "basis_rows": basis.tolist(),
            **({"pinhole": frame["pinhole"]} if lens is not None else {}),
            "silhouette_iou": round(score, 5),
            "photographic_pose_verified": False,
        }
    init = {None: 0, "+Z": 0, "+X": 90, "-Z": 180, "-X": -90}
    if hint not in init or up not in ("+Y", "+Z"):
        raise ValueError("Unsupported camera convention.")
    yaw0 = init[hint]
    offsets = (-15, 0, 15) if hint else (0, -45, 45, -90, 90, -135, 135, 180)
    candidates = []

    def candidate(yaw, pitch):
        basis = _basis(yaw, pitch, up)
        projected, *_ = _fit(verts, basis, low)
        mask = _raster(projected, faces, 96)[0] >= 0
        score = np.count_nonzero(mask & low) / max(1, np.count_nonzero(mask | low))
        deviation = abs((yaw - yaw0 + 180) % 360 - 180)
        return (
            score - 0.012 * deviation / 180 - 0.003 * abs(pitch) / 30,
            score,
            yaw,
            pitch,
        )

    for offset in offsets:
        for pitch in (0, 15, -15):
            candidates.append(candidate(yaw0 + offset, pitch))
    _, _, best_y, best_p = max(candidates)
    candidates += [
        candidate(best_y + dy, best_p + dp) for dy in (-7, 0, 7) for dp in (-7, 0, 7)
    ]
    _, score, yaw, pitch = max(candidates)
    alternative = [
        s for _, s, y, _ in candidates if abs((y - yaw + 180) % 360 - 180) >= 80
    ]
    if score < 0.72:
        raise ValueError(
            f"Foreground and mesh silhouette do not register (IoU {score:.3f})."
        )
    return _basis(yaw, pitch, up), {
        "yaw_degrees": yaw,
        "pitch_degrees": pitch,
        "up": up,
        "camera_hint": hint,
        "silhouette_iou": round(score, 5),
        "alternate_direction_iou": round(max(alternative), 5) if alternative else None,
        "photographic_pose_verified": False,
        "scope": "Orthographic silhouette fit with axis prior. Symmetric and unseen surfaces are ambiguous.",
    }


def _sample(image, xy):
    return np.column_stack(
        [
            ndimage.map_coordinates(
                image[:, :, c], [xy[:, 1], xy[:, 0]], order=1, mode="nearest"
            )
            for c in range(3)
        ]
    ).astype(np.float32)


def _linear(rgb):
    v = rgb.astype(np.float32) / 255
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _srgb(linear):
    return np.uint8(
        np.clip(
            np.rint(
                255
                * np.where(
                    linear <= 0.0031308,
                    12.92 * linear,
                    1.055 * np.maximum(linear, 0) ** (1 / 2.4) - 0.055,
                )
            ),
            0,
            255,
        )
    )


def _surface_graph(mesh, parts):
    """Face neighbours share an edge. Stitch UV seams but keep touching objects apart."""
    f = np.asarray(mesh.faces)
    _, weld = np.unique(mesh.vertices, axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    direct = trimesh.graph.face_adjacency(faces=f)
    edges = weld[f[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)]
    endpoints = np.sort(edges, axis=1)
    key = np.column_stack((np.repeat(parts, 3), endpoints))
    order = np.lexsort(key.T[::-1])
    starts = np.flatnonzero(np.r_[True, np.any(np.diff(key[order], axis=0), axis=1)])
    pair_starts = starts[np.diff(np.r_[starts, len(order)]) == 2]
    first, second = order[pair_starts], order[pair_starts + 1]
    opposite = np.all(edges[first] == edges[second, ::-1], axis=1)
    stitched = np.column_stack((first[opposite] // 3, second[opposite] // 3))
    pairs = np.unique(np.sort(np.vstack((direct, stitched)), axis=1), axis=0)
    a, b = pairs.T
    centres = mesh.triangles_center
    length = np.linalg.norm(centres[a] - centres[b], axis=1)
    normal_agreement = np.einsum("ij,ij->i", mesh.face_normals[a], mesh.face_normals[b])
    cost = np.maximum(length, 1e-12) * (1 + 3 * (1 - np.clip(normal_agreement, -1, 1)))
    graph = csr_matrix(
        (np.tile(cost, 2), (np.r_[a, b], np.r_[b, a])), shape=(len(f), len(f))
    )
    return graph, weld


def _lab(rgb):
    lab = cv2.cvtColor(
        np.float32(rgb).reshape(-1, 1, 3) / 255, cv2.COLOR_RGB2LAB
    ).reshape(-1, 3)
    lab[:, 0] *= 0.65  # A shadow alone is weaker evidence for a material change.
    return lab


def _material_seeds(rgb, weight):
    """Only locally supported, broad colour families seed the hidden surface."""
    lab = _lab(rgb)
    _, groups = np.unique(
        np.rint(lab / 7).astype(np.int16), axis=0, return_inverse=True
    )
    groups = groups.reshape(-1)
    mass = np.bincount(groups, weights=weight)
    centres = np.column_stack(
        [np.bincount(groups, weights=weight * c) / mass for c in lab.T]
    )
    neighbourhood = cKDTree(centres).query_ball_point(centres, 16)
    remaining = mass.copy()
    prototypes = []
    for _ in range(5):
        support = np.array([remaining[idx].sum() for idx in neighbourhood])
        best = int(np.argmax(support))
        if support[best] < 0.12 * mass.sum() and prototypes:
            break
        members = np.asarray(neighbourhood[best])
        members = members[remaining[members] > 0]
        if not len(members):
            break
        select = np.isin(groups, members)
        prototypes.append(np.average(rgb[select], axis=0, weights=weight[select]))
        remaining[members] = 0
    prototypes = np.array(prototypes)
    diff = np.linalg.norm(lab[:, None, :] - _lab(prototypes)[None, :, :], axis=2)
    closest = np.argmin(diff, axis=1)
    eligible = diff[np.arange(len(rgb)), closest] < 18
    # The front retains original markings; only a bounded low-frequency tint
    # remains in the material constraints, so a logo is not copied to the back.
    colours = prototypes[closest] + 0.16 * np.clip(rgb - prototypes[closest], -24, 24)
    return colours, eligible


def _surface_material(
    mesh, parts, basis, first_face, matte, distance, lens_coords=None
):
    faces = np.asarray(mesh.faces)
    graph, weld = _surface_graph(mesh, parts)
    ncomponents, component = connected_components(graph, directed=False)
    z_normal = np.asarray(mesh.face_normals) @ basis[2]
    if lens_coords is not None:
        center, inv = lens_coords
        rays = _norm([0, 0, 1] - (mesh.triangles_center @ basis.T - center) * inv)
        z_normal = np.einsum("ij,ij->i", mesh.face_normals @ basis.T, rays)
    seen = (
        (first_face >= 0)
        & (distance > 3)
        & (z_normal[np.maximum(first_face, 0)] > 0.38)
    )
    if not seen.any():
        raise ValueError("No reliable visible surface material samples.")
    ids = first_face[seen]
    count = np.bincount(ids, minlength=len(faces))
    photo = cv2.medianBlur(np.ascontiguousarray(matte[:, :, :3]), 7)[seen]
    colour = np.column_stack(
        [
            np.bincount(ids, weights=c, minlength=len(faces)) / np.maximum(count, 1)
            for c in photo.T
        ]
    )
    seed = np.zeros(len(faces), bool)
    visible = np.flatnonzero(count > 0)
    order = visible[np.argsort(component[visible], kind="stable")]
    for group in np.split(order, np.flatnonzero(np.diff(component[order])) + 1):
        base, ok = _material_seeds(colour[group], count[group])
        colour[group] = base
        seed[group] = ok
    if not seed.any():
        raise ValueError("No broad material patch could be established.")
    # A multi-source shortest path stays on the original surface. Nearby
    # detached parts and opposite sides of a narrow gap share no paths.
    length, _, nearest = dijkstra(
        graph,
        directed=False,
        indices=np.flatnonzero(seed),
        min_only=True,
        return_predecessors=True,
    )
    supported = np.isfinite(length)
    known = np.maximum(nearest, 0)
    base = np.where(supported[:, None], colour[known], np.full(3, 160.0))
    # The nearest sample supplies an initial condition, not a hard paint
    # boundary on an unseen flank. A Dirichlet solve extends broad material
    # continuously over the graph, with reduced conductance at geometric folds.
    change = (~seed) & supported
    if change.any():
        conductance = graph.copy()
        conductance.data = 1 / np.maximum(conductance.data, 1e-9)
        neighbours = conductance[change, :]
        degree = np.asarray(neighbours.sum(axis=1)).ravel()
        system = (diags(degree) - neighbours[:, change]).tocsr()
        rhs = neighbours[:, seed] @ colour[seed]
        if change.sum() <= 25000:
            base[change] = spsolve(system, rhs, permc_spec="MMD_AT_PLUS_A")
        else:
            precondition = diags(1 / degree)
            for channel in range(3):
                value, code = cg(
                    system,
                    rhs[:, channel],
                    x0=base[change, channel],
                    rtol=1e-5,
                    M=precondition,
                    maxiter=1000,
                )
                if code:
                    raise ValueError(
                        "Hidden surface material diffusion failed to converge."
                    )
                base[change, channel] = value
    if not np.isfinite(base).all():
        raise ValueError("Invalid surface material solution.")
    # A UV seam can split vertices; a component ID prevents interpolation
    # across pieces that merely touch at a point or form coincident shells.
    key = np.column_stack((np.repeat(component, 3), weld[faces.ravel()]))
    _, vertex = np.unique(key, axis=0, return_inverse=True)
    vertex = vertex.reshape(-1)
    area = np.maximum(mesh.area_faces, 1e-15)
    weights = np.bincount(vertex, weights=np.repeat(area, 3))
    corner = np.column_stack(
        [np.bincount(vertex, weights=np.repeat(c * area, 3)) / weights for c in base.T]
    )
    return _linear(corner[vertex].reshape(len(faces), 3, 3)), {
        "method": "Broad observed material colours per connected surface; intrinsic harmonic extension with crease-weighted conductance and fixed visible constraints.",
        "components": int(ncomponents),
        "components_with_photo_seeds": int(len(np.unique(component[seed]))),
        "seed_faces": int(seed.sum()),
        "faces_without_component_evidence": int((~supported).sum()),
        "unsupported_component_material": "low-confidence neutral dielectric",
        "source_face_count": len(faces),
    }


def _field(path, matte):
    if path is None:
        return None
    meshes = _load(path)
    if len(meshes) != 1:
        return None
    mesh = meshes[0]
    material = getattr(mesh.visual, "material", None)
    image = getattr(material, "baseColorTexture", None)
    if image is None or getattr(mesh.visual, "uv", None) is None:
        return None
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    try:
        basis, reg = _registration(verts, faces, matte, "+Z", "+Y")
    except ValueError:
        return None
    _, _, _, mid, height = _fit(verts, basis, matte[:, :, 3] >= 150)
    rng = np.random.default_rng(203)
    chosen = rng.choice(len(faces), size=70000, p=mesh.area_faces / mesh.area)
    ids = faces[chosen]
    uv_rand = rng.random((len(ids), 2))
    uv_rand[uv_rand.sum(axis=1) > 1] = 1 - uv_rand[uv_rand.sum(axis=1) > 1]
    bary = np.column_stack((1 - uv_rand.sum(axis=1), uv_rand))
    pos = np.einsum("ij,ijk->ik", bary, verts[ids])
    normals = _norm(np.einsum("ij,ijk->ik", bary, mesh.vertex_normals[ids]) @ basis.T)
    uv = np.einsum("ij,ijk->ik", bary, mesh.visual.uv[ids])
    tex = np.array(image.convert("RGB"))
    xy = (uv % 1) * [tex.shape[1], tex.shape[0]]
    xy[:, 1] = tex.shape[0] - xy[:, 1]
    xy -= 0.5
    colour = _linear(_sample(tex, xy))
    factor = getattr(material, "baseColorFactor", None)
    if factor is not None:
        colour *= np.asarray(factor[:3]) / 255
    return cKDTree((pos @ basis.T - mid) / height), normals, colour, reg


def _atlas(mesh, n):
    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    opts = xatlas.PackOptions()
    opts.resolution = n
    opts.padding = 6
    opts.bilinear = True
    atlas.generate(pack_options=opts)
    mapping, indices, uv = atlas[0]

    # xatlas can split vertices at UV chart boundaries. The ordered vertex
    # triplet of every original face must be present with the same winding.
    def triangles(rows):
        rows = np.take_along_axis(
            rows, (np.argmin(rows, axis=1)[:, None] + np.arange(3)) % 3, axis=1
        )
        order = np.lexsort(rows.T[::-1])
        return rows[order], order

    atlas_faces, atlas_order = triangles(mapping[indices])
    source_faces, source_order = triangles(np.asarray(mesh.faces))
    if not np.array_equal(atlas_faces, source_faces):
        raise ValueError("UV unwrap altered input faces.")
    source_id = np.empty(len(indices), dtype=int)
    source_id[atlas_order] = source_order
    corner = (
        mapping[indices][:, :, None] == np.asarray(mesh.faces)[source_id][:, None, :]
    ).argmax(axis=2)
    projected = np.column_stack(
        (uv[:, 0] * n - 0.5, (1 - uv[:, 1]) * n - 0.5, np.zeros(len(uv)))
    )
    face, bary, _ = _raster(projected, indices, n)
    return mapping, indices, uv, face, bary, source_id, corner


def paint_shape(
    shape_glb,
    source_image,
    foreground,
    output_dir,
    *,
    baseline_glb=None,
    texture_size=1024,
    camera_hint=None,
    up_axis="+Y",
    camera_frame=None,
    display_from_raw=None,
    sdf_field=None,
):
    """Preserve the supplied geometry and write separate textured GLB/masters."""
    start = time.monotonic()
    display = _display_matrix(display_from_raw)
    if texture_size not in (256, 512, 1024, 2048):
        raise ValueError("Unsupported atlas resolution.")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty dedicated appearance output directory.")
    out.parent.mkdir(parents=True, exist_ok=True)
    matte = _matte(foreground, 768)
    source_agreement = _source_agreement(source_image, matte)
    mask = matte[:, :, 3] >= 150
    distance = ndimage.distance_transform_edt(mask)
    meshes = _load(shape_glb, require_native=sdf_field is not None)
    mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if not len(faces) or not np.isfinite(verts).all():
        raise ValueError("Invalid shape.")
    # Keep the normals that gate colour reprojection separate from the
    # lighting normals. Enabling SDF shading must never repaint a texel.
    export_normals = np.asarray(mesh.vertex_normals)
    polish = None
    if sdf_field is not None:
        from engine.sdf_normals import sdf_vertex_normals

        field_path = Path(sdf_field)
        field_digest = _sha(field_path)
        export_normals, metrics = sdf_vertex_normals(
            mesh, np.load(field_path, mmap_mode="r", allow_pickle=False)
        )
        if _sha(field_path) != field_digest:
            raise ValueError("SDF field changed while estimating shading normals.")
        polish = {
            "raw_shape_sha256": _sha(shape_glb),
            "field_sha256": field_digest,
            **metrics,
        }
    basis, reg = _registration(verts, faces, matte, camera_hint, up_axis, camera_frame)
    lens = lens_parameters(camera_frame)
    lens_coords = _lens_coordinates(verts, basis, lens)
    projected, scale, offset, mid, height = _fit(verts, basis, mask, lens)
    first_face, _, depth = _raster(projected, faces, len(mask))
    parts = np.repeat(np.arange(len(meshes)), [len(m.faces) for m in meshes])
    propagated, propagation_info = _surface_material(
        mesh, parts, basis, first_face, matte, distance, lens_coords
    )
    field = _field(baseline_glb, matte)
    mapping, indices, uv, idmap, weights, source_id, corner = _atlas(mesh, texture_size)
    material_corners = propagated[source_id[:, None], corner]
    rows, cols = np.where(idmap >= 0)
    if not len(rows):
        raise ValueError("UV atlas has no supported samples.")
    atlas = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    v = verts[mapping]
    normals = np.asarray(mesh.vertex_normals)[mapping]
    direct_count = prior_count = 0
    for k in range(0, len(rows), 40000):
        y, x = rows[k : k + 40000], cols[k : k + 40000]
        ids = indices[idmap[y, x]]
        bc = weights[y, x]
        bary = np.column_stack((1 - bc.sum(axis=1), bc))
        point = np.einsum("ij,ijk->ik", bary, v[ids]) @ basis.T
        unit_norm = _norm(np.einsum("ij,ijk->ik", bary, normals[ids]) @ basis.T)
        screen = point
        facing = unit_norm[:, 2]
        if lens_coords is not None:
            center, inv = lens_coords
            screen = project_pinhole(point, center, inv)
            rays = _norm([0, 0, 1] - (point - center) * inv)
            facing = np.einsum("ij,ij->i", unit_norm, rays)
        xy = screen[:, :2] * [scale, -scale] + offset
        xi, yi = np.rint(xy).astype(int).T
        within = (xi >= 0) & (xi < len(mask)) & (yi >= 0) & (yi < len(mask))
        xi = np.clip(xi, 0, len(mask) - 1)
        yi = np.clip(yi, 0, len(mask) - 1)
        # Pixel-centre quantization can shift interpolated depth on a steep
        # triangle. Allow that tolerance only on the same or adjacent surface;
        # an unrelated thin layer can be separated by much less than a pixel.
        first = first_face[yi, xi]
        first_vertices = faces[np.maximum(first, 0)]
        original_vertices = mapping[ids]
        adjacent = (first_vertices[:, :, None] == original_vertices[:, None, :]).any(
            axis=(1, 2)
        )
        tolerance = np.where(adjacent, 1.8 / scale, 1e-6 * height)
        visible = (
            within & (first >= 0) & (np.abs(screen[:, 2] - depth[yi, xi]) <= tolerance)
        )
        direct = (
            visible
            * np.clip((facing - 0.14) / 0.42, 0, 1)
            * np.clip((distance[yi, xi] - 1.5) / 4, 0, 1)
        )
        direct_count += np.count_nonzero(direct > 0.5)
        guess = np.einsum("ij,ijk->ik", bary, material_corners[idmap[y, x]])
        if field is not None:
            tree, normal, colour, _ = field
            dist, index = tree.query((point - mid) / height, k=6)
            align = np.einsum("ij,ikj->ik", unit_norm, normal[index])
            w = (
                np.clip((align - 0.48) / 0.42, 0, 1)
                * np.clip(1 - dist / 0.095, 0, 1) ** 2
            )
            confidence = w.max(axis=1)
            mix = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
            prior = (colour[index] * mix[:, :, None]).sum(axis=1)
            # An inferred baseline frequently drags low-chroma background or
            # unobserved high-contrast speckles onto the far surface. Retain
            # only bounded variation around the observed local material.
            difference = _srgb(prior).astype(float) - _srgb(guess).astype(float)
            luma = difference @ np.array([0.2126, 0.7152, 0.0722])
            chroma = np.linalg.norm(difference - luma[:, None], axis=1)
            confidence *= 0.72 * np.exp(-0.5 * ((luma / 37) ** 2 + (chroma / 40) ** 2))
            guess += confidence[:, None] * (prior - guess)
            prior_count += np.count_nonzero((direct <= 0.5) & (confidence > 0.5))
        photo = _linear(_sample(matte[:, :, :3], xy))
        atlas[y, x] = _srgb(guess + direct[:, None] * (photo - guess))
    # Extrapolate only into unused UV guard bands. Interior and hole pixels
    # of an existing chart are never mixed with another chart.
    nearest = ndimage.distance_transform_edt(
        idmap < 0, return_distances=False, return_indices=True
    )
    atlas = atlas[tuple(nearest)]
    material = trimesh.visual.material.PBRMaterial(
        name="Photograph and inferred material",
        baseColorTexture=Image.fromarray(atlas),
        metallicFactor=0.0,
        roughnessFactor=0.60,
    )
    painted = trimesh.Trimesh(
        vertices=v,
        faces=indices,
        vertex_normals=export_normals[mapping],
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
    )
    if display is not None:
        painted.apply_transform(display)
        painted.vertex_normals = export_normals[mapping] @ display[:3, :3].T
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp)
        shutil.copyfile(shape_glb, stage / "shape-master.glb")
        export_finished(painted, stage)
        stats = {
            "shape_sha256": _sha(shape_glb),
            "geometry_master": "shape-master.glb",
            "original_sha256": _sha(source_image),
            "foreground_sha256": _sha(foreground),
            "baseline_sha256": _sha(baseline_glb) if baseline_glb else None,
            "output_sha256": _sha(stage / "model.glb"),
            "geometry": "Input triangles retained; only UV seams duplicate vertices. An optional rigid display transform is applied after raw-coordinate projection.",
            "display_from_raw": display.tolist() if display is not None else None,
            "source_camera_space": "unmodified geometry master",
            "shading_normals": (
                "Filtered native SDF gradients with guarded raw normals, followed by the optional rigid display rotation."
                if polish is not None
                else "Raw primitive normals (computed before UV splitting when absent); inverse-transpose instance transform, followed by the optional rigid display rotation."
            ),
            "source_faces": len(faces),
            "exported_faces": len(indices),
            "registration": reg,
            "source_agreement": source_agreement,
            "baseline_registration": field[-1] if field is not None else None,
            "base_colour": "One real segmented image, z-buffered visible projection; optional generated baseline surface colours transferred by normalized 3D proximity and normal agreement.",
            "hidden_fallback": "Material propagated along the connected surface; unsupported pieces receive neutral colour. Spatial pattern and true hidden colours unknown.",
            "surface_propagation": propagation_info,
            "texture_size": texture_size,
            "populated_texels": len(rows),
            "mostly_observed_texels": int(direct_count),
            "mostly_baseline_texels": int(prior_count),
            "pbr": "Neutral illustrative dielectric; roughness and metallic response are not measured.",
            "appearance": "needs_review",
            "elapsed_seconds": round(time.monotonic() - start, 3),
        }
        if _sha(stage / "shape-master.glb") != stats["shape_sha256"]:
            raise ValueError("Input geometry changed during export.")
        if polish is not None:
            if polish["raw_shape_sha256"] != stats["shape_sha256"]:
                raise ValueError("Raw shape changed during SDF shading.")
            stats["sdf_shading"] = polish
        (stage / "appearance.json").write_text(
            json.dumps(stats, indent=2), encoding="utf8"
        )
        # Publish the completed directory in one rename; refuse concurrent wins.
        if out.exists():
            out.rmdir()
        os.replace(stage, out)
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("shape", "source", "foreground", "out"):
        p.add_argument(field)
    p.add_argument("--baseline-glb")
    p.add_argument("--texture-size", type=int, default=1024)
    p.add_argument("--camera-hint", choices=("+X", "-X", "+Z", "-Z"))
    p.add_argument("--up-axis", choices=("+Y", "+Z"), default="+Y")
    p.add_argument(
        "--sdf-field",
        type=Path,
        help="Matching native signed-distance .npy grid for shading only",
    )
    p.add_argument(
        "--display-from-raw", type=Path, help="4x4 proper rigid display transform JSON"
    )
    p.add_argument(
        "--camera-frame",
        type=Path,
        help="Registered image_right, image_up and camera_direction_from_origin JSON",
    )
    a = p.parse_args()
    print(
        json.dumps(
            paint_shape(
                a.shape,
                a.source,
                a.foreground,
                a.out,
                baseline_glb=a.baseline_glb,
                texture_size=a.texture_size,
                camera_hint=a.camera_hint,
                up_axis=a.up_axis,
                sdf_field=a.sdf_field,
                display_from_raw=json.loads(
                    a.display_from_raw.read_text(encoding="utf8")
                )
                if a.display_from_raw
                else None,
                camera_frame=json.loads(a.camera_frame.read_text(encoding="utf8"))
                if a.camera_frame
                else None,
            ),
            indent=2,
        )
    )
