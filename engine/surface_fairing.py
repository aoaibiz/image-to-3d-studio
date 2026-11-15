"""Feature- and silhouette-constrained local quadratic fairing of inferred meshes."""

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-14)


def _normals(v, faces):
    a, b, c = v[faces].transpose(1, 0, 2)
    cross = np.cross(b - a, c - a)
    n = np.zeros_like(v)
    for col in faces.T:
        np.add.at(n, col, cross)
    return _unit(n), cross


def _contours(v, faces, extent):
    locked = np.zeros(len(v), bool)
    # Project outer silhouettes and actual negative spaces (e.g. a handle)
    # using one fixed world-to-pixel scale over all three principal views.
    for axes in ([0, 1], [2, 1], [0, 2]):
        coords = v[:, axes]
        uv = np.rint((coords - coords.min(axis=0)) * (384 / extent) + 5).astype("int32")
        mask = np.zeros((395, 395), np.uint8)
        for tri in uv[faces]:
            cv2.fillConvexPoly(mask, tri, 1)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        locked |= distance[uv[:, 1], uv[:, 0]] < 2.1
    return locked


def _quadratic_targets(v, normals, labels, ids, k, scale):
    # Per block, fit a height field around each vertex in its tangent frame.
    # Using a quadratic instead of a local plane retains smoothly curved walls.
    q = v[ids]
    n = normals[ids]
    ex = _unit(np.cross(n, np.eye(3)[np.argmin(np.abs(n), axis=1)]))
    ey = np.cross(n, ex)
    diffs = v[k] - q[:, None, :]
    r = np.maximum(np.linalg.norm(diffs[:, -1], axis=1), scale * 1e-9)
    diffs = diffs / r[:, None, None]
    x = np.einsum("ijk,ik->ij", diffs, ex)
    y = np.einsum("ijk,ik->ij", diffs, ey)
    z = np.einsum("ijk,ik->ij", diffs, n)
    agree = np.einsum("ijk,ik->ij", normals[k], n)
    weight = np.exp(-2.4 * (x * x + y * y)) * np.clip((agree - 0.40) / 0.50, 0, 1)
    weight *= np.abs(z) < 0.40
    weight *= labels[ids, None] == labels[k]
    design = np.stack((np.ones_like(x), x, y, x * x, x * y, y * y), axis=2)

    def solve(w):
        weighted = design * w[:, :, None]
        matrix = design.transpose(0, 2, 1) @ weighted
        matrix += np.eye(6)[None, :, :] * 0.0006
        return np.linalg.solve(matrix, (weighted.transpose(0, 2, 1) @ z[:, :, None]))[
            :, :, 0
        ]

    co = solve(weight)
    residual = z - np.einsum("ijk,ik->ij", design, co)
    # A robust second fit reduces pull toward an adjacent corner or opposite wall.
    # Do not follow a poorly supported patch near a joint.
    robust = weight / (1 + (residual / 0.055) ** 2)
    co = solve(robust)
    supported = robust.sum(axis=1) > 10
    normal = _unit(n - co[:, 1, None] * ex - co[:, 2, None] * ey)
    height = co[:, 0] * r * np.einsum("ij,ij->i", n, normal)
    return height[:, None] * normal * supported[:, None], np.where(
        supported[:, None], normal, n
    )


def fair_surface(mesh):
    """Return (positions, normals, report) without modifying mesh, or None.

    Indexing, transforms, UVs, components and all other mesh data stay with the
    caller. The caller must also verify before/after silhouettes against the input.
    """
    v, iv = np.unique(mesh.vertices, axis=0, return_inverse=True)
    iv = iv.reshape(-1)
    faces = iv[np.asarray(mesh.faces)]
    # The tested bandwidth is for dense marching-cubes surfaces. Do not run a
    # 64-neighbor fit on low-poly assets or unbounded third-party meshes.
    if not 256 <= len(v) <= 150_000 or not np.isfinite(v).all():
        return None
    extent = float(np.ptp(v, axis=0).max())
    if extent <= 1e-12:
        return None
    welded = trimesh.Trimesh(v, faces, process=False)
    edges = welded.edges_unique
    graph = coo_matrix((np.ones(len(edges)), edges.T), shape=(len(v), len(v)))
    _, labels = connected_components(graph, directed=False)
    locked = _contours(v, faces, extent)
    length = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
    pitch = float(np.median(length))
    edge_ids, count = np.unique(welded.edges_unique_inverse, return_counts=True)
    locked[edges[edge_ids[count != 2]].ravel()] = True
    sharp = welded.face_adjacency_edges[welded.face_adjacency_angles > np.deg2rad(52)]
    locked[sharp.ravel()] = True
    for axis in range(3):
        col = v[:, axis]
        locked |= (col == col.min()) | (col == col.max())
    normals0, cross0 = _normals(v, faces)
    if np.any(np.linalg.norm(cross0, axis=1) < extent**2 * 1e-16):
        return None
    ids = np.flatnonzero(~locked)
    if not len(ids):
        return None
    cap = min(extent * 0.010, pitch * 0.66)
    pos = v.copy()
    for _ in range(2):
        normals, _ = _normals(pos, faces)
        tree = cKDTree(pos)
        delta = np.zeros_like(pos)
        for i in range(0, len(ids), 384):
            rows = ids[i : i + 384]
            _, k = tree.query(pos[rows], k=64)
            delta[rows], _ = _quadratic_targets(pos, normals, labels, rows, k, extent)
        proposed = pos - v + 0.88 * delta
        disp = (
            proposed
            * np.minimum(1, cap / np.maximum(np.linalg.norm(proposed, axis=1), 1e-16))[
                :, None
            ]
        )
        # Reject flipped or acutely distorted faces; in very small triangles
        # the allowable displacement can be substantially below the global cap.
        for retry in range(24):
            a, b, c = (v + disp)[faces].transpose(1, 0, 2)
            cross = np.cross(b - a, c - a)
            dots = np.einsum("ij,ij->i", _unit(cross), _unit(cross0))
            bad = (dots < 0.60) | (
                np.linalg.norm(cross, axis=1) < 0.25 * np.linalg.norm(cross0, axis=1)
            )
            if not bad.any():
                break
            disp[np.unique(faces[bad])] *= 0.5 if retry < 8 else 0.0
        else:
            return None
        pos = v + disp
    normals, _ = _normals(pos, faces)
    moved = np.linalg.norm(pos - v, axis=1)
    # Small, skewed triangles have unstable shading even after their geometry
    # passes the orientation test. Use the fitted differential on a well-
    # supported patch. Real open borders, imported split normals and large
    # differences between the patch orientation and local surface stay intact.
    tree = cKDTree(pos)
    shade = normals.copy()
    for i in range(0, len(v), 384):
        rows = np.arange(i, min(i + 384, len(v)))
        _, neighbors = tree.query(pos[rows], k=64)
        _, fit = _quadratic_targets(pos, normals, labels, rows, neighbors, extent)
        agreement = np.sum(normals[rows] * fit, axis=1)
        influence = 0.88 * np.clip((agreement - 0.75) / 0.15, 0, 1)
        shade[rows] = _unit(normals[rows] + influence[:, None] * (fit - normals[rows]))
    shade[edges[edge_ids[count != 2]].ravel()] = normals[
        edges[edge_ids[count != 2]].ravel()
    ]
    original = np.asarray(mesh.vertex_normals)
    preserve = np.sum(original * normals0[iv], axis=1) < 0.90
    updated = _unit(original + (shade - normals0)[iv])
    updated[preserve] = original[preserve]
    return (
        pos[iv],
        updated,
        {
            "moved_unique_vertices": int((moved > 1e-12).sum()),
            "max_displacement_over_diagonal": float(
                moved.max() / np.linalg.norm(np.ptp(v, axis=0))
            ),
            "locked_contour_or_feature_vertices": int(locked.sum()),
            "vertex_components": int(labels.max() + 1),
            "method": "bounded robust quadratic neighborhood fit",
        },
    )
