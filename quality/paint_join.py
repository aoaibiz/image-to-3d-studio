"""Screen for visible pigment edges introduced solely by switching source fields."""

import cv2
import numpy as np


def connected_pairs(first, second, faces):
    same = first == second
    if np.all(same):
        return same
    a, b = faces[first[~same]], faces[second[~same]]
    shared = np.any(a[:, :, None] == b[:, None, :], axis=2).sum(axis=1)
    result = same.copy()
    result[~same] = shared >= 2
    return result


def boundary_runs(rgb_fields, labels, available, depth, normals, scale, face_ids, faces):
    """Compare adjacent output samples with the *same pair* in each source field.

    A real stripe in either input, an occlusion or a hard crease is not evidence
    of a compositing defect. A sustained new edge on a smooth surface is.
    """
    h, w = labels.shape
    evidence = np.zeros((h, w), np.uint8)
    for dy, dx in ((1, 0), (0, 1)):
        a = (slice(0, h - dy), slice(0, w - dx))
        b = (slice(dy, h), slice(dx, w))
        adjacent = (labels[a] != labels[b]) & (labels[a] >= 0) & (labels[b] >= 0)
        y, x = np.nonzero(adjacent)
        if not len(x):
            continue
        yy, xx = y + dy, x + dx
        first, second = labels[y, x], labels[yy, xx]
        comparable = (
            available[first, yy, xx] & available[second, y, x]
            & (abs(depth[y, x] - depth[yy, xx]) * scale < 3)
            & (np.sum(normals[y, x] * normals[yy, xx], axis=-1) > 0.90)
            & connected_pairs(face_ids[y, x], face_ids[yy, xx], faces)
        )
        c00 = rgb_fields[first, y, x].astype(np.int16)
        c11 = rgb_fields[second, yy, xx].astype(np.int16)
        c01 = rgb_fields[first, yy, xx].astype(np.int16)
        c10 = rgb_fields[second, y, x].astype(np.int16)
        blended_edge = abs(c00 - c11).max(axis=1)
        existing_edge = np.maximum(abs(c00 - c01), abs(c10 - c11)).max(axis=1)
        novel = comparable & (blended_edge >= 32) & (blended_edge > 2 * existing_edge + 12)
        evidence[y[novel], x[novel]] = 1
        evidence[yy[novel], xx[novel]] = 1
    # Allow single-pixel sampling gaps. Report the observed run, in image units.
    connected = cv2.dilate(evidence, np.ones((3, 3), np.uint8))
    count, regions, stats, _ = cv2.connectedComponentsWithStats(connected)
    obj_y, obj_x = np.nonzero(labels >= 0)
    span = max(np.ptp(obj_x), np.ptp(obj_y), 1) if len(obj_x) else 1
    area = np.bincount(regions[evidence != 0], minlength=count)
    runs = []
    for index in range(1, count):
        x, y, width, height, _ = map(int, stats[index])
        length = max(width - 2, height - 2)
        if area[index] >= 8 and length >= max(6, span * 0.025):
            runs.append({"xywh": [x, y, width, height], "edge_pixels": int(area[index]),
                         "fraction_of_object_span": round(float(length / span), 4)})
    return sorted(runs, key=lambda r: r["fraction_of_object_span"], reverse=True)[:8]


def inspect(mesh, atlas_faces, rows, cols, original, fields, weights, applied, first):
    """Measure rendered seams in physical view space, independently of UV island layout."""
    from engine import guided_texture as gt

    lookup = np.full(atlas_faces.shape, -1, np.int32)
    lookup[rows, cols] = np.arange(len(rows), dtype=np.int32)
    sources = np.stack([original[rows, cols], *(np.clip(np.rint(f), 0, 255).astype(np.uint8) for f in fields)])
    support = np.stack([np.ones(len(rows), bool), *(w > 0.05 for w in weights)])
    owner = np.where(applied, np.where(first, 1, 2), 0).astype(np.int8)
    uv = gt._uv_pixels(mesh, len(lookup))[:, :2]
    # Texture islands may duplicate vertices. Reconnect their geometric edges,
    # while keeping disjoint layers distinct even when their depths nearly coincide.
    _, welded = np.unique(mesh.vertices, axis=0, return_inverse=True)
    faces = welded[mesh.faces]
    detected = {}
    for yaw in (60, 90, 120, 180, 240, 270, 300):
        _, _, scale, ids, bary, depth = gt._view(mesh, (yaw, 10))
        y, x = np.nonzero(ids >= 0)
        tri = mesh.faces[ids[y, x]]
        b = bary[y, x]
        positions = np.einsum("ij,ijk->ik", np.column_stack([1 - b.sum(1), b]), uv[tri])
        tx, ty = np.rint(positions).astype(int).clip(0, len(lookup) - 1).T
        idx = lookup[ty, tx]
        valid = idx >= 0
        valid[valid] &= connected_pairs(atlas_faces[ty[valid], tx[valid]], ids[y[valid], x[valid]], faces)
        idx, y, x = idx[valid], y[valid], x[valid]
        labels = np.full(ids.shape, -1, np.int8)
        rgb = np.zeros((3, *ids.shape, 3), np.uint8)
        available = np.zeros((3, *ids.shape), bool)
        normals = np.zeros((*ids.shape, 3), np.float32)
        labels[y, x] = owner[idx]
        rgb[:, y, x] = sources[:, idx]
        available[:, y, x] = support[:, idx]
        normals[y, x] = mesh.face_normals[ids[y, x]]
        runs = boundary_runs(rgb, labels, available, depth, normals, scale, ids, faces)
        if runs:
            detected[str(yaw)] = runs
    return {"method": "physical-view-field-switch-v1", "angles_with_new_edges": detected,
            "semantic_continuity": "requires_visual_review"}
