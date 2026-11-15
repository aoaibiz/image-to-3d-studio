"""Refine a textured single-view inference without a subject classifier.

The camera convention and size limits match the local TripoSR exporter. Its
front reference projection has already run. This pass retains every connected
piece, UV chart and material. It cannot reconstruct unseen semantic details.
"""

import struct
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt, maximum_filter

from engine.reference_texture import raster
from engine.surface_fairing import fair_surface


def _camera(vertices, axes, bounds):
    xy = vertices[:, axes] * [1, -1]
    low = xy.min(axis=0)
    extent = np.ptp(xy, axis=0)
    target_low, target_high = np.asarray(bounds)
    scale = np.min((target_high - target_low) / np.maximum(extent, 1e-12))
    offset = target_low + (target_high - target_low - extent * scale) / 2 - low * scale
    return scale, offset


def _mask(vertices, faces, axes, camera, shape):
    scale, offset = camera
    xy = np.ascontiguousarray(
        np.rint((vertices[:, axes] * [1, -1] * scale + offset) * 16), np.int32
    )
    mask = np.zeros(shape, dtype=np.uint8)
    for triangle in xy[faces]:
        cv2.fillConvexPoly(mask, triangle, 1, shift=4)
    return mask.astype(bool)


def _iou(first, second):
    return float(
        np.count_nonzero(first & second) / max(1, np.count_nonzero(first | second))
    )


def _unit(vectors):
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(lengths, 1e-20)


def _smooth_geometry(mesh):
    # Weld only numerically identical seam vertices for adjacency. Write the
    # result back to the original indexing so each texture island stays intact.
    vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    faces = inverse[mesh.faces]
    a, b, c = vertices[faces].transpose(1, 0, 2)
    cross = np.cross(b - a, c - a)
    normals = np.zeros_like(vertices)
    for corner in faces.T:
        np.add.at(normals, corner, cross)
    normals = _unit(normals)
    face_normals = _unit(cross)
    edges = np.concatenate((faces[:, :2], faces[:, 1:], faces[:, ::2]))
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    first, second = edges.T
    delta = vertices[second] - vertices[first]
    edge_length = np.linalg.norm(delta, axis=1)
    scale = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    local_limit = np.full(len(vertices), np.inf)
    feature = np.zeros(len(vertices), bool)
    feature[edges[counts != 2].ravel()] = True
    for corner in faces.T:
        # Protect sharp creases as well as camera tangencies in all three axes.
        feature[corner[np.einsum("ij,ij->i", normals[corner], face_normals) < 0.88]] = (
            True
        )
    feature |= np.min(np.abs(normals), axis=1) < 0.055
    strength = (
        np.clip(
            (np.einsum("ij,ij->i", normals[first], normals[second]) - 0.78) / 0.22, 0, 1
        )
        ** 2
    )
    summed = np.zeros_like(vertices)
    weight = np.zeros(len(vertices))
    for ids, vec in ((first, delta), (second, -delta)):
        np.add.at(summed, ids, vec * strength[:, None])
        np.add.at(weight, ids, strength)
        np.minimum.at(local_limit, ids, edge_length)
    height = np.einsum("ij,ij->i", summed, normals) / np.maximum(weight, 1)
    # No topology change or global inflation, even for very thin inputs. Lock
    # bounding extrema, open borders and silhouettes rather than shrinking them.
    for axis in range(3):
        col = vertices[:, axis]
        feature |= (col == col.min()) | (col == col.max())
    cap = np.minimum(0.12 * local_limit, 0.0019 * scale)
    steps = np.clip(0.47 * height, -cap, cap)
    steps[feature] = 0
    moved = vertices + steps[:, None] * normals

    def new_face_vectors():
        a, b, c = moved[faces].transpose(1, 0, 2)
        return np.cross(b - a, c - a)

    badly_changed = (
        np.einsum("ij,ij->i", _unit(new_face_vectors()), face_normals) < 0.82
    )
    moved[faces[badly_changed].ravel()] = vertices[faces[badly_changed].ravel()]
    final_cross = new_face_vectors()
    if np.any(np.einsum("ij,ij->i", final_cross, cross) <= 0):
        return None
    final_normals = np.zeros_like(vertices)
    for corner in faces.T:
        np.add.at(final_normals, corner, final_cross)
    # Carry the inferred shading across the small deformation. Replacing every
    # imported normal with the welded mean would erase hard shading seams even
    # at vertices whose geometry was correctly held fixed.
    original_normals = np.asarray(mesh.vertex_normals)
    normal_adjustment = (_unit(final_normals) - normals)[inverse]
    updated_normals = _unit(original_normals + normal_adjustment)
    moved_distance = np.linalg.norm(moved - vertices, axis=1)
    return (
        moved[inverse],
        updated_normals,
        {
            "moved_unique_vertices": int(np.count_nonzero(moved_distance > 1e-12)),
            "max_displacement_over_diagonal": float(moved_distance.max() / scale),
        },
    )


def _clean_texture(mesh, reference, foreground):
    # Restrict editing to isolated outliers absent from the photograph's
    # palette. Preserve edges, transparent texels and UV-chart boundaries.
    image = mesh.visual.material.baseColorTexture
    pixels = np.array(image.convert("RGBA"))
    rgb = np.ascontiguousarray(pixels[:, :, :3])
    median = cv2.medianBlur(rgb, 3)
    difference = np.max(np.abs(rgb.astype(np.int16) - median), axis=2)
    # Evaluate neighbors only at the sparse outliers.
    ys, xs = np.where(difference > 46)
    h, w = difference.shape
    inner = (xs > 1) & (xs < w - 2) & (ys > 1) & (ys < h - 2)
    ys, xs = ys[inner], xs[inner]
    if not len(xs):
        return None, 0
    palette = np.zeros((32, 32, 32), bool)
    palette[tuple((reference[foreground, :3] // 8).T)] = True
    palette = maximum_filter(palette, 3, mode="constant")
    keep = (
        ~palette[tuple((rgb[ys, xs] // 8).T)] & palette[tuple((median[ys, xs] // 8).T)]
    )
    ys, xs = ys[keep], xs[keep]
    if not len(xs):
        return None, 0
    neighbors = np.stack(
        [rgb[ys + dy, xs + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]
    )
    close = np.max(np.abs(neighbors.astype(np.int16) - median[ys, xs]), axis=2) <= 21
    keep = (close.sum(0) >= 7) & (pixels[ys, xs, 3] == 255)
    ys, xs = ys[keep], xs[keep]
    if not len(xs):
        return None, 0
    uv = np.asarray(mesh.visual.uv)
    atlas_xy = np.rint(uv * [w - 1, -(h - 1)] + [0, h - 1]).astype(np.int32)
    safe = np.zeros((h, w), np.uint8)
    for tri in atlas_xy[mesh.faces]:
        cv2.fillConvexPoly(safe, tri, 1)
    # Boundary vertices are split by the xatlas exporter, including at charts
    # which happen to abut each other in the flattened bitmap.
    edges = np.sort(mesh.edges, axis=1)
    edge_ids, counts = np.unique(edges, axis=0, return_counts=True)
    for a, b in atlas_xy[edge_ids[counts == 1]]:
        cv2.line(safe, tuple(a), tuple(b), 0, thickness=3)
    edit = safe[ys, xs] != 0
    ys, xs = ys[edit], xs[edit]
    if not len(xs):
        return None, 0
    pixels[ys, xs, :3] = median[ys, xs]
    return Image.fromarray(pixels).convert(image.mode), int(len(xs))


def _pad_atlas(mesh, image):
    """Extend current shaded chart pixels into the unused filtering margin.

    The projection updates front-facing chart interiors after the neural atlas
    was first baked. A GPU bilinear footprint crossing that chart would blend
    fresh paint and stale padding. For each unused texel close to a chart, take
    its nearest *covered* neighbor. No triangles, including inferred back
    faces, are painted over.
    """
    original = np.asarray(image)
    if image.mode == "RGBA" and np.any(original[:, :, 3] != 255):
        return image, 0
    h, w = original.shape[:2]
    uv_pixels = mesh.visual.uv * [w - 1, -(h - 1)] + [0, h - 1]
    coverage = np.zeros((h, w), dtype=bool)
    for face in mesh.faces:
        sample = raster(uv_pixels[face], (h, w))
        if sample is not None:
            coverage[sample[0], sample[1]] = True
    if not coverage.any() or coverage.all():
        return image, 0
    distances, nearest = distance_transform_edt(~coverage, return_indices=True)
    margin = (~coverage) & (distances <= 5)
    adjacent = original[nearest[0, margin], nearest[1, margin]]
    count = int(np.any(original[margin] != adjacent, axis=1).sum())
    if not count:
        return image, 0
    updated = original.copy()
    updated[margin] = adjacent
    return Image.fromarray(updated), count


def build_general(baseline_glb, foreground):
    """Return a finished scene and evidence, or (None, evidence) to retain inference."""
    info = {
        "status": "retained",
        "method": "subject-independent local refinement of inferred surface and baked atlas",
        "geometry_source": "TripoSR inference",
        "unseen_surfaces": "inferred",
        "appearance": "needs_review",
    }
    if not isinstance(foreground, Image.Image):
        return None, dict(info, reason="No reference image")
    source = foreground
    if max(source.size) > 384:
        source = ImageOps.contain(source, (384, 384), Image.Resampling.LANCZOS)
    ref = np.asarray(source.convert("RGBA"))
    mask = ref[:, :, 3] >= 128
    if min(mask.shape) < 8 or mask.all() or mask.sum() < 36:
        return None, dict(info, reason="No reliable foreground silhouette")
    try:
        with Path(baseline_glb).open("rb") as handle:
            # Only the internally generated binary GLB is supported here.
            header = handle.read(12)
            if (
                len(header) != 12
                or header[:8] != b"glTF\x02\x00\x00\x00"
                or struct.unpack_from("<I", header, 8)[0]
                != Path(baseline_glb).stat().st_size
            ):
                return None, dict(info, reason="No valid binary GLB")
            handle.seek(0)
            scene = trimesh.load(handle, file_type="glb", force="scene", process=False)
    except (OSError, ValueError, TypeError, IndexError, KeyError, struct.error):
        return None, dict(info, reason="Could not read inferred mesh")
    if len(scene.geometry) != 1 or len(scene.graph.nodes_geometry) != 1:
        return None, dict(info, reason="Expected a single inference surface")
    mesh = next(iter(scene.geometry.values()))
    node = next(iter(scene.graph.nodes_geometry))
    if not np.allclose(scene.graph[node][0], np.eye(4)) or not isinstance(
        mesh, trimesh.Trimesh
    ):
        return None, dict(info, reason="Unsupported scene transform")
    material = getattr(mesh.visual, "material", None)
    image = getattr(material, "baseColorTexture", None)
    uv = getattr(mesh.visual, "uv", None)
    v = np.asarray(mesh.vertices)
    if (
        len(mesh.faces) < 4
        or len(mesh.faces) > 400000
        or v.shape[0] < 4
        or not np.isfinite(v).all()
        or np.linalg.norm(np.ptp(v, axis=0)) < 1e-10
        or image is None
        or image.mode not in ("RGB", "RGBA")
        or max(image.size) > 2048
        or uv is None
        or len(uv) != len(v)
        or not np.isfinite(uv).all()
        or np.any((uv < 0) | (uv > 1))
    ):
        return None, dict(
            info, reason="Inference surface or atlas unsuitable for refinement"
        )
    y, x = np.where(mask)
    axes = [0, 1]
    camera = _camera(v, axes, [[x.min(), y.min()], [x.max(), y.max()]])
    before = _mask(v, mesh.faces, axes, camera, mask.shape)
    front_iou = _iou(before, mask)
    info["reference_silhouette_iou"] = round(front_iou, 4)
    if front_iou < 0.64:
        return None, dict(info, reason="Inferred outline disagrees with the photograph")
    geometry = {"moved_unique_vertices": 0}
    for operation in (fair_surface, _smooth_geometry):
        smoothing = operation(mesh)
        if smoothing is None:
            continue
        moved, normals, quality = smoothing
        new_mask = _mask(moved, mesh.faces, axes, camera, mask.shape)
        overlaps = [_iou(before, new_mask)]
        for pair in ([2, 1], [0, 2]):
            cam = _camera(v, pair, [[4, 4], [188, 188]])
            overlaps.append(
                _iou(
                    _mask(v, mesh.faces, pair, cam, (192, 192)),
                    _mask(moved, mesh.faces, pair, cam, (192, 192)),
                )
            )
        info["outline_self_iou_front_side_top"] = [round(n, 5) for n in overlaps]
        original_extents = np.ptp(v, axis=0)
        extent_ratio = np.ptp(moved, axis=0) / np.maximum(original_extents, 1e-15)
        extent_ok = np.all(np.abs(extent_ratio[original_extents > 1e-12] - 1) < 0.001)
        info["extent_ratio"] = extent_ratio.tolist()
        if (
            extent_ok
            and min(overlaps) >= 0.995
            and _iou(new_mask, mask) >= front_iou - 0.0015
        ):
            geometry = quality
            break
    corrected, count = _clean_texture(mesh, ref, mask)
    corrected, padding_count = _pad_atlas(
        mesh, corrected if corrected is not None else image
    )
    info.update(
        geometry=geometry,
        corrected_isolated_texels=count,
        updated_padding_texels=padding_count,
        retained_faces=int(len(mesh.faces)),
        retained_disconnected_parts=True,
    )
    if not geometry["moved_unique_vertices"] and not count and not padding_count:
        return None, dict(info, reason="No changes passed the preservation checks")
    if geometry["moved_unique_vertices"]:
        mesh.vertices = moved
        mesh.vertex_normals = normals
    if count or padding_count:
        material.baseColorTexture = corrected
    info["status"] = "refined"
    return scene, info
