"""Transfer a coherent appearance patch onto a supported native PBR surface.

The photograph faces +Z in the registered camera. Computation may rotate the
mesh temporarily. Export retains original glTF space and every non-colour buffer.
"""

import hashlib
import io
import json
import struct
from pathlib import Path
import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
from engine import guided_texture as gt
from engine.triposg_cpu import digest
from engine.visual_brief import json_sha
from engine.shape_appearance import _display_matrix
from engine.finish_export import embed_master, export_finished


def box(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("empty support")
    lo = np.array([x.min(), y.min()])
    hi = np.array([x.max(), y.max()])
    return (lo + hi) / 2, hi - lo


def body(mask):
    mask = mask.astype(bool)
    _, span = box(mask)
    radius = max(8, int(0.11 * min(span)))
    d = ndi.distance_transform_edt(mask)
    ids, n = ndi.label(d > radius)
    if not n:
        raise ValueError("no thick self-supporting surface")
    npx = np.bincount(ids.ravel())
    npx[0] = 0
    core = ids == np.argmax(npx)
    out = (ndi.distance_transform_edt(~core) <= radius) & mask
    if out.sum() < 0.64 * mask.sum():
        raise ValueError("no single dominant solid surface")
    # The convex envelope of the largest supported area may have concave borders but may
    # not hide a structural opening in its interior. Never fill a hole to manufacture support.
    hull = np.zeros_like(out, dtype=np.uint8)
    contours, _ = cv2.findContours(
        out.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.fillConvexPoly(hull, cv2.convexHull(max(contours, key=cv2.contourArea)), 1)
    holes = ndi.binary_fill_holes(mask) & ~mask
    if np.count_nonzero(holes & ndi.binary_erosion(hull, iterations=5)) >= max(
        12, round(0.0001 * mask.sum())
    ):
        raise ValueError("an interior opening crosses the dominant surface")
    return out, holes


def foreground(rgb):
    m = rgb.min(axis=2) < 222
    ids, n = ndi.label(m)
    counts = np.bincount(ids.ravel(), minlength=n + 1)
    keep = counts >= 16
    keep[0] = False
    return keep[ids]


def align(panel, obj):
    source = foreground(panel)
    a, ha = body(source)
    b, hb = body(obj)
    c0, s0 = box(a)
    c1, s1 = box(b)
    scale = float(np.dot(s0, s1) / np.dot(s0, s0))
    if not 0.8 <= scale <= 1.2:
        raise ValueError("unsupported body scale")
    delta = c1 - scale * c0
    matrix = np.column_stack((np.eye(2) * scale, delta))

    def warp(x, it, bv=0):
        return cv2.warpAffine(
            x, matrix, (obj.shape[1], obj.shape[0]), flags=it, borderValue=bv
        )

    aligned = warp(a.astype("uint8"), cv2.INTER_NEAREST) > 0
    score = (aligned & b).sum() / (aligned | b).sum()
    if score < 0.94:
        raise ValueError(f"body agreement {score:.4f} < .94")
    h = warp(ha.astype("uint8"), cv2.INTER_NEAREST).astype(bool) | hb
    inner = aligned & b & obj
    distance = ndi.distance_transform_edt(inner)
    if h.any():
        distance = np.minimum(distance, ndi.distance_transform_edt(~h))
    margin = max(8.0, 0.035 * min(s1))
    support = np.clip((distance - margin) / margin, 0, 1)
    rgb = warp(panel, cv2.INTER_LINEAR, (255, 255, 255))
    return (
        rgb,
        support,
        {
            "solid_iou": float(score),
            "retained_pixel_fraction": float((support > 0).sum() / obj.sum()),
            "margin_px": margin,
            "uniform_scale": scale,
            "offset": delta.tolist(),
        },
    )


def atlas(mesh):
    prior = np.asarray(mesh.visual.material.baseColorTexture)
    tri = np.asarray(mesh.faces, dtype=np.int32)
    ids, bary, _ = gt._raster_bounded(gt._uv_pixels(mesh, len(prior)), tri, len(prior))
    y, x = np.nonzero(ids >= 0)
    q = tri[ids[y, x]]
    b = bary[y, x]
    weights = np.column_stack((1 - b.sum(1), b))
    p = np.einsum("ij,ijk->ik", weights, mesh.vertices[q])
    n = gt._norm(np.einsum("ij,ijk->ik", weights, mesh.vertex_normals[q]))
    return prior, ids, p, n, y, x


def cut(rgb0, rgb1, weight):
    """Find a single low-discrepancy seam in common object coordinates, across UV islands."""
    h = w = 240
    old, new, quality = [
        cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        for a in (rgb0, rgb1, weight)
    ]
    lab0 = cv2.cvtColor(old, cv2.COLOR_RGB2LAB).astype(float)
    lab1 = cv2.cvtColor(new, cv2.COLOR_RGB2LAB).astype(float)
    cost = 1 + np.linalg.norm(lab0 - lab1, axis=2) / 6
    idx = np.arange(w * h, dtype=int).reshape(h, w)
    start = idx.size
    stop = start + 1
    edges = []
    cs = []

    def add(a, b, c):
        edges.append((np.asarray(a).ravel(), np.asarray(b).ravel()))
        cs.append(np.asarray(c).ravel().astype(np.int32))

    for a, b, wt in (
        (idx[1:], idx[:-1], cost[1:] + cost[:-1]),
        (idx[:, 1:], idx[:, :-1], cost[:, 1:] + cost[:, :-1]),
    ):
        add(a, b, wt)
        add(b, a, wt)
    good = quality >= 0.94
    bad = quality <= 0.04
    if good.sum() < 20 or not bad.any():
        raise ValueError("no stable patch core/boundary")
    add(np.full(idx.size, start), idx, np.where(good, 1000000, 8 * quality**2))
    add(idx, np.full(idx.size, stop), np.where(bad, 1000000, 8 * (1 - quality) ** 2))
    a, b0 = (np.concatenate([e[k] for e in edges]) for k in range(2))
    mat = coo_matrix((np.concatenate(cs), (a, b0)), shape=(stop + 1, stop + 1)).tocsr()
    flow = maximum_flow(mat, start, stop, method="dinic")
    residual = mat - flow.flow
    residual.eliminate_zeros()
    reachable = breadth_first_order(
        residual, start, directed=True, return_predecessors=False
    )
    region = np.isin(idx, reachable).astype("float32")
    # About a one-texel antialias at the seam, instead of blending distinguishable motifs together.
    return cv2.resize(
        region, (weight.shape[1], weight.shape[0]), interpolation=cv2.INTER_LINEAR
    )


def _paint(
    original,
    native_common,
    foreground_file,
    camera,
    transform,
    guide_file,
    sheet_file,
    orientation,
    out,
):
    if orientation["hold_projection"]:
        raise ValueError("front-layout veto")
    matched = orientation["inputs"]
    if (
        matched["model_sha256"] != digest(native_common)
        or matched["foreground_sha256"] != digest(foreground_file)
        or matched["sheet_sha256"] != digest(sheet_file)
    ):
        raise ValueError("orientation binding differs")
    common, *_ = gt._model(native_common)
    doc, blob, native, rawmesh = _native(original)
    mesh = rawmesh.copy()
    mesh.apply_transform(_display_matrix(transform))
    if len(mesh.faces) != len(common.faces):
        raise ValueError("different triangle counts")
    tol = np.ptp(mesh.vertices, axis=0).max() * 2e-5
    dist, index = cKDTree(common.triangles_center).query(
        mesh.triangles_center, workers=1
    )
    if dist.max() > tol or len(np.unique(index)) != len(index):
        raise ValueError("registered triangle centers differ")
    corners = np.linalg.norm(
        mesh.triangles[:, :, None, :] - common.triangles[index, None, :, :], axis=3
    )
    if corners.min(axis=2).max() > tol or corners.min(axis=1).max() > tol:
        raise ValueError("registered triangle corners differ")
    if not np.array_equal(
        np.array(gt._image(gt._single(guide_file, 8 * gt.MIB), 2 * gt.SIZE**2)),
        gt._guide_sheet(common),
    ):
        raise ValueError("guide pixels differ from the registered geometry")
    sheet = gt._image(gt._single(sheet_file, 32 * gt.MIB), 12_000_000)
    if (
        sheet.mode != "RGB"
        or min(sheet.size) < 256
        or abs(sheet.width / sheet.height - 2) >= 0.08
    ):
        raise ValueError("Expected an opaque two-panel sheet")
    before, ids, points, normals, y, x = atlas(mesh)
    if not np.array_equal(before[:, :, :3], native) or before.shape != native.shape:
        raise ValueError("native UV texture differs")
    matte = (
        gt._matte(io.BytesIO(gt._single(foreground_file, 32 * gt.MIB)), gt.SIZE)[
            :, :, 3
        ]
        >= 150
    )
    lock, source_fit = gt._source_lock(mesh, matte, camera, ids, points)
    extent = np.ptp(mesh.vertices, axis=0)
    depth_ratio = float(extent[2] / min(extent[:2]))
    if depth_ratio > 0.65:
        raise ValueError("The body is not a shallow broad face.")
    views = []
    stats = {}
    for i, (name, angle) in enumerate(gt.ANGLES.items()):
        view = gt._view(common, angle)
        obj = view[3] >= 0
        rgb = np.array(
            sheet.crop(
                (i * sheet.width // 2, 0, (i + 1) * sheet.width // 2, sheet.height)
            ).resize((gt.SIZE, gt.SIZE), Image.Resampling.LANCZOS)
        )
        try:
            _, support, report = align(rgb, obj)
        except ValueError as e:
            stats[name] = {"hold": str(e)}
            continue
        stats[name] = report
        solid, _ = body(foreground(rgb))
        views.append((name, view, support, rgb, solid))
    if not views:
        raise ValueError("no supported region")
    # Use one ornament field from the least foreshortened valid view. Both cameras
    # can support the geometry without imposing two conflicting ornament layouts.
    chosen = max(views, key=lambda row: float(-row[1][0][2, 2]))
    name, _, _, paint, paint_solid = chosen
    virtual = gt._view(mesh, (180, 0))
    vb, vc, vs, v_ids, vbary, vdepth = virtual
    v_solid, _ = body(v_ids >= 0)

    def row_map(mask):
        rows = np.where(mask.any(axis=1))[0]
        return (
            rows.min(),
            rows.max(),
            mask.argmax(1),
            mask.shape[1] - 1 - mask[:, ::-1].argmax(1),
        )

    t0, t1, tl, tr = row_map(v_solid)
    s0, s1, sl, sr = row_map(paint_solid)

    def field(pt, norm):
        pos = pt @ vb.T
        xy = (pos[:, :2] - vc[:2]) * [vs, -vs] + gt.SIZE / 2
        u, v = np.rint(xy).astype(int).clip(0, gt.SIZE - 1).T
        vf = (xy[:, 1] - t0) / (t1 - t0)
        sy = np.clip(s0 + vf * (s1 - s0), s0, s1)
        srows = np.rint(sy).astype(int)
        frac = (xy[:, 0] - tl[v]) / np.maximum(tr[v] - tl[v], 1)
        sx = sl[srows] + frac * (sr[srows] - sl[srows])
        pigment = gt._sample(paint, np.column_stack((sx, sy)))
        reliability = np.zeros(len(pt), np.float32)
        for _, (b, c, scale, which, _, dep), support, _, _ in views:
            q = pt @ b.T
            uv = (q[:, :2] - c[:2]) * [scale, -scale] + gt.SIZE / 2
            qx, qy = np.rint(uv).astype(int).clip(0, gt.SIZE - 1).T
            visible = (which[qy, qx] >= 0) & (
                np.abs(dep[qy, qx] - q[:, 2]) < 1.7 / scale
            )
            reliability = np.maximum(
                reliability,
                visible * support[qy, qx] * np.clip((norm @ b[2] - 0.13) / 0.28, 0, 1),
            )
        reverse_visible = v_solid[v, u] & (np.abs(vdepth[v, u] - pos[:, 2]) < 1.7 / vs)
        # Leave silhouette, unsupported parts and the branch onto the ring in the native texture.
        reliability *= reverse_visible * np.clip((-norm[:, 2] - 0.18) / 0.37, 0, 1)
        return pigment, reliability, (u, v)

    vy, vx = np.nonzero(v_ids >= 0)
    tri = mesh.faces[v_ids[vy, vx]]
    bw = vbary[vy, vx]
    w = np.column_stack((1 - bw.sum(1), bw))
    vp = np.einsum("ij,ijk->ik", w, mesh.vertices[tri])
    vn = gt._norm(np.einsum("ij,ijk->ik", w, mesh.vertex_normals[tri]))
    vst = np.einsum("ij,ijk->ik", w, mesh.visual.uv[tri])
    tx = vst[:, 0] * (native.shape[1] - 1)
    ty = (1 - vst[:, 1]) * (native.shape[0] - 1)
    old = gt._sample(native, np.column_stack((tx, ty)))
    new, conf, _ = field(vp, vn)
    oldview = np.zeros((gt.SIZE, gt.SIZE, 3), np.uint8)
    newview = oldview.copy()
    confidence = np.zeros(oldview.shape[:2], np.float32)
    oldview[vy, vx] = old
    newview[vy, vx] = new
    confidence[vy, vx] = conf
    seam = cut(oldview, newview, confidence)
    rgb, score, (px, py) = field(points, normals)
    blend = seam[py, px] * np.clip(score / 0.06, 0, 1)
    blend[lock[y, x]] = 0.0
    support = blend > 0
    if np.count_nonzero(support) < max(32, round(0.03 * len(blend))):
        raise ValueError("Insufficient supported hidden surface for a coherent patch")
    updated = native.copy()
    base = native[y, x].astype(float)
    updated[y, x] = (
        np.rint(base + blend[:, None] * (rgb - base)).clip(0, 255).astype(np.uint8)
    )
    delta = updated.astype("int16") - native.astype("int16")
    if not np.any(delta[y, x]):
        raise ValueError("No material change on the inferred surface")
    nearest = ndi.distance_transform_edt(
        ids < 0, return_distances=False, return_indices=True
    )
    gutter = (ids < 0) & ~lock
    updated[gutter] = np.clip(
        native[gutter].astype("int16") + delta[nearest[0][gutter], nearest[1][gutter]],
        0,
        255,
    )
    if np.any(updated[lock] != native[lock]) or np.any(
        updated[y[~support], x[~support]] != native[y[~support], x[~support]]
    ):
        raise ValueError("native paint lock failure")
    rawmesh.visual.material.baseColorTexture = Image.fromarray(updated)
    export_finished(rawmesh, out)
    Image.fromarray(updated).save(out / "texture.png")
    doc["images"][0]["mimeType"] = "image/png"
    others = gt._seal_glb(
        doc, blob, (out / "texture.png").read_bytes(), out / "model.glb"
    )
    embed_master(out / "model.glb", out / "editable.gltf")
    reread, other_blob, got, _ = _native(out / "model.glb")
    if not np.array_equal(got, updated):
        raise ValueError("Native base colour round-trip differs")
    if any(doc[k] != reread[k] for k in doc if k not in ("buffers", "bufferViews")):
        raise ValueError("Native glTF descriptors changed during export")
    color_view = doc["images"][0]["bufferView"]
    for i, (a, b) in enumerate(zip(doc["bufferViews"], reread["bufferViews"])):
        if (
            i != color_view
            and hashlib.sha256(_view_bytes(other_blob, b)).hexdigest()
            != others[i - (i > color_view)]
        ):
            raise ValueError(
                "Native geometry, samplers, or other material data changed"
            )
    ans = {
        "method": "native-reverse-solid-region-v1",
        "appearance": "needs_review",
        "source_image_sha256": matched["source_sha256"],
        "source_glb": digest(original),
        "registered_sha": digest(native_common),
        "output_glb": digest(out / "model.glb"),
        "guide_sha": digest(guide_file),
        "sheet_sha": digest(sheet_file),
        "visible_front_locked": int(lock.sum()),
        "locked_changed": 0,
        "unsupported_surface_changed": 0,
        "supported_uv": int(support.sum()),
        "total_uv": len(support),
        "changed_uv": int(np.any(delta[y, x] != 0, axis=1).sum()),
        "gamut_changed_only_embedded_base_color": True,
        "normals_uv_nodes_material_and_all_other_buffers_bitwise_identical": True,
        "area_source_coverage": stats,
        "source_camera_iou": source_fit,
        "coherent_paint_from": name,
        "source_face_camera": "original world pose, unchanged source glTF nodes",
        "depth_over_lesser_frontal_span": depth_ratio,
        "opaque_core_fraction": float(np.mean(blend[support] > 0.98)),
        "non_base_view_sha256": others,
    }
    return ans


def _view_bytes(blob, view):
    start, size = view.get("byteOffset", 0), view["byteLength"]
    if (
        view.get("buffer") != 0
        or type(start) is not int
        or type(size) is not int
        or start < 0
        or size <= 0
        or start + size > len(blob)
        or "byteStride" in view
    ):
        raise ValueError("Invalid native buffer view")
    return blob[start : start + size]


def _native(path):
    """Read a small fully embedded opaque PBR mesh. Never resolve an external URI."""
    raw = gt._single(path, 48 * gt.MIB)
    if len(raw) < 28 or struct.unpack_from("<4sII", raw) != (b"glTF", 2, len(raw)):
        raise ValueError("Expected a native binary glTF")
    size, tag = struct.unpack_from("<II", raw, 12)
    if not 0 < size <= gt.MIB or tag != 0x4E4F534A or 28 + size > len(raw):
        raise ValueError("Invalid GLB JSON chunk")
    doc = json.loads(raw[20 : 20 + size])
    blob = memoryview(raw)[28 + size :]
    if struct.unpack_from("<II", raw, 20 + size) != (len(blob), 0x004E4942):
        raise ValueError("Invalid GLB BIN chunk")
    try:
        if (
            set(doc)
            - {
                "asset",
                "scene",
                "scenes",
                "nodes",
                "meshes",
                "materials",
                "buffers",
                "bufferViews",
                "accessors",
                "images",
                "textures",
                "samplers",
                "extras",
            }
            or len(doc["buffers"]) != 1
            or "uri" in doc["buffers"][0]
            or doc["buffers"][0]["byteLength"]
            not in range(len(blob) - 3, len(blob) + 1)
            or len(doc["meshes"]) != 1
            or len(doc["meshes"][0]["primitives"]) != 1
            or len(doc["materials"]) != 1
            or len(doc["accessors"]) != 4
        ):
            raise ValueError("Unsupported native mesh structure")
        node = doc["nodes"]
        if (
            not 1 <= len(node) <= 2
            or any(set(n) - {"name", "mesh", "children"} for n in node)
            or sum(n.get("mesh") == 0 for n in node) != 1
            or doc.get("scene", 0) != 0
            or len(doc["scenes"]) != 1
        ):
            raise ValueError("Unsupported native scene transforms")
        roots = doc["scenes"][0]["nodes"]
        if (
            len(roots) != 1
            or roots[0] not in range(len(node))
            or (len(node) == 2 and node[roots[0]].get("children") != [1 - roots[0]])
        ):
            raise ValueError("Unsupported native hierarchy")
        prim = doc["meshes"][0]["primitives"][0]
        attr = prim["attributes"]
        accessor_ids = [*attr.values(), prim.get("indices")]
        if any(type(i) is not int for i in accessor_ids) or set(accessor_ids) != set(
            range(4)
        ):
            raise ValueError(
                "Native positions, normals, UVs and indices must have distinct valid accessors"
            )
        if (
            set(attr) != {"POSITION", "NORMAL", "TEXCOORD_0"}
            or prim.get("mode", 4) != 4
            or prim.get("material") != 0
            or set(prim) - {"attributes", "material", "indices", "mode"}
        ):
            raise ValueError("Native normals, triangles and UVs are required")
        material = doc["materials"][0]
        pbr = material["pbrMetallicRoughness"]
        imgs = doc["images"]
        tex = doc["textures"]
        if (
            not 1 <= len(imgs) <= 2
            or len(tex) != len(imgs)
            or set(material)
            - {
                "name",
                "doubleSided",
                "pbrMetallicRoughness",
                "normalTexture",
                "alphaMode",
            }
            or material.get("alphaMode", "OPAQUE") != "OPAQUE"
            or pbr["baseColorTexture"] != {"index": 0}
            or pbr.get("baseColorFactor", [1] * 4) != [1] * 4
            or set(pbr)
            - {
                "baseColorTexture",
                "baseColorFactor",
                "metallicFactor",
                "roughnessFactor",
            }
            or any(
                t.get("source") != i or set(t) - {"name", "source", "sampler"}
                for i, t in enumerate(tex)
            )
            or len(doc.get("samplers", [])) > 2
        ):
            raise ValueError(
                "Only one opaque base colour and an optional native normal map are supported"
            )
        if len(imgs) == 2:
            normal_map = material.get("normalTexture", {})
            if normal_map.get("index") != 1 or set(normal_map) - {"index", "scale"}:
                raise ValueError("Unsupported native material binding")
        elif "normalTexture" in material:
            raise ValueError("Missing normal map")
        for t in tex:
            if "sampler" in t and (
                type(t["sampler"]) is not int
                or t["sampler"] not in range(len(doc.get("samplers", [])))
            ):
                raise ValueError("Invalid sampler")
        views = doc["bufferViews"]
        if len(views) != 4 + len(imgs):
            raise ValueError(
                "Native accessors and images must have independent storage"
            )
        for v in views:
            _view_bytes(blob, v)
        slots = [a["bufferView"] for a in doc["accessors"]] + [
            i["bufferView"] for i in imgs
        ]
        if sorted(slots) != list(range(len(views))):
            raise ValueError("Native buffer view aliases are unsupported")
        for k, idx, dt, count in (
            ("VEC3", attr["POSITION"], 3, 350000),
            ("VEC3", attr["NORMAL"], 3, 350000),
            ("VEC2", attr["TEXCOORD_0"], 2, 350000),
            ("SCALAR", prim["indices"], 1, 1500000),
        ):
            a = doc["accessors"][idx]
            n = a["count"]
            component = a["componentType"]
            offset = a.get("byteOffset", 0)
            if (
                (component != 5126 if dt > 1 else component not in (5123, 5125))
                or a["type"] != k
                or "sparse" in a
                or a.get("normalized")
                or type(n) is not int
                or not 1 <= n <= count
                or offset < 0
                or offset + n * dt * (2 if component == 5123 else 4)
                > len(_view_bytes(blob, views[a["bufferView"]]))
            ):
                raise ValueError("Invalid or oversized native accessor")
        for i, desc in enumerate(imgs):
            if (
                "uri" in desc
                or set(desc) - {"name", "bufferView", "mimeType"}
                or desc.get("mimeType") not in ("image/jpeg", "image/png")
            ):
                raise ValueError("Native textures must be opaque embedded JPEG or PNG")
            image_bytes = _view_bytes(blob, views[desc["bufferView"]])
            with Image.open(io.BytesIO(image_bytes)) as header:
                if Image.MIME.get(header.format) != desc["mimeType"]:
                    raise ValueError("Native image encoding and MIME type disagree")
            image = gt._image(image_bytes, 1024**2)
            if (
                image.mode != "RGB"
                or image.width != image.height
                or image.width not in (256, 512, 1024)
            ):
                raise ValueError("Unsupported texture format or dimensions")
            if i == 0:
                native = np.array(image)
        # Parse only the validated, embedded payload, without a filesystem resolver.
        scene = trimesh.load(
            io.BytesIO(raw), file_type="glb", force="scene", process=False
        )
        (mesh,) = scene.geometry.values()
        if (
            not len(mesh.faces)
            or len(mesh.faces) > 500000
            or mesh.faces.min() < 0
            or mesh.faces.max() >= len(mesh.vertices)
            or not np.isfinite(mesh.vertices).all()
            or not np.isfinite(mesh.visual.uv).all()
            or np.min(mesh.visual.uv) < 0
            or np.max(mesh.visual.uv) > 1
            or np.max(np.abs(mesh.vertices)) > 1e4
            or not np.isfinite(mesh.vertex_normals).all()
        ):
            raise ValueError(
                "Invalid native positions, normals, faces, or texture coordinates"
            )
        return doc, blob, native, mesh
    except (IndexError, TypeError, KeyError, struct.error) as err:
        raise ValueError("Malformed native PBR GLB") from err


def _admit_observation(layout, receipt, orientation, source_sha256):
    from engine.codex_photo_observations import (
        METHOD,
        METHOD_V2,
        validate_provider_observations as validate_codex,
    )
    from engine.visual_brief import validate_provider_observations as validate_local

    brief = layout["visual_brief"]
    validator = (
        validate_codex if brief.get("method") in (METHOD, METHOD_V2) else validate_local
    )
    observation = validator(brief, source_sha256)
    brief_sha = json_sha(observation)
    if (
        receipt.get("visual_brief_sha256") != brief_sha
        or orientation["inputs"].get("brief_sha256") != brief_sha
    ):
        raise ValueError(
            "Observation identities differ between receipt and orientation review"
        )
    if (
        orientation.get("hold_projection") is not False
        or orientation.get("status") != "needs_review"
    ):
        raise ValueError(
            "Source-facing details or an unresolved orientation veto prevent regional recovery"
        )
    context = observation.get("function_context")
    if context and set(context["interpretation"]["reverse_candidates"]) & {
        "distinct_working_surface",
        "display_or_interface",
        "opening_or_cavity",
    }:
        raise ValueError(
            "The photograph suggests that the opposite side may have a distinct function"
        )
    if "repeated" not in observation["observed_face_cues"]:
        raise ValueError("No independently observed recurring surface decoration")
    return observation


def paint_native_region(
    native_model,
    registered_model,
    source_photo,
    foreground,
    registration,
    guide_dir,
    inferred_sheet,
    layout,
    receipt,
    orientation,
    output_dir,
):
    """Recover a limited appearance proposal only after full-sheet registration has failed.

    Arguments registration, layout, receipt and orientation are the job's sealed
    records. A result is still hypothetical and needs visual review. A veto writes
    no output directory. Callers must retain the native source as a separate master.
    """
    (
        native_model,
        registered_model,
        source_photo,
        foreground,
        guide_dir,
        inferred_sheet,
        output_dir,
    ) = map(
        Path,
        (
            native_model,
            registered_model,
            source_photo,
            foreground,
            guide_dir,
            inferred_sheet,
            output_dir,
        ),
    )
    source_sha = hashlib.sha256(gt._single(source_photo, 32 * gt.MIB)).hexdigest()
    guide_file = guide_dir / "guides.png"
    shape_sha, native_sha, matte_sha, guide_sha, sheet_sha = (
        hashlib.sha256(gt._single(path, limit * gt.MIB)).hexdigest()
        for path, limit in (
            (registered_model, 48),
            (native_model, 48),
            (foreground, 32),
            (guide_file, 8),
            (inferred_sheet, 32),
        )
    )
    layout_guide = json.loads(gt._single(guide_dir / "guide.json", 65536))
    angles = {k: list(v) for k, v in gt.ANGLES.items()}
    expected = {
        "version": gt.VERSION,
        "model_sha256": shape_sha,
        "guide_sha256": guide_sha,
        "views": angles,
        "size": gt.SIZE,
    }
    if layout_guide != expected or (
        layout["photo_sha"],
        layout["model_sha"],
        layout["guide_sha"],
        layout["camera"],
    ) != (source_sha, shape_sha, guide_sha, angles):
        raise ValueError("Wrong photograph, registered body or camera guides")
    if (
        registration.get("mesh_file_sha256") != native_sha
        or registration.get("foreground_file_sha256") != matte_sha
        or registration.get("status") == "poor-fit"
    ):
        raise ValueError(
            "Source-camera registration does not belong to this native model"
        )
    if receipt.get("state") != "completed" or (
        receipt.get("sheet_sha256"),
        receipt.get("geometry_sha256"),
        receipt.get("guide_sha256"),
        receipt.get("photo_sha256"),
    ) != (sheet_sha, shape_sha, guide_sha, source_sha):
        raise ValueError("Native repair cannot use an unbound paint receipt")
    if not (receipt.get("thread_id") or receipt.get("host_tool_call_id")):
        raise ValueError("No completed generation identity")
    evidence = orientation["inputs"]
    if orientation.get("report_sha256") != json_sha(
        {k: v for k, v in orientation.items() if k != "report_sha256"}
    ) or (
        evidence.get("source_sha256"),
        evidence.get("model_sha256"),
        evidence.get("foreground_sha256"),
        evidence.get("sheet_sha256"),
        evidence.get("camera_sha256"),
        evidence.get("angles"),
    ) != (
        source_sha,
        shape_sha,
        matte_sha,
        sheet_sha,
        json_sha(registration["camera_frame"]),
        angles,
    ):
        raise ValueError("Hidden-layout review does not match this proposal")
    observation = _admit_observation(layout, receipt, orientation, source_sha)
    with Image.open(io.BytesIO(gt._single(foreground, 32 * gt.MIB))) as mask:
        if (
            mask.format != "PNG"
            or mask.mode != "RGBA"
            or mask.width * mask.height > 40_000_000
            or getattr(mask, "n_frames", 1) != 1
        ):
            raise ValueError("Expected a bounded segmented foreground")
    with gt._stage(output_dir) as stage:
        info = _paint(
            native_model,
            registered_model,
            foreground,
            registration["camera_frame"],
            registration["upright_transform"],
            guide_file,
            inferred_sheet,
            orientation,
            stage,
        )
        if (
            digest(native_model) != native_sha
            or digest(inferred_sheet) != sheet_sha
            or digest(foreground) != matte_sha
            or digest(registered_model) != shape_sha
        ):
            raise ValueError("Inputs changed while applying native material")
        info.update(
            observation_method=observation["method"],
            visual_brief_sha256=json_sha(observation),
            generation_id=receipt.get("thread_id") or receipt["host_tool_call_id"],
            hidden_layout_sha256=orientation["report_sha256"],
            source_rear_observed=False,
            interpretation="Inferred continuation of photographed decoration; the opposite face may differ.",
        )
        gt._json(stage / "region.json", info)
    return info
