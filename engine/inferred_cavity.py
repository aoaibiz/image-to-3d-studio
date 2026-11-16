"""Build a separately labelled interior hypothesis on an upright, sealed native mesh."""

import hashlib
import io
import json
import struct
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image

METHOD = "source-rim-inferred-cavity-v1"


class IntegrityError(ValueError):
    """Sealed inputs cannot be trusted: do not silently downgrade corruption."""


class Hold(ValueError):
    """Input lacks the conservative geometric or photo cues for this hypothesis."""


def eligible(o):
    role = o.get("function_context", {})
    interpretation = role.get("interpretation", {})
    if interpretation.get("confidence") not in {"low", "moderate"}:
        return False
    if interpretation.get("possible_purpose") != "contain_or_pass":
        return False
    items = [*role.get("observed_cues", []), *o.get("structures", [])]
    if not any(
        x.get("kind") == "rim_or_frame" and x.get("location") == "upper_border"
        for x in items
    ):
        return False
    incompatible = {
        "boundary_seam",
        "hinge_or_pivot",
        "latch_or_fastener",
        "control_or_interface",
        "attachment_point",
        "projecting_assembly",
        "fasteners",
        "screen_or_controls",
        "raised_relief",
        "parallel_wires",
        "hinge",
        "latch",
        "lid",
        "stopper",
    }
    return not any(x.get("kind") in incompatible for x in items)


def read_glb(path):
    if Path(path).stat().st_size > 32_000_000:
        raise Hold("native mesh too large")
    data = Path(path).read_bytes()
    if len(data) < 28 or data[:4] != b"glTF":
        raise Hold("unsupported file")
    jl = struct.unpack_from("<I", data, 12)[0]
    j = json.loads(data[20 : 20 + jl])
    b = data[28 + jl :]
    if any(item.get("uri") for item in j.get("buffers", []) + j.get("images", [])):
        raise Hold("self-contained native GLB required")
    if len(j["meshes"]) != 1 or len(j["meshes"][0]["primitives"]) != 1:
        raise Hold("one native primitive required")
    if j.get("animations") or j.get("skins"):
        raise Hold("animated geometry")
    for node in j["nodes"]:
        if node.get("matrix") or node.get("scale", [1, 1, 1]) != [1, 1, 1]:
            raise Hold("unsupported native basis")
        q = node.get("rotation", [0, 0, 0, 1])
        if abs(q[0]) + abs(q[2]) > 1e-6:
            raise Hold("unsupported native up")
    return j, b


def arr(j, b, index):
    a = j["accessors"][index]
    view = j["bufferViews"][a["bufferView"]]
    if a.get("sparse") or view.get("byteStride"):
        raise Hold("unsupported packed layout")
    t = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[a["componentType"]]
    n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
    return (
        np.frombuffer(
            b,
            dtype=t,
            count=a["count"] * n,
            offset=a.get("byteOffset", 0) + view.get("byteOffset", 0),
        )
        .reshape(-1, n)
        .copy()
    )


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def section(m, y):
    seg = trimesh.intersections.mesh_plane(m, [0, 1, 0], [0, y, 0])[:, :, [0, 2]]
    if len(seg) < 8:
        raise Hold("insufficient wall envelope")
    return seg


def span(s):
    return np.ptp(s.reshape(-1, 2), axis=0)


def envelope(seg, center, angles):
    a = seg[:, 0] - center
    e = seg[:, 1] - seg[:, 0]
    radii = []
    for th in angles:
        d = np.array([np.cos(th), np.sin(th)])
        den = cross(d, e)
        ok = abs(den) > 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            t = cross(a, e) / den
            u = cross(a, d) / den
        hits = np.unique(np.round(t[ok & (t > 0) & (u >= -1e-8) & (u < 1 - 1e-8)], 6))
        if len(hits) != 1:
            raise Hold("section is not a single star-shaped shell")
        radii.append(hits[0])
    return np.array(radii)


def floor_faces(polygon):
    # Concave n-gon from a triangulated cap may have backward-facing radial steps.
    # Clip geometric ears rather than using a center vertex that could overlap.
    ids = list(range(len(polygon)))
    triangles = []
    eps = np.ptp(polygon, axis=0).max() ** 2 * 1e-10
    while len(ids) > 3:
        for k, ib in enumerate(ids):
            ia, ic = ids[k - 1], ids[(k + 1) % len(ids)]
            a, b, c = polygon[[ia, ib, ic]]
            if cross(b - a, c - a) >= -eps:
                continue
            rest = polygon[[i for i in ids if i not in (ia, ib, ic)]]
            inside = (
                (cross(b - a, rest - a) < eps)
                & (cross(c - b, rest - b) < eps)
                & (cross(a - c, rest - c) < eps)
            )
            if np.any(inside):
                continue
            triangles.append([ia, ib, ic])
            ids.pop(k)
            break
        else:
            raise Hold("inferred floor polygon is not safely triangulable")
    triangles.append(ids)
    tri = polygon[np.asarray(triangles)]
    if np.max(cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])) >= -eps:
        raise Hold("degenerate inferred floor")
    return np.asarray(triangles)


def annulus(outer, inner):
    # Join the nearest points across the annulus, unfold the two rings into
    # one weakly simple polygon, and cut ears. Shared bridge indices cancel.
    point = np.vstack([outer, inner])
    ia, ib = np.unravel_index(
        np.linalg.norm(outer[:, None] - inner[None, :], axis=2).argmin(),
        (len(outer), len(inner)),
    )
    oi = np.roll(np.arange(len(outer)), -ia).tolist()
    ii = (len(outer) + (ib - np.arange(len(inner))) % len(inner)).tolist()
    ids = [oi[0], *ii, ii[0], *oi]
    eps = np.ptp(point, axis=0).max() ** 2 * 1e-11
    faces = []
    while len(ids) > 3:
        for k, bid in enumerate(ids):
            aid, cid = ids[k - 1], ids[(k + 1) % len(ids)]
            a, b, c = point[[aid, bid, cid]]
            if cross(b - a, c - a) >= -eps:
                continue
            others = point[[i for i in ids if i not in (aid, bid, cid)]]
            inside = (
                (cross(b - a, others - a) < eps)
                & (cross(c - b, others - b) < eps)
                & (cross(a - c, others - c) < eps)
            )
            if np.any(inside):
                continue
            faces.append([aid, bid, cid])
            ids.pop(k)
            break
        else:
            raise Hold("safe rim transition could not be triangulated")
    faces = np.array([*faces, ids])
    tri = point[faces]
    if np.any(cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) >= -eps):
        raise Hold("crossed rim transition")

    def edges(f):
        e = np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        u, n = np.unique(e, return_counts=True, axis=0)
        return {tuple(x) for x in u[n == 1]}

    expected = set()
    for start, length in [(0, len(outer)), (len(outer), len(inner))]:
        expected.update(
            tuple(sorted((start + i, start + (i + 1) % length))) for i in range(length)
        )
    if edges(faces) != expected:
        raise Hold("transition could not preserve exact rim edges")
    return faces


def make(m):
    lo, hi = m.bounds
    H = m.extents[1]
    top = hi[1]
    if H <= 0 or H / max(m.extents[[0, 2]]) < 0.9:
        raise Hold("flat envelope")
    merged = m.copy()
    merged.merge_vertices(merge_tex=True, merge_norm=True)
    if (
        not merged.is_watertight
        or not merged.is_winding_consistent
        or len(merged.split(only_watertight=False)) != 1
    ):
        raise Hold("native shell is not one closed oriented surface")
    collar = span(section(m, top - 0.02 * H))
    neck = span(section(m, top - 0.1 * H))
    if min(collar / neck) < 1.13 or max(collar) / min(collar) > 1.45:
        raise Hold("flared nearly round collar required")
    upper = m.vertices[m.vertices[:, 1] > top - 0.025 * H][:, [0, 2]]
    c = (upper.min(0) + upper.max(0)) / 2
    r = np.ptp(upper, axis=0) / 2
    centers = m.triangles_center
    rad = np.linalg.norm((centers[:, [0, 2]] - c) / r, axis=1)
    mask = (
        (rad < 0.74) & (m.face_normals[:, 1] > 0.86) & (centers[:, 1] > top - 0.035 * H)
    )
    if mask.sum() < 18:
        raise Hold("no identifiable cap patch")
    tri = m.triangles[mask].reshape(-1, 3)
    tri = np.unique(tri, axis=0)
    fit = np.linalg.lstsq(
        np.c_[tri[:, 0], tri[:, 2], np.ones(len(tri))], tri[:, 1], rcond=None
    )[0]
    resid = tri[:, 1] - (tri[:, 0] * fit[0] + tri[:, 2] * fit[1] + fit[2])
    if np.ptp(resid) > 0.045 * (2 * r.mean()) or np.linalg.norm(fit[:2]) > 0.15:
        raise Hold("top is not a planar cap")
    # Recover exactly the directed geometric boundary of the removed native faces, weld position-only UV seams for topology.
    verts, inverse = np.unique(m.vertices, axis=0, return_inverse=True)
    f = inverse[m.faces[mask]]
    edges = f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    keys = np.sort(edges, axis=1)
    _, rev, n = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    bd = edges[n[rev] == 1]
    nxt = dict(map(tuple, bd.tolist()))
    if len(nxt) != len(bd) or len(set(nxt.values())) != len(bd):
        raise Hold("ambiguous cut border")
    ids = [bd[0, 0]]
    while nxt[ids[-1]] != ids[0] and len(ids) <= len(bd):
        ids.append(nxt[ids[-1]])
    if len(ids) != len(bd):
        raise Hold("multiple boundary loops")
    outer = verts[ids]
    radial = outer[:, [0, 2]] - c
    angles = np.arctan2(radial[:, 1], radial[:, 0])
    a = np.unwrap(np.r_[angles, angles[0]])
    if np.max(np.diff(a)) > 0.45 or abs((a[-1] - a[0]) + 2 * np.pi) > 0.02:
        raise Hold("unstable radial seam")
    # The old triangulated seam is jagged; it ends at the shallow transition.
    # A regular ellipse entirely inside the cut protects all original side faces.
    normalised = radial / r
    e = np.roll(normalised, -1, axis=0) - normalised
    t = np.clip(-np.sum(normalised * e, axis=1) / np.sum(e * e, axis=1), 0, 1)
    inscribed = (
        float(np.min(np.linalg.norm(normalised + t[:, None] * e, axis=1))) * 0.94
    )
    if not 0.44 < inscribed < 0.88:
        raise Hold("irregular seam leaves no usable smooth aperture")
    n = max(48, min(96, 2 * len(outer)))
    angles = -np.arange(n) * 2 * np.pi / n
    aperture = c + inscribed * r * np.stack((np.cos(angles), np.sin(angles)), axis=1)
    bridge = annulus(outer[:, [0, 2]], aperture)
    lip_y = (
        aperture[:, 0] * fit[0] + aperture[:, 1] * fit[1] + fit[2] - 0.009 * min(r) * 2
    )
    lip = np.c_[aperture[:, 0], lip_y, aperture[:, 1]]
    radial = aperture - c
    angles = np.arctan2(radial[:, 1], radial[:, 0])
    native_r = np.linalg.norm(radial, axis=1)
    uvdir = radial / native_r[:, None]
    depth = min(1.05 * collar.mean(), 0.15 * H)
    basey = lip[:, 1].min()
    rings = [lip]
    rad = native_r.copy()
    for z in (0.035, 0.11, 0.25, 0.48, 0.72, 0.88, 0.95, 1.0):
        y = basey - depth * z
        seg = section(m, y)
        env = envelope(seg, c, angles)
        wall = max(0.09 * min(r), 0.009 * H)
        target = native_r * min(1 - 0.14 * z, float(np.min((env - wall) / native_r)))
        if min(target) < 0.35 * min(r):
            raise Hold("insufficient usable lumen")
        if z >= 0.88:
            target *= {0.88: 0.97, 0.95: 0.89, 1.0: 0.72}[z]
        rad = np.minimum(rad, target)
        p = np.empty_like(lip)
        p[:, [0, 2]] = c + rad[:, None] * uvdir
        p[:, 1] = y
        rings.append(p)
    nv = np.vstack([outer, *rings])
    faces = bridge.tolist()
    offset = len(outer)
    for k in range(len(rings) - 1):
        for i in range(n):
            l = offset + k * n + i
            h = offset + k * n + (i + 1) % n
            faces.extend([[l, h, h + n], [l, h + n, l + n]])
    floor = floor_faces(rings[-1][:, [0, 2]]) + offset + (len(rings) - 1) * n
    faces.extend(floor.tolist())
    fm = np.asarray(faces)
    combined = trimesh.Trimesh(
        np.vstack([m.vertices, nv]),
        np.vstack([m.faces[~mask], fm + len(m.vertices)]),
        process=False,
    )
    combined.merge_vertices(merge_tex=True, merge_norm=True)
    if (
        not combined.is_watertight
        or not combined.is_winding_consistent
        or combined.volume <= 0
        or combined.euler_number != merged.euler_number
    ):
        raise Hold("topology/inward orientation")
    # Verify transverse samples of inner triangles against original wall between the construction rings.
    for k in range(1, len(rings) - 1):
        for t in (0.23, 0.5, 0.77):
            p = rings[k] * (1 - t) + rings[k + 1] * t
            env = envelope(section(m, p[0, 1]), c, angles)
            if np.min(env - np.linalg.norm(p[:, [0, 2]] - c, axis=1)) < 0.5 * wall:
                raise Hold("thin wall cross-check")
    stats = dict(
        width=float(collar.mean()),
        depth=float(depth),
        source_faces=len(m.faces),
        removed_cap_faces=int(mask.sum()),
        added_interior_faces=len(fm),
        boundary_vertices=len(outer),
        inner_lip_vertices=n,
        collar_over_neck=(collar / neck).tolist(),
        minimum_design_wall=float(wall),
        connected_watertight=bool(combined.is_watertight),
        euler_before=merged.euler_number,
        euler_after=combined.euler_number,
    )
    return mask, nv, fm, stats


def cap_might_carry_graphics(pixels):
    c = np.asarray(pixels, dtype=float)
    chroma = c - c.mean(axis=1, keepdims=True)
    contrast = np.quantile(
        np.linalg.norm(chroma - np.median(chroma, axis=0), axis=1), 0.95
    )
    luminance = c @ np.array([0.2126, 0.7152, 0.0722])
    return contrast > 0.13 or np.ptp(np.percentile(luminance, [5, 95])) > 0.24


def emit(model, observation, output):
    if not eligible(observation):
        raise Hold("no compatible typed function and rim cues")
    if Path(output).exists():
        raise Hold("candidate destination already exists")
    j, b = read_glb(model)
    prim = j["meshes"][0]["primitives"][0]
    attrs = prim["attributes"]
    pos = arr(j, b, attrs["POSITION"])
    idx = arr(j, b, prim["indices"]).flatten().astype(int).reshape(-1, 3)
    if len(idx) > 120_000 or len(pos) > 240_000 or not np.isfinite(pos).all():
        raise Hold("outside bounded mesh complexity")
    mesh = trimesh.Trimesh(pos, idx, process=False)
    remove, nv, nf, stats = make(mesh)
    native_mat = j["materials"][prim["material"]]["pbrMetallicRoughness"]
    uv = arr(j, b, attrs["TEXCOORD_0"])
    appearance = j["textures"][native_mat["baseColorTexture"]["index"]]
    img = j["images"][appearance["source"]]
    view = j["bufferViews"][img["bufferView"]]
    offs = view.get("byteOffset", 0)
    with Image.open(io.BytesIO(b[offs : offs + view["byteLength"]])) as im:
        if im.width * im.height > 16_777_216:
            raise Hold("oversized source texture")
        rgb = np.asarray(im.convert("RGB"))
    # Sample the paint inside the faces as well as their vertices. A reliable
    # opening hypothesis must not overwrite decorated/marked lids.
    uvtri = uv[idx[remove]]
    for w in [[1 / 3] * 3, [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]:
        u = (uvtri * np.array(w)[None, :, None]).sum(axis=1) % 1
        sample = (
            rgb[
                (u[:, 1] * rgb.shape[0]).astype(int),
                (u[:, 0] * rgb.shape[1]).astype(int),
            ]
            / 255.0
        )
        if cap_might_carry_graphics(sample):
            raise Hold("pre-existing cap appearance may be decorated")
    # glTF V=0 is the upper image row; sample immutable native lip texels.
    uvs = uv[idx[remove]].reshape(-1, 2) % 1
    pixels = (
        rgb[
            (uvs[:, 1] * rgb.shape[0]).astype(int),
            (uvs[:, 0] * rgb.shape[1]).astype(int),
        ]
        / 255.0
    )
    shade = np.median(pixels, axis=0)
    linear = np.where(shade <= 0.04045, shade / 12.92, ((shade + 0.055) / 1.055) ** 2.4)
    rough = float(np.clip(native_mat.get("roughnessFactor", 0.6), 0.3, 0.9))
    material = len(j["materials"])
    j["materials"].append(
        dict(
            name="Inferred glazed interior",
            pbrMetallicRoughness=dict(
                baseColorFactor=[*linear, 1.0], metallicFactor=0, roughnessFactor=rough
            ),
        )
    )
    binary = bytearray(b[: j["buffers"][0]["byteLength"]])

    def add(data, t, n, target):
        while len(binary) % 4:
            binary.append(0)
        o = len(binary)
        binary.extend(data.tobytes())
        bv = len(j["bufferViews"])
        j["bufferViews"].append(
            dict(buffer=0, byteOffset=o, byteLength=len(binary) - o, target=target)
        )
        i = len(j["accessors"])
        d = dict(bufferView=bv, componentType=t, count=len(data), type=n)
        if n == "VEC3":
            d.update(min=data.min(0).tolist(), max=data.max(0).tolist())
        j["accessors"].append(d)
        return i

    prim["indices"] = add(idx[~remove].astype("<u4").ravel(), 5125, "SCALAR", 34963)
    inner = trimesh.Trimesh(nv, nf, process=False)
    normals = inner.vertex_normals.copy()
    # A shallow bottom joins the wall geometrically. Give its planar floor its own
    # normal at the crease so polygon triangulation never writes artificial rays.
    is_floor = inner.triangles[:, :, 1].max(1) <= nv[:, 1].min() + 1e-9
    floor_indices, inverse = np.unique(nf[is_floor], return_inverse=True)
    old_count = len(nv)
    nv = np.vstack((nv, nv[floor_indices]))
    normals = np.vstack((normals, np.tile([0.0, 1.0, 0.0], (len(floor_indices), 1))))
    nf = nf.copy()
    nf[is_floor] = old_count + inverse.reshape(-1, 3)
    dep = np.maximum(0, nv[:, 1].max() - nv[:, 1])
    radius = 0.5 * stats["width"]
    visibility = radius**2 / (radius**2 + dep**2)
    shade = np.repeat((0.46 + 0.54 * visibility)[:, None], 3, axis=1)
    colors = add(shade.astype("<f4"), 5126, "VEC3", 34962)
    p = add(nv.astype("<f4"), 5126, "VEC3", 34962)
    nr = add(normals.astype("<f4"), 5126, "VEC3", 34962)
    ix = add(nf.astype("<u4").ravel(), 5125, "SCALAR", 34963)
    j["meshes"][0]["primitives"].append(
        dict(
            attributes=dict(POSITION=p, NORMAL=nr, COLOR_0=colors),
            indices=ix,
            material=material,
            mode=4,
        )
    )
    j["buffers"][0]["byteLength"] = len(binary)
    binary.extend(b"\0" * ((-len(binary)) % 4))
    jb = json.dumps(j, separators=(",", ":")).encode()
    jb += b" " * ((-len(jb)) % 4)
    out = (
        struct.pack("<4sII", b"glTF", 2, 28 + len(jb) + len(binary))
        + struct.pack("<I4s", len(jb), b"JSON")
        + jb
        + struct.pack("<I4s", len(binary), b"BIN\0")
        + binary
    )
    with Path(output).open("xb") as handle:
        handle.write(out)
    stats.update(
        inferred=True,
        material_srgb=np.median(pixels, axis=0).tolist(),
        protected_native_attributes=True,
        native_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(out).hexdigest(),
    )
    return stats


def canonical_observation_sha(data):
    encoded = json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest(path, limit):
    if (
        path.is_symlink()
        or not path.is_file()
        or not (0 < path.stat().st_size <= limit)
    ):
        raise IntegrityError("missing or oversized sealed input")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _photo_flare(path):
    with Image.open(path) as im:
        if im.width * im.height > 40_000_000 or im.mode != "RGBA":
            raise Hold("a bounded source alpha is required")
        im.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        mask = np.asarray(im.getchannel("A")) >= 150
    ys = np.nonzero(mask.any(axis=1))[0]
    if len(ys) < 80 or not mask.any():
        raise Hold("insufficient source resolution")
    lo, hi = int(ys.min()), int(ys.max())
    height = hi - lo + 1

    def widths(a, b):
        rows = mask[lo + int(a * height) : lo + int(b * height)]
        return (
            rows.shape[1] - np.argmax(rows, axis=1) - np.argmax(rows[:, ::-1], axis=1)
        )

    lip = float(np.quantile(widths(0.008, 0.07), 0.90))
    neck = float(np.median(widths(0.09, 0.16)))
    if neck < 10 or lip / neck < 1.15:
        raise Hold("the source silhouette does not confirm an upper flare")
    return lip / neck


def _source_binding(source, foreground, observation, registration, display_model):
    proof = registration.get("shape_hypothesis_inputs", {})
    for key, path, limit in [
        ("source_sha256", source, 32_000_000),
        ("foreground_sha256", foreground, 32_000_000),
        ("display_model_sha256", display_model, 32_000_000),
    ]:
        if proof.get(key) != _digest(path, limit):
            raise IntegrityError("input hash mismatch: " + key)
    if proof.get("observation_canonical_sha256") != canonical_observation_sha(
        observation
    ):
        raise IntegrityError("sealed observation hash mismatch")

    def valid_sha(x):
        return (
            isinstance(x, str)
            and len(x) == 64
            and all(c in "0123456789abcdef" for c in x)
        )

    if not all(
        valid_sha(proof.get(k))
        for k in ["observer_output_sha256", "geometry_master_sha256"]
    ):
        raise IntegrityError("missing upstream observation or master digest")
    if (
        registration.get("mesh_file_sha256") != proof["geometry_master_sha256"]
        or registration.get("foreground_file_sha256") != proof["foreground_sha256"]
    ):
        raise IntegrityError("source registration seal mismatch")
    if proof.get("display_rotation_xyzw") != [0, 1, 0, 0]:
        raise Hold("unknown display convention")
    frame = registration.get("camera_frame", {})
    direction = np.array(frame.get("camera_direction_from_origin", []))
    up = np.array(frame.get("image_up", []))
    if (
        direction.shape != (3,)
        or up.shape != (3,)
        or not np.isfinite(np.r_[direction, up]).all()
        or abs(direction[1]) > 0.10
        or up[1] < 0.97
    ):
        raise Hold("visible cap or axis disagreement")
    if (
        registration.get("foreground_touches_source_border") != False
        or not 0.88 <= float(registration.get("silhouette_iou", 0)) <= 1
    ):
        raise Hold("insufficient source registration")
    return _photo_flare(foreground)


def editor(*, source, foreground, observation, registration, display_model, output):
    """One conservatively inferred alternative; native bytes and primary output stay intact."""
    source, foreground, display_model, output = map(
        Path, (source, foreground, display_model, output)
    )
    result = dict(
        decision="unchanged",
        method=METHOD,
        reason="",
        evidence=dict(
            opening="uncertain",
            visible_closure=None,
            painted_cap=None,
            negative_controls_passed=False,
        ),
        inferred_dimensions={},
    )
    try:
        flare = _source_binding(
            source, foreground, observation, registration, display_model
        )
        if not eligible(observation):
            raise Hold(
                "No compatible sealed containment and upper-rim evidence, or closure cues are present."
            )
        s = emit(display_model, observation, output)
    except Hold as exc:
        result["reason"] = str(exc)
        return result
    result.update(
        decision="alternative",
        reason="The source shows a flared rim but no measurable interior. Interior depth, wall and finish are inferred.",
        evidence=dict(
            opening="plausible",
            visible_closure=None,
            painted_cap=None,
            cap_colour_pattern_detected=False,
            geometry_supported=True,
            source_registered=True,
            negative_controls_passed=True,
            source_lip_over_neck=flare,
        ),
        inferred_dimensions=dict(
            depth=s["depth"], minimum_design_wall=s["minimum_design_wall"]
        ),
    )
    return result
