"""Transfer bounded pigment detail from two silhouette-guided hypotheses to unseen UVs.

Accepts the display-oriented, single-mesh output of the local shape painter. The
original segmented photo and the raw-to-display fitted camera remain authoritative.
No model execution, network access, image-provider selection or automatic approval.
"""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
import trimesh

from engine.finish_export import embed_master, export_finished
from engine.shape_appearance import _basis, _fit, _matte, _norm, _raster, _sample
from engine.shape_camera import lens_parameters

SIZE = 768
ANGLES = {"rear": (130, 12), "opposite": (-148, 10)}
VERSION = 1
MIB = 1024 * 1024


def _single(path, limit):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > limit:
        raise ValueError("Expected a bounded regular input file")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if not 0 < len(data) <= limit:
        raise ValueError("Input exceeds the file limit or is empty")
    return data


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _image(blob, max_area):
    with Image.open(io.BytesIO(blob)) as image:
        if (
            image.format not in {"JPEG", "PNG", "WEBP"}
            or image.width * image.height > max_area
            or getattr(image, "n_frames", 1) != 1
        ):
            raise ValueError(
                "Unsupported image format, number of frames or pixel count"
            )
        return image.copy()


@contextlib.contextmanager
def _stage(destination):
    target = Path(destination)
    if (
        target.is_symlink()
        or target.exists()
        and (not target.is_dir() or any(target.iterdir()))
    ):
        raise ValueError("Use an empty dedicated output directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        folder = Path(tmp) / "artifacts"
        folder.mkdir()
        yield folder
        if target.exists():
            target.rmdir()  # fails if a concurrent writer created anything
        os.rename(folder, target)


def _json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def _model(path):
    """Decode the painter's narrow native GLB format without external resource resolution."""
    data = _single(path, 48 * MIB)
    if len(data) < 28 or struct.unpack_from("<4sII", data) != (b"glTF", 2, len(data)):
        raise ValueError("Expected a glTF 2 binary")
    size, tag = struct.unpack_from("<II", data, 12)
    if tag != 0x4E4F534A or size > MIB or size + 28 > len(data):
        raise ValueError("Invalid GLB JSON chunk")
    doc = json.loads(data[20 : 20 + size])
    binary = memoryview(data)[28 + size :]
    if struct.unpack_from("<II", data, 20 + size) != (len(binary), 0x004E4942):
        raise ValueError("Invalid GLB binary chunk")
    try:
        buffers, images, meshes, materials, textures = (
            doc[k] for k in ("buffers", "images", "meshes", "materials", "textures")
        )
        if (
            any(len(seq) != 1 for seq in (buffers, images, meshes, materials, textures))
            or "uri" in buffers[0]
            or buffers[0]["byteLength"] > len(binary)
            or "uri" in images[0]
            or images[0]["mimeType"] != "image/png"
            or textures[0] != {"source": 0}
            or any(
                k in doc
                for k in ("skins", "animations", "extensionsUsed", "extensionsRequired")
            )
            or len(meshes[0]["primitives"]) != 1
        ):
            raise ValueError("Expected a single embedded native shape-painter atlas")
        nodes = doc["nodes"]
        if (
            not 1 <= len(nodes) <= 2
            or sum(n.get("mesh") == 0 for n in nodes) != 1
            or any(set(n) - {"name", "mesh", "children"} for n in nodes)
        ):
            raise ValueError("Expected an untransformed native mesh")
        root = doc["scenes"][doc["scene"]]["nodes"]
        if len(root) != 1 or (
            len(nodes) == 2 and nodes[root[0]].get("children") != [1 - root[0]]
        ):
            raise ValueError("Unsupported native scene hierarchy")
        prim = meshes[0]["primitives"][0]
        attr = prim["attributes"]
        if (
            prim.get("mode", 4) != 4
            or prim.get("material") != 0
            or set(attr) != {"POSITION", "NORMAL", "TEXCOORD_0"}
            or "targets" in prim
            or len(doc["accessors"]) != 4
        ):
            raise ValueError(
                "Expected an indexed triangle mesh with native normals and UVs"
            )
        pbr = materials[0]["pbrMetallicRoughness"]
        if (
            pbr["baseColorTexture"] != {"index": 0}
            or pbr.get("baseColorFactor", [1] * 4) != [1] * 4
            or set(pbr) - {"baseColorTexture", "roughnessFactor", "metallicFactor"}
            or set(materials[0]) - {"pbrMetallicRoughness", "name", "doubleSided"}
        ):
            raise ValueError("Unsupported native PBR material")

        views = doc["bufferViews"]
        if (
            len(views) != 5
            or len(
                {a["bufferView"] for a in doc["accessors"]} | {images[0]["bufferView"]}
            )
            != 5
        ):
            raise ValueError("Geometry and texture must have independent buffer views")

        def view(index):
            if type(index) is not int or not 0 <= index < len(views):
                raise ValueError("Invalid buffer view index")
            v = views[index]
            start, length = v.get("byteOffset", 0), v["byteLength"]
            if (
                v.get("buffer") != 0
                or "byteStride" in v
                or min(start, length) < 0
                or start + length > buffers[0]["byteLength"]
            ):
                raise ValueError("Invalid or non-contiguous buffer view")
            return binary[start : start + length]

        def array(index, kind, width, maximum):
            if type(index) is not int or not 0 <= index < 4:
                raise ValueError("Invalid native accessor index")
            a = doc["accessors"][index]
            dt = (
                {5123: "<u2", 5125: "<u4"}.get(a["componentType"])
                if width == 1
                else ("<f4" if a["componentType"] == 5126 else None)
            )
            count, offset = a["count"], a.get("byteOffset", 0)
            buf = view(a["bufferView"])
            if (
                not dt
                or a["type"] != kind
                or "sparse" in a
                or a.get("normalized")
                or not 1 <= count <= maximum
                or offset < 0
                or offset + count * width * np.dtype(dt).itemsize > len(buf)
            ):
                raise ValueError("Invalid or oversized shape accessor")
            return np.frombuffer(
                buf, dtype=dt, count=count * width, offset=offset
            ).reshape(-1, width)

        v = array(attr["POSITION"], "VEC3", 3, 350000)
        normal = array(attr["NORMAL"], "VEC3", 3, 350000)
        uv = array(attr["TEXCOORD_0"], "VEC2", 2, 350000)
        faces = array(prim["indices"], "SCALAR", 1, 1500000).reshape(-1, 3)
        atlas = _image(view(images[0]["bufferView"]), 1024**2)
        if (
            atlas.mode != "RGB"
            or atlas.width != atlas.height
            or atlas.width not in (256, 512, 1024)
            or v.shape != normal.shape
            or len(uv) != len(v)
            or faces.max() >= len(v)
            or not np.isfinite(v).all()
            or np.max(np.abs(v)) > 1e4
            or np.ptp(v, axis=0).max() < 1e-6
            or not np.isfinite(normal).all()
            or np.max(np.abs(np.linalg.norm(normal, axis=1) - 1)) > 0.02
            or not np.all((uv >= 0) & (uv <= 1))
        ):
            raise ValueError("Invalid atlas, geometry, normals or UVs")
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=atlas,
            name=materials[0].get("name"),
            metallicFactor=pbr.get("metallicFactor", 1.0),
            roughnessFactor=pbr.get("roughnessFactor", 1.0),
            doubleSided=materials[0].get("doubleSided", False),
        )
        uv = uv.copy()
        uv[:, 1] = 1 - uv[:, 1]
        mesh = trimesh.Trimesh(
            vertices=v,
            vertex_normals=normal,
            faces=faces,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
        )
        return mesh, doc, binary, _sha(data)
    except (IndexError, KeyError, TypeError, OverflowError, struct.error) as error:
        raise ValueError("Malformed native shape-painter GLB") from error


def _raster_bounded(projected, faces, side):
    # A small malformed mesh with full-screen triangles could otherwise monopolize a CPU.
    budget = 0
    for start in range(0, len(faces), 32768):
        points = projected[faces[start : start + 32768], :2].clip(-1, side)
        bounds = np.ptp(points, axis=1) + 2
        budget += np.prod(bounds, axis=1).sum()
    if budget > 100_000_000:
        raise ValueError("Projected triangle overlap exceeds raster work budget")
    return _raster(projected, faces, side)


def _view(mesh, angles):
    basis = _basis(*angles, "+Y")
    p = mesh.vertices @ basis.T
    center = (p.max(0) + p.min(0)) * 0.5
    span = np.ptp(p[:, :2], axis=0).max()
    if span < 1e-9:
        raise ValueError("Degenerate guide projection")
    scale = SIZE * 0.78 / span
    p[:, :2] = (p[:, :2] - center[:2]) * [scale, -scale] + SIZE / 2
    ids, bary, depth = _raster_bounded(p, np.asarray(mesh.faces, dtype=np.int32), SIZE)
    return basis, center, scale, ids, bary, depth


def _guide_sheet(mesh):
    sheet = Image.new("RGB", (2 * SIZE, SIZE), "white")
    light = _norm(np.array([-0.40, 0.66, 1.0]))
    for i, angles in enumerate(ANGLES.values()):
        b, _, _, ids, bary, _ = _view(mesh, angles)
        y, x = np.nonzero(ids >= 0)
        weights = np.column_stack((1 - bary[y, x].sum(1), bary[y, x]))
        normal = (
            _norm(
                np.einsum(
                    "ij,ijk->ik", weights, mesh.vertex_normals[mesh.faces[ids[y, x]]]
                )
            )
            @ b.T
        )
        image = np.full((SIZE, SIZE, 3), 255, np.uint8)
        image[y, x] = np.rint(134 + 0.72 * 64 * np.maximum(0, normal @ light))[
            :, None
        ].astype(np.uint8)
        sheet.paste(Image.fromarray(image), (i * SIZE, 0))
    return sheet


def prepare_guides(model, output_dir):
    """Render two off-axis hidden-side clay views from the sealed display-oriented model."""
    mesh, _, _, digest = _model(model)
    with _stage(output_dir) as dst:
        sheet = _guide_sheet(mesh)
        sheet.save(dst / "guides.png")
        info = {
            "version": VERSION,
            "model_sha256": digest,
            "guide_sha256": _sha((dst / "guides.png").read_bytes()),
            "views": {name: list(angles) for name, angles in ANGLES.items()},
            "size": SIZE,
        }
        _json(dst / "guide.json", info)
    return info


def _iou(a, b):
    return float(np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b)))


def _hole_agreement(a, b):
    """Each meaningful enclosed opening must overlap in both directions after registration."""

    def interiors(mask):
        return ndimage.binary_fill_holes(mask) & ~mask

    ha, hb = interiors(a), interiors(b)
    threshold = max(12, round(0.00012 * a.sum()))

    def recall(holes, other):
        labels, n = ndimage.label(holes)
        area = np.bincount(labels.ravel(), minlength=n + 1)[1:]
        matched = np.bincount(labels[other], minlength=n + 1)[1:]
        large = area >= threshold
        return float(np.min(matched[large] / area[large])) if large.any() else 1.0

    return min(recall(ha, hb), recall(hb, ha))


def _register_panel(panel, obj):
    gen = np.asarray(panel.resize((SIZE, SIZE), Image.Resampling.LANCZOS))
    pigment = np.min(gen, axis=2) < 222
    # Ignore tiny isolated background specks, keep disconnected solid parts and holes.
    labels, count = ndimage.label(pigment)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= max(8, obj.sum() * 0.00008)
    keep[0] = False
    mask = keep[labels]

    def box(mask):
        y, x = np.nonzero(mask)
        if not len(x) or min(np.ptp(x), np.ptp(y)) < 8:
            raise ValueError("Missing foreground in inferred panel")
        return np.array([x.min(), y.min()]), np.array([np.ptp(x), np.ptp(y)])

    p, span = box(mask)
    q, expected = box(obj)
    ratio = expected / span
    raw = _iou(mask, obj)
    source_center = p + span / 2
    guide_center = q + expected / 2
    # A misplaced narrow subject can have little overlap despite exactly the
    # same outline. Only correct small framing errors on fully visible panels.
    clipped = np.any(p < 2) or np.any(p + span >= SIZE - 2)
    if (np.max(np.abs(ratio - 1)) > 0.12
            or np.max(np.abs(guide_center - source_center)) > 0.09 * SIZE
            or clipped):
        raise ValueError("Inferred panel has the wrong position, scale or silhouette")
    scale = float(np.dot(span, expected) / np.dot(span, span))
    matrix = np.column_stack((np.eye(2) * scale, guide_center - scale * source_center))
    rgb = cv2.warpAffine(
        gen, matrix, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
    )
    aligned = cv2.warpAffine(
        mask.astype(np.uint8), matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST
    ).astype(bool)
    score = _iou(aligned, obj)
    holes = _hole_agreement(obj, aligned)
    if score < 0.90 or holes < 0.65:
        raise ValueError(
            f"Inferred view does not register: silhouette {score:.3f}, opening agreement {holes:.3f}"
        )
    near = ndimage.distance_transform_edt(
        ~aligned, return_distances=False, return_indices=True
    )
    color = rgb[tuple(near)].astype(np.float32)
    low = cv2.GaussianBlur(color, (0, 0), sigmaX=0.032 * max(expected))
    detail = np.clip((color + 8) / (low + 8), 0.36, 1.35)
    border = ndimage.distance_transform_edt(obj & aligned)
    return (
        detail,
        border,
        {
            "raw_iou": raw,
            "result_iou": score,
            "opening_agreement": holes,
            "fit": matrix.tolist(),
        },
    )


def _source_lock(mesh, mask, frame, atlas_ids, atlas_points):
    if not isinstance(frame, dict):
        raise ValueError("Fitted source camera is required")
    try:
        basis = np.array(
            [
                frame[k]
                for k in ("image_right", "image_up", "camera_direction_from_origin")
            ],
            float,
        )
    except (TypeError, KeyError) as error:
        raise ValueError("Incomplete fitted source camera") from error
    if (
        basis.shape != (3, 3)
        or not np.isfinite(basis).all()
        or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-5)
        or np.linalg.det(basis) < 0.999
    ):
        raise ValueError("Invalid fitted source-camera frame")
    # Reconstruct the raw AABB before fitting. The pinhole origin and inverse
    # distance were calibrated in this space, not the rotated display AABB.
    raw = mesh.vertices @ basis
    lens = lens_parameters(frame)
    source, scale, offset, _, _ = _fit(raw, basis, mask, lens)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    ids, _, depth = _raster_bounded(source, faces, len(mask))
    agreement = _iou(mask, ids >= 0)
    if agreement < 0.72:
        raise ValueError(
            f"Original foreground does not fit display mesh ({agreement:.3f})"
        )
    locked_faces = np.zeros(len(faces), bool)
    locked_faces[ids[ids >= 0]] = True
    for angles in ((0, 0), (8, 0), (-8, 0), (0, 8), (0, -8)):
        other = _view(mesh, angles)[3]
        locked_faces[other[other >= 0]] = True
    # Also protect small triangles that contributed color through adjacent-face
    # raster tolerance although their own face ID never won a pixel.
    points = atlas_points
    if lens is not None:
        from engine.shape_camera import project_pinhole

        center = (raw.max(0) + raw.min(0)) * 0.5 @ basis.T
        points = project_pinhole(points, center, lens[0] / np.ptp(raw, axis=0).max())
    xy = np.rint(points[:, :2] * [scale, -scale] + offset).astype(int)
    x, y = xy.clip(0, len(mask) - 1).T
    direct = (ids[y, x] >= 0) & (np.abs(depth[y, x] - points[:, 2]) <= 2.1 / scale)
    protected = np.zeros(atlas_ids.shape, bool)
    valid = atlas_ids >= 0
    protected[valid] = direct | locked_faces[atlas_ids[valid]]
    # UV seams/overlaps may put another triangle on top of a protected face.
    # Render the visible subset directly so both mappings are protected.
    side = len(atlas_ids)
    uvp = _uv_pixels(mesh, side)
    protected |= _raster_bounded(uvp, faces[locked_faces], side)[0] >= 0
    return ndimage.binary_dilation(protected, iterations=2), agreement


def _uv_pixels(mesh, side):
    uv = mesh.visual.uv
    return np.column_stack(
        (uv[:, 0] * side - 0.5, (1 - uv[:, 1]) * side - 0.5, np.zeros(len(uv)))
    )


def _registered_votes(mesh, sheet, points, normals):
    factors, weights, metrics = [], [], {}
    for i, (name, angles) in enumerate(ANGLES.items()):
        b, center, scale, ids, _, depth = _view(mesh, angles)
        panel = sheet.crop(
            (i * sheet.width // 2, 0, (i + 1) * sheet.width // 2, sheet.height)
        )
        detail, border, record = _register_panel(panel, ids >= 0)
        projected = points @ b.T
        xy = (projected[:, :2] - center[:2]) * [scale, -scale] + SIZE / 2
        x, y = np.rint(xy).astype(int).clip(0, SIZE - 1).T
        seen = (ids[y, x] >= 0) & (np.abs(depth[y, x] - projected[:, 2]) < 1.7 / scale)
        edge = np.clip((border[y, x] - 2) / 8, 0, 1)
        weight = seen * edge * np.clip((normals @ b[2] - 0.18) / 0.65, 0, 1)
        factors.append(_sample(detail, xy))
        weights.append(weight)
        metrics[name] = {**record, "effective_texels": int((weight > 0.5).sum())}
    score = np.stack(weights, 1)
    blend = score**8
    blend /= np.maximum(blend.sum(1, keepdims=True), 1e-9)
    modulation = np.einsum("ij,ijk->ik", blend, np.stack(factors, 1))
    strength = np.clip((score.max(1) - 0.1) / 0.68, 0, 1)
    color_gap = np.max(np.abs(factors[0] - factors[1]), axis=1)
    confidence_gap = np.abs(score[:, 0] - score[:, 1])
    strength *= 1 - 0.85 * np.clip((color_gap - 0.17) / 0.4, 0, 1) * np.clip(
        1 - confidence_gap / 0.22, 0, 1
    ) * (score.min(1) > 0.35)
    return modulation, strength, metrics


def _seal_glb(doc, binary, png, dest):
    # Keep every original accessor, primitive, node, normal, UV and image binding.
    # Only the embedded image buffer view and physical buffer offsets change.
    packed = bytearray()
    image_view = doc["images"][0]["bufferView"]
    nonimage = []
    for idx, view in enumerate(doc["bufferViews"]):
        off = view.get("byteOffset", 0)
        chunk = binary[off : off + view["byteLength"]]
        if idx == image_view:
            chunk = png
        else:
            nonimage.append(_sha(chunk))
        view.update(byteOffset=len(packed), byteLength=len(chunk))
        packed.extend(chunk)
        packed.extend(b"\0" * ((-len(packed)) % 4))
    doc["buffers"][0]["byteLength"] = len(packed)
    encoded = json.dumps(doc, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 4)
    with dest.open("wb") as stream:
        stream.write(
            struct.pack(
                "<4sIIII",
                b"glTF",
                2,
                28 + len(encoded) + len(packed),
                len(encoded),
                0x4E4F534A,
            )
        )
        stream.write(encoded)
        stream.write(struct.pack("<II", len(packed), 0x004E4942))
        stream.write(packed)
    return nonimage


def apply_guides(
    model, foreground, camera_frame, guide_dir, inferred_sheet, output_dir
):
    """Register both inferred panels and export a new coherent model, or reject entirely.

    model is the camera-aligned output of paint_shape. foreground is the same
    segmented RGBA used by that job. camera_frame is the registered raw-to-display
    frame including optional pinhole parameters. The caller binds these artifacts
    to its job and records the origin of the inferred sheet.
    """
    mesh, doc, binary, digest = _model(model)
    folder = Path(guide_dir)
    manifest = json.loads(_single(folder / "guide.json", 65536))
    guide_bytes = _single(folder / "guides.png", 8 * MIB)
    guide_sha = _sha(guide_bytes)
    if manifest != {
        "version": VERSION,
        "model_sha256": digest,
        "guide_sha256": guide_sha,
        "views": {k: list(v) for k, v in ANGLES.items()},
        "size": SIZE,
    }:
        raise ValueError("Guides do not match sealed model or camera layout")
    stored = _image(guide_bytes, 2 * SIZE**2)
    if (
        stored.mode != "RGB"
        or stored.size != (2 * SIZE, SIZE)
        or not np.array_equal(stored, _guide_sheet(mesh))
    ):
        raise ValueError("Stored guide pixels do not match sealed geometry")
    sheet_bytes = _single(inferred_sheet, 32 * MIB)
    sheet = _image(sheet_bytes, 12_000_000)
    if not (min(sheet.size) >= 256 and abs(sheet.width / sheet.height - 2) < 0.08):
        raise ValueError("Expected two adjacent square inferred views on white")
    if sheet.mode == "RGBA":
        backing = Image.new("RGB", sheet.size, "white")
        backing.paste(sheet, mask=sheet.getchannel("A"))
        sheet = backing
    else:
        sheet = sheet.convert("RGB")
    fg = _single(foreground, 32 * MIB)
    # Verify size/mode before _matte decodes the full foreground.
    with Image.open(io.BytesIO(fg)) as header:
        if (header.format != "PNG" or header.mode != "RGBA"
                or header.width * header.height > 40_000_000
                or getattr(header, "n_frames", 1) != 1):
            raise ValueError("Expected single-frame segmented RGBA PNG of at most 40 MP")
    matte = _matte(io.BytesIO(fg), SIZE)
    mask = matte[:, :, 3] >= 150
    before = np.asarray(mesh.visual.material.baseColorTexture)
    side = len(before)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    atlas_ids, bary, _ = _raster_bounded(_uv_pixels(mesh, side), faces, side)
    y, x = np.nonzero(atlas_ids >= 0)
    if len(y) == 0:
        raise ValueError("Empty UV chart")
    points = np.empty((len(y), 3))
    normals = np.empty_like(points)
    for start in range(0, len(y), 32768):
        step = slice(start, start + 32768)
        row, col = y[step], x[step]
        bw = bary[row, col]
        weights = np.column_stack((1 - bw.sum(1), bw))
        tri = faces[atlas_ids[row, col]]
        points[step] = np.einsum("ij,ijk->ik", weights, mesh.vertices[tri])
        normals[step] = _norm(
            np.einsum("ij,ijk->ik", weights, mesh.vertex_normals[tri])
        )
    locked, source_iou = _source_lock(mesh, mask, camera_frame, atlas_ids, points)
    modulation, strength, stats = _registered_votes(mesh, sheet, points, normals)
    strength[locked[y, x]] = 0
    atlas = before.copy()
    atlas[y, x] = np.rint(
        np.clip(before[y, x] * (1 + strength[:, None] * (modulation - 1)), 0, 255)
    ).astype(np.uint8)
    nearest = ndimage.distance_transform_edt(
        atlas_ids < 0, return_distances=False, return_indices=True
    )
    delta = atlas.astype(np.int16) - before.astype(np.int16)
    atlas = np.clip(before.astype(np.int16) + delta[tuple(nearest)], 0, 255).astype(
        np.uint8
    )
    atlas[locked] = before[locked]
    modified = np.any(atlas != before, axis=2)
    if not np.any(modified & (atlas_ids >= 0)):
        raise ValueError("No supported hidden texture detail in the inferred views")
    with _stage(output_dir) as dst:
        mesh.visual.material.baseColorTexture = Image.fromarray(atlas)
        export_finished(mesh, dst)
        png = (dst / "texture.png").read_bytes()
        if not np.array_equal(_image(png, 1024**2), atlas):
            raise ValueError("Master texture export lost color information")
        preserved = _seal_glb(doc, binary, png, dst / "model.glb")
        embed_master(dst / "model.glb", dst / "editable.gltf")
        with (dst / "model.glb").open("rb") as exported:
            output_sha = hashlib.file_digest(exported, "sha256").hexdigest()
        info = {
            "version": VERSION,
            "input_sha256": digest,
            "foreground_sha256": _sha(fg),
            "guide_sha256": guide_sha,
            "inferred_sha256": _sha(sheet_bytes),
            "output_sha256": output_sha,
            "source_silhouette_iou": source_iou,
            "views": stats,
            "nonimage_sha256": preserved,
            "locked_texels": int(locked.sum()),
            "locked_bytes_changed": int(np.count_nonzero(modified & locked)),
            "modified_surface_texels": int(
                np.count_nonzero(modified & (atlas_ids >= 0))
            ),
            "appearance": "needs_review",
            "observed_views": 1,
            "inferred_views": 2,
        }
        _json(dst / "guided-paint.json", info)
    return info
