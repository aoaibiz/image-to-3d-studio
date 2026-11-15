"""One source, native SF3D material, and separately authorized observed hidden paint."""

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from engine import triposg_cpu as files
from engine.local_shape_job import StageError

PIPELINE = "sf3d-codex-observed-v1"
ARTIFACTS = {
    "model.glb",
    "editable.gltf",
    "shape-master.glb",
    "finished-native.glb",
    "input-processed.png",
    "generation.json",
    "guided-views.png",
    "guided-layout.json",
    "inferred-paint.png",
    "hidden-review.json",
    "region.json",
    "texture.png",
}


FULL_ARTIFACTS = (ARTIFACTS - {"region.json"}) | {"full-paint.json"}


class NeedsReview(ValueError):
    """A source-shaped model remains privately available, but this finish is withheld."""


def _consent(setup):
    from engine.guided_shape_job import _consent as check

    check(setup, PIPELINE)


def _checked_hash(path, expected):
    if not isinstance(expected, str) or files.digest(path) != expected:
        raise ValueError("保存した画像・形状・受付のチェックサムを確認できません。")


@contextlib.contextmanager
def _request(work, stage, journal, *, completed=None):
    """Only a provider's pre-submit journal can turn a local failure into a used attempt."""
    fence = work / (stage + "-attempt.json")
    if fence.exists():
        old = files.read_json(fence)
        if old.get("stage") != stage:
            raise ValueError("受付の工程が一致しません。")
        if (
            not journal.is_file()
            and not (completed and completed.is_file())
            and old.get("state") != "local_preflight"
        ):
            raise StageError(
                "sf3d_pending",
                "元の受付記録が失われました。重複送信を避けるため停止しました。",
            )
    if journal.is_file() and stage in {"observation", "paint"}:
        original = files.read_json(journal)
        # The underlying providers explicitly recorded that consent was revoked BEFORE launching.
        unused = (
            stage == "observation"
            and original.get("state") == "not_submitted_consent_revoked"
            or stage == "paint"
            and original.get("state") == "blocked_before_submission"
            and original.get("attempts") == 0
        )
        if unused:
            parent = journal.parent.resolve()
            if (
                not parent.is_relative_to(work.resolve())
                or parent == work.resolve()
                or journal.is_symlink()
            ):
                raise ValueError("想定外の受付パスです。")
            archive = Path(tempfile.mkdtemp(prefix="unsent-", dir=work))
            os.replace(parent, archive / parent.name)
    files.atomic_json(fence, {"stage": stage, "state": "entered"})
    try:
        yield
    except (ValueError, RuntimeError, OSError, TypeError, KeyError):
        # All three providers guarantee journal fsync BEFORE their first submission.
        # An existing malformed/partial journal is therefore always uncertain.
        if (
            not journal.exists()
            and not (completed and completed.exists())
            and (
                not journal.parent.exists()
                or all(p.name == "run.lock" for p in journal.parent.iterdir())
            )
        ):
            files.atomic_json(fence, {"stage": stage, "state": "local_preflight"})
        raise
    else:
        files.atomic_json(fence, {"stage": stage, "state": "completed"})


def _registration(native, matte, work):
    from engine.shape_camera import register_camera

    reg_path = work / "registration.json"
    if not reg_path.exists():
        files.atomic_json(reg_path, register_camera(native, matte))
    reg = files.read_json(reg_path)
    _checked_hash(native, reg["mesh_file_sha256"])
    _checked_hash(matte, reg["foreground_file_sha256"])
    if (
        reg["status"] == "poor-fit"
        or reg["camera_frame"]["camera_direction_from_origin"][2] >= 0
    ):
        raise NeedsReview(
            "元写真の正面と公式形状の向きを十分な確度で対応づけられませんでした。"
        )
    return reg


def _proxy(source, native, matte, work, reg):
    from engine.shape_appearance import paint_shape

    proxy = work / "proxy"
    if not proxy.exists():
        with tempfile.TemporaryDirectory(dir=work) as temp:
            staged = Path(temp) / "model"
            paint_shape(
                native,
                source,
                matte,
                staged,
                texture_size=1024,
                camera_frame=reg["camera_frame"],
                display_from_raw=reg["upright_transform"],
            )
            os.replace(staged, proxy)
    record = files.read_json(proxy / "appearance.json")
    for path, key in [
        (native, "shape_sha256"),
        (source, "original_sha256"),
        (matte, "foreground_sha256"),
        (proxy / "model.glb", "output_sha256"),
    ]:
        _checked_hash(path, record[key])
    if record["display_from_raw"] != reg["upright_transform"] or record["registration"][
        "basis_rows"
    ] != [
        reg["camera_frame"][k]
        for k in ("image_right", "image_up", "camera_direction_from_origin")
    ]:
        raise ValueError("中間ガイド形状の座標と登録済みカメラが一致しません。")
    return proxy


def _verify_proposal(receipt, sheet, layout, observations):
    from engine.visual_brief import json_sha

    expected = {
        "state": "completed",
        "recipient": "OpenAI",
        "api_key_route": False,
        "attempts": 1,
        "submission_limit": 1,
        "photo_sha256": layout["photo_sha"],
        "guide_sha256": layout["guide_sha"],
        "geometry_sha256": layout["model_sha"],
        "visual_brief_sha256": json_sha(observations),
        "inferred": True,
        "cameras": layout["camera"],
        "sheet_sha256": files.digest(sheet),
    }
    if (
        not receipt.get("thread_id")
        or any(receipt.get(k) != v for k, v in expected.items())
        or layout.get("visual_brief") != observations
    ):
        raise ValueError("写真・外観解析・形状ガイド・生成結果の受付を照合できません。")


NATIVE_ARTIFACTS = {
    "model.glb",
    "editable.gltf",
    "shape-master.glb",
    "input-processed.png",
    "generation.json",
    "native-inspection.png",
    "native-material.json",
}


def _appearance_mode(observations, submitted, *, full_supported=False):
    # A recovery may only repeat the local analysis of an existing provider attempt.
    if submitted:
        return "guided_region"
    context = observations.get("function_context", {})
    guess = context.get("interpretation", {})
    distinct = {"distinct_working_surface", "display_or_interface", "opening_or_cavity"}
    if distinct.intersection(guess.get("reverse_candidates", [])):
        if full_supported:
            return "inferred_paint"
        raise NeedsReview(
            "写真に写っていない異なる機能面が推定されます。公式モデルを保管しました。"
        )
    if "repeated" in observations["observed_face_cues"]:
        return "guided_region"
    if observations["observed_face_cues"]:
        if full_supported:
            return "inferred_paint"
        raise NeedsReview(
            "局所的な正面の意匠だけを確認しました。裏面へ広げる根拠がありません。"
        )

    unresolved = guess.get("reverse_candidates", ["unknown"]) != [
        "continuous_substrate"
    ]
    functional_parts = any(
        p["kind"] in {"parallel_wires", "fasteners", "screen_or_controls"}
        for p in observations["visible_structures"]
    )
    functional_edges = any(
        c["kind"]
        in {
            "hinge_or_pivot",
            "latch_or_fastener",
            "control_or_interface",
            "aperture_or_recess",
        }
        for c in context.get("observed_cues", [])
    )
    if unresolved and (functional_parts or functional_edges):
        raise NeedsReview("前面の機能部品に対する裏面の仕上げを判断できません。")
    if observations["visible_surround"]["pigment_layout"] not in {
        "uniform",
        "grain_or_mottle",
    }:
        raise NeedsReview("追加塗装の要否を判断できる表面情報がありません。")
    return "native_material"


def _render_native(mesh, texture, basis, mask, lens=None):
    """Render the delivered mesh's actual embedded diffuse and untextured normals on its camera rays."""
    import numpy as np

    from engine import guided_texture as gt

    p, *_ = gt._fit(mesh.vertices, basis, mask, lens)
    ids, bary, _ = gt._raster_bounded(p, np.asarray(mesh.faces, np.int32), len(mask))
    y, x = np.nonzero(ids >= 0)
    if len(y) < 256:
        raise NeedsReview("元モデルの投影面積が小さすぎます。")
    tri = mesh.faces[ids[y, x]]
    w = np.column_stack([1 - bary[y, x].sum(1), bary[y, x]])
    uv = np.einsum("ij,ijk->ik", w, mesh.visual.uv[tri])
    coords = np.column_stack(
        [uv[:, 0] * (texture.shape[1] - 1), (1 - uv[:, 1]) * (texture.shape[0] - 1)]
    )
    diffuse = np.full((*mask.shape, 3), 247, np.uint8)
    diffuse[y, x] = gt._sample(texture, coords).round().clip(0, 255).astype(np.uint8)
    normal = gt._norm(np.einsum("ij,ijk->ik", w, mesh.vertex_normals[tri])) @ basis.T
    clay = diffuse.copy()
    light = gt._norm(np.array([-0.35, 0.5, 0.78]))
    shade = 65 + 162 * np.maximum(0, normal @ light)
    clay[y, x] = np.clip(shade[:, None], 0, 255).astype(np.uint8)
    return diffuse, clay, ids


def _native_inspection(base, reg, observations, report_dir):
    import cv2
    import numpy as np
    from PIL import Image

    from engine import guided_texture as gt
    from engine import hidden_review
    from engine.sf3d_provider import native_glb
    from engine.shape_appearance import _load

    path = base / "model.glb"
    footprint = native_glb(path)
    meshes = _load(path)
    if len(meshes) != 1:
        raise NeedsReview("この公式メッシュ群の材質検査には対応していません。")
    mesh = meshes[0]
    material = mesh.visual.material
    tex = getattr(material, "baseColorTexture", None)
    if tex is None or mesh.visual.uv is None:
        raise NeedsReview("提供元の拡散色テクスチャが確認できません。")
    if (
        getattr(material, "alphaMode", None) not in (None, "OPAQUE")
        or getattr(material, "emissiveTexture", None) is not None
        or (
            getattr(material, "emissiveFactor", None) is not None
            and np.any(np.asarray(material.emissiveFactor) != 0)
        )
        or tex.convert("RGBA").getchannel("A").getextrema() != (255, 255)
    ):
        raise NeedsReview("透明・発光する材質は拡散色だけで評価できません。")
    pixels = np.array(tex.convert("RGB"))
    # Materials with emissive tint, a non-neutral factor or transparency need a full PBR comparison.
    factor = getattr(material, "baseColorFactor", None)
    if factor is not None and not np.all(np.asarray(factor) == 255):
        raise NeedsReview("色係数がついた材質には別途PBR検品が必要です。")
    size = 256
    photo = gt._matte(base / "input-processed.png", size)
    mask = photo[:, :, 3] >= 150
    front = reg["camera_frame"]
    raw_basis = np.array(
        [front[k] for k in ("image_right", "image_up", "camera_direction_from_origin")]
    )
    # SF3D standalone export applies an exact Y-axis half turn at the root.
    basis = raw_basis @ np.diag([-1.0, 1.0, -1.0])
    xyz = mesh.vertices @ basis.T
    span = float(np.ptp(xyz, axis=0).max())
    sections = []
    for lo, hi in ((0.16, 0.36), (0.4, 0.6), (0.64, 0.84)):
        low, high = np.quantile(xyz[:, 1], [lo, hi])
        band = xyz[(xyz[:, 1] >= low) & (xyz[:, 1] <= high)]
        left, right = np.quantile(band[:, 0], [0.2, 0.8])
        core = band[(band[:, 0] >= left) & (band[:, 0] <= right)]
        if len(core) >= 4 and span > 0:
            sections.append(float(np.ptp(np.quantile(core[:, 2], [0.05, 0.95])) / span))
    if not sections or max(sections) < 1e-5:
        raise NeedsReview("中央断面に測定可能な奥行きが見つかりません。")
    front_rgb, front_clay, front_ids = _render_native(
        mesh, pixels, basis, mask, gt.lens_parameters(front)
    )
    overlap = mask & (front_ids >= 0)
    iou = float((overlap.sum()) / (mask | (front_ids >= 0)).sum())
    interior = cv2.erode(overlap.astype("uint8"), np.ones((5, 5), np.uint8)).astype(
        bool
    )
    if iou < 0.72 or interior.sum() < 256:
        raise NeedsReview("写真の輪郭と表示GLBの投影が十分重なりません。")

    # Compare the spatially paired diffuse colour with the real photo, allowing for illumination.
    def lab(rgb):
        return cv2.cvtColor(
            cv2.GaussianBlur(rgb.astype("float32") / 255, (9, 9), 2), cv2.COLOR_RGB2Lab
        )[interior]

    src_lab, model_lab = lab(photo[:, :, :3]), lab(front_rgb)
    errors = np.abs(src_lab - model_lab)
    l = float(np.percentile(errors[:, 0], 75))
    chroma = np.linalg.norm(errors[:, 1:], axis=1)
    median = float(np.median(chroma))
    p90 = float(np.percentile(chroma, 90))
    colour_hold = l > 32 or median > 18 or p90 > 38
    src, src_mask = hidden_review._canonical(photo[:, :, :3], mask)
    images = [front_rgb]
    clays = [front_clay]
    frame = np.zeros(mask.shape, bool)
    frame[12:-12, 12:-12] = True
    angles = {
        "right": (90, 0),
        "back": (180, 0),
        "left": (-90, 0),
        "top": (0, 90),
        "bottom": (0, -90),
        **gt.ANGLES,
    }
    seen = np.unique(front_ids[front_ids >= 0])
    scores = {}
    for name, (yaw, pitch) in angles.items():
        axis = gt._basis(yaw, pitch, "+Y") @ basis
        rgb, clay, ids = _render_native(mesh, pixels, axis, frame)
        images.append(rgb)
        clays.append(clay)
        if name in gt.ANGLES:
            solid = ids >= 0
            target, inner = hidden_review._canonical(rgb, solid)
            hidden = solid & ~np.isin(ids, seen)
            hidden_field, _ = hidden_review._canonical(
                hidden.astype("uint8") * 255, solid
            )
            scores[name] = hidden_review._panel_score(
                src, src_mask, target, inner & (hidden_field >= 160)
            )
    status, hold, reason = hidden_review._diagnosis(list(scores.values()), observations)
    if colour_hold:
        hold, reason = True, "source_diffuse_disagreement"
    # Keep native material and clay diagnostics together, including in a withheld job.
    sheet = Image.fromarray(
        np.concatenate(
            [np.concatenate(images, axis=1), np.concatenate(clays, axis=1)], axis=0
        )
    )
    sheet.save(report_dir / "native-inspection.png")
    report = {
        "decision": "source_diffuse_disagreement"
        if colour_hold
        else "no additional pigment evidenced; inspect native material",
        "mode": "native_material",
        "source_sha256": observations["source_sha256"],
        "model_sha256": footprint["sha256"],
        "inspection_sha256": files.digest(report_dir / "native-inspection.png"),
        "visual_quality": "needs_review",
        "hidden_surface_observed": False,
        "render": "embedded diffuse and neutral geometric normals; no lighting reconstruction",
        "view_order": ["registered front", *angles],
        "image_rows": ["diffuse", "clay"],
        "silhouette_iou": iou,
        "lab_L_error_p75": l,
        "lab_chroma_error_median": median,
        "lab_chroma_error_p90": p90,
        "compared_pixels": int(interior.sum()),
        "hidden_layout_scores": scores,
        "hold_projection": hold,
        "orientation_status": status,
        "orientation_reason": reason,
        "shape": footprint,
        "central_depth_span_fractions": sections,
    }
    files.atomic_json(report_dir / "native-material.json", report)
    if colour_hold:
        raise NeedsReview(
            f"公式モデルと元写真の可視領域の色差が大きすぎます（明度75%点 {l:.1f}・色度90%点 {p90:.1f}）。"
        )
    if hold:
        raise NeedsReview(
            "公式モデルの背面に前面の特徴が重なっています。検品画像を保存しました。"
        )


@dataclass(frozen=True)
class ShapeEditor:
    """Internal adapter identity; the packaged method needs no user-supplied settings."""

    propose: Callable
    implementation_sha256: str
    allow_primary: bool = False


_RELEASED_SHAPE_EDITOR = object()


def _released_shape_editor():
    from engine import inferred_cavity

    return ShapeEditor(
        inferred_cavity.editor, files.digest(Path(inferred_cavity.__file__))
    )


SHAPE_FILES = {
    "unchanged": {"shape-hypothesis.json"},
    "alternative": {
        "shape-hypothesis.json",
        "shape-alternative.glb",
        "shape-alternative.gltf",
    },
    "applied": {"shape-hypothesis.json", "pre-hypothesis.glb"},
}


def _shape_candidate(before, after):
    """Accept only an appended interior and an ordered subset of the original triangles."""
    import numpy as np

    from engine.sf3d_provider import MAX_GLB
    from quality.inspection import load_glb

    if after.stat().st_size > MAX_GLB:
        raise ValueError("補完後のGLBが上限を超えています。")
    _, old, original = load_glb(before)
    raw, new, binary = load_glb(after)
    same_keys = {"buffers", "bufferViews", "accessors", "meshes", "materials"}
    if (
        old.keys() != new.keys()
        or any(old[k] != new[k] for k in old if k not in same_keys)
        or not binary.startswith(original)
        or len(new["buffers"]) != 1
        or set(new["buffers"][0]) != {"byteLength"}
        or not 0 <= len(binary) - new["buffers"][0]["byteLength"] <= 3
        or len(old["meshes"]) != len(new["meshes"])
        or len(new["meshes"]) != 1
    ):
        raise ValueError("元の塗装バッファーまたはシーンが変更されました。")
    extra_attrs = len(new["accessors"]) - len(old["accessors"])
    if extra_attrs not in (4, 5):
        raise ValueError("補完形状の属性構成が不明です。")
    for key, extra in (
        ("bufferViews", extra_attrs),
        ("accessors", extra_attrs),
        ("materials", 1),
    ):
        count = len(old[key])
        if len(new[key]) != count + extra or new[key][:count] != old[key]:
            raise ValueError("元の材質・法線・UVまたは頂点定義が変わりました。")
    old_mesh, new_mesh = old["meshes"][0], new["meshes"][0]
    if (
        any(new_mesh.get(k) != v for k, v in old_mesh.items() if k != "primitives")
        or len(old_mesh["primitives"]) != 1
        or len(new_mesh["primitives"]) != 2
    ):
        raise ValueError("編集したメッシュの構成が不明です。")
    first, interior = new_mesh["primitives"]
    previous = old_mesh["primitives"][0]
    n = len(old["accessors"])
    start = n + (extra_attrs == 5)
    attrs = {"POSITION": start + 1, "NORMAL": start + 2}
    if extra_attrs == 5:
        attrs["COLOR_0"] = n + 1
    if first != {**previous, "indices": n} or interior != {
        "attributes": attrs,
        "indices": start + 3,
        "material": len(old["materials"]),
        "mode": 4,
    }:
        raise ValueError("外面の材質または補完領域の構成が異なります。")
    mat = new["materials"][-1]
    pbr = mat.get("pbrMetallicRoughness", {})
    if set(mat) - {"name", "pbrMetallicRoughness"} or set(pbr) - {
        "baseColorFactor",
        "metallicFactor",
        "roughnessFactor",
    }:
        raise ValueError("想定外の補完材質です。")
    rgba = pbr.get("baseColorFactor", [1] * 4)
    coeff = [*rgba, pbr.get("metallicFactor", 1), pbr.get("roughnessFactor", 1)]
    if (
        len(rgba) != 4
        or rgba[3] != 1
        or any(type(x) not in (int, float) or not 0 <= x <= 1 for x in coeff)
    ):
        raise ValueError("補完材質の色または質感が不正です。")

    def array(doc, buf, acc, kind, width):
        a = doc["accessors"][acc]
        v = doc["bufferViews"][a["bufferView"]]
        types = {5123: "<u2", 5125: "<u4"} if width == 1 else {5126: "<f4"}
        dtype = np.dtype(types[a["componentType"]])
        count = a["count"]
        offset = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        if (
            a["type"] != kind
            or type(count) is not int
            or not 1 <= count <= 1_500_000
            or v["buffer"] != 0
            or v.get("byteStride", width * dtype.itemsize) != width * dtype.itemsize
            or offset < 0
            or offset + width * dtype.itemsize * count > len(buf)
            or a.get("byteOffset", 0) + width * dtype.itemsize * count > v["byteLength"]
        ):
            raise ValueError("補完メッシュのデータ範囲が不正です。")
        return np.frombuffer(buf, dtype, count=count * width, offset=offset).reshape(
            -1, width
        )

    positions = array(new, binary, start + 1, "VEC3", 3)
    normals = array(new, binary, start + 2, "VEC3", 3)
    if extra_attrs == 5:
        colours = array(new, binary, n + 1, "VEC3", 3)
        if (
            colours.shape != positions.shape
            or not np.isfinite(colours).all()
            or not np.all((0 <= colours) & (colours <= 1))
        ):
            raise ValueError("追加面の頂点色が不正です。")
    old_idx = array(old, original, previous["indices"], "SCALAR", 1).reshape(-1, 3)
    outer = array(old, original, previous["attributes"]["POSITION"], "VEC3", 3)
    tolerance = max(1e-6, float(np.ptp(outer, axis=0).max()) * 1e-4)
    kept = array(new, binary, n, "SCALAR", 1).reshape(-1, 3)
    added = array(new, binary, start + 3, "SCALAR", 1).reshape(-1, 3)
    if (
        not np.isfinite(positions).all()
        or normals.shape != positions.shape
        or not np.isfinite(normals).all()
        or np.any(np.abs(np.linalg.norm(normals, axis=1) - 1) > 0.03)
        or added.max() >= len(positions)
        or np.any(positions < outer.min(axis=0) - tolerance)
        or np.any(positions > outer.max(axis=0) + tolerance)
        or len(kept) >= len(old_idx)
    ):
        raise ValueError("補完した頂点・面が不正です。")
    i = 0
    for tri in old_idx:
        if i < len(kept) and np.array_equal(tri, kept[i]):
            i += 1
    if i != len(kept):
        raise ValueError("元の外面三角形が一致しません。")
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "faces": len(kept) + len(added),
        "embedded_images": len(new["images"]),
        "retained_faces": len(kept),
        "removed_faces": len(old_idx) - len(kept),
        "new_faces": len(added),
    }


def _finish_hypothesis(stage, out, source, editor):
    """Stage an opt-in proposal after painting and before atomic publication of settings."""
    from engine.finish_export import embed_master
    from engine.sf3d_provider import native_glb
    from engine.visual_brief import json_sha

    if not isinstance(editor, ShapeEditor) or not re.fullmatch(
        "[0-9a-f]{64}", editor.implementation_sha256
    ):
        raise ValueError("形状補完の実装バージョンが不明です。")
    meta = _validate_result(stage, files.digest(source))
    if meta.get("shape_hypothesis") is not None:
        raise ValueError("補完工程は同じ塗装に繰り返し適用できません。")
    hist = files.read_json(stage / "generation.json")
    before = stage / "model.glb"
    native = native_glb(before)
    observer = hist["photo_observation"]
    raw = out / "sf3d-observed-work/appearance/photo-observer/last-message.json"
    seal = files.read_json(raw.parent.parent / "observation-seal.json")
    source_sha = files.digest(source)
    if (
        source_sha != observer["source_sha256"]
        or source_sha != meta["input_sha256"]
        or observer["state"] != "completed"
        or observer["unseen_surface_observed"] is not False
        or seal.get("receipt") != observer
        or seal.get("visual_brief", {}).get("source_sha256") != source_sha
        or hist["registration"]["foreground_file_sha256"]
        != meta["artifact_sha256"]["input-processed.png"]
        or hist["registration"]["mesh_file_sha256"]
        != meta["artifact_sha256"]["shape-master.glb"]
        or raw.stat().st_size > 49152
    ):
        raise ValueError("形状候補の元写真と観察受付が一致しません。")
    _checked_hash(raw, observer["output_sha256"])
    raw_observation = files.read_json(raw)
    canonical_observation = hashlib.sha256(
        json.dumps(
            raw_observation, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    _checked_hash(before, meta["glb_sha256"])
    original = meta["artifact_sha256"]
    with tempfile.TemporaryDirectory(dir=stage) as temp:
        candidate = Path(temp) / "candidate.glb"
        registered = {
            **hist["registration"],
            "shape_hypothesis_inputs": {
                "source_sha256": source_sha,
                "foreground_sha256": original["input-processed.png"],
                "geometry_master_sha256": original["shape-master.glb"],
                "display_model_sha256": native["sha256"],
                "observer_output_sha256": observer["output_sha256"],
                "observation_canonical_sha256": canonical_observation,
                "display_rotation_xyzw": hist["camera"]["display_rotation_xyzw"],
            },
        }
        result = editor.propose(
            source=source,
            foreground=stage / "input-processed.png",
            observation=raw_observation,
            registration=registered,
            display_model=before,
            output=candidate,
        )
        if not isinstance(result, dict) or result.get("decision") not in SHAPE_FILES:
            raise ValueError("形状候補の判定を検証できません。")
        for name, sha in original.items():
            _checked_hash(stage / name, sha)
        _checked_hash(source, source_sha)
        _checked_hash(raw, observer["output_sha256"])
        mode = result["decision"]
        evidence = result.get("evidence", {})
        dimensions = result.get("inferred_dimensions", {})
        if (
            not isinstance(result.get("method"), str)
            or not result["method"]
            or not isinstance(result.get("reason"), str)
            or not isinstance(evidence, dict)
            or not isinstance(dimensions, dict)
            or any(
                not isinstance(k, str)
                or type(v) not in (int, float)
                or not math.isfinite(v)
                or v <= 0
                for k, v in dimensions.items()
            )
        ):
            raise ValueError("形状補完の根拠または推定寸法が不正です。")
        candidate_info = None
        decision_reason = result["reason"]
        if mode != "unchanged":
            if not dimensions:
                raise ValueError("候補の推定寸法がありません。")
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError("補完候補のGLBがありません。")
            candidate_info = _shape_candidate(before, candidate)
            qualified = (
                evidence.get("opening") in {"observed", "plausible"}
                and evidence.get("visible_closure") is False
                and evidence.get("painted_cap") is False
                and evidence.get("negative_controls_passed") is True
                and evidence.get("source_registered") is True
                and evidence.get("geometry_supported") is True
                and bool(dimensions)
            )
            prohibited = (
                evidence.get("visible_closure") is True
                or evidence.get("painted_cap") is True
            )
            if mode == "applied" and not (qualified and editor.allow_primary is True):
                mode = "alternative"
            alternative_supported = (
                evidence.get("opening") in {"observed", "plausible"}
                and evidence.get("negative_controls_passed") is True
                and evidence.get("geometry_supported") is True
                and evidence.get("source_registered") is True
            )
            if prohibited or (mode == "alternative" and not alternative_supported):
                mode = "unchanged"
                decision_reason = (
                    "visible closure or paint"
                    if prohibited
                    else "insufficient registered evidence for an inferred alternative"
                )
            if mode == "applied":
                os.replace(before, stage / "pre-hypothesis.glb")
                os.replace(candidate, before)
                embed_master(before, stage / "editable.gltf")
            elif mode == "alternative":
                os.replace(candidate, stage / "shape-alternative.glb")
                embed_master(
                    stage / "shape-alternative.glb", stage / "shape-alternative.gltf"
                )
    end = candidate_info if mode == "applied" else native
    _checked_hash(before, end["sha256"])
    report = {
        "version": 1,
        "method": result["method"],
        "decision": mode,
        "reason": decision_reason,
        "evidence": evidence,
        "inferred_dimensions": dimensions,
        "dimensions_observed": False,
        "visual_quality": "needs_review",
        "source_sha256": source_sha,
        "foreground_sha256": original["input-processed.png"],
        "observer_output_sha256": observer["output_sha256"],
        "observations_sha256": json_sha(seal["visual_brief"]),
        "observer_request_id": observer["request_id"],
        "source_camera_sha256": json_sha(hist["registration"]),
        "implementation_sha256": editor.implementation_sha256,
        "painted_before": native,
        "candidate": candidate_info,
        "delivered": end,
    }
    files.atomic_json(stage / "shape-hypothesis.json", report)
    hist.update(shape_hypothesis=report, display_glb_sha256=end["sha256"])
    files.atomic_json(stage / "generation.json", hist)
    names = set(original) | SHAPE_FILES[mode]
    meta.update(
        shape_hypothesis={
            "decision": mode,
            "report": "shape-hypothesis.json",
            **(
                {
                    "alternative_model": "shape-alternative.glb",
                    "alternative_editable": "shape-alternative.gltf",
                }
                if mode == "alternative"
                else {"paint_master": "pre-hypothesis.glb"}
                if mode == "applied"
                else {}
            ),
        },
        triangles=end["faces"],
        embedded_images=end["embedded_images"],
        glb_sha256=end["sha256"],
        artifact_sha256={n: files.digest(stage / n) for n in sorted(names)},
    )
    files.atomic_json(stage / "settings.json", meta)
    return meta


def _publish(stage, out, names, source_sha, *, source=None, shape_editor=None):
    if shape_editor is _RELEASED_SHAPE_EDITOR:
        shape_editor = _released_shape_editor()
    if shape_editor is not None:
        names = _finish_hypothesis(stage, out, source, shape_editor)["artifact_sha256"]
    _validate_result(stage, source_sha)
    # A held preview is a separate product state. Preserve its previous receipt privately
    # when a later, explicitly resumed run succeeds, including its exact original preview.
    if (out / "review-hold.json").is_file():
        work = out / "sf3d-observed-work"
        archive = Path(tempfile.mkdtemp(prefix="previous-hold-", dir=work))
        for name in ("review-hold.json", "review-native.glb", "review-native.png"):
            if (out / name).is_file():
                os.replace(out / name, archive / name)
    for name in sorted(names):
        os.replace(stage / name, out / name)
    os.replace(stage / "settings.json", out / "settings.json")


def _native_finish(
    out, source, base, original, reg, observations, observation, shape_editor=None
):
    work = out / "sf3d-observed-work"
    _native_inspection(base, reg, observations, work)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        stage = Path(tmp)
        for name in ("model.glb", "editable.gltf", "input-processed.png"):
            shutil.copyfile(base / name, stage / name)
        shutil.copyfile(base / "provider-native.glb", stage / "shape-master.glb")
        for name in ("native-inspection.png", "native-material.json"):
            shutil.copyfile(work / name, stage / name)
        hist = {
            "source_sha256": files.digest(source),
            "camera": original["camera"],
            "shape": files.read_json(base / "generation.json"),
            "registration": reg,
            "photo_observation": observation,
            "visual_brief": observations,
            "visual_quality": "needs_review",
            "external_shape_generations": 1,
            "external_photo_observations": 1,
            "external_image_generations": 0,
            "unseen_surface_observed": False,
            "display_glb_sha256": original["glb_sha256"],
        }
        files.atomic_json(stage / "generation.json", hist)
        meta = {
            **original,
            "pipeline": PIPELINE,
            "appearance_mode": "native_material",
            "engine": "Stable Fast 3D + ChatGPT observed native material",
            "geometry_master": "shape-master.glb",
            "visual_quality": "needs_review",
            "texture_size": 1024,
            "photo_observer": observation["method"],
            "artifact_sha256": {
                n: files.digest(stage / n) for n in sorted(NATIVE_ARTIFACTS)
            },
        }
        files.atomic_json(stage / "settings.json", meta)
        _publish(
            stage,
            out,
            NATIVE_ARTIFACTS,
            hist["source_sha256"],
            source=source,
            shape_editor=shape_editor,
        )
    return files.read_json(out / "settings.json")


def _validate_result(out, source_sha):
    from engine.visual_brief import json_sha

    meta = files.read_json(out / "settings.json")
    hashes = meta["artifact_sha256"]
    native = meta.get("appearance_mode") == "native_material"
    modes = {
        "native_material": NATIVE_ARTIFACTS,
        "guided_region": ARTIFACTS,
        "guided_full": FULL_ARTIFACTS,
    }
    if meta.get("appearance_mode") not in modes:
        raise ValueError("不明な仕上げ方式です。")
    required = modes[meta["appearance_mode"]]
    hypothesis = meta.get("shape_hypothesis")
    if hypothesis is not None:
        if (
            not isinstance(hypothesis, dict)
            or hypothesis.get("decision") not in SHAPE_FILES
            or hypothesis.get("report") != "shape-hypothesis.json"
        ):
            raise ValueError("形状補完の判定を照合できません。")
        required = required | SHAPE_FILES[hypothesis["decision"]]
    if (
        set(hashes) != required
        or meta["pipeline"] != PIPELINE
        or meta["input_sha256"] != source_sha
        or meta["auto_finish_complete"] is not True
        or meta["visual_quality"] != "needs_review"
        or meta["glb_sha256"] != hashes["model.glb"]
    ):
        raise ValueError("保存した複合ジョブの成果物を照合できません。")
    for name in required:
        _checked_hash(out / name, hashes[name])
    gen = files.read_json(out / "generation.json")
    paint_sha = hashes["model.glb"]
    if hypothesis is not None:
        report = files.read_json(out / "shape-hypothesis.json")
        mode = hypothesis["decision"]
        paint_sha = (
            hashes["pre-hypothesis.glb"] if mode == "applied" else hashes["model.glb"]
        )
        if (
            report != gen.get("shape_hypothesis")
            or report["decision"] != mode
            or report["painted_before"]["sha256"] != paint_sha
            or report["delivered"]["sha256"] != meta["glb_sha256"]
            or meta["triangles"] != report["delivered"]["faces"]
            or gen["display_glb_sha256"] != meta["glb_sha256"]
            or report["dimensions_observed"] is not False
            or report["source_sha256"] != source_sha
            or report["foreground_sha256"] != hashes["input-processed.png"]
            or report["source_camera_sha256"] != json_sha(gen["registration"])
            or report["observer_request_id"] != gen["photo_observation"]["request_id"]
            or report["observer_output_sha256"]
            != gen["photo_observation"]["output_sha256"]
            or mode == "alternative"
            and report["candidate"]["sha256"] != hashes["shape-alternative.glb"]
            or mode == "applied"
            and report["candidate"] != report["delivered"]
        ):
            raise ValueError("形状補完の写真・観察・編集前後の成果物が一致しません。")
    if native:
        review = files.read_json(out / "native-material.json")
        if (
            gen["source_sha256"] != source_sha
            or gen["external_image_generations"] != 0
            or review["source_sha256"] != source_sha
            or review["model_sha256"] != paint_sha
            or review["inspection_sha256"] != hashes["native-inspection.png"]
            or gen["shape"]["output_sha256"] != paint_sha
            or gen["shape"]["native_sha256"] != hashes["shape-master.glb"]
            or gen["shape"]["camera"] != meta["camera"]
            or review["hold_projection"] is not False
        ):
            raise ValueError("元の形状・観察・無追加塗装の検品証跡が一致しません。")
        return meta
    record_name = (
        "full-paint.json" if meta["appearance_mode"] == "guided_full" else "region.json"
    )
    region = files.read_json(out / record_name)
    if (
        gen["source_sha256"] != source_sha
        or gen["display_glb_sha256"] != meta["glb_sha256"]
        or gen["camera"] != meta["camera"]
        or region["source_glb"] != hashes["shape-master.glb"]
        or region["output_glb"] != hashes["finished-native.glb"]
        or region["source_image_sha256"] != source_sha
        or region["locked_changed"] != 0
        or region["unsupported_surface_changed"] != 0
        or region["normals_uv_nodes_material_and_all_other_buffers_bitwise_identical"]
        is not True
    ):
        raise ValueError(
            "元の公式材質・編集マスター・表示用GLBの対応を照合できません。"
        )
    return meta


def valid(out, settings):
    try:
        return _validate_result(Path(out), settings["input_sha256"]) == settings
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _synthesize(*args):
    from engine.native_full_paint import (
        NativeFullRegistrationError,
        NativeFullReviewHold,
        paint_native_full,
        preflight_native_paint,
    )
    from engine.native_region_paint import NativeRegionQualityHold, paint_native_region

    try:
        paint_native_full(*args)
        return "guided_full", "full-paint.json", None
    except NativeFullReviewHold as exc:
        raise NeedsReview(str(exc)) from exc
    except NativeFullRegistrationError as exc:
        full_reason = str(exc)
    native, _, _, _, registration, _, _, layout, *_ = args
    observation = layout["visual_brief"]
    guesses = (
        observation.get("function_context", {})
        .get("interpretation", {})
        .get("reverse_candidates", [])
    )
    if (
        "repeated" not in observation["observed_face_cues"]
        or {
            "distinct_working_surface",
            "display_or_interface",
            "opening_or_cavity",
        }.intersection(guesses)
        or not preflight_native_paint(native, registration)["region_depth_eligible"]
    ):
        raise NeedsReview("2方向の全体形状が合わず、部分投影の根拠も不足しています。")
    try:
        paint_native_region(*args)
    except NativeRegionQualityHold as exc:
        raise NeedsReview(str(exc)) from exc
    return "guided_region", "region.json", full_reason


def _export(
    out,
    source,
    base,
    original,
    proxy,
    reg,
    guides,
    layout,
    observation,
    proposal,
    orientation,
    shape_editor=None,
):
    from engine.finish_export import embed_master
    from engine.sf3d_provider import native_glb
    from engine.sf3d_shape_job import oriented_glb

    native = base / "provider-native.glb"
    matte = base / "input-processed.png"
    sheet = Path(proposal["sheet"])
    receipt = proposal["provenance"]
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        stage = Path(tmp)
        region_out = stage / "native"
        mode, record_name, full_reason = _synthesize(
            native,
            proxy / "model.glb",
            source,
            matte,
            reg,
            guides,
            sheet,
            layout,
            receipt,
            orientation,
            region_out,
        )
        region = files.read_json(region_out / record_name)
        required = FULL_ARTIFACTS if mode == "guided_full" else ARTIFACTS
        finished = region_out / "model.glb"
        if (
            region["source_glb"] != files.digest(native)
            or region["output_glb"] != files.digest(finished)
            or region["source_image_sha256"] != original["input_sha256"]
            or region["locked_changed"] != 0
            or region["unsupported_surface_changed"] != 0
            or region["changed_uv"] <= 0
            or region[
                "normals_uv_nodes_material_and_all_other_buffers_bitwise_identical"
            ]
            is not True
        ):
            raise ValueError("元の形状・材質と観測済み表面の保護を確認できません。")
        oriented_glb(finished, stage / "model.glb")
        artifact = native_glb(stage / "model.glb")
        if (
            artifact["faces"] != original["triangles"]
            or artifact["embedded_images"] != original["embedded_images"]
        ):
            raise ValueError("公式GLBと表示用GLBの構成が異なります。")
        embed_master(stage / "model.glb", stage / "editable.gltf")
        for p, name in [
            (native, "shape-master.glb"),
            (finished, "finished-native.glb"),
            (matte, "input-processed.png"),
            (region_out / "texture.png", "texture.png"),
            (region_out / record_name, record_name),
            (guides / "guides.png", "guided-views.png"),
            (guides / "request-guide.json", "guided-layout.json"),
            (sheet, "inferred-paint.png"),
        ]:
            shutil.copyfile(p, stage / name)
        files.atomic_json(stage / "hidden-review.json", orientation)
        gen = {
            "source_sha256": original["input_sha256"],
            "shape": files.read_json(base / "generation.json"),
            "camera": original["camera"],
            "registration": reg,
            "photo_observation": observation,
            "paint_application": mode,
            "full_sheet_registration_failure": full_reason,
            "paint": {
                k: receipt[k]
                for k in (
                    "provider",
                    "thread_id",
                    "photo_sha256",
                    "guide_sha256",
                    "geometry_sha256",
                    "sheet_sha256",
                    "visual_brief_sha256",
                    "prompt_sha256",
                    "attempts",
                    "recipient",
                )
            },
            "display_glb_sha256": artifact["sha256"],
            "native_reverse_sha256": region["output_glb"],
            "unseen_surface_observed": False,
            "visual_quality": "needs_review",
            "external_shape_generations": 1,
            "external_photo_observations": 1,
            "external_image_generations": 1,
        }
        files.atomic_json(stage / "generation.json", gen)
        meta = {
            **original,
            "pipeline": PIPELINE,
            "appearance_mode": mode,
            "engine": "Stable Fast 3D + ChatGPT observed native paint",
            "glb_sha256": artifact["sha256"],
            "geometry_master": "shape-master.glb",
            "editable_master": "editable.gltf",
            "texture_size": 1024,
            "visual_quality": "needs_review",
            "photo_observer": observation["method"],
            "artifact_sha256": {n: files.digest(stage / n) for n in sorted(required)},
        }
        files.atomic_json(stage / "settings.json", meta)
        _publish(
            stage,
            out,
            required,
            original["input_sha256"],
            source=source,
            shape_editor=shape_editor,
        )
    return files.read_json(out / "settings.json")


HOLD_REASONS = {
    "source_fit": "元写真の向きと公式モデルの対応に確認が必要です。",
    "surface_uncertain": "写真に写っていない面の材質または機能を判断できません。",
    "visible_material": "公式モデルの形状・可視材質・背面の検品に確認が必要です。",
    "rear_layout": "推定した背面に正面の特徴が重なっている可能性があります。",
    "native_projection": "推定塗装を元の形状・材質に安全に反映できませんでした。",
}


def _save_hold(out, base, key, stage):
    from engine.sf3d_shape_job import valid as native_valid

    original = files.read_json(base / "settings.json")
    if original.get("input_sha256") != key["input_sha256"] or not native_valid(
        base, original
    ):
        raise ValueError("保留モデルの元写真と公式データを検証できません。")
    codes = {
        "registration": "source_fit",
        "native_inspection": "visible_material",
        "orientation": "rear_layout",
        "projection": "native_projection",
    }
    code = codes.get(stage, "surface_uncertain")
    reason = HOLD_REASONS[code]
    preview = base / "model.glb"
    marker = {
        "version": 1,
        **key,
        "shape_sha256": original["artifact_sha256"]["provider-native.glb"],
        "preview_sha256": original["glb_sha256"],
        "reason_code": code,
        "reason": reason,
        "stage": stage,
    }
    report_dir = out / "sf3d-observed-work"
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        stage_dir = Path(tmp)
        shutil.copyfile(preview, stage_dir / "review-native.glb")
        _checked_hash(stage_dir / "review-native.glb", marker["preview_sha256"])
        if (
            stage == "native_inspection"
            and (report_dir / "native-material.json").exists()
        ):
            check = files.read_json(report_dir / "native-material.json")
            if (
                check.get("model_sha256") == marker["preview_sha256"]
                and check.get("source_sha256") == key["input_sha256"]
            ):
                shutil.copyfile(
                    report_dir / "native-inspection.png",
                    stage_dir / "review-native.png",
                )
                _checked_hash(
                    stage_dir / "review-native.png", check["inspection_sha256"]
                )
                marker["image_sha256"] = check["inspection_sha256"]
        files.atomic_json(stage_dir / "review-hold.json", marker)
        os.replace(stage_dir / "review-native.glb", out / "review-native.glb")
        if "image_sha256" in marker:
            os.replace(stage_dir / "review-native.png", out / "review-native.png")
        os.replace(stage_dir / "review-hold.json", out / "review-hold.json")


def _guard_downstream(out, work):
    """Surviving published or later-stage evidence forbids creating replacement provider jobs."""
    account = work / "appearance"
    private_shape = work / "native/sf3d-work/provider/provider.json"
    observation = account / "photo-observer/receipt.json"
    paint = account / "guided-paint/receipt.json"
    state = (
        files.read_json(work / "state.json") if (work / "state.json").is_file() else {}
    )
    # Output is exclusively reserved for this job, including Studio's later HTML, PNG and QA reports.
    published = any(p != work for p in out.iterdir())
    has_native = (work / "native/settings.json").is_file() or private_shape.is_file()
    post_native = (
        published
        or observation.exists()
        or paint.exists()
        or any(
            (work / n).exists()
            for n in (
                "observation-attempt.json",
                "registration.json",
                "proxy",
                "orientation",
                "native-material.json",
                "native-inspection.png",
            )
        )
        or state.get("stage")
        in {
            "observation",
            "registration",
            "native_inspection",
            "guides",
            "paint",
            "orientation",
            "projection",
            "export",
        }
    )
    if post_native and (not (work / "identity.json").is_file() or not has_native):
        raise StageError(
            "sf3d_pending",
            "既存の成果物に対応する公式形状の受付が失われました。再送を停止しました。",
        )
    post_observation = (
        published
        or any(
            (account / n).exists()
            for n in ("guides", "observation-seal.json", "guided-paint")
        )
        or any(
            (work / n).exists()
            for n in (
                "proxy",
                "orientation",
                "registration.json",
                "native-material.json",
            )
        )
        or state.get("stage")
        in {
            "registration",
            "native_inspection",
            "guides",
            "paint",
            "orientation",
            "projection",
            "export",
        }
    )
    if post_observation and not observation.is_file():
        raise StageError(
            "sf3d_pending", "元の写真観察の受付を確認できません。再送を停止しました。"
        )
    hold_path = out / "review-hold.json"
    hold = files.read_json(hold_path) if hold_path.is_file() else {}
    proven_no_paint = (
        (work / "native-material.json").is_file()
        or (out / "native-material.json").is_file()
        or (
            hold.get("stage") in {"registration", "observation", "native_inspection"}
            and not (work / "paint-attempt.json").exists()
        )
    )
    after_paint = (
        (work / "orientation").exists()
        or hold.get("stage") in {"orientation", "projection"}
        or (
            state.get("stage") in {"orientation", "projection"}
            or state.get("stage") == "export"
            and not proven_no_paint
        )
        or any(
            (out / n).exists()
            for n in (
                "finished-native.glb",
                "guided-views.png",
                "guided-layout.json",
                "inferred-paint.png",
                "hidden-review.json",
                "region.json",
                "full-paint.json",
                "texture.png",
            )
        )
    )
    if (after_paint or published and not proven_no_paint) and not paint.is_file():
        raise StageError(
            "sf3d_pending", "元の推定画像の受付を確認できません。再送を停止しました。"
        )


def run(
    source,
    out,
    *,
    setup,
    progress=None,
    cancelled=None,
    shape_editor=_RELEASED_SHAPE_EDITOR,
):
    from engine import guided_shape_job, sf3d_shape_job
    from engine.guided_provider import generate_guided_paint
    from engine.hidden_review import review_hidden_paint
    from engine.sf3d_provider import Pending

    source, out, setup = (Path(p).resolve() for p in (source, out, setup))
    work = out / "sf3d-observed-work"
    work.mkdir(parents=True, exist_ok=True)
    key = {"pipeline": PIPELINE, "input_sha256": files.digest(source)}
    base = work / "native"
    identity = work / "identity.json"
    state = work / "state.json"
    with files.writer_lock(work):
        if identity.exists() and files.read_json(identity) != key:
            raise ValueError("受付済みの元画像が異なります。")
        if (out / "settings.json").is_file():
            return _validate_result(out, key["input_sha256"])
        _guard_downstream(out, work)
        files.atomic_json(identity, key)
        stage = "consent"

        def enter(name):
            nonlocal stage
            stage = name
            if cancelled and cancelled():
                raise InterruptedError(
                    "待機を停止しました。既存の受付から再開できます。"
                )
            _checked_hash(source, key["input_sha256"])
            files.atomic_json(state, {**key, "stage": stage, "status": "running"})
            print("sf3d observed " + name, flush=True)
            if progress:
                progress(name)

        try:
            _consent(setup)
            enter("native")
            with _request(
                work,
                "shape",
                base / "sf3d-work/provider/provider.json",
                completed=base / "settings.json",
            ):
                orig = sf3d_shape_job.run(
                    source,
                    base,
                    setup=setup,
                    progress=progress,
                    cancelled=cancelled,
                    pipeline=PIPELINE,
                )
            if orig["input_sha256"] != key["input_sha256"]:
                raise ValueError("提供元の元画像ハッシュが一致しません。")
            enter("observation")
            # Keep the observer's own downstream markers separate from local native shape artifacts.
            account = work / "appearance"
            account.mkdir(exist_ok=True)
            with _request(work, "observation", account / "photo-observer/receipt.json"):
                observations, observation = guided_shape_job._chatgpt_observations(
                    source, account, key["input_sha256"], setup, pipeline=PIPELINE
                )
            native = base / "provider-native.glb"
            matte = base / "input-processed.png"
            enter("registration")
            reg = _registration(native, matte, work)
            fence = work / "paint-attempt.json"
            submitted = (account / "guided-paint/receipt.json").is_file() or (
                fence.is_file()
                and files.read_json(fence).get("state") != "local_preflight"
            )
            stage = "observation"
            mode = _appearance_mode(observations, submitted, full_supported=True)
            if mode == "native_material":
                enter("native_inspection")
                meta = _native_finish(
                    out,
                    source,
                    base,
                    orig,
                    reg,
                    observations,
                    observation,
                    shape_editor,
                )
                files.atomic_json(
                    state,
                    {
                        **key,
                        "stage": "export",
                        "status": "completed",
                        "visual_quality": "needs_review",
                    },
                )
                print("sf3d observed export", flush=True)
                if progress:
                    progress("export")
                return meta
            enter("guides")
            if not submitted:
                from engine.native_full_paint import preflight_native_paint

                preflight = preflight_native_paint(native, reg)
                if preflight["full_sheet_eligible"] is not True:
                    raise NeedsReview(
                        "元の公式材質で未観測面の推定塗装を検品できません。"
                    )
            proxy = _proxy(source, native, matte, work, reg)
            guides, layout = guided_shape_job._prepare(
                source,
                proxy,
                account,
                {
                    "glb_sha256": files.digest(proxy / "model.glb"),
                    "input_sha256": key["input_sha256"],
                },
                observations,
            )
            if layout.get("visual_brief") != observations:
                raise ValueError("受付した観察結果とガイドが一致しません。")
            enter("paint")
            _consent(setup)
            with _request(work, "paint", account / "guided-paint/receipt.json"):
                proposal = generate_guided_paint(
                    source,
                    guides / "guides.png",
                    guides / "request-guide.json",
                    account,
                    setup=setup,
                    pipeline=PIPELINE,
                )
            sheet = Path(proposal["sheet"])
            _verify_proposal(proposal["provenance"], sheet, layout, observations)
            enter("orientation")
            orientation = review_hidden_paint(
                proxy / "model.glb",
                source,
                base / "input-processed.png",
                sheet,
                observations,
                work / "orientation",
                camera_frame=reg["camera_frame"],
            )
            if orientation["hold_projection"]:
                raise NeedsReview(
                    "推定背面に正面の特徴が残りました。元モデルと画像を保管しました。"
                )
            enter("projection")
            meta = _export(
                out,
                source,
                base,
                orig,
                proxy,
                reg,
                guides,
                layout,
                observation,
                proposal,
                orientation,
                shape_editor,
            )
            print("sf3d observed export", flush=True)
            if progress:
                progress("export")
            files.atomic_json(
                state,
                {
                    **key,
                    "stage": "export",
                    "status": "completed",
                    "visual_quality": "needs_review",
                },
            )
            return meta
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            TypeError,
            ImportError,
        ) as exc:
            if isinstance(exc, NeedsReview):
                _save_hold(out, base, key, stage)
                status, code = "needs_review", "sf3d_observed_review"
            elif (
                isinstance(exc, (Pending, InterruptedError))
                or isinstance(exc, StageError)
                and exc.code == "sf3d_pending"
            ):
                status, code = "waiting", "sf3d_pending"
            else:
                status, code = "failed", "sf3d_observed_error"
            files.atomic_json(
                state, {**key, "stage": stage, "status": status, "reason": str(exc)}
            )
            raise StageError(code, str(exc)) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--setup", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.image, args.output, setup=args.setup)
    except (StageError, ValueError, OSError) as exc:
        files.emit(
            "studio_error",
            code=getattr(exc, "code", "sf3d_observed_error"),
            detail=str(exc),
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
