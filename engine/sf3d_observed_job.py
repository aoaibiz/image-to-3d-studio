"""One source, native SF3D material, and separately authorized observed hidden paint."""

import argparse
import contextlib
import os
import shutil
import tempfile
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


def _publish(stage, out, names, source_sha):
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


def _native_finish(out, source, base, original, reg, observations, observation):
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
        _publish(stage, out, NATIVE_ARTIFACTS, hist["source_sha256"])
    return meta


def _validate_result(out, source_sha):
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
    if native:
        review = files.read_json(out / "native-material.json")
        if (
            gen["source_sha256"] != source_sha
            or gen["external_image_generations"] != 0
            or review["source_sha256"] != source_sha
            or review["model_sha256"] != hashes["model.glb"]
            or review["inspection_sha256"] != hashes["native-inspection.png"]
            or gen["shape"]["output_sha256"] != hashes["model.glb"]
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

    from engine.native_region_paint import paint_native_region

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
    # Validation/identity errors from the native kernel remain real failures.
    paint_native_region(*args)
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
        _publish(stage, out, required, original["input_sha256"])
    return meta


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


def run(source, out, *, setup, progress=None, cancelled=None):
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
                    out, source, base, orig, reg, observations, observation
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
