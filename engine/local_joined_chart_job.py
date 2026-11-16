"""Local TripoSG shape, measured pigment, and at most one sealed surface chart."""

import re
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from engine import guided_texture as gt
from engine import triposg_cpu as sg
from engine.visual_brief import json_sha

# Majority of writable hidden native UV samples, not percentage of a physical item.
MINIMUM_WRITABLE_COVERAGE = 0.5
METHOD = "local-shape-continuity-or-one-chart-v1"
DIAGNOSTIC = "diagnostic_source_model"
REASONS = {
    DIAGNOSTIC: {
        "registration_or_chart_unsupported",
        "proposed_paint_rejected",
        "provider_unavailable",
        "provider_unconfirmed",
    },
    "inferred_chart": {"one_inferred_chart"},
    "local_continuity": {"measured_continuity"},
}


def display_registration(base, source, original):
    """The shipped SG base already contains its rigid camera transform."""
    appearance = sg.read_json(base / "appearance.json")
    generation = sg.read_json(base / "generation.json")
    camera = generation["camera"]
    frame = camera["camera_frame"]
    axes = np.asarray(
        [frame[k] for k in ("image_right", "image_up", "camera_direction_from_origin")],
        float,
    )
    rot = np.asarray(camera["upright_transform"], float)
    baked = np.asarray(appearance["display_from_raw"], float)
    if (
        rot.shape != (4, 4)
        or baked.shape != (4, 4)
        or axes.shape != (3, 3)
        or not np.allclose(rot, baked, atol=1e-6, rtol=0)
        or not np.allclose(axes, rot[:3, :3], atol=1e-6, rtol=0)
        or not np.allclose(rot[3], [0, 0, 0, 1])
        or not np.allclose(rot[:3, 3], 0)
        or not np.allclose(axes @ axes.T, np.eye(3), atol=1e-5)
        or np.linalg.det(axes) < 0.999
        or appearance["output_sha256"] != sg.digest(base / "model.glb")
        or appearance["original_sha256"] != sg.digest(source)
        or original["glb_sha256"] != appearance["output_sha256"]
        or appearance["foreground_sha256"] != sg.digest(base / "input-processed.png")
        or appearance["shape_sha256"] != sg.digest(base / "shape-master.glb")
    ):
        raise ValueError(
            "Local display mesh no longer matches its fitted source-camera geometry"
        )
    fixed = {
        "image_right": [1.0, 0.0, 0.0],
        "image_up": [0.0, 1.0, 0.0],
        "camera_direction_from_origin": [0.0, 0.0, 1.0],
    }
    if "pinhole" in frame:
        fixed["pinhole"] = frame["pinhole"]
    return {
        **camera,
        "camera_frame": fixed,
        "upright_transform": np.eye(4).tolist(),
        "mesh_file_sha256": appearance["output_sha256"],
        "foreground_file_sha256": appearance["foreground_sha256"],
        "coordinate_space": "source-oriented local painted GLB",
        "local_master_sha256": appearance["shape_sha256"],
        "raw_camera_sha256": json_sha(camera),
        "local_source_sha256": appearance["original_sha256"],
    }


def _immutable(path, payload):
    if path.exists():
        if sg.read_json(path) != payload:
            raise ValueError("Stored chart routing evidence differs")
    else:
        sg.atomic_json(path, payload)


def _local_artifacts(directory):
    from engine.guided_shape_job import EDITABLE
    return {name: sg.digest(directory / name) for name in sorted(EDITABLE | {"continuity-paint.json"})}


def _paint_bound(paint, report):
    return (
        paint["source_glb"] == report["base_sha256"]
        and paint["output_glb"] == report["output_sha256"]
        and paint["foreground_sha256"] == report["foreground_sha256"]
        and paint["source_image_sha256"] == report["source_sha256"]
        and paint["visual_brief_sha256"] == report["observation_sha256"]
        and paint["locked_changed"] == 0
        and paint["unsupported_surface_changed"] == 0
        and paint["normals_uv_nodes_material_and_all_other_buffers_bitwise_identical"] is True
        and paint["source_rear_observed"] is False
        and paint["supported_hidden_fraction"] == report["coverage"]
        and type(paint["changed_uv"]) is int and paint["changed_uv"] > 0
    )


def _attempts(work):
    path = work / "guided-paint" / "receipt.json"
    if not path.exists():
        return 0
    receipt = sg.read_json(path)
    if receipt.get("job_recipe") != "joined-intrinsic-strip-v1" or receipt.get(
        "attempts"
    ) not in (0, 1):
        raise ValueError("Unexpected saved paint reservation")
    return receipt["attempts"]


def _result(
    base,
    source,
    work,
    out,
    key,
    original,
    observation,
    observation_receipt,
    decision,
    reason,
    metric=None,
    painted=None,
    manifest=None,
    sheet=None,
    receipt=None,
):
    from engine.guided_shape_job import BASE_FILES, EDITABLE

    expect = files_for(decision, BASE_FILES)
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        stage = Path(tmp)
        for name in BASE_FILES:
            folder = painted if painted is not None and name in EDITABLE else base
            shutil.copyfile(folder / name, stage / name)
        if decision == "inferred_chart":
            shutil.copyfile(
                work / "joined-guides" / "guide.png", stage / "chart-guide.png"
            )
            shutil.copyfile(sheet, stage / "inferred-paint.png")
            shutil.copyfile(painted / "chart-paint.json", stage / "chart-paint.json")
        if decision == "local_continuity":
            shutil.copyfile(
                painted / "continuity-paint.json", stage / "continuity-paint.json"
            )
        sha = sg.digest(stage / "model.glb")
        count = _attempts(work)
        info = dict(
            schema=1,
            pipeline=key["pipeline"],
            decision=decision,
            reason=reason,
            source_sha256=key["input_sha256"],
            foreground_sha256=sg.digest(stage / "input-processed.png"),
            shape_sha256=sg.digest(stage / "shape-master.glb"),
            base_sha256=original["glb_sha256"],
            output_sha256=sha,
            observation_sha256=json_sha(observation) if observation else None,
            paint_attempts=count,
            chart_id=manifest["chart_id"] if manifest else None,
            sheet_sha256=sg.digest(sheet) if sheet else None,
            coverage=metric,
            appearance="needs_review",
        )
        sg.atomic_json(stage / "chart-result.json", info)
        appearance = sg.read_json(stage / "appearance.json")
        appearance.update(
            baseline_output_sha256=original["glb_sha256"],
            output_sha256=sha,
            hidden_paint_history="chart-result.json",
            appearance="needs_review",
        )
        sg.atomic_json(stage / "appearance.json", appearance)
        generation = sg.read_json(stage / "generation.json")
        generation.update(
            hidden_paint_method=METHOD,
            visual_quality="needs_review",
            chart_decision=decision,
            visual_brief_sha256=info["observation_sha256"],
            external_image_request_attempts=count,
            photo_observation=observation_receipt,
            external_photo_observations=int(observation_receipt is not None),
        )
        if receipt is not None:
            generation["hidden_paint_provider"] = {
                k: receipt[k]
                for k in (
                    "provider",
                    "thread_id",
                    "state",
                    "sheet_sha256",
                    "geometry_sha256",
                    "guide_sha256",
                    "photo_sha256",
                    "input_sha256",
                    "directions",
                    "prompt_sha256",
                    "visual_brief_sha256",
                    "elapsed_seconds",
                )
                if k in receipt
            }
        sg.atomic_json(stage / "generation.json", generation)
        hashes = {n: sg.digest(stage / n) for n in expect}
        settings = {
            **original,
            "pipeline": key["pipeline"],
            "chart_decision": decision,
            "glb_sha256": sha,
            "artifact_sha256": hashes,
            "local_base_sha256": original["glb_sha256"],
            "visual_quality": "needs_review",
        }
        # Original high-quality base remains in local-guided-work/base. Final settings commit last.
        for n in sorted(hashes):
            os.replace(stage / n, out / n)
        sg.atomic_json(out / "settings.json", settings)
    return settings


def files_for(decision, base_names):
    if decision not in REASONS:
        raise ValueError("Unrecognised chart decision")
    extra = {"chart-result.json"}
    if decision == "local_continuity":
        extra.add("continuity-paint.json")
    if decision == "inferred_chart":
        extra.update(["chart-guide.png", "chart-paint.json", "inferred-paint.png"])
    return set(base_names) | extra


def valid(directory, settings):
    from engine.guided_shape_job import BASE_FILES
    from engine.pipeline import SG_CHART_PIPELINE

    try:
        d = Path(directory)
        art = settings["artifact_sha256"]
        decision = settings["chart_decision"]
        r = sg.read_json(d / "chart-result.json")
        a = sg.read_json(d / "appearance.json")
        g = sg.read_json(d / "generation.json")
        expected = {
            "schema",
            "pipeline",
            "decision",
            "reason",
            "source_sha256",
            "foreground_sha256",
            "shape_sha256",
            "base_sha256",
            "output_sha256",
            "observation_sha256",
            "paint_attempts",
            "chart_id",
            "sheet_sha256",
            "coverage",
            "appearance",
        }
        if set(r) != expected or type(r["schema"]) is not int:
            return False
        for k in (
            "source_sha256",
            "foreground_sha256",
            "shape_sha256",
            "base_sha256",
            "output_sha256",
            "observation_sha256",
            "chart_id",
            "sheet_sha256",
        ):
            if r[k] is None and k in ("chart_id", "sheet_sha256"):
                continue
            if not isinstance(r[k], str) or not re.fullmatch("[0-9a-f]{64}", r[k]):
                return False
        if r["coverage"] is not None and (
            type(r["coverage"]) not in (float, int) or not 0 <= r["coverage"] <= 1
        ):
            return False
        if decision != DIAGNOSTIC and (
            r["coverage"] is None or r["coverage"] < MINIMUM_WRITABLE_COVERAGE
        ):
            return False
        if (
            settings.get("visual_quality") != "needs_review"
            or settings.get("auto_finish_complete") is not True
        ):
            return False
        if not (
            settings["pipeline"] == r["pipeline"] == SG_CHART_PIPELINE
            and r["schema"] == 1
            and r["decision"] == decision
            and r["reason"] in REASONS[decision]
            and r["appearance"] == "needs_review"
            and r["source_sha256"] == settings["input_sha256"] == a["original_sha256"]
            and r["shape_sha256"]
            == art["shape-master.glb"]
            == a["shape_sha256"]
            == g["shape_result"]["sha256"]
            and r["foreground_sha256"]
            == art["input-processed.png"]
            == a["foreground_sha256"]
            == g["engine"]["input_sha256"]
            and r["output_sha256"]
            == settings["glb_sha256"]
            == art["model.glb"]
            == a["output_sha256"]
            and r["base_sha256"]
            == settings["local_base_sha256"]
            == a["baseline_output_sha256"]
            and g["external_photo_observations"] == 1
            and g["photo_observation"]["source_sha256"] == r["source_sha256"]
            and g.get("visual_brief_sha256") == r["observation_sha256"]
            and g["chart_decision"] == decision
            and type(r["paint_attempts"]) is int
            and 0 <= r["paint_attempts"] <= 1
            and g["external_image_request_attempts"] == r["paint_attempts"]
            and set(art) == files_for(decision, BASE_FILES)
            and all(sg.digest(d / n) == h for n, h in art.items())
        ):
            return False
        if decision == DIAGNOSTIC:
            return r["output_sha256"] == r["base_sha256"]
        if decision == "local_continuity":
            paint = sg.read_json(d / "continuity-paint.json")
            return (
                r["paint_attempts"] == 0
                and _paint_bound(paint, r)
                and paint["method"] == "source-surface-continuity-v1"
                and 0 < paint["supported_hidden_uv"] <= paint["eligible_hidden_uv"]
                and paint["supported_hidden_fraction"] == paint["supported_hidden_uv"] / paint["eligible_hidden_uv"]
            )
        p = sg.read_json(d / "chart-paint.json")
        provider = g["hidden_paint_provider"]
        return (
            r["paint_attempts"] == 1
            and _paint_bound(p, r)
            and p["method"] == "joined-intrinsic-strip-v1"
            and p["source_observed_changed"] == 0
            and p["sheet_sha"]
            == art["inferred-paint.png"]
            == r["sheet_sha256"]
            == provider["sheet_sha256"]
            and p["guide_sha"] == art["chart-guide.png"] == provider["guide_sha256"]
            and p["chart_id"] == r["chart_id"]
            and p["generation_id"] == provider["thread_id"]
            and provider["directions"] == ["joined_chart"]
            and p["visual_brief_sha256"]
            == r["observation_sha256"]
            == provider["visual_brief_sha256"]
        )
    except (ValueError, TypeError, KeyError, OSError):
        return False


def run(source, out, base, work, key, original, observation, public, setup, record):
    from engine.continuity_paint import paint_continuity
    from engine import joined_paint_chart as chart, joined_chart_provider as provider
    from engine.native_full_paint import NativeFullReviewHold
    from engine.guided_provider import TurnaroundProviderError

    args = dict(
        source_photo=source,
        native_model=base / "model.glb",
        foreground=base / "input-processed.png",
        registration=display_registration(base, source, original),
        observation=observation,
    )
    common = (base, source, work, out, key, original, observation, public)
    bind = {
        k: v
        for k, v in (
            ("method", METHOD),
            ("pipeline", key["pipeline"]),
            ("source", key["input_sha256"]),
            ("base", original["glb_sha256"]),
            ("registration", json_sha(args["registration"])),
            ("observation", json_sha(observation)),
        )
    }
    slot = work / "guided-paint"
    checkpoint = work / "chart-route.json"
    if checkpoint.exists():
        choice = sg.read_json(checkpoint)
        if choice.get("binding") != bind or choice.get("path") not in {
            "local",
            "chart",
            "diagnostic",
        }:
            raise ValueError("Bound appearance route changed")
    else:
        if slot.exists():
            raise ValueError("No matching route checkpoint for existing image request")
        print("preparing shape guides", flush=True)
        record("guided_guides")
        local = work / "source-continuity"
        if local.exists():
            # This uncommitted local computation costs no external calls; validate result again.
            with tempfile.TemporaryDirectory(dir=work) as tmp:
                temp = Path(tmp) / "verify"
                info = paint_continuity(**args, output_dir=temp)
                if (
                    info is None
                    or sg.read_json(local / "continuity-paint.json") != info
                    or _local_artifacts(local) != _local_artifacts(temp)
                ):
                    raise ValueError(
                        "Interrupted continuity output differs from recomputation"
                    )
        else:
            info = paint_continuity(**args, output_dir=local)
        local_coverage = float((info or {}).get("supported_hidden_fraction", 0))
        if info and local_coverage >= MINIMUM_WRITABLE_COVERAGE:
            choice = dict(
                binding=bind,
                path="local",
                coverage=local_coverage,
                result=info["output_glb"],
                artifacts=_local_artifacts(local),
            )
        else:
            guides = work / "joined-guides"
            try:
                if guides.exists():
                    manifest = sg.read_json(guides / "chart.json")
                    chart.confirm_chart(
                        **args,
                        guide_bytes=(guides / "guide.png").read_bytes(),
                        manifest=manifest,
                    )
                else:
                    manifest = chart.prepare_chart(
                        **args,
                        output_dir=guides,
                        minimum_hidden_fraction=MINIMUM_WRITABLE_COVERAGE,
                    )
                choice = dict(binding=bind, path="chart", chart_id=manifest["chart_id"])
            except chart.ChartUnavailable:
                choice = dict(binding=bind, path="diagnostic", coverage=local_coverage)
        _immutable(checkpoint, choice)
    if choice["path"] == "local":
        if slot.exists():
            raise ValueError("Unexpected external receipt for locally resolved paint")
        loc = work / "source-continuity"
        stats = sg.read_json(loc / "continuity-paint.json")
        if (
            _local_artifacts(loc) != choice.get("artifacts")
            or stats["output_glb"] != choice["result"]
            or sg.digest(loc / "model.glb") != choice["result"]
        ):
            raise ValueError("Local continuity evidence changed")
        return _result(
            *common, "local_continuity", "measured_continuity", choice["coverage"], loc
        )
    if choice["path"] == "diagnostic":
        if slot.exists():
            raise ValueError("Unexpected image receipt for unsupported domain")
        return _result(
            *common, DIAGNOSTIC, "registration_or_chart_unsupported", choice["coverage"]
        )
    guides = work / "joined-guides"
    manifest = sg.read_json(guides / "chart.json")
    if manifest["chart_id"] != choice["chart_id"]:
        raise ValueError("The accepted chart identity changed")
    intent = work / "paint-intent.json"
    reservation = {"binding": bind, "chart_id": manifest["chart_id"]}
    if intent.exists():
        _immutable(intent, reservation)
        if not (slot / "receipt.json").is_file():
            raise ValueError("以前の塗装受付の記録が欠落しています。重複送信を停止しました。")
    elif (slot / "receipt.json").is_file() and _attempts(work):
        # Retain the independent fence on resume of an older already submitted receipt.
        _immutable(intent, reservation)

    def reserve():
        _immutable(intent, reservation)

    print("requesting inferred paint", flush=True)
    record("guided_provider")
    try:
        result = provider.generate_chart(
            source,
            guides / "guide.png",
            guides / "chart.json",
            work,
            setup=setup,
            pipeline=key["pipeline"],
            before_submission=reserve,
            **{
                k: args[k]
                for k in ["native_model", "foreground", "registration", "observation"]
            },
        )
    except TurnaroundProviderError:
        # Integrity changes remain fatal. Unknown/rejected tool state never authorizes a second call.
        if slot.exists():
            sealed = sg.read_json(slot / "receipt.json")
            if (
                sealed.get("chart_id") != choice["chart_id"]
                or sealed.get("state") == "completed"
            ):
                raise
        # Keep the native model in base and the durable reservation intact. A later resume can
        # still collect copy_pending from the existing receipt; no terminal settings are cached.
        raise
    receipt = result["provenance"]
    sheet = result["sheet"]
    print("applying inferred paint", flush=True)
    record("guided_projection")
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        target = Path(tmp) / "projected"
        try:
            info = chart.apply_chart(
                **args,
                guide_dir=guides,
                sheet_path=sheet,
                receipt=receipt,
                output_dir=target,
            )
        except (chart.ChartUnavailable, NativeFullReviewHold):
            return _result(
                *common,
                DIAGNOSTIC,
                "proposed_paint_rejected",
                manifest=manifest,
                sheet=sheet,
                receipt=receipt,
            )
        return _result(
            *common,
            "inferred_chart",
            "one_inferred_chart",
            info["supported_hidden_fraction"],
            target,
            manifest,
            sheet,
            receipt,
        )
