"""A local photo-painted shape followed by one explicitly authorized paint proposal."""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from engine import local_shape_job, triposg_cpu as sg
from engine.pipeline import GUIDED_PIPELINE, GUIDED_PIPELINES, HOST_PIPELINE, OBSERVATION_METHODS, OBSERVED_PIPELINE, OBSERVED_PIPELINES, SHAPE_PIPELINE, SG_CHART_PIPELINE

EDITABLE = {
    "model.glb",
    "editable.gltf",
    "editable.obj",
    "editable.ply",
    "texture.png",
    "material.mtl",
    "material_0.png",
}
BASE_FILES = EDITABLE | {
    "shape-master.glb",
    "input-processed.png",
    "appearance.json",
    "generation.json",
}
GUIDED_FILES = BASE_FILES | {
    "guided-paint.json",
    "guided-views.png",
    "guided-layout.json",
    "inferred-paint.png",
}
StageError = local_shape_job.StageError


def _consent(setup, pipeline=GUIDED_PIPELINE):
    from studio.configure import valid_agreement

    try:
        if setup.stat().st_size > 32768:
            raise ValueError("設定が無効です。")
        valid = valid_agreement(sg.read_json(setup), pipeline)
    except (OSError, ValueError):
        valid = False
    if not valid:
        raise StageError(
            "guided_consent",
            "OpenAI への画像送信と本人の ChatGPT 利用枠への同意を設定してください。",
        )


def valid_chart(out,settings):
    from engine.local_joined_chart_job import valid
    return valid(out,settings)


def _validate(directory, pipeline, job_key):
    marker = directory / "settings.json"
    if not marker.exists():
        return None
    settings = sg.read_json(marker)
    if pipeline==SG_CHART_PIPELINE:
        if not (settings["pipeline"]==pipeline and settings["auto_finish_complete"] is True and settings["input_sha256"]==job_key["input_sha256"]
                and settings["marching_cubes_resolution"]==job_key["resolution"] and settings["texture_size"]==job_key["texture_size"]
                and valid_chart(directory,settings)):
            raise ValueError("保存した展開図ジョブの入力と出力が一致しません。")
        return settings
    files = settings["artifact_sha256"]
    required = GUIDED_FILES if pipeline in GUIDED_PIPELINES else BASE_FILES
    if not (
        settings["pipeline"] == pipeline
        and settings["auto_finish_complete"] is True
        and settings["input_sha256"] == job_key["input_sha256"]
        and settings["marching_cubes_resolution"] == job_key["resolution"]
        and settings["texture_size"] == job_key["texture_size"]
        and set(files) == required
        and files["model.glb"] == settings["glb_sha256"]
        and all(sg.digest(directory / name) == digest for name, digest in files.items())
    ):
        raise ValueError(
            "保存済みモデルの入力、設定、またはチェックサムが一致しません。"
        )
    appearance = sg.read_json(directory / "appearance.json")
    generation = sg.read_json(directory / "generation.json")
    if not (
        appearance["output_sha256"] == files["model.glb"]
        and appearance["shape_sha256"]
        == files["shape-master.glb"]
        == generation["shape_result"]["sha256"]
        and appearance["foreground_sha256"]
        == files["input-processed.png"]
        == generation["engine"]["input_sha256"]
        and appearance["original_sha256"]
        == job_key["input_sha256"]
        == generation["foreground"]["source_sha256"]
    ):
        raise ValueError("保存した写真、元の立体と完成GLBのつながりを確認できません。")
    if pipeline == OBSERVED_PIPELINE:
        from engine.visual_brief import json_sha
        observation = generation.get("photo_observation", {})
        brief = sg.read_json(directory / "guided-layout.json").get("visual_brief", {})
        if (settings.get("photo_observer") not in OBSERVATION_METHODS
                or observation.get("method") != settings["photo_observer"]
                or brief.get("method") != observation["method"]
                or observation.get("source_sha256") != job_key["input_sha256"]
                or brief.get("source_sha256") != job_key["input_sha256"]
                or json_sha(brief) != generation.get("hidden_paint_provider", {}).get("visual_brief_sha256")
                or generation.get("external_service_requests") != 2):
            raise ValueError("保存した外観解析を元写真・塗装の記録と照合できません。")
    return settings


def vision_runtime_path():
    runtime = Path(os.environ.get(
        "STUDIO_VISUAL_BRIEF_RUNTIME",
        Path(__file__).resolve().parents[1] / ".runtime" / "visual-brief",
    ))
    if not runtime.is_absolute():
        raise ValueError("STUDIO_VISUAL_BRIEF_RUNTIME must be an absolute directory.")
    return runtime


def _observations(source, work, source_sha):
    from engine.visual_brief import analyze_photo, provider_observations

    report = analyze_photo(source, work / "local-observations", runtime=vision_runtime_path())
    return provider_observations(report, source_sha)


def _chatgpt_observations(source, work, source_sha, setup, *, pipeline=OBSERVED_PIPELINE):
    from engine import codex_photo_observations as observer

    if pipeline not in OBSERVED_PIPELINES:
        raise ValueError("この生成方式には写真の外部解析を使用できません。")
    cache = work / "photo-observer"
    # Any downstream checkpoint proves the observation already took place.
    # Recovery reads the original response and must never launch another turn.
    downstream = ("observation-seal.json", "guides", "guided-paint", "chart-route.json", "joined-guides",
                  "source-continuity", "paint-intent.json", "observation-intent.json")
    if pipeline != SG_CHART_PIPELINE:
        downstream+=("base",)
    has_checkpoint=any((work/name).exists() for name in downstream)
    if pipeline == SG_CHART_PIPELINE:
        has_checkpoint |= any((work.parent / name).exists() for name in ("chart-result.json", "model.glb", "generation.json"))
    journal = cache / "receipt.json"
    if pipeline == SG_CHART_PIPELINE and not has_checkpoint and journal.is_file():
        # The standing worker lock serializes this move. Keep the no-send evidence for the audit.
        prior = sg.read_json(journal)
        if (prior.get("state") == "not_submitted_consent_revoked"
                and prior.get("source_sha256") == source_sha and prior.get("method") == observer.METHOD_V2
                and not any((cache / name).exists() for name in ("last-message.json", "events.jsonl"))):
            archive = Path(tempfile.mkdtemp(prefix="observer-no-send-", dir=work))
            cache.rename(archive / "journal")
    if has_checkpoint or journal.exists():
        report = observer.recover_photo(source, cache)
    else:
        _consent(setup, pipeline)

        def consent_now():
            try:
                _consent(setup, pipeline)
            except StageError:
                return False
            if pipeline == SG_CHART_PIPELINE:
                # Invoked after account/CLI preflight, local preparation and the exclusive inner claim.
                sg.atomic_json(work / "observation-intent.json", {"source_sha256": source_sha, "pipeline": pipeline})
            return True

        report = observer.analyze_photo(source, cache, recipe=observer.METHOD_V2,
                                       external_opt_in=True, check_consent=consent_now)
        if report.get("method") != observer.METHOD_V2:
            raise ValueError("受付時の写真解析方式と完了記録が一致しません。")
    observation = observer.provider_observations(report, source_sha)
    if (report["source_sha256"] != source_sha or report["method"] not in OBSERVATION_METHODS
            or report["api_key_route"] is not False or report["state"] != "completed"
            or report["recipient"] != "OpenAI ChatGPT" or report["unseen_surface_observed"] is not False):
        raise ValueError("元写真・送信先・利用方法に対応した完了証跡を確認できません。")
    # Usage counters and local account details stay in the private provider directory.
    public = {k: report[k] for k in (
        "method", "state", "source_sha256", "uploaded_sha256", "request_id", "thread_id",
        "prompt_sha256", "schema_sha256", "flags_sha256", "output_sha256", "events_sha256", "model",
        "recipient", "api_key_route", "unseen_surface_observed", "quality",
    )}
    seal = work / "observation-seal.json"
    frozen = {"receipt": public, "visual_brief": observation}
    if seal.is_file():
        if sg.read_json(seal) != frozen:
            raise ValueError("受付時の写真解析と復元した観察結果が一致しません。")
    else:
        sg.atomic_json(seal, frozen)
    return observation, public


def _prepare(source, base, work, settings, observations=None):
    from engine.guided_texture import prepare_guides

    guides = work / "guides"
    if not guides.exists():
        with tempfile.TemporaryDirectory(dir=work) as tmp:
            stage = Path(tmp) / "guides"
            prepare_guides(base / "model.glb", stage)
            geometry = sg.read_json(stage / "guide.json")
            layout = {
                "model_sha": settings["glb_sha256"],
                "photo_sha": sg.digest(source),
                "guide_sha": sg.digest(stage / "guides.png"),
                "camera": geometry["views"],
            }
            if (
                geometry["model_sha256"] != layout["model_sha"]
                or geometry["guide_sha256"] != layout["guide_sha"]
                or set(layout["camera"]) != {"rear", "opposite"}
            ):
                raise ValueError("ガイドの形状または画像の記録が一致しません。")
            if observations is not None:
                layout["visual_brief"] = observations
            sg.atomic_json(stage / "request-guide.json", layout)
            sg.atomic_json(
                stage / "checksums.json",
                {
                    n: sg.digest(stage / n)
                    for n in ("guide.json", "guides.png", "request-guide.json")
                },
            )
            os.replace(stage, guides)
    layout = sg.read_json(guides / "request-guide.json")
    checks = sg.read_json(guides / "checksums.json")
    if not (
        set(checks) == {"guide.json", "guides.png", "request-guide.json"}
        and all(sg.digest(guides / n) == checks[n] for n in checks)
        and layout["model_sha"] == settings["glb_sha256"]
        and layout["photo_sha"] == sg.digest(source) == settings["input_sha256"]
        and layout["guide_sha"] == checks["guides.png"]
        and sg.read_json(guides / "guide.json")["views"] == layout["camera"]
    ):
        raise ValueError("保存したガイドのチェックサムが一致しません。")
    return guides, layout


def _orientation_check(base, source, sheet, layout, work):
    from engine.hidden_review import review_hidden_paint

    camera = sg.read_json(base / "generation.json")["camera"]["camera_frame"]
    return review_hidden_paint(
        base / "model.glb", source, base / "input-processed.png", sheet,
        layout.get("visual_brief"), work / "orientation", camera_frame=camera,
    )


def _project(base, guides, sheet, receipt, stage, orientation, observation_receipt=None):
    from engine.guided_texture import apply_guides

    camera = sg.read_json(base / "generation.json")["camera"]["camera_frame"]
    apply_guides(
        base / "model.glb", base / "input-processed.png", camera, guides, sheet, stage
    )
    appearance = sg.read_json(base / "appearance.json")
    generation = sg.read_json(base / "generation.json")
    stats = sg.read_json(stage / "guided-paint.json")
    final_hash = sg.digest(stage / "model.glb")
    if not (
        stats.get("input_sha256")
        == receipt["geometry_sha256"]
        == appearance["output_sha256"]
        and stats.get("foreground_sha256") == appearance["foreground_sha256"]
        and stats.get("inferred_sha256") == receipt["sheet_sha256"]
        and stats.get("guide_sha256") == receipt["guide_sha256"]
        and stats.get("output_sha256") == final_hash
        and stats.get("locked_bytes_changed") == 0
        and stats.get("modified_surface_texels", 0) > 0
    ):
        raise ValueError(
            "推定塗装の入力、変更保護、または完成ファイルの検証に失敗しました。"
        )
    if final_hash == receipt["geometry_sha256"]:
        raise ValueError("推定塗装による変更が確認できませんでした。")
    # Retain only the service evidence needed for this job. Machine-specific
    # temporary paths and local account details belong outside deliverables.
    evidence = {
        k: receipt[k]
        for k in (
            "provider",
            "cli_version",
            "state",
            "thread_id",
            "recipient",
            "input_sha256",
            "photo_sha256",
            "guide_sha256",
            "geometry_sha256",
            "sheet_sha256",
            "prompt_sha256",
            "uploaded_png_sha256",
            "original_photograph_count",
            "guide_count",
            "elapsed_seconds",
            "host_tool_call_id",
            "delivery_mode",
            "visual_brief_sha256",
        )
        if k in receipt
    }
    stats.update(
        output_sha256=final_hash,
        baseline_sha256=receipt["geometry_sha256"],
        sheet_sha256=receipt["sheet_sha256"],
        provider=evidence,
        hidden_orientation=orientation,
        visual_quality="needs_review",
    )
    sg.atomic_json(stage / "guided-paint.json", stats)
    appearance.update(
        baseline_output_sha256=appearance["output_sha256"],
        output_sha256=final_hash,
        hidden_paint_history="guided-paint.json",
        appearance="needs_review",
    )
    sg.atomic_json(stage / "appearance.json", appearance)
    generation.update(
        external_generation_calls=1,
        hidden_paint_provider=evidence,
        visual_quality="needs_review",
    )
    if observation_receipt is not None:
        generation.update(photo_observation=observation_receipt, external_photo_observations=1,
                          external_image_generations=1, external_service_requests=2)
    sg.atomic_json(stage / "generation.json", generation)
    for src, dst in (
        (base / "shape-master.glb", "shape-master.glb"),
        (base / "input-processed.png", "input-processed.png"),
        (guides / "guides.png", "guided-views.png"),
        (guides / "request-guide.json", "guided-layout.json"),
        (sheet, "inferred-paint.png"),
    ):
        shutil.copyfile(src, stage / dst)


def run(source, out, resolution=256, texture_size=1024, *, setup, pipeline=GUIDED_PIPELINE):
    from engine.guided_provider import generate_guided_paint

    if resolution not in (128, 256) or texture_size not in (256, 512, 1024):
        raise StageError(
            "guided_guides",
            "対応する解像度は形状128・256、テクスチャ256・512・1024です。",
        )
    source, out, setup = (Path(p).resolve() for p in (source, out, setup))
    key = {
        "pipeline": pipeline,
        "input_sha256": sg.digest(source),
        "resolution": resolution,
        "texture_size": texture_size,
    }
    work = out / "local-guided-work"
    work.mkdir(parents=True, exist_ok=True)
    base = work / "base"
    state = work / "state.json"

    def record(stage, status="running", **extra):
        sg.atomic_json(
            state,
            {
                **key,
                "stage": stage,
                "status": status,
                "local_base": "local-guided-work/base/model.glb"
                if (base / "settings.json").exists()
                else None,
                **extra,
            },
        )

    with sg.writer_lock(work):
        stage = "guided_checkpoint"
        try:
            identity = work / "identity.json"
            if identity.exists() and sg.read_json(identity) != key:
                raise ValueError("保存したジョブの入力または設定が一致しません。")
            sg.atomic_json(identity, key)
            final = _validate(out, pipeline, key)
            if final is not None:
                record("guided_export", "completed")
                return final
            stage = "guided_consent"
            _consent(setup, pipeline)
            stage = "guided_checkpoint"
            record(stage)
            # An existing manifest is part of the immutable provider request, including
            # older requests created before local photograph analysis was available.
            observations, observation_receipt = None, None
            if pipeline == SG_CHART_PIPELINE:
                from engine import local_joined_chart_job as joined
                original=_validate(base,SHAPE_PIPELINE,key)
                if original is None:
                    record("guided_checkpoint")
                    local_shape_job.run(source,base,resolution,texture_size)
                    original=_validate(base,SHAPE_PIPELINE,key)
                    if original is None:raise ValueError("ローカル形状を確認できません。")
                stage="guided_chatgpt_observation";print("observing photo with ChatGPT",flush=True);record(stage)
                observations,observation_receipt=_chatgpt_observations(source,work,key["input_sha256"],setup,pipeline=pipeline)
                stage="guided_projection"
                saved=joined.run(source,out,base,work,key,original,observations,observation_receipt,setup,record)
                if not valid_chart(out,saved):raise ValueError("展開図の出力検証に失敗しました。")
                record("guided_export","completed")
                return saved
            if pipeline == OBSERVED_PIPELINE:
                stage = "guided_chatgpt_observation"
                print("observing photo with ChatGPT", flush=True)
                record(stage)
                observations, observation_receipt = _chatgpt_observations(source, work, key["input_sha256"], setup)
            elif not (work / "guides").exists():
                stage = "guided_observation"
                print("analyzing surface markings", flush=True)
                record(stage)
                observations = _observations(source, work, key["input_sha256"])
            stage = "guided_checkpoint"
            original = _validate(base, SHAPE_PIPELINE, key)
            if original is None:
                local_shape_job.run(source, base, resolution, texture_size)
                original = _validate(base, SHAPE_PIPELINE, key)
                if original is None:
                    raise ValueError("ローカル形状の保存が完了していません。")
            stage = "guided_guides"
            print("preparing shape guides", flush=True)
            record(stage)
            guides, layout = _prepare(source, base, work, original, observations)
            if pipeline == OBSERVED_PIPELINE and layout.get("visual_brief") != observations:
                raise ValueError("保存した形状ガイドとChatGPTで確認した元写真の解析が一致しません。")
            stage = "guided_provider"
            print("requesting inferred paint", flush=True)
            record(stage)
            _consent(setup, pipeline)
            if pipeline == HOST_PIPELINE:
                from engine import host_paint
                request = host_paint.prepare(work, source, guides / "guides.png", layout)
                result = host_paint.result(work, request)
                if result is None:
                    raise StageError("awaiting_host_image", "Codex の画像生成結果を待っています。")
            else:
                options = {"pipeline": pipeline} if pipeline == OBSERVED_PIPELINE else {}
                result = generate_guided_paint(
                    source, guides / "guides.png", guides / "request-guide.json", work, setup=setup, **options,
                )
            sheet = Path(result["sheet"])
            receipt = result["provenance"]
            if pipeline == OBSERVED_PIPELINE:
                from engine.visual_brief import json_sha
                if receipt.get("visual_brief_sha256") != json_sha(observations):
                    raise ValueError("画像生成の受付が、同じ写真のChatGPT解析に結び付いていません。")
            if not (
                receipt.get("state") == "completed"
                and receipt.get("photo_sha256") == layout["photo_sha"]
                and receipt.get("geometry_sha256") == layout["model_sha"]
                and receipt.get("guide_sha256") == layout["guide_sha"]
                and sg.digest(sheet) == receipt.get("sheet_sha256")
                and (receipt.get("host_tool_call_id") if pipeline == HOST_PIPELINE else receipt.get("thread_id"))
            ):
                raise ValueError(
                    "外部生成の写真・形状・結果の記録を確認できませんでした。"
                )
            stage = "guided_orientation"
            print("checking hidden surfaces", flush=True)
            record(stage)
            orientation = _orientation_check(base, source, sheet, layout, work)
            if orientation["hold_projection"]:
                raise StageError(stage, "推定した背面に正面の特徴が重なっています。画像と検品記録を保存しました。")
            stage = "guided_projection"
            print("applying inferred paint", flush=True)
            record(stage)
            with tempfile.TemporaryDirectory(dir=work) as tmp:
                stage_dir = Path(tmp) / "deliverables"
                _project(base, guides, sheet, receipt, stage_dir, orientation, observation_receipt)
                stage = "guided_export"
                print("exporting guided model", flush=True)
                record(stage)
                hashes = {name: sg.digest(stage_dir / name) for name in GUIDED_FILES}
                settings = {
                    **original,
                    "pipeline": pipeline,
                    "artifact_sha256": hashes,
                    "glb_sha256": hashes["model.glb"],
                    "local_base_sha256": original["glb_sha256"],
                    "guided_paint_history": "guided-paint.json",
                    "visual_quality": "needs_review",
                }
                if observation_receipt:
                    settings["photo_observer"] = observation_receipt["method"]
                for name in sorted(hashes):
                    os.replace(stage_dir / name, out / name)
                sg.atomic_json(out / "settings.json", settings)
            record(stage, "completed")
            return settings
        except StageError as exc:
            record(exc.code, "waiting" if exc.code == "awaiting_host_image" else "failed", reason=str(exc))
            raise
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            RuntimeError,
            MemoryError,
        ) as exc:
            record(stage, "failed", reason=str(exc))
            raise StageError(stage, str(exc)) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resolution", type=int, choices=(128, 256), default=256)
    parser.add_argument(
        "--texture-size", type=int, choices=(256, 512, 1024), default=1024
    )
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--pipeline", choices=GUIDED_PIPELINES, default=GUIDED_PIPELINE)
    args = parser.parse_args()
    try:
        run(
            args.image,
            args.output,
            args.resolution,
            args.texture_size,
            setup=args.setup,
            pipeline=args.pipeline,
        )
    except StageError as exc:
        sg.emit("studio_error", code=exc.code, detail=str(exc))
        raise SystemExit(72 if exc.code == "awaiting_host_image" else 1)


if __name__ == "__main__":
    main()
