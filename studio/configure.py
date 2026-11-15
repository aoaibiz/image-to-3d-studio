"""Select the local engine once for a Studio job store."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from engine.pipeline import (GUIDED_PIPELINE, GUIDED_PIPELINES, HOST_PIPELINE, OBSERVED_AGREEMENT,
                             OBSERVED_PIPELINE, OBSERVED_PIPELINES, PIPELINE, SF3D_AGREEMENT,
                             SF3D_OBSERVED_PIPELINE, SF3D_PIPELINE, SF3D_PIPELINES, SHAPE_PIPELINE)

ROOT = Path(__file__).resolve().parents[1]
ENGINES = {SF3D_PIPELINE: SF3D_PIPELINE, "triposr": PIPELINE, "triposg": SHAPE_PIPELINE, "triposg-guided": GUIDED_PIPELINE,
           "triposg-guided-host": HOST_PIPELINE, "triposg-observed": OBSERVED_PIPELINE,
           "sf3d-observed": SF3D_OBSERVED_PIPELINE}


def agreement(pipeline):
    if pipeline == SF3D_OBSERVED_PIPELINE:
        return {"schema": 1, "shape": SF3D_AGREEMENT, "appearance": OBSERVED_AGREEMENT,
                "max_shape_inferences_per_job": 1}
    if pipeline == SF3D_PIPELINE:
        return SF3D_AGREEMENT
    if pipeline == OBSERVED_PIPELINE:
        return OBSERVED_AGREEMENT
    if pipeline == HOST_PIPELINE:
        from engine.host_paint import AGREEMENT
    else:
        from engine.guided_provider import AGREEMENT
    return AGREEMENT


def read_pipeline(directory):
    try:
        doc = json.loads((Path(directory) / "engine.json").read_text(encoding="utf8"))
    except FileNotFoundError:
        return PIPELINE
    if not isinstance(doc, dict) or doc.get("pipeline") not in ENGINES.values():
        raise ValueError(
            "生成設定が無効です。studio.configure でローカルエンジンを設定し直してください。"
        )
    if doc["pipeline"] in SF3D_PIPELINES:
        expected = {"pipeline": doc["pipeline"], "agreement": agreement(doc["pipeline"])}
        if json.dumps(doc, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError("選択した方式の送信先と利用枠を初回設定で確認してください。")
    if doc["pipeline"] in GUIDED_PIPELINES:
        if doc.get("agreement") != agreement(doc["pipeline"]):
            raise ValueError("背面の推定には、元写真と立体の確認画像をOpenAIへ送信する初回設定が必要です。")
    return doc["pipeline"]


def configure(directory, engine, *, allow_send_to_openai=False, chatgpt_photo_analysis=False, allow_official_space=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pipeline = ENGINES[engine]
    if (pipeline in OBSERVED_PIPELINES) != bool(chatgpt_photo_analysis):
        raise ValueError("写真の解析には対象の observed エンジンと --chatgpt-photo-analysis を指定してください。")
    config = {"pipeline": pipeline}
    if (pipeline in SF3D_PIPELINES) != bool(allow_official_space):
        raise ValueError("公式Spaceの評価には SF3D エンジンと --allow-official-space を指定してください。")
    if pipeline in SF3D_PIPELINES:
        config["agreement"] = agreement(pipeline)
    if pipeline in (*GUIDED_PIPELINES, SF3D_OBSERVED_PIPELINE):
        if not allow_send_to_openai:
            raise ValueError("OpenAIへの元写真・確認画像の送信を許可するには --allow-send-to-openai を指定してください。")
        config["agreement"] = agreement(pipeline)
        if pipeline == HOST_PIPELINE:
            from engine.host_paint import setup_token
            setup_token(directory)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf8", dir=directory, delete=False
    ) as f:
        temp = Path(f.name)
        json.dump(config, f)
    try:
        os.replace(temp, directory / "engine.json")
    finally:
        temp.unlink(missing_ok=True)
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=ENGINES, required=True)
    parser.add_argument("--jobs-dir", type=Path, default=ROOT / "jobs" / "studio")
    parser.add_argument("--allow-send-to-openai", action="store_true", help="Send the photo and two geometry guides to OpenAI using your ChatGPT plan.")
    parser.add_argument("--chatgpt-photo-analysis", action="store_true", help="Also send the normalized photo for one ChatGPT text observation per job.")
    parser.add_argument("--allow-official-space", action="store_true", help="Send the segmented photo to Stability AI on Hugging Face using your anonymous ZeroGPU quota.")
    args = parser.parse_args()
    print(configure(args.jobs_dir, args.engine, allow_send_to_openai=args.allow_send_to_openai,
                    chatgpt_photo_analysis=args.chatgpt_photo_analysis, allow_official_space=args.allow_official_space))
