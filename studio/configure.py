"""Select the local engine once for a Studio job store."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from engine.pipeline import GUIDED_PIPELINE, GUIDED_PIPELINES, HOST_PIPELINE, PIPELINE, SHAPE_PIPELINE

ROOT = Path(__file__).resolve().parents[1]
ENGINES = {"triposr": PIPELINE, "triposg": SHAPE_PIPELINE, "triposg-guided": GUIDED_PIPELINE,
           "triposg-guided-host": HOST_PIPELINE}


def agreement(pipeline):
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
    if doc["pipeline"] in GUIDED_PIPELINES:
        if doc.get("agreement") != agreement(doc["pipeline"]):
            raise ValueError("背面の推定には、元写真と立体の確認画像をOpenAIへ送信する初回設定が必要です。")
    return doc["pipeline"]


def configure(directory, engine, *, allow_send_to_openai=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pipeline = ENGINES[engine]
    config = {"pipeline": pipeline}
    if pipeline in GUIDED_PIPELINES:
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
    args = parser.parse_args()
    print(configure(args.jobs_dir, args.engine, allow_send_to_openai=args.allow_send_to_openai))
