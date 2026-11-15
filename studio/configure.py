"""Select the local engine once for a Studio job store."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from engine.pipeline import PIPELINE, SHAPE_PIPELINE

ROOT = Path(__file__).resolve().parents[1]
ENGINES = {"triposr": PIPELINE, "triposg": SHAPE_PIPELINE}


def read_pipeline(directory):
    try:
        doc = json.loads((Path(directory) / "engine.json").read_text(encoding="utf8"))
    except FileNotFoundError:
        return PIPELINE
    if not isinstance(doc, dict) or doc.get("pipeline") not in ENGINES.values():
        raise ValueError(
            "生成設定が無効です。studio.configure でローカルエンジンを設定し直してください。"
        )
    return doc["pipeline"]


def configure(directory, engine):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pipeline = ENGINES[engine]
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf8", dir=directory, delete=False
    ) as f:
        temp = Path(f.name)
        json.dump({"pipeline": pipeline}, f)
    try:
        os.replace(temp, directory / "engine.json")
    finally:
        temp.unlink(missing_ok=True)
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=ENGINES, required=True)
    parser.add_argument("--jobs-dir", type=Path, default=ROOT / "jobs" / "studio")
    args = parser.parse_args()
    print(configure(args.jobs_dir, args.engine))
