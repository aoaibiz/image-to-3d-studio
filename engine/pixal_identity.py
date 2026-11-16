"""Pinned identity, usable in the lightweight Studio service process."""

import re

SPACE = "victor/pixal3d-studio"
REVISION = "ef1cb838893269417b3eaa81dd8fc635f65a5180"
ORIGIN = "https://victor-pixal3d-studio.hf.space"
SEED = 1234


def matches_completed(receipt):
    asset = receipt.get("asset_id")
    event = receipt.get("event_id")
    return (
        receipt.get("state") == "completed" and receipt.get("provider_space") == SPACE
        and receipt.get("revision") == REVISION and receipt.get("origin") == ORIGIN
        and receipt.get("seed") == SEED and receipt.get("resolution") == 1024
        and receipt.get("quality_setting") == "balanced"
        and receipt.get("mock") is False and receipt.get("share_to_community") is False
        and isinstance(event, str) and re.fullmatch(r"[a-f0-9]{32}", event) is not None
        and isinstance(asset, str) and re.fullmatch(r"[a-f0-9]{12}", asset) is not None
        and receipt.get("download_url") == f"{ORIGIN}/community-assets/{asset}/model.glb"
    )
