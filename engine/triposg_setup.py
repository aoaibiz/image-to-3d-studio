"""Install the optional native Windows TripoSG runtime. Invoked by setup-triposg.ps1."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile


CODE = "fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c"
MODEL = "2c1c516d22d58db486a058d98d31bb6177344e06"
ARCHIVE = (16411924, "cbb21f7dbb9f6759b84f00b5bcdce5c75a5130ae3a63d012646cd274fd945062")
CODE_URL = "https://codeload.github.com/VAST-AI-Research/TripoSG/zip/" + CODE
MODEL_URL = "https://huggingface.co/VAST-AI/TripoSG/resolve/" + MODEL + "/"
WEIGHTS = {
    "README.md": (1465, "0f4c34a1a17acdbcff02c259d7f4f7b5df1b8956395c625c28da97930eb82ef3"),
    "model_index.json": (525, "750af638d10fc67a5f43d19ef0a0d1d3d446174e24b73bddd0a49226016c80c2"),
    "feature_extractor_dinov2/preprocessor_config.json": (436, "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"),
    "image_encoder_dinov2/config.json": (1104, "8ed81b0d5ffd2c8b2cecd7dc1326d3eefa409bc9d525537b6b2a3c70a992fd02"),
    "image_encoder_dinov2/model.safetensors": (1217522888, "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23"),
    "scheduler/scheduler_config.json": (158, "ea7c9b4d502dcaba97cd85291cfd7dbd3c6c82cbc4c775af5c6d9224864279c9"),
    "transformer/config.json": (191, "54ad779e68f9bc7303122cdf291f6db48dec7c67e41d28c05ac359daf914128a"),
    "transformer/diffusion_pytorch_model.safetensors": (5758280104, "9192b5923f7b605b394192809aa2ceb73bf0f4009674d8e3b999b45bb97d4bf2"),
    "vae/config.json": (335, "2fb6ba9b364d42c9681879dca0f7edcb978f2db974a1ca4b15d28f9b15a5baed"),
    "vae/diffusion_pytorch_model.safetensors": (970685468, "a2e667c24927a5a35e5f19fcb4c75890e9399aa966b6db8131d7df733a750c8b"),
}


def sha(path):
    with path.open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def fetch_range(url, part, size):
    start = part.stat().st_size if part.exists() else 0
    end = min(start + 64 * 1024**2, size) - 1
    # Fetch a fresh signed CDN redirect for every range, also behind HTTP caches.
    req = urllib.request.Request(f"{url}?download=true&offset={start}&t={time.time_ns()}",
                                 headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=60) as response:
        if response.status == 206 and response.headers.get("Content-Range") == f"bytes {start}-{end}/{size}":
            expected = end - start + 1
        elif response.status == 200 and start == 0:
            expected = size
        else:
            raise ValueError(f"Server did not confirm requested byte range at offset {start}.")
        with part.open("ab") as target:
            remaining = expected
            while remaining:
                block = response.read(min(1024**2, remaining))
                if not block:
                    raise OSError("Incomplete response. Resume from the last byte and verify the final SHA-256.")
                target.write(block)
                remaining -= len(block)
            if response.read(1):
                target.truncate(start)
                raise ValueError("Server returned bytes outside the confirmed range.")


def acquire(dst, spec, url, cache=None, offline=False):
    size, digest = spec
    if dst.exists():
        if dst.stat().st_size == size and sha(dst) == digest:
            return
        raise ValueError(f"Existing file failed its pinned checksum. Move it aside and rerun: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".partial")
    if cache is not None:
        if cache.stat().st_size != size:
            raise ValueError(f"Cached file has the wrong size: {cache}")
        # A separate copy ensures later cache edits cannot alter the installation.
        shutil.copyfile(cache, part)
    elif not offline:
        failures = 0
        while not part.exists() or part.stat().st_size < size:
            try:
                fetch_range(url, part, size)
                failures = 0
            except OSError:
                failures += 1
                if failures >= 3:
                    raise
                time.sleep(failures)
    elif not part.exists() or part.stat().st_size != size:
        raise FileNotFoundError(f"Offline setup needs a verified local copy of {dst.name}")
    if part.stat().st_size != size or sha(part) != digest:
        part.unlink()
        raise ValueError(f"Pinned SHA-256 or size mismatch for {dst.name}. Check the cache or rerun the download.")
    part.replace(dst)


def source_files(archive, target):
    hashes = {}
    prefix = f"TripoSG-{CODE}/"
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = member.filename.removeprefix(prefix)
            if member.is_dir() or not (name.startswith("triposg/") or name in {"LICENSE", "NOTICE", "README.md"}):
                continue
            dest = target / name
            dest.resolve().relative_to(target.resolve())
            data = zf.read(member)
            digest = hashlib.sha256(data).hexdigest()
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                temp = dest.with_name(dest.name + ".partial")
                temp.write_bytes(data)
                temp.replace(dest)
            elif sha(dest) != digest:
                raise ValueError(f"Installed source differs from pinned code: {name}. Move it aside and rerun.")
            hashes[name] = digest
    # A prior ordinary Python import may have cached these same source modules.
    # Discard only their rebuildable bytecode. Reject any other unpinned module.
    for path in target.rglob("*"):
        if path.is_file() and path.relative_to(target).as_posix() not in hashes:
            source = path.parent.parent / (path.name.split(".")[0] + ".py")
            if path.parent.name == "__pycache__" and path.suffix == ".pyc" and source.relative_to(target).as_posix() in hashes:
                path.unlink()
            else:
                raise ValueError(f"Unexpected file in installed source: {path}. Move it aside and rerun.")
    return hashes


PROBE = """
import importlib.metadata, json, sys
sys.path.insert(0, sys.argv[1])
import torch, scipy.ndimage, skimage.measure, trimesh, psutil
from PIL import Image
from transformers import Dinov2Model, BitImageProcessor
from accelerate import init_empty_weights
from safetensors import safe_open
from triposg.models.transformers.triposg_transformer import TripoSGDiTModel
from triposg.models.autoencoders.autoencoder_kl_triposg import TripoSGVAEModel
from triposg.schedulers.scheduling_rectified_flow import RectifiedFlowScheduler
torch.set_num_threads(1)
x = torch.ones((32, 32), dtype=torch.bfloat16, device='cpu')
if not bool(((x @ x) == 32).all()) or torch.version.cuda is not None:
    raise RuntimeError('Expected a working BF16 CPU matrix operation using the CPU wheel')
print(json.dumps({'python': sys.version.split()[0], 'torch': torch.__version__,
                  'cpu_capability': torch.backends.cpu.get_cpu_capability(), 'bf16_matmul': True,
                  'physical_ram_bytes': psutil.virtual_memory().total,
                  'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}))
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--uv", required=True)
    p.add_argument("--weights-cache", type=Path)
    p.add_argument("--source-archive", type=Path)
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()
    if (os.name != "nt" or platform.machine().upper() not in {"AMD64", "X86_64"}
            or sys.maxsize < 2**32 or sys.version_info[:3] != (3, 11, 15)):
        raise ValueError("Use 64-bit Windows x86-64 with CPython 3.11.15.")
    root = args.runtime.resolve()
    if (root / ".image-to-3d-triposg-v1").read_text() != "image-to-3d-studio TripoSG v1":
        raise ValueError("Launch via setup-triposg.ps1 using a dedicated runtime directory.")
    manifest = root / "manifest.json"
    manifest.unlink(missing_ok=True)  # A failed revalidation must not leave a ready receipt.
    missing_bytes = 0
    for name, (size, _) in WEIGHTS.items():
        target = root / "weights" / name
        if not target.exists():
            part = target.with_name(target.name + ".partial")
            missing_bytes += max(0, size - (part.stat().st_size if part.exists() else 0))
    if shutil.disk_usage(root).free < missing_bytes:
        raise OSError(f"Insufficient disk space for the missing model assets ({missing_bytes:,} bytes).")
    lock = Path(__file__).with_name("triposg-runtime.lock")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("UV_HTTP_TIMEOUT", "180")
    env.setdefault("UV_CONCURRENT_DOWNLOADS", "4")
    uv = [args.uv, "--no-config"] + (["--offline"] if args.offline else [])
    python = root / "venv/Scripts/python.exe"
    if not python.exists():
        subprocess.run([*uv, "venv", "--python", sys.executable, str(root / "venv")], check=True, env=env)
    subprocess.run([*uv, "pip", "sync", "--python", str(python), "--require-hashes",
                    "--only-binary", ":all:", "--default-index", "https://pypi.org/simple", str(lock)],
                   check=True, env=env)
    archive = root / "archives" / f"{CODE}.zip"
    acquire(archive, ARCHIVE, CODE_URL, args.source_archive, args.offline)
    sources = source_files(archive, root / "source")
    probe = subprocess.check_output([str(python), "-I", "-B", "-c", PROBE, str(root / "source")],
                                    env=env, text=True)
    for name, spec in WEIGHTS.items():
        print(f"Verifying {name} ({spec[0]:,} bytes)", flush=True)
        cache = args.weights_cache / name if args.weights_cache else None
        acquire(root / "weights" / name, spec, MODEL_URL + name, cache, args.offline)
    receipt = {
        "schema": 1, "engine": "TripoSG", "status": "runtime-verified",
        "source": {"path": "source", "revision": CODE, "archive_sha256": ARCHIVE[1], "files": sources},
        "weights": {"path": "weights", "revision": MODEL, "files": {
            name: {"size": size, "sha256": digest} for name, (size, digest) in WEIGHTS.items()}},
        "python": "venv/Scripts/python.exe", "wheel_lock_sha256": sha(lock),
        "diagnostics": json.loads(probe), "generation_tested_by_installer": False,
    }
    temp = manifest.with_suffix(".json.partial")
    temp.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temp.replace(manifest)
    print(f"Runtime checks complete. Receipt: {manifest}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        sys.exit(f"TripoSG setup: {exc}")
