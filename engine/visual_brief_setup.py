"""Pinned assets for the optional local photo-description engine."""

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REVISION = "ea6a11058182570be6436b9a2e4ee7f7b49f908d"
RELEASE = "b12420"
WEIGHTS = "Qwen3-VL-2B-Instruct-Q8_0.gguf"
PROJECTOR = "mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf"
EXECUTABLE = "llama-mtmd-cli.exe"
BUNDLE = f"llama-{RELEASE}-bin-win-cpu-x64.zip"
NATIVE_SHA = "f8ca923f2d3d77efe1465159c62ff47b1f9b0eb89bfb3ef242315ee689665d8c"
HF = f"https://huggingface.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF/resolve/{REVISION}"
ASSETS = {
    WEIGHTS: (
        1834427296,
        "b7802e29f71a9e5b5e3f83f613df898a2204342dcea71a231ea501d481813c39",
        f"{HF}/{WEIGHTS}",
    ),
    PROJECTOR: (
        445053056,
        "69066c8f279ec85ff48ab4059f6ebba0d2932ca57667f2bbdac7d9805bca9e7b",
        f"{HF}/{PROJECTOR}",
    ),
    BUNDLE: (
        25143187,
        "e8ece49057f20a7a884527caaae82c69ecf09656b6f2dd706d376f74572dfe4c",
        f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}/{BUNDLE}",
    ),
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binary_tree(bin_dir):
    selected = [
        p
        for p in bin_dir.iterdir()
        if p.name.lower().endswith(".dll") or p.name.lower() == EXECUTABLE
    ]
    h = hashlib.sha256()
    for path in sorted(selected, key=lambda p: p.name.lower()):
        h.update(f"{path.name.lower()} {digest(path)}\n".encode())
    return h.hexdigest()


def verify(runtime):
    runtime = Path(runtime)
    for name in (WEIGHTS, PROJECTOR):
        path = runtime / name
        size, expected, _ = ASSETS[name]
        if (
            not path.is_file()
            or path.stat().st_size != size
            or digest(path) != expected
        ):
            raise ValueError(f"Local visual brief needs the pinned {name}.")
    if binary_tree(runtime / "bin") != NATIVE_SHA:
        raise ValueError("Local visual brief native binary checksums differ.")
    return runtime / "bin" / EXECUTABLE


def download(name, dest):
    size, expected, url = ASSETS[name]
    with urllib.request.urlopen(url, timeout=90) as response, dest.open("xb") as output:
        written = 0
        while chunk := response.read(1024**2):
            written += len(chunk)
            if written > size:
                raise ValueError(f"Oversized asset {name}.")
            output.write(chunk)
    if written != size or digest(dest) != expected:
        raise ValueError(f"Asset checksum mismatch: {name}.")


def install(runtime, *, assets_dir=None):
    """Fresh atomic directory; an existing, different environment is never overwritten."""
    runtime = Path(runtime).resolve()
    if runtime.exists():
        verify(runtime)
        return runtime
    runtime.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime.parent) as tmp:
        stage = Path(tmp) / "runtime"
        stage.mkdir()
        (stage / "bin").mkdir()
        for name in ASSETS:
            if assets_dir is None:
                download(name, stage / name)
            else:
                src = Path(assets_dir) / name
                size, expected, _ = ASSETS[name]
                if src.stat().st_size != size or digest(src) != expected:
                    raise ValueError(f"Offline asset checksum mismatch: {name}.")
                shutil.copyfile(src, stage / name)
        with zipfile.ZipFile(stage / BUNDLE) as z:
            for item in z.infolist():
                path = Path(item.filename)
                if item.is_dir() or not (
                    path.suffix.lower() == ".dll" or path.name.lower() == EXECUTABLE
                ):
                    continue
                # The zip's fingerprint is pinned, and only immediate filenames are extracted.
                if len(path.parts) != 1:
                    raise ValueError("Unexpected native bundle structure.")
                with z.open(item) as src, (stage / "bin" / path.name).open("xb") as out:
                    shutil.copyfileobj(src, out)
        verify(stage)
        (stage / BUNDLE).unlink()
        os.rename(stage, runtime)
    return runtime


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument(
        "--assets-dir",
        type=Path,
        help="Use pinned previously downloaded files offline.",
    )
    args = parser.parse_args()
    print(install(args.runtime, assets_dir=args.assets_dir))
