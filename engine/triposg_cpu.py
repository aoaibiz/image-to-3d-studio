"""Staged, offline CPU shape inference using the official TripoSG checkpoint."""

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


SOURCE_REVISION = "fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c"
WEIGHTS_REVISION = "2c1c516d22d58db486a058d98d31bb6177344e06"
WEIGHT_SHA256 = {
    "image_encoder_dinov2/model.safetensors": "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23",
    "transformer/diffusion_pytorch_model.safetensors": "9192b5923f7b605b394192809aa2ceb73bf0f4009674d8e3b999b45bb97d4bf2",
    "vae/diffusion_pytorch_model.safetensors": "a2e667c24927a5a35e5f19fcb4c75890e9399aa966b6db8131d7df733a750c8b",
}
SOURCE_TREE = "7de3d8ce787544d94246daeb8c13843c34dff7d78404535fdbd9c09326748673"
CONFIG_TREE = "a3058aab39def64c283e2b36fa43e3d5cf68a6ec5c2d39f17313772a3058e14c"
STAGE_START_GIB = 6
MEMORY_WAIT_SECONDS = 600


def emit(event, **details):
    print(json.dumps({"event": event, **details}, ensure_ascii=True), flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def tree_digest(base, paths):
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(base).as_posix()):
        h.update(path.relative_to(base).as_posix().encode() + b"\0")
        h.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def verify_runtime(runtime):
    source, weights = runtime / "source", runtime / "weights"
    if tree_digest(source, (source / "triposg").rglob("*.py")) != SOURCE_TREE:
        raise ValueError("TripoSG本体が指定バージョンと一致しません。セットアップを再実行してください。")
    configurations = ("model_index.json", "feature_extractor_dinov2/preprocessor_config.json",
                      "image_encoder_dinov2/config.json", "scheduler/scheduler_config.json",
                      "transformer/config.json", "vae/config.json")
    if tree_digest(weights, [weights / name for name in configurations]) != CONFIG_TREE:
        raise ValueError("TripoSGのモデル設定が指定バージョンと一致しません。")
    for name, expected in WEIGHT_SHA256.items():
        if digest(weights / name) != expected:
            raise ValueError(f"TripoSGの重みが破損しています。再取得してください ({name})")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@contextlib.contextmanager
def writer_lock(work):
    """The kernel releases this lock when the owner exits, including after a crash."""
    stream = (work / "run.lock").open("a+b")
    try:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"0")
                stream.flush()
                stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError("このジョブはすでに実行中です。既存ジョブに再接続してください。") from exc
    try:
        yield
    finally:
        stream.close()


def receipt_path(path):
    return path.with_name(path.name + ".json")


def valid_artifact(path, job_key, **expected):
    try:
        meta = read_json(receipt_path(path))
        if (isinstance(meta, dict) and meta.get("job_key") == job_key
                and all(meta.get(key) == value for key, value in expected.items())
                and meta.get("sha256") == digest(path)):
            return meta
    except (OSError, KeyError, ValueError):
        pass
    return None


def seal(path, job_key, **extra):
    info = {"sha256": digest(path), "job_key": job_key, **extra}
    atomic_json(receipt_path(path), info)
    return info


def save_tensor(tensor, path, job_key, **extra):
    from safetensors.torch import save_file
    import torch

    if not torch.isfinite(tensor).all():
        raise ValueError("形状推論に非有限値が含まれています。")
    tmp = path.with_suffix(".partial")
    save_file({"tensor": tensor.detach().contiguous()}, str(tmp))
    os.replace(tmp, path)
    seal(path, job_key, **extra)


def read_tensor(path):
    from safetensors.torch import load_file

    return load_file(str(path), device="cpu")["tensor"]


def latest_step(work, job_key):
    candidates = []
    for slot in ("flow-a.safetensors", "flow-b.safetensors"):
        path = work / slot
        meta = valid_artifact(path, job_key)
        if meta and isinstance(meta.get("step"), int) and 1 <= meta["step"] <= 50:
            candidates.append((meta["step"], path))
    return max(candidates, default=(0, None))


def prepare_input(source):
    import numpy as np
    from PIL import Image, ImageOps
    from scipy.ndimage import binary_propagation

    with Image.open(source) as opened:
        if opened.width * opened.height > 40_000_000:
            raise ValueError("入力は4000万画素以内に縮小してください。")
        image = ImageOps.exif_transpose(opened).convert("RGBA")
    alpha = image.getchannel("A")
    if alpha.getextrema()[0] == 255:
        # The normal preprocessing path also accepts its square, neutral-background RGB.
        rgb = np.asarray(image, dtype=np.int16)[:, :, :3]
        border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
        bg = np.median(border, axis=0).astype(np.float32)
        if np.mean(np.max(abs(border - bg), axis=1) < 12) < 0.92:
            raise ValueError("背景分離済みのRGB画像、または透過PNGを渡してください。")
        matches = np.max(abs(rgb - bg), axis=2) < 5
        edge = np.zeros_like(matches)
        edge[[0, -1]] = matches[[0, -1]]
        edge[:, [0, -1]] = matches[:, [0, -1]]
        alpha = Image.fromarray((~binary_propagation(edge, mask=matches)).astype("uint8") * 255)
        image.putalpha(alpha)
    # Low-alpha background noise must not determine the subject's framing.
    # Retain the original soft matte when compositing the resulting viewport.
    box = alpha.point(lambda value: 255 if value >= 150 else 0).getbbox()
    if not box:
        raise ValueError("入力画像に十分な不透明度の前景がありません。")
    width, height = box[2] - box[0], box[3] - box[1]
    side = max(1, int(max(width, height) * 1.2))
    left = box[0] - (side - width) // 2
    top = box[1] - (side - height) // 2
    region = (max(0, left), max(0, top), min(image.width, left + side), min(image.height, top + side))
    patch = image.crop(region)
    # A very wide photograph can have few pixels but an enormous square bounding canvas.
    # Keep ordinary images pixel-identical and downsample the intersecting strip before padding.
    canvas_side = side if side <= 4096 else 768
    scale = canvas_side / side
    if scale < 1:
        size = tuple(max(1, round(value * scale)) for value in patch.size)
        patch = patch.resize(size, Image.Resampling.BICUBIC)
    padded = Image.new("RGB", (canvas_side, canvas_side), "white")
    offset = (round((region[0] - left) * scale), round((region[1] - top) * scale))
    padded.paste(patch, offset, patch.getchannel("A"))
    padded.thumbnail((768, 768), Image.Resampling.BICUBIC)
    return padded


class RunnerVersionMismatch(ValueError):
    """An unfinished job belongs to another installed provider version."""


class RamShortfall(MemoryError):
    def __init__(self, available_gib, required_gib):
        self.available_gib = available_gib
        self.required_gib = required_gib
        super().__init__(f"空きRAMが{available_gib:.2f} GiBです。{required_gib:g} GiB以上で同じジョブを再開できます。")


class MemoryWaitExpired(RamShortfall):
    def __str__(self):
        return f"空きRAM待機が{MEMORY_WAIT_SECONDS}秒に達しました。{super().__str__()}"


def available_ram_gib():
    import psutil

    return psutil.virtual_memory().available / 1024**3


def guard_ram(gib):
    available = available_ram_gib()
    if available < gib:
        raise RamShortfall(available, gib)


def load_network(cls, weights, name):
    """Stream one official FP32 tensor at a time into CPU bfloat16 parameters."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    import torch

    folder = weights / name
    with init_empty_weights(include_buffers=False):
        network = cls.from_config(cls.load_config(str(folder)))
    path = folder / "diffusion_pytorch_model.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as source:
        keys = source.keys()
    if set(dict(network.named_parameters())) - set(keys):
        raise ValueError(f"モデル {name} の重みが欠落しています。")
    for i, key in enumerate(keys):
        with safe_open(str(path), framework="pt", device="cpu") as source:
            set_module_tensor_to_device(network, key, "cpu", value=source.get_tensor(key), dtype=torch.bfloat16)
        if i % 100 == 0:
            guard_ram(1.75)
    return network.eval()


def encode(work, weights, job_key):
    import torch
    from transformers import BitImageProcessor, Dinov2Model

    image = prepare_input(work / "input.image")
    image.save(work / "condition.png")
    processor = BitImageProcessor.from_pretrained(str(weights / "feature_extractor_dinov2"), local_files_only=True)
    network = Dinov2Model.from_pretrained(str(weights / "image_encoder_dinov2"), torch_dtype=torch.bfloat16,
                                          low_cpu_mem_usage=True, use_safetensors=True, local_files_only=True).eval()
    pixels = processor(image, return_tensors="pt").pixel_values.to(torch.bfloat16)
    embedding = network(pixels).last_hidden_state
    save_tensor(torch.cat([torch.zeros_like(embedding), embedding]), work / "condition.safetensors", job_key)


def denoise(work, weights, job_key, settings):
    import torch
    from triposg.models.transformers.triposg_transformer import TripoSGDiTModel
    from triposg.schedulers.scheduling_rectified_flow import RectifiedFlowScheduler

    scheduler = RectifiedFlowScheduler.from_pretrained(str(weights / "scheduler"), local_files_only=True)
    scheduler.set_timesteps(settings["steps"], device="cpu")
    resume, checkpoint = latest_step(work, job_key)
    if checkpoint:
        noise = read_tensor(checkpoint)
        emit("resumed", stage="denoise", step=resume, total=settings["steps"])
    else:
        rng = torch.Generator(device="cpu").manual_seed(settings["seed"])
        noise = torch.randn((1, settings["tokens"], 64), generator=rng, dtype=torch.bfloat16)
    if resume < settings["steps"]:
        model = load_network(TripoSGDiTModel, weights, "transformer")
        embedding = read_tensor(work / "condition.safetensors")
        for i in range(resume, settings["steps"]):
            guard_ram(1.75)
            t = scheduler.timesteps[i]
            pred = model(torch.cat([noise, noise]), t.expand(2), encoder_hidden_states=embedding, return_dict=False)[0]
            negative, positive = pred.chunk(2)
            pred = negative + settings["guidance"] * (positive - negative)
            noise = scheduler.step(pred, t, noise, return_dict=False)[0].to(torch.bfloat16)
            name = "flow-a.safetensors" if i % 2 == 0 else "flow-b.safetensors"
            save_tensor(noise, work / name, job_key, step=i + 1)
            emit("progress", stage="denoise", step=i + 1, total=settings["steps"])
    save_tensor(noise, work / "shape.safetensors", job_key)


def subdivide(field):
    """Trilinear refinement with exactly coincident parent and child grid vertices."""
    import numpy as np

    side = field.shape[0] * 2 - 1
    up = np.empty((side,) * 3, dtype=np.float32)
    up[::2, ::2, ::2] = field
    up[1::2, ::2, ::2] = (field[:-1] + field[1:]) * 0.5
    up[:, 1::2, ::2] = (up[:, :-2:2, ::2] + up[:, 2::2, ::2]) * 0.5
    up[:, :, 1::2] = (up[:, :, :-2:2] + up[:, :, 2::2]) * 0.5
    return up


def extract(work, weights, job_key, settings):
    import numpy as np
    import torch
    import trimesh
    from scipy import ndimage
    from skimage.measure import marching_cubes
    from triposg.models.autoencoders.autoencoder_kl_triposg import TripoSGVAEModel

    vae = load_network(TripoSGVAEModel, weights, "vae")
    z = vae.post_quant(read_tensor(work / "shape.safetensors"))
    limit = 1.005
    kv = None
    field = None
    # Begin with a dense 65³ field so isolated parts and narrow gaps can seed refinement.
    for level in (64, 128, 256):
        field_path = work / f"field-{level}.npy"
        if valid_artifact(field_path, job_key, cells=level):
            field = np.load(field_path, allow_pickle=False)
        else:
            if field is None:
                field = np.empty((level + 1,) * 3, dtype=np.float32)
                indices = np.indices(field.shape).reshape(3, -1).T
            else:
                # Re-query the learned SDF on both sides of the coarse zero set.
                lower = ndimage.minimum_filter(field, 3)
                upper = ndimage.maximum_filter(field, 3)
                halo = ndimage.binary_dilation((lower <= 0) & (upper >= 0))
                parent = np.arange(level + 1) // 2
                near = halo[np.ix_(parent, parent, parent)]
                near[::2, ::2, ::2] = False
                indices = np.argwhere(near)
                field = subdivide(field)
            for first in range(0, len(indices), 2048):
                guard_ram(1.75)
                batch = indices[first:first + 2048]
                xyz = torch.from_numpy(batch.astype(np.float32) * (2 * limit / level) - limit)[None]
                queries = vae.embedder(xyz).to(z.dtype)
                scores, kv = vae.decoder(z, queries, kv)
                field[tuple(batch.T)] = scores[0, :, 0].float().numpy()
                if first % 131072 == 0:
                    emit("progress", stage="surface", cells=level, queried=min(first + 2048, len(indices)),
                         total=len(indices))
            if not np.isfinite(field).all() or not field.min() < 0 < field.max():
                raise ValueError("推定した距離場に立体の境界面がありません。")
            tmp = field_path.with_suffix(".part")
            with tmp.open("wb") as dest:
                np.save(dest, field, allow_pickle=False)
            os.replace(tmp, field_path)
            seal(field_path, job_key, cells=level)
        if level == settings["cells"]:
            break
    vertices, faces, _, _ = marching_cubes(field, level=0, spacing=(2 * limit / level,) * 3, method="lewiner")
    mesh = trimesh.Trimesh(vertices - limit, faces, process=True)
    if len(mesh.faces) < 8 or np.min(mesh.extents) <= 0:
        raise ValueError("推定結果から立体メッシュを抽出できませんでした。")
    if mesh.volume < 0:
        mesh.invert()
    result = work / "raw-shape.glb"
    tmp = result.with_suffix(".part")
    tmp.write_bytes(mesh.export(file_type="glb"))
    os.replace(tmp, result)
    seal(result, job_key, cells=level, faces=len(mesh.faces), vertices=len(mesh.vertices),
         watertight=bool(mesh.is_watertight), appearance="untextured", visual_quality="needs_review")


def watch_supervisor(pipe_fd, on_lost=None):
    """Notice anonymous-pipe EOF without locking Windows' CRT during native DLL imports."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        import msvcrt

        peek = ctypes.WinDLL("kernel32", use_last_error=True).PeekNamedPipe
        peek.argtypes = (wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                         ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
                         ctypes.POINTER(wintypes.DWORD))
        peek.restype = wintypes.BOOL
        handle = msvcrt.get_osfhandle(pipe_fd)
        while peek(handle, None, 0, None, None, None):
            time.sleep(0.4)
    else:
        import select

        select.select([pipe_fd], [], [])
    try:
        if on_lost is not None:
            on_lost()
    finally:
        os._exit(130)


def stage_worker(stage, runtime, work):
    identity = read_json(work / "identity.json")
    if sys.stdin.buffer.readline().strip() != identity["job_key"].encode("ascii"):
        raise RuntimeError("ステージはジョブ管理プロセスから起動してください。")
    start = time.monotonic()
    def parent_lost():
        record_stage(work, stage, identity["job_key"], start, "interrupted", 130, reason="parent_lost")
    threading.Thread(target=watch_supervisor, args=(sys.stdin.fileno(), parent_lost), daemon=True).start()
    sys.path.insert(0, str(runtime / "source"))
    guard_ram(STAGE_START_GIB)
    import torch

    settings = identity["settings"]
    torch.set_num_threads(settings["threads"])
    torch.set_num_interop_threads(1)
    with torch.inference_mode():
        weights = runtime / "weights"
        key = identity["job_key"]
        if stage == "encode":
            encode(work, weights, key)
        elif stage == "denoise":
            denoise(work, weights, key, settings)
        else:
            extract(work, weights, key, settings)


def record_stage(work, stage, job_key, start, outcome, code, **details):
    record = {"stage": stage, "outcome": outcome, "seconds": round(time.monotonic() - start, 3),
              "exit_code": code, "job_key": job_key, "finished_unix": time.time(), **details}
    if outcome != "completed":
        # This exit-time sample is separate from the worker's pre-exit memory guard.
        with contextlib.suppress(ImportError, OSError, MemoryError):
            record["available_gib_at_exit"] = round(available_ram_gib(), 3)
    with (work / "timings.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def relay_worker(stream, stage, failure):
    """Forward the local worker's typed events without persisting arbitrary process output."""
    for line in stream:
        try:
            msg = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if not isinstance(msg, dict) or msg.get("stage") != stage:
            continue
        event = msg.get("event")
        if event in ("progress", "resumed"):
            counts = {k: msg[k] for k in ("step", "total", "cells", "queried")
                      if type(msg.get(k)) is int and msg[k] >= 0}
            emit(event, stage=stage, **counts)
        elif event == "resource_shortfall":
            values = [msg.get(k) for k in ("available_gib", "required_gib")]
            if all(type(v) in (int, float) and math.isfinite(v) for v in values) and 0 <= values[0] <= values[1]:
                failure.update(reason=event, available_gib=values[0], required_gib=values[1])
                emit(event, stage=stage, available_gib=values[0], required_gib=values[1])
        elif event == "worker_failure" and msg.get("reason") in ("dependency_unavailable", "worker_error"):
            failure["reason"] = msg["reason"]


def supervise_stage(stage, runtime, work, interpreter, cancel_file, settings, job_key):
    if cancel_file is not None and cancel_file.exists():
        raise InterruptedError("ジョブはキャンセルされています。途中状態を保存しました。")
    command = [str(interpreter), str(Path(__file__).resolve()), "_stage", stage, str(runtime), str(work)]
    env = dict(os.environ, OMP_NUM_THREADS=str(settings["threads"]), OPENBLAS_NUM_THREADS="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONNOUSERSITE="1")
    emit("stage", stage=stage)
    start = time.monotonic()
    outcome = "interrupted"
    failure = {}
    code = None
    child = None
    reader = None
    try:
        try:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=env)
        except OSError as exc:
            failure["reason"] = "worker_launch_failed"
            raise RuntimeError("形状生成の子プロセスを開始できません。ランタイムを確認してください。") from exc
        reader = threading.Thread(target=relay_worker, args=(child.stdout, stage, failure), daemon=True)
        reader.start()
        child.stdin.write(job_key.encode("ascii") + b"\n")
        child.stdin.flush()
        while True:
            try:
                code = child.wait(timeout=0.75)
                reader.join()
                if code:
                    if code == 75 and failure.get("reason") == "resource_shortfall":
                        raise RamShortfall(failure["available_gib"], failure["required_gib"])
                    failure.setdefault("reason", "native_exit" if code < 0 or code >= 0xC0000000 else "worker_exit")
                    raise RuntimeError(f"工程 {stage} が中断しました（{failure['reason']}・終了コード {code}）。"
                                       "同じジョブから再開できます。")
                outcome = "completed"
                break
            except subprocess.TimeoutExpired:
                if cancel_file is not None and cancel_file.exists():
                    outcome = "cancelled"
                    failure["reason"] = "cancelled"
                    raise InterruptedError("ジョブをキャンセルしました。途中状態を保存しました。")
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
            code = child.wait()
            with contextlib.suppress(OSError):
                child.stdin.close()
            if reader is not None:
                reader.join()
            child.stdout.close()
        if outcome == "interrupted":
            failure.setdefault("reason", "worker_exit")
        record_stage(work, stage, job_key, start, outcome, code, **failure)


def wait_for_memory(stage, shortfall, cancel_file):
    # The child checks before importing torch; leave 0.5 GiB margin for startup fluctuations.
    required = max(STAGE_START_GIB, shortfall.required_gib) + 0.5
    deadline = time.monotonic() + MEMORY_WAIT_SECONDS
    emit("waiting_for_memory", stage=stage, available_gib=shortfall.available_gib,
         required_gib=required, max_wait_seconds=MEMORY_WAIT_SECONDS)
    consecutive = 0
    while True:
        if cancel_file is not None and cancel_file.exists():
            raise InterruptedError("空きRAMの待機をキャンセルしました。同じジョブから再開できます。")
        available = available_ram_gib()
        if time.monotonic() >= deadline:
            raise MemoryWaitExpired(available, required)
        consecutive = consecutive + 1 if available >= required else 0
        if consecutive >= 2:
            return
        time.sleep(min(2, max(0, deadline - time.monotonic())))


def run_stage(stage, runtime, work, interpreter, cancel_file, settings, job_key):
    try:
        supervise_stage(stage, runtime, work, interpreter, cancel_file, settings, job_key)
    except RamShortfall as exc:
        start = time.monotonic()
        try:
            wait_for_memory(stage, exc, cancel_file)
        except (MemoryWaitExpired, InterruptedError) as stop:
            if isinstance(stop, MemoryWaitExpired):
                record_stage(work, stage, job_key, start, "paused", None, reason="memory_wait_timeout",
                             available_gib=stop.available_gib, required_gib=stop.required_gib)
            else:
                record_stage(work, stage, job_key, start, "paused", None, reason="cancelled")
            raise
        supervise_stage(stage, runtime, work, interpreter, cancel_file, settings, job_key)


def runtime_python(runtime):
    rel = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    interpreter = runtime / "venv" / rel
    if not interpreter.is_file():
        raise FileNotFoundError("TripoSG用Pythonが見つかりません。オプションのセットアップを実行してください。")
    return interpreter


def installed_versions(interpreter):
    packages = ("torch", "diffusers", "transformers", "accelerate", "safetensors", "scikit-image", "trimesh")
    command = "import importlib.metadata as m,json,sys;print(json.dumps({p:m.version(p) for p in sys.argv[1:]}))"
    result = subprocess.run([str(interpreter), "-I", "-c", command, *packages],
                            check=True, capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)


def stage_receipt(stage, artifact, job_key, cells):
    if stage == "surface":
        if not valid_artifact(artifact.parent / f"field-{cells}.npy", job_key, cells=cells):
            return None
        return valid_artifact(artifact, job_key, cells=cells)
    return valid_artifact(artifact, job_key)


def generate(image, work, output, runtime, cancel_file=None, seed=42, cells=256, threads=4):
    """Return the hash-bound raw geometry receipt. Never assert that its appearance passed review."""
    image, work, output, runtime = (Path(value).resolve() for value in (image, work, output, runtime))
    cancel_file = Path(cancel_file).resolve() if cancel_file else None
    if output == image or output.suffix.lower() != ".glb":
        raise ValueError("入力と別の.glb出力先を指定してください。")
    if cells not in (128, 256) or not 1 <= threads <= 8 or not 0 <= seed < 2**32:
        raise ValueError("形状生成の設定値が範囲外です。")
    work.mkdir(parents=True, exist_ok=True)
    with writer_lock(work):
        if cancel_file is not None and cancel_file.exists():
            raise InterruptedError("ジョブはキャンセルされています。再開時はキャンセル状態を解除してください。")
        interpreter = runtime_python(runtime)
        emit("stage", stage="verify")
        verify_runtime(runtime)
        settings = {"steps": 50, "tokens": 2048, "guidance": 7.0, "seed": seed, "cells": cells, "threads": threads}
        identity = {"engine": "TripoSG", "source_revision": SOURCE_REVISION, "weights_revision": WEIGHTS_REVISION,
                    "weights_sha256": WEIGHT_SHA256, "source_tree": SOURCE_TREE, "configuration_tree": CONFIG_TREE,
                    "runner_sha256": digest(Path(__file__).resolve()), "input_sha256": digest(image), "settings": settings,
                    "versions": installed_versions(interpreter), "device": "cpu", "dtype": "bfloat16"}
        job_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        identity["job_key"] = job_key
        state = work / "identity.json"
        if state.is_file():
            previous = read_json(state)
            if previous != identity:
                if isinstance(previous, dict) and previous.get("runner_sha256") != identity["runner_sha256"]:
                    raise RunnerVersionMismatch("このジョブの再開には、以前のリリースの triposg_cpu.py "
                                     f"(SHA256={previous.get('runner_sha256')}) と元の入力・設定・環境が必要です。"
                                     "このバージョンで新規生成する場合は別の作業場所を指定してください。")
                raise ValueError("作業場所に別の入力・設定・環境のジョブがあります。新しい作業場所を指定してください。")
        atomic_json(state, identity)
        snapshot = work / "input.image"
        if not snapshot.is_file() or digest(snapshot) != identity["input_sha256"]:
            tmp = snapshot.with_suffix(".part")
            shutil.copyfile(image, tmp)
            if digest(tmp) != identity["input_sha256"]:
                raise ValueError("読込中に入力画像が変更されました。")
            os.replace(tmp, snapshot)
        stages = (("encode", "condition.safetensors"), ("denoise", "shape.safetensors"), ("surface", "raw-shape.glb"))
        for stage, name in stages:
            artifact = work / name
            if cancel_file is not None and cancel_file.exists():
                raise InterruptedError("ジョブをキャンセルしました。途中状態を保存しました。")
            result = stage_receipt(stage, artifact, job_key, cells)
            if not result:
                run_stage(stage, runtime, work, interpreter, cancel_file, settings, job_key)
                result = stage_receipt(stage, artifact, job_key, cells)
                if not result:
                    raise RuntimeError(f"工程 {stage} の成果物を検証できませんでした。")
            else:
                emit("resumed", stage=stage)
        raw = work / "raw-shape.glb"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output != raw:
            tmp = output.with_name(f".{output.name}.{job_key[:16]}.part")
            shutil.copyfile(raw, tmp)
            os.replace(tmp, output)
        emit("complete", **result, file=str(output))
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    run = commands.add_parser("generate")
    for arg in ("image", "work-dir", "output", "runtime"):
        run.add_argument("--" + arg, required=True, type=Path)
    run.add_argument("--cancel-file", type=Path)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--cells", type=int, choices=(128, 256), default=256)
    run.add_argument("--threads", type=int, default=4)
    worker = commands.add_parser("_stage", help=argparse.SUPPRESS)
    worker.add_argument("stage", choices=("encode", "denoise", "surface"))
    worker.add_argument("runtime", type=Path)
    worker.add_argument("work", type=Path)
    args = ap.parse_args()
    if args.command == "_stage":
        try:
            stage_worker(args.stage, args.runtime, args.work)
        except RamShortfall as exc:
            emit("resource_shortfall", stage=args.stage, available_gib=exc.available_gib, required_gib=exc.required_gib)
            raise SystemExit(75)
        except (ImportError, OSError) as exc:
            missing = isinstance(exc, ImportError) or getattr(exc, "winerror", None) in (126, 127, 193, 1114)
            emit("worker_failure", stage=args.stage, reason="dependency_unavailable" if missing else "worker_error")
            raise SystemExit(1)
        except Exception:
            emit("worker_failure", stage=args.stage, reason="worker_error")
            raise SystemExit(1)
    else:
        generate(args.image, args.work_dir, args.output, args.runtime, args.cancel_file, args.seed, args.cells, args.threads)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, InterruptedError):
        emit("cancelled", resume="同じ入力・設定・作業場所で再開できます。")
        raise SystemExit(130)
    except Exception as exc:
        emit("error", message=str(exc), resume="原因を解消して同じジョブを再開してください。")
        raise SystemExit(1)
