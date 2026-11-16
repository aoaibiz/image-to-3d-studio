"""Read-only local generation preflight. No network requests or credential reads."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SHA256 = "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee"
U2NET_SHA256 = "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491"

# Importing rembg starts pymatting's Numba compilation on a new installation.
# Check the installed session and native execution provider without compiling it.
RUNTIME_PROBE = """
from importlib.util import find_spec
from pathlib import Path
import torch, PIL, xatlas, trimesh, skimage, onnxruntime
rembg = find_spec('rembg')
if not rembg or not rembg.origin or not (Path(rembg.origin).parent / 'sessions/u2net.py').is_file():
    raise SystemExit('rembg U2Net session is missing')
if 'CPUExecutionProvider' not in onnxruntime.get_available_providers():
    raise SystemExit('ONNX Runtime CPU provider is missing')
print(torch.__version__)
"""


def runtime_python(runtime):
    return runtime / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def runtime_validator(runtime):
    return runtime / "validator" / ("gltf_validator.exe" if os.name == "nt" else "gltf_validator")


def executable_file(path):
    return path.is_file() and os.access(path, os.X_OK)


def _local_output(command):
    return subprocess.run(command, capture_output=True, check=True, text=True, timeout=15).stdout


def local_memory():
    if os.name == "nt":

        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                (name, ctypes.c_uint64)
                for name in (
                    "physical",
                    "available",
                    "page",
                    "free_page",
                    "virtual",
                    "free_virtual",
                    "extended",
                )
            ]

        state = Memory()
        state.length = ctypes.sizeof(state)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            return {"total_bytes": state.physical, "available_bytes": state.available}
    if sys.platform.startswith("linux"):
        try:
            lines = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
            fields = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in lines
                      if line.startswith(("MemTotal:", "MemAvailable:"))}
            return {"total_bytes": fields["MemTotal"], "available_bytes": fields.get("MemAvailable")}
        except (OSError, ValueError, KeyError, IndexError):
            return None
    if sys.platform == "darwin":
        try:
            total = int(_local_output(["/usr/sbin/sysctl", "-n", "hw.memsize"]))
            return {"total_bytes": total, "available_bytes": None}
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    return None


def gpu_inventory():
    try:
        if os.name == "nt":
            cmd = ("Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM "
                   "| ConvertTo-Json -Compress")
            return json.loads(_local_output(["powershell.exe", "-NoProfile", "-Command", cmd]))
        if sys.platform == "darwin":
            displays = json.loads(_local_output(["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"]))
            return [{"name": gpu.get("sppci_model") or gpu.get("_name"),
                     "reported_memory": gpu.get("spdisplays_vram") or gpu.get("spdisplays_vram_shared")}
                    for gpu in displays["SPDisplaysDataType"]]
        if sys.platform.startswith("linux") and shutil.which("nvidia-smi"):
            result = _local_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
            gpus = []
            for line in result.splitlines():
                name, memory = line.rsplit(",", 1)
                gpus.append({"name": name.strip(), "memory_bytes": int(memory.strip()) * 1024**2})
            return gpus
    except (OSError, KeyError, ValueError, subprocess.SubprocessError):
        pass
    if sys.platform.startswith("linux"):
        gpus = []
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
            if not card.name[4:].isdigit():
                continue
            device = card / "device"
            try:
                info = dict(line.split("=", 1) for line in (device / "uevent").read_text().splitlines()
                            if "=" in line)
                gpu = {"device": card.name, "driver": info.get("DRIVER"), "pci_id": info.get("PCI_ID")}
                vram = device / "mem_info_vram_total"
                gpu["memory_bytes"] = int(vram.read_text()) if vram.is_file() else None
                gpus.append(gpu)
            except (OSError, ValueError):
                continue
        return gpus or None
    return None


def verify_weight(path, expected, check_hash):
    if not path.is_file():
        return "missing"
    if not check_hash:
        return "unchecked"
    try:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        return "unreadable"
    return "verified" if digest == expected else "mismatch"


def diagnose(check_hash=False):
    runtime = ROOT / ".runtime"
    weights = runtime / "model" / "model.ckpt"
    foreground_weights = runtime / "u2net" / "u2net.onnx"
    python = runtime_python(runtime)
    requirements = [
        python,
        weights,
        runtime / "model" / "config.yaml",
        runtime / "model" / "dino-config.json",
        runtime / "TripoSR" / "tsr" / "system.py",
    ]
    missing = [
        str(path.relative_to(ROOT)) for path in requirements if not path.is_file()
    ]
    checksum = verify_weight(weights, WEIGHT_SHA256, check_hash)
    foreground_checksum = verify_weight(foreground_weights, U2NET_SHA256, check_hash)
    dependencies = "unchecked"
    if python.is_file():
        try:
            probe = subprocess.run(
                [
                    str(python),
                    "-I",
                    "-c",
                    RUNTIME_PROBE,
                ],
                capture_output=True,
                text=True,
                timeout=45,
            )
            dependencies = (
                probe.stdout.strip() if probe.returncode == 0 else "import_failed"
            )
        except (OSError, subprocess.SubprocessError):
            dependencies = "unavailable"
    alpha_ready = (
        not missing
        and checksum == "verified"
        and dependencies not in {"unchecked", "import_failed", "unavailable"}
    )
    if foreground_checksum == "missing":
        missing.append(str(foreground_weights.relative_to(ROOT)))
    photo_ready = alpha_ready and foreground_checksum == "verified"
    return {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "ram": local_memory(),
        "graphics": gpu_inventory(),
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "local_cpu": {
            "engine": "TripoSR",
            "files_missing": missing,
            "weight_verification": checksum,
            "foreground_weight_verification": foreground_checksum,
            "input_readiness": {
                "transparent": alpha_ready,
                "opaque_photo": photo_ready,
            },
            "torch_probe": dependencies,
            "preflight": "ready_to_try"
            if photo_ready
            else "setup_or_verification_required",
            "setup": "engine/setup.ps1" if os.name == "nt" else "sh engine/setup-posix.sh",
        },
        "trellis2": {
            "implemented": False,
            "official_requirement": "Linux; NVIDIA GPU with >=24 GB memory",
        },
        "tripo_cloud": {"implemented": False, "credit_balance": "not_queried"},
        "privacy": "Inference runs on this computer. Setup retrieves code, dependencies and weights.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-weights", action="store_true")
    args = parser.parse_args()
    print(json.dumps(diagnose(args.verify_weights), ensure_ascii=False, indent=2))
