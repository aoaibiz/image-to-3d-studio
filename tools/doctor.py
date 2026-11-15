"""Read-only local generation preflight. No network requests or credential reads."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SHA256 = "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee"


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
    return None


def gpu_inventory():
    if os.name != "nt":
        return None
    cmd = (
        "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM "
        "| ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        return json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def diagnose(check_hash=False):
    runtime = ROOT / ".runtime"
    weights = runtime / "model" / "model.ckpt"
    python = (
        runtime / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    requirements = [
        python,
        weights,
        runtime / "model" / "config.yaml",
        runtime / "TripoSR" / "tsr" / "system.py",
    ]
    missing = [
        str(path.relative_to(ROOT)) for path in requirements if not path.is_file()
    ]
    checksum = "unchecked"
    if weights.is_file() and check_hash:
        with weights.open("rb") as stream:
            checksum = (
                "verified"
                if hashlib.file_digest(stream, "sha256").hexdigest() == WEIGHT_SHA256
                else "mismatch"
            )
    dependencies = "unchecked"
    if python.is_file():
        try:
            probe = subprocess.run(
                [
                    str(python),
                    "-c",
                    "import torch, PIL, xatlas, trimesh, skimage; print(torch.__version__)",
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
    eligible = (
        not missing
        and checksum == "verified"
        and dependencies not in {"unchecked", "import_failed", "unavailable"}
    )
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
            "torch_probe": dependencies,
            "preflight": "ready_to_try"
            if eligible
            else "setup_or_verification_required",
            "setup": "engine/setup.ps1",
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
