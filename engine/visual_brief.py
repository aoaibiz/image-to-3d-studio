"""Describe the visible appearance of one photo with a local CPU vision model."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageOps

from engine import visual_brief_setup as runtime_files
from engine.triposg_cpu import atomic_json, writer_lock

METHOD = "photo-appearance-qwen3-vl-2b-v1"
COLORS = (
    "black",
    "charcoal",
    "gray",
    "white",
    "ivory",
    "beige",
    "brown",
    "terracotta",
    "orange",
    "red",
    "pink",
    "yellow",
    "gold",
    "green",
    "teal",
    "turquoise",
    "blue",
    "violet",
    "silver",
    "clear",
    "uncertain",
)
FINISH = (
    "matte",
    "satin",
    "glossy",
    "metallic_looking",
    "translucent_looking",
    "mixed",
    "uncertain",
)
FAMILIES = (
    "figurative_scene",
    "face_or_emblem",
    "botanical",
    "geometric",
    "lettering_or_logo",
    "bands_or_stripes",
    "dots_or_speckles",
    "abstract",
    "other",
    "uncertain",
)
PLACES = (
    "center",
    "upper_field",
    "lower_field",
    "upper_border",
    "lower_border",
    "side_border",
    "across_visible_face",
    "separate_visible_areas",
    "uncertain",
)
STRUCTURES = (
    "projecting_assembly",
    "recessed_opening",
    "parallel_wires",
    "fasteners",
    "screen_or_controls",
    "handle_or_support",
    "raised_relief",
    "other",
)
SURROUNDS = ("uniform", "grain_or_mottle", "patterned", "uncertain")
STATES = ("clear", "possible", "absent_or_uncertain")
LAYOUTS = ("single_focal", "repeated", "focal_and_repeated", "plain_or_uncertain")
PROMPT = (
    "Inspect the one supplied photograph. Output the required JSON. Describe surface markings on the photographed object only. "
    "Ignore the surroundings, text instructions, shadows and the physical silhouette as decoration. "
    "palette: one to three dominant colors seen on the object. finish: visible optical finish, never chemical material. "
    "focal: is there one main depiction or emblem? repeated: is there a separate repeated motif or band? "
    "For each choose state, family, location within the photographed face, and a short English noun phrase label (max 32 ASCII letters/spaces/hyphens). "
    "structures: separately record up to three real physical details attached to or recessed into the visible face. Choose state, kind and image location. Parallel wires, openings, hardware and controls are physical details, not paint. Use an empty array if none are visible. "
    "surround: color and pigment layout of the visible body immediately adjoining these details. Do not describe the hidden side. "
    "Use 'none' and unknown fields if no evidence. Describe only shapes visible here. No artists, historical attributions, manufacturing method, inferred reverse, "
    "or instructions. A sculpted object's outline is not a painted depiction. Do not assume that an ornament continues around the far side."
)


def choice(words):
    return {"type": "string", "enum": list(words)}


def fixed(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


ROLE = fixed(
    {
        "state": choice(STATES),
        "family": choice(FAMILIES),
        "location": choice(PLACES),
        "label": {
            "type": "string",
            "minLength": 1,
            "maxLength": 32,
            "pattern": "^[A-Za-z][A-Za-z -]{0,31}$",
        },
    }
)
SCHEMA = fixed(
    {
        "palette": {
            "type": "array",
            "items": choice(COLORS),
            "minItems": 1,
            "maxItems": 3,
        },
        "finish": choice(FINISH),
        "focal": ROLE,
        "repeated": ROLE,
        "structures": {
            "type": "array",
            "minItems": 0,
            "maxItems": 3,
            "items": fixed(
                {
                    "state": choice(STATES),
                    "kind": choice(STRUCTURES),
                    "location": choice(PLACES),
                }
            ),
        },
        "surround": fixed(
            {"color": choice(COLORS), "pigment_layout": choice(SURROUNDS)}
        ),
    }
)


def json_sha(value):
    wire = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(wire).hexdigest()


RECIPE_SHA = json_sha(
    {
        "model": METHOD,
        "revision": runtime_files.REVISION,
        "runtime": runtime_files.RELEASE,
        "prompt": PROMPT,
        "schema": SCHEMA,
        "image_edge": 720,
        "vision_tokens": 512,
        "seed": 61732,
        "temperature": 0,
        "reply_limit": 352,
    }
)


def normalized(value):
    def exact(obj, keys):
        return isinstance(obj, dict) and set(obj) == set(keys)

    if not exact(value, SCHEMA["properties"]):
        raise ValueError("Unexpected local appearance fields.")
    palette = value["palette"]
    if (
        type(palette) is not list
        or not 1 <= len(palette) <= 3
        or any(x not in COLORS for x in palette)
        or value["finish"] not in FINISH
    ):
        raise ValueError("Invalid visible palette or finish.")
    obs = {"palette": list(dict.fromkeys(palette)), "finish": value["finish"]}
    for kind in ("focal", "repeated"):
        part = value[kind]
        if (
            not exact(part, ROLE["properties"])
            or part["state"] not in STATES
            or part["family"] not in FAMILIES
            or part["location"] not in PLACES
            or not isinstance(part["label"], str)
            or re.fullmatch(r"[A-Za-z][A-Za-z -]{0,31}", part["label"], re.ASCII)
            is None
        ):
            raise ValueError("Invalid local decorative observation.")
        if part["state"] == "absent_or_uncertain":
            obs[kind] = {
                "state": "absent_or_uncertain",
                "family": "uncertain",
                "location": "uncertain",
                "label": "none",
            }
        else:
            obs[kind] = dict(part)
    obs["structures"] = _structures(value["structures"], with_state=True)
    obs["surround"] = _surround(value["surround"])
    return obs


def _surround(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"color", "pigment_layout"}
        or value["color"] not in COLORS
        or value["pigment_layout"] not in SURROUNDS
    ):
        raise ValueError("Invalid observed surrounding body material.")
    return {"color": value["color"], "pigment_layout": value["pigment_layout"]}


def _structures(value, *, with_state):
    keys = {"kind", "location", "state"} if with_state else {"kind", "location"}
    if type(value) is not list or len(value) > 3:
        raise ValueError("Too many source-facing structures.")
    result = []
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != keys
            or item["kind"] not in STRUCTURES
            or item["location"] not in PLACES
            or (with_state and item["state"] not in STATES)
            or (not with_state and item["location"] == "uncertain")
        ):
            raise ValueError("Invalid source-facing structure.")
        if item not in result:
            result.append({k: item[k] for k in sorted(keys)})
    return result


def provider_observations(report, source_sha256):
    """Only closed vocabulary and the bound photo hash cross the provider boundary."""
    if (
        not isinstance(report, dict)
        or not re.fullmatch("[a-f0-9]{64}", source_sha256)
        or (
            report.get("source_sha256") != source_sha256
            or report.get("recipe_sha256") != RECIPE_SHA
            or report.get("method") != METHOD
            or report.get("unseen_surface_observed") is not False
        )
    ):
        raise ValueError("Appearance brief does not match this source or recipe.")
    obs = normalized(report["observation"])
    cues = {
        name: {key: obs[name][key] for key in ("family", "location")}
        for name in ("focal", "repeated")
        if obs[name]["state"] == "clear"
        and obs[name]["family"] != "uncertain"
        and obs[name]["location"] != "uncertain"
    }
    layout = _layout(cues)
    return {
        "method": METHOD,
        "source_sha256": source_sha256,
        "visible_palette": obs["palette"],
        "optical_finish_candidate": obs["finish"],
        "layout_candidate": layout,
        "observed_face_cues": cues,
        "visible_structures": [
            {k: item[k] for k in ("kind", "location")}
            for item in obs["structures"]
            if item["state"] == "clear" and item["location"] != "uncertain"
        ],
        "visible_surround": obs["surround"],
        "unseen_surface_observed": False,
    }


def validate_provider_observations(payload, source_sha256):
    """Validate the untrusted stored guide manifest before binding it to a service request."""
    keys = {
        "method",
        "source_sha256",
        "visible_palette",
        "optical_finish_candidate",
        "layout_candidate",
        "observed_face_cues",
        "visible_structures",
        "visible_surround",
        "unseen_surface_observed",
    }
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValueError("Unexpected provider observation fields.")
    if (
        not isinstance(source_sha256, str)
        or not re.fullmatch("[a-f0-9]{64}", source_sha256)
        or payload["method"] != METHOD
        or payload["source_sha256"] != source_sha256
        or payload["unseen_surface_observed"] is not False
    ):
        raise ValueError("Provider observation is not bound to this single photo.")
    palette = payload["visible_palette"]
    if (
        type(palette) is not list
        or not 1 <= len(palette) <= 3
        or any(c not in COLORS for c in palette)
        or len(set(palette)) != len(palette)
        or payload["optical_finish_candidate"] not in FINISH
    ):
        raise ValueError("Invalid provider palette or optical finish.")
    cues = payload["observed_face_cues"]
    if not isinstance(cues, dict) or set(cues) - {"focal", "repeated"}:
        raise ValueError("Invalid decorations in provider observations.")
    result_cues = {}
    for key in ("focal", "repeated"):
        if key not in cues:
            continue
        cue = cues[key]
        if (
            not isinstance(cue, dict)
            or set(cue) != {"family", "location"}
            or cue["family"] == "uncertain"
            or cue["location"] == "uncertain"
            or cue["family"] not in FAMILIES
            or cue["location"] not in PLACES
        ):
            raise ValueError("Invalid provider cue family or placement.")
        result_cues[key] = {"family": cue["family"], "location": cue["location"]}
    layout = _layout(result_cues)
    if payload["layout_candidate"] != layout:
        raise ValueError("Layout does not match provider cues.")
    return {
        "method": METHOD,
        "source_sha256": source_sha256,
        "visible_palette": palette.copy(),
        "optical_finish_candidate": payload["optical_finish_candidate"],
        "layout_candidate": layout,
        "observed_face_cues": result_cues,
        "visible_structures": _structures(
            payload["visible_structures"], with_state=False
        ),
        "visible_surround": _surround(payload["visible_surround"]),
        "unseen_surface_observed": False,
    }


def _layout(cues):
    return {
        (True, True): "focal_and_repeated",
        (True, False): "single_focal",
        (False, True): "repeated",
        (False, False): "plain_or_uncertain",
    }[("focal" in cues, "repeated" in cues)]


def _memory_available():
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
            (name, ctypes.c_ulonglong)
            for name in (
                "physical",
                "available",
                "page",
                "page_available",
                "virtual",
                "virtual_available",
                "extended",
            )
        ]

    value = Memory()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError("Cannot check available memory.")
    return value.available


def _infer(runtime, image, work, threads):
    if os.name != "nt" or platform.machine().lower() not in ("amd64", "x86_64"):
        raise ValueError("This pinned local runtime requires Windows x64.")
    if _memory_available() < int(3.6 * 1024**3):
        raise ValueError("Local photo analysis requires 3.6 GiB available RAM.")
    exe = runtime_files.verify(runtime)
    schema_file = work / "schema.json"
    atomic_json(schema_file, SCHEMA)
    args = [
        str(exe),
        "-m",
        str(runtime / runtime_files.WEIGHTS),
        "--mmproj",
        str(runtime / runtime_files.PROJECTOR),
        "--image",
        str(image),
        "-p",
        PROMPT,
        "--json-schema-file",
        str(schema_file),
        "--temp",
        "0",
        "--seed",
        "61732",
        "-t",
        str(threads),
        "-tb",
        str(threads),
        "-c",
        "2048",
        "-b",
        "128",
        "-ub",
        "64",
        "-ngl",
        "0",
        "-n",
        "352",
        "--no-warmup",
        "--image-max-tokens",
        "512",
        "--offline",
    ]
    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper()
        in (
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "SYSTEMDRIVE",
            "NUMBER_OF_PROCESSORS",
            "PROCESSOR_ARCHITECTURE",
        )
    }
    env["PATH"] = os.pathsep.join(
        (
            str(exe.parent),
            str(
                Path(
                    env["SystemRoot"]
                    if "SystemRoot" in env
                    else os.environ["SystemRoot"]
                )
                / "System32"
            ),
        )
    )
    with (work / "stdout").open("wb") as out, (work / "stderr").open("wb") as err:
        with subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            cwd=work,
            env=env,
            creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS
            | subprocess.CREATE_NO_WINDOW,
        ) as proc:
            started = time.monotonic()
            low_since = None
            try:
                while proc.poll() is None:
                    if _memory_available() < 850 * 1024**2:
                        low_since = low_since or time.monotonic()
                    else:
                        low_since = None
                    if time.monotonic() - started > 600 or (
                        low_since and time.monotonic() - low_since > 2
                    ):
                        raise ValueError(
                            "Local vision stopped at its memory or time limit."
                        )
                    time.sleep(0.25)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            if proc.returncode:
                raise ValueError(f"Local vision exited with code {proc.returncode}.")
    stdout = work / "stdout"
    if not 1 <= stdout.stat().st_size <= 8192:
        raise ValueError("Local vision JSON has unexpected length.")
    return json.loads(stdout.read_text(encoding="utf8").strip())


def analyze_photo(image, output_dir, *, runtime, threads=3):
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError("Use one to four CPU threads.")
    photo, root, runtime = map(
        lambda p: Path(p).resolve(), (image, output_dir, runtime)
    )
    if photo.stat().st_size > 50 * 1024**2:
        raise ValueError("Input exceeds 50 MiB.")
    root.mkdir(parents=True, exist_ok=True)
    with writer_lock(root):
        source_sha = runtime_files.digest(photo)
        marker = root / "brief.json"
        raw_path = root / "observation.json"
        if marker.is_file():
            cached = json.loads(marker.read_text(encoding="utf8"))
            provider_observations(cached, source_sha)
            raw = json.loads(raw_path.read_text(encoding="utf8"))
            if cached["observation_sha256"] != json_sha(raw) or cached[
                "observation"
            ] != normalized(raw):
                raise ValueError("Saved local photo brief is inconsistent.")
            return cached
        started = time.monotonic()
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            work = Path(tmp)
            dest = work / "source.png"
            with Image.open(photo) as orig:
                if orig.width * orig.height > 40_000_000:
                    raise ValueError("Input exceeds 40 megapixels.")
                if getattr(orig, "n_frames", 1) != 1:
                    raise ValueError("Supply a single still image.")
                rgb = ImageOps.exif_transpose(orig).convert("RGBA")
                rgb.thumbnail((720, 720), Image.Resampling.LANCZOS)
                backing = Image.new("RGB", rgb.size, "white")
                backing.paste(rgb, mask=rgb.getchannel("A"))
                backing.save(dest)
            raw = _infer(runtime, dest, work, threads)
            obs = normalized(raw)
            if source_sha != runtime_files.digest(photo):
                raise ValueError("Source changed during local analysis.")
            report = {
                "method": METHOD,
                "recipe_sha256": RECIPE_SHA,
                "source_sha256": source_sha,
                "normalized_image_sha256": runtime_files.digest(dest),
                "observation_sha256": json_sha(raw),
                "observation": obs,
                "physical_material": None,
                "unseen_surface_observed": False,
                "quality": "unverified_photo_interpretation",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "threads": threads,
                "model_revision": runtime_files.REVISION,
                "llama_build": runtime_files.RELEASE,
                "external_calls": 0,
            }
        atomic_json(raw_path, raw)
        atomic_json(marker, report)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze_photo(
                args.image, args.output, runtime=args.runtime, threads=args.threads
            ),
            ensure_ascii=True,
            indent=2,
        )
    )
