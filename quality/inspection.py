"""Conservative GLB geometry inspection; appearance always requires human review."""

import argparse, hashlib, json, math, struct, subprocess, sys
from pathlib import Path


def load_glb(path):
    raw = Path(path).read_bytes()
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise ValueError("not a GLB")
    _, version, length = struct.unpack_from("<4sII", raw)
    if version != 2 or length != len(raw):
        raise ValueError("invalid GLB header")
    chunks, offset = {}, 12
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise ValueError("truncated chunk header")
        size, kind = struct.unpack_from("<II", raw, offset)
        offset += 8
        if offset + size > len(raw):
            raise ValueError("truncated chunk")
        chunks[kind] = raw[offset : offset + size]
        offset += size
    doc = json.loads(chunks[0x4E4F534A].decode("utf-8").rstrip(" \x00"))
    return raw, doc, chunks.get(0x004E4942, b"")


def multiply(a, b):
    return [
        sum(a[k * 4 + r] * b[c * 4 + k] for k in range(4))
        for c in range(4)
        for r in range(4)
    ]


def matrix(node):
    if "matrix" in node:
        return node["matrix"]
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    sx, sy, sz = node.get("scale", [1, 1, 1])
    tx, ty, tz = node.get("translation", [0, 0, 0])
    return [
        (1 - 2 * (y * y + z * z)) * sx,
        2 * (x * y + z * w) * sx,
        2 * (x * z - y * w) * sx,
        0,
        2 * (x * y - z * w) * sy,
        (1 - 2 * (x * x + z * z)) * sy,
        2 * (y * z + x * w) * sy,
        0,
        2 * (x * z + y * w) * sz,
        2 * (y * z - x * w) * sz,
        (1 - 2 * (x * x + y * y)) * sz,
        0,
        tx,
        ty,
        tz,
        1,
    ]


def accessor(doc, binary, index):
    a = doc["accessors"][index]
    if "sparse" in a or "bufferView" not in a:
        raise ValueError("sparse or absent accessor unsupported")
    v = doc["bufferViews"][a["bufferView"]]
    if v.get("buffer", 0) != 0:
        raise ValueError("external buffers unsupported")
    component = {
        5120: ("b", 1),
        5121: ("B", 1),
        5122: ("h", 2),
        5123: ("H", 2),
        5125: ("I", 4),
        5126: ("f", 4),
    }
    fmt, size = component[a["componentType"]]
    n = {"SCALAR": 1, "VEC3": 3}[a["type"]]
    stride = v.get("byteStride", size * n)
    start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    end = start + max(0, a["count"] - 1) * stride + size * n
    if (
        stride < size * n
        or start < v.get("byteOffset", 0)
        or end > v.get("byteOffset", 0) + v["byteLength"]
        or end > len(binary)
    ):
        raise ValueError("accessor outside buffer view")
    values = [
        struct.unpack_from("<" + fmt * n, binary, start + i * stride)
        for i in range(a["count"])
    ]
    if a.get("normalized"):
        raise ValueError("normalized positions unsupported")
    return values


def triangles(doc, binary):
    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    out = []
    if "skins" in doc or "animations" in doc:
        raise ValueError("animated or skinned geometry unsupported")

    def visit(i, parent, ancestors):
        if i in ancestors:
            raise ValueError("cycle in node graph")
        node = doc["nodes"][i]
        m = multiply(parent, matrix(node))
        if "mesh" in node:
            for p in doc["meshes"][node["mesh"]]["primitives"]:
                if p.get("mode", 4) != 4 or "targets" in p:
                    raise ValueError("unsupported primitive")
                if "extensions" in p:
                    raise ValueError("compressed or extended primitive unsupported")
                points = accessor(doc, binary, p["attributes"]["POSITION"])
                pts = [
                    tuple(
                        sum(m[c * 4 + r] * v[c] for c in range(3)) + m[12 + r]
                        for r in range(3)
                    )
                    for v in points
                ]
                if any(not math.isfinite(x) for v in pts for x in v):
                    raise ValueError("non-finite coordinate")
                ix = (
                    [v[0] for v in accessor(doc, binary, p["indices"])]
                    if "indices" in p
                    else list(range(len(pts)))
                )
                if len(ix) % 3:
                    raise ValueError("invalid triangle indices")
                out.extend(
                    (pts[ix[j]], pts[ix[j + 1]], pts[ix[j + 2]])
                    for j in range(0, len(ix), 3)
                )
        for child in node.get("children", []):
            visit(child, m, ancestors | {i})

    scenes = doc.get("scenes", [])
    if not scenes:
        raise ValueError("no scene")
    for root in scenes[doc.get("scene", 0)].get("nodes", []):
        visit(root, identity, set())
    if not out:
        raise ValueError("no triangles")
    return out


def inspect_geometry(tris, profile):
    if (
        not profile
        or profile.get("subject") != "cat"
        or profile.get("up_axis") not in (0, 1, 2)
        or profile.get("depth_axis") not in (0, 1, 2)
        or profile["up_axis"] == profile["depth_axis"]
    ):
        return {
            "status": "review",
            "reason": "Explicit cat subject, distinct up/depth axes, and segmented head/body regions required.",
        }
    regions = profile.get("regions", {})
    axis = profile["up_axis"]
    depth = profile["depth_axis"]
    results = {}
    for name in ("head", "body"):
        region = regions.get(name, {})
        bounds = region.get("bounds")
        levels = region.get("levels")
        threshold = region.get("min_depth_ratio")
        if not (
            isinstance(bounds, list)
            and len(bounds) == 3
            and all(
                isinstance(b, list)
                and len(b) == 2
                and all(isinstance(x, (int, float)) and math.isfinite(x) for x in b)
                and b[0] < b[1]
                for b in bounds
            )
            and isinstance(levels, list)
            and len(levels) >= 3
            and all(
                isinstance(x, (int, float))
                and math.isfinite(x)
                and bounds[axis][0] < x < bounds[axis][1]
                for x in levels
            )
            and isinstance(threshold, (int, float))
            and math.isfinite(threshold)
            and 0 < threshold <= 1
        ):
            return {
                "status": "review",
                "reason": f"{name} needs finite explicit world-space bounds, at least three interior levels and a depth ratio threshold.",
            }
        slices = []
        for level in levels:
            level_sample = level + (bounds[axis][1] - bounds[axis][0]) * 1e-8
            width = 3 - axis - depth
            segments = []
            for tri in tris:
                hits = []
                for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    delta = b[axis] - a[axis]
                    if abs(delta) < 1e-12:
                        continue
                    t = (level_sample - a[axis]) / delta
                    if 0 <= t < 1:
                        point = tuple(a[k] + t * (b[k] - a[k]) for k in range(3))
                        if all(
                            bounds[k][0] <= point[k] <= bounds[k][1] for k in range(3)
                        ):
                            hits.append(point)
                if len(hits) == 2 and hits[0] != hits[1]:
                    segments.append((hits[0], hits[1]))
            if not segments:
                slices.append(
                    {
                        "level": level,
                        "status": "review",
                        "reason": "no usable cross-section segments",
                    }
                )
                continue
            xs = [p[width] for segment in segments for p in segment]
            low, high = min(xs), max(xs)
            span = high - low
            if span <= 1e-12:
                slices.append(
                    {"level": level, "status": "review", "reason": "zero section width"}
                )
                continue
            scans = []
            ambiguous = False
            for fraction in (0.2, 0.35, 0.5, 0.65, 0.8):
                x = low + fraction * span
                depths = []
                for a, b in segments:
                    dx = b[width] - a[width]
                    if abs(dx) < 1e-12:
                        continue
                    t = (x - a[width]) / dx
                    if 0 <= t < 1:
                        depths.append(a[depth] + t * (b[depth] - a[depth]))
                depths.sort()
                unique = []
                for value in depths:
                    if not unique or abs(value - unique[-1]) > max(span, 1e-9) * 1e-7:
                        unique.append(value)
                if len(unique) > 2:
                    ambiguous = True
                if len(unique) >= 2:
                    scans.append(unique[-1] - unique[0])
                elif len(unique) == 1:
                    scans.append(0.0)
            if len(scans) < 3:
                slices.append(
                    {
                        "level": level,
                        "status": "review",
                        "reason": "insufficient local depth scans",
                    }
                )
                continue
            scans.sort()
            thickness = scans[len(scans) // 2]
            ratio = thickness / span
            # A tiny measured thickness is a definite failure; thick-looking disconnected
            # plates or multiple overlapping contours can never become a pass.
            degree = {}
            for a, b in segments:
                keys = [
                    tuple(round(v / max(span, 1e-9), 7) for v in point)
                    for point in (a, b)
                ]
                if keys[0] == keys[1]:
                    continue
                for key in keys:
                    degree[key] = degree.get(key, 0) + 1
            closed = all(n == 2 for n in degree.values())
            status = (
                "fail"
                if ratio < threshold
                else ("review" if ambiguous or not closed else "pass")
            )
            slices.append(
                {
                    "level": level,
                    "depth": thickness,
                    "width": span,
                    "depth_ratio": ratio,
                    "status": status,
                    "local_depths": scans,
                    "closed": closed,
                    "ambiguous": ambiguous,
                }
            )
        statuses = [s["status"] for s in slices]
        results[name] = {
            "status": "fail"
            if "fail" in statuses
            else ("review" if "review" in statuses else "pass"),
            "slices": slices,
        }
    statuses = [r["status"] for r in results.values()]
    return {
        "status": "fail"
        if "fail" in statuses
        else ("review" if "review" in statuses else "pass"),
        "regions": results,
        "note": "Geometric heuristic only; segment boundaries and thresholds require human judgment.",
    }


def inspect(path, profile=None, validator=None):
    raw = Path(path).read_bytes()
    result = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "format": {"status": "unknown", "reason": "Khronos validator not run"},
        "geometry": {"status": "review"},
        "appearance": {
            "status": "pending_review",
            "reason": "Human inspection of hash-matched material and clay views required.",
        },
    }
    if validator:
        try:
            proc = subprocess.run(
                [str(validator), "-o", str(path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            report = json.loads(proc.stdout)
            errors = report["issues"]["numErrors"]
            result["format"] = {
                "status": "pass" if errors == 0 and proc.returncode == 0 else "fail",
                "errors": errors,
                "exit_code": proc.returncode,
                "report": report,
            }
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError) as e:
            result["format"] = {"status": "unknown", "reason": str(e)}
    try:
        _, doc, binary = load_glb(path)
        result["geometry"] = inspect_geometry(triangles(doc, binary), profile)
    except (
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        struct.error,
        UnicodeError,
    ) as e:
        result["geometry"] = {"status": "review", "reason": str(e)}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("glb")
    p.add_argument("--profile")
    p.add_argument("--validator")
    a = p.parse_args()
    profile = (
        json.loads(Path(a.profile).read_text(encoding="utf-8")) if a.profile else None
    )
    print(json.dumps(inspect(a.glb, profile, a.validator), indent=2))


if __name__ == "__main__":
    main()
