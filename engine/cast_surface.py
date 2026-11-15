"""Construct a bounded continuous-field mesh from explicit geometric primitives."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes

MAX_INPUT_BYTES = 128 * 1024
MAX_PRIMITIVES = 64
MAX_SEGMENTS = 128
MAX_GRID_POINTS = 4_000_000
MAX_WORK = 250_000_000
CHUNK_POINTS = 100_000
MAX_COORD = 10_000.0


def number(v, name, *, positive=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{name} must be a finite number")  # noqa: TRY004
    if abs(v) > MAX_COORD or not math.isfinite(v) or (positive and v <= 0):
        raise ValueError(f"{name} out of range")
    return float(v)


def vec3(v, name, *, positive=False):
    if not isinstance(v, list) or len(v) != 3:
        raise ValueError(f"{name} must be an array of three numbers")
    return np.array([number(x, name, positive=positive) for x in v], dtype=np.float64)


def keys(obj, required, optional=()):
    if (
        not isinstance(obj, dict)
        or not set(required) <= obj.keys()
        or obj.keys() - set(required) - set(optional)
    ):
        raise ValueError(f"expected fields {required}, optional {optional}")


def parse_recipe(data):
    keys(data, ("pitch", "blend", "primitives"))
    pitch = number(data["pitch"], "pitch", positive=True)
    blend = number(data["blend"], "blend")
    if not 0.0001 <= pitch <= 100 or not 0 <= blend <= 100:
        raise ValueError("pitch or blend outside supported range")
    items = data["primitives"]
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_PRIMITIVES:
        raise ValueError("primitives count out of range")
    parsed = []
    lower, upper = [], []
    segments = 0
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("primitive must be an object")  # noqa: TRY004
        kind = item.get("type")
        if kind == "ellipsoid":
            keys(item, ("type", "center", "radii"))
            c, r = (
                vec3(item["center"], "center"),
                vec3(item["radii"], "radii", positive=True),
            )
            if min(r) < 2 * pitch:
                raise ValueError("radii must be at least twice pitch")
            parsed.append((kind, c, r))
            lower.append(c - r)
            upper.append(c + r)
            segments += 1
        elif kind == "tube":
            keys(item, ("type", "points", "radii"))
            pts, radii = item["points"], item["radii"]
            if (
                not isinstance(pts, list)
                or not 2 <= len(pts) <= MAX_SEGMENTS + 1
                or not isinstance(radii, list)
                or len(pts) != len(radii)
            ):
                raise ValueError("tube needs matching points and radii arrays")
            p = np.array([vec3(x, "point") for x in pts])
            r = np.array([number(x, "radius", positive=True) for x in radii])
            if min(r) < 2 * pitch or np.any(
                np.linalg.norm(np.diff(p, axis=0), axis=1) < 1e-9
            ):
                raise ValueError("tube radii too small or duplicate consecutive points")
            parsed.append((kind, p, r))
            lower.append(np.min(p - r[:, None], axis=0))
            upper.append(np.max(p + r[:, None], axis=0))
            segments += len(p) - 1
        else:
            raise ValueError("unknown primitive type")
    if segments > MAX_SEGMENTS:
        raise ValueError("too many primitive segments")
    margin = max(3 * pitch, blend * 2)
    origin = np.min(lower, axis=0) - margin
    end = np.max(upper, axis=0) + margin
    dims = np.ceil((end - origin) / pitch).astype(np.int64) + 1
    grid = math.prod(int(d) for d in dims)
    if np.any(dims < 3) or grid > MAX_GRID_POINTS or grid * segments > MAX_WORK:
        raise ValueError(
            "grid or work limit exceeded; increase pitch or simplify the recipe"
        )
    return pitch, blend, parsed, origin, dims, grid, segments


def smooth_min(a, b, width):
    if width == 0:
        return np.minimum(a, b)
    h = np.maximum(width - np.abs(a - b), 0) / width
    return np.minimum(a, b) - width * 0.25 * h * h


def build_mesh(data):
    pitch, blend, parts, origin, dims, count, segments = parse_recipe(data)
    field = np.empty(count, dtype=np.float32)
    # Flat-index chunks bound all temporary arrays independently of grid size.
    yz = int(dims[1] * dims[2])
    nz = int(dims[2])
    for start in range(0, count, CHUNK_POINTS):
        stop = min(count, start + CHUNK_POINTS)
        ix = np.arange(start, stop, dtype=np.int64)
        x = origin[0] + (ix // yz) * pitch
        y = origin[1] + ((ix % yz) // nz) * pitch
        z = origin[2] + (ix % nz) * pitch
        f = None
        for kind, a, r in parts:
            if kind == "ellipsoid":
                d = (
                    np.sqrt(
                        ((x - a[0]) / r[0]) ** 2
                        + ((y - a[1]) / r[1]) ** 2
                        + ((z - a[2]) / r[2]) ** 2
                    )
                    - 1
                ) * min(r)
            else:
                d = np.full(stop - start, np.inf)
                for i in range(len(a) - 1):
                    delta = a[i + 1] - a[i]
                    t = np.clip(
                        (
                            (x - a[i, 0]) * delta[0]
                            + (y - a[i, 1]) * delta[1]
                            + (z - a[i, 2]) * delta[2]
                        )
                        / np.dot(delta, delta),
                        0,
                        1,
                    )
                    candidate = np.sqrt(
                        (x - a[i, 0] - t * delta[0]) ** 2
                        + (y - a[i, 1] - t * delta[1]) ** 2
                        + (z - a[i, 2] - t * delta[2]) ** 2
                    ) - (r[i] + t * (r[i + 1] - r[i]))
                    np.minimum(d, candidate, out=d)
            f = d if f is None else smooth_min(f, d, blend)
        field[start:stop] = f
    volume = field.reshape(tuple(dims))
    if any(
        np.any(face <= 0)
        for face in (
            volume[0],
            volume[-1],
            volume[:, 0],
            volume[:, -1],
            volume[:, :, 0],
            volume[:, :, -1],
        )
    ):
        raise ValueError("surface reaches grid boundary; reduce blend")
    vertices, faces, _, _ = marching_cubes(
        volume, level=0, spacing=(pitch,) * 3, allow_degenerate=False
    )
    vertices += origin
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh, multibody=True)
    return mesh, {
        "grid_points": count,
        "grid_shape": dims.tolist(),
        "segments": segments,
        "pitch": pitch,
        "blend": blend,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.recipe.stat().st_size > MAX_INPUT_BYTES:
            raise ValueError("recipe too large")
        if args.output.suffix.lower() not in (".ply", ".glb"):
            raise ValueError("output must be .ply or .glb")
        if args.output.exists():
            raise ValueError("output already exists")
        data = json.loads(
            args.recipe.read_text(encoding="utf8"),
            parse_constant=lambda x: (_ for _ in ()).throw(
                ValueError("non-finite JSON number")
            ),
        )
        mesh, info = build_mesh(data)
        blob = mesh.export(file_type=args.output.suffix[1:].lower())
        with args.output.open("xb") as f:
            f.write(blob)
        print(
            json.dumps(
                {
                    **info,
                    "vertices": len(mesh.vertices),
                    "faces": len(mesh.faces),
                    "watertight": mesh.is_watertight,
                }
            )
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"cast_surface: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
