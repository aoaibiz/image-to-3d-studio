"""Conservative, unlit texture-region paint review; never an appearance verdict."""

import argparse
import hashlib
import io
import json
import math
import struct
from pathlib import Path

from PIL import Image
from inspection import load_glb


MAX_GLB_BYTES = 64 * 1024 * 1024
MAX_CONFIG_BYTES = 64 * 1024
MAX_REGIONS = 32
MAX_PALETTE = 16
MAX_REGION_PIXELS = 250_000
MAX_WORK = 2_000_000


def validate_config(config):
    if not isinstance(config, dict) or not isinstance(config.get("regions"), list):
        raise ValueError("config must contain a regions list")
    regions = config["regions"]
    if len(regions) > MAX_REGIONS:
        raise ValueError("too many regions")
    work = 0
    for spec in regions:
        if not isinstance(spec, dict):
            raise ValueError("each region must be an object")
        name = spec.get("name")
        if not isinstance(name, str) or not name or len(name) > 128:
            raise ValueError(
                "region name must be a nonempty string up to 128 characters"
            )
        if ("material_index" in spec) == ("material" in spec):
            raise ValueError("specify exactly one material name or index")
        if "material_index" in spec and (
            type(spec["material_index"]) is not int or spec["material_index"] < 0
        ):
            raise ValueError("invalid material index")
        if "material" in spec and (
            not isinstance(spec["material"], str)
            or not spec["material"]
            or len(spec["material"]) > 128
        ):
            raise ValueError("invalid material name")
        rect = spec.get("rect")
        if (
            not isinstance(rect, list)
            or len(rect) != 4
            or any(type(v) is not int or v < 0 for v in rect)
            or rect[0] >= rect[2]
            or rect[1] >= rect[3]
        ):
            raise ValueError("rect must be four nonnegative ordered integers")
        area = (rect[2] - rect[0]) * (rect[3] - rect[1])
        if area > MAX_REGION_PIXELS:
            raise ValueError("region exceeds pixel limit")
        palette = spec.get("palette")
        if (
            not isinstance(palette, list)
            or not 1 <= len(palette) <= MAX_PALETTE
            or any(
                not isinstance(c, list)
                or len(c) != 3
                or any(type(v) is not int or not 0 <= v <= 255 for v in c)
                for c in palette
            )
        ):
            raise ValueError("invalid palette")
        tolerance = spec.get("tolerance")
        maximum = spec.get("max_island_pixels")
        if (
            type(tolerance) not in (int, float)
            or not math.isfinite(tolerance)
            or not 0 <= tolerance <= 441
            or type(maximum) is not int
            or not 1 <= maximum <= 100000
        ):
            raise ValueError("invalid thresholds")
        work += area * len(palette)
        if work > MAX_WORK:
            raise ValueError("total pixel-palette work limit exceeded")


def scan(image, region, palette, tolerance, max_pixels, factor=(1, 1, 1, 1)):
    """Find 4-connected off-palette islands, using RGB distance in sRGB bytes."""
    x0, y0, x1, y1 = region
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise ValueError("region outside texture")
    if (x1 - x0) * (y1 - y0) > MAX_REGION_PIXELS:
        raise ValueError("region exceeds pixel limit")
    if not palette or any(
        len(c) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in c)
        for c in palette
    ):
        raise ValueError("palette must contain RGB byte triples")
    if not 0 <= tolerance <= 441 or not 1 <= max_pixels <= 100000:
        raise ValueError("invalid threshold")

    def effective(v, f):
        s = v / 255
        linear = s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4
        linear *= f
        return round(
            255
            * (
                12.92 * linear
                if linear <= 0.0031308
                else 1.055 * linear ** (1 / 2.4) - 0.055
            )
        )

    tables = [[effective(v, f) for v in range(256)] for f in factor[:3]]
    im = image.convert("RGBA")
    px = im.load()
    bad = set()
    cache = {}
    threshold = tolerance * tolerance
    for y in range(y0, y1):
        for x in range(x0, x1):
            r, g, b, a = px[x, y]
            if a != 255:
                raise ValueError("non-opaque texels in region unsupported")
            rgb = (r, g, b)
            if rgb not in cache:
                rr, gg, bb = tables[0][r], tables[1][g], tables[2][b]
                off = all(
                    (rr - c[0]) ** 2 + (gg - c[1]) ** 2 + (bb - c[2]) ** 2 > threshold
                    for c in palette
                )
                if len(cache) < 65536:
                    cache[rgb] = off
            else:
                off = cache[rgb]
            if off:
                bad.add((x, y))
    islands = []
    large = []
    while bad:
        start = bad.pop()
        todo = [start]
        count = 0
        xs = []
        ys = []
        while todo:
            x, y = todo.pop()
            count += 1
            xs.append(x)
            ys.append(y)
            for pt in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if pt in bad:
                    bad.remove(pt)
                    todo.append(pt)
        item = {
            "pixels": count,
            "bbox": [min(xs), min(ys), max(xs) + 1, max(ys) + 1],
            "touches_region_edge": min(xs) == x0
            or min(ys) == y0
            or max(xs) == x1 - 1
            or max(ys) == y1 - 1,
        }
        (islands if count <= max_pixels else large).append(item)
    return {
        "small_off_palette_islands": sorted(
            islands, key=lambda a: (a["bbox"][1], a["bbox"][0])
        ),
        "large_off_palette_components": large,
    }


def inspect(path, config):
    validate_config(config)
    if Path(path).stat().st_size > MAX_GLB_BYTES:
        raise ValueError("GLB exceeds byte limit")
    raw, doc, binary = load_glb(path)
    if len(raw) > MAX_GLB_BYTES:
        raise ValueError("GLB exceeds byte limit")
    result = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "appearance": "pending_human_review",
        "regions": [],
    }
    for spec in config.get("regions", []):
        row = {
            "name": spec.get("name"),
            "material": spec.get("material"),
            "status": "unmeasured",
        }
        try:
            matches = [
                (i, m)
                for i, m in enumerate(doc.get("materials", []))
                if (
                    i == spec["material_index"]
                    if "material_index" in spec
                    else m.get("name") == spec["material"]
                )
            ]
            if len(matches) != 1:
                raise ValueError("material name missing or not unique")
            idx, material = matches[0]
            prims = [
                p
                for mesh in doc.get("meshes", [])
                for p in mesh.get("primitives", [])
                if p.get("material") == idx
            ]
            if not prims or any(
                "TEXCOORD_0" not in p.get("attributes", {})
                or "COLOR_0" in p.get("attributes", {})
                or p.get("extensions")
                for p in prims
            ):
                raise ValueError(
                    "material requires uncompressed TEXCOORD_0 primitives without vertex colors"
                )
            if (
                material.get("extensions")
                or material.get("alphaMode", "OPAQUE") != "OPAQUE"
            ):
                raise ValueError("extended or non-opaque material unsupported")
            pbr = material.get("pbrMetallicRoughness", {})
            factor = pbr.get("baseColorFactor", [1, 1, 1, 1])
            if (
                len(factor) != 4
                or any(
                    not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in factor
                )
                or factor[3] != 1
            ):
                raise ValueError("invalid or transparent baseColorFactor")
            texinfo = pbr["baseColorTexture"]
            if texinfo.get("texCoord", 0) != 0 or texinfo.get("extensions"):
                raise ValueError("texture transform or alternate UV unsupported")
            texture = doc["textures"][texinfo["index"]]
            if texture.get("extensions"):
                raise ValueError("texture extensions unsupported")
            image = doc["images"][texture["source"]]
            if "bufferView" not in image or image.get("mimeType") not in (
                "image/png",
                "image/jpeg",
            ):
                raise ValueError("only embedded PNG/JPEG supported")
            view = doc["bufferViews"][image["bufferView"]]
            off = view.get("byteOffset", 0)
            length = view["byteLength"]
            if (
                view.get("buffer", 0) != 0
                or off < 0
                or length <= 0
                or off + length > len(binary)
            ):
                raise ValueError("invalid image buffer view")
            with Image.open(io.BytesIO(binary[off : off + length])) as im:
                if im.width * im.height > 16777216:
                    raise ValueError("texture exceeds size limit")
                rect = spec["rect"]
                pal = spec["palette"]
                tol = spec["tolerance"]
                maximum = spec["max_island_pixels"]
                if len(rect) != 4 or any(type(v) is not int for v in rect):
                    raise ValueError("rect must be four integer pixels")
                row.update(scan(im, rect, pal, tol, maximum, factor))
                row.update(status="review", texture_size=list(im.size), rect=rect)
        except (
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            OSError,
            AttributeError,
            OverflowError,
            struct.error,
            UnicodeError,
            Image.DecompressionBombError,
        ) as e:
            row["reason"] = str(e)
        result["regions"].append(row)
    if not result["regions"]:
        result["note"] = "No explicit regions configured; nothing measured."
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("glb")
    p.add_argument("--config", required=True)
    a = p.parse_args()
    try:
        with open(a.config, "rb") as file:
            data = file.read(MAX_CONFIG_BYTES + 1)
        if len(data) > MAX_CONFIG_BYTES:
            raise ValueError("config exceeds byte limit")
        result = inspect(a.glb, json.loads(data.decode("utf-8-sig")))
        print(json.dumps(result, indent=2))
        return 0
    except (
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        OSError,
        AttributeError,
        OverflowError,
        struct.error,
        UnicodeError,
        Image.DecompressionBombError,
    ) as e:
        print(json.dumps({"appearance": "pending_human_review", "error": str(e)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
