"""Describe the final GLB, independently of the pre-finish mesh."""

import json
import struct
from pathlib import Path


def describe_glb(path):
    with Path(path).open("rb") as stream:
        header = stream.read(20)
        if len(header) != 20:
            raise ValueError("truncated GLB")
        magic, version, length, json_length, kind = struct.unpack("<4sIIII", header)
        if (
            magic != b"glTF"
            or version != 2
            or length != Path(path).stat().st_size
            or kind != 0x4E4F534A
            or json_length > length - 20
        ):
            raise ValueError("invalid GLB header")
        doc = json.loads(stream.read(json_length).decode("utf-8").rstrip(" \0"))
    vertices = triangles = 0
    for mesh in doc.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if primitive.get("mode", 4) != 4:
                raise ValueError("non-triangle GLB primitive unsupported")
            accessors = doc["accessors"]
            count = accessors[primitive["attributes"]["POSITION"]]["count"]
            indices = (
                accessors[primitive["indices"]]["count"]
                if "indices" in primitive
                else count
            )
            if (
                type(count) is not int
                or type(indices) is not int
                or count < 0
                or indices < 0
                or indices % 3
            ):
                raise ValueError("invalid GLB primitive counts")
            vertices += count
            triangles += indices // 3
    if not vertices or not triangles:
        raise ValueError("empty GLB")
    return {
        "vertices": vertices,
        "faces": triangles,
        "count_scope": "sum of mesh primitives; instances are not multiplied",
        "material_count": len(doc.get("materials", [])),
        "materials": [
            {
                "name": material.get("name"),
                "pbrMetallicRoughness": material.get("pbrMetallicRoughness", {}),
            }
            for material in doc.get("materials", [])
        ],
        "texture_count": len(doc.get("textures", [])),
        "image_count": len(doc.get("images", [])),
    }


def finalise_metadata(
    meta, final_glb, this_attempt_seconds, finishing_seconds, resumed, material_images
):
    result = dict(meta)
    baseline_elapsed = float(result.get("elapsed_seconds", 0))
    baseline = {
        "vertices": result.pop("vertices"),
        "faces": result.pop("faces"),
        "coordinates": result.pop("coordinates"),
        "export_material": result.pop("export_material"),
        "pre_finish_elapsed_seconds": baseline_elapsed,
    }
    stats = describe_glb(final_glb)
    result["baseline"] = baseline
    result["vertices"] = stats["vertices"]
    result["faces"] = stats["faces"]
    result["final_geometry"] = {
        "vertices": stats["vertices"],
        "faces": stats["faces"],
        "count_scope": stats["count_scope"],
    }
    result["final_materials"] = {
        "material_count": stats["material_count"],
        "texture_count": stats["texture_count"],
        "image_count": stats["image_count"],
        "exported_image_sizes": material_images,
    }
    result["export_material"] = {"source": "final GLB", "materials": stats["materials"]}
    result["coordinates"] = (
        "Final glTF scene coordinates; see finish_measurements.projection when available."
    )
    result["texture_size_scope"] = (
        "Requested baseline baking size; final material image sizes are listed separately."
    )
    result["this_attempt_elapsed_seconds"] = round(this_attempt_seconds, 2)
    result["finishing_elapsed_seconds"] = round(finishing_seconds, 2)
    result["elapsed_seconds"] = round(
        this_attempt_seconds + (baseline_elapsed if resumed else 0), 2
    )
    result["elapsed_scope"] = (
        "Known active time: baseline plus current attempt; excludes downtime and earlier failed attempts."
        if resumed
        else "Current attempt including baseline and finishing."
    )
    result["resumed_from_checkpoint"] = bool(resumed)
    return result
