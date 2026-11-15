import unittest
import json
import io
import subprocess
import sys
import struct
import zlib
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
from paint_audit import scan, validate_config, MAX_CONFIG_BYTES, MAX_GLB_BYTES


class PaintAuditTest(unittest.TestCase):
    def test_triangle_and_intended_stripe(self):
        im = Image.new("RGBA", (40, 40), (220, 180, 100, 255))
        d = ImageDraw.Draw(im)
        d.rectangle((20, 0, 24, 39), fill=(40, 40, 40, 255))
        d.polygon([(4, 4), (9, 4), (4, 9)], fill=(20, 150, 180, 255))
        result = scan(im, [0, 0, 40, 40], [[220, 180, 100], [40, 40, 40]], 5, 50)
        self.assertEqual(len(result["small_off_palette_islands"]), 1)
        self.assertEqual(result["small_off_palette_islands"][0]["bbox"], [4, 4, 10, 10])
        self.assertEqual(result["large_off_palette_components"], [])

    def test_large_unexpected_region_is_reported_not_passed(self):
        im = Image.new("RGBA", (20, 20), (40, 40, 40, 255))
        result = scan(im, [0, 0, 20, 20], [[200, 200, 200]], 5, 10)
        self.assertEqual(result["large_off_palette_components"][0]["pixels"], 400)

    def test_transparent_is_unmeasured(self):
        with self.assertRaises(ValueError):
            scan(Image.new("RGBA", (2, 2)), [0, 0, 2, 2], [[0, 0, 0]], 1, 3)

    def test_factor_is_applied_in_linear_space(self):
        im = Image.new("RGB", (1, 1), "white")
        self.assertEqual(
            scan(im, [0, 0, 1, 1], [[188, 188, 188]], 1, 1, (0.5, 0.5, 0.5, 1))[
                "small_off_palette_islands"
            ],
            [],
        )


class InputTest(unittest.TestCase):
    def test_bad_structures_and_limits(self):
        base = {
            "name": "body",
            "material_index": 0,
            "rect": [0, 0, 20, 20],
            "palette": [[1, 2, 3]],
            "tolerance": 2,
            "max_island_pixels": 10,
        }
        bad = [
            None,
            {},
            {"regions": None},
            {"regions": [None]},
            {"regions": [{}]},
            {"regions": [dict(base, palette=[None])]},
            {"regions": [dict(base, palette=[[1, 2, 3]] * 17)]},
            {"regions": [dict(base, rect=[0, 0, 501, 501])]},
            {"regions": [base] * 33},
            {"regions": [dict(base, tolerance=float("nan"))]},
            {"regions": [dict(base, material_index=-1)]},
            {"regions": [dict(base, rect=[0, 0, 500, 500], palette=[[1, 2, 3]] * 9)]},
        ]
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_config(value)

    def test_cli_handles_invalid_inputs_without_traceback(self):
        script = Path(__file__).with_name("paint_audit.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glb = root / "model.glb"
            conf = root / "config.json"
            glb.write_bytes(b"bad")
            for data in (
                b'{"regions":[null]}',
                b"{",
                b"x" * (MAX_CONFIG_BYTES + 1),
                b"\xff",
            ):
                conf.write_bytes(data)
                proc = subprocess.run(
                    [sys.executable, str(script), str(glb), "--config", str(conf)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 2)
                self.assertNotIn("Traceback", proc.stderr)
                self.assertIn("error", json.loads(proc.stdout))
            conf.write_text('{"regions":[]}', encoding="utf-8")
            with glb.open("wb") as f:
                f.truncate(MAX_GLB_BYTES + 1)
            proc = subprocess.run(
                [sys.executable, str(script), str(glb), "--config", str(conf)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("byte limit", json.loads(proc.stdout)["error"])

    def test_decompression_bomb_embedded_image_is_unmeasured(self):
        # A tiny, valid PNG header declares enormous dimensions; no large image is allocated.
        payload = struct.pack(">IIBBBBB", 50000, 50000, 8, 2, 0, 0, 0)
        source = io.BytesIO()
        Image.new("RGB", (1, 1)).save(source, format="PNG")
        png = (
            source.getvalue()[:8]
            + struct.pack(">I", len(payload))
            + b"IHDR"
            + payload
            + struct.pack(">I", zlib.crc32(b"IHDR" + payload))
            + source.getvalue()[33:]
        )
        doc = {
            "asset": {"version": "2.0"},
            "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
            "meshes": [
                {"primitives": [{"material": 0, "attributes": {"TEXCOORD_0": 0}}]}
            ],
            "textures": [{"source": 0}],
            "images": [{"bufferView": 0, "mimeType": "image/png"}],
            "buffers": [{"byteLength": len(png)}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(png)}],
        }
        encoded = json.dumps(doc).encode()
        encoded += b" " * ((-len(encoded)) % 4)
        binary = png + b"\0" * ((-len(png)) % 4)
        glb = (
            struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(encoded) + 8 + len(binary))
            + struct.pack("<I4s", len(encoded), b"JSON")
            + encoded
            + struct.pack("<I4s", len(binary), b"BIN\0")
            + binary
        )
        config = {
            "regions": [
                {
                    "name": "body",
                    "material_index": 0,
                    "rect": [0, 0, 1, 1],
                    "palette": [[0, 0, 0]],
                    "tolerance": 1,
                    "max_island_pixels": 1,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.glb"
            profile = root / "config.json"
            model.write_bytes(glb)
            profile.write_text(json.dumps(config), encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("paint_audit.py")),
                    str(model),
                    "--config",
                    str(profile),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0)
            self.assertNotIn("Traceback", proc.stderr)
            result = json.loads(proc.stdout)
            self.assertEqual(result["appearance"], "pending_human_review")
            self.assertEqual(result["regions"][0]["status"], "unmeasured")
            self.assertIn("decompression bomb", result["regions"][0]["reason"].lower())


if __name__ == "__main__":
    unittest.main()
