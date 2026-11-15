import json, struct, tempfile, unittest
from pathlib import Path
from inspection import inspect


class TestGlb(unittest.TestCase):
    def test_actual_glb_and_unknown_validation(self):
        vertices = [(-1, 0, 0), (1, 0, 0), (0, 1, 0)]
        binary = b"".join(struct.pack("<fff", *v) for v in vertices)
        doc = {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": len(binary)}],
            "bufferViews": [{"buffer": 0, "byteLength": len(binary)}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}
            ],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}],
            "scene": 0,
        }
        raw = json.dumps(doc).encode()
        raw += b" " * ((-len(raw)) % 4)
        glb = (
            struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(raw) + 8 + len(binary))
            + struct.pack("<II", len(raw), 0x4E4F534A)
            + raw
            + struct.pack("<II", len(binary), 0x004E4942)
            + binary
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sample.glb"
            path.write_bytes(glb)
            result = inspect(path)
        self.assertEqual(result["format"]["status"], "unknown")
        self.assertEqual(result["geometry"]["status"], "review")
        self.assertEqual(result["appearance"]["status"], "pending_review")
        self.assertEqual(len(result["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
