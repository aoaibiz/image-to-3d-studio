import json
import struct
import tempfile
import unittest
from pathlib import Path
from engine.output_metadata import describe_glb, finalise_metadata


class OutputMetadataTests(unittest.TestCase):
    def test_final_counts_materials_and_resume_timing(self):
        doc = {
            "asset": {"version": "2.0"},
            "accessors": [{"count": 10}, {"count": 12}, {"count": 3}],
            "meshes": [
                {
                    "primitives": [
                        {"attributes": {"POSITION": 0}, "indices": 1},
                        {"attributes": {"POSITION": 2}},
                    ]
                }
            ],
            "materials": [{}, {}],
            "textures": [{}],
            "images": [{}],
        }
        raw = json.dumps(doc).encode()
        raw += b" " * (-len(raw) % 4)
        data = (
            struct.pack("<4sIIII", b"glTF", 2, 20 + len(raw), len(raw), 0x4E4F534A)
            + raw
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.glb"
            path.write_bytes(data)
            self.assertEqual(describe_glb(path)["vertices"], 13)
            base = {
                "vertices": 999,
                "faces": 888,
                "coordinates": "old",
                "export_material": {"roughness_factor": 0.52},
                "texture_size": 1024,
                "elapsed_seconds": 100.0,
            }
            result = finalise_metadata(
                base, path, 5, 3, True, {"texture.png": [512, 512]}
            )
            self.assertEqual((result["vertices"], result["faces"]), (13, 5))
            self.assertEqual(result["baseline"]["vertices"], 999)
            self.assertEqual(result["baseline"]["pre_finish_elapsed_seconds"], 100)
            self.assertEqual(result["elapsed_seconds"], 105)
            self.assertEqual(result["finishing_elapsed_seconds"], 3)
            self.assertEqual(result["texture_size"], 1024)
            self.assertEqual(
                result["final_materials"]["exported_image_sizes"]["texture.png"],
                [512, 512],
            )
            self.assertEqual(result["final_materials"]["material_count"], 2)
            self.assertEqual(result["export_material"]["source"], "final GLB")
            self.assertEqual(len(result["export_material"]["materials"]), 2)
            self.assertEqual(
                finalise_metadata(base, path, 120, 20, False, {})["elapsed_seconds"],
                120,
            )

    def test_rejects_non_triangle_primitive(self):
        doc = {
            "accessors": [{"count": 3}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": 1}]}],
        }
        raw = json.dumps(doc).encode()
        raw += b" " * (-len(raw) % 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.glb"
            path.write_bytes(
                struct.pack("<4sIIII", b"glTF", 2, 20 + len(raw), len(raw), 0x4E4F534A)
                + raw
            )
            with self.assertRaises(ValueError):
                describe_glb(path)


if __name__ == "__main__":
    unittest.main()
