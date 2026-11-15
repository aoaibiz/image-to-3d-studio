import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import trimesh

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "cast_surface", ROOT / "engine" / "cast_surface.py"
)
cast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cast)
RECIPE = {
    "pitch": 0.04,
    "blend": 0.08,
    "primitives": [
        {"type": "ellipsoid", "center": [0, 0, 0], "radii": [0.5, 0.5, 0.5]},
        {
            "type": "tube",
            "points": [[0.35, 0, 0], [0.7, 0.1, 0], [0.9, 0.3, 0]],
            "radii": [0.18, 0.18, 0.14],
        },
        {"type": "ellipsoid", "center": [2, 0, 0], "radii": [0.3, 0.3, 0.3]},
    ],
}


class CastSurfaceTests(unittest.TestCase):
    def test_join_and_separate_components_are_closed(self):
        mesh, info = cast.build_mesh(RECIPE)
        components = mesh.split(only_watertight=False)
        self.assertEqual(len(components), 2)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(all(m.volume > 0 for m in components))
        boxes = sorted((m.bounds for m in components), key=lambda b: b[0, 0])
        self.assertGreater(boxes[1][0, 0] - boxes[0][1, 0], 0.4)
        self.assertLessEqual(info["grid_points"], cast.MAX_GRID_POINTS)

    def test_invalid_numbers_and_resource_limits(self):
        for mutation in (
            "nan",
            "zero",
            "overflow",
            "duplicate",
            "hugegrid",
            "primitive_count",
            "unknown",
            "huge_integer",
        ):
            d = copy.deepcopy(RECIPE)
            if mutation == "nan":
                d["pitch"] = float("nan")
            elif mutation == "zero":
                d["primitives"][0]["radii"][0] = 0
            elif mutation == "overflow":
                d["primitives"][0]["center"][0] = 1e308
            elif mutation == "duplicate":
                d["primitives"][1]["points"][1] = d["primitives"][1]["points"][0]
            elif mutation == "hugegrid":
                d["pitch"] = 0.0001
            elif mutation == "primitive_count":
                d["primitives"] = d["primitives"] * 30
            elif mutation == "huge_integer":
                d["pitch"] = 10**400
            else:
                d["primitives"][0]["url"] = "https://example.com"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                cast.build_mesh(d)

    def test_cli_exports_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            recipe = path / "input.json"
            out = path / "output.glb"
            recipe.write_text(json.dumps(RECIPE), encoding="utf8")
            cmd = [
                sys.executable,
                str(ROOT / "engine" / "cast_surface.py"),
                str(recipe),
                str(out),
            ]
            first = subprocess.run(
                cmd, check=False, capture_output=True, text=True, timeout=60
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            before = out.read_bytes()
            scene = trimesh.load(out, force="scene")
            self.assertGreater(len(scene.geometry), 0)
            second = subprocess.run(
                cmd, check=False, capture_output=True, text=True, timeout=60
            )
            self.assertEqual(second.returncode, 2)
            self.assertEqual(before, out.read_bytes())
            recipe.write_text('{"pitch":NaN}', encoding="utf8")
            third = subprocess.run(
                [*cmd[:-1], str(path / "bad.ply")],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(third.returncode, 2)
            self.assertFalse((path / "bad.ply").exists())


if __name__ == "__main__":
    unittest.main()
