import base64
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from engine import sf3d_shape_job as job, sf3d_provider as provider
from engine.pipeline import SF3D_PIPELINE
from studio import configure
from studio.core import JobCore
from quality.test_sf3d_provider import Replay, fixture_glb, png


def unpack(glb):
    n = struct.unpack_from("<I", glb, 12)[0]
    return json.loads(glb[20 : 20 + n]), glb[20 + n + 8 :]


class NativeJobTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.image = self.root / "input.png"
        self.image.write_bytes(png())
        self.setup = self.root / "engine.json"
        self.out = self.root / "out"
        self.api = Replay(fixture_glb())
        self.api.close = lambda: None

    def configured(self):
        configure.configure(self.root, SF3D_PIPELINE, allow_official_space=True)

    def test_authorization_is_specific_and_checked_before_upload(self):
        with self.assertRaises(ValueError):
            configure.configure(self.root, SF3D_PIPELINE, allow_send_to_openai=True)
        with mock.patch.object(provider, "Space", return_value=self.api):
            with self.assertRaises(job.ConsentError):
                job.run(self.image, self.out, setup=self.setup)
            self.configured()

            def revoke(stage):
                if stage == "upload_unknown":
                    configure.configure(self.root, "triposr")

            with self.assertRaises(job.ConsentError):
                job.run(self.image, self.out, setup=self.setup, progress=revoke)
            self.assertEqual(self.api.uploads, 0)
            self.assertEqual(
                json.loads(
                    (self.out / "sf3d-work/provider/provider.json").read_bytes()
                )["stage"],
                "prepared",
            )

    def test_source_oriented_materials_and_resume(self):
        self.configured()
        with mock.patch.object(provider, "Space", return_value=self.api):
            settings = job.run(self.image, self.out, setup=self.setup)
            self.assertEqual(self.api.joins, [4, 5])
            configure.configure(self.root, "triposr")
            self.assertEqual(job.run(self.image, self.out, setup=self.setup), settings)
            self.assertEqual(self.api.joins, [4, 5])
        self.assertEqual((self.out / "provider-native.glb").read_bytes(), self.api.glb)
        raw, rawbin = unpack(self.api.glb)
        rotated, finalbin = unpack((self.out / "model.glb").read_bytes())
        self.assertEqual(finalbin, rawbin)
        for name in ("meshes", "textures", "materials", "images", "accessors"):
            self.assertEqual(raw[name], rotated[name])
        self.assertEqual(
            rotated["nodes"][rotated["scenes"][0]["nodes"][0]]["rotation"], [0, 1, 0, 0]
        )
        master = json.loads((self.out / "editable.gltf").read_bytes())
        self.assertEqual(
            base64.b64decode(master["buffers"][0]["uri"].split(",", 1)[1]),
            rawbin[: raw["buffers"][0]["byteLength"]],
        )
        self.assertEqual(master["materials"], raw["materials"])
        self.assertTrue(JobCore._artifacts_valid(self.out, settings))
        self.assertNotIn("marching_cubes_resolution", settings)
        self.assertEqual(settings["visual_quality"], "needs_review")
        (self.out / "provider-native.glb").write_bytes(b"corrupt")
        self.assertFalse(JobCore._artifacts_valid(self.out, settings))
        (self.out / "sf3d-work/provider/provider.json").unlink()
        with self.assertRaisesRegex(ValueError, "再送"):
            job.run(self.image, self.out, setup=self.setup)

    @unittest.skipUnless(
        os.getenv("STUDIO_SF3D_REPLAY"), "Set the private offline receipt directory."
    )
    def test_real_material_packages(self):
        root = Path(os.environ["STUDIO_SF3D_REPLAY"])
        for i, folder in enumerate((root, root / "fresh")):
            rec = json.loads((folder / "receipt.json").read_bytes())
            png_path = folder / "derived-input.png"
            native = folder / "model.glb"
            self.assertEqual(
                job.files.digest(Path(rec["source"])), rec["source_sha256"]
            )
            self.assertEqual(job.files.digest(png_path), rec["upload_sha256"])
            receipt = {
                "source_sha256": rec["source_sha256"],
                "foreground_sha256": rec["upload_sha256"],
                "artifact": provider.native_glb(native),
                "settings": rec["settings"],
                "generation_event_id": rec["sf3d_event_id"],
            }
            settings = job.finalize(
                native,
                png_path,
                {
                    "source_sha256": rec["source_sha256"],
                    "selection": {"method": "archived foreground"},
                },
                {"method": rec["model_input_preparation"]},
                receipt,
                self.root / str(i),
            )
            self.assertTrue(job.valid(self.root / str(i), settings))
            self.assertEqual(
                settings["artifact_sha256"]["provider-native.glb"], rec["glb_sha256"]
            )
            a, b = (
                unpack(native.read_bytes()),
                unpack((self.root / str(i) / "model.glb").read_bytes()),
            )
            self.assertEqual(a[1], b[1])
            self.assertEqual(a[0]["materials"], b[0]["materials"])
