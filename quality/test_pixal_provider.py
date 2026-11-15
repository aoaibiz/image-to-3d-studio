import io
import json
import shutil
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import requests
import trimesh
from PIL import Image

from engine import pixal_provider as provider

EVENT = "1" * 32
ASSET = "a" * 12
FILE_URL = provider.ORIGIN + "/community-assets/" + ASSET + "/model.glb"


class PixalProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.work = root / "work"
        self.photo = root / "private_filename.jpg"
        original = Image.new("RGB", (80, 40), "#982020")
        exif = Image.Exif()
        exif[270] = "sensitive unpublished metadata"
        original.save(self.photo, exif=exif)
        model = trimesh.creation.icosphere(subdivisions=1)
        pbr = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (8, 8), "#762000"),
            metallicRoughnessTexture=Image.new("RGB", (8, 8), "#002255"))
        model.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(model.vertices), 2)), material=pbr)
        self.mesh = root / "fixture.glb"
        model.export(self.mesh)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.preflight = self.stack.enter_context(patch.object(provider, "preflight", return_value={
            "provider_space":provider.SPACE, "revision":provider.REVISION, "origin":provider.ORIGIN}))
        self.stack.enter_context(patch.object(requests.sessions.Session, "request", side_effect=AssertionError("network disabled")))
        self.calls = []
        self.api = self.stack.enter_context(patch.object(provider, "_json", side_effect=self.request))
        self.wait = self.stack.enter_context(patch.object(provider, "_wait", return_value=(ASSET, FILE_URL)))
        self.download = self.stack.enter_context(patch.object(provider, "_download", side_effect=self.send_file))

    def request(self, session, method, url, **kwargs):
        self.assertEqual(method,"POST")
        self.calls.append(url)
        if url.endswith("/upload"):
            filename, body, mime = kwargs["files"]["files"]
            self.assertEqual((filename,mime),("input.png","image/png"))
            data = body.read()
            self.assertNotIn(b"sensitive unpublished metadata", data)
            with Image.open(io.BytesIO(data)) as im:
                self.assertFalse(im.getexif())
            return ["/tmp/gradio/" + "b"*64 + "/input.png"]
        self.assertEqual(url, provider.CALL)
        self.assertEqual(kwargs["json"]["data"][1:],["",provider.SEED,1024,"balanced",-1.0,False])
        return {"event_id":EVENT}

    def send_file(self, session, url, dest, limit=provider.MAX_GLB):
        if url == FILE_URL:
            shutil.copyfile(self.mesh,dest)
        elif url == FILE_URL.replace("model.glb","input.png"):
            shutil.copyfile(self.work/"input.png",dest)
        else:
            self.fail("foreign URL")

    def test_consent_and_exact_pbr_resume(self):
        with self.assertRaises(PermissionError):
            provider.generate(self.photo,self.work)
        with self.assertRaisesRegex(ValueError, "approved source digest"):
            provider.generate(self.photo,self.work,allow_public_cc0=True,expected_source_sha256="0"*64)
        self.assertFalse(self.calls)
        meta = provider.generate(self.photo,self.work,allow_public_cc0=True)
        self.assertEqual(meta["state"],"completed")
        self.assertEqual(meta["sha256"], provider.digest(self.mesh))
        self.assertEqual(meta["material"]["triangles"],80)
        self.assertEqual(meta["event_id"],EVENT)
        self.assertEqual(meta["quality"],"needs_review")
        self.assertEqual(provider.generate(self.photo,self.work),meta)
        self.assertEqual(len(self.calls),2)
        (self.work/"provider.json").unlink()
        with self.assertRaises(provider.Pending):
            provider.generate(self.photo,self.work,allow_public_cc0=True)
        self.assertEqual(len(self.calls),2)

    def test_timeout_uses_event_only_and_wrong_returned_photo_fails(self):
        self.wait.side_effect = [requests.Timeout("interrupted"),(ASSET,FILE_URL)]
        with self.assertRaises(provider.Pending):
            provider.generate(self.photo,self.work,allow_public_cc0=True)
        self.download.side_effect = lambda session,url,dest,*args: dest.write_bytes(b"unrelated")
        with self.assertRaisesRegex(ValueError,"different uploaded image"):
            provider.generate(self.photo,self.work)
        self.assertEqual(self.wait.call_args.args[1], EVENT)
        self.assertEqual(len(self.calls),2)
        self.assertFalse((self.work/"model.glb").exists())

    def test_submit_uncertainty_and_old_journal_never_post_twice(self):
        self.api.side_effect = lambda *a,**kw: self.request(*a,**kw) if a[2].endswith("/upload") else (_ for _ in ()).throw(requests.Timeout())
        with self.assertRaises(provider.Pending):
            provider.generate(self.photo,self.work,allow_public_cc0=True)
        receipt = self.work/"provider.json"
        prior = json.loads(receipt.read_bytes())
        self.assertEqual(prior["state"],"submission_unknown")
        prior["state"]="uploaded"  # A previously saved journal survived a crash.
        receipt.write_text(json.dumps(prior))
        with self.assertRaises(provider.Pending):
            provider.generate(self.photo,self.work,allow_public_cc0=True)
        self.assertEqual(self.api.call_count,2)

    def test_foreign_or_shared_record_and_unpainted_models_rejected(self):
        local="/community-assets/"+ASSET
        asset=dict(id=ASSET,shared=False,source_space=provider.SPACE, model_filename="model.glb",
                   model_url=local+"/model.glb",input_url=local+"/input.png",status="complete",
                   seed=provider.SEED,resolution=1024,quality="balanced")
        for overrides in ({}, {"shared":True},{"model_url":"https://example.org/model.glb"},
                          {"id":"../sample"},{"input_url":None}):
            data=[json.dumps(dict(ok=True,mode="fast_final",asset={**asset,**overrides}))]
            if overrides:
                with self.assertRaises(ValueError):provider._asset(data)
            else:
                self.assertEqual(provider._asset(data),(ASSET,FILE_URL))
        blank = self.mesh.with_name("clay.glb")
        trimesh.creation.box().export(blank)
        with self.assertRaises((ValueError,KeyError)):provider.inspect_pbr(blank)


if __name__ == "__main__":
    unittest.main()
