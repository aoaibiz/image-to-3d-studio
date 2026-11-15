import hashlib
import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from engine import sf3d_provider as sf


def png():
    im = Image.new("RGBA", (24, 24))
    im.paste((220, 110, 20, 255), (3, 4, 21, 20))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def fixture_glb(texture_bytes=None, mime="image/png", last_index=2):
    body = bytearray()
    views = []

    def add(raw):
        start = len(body)
        body.extend(raw)
        body.extend(b"\0" * (-len(body) % 4))
        views.append({"buffer": 0, "byteOffset": start, "byteLength": len(raw)})
        return len(views) - 1

    access = []
    for kind, ctype, data in [
        ("VEC3", 5126, struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)),
        ("VEC2", 5126, struct.pack("<6f", 0, 0, 1, 0, 0, 1)),
        ("VEC3", 5126, struct.pack("<9f", 0, 0, 1, 0, 0, 1, 0, 0, 1)),
        ("SCALAR", 5123, struct.pack("<3H", 0, 1, last_index)),
    ]:
        access.append(
            {"bufferView": add(data), "type": kind, "componentType": ctype, "count": 3}
        )
    texture = add(texture_bytes if texture_bytes is not None else png())
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "buffers": [{"byteLength": len(body)}],
        "bufferViews": views,
        "accessors": access,
        "images": [{"bufferView": texture, "mimeType": mime}],
        "textures": [{"source": 0}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1, "NORMAL": 2},
                        "indices": 3,
                        "material": 0,
                    }
                ]
            }
        ],
    }
    j = json.dumps(doc).encode()
    j += b" " * (-len(j) % 4)
    return (
        struct.pack("<4sIIII", b"glTF", 2, 28 + len(j) + len(body), len(j), 0x4E4F534A)
        + j
        + struct.pack("<II", len(body), 0x004E4942)
        + body
    )


class Replay:
    # Live traces retained event kinds, IDs and outputs but not response envelopes
    # or Gradio temp paths. Those two pieces are synthetic in this offline test.
    def __init__(self, glb, receipt=None):
        self.glb = glb
        self.receipt = receipt or {
            "preprocess_event_id": "a" * 32,
            "sf3d_event_id": "b" * 32,
            "events": [
                {
                    "stage": s,
                    "type": t,
                    "success": True if t == "process_completed" else None,
                }
                for s in ("preprocess", "sf3d")
                for t in ("estimation", "process_starts", "process_completed")
            ],
        }
        self.fn = None
        self.joins = []
        self.lose_ack = False
        self.uploads = 0
        self.interrupt_stream = False
        self.downloads = 0
        self.events_seen = []
        self.file = "/tmp/gradio/" + "c" * 40 + "/fixture.glb"
        self.work = None

    def configuration(self):
        return {
            "gradio_version": "4.41.0",
            "protocol": "sse_v3",
            "requires_bg_remove": 4,
            "run_button": 5,
            "space_head_revision": self.receipt.get("head_revision"),
            "runtime_revision": None,
        }

    def upload(self, png_bytes):
        self.uploads += 1
        self.last_upload = png_bytes
        return "/tmp/gradio/" + "d" * 40 + "/foreground.png"

    def join(self, session, fn, data):
        if self.work:
            expected = "preprocess_waiting" if fn == 4 else "generation_waiting"
            assert (
                json.loads((self.work / "provider.json").read_bytes())["stage"]
                == expected
            )
        self.joins.append(fn)
        self.fn = fn
        if fn == 5 and self.lose_ack:
            self.lose_ack = False
            raise TimeoutError("lost acknowledgement")
        return self.receipt[("preprocess_event_id" if fn == 4 else "sf3d_event_id")]

    def events(self, session):
        step = "preprocess" if self.fn == 4 else "sf3d"
        eid = self.receipt[step + "_event_id"]
        yield {"msg": "heartbeat"}
        for rec in self.receipt["events"]:
            if rec["stage"] != step:
                continue
            if self.interrupt_stream and step == "sf3d":
                self.interrupt_stream = False
                raise TimeoutError("stream disconnected")
            self.events_seen.append(rec["type"])
            items = [None] * 6
            if step == "preprocess":
                items[0] = {"value": "Run"}
            else:
                items[4] = {
                    "value": {
                        "path": self.file,
                        "url": sf.ORIGIN + "/file=" + self.file,
                    }
                }
            yield {
                "event_id": eid,
                "msg": rec["type"],
                "success": rec["success"],
                "output": {"data": items},
            }

    def download(self, path, dest):
        assert path == self.file
        self.downloads += 1
        dest.write_bytes(self.glb)


class ProviderTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.work = self.root / "work"
        self.rgba = self.root / "source.png"
        self.rgba.write_bytes(png())
        self.src = hashlib.sha256(self.rgba.read_bytes()).hexdigest()
        self.api = Replay(fixture_glb())
        self.api.work = self.work

    def call(self, allow=True, cancelled=None):
        return sf._generate(
            self.rgba, self.src, self.work, allow, None, cancelled, self.api
        )

    def test_requires_explicit_selection_and_cancels_before_upload(self):
        for flag in (None, False, 1, "true"):
            with self.assertRaises(PermissionError):
                self.call(flag)
        with self.assertRaises(InterruptedError):
            self.call(cancelled=lambda: True)
        self.assertEqual((self.api.joins, self.api.uploads), ([], 0))

    def test_native_bytes_retained_source_bound_and_no_repost(self):
        result = self.call()
        self.assertEqual(result["artifact"]["faces"], 1)
        self.assertNotIn("session_hash", result)
        self.assertEqual((self.work / "model.glb").read_bytes(), self.api.glb)
        self.assertEqual(self.call(False), result)
        self.assertEqual(self.api.joins, [4, 5])
        self.assertEqual(self.api.downloads, 1)
        self.src = "1" * 64
        with self.assertRaises(ValueError):
            self.call()
        self.assertEqual(self.api.joins, [4, 5])

    def test_hidden_and_faint_rgb_are_scrubbed_before_upload(self):
        with Image.open(self.rgba) as im:
            im.putpixel((0, 0), (205, 75, 149, 0))
            im.putpixel((1, 0), (205, 75, 149, 31))
            im.save(self.rgba)
        self.call()
        with Image.open(io.BytesIO(self.api.last_upload)) as clean:
            for x in (0, 1):
                self.assertEqual(clean.getpixel((x, 0)), (0, 0, 0, 0))
            self.assertEqual(clean.getpixel((6, 6)), (220, 110, 20, 255))

    def test_cancel_during_config_or_just_before_join_does_not_submit(self):
        active = [False]
        config = self.api.configuration

        def stop():
            active[0] = True
            return config()

        self.api.configuration = stop
        with self.assertRaises(InterruptedError):
            self.call(cancelled=lambda: active[0])
        self.assertEqual(self.api.uploads, 0)
        self.api.configuration = config
        active[0] = False

        def stop_join(stage):
            if stage == "generation_waiting":
                active[0] = True

        with self.assertRaises(InterruptedError):
            sf._generate(
                self.rgba,
                self.src,
                self.work,
                True,
                stop_join,
                lambda: active[0],
                self.api,
            )
        self.assertEqual(self.api.joins, [4])
        self.call()
        self.assertEqual(self.api.joins, [4, 5])

    def test_lost_join_ack_and_disconnected_stream_resume_only_via_get(self):
        self.api.lose_ack = True
        with self.assertRaises(sf.Pending):
            self.call()
        self.assertEqual(self.api.joins, [4, 5])
        self.api.interrupt_stream = True
        with self.assertRaises(sf.Pending):
            self.call()
        self.assertEqual(self.call()["generation_event_id"], "b" * 32)
        self.assertEqual(self.api.joins, [4, 5])

    def test_abnormal_result_is_terminal_and_never_downloaded(self):
        self.api.file = "/tmp/gradio/no-trusted-path/invalid.txt"
        with self.assertRaises(sf.RemoteFailure):
            self.call()
        with self.assertRaises(sf.RemoteFailure):
            self.call()
        self.assertEqual(self.api.downloads, 0)
        self.assertEqual(self.api.joins, [4, 5])

    def test_untrusted_file_and_interrupted_download_are_not_exposed(self):
        known = "/tmp/gradio/" + "e" * 40 + "/model.glb"
        for url in (
            "https://example.org/file=" + known,
            sf.ORIGIN + "/file=" + known + "?wrong",
            sf.ORIGIN + "/different",
        ):
            with self.assertRaises(ValueError):
                sf.output_path(
                    [None] * 4 + [{"value": {"path": known, "url": url}}, None]
                )
        self.api.glb = self.api.glb[:24]
        with self.assertRaises(ValueError):
            self.call()
        self.assertFalse((self.work / "model.glb").exists())
        self.api.glb = fixture_glb()
        self.call()
        self.assertEqual(self.api.joins, [4, 5])
        self.assertEqual(self.api.downloads, 2)

    def test_corrupt_texture_and_out_of_range_mesh_rejected(self):
        jpeg = io.BytesIO()
        Image.new("RGB", (24, 24), "red").save(jpeg, "JPEG")
        for bad in (
            fixture_glb(jpeg.getvalue()[:-8], "image/jpeg"),
            fixture_glb(last_index=7),
        ):
            self.api.glb = bad
            with self.assertRaises((OSError, ValueError)):
                self.call()
            self.assertFalse((self.work / "model.glb").exists())
            self.assertEqual(self.api.joins, [4, 5])

    @unittest.skipUnless(
        os.getenv("STUDIO_SF3D_REPLAY"),
        "Set STUDIO_SF3D_REPLAY to the separate two-job local evidence directory.",
    )
    def test_two_genuine_event_traces_and_native_glbs_offline(self):
        evidence = Path(os.environ["STUDIO_SF3D_REPLAY"])
        for i, (folder, sha) in enumerate(
            (
                (
                    evidence,
                    "fa514f4c7e13423e95d74c5f1aa997c8f00d1fd796779753bee97b6ad5e7c961",
                ),
                (
                    evidence / "fresh",
                    "0f9722e8a1adf270eabe4cdf97f97cb07b4fd4a4339024cfc31ba22697831db4",
                ),
            )
        ):
            with self.subTest(i=i):
                rec = json.loads((folder / "receipt.json").read_bytes())
                self.assertEqual(rec["glb_sha256"], sha)
                self.api = Replay((folder / "model.glb").read_bytes(), rec)
                self.work = self.root / str(i)
                self.api.work = self.work
                self.rgba = folder / "derived-input.png"
                # The original private experiment retained faint background pixels.
                # Replays use the same recorded input but the new adapter clears them.
                self.assertEqual(
                    hashlib.sha256(self.rgba.read_bytes()).hexdigest(),
                    rec["upload_sha256"],
                )
                self.src = rec["source_sha256"]
                result = self.call()
                self.assertEqual(
                    result["upload_sha256"],
                    hashlib.sha256(self.api.last_upload).hexdigest(),
                )
                self.assertEqual(result["artifact"]["sha256"], sha)
                self.assertEqual(result["generation_event_id"], rec["sf3d_event_id"])
                self.assertEqual(self.api.joins, [4, 5])
                self.assertEqual(
                    self.api.events_seen, [v["type"] for v in rec["events"]]
                )
                self.assertEqual(result["artifact"]["bytes"], rec["glb_size"])
