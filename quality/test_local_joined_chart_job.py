"""Durability/consent tests for the normal SG-chart worker. No inference or network."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine import guided_shape_job as job
from engine import joined_paint_chart as chart
from engine import joined_chart_provider as provider
from engine import local_joined_chart_job as local
from engine import triposg_cpu as sg
from engine.pipeline import SG_CHART_PIPELINE
from studio.configure import agreement


real_observer = job._chatgpt_observations


class LocalJoinedJobTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup)
        self.root=Path(t.name);self.src=self.root/"source.png";self.src.write_bytes(b"sealed synthetic input")
        self.out=self.root/"out"; self.cfg=self.root/"store.json"
        sg.atomic_json(self.cfg,{"pipeline":SG_CHART_PIPELINE,"agreement":agreement(SG_CHART_PIPELINE)})
        self.obs={"method":"codex-photo-observations-v2","source_sha256":sg.digest(self.src)}
        self.public={**self.obs}
        self.start(patch("engine.guided_provider.preflight", return_value={"available":True}))
        self.shape=self.start(patch.object(job.local_shape_job,"run",side_effect=self.baseline))
        self.vision=self.start(patch.object(job,"_chatgpt_observations",return_value=(self.obs,self.public)))
        self.legacy=self.start(patch("engine.guided_provider.generate_guided_paint",side_effect=AssertionError("two views cannot run")))
        self.cont=self.start(patch("engine.continuity_paint.paint_continuity",return_value=None))
    def start(self,mock):
        f=mock.start();self.addCleanup(mock.stop);return f
    def baseline(self,src,base,resolution,texture_size):
        base.mkdir(parents=True)
        for name in job.BASE_FILES:(base/name).write_bytes(("local "+name).encode())
        # Valid path-level fixture, deliberately not an inference test.
        m=sg.digest(base/"model.glb");sh=sg.digest(base/"shape-master.glb");fg=sg.digest(base/"input-processed.png")
        ident=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        frame=dict(image_right=[1,0,0],image_up=[0,1,0],camera_direction_from_origin=[0,0,1],pinhole={"inverse_distance":.23,"magnification":1.02,"shift_fraction":[0,.01]})
        sg.atomic_json(base/"appearance.json",dict(output_sha256=m,shape_sha256=sh,foreground_sha256=fg,original_sha256=sg.digest(src),display_from_raw=ident))
        sg.atomic_json(base/"generation.json",dict(shape_result={"sha256":sh},engine={"input_sha256":fg},foreground={"source_sha256":sg.digest(src)},
            camera=dict(camera_frame=frame,upright_transform=ident,status="candidate")))
        sg.atomic_json(base/"settings.json",dict(pipeline=job.SHAPE_PIPELINE,auto_finish_complete=True,input_sha256=sg.digest(src),
            marching_cubes_resolution=resolution,texture_size=texture_size,glb_sha256=m,artifact_sha256={n:sg.digest(base/n) for n in job.BASE_FILES}))
    def run_job(self):
        return job.run(self.src,self.out,256,1024,setup=self.cfg,pipeline=SG_CHART_PIPELINE)
    def test_unsupported_returns_local_diagnostic_without_paint_and_survives_restart(self):
        with patch.object(chart,"prepare_chart",side_effect=chart.ChartUnavailable("fixture unchartable")),patch.object(provider,"generate_chart",side_effect=AssertionError("no image slot")):
            actual=self.run_job(); recovered=self.run_job()
        self.assertEqual(actual,recovered);self.assertEqual(actual["chart_decision"],local.DIAGNOSTIC)
        self.assertEqual(sg.read_json(self.out/"chart-result.json")["paint_attempts"],0)
        self.shape.assert_called_once();self.vision.assert_called_once();self.assertTrue(job.valid_chart(self.out,actual))
        report=sg.read_json(self.out/"chart-result.json");report["private_path"]="untrusted.txt"
        sg.atomic_json(self.out/"chart-result.json",report)
        actual["artifact_sha256"]["chart-result.json"]=sg.digest(self.out/"chart-result.json")
        self.assertFalse(job.valid_chart(self.out,actual))
    def test_surviving_partial_pigment_never_prevents_geometry_review(self):
        # report from an upstream local method that only covered a thin strip
        self.cont.return_value=dict(supported_hidden_fraction=.1,output_glb="a"*64)
        with patch.object(chart,"prepare_chart",side_effect=chart.ChartUnavailable("fixture blocked")) as prep:
            actual=self.run_job()
        self.assertEqual(actual["chart_decision"],local.DIAGNOSTIC)
        self.assertEqual(sg.read_json(self.out/"chart-result.json")["coverage"],.1);prep.assert_called_once()
    def test_pose_drift_stops_before_chart_and_source_mesh_is_reused(self):
        base=self.root/"base";self.baseline(self.src,base,256,1024)
        info=sg.read_json(base/"settings.json");reg=local.display_registration(base,self.src,info)
        self.assertEqual(reg["camera_frame"]["pinhole"]["inverse_distance"],.23)
        g=sg.read_json(base/"generation.json");g["camera"]["upright_transform"][0][0]=-.9;sg.atomic_json(base/"generation.json",g)
        with self.assertRaises(ValueError):local.display_registration(base,self.src,info)
    def test_generated_but_vetoed_result_is_not_promoted_or_resubmitted(self):
        def guides(output_dir,**kwargs):
            output_dir.mkdir();(output_dir/"guide.png").write_bytes(b"guide")
            m={"chart_id":"a"*64};sg.atomic_json(output_dir/"chart.json",m);return m
        def proposed(_a,_b,_c,work,**kwargs):
            slot=work/"guided-paint";slot.mkdir()
            sheet=slot/"sheet.png";sheet.write_bytes(b"checkpointed synthesis")
            r=dict(job_recipe=chart.METHOD,chart_id="a"*64,state="completed",attempts=1,thread_id="fixture",sheet_sha256=sg.digest(sheet))
            sg.atomic_json(slot/"receipt.json",r);return dict(sheet=sheet,provenance=r)
        with patch.object(chart,"prepare_chart",side_effect=guides),patch.object(provider,"generate_chart",side_effect=proposed) as submit,patch.object(chart,"apply_chart",side_effect=chart.ChartUnavailable("failed immutable margin")):
            settings=self.run_job();self.run_job()
        r=sg.read_json(self.out/"chart-result.json")
        self.assertEqual((r["decision"],r["reason"],r["paint_attempts"]),(local.DIAGNOSTIC,"proposed_paint_rejected",1));self.assertEqual(r["output_sha256"],r["base_sha256"]);submit.assert_called_once();self.assertTrue(job.valid_chart(self.out,settings))
    def test_missing_photo_journal_after_source_continuity_recovers_only(self):
        from engine import codex_photo_observations as observer
        work=self.root/"work"; (work/"source-continuity").mkdir(parents=True)
        with patch.object(observer,"analyze_photo",side_effect=AssertionError("duplicate")), patch.object(observer,"recover_photo",side_effect=ValueError("missing journal")):
            with self.assertRaisesRegex(ValueError,"missing journal"):
                real_observer(self.src,work,sg.digest(self.src),self.cfg,pipeline=SG_CHART_PIPELINE)

    def test_no_observer_intent_until_consent_after_local_preflight(self):
        from engine import codex_photo_observations as observer
        work=self.root/"observation";work.mkdir()
        with patch.object(observer,"analyze_photo",side_effect=ValueError("local preflight")) as analyze:
            for _ in range(2):
                with self.assertRaisesRegex(ValueError,"local preflight"):
                    real_observer(self.src,work,sg.digest(self.src),self.cfg,pipeline=SG_CHART_PIPELINE)
            self.assertEqual(analyze.call_count,2)
            self.assertFalse((work/"observation-intent.json").exists())
            self.assertTrue(analyze.call_args.kwargs["check_consent"]())
            self.assertTrue((work/"observation-intent.json").exists())

    def test_lost_remote_slot_never_admits_second_paint(self):
        def guides(output_dir,**kwargs):
            output_dir.mkdir();(output_dir/"guide.png").write_bytes(b"guide")
            m={"chart_id":"b"*64};sg.atomic_json(output_dir/"chart.json",m);return m
        def interrupt(a,b,c,work,**kwargs):
            slot=work/"guided-paint";slot.mkdir()
            sg.atomic_json(slot/"receipt.json",{"attempts":1,"state":"submitted"})
            kwargs["before_submission"]()
            raise KeyboardInterrupt
        with patch.object(chart,"prepare_chart",side_effect=guides),patch.object(provider,"generate_chart",side_effect=interrupt) as remote:
            with self.assertRaises(KeyboardInterrupt): self.run_job()
            work=self.out/"local-guided-work"
            (work/"guided-paint").rename(work/"lost-journal-for-test")
            with self.assertRaises(job.StageError): self.run_job()
            self.assertEqual(remote.call_count,1)

    def paint_fixture(self, output_dir, args, method):
        output_dir.mkdir()
        for name in job.EDITABLE:
            (output_dir/name).write_bytes(("painted " + name).encode())
        return dict(
            output_glb=sg.digest(output_dir/"model.glb"), source_glb=sg.digest(args["native_model"]),
            foreground_sha256=sg.digest(args["foreground"]), source_image_sha256=sg.digest(args["source_photo"]),
            visual_brief_sha256=local.json_sha(args["observation"]), method=method,
            supported_hidden_fraction=.75, locked_changed=0, unsupported_surface_changed=0,
            normals_uv_nodes_material_and_all_other_buffers_bitwise_identical=True,
            source_rear_observed=False, changed_uv=15)

    def test_applied_local_material_requires_bound_source_and_reopened_evidence(self):
        def accepted(output_dir, **args):
            stats=self.paint_fixture(output_dir,args,"source-surface-continuity-v1")
            stats.update(eligible_hidden_uv=100, supported_hidden_uv=75)
            sg.atomic_json(output_dir/"continuity-paint.json",stats)
            return stats
        self.cont.side_effect=accepted
        with patch.object(provider,"generate_chart",side_effect=AssertionError("unexpected external request")):
            settings=self.run_job()
            self.assertEqual(settings["chart_decision"],"local_continuity")
            self.assertTrue(job.valid_chart(self.out,settings))
        path=self.out/"continuity-paint.json"
        stats=sg.read_json(path);stats["visual_brief_sha256"]="a"*64;sg.atomic_json(path,stats)
        settings["artifact_sha256"][path.name]=sg.digest(path)
        self.assertFalse(job.valid_chart(self.out,settings))

    def test_single_inferred_output_agrees_with_all_source_and_sheet_digests(self):
        def guides(output_dir,**kwargs):
            output_dir.mkdir();(output_dir/"guide.png").write_bytes(b"guide")
            m={"chart_id":"b"*64};sg.atomic_json(output_dir/"chart.json",m);return m
        def proposed(source,guide,manifest,work,**kwargs):
            slot=work/"guided-paint";slot.mkdir()
            sheet=slot/"sheet.png";sheet.write_bytes(b"synthetic integration only")
            receipt=dict(job_recipe=chart.METHOD,chart_id="b"*64,state="completed",attempts=1,
                         thread_id="fixture",sheet_sha256=sg.digest(sheet),guide_sha256=sg.digest(guide),
                         directions=["joined_chart"],visual_brief_sha256=local.json_sha(self.obs))
            sg.atomic_json(slot/"receipt.json",receipt);kwargs["before_submission"]()
            return dict(sheet=sheet,provenance=receipt)
        def projected(output_dir,**kwargs):
            stats=self.paint_fixture(output_dir,kwargs,chart.METHOD)
            receipt=kwargs["receipt"]
            stats.update(source_observed_changed=0,sheet_sha=receipt["sheet_sha256"],guide_sha=receipt["guide_sha256"],
                         chart_id=receipt["chart_id"],generation_id=receipt["thread_id"])
            sg.atomic_json(output_dir/"chart-paint.json",stats)
            return stats
        with patch.object(chart,"prepare_chart",side_effect=guides),patch.object(provider,"generate_chart",side_effect=proposed) as send,patch.object(chart,"apply_chart",side_effect=projected):
            settings=self.run_job(); self.run_job()
            self.assertEqual(settings["chart_decision"],"inferred_chart")
            self.assertTrue(job.valid_chart(self.out,settings));send.assert_called_once()
        path=self.out/"chart-paint.json"
        stats=sg.read_json(path);stats["unsupported_surface_changed"]=3;sg.atomic_json(path,stats)
        settings["artifact_sha256"][path.name]=sg.digest(path)
        self.assertFalse(job.valid_chart(self.out,settings))

    def test_corrupt_saved_editable_master_cannot_be_resealed(self):
        def accepted(output_dir,**kwargs):
            output_dir.mkdir()
            for name in job.EDITABLE: (output_dir/name).write_bytes(("edited "+name).encode())
            p=dict(output_glb=sg.digest(output_dir/"model.glb"),supported_hidden_fraction=.9)
            sg.atomic_json(output_dir/"continuity-paint.json",p)
            return p
        self.cont.side_effect=accepted
        with patch.object(local,"_result",side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): self.run_job()
        (self.out/"local-guided-work"/"source-continuity"/"texture.png").write_bytes(b"changed")
        with self.assertRaises(job.StageError): self.run_job()
        self.assertFalse((self.out/"settings.json").exists())

    def test_surface_consent_does_not_authorise_two_views(self):
        provider.chart_consent(self.cfg,SG_CHART_PIPELINE)
        # Access real function instead of setUp's submission gate fixture.
        self.assertNotEqual(agreement(SG_CHART_PIPELINE),agreement(job.OBSERVED_PIPELINE))
        sg.atomic_json(self.cfg,{"pipeline":job.OBSERVED_PIPELINE,"agreement":agreement(job.OBSERVED_PIPELINE)})
        with self.assertRaises(provider.provider.TurnaroundProviderError):provider.chart_consent(self.cfg,SG_CHART_PIPELINE)

if __name__=="__main__":
    unittest.main()
