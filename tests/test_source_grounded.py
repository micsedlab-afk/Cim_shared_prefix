"""Physical-geometry, GQA, fail-closed and upstream arithmetic regression checks."""
import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from cim_shared_prefix.contracts import Workload
from cim_shared_prefix.source_grounded import (
  Llama31,MacroSpec,PhysicalCIM,placement,counts,sram_storage,RateSensitivity,
  s2_components,GQAGPUPoint,compare,REQUIRED_NON_SCORE_STAGES,sram_placement)
from cim_shared_prefix.gqa_native import gqa_shapes,run_native_fp16
from run_native_sinangil_check import run as run_source_macro


def w(p=2048,b=1,o=32,s=128): return Workload(b,p,s,32,8,128,32,o,key_bits=4)
def rates(bw=1e12):return RateSensitivity(bw,bw,bw,bw,bw,1e12,1e12,0,{
    k:'named and unqualified unit-test sensitivity' for k in
    ('prefix_link','sram_to_gpu','query_link','score_link','program','reduction_adds','partial_scales','quantization','cim_controller_s_per_layer','sram_controller_s_per_layer')},
    sram_write_Bps=bw,sram_write_provenance='synthetic SRAM write service')
def gpu_point(work:Workload,index:int,*,complete:bool=False):
    st={k:(1e-6 if complete else None) for k in REQUIRED_NON_SCORE_STAGES}
    return GQAGPUPoint(work.prefix_tokens+work.suffix_tokens+index+1,work.batch,32,8,128,4096,14336,32,128256,'fp16',
                        st,2e-6 if complete else None,1e-6 if complete else None,3e-6 if complete else None,
                        'named unit-test-only fabricated profile',score_precision='fp16',operator_evidence='synthetic_unit_test')

class GroundedTests(unittest.TestCase):
    def test_exact_model(self):
        Llama31().validate(w())
        with self.assertRaises(ValueError):Llama31().validate(Workload(1,2048,128,32,32,128,32,1))
    def test_macro_facts(self):
        a=MacroSpec();self.assertEqual(a.capacity_bytes,512);self.assertEqual(a.rows*a.logical_outputs,1024)
        self.assertAlmostEqual((2*1024)/a.native_access_s/1e9,372.36363636,places=5)
        with self.assertRaises(ValueError):replace(a,area_mm2=.0001)
    def test_full_macro_count_matches_area_branch(self):
        a=placement(w(o=1),PhysicalCIM(131072,4096))
        self.assertEqual(a.macros_required,131072)
        self.assertEqual(a.layer_total_macros,4096)
        self.assertEqual(a.initial_placement_bytes,67108864)
        self.assertEqual(a.recurring_refill_bytes_per_token,0)
        self.assertAlmostEqual(a.complete_macro_area_required_mm2,419.4304)
    def test_native_calls_and_reductions(self):
        a=counts(w(o=1),PhysicalCIM(131072,4096))
        self.assertEqual(a['native_macro_calls_per_token'],1048576)
        self.assertEqual(a['logical_signed_qk_macs_per_token']*4,1073741824)
        self.assertEqual(a['complete_score_bytes_per_token'],8388608)
        self.assertEqual(a['digital_signed_combines_per_token'],7*(8388608//4))
        self.assertEqual(a['digital_partial_rescale_multiplies_per_token'],8*(8388608//4))
    def test_sram_area_reference(self):
        a=sram_storage(w(o=1));self.assertEqual(a['macro_count'],1024)
        self.assertAlmostEqual(a['total_macro_area_mm2'],9.192995,places=5)
    def test_gqa_shared_K_not_four_copies(self):
        a=counts(w(b=1),PhysicalCIM(131072,1024))
        b=counts(w(b=8),PhysicalCIM(131072,1024))
        self.assertEqual(a['signed_physical_key_bytes'],b['signed_physical_key_bytes'])
        self.assertEqual(a['native_macro_calls_per_token']*8,b['native_macro_calls_per_token'])
        self.assertEqual(a['complete_score_bytes_per_token']*8,b['complete_score_bytes_per_token'])
    def test_capacity_and_scratch(self):
        a=placement(w(),PhysicalCIM(1024,64));self.assertEqual(sum(a.pinned_per_layer),1023)
        self.assertEqual(a.scratch_macros,1);self.assertEqual(sum(a.streamed_per_layer),131072-1023)
        self.assertGreater(a.native_single_scratch_schedule_s_per_token,a.native_access_floor_s_per_token)
        self.assertLessEqual(sum(a.pinned_per_layer)+a.scratch_macros,a.macro_budget)
    def test_one_macro_budget_feasible(self):
        a=placement(w(),PhysicalCIM(1,1));self.assertEqual(sum(a.pinned_per_layer),0)
        self.assertEqual(a.scratch_macros,1)
    def test_invalid_cap_and_ports(self):
        with self.assertRaises(ValueError):PhysicalCIM(0,1)
        with self.assertRaises(ValueError):PhysicalCIM(1,0)
    def test_signed_q8_double_calls(self):
        a=placement(w(),PhysicalCIM(131072,256,q_bits=4));b=placement(w(),PhysicalCIM(131072,256,q_bits=8))
        self.assertEqual(b.native_macro_calls_per_token,2*a.native_macro_calls_per_token)
        self.assertEqual(a.macros_required,b.macros_required)
    def test_amortized_pins(self):
        a=s2_components(w(o=1),PhysicalCIM(131072,256),rates())
        b=s2_components(w(o=32),PhysicalCIM(131072,256),rates())
        self.assertAlmostEqual(a['amortized_initial_programming_s']/32,b['amortized_initial_programming_s'])
        self.assertEqual(a['refill_s'],b['refill_s'])
    def test_macro_bound_not_complete(self):
        a=compare(w(),PhysicalCIM(1024,64))
        self.assertIsNone(a['S2_CIM']['tpot_s']);self.assertIsNone(a['S0_over_S2'])
        self.assertEqual(a['source_status']['periphery'],'unspecified')
    def test_periphery_explicit_unqualified(self):
        a=compare(w(),PhysicalCIM(1024,64),rates=rates())
        self.assertIsNone(a['S2_CIM']['tpot_s'])
        self.assertGreater(a['S2_CIM']['sensitivity_periphery_and_macro_s'],0)
        self.assertIn(a['S2_CIM']['sensitivity_bottleneck_excluding_GPU'],a['S2_CIM']['sensitivity_components_s'])
    def test_rates_missing_source(self):
        with self.assertRaises(ValueError):replace(rates(),provenance={'program':'fake'})
    def test_full_gpu_profile_requires_real_geometry(self):
        p=gpu_point(w(),0,complete=True);p.validate(w(),0)
        with self.assertRaises(ValueError):replace(p,kv_heads=32).validate(w(),0)
        with self.assertRaises(ValueError):replace(p,ffn_dim=16384).validate(w(),0)
    def test_fp16_not_relabelled_bf16(self):
        p=gpu_point(w(),0,complete=True)
        with self.assertRaises(ValueError):replace(p,gpu_precision='bf16').validate(w(),0)
        with self.assertRaises(ValueError):replace(p,prefix_gpu_precision='fp16').validate(w(),0)
    def test_missing_gpu_operator_rejected_for_full_tpot(self):
        ww=w(o=1);p=gpu_point(ww,0)
        result=compare(ww,PhysicalCIM(131072,256),gpu_points=(p,),rates=rates())
        self.assertTrue(all(result[k]['tpot_s'] is None for k in ('S0_GPU','S1_SRAM','S2_CIM')))
    def test_complete_synthetic_gpu_gives_explicit_conditional_only(self):
        ww=w(o=2);ps=tuple(gpu_point(ww,i,complete=True) for i in range(2))
        result=compare(ww,PhysicalCIM(131072,256),gpu_points=ps,rates=rates())
        self.assertGreater(result['S2_CIM']['tpot_s'],0)
        self.assertGreater(result['S1_SRAM']['tpot_s'],0)
        self.assertEqual(result['evidence_class'],'conditional_model_with_explicit_unqualified_periphery')
        self.assertGreater(result['S0_over_S2'],0)
    def test_gpu_partial_points_rejected(self):
        ww=w(o=2)
        with self.assertRaises(ValueError):compare(ww,PhysicalCIM(1024,1),gpu_points=(gpu_point(ww,0),),rates=rates())
    def test_s2_prefix_score_not_charged_twice(self):
        ww=w(o=1);p=gpu_point(ww,0,complete=True)
        v=compare(ww,PhysicalCIM(131072,256),gpu_points=(p,),rates=rates())
        # S2 uses suffix score only. S0 includes source full score. S1 includes Q4K4 GPU prefix score.
        base=p.non_score_total_s(ww.layers)
        x=s2_components(ww,PhysicalCIM(131072,256),rates())
        self.assertAlmostEqual(v['S2_CIM']['tpot_s'],base+ww.layers*p.score_suffix_s+sum(x.values()))
        self.assertAlmostEqual(v['S0_GPU']['tpot_s'],base+ww.layers*p.score_full_s)
    def test_exact_gqa_batched_matmul_shapes(self):
        sh=gqa_shapes(w(o=1),0)
        self.assertEqual(sh['k_proj']['input_b'],(4096,1024))
        self.assertEqual(sh['gate_proj']['input_b'],(4096,14336))
        self.assertEqual(sh['full_score']['input_a'],(1,8,4,128))
        self.assertEqual(sh['full_score']['input_b'],(1,8,128,2177))
        self.assertEqual(sh['attention_apply']['input_a'],(1,8,4,2177))
    def test_no_silent_native_fallback(self):
        with self.assertRaises(FileNotFoundError):run_native_fp16(w(o=1),0,Path('/nonexistent_magma'))
    def test_sram_finite_refills(self):
        ww=w(o=1)
        a=sram_placement(ww,64)
        self.assertEqual(a.required_macros,1024)
        self.assertEqual(sum(a.pinned_per_layer),63)
        self.assertEqual(sum(a.streamed_per_layer),961)
        self.assertEqual(a.scratch_macros,1)
        self.assertEqual(a.physical_key_bytes_transferred_to_gpu_per_token,33554432)

    def test_sram_residency_affects_s1_only(self):
        ww=w(o=2);p=tuple(gpu_point(ww,i,complete=True) for i in range(2))
        full=compare(ww,PhysicalCIM(131072,256),gpu_points=p,rates=rates(),sram_budget_macros=1024)
        part=compare(ww,PhysicalCIM(131072,256),gpu_points=p,rates=rates(),sram_budget_macros=64)
        self.assertGreater(part['S1_SRAM']['tpot_s'],full['S1_SRAM']['tpot_s'])
        self.assertEqual(part['S2_CIM']['tpot_s'],full['S2_CIM']['tpot_s'])

    def test_manifest_runner_without_gpu_has_null_speedup(self):
        from run_matched_system import read_manifest
        ww=w(o=1)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'scenario.json'
            p.write_text(json.dumps({'schema':'cim-shared-prefix-matched-system-request/v1',
                'workload':{'batch':ww.batch,'prefix_tokens':ww.prefix_tokens,
                            'suffix_tokens':ww.suffix_tokens,'query_heads':32,'kv_heads':8,
                            'head_dim':128,'layers':32,'output_tokens':1,'key_bits':4},
                'cim_physical':{'macro_budget':4096,'active_macro_ports':256}}))
            result=read_manifest(p)
            self.assertEqual(result['S0_GPU']['status'],'unsupported_without_exact_GQA_GPU_stages')
            self.assertIsNone(result['S0_over_S2'])

    def test_manifest_rejects_parent_gpu_file_escape(self):
        from run_matched_system import read_manifest
        ww=w(o=1)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'scenario.json'
            p.write_text(json.dumps({'schema':'cim-shared-prefix-matched-system-request/v1',
                'workload':{'batch':1,'prefix_tokens':2048,'suffix_tokens':128,'query_heads':32,
                            'kv_heads':8,'head_dim':128,'layers':32,'output_tokens':1,'key_bits':4},
                'cim_physical':{'macro_budget':4096,'active_macro_ports':256},
                'gpu_profile_files':['../nonexistent.json']}))
            with self.assertRaises(ValueError):read_manifest(p)

    def test_native_source_ideal_macro_executed(self):
        f=ROOT/'_upstream/shared_prefix_cim/experiments/sram2llm_shared_prefix/sinangil_ideal_macro.py'
        if not f.is_file():self.skipTest('credentialed checkout or SHA-verified local copy unavailable')
        r=run_source_macro(ROOT)
        self.assertTrue(r['source_verified']);self.assertEqual(r['max_integer_dot_error'],0)
        self.assertEqual(r['native_macro_calls'],96);self.assertEqual(r['samples'],192)

if __name__=='__main__': unittest.main()
