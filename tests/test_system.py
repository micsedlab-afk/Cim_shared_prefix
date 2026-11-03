"""Executable, source-labeled synthetic fixtures; no measured timing claims."""
import sys
import unittest
from dataclasses import replace
from math import isclose
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cim_shared_prefix import Workload, SignedMapping, ScoreInterface
from cim_shared_prefix.system import (SystemInputs, evaluate_system, plan_residency, PathCalibration,
    GPUBlock, REQUIRED_GPU_STAGES, evaluate_decoder, OverlapEvidence, magma_a100_block)


def fixture(w, i):
    stages = {stage: (0.00002 if stage == 'qkv_proj' else 0.00001)
              for stage in REQUIRED_GPU_STAGES}
    stages['attn_score'] = 0.00008  # excluded from the accelerated path
    return GPUBlock(w.prefix_tokens+w.suffix_tokens+i+1, w.batch, w.query_heads, w.kv_heads,
                    w.head_dim, w.query_heads*w.head_dim, w.head_dim*4*w.query_heads,
                    'fp16', stages, 0 if w.suffix_tokens+i == 0 else 0.000002,
                    0.000003, 'synthetic LLMCompass fixture only',
                    'geometry zero suffix' if w.suffix_tokens+i == 0 else 'synthetic suffix score',
                    'synthetic structural operators', 'HBM2e', 'fixture-commit')


def timing():
    return PathCalibration(1e11, 1e10, 1e10, 1e10, 1e10,
                           1e-10, 2e-10, 3e-7, 16,
                           provenance={k:'synthetic '+k for k in (
                              'array','prefix_link','query_link','score_link','program','local_reduce','gpu_reduce','gpu_quant')})


class SystemLegacy(unittest.TestCase):
    def setUp(self):
        self.w = Workload(1, 128, 32, 8, 2, 64, 4, 8)
        self.params = SystemInputs(1e9,1e10,1e10,.005,.008,0.,1e-9,0.,'test synthetic parameters','test synthetic GPU')

    def test_16_gib_large_completed_scores(self):
        # Workstream contract: 32 layers, batch=1024, 32 heads, prefix=4096, FP32 scores.
        large=Workload(1024,4096,0,32,32,128,32,1)
        self.assertEqual(ScoreInterface().bytes_per_generated_token(large),16*2**30)

    def test_full_breakdown(self):
        r = evaluate_system(self.w, self.params)
        self.assertGreater(r.physical_qk_compute_s, 0)
        self.assertEqual(r.recurring_refill_s, 0)
        self.assertTrue(isclose(r.amortized_initial_programming_s, .001))
        self.assertTrue(isclose(r.total_s, sum(vars(r).values())))
        self.assertEqual(r.partial_merge_s, 0)

    def test_refill_and_partial_transfer(self):
        full = evaluate_system(self.w, self.params)
        partial = evaluate_system(self.w, replace(self.params,prefix_miss_fraction_per_token=1.),scores=ScoreInterface(partials_per_score=2))
        self.assertGreater(partial.completed_score_transfer_s,full.completed_score_transfer_s)
        self.assertGreater(partial.partial_merge_s,0)
        self.assertGreater(partial.recurring_refill_s,0)

    def test_signed_mapping_affects_compute(self):
        a = evaluate_system(self.w,self.params)
        b = evaluate_system(self.w,self.params,mapping=SignedMapping(1,1))
        self.assertTrue(isclose(a.physical_qk_compute_s,4*b.physical_qk_compute_s))
        self.assertTrue(isclose(a.amortized_initial_placement_s,2*b.amortized_initial_placement_s))

    def test_rejects_invalid_sources(self):
        with self.assertRaises(ValueError): replace(self.params,score_link_bytes_per_second=0)
        with self.assertRaises(ValueError): replace(self.params,source='')


class ResidencyTests(unittest.TestCase):
    def setUp(self):
        self.w = Workload(2, 128, 0, 4, 2, 64, 3, 2)

    def test_full_residency_no_stream(self):
        p=plan_residency(self.w,1_000_000,tile_tokens=64)
        self.assertEqual(p.streamed_tiles,())
        self.assertEqual(p.scratch_bytes,0)
        self.assertEqual(p.total_prefix_bytes,2*self.w.prefix_key_bytes_all_layers)

    def test_finite_residency_refills(self):
        p=plan_residency(self.w,20_000,tile_tokens=64)
        self.assertTrue(p.pinned_tiles)
        self.assertTrue(p.streamed_tiles)
        self.assertLessEqual(p.occupancy_bytes,p.capacity_bytes)
        self.assertEqual(len(p.pinned_tiles)+len(p.streamed_tiles),6)

    def test_no_capacity_for_one_tile(self):
        with self.assertRaises(ValueError): plan_residency(self.w,4,tile_tokens=64)

    def test_shared_prefix_not_batch_replicated(self):
        p=plan_residency(self.w,1_000_000,tile_tokens=64)
        doubled=plan_residency(replace(self.w,batch=8),1_000_000,tile_tokens=64)
        self.assertEqual(p.total_prefix_bytes,doubled.total_prefix_bytes)

    def test_last_tile_accounted_once(self):
        w=replace(self.w,prefix_tokens=129)
        p=plan_residency(w,1_000_000,tile_tokens=64)
        self.assertEqual(len(p.pinned_tiles),9)
        self.assertEqual(p.total_prefix_bytes,2*w.prefix_key_bytes_all_layers)

    def test_plan_detects_missing_tile(self):
        p=plan_residency(self.w,1_000_000,tile_tokens=64)
        with self.assertRaises(ValueError): replace(p,pinned_tiles=p.pinned_tiles[:-1]).verify(self.w,SignedMapping())


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.w=Workload(2,128,0,4,2,64,3,2)
        self.gpus=tuple(fixture(self.w,i) for i in range(self.w.output_tokens))

    def test_full_path_all_components(self):
        p=plan_residency(self.w,20_000,tile_tokens=64)
        r=evaluate_decoder(self.w,p,timing(),self.gpus)
        self.assertEqual(len(r.tokens),2)
        self.assertGreater(r.tokens[0].initial_placement_s,0)
        self.assertEqual(r.tokens[1].initial_placement_s,0)
        self.assertEqual(r.tokens[0].recurring_refill_s,r.tokens[1].recurring_refill_s)
        self.assertGreater(r.tokens[0].local_signed_reduce_s,0)
        self.assertGreater(r.tokens[0].query_transfer_s,0)
        self.assertEqual(r.tokens[0].gpu_partial_merge_s,0)
        self.assertTrue(isclose(r.total_decode_s, sum(t.serial_s for t in r.tokens)))

    def test_full_residency_has_zero_refill(self):
        p=plan_residency(self.w,1_000_000,tile_tokens=64)
        r=evaluate_decoder(self.w,p,timing(),self.gpus)
        self.assertTrue(all(t.recurring_programming_s==0 and t.recurring_refill_s==0 for t in r.tokens))

    def test_refill_increases_token_tpot(self):
        fast=evaluate_decoder(self.w,plan_residency(self.w,1_000_000,tile_tokens=64),timing(),self.gpus)
        miss=evaluate_decoder(self.w,plan_residency(self.w,20_000,tile_tokens=64),timing(),self.gpus)
        self.assertGreater(miss.tokens[1].serial_s,fast.tokens[1].serial_s)

    def test_signed_reduction_and_link_partial(self):
        p=plan_residency(self.w,1_000_000,tile_tokens=64)
        merged=evaluate_decoder(self.w,p,timing(),self.gpus)
        partial=evaluate_decoder(self.w,p,timing(),self.gpus,scores=ScoreInterface(partials_per_score=2))
        self.assertEqual(merged.tokens[0].local_signed_reduce_s,partial.tokens[0].local_signed_reduce_s*1.5)
        self.assertGreater(partial.tokens[0].gpu_partial_merge_s,0)
        self.assertEqual(partial.tokens[0].completed_score_transfer_s,2*merged.tokens[0].completed_score_transfer_s)
        with self.assertRaises(ValueError): evaluate_decoder(self.w,p,timing(),self.gpus,scores=ScoreInterface(partials_per_score=5))

    def test_exact_context_per_token(self):
        p=plan_residency(self.w,1_000_000,tile_tokens=64)
        with self.assertRaises(ValueError): evaluate_decoder(self.w,p,timing(),(self.gpus[0],self.gpus[0]))

    def test_gqa_geometry_is_checked(self):
        p=plan_residency(self.w,1_000_000,tile_tokens=64)
        with self.assertRaises(ValueError): evaluate_decoder(self.w,p,timing(),(replace(self.gpus[0],kv_heads=4),self.gpus[1]))

    def test_missing_stages_rejected(self):
        p=plan_residency(self.w,1_000_000,tile_tokens=64)
        bad=dict(self.gpus[0].stages_s);bad.pop('softmax')
        with self.assertRaises(ValueError): evaluate_decoder(self.w,p,timing(),(replace(self.gpus[0],stages_s=bad),self.gpus[1]))

    def test_missing_timing_sources_rejected(self):
        with self.assertRaises(ValueError): replace(timing(),provenance={'array':'synthetic'})

    def test_unqualified_overlap_rejected(self):
        p=plan_residency(self.w,20_000,tile_tokens=64)
        with self.assertRaises(ValueError): evaluate_decoder(self.w,p,timing(),self.gpus,overlap=OverlapEvidence(True,False,True,'not proof'))

    def test_qualified_prefetch_overlap_bounded(self):
        p=plan_residency(self.w,20_000,tile_tokens=64)
        serial=evaluate_decoder(self.w,p,timing(),self.gpus)
        concurrent=evaluate_decoder(self.w,p,timing(),self.gpus,overlap=OverlapEvidence(True,True,True,'synthetic independent paths'))
        self.assertGreater(serial.total_decode_s,concurrent.total_decode_s)
        self.assertGreaterEqual(concurrent.total_decode_s,serial.total_decode_s-sum(g.stages_s['qkv_proj']*self.w.layers for g in self.gpus))
        self.assertEqual(concurrent.timing_policy,'optimistic-first-tile-qkv-overlap-sensitivity')

    def test_physical_overlap_needs_receipt(self):
        p=plan_residency(self.w,20_000,tile_tokens=64)
        with self.assertRaises(ValueError):
            evaluate_decoder(self.w,p,timing(),self.gpus,overlap=OverlapEvidence(True,True,True,'declared physical trace',evidence_kind='physical_trace'))

    def test_magma_a100_gqa_rejected(self):
        w=Workload(1,127,0,16,8,128,1,1)
        with self.assertRaises(ValueError): magma_a100_block({},w,0,suffix_score_s=0,suffix_source='none',structural_s_per_token=0,structural_source='synthetic')

    def test_magma_a100_native_profile_and_full_score_exclusion(self):
        w=Workload(1,127,0,16,16,128,1,1)
        stages=[{'stage':s,'latency_seconds':(10e-6 if s=='attn_score' else 1e-6)} for s in REQUIRED_GPU_STAGES]
        d={'schema':'magma.llmcompass-native-a100-point/v1','native_llmcompass_timing':True,'magma_latency_provider_used':False,
           'point':{'context_length':128,'batch_size':1,'heads':16,'d_head':128,'d_model':2048,'ffn_dim':8192,'precision':'fp16'},
           'memory':{'protocol':'HBM2e'},'git_commit':'synthetic','stages':stages,
           'total_block_latency_seconds':sum(s['latency_seconds'] for s in stages)}
        gpu=magma_a100_block(d,w,0,suffix_score_s=None,suffix_source='',structural_s_per_token=0,structural_source='synthetic missing structural costs assumed 0')
        self.assertTrue(isclose(gpu.other_gpu_s_per_layer,11e-6))
        r=evaluate_decoder(w,plan_residency(w,1_000_000,tile_tokens=64),timing(),(gpu,))
        self.assertTrue(isclose(r.tokens[0].gpu_other_s,11e-6))
        self.assertEqual(r.tokens[0].gpu_suffix_qk_s,0)
        with self.assertRaises(ValueError): magma_a100_block(d,replace(w,suffix_tokens=1),0,suffix_score_s=None,suffix_source='',structural_s_per_token=0,structural_source='synthetic')


if __name__ == '__main__': unittest.main()
