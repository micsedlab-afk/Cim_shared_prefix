"""Numerical oracle and finite 7nm K-CIM/V-SRAM end-to-end prefix engine tests."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from cim_shared_prefix.contracts import Workload
from cim_shared_prefix.prefix_context_engine import (
    ContextRates,simulate_prefix_context,online_prefix_context,
    fuse_prefix_suffix,_blocking_pipeline,
)

def rates(**overrides):
    kw=dict(
        key_supply_Bps=1e12,key_program_Bps=1e12,
        v_supply_Bps=1e12,v_program_Bps=1e12,
        query_Bps=1e12,local_v_read_Bps=16e12,
        exp_per_s=1e12,scalar_ops_per_s=1e12,
        pv_mac_per_s=64e12,context_output_Bps=1e12,
        partial_scalings_per_s=1e12,signed_adds_per_s=1e12,
        source="test-defined assumed digital rates, not physical synthesis",
    )
    kw.update(overrides)
    return ContextRates(**kw)


class NearMemoryPrefixContextTests(unittest.TestCase):
    def setUp(self):
        self.w=Workload(1,2048,128,32,8,128,32,128)

    def pipe(self,w=None,**overrides):
        kw=dict(area_mm2=16,support_mm2=.5,ports=256,rates=rates(),
                gpu_prefix_qk_s=1.0626699362432564e-6)
        kw.update(overrides)
        return simulate_prefix_context(w or self.w,**kw)

    def test_physical_k_v_storage_and_endpoint(self):
        p=self.pipe()
        self.assertIsNotNone(p)
        self.assertEqual(p.cim_k_macros,4096)
        self.assertEqual(p.sram_v_macros,128)
        self.assertEqual(p.full_v_bytes_per_layer,4194304)
        self.assertEqual(p.per_token_v_read_bytes,4194304)
        self.assertEqual(p.prefix_score_link_bytes_avoided,262144)
        self.assertEqual(p.context_and_lse_link_bytes,8320)
        self.assertLessEqual(p.total_physical_area_mm2,16)
        self.assertGreater(p.sram_pipeline_macros,0)
        self.assertEqual(p.max_pinned_layers,1)
        self.assertGreaterEqual(p.total_waves,16)
        self.assertGreater(p.mean_context_s,p.warm_attention_context_s)
        self.assertGreater(p.cold_KV_supply_and_program_s,0)
        self.assertGreater(p.warm_attention_context_s,0)
        self.assertEqual(len(p.stage_totals_s),6)

    def test_storage_and_bandwidth_causality(self):
        fast=self.pipe()
        slow=self.pipe(rates=rates(local_v_read_Bps=1e12))
        self.assertIsNotNone(fast)
        self.assertGreater(slow.mean_context_s,fast.mean_context_s)
        self.assertGreater(slow.mean_context_s,1e-6)
        short=self.pipe(w=Workload(1,2048,128,32,8,128,32,1))
        self.assertGreater(short.mean_context_s,fast.mean_context_s)
        self.assertIsNone(self.pipe(area_mm2=14))
        self.assertIsNotNone(self.pipe(area_mm2=15))

    def test_zero_adc_or_device_claim_never_inferred(self):
        with self.assertRaisesRegex(ValueError,"V remains FP16"):
            self.pipe(v_bits=4)
        with self.assertRaisesRegex(ValueError,"positive finite"):
            self.pipe(rates=rates(local_v_read_Bps=-1))
        with self.assertRaisesRegex(ValueError,"tiled"):
            self.pipe(w=Workload(1,2047,128,32,8,128,32,128))
        p=self.pipe(gpu_prefix_qk_s=None)
        self.assertIsNone(p.prefix_context_vs_gpu_QK_model_difference_s)

    def test_streaming_softmax_pv_matches_full_attention(self):
        scores=[-1000.,-5.,0.,2.,11.,-8.,1000.,.1,2.4,-1.7,3.8]
        values=[[float((i*7+j*3)%17-8)/9 for j in range(5)]
                for i in range(len(scores))]
        m=max(scores)
        e=[math.exp(x-m) for x in scores]
        den=sum(e)
        expected=[sum(a*v[j] for a,v in zip(e,values))/den for j in range(5)]
        expected_lse=m+math.log(den)
        for tile in (1,2,3,4,16):
            result,lse=online_prefix_context(scores,values,tile_size=tile)
            for a,b in zip(result,expected):
                self.assertAlmostEqual(a,b,places=12)
            self.assertAlmostEqual(lse,expected_lse,places=12)

    def test_lse_fusion_matches_direct_softmax_for_suffix(self):
        ps=[-12,0,4,2,1]
        vs=[[float(i+j)/9 for j in range(3)] for i in range(5)]
        ss=[3,2,5]
        vv=[[float(i-j)/7 for j in range(3)] for i in range(3)]
        po,pl=online_prefix_context(ps,vs,tile_size=2)
        so,sl=online_prefix_context(ss,vv,tile_size=2)
        combined=fuse_prefix_suffix(po,pl,so,sl)
        full,_=online_prefix_context(ps+ss,vs+vv,tile_size=3)
        for a,b in zip(combined,full):
            self.assertAlmostEqual(a,b,places=12)

    def test_no_infinite_buffer_score_link_or_unqualified_full_tpot(self):
        p=self.pipe()
        self.assertGreater(p.sram_pipeline_area_mm2,0)
        self.assertLess(p.context_and_lse_link_bytes,p.prefix_score_link_bytes_avoided)
        self.assertIn("unqualified",p.evidence)
        self.assertNotIn("speedup",p.json_record())


if __name__=="__main__":
    unittest.main()
