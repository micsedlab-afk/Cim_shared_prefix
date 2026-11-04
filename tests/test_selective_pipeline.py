"""Qualify finite-buffer selective pinned-layer pipeline against source evidence."""
from __future__ import annotations

import json
from math import isclose
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from cim_shared_prefix.contracts import Workload
from cim_shared_prefix.integrated_schedule import SerialRates
from cim_shared_prefix.selective_pipeline import (
    resident_pipeline, optimize_selective, _wave_dag,
    ResidentWave,
)

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"results/integration/source_native_a100_fp16_gqa_qk_128steps.json"
R=SerialRates(1e12,1e12,1e12,1e12,1e12,1e12,
              source="test scenario preserving six unqualified rates")


class SelectivePipelineTests(unittest.TestCase):
    def setUp(self):
        self.w=Workload(1,2048,128,32,8,128,32,128)
        source=json.loads(SRC.read_text())
        self.full=tuple(s["full_score_s"] for s in source["steps"])
        self.suffix=tuple(s["suffix_score_s"] for s in source["steps"])

    def make_pipeline(self, *,budget=16,ports=256,stage_area=.1, rates=R):
        return resident_pipeline(
            self.w,area_mm2=budget,support_mm2=.5,
            active_ports=ports,rates=rates,
            operand_stage_support_mm2=stage_area,
        )

    def test_pinned_layer_cold_amortization_and_all_128_source_steps(self):
        p=self.make_pipeline()
        self.assertIsNotNone(p)
        self.assertEqual(p.macro_count_one_layer,4096)
        self.assertEqual(p.pinned_layers_max,1)
        self.assertEqual(p.groups_per_wave,64)
        self.assertEqual(len(p.waves),16)
        self.assertEqual(p.physical_storage_macros,10)
        self.assertAlmostEqual(p.physical_stage_buffer_area_mm2,
                               10*(512*512)/(29.2e6),places=12)
        self.assertLessEqual(p.physical_stage_buffer_area_mm2,p.stage_buffer_limit_mm2)
        self.assertAlmostEqual(p.warm_pipelined_s_per_layer*1e6,.783872,places=8)
        self.assertAlmostEqual(p.cold_source_plus_program_s_per_layer*1e6,4.194304,places=8)
        self.assertGreater(p.warm_nonoverlapped_s_per_layer,p.warm_pipelined_s_per_layer)

        result=optimize_selective(
            self.w,p,full_qk_s_per_layer=self.full,
            suffix_qk_s_per_layer=self.suffix,area_budget_mm2=16,
            support_mm2=.5,source="native source A100 FP16 GQA roofline",
        )
        self.assertEqual(result.policy_selected_layers,1)
        self.assertEqual(len(result.per_token),128)
        self.assertTrue(isclose(result.candidate_minus_gpu_s_per_token*1e6,
                                -.2440211083864641,rel_tol=1e-9))
        self.assertTrue(isclose(result.maximum_extra_setup_s_to_retain_advantage*1e6,
                                .2440211083864641,rel_tol=1e-9))
        self.assertEqual(result.minimum_outputs_to_amortize_cold,16)
        self.assertGreater(result.per_token[0].model_hybrid_minus_gpu_s,0)
        self.assertLess(result.per_token[-1].model_hybrid_minus_gpu_s,0)
        self.assertIsNone(result.json_record().get("full_decoder_TPOT_s"))

    def test_finite_two_wave_buffer_required_and_no_unfunded_staging(self):
        self.assertIsNone(self.make_pipeline(stage_area=.005))
        reduced=self.make_pipeline(stage_area=.08)
        self.assertIsNotNone(reduced)
        self.assertLess(reduced.groups_per_wave,64)
        self.assertIsNotNone(self.make_pipeline(stage_area=.1))
        self.assertIsNone(self.make_pipeline(budget=13))
        self.assertIsNotNone(self.make_pipeline(budget=14))

    def test_nonregression_policy_gpu_fallback_at_short_output_or_slow_ports(self):
        for o in (1,8):
            w=Workload(1,2048,128,32,8,128,32,o)
            p=resident_pipeline(w,area_mm2=16,support_mm2=.5,
                                active_ports=256,rates=R,
                                operand_stage_support_mm2=.1)
            z=optimize_selective(w,p,full_qk_s_per_layer=self.full[:o],
                                 suffix_qk_s_per_layer=self.suffix[:o],
                                 area_budget_mm2=16,support_mm2=.5,source="fixture")
            self.assertEqual(z.policy_selected_layers,0)
            self.assertTrue(isclose(z.conditional_qk_speedup,1.0,rel_tol=1e-9))
        p=self.make_pipeline(ports=128)
        result=optimize_selective(
            self.w,p,full_qk_s_per_layer=self.full,
            suffix_qk_s_per_layer=self.suffix,
            area_budget_mm2=16,support_mm2=.5,source="fixture")
        self.assertEqual(result.policy_selected_layers,0)

    def test_exact_setup_margin_and_positive_threshold(self):
        p=self.make_pipeline()
        z=optimize_selective(
            self.w,p,full_qk_s_per_layer=self.full,
            suffix_qk_s_per_layer=self.suffix,
            area_budget_mm2=16,support_mm2=.5,source="fixture",
            extra_setup_s_per_layer_per_token=0.1e-6)
        self.assertEqual(z.policy_selected_layers,1)
        self.assertLess(z.candidate_minus_gpu_s_per_token,0)
        z=optimize_selective(
            self.w,p,full_qk_s_per_layer=self.full,
            suffix_qk_s_per_layer=self.suffix,
            area_budget_mm2=16,support_mm2=.5,source="fixture",
            extra_setup_s_per_layer_per_token=0.3e-6)
        self.assertEqual(z.policy_selected_layers,0)
        self.assertAlmostEqual(z.candidate_minus_gpu_s_per_token,0.0,places=12)

    def test_blocking_flowshop_on_slow_downstream_stage(self):
        # A deliberately slow scaling service may not allow infinite waves
        # to accumulate in an unbounded intermediate memory.
        waves=tuple(ResidentWave(
            1,1,1.0,10.0,1.0,1.0,16,16,64) for _ in range(10))
        t=_wave_dag(waves)
        self.assertGreaterEqual(t,100.0)
        self.assertLessEqual(t,sum(13.0 for _ in range(10)))

    def test_geometry_and_rate_fail_closed(self):
        with self.assertRaises(ValueError):
            self.make_pipeline(ports=0)
        p=self.make_pipeline()
        with self.assertRaisesRegex(ValueError,"per output token"):
            optimize_selective(self.w,p,full_qk_s_per_layer=self.full[:-1],
                               suffix_qk_s_per_layer=self.suffix,
                               area_budget_mm2=16,support_mm2=.5,source="fixture")
        with self.assertRaisesRegex(ValueError,"invalid GPU"):
            optimize_selective(self.w,p,full_qk_s_per_layer=self.suffix,
                               suffix_qk_s_per_layer=self.full,
                               area_budget_mm2=16,support_mm2=.5,source="fixture")


if __name__=="__main__":
    unittest.main()
