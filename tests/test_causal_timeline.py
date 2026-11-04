"""Executable token timeline and complete-decoder composition tests."""
import unittest
from dataclasses import replace
from math import isclose
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cim_shared_prefix.contracts import Workload
from cim_shared_prefix.integrated_schedule import SerialRates, best_serial
from cim_shared_prefix.residency_area import ExternalArea, residency_plan
from cim_shared_prefix.causal_timeline import (
    GPUStagePoint, GPUSourceSeries, SRAMRates, simulate,
)

Z = ExternalArea(0, 0, 0, 0, 0, 0)
R = SerialRates(1e12, 1e12, 1e12, 1e12, 1e12, 1e12)


class TimelineTests(unittest.TestCase):
    def test_full_reference_128_causal_steps_exact_aggregate(self):
        w = Workload(1, 2048, 128, 32, 8, 128, 32, 128)
        support = ExternalArea(*(0.5 / 6 for _ in range(6)))
        result = best_serial(w, budget_mm2=16, support=support,
                             rates=R, active_macro_ports=256)
        self.assertIsNotNone(result)
        plan, trace = result
        out = simulate(w, plan, R, active_macro_ports=256)
        self.assertEqual(len(out.token_steps), 128)
        self.assertEqual(plan.instantiated_macros, 4842)
        self.assertEqual(plan.pinned_layers, 1)
        self.assertEqual(plan.scratch_tile_units, 373)
        self.assertEqual(out.cim_total_program_bytes, plan.programming_bytes_total)
        self.assertAlmostEqual(out.cim_mean_prefix_score_s,
                               trace.serial_completed_score_s, places=12)
        self.assertAlmostEqual(out.cim_mean_prefix_score_s,
                               0.000193859616, places=11)
        self.assertIsNone(out.s0_model_mean_tpot_s)
        self.assertIsNone(out.s2_model_mean_tpot_s)
        self.assertIsNone(out.s0_over_s2_conditional)
        self.assertGreater(out.cim_first_prefix_score_s,
                           out.cim_steady_mean_prefix_score_s)
        self.assertEqual(out.token_steps[0].cim_program_bytes,
                         plan.valid_key_layer_bytes * 32)
        self.assertEqual(out.token_steps[1].cim_program_bytes,
                         plan.valid_key_layer_bytes * 31)

    def test_small_integer_scheduler_agrees_with_serial_source(self):
        for layers in (1, 2, 3):
            for prefix in (16, 32, 48):
                w = Workload(2, prefix, 0, 4, 2, 64, layers, 4)
                for n in (4, 8, 14):
                    for ports in (1, 2, 3, n):
                        plan = residency_plan(w, kind="cim", budget_mm2=n * 0.0032+1e-8,
                                              support=Z)
                        if plan is None:
                            continue
                        result = simulate(w, plan, R, active_macro_ports=ports)
                        self.assertGreater(result.cim_mean_prefix_score_s, 0)
                        self.assertEqual(result.cim_total_program_bytes,
                                         plan.programming_bytes_total)

    def test_source_gpu_full_decoder_s0_s1_s2_endpoint(self):
        w = Workload(1, 32, 4, 4, 2, 64, 3, 3)
        budget = 0.1
        cp = residency_plan(w, kind="cim", budget_mm2=budget, support=Z)
        sp = residency_plan(w, kind="sram", budget_mm2=budget, support=Z)
        self.assertIsNotNone(cp)
        self.assertIsNotNone(sp)
        pt = GPUStagePoint(full_attention_s=0.000006,
                           score_remainder_s=0.000005,
                           other_decoder_per_layer_s=0.00012,
                           terminal_once_s=0.00007,
                           prefix_q4k4_s=0.000004)
        series = GPUSourceSeries(
            model_id="fixture", model_revision="fixture-rev",
            gpu_device_or_model="fixture-source-model", gpu_precision="fp16",
            evidence="sensitivity", source="explicit arithmetic-test fixture",
            batch=1, prefix_tokens=32, suffix_tokens=4,
            query_heads=4, kv_heads=2, head_dim=64, layers=3,
            steps=(pt, pt, pt),
        )
        rates = SRAMRates(1e12, 1e12, 1e12, 1e12, "fixture SRAM rates")
        result = simulate(w, cp, R, active_macro_ports=4,
                          model_id="fixture", model_revision="fixture-rev",
                          gpu=series, sram_plan=sp, sram_rates=rates)
        self.assertIsNotNone(result.s0_model_mean_tpot_s)
        self.assertIsNotNone(result.s1_model_mean_tpot_s)
        self.assertIsNotNone(result.s2_model_mean_tpot_s)
        first = result.token_steps[0]
        self.assertTrue(isclose(first.s0_full_decoder_s,
                                3 * (0.00012 + 0.000006) + 0.00007))
        self.assertTrue(isclose(first.s2_full_decoder_s,
                                3 * (0.00012 + 0.000005) + 0.00007 +
                                first.cim_prefix_complete_score_s))
        self.assertTrue(isclose(first.s1_full_decoder_s,
                                3 * (0.00012 + 0.000004 + 0.000005) +
                                0.00007 + first.sram_prefix_path_s))
        bad = replace(series, kv_heads=4)
        with self.assertRaisesRegex(ValueError, "shape"):
            simulate(w, cp, R, active_macro_ports=4, model_id="fixture",
                     model_revision="fixture-rev", gpu=bad)

    def test_fail_closed_for_incomplete_sram_and_precision(self):
        w = Workload(1, 16, 0, 1, 1, 64, 2, 1)
        cp = residency_plan(w, kind="cim", budget_mm2=0.1, support=Z)
        with self.assertRaises(ValueError):
            simulate(w, cp, R, active_macro_ports=0)
        with self.assertRaises(ValueError):
            simulate(w, cp, R, active_macro_ports=2,
                     sram_rates=SRAMRates(1e11, 1e11, 1e11, 1e11, "fixture"))
        gp = GPUSourceSeries(
            model_id="fixture", model_revision="v1", gpu_device_or_model="sim",
            gpu_precision="int4", evidence="native_simulator", source="fixture",
            batch=1, prefix_tokens=16, suffix_tokens=0,
            query_heads=1, kv_heads=1, head_dim=64, layers=2,
            steps=(GPUStagePoint(1e-5, 1e-5, 1e-5, 1e-5),),
        )
        with self.assertRaisesRegex(ValueError, "FP16/BF16"):
            simulate(w, cp, R, active_macro_ports=2, model_id="fixture",
                     model_revision="v1", gpu=gp)


if __name__ == "__main__":
    unittest.main()
