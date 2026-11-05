"""Cross-check published manuscript inputs and incremental S2 placement model."""
import math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.ojcas_published_placement_delta import (
    PAPER_A100_HBM_BYTES_PER_SECOND,PAPER_S2_PROGRAM_WAVE_SECONDS,
    PAPER_S2_MACRO_ACCESS_SECONDS,
    paper_stage_components,optimize_published_source,
    first_pin_critical_hbm_Bps,execute_728
)
from cim_shared_prefix.ojcas_manuscript_baseline import (
    manuscript_grid,PAPER_MAX_QK_STAGE_SPEEDUP,paper_partial_energy
)
from cim_shared_prefix.ojcas_placement_extension import candidates

class OriginalPaperPublishedPlacementTests(unittest.TestCase):
    def test_only_paper_table2_and_section3B_hardware_rates(self):
        self.assertEqual(PAPER_A100_HBM_BYTES_PER_SECOND,2039e9)
        self.assertEqual(PAPER_S2_PROGRAM_WAVE_SECONDS,64*200e-12)
        self.assertEqual(PAPER_S2_MACRO_ACCESS_SECONDS,5.5e-9)

    def test_programming_is_one_wave_not_one_tile_with_parallel_engines(self):
        c=candidates(1,2048,16,decode_batch_steps=128)
        base=paper_stage_components(c[0])
        pin1=paper_stage_components(c[1])
        self.assertEqual((base.pin_layers,pin1.pin_layers),(0,1))
        self.assertEqual(base.programming_wave_count,128*32)
        self.assertEqual(pin1.programming_wave_count,128*32-127)
        self.assertEqual(base.macro_compute_wave_units,pin1.macro_compute_wave_units)
        self.assertEqual(base.k4_transfer_bytes-pin1.k4_transfer_bytes,127*65536*16)
        self.assertAlmostEqual(
            (base.three_component_serial_seconds-pin1.three_component_serial_seconds)*1e6/128,
            (127*16*65536/2039e9 +127*64*200e-12)*1e6/128,places=12)
        self.assertEqual(
            (base.k4_transfer_bytes-pin1.k4_transfer_bytes)//65536,2032)

    def test_optimization_A8_A12_A16_B1_2048_model_matches_direct(self):
        b8,c8,_=optimize_published_source(1,2048,8)
        b12,c12,_=optimize_published_source(1,2048,12)
        b16,c16,_=optimize_published_source(1,2048,16)
        self.assertEqual((c8.pin_layers,c12.pin_layers,c16.pin_layers),(0,0,1))
        self.assertGreater(
            paper_stage_components(candidates(1,2048,8,decode_batch_steps=128)[1]).three_component_serial_seconds,
            b8.three_component_serial_seconds)
        self.assertGreater(
            paper_stage_components(candidates(1,2048,12,decode_batch_steps=128)[1]).three_component_serial_seconds,
            b12.three_component_serial_seconds)
        self.assertLess(c16.three_component_serial_seconds,b16.three_component_serial_seconds)
        self.assertEqual(c16.macro_compute_seconds,b16.macro_compute_seconds)
        self.assertEqual(c16.active_engines,39)

    def test_first_pin_bandwidth_threshold_explains_A8_and_A12(self):
        a8=first_pin_critical_hbm_Bps(1,2048,8)/1e9
        a12=first_pin_critical_hbm_Bps(1,2048,12)/1e9
        a16=first_pin_critical_hbm_Bps(1,2048,16)
        self.assertTrue(100<a8<300)
        self.assertTrue(800<a12<1100)
        self.assertTrue(math.isinf(a16))
        self.assertTrue(a8<a12<2039)
        self.assertIsNone(first_pin_critical_hbm_Bps(1,32768,16))
        self.assertEqual(first_pin_critical_hbm_Bps(1,2048,8,output_batch_steps=1),0.0)

    def test_original_grid_and_old_manuscript_results_not_modified(self):
        rows,m=execute_728(output_batch_steps=128)
        old,s=manuscript_grid()
        self.assertEqual(len(rows),728)
        self.assertEqual({(x["B"],x["Lp"],x["Aadd_mm2"]) for x in rows},
                         {(x["B"],x["Lp"],x["Aadd_mm2"]) for x in old})
        self.assertTrue(all(x["original_engine_macros"]==128 and
                            x["original_engine_mm2"]==0.4096 and
                            x["original_s2_model_historical_stage_speedup"] is None and
                            x["full_decoder_tpot_us"] is None for x in rows))
        self.assertEqual(m["original_max_modeled_s2_QK_stage_speedup"],1.97)
        self.assertEqual(s["paper_favorable_counts_by_area"]["8"],60)
        self.assertEqual(s["paper_favorable_counts_by_area"]["12"],60)
        self.assertAlmostEqual(paper_partial_energy()["crossing_B_s1_hd"],
                               40.85689802913453)
        for r in rows:
            self.assertGreaterEqual(r["three_component_serial_reduction_fraction"],-1e-12)
            self.assertLessEqual(r["policy_three_component_serial_us_per_step"],
                                 r["baseline_three_component_serial_us_per_step"]+1e-11)
            self.assertLessEqual(r["baseline_independent_resource_floor_us_per_step"],
                                 r["baseline_three_component_serial_us_per_step"]+1e-11)
        self.assertEqual(sum(x["strict_no_wave_penalty_pareto"] for x in rows),117)

    def test_no_warm_reuse_cannot_reduce_source_compulsory_steps(self):
        for a in (8,12,16):
            original,best,_=optimize_published_source(1,2048,a,output_batch_steps=1)
            self.assertEqual(best.pin_layers,0)
            self.assertEqual(best.three_component_serial_seconds,
                             original.three_component_serial_seconds)

    def test_peak_HBM_source_sensitivity_is_not_relabelled_gpu_measurement(self):
        _,_,x=optimize_published_source(1,2048,16,hbm_bytes_per_second=2039e9)
        c=x[0]
        self.assertAlmostEqual(c.k4_supply_seconds*1e6/128,
                                (32*2048*8*128*4/8)/(2039e9)*1e6,places=12)
        with self.assertRaises(ValueError):
            paper_stage_components(candidates(1,2048,16,decode_batch_steps=128)[0],
                                   hbm_bytes_per_second=float('nan'))
        with self.assertRaises(ValueError):
            paper_stage_components(candidates(1,2048,16,decode_batch_steps=128)[0],
                                   hbm_bytes_per_second=0)

if __name__=="__main__":unittest.main()
