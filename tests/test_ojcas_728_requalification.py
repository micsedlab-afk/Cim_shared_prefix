"""Independent cross-checks for original OJCAS-2026-0120 axes and silicon mapping."""
import unittest
import sys
from pathlib import Path
from math import isclose

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.ojcas_paper_728_audit import (
    AREAS_MM2,BATCHES,PREFIXES,full_grid,coordinate,
    area_capacity,energy_crossovers,
)

class OJCAS728AuditTests(unittest.TestCase):
    def test_original_grid_and_absent_unqualified_speedups(self):
        rows,s=full_grid()
        self.assertEqual(len(rows),13*7*8)
        self.assertEqual(len({(x["B"],x["Lp"],x["Aadd_mm2"]) for x in rows}),728)
        self.assertEqual(list(PREFIXES),[512,1024,2048,4096,8192,16384,32768])
        self.assertEqual(list(AREAS_MM2),[1,2,3,4,6,8,12,16])
        self.assertEqual(s["original_manuscript_evidence"]["reported_S2_win_counts_by_area_PAPER_ONLY"],
            {"1":0,"2":11,"3":14,"4":22,"6":52,"8":60,"12":60,"16":57})
        self.assertEqual(s["requalified_original_S2_speedup_coordinates"],0)
        self.assertEqual(s["qualified_INT4_A100_GPU_stage_times"],0)
        self.assertTrue(all(x["corrected_qualified_S2_speedup"] is None and
                            x["primary_A100_INT4_source_operator_time_s"] is None
                            for x in rows))

    def test_optional_signed_bank_replication_per_original_area_budget(self):
        old=[area_capacity(a).paper_unsigned_tile_engine_count for a in AREAS_MM2]
        new=[area_capacity(a).corrected_signed_tile_engine_count_macro_only for a in AREAS_MM2]
        self.assertEqual(old,[2,4,7,9,14,19,29,39])
        self.assertEqual(new,[1,2,3,4,7,9,14,19])
        self.assertTrue(all(n<=o for n,o in zip(new,old)))
        a=area_capacity(8)
        self.assertAlmostEqual(a.paper_macro_area_per_engine_mm2,.4096)
        self.assertAlmostEqual(a.signed_macro_area_per_engine_mm2,.8192)
        self.assertEqual(a.cim_physical_macros_max_no_support,2500)
        self.assertEqual(a.cim_physical_signed_K_storage_bytes_max_no_support,1280000)
        self.assertEqual(area_capacity(1).corrected_signed_tile_engine_count_with_0p5mm2_support,0)

    def test_S1_8mm2_matches_manuscript_and_tile_fraction(self):
        a=area_capacity(8)
        self.assertEqual(a.s1_HD_macro_count,891)
        self.assertEqual(a.s1_HD_capacity_bytes,891*32768)
        self.assertEqual(a.s1_DP_macro_count,1464)
        self.assertAlmostEqual(a.s1_HD_capacity_bytes/2**20,27.84375)
        self.assertAlmostEqual(a.s1_DP_capacity_bytes/2**20,5.71875)
        x=coordinate(1,2048,8)
        self.assertAlmostEqual(x["S1_HD_resident_fraction_128token_tiles"],445/512)
        self.assertAlmostEqual(x["S1_DP_resident_fraction_128token_tiles"],91/512)

    def test_physical_programming_is_distinct_from_upstream_HBM_K4_supply(self):
        x=coordinate(1,2048,8)
        self.assertEqual(x["source_unique_K4_bytes_all_layers"],32*2**20)
        self.assertEqual(x["physical_Kplus_Kminus_bytes_all_layers"],64*2**20)
        self.assertEqual(x["macro_key_residency_bytes_optimistic_no_support"],1280000)
        self.assertEqual(x["minimum_warm_program_BYTES_per_token"],64*2**20-1280000)
        self.assertEqual(x["minimum_cold_program_BYTES_one_prefix"],64*2**20)
        self.assertLess(x["minimum_warm_program_BYTES_per_token"],
                        x["minimum_cold_program_BYTES_one_prefix"])
        self.assertEqual(x["completed_FP32_score_BYTES_per_token"],8*2**20)
        self.assertEqual(x["paper_two_d64_half_partial_BYTES_per_token"],16*2**20)
        self.assertEqual(x["eight_raw_signed_d64_partial_BYTES_per_token"],64*2**20)

    def test_engine_wave_and_resource_floor_separately_labeled(self):
        x=coordinate(1,2048,8)
        self.assertEqual(x["paper_128token_tile_waves"],1)
        self.assertEqual(x["corrected_signed_128token_tile_waves_macro_only"],2)
        self.assertEqual(x["physical_signed_native_unsigned_calls_per_layer"],32768)
        self.assertEqual(x["physical_signed_native_unsigned_calls_per_token_all_layers"],32768*32)
        self.assertAlmostEqual(x["corrected_ideal_128token_wave_compute_us_all_layers"],2.816)
        self.assertAlmostEqual(x["corrected_optimistic_universal_macro_RESOURCE_floor_us_all_layers"],2.464)
        self.assertLessEqual(
            x["corrected_optimistic_universal_macro_RESOURCE_floor_us_all_layers"],
            x["corrected_ideal_128token_wave_compute_us_all_layers"])

    def test_macro_energy_eq10_correction_not_system_energy(self):
        e=energy_crossovers()
        self.assertEqual(e["paper_native_unsigned_calls_per_128token_tile_B1"],512)
        self.assertEqual(e["signed_native_unsigned_calls_per_128token_tile_B1"],2048)
        self.assertEqual(e["access_count_correction_factor"],4)
        self.assertAlmostEqual(e["original_HD_nominal_energy_crossing_B"],40.85689802913453)
        self.assertAlmostEqual(e["original_DP_nominal_energy_crossing_B"],25.44644387317909)
        self.assertAlmostEqual(e["corrected_HD_nominal_energy_crossing_B"],10.214224507283632)
        self.assertAlmostEqual(e["corrected_DP_nominal_energy_crossing_B"],6.361610968294772)

    def test_score_output_increases_with_batch_not_key_storage(self):
        a=coordinate(1,2048,8);b=coordinate(128,2048,8)
        self.assertEqual(a["physical_Kplus_Kminus_bytes_all_layers"],
                         b["physical_Kplus_Kminus_bytes_all_layers"])
        self.assertEqual(a["minimum_warm_program_BYTES_per_token"],
                         b["minimum_warm_program_BYTES_per_token"])
        self.assertEqual(b["paper_two_d64_half_partial_BYTES_per_token"],
                         128*a["paper_two_d64_half_partial_BYTES_per_token"])
        with self.assertRaises(ValueError):
            coordinate(3,2048,8)
        with self.assertRaises(ValueError):
            area_capacity(5)

if __name__=="__main__":unittest.main()
