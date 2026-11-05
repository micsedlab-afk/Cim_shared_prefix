"""Regression contract: published OJCAS results must remain authoritative."""
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.ojcas_manuscript_baseline import (
    manuscript_grid,paper_capacity,paper_partial_energy,saved_stage_speedup_margin,
    AREAS_MM2,CIM_ENGINE_AREA_MM2,CIM_MACROS_PER_ENGINE
)

class OriginalSubmittedScienceTests(unittest.TestCase):
    def test_no_unapproved_signed_engine_resizing(self):
        self.assertEqual(CIM_MACROS_PER_ENGINE,128)
        self.assertAlmostEqual(CIM_ENGINE_AREA_MM2,0.4096,places=12)
        self.assertEqual([paper_capacity(a).s2_original_engine_count for a in AREAS_MM2],
                         [2,4,7,9,14,19,29,39])

    def test_original_728_table5_and_stage_limit_preserved(self):
        rows,s=manuscript_grid()
        self.assertEqual(len(rows),728)
        self.assertEqual(len({(r["B"],r["Lp"],r["Aadd_mm2"]) for r in rows}),728)
        self.assertEqual(s["paper_favorable_counts_by_area"],
                         {"1":0,"2":11,"3":14,"4":22,"6":52,"8":60,"12":60,"16":57})
        self.assertEqual(s["total_reported_favorable_stage_points"],276)
        self.assertAlmostEqual(s["paper_favorable_coverage_pct_by_area"]["8"],65.934065934,places=7)
        self.assertAlmostEqual(s["paper_favorable_coverage_pct_by_area"]["12"],65.934065934,places=7)
        self.assertEqual(s["paper_max_modeled_QK_stage_speedup"],1.97)
        self.assertTrue(all(r["individual_s2_stage_speedup"] is None for r in rows))
        self.assertTrue(all(r["original_s2_engine_area_mm2"]==0.4096 for r in rows))
        self.assertIn("elsewhere",s["signed_q4k4_boundary"])

    def test_original_storage_not_altered(self):
        a=paper_capacity(8)
        self.assertEqual(a.s1_hd_macro_count,891)
        self.assertEqual(a.s1_dp_macro_count,1464)
        self.assertAlmostEqual(a.s1_hd_capacity_bytes/1048576,27.84375)
        self.assertAlmostEqual(a.s1_dp_capacity_bytes/1048576,5.71875)
        rows,_=manuscript_grid()
        p=next(x for x in rows if x["B"]==1 and x["Lp"]==2048 and x["Aadd_mm2"]==8)
        self.assertAlmostEqual(p["s1_hd_resident_tile_fraction"],445/512)
        self.assertAlmostEqual(p["s1_dp_resident_tile_fraction"],91/512)

    def test_original_macro_energy_kept_unchanged(self):
        e=paper_partial_energy()
        self.assertAlmostEqual(e["s1_hd_write_and_read_pJ"],122060.8)
        self.assertAlmostEqual(e["s1_dp_write_and_read_pJ"],76021.76)
        self.assertAlmostEqual(e["s2_macro_compute_at_B1_pJ"],2987.52)
        self.assertAlmostEqual(e["crossing_B_s1_hd"],40.85689802913453,places=10)
        self.assertAlmostEqual(e["crossing_B_s1_dp"],25.44644387317909,places=10)
        self.assertNotIn("4x",e["scope"])

    def test_additional_overhead_has_its_own_gate_not_retroactive_rescaling(self):
        margin=saved_stage_speedup_margin(1.97)
        self.assertAlmostEqual(margin,1-1/1.97)
        self.assertTrue(0<margin<.5)
        for bad in (1.0,.8,-1):
            with self.assertRaises(ValueError):saved_stage_speedup_margin(bad)

if __name__=="__main__":unittest.main()
