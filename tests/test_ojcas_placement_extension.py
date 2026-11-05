"""Independent physical-tiling and causal-placement tests for paper S2."""
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.ojcas_placement_extension import (
    candidates,dimensionless_cost,optimize,first_pin_threshold,run_grid
)
from cim_shared_prefix.ojcas_manuscript_baseline import manuscript_grid


class OriginalEnginePlacementTests(unittest.TestCase):
    def test_original_engine_preserved_and_all_layers_accounted(self):
        old=candidates(1,2048,8,decode_batch_steps=128)
        self.assertEqual(len(old),2)
        p0,p1=old
        self.assertEqual((p0.total_engines,p0.tiles_per_layer),(19,16))
        self.assertEqual((p0.pinned_complete_layers,p0.streamed_engines),(0,19))
        self.assertEqual((p1.pinned_complete_layers,p1.streamed_engines),(1,3))
        self.assertEqual(p0.compute_tile_waves_per_batch_step,32)
        self.assertEqual(p1.compute_tile_waves_per_batch_step,187)
        self.assertEqual(p0.K4_tile_writes_total,32*16*128)
        self.assertEqual(p1.K4_tile_writes_total,32*16+127*31*16)
        self.assertEqual(p1.K4_tile_writes_saved_vs_no_pin,16*127)
        self.assertEqual(p0.source_K4_programming_bytes_total,
                         p0.K4_tile_writes_total*65536)
        self.assertEqual(p1.source_K4_programming_bytes_total,
                         p1.K4_tile_writes_total*65536)

    def test_opportunity_threshold_is_not_misreported_speedup(self):
        t=first_pin_threshold(1,2048,8,decode_batch_steps=128)
        self.assertAlmostEqual(t,(187-32)*128/(16*127),places=12)
        self.assertEqual(optimize(1,2048,8,decode_batch_steps=128,
                                  lambda_write_per_tile=1.)[1].pinned_complete_layers,0)
        a,b,gain=optimize(1,2048,8,decode_batch_steps=128,
                           lambda_write_per_tile=10.)
        self.assertEqual(b.pinned_complete_layers,1)
        self.assertAlmostEqual(gain,(659456-658976)/659456,places=12)
        self.assertEqual(optimize(8,2048,8,decode_batch_steps=128,
                                  lambda_write_per_tile=10.)[1].pinned_complete_layers,0)

    def test_extra_area_changes_frontier_without_macros_replication(self):
        p=candidates(1,2048,16,decode_batch_steps=128)
        self.assertEqual(len(p),3)
        self.assertEqual([x.streamed_engines for x in p],[39,23,7])
        self.assertEqual([x.compute_tile_waves_per_batch_step for x in p],[32,32,92])
        self.assertEqual([x.K4_tile_writes_total for x in p],[65536,63504,61472])
        self.assertEqual(optimize(1,2048,16,decode_batch_steps=128,
                                  lambda_write_per_tile=1.)[1].pinned_complete_layers,1)
        self.assertEqual(optimize(1,2048,16,decode_batch_steps=128,
                                  lambda_write_per_tile=10.)[1].pinned_complete_layers,2)
        self.assertEqual(optimize(4,2048,16,decode_batch_steps=128,
                                  lambda_write_per_tile=10.)[1].pinned_complete_layers,1)

    def test_no_warm_decode_reuse_no_pinned_advantage(self):
        for area in (1,8,12,16):
            for b in (1,8,128):
                a,p,g=optimize(b,2048,area,decode_batch_steps=1,
                               lambda_write_per_tile=1000.)
                self.assertEqual(p.pinned_complete_layers,0)
                self.assertEqual(g,0)
        self.assertTrue(math.isinf(first_pin_threshold(
            1,2048,8,decode_batch_steps=1)))
        self.assertIsNone(first_pin_threshold(
            1,32768,16,decode_batch_steps=128))

    def test_source_endpoints_and_arithmetic_validation(self):
        rows,meta=run_grid()
        old,oldmeta=manuscript_grid()
        self.assertEqual(len(rows),728)
        self.assertEqual(
            {(x["B"],x["Lp"],x["Aadd_mm2"]) for x in rows},
            {(x["B"],x["Lp"],x["Aadd_mm2"]) for x in old})
        self.assertEqual(meta["original_S2_728_stage_results"],
                         "preserved unchanged in ojcas_manuscript_baseline.py")
        self.assertTrue(all(x["original_engine_macros"]==128 for x in rows))
        self.assertTrue(all(x["original_manuscript_S2_stage_speedup"] is None for x in rows))
        for x in rows:
            for val in (1.,10.,40.):
                self.assertGreaterEqual(x[f"lambda_{val:g}_dimensionless_cost_gain_fraction"],-1e-12)
                self.assertLessEqual(x[f"lambda_{val:g}_dimensionless_cost_gain_fraction"],1.+1e-12)
        self.assertEqual(oldmeta["paper_max_modeled_QK_stage_speedup"],1.97)

    def test_reject_unphysical_inputs(self):
        for kwargs in [
            dict(batch=3,prefix=2048,area_mm2=8,decode_batch_steps=128),
            dict(batch=1,prefix=2049,area_mm2=8,decode_batch_steps=128),
            dict(batch=1,prefix=2048,area_mm2=5,decode_batch_steps=128),
            dict(batch=1,prefix=2048,area_mm2=8,decode_batch_steps=0),
        ]:
            with self.assertRaises(ValueError):candidates(**kwargs)
        p=candidates(1,2048,8,decode_batch_steps=128)[0]
        for lam in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):dimensionless_cost(p,lam)

if __name__=="__main__":unittest.main()
