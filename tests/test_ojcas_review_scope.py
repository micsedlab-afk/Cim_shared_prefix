"""Checks source capacities, reuse amortization and full-decoder bounds."""
import math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.ojcas_review_scope import (
    decoder_scope,s1_placement_amortization,reference_sensitivity
)

class OJCASReviewerScopeTests(unittest.TestCase):
    def test_original_S1_cold_placement_excluded_from_timed_decode(self):
        hd=s1_placement_amortization("S1-HD",2048,8,
            decode_steps=128,reuse_batches=1,interface_Bps_sensitivity=1e12)
        dp=s1_placement_amortization("S1-DP",2048,8,
            decode_steps=128,reuse_batches=1,interface_Bps_sensitivity=1e12)
        self.assertEqual((hd.pin_capacity_full_K4_tiles,dp.pin_capacity_full_K4_tiles),(445,91))
        self.assertEqual(hd.physical_placed_K4_bytes_once,445*65536)
        self.assertEqual(dp.physical_placed_K4_bytes_once,91*65536)
        self.assertEqual(hd.physical_placed_K4_bytes_once,29163520)
        self.assertEqual(dp.physical_placed_K4_bytes_once,5963776)
        self.assertEqual(hd.amortized_K4_bytes_per_batch_step,227840)
        self.assertEqual(dp.amortized_K4_bytes_per_batch_step,46592)
        self.assertAlmostEqual(hd.placement_us_per_batch_step_at_unqualified_interface_rate,.22784)
        self.assertAlmostEqual(dp.placement_us_per_batch_step_at_unqualified_interface_rate,.046592)

    def test_reuse_reduces_only_excluded_initial_S1_placement_cost(self):
        a=s1_placement_amortization("S1-HD",2048,8,
            decode_steps=128,reuse_batches=1,interface_Bps_sensitivity=1e12)
        b=s1_placement_amortization("S1-HD",2048,8,
            decode_steps=128,reuse_batches=8,interface_Bps_sensitivity=1e12)
        self.assertEqual(a.physical_placed_K4_bytes_once,b.physical_placed_K4_bytes_once)
        self.assertEqual(a.placement_us_per_batch_step_at_unqualified_interface_rate,
                         8*b.placement_us_per_batch_step_at_unqualified_interface_rate)
        self.assertEqual(b.amortized_K4_bytes_per_batch_step,28480)

    def test_full_decoder_scope_is_exact_Amdahl_with_stage_only_input(self):
        for f in (.01,.05,.1,.2,.4,.6):
            x=decoder_scope(f)
            self.assertAlmostEqual(x.no_added_overhead_upper_envelope,
                                   1/(1-f+f/1.97),places=12)
            self.assertAlmostEqual(x.max_additional_overhead_fraction_to_preserve_full_speedup,
                                   f*(1-1/1.97),places=12)
            y=decoder_scope(f,extra_overhead_fraction=x.max_additional_overhead_fraction_to_preserve_full_speedup)
            self.assertAlmostEqual(y.resulting_full_decoder_speedup_conditional,1.,places=12)
        self.assertAlmostEqual(decoder_scope(.1).resulting_full_decoder_speedup_conditional,
                               1/(.9+.1/1.97),places=12)
        self.assertEqual(decoder_scope(0.).resulting_full_decoder_speedup_conditional,1.)

    def test_concrete_scope_data_are_assumptions_not_empirical_TPOT(self):
        d=reference_sensitivity()
        self.assertEqual(len(d["S1_placement_sensitivities"]),6)
        self.assertEqual(len(d["decoder_scope_sensitivities"]),6)
        self.assertIsNone(d["qualified_model_level_tokens_per_second"])
        self.assertIsNone(d["qualified_gpu_end_to_end_TPOT"])
        self.assertIn("no duplicated S2 writes",d["important_scope"])
        self.assertIn("NO measured decoder timing",d["evidence_class"])

    def test_reject_invalid_source_and_unbounded_rates(self):
        for kind,p,a,o,r,bw in [
            ("S1-unknown",2048,8,128,1,1e12),
            ("S1-HD",2049,8,128,1,1e12),
            ("S1-HD",2048,9,128,1,1e12),
            ("S1-HD",2048,8,0,1,1e12),
            ("S1-DP",2048,8,128,0,1e12),
            ("S1-DP",2048,8,128,1,float("inf")),
            ("S1-DP",2048,8,128,1,0),
        ]:
            with self.assertRaises(ValueError):
                s1_placement_amortization(kind,p,a,decode_steps=o,reuse_batches=r,
                                          interface_Bps_sensitivity=bw)
        for f in (-.1,1.1,float("nan")):
            with self.assertRaises(ValueError):decoder_scope(f)
        with self.assertRaises(ValueError):decoder_scope(.1,extra_overhead_fraction=-.01)

if __name__=="__main__":unittest.main()
