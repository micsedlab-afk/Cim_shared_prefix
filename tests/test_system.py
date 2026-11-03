import unittest
from math import isclose
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cim_shared_prefix import Workload, SignedMapping, ScoreInterface
from cim_shared_prefix.system import SystemInputs, evaluate_system


class SystemTests(unittest.TestCase):
    def setUp(self):
        self.w = Workload(1, 128, 32, 8, 2, 64, 4, 8)
        self.params = SystemInputs(
            physical_array_evaluations_per_second=1e9,
            prefix_link_bytes_per_second=1e10,
            score_link_bytes_per_second=1e10,
            gpu_remaining_seconds_per_token=.005,
            initial_programming_seconds=.008,
            refill_programming_seconds_per_token=.0,
            merge_seconds_per_partial=1e-9,
            prefix_miss_fraction_per_token=0.,
            source="test synthetic parameters",
            gpu_source="test synthetic GPU",
        )

    def test_full_breakdown(self):
        r = evaluate_system(self.w, self.params)
        self.assertGreater(r.physical_qk_compute_s, 0)
        self.assertEqual(r.recurring_refill_s, 0)
        self.assertTrue(isclose(r.amortized_initial_programming_s, .001))
        self.assertTrue(isclose(r.total_s, sum(vars(r).values())))
        self.assertEqual(r.partial_merge_s, 0)

    def test_refill_and_partial_transfer(self):
        from dataclasses import replace
        full = evaluate_system(self.w, self.params)
        partial = evaluate_system(
            self.w,
            replace(self.params, prefix_miss_fraction_per_token=1.),
            scores=ScoreInterface(partials_per_score=2),
        )
        self.assertGreater(partial.completed_score_transfer_s, full.completed_score_transfer_s)
        self.assertGreater(partial.partial_merge_s, 0)
        self.assertGreater(partial.recurring_refill_s, 0)

    def test_signed_mapping_affects_compute(self):
        signed = evaluate_system(self.w, self.params)
        ideal = evaluate_system(self.w, self.params,
                                mapping=SignedMapping(1, 1))
        self.assertTrue(isclose(signed.physical_qk_compute_s,
                                4 * ideal.physical_qk_compute_s))
        self.assertTrue(isclose(signed.amortized_initial_placement_s,
                                2 * ideal.amortized_initial_placement_s))

    def test_rejects_unknown_or_invalid_sources(self):
        from dataclasses import replace
        with self.assertRaises(ValueError):
            replace(self.params, score_link_bytes_per_second=0)
        with self.assertRaises(ValueError):
            replace(self.params, source="")

if __name__ == "__main__":
    unittest.main()
