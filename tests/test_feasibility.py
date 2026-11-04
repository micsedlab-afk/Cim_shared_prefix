import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cim_shared_prefix import Workload
from cim_shared_prefix.feasibility import LinkCeilings, PhysicalBudget, lower_bound


class NecessaryBoundTests(unittest.TestCase):
    def setUp(self):
        self.w = Workload(1, 2048, 128, 32, 8, 128, 32, 128)
        self.links = LinkCeilings(1e12, 1e12, 1e12, "illustrative 1 TB/s sensitivities")

    def test_published_full_residency_geometry(self):
        out = lower_bound(self.w, PhysicalBudget(16, 0, "16 mm2 macro-only"), self.links)
        self.assertEqual(out.layer_macros_full, 4096)
        self.assertEqual(out.model_macros_full, 131072)
        self.assertEqual(out.mandatory_native_calls_per_token, 1048576)
        self.assertEqual(out.completed_score_bytes_per_token, 8 * 1024**2)
        self.assertEqual(out.query_bytes_per_token_min, 65536)
        self.assertEqual(out.programming_bytes_total_min,
                         64 * 1024**2 + 127 * (64 * 1024**2 -
                         out.macro_count_max * 512))
        self.assertEqual(out.elapsed_floor_s, max(
            out.macro_compute_floor_s, out.programming_floor_mean_s,
            out.query_floor_s, out.score_link_floor_s))

    def test_more_physical_macros_cannot_worsen_resource_bound(self):
        tiny = lower_bound(self.w, PhysicalBudget(1, 0), self.links)
        large = lower_bound(self.w, PhysicalBudget(16, 0), self.links)
        self.assertGreater(tiny.elapsed_floor_s, large.elapsed_floor_s)
        self.assertGreater(tiny.programming_floor_mean_s,
                           large.programming_floor_mean_s)

    def test_physical_port_cap_restricts_parallelism(self):
        ideal = lower_bound(self.w, PhysicalBudget(16, 0), self.links)
        ports = lower_bound(self.w, PhysicalBudget(16, 0), self.links,
                            active_macro_port_cap=256)
        self.assertGreater(ports.macro_compute_floor_s, ideal.macro_compute_floor_s)
        self.assertEqual(ports.programming_floor_mean_s, ideal.programming_floor_mean_s)

    def test_conditional_not_physical_claim(self):
        r = lower_bound(self.w, PhysicalBudget(8, 0), self.links)
        self.assertEqual(r.compare_gpu(r.elapsed_floor_s / 2, "test GPU fixture"),
                         "conditionally_excluded_for_stated_rates")
        self.assertEqual(r.compare_gpu(r.elapsed_floor_s * 2, "test GPU fixture"),
                         "not_excluded_by_lower_bound")
        self.assertIn("conditional", r.evidence)

    def test_area_and_source_validation(self):
        self.assertIsNone(lower_bound(self.w, PhysicalBudget(.001, 0), self.links))
        with self.assertRaises(ValueError):
            LinkCeilings(0, 1e12, 1e12, "invalid")
        with self.assertRaises(ValueError):
            PhysicalBudget(1, -1)
        with self.assertRaises(ValueError):
            lower_bound(self.w, PhysicalBudget(8, 0), self.links,
                        active_macro_port_cap=0)


if __name__ == "__main__":
    unittest.main()
