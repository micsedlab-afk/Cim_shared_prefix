import unittest
from math import isclose
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cim_shared_prefix import Workload, SignedMapping, ScoreInterface, AreaItem, AreaLedger


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.w = Workload(batch=128, prefix_tokens=32768, suffix_tokens=128,
                          query_heads=32, kv_heads=8, head_dim=128, layers=32,
                          output_tokens=128)

    def test_prefix_shared_across_batch(self):
        self.assertEqual(self.w.gqa_ratio, 4)
        self.assertEqual(self.w.prefix_key_bytes_per_layer, 32768 * 8 * 128 // 2)
        self.assertEqual(self.w.prefix_scores_per_layer, 128 * 32 * 32768)
        w2 = Workload(**{**self.w.__dict__, "batch": 1})
        self.assertEqual(w2.prefix_key_bytes_per_layer, self.w.prefix_key_bytes_per_layer)

    def test_full_model_score_endpoint(self):
        interface = ScoreInterface(bits_per_partial=32, partials_per_score=1)
        self.assertEqual(interface.bytes_per_generated_token(self.w), 16 * 1024**3)
        self.assertEqual(ScoreInterface(partials_per_score=2).bytes_per_generated_token(self.w), 32 * 1024**3)
        self.assertTrue(isclose(interface.transfer_seconds(self.w, 1e12), 16 * 1024**3 / 1e12))

    def test_signed_split(self):
        mapping = SignedMapping()
        self.assertEqual(mapping.physical_key_bytes_all_layers(self.w),
                         2 * self.w.prefix_key_bytes_all_layers)
        self.assertEqual(mapping.physical_array_work_per_generated_token(self.w),
                         4 * self.w.total_prefix_qk_mac_per_generated_token)

    def test_area_accounting(self):
        ledger = AreaLedger((AreaItem("macro", 2, .4, "published paper"),
                             AreaItem("controller", 1, .05, "parametric")))
        self.assertTrue(ledger.within(.85))
        self.assertFalse(ledger.within(.84))
        with self.assertRaises(ValueError):
            AreaLedger((AreaItem("ADC", 1, .01, "macro", True),))

    def test_invalid_geometries(self):
        with self.assertRaises(ValueError):
            Workload(1, 128, 0, 32, 7, 128, 32, 64)
        with self.assertRaises(ValueError):
            ScoreInterface(bits_per_partial=7)

if __name__ == "__main__":
    unittest.main()
