import unittest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_ojcas_resource_screen import run

class ResourceScreenTests(unittest.TestCase):
    def test_reference_rate_cap_and_overlap_safe_bound(self):
        with TemporaryDirectory() as tmp:
            rows,s=run(Path(tmp))
            self.assertEqual(len(rows),728)
            r={x["B"]:x for x in s["reference_P2048_A8_rows"]}
            x=r[1]
            self.assertAlmostEqual(
                x["signed_programming_mean_us_per_token_O128_at_rate_sensitivity"],65.838864)
            self.assertAlmostEqual(
                x["two_d64_partial_link_us_per_token_at_rate_sensitivity"],16.777216)
            self.assertAlmostEqual(
                x["optimistic_native_macro_resource_us_per_token"],2.464)
            self.assertEqual(x["largest_necessary_resource_at_rate_sensitivity"],
                             "physical_signed_macro_programming")
            self.assertAlmostEqual(r[4]["two_d64_partial_link_us_per_token_at_rate_sensitivity"],67.108864)
            self.assertEqual(r[4]["largest_necessary_resource_at_rate_sensitivity"],
                             "paper_GPU_d64_half_partial_link")
            self.assertEqual(r[128]["largest_necessary_resource_at_rate_sensitivity"],
                             "paper_GPU_d64_half_partial_link")
            self.assertAlmostEqual(
                r[128]["two_d64_partial_link_us_per_token_at_rate_sensitivity"],2147.483648)
            self.assertFalse(s["conditional_full_TPOT_qualified"])
            self.assertFalse(s["hardware_speedup_qualified"])
            self.assertTrue(all(x["qualified_a100_INT4_speedup"] is None for x in rows))
    def test_reject_unsupported_rates(self):
        with TemporaryDirectory() as tmp:
            for z in (0,-1):
                with self.assertRaises(ValueError):
                    run(Path(tmp),z)

if __name__=="__main__":unittest.main()
