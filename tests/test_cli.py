import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cim_shared_prefix import Workload
from cim_shared_prefix.cli import geometry_report


class CliTests(unittest.TestCase):
    def test_manuscript_large_case(self):
        w = Workload(128, 32768, 128, 32, 8, 128, 32, 128)
        report = geometry_report(w)
        self.assertEqual(report["score_link_bytes_per_token"], 16 * 1024**3)
        self.assertEqual(report["signed_native_array_work_per_token"],
                         4 * report["logical_prefix_qk_macs_per_token"])
        self.assertTrue(report["timing"].startswith("unsupported"))
        self.assertEqual(geometry_report(w, partials_per_score=2)["score_link_bytes_per_token"],
                         32 * 1024**3)

if __name__ == "__main__":
    unittest.main()
