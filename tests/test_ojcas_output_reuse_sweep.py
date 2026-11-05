"""Check temporal reuse scenarios independently of paper 728 results."""
import sys,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_ojcas_output_reuse_sweep import run,OUTPUT_LENGTHS,PREFIX_REUSE_BATCHES

class OJCASOutputReuseTests(unittest.TestCase):
    def test_frozen_728_geometry_across_output_and_reuse(self):
        with TemporaryDirectory() as d:
            rows,m=run(Path(d))
            self.assertEqual(len(rows),5824)
            self.assertEqual(len(m["sections"]),8)
            self.assertTrue(all(s["O"] in (1,32,128,512) for s in m["sections"]))
            self.assertTrue(all(s["reuse_batches"] in (1,8) for s in m["sections"]))
            self.assertTrue(all(x["original_macro_engine_count"]==128 and
                                x["original_paper_728_speedup_unmodified"] for x in rows))
            self.assertTrue(all(x["measured_A100_INT4_TPOT"] is None for x in rows))
            byN={s["total_step_count"]:s["positive_coordinates"] for s in m["sections"]}
            self.assertEqual(byN[1],0)
            self.assertEqual(byN[128],117)
            self.assertGreaterEqual(byN[512],byN[128])
            self.assertGreaterEqual(byN[4096],byN[512])
            for x in rows:
                self.assertGreaterEqual(x["three_component_source_savings_us_per_step"],0.)
                self.assertLessEqual(x["selected_three_component_us_per_step"],
                                     x["baseline_three_component_us_per_step"]+1e-9)
if __name__=="__main__":
    unittest.main()
