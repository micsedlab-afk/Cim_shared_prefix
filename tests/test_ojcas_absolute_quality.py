import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import csv

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from export_ojcas_absolute_quality import export,REFERENCE

class OJCASAbsoluteQualityTests(unittest.TestCase):
    def test_real_archived_absolute_ppl_and_nonmonotonic_deltas(self):
        with TemporaryDirectory() as tmp:
            s=export(REFERENCE,Path(tmp))
            self.assertEqual(len(s["rows"]),3)
            self.assertTrue(s["nonmonotonic_R3_lower_than_R2_at_2048"])
            self.assertAlmostEqual(s["rows"][1]["R0_BF16_ppl"],5.4807592498153035)
            self.assertAlmostEqual(s["rows"][1]["R3_circuit_ppl"],5.665858560372165)
            self.assertAlmostEqual(s["rows"][1]["R3_vs_R2_pct"],-1.348,places=2)
            with (Path(tmp)/"ojcas_absolute_ppl_R0_R3.csv").open() as fd:
                rows=list(csv.DictReader(fd))
            self.assertEqual(len(rows),12)
            self.assertEqual({int(x["scored_tokens"]) for x in rows},{1024})
    def test_missing_or_unmatched_evidence_fails_closed(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                export(Path(__file__).resolve(),Path(tmp))
if __name__=="__main__":unittest.main()
