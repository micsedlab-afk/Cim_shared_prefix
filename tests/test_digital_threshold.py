import unittest
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cim_shared_prefix import Workload
from cim_shared_prefix.digital_threshold import (
    EvidenceValue, DigitalRequirement, digital_break_even,
)


def area(value):
    return EvidenceValue(value, "mm2", "synthetic sensitivity test", "sensitivity")


def time(value):
    return EvidenceValue(value, "seconds", "synthetic sensitivity test", "sensitivity")


class DigitalThresholdTests(unittest.TestCase):
    def setUp(self):
        self.w = Workload(2, 100, 32, 4, 2, 8, 2, 16, key_bits=4)
        self.kw = dict(
            added_area_budget_mm2=1.0,
            digital_sram_area=area(0.25),
            digital_other_area=area(0.25),
            cim_completed_score_stage_time=time(1.0),
            digital_nonmac_stage_time=time(0.5),
            query_bits=4,
            accumulator_bits=32,
        )

    def test_algebra_and_no_false_area_qualification(self):
        r = digital_break_even(self.w, **self.kw)
        self.assertEqual(r.status, "conditional_break_even")
        self.assertFalse(r.qualified_isoarea)
        self.assertEqual(r.core_area_available_mm2, 0.5)
        self.assertEqual(r.compute_time_available_s, 0.5)
        self.assertEqual(r.logical_mac_per_generated_token, 2 * 4 * 100 * 8 * 2)
        self.assertEqual(r.required_mac_per_second, r.logical_mac_per_generated_token / 0.5)
        self.assertEqual(r.required_mac_per_second_per_mm2, r.logical_mac_per_generated_token / 0.25)

    def test_no_room_for_digital_mac(self):
        r = digital_break_even(self.w, **{**self.kw, "digital_other_area": area(0.75)})
        self.assertEqual(r.status, "no_core_area")
        self.assertIsNone(r.required_mac_per_second_per_mm2)

    def test_insufficient_time(self):
        r = digital_break_even(
            self.w, **{**self.kw, "digital_nonmac_stage_time": time(1.1)})
        self.assertEqual(r.status, "no_compute_time")
        self.assertIsNone(r.required_mac_per_second_per_mm2)

    def test_rejects_missing_sources_and_bad_units(self):
        with self.assertRaises(ValueError):
            EvidenceValue(0.1, "mm2", "", "modeled")
        with self.assertRaises(ValueError):
            digital_break_even(self.w, **{**self.kw, "digital_sram_area": time(0.1)})
        with self.assertRaises(ValueError):
            digital_break_even(self.w, **{**self.kw, "query_bits": 8})

    def test_never_claims_isoarea_from_a_formula(self):
        with self.assertRaises(ValueError):
            DigitalRequirement("conditional_break_even", 1, 1.0, 1.0, 1.0, 1.0,
                               qualified_isoarea=True)

if __name__ == "__main__":
    unittest.main()
