import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cim_shared_prefix.gpu_saving_boundary import from_integrated_architecture_row

class PhysicalThresholdTests(unittest.TestCase):
    def test_exact_reference_derived_boundary(self):
        example={
            "model":"meta-llama/Llama-3.1-8B",
            "evidence":"published_complete_macro_plus_conditional_serial_rate_sensitivity",
            "serial_completed_prefix_scores_s":0.000193859616,
            "optimistic_resource_lower_bound_s":0.00006464862,
            "stage_completed_score_delivery_s":0.000008388608,
            "stage_source_key_supply_s":0.000065028096,
            "stage_macro_programming_s":0.000065028096,
        }
        t=from_integrated_architecture_row(example)
        self.assertAlmostEqual(t.required_gpu_attention_saving_us_per_layer,6.058113,places=8)
        self.assertAlmostEqual(t.optimistic_floor_us_per_layer,2.020269375,places=8)
        self.assertEqual(t.compare_gpu(8e-6,1e-6,gpu_source="fixture"),"CONDITIONALLY_SERIAL_CIM_FASTER")
        self.assertEqual(t.compare_gpu(6e-6,1e-6,gpu_source="fixture"),"SERIAL_CIM_NOT_FASTER")

    def test_reject_unqualified_and_impossible_latencies(self):
        row={"model":"meta-llama/Llama-3.1-8B","evidence":"synthetic",
             "serial_completed_prefix_scores_s":.0001,
             "optimistic_resource_lower_bound_s":.0002,
             "stage_completed_score_delivery_s":.00001,
             "stage_source_key_supply_s":.00001,
             "stage_macro_programming_s":.00001}
        with self.assertRaises(ValueError):
            from_integrated_architecture_row(row)
        row["evidence"]="published_complete_macro_plus_conditional_serial_rate_sensitivity"
        with self.assertRaisesRegex(ValueError,"ordering"):
            from_integrated_architecture_row(row)


if __name__=="__main__":
    unittest.main()
