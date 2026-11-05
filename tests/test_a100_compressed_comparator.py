"""Scientific unit/endpoint guards on A100 compressed-vs-exact study."""
import math
import unittest
from cim_shared_prefix.a100_compressed_comparator import (
    point, unfused_source_operator_us, fused_floor_us
)
from cim_shared_prefix.source_roofline_corrected import A100_HBM_BPS,A100_L2_BPS

class A100CompressedTests(unittest.TestCase):
    def test_frozen_reference_sizes_and_placements(self):
        x=point()
        self.assertEqual(x.batch,1)
        self.assertEqual(x.exact_prefix_KV_fp16_bytes_all_layers,268435456)
        self.assertEqual(x.fully_compressed_FP16_KV_bytes_if_all_layers,8388608)
        self.assertEqual(x.fully_compressed_packed_K4_FP16_V_and_metadata_bytes,5308416)
        self.assertEqual(x.compressed_FP16_KV_bytes_selected_layers,4*262144)
        self.assertEqual(x.exact_FP16_KV_bytes_unselected_layers,28*8388608)
        self.assertEqual(x.total_FP16_KV_bytes_selective_hybrid,235929600)
        self.assertTrue(x.full_compressed_fp16_KV_fits_nominal_A100_L2)
        self.assertFalse(x.hybrid_FP16_KV_fits_nominal_A100_L2)
        self.assertAlmostEqual(x.published_K_V_metadata_macro_area_mm2_if_all_layers,
            14.274279452054795,places=10)

    def test_exact_A100_source_model_matches_frozen_corrected_HBM_and_L2(self):
        x=point()
        self.assertAlmostEqual(x.exact_GPU_HBM_unfused_operator_source_us_per_layer,
                               4.443527,places=5)
        self.assertAlmostEqual(x.exact_GPU_L2_unfused_operator_source_us_per_layer,
                               1.29471,places=5)
        self.assertAlmostEqual(x.exact_GPU_HBM_fused_optimistic_floor_us_per_layer,
                               4.114078,places=5)
        self.assertAlmostEqual(x.exact_GPU_L2_fused_optimistic_floor_us_per_layer,
                               1.161985,places=5)

    def test_compressed_A100_reference_no_H100_crosswalk(self):
        x=point()
        self.assertAlmostEqual(x.compressed_GPU_HBM_unfused_operator_source_us_per_layer,
                               0.1466444335458558,places=10)
        self.assertAlmostEqual(x.compressed_GPU_HBM_fused_optimistic_floor_us_per_layer,
                               0.12856498283472292,places=10)
        self.assertAlmostEqual(x.compressed_GPU_L2_fused_optimistic_floor_us_per_layer,
                               0.03631205673758865,places=10)
        self.assertFalse(x.measured_a100)
        self.assertIsNone(x.matched_signed_q4k4_gpu_timing_us)
        self.assertIsNone(x.full_decoder_tpot_us)
        self.assertIsNone(x.hardware_speedup)
        self.assertFalse(x.digital_area_qualified)

    def test_same_selected_layer_is_not_all_layer_quality(self):
        for n in (0,1,4,32):
            x=point(selected_layers=n)
            self.assertEqual(x.total_FP16_KV_bytes_selective_hybrid,
                n*262144+(32-n)*8388608)
        self.assertFalse(point(selected_layers=4).hybrid_FP16_KV_fits_nominal_A100_L2)
        self.assertTrue(point(selected_layers=32).hybrid_FP16_KV_fits_nominal_A100_L2)

    def test_floor_vs_source_and_no_precision_conflation(self):
        for b in (1,2,4,8,16,32):
            x=point(batch=b)
            self.assertLessEqual(x.compressed_GPU_HBM_fused_optimistic_floor_us_per_layer,
                x.compressed_GPU_HBM_unfused_operator_source_us_per_layer)
            self.assertLessEqual(x.compressed_GPU_L2_fused_optimistic_floor_us_per_layer,
                x.compressed_GPU_L2_unfused_operator_source_us_per_layer)
            self.assertGreater(x.exact_GPU_HBM_unfused_operator_source_us_per_layer,
                x.compressed_GPU_HBM_unfused_operator_source_us_per_layer)
            self.assertTrue(all(math.isfinite(y) and y>0 for y in [
                x.compressed_GPU_HBM_unfused_operator_source_us_per_layer,
                x.compressed_GPU_L2_unfused_operator_source_us_per_layer]))
        for kwargs in [
            {"batch":0},{"prefix":15},{"prefix":2049},{"reps":1},
            {"reps":66},{"reps":2064},{"selected_layers":33},
            {"selected_layers":-1}]:
            with self.assertRaises(ValueError):
                point(**kwargs)

if __name__=="__main__":
    unittest.main()
