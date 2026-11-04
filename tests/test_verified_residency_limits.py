"""Independent integer macro/tile, cache pressure and weight-traffic checks."""
from __future__ import annotations

from itertools import product
from math import floor, isclose
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from cim_shared_prefix.area_optimizer import CIM_7NM,SRAM_7NM
from cim_shared_prefix.verified_residency_limits import (
    capacity_limit,minimum_touched_layers_for_persistent_tiles,
    dense_llama_weight_parameters,fp16_decoder_weight_bytes,
    A100_L2_NOMINAL_BYTES,A100_80GB_SXM_HBM_BPS,
    fused_gpu_floor_and_requirements,A100_L2_MODEL_BPS,
    cross_model_macro_bound,
)


class PhysicalResidencyProofTests(unittest.TestCase):
    def test_permanent_full_signed_k_and_v_storage(self):
        r=capacity_limit(2048,16.0)
        self.assertEqual(r.k_cim_macros_per_full_layer,4096)
        self.assertEqual(r.v_sram_macros_per_full_layer,128)
        self.assertAlmostEqual(r.k_cim_area_mm2_per_layer,13.1072,places=12)
        self.assertAlmostEqual(r.v_sram_area_mm2_per_layer,
                               128*SRAM_7NM.area_mm2,places=12)
        self.assertAlmostEqual(r.complete_kv_macro_area_mm2_per_layer,
                               14.256324383561644,places=10)
        self.assertEqual(r.max_complete_layers_without_peripherals,1)
        self.assertEqual(r.signed_k_and_v_macros_per_16token_tile,(32,1))
        self.assertEqual(r.max_persistent_16token_layer_tiles_without_peripherals,143)
        self.assertEqual(r.minimum_touched_layers_to_pack_maximum_tiles,2)
        self.assertAlmostEqual(r.persistent_prefix_token_layer_fraction_ceiling,
                               143/4096,places=15)
        self.assertLessEqual(143*r.physical_area_per_16token_tile_mm2,16)
        self.assertGreater(144*r.physical_area_per_16token_tile_mm2,16)
        self.assertEqual(r.max_removed_gpu_KV_bytes_under_persistent_tile_residency,
                         143*65536)
        self.assertEqual(r.gpu_fp16_prefix_KV_bytes_per_layer,8388608)
        self.assertEqual(r.gpu_fp16_prefix_KV_bytes_all_layers,268435456)
        self.assertFalse(r.gpu_prefix_KV_fits_entirely_in_nominal_A100_L2)
        self.assertEqual(r.score_return_bytes_per_layer,262144)
        self.assertEqual(r.prefix_context_LSE_return_bytes_per_layer,8320)
        self.assertAlmostEqual(r.context_vs_score_bytes_ratio,262144/8320,places=14)

    def test_exact_native_weight_operands_and_all_HBM_byte_envelope(self):
        p=dense_llama_weight_parameters()
        self.assertEqual(p["Q_projection_per_layer"],4096*4096)
        self.assertEqual(p["K_projection_per_layer"],4096*1024)
        self.assertEqual(p["V_projection_per_layer"],4096*1024)
        self.assertEqual(p["O_projection_per_layer"],4096*4096)
        self.assertEqual(p["MLP_up_per_layer"],4096*14336)
        self.assertEqual(p["LM_head_once_per_token"],4096*128256)
        self.assertEqual(fp16_decoder_weight_bytes(),15009316864)
        x=capacity_limit(2048,16)
        self.assertEqual(x.ideal_weight_plus_KV_HBM_bytes_per_token,15277752320)
        self.assertAlmostEqual(x.optimistic_fraction_of_full_decoder_HBM_bytes_removed,
                               0.0006134179821551132,places=15)
        self.assertAlmostEqual(x.traffic_only_ideal_speedup_ceiling,
                               1.0006137944947355,places=13)
        self.assertAlmostEqual(x.gpu_fp16_prefix_KV_bytes_per_layer/A100_80GB_SXM_HBM_BPS*1e6,
                               4.114079450711134,places=12)

    def test_capacity_context_length_cache_and_io_tradeoff(self):
        samples={64:(32,True),128:(17,True),256:(8,True),
                 320:(7,False),512:(4,False),
                 1024:(2,False),2048:(1,False),2304:(0,False),
                 4096:(0,False)}
        for p,(maxlayers,fits) in samples.items():
            r=capacity_limit(p,16)
            self.assertEqual(r.max_complete_layers_without_peripherals,maxlayers)
            self.assertEqual(r.gpu_prefix_KV_fits_entirely_in_nominal_A100_L2,fits)
            self.assertLessEqual(r.max_persistent_16token_layer_tiles_without_peripherals
                                 *r.physical_area_per_16token_tile_mm2,16+1e-10)
            self.assertLessEqual(r.persistent_prefix_token_layer_fraction_ceiling,1)
        self.assertLess(capacity_limit(64,16).context_vs_score_bytes_ratio,1)
        self.assertGreater(capacity_limit(128,16).context_vs_score_bytes_ratio,1)
        self.assertEqual(capacity_limit(4096,16).max_complete_layers_without_peripherals,0)
        self.assertGreater(capacity_limit(4096,16).max_persistent_16token_layer_tiles_without_peripherals,0)

    def test_full_model_KV_capacity_and_external_score_reduction_are_disjoint(self):
        # Any positive score-output reduction requires P>65, because
        # B*Hq*P*4 > B*Hq*(D*2+4) iff P>65.
        # In 16-token-aligned native mapping, the smallest such P is 80.
        # At P64 all 32 layers fit in a 16mm2 macro-only budget, but the
        # context+LSE output (8320B) exceeds QK-score output (8192B).
        for prefix in range(16,4097,16):
            r=capacity_limit(prefix,16)
            if r.max_complete_layers_without_peripherals==32:
                self.assertLessEqual(r.context_vs_score_bytes_ratio,1.0)
            if r.context_vs_score_bytes_ratio>1:
                self.assertLess(r.max_complete_layers_without_peripherals,32)
        self.assertEqual(capacity_limit(64,16).max_complete_layers_without_peripherals,32)
        self.assertEqual(capacity_limit(80,16).max_complete_layers_without_peripherals,28)

    def test_removing_V_does_not_enable_two_complete_signed_K_layers_at_P2048(self):
        r=capacity_limit(2048,16)
        self.assertGreater(2*r.k_cim_area_mm2_per_layer,16)
        self.assertGreater(2*r.complete_kv_macro_area_mm2_per_layer,16)
        self.assertGreater(32*r.complete_kv_macro_area_mm2_per_layer,16*28)
        self.assertEqual(r.max_complete_layers_without_peripherals,1)

    def test_exact_minimum_number_of_touched_layers_exhaustive_small(self):
        for layers in (1,2,3,4):
            for tiles_per_layer in (1,2,3):
                for target in range(layers*tiles_per_layer+1):
                    best=min(sum(n>0 for n in alloc)
                             for alloc in product(range(tiles_per_layer+1),repeat=layers)
                             if sum(alloc)==target)
                    self.assertEqual(
                        minimum_touched_layers_for_persistent_tiles(target,tiles_per_layer),
                        best)

    def test_ideal_fused_GPU_source_rate_lower_floors(self):
        h=fused_gpu_floor_and_requirements(2048,1,16,"HBM")
        l=fused_gpu_floor_and_requirements(2048,1,16,"L2")
        self.assertEqual(h.native_FP16_prefix_KV_bytes_per_layer,8388608)
        self.assertEqual(h.native_attention_QK_plus_PV_flops_per_layer,33554432)
        self.assertEqual(h.digital_PV_MACs_per_layer,8388608)
        self.assertEqual(h.resident_FP16_V_read_bytes_per_layer,4194304)
        self.assertAlmostEqual(h.ideal_fused_GPU_prefix_floor_s*1e6,
                               4.114079450711134,places=11)
        self.assertAlmostEqual(h.necessary_local_V_read_bandwidth_Bps,
                               1.0195e12,delta=1)
        self.assertAlmostEqual(h.necessary_digital_PV_MAC_per_s,
                               2.039e12,delta=1)
        self.assertAlmostEqual(l.necessary_local_V_read_bandwidth_Bps,
                               A100_L2_MODEL_BPS/2,delta=1)
        self.assertAlmostEqual(l.necessary_digital_PV_MAC_per_s,
                               A100_L2_MODEL_BPS,delta=1)
        self.assertLess(l.ideal_fused_GPU_prefix_floor_s,h.ideal_fused_GPU_prefix_floor_s)
        self.assertIsNotNone(l.necessary_PV_MAC_density_per_s_per_mm2)
        self.assertGreater(l.necessary_PV_MAC_density_per_s_per_mm2,
                           h.necessary_PV_MAC_density_per_s_per_mm2)
        with self.assertRaises(ValueError):
            fused_gpu_floor_and_requirements(2048,1,16,"unknown")
        with self.assertRaises(ValueError):
            fused_gpu_floor_and_requirements(2048,0,16,"HBM")

    def test_native_OPT_MHA_and_Mistral_GQA_SWA_source_geometry(self):
        llama=cross_model_macro_bound(
            "meta-llama/Llama-3.1-8B",2048,128,128,16)
        mistral=cross_model_macro_bound(
            "mistralai/Mistral-7B-v0.1",2048,128,128,16)
        self.assertEqual(llama.attention_kind,"GQA")
        self.assertEqual(mistral.attention_kind,"GQA")
        self.assertEqual(llama.physical_K_macros_per_layer,4096)
        self.assertEqual(mistral.physical_K_macros_per_layer,4096)
        self.assertEqual(mistral.maximum_complete_layers_no_peripherals,1)
        mistral_swa=cross_model_macro_bound(
            "mistralai/Mistral-7B-v0.1",4096,128,128,16)
        self.assertEqual(mistral_swa.first_decode_visible_prefix_tokens,3967)
        self.assertEqual(mistral_swa.maximum_complete_layers_no_peripherals,0)
        opt=cross_model_macro_bound("facebook/opt-2.7b",512,128,128,16)
        self.assertEqual(opt.attention_kind,"MHA")
        self.assertEqual(opt.kv_heads,32)
        self.assertEqual(opt.head_dim,80)
        self.assertEqual(opt.physical_K_macros_per_layer,4096)
        self.assertEqual(opt.FP16_V_SRAM_macros_per_layer,80)
        self.assertEqual(opt.maximum_complete_layers_no_peripherals,1)
        opt1024=cross_model_macro_bound("facebook/opt-2.7b",1024,128,128,16)
        self.assertEqual(opt1024.maximum_complete_layers_no_peripherals,0)
        with self.assertRaisesRegex(ValueError,"context limit"):
            cross_model_macro_bound("facebook/opt-2.7b",2048,128,128,16)
        with self.assertRaisesRegex(ValueError,"unknown"):
            cross_model_macro_bound("invalid",256,128,128,16)

    def test_no_setup_or_digital_area_implicitly_claimed(self):
        z=capacity_limit(2048,0.0)
        self.assertEqual(z.max_complete_layers_without_peripherals,0)
        self.assertEqual(z.max_persistent_16token_layer_tiles_without_peripherals,0)
        self.assertAlmostEqual(z.traffic_only_ideal_speedup_ceiling,1.0,places=15)
        self.assertAlmostEqual(z.optimistic_fraction_of_full_decoder_HBM_bytes_removed,
                               0.0,places=15)
        with self.assertRaises(ValueError):
            capacity_limit(2047,16)
        with self.assertRaises(ValueError):
            capacity_limit(2048,-1)
        with self.assertRaises(ValueError):
            minimum_touched_layers_for_persistent_tiles(1,0)

    def test_monotonicity_without_scheduling_assumptions(self):
        for p in (128,256,512,1024,2048,4096):
            cases=[capacity_limit(p,area) for area in (1,2,4,8,16,32,64)]
            self.assertEqual(
                sorted(x.max_complete_layers_without_peripherals for x in cases),
                [x.max_complete_layers_without_peripherals for x in cases])
            self.assertEqual(
                sorted(x.max_persistent_16token_layer_tiles_without_peripherals for x in cases),
                [x.max_persistent_16token_layer_tiles_without_peripherals for x in cases])


if __name__=="__main__":
    unittest.main()
