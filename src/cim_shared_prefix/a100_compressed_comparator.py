"""A100 FP16 source-model comparators for physically compressed shared-prefix KV.

All results are operator source estimates/optimistic floors; none are real GPU
measurements or exact GPU-vs-Q4K4 matched-quality execution.

The A100_80GB_fp16/LLMCompass arithmetic and bandwidth constants are imported
from the original byte-corrected MAGMA source adapter. No H100 device values.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from math import ceil
from .source_roofline_corrected import (
    A100_HBM_BPS, A100_L2_BPS, A100_MATMUL_FLOPS_PER_S,
    A100_VECTOR_FLOPS_PER_S, FP16_WORD_BYTES, SOFTMAX_MODEL_FLOPS_PER_ELEMENT,
)
from .verified_residency_limits import LLAMA
from .compact_prefix_cim import macro_summary_area
from .contracts import Workload
from .area_optimizer import allocate_cim, cim_native_floor

A100_L2_NOMINAL_BYTES = 40_000_000
GPU_DENSE_FP16_FLOPS_PER_S = 312e12
QUERY_HEADS, KV_HEADS, D, L = 32, 8, 128, 32
assert (LLAMA.query_heads, LLAMA.kv_heads, LLAMA.head_dim, LLAMA.layers) == (
    QUERY_HEADS,KV_HEADS,D,L
)

@dataclass(frozen=True)
class A100CompressedPoint:
    batch: int
    original_prefix: int
    representatives: int
    selected_layers: int
    exact_prefix_KV_fp16_bytes_all_layers: int
    compressed_FP16_KV_bytes_selected_layers: int
    exact_FP16_KV_bytes_unselected_layers: int
    total_FP16_KV_bytes_selective_hybrid: int
    fully_compressed_FP16_KV_bytes_if_all_layers: int
    fully_compressed_packed_K4_FP16_V_and_metadata_bytes: int
    full_compressed_fp16_KV_fits_nominal_A100_L2: bool
    hybrid_FP16_KV_fits_nominal_A100_L2: bool
    published_K_V_metadata_macro_area_mm2_if_all_layers: float
    selected_layer_K_V_metadata_macro_only_area_mm2: float
    cim_macro_count_per_layer: int
    minimum_local_FP16_V_bytes_per_layer: int
    digital_PV_MACs_per_layer: int
    necessary_V_read_Bps_vs_HBM_source_us: float
    necessary_V_read_Bps_vs_L2_source_us: float
    necessary_digital_PV_MACps_vs_HBM_source_us: float
    necessary_digital_PV_MACps_vs_L2_source_us: float
    cim_signed_unsigned_array_work_per_layer: int
    cim_macro_only_optimistic_floor_us_per_layer: float
    cim_macro_only_below_A100_HBM_unfused_source_model: bool
    cim_macro_only_below_A100_L2_unfused_source_model: bool
    compressed_GPU_HBM_unfused_operator_source_us_per_layer: float
    compressed_GPU_L2_unfused_operator_source_us_per_layer: float
    compressed_GPU_HBM_fused_optimistic_floor_us_per_layer: float
    compressed_GPU_L2_fused_optimistic_floor_us_per_layer: float
    exact_GPU_HBM_unfused_operator_source_us_per_layer: float
    exact_GPU_L2_unfused_operator_source_us_per_layer: float
    exact_GPU_HBM_fused_optimistic_floor_us_per_layer: float
    exact_GPU_L2_fused_optimistic_floor_us_per_layer: float
    comparison_scope: str = (
        "A100 80GB FP16 HBM2e LLMCompass byte-corrected operator source; "
        "GPU compressed FP16 centroid execution has not been numerically "
        "equated to ideal CIM signed Q4K4 O4"
    )
    measured_a100: bool = False
    matched_signed_q4k4_gpu_timing_us: None = None
    full_decoder_tpot_us: None = None
    hardware_speedup: None = None
    digital_area_qualified: bool = False

    def json_record(self):
        return asdict(self)


def unfused_source_operator_us(batch: int, prefix: int, bw: float) -> float:
    """Native grouped QK / softmax / PV, per-layer serialized source model.

    Native GQA: 8 groups each with G=4*B queries against one K per KV head.
    Count correction, query quantization, FP16/INT4 conversions and kernel
    launch/graph costs are omitted, hence not a performance claim.
    """
    m = batch*(QUERY_HEADS//KV_HEADS)
    elems = KV_HEADS*(m*D + D*prefix + m*prefix)
    flops = KV_HEADS*2*m*D*prefix
    mm = max(flops/A100_MATMUL_FLOPS_PER_S,
             FP16_WORD_BYTES*elems/bw)
    sm_elems = QUERY_HEADS*batch*prefix
    sm = max(sm_elems*SOFTMAX_MODEL_FLOPS_PER_ELEMENT/A100_VECTOR_FLOPS_PER_S,
             3*FP16_WORD_BYTES*sm_elems/bw)
    return (2*mm+sm)*1e6


def fused_floor_us(batch: int, prefix: int, bw: float) -> float:
    """Bandwidth + dense peak necessary floor, no materialized QK scores."""
    kv_bytes = 2*KV_HEADS*prefix*D*FP16_WORD_BYTES
    flops = 4*batch*QUERY_HEADS*prefix*D
    return max(kv_bytes/bw, flops/GPU_DENSE_FP16_FLOPS_PER_S)*1e6


def point(batch: int = 1, prefix: int = 2048, reps: int = 64,
          selected_layers: int = 4) -> A100CompressedPoint:
    if type(batch) is not int or batch<1 or type(prefix) is not int or prefix<16 or prefix%16:
        raise ValueError("invalid batch or prefix")
    if type(reps) is not int or reps<16 or reps>prefix or reps%16:
        raise ValueError("aligned representative count required")
    if type(selected_layers) is not int or selected_layers<0 or selected_layers>L:
        raise ValueError("invalid compressed-layer count")
    orig_per_layer = 2*KV_HEADS*prefix*D*FP16_WORD_BYTES
    cmpr_per_layer = 2*KV_HEADS*reps*D*FP16_WORD_BYTES
    exact_all=L*orig_per_layer
    cmpr_selected=selected_layers*cmpr_per_layer
    exact_other=(L-selected_layers)*orig_per_layer
    hybrid=cmpr_selected+exact_other
    full=L*cmpr_per_layer
    # Q4 K is 4 bits; V is FP16; metadata holds uint16 count + FP16 scale.
    packed_per_layer=KV_HEADS*reps*D//2 + KV_HEADS*reps*D*2 + KV_HEADS*reps*4
    budget=macro_summary_area(L,KV_HEADS,D,reps)
    selected_budget=(macro_summary_area(selected_layers,KV_HEADS,D,reps)
                     if selected_layers else None)
    # This is the same native Q4xK4 K+/K- macro mapping already audited in
    # the main area/signed arithmetic workstream. A single layer is active.
    one_layer=Workload(batch,reps,128,QUERY_HEADS,KV_HEADS,D,1,1)
    allocation=allocate_cim(one_layer)
    physical_floor=cim_native_floor(one_layer,allocation)
    macro_floor_us=physical_floor.best_case_macro_compute_floor_s*1e6
    # Independent necessary stage-rate conditions at the same completed
    # context output endpoint; they do NOT subtract macro time or add
    # serial latencies, since pipeline overlap has not been established.
    vbytes=KV_HEADS*reps*D*FP16_WORD_BYTES
    pv_macs=batch*QUERY_HEADS*reps*D
    hbm_target=unfused_source_operator_us(batch,reps,A100_HBM_BPS)*1e-6
    l2_target=unfused_source_operator_us(batch,reps,A100_L2_BPS)*1e-6
    return A100CompressedPoint(
        batch=batch,original_prefix=prefix,representatives=reps,
        selected_layers=selected_layers,
        exact_prefix_KV_fp16_bytes_all_layers=exact_all,
        compressed_FP16_KV_bytes_selected_layers=cmpr_selected,
        exact_FP16_KV_bytes_unselected_layers=exact_other,
        total_FP16_KV_bytes_selective_hybrid=hybrid,
        fully_compressed_FP16_KV_bytes_if_all_layers=full,
        fully_compressed_packed_K4_FP16_V_and_metadata_bytes=L*packed_per_layer,
        full_compressed_fp16_KV_fits_nominal_A100_L2=full<=A100_L2_NOMINAL_BYTES,
        hybrid_FP16_KV_fits_nominal_A100_L2=hybrid<=A100_L2_NOMINAL_BYTES,
        published_K_V_metadata_macro_area_mm2_if_all_layers=budget[
            "signed_K_V_metadata_macro_area_mm2"],
        selected_layer_K_V_metadata_macro_only_area_mm2=(
            selected_budget["signed_K_V_metadata_macro_area_mm2"]
            if selected_budget else 0.0),
        cim_macro_count_per_layer=allocation.macro_count,
        minimum_local_FP16_V_bytes_per_layer=vbytes,
        digital_PV_MACs_per_layer=pv_macs,
        necessary_V_read_Bps_vs_HBM_source_us=vbytes/hbm_target,
        necessary_V_read_Bps_vs_L2_source_us=vbytes/l2_target,
        necessary_digital_PV_MACps_vs_HBM_source_us=pv_macs/hbm_target,
        necessary_digital_PV_MACps_vs_L2_source_us=pv_macs/l2_target,
        cim_signed_unsigned_array_work_per_layer=physical_floor.native_unsigned_mav_per_token,
        cim_macro_only_optimistic_floor_us_per_layer=macro_floor_us,
        cim_macro_only_below_A100_HBM_unfused_source_model=(
            macro_floor_us<unfused_source_operator_us(batch,reps,A100_HBM_BPS)),
        cim_macro_only_below_A100_L2_unfused_source_model=(
            macro_floor_us<unfused_source_operator_us(batch,reps,A100_L2_BPS)),
        compressed_GPU_HBM_unfused_operator_source_us_per_layer=unfused_source_operator_us(
            batch,reps,A100_HBM_BPS),
        compressed_GPU_L2_unfused_operator_source_us_per_layer=unfused_source_operator_us(
            batch,reps,A100_L2_BPS),
        compressed_GPU_HBM_fused_optimistic_floor_us_per_layer=fused_floor_us(
            batch,reps,A100_HBM_BPS),
        compressed_GPU_L2_fused_optimistic_floor_us_per_layer=fused_floor_us(
            batch,reps,A100_L2_BPS),
        exact_GPU_HBM_unfused_operator_source_us_per_layer=unfused_source_operator_us(
            batch,prefix,A100_HBM_BPS),
        exact_GPU_L2_unfused_operator_source_us_per_layer=unfused_source_operator_us(
            batch,prefix,A100_L2_BPS),
        exact_GPU_HBM_fused_optimistic_floor_us_per_layer=fused_floor_us(
            batch,prefix,A100_HBM_BPS),
        exact_GPU_L2_fused_optimistic_floor_us_per_layer=fused_floor_us(
            batch,prefix,A100_L2_BPS),
    )
