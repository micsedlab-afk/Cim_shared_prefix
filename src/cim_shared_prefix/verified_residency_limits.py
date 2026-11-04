"""Source-derived, rate-free KV residency and ideal-GPU memory/work limits.

This module deliberately has NO assumed CIM ports, off-macro digital area,
communication rates, DAC/ADC latency or accelerator speedup. It answers:
how much complete signed K4 + FP16 V can stay resident under published
macro areas; what fraction of exact shared-prefix data/work could be
offloaded with *zero* execution overhead; and how much of a native FP16
dense decoder's ideal HBM byte volume is that fraction.

Only the source model's Llama-3.1-8B native architecture is supported:
32Q / 8KV / D128 / hidden4096 / intermediate14336 / layers32 / vocab128256.
A100 80GB SXM bandwidth 2039 GB/s and L2 capacity 40 MB are published
GPU specifications. All conditional comparisons state their data placement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, floor, isfinite

from .area_optimizer import CIM_7NM, SRAM_7NM
from .profiles import PROFILES

LLAMA=next(p for p in PROFILES if p.model_id=="meta-llama/Llama-3.1-8B")
if (LLAMA.hidden_size,LLAMA.query_heads,LLAMA.kv_heads,
    LLAMA.layers,LLAMA.head_dim)!=(4096,32,8,32,128):
    raise AssertionError("native pinned Llama geometry changed")

# Llama-3.1-8B pinned native config, source available through LLAMA.source_url.
INTERMEDIATE_SIZE=14336
VOCAB_SIZE=128256
OUTPUT_EMBEDDINGS_TIED=False
FP16_BYTES=2
LLAMA_CONFIG_SOURCE=LLAMA.source_url
A100_80GB_SXM_HBM_BPS=2039e9
A100_L2_NOMINAL_BYTES=40_000_000
# L2 interface and FP16 dense peak are from vendored LLMCompass A100 source;
# throughput is a model peak, not a CUDA measurement.
A100_L2_MODEL_BPS=5120*1.41e9
A100_FP16_DENSE_FLOPS_PER_S=312e12
A100_PUBLIC_DOC="https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/"
A100_SOURCE="https://www.nvidia.com/en-us/data-center/a100/"


@dataclass(frozen=True)
class ResidentLimit:
    prefix_tokens: int
    area_budget_mm2: float
    k_cim_macros_per_full_layer: int
    v_sram_macros_per_full_layer: int
    k_cim_area_mm2_per_layer: float
    v_sram_area_mm2_per_layer: float
    complete_kv_macro_area_mm2_per_layer: float
    max_complete_layers_without_peripherals: int
    signed_k_and_v_macros_per_16token_tile: tuple[int, int]
    physical_area_per_16token_tile_mm2: float
    max_persistent_16token_layer_tiles_without_peripherals: int
    persistent_prefix_token_layer_fraction_ceiling: float
    minimum_touched_layers_to_pack_maximum_tiles: int
    gpu_fp16_prefix_KV_bytes_per_layer: int
    gpu_fp16_prefix_KV_bytes_all_layers: int
    gpu_prefix_KV_fits_entirely_in_nominal_A100_L2: bool
    score_return_bytes_per_layer: int
    prefix_context_LSE_return_bytes_per_layer: int
    context_vs_score_bytes_ratio: float
    max_removed_gpu_KV_bytes_under_persistent_tile_residency: int
    source_fp16_transformer_and_lm_head_weight_bytes: int
    ideal_weight_plus_KV_HBM_bytes_per_token: int
    optimistic_fraction_of_full_decoder_HBM_bytes_removed: float
    traffic_only_ideal_speedup_ceiling: float
    uniform_layer_prefix_work_fraction_if_full_layers_offloaded: float
    uniform_layer_prefix_attention_speedup_ceiling: float | None
    evidence: str = (
        "source_7nm_K_CIM_macro_plus_7nm_V_SRAM_macro_capacity_ceiling;"
        "native_Llama_geometry;A100_spec;zero_periphery_zero_processing"
    )

    def json_record(self)->dict:
        return asdict(self)


def dense_llama_weight_parameters()->dict[str,int]:
    """Decoder matrices and untied LM head, excluding lookup-only embeddings.

    Parameter sizes are from native config. Per-token matmul reads every
    projection weight once in the explicit all-HBM FP16 streaming scenario.
    No weight reuse or prefetch savings are assumed for this *byte model*.
    """
    h=LLAMA.hidden_size
    kv=LLAMA.kv_heads*LLAMA.head_dim
    return {
        "Q_projection_per_layer":h*h,
        "K_projection_per_layer":h*kv,
        "V_projection_per_layer":h*kv,
        "O_projection_per_layer":h*h,
        "MLP_gate_per_layer":h*INTERMEDIATE_SIZE,
        "MLP_up_per_layer":h*INTERMEDIATE_SIZE,
        "MLP_down_per_layer":INTERMEDIATE_SIZE*h,
        "LM_head_once_per_token":h*VOCAB_SIZE,
    }


def fp16_decoder_weight_bytes()->int:
    p=dense_llama_weight_parameters()
    return FP16_BYTES*(LLAMA.layers*sum(v for k,v in p.items() if k!="LM_head_once_per_token")
                       +p["LM_head_once_per_token"])


def _finite_nonnegative(v:float,name:str)->None:
    if isinstance(v,bool) or not isinstance(v,(float,int)) or not isfinite(v) or v<0:
        raise ValueError(name+" must be finite nonnegative")


def minimum_touched_layers_for_persistent_tiles(n_tiles:int,per_layer:int)->int:
    """Exact minimum number of partitioned layers for n identical tiles.

    For any strictly positive fixed touched-layer fusion/setup overhead,
    packing complete layers first minimizes that overhead without affecting
    how many persistent token-layer tiles are represented.
    """
    if any(type(x) is not int or x<0 for x in (n_tiles,per_layer)) or per_layer==0:
        raise ValueError("valid positive integer tile geometry required")
    return ceil(n_tiles/per_layer)


def capacity_limit(prefix_tokens:int,area_budget_mm2:float)->ResidentLimit:
    if type(prefix_tokens) is not int or prefix_tokens<=0 or prefix_tokens%16:
        raise ValueError("prefix length must be positive, 16-token aligned")
    if prefix_tokens>LLAMA.context_limit:
        raise ValueError("exceeds native model context")
    _finite_nonnegative(area_budget_mm2,"area")
    hkv=LLAMA.kv_heads
    d=LLAMA.head_dim
    L=LLAMA.layers
    # Native published macro contains 64 rows and 16 4-bit weights per row.
    # Physically separate K+ and K- (four unsigned evals per signed Q4K4).
    k_tile_macros=hkv*ceil(d/64)*2
    v_tile_bytes=hkv*16*d*FP16_BYTES
    v_tile_macros=ceil(v_tile_bytes/(SRAM_7NM.capacity_bits//8))
    ntiles=prefix_tokens//16
    k_macros=k_tile_macros*ntiles
    v_macros=v_tile_macros*ntiles
    k_area=k_macros*CIM_7NM.area_mm2
    v_area=v_macros*SRAM_7NM.area_mm2
    per_layer_area=k_area+v_area
    physical_area_per_tile=k_tile_macros*CIM_7NM.area_mm2+v_tile_macros*SRAM_7NM.area_mm2

    max_layer=min(L, floor((area_budget_mm2+1e-12)/per_layer_area))
    max_tiles=min(L*ntiles,floor((area_budget_mm2+1e-12)/physical_area_per_tile))
    selected_touched=minimum_touched_layers_for_persistent_tiles(max_tiles,ntiles)
    # Original GPU FP16 K and V, each stored per native KV head. No
    # replicated per-query-head prefix and no B copies (shared across batch).
    gpu_kv_per_tile=2*hkv*16*d*FP16_BYTES
    gpu_kv_per_layer=gpu_kv_per_tile*ntiles
    gpu_kv_full_model=gpu_kv_per_layer*L
    weight_bytes=fp16_decoder_weight_bytes()
    baseline_hbm_bytes=weight_bytes+gpu_kv_full_model
    saved_gpu_bytes=max_tiles*gpu_kv_per_tile
    if saved_gpu_bytes>=baseline_hbm_bytes:
        raise AssertionError("invalid all-HBM byte denominator")
    context_bytes=LLAMA.query_heads*(d*FP16_BYTES+4)
    scores_bytes=LLAMA.query_heads*prefix_tokens*4
    uniform_fraction=max_layer/L
    return ResidentLimit(
        prefix_tokens=prefix_tokens,
        area_budget_mm2=area_budget_mm2,
        k_cim_macros_per_full_layer=k_macros,
        v_sram_macros_per_full_layer=v_macros,
        k_cim_area_mm2_per_layer=k_area,
        v_sram_area_mm2_per_layer=v_area,
        complete_kv_macro_area_mm2_per_layer=per_layer_area,
        max_complete_layers_without_peripherals=max_layer,
        signed_k_and_v_macros_per_16token_tile=(k_tile_macros,v_tile_macros),
        physical_area_per_16token_tile_mm2=physical_area_per_tile,
        max_persistent_16token_layer_tiles_without_peripherals=max_tiles,
        persistent_prefix_token_layer_fraction_ceiling=max_tiles/(L*ntiles),
        minimum_touched_layers_to_pack_maximum_tiles=selected_touched,
        gpu_fp16_prefix_KV_bytes_per_layer=gpu_kv_per_layer,
        gpu_fp16_prefix_KV_bytes_all_layers=gpu_kv_full_model,
        gpu_prefix_KV_fits_entirely_in_nominal_A100_L2=(
            gpu_kv_full_model<=A100_L2_NOMINAL_BYTES
        ),
        score_return_bytes_per_layer=scores_bytes,
        prefix_context_LSE_return_bytes_per_layer=context_bytes,
        context_vs_score_bytes_ratio=scores_bytes/context_bytes,
        max_removed_gpu_KV_bytes_under_persistent_tile_residency=saved_gpu_bytes,
        source_fp16_transformer_and_lm_head_weight_bytes=weight_bytes,
        ideal_weight_plus_KV_HBM_bytes_per_token=baseline_hbm_bytes,
        optimistic_fraction_of_full_decoder_HBM_bytes_removed=saved_gpu_bytes/baseline_hbm_bytes,
        traffic_only_ideal_speedup_ceiling=baseline_hbm_bytes/(baseline_hbm_bytes-saved_gpu_bytes),
        uniform_layer_prefix_work_fraction_if_full_layers_offloaded=uniform_fraction,
        uniform_layer_prefix_attention_speedup_ceiling=(
            L/(L-max_layer) if max_layer<L else None
        ),
    )


@dataclass(frozen=True)
class FusedGPUFloorAndNearMemoryRequirements:
    """Best-case fused GPU prefix and necessary local SRAM/digital rates.

    This is a *necessary* memory/computation lower floor, not a measured
    fused attention latency. It ignores kernel launches, score materialization,
    exp/reduction costs, Q input/output and cache miss effects. Requirements
    for near-memory PV are rates that would have to be exceeded to match
    this floor with all other near-memory work unrealistically free.
    """
    prefix_tokens: int
    batch: int
    source_memory_placement: str
    modeled_available_GPU_KV_bandwidth_Bps: float
    native_FP16_prefix_KV_bytes_per_layer: int
    native_attention_QK_plus_PV_flops_per_layer: int
    ideal_fused_GPU_prefix_floor_s: float
    resident_FP16_V_read_bytes_per_layer: int
    digital_PV_MACs_per_layer: int
    necessary_local_V_read_bandwidth_Bps: float
    necessary_digital_PV_MAC_per_s: float
    maximum_area_left_after_KV_macros_mm2: float
    necessary_PV_MAC_density_per_s_per_mm2: float | None
    evidence: str = ("A100_peak_KV_byte_and_compute_floor;published_7nm_"
                     "storage_capacity;near_memory_rate_necessity_not_sufficiency")

    def json_record(self)->dict:
        return asdict(self)


def fused_gpu_floor_and_requirements(prefix_tokens:int,batch:int,
                                    area_budget_mm2:float,
                                    placement:str)->FusedGPUFloorAndNearMemoryRequirements:
    """Same native GQA geometry, exact prefix KV size and FP16 FMA count.

    Memory placements "HBM" and "L2" are separate ideal cases and MUST
    not be merged into a fabricated measured GPU reference.
    """
    if placement not in ("HBM","L2"):
        raise ValueError("source GPU memory placement must be HBM or L2")
    if type(batch) is not int or batch<=0:
        raise ValueError("positive integer batch required")
    c=capacity_limit(prefix_tokens,area_budget_mm2)
    bw=A100_80GB_SXM_HBM_BPS if placement=="HBM" else A100_L2_MODEL_BPS
    kv_bytes=c.gpu_fp16_prefix_KV_bytes_per_layer
    flops=4*batch*LLAMA.query_heads*prefix_tokens*LLAMA.head_dim
    floor_s=max(kv_bytes/bw,flops/A100_FP16_DENSE_FLOPS_PER_S)
    v_bytes=LLAMA.kv_heads*prefix_tokens*LLAMA.head_dim*FP16_BYTES
    macs=batch*LLAMA.query_heads*prefix_tokens*LLAMA.head_dim
    remaining=area_budget_mm2-c.complete_kv_macro_area_mm2_per_layer
    density=(macs/floor_s/remaining if remaining>0 else None)
    return FusedGPUFloorAndNearMemoryRequirements(
        prefix_tokens=prefix_tokens,batch=batch,
        source_memory_placement=placement,
        modeled_available_GPU_KV_bandwidth_Bps=bw,
        native_FP16_prefix_KV_bytes_per_layer=kv_bytes,
        native_attention_QK_plus_PV_flops_per_layer=flops,
        ideal_fused_GPU_prefix_floor_s=floor_s,
        resident_FP16_V_read_bytes_per_layer=v_bytes,
        digital_PV_MACs_per_layer=macs,
        necessary_local_V_read_bandwidth_Bps=v_bytes/floor_s,
        necessary_digital_PV_MAC_per_s=macs/floor_s,
        maximum_area_left_after_KV_macros_mm2=max(0.0,remaining),
        necessary_PV_MAC_density_per_s_per_mm2=density,
    )
