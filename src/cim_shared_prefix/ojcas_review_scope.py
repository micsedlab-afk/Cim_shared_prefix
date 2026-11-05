"""OJCAS reviewer extensions: *additive* S1 placement and decoder-scope limits.

Submitted S2 128-macro QK-stage results remain unchanged. S1 pre-decode
placement was excluded in the manuscript, and S2 active-layer supply was
already timed: NEVER charge S2 K placement twice.

All cost/rate inputs are explicit sensitivities. None represents observed
A100 INT4 kernels, fabricated 7-nm periphery, or full-decoder TPOT.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite
from typing import Literal

from .ojcas_manuscript_baseline import (
    PAPER_MAX_QK_STAGE_SPEEDUP, N_LAYERS, PREFIXES, AREAS_MM2,
    S1_K_TILE_BYTES, paper_capacity, saved_stage_speedup_margin,
)

SRAMType=Literal["S1-HD","S1-DP"]


@dataclass(frozen=True)
class PlacementAmortization:
    technology:str
    prefix_tokens:int
    area_mm2:int
    decode_steps_per_batch:int
    consecutive_batches_sharing_prefix:int
    pin_capacity_full_K4_tiles:int
    physical_placed_K4_bytes_once:int
    amortized_K4_bytes_per_batch_step:float
    cold_placement_us_at_unqualified_interface_rate:float
    placement_us_per_batch_step_at_unqualified_interface_rate:float
    interface_rate_Bps_sensitivity:float
    source:str="original S1 published macro capacity; full 128-token key tiles only"
    area_metric:str="published SRAM macro-only area; external controller unqualified"


def s1_placement_amortization(kind:SRAMType, prefix:int, area_mm2:int,
                              *, decode_steps:int, reuse_batches:int,
                              interface_Bps_sensitivity:float)->PlacementAmortization:
    """One signedness-neutral single-copy packed K4 placement per common prefix.

    We place a maximal whole-tile S1 working set *once* before all R
    consecutive decode batches. Extra per-token SRAM reads/misses are already
    part of original S1 and therefore are NOT recharged here.

    The optimistic placement-only transfer time is bytes / assumed external
    interface rate; physical paths, actual clock and energy remain unqualified.
    """
    if kind not in ("S1-HD","S1-DP"):raise ValueError("unsupported original SRAM path")
    if type(prefix) is not int or prefix not in PREFIXES:raise ValueError("invalid prefix")
    if type(area_mm2) is not int or area_mm2 not in AREAS_MM2:raise ValueError("invalid area")
    if type(decode_steps) is not int or decode_steps<1:raise ValueError("invalid decode length")
    if type(reuse_batches) is not int or reuse_batches<1:raise ValueError("invalid prefix reuse")
    if (not isinstance(interface_Bps_sensitivity,(int,float)) or
        isinstance(interface_Bps_sensitivity,bool) or not isfinite(interface_Bps_sensitivity)
        or interface_Bps_sensitivity<=0):
        raise ValueError("positive finite interface-rate sensitivity required")
    physical=paper_capacity(area_mm2)
    capacity=physical.s1_hd_capacity_bytes if kind=="S1-HD" else physical.s1_dp_capacity_bytes
    total_tiles=N_LAYERS*ceil(prefix/128)
    tile_count=min(total_tiles,capacity//S1_K_TILE_BYTES)
    bytes_total=tile_count*S1_K_TILE_BYTES
    steps=decode_steps*reuse_batches
    return PlacementAmortization(
        technology=kind,prefix_tokens=prefix,area_mm2=area_mm2,
        decode_steps_per_batch=decode_steps,
        consecutive_batches_sharing_prefix=reuse_batches,
        pin_capacity_full_K4_tiles=tile_count,
        physical_placed_K4_bytes_once=bytes_total,
        amortized_K4_bytes_per_batch_step=bytes_total/steps,
        cold_placement_us_at_unqualified_interface_rate=bytes_total/interface_Bps_sensitivity*1e6,
        placement_us_per_batch_step_at_unqualified_interface_rate=bytes_total/steps/interface_Bps_sensitivity*1e6,
        interface_rate_Bps_sensitivity=interface_Bps_sensitivity,
    )


@dataclass(frozen=True)
class DecoderScope:
    qk_share_of_original_full_decoder:float
    s2_stage_speedup_original_model:float
    additional_overhead_fraction_of_original_decoder:float
    resulting_full_decoder_speedup_conditional:float
    no_added_overhead_upper_envelope:float
    max_additional_overhead_fraction_to_preserve_full_speedup:float
    provenance:str="algebraic Amdahl on original modeled QK stage, NOT a measured full-decoder result"


def decoder_scope(qk_fraction:float, *,
                  stage_speedup:float=PAPER_MAX_QK_STAGE_SPEEDUP,
                  extra_overhead_fraction:float=0.0)->DecoderScope:
    """Amdahl: original decoder is normalized to 1.

    T_S0,all = T_nonQK + T_QK.
    T_S2,all = (1-f) + f / original_S2_QK_stage_speedup + Δnew
    Δnew accounts only costs omitted by original S2; existing S2 programming
    and transfers are already in that stage result. No double charging.
    """
    for name,v in (("qk_fraction",qk_fraction),("extra_overhead_fraction",extra_overhead_fraction)):
        if not isinstance(v,(float,int)) or isinstance(v,bool) or not isfinite(v) or v<0:
            raise ValueError(f"invalid {name}")
    if qk_fraction>1:raise ValueError("QK cannot exceed total baseline decode")
    if stage_speedup<=1 or not isfinite(stage_speedup):
        raise ValueError("stage speedup must be existing favorable modeled point")
    neutral=1-qk_fraction+qk_fraction/stage_speedup
    full=1/(neutral+extra_overhead_fraction)
    headroom=qk_fraction*saved_stage_speedup_margin(stage_speedup)
    return DecoderScope(
        qk_share_of_original_full_decoder=qk_fraction,
        s2_stage_speedup_original_model=stage_speedup,
        additional_overhead_fraction_of_original_decoder=extra_overhead_fraction,
        resulting_full_decoder_speedup_conditional=full,
        no_added_overhead_upper_envelope=1/neutral,
        max_additional_overhead_fraction_to_preserve_full_speedup=headroom,
    )


def reference_sensitivity()->dict:
    s1=[]
    for kind in ("S1-HD","S1-DP"):
        for r in (1,8,64):
            p=s1_placement_amortization(kind,2048,8,decode_steps=128,
                  reuse_batches=r,interface_Bps_sensitivity=1e12)
            s1.append(asdict(p))
    amp=[]
    for f in (.01,.05,.10,.20,.40,.60):
        d=decoder_scope(f)
        amp.append(asdict(d))
    return {
        "schema":"ojcas_original_S1_placement_and_S2_decoder_scope_overlay/v1",
        "reference":"submitted 128-macro S2, original S1-HD/S1-DP and max modeled S2 QK stage speedup 1.97",
        "S1_placement_sensitivities":s1,
        "decoder_scope_sensitivities":amp,
        "evidence_class":"algebraic source-macro capacity and explicit 1TBps interface sensitivity; NO measured decoder timing",
        "important_scope":"S2 already charges active-layer programming; no duplicated S2 writes. Only S1 excluded pre-decode placement is added. External sign/reconstruction remains a future separately costed component.",
        "input_precision":"original 4-bit packed K source, not new FP16/A100 swap or on-macro signed replication",
        "qualified_model_level_tokens_per_second":None,
        "qualified_gpu_end_to_end_TPOT":None,
        "qualified_additional_periphery_timing":None,
    }
