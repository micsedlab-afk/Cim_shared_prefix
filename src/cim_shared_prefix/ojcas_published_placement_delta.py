"""Paper-native A100/HBM + published 7-nm S2 placement-delta experiment.

Keeps the ORIGINAL 128-macro S2 engine, external signed arithmetic, 728
coordinates and all submitted speedups UNCHANGED.

Studies only three S2 components ALREADY accounted for in the submitted QK
path: K4 active-layer HBM supply, 64-row CIM programming, and native CIM
score accesses. Their incremental difference under a *new* pin/stream
scheduler is computed with original manuscript published model primitives.

This is not a rerun of the original S0/S1/S2 model. The full S2 QK path
also has GPU partial-score merge etc., which are unchanged and cancel
ONLY in a serial, non-overlapped same-endpoint delta comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import isfinite, inf
from typing import Iterable

from .ojcas_manuscript_baseline import (
    AREAS_MM2, BATCHES, PREFIXES, N_LAYERS,
    CIM_MACROS_PER_ENGINE, CIM_ENGINE_AREA_MM2,
    S1_K_TILE_BYTES, PAPER_MAX_QK_STAGE_SPEEDUP,
)
from .ojcas_placement_extension import Candidate, candidates, no_compute_penalty_pinning

# Manuscript Table 2, page 4 — A100 HBM input, not measured sustained BW.
PAPER_A100_HBM_BYTES_PER_SECOND = 2039e9
# Manuscript Section III-B, page 4 — 200 ps per programmed row, 64 rows.
PAPER_S2_PROGRAM_ROW_SECONDS = 200e-12
PAPER_S2_PROGRAM_ROWS = 64
PAPER_S2_PROGRAM_WAVE_SECONDS = PAPER_S2_PROGRAM_ROW_SECONDS * PAPER_S2_PROGRAM_ROWS
# Manuscript Table 2 — source-fabricated unsigned 7-nm macro access.
PAPER_S2_MACRO_ACCESS_SECONDS = 5.5e-9
# One macro engine computes eight KV heads, four queries per head and batch.
ORIGINAL_GQA_QUERY_REUSE = 4

@dataclass(frozen=True)
class PaperStageDelta:
    batch: int
    prefix: int
    area_mm2: int
    output_batch_steps: int
    pin_layers: int
    active_engines: int
    k4_transfer_bytes: int
    programming_wave_count: int
    macro_compute_wave_units: int
    k4_supply_seconds: float
    program_seconds: float
    macro_compute_seconds: float
    three_component_serial_seconds: float
    three_component_independent_resource_floor_seconds: float

    def record(self): return asdict(self)


def paper_stage_components(plan: Candidate, *,
                           hbm_bytes_per_second: float = PAPER_A100_HBM_BYTES_PER_SECOND
                           ) -> PaperStageDelta:
    """All N causal decode steps for the batch, same exact QK score endpoint.

    Sequential layers; pinned engines cannot also be scratch engines.
    Source K4 tile bytes are the logical packed-K representation already in
    the manuscript. 64-row row-programming happens ONCE PER PARALLEL WAVE,
    not once per K4 tile: avoid previous normalized write-price conflation.

    One-time load of pinned layers is charged on the first decoded step.
    Warm pinned layers are reused; all other tiles are supplied/programmed
    each decode step. S2 in original paper already charges these services.

    Independent max() lower bound gives ideal unlimited overlap among
    independent supplies, programmer and compute resources. Sum() is the
    specific explicitly serialized reference, *not* real hardware latency.
    """
    if (not isinstance(hbm_bytes_per_second,(float,int)) or
        isinstance(hbm_bytes_per_second,bool) or
        not isfinite(hbm_bytes_per_second) or hbm_bytes_per_second<=0):
        raise ValueError("positive finite HBM sensitivity required")
    N = plan.decode_batch_steps
    k = plan.pinned_complete_layers
    waves_per_step = plan.compute_tile_waves_per_batch_step
    # First step loads and programs pinned layers exactly once. Thereafter
    # only streamed layers require row programming.
    program_waves = N * waves_per_step - (N - 1)*k
    compute_units = N * waves_per_step * plan.batch
    transfer_bytes = plan.K4_tile_writes_total * S1_K_TILE_BYTES
    assert program_waves>=0 and transfer_bytes>=0
    assert compute_units==plan.compute_wave_units_total
    supply = transfer_bytes / hbm_bytes_per_second
    program = program_waves * PAPER_S2_PROGRAM_WAVE_SECONDS
    compute = compute_units * ORIGINAL_GQA_QUERY_REUSE * PAPER_S2_MACRO_ACCESS_SECONDS
    return PaperStageDelta(
        batch=plan.batch, prefix=plan.prefix_tokens,area_mm2=plan.area_mm2,
        output_batch_steps=N,pin_layers=k,active_engines=plan.total_engines,
        k4_transfer_bytes=transfer_bytes,
        programming_wave_count=program_waves,
        macro_compute_wave_units=compute_units,
        k4_supply_seconds=supply,
        program_seconds=program,
        macro_compute_seconds=compute,
        three_component_serial_seconds=supply+program+compute,
        three_component_independent_resource_floor_seconds=max(supply,program,compute),
    )


def optimize_published_source(batch:int,prefix:int,area:int,*,
                              output_batch_steps:int=128,
                              hbm_bytes_per_second:float=PAPER_A100_HBM_BYTES_PER_SECOND
                              )->tuple[PaperStageDelta,PaperStageDelta,tuple[PaperStageDelta,...]]:
    plans=candidates(batch,prefix,area,decode_batch_steps=output_batch_steps)
    parts=tuple(paper_stage_components(x,hbm_bytes_per_second=hbm_bytes_per_second)
                for x in plans)
    control=parts[0]
    best=min(parts,key=lambda x:(x.three_component_serial_seconds,x.pin_layers))
    return control,best,parts


def first_pin_critical_hbm_Bps(batch:int,prefix:int,area:int,*,
                               output_batch_steps:int=128)->float | None:
    """Analytic crossover: positive pin benefit when HBM BW < critical value.

    If one pinned layer is impossible -> null.
    If compute+program overhead <=0, pin wins for any finite positive HBM
    rate with a warm decode; +inf means always, 0 means no opportunity.
    """
    plans=candidates(batch,prefix,area,decode_batch_steps=output_batch_steps)
    if len(plans)<2: return None
    p0,p1=plans[:2]
    old=paper_stage_components(p0)
    new=paper_stage_components(p1)
    saved=old.k4_transfer_bytes-new.k4_transfer_bytes
    extra=(new.program_seconds+new.macro_compute_seconds
            -old.program_seconds-old.macro_compute_seconds)
    if saved==0: return 0.0
    if extra<=0: return inf
    return saved/extra


def _rel_change(original:float, candidate:float)->float:
    return (original-candidate)/original


def execute_728(*, output_batch_steps:int=128,
                hbm_bytes_per_second:float=PAPER_A100_HBM_BYTES_PER_SECOND
                )->tuple[list[dict],dict]:
    rows=[]
    stats={}
    for area in AREAS_MM2:
        count_win=0
        pareto_win=0
        source_best=[]
        for prefix in PREFIXES:
            for batch in BATCHES:
                control,best,all_options=optimize_published_source(
                    batch,prefix,area,output_batch_steps=output_batch_steps,
                    hbm_bytes_per_second=hbm_bytes_per_second)
                free=no_compute_penalty_pinning(batch,prefix,area,
                    decode_batch_steps=output_batch_steps)
                free_stage=paper_stage_components(free,
                    hbm_bytes_per_second=hbm_bytes_per_second)
                strict=best.three_component_serial_seconds < control.three_component_serial_seconds - 1e-18
                count_win+=int(strict)
                pareto_win+=int(
                    free_stage.k4_transfer_bytes < control.k4_transfer_bytes and
                    free_stage.macro_compute_wave_units==control.macro_compute_wave_units)
                gain=_rel_change(control.three_component_serial_seconds,
                                 best.three_component_serial_seconds)
                if gain < -1e-12:raise AssertionError("selected worse than control")
                if strict:source_best.append(gain)
                br=first_pin_critical_hbm_Bps(batch,prefix,area,
                    output_batch_steps=output_batch_steps)
                rows.append({
                    "B":batch,"Lp":prefix,"Aadd_mm2":area,"Lu":128,
                    "original_engine_macros":CIM_MACROS_PER_ENGINE,
                    "original_engine_mm2":CIM_ENGINE_AREA_MM2,
                    "original_s2_model_historical_stage_speedup":None,
                    "control_pin_layers":0,
                    "chosen_pin_layers":best.pin_layers,
                    "original_engine_count":control.active_engines,
                    "baseline_K4_transfer_bytes_all_outputs":control.k4_transfer_bytes,
                    "policy_K4_transfer_bytes_all_outputs":best.k4_transfer_bytes,
                    "baseline_program_waves_all_outputs":control.programming_wave_count,
                    "policy_program_waves_all_outputs":best.programming_wave_count,
                    "baseline_macro_wave_units_all_outputs":control.macro_compute_wave_units,
                    "policy_macro_wave_units_all_outputs":best.macro_compute_wave_units,
                    "baseline_K4_HBM_supply_us_per_decode_step":control.k4_supply_seconds*1e6/output_batch_steps,
                    "policy_K4_HBM_supply_us_per_decode_step":best.k4_supply_seconds*1e6/output_batch_steps,
                    "baseline_row_program_us_per_decode_step":control.program_seconds*1e6/output_batch_steps,
                    "policy_row_program_us_per_decode_step":best.program_seconds*1e6/output_batch_steps,
                    "baseline_native_CIM_us_per_decode_step":control.macro_compute_seconds*1e6/output_batch_steps,
                    "policy_native_CIM_us_per_decode_step":best.macro_compute_seconds*1e6/output_batch_steps,
                    "baseline_three_component_serial_us_per_step":control.three_component_serial_seconds*1e6/output_batch_steps,
                    "policy_three_component_serial_us_per_step":best.three_component_serial_seconds*1e6/output_batch_steps,
                    "baseline_independent_resource_floor_us_per_step":control.three_component_independent_resource_floor_seconds*1e6/output_batch_steps,
                    "policy_independent_resource_floor_us_per_step":best.three_component_independent_resource_floor_seconds*1e6/output_batch_steps,
                    "three_component_serial_reduction_fraction":gain,
                    "strict_paper_source_better":strict,
                    "strict_no_wave_penalty_pareto":free_stage.k4_transfer_bytes < control.k4_transfer_bytes,
                    "first_pin_critical_HBM_GBps":(br/1e9 if br is not None and isfinite(br) else None),
                    "first_pin_beneficial_at_2039_GBps":(
                        len(all_options)>1 and
                        all_options[1].three_component_serial_seconds
                        <control.three_component_serial_seconds-1e-18),
                    "a100_int4_gpu_latency_measured":None,
                    "full_decoder_tpot_us":None,
                })
        stats[str(area)]={
            "coordinates":91,"source_serial_positive_delta_cases":count_win,
            "source_serial_mean_gain_positive_only":(sum(source_best)/len(source_best) if source_best else 0.0),
            "parameter_free_no_compute_wave_penalty_cases":pareto_win,
        }
    assert len(rows)==728
    return rows, {
        "schema":"ojcas_original_S2_128macro_published_stage_placement_delta/v1",
        "experimental_class":"executed paper-source timing component comparison under serialized source stages, NOT full S2 QK or S0-S2 speedup",
        "reference":"submitted OJCAS 2026 paper, Table2/Section III-B; OJCAS original 128-macro baseline unchanged",
        "A100_source":"A100 2039 GB/s nominal HBM bandwidth cap from submitted Table 2, not sustained measurement",
        "native_CIM_access_ns":PAPER_S2_MACRO_ACCESS_SECONDS*1e9,
        "programming_per_row_ns":PAPER_S2_PROGRAM_ROW_SECONDS*1e9,
        "programming_64row_wave_ns":PAPER_S2_PROGRAM_WAVE_SECONDS*1e9,
        "native_queries_per_prefix_K_group_per_batch":ORIGINAL_GQA_QUERY_REUSE,
        "precision":"original packed K4, original external signed Q4K4 handling",
        "removed_vs_original":"none; this is a separately reported placement-policy overlay",
        "unchanged_common_stages":"suffix QK, GPU merge of two d64 score partials, all remaining GPU decoder operations and external signed arithmetic; therefore not included in incremental component delta",
        "original_submitted_coverage_at_area_8_12":"60/91 at each; unchanged and NOT reinterpreted as new counts",
        "original_max_modeled_s2_QK_stage_speedup":PAPER_MAX_QK_STAGE_SPEEDUP,
        "new_s2_hardware_speedup":None,
        "new_full_decoder_TPOT":None,
        "overlap_model":"three named serial modeled components; independent resource floor is max, not achievable hardware schedule",
        "output_steps_per_batch":output_batch_steps,
        "additional_sram_controller_area":"not modeled, no new silicon area assigned",
        "by_area":stats,
    }
