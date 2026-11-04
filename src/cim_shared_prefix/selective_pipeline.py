"""Causal finite-buffer selective offload of *fully resident* prefix-QK layers.

This is a physically charged *conditional model*, grounded in the published
7nm 64x64 8T CIM macro for positive/negative signed Q4K4 mapping.

Optimization: retain complete signed-key banks for chosen decoder layers
across O autoregressive outputs; all unselected layers execute on GPU.
For each retained layer, process 16-token x KV-head groups in bounded waves,
overlapping unsigned macro work, off-macro scaling, signed reduction and
completed-score output on distinct resources. Causality holds *within*
each wave; decoder layers themselves are sequential.

No offloading on nonresident layers: this avoids steady reprogramming that
dominates the conventional full-model S2 policy. Physical hardware ports,
off-macro rates and support area are explicitly conditional sensitivities.
The GPU QK savings are source-model inputs, never fabricated here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import ceil, floor, isfinite
from typing import Sequence

from .area_optimizer import CIM_7NM, SRAM_7NM, CIM_PUBLISHED_ACCESS_S
from .contracts import Workload
from .integrated_schedule import SerialRates


def _positive(value: float, name: str) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or value <= 0):
        raise ValueError(name + " must be finite positive")


@dataclass(frozen=True)
class ResidentWave:
    groups: int
    native_cycles: int
    unsigned_compute_s: float
    signed_partial_scaling_s: float
    signed_reduction_s: float
    completed_score_transfer_s: float
    scores: int
    raw_adc_bytes: int
    completed_score_bytes: int


@dataclass(frozen=True)
class ResidentPipeline:
    macro_count_one_layer: int
    macro_area_one_layer_mm2: float
    pinned_layers_max: int
    physical_storage_macros: int
    physical_stage_buffer_area_mm2: float
    stage_buffer_bytes: int
    stage_buffer_limit_mm2: float
    active_macro_ports: int
    groups_per_wave: int
    microbatch_size: int
    waves: tuple[ResidentWave, ...]
    query_delivery_s: float
    warm_pipelined_s_per_layer: float
    warm_nonoverlapped_s_per_layer: float
    cold_source_plus_program_s_per_layer: float
    status: str = "conditional_causal_wave_pipeline_unqualified_external_rates"

    def token_cost(self, *, first_token: bool) -> float:
        return (self.warm_pipelined_s_per_layer +
                (self.cold_source_plus_program_s_per_layer if first_token else 0))


@dataclass(frozen=True)
class Decision:
    token_index: int
    gpu_full_qk_s_per_layer: float
    gpu_suffix_qk_s_per_layer: float
    gpu_full_qk_32layers_s: float
    selected_cim_layers: int
    model_hybrid_qk_service_s: float
    model_gpu_only_qk_service_s: float
    model_hybrid_minus_gpu_s: float


@dataclass(frozen=True)
class SelectiveResult:
    source: str
    workload: dict
    area_budget_mm2: float
    external_support_mm2_sensitivity: float
    extra_setup_per_layer_per_token_s_sensitivity: float
    pinned_physical_layer_count: int
    policy_selected_layers: int
    baseline_mean_gpu_qk_service_s: float
    candidate_mean_hybrid_qk_service_s: float
    candidate_minus_gpu_s_per_token: float
    conditional_qk_speedup: float
    maximum_extra_setup_s_to_retain_advantage: float | None
    minimum_outputs_to_amortize_cold: int | None
    pipeline: ResidentPipeline
    per_token: tuple[Decision, ...]
    evidence: str = ("source-native_gpu_fp16_roofline_vs_7nm_Q4K4O4_macro; "
                     "common_non_QK_stages_cancel_by_assumption; "
                     "not_physical_full_decoder_speedup")

    def json_record(self) -> dict:
        return asdict(self)


def _wave_dag(waves: Sequence[ResidentWave]) -> float:
    """Exact earliest finish of compute -> scale -> reduce -> output.

    Each stage is a distinct serial shared resource. A stage job for wave k
    cannot begin until its preceding stage for k has finished, or the same
    stage has released wave k-1, whichever happens later. The 2-wave
    intermediate storage charged elsewhere permits the schedule.
    """
    start = [[], [], [], []]
    finish = [[], [], [], []]
    for j, wave in enumerate(waves):
        durations = (wave.unsigned_compute_s,
                     wave.signed_partial_scaling_s,
                     wave.signed_reduction_s,
                     wave.completed_score_transfer_s)
        for stage, duration in enumerate(durations):
            prior_job = finish[stage][j - 1] if j else 0.0
            prior_stage = finish[stage - 1][j] if stage else 0.0
            begin = max(prior_job, prior_stage)
            end = begin + duration
            # Blocking-after-service: every adjacent producer/consumer pair
            # has two physical complete-wave buffers. The producer cannot
            # finish wave j while both slots are held by waves j-2,j-1.
            if stage < 3 and j >= 2:
                end = max(end, start[stage + 1][j - 2])
            start[stage].append(begin)
            finish[stage].append(end)
    return finish[-1][-1] if waves else 0.0


def resident_pipeline(
    w: Workload,
    *,
    area_mm2: float,
    support_mm2: float,
    active_ports: int,
    rates: SerialRates,
    operand_stage_support_mm2: float | None = None,
) -> ResidentPipeline | None:
    """Reserve full sign-plane banks; do not assume fractional macro area.

    We allocate a *complete* positive and negative K bank for each selected
    layer and explicitly stage two waves of intermediate digitized partials
    and reconstructed score words in complete published SRAM macros.
    Off-macro digital/scaling services must be independent for overlap.
    """
    if w.key_bits != 4 or w.query_heads % w.kv_heads:
        raise ValueError("published native unsigned Q4K4 split required")
    if type(active_ports) is not int or active_ports < 1:
        raise ValueError("active_ports must be a positive integer")
    _positive(area_mm2, "area_mm2")
    if not isinstance(support_mm2, (float, int)) or not isfinite(support_mm2) or support_mm2 < 0:
        raise ValueError("support_mm2 must be finite nonnegative")
    if operand_stage_support_mm2 is None:
        # Existing architecture sweep allocates support equally to six roles.
        operand_stage_support_mm2 = support_mm2 / 6
    if (operand_stage_support_mm2 < 0 or operand_stage_support_mm2 > support_mm2):
        raise ValueError("operand staging must be funded from support budget")

    d_tiles = ceil(w.head_dim / 64)
    p_tiles = ceil(w.prefix_tokens / 16)
    groups = w.kv_heads * p_tiles
    macro_per_group = 2 * d_tiles
    layer_macros = groups * macro_per_group
    macro_area = layer_macros * CIM_7NM.area_mm2
    max_pin = max(0, floor((area_mm2 - support_mm2 + 1e-12) / macro_area))
    max_pin = min(w.layers, max_pin)
    if max_pin == 0:
        return None

    ports = min(layer_macros * max_pin, active_ports)
    max_groups = min(groups, ports // macro_per_group)
    if max_groups < 1:
        return None
    q = w.batch * w.gqa_ratio
    total_scores = w.batch * w.query_heads * w.prefix_tokens
    qbytes = ceil(w.batch * w.query_heads * w.head_dim * 4 / 8)
    query_time = qbytes / rates.query_Bps

    # Search complete query microbatch sizes and physical wave widths.
    # K is persistently resident; reusing it for each microbatch requires
    # no reprogramming. All B queries are serviced exactly once. Smaller
    # microbatches reduce the buffer demand without inventing extra ports.
    # Include all divisors of B, since incomplete last batches cannot improve
    # worst-case staging or per-wave compute in this homogeneous source model.
    chunk_candidates=[chunk for chunk in range(1,w.batch+1) if w.batch%chunk==0]
    feasible = []
    for microbatch in chunk_candidates:
        for width in range(1,max_groups+1):
            waves: list[ResidentWave] = []
            for batch_start in range(0,w.batch,microbatch):
                this_batch=min(microbatch,w.batch-batch_start)
                active_q=this_batch*w.gqa_ratio
                remaining=groups
                while remaining:
                    take=min(remaining,width)
                    scores=16*take*active_q
                    # Native signed Q4xK4 = Q+/Q- x K+/K-;
                    # both K planes have independent physical macros.
                    cycle=2*active_q
                    vals=scores*4*d_tiles
                    adds=scores*(4*d_tiles-1)
                    raw_bytes=ceil(vals*4/8)
                    complete_bytes=scores*4
                    waves.append(ResidentWave(
                        groups=take,native_cycles=cycle,
                        unsigned_compute_s=cycle*CIM_PUBLISHED_ACCESS_S,
                        signed_partial_scaling_s=vals/rates.partial_scalings_per_s,
                        signed_reduction_s=adds/rates.signed_reduction_adds_per_s,
                        completed_score_transfer_s=complete_bytes/rates.completed_score_Bps,
                        scores=scores,raw_adc_bytes=raw_bytes,
                        completed_score_bytes=complete_bytes,
                    ))
                    remaining-=take
            if sum(x.scores for x in waves)!=total_scores:
                raise AssertionError("missing or duplicated batch/prefix scores")

            # Three independent interstage FIFOs, each storing at most two
            # whole waves: raw O4 codes, FP32 signed partials, completed scores.
            max_intermediates=max(
                x.raw_adc_bytes +
                x.scores*4*d_tiles*4 +
                x.completed_score_bytes for x in waves
            )
            stage_bytes=2*max_intermediates
            storage_macros=ceil(stage_bytes*8/SRAM_7NM.capacity_bits)
            buffer_area=storage_macros*SRAM_7NM.area_mm2
            if buffer_area>operand_stage_support_mm2+1e-12:
                continue
            warm=query_time+_wave_dag(waves)+rates.controller_s_per_layer
            serial=query_time+sum(
                x.unsigned_compute_s+x.signed_partial_scaling_s+
                x.signed_reduction_s+x.completed_score_transfer_s for x in waves
            )+rates.controller_s_per_layer
            feasible.append((warm,-width,-microbatch,width,microbatch,
                             tuple(waves),storage_macros,buffer_area,
                             stage_bytes,serial))

    if not feasible:
        return None
    warm,_negwidth,_negchunk,chosen_width,chosen_microbatch,waves,storage_macros,buffer_area,stage_bytes,serial=min(feasible)
    # Positive and negative key-bank planes each occupy one complete 512B
    # macro image. Source supply precedes programming on the same first use.
    key_bytes = layer_macros * (CIM_7NM.capacity_bits // 8)
    cold = (key_bytes/rates.prefix_supply_Bps +
            key_bytes/rates.programming_Bps +
            0.0)
    return ResidentPipeline(
        macro_count_one_layer=layer_macros,
        macro_area_one_layer_mm2=macro_area,
        pinned_layers_max=max_pin,
        physical_storage_macros=storage_macros,
        physical_stage_buffer_area_mm2=buffer_area,
        stage_buffer_bytes=stage_bytes,
        stage_buffer_limit_mm2=operand_stage_support_mm2,
        active_macro_ports=ports,
        groups_per_wave=chosen_width,
        microbatch_size=chosen_microbatch,
        waves=tuple(waves),
        query_delivery_s=query_time,
        warm_pipelined_s_per_layer=warm,
        warm_nonoverlapped_s_per_layer=serial,
        cold_source_plus_program_s_per_layer=cold,
    )


def optimize_selective(
    w: Workload,
    pipeline: ResidentPipeline,
    *,
    full_qk_s_per_layer: Sequence[float],
    suffix_qk_s_per_layer: Sequence[float],
    area_budget_mm2: float,
    support_mm2: float,
    source: str,
    extra_setup_s_per_layer_per_token: float = 0.0,
) -> SelectiveResult:
    """Select zero or all feasible identical pinned layers for a source series.

    Each remaining layer stays on GPU. No token/device overlap is credited.
    A single cold provisioning is amortized across w.output_tokens, even if
    the policy finally declines offload. We model layer-homogeneous source
    QK operations; per-layer heterogeneous inputs require separate selection.
    """
    if len(full_qk_s_per_layer) != w.output_tokens or len(suffix_qk_s_per_layer) != w.output_tokens:
        raise ValueError("one native source model point per output token required")
    if not source.strip():
        raise ValueError("source identity required")
    if isinstance(extra_setup_s_per_layer_per_token, bool) or not isinstance(
        extra_setup_s_per_layer_per_token, (int, float)
    ) or not isfinite(extra_setup_s_per_layer_per_token) or extra_setup_s_per_layer_per_token < 0:
        raise ValueError("nonnegative extra setup sensitivity required")
    for full, suffix in zip(full_qk_s_per_layer, suffix_qk_s_per_layer):
        if (not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and isfinite(x) and x >= 0 for x in (full,suffix))
                or suffix > full):
            raise ValueError("invalid GPU full/suffix source-model QK latencies")
    gpu_saving = sum(x-y for x,y in zip(full_qk_s_per_layer,suffix_qk_s_per_layer))
    cim_cost = (w.output_tokens *
                (pipeline.warm_pipelined_s_per_layer+extra_setup_s_per_layer_per_token)
                + pipeline.cold_source_plus_program_s_per_layer)
    beneficial = gpu_saving > cim_cost
    selected = pipeline.pinned_layers_max if beneficial else 0
    points = []
    for i,(full,suffix) in enumerate(zip(full_qk_s_per_layer, suffix_qk_s_per_layer)):
        offload = (pipeline.token_cost(first_token=(i==0))+
                   extra_setup_s_per_layer_per_token)
        gpu_total = w.layers * full
        hybrid = ((w.layers-selected)*full + selected*(suffix+offload))
        points.append(Decision(
            token_index=i,gpu_full_qk_s_per_layer=full,
            gpu_suffix_qk_s_per_layer=suffix,
            gpu_full_qk_32layers_s=gpu_total,
            selected_cim_layers=selected,
            model_hybrid_qk_service_s=hybrid,
            model_gpu_only_qk_service_s=gpu_total,
            model_hybrid_minus_gpu_s=hybrid-gpu_total,
        ))
    avg_gpu = sum(x.model_gpu_only_qk_service_s for x in points)/w.output_tokens
    avg_hybrid = sum(x.model_hybrid_qk_service_s for x in points)/w.output_tokens
    gain_without_setup = (gpu_saving/w.output_tokens -
                          pipeline.warm_pipelined_s_per_layer -
                          pipeline.cold_source_plus_program_s_per_layer/w.output_tokens)
    max_setup = gain_without_setup if selected > 0 else None
    # A single constant average GPU QK saving allows an analytical
    # break-even output count, with source's step differences retained in
    # exact chosen-policy decisions above.
    avg_saving = gpu_saving / w.output_tokens
    steady_gain = (avg_saving-pipeline.warm_pipelined_s_per_layer-
                   extra_setup_s_per_layer_per_token)
    min_outputs = (floor(pipeline.cold_source_plus_program_s_per_layer/steady_gain)+1
                   if steady_gain>0 else None)
    return SelectiveResult(
        source=source,
        workload={"batch":w.batch,"prefix_tokens":w.prefix_tokens,
                  "suffix_tokens":w.suffix_tokens,"query_heads":w.query_heads,
                  "kv_heads":w.kv_heads,"head_dim":w.head_dim,
                  "layers":w.layers,"output_tokens":w.output_tokens},
        area_budget_mm2=area_budget_mm2,
        external_support_mm2_sensitivity=support_mm2,
        extra_setup_per_layer_per_token_s_sensitivity=extra_setup_s_per_layer_per_token,
        pinned_physical_layer_count=pipeline.pinned_layers_max,
        policy_selected_layers=selected,
        baseline_mean_gpu_qk_service_s=avg_gpu,
        candidate_mean_hybrid_qk_service_s=avg_hybrid,
        candidate_minus_gpu_s_per_token=avg_hybrid-avg_gpu,
        conditional_qk_speedup=avg_gpu/avg_hybrid,
        maximum_extra_setup_s_to_retain_advantage=max_setup,
        minimum_outputs_to_amortize_cold=min_outputs,
        pipeline=pipeline,
        per_token=tuple(points),
    )
