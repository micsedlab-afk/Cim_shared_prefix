"""Completed-QK digital traffic, bandwidth, and MAC-area break-even boundaries.

Aggregate bandwidths and reduction rate are externally sourced sensitivities;
SCALE-Sim can supply schedule evidence but never physical MAC/core area.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf, isfinite
from typing import Literal

from .contracts import ScoreInterface, Workload
from .digital_threshold import Basis, DigitalRequirement, EvidenceValue
from .area_optimizer import AreaComparison, break_even_at_physical_area


@dataclass(frozen=True)
class RateEvidence:
    value: float
    unit: Literal['bytes/s', 'adds/s']
    source: str
    basis: Basis
    def __post_init__(self) -> None:
        if not isfinite(self.value) or self.value <= 0:
            raise ValueError('positive finite performance rate required')
        if not self.source.strip() or self.basis not in ('measured', 'published', 'modeled', 'sensitivity'):
            raise ValueError('rate provenance required')


@dataclass(frozen=True)
class DigitalSchedule:
    key_read_reuse: int
    reduction_tile_k: int
    sram_aggregate_bandwidth: RateEvidence
    score_link_bandwidth: RateEvidence
    reduction_throughput: RateEvidence
    controller_latency: EvidenceValue
    overlap: Literal['serial', 'optimistic_pipeline'] = 'serial'
    def __post_init__(self) -> None:
        if self.key_read_reuse < 1 or self.reduction_tile_k < 1:
            raise ValueError('reuse/tile must be positive')
        if (self.sram_aggregate_bandwidth.unit != 'bytes/s' or
                self.score_link_bandwidth.unit != 'bytes/s' or
                self.reduction_throughput.unit != 'adds/s' or
                self.controller_latency.unit != 'seconds'):
            raise ValueError('bandwidth/reduction/control units mismatch')
        if self.overlap not in ('serial', 'optimistic_pipeline'):
            raise ValueError('unknown overlap policy')


@dataclass(frozen=True)
class DigitalTraffic:
    stored_prefix_bytes: int
    key_sram_read_bytes: int
    query_sram_read_bytes: int
    completed_score_bytes: int
    local_reduction_adds: int
    logical_signed_q4_mac: int
    score_partial_tiles: int


def digital_traffic(workload: Workload, *, read_reuse: int,
                    reduction_tile_k: int = 64,
                    score: ScoreInterface = ScoreInterface()) -> DigitalTraffic:
    if workload.key_bits != 4 or score.partials_per_score != 1 or score.bits_per_partial != 32:
        raise ValueError('requires signed Q4 K4 and 32-bit completed scores')
    if reduction_tile_k < 1 or read_reuse < 1 or read_reuse > workload.batch * workload.gqa_ratio:
        raise ValueError('invalid K reuse across batch and grouped query heads')
    # All query heads in a KV group use the same shared K, but each Q is distinct.
    # SRAM reads are explicitly sensitive to the mapped reuse factor.
    k_passes = ceil((workload.batch * workload.gqa_ratio) / read_reuse)
    k_bytes = workload.prefix_key_bytes_all_layers * k_passes
    q_bytes = ceil(workload.layers * workload.batch * workload.query_heads *
                   workload.head_dim * 4 / 8)
    partials = ceil(workload.head_dim / reduction_tile_k)
    scores = workload.layers * workload.prefix_scores_per_layer
    return DigitalTraffic(workload.prefix_key_bytes_all_layers, k_bytes, q_bytes,
                          score.bytes_per_generated_token(workload),
                          scores * (partials - 1),
                          workload.total_prefix_qk_mac_per_generated_token,
                          partials)


@dataclass(frozen=True)
class DigitalStageTimes:
    sram_s: float
    reduction_s: float
    transfer_s: float
    controller_s: float
    nonmac_total_s: float
    overlap: str


def digital_stage_times(traffic: DigitalTraffic, schedule: DigitalSchedule) -> DigitalStageTimes:
    m = ((traffic.key_sram_read_bytes + traffic.query_sram_read_bytes) /
         schedule.sram_aggregate_bandwidth.value)
    r = traffic.local_reduction_adds / schedule.reduction_throughput.value
    o = traffic.completed_score_bytes / schedule.score_link_bandwidth.value
    c = schedule.controller_latency.value
    t = (m + r + o + c) if schedule.overlap == 'serial' else (max(m, r, o) + c)
    return DigitalStageTimes(m, r, o, c, t, schedule.overlap)


@dataclass(frozen=True)
class Boundary:
    status: str
    requirement: DigitalRequirement
    traffic: DigitalTraffic
    nonmac: DigitalStageTimes
    minimum_sram_bandwidth_bytes_s: float
    minimum_score_link_bandwidth_bytes_s: float
    minimum_reduction_adds_s: float
    qualified_digital_isoarea: bool = False
    def __post_init__(self) -> None:
        if self.qualified_digital_isoarea:
            raise ValueError('conditional throughput/area boundary is not iso-area evidence')


def _minimum_rate(work: int, deadline: float) -> float:
    if work == 0:
        return 0.0
    return work / deadline if deadline > 0 else inf


def architecture_boundary(workload: Workload, area: AreaComparison, *,
                          cim_completed_score: EvidenceValue,
                          schedule: DigitalSchedule) -> Boundary:
    """Required digital MAC/s/mm2 and one-at-a-time bandwidth thresholds.

    CIM time must be a FULL complete-score end point, including CIM native
    calls, signed merge and output transfers. When unavailable, pass a labeled
    sensitivity target. Digital stages are disjoint from MAC computation.
    Each minimum-rate calculation fixes the remaining schedule parameters.
    """
    if cim_completed_score.unit != 'seconds':
        raise ValueError('CIM score time must use seconds')
    traffic = digital_traffic(workload, read_reuse=schedule.key_read_reuse,
                             reduction_tile_k=schedule.reduction_tile_k)
    times = digital_stage_times(traffic, schedule)
    req = break_even_at_physical_area(
        workload, area, cim_completed_score=cim_completed_score,
        digital_nonmac_score=EvidenceValue(times.nonmac_total_s, 'seconds',
                                          'digital_traffic and explicit schedule rates', 'modeled'))
    d = cim_completed_score.value - times.controller_s
    if schedule.overlap == 'serial':
        sm = d - times.reduction_s - times.transfer_s
        so = d - times.reduction_s - times.sram_s
        sr = d - times.transfer_s - times.sram_s
    else:
        sm = d if max(times.reduction_s, times.transfer_s) < d else -1
        so = d if max(times.reduction_s, times.sram_s) < d else -1
        sr = d if max(times.transfer_s, times.sram_s) < d else -1
    return Boundary(req.status, req, traffic, times,
                    _minimum_rate(traffic.key_sram_read_bytes + traffic.query_sram_read_bytes, sm),
                    _minimum_rate(traffic.completed_score_bytes, so),
                    _minimum_rate(traffic.local_reduction_adds, sr))
