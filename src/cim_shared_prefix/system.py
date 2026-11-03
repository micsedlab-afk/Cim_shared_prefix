"""Source-driven SRAM-CIM to completed-score and full decoder latency accounting.

Every performance parameter must be supplied from a named measured/model source.
No architectural timing or bandwidth constant is silently hardcoded.
The conservative scheduler serializes compute, merge and communications.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .contracts import Workload, SignedMapping, ScoreInterface


def _nonnegative(name: str, value: float, *, positive: bool = False) -> None:
    if not isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name}: finite {'positive' if positive else 'nonnegative'} value required")


@dataclass(frozen=True)
class SystemInputs:
    physical_array_evaluations_per_second: float
    prefix_link_bytes_per_second: float
    score_link_bytes_per_second: float
    gpu_remaining_seconds_per_token: float
    initial_programming_seconds: float
    refill_programming_seconds_per_token: float
    merge_seconds_per_partial: float
    prefix_miss_fraction_per_token: float
    source: str
    # Identifies the allocation of prefix tiles, the GPU model, units and source.
    gpu_source: str

    def __post_init__(self) -> None:
        for name in ("physical_array_evaluations_per_second",
                     "prefix_link_bytes_per_second", "score_link_bytes_per_second"):
            _nonnegative(name, getattr(self, name), positive=True)
        for name in ("gpu_remaining_seconds_per_token", "initial_programming_seconds",
                     "refill_programming_seconds_per_token", "merge_seconds_per_partial"):
            _nonnegative(name, getattr(self, name))
        if not isfinite(self.prefix_miss_fraction_per_token) or not 0 <= self.prefix_miss_fraction_per_token <= 1:
            raise ValueError("prefix_miss_fraction_per_token must lie in [0, 1]")
        if not self.source or not self.gpu_source:
            raise ValueError("source and gpu_source are required")


@dataclass(frozen=True)
class LatencyBreakdown:
    """Seconds per generated token, averaged across output_tokens."""
    amortized_initial_placement_s: float
    amortized_initial_programming_s: float
    recurring_refill_s: float
    recurring_programming_s: float
    physical_qk_compute_s: float
    partial_merge_s: float
    completed_score_transfer_s: float
    gpu_remaining_s: float

    @property
    def total_s(self) -> float:
        return sum(vars(self).values())

    @property
    def qk_path_s(self) -> float:
        return self.total_s - self.gpu_remaining_s


def evaluate_system(
    workload: Workload,
    params: SystemInputs,
    mapping: SignedMapping = SignedMapping(),
    scores: ScoreInterface = ScoreInterface(),
) -> LatencyBreakdown:
    """Compute physical QK cost and overall decode interval under serial scheduling.

    Prefix initially placed once across the measured output horizon. Miss fraction
    refers to additional physical prefix bytes moved each generated token, after
    the initial placement; it must be sourced from a scheduling/residency model.
    No hidden overlap is credited. Merge only counts inter-partial reductions:
    scores.partials_per_score=1 implies reduction completed before the link,
    whose latency must already be included in the physical QK/core timing.
    """
    key_bytes = mapping.physical_key_bytes_all_layers(workload)
    initial_placement = key_bytes / params.prefix_link_bytes_per_second
    refill = key_bytes * params.prefix_miss_fraction_per_token / params.prefix_link_bytes_per_second
    physical_compute = (mapping.physical_array_work_per_generated_token(workload) /
                        params.physical_array_evaluations_per_second)
    intermediate_merge = (workload.layers * workload.prefix_scores_per_layer *
                          max(0, scores.partials_per_score - 1) *
                          params.merge_seconds_per_partial)
    return LatencyBreakdown(
        amortized_initial_placement_s=initial_placement / workload.output_tokens,
        amortized_initial_programming_s=params.initial_programming_seconds / workload.output_tokens,
        recurring_refill_s=refill,
        recurring_programming_s=params.refill_programming_seconds_per_token,
        physical_qk_compute_s=physical_compute,
        partial_merge_s=intermediate_merge,
        completed_score_transfer_s=scores.transfer_seconds(
            workload, params.score_link_bytes_per_second),
        gpu_remaining_s=params.gpu_remaining_seconds_per_token,
    )
