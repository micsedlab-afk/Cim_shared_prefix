"""Necessary, optimistic resource bounds for source-defined Q4/K4 SRAM-CIM.

This is a *conditional feasibility screen*, not an implemented 7-nm accelerator.
Overlappable hardware-resource durations combine by max, NOT serial sum.
Published macro evidence does not establish physical interface bandwidth.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite
from typing import Literal

from .contracts import Workload, SignedMapping, ScoreInterface
from .area_optimizer import CIM_7NM, CIM_PUBLISHED_ACCESS_S

Evidence = Literal["sensitivity", "measured_system"]


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class LinkCeilings:
    """Upper bounds on deliverable throughput, in bytes/s.

    Sensitivity is the default and never establishes physical implementation.
    The same name `bandwidth` has no implicit shared-bus arbitration; max
    across resource times permits optimistic complete overlap.
    """
    programming_Bps: float
    query_Bps: float
    completed_score_Bps: float
    source: str
    evidence: Evidence = "sensitivity"

    def __post_init__(self) -> None:
        for name in ("programming_Bps", "query_Bps", "completed_score_Bps"):
            _positive(getattr(self, name), name)
        if not self.source.strip() or self.evidence not in ("sensitivity", "measured_system"):
            raise ValueError("bandwidth limits need provenance and evidence class")


@dataclass(frozen=True)
class PhysicalBudget:
    """Incremental accelerator area; support is a compulsory reserved portion."""
    total_mm2: float
    reserved_support_mm2: float
    source: str = "explicit area-sensitivity; support floor is unqualified"

    def __post_init__(self) -> None:
        _positive(self.total_mm2, "total_mm2")
        if not isinstance(self.reserved_support_mm2, (int, float)) or not isfinite(self.reserved_support_mm2) or self.reserved_support_mm2 < 0:
            raise ValueError("reserved_support_mm2 must be nonnegative")
        if not self.source.strip():
            raise ValueError("area needs an explicit source or sensitivity declaration")

    @property
    def macro_count_max(self) -> int:
        return max(0, floor((self.total_mm2 - self.reserved_support_mm2 + 1e-12) /
                            CIM_7NM.area_mm2))


@dataclass(frozen=True)
class NecessaryBound:
    """A lower bound in seconds per output token for the specified datapath."""
    macro_count_max: int
    layer_macros_full: int
    model_macros_full: int
    mandatory_native_calls_per_token: int
    macro_compute_floor_s: float
    programming_bytes_total_min: int
    programming_floor_mean_s: float
    query_bytes_per_token_min: int
    query_floor_s: float
    completed_score_bytes_per_token: int
    score_link_floor_s: float
    elapsed_floor_s: float
    evidence: str = "conditional_macro_calibrated_resource_lower_bound"

    def compare_gpu(self, gpu_tpot_s: float, gpu_source: str) -> str:
        """Return only a conditional screen; no unverified physical claims."""
        _positive(gpu_tpot_s, "gpu_tpot_s")
        if not gpu_source.strip():
            raise ValueError("GPU timing requires a source")
        return ("conditionally_excluded_for_stated_rates" if
                self.elapsed_floor_s > gpu_tpot_s else "not_excluded_by_lower_bound")


def lower_bound(
    w: Workload,
    budget: PhysicalBudget,
    links: LinkCeilings,
    *,
    active_macro_port_cap: int | None = None,
    mapping: SignedMapping = SignedMapping(),
) -> NecessaryBound | None:
    """Optimistic floor for mapped Q4/K4 prefix QK with GPU-side softmax/PV.

    Requires all native 4b signed-key-plane contents accessible per token and
    32-bit postprocessed score words crossing CIM->GPU boundary. Two physical
    K planes and four unsigned evaluations follow the pinned numerical backend.

    Program traffic lower bound uses *valid* K payload only, never padded
    full-macro images; assumes no compression of keys across the link. The
    first token is cold. Thereafter, at most M*512 bytes can remain resident,
    so at least max(0, valid_K-M*512) must be reprogrammed each step.

    O4 ADC quantization remains a numerical approximation; the 32b transfer
    word does not imply an exact signed 32b accumulator. Excludes all additional
    overheads, and assumes independent full overlap across physical resources.
    For sliding-window models, call with per-token effective visible-prefix
    geometries or use a separately validated SWA-aware aggregation.
    """
    if w.key_bits != 4 or mapping.key_storage_factor != 2 or mapping.array_evaluations_per_logical_operation != 4:
        raise ValueError("requires pinned 2-plane signed Q4/K4 mapping")
    if active_macro_port_cap is not None and (
        not isinstance(active_macro_port_cap, int) or isinstance(active_macro_port_cap, bool)
        or active_macro_port_cap <= 0
    ):
        raise ValueError("active_macro_port_cap must be a positive integer")
    n = budget.macro_count_max
    if n == 0:
        return None
    ports = min(n, active_macro_port_cap) if active_macro_port_cap is not None else n
    d_tiles = ceil(w.head_dim / 64)
    p_tiles = ceil(w.prefix_tokens / 16)
    layer_macros = 2 * w.kv_heads * d_tiles * p_tiles
    full_macros = layer_macros * w.layers
    calls_layer = 4 * w.batch * w.query_heads * d_tiles * p_tiles
    mandatory_calls = calls_layer * w.layers
    # Decoder layers are causally ordered; each service round executes at most
    # 'ports' unsigned native macro evaluations, even with ideal programming.
    compute_s = w.layers * ceil(calls_layer / ports) * CIM_PUBLISHED_ACCESS_S

    # At least one copy of both fixed signed K planes must be installed during
    # cold-start. Resident physical macros cannot retain > 512 B each.
    valid_key_bytes = w.prefix_key_bytes_all_layers * 2
    missed_each_following = max(0, valid_key_bytes - n * (4096 // 8))
    program_total = valid_key_bytes + (w.output_tokens - 1) * missed_each_following
    program_time = (program_total / w.output_tokens) / links.programming_Bps

    query_bits = 4 * w.layers * w.batch * w.query_heads * w.head_dim
    query_bytes = ceil(query_bits / 8)
    query_time = query_bytes / links.query_Bps
    completed_bytes = ScoreInterface().bytes_per_generated_token(w)
    score_time = completed_bytes / links.completed_score_Bps

    return NecessaryBound(
        macro_count_max=n,
        layer_macros_full=layer_macros,
        model_macros_full=full_macros,
        mandatory_native_calls_per_token=mandatory_calls,
        macro_compute_floor_s=compute_s,
        programming_bytes_total_min=program_total,
        programming_floor_mean_s=program_time,
        query_bytes_per_token_min=query_bytes,
        query_floor_s=query_time,
        completed_score_bytes_per_token=completed_bytes,
        score_link_floor_s=score_time,
        elapsed_floor_s=max(compute_s, program_time, query_time, score_time),
    )
