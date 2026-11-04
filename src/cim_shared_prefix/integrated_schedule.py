"""Physically allocated serial Q4/K4 completed-prefix-score schedule.

This is an attainable *conditional* layer-serial order under the specified
bandwidths and active-port caps. A published 5.5 ns unsigned macro access is
the only sourced native timing. Controller/rate/area terms remain sensitivities.
No GPU full-decoder latency or physically qualified 7nm accelerator is implied.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite

from .area_optimizer import CIM_PUBLISHED_ACCESS_S
from .contracts import Workload
from .residency_area import Residency, ExternalArea, accounting, enumerate_plans
from .feasibility import LinkCeilings, PhysicalBudget, lower_bound


def _rate(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive finite")


@dataclass(frozen=True)
class SerialRates:
    """Independent sequential physical services; all values explicit sensitivities."""
    prefix_supply_Bps: float
    programming_Bps: float
    query_Bps: float
    completed_score_Bps: float
    signed_reduction_adds_per_s: float
    partial_scalings_per_s: float
    controller_s_per_layer: float = 0.0
    source: str = "explicit unqualified interface and digital-rate sensitivities"

    def __post_init__(self):
        for key in ("prefix_supply_Bps", "programming_Bps", "query_Bps",
                    "completed_score_Bps", "signed_reduction_adds_per_s",
                    "partial_scalings_per_s"):
            _rate(getattr(self, key), key)
        if (not isfinite(self.controller_s_per_layer) or self.controller_s_per_layer < 0
                or not self.source.strip()):
            raise ValueError("controller timing and rate provenance required")

    def ceilings(self) -> LinkCeilings:
        return LinkCeilings(self.programming_Bps, self.query_Bps,
                            self.completed_score_Bps, self.source)


@dataclass(frozen=True)
class SerialTrace:
    """One complete prefix QK-to-32-bit-completed-scores stage per output token."""
    physical_macros: int
    signed_tile_units_per_layer: int
    pinned_layers: int
    scratch_tile_units: int
    active_macro_ports: int
    native_unsigned_calls_per_token: int
    native_access_slots_per_token: int
    programming_bytes_total: int
    query_bytes_per_token: int
    completed_score_bytes_per_token: int
    signed_reduction_adds_per_token: int
    partial_scalings_per_token: int
    stages_s: dict[str, float]
    serial_completed_score_s: float
    optimistic_resource_floor_s: float
    floor_certificate: str
    status: str = "conditional_serial_schedule; support/rates unqualified; GPU remainder unknown"


def _native_slots(w: Workload, plan: Residency, active_ports: int) -> int:
    """Constructive schedule: program a wave, then service its two signed planes.

    Ignore physical query replicas when computing an achievable schedule. This
    is conservative and keeps every charged macro occupied at most once/slot.
    With a single active port, evaluate the four signed products serially.
    With >=2, execute K+ and K- for each Q sign in pairs, then reduce.
    """
    if plan.layout.kind != "cim" or plan.layout.macros_per_tile_unit < 2:
        raise ValueError("signed CIM plan required")
    q = w.batch * w.gqa_ratio
    tiles = plan.layout.tile_units_per_layer
    total_slots = 0
    for layer in range(w.layers):
        scratch = tiles if layer < plan.pinned_layers else plan.scratch_tile_units
        if scratch < 1:
            raise ValueError("incomplete scratch residency")
        left = tiles
        while left:
            wave = min(scratch, left)
            if active_ports == 1:
                total_slots += wave * q * 4
            else:
                parallel_tiles = min(wave, active_ports // 2)
                total_slots += ceil(wave / parallel_tiles) * q * 2
            left -= wave
    return total_slots


def serial_schedule(w: Workload, plan: Residency, rates: SerialRates,
                    *, active_macro_ports: int | None = None) -> SerialTrace:
    if plan.layout.kind != "cim" or not plan.cold_start:
        raise ValueError("CIM cold-start plan required for matched lower-bound certificate")
    if plan.output_tokens != w.output_tokens:
        raise ValueError("output-token schedule mismatch")
    if plan.layout.full_layer_macros != 2 * w.kv_heads * ceil(w.head_dim / 64) * ceil(w.prefix_tokens / 16):
        raise ValueError("signed physical geometry mismatch")
    if active_macro_ports is not None and (type(active_macro_ports) is not int or active_macro_ports <= 0):
        raise ValueError("positive integer port limit required")
    active = min(plan.instantiated_macros, active_macro_ports or plan.instantiated_macros)
    a = accounting(w)
    slots = _native_slots(w, plan, active)
    scalings = w.layers * w.prefix_scores_per_layer * 4 * ceil(w.head_dim / 64)
    stages = {
        "source_key_supply": plan.programming_bytes_per_generated_token / rates.prefix_supply_Bps,
        "macro_programming": plan.programming_bytes_per_generated_token / rates.programming_Bps,
        "signed_unsigned_macro_evaluations": slots * CIM_PUBLISHED_ACCESS_S,
        "query_delivery": a.query_input_bytes_per_token / rates.query_Bps,
        "signed_partial_scaling": scalings / rates.partial_scalings_per_s,
        "signed_reduction": a.reduction_additions_per_token / rates.signed_reduction_adds_per_s,
        "completed_score_delivery": a.complete_score_bytes_per_token / rates.completed_score_Bps,
        "controller": w.layers * rates.controller_s_per_layer,
    }
    floor = lower_bound(w, PhysicalBudget(plan.budget_mm2, plan.support_mm2),
                        rates.ceilings(), active_macro_port_cap=active)
    if floor is None:
        raise AssertionError("feasible allocation has no macro lower bound")
    total = sum(stages.values())
    if total + max(1e-15, abs(total) * 1e-12) < floor.elapsed_floor_s:
        raise AssertionError("attainable schedule violates independently derived lower bound")
    return SerialTrace(
        plan.instantiated_macros, plan.layout.tile_units_per_layer,
        plan.pinned_layers, plan.scratch_tile_units, active,
        a.native_macro_accesses_per_token, slots, plan.programming_bytes_total,
        a.query_input_bytes_per_token, a.complete_score_bytes_per_token,
        a.reduction_additions_per_token, scalings, stages, total, floor.elapsed_floor_s,
        "max(causal per-layer macro service, compulsory averaged programming, query, score links)",
    )


def best_serial(w: Workload, *, budget_mm2: float, support: ExternalArea,
                rates: SerialRates, active_macro_ports: int | None = None,
                copies: int = 1) -> tuple[Residency, SerialTrace] | None:
    """Enumerate all whole-layer pin counts; return fastest feasible serial policy."""
    choices = []
    for plan in enumerate_plans(w, kind="cim", budget_mm2=budget_mm2,
                                support=support, cold_start=True):
        if plan.layout.macros_per_tile_unit != 2 * copies:
            if copies != 1:
                raise ValueError("replicated query storage needs a separate explicit plan")
            continue
        trace = serial_schedule(w, plan, rates, active_macro_ports=active_macro_ports)
        choices.append((plan, trace))
    return min(choices, key=lambda item: (item[1].serial_completed_score_s,
                                         item[0].instantiated_macros)) if choices else None
