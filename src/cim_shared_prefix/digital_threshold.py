"""Digital SRAM+MAC break-even requirement without invented physical MAC area.

This module deliberately does NOT estimate area per digital multiplier.
It computes the *required* physical performance density for a digital
implementation that occupies the unallocated part of an incremental
SRAM/CIM area budget. Results are conditional, never iso-area measurements.

All external numeric inputs must carry their physical source or be explicitly
labeled as sensitivities; the test fixtures are synthetic only.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from .contracts import Workload

Basis = Literal["measured", "published", "modeled", "sensitivity"]


@dataclass(frozen=True)
class EvidenceValue:
    value: float
    unit: Literal["mm2", "seconds"]
    source: str
    basis: Basis

    def __post_init__(self) -> None:
        if not isfinite(self.value) or self.value < 0:
            raise ValueError("area/time values must be finite and nonnegative")
        if self.basis not in {"measured", "published", "modeled", "sensitivity"}:
            raise ValueError("unknown evidence basis")
        if not self.source or not self.source.strip():
            raise ValueError("every numeric input requires a traceable source or sensitivity label")


@dataclass(frozen=True)
class DigitalRequirement:
    status: Literal["conditional_break_even", "no_core_area", "no_compute_time"]
    logical_mac_per_generated_token: int
    core_area_available_mm2: float
    compute_time_available_s: float
    required_mac_per_second: float | None
    required_mac_per_second_per_mm2: float | None
    qualified_isoarea: bool = False

    def __post_init__(self) -> None:
        if self.qualified_isoarea:
            raise ValueError("break-even algebra cannot constitute physical iso-area evidence")


def digital_break_even(
    workload: Workload,
    *,
    added_area_budget_mm2: float,
    digital_sram_area: EvidenceValue,
    digital_other_area: EvidenceValue,
    cim_completed_score_stage_time: EvidenceValue,
    digital_nonmac_stage_time: EvidenceValue,
    query_bits: int = 4,
    accumulator_bits: int = 32,
) -> DigitalRequirement:
    """Minimum digital MAC/s/mm² to beat a completed-score CIM stage.

    CIM stage timing must include computation, programming, any partial
    merge and score transfers up to the *same* GPU-facing endpoint.

    Digital non-MAC timing must include SRAM reads, local operand staging,
    output reduction/accumulation, score link and digital control. There
    is no assumed compute/communication overlap.

    The threshold is only *conditional* on the user-provided digital
    SRAM/support area and non-MAC time. Even if sources happen to be
    'measured', this function cannot qualify a digital implementation
    without a documented complete physical implementation.
    """
    if not isfinite(added_area_budget_mm2) or added_area_budget_mm2 <= 0:
        raise ValueError("incremental area budget must be finite and positive")
    for name, entry, unit in (
        ("digital_sram_area", digital_sram_area, "mm2"),
        ("digital_other_area", digital_other_area, "mm2"),
        ("cim_completed_score_stage_time", cim_completed_score_stage_time, "seconds"),
        ("digital_nonmac_stage_time", digital_nonmac_stage_time, "seconds"),
    ):
        if entry.unit != unit:
            raise ValueError(f"{name} must have units {unit}")
    if query_bits != 4 or workload.key_bits != 4:
        raise ValueError("this Q4 x K4 comparison requires query_bits=key_bits=4")
    if accumulator_bits < 16:
        raise ValueError("requires a sufficiently wide signed accumulator")

    core_area = added_area_budget_mm2 - digital_sram_area.value - digital_other_area.value
    available_time = cim_completed_score_stage_time.value - digital_nonmac_stage_time.value
    work = workload.total_prefix_qk_mac_per_generated_token
    if core_area <= 0:
        return DigitalRequirement("no_core_area", work, core_area, available_time, None, None)
    if available_time <= 0:
        return DigitalRequirement("no_compute_time", work, core_area, available_time, None, None)
    rate = work / available_time
    return DigitalRequirement(
        status="conditional_break_even",
        logical_mac_per_generated_token=work,
        core_area_available_mm2=core_area,
        compute_time_available_s=available_time,
        required_mac_per_second=rate,
        required_mac_per_second_per_mm2=rate / core_area,
    )
