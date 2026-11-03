"""Source-qualified physical macro accounting and conditional Q4xK4 boundary.

All reported area belongs to incremental accelerator blocks, excluding GPU die.
Macro area is the complete reported physical macro, not a bit-cell estimate.
Unknown support/compute blocks remain explicit sensitivity inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Literal

from .contracts import SignedMapping, ScoreInterface, Workload
from .digital_threshold import DigitalRequirement, EvidenceValue, digital_break_even

YOKOYAMA_DOI = 'https://doi.org/10.1109/VLSICircuits18222.2020.9162985'
DONG_DOI = 'https://doi.org/10.1109/ISSCC19947.2020.9062985'
DONG_JSSC_DOI = 'https://doi.org/10.1109/JSSC.2020.3031290'


@dataclass(frozen=True)
class PhysicalMacro:
    name: str
    capacity_bits: int
    area_mm2: float
    technology_nm: int
    source: str
    scope: str
    porting: str
    precision: str
    def __post_init__(self) -> None:
        if self.capacity_bits <= 0 or self.area_mm2 <= 0 or not isfinite(self.area_mm2):
            raise ValueError('physical macro capacity/area must be positive')
        if not all((self.source, self.scope, self.porting, self.precision)):
            raise ValueError('macro metadata required')


# Renesas 7nm 512x512 *complete SRAM macro*; area recovered from the
# PUBLISHED MACRO density (29.2 Mbit/mm2), not from bit-cell density.
SRAM_7NM = PhysicalMacro(
    'Yokoyama et al. VLSI 2020 high-density SRAM', 512 * 512,
    (512 * 512) / (29.2 * 1e6), 7, YOKOYAMA_DOI,
    'fabricated 512x512 SRAM macro, 29.2 Mbit/mm2 macro density',
    'high-density SRAM; port bandwidth not established for this QK workload',
    'storage; no digital MACs',
)
CIM_7NM = PhysicalMacro(
    'Dong/Sinangil et al. ISSCC 2020 CIM', 64 * 64, 0.0032, 7,
    DONG_DOI, 'fabricated 64x64 8T CIM macro, including ADC/SA, counters, test/reconfiguration',
    'foundry two-port 8T SRAM', '1024 parallel unsigned 4b x 4b MAV, 4b ADC',
)
CIM_NATIVE_MAC_PER_CYCLE = 64 * 16
CIM_PUBLISHED_ACCESS_S = 5.5e-9
CIM_PUBLISHED_GOPS = 372.4  # 2 operations per native MAV, not 372.4 GMAC/s.


@dataclass(frozen=True)
class StorageAllocation:
    macro: PhysicalMacro
    copies: int
    macro_count: int
    stored_logical_key_bits: int
    physical_reserved_bits: int
    area_mm2: float
    mapping: str
    @property
    def utilization(self) -> float:
        return self.stored_logical_key_bits / self.physical_reserved_bits


def allocate_sram(workload: Workload, *, copies: int = 1) -> StorageAllocation:
    """One KV-head per allocation group; no fractional/bit-cell scaling.

    A high-density SRAM macro may need banking/ports beyond this capacity floor.
    """
    if not isinstance(copies, int) or isinstance(copies, bool) or copies < 1:
        raise ValueError('copies must be a positive integer')
    bits_per_head = workload.prefix_tokens * workload.head_dim * workload.key_bits
    n = workload.layers * workload.kv_heads * copies * ceil(bits_per_head / SRAM_7NM.capacity_bits)
    bits = workload.prefix_key_bits_per_layer * workload.layers * copies
    return StorageAllocation(SRAM_7NM, copies, n, bits, n * SRAM_7NM.capacity_bits,
                             n * SRAM_7NM.area_mm2, 'head-separated, integer SRAM macros; Q4 twos-complement keys')


def allocate_cim(workload: Workload, *, copies: int = 1,
                 signed: SignedMapping = SignedMapping()) -> StorageAllocation:
    """64 reduction rows x 16 Q4 weights per tile, per KV head and layer.

    The signed numerical adapter stores physically separate positive/negative
    K subarrays. Four unsigned array evaluations per logical signed Q4xK4
    operation remain mandatory; nothing assumes the native macro is signed.
    """
    if workload.key_bits != 4:
        raise ValueError('published CIM is 4-bit-weight only')
    if not isinstance(copies, int) or isinstance(copies, bool) or copies < 1:
        raise ValueError('copies must be a positive integer')
    tiles_per_head = (ceil(workload.head_dim / 64) * ceil(workload.prefix_tokens / 16)
                      * signed.key_storage_factor)
    n = workload.layers * workload.kv_heads * copies * tiles_per_head
    bits = workload.prefix_key_bits_per_layer * workload.layers * signed.key_storage_factor * copies
    return StorageAllocation(CIM_7NM, copies, n, bits, n * CIM_7NM.capacity_bits,
                             n * CIM_7NM.area_mm2,
                             '64 reduction rows x 16 output scores; signed K split; 4 unsigned evaluations')


@dataclass(frozen=True)
class CIMPhysicalExecution:
    native_unsigned_mav_per_token: int
    physical_unsigned_evaluations_per_logical_mac: int
    signed_partials_per_score: int
    output_score_bytes_per_token: int
    best_case_macro_compute_floor_s: float
    note: str = 'optimistic macro-only throughput lower bound, NOT completed-score latency'


def cim_native_floor(workload: Workload, allocation: StorageAllocation,
                     signed: SignedMapping = SignedMapping(),
                     score: ScoreInterface = ScoreInterface()) -> CIMPhysicalExecution:
    if allocation.macro != CIM_7NM or score.partials_per_score != 1:
        raise ValueError('CIM floor requires CIM allocation and merged 32b scores')
    if score.bits_per_partial != 32:
        raise ValueError('completed score endpoint requires 32-bit signed scores')
    work = signed.physical_array_work_per_generated_token(workload)
    # Full macro occupancy, all banks simultaneously active, no pipeline/tiling overhead:
    # this is a FLOOr. Physical mapping and read/write/merge are unresolved.
    t = ceil(work / (allocation.macro_count * CIM_NATIVE_MAC_PER_CYCLE)) * CIM_PUBLISHED_ACCESS_S
    return CIMPhysicalExecution(
        work, signed.array_evaluations_per_logical_operation,
        ceil(workload.head_dim / 64) * signed.array_evaluations_per_logical_operation,
        score.bytes_per_generated_token(workload), t)


SUPPORT_COMPONENTS = ('controller', 'operand_staging', 'signed_reduction',
                      'score_interface', 'interconnect', 'clock_and_misc')


@dataclass(frozen=True)
class SupportArea:
    """Every external block must be passed, even if its bound is zero.

    A zero-area sensitivity is an optimistic lower bound, never physical proof.
    """
    controller: EvidenceValue
    operand_staging: EvidenceValue
    signed_reduction: EvidenceValue
    score_interface: EvidenceValue
    interconnect: EvidenceValue
    clock_and_misc: EvidenceValue
    def __post_init__(self) -> None:
        for name in SUPPORT_COMPONENTS:
            if getattr(self, name).unit != 'mm2':
                raise ValueError(f'{name}: mm2 required')
    @property
    def total_mm2(self) -> float:
        return sum(getattr(self, n).value for n in SUPPORT_COMPONENTS)
    @property
    def physical_coverage_qualified(self) -> bool:
        # Evidence class is necessary but not sufficient for actual layout
        # qualification; area policy disallows using this as an isoarea gate.
        return False


@dataclass(frozen=True)
class AreaComparison:
    cim: StorageAllocation
    digital: StorageAllocation
    cim_external: SupportArea
    digital_external: SupportArea
    cim_total_mm2: float
    digital_nonmac_mm2: float
    available_digital_core_mm2: float
    qualified_digital_isoarea: bool = False
    def __post_init__(self) -> None:
        if self.qualified_digital_isoarea:
            raise ValueError('requires complete physically implemented digital comparator')


def compare_physical_areas(workload: Workload, *,
                           cim_external: SupportArea, digital_external: SupportArea,
                           cim_copies: int = 1, digital_copies: int = 1,
                           signed: SignedMapping = SignedMapping()) -> AreaComparison:
    cim = allocate_cim(workload, copies=cim_copies, signed=signed)
    digital = allocate_sram(workload, copies=digital_copies)
    cim_total = cim.area_mm2 + cim_external.total_mm2
    digital_nonmac = digital.area_mm2 + digital_external.total_mm2
    return AreaComparison(cim, digital, cim_external, digital_external,
                          cim_total, digital_nonmac, cim_total - digital_nonmac)


def break_even_at_physical_area(workload: Workload, area: AreaComparison, *,
                                cim_completed_score: EvidenceValue,
                                digital_nonmac_score: EvidenceValue) -> DigitalRequirement:
    return digital_break_even(
        workload, added_area_budget_mm2=area.cim_total_mm2,
        digital_sram_area=EvidenceValue(area.digital.area_mm2, 'mm2',
                                        SRAM_7NM.source, 'published'),
        digital_other_area=EvidenceValue(area.digital_external.total_mm2, 'mm2',
                                         'explicit per-component area sensitivities', 'sensitivity'),
        cim_completed_score_stage_time=cim_completed_score,
        digital_nonmac_stage_time=digital_nonmac_score,
        query_bits=4, accumulator_bits=32)
