"""Physical macro residency and conditional complete-score bandwidth boundaries.

Macro-only area uses the 7-nm published *complete macro* footprints in
area_optimizer.py.  Digital MAC silicon area/clock/ports are unknown and are
never derived here.  Every reported cycle count is an optimistic scheduling
bound, not a validated signed-attention datapath or real system measurement.

Pinned layers consume dedicated macros.  Any nonresident layer must have a
separate programmable scratch slot, even when the budget holds a full layer.
The first token programs every layer in a cold start; subsequent tokens load
all unpinned layers. Each layer is visited serially in causal decoder order.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, inf, isfinite
from typing import Literal

from .area_optimizer import CIM_7NM, SRAM_7NM, CIM_PUBLISHED_ACCESS_S
from .contracts import Workload, SignedMapping, ScoreInterface

Kind = Literal['cim', 'sram']
ProgramPolicy = Literal['valid_bits', 'full_macros']
SixSupport = ('controller', 'operand_staging', 'signed_reduction',
              'score_interface', 'interconnect', 'clock_and_misc')


@dataclass(frozen=True)
class ExternalArea:
    """Unmeasured integration blocks in mm2; only explicit sensitivity input.

    Every block must be included in a real physical integration audit.  A zero
    value is a purely optimistic bound, not permission to omit the component.
    """
    controller: float
    operand_staging: float
    signed_reduction: float
    score_interface: float
    interconnect: float
    clock_and_misc: float
    source: str = 'explicit external-area sensitivity; no physical layout'
    evidence: str = 'sensitivity'

    def __post_init__(self):
        if self.evidence != 'sensitivity' or not self.source.strip():
            raise ValueError('unqualified physical support must be labeled sensitivity')
        if any(not isfinite(getattr(self, n)) or getattr(self, n) < 0 for n in SixSupport):
            raise ValueError('all support-area fields must be finite, nonnegative mm2')

    @property
    def total_mm2(self):
        return sum(getattr(self, name) for name in SixSupport)


@dataclass(frozen=True)
class Layout:
    kind: Kind
    macro_name: str
    macro_source: str
    macro_area_mm2: float
    macros_per_tile_unit: int
    tile_units_per_layer: int
    full_layer_macros: int
    all_layer_macros: int
    full_layer_macro_mm2: float
    all_layer_macro_mm2: float
    logical_key_bits_per_layer: int
    signed_physical_key_bits_per_layer: int
    useful_64x16_occupancy: float | None


def layer_layout(workload: Workload, *, kind: Kind,
                 copies: int = 1, signed: SignedMapping = SignedMapping()) -> Layout:
    if workload.key_bits != 4:
        raise ValueError('comparisons require 4-bit keys')
    if not isinstance(copies, int) or isinstance(copies, bool) or copies < 1:
        raise ValueError('copies must be a positive integer')
    if signed.key_storage_factor != 2 or signed.array_evaluations_per_logical_operation != 4:
        raise ValueError('signed Q4xK4 adapter requires physical 2x K / four unsigned evaluations')
    if kind == 'cim':
        # Each 64x64 8T macro contains sixteen 4-bit K column vectors of D=64.
        # Two independently instantiated macros store the K+ and K- planes.
        unit = 2 * copies
        tiles = workload.kv_heads * ceil(workload.head_dim / 64) * ceil(workload.prefix_tokens / 16)
        macro = CIM_7NM
        occupy = (workload.head_dim * workload.prefix_tokens) / (
            ceil(workload.head_dim / 64) * 64 * ceil(workload.prefix_tokens / 16) * 16)
        physical_bits = 2 * workload.prefix_key_bits_per_layer * copies
    elif kind == 'sram':
        unit = copies
        tiles = workload.kv_heads * ceil(
            workload.prefix_tokens * workload.head_dim * workload.key_bits / SRAM_7NM.capacity_bits)
        macro = SRAM_7NM
        occupy = None
        physical_bits = workload.prefix_key_bits_per_layer * copies
    else:
        raise ValueError('unknown macro kind')
    n = tiles * unit
    return Layout(kind, macro.name, macro.source, macro.area_mm2, unit, tiles, n,
                  workload.layers * n, n * macro.area_mm2,
                  workload.layers * n * macro.area_mm2,
                  workload.prefix_key_bits_per_layer, physical_bits, occupy)


@dataclass(frozen=True)
class Residency:
    layout: Layout
    budget_mm2: float
    support_mm2: float
    pinned_layers: int
    scratch_tile_units: int
    instantiated_macros: int
    physical_macro_mm2: float
    complete_mm2: float
    unused_budget_mm2: float
    stream_waves_per_unpinned_layer: int
    full_active_layer_in_scratch: bool
    cold_start: bool
    output_tokens: int
    programming_policy: ProgramPolicy
    programming_visits_total: int
    programming_bytes_total: int
    programming_bytes_per_generated_token: float
    physical_full_layer_bytes: int
    valid_key_layer_bytes: int

    @property
    def first_token_programming_bytes(self) -> int:
        per_layer = (self.physical_full_layer_bytes if self.programming_policy == 'full_macros'
                     else self.valid_key_layer_bytes)
        layers = (self.programming_visits_total -
                  (self.output_tokens - 1) * (self.layout.all_layer_macros //
                    self.layout.full_layer_macros - self.pinned_layers))
        return layers * per_layer

    @property
    def steady_token_programming_bytes(self) -> int:
        per_layer = (self.physical_full_layer_bytes if self.programming_policy == 'full_macros'
                     else self.valid_key_layer_bytes)
        return (self.layout.all_layer_macros // self.layout.full_layer_macros -
                self.pinned_layers) * per_layer


def residency_plan(workload: Workload, *, kind: Kind, budget_mm2: float,
                   support: ExternalArea, pinned_layers: int = 0,
                   copies: int = 1, cold_start: bool = True,
                   programming_policy: ProgramPolicy = 'valid_bits') -> Residency | None:
    """Allocate each whole macro exactly once; None means physically infeasible.

    Instantiated macro units are fixed across all decoded tokens. For R pinned
    complete layers and L>R, scratch must hold at least ONE tile unit.  Spare
    space below a tile is unusable.  Tiling/loading operates sequentially.
    """
    if not isfinite(budget_mm2) or budget_mm2 <= 0:
        raise ValueError('finite positive incremental area budget required')
    if not isinstance(pinned_layers, int) or isinstance(pinned_layers, bool) or not (0 <= pinned_layers <= workload.layers):
        raise ValueError('pinned_layers outside layer count')
    if programming_policy not in ('valid_bits', 'full_macros'):
        raise ValueError('unknown programming payload assumption')
    layout = layer_layout(workload, kind=kind, copies=copies)
    available_area = budget_mm2 - support.total_mm2
    # Tolerance only for binary floating point error at exact macro boundaries.
    total_units = floor((available_area + 1e-12) /
                        (layout.macros_per_tile_unit * layout.macro_area_mm2))
    reserved = pinned_layers * layout.tile_units_per_layer
    if reserved > total_units:
        return None
    if pinned_layers == workload.layers:
        scratch = 0
        wave_count = 0
    else:
        scratch = min(layout.tile_units_per_layer, total_units - reserved)
        if scratch < 1:
            return None
        wave_count = ceil(layout.tile_units_per_layer / scratch)
    n = (reserved + scratch) * layout.macros_per_tile_unit
    physical_area = n * layout.macro_area_mm2
    cost = support.total_mm2 + physical_area
    if cost > budget_mm2 + 1e-10:
        raise AssertionError('physical allocation exceeded budget')
    macros_per_layer = layout.full_layer_macros
    full_bytes = (macros_per_layer * (CIM_7NM.capacity_bits if kind == 'cim'
                                    else SRAM_7NM.capacity_bits)) // 8
    valid_bytes = ceil(layout.signed_physical_key_bits_per_layer / 8)
    bytes_per_layer = full_bytes if programming_policy == 'full_macros' else valid_bytes
    # A cold first-token pass must first load ALL L layer keys, including pins.
    # In warm mode only persistent pinned state is initially preloaded.
    visits = (workload.layers + (workload.output_tokens - 1) *
              (workload.layers - pinned_layers)) if cold_start else (
              workload.output_tokens * (workload.layers - pinned_layers))
    programming = bytes_per_layer * visits
    return Residency(layout, budget_mm2, support.total_mm2, pinned_layers,
                     scratch, n, physical_area, cost, budget_mm2 - cost,
                     wave_count, scratch == layout.tile_units_per_layer,
                     cold_start, workload.output_tokens, programming_policy,
                     visits, programming, programming / workload.output_tokens,
                     full_bytes, valid_bytes)


@dataclass(frozen=True)
class WorkAccounting:
    logical_signed_mac_per_token: int
    native_unsigned_mac_per_token: int
    native_issued_padded_mac_per_token: int
    native_macro_accesses_per_token: int
    signed_partial_values_per_score: int
    reduction_additions_per_token: int
    complete_score_bytes_per_token: int
    query_input_bytes_per_token: int
    key_read_bytes_per_token: int
    mapping_occupancy: float


def accounting(workload: Workload, *, digital_reuse: int = 1,
               digital_reduction_tile_k: int = 64,
               score: ScoreInterface = ScoreInterface()) -> WorkAccounting:
    if workload.key_bits != 4 or score.partials_per_score != 1 or score.bits_per_partial != 32:
        raise ValueError('signed Q4K4 and completed 32-bit score endpoint mandatory')
    q = workload.batch * workload.gqa_ratio
    if not isinstance(digital_reuse, int) or not 1 <= digital_reuse <= q:
        raise ValueError('reuse outside available batch/GQA query sharing')
    if digital_reduction_tile_k < 1:
        raise ValueError('bad digital tile')
    dl = layer_layout(workload, kind='cim')
    total_tiles = dl.tile_units_per_layer * workload.layers
    # Four native unsigned evaluations for every signed dot-product tile.
    padded = total_tiles * q * 4 * 64 * 16
    w = workload.total_prefix_qk_mac_per_generated_token
    s = workload.layers * workload.prefix_scores_per_layer
    # The four products at each D/64 segment must be merged with correct signs.
    partials = 4 * ceil(workload.head_dim / 64)
    key_reads = workload.prefix_key_bytes_all_layers * ceil(q / digital_reuse)
    input_bytes = ceil(workload.layers * workload.batch * workload.query_heads *
                       workload.head_dim * 4 / 8)
    return WorkAccounting(w, 4 * w, padded, total_tiles * q * 4,
                          partials, s * (partials - 1),
                          score.bytes_per_generated_token(workload),
                          input_bytes, key_reads, dl.useful_64x16_occupancy)


def cim_optimistic_compute_floor_s(workload: Workload, plan: Residency,
                                   *, copies: int | None = None, sign_plane_parallelism: int = 2) -> float:
    """Two signed planes can run concurrently only in this optimistic case.

    Every sign product still consumes a macro access. For 2-way sign-plane
    parallelism, four accesses become two serial time slots on two macros;
    1-way requires four slots. Repeated queries cannot share an analog MAC.
    """
    if plan.layout.kind != 'cim':
        raise ValueError('CIM plan required')
    if sign_plane_parallelism not in (1, 2):
        raise ValueError('only serial or paired sign-bank access is modeled')
    physical_copies = plan.layout.macros_per_tile_unit // 2
    if copies is None:
        copies = physical_copies
    if copies != physical_copies:
        raise ValueError('query copies must match physically allocated signed key replicas')
    q = workload.batch * workload.gqa_ratio
    query_rounds = ceil(q / copies)
    unsigned_slots_per_query = 4 // sign_plane_parallelism
    waves_total = (plan.pinned_layers +
                   (workload.layers - plan.pinned_layers) * plan.stream_waves_per_unpinned_layer)
    return waves_total * query_rounds * unsigned_slots_per_query * CIM_PUBLISHED_ACCESS_S


@dataclass(frozen=True)
class RateSet:
    programming_Bps: float
    sram_Bps: float
    output_Bps: float
    query_Bps: float
    reduction_adds_per_s: float
    controller_s: float = 0.0
    source: str = 'explicit link/compute-rate sensitivity; not measured hardware'

    def __post_init__(self):
        for key in ('programming_Bps', 'sram_Bps', 'output_Bps', 'query_Bps', 'reduction_adds_per_s'):
            if not isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not isfinite(self.controller_s) or self.controller_s < 0 or not self.source.strip():
            raise ValueError('invalid controller latency or provenance')


@dataclass(frozen=True)
class StageTimes:
    compute_s: float
    programming_s: float
    sram_read_s: float
    reduction_s: float
    query_s: float
    output_s: float
    control_s: float
    serial_completed_score_s: float


def cim_serial_stages(workload: Workload, plan: Residency, rates: RateSet,
                      *, sign_plane_parallelism: int = 2) -> StageTimes:
    x = accounting(workload)
    comp = cim_optimistic_compute_floor_s(workload, plan,
                                         sign_plane_parallelism=sign_plane_parallelism)
    prog = plan.programming_bytes_per_generated_token / rates.programming_Bps
    redu = x.reduction_additions_per_token / rates.reduction_adds_per_s
    query = x.query_input_bytes_per_token / rates.query_Bps
    output = x.complete_score_bytes_per_token / rates.output_Bps
    return StageTimes(comp, prog, 0.0, redu, query, output, rates.controller_s,
                      comp + prog + redu + query + output + rates.controller_s)


def digital_serial_stages(workload: Workload, plan: Residency, rates: RateSet,
                          *, reuse: int = 1, reduction_tile_k: int = 64) -> StageTimes:
    if plan.layout.kind != 'sram':
        raise ValueError('SRAM plan required')
    x = accounting(workload, digital_reuse=reuse,
                   digital_reduction_tile_k=reduction_tile_k)
    # Full signed-MAC execution time is intentionally absent; it is the
    # explicit throughput/area requirement obtained from remaining time.
    prog = plan.programming_bytes_per_generated_token / rates.programming_Bps
    sram = (x.key_read_bytes_per_token + x.query_input_bytes_per_token) / rates.sram_Bps
    adds = (workload.layers * workload.prefix_scores_per_layer *
            (ceil(workload.head_dim / reduction_tile_k) - 1))
    redu = adds / rates.reduction_adds_per_s
    query = x.query_input_bytes_per_token / rates.query_Bps
    output = x.complete_score_bytes_per_token / rates.output_Bps
    return StageTimes(0.0, prog, sram, redu, query, output, rates.controller_s,
                      prog + sram + redu + query + output + rates.controller_s)


def min_rate_for_deadline(bytes_or_ops: float, deadline_s: float,
                          other_time_s: float) -> float:
    """Necessary one-at-a-time rate, holding all other stages fixed.

    Unbudgeted digital MAC execution is ZERO for this optimistic condition.
    A strictly positive amount of compute time requires a greater rate.
    """
    if bytes_or_ops < 0 or deadline_s <= 0 or other_time_s < 0:
        raise ValueError('invalid work/time')
    if bytes_or_ops == 0:
        return 0.0 if other_time_s < deadline_s else inf
    return bytes_or_ops / (deadline_s - other_time_s) if deadline_s > other_time_s else inf


@dataclass(frozen=True)
class Boundary:
    target_s: float
    kind: Kind
    status: str
    macro_count: int
    macro_mm2: float
    support_mm2: float
    mac_core_area_mm2: float | None
    pinned_layers: int
    waves_per_stream_layer: int
    programming_bytes_per_token: float
    complete_score_bytes_per_token: int
    signed_useful_mac_per_token: int
    optimistic_native_compute_s: float | None
    nonmac_s: float
    full_time_lower_bound_s: float
    minimum_program_Bps: float
    minimum_sram_Bps: float | None
    minimum_output_Bps: float
    required_mac_per_s_per_mm2: float | None
    qualified_digital_isoarea: bool = False

    def __post_init__(self):
        if self.qualified_digital_isoarea:
            raise ValueError('conditional MAC-area requirement cannot qualify digital iso-area')


def boundary(workload: Workload, plan: Residency, rates: RateSet,
             *, target_s: float, reuse: int = 1,
             sign_plane_parallelism: int = 2) -> Boundary:
    if not isfinite(target_s) or target_s <= 0:
        raise ValueError('positive deadline sensitivity required')
    x = accounting(workload, digital_reuse=reuse)
    if plan.layout.kind == 'cim':
        s = cim_serial_stages(workload, plan, rates,
                              sign_plane_parallelism=sign_plane_parallelism)
        p_other = s.serial_completed_score_s - s.programming_s
        o_other = s.serial_completed_score_s - s.output_s
        mp = min_rate_for_deadline(plan.programming_bytes_per_generated_token,
                                   target_s, p_other)
        mo = min_rate_for_deadline(x.complete_score_bytes_per_token, target_s, o_other)
        status = 'macro_only_optimistic_target_met' if s.serial_completed_score_s <= target_s else 'time_bound_missed'
        return Boundary(target_s, 'cim', status, plan.instantiated_macros,
                        plan.physical_macro_mm2, plan.support_mm2, None,
                        plan.pinned_layers, plan.stream_waves_per_unpinned_layer,
                        plan.programming_bytes_per_generated_token,
                        x.complete_score_bytes_per_token, x.logical_signed_mac_per_token,
                        s.compute_s, s.serial_completed_score_s-s.compute_s,
                        s.serial_completed_score_s, mp, None, mo, None)
    s = digital_serial_stages(workload, plan, rates, reuse=reuse)
    mac_area = plan.budget_mm2 - plan.complete_mm2
    rem = target_s - s.serial_completed_score_s
    if mac_area <= 0:
        status, density = 'no_mac_core_area', None
    elif rem <= 0:
        status, density = 'no_MAC_time', None
    else:
        status, density = 'conditional_MAC_density', x.logical_signed_mac_per_token / (rem * mac_area)
    mem_bytes = x.key_read_bytes_per_token + x.query_input_bytes_per_token
    mp = min_rate_for_deadline(plan.programming_bytes_per_generated_token,
                               target_s, s.serial_completed_score_s - s.programming_s)
    ms = min_rate_for_deadline(mem_bytes, target_s,
                               s.serial_completed_score_s - s.sram_read_s)
    mo = min_rate_for_deadline(x.complete_score_bytes_per_token, target_s,
                               s.serial_completed_score_s - s.output_s)
    return Boundary(target_s, 'sram', status, plan.instantiated_macros,
                    plan.physical_macro_mm2, plan.support_mm2, mac_area,
                    plan.pinned_layers, plan.stream_waves_per_unpinned_layer,
                    plan.programming_bytes_per_generated_token,
                    x.complete_score_bytes_per_token, x.logical_signed_mac_per_token,
                    None, s.serial_completed_score_s, s.serial_completed_score_s,
                    mp, ms, mo, density)


def enumerate_plans(workload: Workload, *, kind: Kind, budget_mm2: float,
                    support: ExternalArea, cold_start: bool = True,
                    programming_policy: ProgramPolicy = 'valid_bits') -> tuple[Residency, ...]:
    # Upper bound from full-layer physical macro area avoids enumerating L
    # impossible allocations when only small budgets are available.
    layout = layer_layout(workload, kind=kind)
    rmax = min(workload.layers, int((budget_mm2-support.total_mm2+1e-12) /
                                  layout.full_layer_macro_mm2))
    if rmax < 0:
        return ()
    return tuple(p for r in range(rmax + 1)
                 if (p := residency_plan(workload, kind=kind, budget_mm2=budget_mm2,
                                         support=support, pinned_layers=r,
                                         cold_start=cold_start,
                                         programming_policy=programming_policy)) is not None)


def best_cim(workload: Workload, *, budget_mm2: float, support: ExternalArea,
             rates: RateSet, cold_start: bool = True,
             programming_policy: ProgramPolicy = 'valid_bits') -> tuple[Residency, StageTimes] | None:
    plans = enumerate_plans(workload, kind='cim', budget_mm2=budget_mm2,
                            support=support, cold_start=cold_start,
                            programming_policy=programming_policy)
    if not plans:
        return None
    return min(((p, cim_serial_stages(workload, p, rates)) for p in plans),
               key=lambda x: x[1].serial_completed_score_s)


def best_digital_requirement(workload: Workload, *, budget_mm2: float,
                             support: ExternalArea, rates: RateSet,
                             target_s: float, reuse: int,
                             cold_start: bool = True,
                             programming_policy: ProgramPolicy = 'valid_bits') -> tuple[Residency, Boundary] | None:
    plans = enumerate_plans(workload, kind='sram', budget_mm2=budget_mm2,
                            support=support, cold_start=cold_start,
                            programming_policy=programming_policy)
    if not plans:
        return None
    candidates = [(p, boundary(workload, p, rates, target_s=target_s, reuse=reuse))
                  for p in plans]
    feasible = [(p, b) for p, b in candidates if b.required_mac_per_s_per_mm2 is not None]
    if not feasible:
        return max(candidates, key=lambda item: (item[1].mac_core_area_mm2 or 0)
                   * (item[1].target_s - item[1].nonmac_s))
    return min(feasible, key=lambda item: item[1].required_mac_per_s_per_mm2)