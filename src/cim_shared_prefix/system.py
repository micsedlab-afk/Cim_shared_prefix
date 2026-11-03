"""Finite SRAM-CIM placement and completed-score decoder timing.

No hardware performance default is assumed. The LLMCompass adapter accepts only
geometry-matched native stage records; additional GPU structural costs and the
suffix QK score must be explicitly accounted for. A model with missing evidence
fails closed rather than substituting zero. All values have SI units.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import ceil, isfinite
from typing import Mapping

from .contracts import Workload, SignedMapping, ScoreInterface


def _nonnegative(name: str, value: float, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name}: finite {'positive' if positive else 'nonnegative'} value required")


@dataclass(frozen=True)
class SystemInputs:
    """Legacy, explicit serial input contract retained for baseline reproducibility."""
    physical_array_evaluations_per_second: float
    prefix_link_bytes_per_second: float
    score_link_bytes_per_second: float
    gpu_remaining_seconds_per_token: float
    initial_programming_seconds: float
    refill_programming_seconds_per_token: float
    merge_seconds_per_partial: float
    prefix_miss_fraction_per_token: float
    source: str
    gpu_source: str

    def __post_init__(self) -> None:
        for name in ("physical_array_evaluations_per_second", "prefix_link_bytes_per_second", "score_link_bytes_per_second"):
            _nonnegative(name, getattr(self, name), positive=True)
        for name in ("gpu_remaining_seconds_per_token", "initial_programming_seconds", "refill_programming_seconds_per_token", "merge_seconds_per_partial"):
            _nonnegative(name, getattr(self, name))
        _nonnegative("prefix_miss_fraction_per_token", self.prefix_miss_fraction_per_token)
        if self.prefix_miss_fraction_per_token > 1:
            raise ValueError("prefix_miss_fraction_per_token must lie in [0, 1]")
        if not self.source or not self.gpu_source:
            raise ValueError("source and gpu_source are required")


@dataclass(frozen=True)
class LatencyBreakdown:
    """Legacy serial average seconds per generated decoding step."""
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


def evaluate_system(workload: Workload, params: SystemInputs,
                    mapping: SignedMapping = SignedMapping(),
                    scores: ScoreInterface = ScoreInterface()) -> LatencyBreakdown:
    """Historical synthetic-fraction model; use evaluate_decoder for residency."""
    key_bytes = mapping.physical_key_bytes_all_layers(workload)
    merge = workload.layers * workload.prefix_scores_per_layer * max(0, scores.partials_per_score-1) * params.merge_seconds_per_partial
    return LatencyBreakdown(
        amortized_initial_placement_s=key_bytes / params.prefix_link_bytes_per_second / workload.output_tokens,
        amortized_initial_programming_s=params.initial_programming_seconds / workload.output_tokens,
        recurring_refill_s=key_bytes * params.prefix_miss_fraction_per_token / params.prefix_link_bytes_per_second,
        recurring_programming_s=params.refill_programming_seconds_per_token,
        physical_qk_compute_s=mapping.physical_array_work_per_generated_token(workload) / params.physical_array_evaluations_per_second,
        partial_merge_s=merge,
        completed_score_transfer_s=scores.transfer_seconds(workload, params.score_link_bytes_per_second),
        gpu_remaining_s=params.gpu_remaining_seconds_per_token)


@dataclass(frozen=True)
class PrefixTile:
    layer: int
    token_start: int
    token_count: int
    physical_bytes: int


@dataclass(frozen=True)
class ResidencyPlan:
    capacity_bytes: int
    tile_tokens: int
    scratch_bytes: int
    pinned_tiles: tuple[PrefixTile, ...]
    streamed_tiles: tuple[PrefixTile, ...]
    placement_source: str

    @property
    def resident_bytes(self) -> int:
        return sum(t.physical_bytes for t in self.pinned_tiles)

    @property
    def streamed_bytes_per_token(self) -> int:
        return sum(t.physical_bytes for t in self.streamed_tiles)

    @property
    def total_prefix_bytes(self) -> int:
        return self.resident_bytes + self.streamed_bytes_per_token

    @property
    def occupancy_bytes(self) -> int:
        return self.resident_bytes + self.scratch_bytes

    def verify(self, w: Workload, mapping: SignedMapping) -> None:
        if self.capacity_bytes <= 0 or self.tile_tokens <= 0 or not self.placement_source:
            raise ValueError("invalid SRAM residency source/capacity")
        expected = {(layer, start): (min(self.tile_tokens, w.prefix_tokens - start),
                      ceil(min(self.tile_tokens, w.prefix_tokens - start) * w.kv_heads * w.head_dim * w.key_bits / 8) * mapping.key_storage_factor)
                    for layer in range(w.layers) for start in range(0, w.prefix_tokens, self.tile_tokens)}
        observed = {}
        for t in self.pinned_tiles + self.streamed_tiles:
            k = (t.layer, t.token_start)
            if k in observed:
                raise ValueError("duplicate prefix tile")
            observed[k] = (t.token_count, t.physical_bytes)
        if observed != expected or self.occupancy_bytes > self.capacity_bytes:
            raise ValueError("invalid tile coverage or overcommitted SRAM")
        if self.streamed_tiles and self.scratch_bytes < max(t.physical_bytes for t in self.streamed_tiles):
            raise ValueError("streamed tile lacks scratch space")
        if not self.streamed_tiles and self.scratch_bytes:
            raise ValueError("unused scratch reservation")


def plan_residency(w: Workload, capacity_bytes: int, *, tile_tokens: int,
                   mapping: SignedMapping = SignedMapping(),
                   source: str = "explicit pinned-tile + single streaming scratch scheduler") -> ResidencyPlan:
    """Pin reusable K tiles; stream all remaining tiles once per decode step.

    Tiles are indivisible. When full K exceeds capacity, reserve one max-size
    scratch tile. Pin largest complete tiles first, with deterministic tie-breaks.
    A streamed tile is overwritten and programmed on every token. No optimistic
    implicit cache reuse or prefetch is credited.
    """
    if isinstance(capacity_bytes, bool) or not isinstance(capacity_bytes, int) or capacity_bytes <= 0:
        raise ValueError("capacity_bytes must be positive integer")
    if isinstance(tile_tokens, bool) or not isinstance(tile_tokens, int) or tile_tokens <= 0:
        raise ValueError("tile_tokens must be positive integer")
    if not source:
        raise ValueError("residency source required")
    tiles = tuple(PrefixTile(layer, start, min(tile_tokens, w.prefix_tokens-start),
                             ceil(min(tile_tokens, w.prefix_tokens-start)*w.kv_heads*w.head_dim*w.key_bits/8)*mapping.key_storage_factor)
                  for layer in range(w.layers) for start in range(0, w.prefix_tokens, tile_tokens))
    total = sum(t.physical_bytes for t in tiles)
    if total <= capacity_bytes:
        result = ResidencyPlan(capacity_bytes, tile_tokens, 0, tiles, (), source)
    else:
        biggest = max(t.physical_bytes for t in tiles)
        if biggest > capacity_bytes:
            raise ValueError("SRAM capacity cannot accommodate one programmed tile")
        remaining = capacity_bytes - biggest
        pinned = []
        streamed = []
        for t in sorted(tiles, key=lambda a: (-a.physical_bytes, a.layer, a.token_start)):
            if t.physical_bytes <= remaining:
                pinned.append(t)
                remaining -= t.physical_bytes
            else:
                streamed.append(t)
        # A smaller last tile may be streamed, so scratch need only fit the largest miss.
        scratch = max(t.physical_bytes for t in streamed) if streamed else 0
        result = ResidencyPlan(capacity_bytes, tile_tokens, scratch, tuple(pinned), tuple(streamed), source)
    result.verify(w, mapping)
    return result


REQUIRED_GPU_STAGES = (
    "qkv_proj", "attn_score", "softmax", "attn_apply", "out_proj",
    "layernorm0", "mlp_fc1", "gelu", "mlp_fc2", "layernorm1",
    "allreduce_mha", "allreduce_ffn",
)


@dataclass(frozen=True)
class GPUBlock:
    """One complete LLMCompass block at one exact decode context."""
    context_length: int
    batch: int
    query_heads: int
    kv_heads: int
    head_dim: int
    d_model: int
    ffn_dim: int
    precision: str
    stages_s: Mapping[str, float]
    suffix_score_s: float
    structural_s_per_token: float
    source: str
    suffix_source: str
    structural_source: str
    memory_protocol: str
    provenance: str

    def validate(self, w: Workload, token_index: int) -> None:
        if (self.batch, self.query_heads, self.kv_heads, self.head_dim) != (w.batch, w.query_heads, w.kv_heads, w.head_dim):
            raise ValueError("LLMCompass batch/head/GQA geometry mismatch")
        if self.context_length != w.prefix_tokens + w.suffix_tokens + token_index + 1:
            raise ValueError("LLMCompass context must match this generated token exactly")
        if self.d_model != self.query_heads * self.head_dim or self.ffn_dim <= 0:
            raise ValueError("LLMCompass model width/FFN mismatch")
        if self.precision not in ("fp16", "bf16"):
            raise ValueError("unsupported GPU precision")
        if not self.source or not self.suffix_source or not self.structural_source or not self.memory_protocol or not self.provenance:
            raise ValueError("missing GPU stage provenance")
        if set(self.stages_s) != set(REQUIRED_GPU_STAGES):
            raise ValueError("GPU stage coverage incomplete or duplicated")
        for name, value in self.stages_s.items():
            _nonnegative(name, value)
        _nonnegative("suffix_score_s", self.suffix_score_s)
        _nonnegative("structural_s_per_token", self.structural_s_per_token)
        if w.suffix_tokens + token_index == 0 and self.suffix_score_s != 0:
            raise ValueError("suffix score must be zero for empty suffix")

    @property
    def other_gpu_s_per_layer(self) -> float:
        return sum(v for k, v in self.stages_s.items() if k != "attn_score")


def magma_a100_block(payload: Mapping[str, object], w: Workload, token_index: int,
                     *, suffix_score_s: float | None, suffix_source: str,
                     structural_s_per_token: float | None, structural_source: str) -> GPUBlock:
    """Read MAGMA's native LLMCompass A100 point schema, without extrapolating.

    MAGMA's block has H=16, d_head=128, MHA and FP16/HBM2e. A full-attention
    score record is retained for provenance but never charged along with CIM
    prefix QK. A *separate*, matched suffix score timing is mandatory when the
    suffix is nonempty. Reshape/cache/sampling costs require separate evidence.
    """
    if payload.get("schema") != "magma.llmcompass-native-a100-point/v1" or payload.get("native_llmcompass_timing") is not True or payload.get("magma_latency_provider_used") is not False:
        raise ValueError("expected genuine native MAGMA A100/LLMCompass point")
    point, memory, records = payload.get("point"), payload.get("memory"), payload.get("stages")
    if not isinstance(point, Mapping) or not isinstance(memory, Mapping) or not isinstance(records, list):
        raise ValueError("malformed native LLMCompass point")
    if memory.get("protocol") != "HBM2e" or point.get("precision") != "fp16" or point.get("heads") != 16 or point.get("d_head") != 128 or point.get("d_model") != 2048 or point.get("ffn_dim") != 8192:
        raise ValueError("native A100 geometry or protocol unsupported")
    if w.query_heads != w.kv_heads:
        raise ValueError("MAGMA native A100 path only supports MHA; no GQA substitution")
    if suffix_score_s is None and w.suffix_tokens + token_index > 0:
        raise ValueError("suffix score missing; cannot subtract full LLMCompass score")
    if structural_s_per_token is None:
        raise ValueError("uncounted structural GPU operations require explicit cost")
    stages = {}
    for rec in records:
        if not isinstance(rec, Mapping):
            raise ValueError("invalid LLMCompass stage entry")
        name = rec.get("stage")
        if name in REQUIRED_GPU_STAGES:
            if name in stages:
                raise ValueError("duplicate LLMCompass stage")
            stages[str(name)] = rec.get("latency_seconds")
    b = GPUBlock(int(point["context_length"]), int(point["batch_size"]), 16, 16, 128, 2048, 8192,
                 "fp16", stages, 0.0 if suffix_score_s is None else suffix_score_s,
                 structural_s_per_token, "MAGMA-gpt src/magma/llmcompass_native_a100.py",
                 suffix_source or ("geometry: no suffix score" if w.suffix_tokens + token_index == 0 else ""),
                 structural_source, "HBM2e", str(payload.get("git_commit") or ""))
    b.validate(w, token_index)
    full = payload.get("total_block_latency_seconds")
    if not isinstance(full, (int, float)) or not isfinite(full) or abs(sum(stages.values()) - full) > max(1e-15, full*1e-9):
        raise ValueError("native LLMCompass stage sum mismatch")
    return b


CALIBRATION_FIELDS = ("array", "prefix_link", "query_link", "score_link", "program", "local_reduce", "gpu_reduce", "gpu_quant")


@dataclass(frozen=True)
class PathCalibration:
    array_evals_per_second: float
    prefix_link_bytes_per_second: float
    query_link_bytes_per_second: float
    score_link_bytes_per_second: float
    program_bytes_per_second: float
    local_reduction_seconds_per_combine: float
    gpu_reduction_seconds_per_combine: float
    query_quant_seconds_per_layer: float
    query_bits: int
    prefix_transfer_setup_s_per_tile: float = 0.0
    programming_setup_s_per_tile: float = 0.0
    score_transfer_setup_s_per_layer: float = 0.0
    query_transfer_setup_s_per_layer: float = 0.0
    provenance: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        for name in ("array_evals_per_second", "prefix_link_bytes_per_second", "query_link_bytes_per_second", "score_link_bytes_per_second", "program_bytes_per_second"):
            _nonnegative(name, getattr(self, name), positive=True)
        for name in ("local_reduction_seconds_per_combine", "gpu_reduction_seconds_per_combine", "query_quant_seconds_per_layer", "prefix_transfer_setup_s_per_tile", "programming_setup_s_per_tile", "score_transfer_setup_s_per_layer", "query_transfer_setup_s_per_layer"):
            _nonnegative(name, getattr(self, name))
        if self.query_bits not in (4, 8, 16):
            raise ValueError("query precision unsupported")
        if not isinstance(self.provenance, Mapping) or any(not self.provenance.get(k) for k in CALIBRATION_FIELDS):
            raise ValueError("all calibration sources (including zero-cost assumptions) must be explicit")


@dataclass(frozen=True)
class OverlapEvidence:
    """Permission to hide only the first streamed tile behind QKV, per layer.

    Requires two physical resources and proof that prefetch of prefix K can run
    before query readiness. It does not grant overlap for later streamed tiles,
    result transfer, reductions, softmax or other decoder dependencies.
    """
    independent_dma_and_gpu: bool
    independent_programmer_and_gpu: bool
    prefix_known_before_query: bool
    source: str
    evidence_kind: str = "sensitivity"
    schedule_receipt: str = ""

    def qualified(self) -> bool:
        return bool(self.independent_dma_and_gpu and self.independent_programmer_and_gpu
                    and self.prefix_known_before_query and self.source)


@dataclass(frozen=True)
class TokenLatency:
    token_index: int
    initial_placement_s: float
    initial_programming_s: float
    recurring_refill_s: float
    recurring_programming_s: float
    physical_qk_s: float
    query_quant_s: float
    query_transfer_s: float
    local_signed_reduce_s: float
    completed_score_transfer_s: float
    gpu_partial_merge_s: float
    gpu_suffix_qk_s: float
    gpu_other_s: float
    gpu_structural_s: float
    overlap_credit_s: float = 0.0

    @property
    def serial_s(self) -> float:
        return sum(v for k, v in asdict(self).items() if k not in ("token_index", "overlap_credit_s"))

    @property
    def scheduled_s(self) -> float:
        return self.serial_s - self.overlap_credit_s


@dataclass(frozen=True)
class DecoderResult:
    residence: ResidencyPlan
    tokens: tuple[TokenLatency, ...]
    timing_policy: str
    sources: tuple[str, ...]

    @property
    def serial_mean_tpot_s(self) -> float:
        return sum(t.serial_s for t in self.tokens) / len(self.tokens)

    @property
    def scheduled_mean_tpot_s(self) -> float:
        return sum(t.scheduled_s for t in self.tokens) / len(self.tokens)

    @property
    def first_token_s(self) -> float:
        return self.tokens[0].scheduled_s

    @property
    def total_decode_s(self) -> float:
        return sum(t.scheduled_s for t in self.tokens)

    @property
    def aggregate_components_s(self) -> dict[str, float]:
        return {k: sum(getattr(t, k) for t in self.tokens)/len(self.tokens)
                for k in TokenLatency.__dataclass_fields__ if k != "token_index"}


def evaluate_decoder(w: Workload, plan: ResidencyPlan, timing: PathCalibration,
                     gpu_blocks: tuple[GPUBlock, ...], *,
                     mapping: SignedMapping = SignedMapping(),
                     scores: ScoreInterface = ScoreInterface(),
                     overlap: OverlapEvidence | None = None) -> DecoderResult:
    """One causal decoder-step schedule per output token, full layer coverage.

    Initial pinning is charged only on the first token; streams are refilled and
    reprogrammed on *every* token. No softmax overlap with prefix transfer is
    possible because completed scores must reach the GPU before softmax.
    """
    plan.verify(w, mapping)
    if len(gpu_blocks) != w.output_tokens:
        raise ValueError("need exactly one geometry-matched GPU block per output token")
    if scores.partials_per_score > mapping.array_evaluations_per_logical_operation:
        raise ValueError("link partial count exceeds physical signed product count")
    if overlap and not overlap.qualified():
        raise ValueError("overlap requires physical concurrency evidence")
    for index, gpu in enumerate(gpu_blocks):
        gpu.validate(w, index)
        # A non-FP16 CIM query requires a separately sourced quantization cost.
        # The precision and cost appear explicitly in PathCalibration.
    def transfer(tiles: tuple[PrefixTile, ...]) -> float:
        return sum(t.physical_bytes/timing.prefix_link_bytes_per_second + timing.prefix_transfer_setup_s_per_tile for t in tiles)
    def program(tiles: tuple[PrefixTile, ...]) -> float:
        return sum(t.physical_bytes/timing.program_bytes_per_second + timing.programming_setup_s_per_tile for t in tiles)
    initial_transfer, initial_program = transfer(plan.pinned_tiles), program(plan.pinned_tiles)
    recurring_transfer, recurring_program = transfer(plan.streamed_tiles), program(plan.streamed_tiles)
    query_bytes = ceil(w.batch*w.query_heads*w.head_dim*timing.query_bits/8)
    query_transfer = w.layers*(query_bytes/timing.query_link_bytes_per_second + timing.query_transfer_setup_s_per_layer)
    score_transfer = w.layers*timing.score_transfer_setup_s_per_layer + scores.bytes_per_generated_token(w)/timing.score_link_bytes_per_second
    combinations_local = mapping.array_evaluations_per_logical_operation - scores.partials_per_score
    combinations_gpu = scores.partials_per_score - 1
    reductions_local = w.layers*w.prefix_scores_per_layer*combinations_local*timing.local_reduction_seconds_per_combine
    reductions_gpu = w.layers*w.prefix_scores_per_layer*combinations_gpu*timing.gpu_reduction_seconds_per_combine
    qk = mapping.physical_array_work_per_generated_token(w)/timing.array_evals_per_second
    rows = []
    first_by_layer = {}
    for t in plan.streamed_tiles:
        current = first_by_layer.get(t.layer)
        if current is None or t.token_start < current.token_start:
            first_by_layer[t.layer] = t
    for i, gpu in enumerate(gpu_blocks):
        credit = 0.0
        if overlap:
            # Only one first scratch-tile prefetch can precede each layer's QKV.
            for layer, t in first_by_layer.items():
                prefetch = (t.physical_bytes/timing.prefix_link_bytes_per_second + timing.prefix_transfer_setup_s_per_tile
                            + t.physical_bytes/timing.program_bytes_per_second + timing.programming_setup_s_per_tile)
                credit += min(prefetch, gpu.stages_s["qkv_proj"])
        row = TokenLatency(i,
            initial_transfer if i == 0 else 0.0, initial_program if i == 0 else 0.0,
            recurring_transfer, recurring_program, qk,
            w.layers*timing.query_quant_seconds_per_layer, query_transfer,
            reductions_local, score_transfer, reductions_gpu,
            gpu.suffix_score_s*w.layers, gpu.other_gpu_s_per_layer*w.layers,
            gpu.structural_s_per_token, credit)
        if row.scheduled_s <= 0:
            raise ValueError("nonpositive complete decoder latency")
        rows.append(row)
    if overlap and overlap.evidence_kind == "physical_trace" and not overlap.schedule_receipt:
        raise ValueError("physical overlap requires an auditable schedule receipt")
    policy = ("trace-supported-first-tile-qkv-overlap" if overlap and overlap.evidence_kind == "physical_trace"
              else "optimistic-first-tile-qkv-overlap-sensitivity" if overlap else "serial-no-overlap")
    return DecoderResult(plan, tuple(rows), policy,
                         tuple(sorted(set([g.source for g in gpu_blocks] + [g.suffix_source for g in gpu_blocks]
                                          + [g.structural_source for g in gpu_blocks] + list(timing.provenance.values())
                                          + [plan.placement_source]))))
