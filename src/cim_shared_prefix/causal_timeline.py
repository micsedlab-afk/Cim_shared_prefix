"""Causal, token-resolved execution of the published-7nm macro scheduling model.

This runs the actual scheduling arithmetic, token by token and layer by layer.
It preserves cold-start versus persistent-key traffic, explicit sign planes,
port-limited macro service, reduction, and complete score communication.

Optional, externally supplied GPU operator profiles compose S0/S1/S2 at the
same final attention-context endpoint. No GPU latency is invented.
Every duration is conditional on its declared resource-rate source.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite
from typing import Literal

from .area_optimizer import CIM_PUBLISHED_ACCESS_S
from .contracts import Workload
from .integrated_schedule import SerialRates, serial_schedule
from .residency_area import Residency, accounting


def _finite_nonnegative(value: float | None, name: str, *, required: bool = True) -> None:
    if value is None and not required:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ValueError(name + " must be finite and nonnegative")


@dataclass(frozen=True)
class GPUStagePoint:
    """Per-layer source GPU timing at one generated-token context.

    full_attention: complete attention-context output with native full Q/K/V.
    remainder: complete attention-context output from already-delivered prefix
    FP32 score words, including suffix QK and softmax/PV.
    other: Q/K/V projections, norms, RoPE, MLP, KV update etc, excluding any
    attention-context operation. Terminal once-per-token covers LM head/sampling.
    prefix_q4k4: optional S1 signed GPU prefix score and its quantization.
    All entries originate from one declared measured or simulated source.
    """
    full_attention_s: float
    score_remainder_s: float
    other_decoder_per_layer_s: float
    terminal_once_s: float
    prefix_q4k4_s: float | None = None

    def __post_init__(self) -> None:
        for key in ("full_attention_s", "score_remainder_s",
                    "other_decoder_per_layer_s", "terminal_once_s"):
            _finite_nonnegative(getattr(self, key), key)
        _finite_nonnegative(self.prefix_q4k4_s, "prefix_q4k4_s", required=False)


@dataclass(frozen=True)
class GPUSourceSeries:
    """Provenance-bound GPU simulation/measurement outputs, never fabricated here."""
    model_id: str
    model_revision: str
    gpu_device_or_model: str
    gpu_precision: str
    evidence: Literal["native_simulator", "actual_gpu_kernel", "sensitivity"]
    source: str
    batch: int
    prefix_tokens: int
    suffix_tokens: int
    query_heads: int
    kv_heads: int
    head_dim: int
    layers: int
    steps: tuple[GPUStagePoint, ...]

    def verify(self, w: Workload, *, model_id: str, revision: str) -> None:
        if (self.model_id, self.model_revision) != (model_id, revision):
            raise ValueError("GPU model identity/revision mismatch")
        if (self.batch, self.prefix_tokens, self.suffix_tokens,
            self.query_heads, self.kv_heads, self.head_dim, self.layers) != (
            w.batch, w.prefix_tokens, w.suffix_tokens,
            w.query_heads, w.kv_heads, w.head_dim, w.layers):
            raise ValueError("GPU GQA/MHA shape mismatch")
        if len(self.steps) != w.output_tokens:
            raise ValueError("one explicit GPU stage record per output token required")
        if not self.source.strip() or not self.gpu_device_or_model.strip():
            raise ValueError("GPU source provenance missing")
        if self.evidence not in ("native_simulator", "actual_gpu_kernel", "sensitivity"):
            raise ValueError("unsupported GPU evidence class")
        if self.gpu_precision not in ("fp16", "bf16"):
            raise ValueError("only explicit FP16/BF16 GPU reference variants")


@dataclass(frozen=True)
class SRAMRates:
    """External rate assumptions for the conventional SRAM path."""
    prefix_supply_Bps: float
    sram_program_Bps: float
    sram_read_Bps: float
    sram_to_gpu_Bps: float
    source: str

    def __post_init__(self) -> None:
        for key in ("prefix_supply_Bps", "sram_program_Bps",
                    "sram_read_Bps", "sram_to_gpu_Bps"):
            v = getattr(self, key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v <= 0:
                raise ValueError(key + " must be positive finite")
        if not self.source.strip():
            raise ValueError("SRAM rates require provenance")


@dataclass(frozen=True)
class TokenStep:
    index: int
    first_token: bool
    cim_stage_seconds: dict[str, float]
    cim_prefix_complete_score_s: float
    cim_program_bytes: int
    sram_prefix_path_s: float | None
    s0_full_decoder_s: float | None
    s1_full_decoder_s: float | None
    s2_full_decoder_s: float | None


@dataclass(frozen=True)
class Execution:
    status: str
    evidence: str
    cim_macro_count: int
    cim_macro_area_mm2: float
    cim_external_area_mm2_sensitivity: float
    pinned_layers: int
    scratch_tile_units: int
    cim_total_program_bytes: int
    token_steps: tuple[TokenStep, ...]
    cim_mean_prefix_score_s: float
    cim_first_prefix_score_s: float
    cim_steady_mean_prefix_score_s: float | None
    s0_model_mean_tpot_s: float | None
    s1_model_mean_tpot_s: float | None
    s2_model_mean_tpot_s: float | None
    s0_over_s2_conditional: float | None

    def json_record(self) -> dict:
        return asdict(self)


def _slots_one_layer(w: Workload, plan: Residency, ports: int, layer: int) -> int:
    """Physical signed K-bank pair stays resident within each streaming wave."""
    tiles = plan.layout.tile_units_per_layer
    scratch = tiles if layer < plan.pinned_layers else plan.scratch_tile_units
    if scratch < 1:
        raise ValueError("no writable/usable pair for the layer")
    q = w.batch * w.gqa_ratio
    remaining, slots = tiles, 0
    while remaining:
        wave = min(scratch, remaining)
        if ports == 1:
            slots += wave * q * 4
        else:
            simultaneous_pairs = min(wave, ports // 2)
            slots += ceil(wave / simultaneous_pairs) * q * 2
        remaining -= wave
    return slots


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def simulate(
    w: Workload,
    plan: Residency,
    rates: SerialRates,
    *,
    active_macro_ports: int,
    model_id: str = "",
    model_revision: str = "",
    gpu: GPUSourceSeries | None = None,
    sram_plan: Residency | None = None,
    sram_rates: SRAMRates | None = None,
) -> Execution:
    """Execute a legal layer/token serial timeline with explicit physical occupancy.

    A declared GPU source is required for any S0/S1/S2 full decoder number.
    S1 additionally requires a conventional SRAM allocation and signed Q4K4
    prefix GPU kernel cost; it shares the GPU attention remainder with S2.
    The GPU operator profile includes non-attention decoder work once per
    layer; terminal stages execute once per token.
    """
    if plan.layout.kind != "cim" or not plan.cold_start or w.output_tokens != plan.output_tokens:
        raise ValueError("cold-start CIM allocation must match workload")
    if type(active_macro_ports) is not int or active_macro_ports < 1:
        raise ValueError("physical active port limit must be a positive integer")
    if (sram_plan is None) != (sram_rates is None):
        raise ValueError("SRAM allocation and rates must be supplied together")
    if sram_plan is not None:
        if (sram_plan.layout.kind != "sram" or not sram_plan.cold_start or
                sram_plan.output_tokens != w.output_tokens):
            raise ValueError("SRAM allocation must use matching cold-start workload")
        if abs(sram_plan.budget_mm2 - plan.budget_mm2) > 1e-9:
            raise ValueError("S1/S2 must use the same incremental area budget")
    if gpu is not None:
        gpu.verify(w, model_id=model_id, revision=model_revision)

    ports = min(active_macro_ports, plan.instantiated_macros)
    if ports < 1:
        raise ValueError("no executable macro port")
    ac = accounting(w)
    score_per_layer = ac.complete_score_bytes_per_token // w.layers
    scores_per_layer = w.prefix_scores_per_layer
    partial_values_per_score = ac.signed_partial_values_per_score
    query_total = ac.query_input_bytes_per_token
    stages = (
        "source_key_supply", "macro_programming",
        "signed_unsigned_macro_evaluations", "query_delivery",
        "signed_partial_scaling", "signed_reduction",
        "completed_score_delivery", "controller"
    )
    steps: list[TokenStep] = []
    nbytes_written = 0
    for token in range(w.output_tokens):
        token_stages = {key: 0.0 for key in stages}
        prog = 0
        sram_time = 0.0 if sram_plan is not None else None
        for layer in range(w.layers):
            # Physical keys are written once on the first token. A pinned full
            # layer persists; all nonpinned layers must be refilled every token.
            load = token == 0 or layer >= plan.pinned_layers
            bytes_written = (plan.valid_key_layer_bytes if
                             plan.programming_policy == "valid_bits" else
                             plan.physical_full_layer_bytes) if load else 0
            prog += bytes_written
            token_stages["source_key_supply"] += bytes_written / rates.prefix_supply_Bps
            token_stages["macro_programming"] += bytes_written / rates.programming_Bps
            token_stages["signed_unsigned_macro_evaluations"] += (
                _slots_one_layer(w, plan, ports, layer) * CIM_PUBLISHED_ACCESS_S
            )
            # Integer byte distribution avoids fractional transfers if a
            # small synthetic test case has non-byte-aligned per-layer Q.
            q_layer_bytes = query_total // w.layers + int(layer < query_total % w.layers)
            token_stages["query_delivery"] += q_layer_bytes / rates.query_Bps
            token_stages["signed_partial_scaling"] += (
                scores_per_layer * partial_values_per_score / rates.partial_scalings_per_s
            )
            token_stages["signed_reduction"] += (
                scores_per_layer * (partial_values_per_score - 1) /
                rates.signed_reduction_adds_per_s
            )
            token_stages["completed_score_delivery"] += (
                score_per_layer / rates.completed_score_Bps
            )
            token_stages["controller"] += rates.controller_s_per_layer

            if sram_plan is not None:
                assert sram_rates is not None
                sram_load = token == 0 or layer >= sram_plan.pinned_layers
                write = (sram_plan.valid_key_layer_bytes if
                         sram_plan.programming_policy == "valid_bits" else
                         sram_plan.physical_full_layer_bytes) if sram_load else 0
                # SRAM stores K4 once, while every token needs the active
                # prefix key once at the GPU signed Q4K4 kernel.
                key = w.prefix_key_bytes_per_layer
                sram_time += (write / sram_rates.prefix_supply_Bps +
                              write / sram_rates.sram_program_Bps +
                              key / sram_rates.sram_read_Bps +
                              key / sram_rates.sram_to_gpu_Bps)
        nbytes_written += prog
        prefix_time = sum(token_stages.values())
        gpu_point = gpu.steps[token] if gpu is not None else None
        s0 = s1 = s2 = None
        if gpu_point is not None:
            common = w.layers * gpu_point.other_decoder_per_layer_s + gpu_point.terminal_once_s
            s0 = common + w.layers * gpu_point.full_attention_s
            s2 = common + w.layers * gpu_point.score_remainder_s + prefix_time
            if (sram_time is not None and gpu_point.prefix_q4k4_s is not None):
                s1 = common + w.layers * (
                    gpu_point.prefix_q4k4_s + gpu_point.score_remainder_s
                ) + sram_time
        steps.append(TokenStep(token, token == 0, token_stages,
                               prefix_time, prog, sram_time, s0, s1, s2))

    if nbytes_written != plan.programming_bytes_total:
        raise AssertionError("causal programming differs from physical residency plan")
    # Independently cross-check temporal execution against the integrated
    # algebraic per-token serial scheduler, with no averaging shortcuts.
    reference = serial_schedule(w, plan, rates, active_macro_ports=ports)
    averaged = _mean([x.cim_prefix_complete_score_s for x in steps])
    if abs(averaged - reference.serial_completed_score_s) > max(
        1e-14, 1e-11 * reference.serial_completed_score_s
    ):
        raise AssertionError("token timeline disagrees with source macro scheduler")
    for stage in stages:
        avg_stage = _mean([x.cim_stage_seconds[stage] for x in steps])
        if abs(avg_stage - reference.stages_s[stage]) > max(
            1e-14, 1e-11 * reference.stages_s[stage]
        ):
            raise AssertionError("token-resolved stage mismatch: " + stage)

    def maybe_mean(name: str) -> float | None:
        xs = [getattr(x, name) for x in steps]
        return _mean(xs) if all(x is not None for x in xs) else None

    s0_mean = maybe_mean("s0_full_decoder_s")
    s2_mean = maybe_mean("s2_full_decoder_s")
    ratio = s0_mean / s2_mean if s0_mean is not None and s2_mean is not None and s2_mean > 0 else None
    return Execution(
        status="SOURCE_MACRO_EXECUTED_CAUSAL_SIMULATION_WITH_CONDITIONAL_RATES",
        evidence="published 7nm macro geometry/access + explicit external rates; GPU numbers only if source series supplied",
        cim_macro_count=plan.instantiated_macros,
        cim_macro_area_mm2=plan.physical_macro_mm2,
        cim_external_area_mm2_sensitivity=plan.support_mm2,
        pinned_layers=plan.pinned_layers,
        scratch_tile_units=plan.scratch_tile_units,
        cim_total_program_bytes=nbytes_written,
        token_steps=tuple(steps),
        cim_mean_prefix_score_s=averaged,
        cim_first_prefix_score_s=steps[0].cim_prefix_complete_score_s,
        cim_steady_mean_prefix_score_s=_mean([
            x.cim_prefix_complete_score_s for x in steps[1:]
        ]) if len(steps) > 1 else None,
        s0_model_mean_tpot_s=s0_mean,
        s1_model_mean_tpot_s=maybe_mean("s1_full_decoder_s"),
        s2_model_mean_tpot_s=s2_mean,
        s0_over_s2_conditional=ratio,
    )
