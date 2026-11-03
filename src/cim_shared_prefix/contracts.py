"""Pure, unit-explicit contracts; no undocumented hardware performance defaults.

Layer-local versus full-model quantities and shared-prefix versus per-request
storage are deliberately separate. All time/cost parameters enter in workstreams.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite


@dataclass(frozen=True)
class Workload:
    batch: int
    prefix_tokens: int
    suffix_tokens: int
    query_heads: int
    kv_heads: int
    head_dim: int
    layers: int
    output_tokens: int
    key_bits: int = 4

    def __post_init__(self) -> None:
        values = (self.batch, self.prefix_tokens, self.query_heads, self.kv_heads,
                  self.head_dim, self.layers, self.output_tokens, self.key_bits)
        if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in values):
            raise ValueError("positive integer geometry and key precision required")
        if not isinstance(self.suffix_tokens, int) or self.suffix_tokens < 0:
            raise ValueError("suffix_tokens must be nonnegative integer")
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be a multiple of kv_heads")
        if self.key_bits not in (2, 4, 8, 16):
            raise ValueError("unsupported key precision")

    @property
    def gqa_ratio(self) -> int:
        return self.query_heads // self.kv_heads

    @property
    def prefix_key_bits_per_layer(self) -> int:
        # One shared prefix per batch: GQA means only kv_heads sets of K.
        return self.prefix_tokens * self.kv_heads * self.head_dim * self.key_bits

    @property
    def prefix_key_bytes_per_layer(self) -> int:
        return ceil(self.prefix_key_bits_per_layer / 8)

    @property
    def prefix_key_bytes_all_layers(self) -> int:
        return self.layers * self.prefix_key_bytes_per_layer

    @property
    def prefix_scores_per_layer(self) -> int:
        return self.batch * self.query_heads * self.prefix_tokens

    @property
    def prefix_qk_mac_per_layer(self) -> int:
        return self.prefix_scores_per_layer * self.head_dim

    @property
    def total_prefix_qk_mac_per_generated_token(self) -> int:
        return self.layers * self.prefix_qk_mac_per_layer


@dataclass(frozen=True)
class SignedMapping:
    """Physical signed split multipliers; numbers must match numerical backend."""
    key_storage_factor: int = 2
    array_evaluations_per_logical_operation: int = 4

    def __post_init__(self) -> None:
        if self.key_storage_factor < 1 or self.array_evaluations_per_logical_operation < 1:
            raise ValueError("mapping factors must be >= 1")

    def physical_key_bytes_all_layers(self, workload: Workload) -> int:
        return workload.prefix_key_bytes_all_layers * self.key_storage_factor

    def physical_array_work_per_generated_token(self, workload: Workload) -> int:
        return (workload.total_prefix_qk_mac_per_generated_token *
                self.array_evaluations_per_logical_operation)


@dataclass(frozen=True)
class ScoreInterface:
    """Bytes crossing the SRAM/CIM-to-GPU boundary for prefix QK alone.

    partials_per_score=1 means a completed score is merged before the link;
    larger values mean unresolved partial scores cross the link for GPU merge.
    """
    bits_per_partial: int = 32
    partials_per_score: int = 1

    def __post_init__(self) -> None:
        if self.bits_per_partial < 1 or self.bits_per_partial % 8:
            raise ValueError("score precision must be byte-aligned")
        if self.partials_per_score < 1:
            raise ValueError("partials_per_score must be >= 1")

    def bytes_per_generated_token(self, workload: Workload) -> int:
        return (workload.layers * workload.prefix_scores_per_layer *
                self.partials_per_score * (self.bits_per_partial // 8))

    def transfer_seconds(self, workload: Workload, bandwidth_bytes_per_second: float) -> float:
        if not isfinite(bandwidth_bytes_per_second) or bandwidth_bytes_per_second <= 0:
            raise ValueError("link bandwidth must be finite and positive")
        return self.bytes_per_generated_token(workload) / bandwidth_bytes_per_second


@dataclass(frozen=True)
class AreaItem:
    name: str
    count: int
    area_mm2_each: float
    source: str
    inside_published_macro: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.source:
            raise ValueError("area component requires name and provenance")
        if self.count < 0 or not isfinite(self.area_mm2_each) or self.area_mm2_each < 0:
            raise ValueError("area entries must be nonnegative and finite")

    @property
    def total_mm2(self) -> float:
        return self.count * self.area_mm2_each


@dataclass(frozen=True)
class AreaLedger:
    components: tuple[AreaItem, ...]

    def __post_init__(self) -> None:
        names = [entry.name for entry in self.components]
        if len(names) != len(set(names)):
            raise ValueError("duplicate area component name")
        if any(entry.inside_published_macro for entry in self.components):
            raise ValueError("on-macro components must not be counted separately")

    @property
    def total_mm2(self) -> float:
        return sum(entry.total_mm2 for entry in self.components)

    def within(self, budget_mm2: float) -> bool:
        if not isfinite(budget_mm2) or budget_mm2 < 0:
            raise ValueError("budget must be finite and nonnegative")
        return self.total_mm2 <= budget_mm2 + 1e-12
