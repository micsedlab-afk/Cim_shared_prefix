"""Source-grounded LLM geometry and explicitly geometry-only sensitivity sweeps.

Prefix keys are shared across batch; all reported score traffic is a *completed*
QK-score endpoint, prior to scale/softmax, without claiming any measured TPOT.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    revision: str
    query_heads: int
    kv_heads: int
    hidden_size: int
    layers: int
    context_limit: int
    source_url: str
    sliding_window: int | None = None

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.query_heads:
            raise ValueError("nonintegral head dimension")
        return self.hidden_size // self.query_heads

    @property
    def gqa_ratio(self) -> int:
        if self.query_heads % self.kv_heads:
            raise ValueError("invalid GQA geometry")
        return self.query_heads // self.kv_heads


PROFILES = (
    ModelProfile("meta-llama/Llama-3.1-8B",
                 "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b", 32, 8,
                 4096, 32, 131072,
                 "https://huggingface.co/meta-llama/Llama-3.1-8B/blob/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/config.json"),
    ModelProfile("mistralai/Mistral-7B-v0.1",
                 "17688039ea7b7001c702797eed2ab7716a0cc3c2", 32, 8,
                 4096, 32, 32768,
                 "https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/17688039ea7b7001c702797eed2ab7716a0cc3c2/config.json",
                 sliding_window=4096),
    ModelProfile("facebook/opt-2.7b", "main (config; re-pin before execution)",
                 32, 32, 2560, 32, 2048,
                 "https://huggingface.co/facebook/opt-2.7b/blob/main/config.json"),
)


def visible_prefix(profile: ModelProfile, prefix: int, suffix: int, output_index: int) -> int:
    """Number of prefix K visible to the next generated token (0-indexed).

    Sliding window includes the current query position; suffix and already
    generated tokens occupy slots ahead of prefix. At W=4096, S=128, token 0
    can attend at most W-1-S=3967 prefix keys.
    """
    if min(prefix, suffix, output_index) < 0:
        raise ValueError("negative geometry")
    if profile.sliding_window is None:
        return prefix
    return min(prefix, max(0, profile.sliding_window - 1 - suffix - output_index))


def geometry_row(profile: ModelProfile, *, prefix: int, suffix: int, batch: int,
                 output: int, ratio: int, key_bits: int = 4,
                 signed_storage_factor: int = 2,
                 physical_evaluations_factor: int = 4,
                 score_bits: int = 32) -> dict:
    if (any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in
            (prefix, batch, output, ratio, key_bits, signed_storage_factor,
             physical_evaluations_factor, score_bits)) or
            not isinstance(suffix, int) or suffix < 0):
        raise ValueError("invalid positive integer geometry")
    if profile.query_heads % ratio:
        raise ValueError("GQA ratio must divide query heads")
    if prefix + suffix + output > profile.context_limit:
        raise ValueError("total context exceeds source model configuration")
    if score_bits % 8:
        raise ValueError("completed QK score width must be byte aligned")
    kv_heads = profile.query_heads // ratio
    visible = [visible_prefix(profile, prefix, suffix, i) for i in range(output)]
    n_prefix_scores = sum(visible) * batch * profile.query_heads * profile.layers
    logical_macs = n_prefix_scores * profile.head_dim
    logical_key_bits = prefix * kv_heads * profile.head_dim * key_bits * profile.layers
    # Logical prefix K shared across batch, before any physical sign split.
    return {
        "model_id": profile.model_id, "model_revision": profile.revision,
        "evidence_class": "geometry_only_native" if ratio == profile.gqa_ratio else "geometry_only_counterfactual_gqa",
        "attention": "MHA" if ratio == 1 else "GQA", "native_gqa_ratio": profile.gqa_ratio,
        "gqa_ratio": ratio, "query_heads": profile.query_heads, "kv_heads": kv_heads,
        "head_dim": profile.head_dim, "layers": profile.layers,
        "prefix_tokens": prefix, "suffix_tokens": suffix, "batch": batch,
        "output_tokens": output, "context_limit": profile.context_limit,
        "sliding_window": profile.sliding_window if profile.sliding_window is not None else "none",
        "visible_prefix_first": visible[0], "visible_prefix_last": visible[-1],
        "effective_prefix_key_scores_all_outputs": n_prefix_scores,
        "prefix_qk_logical_macs_all_outputs": logical_macs,
        "signed_array_evaluations_all_outputs": logical_macs * physical_evaluations_factor,
        "logical_shared_prefix_k_bytes": (logical_key_bits + 7) // 8,
        "physical_signed_prefix_k_bytes": ((logical_key_bits + 7) // 8) * signed_storage_factor,
        "completed_score_transfer_bytes_all_outputs": n_prefix_scores * score_bits // 8,
        "qk_score_bits": score_bits, "key_bits": key_bits,
        "source_url": profile.source_url,
        "numerical_quality": "not_evaluated", "timing": "not_evaluated",
    }


def sweep_rows(*, prefixes: tuple[int, ...] = (512, 2048, 16384),
               suffixes: tuple[int, ...] = (32, 128, 256),
               batches: tuple[int, ...] = (1, 8, 128),
               outputs: tuple[int, ...] = (16, 128, 256),
               ratios: tuple[int, ...] = (1, 2, 4, 8),
               profiles: tuple[ModelProfile, ...] = PROFILES) -> list[dict]:
    rows = []
    for profile in profiles:
        # OPT has a 2K context; retain useful comparable prefix sensitivities.
        lengths = (256, 512, 1024) if profile.model_id.startswith("facebook/opt") else prefixes
        for prefix, suffix, batch, output, ratio in product(lengths, suffixes, batches, outputs, ratios):
            if prefix + suffix + output > profile.context_limit:
                continue
            rows.append(geometry_row(profile, prefix=prefix, suffix=suffix,
                                     batch=batch, output=output, ratio=ratio))
    return rows


def export_sweep(destination: str | Path, **kwargs) -> list[dict]:
    rows = sweep_rows(**kwargs)
    if not rows:
        raise ValueError("empty geometry sweep")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows
