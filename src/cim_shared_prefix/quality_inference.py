"""Scalar, source-grounded R0--R3 attention-injection primitives.

Execution depends on Transformers 4.46.2 and the original SRAM2LLM
Sinangil backend. Imports are deliberately lazy so CPU-only geometry and
source-integrity tests can run without gated model weights. R0 delegates
exactly to the unmodified attention kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1
from math import sqrt
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import torch
import torch.nn.functional as F

MODEL_REVISIONS = {
    "mistral": ("mistralai/Mistral-7B-v0.1", "17688039ea7b7001c702797eed2ab7716a0cc3c2"),
    "opt": ("facebook/opt-2.7b", "905a4b602cda5c501f1b3a2650a4152680238254"),
}
TRANSFORMERS_VERSION = "4.46.2"
# Frozen upstream installed Git blob checks (Mistral pin was independently
# qualified by shared_prefix_cim/task/mistral7b-surrogate-20260918).
MISTRAL_SOURCE_BLOB = "321d3dc0daf378b18e9bde140205c0e2fb2c9e7b"
CACHE_SOURCE_BLOB = "0f696cc3ac6a4d63ac3cf071ef620a9a083b5a05"
OPT_SOURCE_BLOB = "e4ef510f099d665e2a69e1b45c2f7c0dfd8cf4b2"


def _git_blob_sha(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


@dataclass
class Intervention:
    model: str
    enabled: bool = False
    arm: str = "R0"
    reference_kernel: str = "native"  # switch to eager for model-quality R0
    prefix: int = 0
    suffix: int = 0
    window_id: str = ""
    executor: Any = None
    adc_policies: dict[int, Any] = field(default_factory=dict)
    fig16_response: Any = None
    capture: Callable[[int, Any], None] | None = None
    # Only active suffix-query keys are dispatched to CIM. Raw masked/evicted
    # keys stay unmodified and are never counted as executed in the trace.
    active_scores: dict[int, int] = field(default_factory=dict)
    source_calls: dict[int, int] = field(default_factory=dict)

    def validate(self) -> None:
        if self.model not in MODEL_REVISIONS or self.arm not in ("R0", "R1", "R2", "R3") or self.reference_kernel not in ("native", "eager"):
            raise ValueError("unsupported model or arm")
        if self.prefix <= 0 or self.suffix <= 0:
            raise ValueError("intervention requires positive prefix and suffix")
        if self.arm in ("R2", "R3") and not self.adc_policies:
            raise ValueError("model-specific ADC calibration required for R2/R3")
        if self.arm == "R3" and self.fig16_response is None:
            raise ValueError("R3 requires characterized Fig.16 response")


def active_prefix_bounds(prefix: int, query_index: int, window: int | None) -> tuple[int, int]:
    """Indices of *original prefix keys* actually visible to suffix query j.

    Query absolute position is prefix+j. A sliding window W includes that
    query: key position k is visible iff prefix+j-W < k <= prefix+j.
    """
    if prefix < 0 or query_index < 0 or (window is not None and window <= 0):
        raise ValueError("invalid attention geometry")
    return (0 if window is None else min(prefix, max(0, prefix + query_index - window + 1)), prefix)


def _effective_mask(mask: torch.Tensor | None, *, prefix: int, qlen: int,
                    klen: int, window: int | None, device: torch.device) -> torch.Tensor:
    """Reject causal-future and sliding-evicted keys, regardless of cache type."""
    qindex = torch.arange(qlen, device=device).view(1, 1, qlen, 1)
    kindex = torch.arange(klen, device=device).view(1, 1, 1, klen)
    active = kindex <= prefix + qindex
    if window is not None:
        active &= kindex >= prefix + qindex - window + 1
    if mask is not None:
        if mask.ndim != 4 or mask.shape[-2] != qlen or mask.shape[-1] < klen:
            raise ValueError("source attention mask does not cover all suffix queries")
        # HF 4.46.2 supplies a finite additive mask (0 or dtype min).
        active &= mask[..., :klen] == 0
    return active


def _unrepeat_k(repeated_k: torch.Tensor, ratio: int) -> torch.Tensor:
    if ratio < 1 or repeated_k.shape[1] % ratio:
        raise ValueError("invalid MHA/GQA head mapping")
    unique = repeated_k[:, ::ratio]
    # This is a numerical identity requirement: no inferred KV copies.
    if not torch.equal(repeated_k, unique.repeat_interleave(ratio, dim=1)):
        raise ValueError("SDPA K heads do not match source GQA repeat_kv")
    return unique


def _pad_physical_head(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sinangil backend has exactly two 64-row groups; OPT has 80 active rows.

    Zero extension keeps the per-group max-absolute quantizer unchanged for
    both active OPT groups. It is an *explicit two-tile physical mapping*,
    not an 80-row macro or an invented hardware implementation.
    """
    if q.shape[-1] != k.shape[-1] or q.shape[-1] not in (80, 128):
        raise ValueError("Sinangil backend supports OPT 80/128-d Mistral via 128 rows")
    padding = 128 - q.shape[-1]
    return (F.pad(q, (0, padding)), F.pad(k, (0, padding))) if padding else (q, k)


def _dispatch_backend(q: torch.Tensor, unique_k: torch.Tensor, intervention: Intervention,
                      layer: int, query_index: int) -> torch.Tensor:
    """Execute exact upstream Q4/K4 mapping on the *visible* prefix region."""
    if intervention.executor is None:
        raise ValueError("R1--R3 require original SinangilQKExecutor")
    from sram2llm.sinangil_qk import make_quant_state

    q, unique_k = _pad_physical_head(q, unique_k)
    state = make_quant_state(q, unique_k, query_bits=4)
    if intervention.capture is not None:
        intervention.capture(layer, state)
    window = f"{intervention.window_id}:q{query_index}"
    args = {"layer_id": str(layer), "window_id": window}
    arm = intervention.arm
    if arm == "R1":
        result = intervention.executor.execute_r1(state, **args)
    elif arm == "R2":
        result = intervention.executor.execute_r2(state, policy=intervention.adc_policies[layer], **args)
    elif arm == "R3":
        result = intervention.executor.execute_r3(
            state, policy=intervention.adc_policies[layer],
            response=intervention.fig16_response, **args,
        )
        if result.diagnostics.get("source_domain_violation_count", 0):
            raise ValueError("R3 pulse count exceeds digitized Fig.16 support")
    else:
        raise ValueError("R0 must delegate to the native SDPA kernel")
    expected = (q.shape[0], q.shape[1], q.shape[2], unique_k.shape[2])
    if tuple(result.raw_scores.shape) != expected or not torch.isfinite(result.raw_scores).all():
        raise ValueError("Sinangil backend returned incompatible/nonfinite QK scores")
    return result.raw_scores


def intervene_sdpa(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    *, native_sdpa: Callable, intervention: Intervention, layer: int,
    head_ratio: int, window: int | None, unscaled_query: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None, dropout_p: float = 0.0,
    is_causal: bool = False, scale: float | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Replace visible prefix scores; preserve model projection/RoPE/KV flow.

    R0-native qualifies exact source SDPA delegation; the published R0-eager
    follows the original SRAM2LLM reference arithmetic with FP32 softmax.
    R1--R3 retain that eager computation path and replace prefix QK scores.
    Source-supplied masks are reinforced with Mistral sliding-window limits.
    """
    intervention.validate()
    intervention.source_calls[layer] = intervention.source_calls.get(layer, 0) + 1
    args = {"attn_mask": attn_mask, "dropout_p": dropout_p,
            "is_causal": is_causal}
    if scale is not None:
        args["scale"] = scale
    args.update(kwargs)
    if intervention.arm == "R0" and intervention.reference_kernel == "native":
        baseline = native_sdpa(query, key, value, **args)
        if intervention.capture is not None:
            # Development calibration observes post-RoPE Q/K using R0 scores;
            # it cannot change any evaluated held-out logits.
            if query.shape[-2] != intervention.suffix or key.shape[-2] != intervention.prefix + intervention.suffix:
                raise ValueError("calibration source cache differs from frozen geometry")
            unique = _unrepeat_k(key, head_ratio)
            raw_q = query if unscaled_query is None else unscaled_query
            from sram2llm.sinangil_qk import make_quant_state
            for j in range(min(8, query.shape[-2])):
                low, high = active_prefix_bounds(intervention.prefix, j, window)
                if low < high:
                    qj, kj = _pad_physical_head(raw_q[..., j:j+1, :], unique[..., low:high, :])
                    intervention.capture(layer, make_quant_state(qj, kj, query_bits=4))
        return baseline
    if dropout_p != 0.0 or kwargs:
        raise ValueError("numerical-quality study requires eval dropout=0 and supported SDPA arguments")
    b, h, t, d = query.shape
    if b != 1 or t != intervention.suffix or key.shape != value.shape:
        raise ValueError("quality evaluation is scalar and requires matching K/V")
    if key.shape[-2] != intervention.prefix + intervention.suffix:
        raise ValueError("source KV cache length changed unexpectedly")
    if h % head_ratio or key.shape[1] != h:
        raise ValueError("non-native GQA head geometry")
    if unscaled_query is None:
        unscaled_query = query
    if unscaled_query.shape != query.shape:
        raise ValueError("unscaled model projection does not match SDPA Q")
    source_scaling = 1.0 / sqrt(d) if scale is None else scale
    backend_scaling = 1.0 / sqrt(d)  # OPT q_proj is unscaled here; source SDPA q was scaled.
    orig = torch.matmul(query, key.transpose(-2, -1)) * source_scaling
    merged = orig.clone()
    unique_k = _unrepeat_k(key, head_ratio)
    active_mask = _effective_mask(attn_mask, prefix=intervention.prefix,
                                   qlen=t, klen=key.shape[-2], window=window,
                                   device=query.device)
    if intervention.model == "mistral" and window != 4096:
        raise ValueError("Mistral-v0.1 requires its source 4096-token sliding window")
    if intervention.model == "opt" and window is not None:
        raise ValueError("OPT must use full causal attention")
    # One batched native call is valid if every prefix key is visible at every
    # suffix position. Otherwise dispatch per query with sliced physical K.
    full_prefix_visible = bool(active_mask[..., :intervention.prefix].all())
    if intervention.arm == "R0":
        if intervention.capture is not None:
            from sram2llm.sinangil_qk import make_quant_state
            for j in range(min(8, t)):
                low,high=active_prefix_bounds(intervention.prefix,j,window)
                if low<high:
                    qj,kj=_pad_physical_head(unscaled_query[...,j:j+1,:],unique_k[...,low:high,:])
                    intervention.capture(layer,make_quant_state(qj,kj,query_bits=4))
    elif full_prefix_visible:
        result = _dispatch_backend(unscaled_query, unique_k[..., :intervention.prefix, :],
                                   intervention, layer, 0).to(orig.dtype)
        merged[..., :intervention.prefix] = result * backend_scaling
        intervention.active_scores[layer] = intervention.active_scores.get(layer, 0) + b * h * t * intervention.prefix
    else:
        for j in range(t):
            low, high = active_prefix_bounds(intervention.prefix, j, window)
            if low == high:
                continue
            # Full causal/sliding mask may also contain padding. Sparse
            # padding within a prefix is rejected rather than silently charged.
            visible = active_mask[..., j:j+1, low:high]
            if not bool(visible.all()):
                raise ValueError("unexpected masked prefix keys inside active sliding interval")
            result = _dispatch_backend(unscaled_query[..., j:j+1, :],
                                       unique_k[..., low:high, :], intervention,
                                       layer, j).to(orig.dtype)
            merged[..., j:j+1, low:high] = result * backend_scaling
            intervention.active_scores[layer] = intervention.active_scores.get(layer, 0) + b * h * (high-low)
    # Preserve arbitrary model-provided additive masks and force physical
    # eviction even if an upstream cache implementation changes.
    if attn_mask is not None:
        merged += attn_mask[..., :key.shape[-2]]
    merged = merged.masked_fill(~active_mask, torch.finfo(merged.dtype).min)
    prob = F.softmax(merged, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(prob, value)


def install_model_adapter(model: Any, kind: str, controller: Intervention) -> None:
    """Wrap *existing* SDPA modules; preserve state dict, prefill and RoPE.

    Only one model may execute within the scope; intercepting a process-global
    torch function is deliberately restricted to synchronous inference.
    """
    import transformers
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError("Requires the archived Transformers==4.46.2 source")
    if kind == "mistral":
        from transformers.models.mistral import modeling_mistral
        if _git_blob_sha(modeling_mistral.__file__) != MISTRAL_SOURCE_BLOB:
            raise RuntimeError("Mistral source differs from archived source pin")
        if _git_blob_sha(transformers.cache_utils.__file__) != CACHE_SOURCE_BLOB:
            raise RuntimeError("DynamicCache source differs from archived source pin")
        base = modeling_mistral.MistralSdpaAttention
        layers = model.model.layers
        window = model.config.sliding_window
        assert window == 4096
        ratio = model.config.num_attention_heads // model.config.num_key_value_heads
    elif kind == "opt":
        from transformers.models.opt import modeling_opt
        if _git_blob_sha(modeling_opt.__file__) != OPT_SOURCE_BLOB:
            raise RuntimeError("OPT source differs from frozen Transformers 4.46.2 pin")
        base = modeling_opt.OPTSdpaAttention
        layers = model.model.decoder.layers
        window = None
        ratio = 1
        assert model.config.max_position_embeddings == 2048
    else:
        raise ValueError("unknown model architecture")
    controller.model = kind
    for layer_index, block in enumerate(layers):
        module = block.self_attn
        if not isinstance(module, base):
            raise TypeError("source model must instantiate SDPA attention")
        original = module.forward
        def wrapped(*args, _native_forward=original, _layer=layer_index,
                    _module=module, **kw):
            if not controller.enabled:
                return _native_forward(*args, **kw)
            controller.source_calls.pop(_layer, None)
            def patched(q, k, v, **sdpa_kw):
                # OPT's SDPA Q is pre-scaled. Obtain exact unscaled q_proj for
                # Q4/K4 quantization before applying native scaling afterwards.
                uq = None
                if kind == "opt" and (controller.arm != "R0" or controller.capture is not None):
                    hidden = args[0] if args else kw["hidden_states"]
                    uq = _module._shape(_module.q_proj(hidden), -1, hidden.shape[0])
                return intervene_sdpa(q, k, v, native_sdpa=native_sdpa,
                                      intervention=controller, layer=_layer,
                                      head_ratio=ratio, window=window,
                                      unscaled_query=uq, **sdpa_kw)
            native_sdpa = F.scaled_dot_product_attention
            with patch.object(F, "scaled_dot_product_attention", patched):
                result = _native_forward(*args, **kw)
            if controller.source_calls.get(_layer) != 1:
                raise RuntimeError("exactly one native SDPA call required per active attention layer")
            return result
        # No model weights altered. Instance-level forward wrapper is active
        # for both prefill and suffix, but R0 unchanged in attention math.
        module.forward = wrapped
