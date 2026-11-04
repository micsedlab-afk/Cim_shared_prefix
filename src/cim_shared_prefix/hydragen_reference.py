"""Hydragen-style *functional* exact shared-prefix attention reference.

Flattening batch x GQA query groups into a shared-prefix GEMM is the competitive
GPU baseline algorithm. Separate suffix attention is fused using logsumexp.
This CPU-capable reference qualifies math and shapes only; GPU kernel timing and
full-decoder TPOT require real matched GPU records.
"""
from __future__ import annotations

from math import sqrt


def prefix_suffix_attention(q, prefix_k, prefix_v, suffix_k, suffix_v, *, scale=None):
    """Return exact fused attention for one decoding position per request.

    q:          [batch, query_heads, head_dim]
    prefix K/V: [kv_heads, shared_prefix_tokens, head_dim]
    suffix K/V: [batch, kv_heads, unique_tokens, head_dim]
    GQA is native: one K/V group serves query_heads/kv_heads query heads.
    The caller supplies only causally *visible* keys (including sliding window).
    """
    import torch

    if q.ndim != 3 or prefix_k.ndim != 3 or prefix_v.ndim != 3 or suffix_k.ndim != 4 or suffix_v.ndim != 4:
        raise ValueError("invalid query/prefix/suffix ranks")
    b, hq, d = q.shape
    hk, p, dk = prefix_k.shape
    if hq % hk or hk <= 0 or d != dk or prefix_v.shape != prefix_k.shape:
        raise ValueError("invalid native MHA/GQA shared-prefix geometry")
    if suffix_k.shape[:2] != (b, hk) or suffix_k.shape[-1] != d or suffix_v.shape != suffix_k.shape:
        raise ValueError("invalid batch-distinct suffix geometry")
    if p == 0 and suffix_k.shape[2] == 0:
        raise ValueError("at least one causal key is required")
    if not all(t.device == q.device for t in (prefix_k, prefix_v, suffix_k, suffix_v)):
        raise ValueError("attention tensors must share the same device")

    # Float32 accumulation for the functional reference; numerical GPU precision
    # must be separately reported for measured BF16/FP16 kernels.
    bf = q.float().reshape(b, hk, hq // hk, d)
    kp, vp = prefix_k.float(), prefix_v.float()
    ks, vs = suffix_k.float(), suffix_v.float()
    alpha = 1.0 / sqrt(d) if scale is None else float(scale)
    if alpha <= 0:
        raise ValueError("positive scale required")

    if p:
        # Shared-prefix computation is a grouped GEMM over B x GQA queries.
        scores_p = torch.einsum("bhgd,hpd->bhgp", bf, kp) * alpha
        lse_p = torch.logsumexp(scores_p, dim=-1)
        output_p = torch.einsum("bhgp,hpd->bhgd",
                                torch.softmax(scores_p, dim=-1), vp)
    else:
        lse_p = torch.full(bf.shape[:-1], float("-inf"), dtype=bf.dtype, device=bf.device)
        output_p = torch.zeros_like(bf)

    if suffix_k.shape[2]:
        scores_s = torch.einsum("bhgd,bhsd->bhgs", bf, ks) * alpha
        lse_s = torch.logsumexp(scores_s, dim=-1)
        output_s = torch.einsum("bhgs,bhsd->bhgd",
                                torch.softmax(scores_s, dim=-1), vs)
    else:
        lse_s = torch.full(bf.shape[:-1], float("-inf"), dtype=bf.dtype, device=bf.device)
        output_s = torch.zeros_like(bf)

    full_lse = torch.logaddexp(lse_p, lse_s)
    fused = (torch.exp(lse_p-full_lse)[..., None] * output_p +
             torch.exp(lse_s-full_lse)[..., None] * output_s)
    return fused.reshape(b, hq, d).to(dtype=q.dtype)
