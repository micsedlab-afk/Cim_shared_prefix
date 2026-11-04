"""Real CUDA shared-prefix attention candidates and strict endpoint definitions.

The existing hydragen_reference is the numerical oracle. These functions
execute full attention QK/softmax/PV and return the same [B,Hq,1,D] tensor.
The grouped prefix GEMMs reuse one [Hkv,P,D] K/V across the B*GQA queries;
suffix K/V remain batch-private. No K/V is duplicated for GQA.

This file deliberately contains NO latency constants. Timing only comes from
CUDA Events in scripts/benchmark_gpu_attention.py. CUDA graph/torch.compile
improve execution only if supported and measured on an actual CUDA device.
"""
from __future__ import annotations
from math import sqrt
from functools import lru_cache

import torch
from torch.nn import functional as F


def check_inputs(q, pk, pv, sk, sv):
    if not all(isinstance(t, torch.Tensor) for t in (q, pk, pv, sk, sv)):
        raise TypeError("five PyTorch tensors required")
    if q.ndim != 4 or q.shape[-2] != 1 or pk.ndim != 3 or pv.ndim != 3 or sk.ndim != 4 or sv.ndim != 4:
        raise ValueError("expected Q[B,Hq,1,D], shared prefix K/V[Hkv,P,D], suffix K/V[B,Hkv,U,D]")
    b, hq, _, d = q.shape
    hk, p, kd = pk.shape
    if hk < 1 or hq % hk or kd != d or pv.shape != pk.shape:
        raise ValueError("native query/KV head ratio and prefix shape mismatch")
    if sk.shape[:2] != (b, hk) or sk.shape[-1] != d or sv.shape != sk.shape:
        raise ValueError("batch-private suffix shape mismatch")
    if p == 0 and sk.shape[-2] == 0:
        raise ValueError("no causal keys")
    if len({t.dtype for t in (q, pk, pv, sk, sv)}) != 1 or len({t.device for t in (q, pk, pv, sk, sv)}) != 1:
        raise ValueError("all inputs must have identical dtype and device")
    if not q.dtype.is_floating_point:
        raise ValueError("floating-point attention precision required")
    return b, hq, hk, d, p, sk.shape[-2]


def full_kv(q, pk, pv, sk, sv):
    """Materialize a contiguous full-cache reference OUTSIDE the timed kernel."""
    b, hq, hk, d, p, u = check_inputs(q, pk, pv, sk, sv)
    k = torch.cat((pk.unsqueeze(0).expand(b, -1, -1, -1), sk), dim=-2).contiguous()
    v = torch.cat((pv.unsqueeze(0).expand(b, -1, -1, -1), sv), dim=-2).contiguous()
    return k, v


def full_sdpa(q, k, v):
    """Native PyTorch SDPA, FlashAttention kernel enforced separately by runner."""
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape or k.shape[0] != q.shape[0]:
        raise ValueError("incompatible full-attention tensors")
    if q.shape[1] % k.shape[1]:
        raise ValueError("invalid GQA ratio")
    return F.scaled_dot_product_attention(q, k, v, is_causal=False,
                                           dropout_p=0.0, enable_gqa=q.shape[1] != k.shape[1])


def grouped_prefix_scores(q, pk):
    """Q4-independent GPU BF16/FP16 QK kernel, one batched GEMM per KV head."""
    b, hq, one, d = q.shape
    hk, p, dim = pk.shape
    if one != 1 or dim != d or hq % hk:
        raise ValueError("invalid grouped prefix QK")
    ratio = hq // hk
    # B*GQA query rows share one K copy per physical KV head.
    qs = q[:, :, 0].reshape(b, hk, ratio, d).permute(1, 0, 2, 3)
    qs = qs.reshape(hk, b * ratio, d).contiguous()
    scores = torch.bmm(qs, pk.transpose(-1, -2))
    return scores.reshape(hk, b, ratio, p).permute(1, 0, 2, 3).reshape(b, hq, p)


def shared_prefix_gemm_attention(q, pk, pv, sk, sv):
    """Shared-prefix GEMM + exact separate suffix attention and logsumexp merge.

    All softmax reductions use FP32. QK and PV GEMMs run at the input dtype.
    This is a genuine GPU candidate suitable for torch.compile/Triton fusion;
    it is NOT classified as faster than SDPA unless the benchmark proves it.
    """
    b, hq, hk, d, p, u = check_inputs(q, pk, pv, sk, sv)
    g = hq // hk
    qs = q[:, :, 0].reshape(b, hk, g, d).permute(1, 0, 2, 3)
    qs = qs.reshape(hk, b * g, d).contiguous()
    qs_suffix = q[:, :, 0].reshape(b * hk, g, d).contiguous()
    scale = 1.0 / sqrt(d)

    if p:
        ps = torch.bmm(qs, pk.transpose(-1, -2)).float() * scale
        plse = torch.logsumexp(ps, dim=-1).reshape(hk, b, g).permute(1, 0, 2)
        pp = torch.softmax(ps, dim=-1).to(dtype=q.dtype)
        pout = torch.bmm(pp, pv).reshape(hk, b, g, d).permute(1, 0, 2, 3).float()
    else:
        plse = torch.full((b, hk, g), float("-inf"), dtype=torch.float32, device=q.device)
        pout = torch.zeros((b, hk, g, d), dtype=torch.float32, device=q.device)

    if u:
        ss = torch.bmm(qs_suffix, sk.reshape(b * hk, u, d).transpose(-1, -2)).float() * scale
        slse = torch.logsumexp(ss, dim=-1).reshape(b, hk, g)
        sp = torch.softmax(ss, dim=-1).to(dtype=q.dtype)
        sout = torch.bmm(sp, sv.reshape(b * hk, u, d)).reshape(b, hk, g, d).float()
    else:
        slse = torch.full((b, hk, g), float("-inf"), dtype=torch.float32, device=q.device)
        sout = torch.zeros((b, hk, g, d), dtype=torch.float32, device=q.device)

    total = torch.logaddexp(plse, slse)
    mixed = torch.exp(plse - total).unsqueeze(-1) * pout + torch.exp(slse - total).unsqueeze(-1) * sout
    return mixed.reshape(b, hq, 1, d).to(q.dtype)


@lru_cache(maxsize=2)
def make_candidate(name: str):
    if name == "grouped_gemm":
        return shared_prefix_gemm_attention
    if name == "compiled_grouped_gemm":
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile unavailable")
        # Tensor geometry is static at one benchmarked decode position.
        return torch.compile(shared_prefix_gemm_attention, mode="default",
                             fullgraph=True, dynamic=True)
    raise ValueError("unknown candidate")


def completed_prefix_scores_gpu_remainder(q, completed_prefix_qk, pv, sk, sv):
    """GPU suffix QK + completed-prefix softmax/PV + logsumexp fusion.

    The supplied prefix scores are unscaled completed FP32 transfer words
    ALREADY GPU-resident; transfer and CIM time are separate physical stages.
    """
    b, hq, hk, d, p, u = check_inputs(q, pv, pv, sk, sv)
    if completed_prefix_qk.shape != (b, hq, p) or completed_prefix_qk.dtype != torch.float32:
        raise ValueError("completed QK must be FP32 unscaled GPU-resident words")
    if completed_prefix_qk.device != q.device:
        raise ValueError("score GPU residency mismatch")
    g = hq // hk
    alpha = 1.0 / sqrt(d)
    if p:
        ps = (completed_prefix_qk.reshape(b, hk, g, p)
              .permute(1, 0, 2, 3).reshape(hk, b*g, p) * alpha)
        plse = torch.logsumexp(ps, dim=-1).reshape(hk, b, g).permute(1, 0, 2)
        pp = torch.softmax(ps, dim=-1).to(q.dtype)
        pout = torch.bmm(pp, pv).reshape(hk, b, g, d).permute(1, 0, 2, 3).float()
    else:
        plse = torch.full((b,hk,g), float("-inf"), dtype=torch.float32, device=q.device)
        pout = torch.zeros((b,hk,g,d), dtype=torch.float32, device=q.device)
    if u:
        qg = q[:, :, 0].reshape(b*hk, g, d).contiguous()
        ss = torch.bmm(qg, sk.reshape(b*hk,u,d).transpose(-1,-2)).float() * alpha
        slse = torch.logsumexp(ss, dim=-1).reshape(b,hk,g)
        sp = torch.softmax(ss, dim=-1).to(q.dtype)
        sout = torch.bmm(sp, sv.reshape(b*hk,u,d)).reshape(b,hk,g,d).float()
    else:
        slse = torch.full((b,hk,g), float("-inf"), dtype=torch.float32, device=q.device)
        sout = torch.zeros((b,hk,g,d), dtype=torch.float32, device=q.device)
    denom = torch.logaddexp(plse, slse)
    return (torch.exp(plse-denom).unsqueeze(-1)*pout +
            torch.exp(slse-denom).unsqueeze(-1)*sout).reshape(b,hq,1,d).to(q.dtype)
