#!/usr/bin/env python3
"""Measure real CUDA kernel service of optimized Hydragen-style attention.

No CPU timings, static FLOP estimates, LLMCompass predictions or H100 results
are substituted. Runs an explicitly selected native GQA/MHA model geometry
and CUDA dtype, requiring numerical agreement with full SDPA and the existing
CPU functional oracle. Model activations default to random Q/K/V: these are
physical CUDA-kernel measurements, NOT pretrained full-model measurements.

Usage:
  python scripts/benchmark_gpu_attention.py --model llama --prefix 2048 \
    --suffix 128 --batch 1 --output-tokens 128 --steps all \
    --precision bf16 --out results/gpu/llama_b1.json

GPU-optimized candidates: grouped shared-prefix GEMM, and torch.compile
(reduce-overhead) + Triton/Inductor where available. Both receive one shared
K/V copy per KV-head; the suffix remains request-private. The full-SDPA
Flash kernel is the competitive full-attention reference; full cache tensors
are assembled before, never inside, the timed operation.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cim_shared_prefix.profiles import PROFILES, visible_prefix
from cim_shared_prefix.gpu_attention import (
    full_kv, full_sdpa, grouped_prefix_scores, completed_prefix_scores_gpu_remainder, make_candidate,
    shared_prefix_gemm_attention,
)
from cim_shared_prefix.hydragen_reference import prefix_suffix_attention

MODEL_IDS = {"llama": "meta-llama/Llama-3.1-8B",
             "mistral": "mistralai/Mistral-7B-v0.1",
             "opt": "facebook/opt-2.7b"}
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
SCHEMA = "cim-shared-prefix-real-CUDA-attention/v1"


def _hardware():
    result = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(dev)
        result.update(device_index=dev, gpu_name=p.name,
                      device_total_bytes=p.total_memory, compute_capability=f"{p.major}.{p.minor}")
        try:
            run = subprocess.run(["nvidia-smi", "--query-gpu=uuid,pci.bus_id,driver_version",
                                  "--format=csv,noheader"], capture_output=True, check=True,
                                 text=True, timeout=10)
            result["device_identifiers_raw"] = run.stdout.strip()
        except (FileNotFoundError, subprocess.SubprocessError):
            result["device_identifiers_raw"] = None
    return result


def _profiling_receipt(fn):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=False) as prof:
        fn()
        torch.cuda.synchronize()
    ev = prof.events()
    kernels = sorted({str(x.name) for x in ev
                      if "cuda" in str(getattr(x, "device_type", "")).lower()
                      or str(getattr(x, "device_type", "")).endswith(".CUDA")})
    cpu_ops = sorted({str(x.key) for x in ev if hasattr(x, "key")})
    return {"cuda_kernel_count": len(kernels),
            "cuda_kernel_names": kernels[:80],
            "cuda_kernel_names_truncated": len(kernels) > 80,
            "source": "PyTorch Kineto runtime profiler; actual CUDA execution",
            "flash_kernel_observed": any("flash" in x.lower() for x in kernels + cpu_ops),
            "grouped_gemm_observed": any(any(z in x.lower() for z in
                                           ("gemm", "bmm", "matmul", "cublas"))
                                         for x in kernels + cpu_ops)}


def _event_samples(fn, *, mode: str, warmup: int, trials: int, repeats: int):
    if mode == "graph":
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_result = fn()
        # Hold output and graph allocations alive throughout all timings.
        assert static_result is not None
        task = graph.replay
    else:
        task = fn
    for _ in range(warmup):
        task()
    torch.cuda.synchronize()
    vals = []
    for _ in range(trials):
        a = torch.cuda.Event(enable_timing=True)
        z = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(repeats):
            task()
        z.record()
        z.synchronize()
        elapsed_ms = a.elapsed_time(z)
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            raise RuntimeError("invalid real CUDA elapsed time")
        vals.append(elapsed_ms * 1000.0 / repeats)
    return {"round_us": vals, "median_us": statistics.median(vals),
            "min_us": min(vals), "max_us": max(vals),
            "event_span": f"{repeats} real CUDA executions per round",
            "graph_captured": mode == "graph"}


def run_case(profile, *, batch, prefix, suffix, output_tokens, step,
             precision, modes, candidates, warmup, trials, repeats,
             max_error_abs: float | None = None):
    if not 0 <= step < output_tokens:
        raise ValueError("token step outside output range")
    if prefix + suffix + output_tokens > profile.context_limit:
        raise ValueError("model context limit violated")
    p = visible_prefix(profile, prefix, suffix, step)
    if p <= 0:
        return {"token_step": step, "status": "NO_ACTIVE_SHARED_PREFIX",
                "active_prefix": p, "measurements": []}
    s = suffix + step + 1
    if profile.sliding_window is not None:
        s = min(s, profile.sliding_window)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(20260917 + 11 * step + 421 * batch + 23 * p)
    dtype = DTYPES[precision]
    hq, hk, d = profile.query_heads, profile.kv_heads, profile.head_dim
    def sample(shape):
        return torch.randn(shape, generator=gen, dtype=dtype, device="cuda") * 0.06
    q = sample((batch, hq, 1, d))
    pk, pv = sample((hk, p, d)), sample((hk, p, d))
    sk, sv = sample((batch, hk, s, d)), sample((batch, hk, s, d))
    k, v = full_kv(q, pk, pv, sk, sv)
    # Completed, unscaled 32-bit score words physically resident on GPU.
    # Score transfer from CIM must be charged by the architecture model.
    injected_completed_scores = grouped_prefix_scores(q, pk).float()
    expected = prefix_suffix_attention(q[:, :, 0], pk, pv, sk, sv)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        flash = full_sdpa(q, k, v)
        torch.cuda.synchronize()
    err_baseline = float((flash.float() - expected.float().unsqueeze(-2)).abs().max())
    tol = 0.035 if precision == "bf16" else 0.008
    if not torch.allclose(flash.float(), expected.float().unsqueeze(-2), atol=tol, rtol=tol):
        raise RuntimeError(f"FLASH_ORACLE_MISMATCH max_abs={err_baseline}, dtype={precision}")
    record = {
        "token_step": step, "active_prefix_tokens": p, "visible_suffix_tokens": s,
        "full_attention_tokens": p+s,
        "model_id": profile.model_id, "model_revision": profile.revision,
        "native_gqa_ratio": profile.gqa_ratio, "precision": precision,
        "batch": batch, "head_dim": d, "query_heads": hq, "kv_heads": hk,
        "reference_absolute_error": err_baseline,
        "completed_score_word_bits": 32,
        "GPU_remainder_input": "unscaled FP32 completed QK scores resident on GPU; physical output-link excluded",
        "source_data": "random tensor activations; actual GPU CUDA event kernels, NOT pretrained inference",
        "measurements": [], "status": "MEASURED_ATTENTION_KERNEL_ONLY",
    }
    mode_names = [("full_flash_sdpa", lambda: full_sdpa(q, k, v)),
                  ("prefix_qk_only", lambda: grouped_prefix_scores(q, pk)),
                  ("gpu_score_to_attention_remainder",
                   lambda: completed_prefix_scores_gpu_remainder(
                       q, injected_completed_scores, pv, sk, sv))]
    for name in candidates:
        if name == "compiled_grouped_gemm":
            mode_names.append((name, make_candidate(name)))
        elif name == "grouped_gemm":
            mode_names.append((name, shared_prefix_gemm_attention))
        else:
            raise ValueError("invalid candidate " + name)
    for mode in modes:
        for name, base_fn in mode_names:
            fn = base_fn if name in ("full_flash_sdpa", "prefix_qk_only", "gpu_score_to_attention_remainder") else (
                lambda call=base_fn: call(q, pk, pv, sk, sv))
            if name == "prefix_qk_only":
                endpoint = "prefix_completed_QK_scores_pre_scale_softmax"
            else:
                endpoint = "full_prefix_suffix_attention_output"
            if name == "full_flash_sdpa":
                def invoke(f=fn):
                    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                        return f()
                fn = invoke
            try:
                out = fn()
                torch.cuda.synchronize()
                if endpoint == "full_prefix_suffix_attention_output":
                    abs_error = float((out.float() - flash.float()).abs().max())
                    if not torch.allclose(out.float(), flash.float(), atol=tol, rtol=tol):
                        raise RuntimeError(f"NUMERICAL_MISMATCH abs={abs_error}")
                else:
                    abs_error = None
                receipt = _profiling_receipt(fn)
                if receipt["cuda_kernel_count"] == 0:
                    raise RuntimeError("GPU kernel evidence missing")
                if name == "full_flash_sdpa" and not receipt["flash_kernel_observed"]:
                    raise RuntimeError("FlashAttention kernel could not be independently verified")
                result = _event_samples(fn, mode=mode, warmup=warmup,
                                        trials=trials, repeats=repeats)
                record["measurements"].append({
                    "name": name, "endpoint": endpoint, "execution_mode": mode,
                    "status": "GPU_KERNEL_EVENT_MEASURED",
                    "max_absolute_error_vs_full_SDPA": abs_error,
                    **result, "kernel_proof": receipt,
                })
            except (RuntimeError, ValueError, OSError) as err:
                record["measurements"].append({
                    "name": name, "endpoint": endpoint, "execution_mode": mode,
                    "status": "UNAVAILABLE_OR_FAILED",
                    "reason": type(err).__name__ + ": " + str(err)[:1000],
                    "median_us": None,
                })
    record["complete_pairs"] = [
        {"execution_mode": mode,
         "native_full_flash_us": next((x["median_us"] for x in record["measurements"]
                                      if x["name"] == "full_flash_sdpa" and
                                      x["execution_mode"] == mode and
                                      x["status"] == "GPU_KERNEL_EVENT_MEASURED"), None),
         "optimized_hydragen_us": min((
             x["median_us"] for x in record["measurements"]
             if x["name"] in ("grouped_gemm", "compiled_grouped_gemm")
             and x["execution_mode"] == mode and
             x["status"] == "GPU_KERNEL_EVENT_MEASURED"
         ), default=None)}
        for mode in modes
    ]
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=MODEL_IDS)
    ap.add_argument("--prefix", type=int, default=2048)
    ap.add_argument("--suffix", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--steps", default="0,63,127", help="comma-separated decode indices or 'all'")
    ap.add_argument("--precision", required=True, choices=DTYPES)
    ap.add_argument("--candidates", default="grouped_gemm,compiled_grouped_gemm")
    ap.add_argument("--modes", default="graph", help="graph or direct, comma-separated")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.prefix <= 0 or a.suffix < 0 or a.batch < 1 or a.output_tokens < 1:
        ap.error("positive model geometry required")
    if min(a.warmup, a.trials, a.repeats) < 1:
        ap.error("warmups/trials/repeats must be positive")
    profile = next(p for p in PROFILES if p.model_id == MODEL_IDS[a.model])
    steps = list(range(a.output_tokens)) if a.steps == "all" else sorted(set(map(int, a.steps.split(","))))
    modes = [m.strip() for m in a.modes.split(",")]
    if not modes or any(m not in ("graph", "direct") for m in modes):
        ap.error("--modes must use graph/direct")
    names = [s.strip() for s in a.candidates.split(",") if s.strip()]
    if any(n not in ("grouped_gemm", "compiled_grouped_gemm") for n in names):
        ap.error("invalid candidate name")
    hw = _hardware()
    out = {"schema": SCHEMA, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "hardware": hw, "model_id": profile.model_id,
           "model_revision": profile.revision,
           "precision": a.precision,
           "requested": {"prefix": a.prefix, "suffix": a.suffix, "batch": a.batch,
                         "output_tokens": a.output_tokens, "steps": steps,
                         "trials": a.trials, "repeats": a.repeats},
           "activation_source": "shape-matched synthetic tensors (not pretrained model activations)",
           "numerical_reference": "hydragen_reference + forced flash SDPA",
           "source_provenance": "CUDA events and Kineto kernel names on actual device",
           "measurements": [], "full_decoder_tpot_s": None,
           "SRAM_vs_GPU_speedup": None, "CIM_vs_GPU_speedup": None}
    if not hw["cuda_available"]:
        out["status"] = "UNAVAILABLE_NO_CUDA_DEVICE"
    else:
        for step in steps:
            out["measurements"].append(run_case(
                profile, batch=a.batch, prefix=a.prefix, suffix=a.suffix,
                output_tokens=a.output_tokens, step=step, precision=a.precision,
                modes=modes, candidates=names, warmup=a.warmup,
                trials=a.trials, repeats=a.repeats))
        out["status"] = ("GPU_KERNEL_MEASUREMENTS_RECORDED"
                         if any(any(m.get("status") == "GPU_KERNEL_EVENT_MEASURED"
                                    for m in r.get("measurements", []))
                                for r in out["measurements"])
                         else "GPU_AVAILABLE_BUT_NO_QUALIFIED_KERNEL_RECORD")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, allow_nan=False, sort_keys=True) + "\n")
    print(json.dumps({"status": out["status"], "device": hw.get("gpu_name"),
                      "coordinates": len(out["measurements"]), "output": str(a.out)}))
    if out["status"] != "GPU_KERNEL_MEASUREMENTS_RECORDED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
