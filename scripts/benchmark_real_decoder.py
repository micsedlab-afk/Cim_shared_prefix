#!/usr/bin/env python3
"""Measure real pretrained Hugging Face decoder CUDA timings with source geometry.

This is a *native full-model GPU* acquisition, separate from the controlled
attention-kernel comparison. Requires pinned model weights, a CUDA GPU and
Transformers 4.46.2. Reports true CUDA Event per-token decode time; optional
instrumented leaf-operator timing is obtained on a SEPARATE forward, never
silently summed to claim exact uninstrumented TPOT.

The cached prefill is excluded from decoder TPOT. Each measured output step
uses a real KV cache, greedy actual-model generated tokens and the exact
prefix+suffix+step context; 128 output steps means 128 post-prefill decode
executions. The HF baseline is conventional (not Hydragen), and cannot
substitute for the optimized Hydragen GPU baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cim_shared_prefix.profiles import PROFILES
from cim_shared_prefix.gpu_attention import full_sdpa

NAMES = {"llama": "meta-llama/Llama-3.1-8B",
         "mistral": "mistralai/Mistral-7B-v0.1",
         "opt": "facebook/opt-2.7b"}
EXTERNAL_MISTRAL_H100_REV = "27d67f1b5f57dc0953326b2601d68371d40ea8da"
NATIVE_TRANSFORMERS = "4.46.2"


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fd:
        for block in iter(lambda: fd.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_exact_snapshot(path, model_id, revision):
    """Require exact HF snapshot name plus matching config, no implicit retag."""
    snap = Path(path).resolve()
    if not snap.is_dir() or snap.name != revision:
        raise RuntimeError("SNAPSHOT_NOT_EXACT_PIN: expected snapshot directory named " + revision)
    if not (snap/"config.json").is_file():
        raise RuntimeError("MODEL_CONFIG_MISSING")
    config = json.loads((snap/"config.json").read_text())
    if config.get("_name_or_path") not in (None, model_id) and not str(config.get("_name_or_path")).endswith(model_id):
        # _name_or_path can be a local string; a mismatch is a provenance issue.
        raise RuntimeError("MODEL_ID_MISMATCH")
    shards = sorted(snap.glob("*.safetensors"))
    if not shards:
        raise RuntimeError("REAL_SAFETENSORS_WEIGHTS_MISSING")
    return {"source_path": str(snap), "revision": revision,
            "config_sha256": _sha256(snap/"config.json"),
            "weights_sha256": {p.name: _sha256(p) for p in shards},
            "weight_source": "local full pretrained revision-pinned safetensors"}


def check_config(config, profile):
    if config.num_hidden_layers != profile.layers or config.num_attention_heads != profile.query_heads:
        raise RuntimeError("decoder layer/query geometry mismatch")
    hk = getattr(config, "num_key_value_heads", config.num_attention_heads)
    if hk != profile.kv_heads or config.hidden_size != profile.hidden_size:
        raise RuntimeError("native GQA/MHA geometry mismatch")
    dim = (getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)
    if dim != profile.head_dim:
        raise RuntimeError("native head dim mismatch")
    if int(config.max_position_embeddings) != profile.context_limit:
        raise RuntimeError("model context differs from source profile")
    if profile.sliding_window is not None and config.sliding_window != profile.sliding_window:
        raise RuntimeError("Mistral 4096 SWA model mismatch")
    return {"layers": config.num_hidden_layers, "query_heads": config.num_attention_heads,
            "kv_heads": hk, "head_dim": dim,
            "sliding_window": profile.sliding_window, "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "intermediate_size": getattr(config, "intermediate_size", None)}


def make_inputs(torch, vocab, *, p, u, b, seed, path=None):
    if path:
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict) or len(raw.get("prefix_ids", [])) != p or len(raw.get("suffix_ids", [])) != b:
            raise ValueError("input token fixture differs from requested batch/prefix geometry")
        if any(len(x) != u for x in raw["suffix_ids"]):
            raise ValueError("input suffix lengths differ")
        token_rows = [raw["prefix_ids"] + suffix for suffix in raw["suffix_ids"]]
        source = "provided exact token ids"
    else:
        rng = torch.Generator().manual_seed(seed)
        prefix = torch.randint(10, vocab, (p,), generator=rng).tolist()
        suffix = torch.randint(10, vocab, (b, u), generator=rng).tolist()
        token_rows = [prefix + row for row in suffix]
        source = "deterministic token IDs on real pretrained weights; synthetic text stimulus"
    ids = torch.tensor(token_rows, dtype=torch.long, device="cuda:0")
    return ids, source, hashlib.sha256(ids.cpu().numpy().astype("<i8").tobytes()).hexdigest()


def _gpu_identification(torch):
    p = torch.cuda.get_device_properties(0)
    out = {"name": p.name, "cc": f"{p.major}.{p.minor}",
           "memory_bytes": p.total_memory, "torch": torch.__version__,
           "cuda": torch.version.cuda}
    try:
        x = subprocess.run(["nvidia-smi", "--query-gpu=uuid,pci.bus_id,driver_version",
                            "--format=csv,noheader"], text=True,
                           capture_output=True, timeout=10, check=True)
        out["physical_identifiers"] = x.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out["physical_identifiers"] = None
    return out


def _leaf_targets(model):
    import torch
    specs = []
    for name, mod in model.named_modules():
        if not name:
            continue
        kind = type(mod).__name__.lower()
        if (isinstance(mod, torch.nn.Linear) or isinstance(mod, torch.nn.LayerNorm)
                or any(z in kind for z in ("rmsnorm", "rotaryembedding", "silu"))):
            specs.append((name, mod))
    return specs


def _instrument_one(model, prompt, *, mode, prefix, suffix, torch):
    """Separate hooked CUDA Event and Kineto probe for one cached token."""
    from torch.profiler import profile as profiler, ProfilerActivity
    # Instrumented probe is never included in production uninstrumented TPOT.
    timings = defaultdict(list)
    handles = []
    starts = {}
    for name, mod in _leaf_targets(model):
        def hook_pre(module, args, key=name):
            begin = torch.cuda.Event(enable_timing=True)
            begin.record()
            starts[key] = begin
        def hook_post(module, args, out, key=name):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            timings[key].append((starts.pop(key), end))
        handles.append(mod.register_forward_pre_hook(hook_pre))
        handles.append(mod.register_forward_hook(hook_post))
    try:
        with torch.inference_mode():
            warmed = model(input_ids=prompt, use_cache=True, return_dict=True)
            next_ids = warmed.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            # Only measure leaf hooks during decode; remove prefill records.
            timings.clear()
            with profiler(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                          record_shapes=True) as prof:
                result = model(input_ids=next_ids, past_key_values=warmed.past_key_values,
                               use_cache=True, return_dict=True)
                torch.cuda.synchronize()
        events = {name: [float(start.elapsed_time(end)) for start, end in times]
                  for name, times in timings.items()}
        kernel_names = sorted({str(e.name) for e in prof.events()
                               if "CUDA" in str(getattr(e, "device_type", ""))})
        return {
            "status": "ACTUAL_PRETRAINED_WEIGHT_DECODE_OPERATOR_HOOKED",
            "leaf_module_gpu_event_ms": events,
            "leaf_operator_calls": sum(len(v) for v in events.values()),
            "kernel_names": kernel_names[:100],
            "kernel_names_truncated": len(kernel_names)>100,
            "flash_kernel_observed": any("flash" in n.lower() for n in kernel_names),
            "limitation": "hook instrumentation overhead; leaf module service only; functional RoPE/KV/softmax/PV and sampling may not have standalone module boundary",
            "incomplete_gpu_stage_timing": True,
        }
    finally:
        for h in handles:
            h.remove()


def acquire(a):
    import torch
    if not torch.cuda.is_available():
        return {"status":"UNAVAILABLE_NO_CUDA_GPU", "full_decoder_tpot_s":None,
                "actual_output_steps":0}
    import transformers
    if transformers.__version__ != NATIVE_TRANSFORMERS:
        raise RuntimeError(f"requires frozen Transformers {NATIVE_TRANSFORMERS}, got {transformers.__version__}")
    from transformers import AutoModelForCausalLM
    profile = next(x for x in PROFILES if x.model_id == NAMES[a.model])
    rev = profile.revision
    if a.mistral_h100_acquisition_revision:
        if a.model != "mistral":
            raise ValueError("H100 alternative revision only for Mistral")
        rev = EXTERNAL_MISTRAL_H100_REV
    if a.prefix+a.suffix+a.output_tokens>profile.context_limit:
        raise ValueError("input+output exceeds model context limit")
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    precision = {"bf16": torch.bfloat16, "fp16":torch.float16}[a.precision]
    if precision == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable on allocated CUDA GPU")
    evidence = read_exact_snapshot(a.model_root, profile.model_id, rev)
    model = AutoModelForCausalLM.from_pretrained(
        str(a.model_root), torch_dtype=precision,
        attn_implementation="sdpa", local_files_only=True,
        low_cpu_mem_usage=True).to("cuda:0").eval()
    if model.config._attn_implementation != "sdpa":
        raise RuntimeError("native SDPA attention implementation required")
    shape = check_config(model.config, profile)
    inputs, stimulus_source, stimulus_sha = make_inputs(
        torch, model.config.vocab_size, p=a.prefix, u=a.suffix,
        b=a.batch, seed=a.seed, path=a.tokens_json)
    hw = _gpu_identification(torch)
    rounds = []
    for _ in range(a.rounds):
        with torch.inference_mode():
            pre = model(input_ids=inputs, use_cache=True, return_dict=True)
            next_ids=pre.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            starts=[]; ends=[]
            out=pre
            for step in range(a.output_tokens):
                start=torch.cuda.Event(enable_timing=True)
                stop=torch.cuda.Event(enable_timing=True)
                start.record()
                out=model(input_ids=next_ids, past_key_values=out.past_key_values,
                          use_cache=True, return_dict=True)
                stop.record()
                starts.append(start); ends.append(stop)
                next_ids=out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
        ms=[float(st.elapsed_time(en)) for st,en in zip(starts,ends)]
        if len(ms)!=a.output_tokens or any(not math.isfinite(x) or x<=0 for x in ms):
            raise RuntimeError("invalid CUDA kernel timeline")
        rounds.append({"gpu_event_ms_per_decode_step":ms,
                       "gpu_decode_total_ms":sum(ms)})
    per_token=[statistics.median([r["gpu_event_ms_per_decode_step"][i] for r in rounds])
               for i in range(a.output_tokens)]
    probe = _instrument_one(model, inputs, mode="sdpa", prefix=a.prefix,
                            suffix=a.suffix, torch=torch) if a.profile_modules else None
    return {
        "status": "MEASURED_PRETRAINED_GPU_DECODER",
        "model_id": profile.model_id, "weight_revision": rev,
        "profile_primary_revision": profile.revision,
        "alternative_h100_acquisition_revision": rev!=profile.revision,
        "operator_precision": a.precision,
        "device": hw, "source_weights": evidence,
        "source_model_geometry": shape,
        "stimulus": stimulus_source, "stimulus_token_sha256": stimulus_sha,
        "prefix_shared_across_batch":True,
        "real_autoregressive_KV":True,
        "context_first_step":a.prefix+a.suffix+1,
        "context_last_step":a.prefix+a.suffix+a.output_tokens,
        "batch":a.batch, "prefix":a.prefix, "suffix":a.suffix,
        "output_steps":a.output_tokens,
        "rounds":rounds, "per_step_median_gpu_ms":per_token,
        "median_round_decode_total_ms": statistics.median(x["gpu_decode_total_ms"] for x in rounds),
        "mean_post_prefill_gpu_tpot_s":statistics.median(x["gpu_decode_total_ms"] for x in rounds)/(a.output_tokens*1000.0),
        "instrumented_operator_probe":probe,
        "attention_reference": "unoptimized native full-cache HF SDPA; NOT optimized Hydragen",
        "GPU_SRAM_CIM_tpot_or_speedup":None,
        "precision_qualification": "actual weights and exact dtype; alternative model revision separately tagged",
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",required=True,choices=NAMES)
    ap.add_argument("--model-root",type=Path,required=True)
    ap.add_argument("--precision",choices=("fp16","bf16"),required=True)
    ap.add_argument("--prefix",type=int,default=2048)
    ap.add_argument("--suffix",type=int,default=128)
    ap.add_argument("--batch",type=int,default=1)
    ap.add_argument("--output-tokens",type=int,default=128)
    ap.add_argument("--rounds",type=int,default=3)
    ap.add_argument("--seed",type=int,default=20260917)
    ap.add_argument("--tokens-json",type=Path)
    ap.add_argument("--profile-modules",action="store_true")
    ap.add_argument("--mistral-h100-acquisition-revision",action="store_true")
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    if min(a.prefix,a.batch,a.output_tokens,a.rounds)<1 or a.suffix<0:
        ap.error("positive geometry/rounds required")
    payload = {"schema":"cim-shared-prefix-real-model-GPU-decoder/v1",
               "acquired_utc":datetime.now(timezone.utc).isoformat(),
               "requested_model":NAMES[a.model], "requested_precision":a.precision,
               "results":acquire(a)}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(payload,indent=2,allow_nan=False,sort_keys=True)+"\n")
    print(json.dumps({"status":payload["results"]["status"],"output":str(a.out)}))
    if payload["results"]["status"]!="MEASURED_PRETRAINED_GPU_DECODER":
        raise SystemExit(2)


if __name__=="__main__":
    main()
