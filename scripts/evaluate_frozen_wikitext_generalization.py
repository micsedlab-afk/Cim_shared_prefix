#!/usr/bin/env python3
"""Frozen-policy WikiText-2 held-out qualification for the 135M pretrained pilot.

Real model, independent standard corpus, no quality retuning, CUDA/speedup
or ADC-silicon inference. Source comparison is fully teacher-forced.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from evaluate_pretrained_compact_prefix import (
    MODEL, REVISION, replace_attention_prefix, nll_and_logits,
)
from cim_shared_prefix.compact_prefix_cim import macro_summary_area

DATASET = "Salesforce/wikitext"
DATASET_REV = "b08601e"
DATASET_PATH = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
DATASET_SHA256 = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"
FROZEN_LAYERS = (12, 13, 21, 27)
P, O, M = 128, 32, 64
WINDOW_COUNT = 12
OFFSET_START = 1024
OFFSET_STRIDE = 4096
SEED = 20261105


def percentile_ci(x: np.ndarray, rng: np.random.Generator, repeats: int = 10000) -> list[float]:
    idx = rng.integers(0, len(x), size=(repeats, len(x)))
    return [float(q) for q in np.quantile(x[idx].mean(axis=1), [0.025, 0.975])]


def evaluate(args):
    import torch
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(22848)
    np.random.seed(22848)
    torch.set_num_threads(args.threads)

    dataset_file = Path(hf_hub_download(
        repo_id=DATASET, repo_type="dataset", filename=DATASET_PATH,
        revision=DATASET_REV))
    actual_sha = hashlib.sha256(dataset_file.read_bytes()).hexdigest()
    if actual_sha != DATASET_SHA256:
        raise RuntimeError("frozen WikiText parquet hash mismatch: " + actual_sha)
    t = pq.read_table(dataset_file, columns=["text"])
    corpus = "\n".join(str(x or "") for x in t.column("text").to_pylist())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float32,
        attn_implementation="eager", trust_remote_code=False).eval()
    if (model.config.num_hidden_layers, model.config.num_attention_heads,
            model.config.num_key_value_heads, model.config.hidden_size) != (30, 9, 3, 576):
        raise RuntimeError("frozen pretrained model geometry changed")
    ids = tokenizer.encode(corpus, add_special_tokens=False)
    offsets = [OFFSET_START+i*OFFSET_STRIDE for i in range(WINDOW_COUNT)]
    if offsets[-1]+P+O > len(ids):
        raise RuntimeError("insufficient corpus tokens for frozen nonoverlapping offsets")
    inputs = [torch.tensor([ids[x:x+P+O]], dtype=torch.long) for x in offsets]
    policies = {
        "exact": [0]*30,
        "strict_four": [M if i in FROZEN_LAYERS else 0 for i in range(30)],
        "first_four": [M if i<4 else 0 for i in range(30)],
    }
    results = []
    for idx, (offset, inp) in enumerate(zip(offsets, inputs)):
        losses = {}
        for name, selection in policies.items():
            with replace_attention_prefix(model, P, selection, mode="q4k4"):
                loss, _ = nll_and_logits(model, inp, P)
            if not math.isfinite(loss):
                raise RuntimeError(f"non-finite pretrained NLL: {idx}/{name}")
            losses[name] = float(loss)
        results.append({"window": idx, "token_offset": offset,
                        "targets": O-1, **losses,
                        "strict_delta_NLL": losses["strict_four"]-losses["exact"],
                        "first_delta_NLL": losses["first_four"]-losses["exact"]})
        print(f"HELDOUT_WINDOW {idx+1}/{WINDOW_COUNT} " +
              json.dumps(results[-1], sort_keys=True), flush=True)

    rng = np.random.default_rng(SEED)
    strict = np.array([r["strict_delta_NLL"] for r in results])
    first = np.array([r["first_delta_NLL"] for r in results])
    pair = strict - first
    def stats(x):
        return {"mean_delta_NLL": float(np.mean(x)),
                "mean_delta_NLL_window_bootstrap_95ci": percentile_ci(x, rng),
                "relative_PPL_pct_from_mean_delta_NLL": float(100*np.expm1(np.mean(x))),
                "windows_with_positive_delta_NLL": int(np.sum(x > 0))}
    macro = macro_summary_area(4, 3, 64, M)
    output = {
        "schema":"predeclared_wikitext2_test_135m_strict_vs_first4/v1",
        "status":"ACTUAL_PRETRAINED_CPU_INFERENCE",
        "checkpoint":MODEL, "revision":REVISION,
        "dataset":DATASET, "dataset_revision":DATASET_REV,
        "dataset_path":DATASET_PATH, "dataset_sha256":actual_sha,
        "evaluation":"teacher-forced causal post-prefix NLL, real weights",
        "prefix":P, "output_tokens":O, "heldout_targets":WINDOW_COUNT*(O-1),
        "window_count":WINDOW_COUNT, "window_offsets":offsets,
        "calibration":"frozen from 2026-11-05 independent authored domains; no WikiText calibration",
        "admitted_layers":list(FROZEN_LAYERS), "centroids":M,
        "arithmetic":"ideal signed Q4K4, FP16 centroid V and FP16 scales; no characterized O4 ADC",
        "exact_fallback":"all 26 unadmitted layers and exact per-request suffix",
        "macro_only_area_mm2":macro["signed_K_V_metadata_macro_area_mm2"],
        "results_by_window":results,
        "strict_four":stats(strict), "first_four":stats(first),
        "strict_minus_first":stats(pair),
        "uncertainty":"percentile bootstrap across 12 nonoverlapping text windows, descriptive only",
        "no_claims":["Llama-8B accuracy","GPU performance","silicon digital area","hardware O4 ADC accuracy",
                     "hardware speedup or energy","statistical independence of windows"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True)+"\n")
    print("FROZEN_WIKITEXT_QUALITY "+json.dumps({
        "mean_strict_delta_NLL":output["strict_four"]["mean_delta_NLL"],
        "mean_first_delta_NLL":output["first_four"]["mean_delta_NLL"],
        "strict_minus_first":output["strict_minus_first"],
        "targets":output["heldout_targets"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--threads",type=int,default=4)
    evaluate(p.parse_args())
