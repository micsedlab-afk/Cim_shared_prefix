# Frozen shared-prefix compression qualification — 5 November 2026

## Fixed scientific conditions

- Preserve the published 7-nm Sinangil signed-K4 CIM and Yokoyama FP16-V SRAM macro geometries. Do not introduce new 7-nm synthesis or fictitious peripheral/MAC areas.
- **Primary comparator:** MAGMA/LLMCompass A100 80GB FP16 HBM2e source model, preserving the original 7-nm architectural study. No A100 CUDA kernel measurements are available in this execution environment. H100 acquisition evidence, where historically available, remains separate and must never replace or be relabeled as the primary A100 comparator.
- No assumed GPU kernel timing, silicon throughput, energy or end-to-end speedup. Missing measurements remain null.
- A complete comparator must include **exact optimized GPU**, **same-quality compressed GPU**, and **compressed CIM+GPU**, each with the same model, decoder endpoint and batch/precision.
- Frozen calibration layers cannot be retuned using either held-out WikiText test split. No statistical claims across 8B from 135M.

## New actual pretrained CPU inference

Checkpoint: HuggingFaceTB/SmolLM2-135M at 93efa2f097d58c2a74874c7e644dbc9b0cee75a2. Architecture 30 layers / GQA 9Q:3KV / head_dim64. FP32 base, physically rounded FP16 centroid V and scales; ideal integer Q4K4 without characterized O4 ADC. Exact suffix and 26-layer fallback. Frozen strict subset [12,13,21,27], compared at identical 4 compressed layers with fixed first four.

Data: independent Salesforce/wikitext WikiText-2 raw test parquet, revision b08601e, SHA256 5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91. Predeclared nonoverlapping test windows; no selection or optimization on held-out data.

| P / M | Distinct windows / targets | Frozen selected ΔNLL | Fixed first four ΔNLL | Frozen - fixed ΔNLL; 95% window-bootstrap CI | Source |
|---|---|---:|---:|---|---|
| 128 / 64 (2×) | 12 / 372 | +0.001941333 | +0.014471898 | -0.012530565 [-0.053542758,+0.028198135] | [successful workflow 42006630402](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/42006630402) |
| 2048 / 64 (32×) | 6 / 186 | +0.055205325 | +0.095678826 | -0.040473501 [-0.110155066,+0.032628616] | [successful workflow 42007059366](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/42007059366) |

The P2048 result is +5.67% PPL for strict-four and +10.04% for first-four. Both confidence intervals on **strict-minus-fixed** include zero; no statistically established quality improvement. The windows differ between P128 and P2048, so these cannot support a paired causal conclusion about the compression ratio. The strict subset was **calibrated at P128** and frozen when tested at P2048.

Immutable row-level evidence is in `results/quality/frozen_wikitext2_P128_M64_20261105.json` and `results/quality/frozen_wikitext2_P2048_M64_20261105.json`; reproducible scripts/workflows are included. The 135M results do not qualify full-model Llama8B quality or characterized ADC effects.

## Source-macro geometry and decisive GPU fairness issue

At native Llama-3.1-8B P2048, 32 layers, 8 KV heads, D128, all 32 layers retained:
- 64 representatives (32×): 4096 signed K CIM macros, 128 V SRAM macros, 2 metadata macros; **14.274279 mm²** macro-only.
- 80 representatives: **17.847338 mm²**, exceeding 16 mm² before digital/controller area.
- 1024 representatives (2×): **228.388471 mm²** macro-only.

Thus extrapolating the 135M P128/M64 (2×) results to 8B full-residency P2048/M64 (32×) is not valid.

Original Llama-8B dense FP16 shared-prefix K/V at P2048 contains 256 MiB. A GPU implementation of the **same compressed representation** at M64 would contain **8 MiB FP16 K/V**, or **5.0625 MiB** with packed signed K4 / FP16 V plus 4-byte-per-representative metadata. These payloads fit within the **A100 nominal 40 MB L2** capacity when all 32 layers are compressed; cache persistence under model weights and decoder execution is unmeasured. For the actually tested **four compressed layers**, the remaining 28 exact-prefix layers contain an additional 224 MiB FP16 K/V. The resulting 225 MiB hybrid KV representation does **not** fit in nominal A100 L2. The all-layer 8 MiB statement is purely hypothetical until 32-layer Llama-8B quality is measured. A100 device specification: https://www.nvidia.com/en-us/data-center/a100/.

The attention-quality matched compressed GPU must therefore be measured, together with exact Hydragen/Flash and physically supported CIM peripheral rates. No real GPU performance or system-speedup result was generated in the CPU-only run.

## Decision

Keep hardware feasibility/quality admission as a research direction. It currently qualifies storage capacity and small-model numerical behavior only. A publication-quality accelerator advantage requires true 8B inference at 32×, quantified O4 ADC loss, and a **primary A100 FP16** comparator covering optimized exact and compressed attention at the same output endpoint; the source-only model cannot stand in for measured kernels. Matched signed Q4K4 execution is required for a numerical-fidelity claim; an FP16-centroid GPU source timing alone is a different precision path. Stop hypothetical deadline sweeps, fabricated digital iso-area and cross-model GPU timing substitutions.
