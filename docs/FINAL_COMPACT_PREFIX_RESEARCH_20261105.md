# Completed compressed shared-prefix CIM study — 5 November 2026

## Conclusion

**A physically mapped compact-prefix architecture is implemented and tested on actual pretrained weights, but an accelerator-speedup claim is not yet supported.** The most reliable result is a **strictly quality-gated layer admission** policy that reduces held-out NLL damage substantially at equal offloaded-layer count compared with a fixed first-layer policy. Uniform prefix compression and attention-output-error-only allocation were rejected by direct pretrained-model experiments.

This is a *research prototype* and an empirical algorithmic finding, not proof of a published-novel architecture or a fabricated 7-nm accelerator.

### Verified experiment and evidence trace

- Model: `HuggingFaceTB/SmolLM2-135M` revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, original full pretrained weights, CPU source execution, FP32 base weights, 30 layers, GQA Hq9/Hkv3, d64. `transformers==4.46.3`, `torch==2.5.1`, `numpy==1.26.4`; seed 22848.
- P=128 prefix tokens, 32-token tail, **31 next-token NLL targets** on each non-repeated authored passage. Domains: calibration A engineering, calibration B coastal survey, held-out C agriculture. Only C is used for held-out measurement.
- Shared prefix is grouped *by KV head* after applying real RoPE. Deterministic farthest-point plus Lloyd centroids, count-corrected attention `softmax(qKc^T/sqrt(d)+log(count)) Vc`, 64 representatives per selected layer. Query-specific suffix attention is exact and joined by stable prefix/suffix LSE. All unselected layers execute exact pretrained attention (software CPU in these tests).
- CIM arithmetic proxy: ideal signed integer Q4×K4 with explicit scales; **physically stored FP16 centroid values**. The published Sinangil 4-bit O4 ADC nonlinearity is **not** modeled in this new inference experiment. The exact full-P control checks reconstruction against pretrained attention.
- Calibration: isolated layer modifications measured on A and B. **Strict** candidate gate: each admitted layer must have non-increasing standalone NLL in both; the joint admitted set must stay within +0.05 NLL on each. The set is ranked by worst single-layer calibration impact. Calibration and held-out C are separate. Candidate counts are sampled from 0,1,2,4,6,8,12,16,24,30; this is a tested greedy policy, **not** a globally optimal search over 30-layer subsets.
- Source execution receipts: [exact full-P/naive compression control](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/42002167352); [single-calibration admission](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/42002167203); [two-domain permissive admission](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/42002167146); [two-domain strict admission](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/42002167366); [numeric and macro-DP qualification](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/42002167150). All five succeeded on the final FP16-V implementation at commit `6037b3815d4cd02845aac9c84e83212a769c9343`.

## A. Real held-out decoder quality — same checkpoint and held-out targets

All rows below use P128 and 64 centroid representatives per selected layer. **Held-out dataset: one independent, non-repeated authored passage, 31 next-token prediction targets.**

| Policy | Selected layers | Held-out NLL | ΔNLL vs exact | Relative perplexity increase (exp ΔNLL −1) |
|---|---:|---:|---:|---:|
| Pretrained exact attention | 0 | **3.257782** | 0 | 0% |
| **Strict calibration-gated subset** | **4** (12,13,21,27) | **3.291909** | **+0.034127** | **+3.47%** |
| Fixed first four layers, same K4/V16 storage count | 4 | 3.391042 | +0.133260 | +14.25% |
| Two-domain permissive +0.05 budget, with no individual-layer gate | 24 | 3.552132 | +0.294350 | +34.23% |
| One-domain calibration admitting all layers | 30 | 3.839710 | +0.581928 | +78.95% |

**Effect:** strict layer selection reduces the third-domain NLL penalty by **74.39%** compared with fixed first-four compression at the **same number of representatives and layers**. This is a single held-out-passage result, not an estimated population confidence interval. It does not assert a latency gain.

**Why strict gating was needed:** the more permissive two-domain calibration selected 24 layers and kept training-side ΔNLL within +0.05, but suffered **+0.29435 NLL** on held-out C. This is direct evidence of calibration overfitting; the +0.05 calibration budget does not give a universal generalization guarantee.

A separate eight-token earlier functional control (with repeated toy text) reconstructed all-P attention with maximum logit difference below 4e−5; its NLL must not be mixed with the 31-target held-out comparisons.

## B. Physical SRAM-CIM capacity

Native published 7-nm Sinangil Q4K4 K± signed banks, 7-nm Yokoyama conventional FP16 V macro, plus 16-bit centroid count and scale metadata (whole macros only). This counts *storage macros*, **not** support circuitry or digital throughput.

| Workload geometry | Stored prefix representation | Full-layer physical K/V + metadata macro area | Meaning |
|---|---|---:|---|
| **Llama-3.1-8B GQA, P2048**, all 32 layers | M64 per layer | **14.274279 mm²** | 4096 K CIM macros + 128 FP16 V SRAM macros + 2 metadata SRAM macros; full 32-layer coverage *by compressed representatives* |
| **SmolLM2-135M GQA, P128**, all 30 layers | M64 per layer | **3.390282 mm²** | Real experimental checkpoint; 2× fewer prefix positions; held-out NLL loss unacceptable when uniformly applied |
| SmolLM2-135M, P128, selected 4/30 layers | M64 per selected layer | **0.459818 mm²** | 96 K CIM macros + 16 V SRAM macros + 1 metadata macro; other 26 layers execute exactly |

The Llama P2048 result provides a **32× reduction in represented prefix positions** and reverses the earlier permanent-residency impossibility *at storage capacity only*. Its numerical effect is currently unmeasured on the Llama-8B checkpoint; transferring the 135M quality results to Llama would be invalid. The 1.7257 mm² remaining under 16 mm² is **unallocated**, not evidence that the softmax, PV, controller, clock and interface can be built in that area.

## C. Contribution relative to existing literature

- [Hydragen](https://arxiv.org/abs/2402.05099) already introduced exact prefix/suffix attention decomposition, query batching, and LSE fusion. These are baselines and reusable components, **not our contribution**.
- [Multipole Attention, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/359f442022ef31bcd9e41959576107ef-Abstract-Conference.html) uses key clustering, key/value centroids, multiplicity and importance-aware exact/approximate attention. Our centroid formula **is prior art**, and a quality-gated layer selector is not automatically a new attention mechanism.
- The candidate contribution requiring a stronger demonstration is **cross-layer adaptive exact/CIM placement under the complete signed-K4/FP16-V 7-nm macro budget, optimizing real decoder-quality loss while reusing shared prefixes**. The exact integer-capacity DP minimizes an empirical attention-error proxy, but the real-model experiment shows that proxy optimization does not necessarily improve NLL. The strict NLL-based selection is a tested heuristic, **not globally optimal**.

## D. Reviewer responses — concrete evidence

| Reviewer concern | Result | Status |
|---|---|---|
| Novelty over Hydragen and Multipole | Explicit prior-art boundary; new research target is *joint* device-resource admission and model-quality placement | Hypothesis; novelty needs a broader prior-art comparison |
| Physically grounded K/V capacity | Complete signed-K CIM + FP16 V + metadata allocation, 14.274 mm² for Llama-8B M64 | Source-macro storage qualified |
| Numeric quality, quantization and ADC | Pretrained 135M NLL measured under ideal Q4K4 with physically rounded FP16 V; full-P oracle control passed | Genuine small-model pilot; characterized O4 ADC still absent |
| Meaningful baseline | Exact pretrained model, uniform 30, permissive 24, fixed first-four and strict-four at matched layer count | Completed on 31-target third domain |
| Stability / held-out generalization | 2-domain calibration / third-domain test. Permissive admission +0.294 NLL; strict +0.034 NLL | Pilot with single unseen passage; no confidence interval |
| Full system and optimized GPU | Unselected attention remains exact, but CPU execution only; no GPU kernel timing or end-to-end memory/energy measurement | Not demonstrated |
| Scaling to 8B and long contexts | Exact native 8B storage mapping at P2048; no 8B compressed attention NLL | Architecture geometry only |
| Reproducibility | Pinned checkpoint/dependencies, deterministic clustering, complete code/scripts and successful five workflow executions | Executed and versioned |

## Scientific decision

**Supported:** The original full-prefix K/V occupancy obstacle can be reduced *algebraically* through compressed representations. Exact fallback and dual-domain quality gating protect real pretrained outputs better than fixed-index or permissive compression on the performed test. This gives a defensible *early-stage combined architecture/methodology direction*.

**Not supported:** a 7-nm *working* full-chip implementation, improved GPU/CIM tokens/s or tokens/J, significant accuracy preservation at M64 across all Llama-8B layers, a generalizable model-quality guarantee, or new centroid-attention theory.

For a publishable competitive claim, use the existing H100 infrastructure to test genuine Llama/Mistral checkpoint quality with a meaningful held-out corpus and model multiple exact/CIM configurations **including measured or source-supported peripheral operation rates**. Do not replace those measurements with arbitrary latency constants. The four-layer 135M pilot is a positive controlled result, not the final paper-level performance claim.
