# A100-primary independent scientific re-audit — 5 November 2026

## Frozen paper contract

Primary baseline: **A100 80GB FP16 HBM2e**, through the pinned MAGMA/LLMCompass source model. Original native workload: Llama-3.1-8B **GQA 32Q / 8KV / D128**, P2048, suffix128, batch {1,2,4,8,16,32}, O128. Do not substitute H100 evidence. Llama checkpoint BF16 numerical evaluations are a **separate precision protocol**; no cross-precision timing/quality equivalence is inferred.

CIM: published **7-nm Sinangil/Dong 64x64 unsigned I4W4O4** 0.0032 mm² complete macro, 5.5 ns native access, signed K+/K- banks and **four native unsigned evaluations per signed Q4xK4 MAC**, 4-bit ADC. FP16 V stored in published **7-nm Yokoyama 512x512 SRAM**, area recovered from 29.2 Mbit/mm² full macro density. Controller, interface, digital softmax/PV and physical power remain unqualified. No fabricated 7-nm synthesis or iso-area digital MAC.

A complete scientific comparator needs (i) A100 optimized exact shared-prefix attention (Hydragen/Flash-style), (ii) A100 *numerically matched* same-compressed attention, (iii) SRAM-only and digital SRAM+MAC under identical storage/endpoint/rate constraints, and (iv) CIM+GPU. This source model computes only FP16 operator sums and ideal floors; **signed Q4K4 GPU timing and quality matching remain null**. Real CUDA device access is absent; there is no real A100 kernel or model-level TPOT measurement.

## Corrected source-executed comparison

Source: `src/cim_shared_prefix/a100_compressed_comparator.py`, `scripts/run_a100_compressed_comparison.py`; five explicit tests in `tests/test_a100_compressed_comparator.py`. [Execution receipt](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/42010466020) on the corrected A100 model. All units below are **µs per layer per generated-token group**, not physical CUDA measurements; "unfused" means serialized source QK + softmax + PV, and fused is an optimistic FLOP/KV-byte floor.

Llama P2048, M64 (32x compression). The FP16 GPU compressed attention uses the **same number of centroids**, but its floating-point QK path is not verified equal in accuracy to signed Q4K4/O4 CIM.

| B | A100 compressed HBM unfused source | A100 compressed L2 unfused source | Published signed CIM K-Q macro-only optimistic floor |
|---:|---:|---:|---:|
| 1 | 0.146644 | 0.042658 | 0.044000 |
| 2 | 0.164724 | 0.049004 | 0.088000 |
| 4 | 0.200883 | 0.061697 | 0.176000 |
| 8 | 0.273201 | 0.087082 | 0.352000 |
| 16 | 0.417836 | 0.137851 | 0.704000 |
| 32 | 0.707107 | 0.239391 | 1.408000 |

The CIM floor reuses the native `allocate_cim` and `cim_native_floor` proof: for B=1, 128 active signed K macros / layer, 1,048,576 native unsigned MACs / token / layer, 1024 unsigned MACs per macro access, 5.5 ns per native cycle -> **8 minimum ideal macro waves = 44 ns**. This includes neither FP16 V read, stable softmax, digital PV, output transfer, nor setup/programming; it is **not a complete accelerator timing model**. All macros are assumed simultaneously available, at full utilization.

Under the optimistic serialized *source model only*, the CIM QK-only macro floor already exceeds the A100 compressed **full unfused** L2 source operator estimate for every listed B. The same is true against HBM source operators at B>=8. This is a conditional **model-relative fail-to-beat** statement. It is not a physical speedup bound or exclusion against real optimized A100; actual cache behavior, kernel launch, fusion and model efficiency are unmeasured. At B1–B4 HBM source, a positive candidate requires enough measured physical budget for every missing CIM/digital/interface stage; the 44 ns lower floor alone cannot establish a win.

At B1, the **A100 exact P2048** source unfused QK+softmax+PV is 4.443527 µs with HBM or 1.294710 µs with L2. At **M64**, the source unfused GPU time is 0.146644 µs HBM or 0.042658 µs L2. The **A100 compressed ideal fused** floor is 0.128565 µs HBM or 0.036312 µs L2. Therefore most of the uncompressed-to-compressed GPU model gain is available **without any CIM**. These are source-recomputed numbers, and there is no measured inference throughput improvement.

Independent necessary per-stage throughput, **without assuming serial stages or subtracting their times** (B1): V read 131072 B per layer, PV 262144 MACs per layer. To beat A100 compressed HBM unfused source time 0.146644 µs, V read must reach at least **0.893808 TB/s** and PV **1.787616 TMAC/s**, each if all other work were free. For the L2 source time 0.042658 µs, thresholds are **3.072605 TB/s** and **6.145211 TMAC/s**; the 44 ns macro floor already misses that modeled time. These are **necessary, not sufficient** rates; no real 7-nm V-port/PV MAC implementation meets a verified threshold in this repo.

## Physical residency, proper comparison case

- Full Llama P2048 FP16 prefix KV: 256 MiB across 32 layers.
- Hypothetical **all 32 layers compressed** to M64: 8 MiB FP16 GPU KV, or 5.0625 MiB packed signed K4 / FP16 V / metadata. This fits within **nominal A100 40 MB L2 capacity only as an aggregate size condition**, without proved residency.
- **Actual small-model quality pilot selected only four layers**. At native Llama geometry, an analogous 4-layer compressed / 28-layer exact hybrid requires **225 MiB GPU FP16 KV**, well above nominal A100 L2. Physical 7-nm signed K / FP16 V / metadata storage for just four hypothetical Llama layers is **1.791018 mm²** macro-only; all 32 compressed Llama layers are **14.274279 mm²** macro-only. Their **8B numerical quality is entirely untested**.
- With an *explicit* FP16 B1 all-HBM all-weight-streaming decoder byte model, four compressed and 28 exact Llama layers eliminate only 32,505,856 B out of 15,277,752,320 B/token (0.21277%). The **traffic-only** zero-overhead speedup ceiling is 1.002132x in that defined model; no universal GPU latency/energy bound follows. Across all 32 compressed layers the hypothetical traffic-only ceiling is 1.017316x, but full model-quality viability remains unestablished.

The published V SRAM ports/bandwidth, clock, controller, reduction, signed I/O, provisioning, and precise technology-specific extra area are not synthesized or measured. A 16-mm² macro fit is not a full-chip implementation result.

## Direct reviewer checklist

| Reviewer issue | Status after re-audit |
|---|---|
| Fair GPU baseline/Hydragen | **Improved** A100 source exact versus same-compressed alternative, both HBM/L2, same GQA and P; fused kernel execution and numerical equality to Q4 still OPEN |
| Novelty vs Hydragen, Multipole Attention, MoSKA, Ouroboros, CIM-Tuner | Prefix clustering/fusion/batching/placement are prior art; physically constrained quality admission remains **hypothesis**, not published superiority |
| Physical 7-nm signed operation and ADC | K+/- macro maps and 4 unsigned accesses are accounted; O4 characterized ADC response **not in new compressed inference**, so quality is upper-optimistic |
| Complete physical area and interface | Macro-only data qualified; external digital/bus area and rates remain **unqualified** |
| Storage capacity versus 32x numerical quality | Physical full-32-layer capacity exists only with highly compressed representation; actual 135M four-layer results cannot be extrapolated to Llama8B all layers |
| Full decoder time/power/speedup | **UNSUPPORTED** (null, never synthetic) |
| Conventional SRAM+digital iso-area | Published macro capacity and break-even thresholds only, no unsupported matched physical digital silicon |

## Research decision

Most defensible contribution is a **quantitative, reproducible, physically constrained feasibility frontier** for signed Q4K4 shared-prefix attention with selective model-quality admission and fair conventional GPU storage alternatives. The new critical result is that **compression makes A100 substantially faster too**, and signed CIM floor plus missing V/digital hardware make accelerator competitiveness unproven. A positive architectural claim requires real 8B quality (including O4 ADC), matched A100 exact/FP16-compressed/Q4K4 implementations, and physically credible peripheral/transfer throughput. Until then, report the result as rigorous feasibility and limitations, not tokens/s, speedup, energy, or new centroid theory.

All original studies/tests preserved; no unrelated model/resource substitutions.
