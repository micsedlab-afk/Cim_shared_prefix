# Verified capacity and GPU-side traffic envelope — 4 November 2026

**Primary result:** This is a source-derived, assumption-minimized physical-capacity analysis for the native Llama-3.1-8B shared-prefix workload. It uses the **complete published 7-nm SRAM-CIM and SRAM macro areas** and **native pinned Llama architecture**. It does not use guessed 0.5 mm² peripheral support, 256 active ports, 16 TB/s SRAM, 64 TMAC/s, zero controller latency or synthetic GPU kernel times.

**Executed source:** `src/cim_shared_prefix/verified_residency_limits.py`, `scripts/run_verified_capacity_study.py`, `tests/test_verified_residency_limits.py`. [Independent 7-test and 136-case physical-source CI run 41916282585](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/41916282585). The workflow uploads a 136-case source-capacity CSV, 34-case HBM/L2 fused-GPU necessary-rate CSV and exact JSON with the complete native model weight inventory.

## Sources and mathematical contract

| Parameter | Source | Status |
|---|---|---|
| 7-nm unsigned 4×4 SRAM-CIM 64×64, area **0.0032 mm²/macro** | Dong/Sinangil, ISSCC 2020 and JSSC 2021, DOI 10.1109/JSSC.2020.3031290 | Published complete macro |
| Conventional 7-nm 512×512 SRAM, **29.2 Mbit/mm² macro density** | Yokoyama et al., VLSI 2020, DOI 10.1109/VLSICircuits18222.2020.9162985 | Published macro density |
| Llama-3.1-8B h4096, intermediate14336, Hq32, Hkv8, D128, L32, vocab128256, untied LM head | [Pinned config](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/config.json) | Source model |
| GPU A100 80GB SXM HBM2e **2039 GB/s**, FP16 peak **312 TFLOPS**, nominal L2 **40 MB** | [NVIDIA A100 product](https://www.nvidia.com/en-us/data-center/a100/) and [NVIDIA performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/) | Published GPU specifications |
| GPU A100 L2 bandwidth `5120 B/cycle × 1.41 GHz = 7.2192 TB/s` | Source [MAGMA/LLMCompass A100 device model](https://github.com/micsedlab-afk/MAGMA-gpt) | **Analytical GPU model**, not measured cache throughput |
| Accelerator area **16 mm²** | Research design budget | **Budget constraint**, not fabricated die area |
| Circuit digital area, clocks/ports, transfer/programming bandwidth, MCU runtime | No applicable 7-nm system physical evidence | **Excluded from rate-free capacity results** |

All physical 16-token tile results use exact complete macro allocation for two K sign-plane banks and **FP16 V**. The native signed Q4×K4 mapping requires four unsigned evaluations; the rejected affine-offset mapping is excluded.

## Result A — persistent complete-layer feasibility

For Llama GQA, one 16-prefix-token tile of a given layer occupies exactly **32 CIM macros** (K+/K− × 8 KV heads × 2 depth tiles) and **one conventional SRAM macro** (8 heads × 16 tokens × 128 dimensions × FP16 V = 32 KiB). Therefore

```text
A_tile = 32 × 0.0032 mm² + 262144 bits / (29.2e6 bits/mm²)
       = 0.11137753424657534 mm² per (16 prefix tokens × decoder layer)

A_KV_layer(P) = ceil(P/16) × A_tile, for P divisible by 16

L_full_max(A,P) = min(32, floor(A / A_KV_layer(P)))
```

This is a **permissive necessary capacity bound**: **no area** is allocated to control, interface, DAC/ADC outside the reported macro, buffers, clock or digital softmax/PV. Real allocations can only fit **fewer** complete layers.

| Prefix P | K+V complete-macro area/layer | Most complete layers in 16 mm² | Full-model FP16 prefix KV | Fits in nominal 40MB A100 L2 by bytes alone? |
|---:|---:|---:|---:|---|
| 64 | 0.445510 mm² | 32 | 8 MiB | Yes |
| 128 | 0.891020 mm² | 17 | 16 MiB | Yes |
| 256 | 1.782041 mm² | 8 | 32 MiB | Yes |
| 320 | 2.227551 mm² | 7 | 40 MiB | Approximately at/above L2 nominal capacity* |
| 512 | 3.564081 mm² | 4 | 64 MiB | No |
| 1024 | 7.128162 mm² | 2 | 128 MiB | No |
| 2048 | 14.256324 mm² | **1** | **256 MiB** | No |
| 2304 | 16.038365 mm² | **0** | 288 MiB | No |
| 4096 | 28.512649 mm² | **0** | 512 MiB | No |

*The published 40MB label does not specify whether an effective 40,000,000 B or 40 MiB are available to this workload. Cache-byte *capacity* alone also does not establish residency/hit rate; weights and other live data compete for L2.

At P2048, signed K macro area is 13.1072 mm²; V macro area is 1.1491244 mm²; their sum is **14.2563244 mm²**. Hence **1.7436756 mm² remains for ALL support+buffers+digital work**, not a randomly assumed 0.5 mm². The previous buffered candidate's 27 complete conventional SRAM macros add 0.2423934 mm², leaving **1.5012822 mm²** for still-uncharacterized circuits.

## Result B — stronger bound allowing arbitrary partial-prefix placement

A scheduler may store **partial prefixes** across several layers rather than complete K/V layers. For 16-token tiles with identical native geometry, the total number of *simultaneously persistent* layer×prefix tiles in a 16 mm² memory-only budget is bounded by:

```text
N_persistent_tiles_max = floor(16 / A_tile) = 143 tiles

P2048 : complete full-model prefix tiles = 32 × (2048 / 16) = 4096
Fraction_max = 143/4096 = 3.4912109375%

Removable GPU FP16 K+V bytes <= 143 × 65536 B = 9371648 B (8.9375 MiB)
```

This is valid for **any choice of permanently cached 16-token K+V tiles** with the defined macro organization, not just the previously implemented fully resident layer policy. An exact packing proof establishes that concentrating these 143 equal tiles into **one complete layer (128 tiles) plus 15 tiles of another** minimizes the count of touched layers when each touched layer incurs any positive fixed setup/fusion cost. Small-case exhaustive allocation tests passed.

**Do not interpret 143 tiles as physically attainable.** At 143 tiles, macro-only area is approximately 15.93 mm², leaving about 0.07 mm² for all buffers and periphery. The bound is deliberately optimistic. A schedule that reprograms tiles during every output token is a different nonresident architecture; its programming costs are not covered by this occupancy bound.

This also means a 4K prefix still permits *partial* KV residency even though **zero complete K/V layers** can fit. Previous conclusions implying no offload of any kind at 4K were overly broad.

## Result C — GPU-side memory traffic and ideal fused attention

Native Llama-3.1-8B FP16 **matmul weight inventory** (one copy of seven matrices per decoder layer: Q,K,V,O, gate, up, down, plus untied LM head) totals:

```text
Transformer projection + MLP parameters : 6,979,321,856
LM-head parameters                      :   525,336,576
FP16 dense matmul weight bytes          : 15,009,316,864 B/token
FP16 full-model prefix KV bytes, P2048  :    268,435,456 B/token (256 MiB)
FP16 ideal streamed total               : 15,277,752,320 B/token
```

This is explicitly a **B=1, every dense matrix weight streamed once from HBM per generated token** memory-traffic model. It excludes embedding lookup tables, the suffix KV, activation traffic and any nonmatmul overhead. It does **not** claim observed bandwidth, latency or cache behavior. Weight reuse, weight quantization and multi-request batching can change the traffic composition.

In this model, the **most** a 16 mm² persistent-tile CIM accelerator could remove from GPU HBM is **9,371,648 B**, or only:

```text
9,371,648 / 15,277,752,320 = 0.0613418% of ideal FP16 HBM bytes/token

Idealized HBM-traffic-only speedup ceiling = 1.0006137945×
```

This is a bound on the **specified B1 FP16 streaming-byte model**, not on GPU execution latency under arbitrary scheduling. Hardware offload or asynchronous overlapping stages may affect runtime differently; no universal speedup impossibility theorem is claimed.

Even removing every one of Llama's 32 full prefix K/V layers (requiring far more than 16 mm²) would eliminate at most **1.757%** of those streamed bytes at P2048. This demonstrates how dense weight streaming dominates the particular B1 decode traffic model.

### Optimistic fused GPU lower floor (no score materialization)

Compare against a stronger ideal **fused QK + softmax + PV** GPU model that **never materializes P-length score matrices**, shares K and V across GQA query heads and request batch, and reads each native FP16 K and V element **once**. The necessary floor for a source placement is

```text
T_fused_GPU >= max( bytes_native_KV / BW_GPU_source,
                    4 × B × Hq × P × D / dense_GPU_FP16_FLOP_s )
```

This deliberately **ignores** kernel launch, softmax, memory transaction efficiency, Q/O traffic and every other positive cost. It cannot be labeled a measured Hydragen latency.

B=1, P2048, **one decoder layer**:

| GPU source placement | Ideal fused lower floor | Minimum local V read bandwidth to match this floor with all other CIM work free | Minimum digital PV MAC/s to match floor with all other work free |
|---|---:|---:|---:|
| A100 SXM HBM2e, 2039 GB/s published | **4.114079 µs** | **1.0195 TB/s** | **2.039 TMAC/s** |
| L2 7.2192 TB/s analytical source model | **1.161986 µs** | **3.6096 TB/s** | **7.2192 TMAC/s** |

These are **necessary, never sufficient** conditions. At P2048, each `KV` layer requires K/Q4 CIM and V/FP16 storage, and the local digital engine must also execute score conversion, stable softmax, communications and control. The published macro characterization provides no validated bandwidth/area for those external digital units. Allocating the entire **1.7436756 mm²** free after K+V macros to PV would require at least **1.17 TMAC/s/mm²** (HBM ideal-floor target) or **4.14 TMAC/s/mm²** (L2 ideal-floor target), before any other support circuitry.

The 40 MB L2 can hold only a minority of the **256 MiB** complete Llama prefix K/V simultaneously at P2048, but GPU attention scheduling and L2 pinning/hit behavior are not simulated or measured here.

## Actual optimization conclusion

There is a **capacity-versus-opportunity inversion** under the fixed 16 mm² budget:

- **Short prefixes:** the accelerator can keep more complete layers resident, but the entire FP16 model prefix KV may also fit in nominal GPU L2, and output-context transfer may approach or exceed score-return size (at P64, the scores are 8192B whereas context+LSE is 8320B).
- **Long prefixes:** GPU memory pressure grows, but 7-nm signed K4 and FP16 V capacity restricts the accelerator to ≤1 complete layer at P2048 and zero complete layers at P≥2304, unless it uses a partial-prefix or reprogrammed policy.
- **Arbitrary persistent partial-prefix scheduling:** even with every nonmemory component free, the absolute removable FP16 GPU prefix KV data is bounded by area. It cannot grow with P once the 143 physical tile limit is saturated.
- **B1 dense weight-dominated decode:** one permitted 16 mm² accelerator can remove <0.062% of an ideal FP16 full-decoder streamed-byte workload at P2048. A large full-model speedup cannot be inferred from isolated context-QK microseconds.

These results **identify a strong architectural limitation**, without introducing speculative clocks/ports/throughput. They do not alone constitute a novel accelerator design. The possible scholarly contribution is a **published-macro-calibrated storage/traffic upper-envelope methodology** exposing which offload endpoints can be relevant. Its novelty should be positioned relative to Hydragen, CIM-Tuner and SRAM-CIM attention literature, without claiming those algorithms as new.

## Scientific implications and next architecture candidates

The physically reliable way to seek a larger fraction of model time is to change one of the constraints in a source-justified manner: (i) verified higher KV storage density or digital operation density; (ii) a larger evidenced macro-area budget; (iii) target a workload with a greater relative attention/weight footprint, e.g. much longer context and/or a smaller weight model; (iv) explore execution paths reducing *weight* movement instead of prefix KV alone; or (v) incorporate source-matched KV compression with a separately evaluated numerical-quality protocol. None has been claimed as already implemented or beneficial.

Do not publish the historical 0.929 µs point as a silicon performance result. Its 16TB/s and 64TMAC/s near-memory resources were **arbitrary sensitivity values**. The valid outputs retained here are native macro counts, bounds, source GPU specifications, exact byte volumes, and separately labeled necessary-rate inequalities.
