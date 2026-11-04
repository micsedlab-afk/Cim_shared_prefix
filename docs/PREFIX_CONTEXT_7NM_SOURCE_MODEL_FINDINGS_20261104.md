# Executed prefix-context study — 7-nm CIM QK + FP16 V SRAM + digital online attention

**Date:** 4 November 2026. **Current evidence:** a locally executed independent Python replication, published macro specifications, original MAGMA/LLMCompass source. The corresponding exact repository module/CI jobs have **passed**, and neither a measured GPU nor a fabricated integrated accelerator has been tested.

## 1. Architecture and numerical semantics

Keep the original, approved **positive/negative signed Q4×K4** mapping with two independently instantiated K sign planes and four native unsigned evaluations; do not revive rejected affine-offset mappings. One Llama 3.1 8B layer is fully pinned in complete published 7-nm Sinangil macros.

The larger functional endpoint is **completed prefix attention**: signed Q4×K4 O4-digitized QK scores → off-macro scaling/reduction → stable *online softmax* → FP16 V SRAM fetch → digital PV (FP32 accumulator) → one FP16 prefix context and FP32 LSE per query head. GPU suffix attention and logsumexp fusion follow at the same context-output endpoint. No V quantization is introduced; previous R0–R3 whole-model numerical results **do not qualify this new selective policy**.

The CPU numerical oracle `online_prefix_context()` and `fuse_prefix_suffix()` test mathematical correctness for explicitly supplied reconstructed scores and FP16 V values. Those tests do not establish analog Q4×K4/O4 end-to-end inference equivalence.

## 2. Full-macro area and communication

Reference: native Llama-3.1-8B GQA 32Q/8KV/D128/L32, P2048, B1, O128; 16 mm² incremental budget, with 0.5 mm² assumed support.

| Resource | Count/size | Macro-only area |
|---|---:|---:|
| One layer K+/K− CIM banks | 4,096 × complete Sinangil 64×64 7nm | 13.107200 mm² |
| Resident FP16 V | 4 MiB = 128 × 512×512 Yokoyama 7nm SRAM | 1.149125 mm² |
| Bounded two-wave digitized/partial/V FIFOs and streaming FP32 state | 27 complete 512×512 SRAM macros | 0.242394 mm² |
| Assumed complete external support | controller/digital/clock/etc. | 0.500000 mm² |
| **Total** | area rounded to whole macros | **14.998718 mm²** |

With the full-prefix QK-only endpoint, the engine would transfer 262,144 B/layer/token (32-bit reconstructed scores). The combined prefix-context+LSE endpoint transfers only **8,320 B**, a **31.51×** output traffic reduction. V is read from on-chip SRAM (4,194,304 B/selected layer/token for B1).

## 3. Executed 128-output causal model

The reference assumes an explicitly nonphysical rate set: 1 TB/s independent external supply/write/query/context, 16 TB/s local V SRAM read, 256 active CIM ports, 1 Texp/s softmax exponential, 1 Tscalar-op/s online scaling, 1 Tpartial/s signed reduction, **64 TMAC/s** digital PV, zero controller/setup delay, and no invalid V reuse. Distinct stage resources overlap *across* waves with finite double buffering and causal per-wave dependencies.

| Modeled quantity | Result |
|---|---:|
| First-time K+V key/value supply and programming | 12.582912 µs once |
| Warm prefix-context service | 0.830632 µs/token |
| Cold cost amortized across 128 outputs | 0.098304 µs/token |
| **Mean conditional prefix-context service** | **0.928936 µs/token** |
| GPU source A100 grouped QK, *legacy element-count bug* | 1.062670 µs/layer |
| GPU source A100 FP16 grouped QK, **byte-corrected HBM model** | **2.125340 µs/layer** |
| GPU source A100 FP16 grouped QK, **byte-corrected L2 model** | **0.600284 µs/layer** |
| Source A100 FP16 *unfused* QK+Softmax+PV, byte-corrected HBM | 4.443527 µs/layer |
| Source A100 FP16 *unfused* QK+Softmax+PV, byte-corrected L2 | 1.294710 µs/layer |

This comparison clearly separates **GPU source-model HBM versus L2 residency**. The 0.929 µs conditional prefix engine is quicker than the modeled HBM QK-only step but slower than the optimistic L2 QK-only model. It is quicker than both *unfused* full-prefix GPU decompositions. **Neither comparator represents a qualified optimized fused Hydragen kernel**, so none establishes a physical GPU–CIM hardware speedup or decoder TPOT.

### Critical discovery: upstream Matmul FP16 units

In vendored LLMCompass `software_model/matmul.py`, `Matmul.__call__` sets `io_count = M*K + K*N + M*N`, i.e. **elements**. The original `Matmul.roofline_model` divides `io_count` directly by `device.io_module.bandwidth` in **bytes/s**, without multiplying by `DataType.word_size=2` for FP16. We preserve original results and independently correct the memory term:

```text
T_Matmul_corrected = max(2*M*K*N / source_device_FLOP_s,
                         2*(M*K + K*N + M*N) / source_device_B_s)
```

Group by KV head and multiply by H_KV; Softmax's original roofline already counts `word_size` in memory traffic. The source-device constants (A100 FP16) are `io=2039e9 B/s`, `L2=5120*1.41e9 B/s`, `tensor FLOP/s=108*4*16*16*2*1.41e9`, `vector FLOP/s=108*512*1.41e9`, and `112 flops/softmax element`.

This correction changes the earlier GPU source-model comparisons. The original uncorrected results remain as historical artifacts; do not relabel them without recomputation. GPU HBM/L2 are alternative memory-placement *simulations*, not observed cache states.

## 4. Performance boundaries

The independent 72-case B1–B8 / V-bandwidth 1–32 TB/s / PV rate 4–64 TMAC/s sensitivity has:

| Comparison target | Favorable conditional cases / 72 |
|---|---:|
| Byte-corrected A100 HBM QK-only source proxy | 16 |
| Byte-corrected A100 L2 QK-only source proxy | 0 |
| Byte-corrected HBM *unfused* complete prefix | 58 |
| Byte-corrected L2 *unfused* complete prefix | 8 |

All 16 HBM-QK-proxy favorable cases occur at B1 or B2. In the B1 128-output case, positive margin versus HBM QK alone first appears with a modeled local V rate of 4 TB/s and PV throughput **above approximately 4.392 TMAC/s** (holding other assumed rates). At 8 TB/s local V read, the PV threshold falls to **~4.364 TMAC/s**. Against the L2 unfused full-prefix source model, a stronger 8 TB/s and **~7.946 TMAC/s** combination is required. No rate at or below the investigated 32 TB/s and 128 TMAC/s beats the L2 QK-only source proxy.

These thresholds are *conditional rate requirements*, not synthesized 7-nm digital capabilities. A 16 TB/s shared V rate implies 125 GB/s per installed V SRAM macro on average. A 64 TMAC/s PV block inside at most 0.5 mm² external support would require at least 128 TMAC/s/mm² of MAC throughput density before charging the controller, softmax units and communication. Neither rate has a qualified physical implementation.

## 5. Reproducibility and publication boundary

Repository source: `src/cim_shared_prefix/prefix_context_engine.py`,
`src/cim_shared_prefix/source_roofline_corrected.py`,
`scripts/run_prefix_context_sweep.py`,
`tests/test_prefix_context_engine.py`,
`tests/test_source_roofline_corrected.py`.

Independent local execution generated downloadable `prefix_context_independent_sweep.csv`,
`prefix_context_corrected_gpu_comparison.csv`,
`prefix_context_required_rate_frontier.csv`, and `prefix_context_rate_frontier.svg`. The committed feature branch records the independently executed reference in `results/integration/prefix_context_independent_local_20261104.json`.

**Final qualification:** [Source-model integration CI 41868345731](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/41868345731) completed with 156 discovered tests (155 pass, 1 optional skip), 576-case sweep and archived output artifacts. [Native MAGMA model CI 41868354178](https://github.com/micsedlab-afk/MAGMA-gpt/actions/runs/41868354178) passed and executed source QK–softmax–PV with corrected FP16 bytes. These checks validate **the executed simulation implementation** and its provenance, not silicon/digital area or optimized fused Hydragen performance. See `docs/FINAL_RESULTS_AND_REVIEWER_MATRIX_20261104.md` for the reviewer-ordered final evidence and conclusions.

**Scientific decision:** the full-prefix attention endpoint materially reduces communication and avoids introducing new V approximation, but the modeled benefit is sensitive to highly aggressive unqualified local SRAM and digital PV throughput. The source-model byte correction makes GPU memory placement the deciding axis. A robust 7-nm silicon implementation win is **not** established by these results. The constructive next architectural direction is reducing the requirement for V SRAM bandwidth/digital MAC throughput or co-designing a fuller KV-resident attention engine with documented physical rates; inflated GPU rooflines or unsupported periphery rates are not acceptable alternatives.
