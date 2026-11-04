# Executed source-native GPU/CIM model comparison

**Reference:** Llama-3.1-8B GQA, 32 query heads / 8 KV heads, head dimension 128, 32 decoder layers, shared prefix 2,048, suffix 128, B=1, O=128.

**Work executed:** The original `micsedlab-afk/MAGMA-gpt` vendored `BatchedMatmul.roofline_model()` source computed 128 native A100 FP16/HBM2e GQA full-QK and suffix-QK operator points on a CPU runner. The complementary signed Q4/K4 7-nm CIM model executed all 128 causal tokens, including finite 4,842-macro allocation, persistent programming and source-key transport, four unsigned evaluations, partial scaling and reductions, and 32-bit completed-score output.

## Quantitative paired result

| Quantity | Mean modeled latency |
|---|---:|
| Source-native A100 FP16 full-context QK | 1.162365866 µs/layer |
| Source-native A100 FP16 suffix-only QK | 0.101704757 µs/layer |
| Modeled GPU QK work potentially removed by offload | **1.060661108 µs/layer** |
| Optimistic 7-nm CIM necessary resource floor | 2.020269375 µs/layer |
| Executed 7-nm CIM serial policy | 6.058113 µs/layer |
| GPU-model QK saving across 32 layers | 33.941155468 µs/token |
| 7-nm CIM serial prefix-QK service | 193.859616 µs/token |
| **S2−S0 modeled attention contribution (conditional)** | **+159.918461 µs/token** |

The paired expression is:

```text
S2 - S0 = CIM_prefix_QK_completed_score_service
          - (GPU_full_QK - GPU_suffix_QK)
          + additional_GPU_score_assembly
```

Here `additional_GPU_score_assembly=0` favors CIM, and the remaining GPU decoder operators and softmax/PV are shared/canceled in this restricted operator decomposition. The result is **conditional negative performance**: the chosen source-modeled A100 FP16 GPU QK saving does not amortize the published-macro 7-nm CIM serial policy at the example interface rates. It does **not** establish end-to-end TPOT, a physically fabricated A100/CIM pair, or an impossibility theorem against optimized Hydragen kernels.

The original source roofline omits GPU kernel launches, FlashAttention fusion and practical QK/PV memory scheduling; subtracting two GPU roofline estimates **cannot serve as a universal upper bound on their real latency difference**. The CIM requires Q4/K4/O4 quantization with known numerical degradation relative to FP16/BF16. The link rates, 0.5-mm² support area, 256 active ports, reduction throughput and controller=0 are explicitly *architectural sensitivity inputs*.

## Reproduction and code evidence

1. Native MAGMA source execution (128 steps): [Actions run 41846791710](https://github.com/micsedlab-afk/MAGMA-gpt/actions/runs/41846791710), script `MAGMA-gpt@research/cim-native-gqa-20261104:scripts/research_cim_native_gqa_a100.py`, source method `BatchedMatmul.roofline_model`. The vendor `matmul.py` was restored unchanged.
2. Committed source point data: `results/integration/source_native_a100_fp16_gqa_qk_128steps.json`.
3. Token-causal CIM simulation: `scripts/run_causal_7nm_sim.py` / `src/cim_shared_prefix/causal_timeline.py`.
4. Native/7-nm comparison: `scripts/compare_native_a100_model_7nm.py`. All 128 paired point records are emitted in the downloadable CI artifact.
5. [Integrated CI run 41847831215](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/41847831215): **141 discovered tests; 140 passed; 1 skipped; 0 failures**. The paired study and 2,304-source-coordinate saving threshold generator executed successfully.
6. Committed condensed outcome: `results/integration/paired_a100_model_7nm_cim_summary.json`.

These are numerically reproducible and bounded simulation outcomes. All physical unsupported components remain identified sensitivities. There are **no reported measured A100 times, BF16 timings, fabricated accelerator peripheral layouts, digital iso-area results, or end-to-end hardware speedups**.
