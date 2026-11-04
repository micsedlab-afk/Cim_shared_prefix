# Source-model-conditional selective and pipelined 7-nm CIM scheduling

**Date:** 4 November 2026. **Executed code and tests:** [integration verification 41857191050](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/41857191050). **Source GPU execution:** [MAGMA native grouped FP16 roofline 41855617524](https://github.com/micsedlab-afk/MAGMA-gpt/actions/runs/41855617524).

## Scientific objective and exact scope

Optimize the **approved positive/negative signed Q4/K4 Sinangil macro mapping** while retaining physical storage and scheduling costs. Do not use the previously rejected affine-offset scheme: the original [Gate-A decision](https://github.com/micsedlab-afk/shared_prefix_cim/blob/a8a3f7238c471da2f3b7e6fe5647f8f25b8d84c4/analysis/sram2llm/SINANGIL_GATE_A_DECISION.md) reports 57.617% Q4 Fig.16 support violations and 100% ADC top-code saturation under its provisional diagnostic calibration. Its apparent 4× operation-count reduction is not a qualified speedup.

Under P=2048, B=1, O=128, Q32/KV8/D128/L32, 16 mm² incremental area, 0.5 mm² assumed external support, 256 active ports, 1 TB/s independent supply/program/query/output links and 1 Toperation/s scaling/reduction:

- Full positive/negative key storage for **one** decoder layer: 4,096 complete 512-byte Sinangil CIM macros = **13.1072 mm²**.
- Double-buffer raw O4 codes, FP32 signed intermediate partials and completed 32b scores: **10** complete 7-nm Yokoyama SRAM macros = **0.089775 mm²**, charged within an explicitly reserved **0.1 mm²** operand-staging slice of the 0.5-mm² assumed support budget. The other 0.4 mm² remains an unqualified allowance for control, clocks, delivery and reduction circuitry.
- Remaining 31 layers run unchanged on the source-modeled GPU. Other resident macro space is *not* charged twice or treated as simultaneously populated with multiple whole layers.
- Execute **64 independent 16-token × KV-head score groups per physical wave**, 256 active CIM ports = 64 groups × (2 D tiles × 2 K sign banks).
- Four native unsigned evaluations per Q4/K4 signed dot tile (both Q signs across two K planes). No circuit modification.
- Serial dependency per wave: native compute → scaling → signed reduction → 32b completed-score link. Distinct physical service resources may overlap across successive waves, with **finite two-wave** blocking per stage.
- Program K+/K− macro images **once**, charged to the first output token; no paid-per-token streaming for the 31 layers that stay on GPU.
- The optimizer enumerates all legal query microbatch sizes and wave group widths in this particular batch-major pipeline family. The dispatch rule chooses only positive amortized source-model saving; otherwise GPU-only.

## Executed reference result

| Stage / metric | Conditional simulation |
|---|---:|
| Compute per 64-group wave | 44.000 ns |
| Partial scaling per wave | 32.768 ns |
| Signed reduction per wave | 28.672 ns |
| Completed score transfer per wave | 16.384 ns |
| Number of waves | 16 |
| Complete resident-layer warm service | **0.783872 µs/token** |
| Key-source + initial macro programming | **4.194304 µs once** |
| Mean over 128 tokens | **0.816640 µs/token** |
| Source-native A100 FP16 grouped *single shared K* prefix QK roofline | **1.062670 µs/layer** |
| **Net simulated prefix-QK stage saving** | **0.246030 µs/generated token** |
| Maximum unmodeled per-token setup before advantage disappears | **0.246030 µs** |
| Minimum amortization horizon (constant source geometry) | **16 output tokens** |

This is a **source-model-conditional positive QK contribution**, **not** an established full-attention or end-to-end decoder speedup. The native A100 grouped roofline is computed by Matmul with M=B×G, K=128, N=2048, K read once per KV head. It omits kernel launches and is not a real optimized Hydragen/FlashAttention timing. The model charges a materialized score tensor to the source GPU; fused GPU attention may avoid that traffic.

The previously executed full 32-layer CIM serial policy was **+159.918 µs/token** slower than the old (nonfused) A100 FP16 source-model comparison. The new selective pipeline avoids that repeated programming by choosing *one resident layer* and leaving all other layers on the GPU. Relative to the new grouped source comparator, the whole-layer pinned **nonoverlapped** schedule would cost approximately 1.984000 µs per token, or **+0.921330 µs** versus source prefix QK. The optimized finite-buffer wave pipeline reduces its mean to 0.816640 µs, crossing the modeled break-even.

## Shared-prefix batch scaling, B=1–32

Source-native GPU query batching shares K rather than multiplying its physical reads by B. Both simulated candidates use exact native GQA geometry. A complete physical SRAM buffer is reserved for each candidate, and the optimizer searches batch and group tiling.

| B | Grouped GPU prefix QK roofline (µs/layer) | Conditional resident-layer CIM mean (µs/token) | Selected path |
|---:|---:|---:|---|
| 1 | 1.062670 | 0.816640 | CIM, 1 layer |
| 2 | 1.096820 | 1.522688 | GPU |
| 4 | 1.165120 | 2.934784 | GPU |
| 8 | 1.301720 | 5.758976 | GPU |
| 16 | 1.574921 | 11.407360 | GPU |
| 32 | 2.121322 | 22.704128 | GPU |

The optimized query microbatch is 1 in all six cases. This avoids the original unoptimized wave-buffer explosion as B increases. For B=8, the model improves resident-layer service from **45.183 µs to 5.759 µs**, but the grouped GPU model is still faster.

## Larger conditional sweep and provenance

The executed grid includes B=1, budgets 13–16 mm², port caps {64,128,256,512}, link rates {100,500,1000 GB/s}, output lengths {1,8,16,32,128}, and unmodeled additional setup {0,0.1,0.3 µs}:

- **720** physical and timing sensitivity coordinates.
- **48** choose CIM offload under the specified source comparison.
- **492** choose GPU-only.
- **180** cannot fit one fully resident signed K layer plus the required buffers.

Results and source: `results/integration/selective_pipeline_20261104.json`,
`results/integration/native_grouped_a100_fp16_prefix_roofline.json`,
`src/cim_shared_prefix/selective_pipeline.py`,
`scripts/run_selective_pipelined_study.py`,
`scripts/compare_grouped_gpu_7nm_selective.py`. The Actions run above contains the full 720-row CSV, grouped model comparison CSV/JSON, exact source-macro trace and tests.

## Conclusion and physical limits

**A source-relative positive region exists at B=1**, with fully resident single-layer signed K and finite-buffer internal pipelining. It is **small and not robust to more than 0.246 µs of additional setup**, including missing Q4 quantization, DMA launch, score-assembly, controller or integration delays. No direct data here establish the required multi-macro 256-port physical design, 1 TB/s programming and score links, digital reductions at 1 Toperation/s, BF16 precision equivalence or the optimized Hydragen timing. The 96 Llama R0–R3 observations qualify the original all-layer split-mapping numerical protocol; **single-layer selective PPL is not evaluated**. The GPU source is an A100 **model**, not physical A100 access.

This is a falsifiable allocation+dispatch/causal-scheduling result, not generic optimizer novelty (CIM-Tuner already covers area-constrained scheduling), and cannot yet support a significant full-system speedup. Larger headline speedup would require changing the offload endpoint beyond prefix QK or obtaining additional source-grounded advantages; those are separate architectures and cannot be inserted without accounting for complete scores, V, softmax and hardware costs.
