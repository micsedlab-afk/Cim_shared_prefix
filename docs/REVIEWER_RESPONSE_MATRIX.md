# Reviewer concern -> result -> qualification matrix (4 November 2026)

**Authoritative detailed analysis:** [FINAL_RESULTS_AND_REVIEWER_MATRIX_20261104.md](FINAL_RESULTS_AND_REVIEWER_MATRIX_20261104.md). Existing literature comparisons and prior review framing are retained. The result is **published-macro-calibrated 7-nm simulation** with source-native A100 FP16 GQA modeling; no fabricated 7-nm integrated accelerator or real GPU kernel is asserted.

| Existing reviewer issue | Direct result / remedy | Qualification |
|---|---|---|
| Hydragen, shared-prefix exact attention | True context-output offload with stable suffix logsumexp; 31.5077x smaller link than QK-only. GPU reference supports native shared-K/GQA, byte-corrected FP16 HBM/L2 model. Still lacks optimized fused Hydragen performance. | Functional/method: supported; competitive timing OPEN |
| MoSKA sparse/shared-KV baseline | Dense-prefix architecture, explicit shared-K query batching. MoSKA sparse/FP8 regimes are different; shared-K batching is not claimed as our novelty. | Scope corrected |
| S2-CIM measured 28-nm digital acceleration | Separate process, digital/compressed precision and sparse workload. No node-scaled speedup comparisons. | Scope corrected |
| CIM-Tuner area-aware optimizer novelty | Resource-schedule optimization is prior art. Claim specific physical signed Q4K4 + KV residency + complete prefix-context feasibility methodology. | Claim corrected |
| Ouroboros KV scheduling | Dynamic KV management and token pipelines are established prior art; report only measured-macro-mapped feasibility thresholds. | Claim corrected |
| Signed Q4×K4/O4 physical operation | Preserve positive/negative banks, 4 unsigned accesses, signed reductions, characterized O4 ADC effects. Original affine-offset Gate-A mapping rejected. | Modeled mapping supported; direct silicon behavior outside scope |
| Complete 7-nm area | 4096 K CIM + 128 V SRAM + 27 finite FIFO/state SRAM macros, total 14.998718 mm² **including 0.5 mm² assumed digital/control support**. | Complete macro counts; periphery unqualified |
| Valid lower bound and overlap | Independent necessary maxima retained; finite two-wave causal pipeline is constructive with stage backpressure. Conditional serial schedules never treated as universal lower bounds. | Simulated/checked |
| Cold programming, V reads, output | 12.582912 µs one-time K+V provisioning, 4MiB V read/token/layer, 262144→8320 B external output, 0.928936 µs mean under stated reference rates. | Simulation with stated rates |
| Fair conventional SRAM/digital MAC iso-area | Yokoyama published 7-nm SRAM capacity used; digital PV 64TMAC/s in ≤0.5mm² would require **≥128 TMAC/s/mm²**, local V 16TB/s =125GB/s/macro. Neither throughput/area synthesized. | Break-even density quantified; physical feasibility open |
| Correct GPU precision, GQA, full attention | Original LLMCompass FP16 Matmul IO element/byte mismatch corrected; [MAGMA source rerun passed](https://github.com/micsedlab-afk/MAGMA-gpt/actions/runs/41868354178). GPU HBM QK=2.12534µs; L2 QK=0.60028µs; HBM unfused full=4.44353µs; L2 unfused full=1.29471µs. Not optimized Hydragen. | Native source operator model passed; measured fused/full TPOT open |
| Numerical degradation / ADC | Genuine archived 96/96 Llama R0-R3 evaluations, P512 +2.11%, P2048 +3.38%, P16384 +6.43% PPL R0→R3. P2048 R3−R2 nonsignificant negative mean. New single-layer context policy numerically untested. | Historical inference supported; new path quality open |
| Mistral/OPT workload coverage | Native geometry preserved but no new evaluated numerical inference (0/96 for each). | Open; no extrapolation |
| Reproducibility and evidence trace | [CIM source run 41868345731](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/41868345731): 156 discovered tests, 155 pass, 1 skipped, full 576-case sweep archived. Independent 72-case algebraic replication. | Source simulation qualified |
| Full decoder, energy, silicon claim | Complete S0/S1/S2 end-to-end latency and GPU speedup remain null; no device energy or 7-nm digital synthesis produced. No fictitious claims. | Publication scope restricted |

## Numeric comparison for B=1, P=2048, O=128, 16mm²

The new conditional CIM QK/softmax/PV **prefix-context** model averages **0.928936 µs** (cold supply/programming amortized), assuming 256 concurrent native macro ports, 16 TB/s local V reads, 64 TMAC/s digital PV, 1 TB/s external supply/programming/output and zero controller/setup. The corrected A100 source-model QK-only HBM (2.125340µs) is slower, but its optimistic L2 model (0.600284µs) is faster. The source-model *unfused* full-prefix operator sums (HBM 4.443527µs, L2 1.294710µs) are slower than the hypothetical CIM path. These comparisons do **not** establish an advantage over optimized fused Hydragen or full-decoder TPOT.

## Tested frontier

Full source-model grid **576 cases**: 92 below corrected HBM GPU QK proxy, 244 below corrected HBM unfused full-prefix only, 240 slower than or equal to that unfused model. Independent 72-case B1–B8 slice: 16 below corrected HBM QK-only proxy, **zero below L2 QK-only**, 58 below HBM unfused full-prefix, 8 below L2 unfused full-prefix. The rate assumptions in favorable cases remain hypothetical.

**Defensible contribution:** completed prefix-attention endpoint, finite 7-nm macro area and storage residency, causal schedule, source-calibrated and byte-corrected GPU alternatives, and resource-rate feasibility boundaries. **No system speedup or silicon energy claim.**
