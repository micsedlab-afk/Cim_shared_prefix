# Final integration — scientific qualification gates

Branch: `integration/final-framework`. The three completed workstreams were merged by PRs #1, #2 and #3. Upstream results are preserved with their original evidence labels.

## Correct contribution

Develop a **source-calibrated, conditional physical feasibility envelope** for Q4/K4 SRAM-CIM offload of dense, shared-prefix attention from a GPU. The key novelty candidate is a rigorous *exclusion condition* under specific area, score-transport and programming-bandwidth constraints, together with attainable execution schedules where source evidence permits. Do not advertise a generic compute/storage mapping optimizer as original: CIM-Tuner (DATE 2026) already co-explores the macro area, storage/compute partition and spatial/temporal mapping. Ouroboros (ASPLOS 2026) already develops wafer-scale CIM KV mapping and dynamic management.

## Critical physical-model reconciliation

The `source_grounded.placement()` baseline reserves **one streaming scratch macro**, spreading the remaining macros across all layers; this is a valid *fixed/poor residency policy*, never an optimal schedule or universal lower bound. The `residency_area.residency_plan()` branch enumerates `pinned_layers` and can allocate nearly all provisioned macros to scratch, finishing a streamed layer in fewer waves. Compare identical physical macro count, concurrent ports, scratch/mapping policy and bandwidths before comparing their times. For P=2,048 Llama with 4,096 provisioned macros, the two implementations currently use different scratch allocations and their quoted 5.758 ms versus ~0.09 ms **must not be reported as an implementation speedup**. One is a deliberately serial policy; the other is an optimistic macro-level capacity/parallelism sensitivity.

All new comparisons must share: total incremental area; 2 physical signed K planes; four unsigned Q4/K4 evaluations; layer order; macro port limit; partial reduction; Q input; completed-score transfer; GPU remainder. No area fabricated for MAC, peripheral or interface.

## Sound lower-bound mathematics

For a fixed source-defined Q4/K4 mapping, `N_native` mandatory native macro evaluations, `M` provisioned physical macros, `P_active<=M` independent active macro ports, and macro time `t_macro=5.5ns`, a per-layer optimistic compute-service floor is `ceil(N_native,layer / P_active) * t_macro`. The decoder has sequential layers; add this floor over causally ordered layers.

Compulsory traffic, macro programming, Q input and score output each independently create a *resource lower bound*. **Use max of individually unavoidable resource durations**, plus only strictly justified causal-path combinations, for an end-to-end latency **lower** bound. A serial **sum** of stage times is a *candidate schedule time/upper bound under specified rates*, NOT a universal lower bound when overlap is possible. Earlier branch `residency_area` calls its sum a serial 'bound'; integrate it as a serial conditional model, not as a proof of exclusion.

To exclude CIM from beating a measured qualified GPU baseline, the optimistic lower bound must already exceed that GPU's comparable total decoder time. If programming/link bandwidths are user-provided sensitivities, exclusion is **conditional on those bandwidth caps**, not physically established on a 7-nm accelerator. If GPU reference timing is absent, publish the required interface-bandwidth and GPU-time thresholds; no speedup claim.

## Numerical/attention semantics

The source macro has a 4-bit output ADC for unsigned averages. The adopted R2/R3 attention is *quantized and response-characterized*, not mathematically exact BF16 or lossless signed 32-bit QK. 32-bit denotes the **postprocessed transfer word width**, not reconstructed arithmetic precision. Keep GPU reference precision and evaluated numerical deviation explicit.

Hydragen (arXiv:2402.05099v2) implements exact shared-prefix attention with fused prefix/suffix attention+logsumexp, whereas a QK-only CIM with GPU softmax/PV must return many completed scores; compare complete attention outputs and total decoder time. MoSKA (arXiv:2511.06010v1) uses shared KV GEMM and sparse routing. S2-CIM (IEEE TCSI DOI:10.1109/TCSI.2026.3742618) is a measured 28nm **digital** CIM implementation with sparse KV selection and W4A8KV8; its reported model TPOT comes from traces/projected system timing, not a matching Q4/K4 exact dense offload comparison. Ouroboros uses ASAP7/CACTI and node-scaled fabric components: none is suitable for extracting a fabricated 7nm MAC area by subtraction or scaling.

## Acceptance evidence

1. Same reference workload, same physical macro count, same scheduling policy and rates reproduces equal geometry and traffic in area and system modules, or discrepancies are explained by explicit differences.
2. Score transfer and K streaming counted at appropriate causal stages; streaming/residency changes across generated tokens.
3. A physical-exclusion claim uses a mathematically valid **lower bound** with all prerequisite evidence; no serial sum misidentified as lower bound.
4. GPU baseline: matched model, GQA, exact suffix geometry, batch, length, precision and complete attention implementation. Without native results, TPOT/speedups stay null.
5. Quality remains evaluated Llama 96/96 only; OPT and Mistral 0/96 until BF16-capable GPU inference; Mistral sliding context applies to score traffic and residency.
6. Full repository test suite + source reruns are required before publication qualification. PR merge success does not imply tests pass.

## Remaining execution gates

- **Integration:** cross-workstream physical scheduler; valid lower-bound certificate; publication comparison model and tests.
- **GPU execution:** MAGMA/LLMCompass or direct GPU matched GQA stage measurement; system TPOT currently 0 qualified coordinates.
- **Quality extension:** OPT/Mistral CUDA inference 0/192; architecture sweeps can proceed without asserting their PPL.
- **Digital:** no qualified 7-nm physical digital iso-area, publish threshold/sensitivity only.
