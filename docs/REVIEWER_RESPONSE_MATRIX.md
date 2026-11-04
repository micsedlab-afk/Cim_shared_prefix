# Reviewer response and claim-qualification matrix

| Reviewer concern | Evidence / response | Status |
|---|---|---|
| Novelty relative to **Hydragen** (Juravsky et al., 2024, arXiv:2402.05099) | Hydragen provides exact shared-prefix/suffix fused attention with an efficient batched-prefix GPU execution. We implement a mathematical batched-prefix + logsumexp CPU-capable reference and test exact outputs against full-context attention. Fair full GPU performance still needs measured matching GQA and full decoder. The contribution candidate is the boundary for whether *QK-only* CIM offload can overcome completed-score return and programming costs. | Functional baseline: complete; performance baseline: **OPEN** |
| Novelty relative to **MoSKA** (Rhee et al., arXiv:2511.06010) | Shared-KV GEMM batching and sparse MoE-style routing are prior art. Our dense-prefix Q4/K4 physical frontier is a different workload path; no performance comparison to MoSKA is asserted. | Scoped |
| Novelty relative to **S²-CIM** (Liu et al., IEEE TCSI 2026, DOI:10.1109/TCSI.2026.3742618) | Measured 28-nm *digital* CIM exploits KV sparsity, compression and nonlinear operators; its performance and area cannot be mapped to the present 7-nm analog/O4 native macro without matched hardware and algorithms. | Scoped; direct comparison **OPEN** |
| Novelty relative to **CIM-Tuner** (Chen et al., DATE 2026, DOI:10.23919/DATE69613.2026.11539276) | Area-constrained compute/storage co-exploration and hierarchical mapping are already established. We make no optimizer novelty claim, restricting new claims to source-grounded GPU–SRAM-CIM physical feasibility and exclusion certificates. | Claim corrected |
| Novelty relative to **Ouroboros** (Liu et al., ASPLOS 2026; arXiv:2603.02737) | Wafer-scale in-situ CIM with token pipelining and dynamic KV management already addresses scheduling and mapping. Its scaled component estimates do not qualify 7-nm physical digital MAC area. | Claim corrected |
| Signed representation and native hardware precision | Two K-plane macros per logical tile, four unsigned Q4/K4 evaluations, off-macro signed partial reduction, O4 ADC response; transferred 32-bit completed scores are postprocessed words, not exact recovered 32-bit accumulation. | Geometry tested; circuit end-to-end **OPEN** |
| Physical macro count and 1–16 mm² support | All allocations use published complete 0.0032 mm² macro footprints and dedicated scratch capacity; no macro is multiplied by number of streamed layers. External area is an explicit 0 or 0.5 mm² scenario, **not** physically validated. | Conditional coverage |
| Lower-bound correctness | Causal native service floors and independent compulsory resource floors combine by max; tests exhaustively enumerate small residency cases and verify constructive serial schedules cannot beat the necessary floor. | Tested; physical rate measurements **OPEN** |
| Programming, output and reduction completeness | Constructive serial stages charge cold/warm K supply, every write, all native signed evaluations, query bytes, partial scales/reductions and 32-bit completed score traffic. First/last effective Mistral prefix checked with SWA. | Analytical model completed |
| Fair SRAM+digital MAC comparison | Published 512x512 7-nm SRAM density is used. Digital MAC silicon/core clock/ports unresolved: report required MAC/s/mm² and interface bandwidth only. | Sensitivity; physical iso-area **OPEN** |
| Hydragen-style GPU and full decoder | CPU functional equivalence tests pass when CI succeeds. Native MAGMA/LLMCompass GQA/precision-matched GPU full scores, suffix, softmax/PV and all decoder operators are still missing. | **OPEN; TPOT null** |
| Numerical degradation and reproducibility | Recovered Llama R0-R3 96/96: absolute NLL/PPL and paired 95% intervals. Nonmonotonic R2→R3 is retained, no monotonicity presumption. OPT/Mistral 0/96 each; full inference and Mistral ADC-specific calibration missing. | Llama evidence retained; new inference **OPEN** |
| Publication figures and source regeneration | The deterministic integrated architecture script exports 1–16 mm² CSV, conditional bandwidth/latency SVGs, source and quality manifest. CI runs legacy studies and complete unit tests. | Automated conditional outputs |
| Headline full-system energy/latency savings | No direct matched GPU per-operator measurements, integrated physical ports, or validated control/reduction silicon. Full-decoder TPOT, energy and speedup claims remain null. | **NOT READY FOR PERFORMANCE PUBLICATION** |

## Proposed claim

A source-grounded, reproducible boundary analysis for dense shared-prefix
Q4/K4 attention offloading to a finite 7-nm SRAM-CIM macro budget,
incorporating signed-plane replication, dynamic K residency, programming,
causal layer execution and completed-score transfer. The result includes
sound necessary resource bounds, constructive conditional schedules and
required interface bandwidths. Full-system benefit is an experimental
question pending a competitive Hydragen-style GPU baseline.

## Distinct limitations

1. The source macro is measured, while integrated physical rates, port count,
   controller, signed reductions and support area are sensitivities.
2. Functional Hydragen equivalence does not supply a high-performance GPU
   implementation or kernel times.
3. Llama numerical evidence is evaluated but circuit response is emulated;
   quality cannot be transferred to OPT/Mistral without real evaluation.
4. Our 2 ms threshold is hypothetical. A conditional deadline result must
   never be misreported as measured GPU-vs-CIM speedup.
