# Cim_shared_prefix — OJCAS revision framework

**Manuscript baseline (authoritative):** [OJCAS baseline lock](docs/OJCAS_MANUSCRIPT_BASELINE_LOCK_20261105.md). Preserve the submitted S0 A100 modeled INT4/S1-HD/S1-DP/S2 design (128 macros/engine, 0.4096 mm²; 728 points; 65.9% modeled favorable coverage; 1.97× stage maximum; original Eq.10 macro-energy). Signed Q4×K4 handling is outside the S2 128-macro core. Extra signed-bank storage in later source modules is an optional alternate physical mapping, not a mandatory correction. The submitted results must not be replaced by optional modeling extensions.

Source-grounded SRAM versus SRAM-CIM shared-prefix QKᵀ evaluation, anchored to the original OJCAS S0/S1-HD/S1-DP/S2 macro-area comparison.

## Scope

- **Original S0 GPU comparator:** the submitted A100 80GB 4-bit modeled score path; **additional** A100 FP16 HBM2e MAGMA/LLMCompass analysis is a distinct precision configuration. Do not substitute H100, FP16 or unrelated kernels for the manuscript's S0.
- **Separate two architectural endpoints:** (i) completed signed Q4K4 prefix QK scores at scale/softmax, and (ii) optional prefix context + logsumexp returned after FP16-V SRAM, streaming softmax and digital PV. The two have different area, bytes and latency and must not be conflated.
- Evaluate exact-prefix and compressed-prefix alternatives on the GPU as well as CIM; a compressed FP16 GPU operator source model is not a numerical substitute for matched Q4K4 GPU execution.
- Compare the original A100 S0, S1-HD, S1-DP and SRAM-CIM S2 on the paper's common published-macro area budgets and shared-prefix QKᵀ endpoint.
- Include prefix placement/refill, mapping, signed arithmetic, reduction, score transport and complete decoder latency.
- Preserve the paper's **macro-area** scope and the distinction between source model parameters and measurements; do not present model sensitivities as measured acceleration.
- Generate architectural comparisons and absolute numerical-quality evidence from preserved project data.

## Frozen ladder

1. **L0** Common geometry, provenance, units and physical-accounting contracts.
2. **L1** System path: placement, refill, score output/merge and end-to-end TPOT.
3. **L2** Original macro-area comparison: SRAM/6T/8T and SRAM-CIM under the submitted allocation budgets.
4. **L3** Quality and model profiles: archived R0–R3 results; suffix/GQA/MHA sensitivity.
5. **L4** Integrate L1–L3; regenerate the architecture sweep; publication figures/tables; reviewer responses.

Independent implementation branches: `work/system-path`, `work/area-digital`, `work/quality-workloads`. Integration: `integration/final-framework`.

All branches start from `main`. Read `docs/AREA_EVIDENCE_POLICY.md` first. Do not change another workstream's files; merge into integration only after executable tests.

## Source acquisition

The upstream research implementations live in other repositories and must be checked out in each developer workspace; never commit access tokens, gated model weights, original result archives or private dependency code into this **public** repository. Reuse through imports/adapters and pinned revisions:

- `micsedlab-afk/shared_prefix_cim` branch `task/sram2llm-sinangil7nm-20260919`, commit `a8a3f7238c471da2f3b7e6fe5647f8f25b8d84c4`
- `micsedlab-afk/SRAM2LLM`
- `micsedlab-afk/MAGMA-gpt`
- `micsedlab-afk/HBM4PiM`

Workstation-side code/results may supersede repository snapshots when identified and tracked in evidence manifests. Do not substitute unrelated FP16 GPU calibration for INT4 hardware results.

## Scientific rules

- Every stage reports its endpoint, units, evidence source, assumption class and overlap policy.
- Complete scores must include any signed/multi-partial merge before softmax.
- Signed processing/reconstruction is handled outside the submitted 128-macro S2 core. Count additional banks/evaluations only for explicitly distinct physical implementations; maintain separate accounting without overwriting original results.
- Report area consistently as the published complete-macro area used by the original manuscript.
- No fabricated measured times, PPL, power, or foundry area estimates.
- GPU-only and end-to-end numbers are labeled calibrated, modeled, bounded, or measured as appropriate.

Run local unit tests with `python -m unittest discover -s tests -v`.
