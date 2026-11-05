# Cim_shared_prefix — OJCAS revision framework

Integrated source-grounded SRAM, SRAM-CIM and SRAM + digital-MAC analysis for shared-prefix LLM attention.

## Scope

- **Primary GPU comparator:** A100 80GB FP16 HBM2e MAGMA/LLMCompass source model, preserving original native GQA and byte-correct operator counts. H100 measurements belong to a separate calibration study and must never substitute for A100.
- **Separate two architectural endpoints:** (i) completed signed Q4K4 prefix QK scores at scale/softmax, and (ii) optional prefix context + logsumexp returned after FP16-V SRAM, streaming softmax and digital PV. The two have different area, bytes and latency and must not be conflated.
- Evaluate exact-prefix and compressed-prefix alternatives on the GPU as well as CIM; a compressed FP16 GPU operator source model is not a numerical substitute for matched Q4K4 GPU execution.
- Compare GPU, SRAM-only and SRAM-CIM at fully documented modeled area. Include SRAM+digital MAC only as a matched-area implementation when complete comparable physical area evidence exists; otherwise publish a conditional digital break-even threshold.
- Include prefix placement/refill, mapping, signed arithmetic, reduction, score transport and complete decoder latency.
- Classify all physical area evidence. Sensitivity parameters never qualify as physically comparable area. **No 7-nm synthesis and no invented digital-MAC area.**
- Generate architectural comparisons and absolute numerical-quality evidence from preserved project data.

## Frozen ladder

1. **L0** Common geometry, provenance, units and physical-accounting contracts.
2. **L1** System path: placement, refill, score output/merge and end-to-end TPOT.
3. **L2** Iso-area: SRAM/6T/8T, signed SRAM-CIM and digital MAC; constrained allocation.
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
- Signed mapping expands K storage and array evaluations according to the actual numerical mapping.
- Budget includes macros and any support logic charged outside published macro-area figures.
- No fabricated measured times, PPL, power, or foundry area estimates.
- GPU-only and end-to-end numbers are labeled calibrated, modeled, bounded, or measured as appropriate.

Run local unit tests with `python -m unittest discover -s tests -v`.
