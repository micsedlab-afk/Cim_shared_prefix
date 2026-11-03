# Branch contracts and integration gates

All branches fork from the common main baseline. Never alter common source contracts without explicit integration review.

## Work/system-path
Ownership: `src/cim_shared_prefix/system.py`, `tests/test_system.py`, `docs/system_method.md`.
Inputs: `Workload`, signed mapping, completed-score transport endpoint, source-reported GPU stages and explicit bandwidths.
Outputs: prefix placement/refill, merge, transfer, per-stage nonoverlap/overlap timing and full decode token latency. Include meaningful zero-data and absent-metric handling.

## Work/area-digital
Ownership: `src/cim_shared_prefix/area_optimizer.py`, `tests/test_area_optimizer.py`, `docs/area_method.md`.
Inputs: published macro specifications; explicit digital MAC/core/scratch/controller cost sources; standard `AreaLedger`.
Outputs: feasible iso-area alternatives and constrained bank/compute allocation with winning-region analysis. No foundry synthesis.

## Work/quality-workloads
Ownership: `src/cim_shared_prefix/quality.py`, `src/cim_shared_prefix/profiles.py`, `tests/test_quality.py`, `docs/quality_method.md`.
Inputs: archived R0/R1/R2/R3 numerical evidence; source identity and MHA/GQA model geometry.
Outputs: absolute perplexities with uncertainty, profile/suffix sweep configs, explanation of baseline quality, no invented values.

## Integration/final-framework
Ownership: `src/cim_shared_prefix/runner.py`, `src/cim_shared_prefix/report.py`, `docs/reviewer_matrix.md`, publication outputs.
Integrate workstreams only after tests; preserve source/provenance; full 728-point sweep with additional model and digital baseline; 5 figures and 2 tables.

## Acceptance
- Unit tests pass; no hard-coded made-up calibration data.
- Model 32-bit per-score transfer at the standard large geometry equals 16 GiB for 32 layers.
- Signed split Q4/K4 storage and native call multipliers carried through area and latency.
- Every numerical scenario reports all four comparators at matched integrated area.
- Missing data are marked unsupported; no silently omitted transfers or zero-valued unsupported energy.
