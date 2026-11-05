# OJCAS-2026-0120 — Submitted manuscript baseline and reviewer extensions

**Authoritative experiment:** *Where Is the Sweet Spot for SRAM-CIM in Shared-Prefix LLM Attention?* Submitted 2026; rejection comments received 3 November 2026.

## Manuscript evidence — preserve as submitted

S0: modeled A100 original 4-bit attention baseline. S1-HD: 7-nm high-density conventional SRAM. S1-DP: 7-nm 2RW SRAM. S2: 7-nm SRAM-CIM shared-prefix QKᵀ; remaining attention and unique suffix run on conventional execution path.

**Signedness boundary:** The paper's 128-macro S2 engine counts the unsigned 7-nm macro geometry for shared-prefix score computation; signed-Q4/K4 sign handling and partial reconstruction occur outside the core in the architectural/numerical path. Keep this distinction in the methodology. The original timing and macro-energy figures remain those of the **submitted model**, without retroactively charging a second bank or multiplying the published Eq. (10) coefficient. The original manuscript's hardware metric is published-macro area; the numerical sign-handling boundary and the original 728-point results remain unchanged.

### Locked values

| Property | Original manuscript |
|---|---:|
| S2 macros / 128-token engine | **128** |
| S2 engine macro-only area | **0.4096 mm²** |
| Sweep | **728 = 13 B × 7 prefix lengths × 8 area budgets** |
| Favorable S2 points / 91, area 1/2/3/4/6/8/12/16 mm² | **0 / 11 / 14 / 22 / 52 / 60 / 60 / 57** |
| Modeled favorable share at area 8 and 12 mm² | **60/91 ≈ 65.9%** |
| Maximum modeled *shared-prefix QKᵀ* stage speedup | **1.97×** |
| S1-HD / S1-DP original partial macro-energy crossover | **B≈40.86 / 25.45** |
| S1-HD / S1-DP capacity at area 8 mm² | **27.84375 / 5.71875 MiB** |

These numbers are archived **modeled results of the submitted paper**. They are not physical-silicon measurements, end-to-end decoder speedups, or integrated chip energy. Exact per-coordinate S0/S1/S2 time samples are not available in the present reconstructed repository, so no new per-coordinate timing values or revalidation claim is invented.

Runnable source and guard: `src/cim_shared_prefix/ojcas_manuscript_baseline.py`, `scripts/run_ojcas_manuscript_baseline.py`, `tests/test_ojcas_manuscript_baseline.py`. Outputs: **submitted_OJCAS_728_model_grid.csv**, **submitted_OJCAS_baseline_lock.json**.

## Previous 256-macro / fourfold-energy exploration

The code and results in `src/cim_shared_prefix/ojcas_paper_728_audit.py` and `docs/OJCAS_2026_0120_EXECUTED_728_AUDIT_20261105.md` belong to an **alternative explicit on-macro signed-bank mapping sensitivity**. It provisions separate K+ and K− banks and charges four native unsigned operations. This is a valid conditional architecture to analyze, but it **is not the submitted S2 implementation** and cannot displace its area, latency, energy, favorable region or associated scientific results. The alternate programming/partial-score transport floors cannot be asserted as unconditional problems in the original model because processing location and interface schedule differ.

The original shared-prefix QK study remains the reference for any subsequent placement or latency extension under the same macro model.

## Reviewer-driven extensions, without changing the control

1. **R1 placement:** document the already timed S2 active-layer transfer and row programming, separately quantify S1 excluded pre-decode placement. Specify how external sign processing/partial reconstruction is implemented, where it runs and its actual energy/latency/area or documented sensitivity; preserve the original model values in parallel.
2. **R1/R3 full decoder:** append shared-suffix, attention softmax and PV, projections, FFN, norms and GPU remainder to the **unchanged** QK-stage model; preserve stage-only speedup and report full-TPOT separately.
3. **R1 numerics:** retain original evaluated Llama R0–R3 and absolute PPL with confidence intervals; explain nonmonotonic R2→R3 without modifying hardware precision or fitting the test samples.
4. **R1/R3 area definition:** retain published 7-nm complete-macro areas, the original 128-macro engine and the paper's explicitly **macro-area** comparison. Report modeled area with that exact scope.
5. **R2 design boundary:** present original S0/S1-HD/S1-DP/S2 favorable points and clarify the macro-capacity/suffix/batch tradeoffs. Analyze added costs as sensitivity overlays, never replacements for original curves.
6. **R3 generality:** extend the original A100 INT4/S0/S1/S2 score-stage comparison to other native model geometries and prefix/suffix ratios. Keep any A100 FP16 study separately labeled.

**No unrelated replacements:** speculative centroid compression, 135M small-model selective-layer policies and H100 measurements are optional, separately labeled research. They must not change the OJCAS original assumptions or be promoted as equivalent performance evidence.
