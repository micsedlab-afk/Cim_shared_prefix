# OJCAS reviewer placement and scope extension — 5 November 2026

The submitted S2 model is fixed: 128 macros per engine, 0.4096 mm2 per engine, unchanged 728 coordinate results and original 1.97x modeled QK-stage maximum. Original signed handling remains external to the CIM core.

## Execution

[All 16 independent tests passed](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/42047435257). The workflow artifact contains 728 placement rows and a separate S1 cold-placement and decoder-scope JSON. A checked summary is in `results/reviewer/ojcas_original_engine_summary_20261105.json`.

## Finite residency result

For P2048, B1, 128 decode steps and illustrative normalized write cost lambda=10:

| Budget | Original engines | Selected resident layers | Compute waves per step, before to after | Write reduction | Normalized objective gain |
|---|---:|---:|---:|---:|---:|
| 8 mm2 | 19 | 1 | 32 to 187 | 3.10% | 0.073% |
| 12 mm2 | 29 | 1 | 32 to 63 | 3.10% | 2.480% |
| 16 mm2 | 39 | 2 | 32 to 92 | 6.20% | 4.998% |

**Parameter-free Pareto result:** Across the 728 original coordinates, 117 admit a feasible whole-layer pinning policy that strictly reduces packed-K4 tile programming while leaving the per-step compute-wave count unchanged. Counts by the original eight area budgets (1,2,3,4,6,8,12,16 mm2) are 0, 0, 0, 13, 13, 26, 26, 39. This proof covers those two logical resource counts; external control overhead and physical execution remain unmeasured.

At 8 mm2, the threshold write-to-compute ratio for pinning one layer is 9.764 at B1, 39.055 at B4, and 78.110 at B8. Original S2 active-layer programming is not charged again.

## S1 initial placement and decoder scope

At 8 mm2, P2048, original S1-HD/DP cold packed-K4 placement is 29,163,520/5,963,776 bytes. At 128 decode steps, these amortize to 227,840/46,592 bytes per batch step; eight consecutive batches sharing the prefix reduce each by eight. At a hypothetical 1 TB/s interface, the R1 amortized times are 0.22784/0.046592 microseconds.

With original maximum QK-stage modeled speedup 1.97x, the no-added-cost Amdahl decoder ceiling equals 1/(1-f+f/1.97): 1.02524x, 1.05179x, and 1.10923x at hypothetical baseline QK shares f=5%, 10%, and 20%.

All pinning gains are normalized logical cost sensitivities. The 1 TB/s rate and f values are unmeasured. No physical TPOT, GPU speedup, energy advantage, or novel circuit is claimed. Original OJCAS results and R0–R3 numerical evidence are preserved.
