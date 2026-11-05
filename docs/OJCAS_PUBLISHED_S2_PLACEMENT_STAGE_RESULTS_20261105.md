# OJCAS reviewer extension — published-input S2 placement (2026-11-05)

This executed extension retains the submitted OJCAS-2026-0120 S0/S1-HD/S1-DP/S2 comparison **unchanged**. S2 keeps **128 macros/engine (0.4096 mm²)**, original external signed processing, and the original 728 modeled QK-stage results (60/91 favorable at 8 and 12 mm²; maximum 1.97×). These original speedups are not recalculated or replaced.

## What was executed

[GitHub Actions run 42051149482](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/42051149482): 25 dedicated tests passed. Artifacts contain 728-row stage breakdown and 5,824-row output/reuse sensitivity with machine-readable manifests. The condensed numbers are preserved in `results/reviewer/ojcas_published_S2_placement_stage_delta_20261105.json`.

The only newly modeled latency components are **the original S2's active-layer K4 supply from A100 HBM, 64-row programming, and native macro accesses**. Inputs are from submitted Table 2 / Section III-B: nominal A100 HBM **2,039 GB/s**, original row-program time **200 ps/row × 64**, and published native CIM service **5.5 ns/access**. The source model counts one 128-token key tile = 65,536 B, and original GQA four queries per KV-head per batch.

The new policy may retain *k* complete prefix layers in the existing engines, reserving the remaining engines for streamed layers. Cold programming is charged once, then every unpinned layer is streamed again for each generated batch step. The new and original configurations use the same serial source timing convention. GPU d64 merging, signed handling outside the original core, suffix and other decoder operations are unchanged common costs; their absolute latency is **not inferred**.

## Results for P=2048, O=128 (µs per batch decode step)

| Aadd | B | Resident layers | Fully streamed three-component source | Placement-selected three-component source | Change |
|---|---:|---:|---:|---:|---:|
| 8 mm² | 1 | 0 | 17.569918 | 17.569918 | 0 |
| 12 mm² | 1 | 0 | 17.569918 | 17.569918 | 0 |
| 16 mm² | 1 | 1 | 17.569918 | 17.046976 | −0.522942 (−2.976%) |
| 16 mm² | 4 | 1 | 19.681918 | 19.158976 | −0.522942 (−2.657%) |

At 8 mm², P=512, B=1, pinning three complete layers improves the same source component from **5.227679 to 4.806898 µs** (−8.049%). At the manuscript's original **maximum-speedup point**, B=128, P=32768, 16 mm², no complete layer can be pinned and **there is no gain**.

**Across 728 original coordinates:** source-stage reduction is positive for **117/728**, exactly the cases with *fewer K4 program writes and no extra native compute waves*. Positive cases by increasing budget 1,2,3,4,6,8,12,16 mm²: **0,0,0,13,13,26,26,39**. Mean source-component reductions *among improved points only* are respectively 1.083%, 2.166%, 2.332%, 4.664% and 4.895% for budgets 4,6,8,12,16 mm². They are not full S2 QK speedups or end-to-end TPOT.

### First-layer pinning boundary from the original source inputs

At P=2048, B=1 and 128 output steps, the effective HBM bandwidth below which pinning one layer becomes beneficial under this **serial source model** is:

- **8 mm²: below 193.33 GB/s**.
- **12 mm²: below 975.88 GB/s**.
- **16 mm²: any finite bandwidth**, because one layer can be pinned without adding compute waves and its subsequent programming is avoided.

The manuscript's **2,039-GB/s nominal HBM** value therefore selects no pinned layer at 8 or 12 mm². This threshold is model-specific; sustained data rates on A100 are not measured here.

### Output length and cross-batch prefix reuse

Executed eight scenarios (O=1/32/128/512, prefix held for R=1 or 8 consecutive batches) covering **5,824 source-model coordinates**. For a prefix used for **one single decode step** (O1/R1), none gains from persistent residency. For O32/R1, O128/R1, O512/R1 or any O/R yielding at least eight steps in this tested matrix, the same **117/728** coordinates reduce the identified three-component source time. This assumes that the resident prefix is retained across batches and not evicted by other work.

## Reviewer significance and uncertainty

- **R1 placement:** identifies precisely when initial versus repeated S2 programming is avoided; original active-layer programming is already included, so no double charging.
- **R2 sweet spot:** reveals the discrete engine-capacity/batch/HBM-bandwidth boundary for optional whole-layer pinning.
- **R3 latency:** separates source K-transfer, programming and native array execution and produces a falsifiable bandwidth threshold.
- **Area:** the 128-macro engines are unchanged. New external controller/interconnect area is not synthesized, so the original macro-area comparison remains macro-only; *policy-level* resource gains are subject to implementation overhead.

At 16 mm² the original 39 engines consume 15.9744 mm², leaving only 0.0256 mm² within the macro budget. This **arithmetic slack is not evidence that any additional controller fits**. The 0.522942 µs/step source delta is the maximum *additional non-overlappable policy overhead* that can be introduced at the P2048/B1/16-mm² point before the three-component serial benefit vanishes.

**Unqualified:** measured A100 INT4 kernels, physically achieved HBM/programming overlap, controller area/energy, external signed reconstruction latency, source-to-GPU score transfer, S0/S1/S2 pointwise full time, actual decoder TPOT, and competitive digital SRAM+MAC implementation. No new hardware speedup, energy or quality result is claimed. The original R0–R3 fidelity evidence is preserved.

## Executable sources

`src/cim_shared_prefix/ojcas_published_placement_delta.py`, `src/cim_shared_prefix/ojcas_placement_extension.py`, `scripts/run_ojcas_published_placement_delta.py`, `scripts/run_ojcas_output_reuse_sweep.py`; unit tests and archived results accompany them.
