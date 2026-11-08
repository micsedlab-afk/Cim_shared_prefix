# IEEE TETC submission audit and revision plan — 8 November 2026

**Target:** IEEE Transactions on Emerging Topics in Computing (TETC), regular technical-track paper, primary Track 1 (Emerging Computing Architectures and Paradigms), secondary topical alignment Track 2 (AI-Driven and Intelligent Computing Systems).

**Source baseline:** integration/final-framework at e93392ecd7415ecc3ee379f10a0e8b37c4430f98 (6 November 2026), original eight-page "Where Is the Sweet Spot for SRAM-CIM in Shared-Prefix LLM Attention?" PDF, and the merged TCAS-AI editorial packets docs/TCASAI_MINIMAL_REVISION_AND_SOTA_20261106.md and docs/TCASAI_SOURCE_ANCHORED_MANUSCRIPT_AUDIT_AND_PATCH_20261106.md. This plan changes no manuscript result, physical architecture, assumption, source file, or numerical record.

## Editorial decision

TETC is an appropriate technical venue because its Track 1 explicitly lists memory-centric/non-von-Neumann systems and Track 2 lists emerging AI accelerator architectures. Its accepted paper Sharma et al., "What, When, Where to Compute-in-Memory for Efficient Matrix Multiplication During Machine Learning Inference," IEEE TETC 13(3), 1215–1229 (2025), DOI 10.1109/TETC.2025.3574508, is a **direct methodological novelty comparator**, not simply favorable journal precedent. Sharma et al. already studied architectural CIM integration choices using analytical and iso-area-style methodology across matrix multiplication. Also review "Evaluating Compute in Memory Architectures for Matrix Multiplication: A Dataflow-Centric Perspective," ISPASS 2025, DOI 10.1109/ISPASS64960.2025.00056. Our differentiator must be **the shared-prefix runtime-K storage-versus-computation decision, with its repeated-query reuse, finite complete-prefix capacity, GQA geometry, conventional GPU/SRAM pointwise comparator, and numerical-fidelity evaluation**. Do not claim the first CIM sweet-spot, area-constrained CIM allocation, or KV-sharing-aware CIM hardware.

## Locked manuscript evidence — preserve

- Four configurations at equal *incremental published-macro area*: S0 original LLMCompass-predicted A100 INT4; S1-HD published 7-nm 6T SRAM; S1-DP published 7-nm 2RW 8T SRAM; S2 published 7-nm 128-macro/engine SRAM-CIM (0.4096 mm²/engine), external signed Q4/K4 handling as in submitted source model.
- Endpoint: modeled shared-prefix **QK^T score stage**. The pointwise comparator is min(S0, S1-HD, S1-DP), not S1 alone. Suffix attention, softmax/PV, projections, FFN and LM head are not included in the original timed endpoint.
- 728 = 13 batches × 7 shared-prefix lengths × 8 area budgets. Favorable points per 91 workload shapes at areas [1,2,3,4,6,8,12,16] mm²: [0,11,14,22,52,60,60,57]. Thus 65.9% *sampled-grid coverage* at **both 8 and 12 mm²**; 62.6% at 16 mm². This is not trace-weighted serving probability or a universal optimal area. Peak individually modeled QK-stage speedup is 1.97× at B=128, prefix=32k, area=16 mm²; peak coverage area and peak point differ.
- Source model: A100 1248 dense INT4 TOPS, 2039 GB/s HBM and 40 MB L2. One S2 engine 128 published macros at 0.0032 mm² each; original external sign-handling model. Periphery/digital conversion/interconnect area unqualified.
- S1-HD/DP capacities at 8 mm²: 27.84375 MiB / 5.71875 MiB. S1 ideal conflict-free banking and 4.3-TB/s assumed interface ceiling. S1 initial placement excluded from timed decode; S2 active-layer K supply/programming and partial-score GPU combination included according to source model.
- Original energy Fig.4: S1 *one tile write+read* versus S2 *published-macro compute access*. Partial local-memory energy only; source-model crossover near B=41 (HD) and B=25 (DP), not whole-system energy.
- Real archived Llama-3.1-8B R0–R3 evaluation: 96/96 original WikiText-2 units; disjoint calibration/test and 15 PG-19 excerpts with frozen ADC settings. Absolute WikiText subset PPL R0/R1/R2/R3: P512 6.699/6.721/6.798/6.840; P2048 5.481/5.564/5.743/5.666; P16384 5.154/5.256/5.425/5.485. At P2048 R3-R2 relative PPL -1.348%, paired CI [-3.294%,+0.663%] crosses zero. R1-R3 at P2048 +1.833% [0.050%,3.490%].

## Technical audit at 8 November

**Qualified/reusable:** original manuscript tables/figures as submitted-model evidence; reproduced original geometry and SRAM capacities; source-locked 728 coordinates and aggregate Table 5; executed 728-row *geometry/accounting* reconstruction; real archived Llama numerical-quality CSV with paired CIs; completed source-model placements and rate sensitivities; CI success on integrated HEAD e93392e (GitHub Actions, 6 November).

**Critical gap:** the repository's reconstructed 728 rows intentionally set per-coordinate original stage times, win flags and speedups to null. The original per-coordinate timing/winner map was **not independently regenerated**. A successful 728-row geometry run is not validation of all 728 original comparisons. Recover the generating script/raw points or reproduce exactly using the frozen published assumptions; retain old claims as historical modeled observations until such reproduction is complete.

**Other gaps:** no real matched optimized Hydragen/PAT A100/CUDA benchmark; no physically qualified external signed-handling or periphery area/latency; no measured complete-decoder TPOT or chip energy; no OPT/Mistral real R0–R3 inference (0/96 each). No 7-nm synthesized digital SRAM+MAC comparator. These MUST NOT be represented as measured or completed.

**Separate experiments:** 117/728 Pareto placement points (reduced K4 tile writes at unchanged modeled wave count) are not linked to original winning 276/728 points; 17.570→17.047 µs component delta at a selected area is not an overall QK or TPOT improvement; 256-macro signed-bank/four-unsigned-call design is a distinct physical sensitivity, not the submitted 128-macro design; 2304 conditional Llama/Mistral/OPT timing-sensitivity cases use alternative physical assumptions/endpoints and no qualified GPU TPOT. Do not mix these into the 65.9% statement.

## Manuscript transformation map

| Section | Action for TETC | Original evidence to retain |
|---|---|---|
| Title | Retain original working title; optional shorter subtitle "An Iso-Area Architectural Analysis" only after review | Core question |
| Abstract | Adapt November 6 audited Patch 1: highlight *selection boundary* and qualified QK stage; describe sampled-grid coverage and peak separately | 728, 65.9%, 1.97×, real numerical quality |
| Introduction | Emphasize hardware allocation decision, why prefix workload changes CIM/standard SRAM comparisons; sharpen claim against Sharma TETC 2025 and ISPASS 2025 | Existing motivation and 3 contributions |
| Related Work (II.C) | Extend audited Patch 2 to explicitly contrast Sharma 2025, ISPASS 2025, CIM-Tuner, Hydragen/PAT/MoSKA, Ouroboros, 2026 S²-CIM; use a compact comparison table if space | Existing references and accurate differentiation |
| Problem/Metrics (II.A, B, D) | Preserve equations and operational intensity I_K=16B op/byte; state decision indicator for S2 favorable x, coverage is count/91 on *uniform tested grid* | Eqs. 1–4, Table 1 |
| Methodology (III) | Preserve source macro and A100 parameters; adopt Patch 3 clarifying timed S1 vs S2 placement. Clearly delineate "published macro area" from total-chip area and BF16 numerics from INT4 timing. | Tables 2,3; Eqs. 5–12 |
| Results (IV.A) | Keep Table 5; replace overlapping exposition with audited Patch 4. Add explicit examples/phase-map once original row-level times recovered. In Fig.6, S1 winner alone is **not** the full Tconv comparator; labels/prose must reflect this. | Figures 2–4,6, Table 5 |
| Numerical fidelity (IV.B) | Insert absolute PPL 3×4 table and audited Patch 5. Clarify nonmonotonic R2/R3 paired CI and held-out subset. | Figure 5, Table 4, source quality |
| Design implications (IV.C) | Reorganize as architectural design rules with their assumptions and failure regimes. Report only sourced bounded sensitivity for S1 initial placement, S2 external support and GPU comparator improvement. Avoid universal 8–12 mm² recommendation. | Original IV.C mechanisms |
| Conclusion (V) | Replace via audited Patch 6, emphasizing scope-bound selection map and uncertainty; no whole-LLM speedup claim. | Original findings |
| Figures | Redraw Figure 1 at readable IEEE size; prefer 2D crossovers/heatmap over tiny 3D bars **only if raw source data are available**; enlarge Figs 2/6 labels; no digitizing/reconstructing results from figures. | Underlying data unchanged |
| Template | Migrate IEEE Access-style eight-page layout ("VOLUME XX") to TETC/IEEE Computer Society Transactions template via official IEEE Template Selector. | Content, citations, figures |

## Minimal defensible additions, ordered

**P0 — scientific/reproducibility before submission**
1. Recover exact original per-coordinate T_GPU, T_HD, T_DP, T_CIM and S2 favorable flag; run 728-point tests against Table 5 and named extrema. If source data irrecoverable, rebuild *original* frozen A100 INT4 and macro model and rerun with explicit provenance, documenting any discrepancies. Preserve original published claims for audit, do not silently substitute new numbers.
2. Show a meaningful phase boundary: 2D B×Lp decision region for at least selected area points and interpret engine count versus SRAM capacity/GPU efficiency. Report coverage as a sampled-grid property, identify cases where conventional wins, and distinguish max win from max coverage.
3. Establish a **novelty gap paragraph and comparison table** against Sharma TETC 2025/ISPASS 2025 and the recent LLM/CIM papers, making no first-of-kind claims.
4. Audit all original comparison assumptions and units: S1 cold placement vs S2 active programming, L2/HBM/GPU modeled utilization, Q4/K4 sign processing, score transport endpoint, macro-only area and energy. Present an explicit scope/assumption table.

**P1 — tight robustness and editorial value**
5. Once timings are available, evaluate at least one **small, labeled sensitivity overlay** for external S2 processing/transport and S1-placement amortization over output tokens. Show which favorable cases survive the assumed overhead; do not rewrite original 728 results.
6. Use source-supported GPU best-case/optimism sensitivity; only call it a competitive GPU baseline if actual matched optimized shared-prefix kernels are executed. PAT/Hydragen literature numbers with differing geometry/dtype are context, not speedups.
7. Add original absolute PPL table with paired CI; no need to repeat all new OPT/Mistral quality or an entire end-to-end decoder study for this QK-boundary paper.
8. Visually replace Figure 1 and repair figure legibility, placement and captions; compile with official TETC/IEEE Transactions style.

**P2 — optional strengthening only if results available**
9. One additional native GQA/MHA/SWA geometry or parameter-invariance check with explicit scope and source model. Existing 2304 conditional points cannot be relabeled validated TETC wins or substituted for original 728.
10. Real A100 CUDA comparison at a small set of matched B/P, native dtype and identical functional endpoint; useful strengthening, but no invented timing or H100/FP16-to-A100/INT4 substitution.
11. Supplementary artifact: original raw 728 points, source parameter JSON, quality CSV/provenance, figure-generation scripts, software environment, CI receipt and README.

## TETC packaging and submission

- Track 1 regular paper; no special-section deadline required. Technical tracks are open for continuing submissions: https://www.computer.org/digital-library/journals/ec/technical-tracks .
- Use official IEEE/Computer Society journal template and author guidance: https://www.computer.org/publications/author-resources ; https://www.computer.org/csdl/journal/ec/write-for-us/15071 .
- IEEE Computer Society regular Transactions baseline: 12 formatted double-column pages inclusive of references, with overlength fees beyond 12 as stated on the current publisher guidance. A **10–12-page target** is editorial, not permission to pad. Historical TETC policy permits up to 16 pages with mandatory charges; verify current live author portal before final upload.
- Retain compact relevant references; historical TETC technical-track guidance says <=45 bibliography items, subject to current journal-specific author page.
- Original editable LaTeX source and source figures are not in the currently audited GitHub repository/available uploads. Locate actual manuscript project before producing camera-ready layout and visual diff; do not reconstruct text from PDF as a substituted source.
- Prepare PDF, source TeX, accurate supplementary data/code, author metadata/ORCID and manuscript declaration; obtain coauthor approval. Upload under the appropriate "regular paper"/technical track via https://mc.manuscriptcentral.com/tetc-cs or the linked IEEE Author Portal. Traditional hybrid subscription submission is an option; OA choice optional. No simultaneous publication review.
- Prior submission/rejection need not be written into paper text. If submission system asks about prior submission history, answer truthfully.
- Keep acknowledgement/AI-use disclosures consistent with IEEE publisher policies and actual assistance; review original disclosure wording without falsely asserting non-use.

## Decision gate

**Journal fit: yes. Submission readiness: not yet.** Earliest scientifically defensible freeze requires (i) a genuinely reproducible original pointwise score-stage comparison, (ii) an explicit novelty contrast against 2025 TETC analytical CIM work, (iii) carefully scoped performance/area/energy and quality boundaries, (iv) readable TETC-compliant LaTeX layout. More hypothetical deadline sweeps, extra models without inference, uncontrolled digital synthesis and broad full-decoder claims are not prerequisites for this focused manuscript.

### Evidence links

- Integrated baseline: https://github.com/micsedlab-afk/Cim_shared_prefix/tree/integration/final-framework
- November 6 source-anchored draft: docs/TCASAI_SOURCE_ANCHORED_MANUSCRIPT_AUDIT_AND_PATCH_20261106.md
- November 6 minimal draft: docs/TCASAI_MINIMAL_REVISION_AND_SOTA_20261106.md
- Original manuscript baseline lock: docs/OJCAS_MANUSCRIPT_BASELINE_LOCK_20261105.md
- Row-level reconstruction caveat: src/cim_shared_prefix/ojcas_manuscript_baseline.py
- GPU gaps: docs/GPU_PUBLICATION_GATES.md; results/gpu/EVIDENCE_AUDIT.json
- Llama quality: results/quality/absolute_quality.csv; results/quality/paired_effects.csv
- 2025 TETC CIM analysis: https://doi.org/10.1109/TETC.2025.3574508
- ISPASS iso-area CIM: https://doi.org/10.1109/ISPASS64960.2025.00056
- CIM-Tuner: https://doi.org/10.23919/DATE69613.2026.11539276
- S²-CIM: https://doi.org/10.1109/TCSI.2026.3742618
