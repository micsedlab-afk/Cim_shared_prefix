# TCAS-AI — source-anchored manuscript audit and surgical revision

**Basis and status (6 November 2026).** Read and visually inspected all eight pages of the attached PDF, *Where Is the Sweet Spot for SRAM-CIM in Shared-Prefix LLM Attention?* (PDF generated 5 November 2026); checked the active `micsedlab-afk/Cim_shared_prefix` `integration/final-framework` repository at commit `85b2e10018ef3377211b86c9e41eaae8fb109a27`, including original-model locks, current timing extensions, archived quality statistics, and regression receipts; and cross-checked directly relevant 2024–2026 papers and NVIDIA's A100 specification. This packet is an editorial revision of the **actual attached manuscript**. No data, original architecture, figure or experiment is replaced here. There is no original editable LaTeX project among the supplied files, so this is a precisely anchored text patch rather than a reconstructed, potentially unfaithful PDF.

## A. Decision and evidence audit

**Central article claim.** At equal **incremental published-macro area**, for shared-prefix **QKᵀ** and the submitted 4-bit LLMCompass/A100 source model, the preferred option changes with batch size `B`, shared-prefix length `Lp`, and added macro area `Aadd`. The conventional comparator is the **pointwise minimum** of `S0`, `S1-HD` and `S1-DP`; the fourth candidate is the submitted `S2`. The research output is a workload-dependent **selection boundary**, not a new SRAM cell, a new shared-prefix algorithm, or measured end-to-end decoder acceleration.

**Preserve without alteration:** S0 A100 INT4 roofline/model (1,248 dense INT4 TOPS, 2,039 GB/s HBM and 40-MB L2), S1-HD/DP 7-nm published SRAM configurations, S2 128 complete macros/engine (0.4096 mm²), original external signed-handling model, all 728 workload-area points, Fig. 2–6 numerical results, Table 5, macro-level energy ratios and R0–R3 quality experiments. The A100 dense INT4 specification is independently supported by NVIDIA's Ampere architecture documentation. Any A100 FP16, H100, 256-macro sign-bank, compressed-prefix or conditional model is outside the fixed manuscript baseline.

**Verified headline values:** 13 `B` × 7 `Lp` × 8 `Aadd` = 728; Table 5 S2-winning counts [0, 11, 14, 22, 52, 60, 60, 57] per 91 shapes for budgets [1, 2, 3, 4, 6, 8, 12, 16] mm²; 60/91 = **65.9%** at BOTH 8 and 12 mm²; 57/91 = **62.6%** at 16 mm². These are **fractions of the uniformly sampled design grid** rather than workload-weighted serving wins. Maximum individual modeled QK-stage speedup **1.97×** occurs at B=128, P=32k, Aadd=16 mm². The best coverage budget and best individual point differ. At Aadd=8 mm² S1-HD holds 27.844 MiB, covering 86.9% of P=2k's 32-MiB K footprint but only 5.4% of P=32k's 512-MiB footprint. The paper's partial macro-energy crossovers remain approximately B=41 (S1-HD) and B=25 (S1-DP). Neither is total-system energy.

**Quality record:** archived actual Llama-3.1-8B R0–R3 evaluation, eight disjoint WikiText-2 held-out sequences, 1,024 scored tokens per arm and prefix, plus 15 held-out PG-19 excerpts under frozen WikiText-derived per-layer ADC calibration. `results/quality/absolute_quality.csv` and `paired_effects.csv` corroborate Table 4 and Figure 5. WikiText-2 absolute PPL:

| Prefix | R0 BF16 | R1 Q4/K4 | R2 mapped ADC | R3 circuit response |
|---:|---:|---:|---:|---:|
| 512 | 6.699 | 6.721 | 6.798 | 6.840 |
| 2,048 | 5.481 | 5.564 | 5.743 | 5.666 |
| 16,384 | 5.154 | 5.256 | 5.425 | 5.485 |

At P=2k, R3 is numerically below R2; the paired **R2→R3** relative-PPL 95% CI is **[−3.294%, +0.663%]** (point estimate −1.348%), which spans zero. The increase in R3 versus R1 is **+1.833%** (95% paired CI [+0.050%, +3.490%]) at P=2k. This distinction should remain explicit. These are **subset PPL**, not the full WikiText-2 corpus baseline; R0 PPL differences across prefix lengths are partly differences in evaluation context and should not be interpreted as a model improving merely by adding prefix tokens.

**Evidence limitations that must not be paper claims:** the repository does not contain a newly executed, fully matched A100 INT4-versus-CIM timing table with the 728 original per-coordinate values; original published figures remain the reported modeled outcomes. The reported stage time omits suffix attention, softmax/PV, projection, FFN and LM head because the stated endpoint is shared-prefix QKᵀ. The 117/728 new placement candidates mean *reduced K4 tile programming at no increase in modeled native compute waves*; it is **unknown** how many of these overlap the original Table 5's 276 S2-favorable coordinates. The published-input stage delta at P=2k/B1/Aadd=16 mm² (17.570→17.047 μs per step) covers **K4 supply + row programming + native macro execution only**. It cannot be added to or subtracted from every original S2 speedup without pointwise records and a reconciled schedule. Consequently **do not include 117/728 or 0.523 μs as a new claimed extension in the main eight-page manuscript**. Retain these in internal research material, and use the qualitative finite-residency insight only if necessary.

## B. What the PDF already does well — avoid redundant revisions

- Pages 1–2 already explain why larger `Aadd` benefits **both** SRAM residency and SRAM-CIM engine count, and they explicitly compare `S2` with the pointwise minimum `S0/S1-HD/S1-DP`. Keep Equations (1)–(4), Table 1 and contribution structure.
- Page 3 already derives **I_K = 16B op/byte** for g=4 and explains that K working-set size is independent of B. Keep Equation (2), not a second derivation.
- Pages 4–5 already disclose **S1 perfect banking, 4.3 TB/s ceiling, initial placement excluded**, and **S2 active-layer K refill, 200-ps/row programming and GPU d64 reduction included**. Clarify their boundary in one sentence; do not add a speculative processor, peripheral or sign-plane design.
- Pages 5–7 already contain R0–R3 decomposition and calibrated/held-out evaluation. The deficiency is the **missing absolute PPL values beside relative changes**, not a need to redo the inference pipeline.
- Pages 6–7 already discuss the Table 5 nonmonotonicity and Figure 6 roofline; replace overlapping text with **one explicit answer to the area sweet-spot question** rather than adding another 'three regimes' section.

## C. Required edits, ordered by impact

| Priority | Exact location in attached PDF | Actual issue | Minimal action |
|---|---|---|---|
| P0 | Page 3, Section II.C, last CIM-related paragraph | Many works are enumerated without an explicit technical comparison; newest S²-CIM absent | Replace that paragraph with Patch 2; add one verified new citation |
| P0 | Page 6, Section IV.A directly after Table 5 | Coverage plateau is reported but 'sweet spot' is insufficiently defined and peak speedup is elsewhere | Replace two overlapping paragraphs with Patch 4 |
| P0 | Page 2, Figure 1 | Figure is visually inconsistent: overlapping/undersized labels, ornamental arrows/wall, inconsistent CiM/CIM capitalization; the axis sketch is not a reproducible roofline | Redraw as a simple S0/S1/S2 architecture comparison with clean labels, **not** a quantitative performance claim. Keep same roles and dataflow. |
| P0 | Page 7, Section IV.B after first WikiText-2 result paragraph | Relative PPL alone obscures magnitude and baseline | Insert compact absolute-PPL table + Patch 5 (no new inference) |
| P1 | Page 4, end of III.A / start III.B | Readers may conflate excluded S1 *cold* placement with modeled S2 *active-layer* refill | Insert Patch 3 as a two-sentence assumption boundary |
| P1 | Page 1 abstract and page 2 contributions | Present work already has an adequate abstract and objective, but 'sweet spot' could be defined more sharply | Replace abstract with Patch 1 **only if needed**; the new text retains all evidence and scope |
| P1 | Page 8, Conclusion | Repeats headline figures without distinguishing coverage from peak and stage scope | Replace conclusion paragraph with Patch 6; preserve original numerical conclusion |
| P2 | Page 5, Fig. 2 | Four 3-D bar clusters are difficult to read at IEEE two-column scale | Prefer improving font/crop from the original figure source; **do not reconstruct 728 numerical values from screenshots** |
| P2 | Page 7, Fig. 6 | Dotted/colored curves and legend are small; figure describes `S1` winner rather than necessarily full pointwise `Tconv` | Enlarge labels if original artwork available; ensure prose explicitly distinguishes S1-only comparison from Eq. (3) |

### Patch 1 — Abstract (complete ready-to-paste replacement, optional)

> Shared-prefix LLM decoding exposes a choice between storing key vectors in additional SRAM and using the same area for SRAM compute-in-memory (CIM) attention-score computation. The preferred allocation depends on shared-key reuse, storage capacity, and conventional GPU efficiency. This paper characterizes that selection boundary for shared-prefix QKᵀ using an LLMCompass-predicted NVIDIA A100/HBM path (S0), high-density SRAM (S1-HD), dual-port SRAM (S1-DP), and a published 7-nm SRAM-CIM macro (S2), under a 4-bit source timing model. We compare all alternatives at equal added published-macro area across 728 combinations of batch size, prefix length, and area budget. S2 has lower modeled QKᵀ latency than the fastest conventional configuration for 65.9% of the sampled workload points at 8 and 12 mm², while the maximum individual modeled speedup is 1.97× at 16 mm². The difference between favorable coverage and peak speedup reflects the competition between finite SRAM key residency, discrete CIM engine parallelism, and batch-dependent GPU utilization. We also quantify shared-prefix K traffic reduction and compare SRAM read/write with CIM computation energy at the macro level. For Llama-3.1-8B, a separate R0–R3 numerical evaluation isolates Q4/K4 quantization, ADC quantization, and published circuit-response effects using ADC parameters calibrated on one WikiText-2 subset and frozen for disjoint WikiText-2 and PG-19 evaluation. Relative to R1 Q4/K4 arithmetic, the full R3 model increases WikiText-2 perplexity by approximately 1.8% for 512- and 2,048-token prefixes and 4.4% for 16,384-token prefixes.

**Audit:** no full-decoder speedup, no measured A100 CUDA run, no full-system energy, no novel circuit claim. The current abstract is already sound; this is primarily a precision/style replacement, not a required wholesale rewrite.

### Patch 2 — Section II.C, final comparison paragraph (high priority)

Replace the **last CIM architecture paragraph** beginning “Attention-oriented CIM architectures address the same data-movement pressure ...” and ending “... pointwise in terms of shared-prefix QKᵀ latency.” Retain the first two prefix-kernel paragraphs and existing references to the underlying circuit studies; citations to individual circuit papers may remain in a shortened first sentence. Insert:

> Existing attention-oriented CIM accelerators differ in their treatment of precision, KV state, and execution scope. Analog attention acceleration [8], KV-CIM [9], fabricated eDRAM attention processing [10], HyAtt [11], CINELL [12], HARDSEA [25], the attention-score CIM macro in [26], and UniCAIM [13] target local arithmetic or sparse-attention processing using different memory technologies. Ouroboros [14] and CHIPSMORE [16] additionally address inference scheduling and KV organization at the system level. CIM-Tuner [15] optimizes SRAM-CIM compute/storage allocation and mapping within an accelerator, while the recently published S²-CIM [new citation] demonstrates a fabricated KV-sharing-aware CIM accelerator with access- and computation-level specialization. Conventional prefix-aware GPU execution is addressed by Hydragen [4], MoSKA [7], and PAT [6]. The question considered here is narrower: for the *same shared-prefix QKᵀ workload and an equal incremental published-macro-area budget*, when should the area support conventional SRAM K residency, and when should it support CIM arithmetic? This cross-alternative boundary is evaluated against the pointwise fastest S0/S1-HD/S1-DP configuration, with numerical fidelity treated separately from timing.

**Technical boundary:** CIM-Tuner has already studied constrained CIM resource allocation; do **not** claim 'first area-optimal mapping'. S²-CIM has already demonstrated KV-sharing-aware CIM in a chip; do **not** claim 'first KV-sharing accelerator'. PAT has real GPU inference benchmarks; do **not** claim that the modeled A100 S0 matches or outperforms its measured kernel. Different process/precision/endpoints cannot be put on a single speedup or energy-per-token leaderboard.

**New bibliographic entry (use a BibTeX key; let IEEE assign the number) — verified publisher metadata:** B. Liu *et al*., “S²-CIM: A 28-nm 68.9-μJ/Token Digital CIM Accelerator With Statistics-Aware Access and Computation for KV Sharing-Enabled LLM Inference,” *IEEE Transactions on Circuits and Systems I: Regular Papers*, early access, 19 Oct. 2026, doi: 10.1109/TCSI.2026.3742618. **Editorial check:** the publisher's title includes '68.9 μJ/Token' while its abstract states an average '172.5 μJ/token'. To avoid perpetuating this apparent title/abstract mismatch, cite the paper without repeating either energy figure in our text.

### Patch 3 — Section III.A/B, assumption boundary (two sentences)

Insert after the existing sentence “prefix placement is completed before the timed decode interval,” or use these two sentences to replace the current adjacent S1/S2 placement explanations without duplication:

> The shared-prefix K state is assumed to be available after conventional prefill; initial placement into S1 is therefore treated as a pre-decode operation and excluded from the reported QKᵀ latency. In contrast, S2 active-layer K transfer, 64-row programming, CIM computation, and GPU combination of the two d=64 partial scores are included in T_CIM, as specified in Section III.B.

**Audit:** This states the existing modeling boundary clearly. It does **not** imply S1 placement is physically free. A numerical placement time would require a specified load path and reuse horizon; none should be invented. Signed-Q4/K4 numerical decomposition remains per Equations (11)–(12), outside the original 128-macro engine definition.

### Patch 4 — Section IV.A, interpretation immediately following Table 5 (high priority)

Replace the two adjacent paragraphs beginning “The complete coverage is summarized in Table 5 ...” and “Larger Lp increases the shared-prefix K footprint ...” with one concentrated paragraph. Keep the detailed Fig. 3 and Fig. 6 interpretations that follow in the paper.

> Table 5 identifies the *coverage* sweet spot for the evaluated workload grid. The number of S2-favorable cases increases from 22/91 at 4 mm² to 60/91 at 8 mm², remains 60/91 at 12 mm², and decreases to 57/91 at 16 mm². These values count equally weighted (B,Lp) combinations and should not be interpreted as the fraction of requests accelerated in a particular serving trace. The change is consistent with two competing area benefits: additional complete CIM engines reduce the number of prefix tile waves, while larger conventional SRAM capacity increases the fraction of K accesses served locally. Batch size also affects both shared-key reuse (I_K=16B op/byte) and conventional GPU matrix utilization. Thus 8–12 mm² maximize the *sampled favorable region*, whereas the maximum individual modeled speedup of 1.97× occurs at B=128, Lp=32k and 16 mm². The preferred area is workload-dependent; the coverage maximum is not a universal area optimum.

**Audit:** the underlying winner map is preserved, and no new numerical values are derived. This replacement avoids repeating the same mechanism again in Section IV.C. The actual original *per-coordinate timing records* are not reconstructed in the active repo, so no further precision or newly claimed crossover percentages should be introduced.

### Patch 5 — Section IV.B, absolute Figure 5 PPL and R2→R3 interpretation (high priority)

Insert after the sentence reporting relative R1→R3 and R0→R3 WikiText-2 perplexity changes. A tiny table may be used in the column, or keep the same values inline if page count is tight:

| Lp | R0 | R1 | R2 | R3 |
|---:|---:|---:|---:|---:|
| 512 | 6.699 | 6.721 | 6.798 | 6.840 |
| 2k | 5.481 | 5.564 | 5.743 | 5.666 |
| 16k | 5.154 | 5.256 | 5.425 | 5.485 |

> Table X gives the absolute perplexities underlying the relative changes in Fig. 5 for the eight held-out WikiText-2 sequences (1,024 scored tokens per configuration). These values describe the evaluated subset, not a complete-corpus benchmark. At Lp=2k, R3 yields a lower point-estimate PPL than R2 (5.666 versus 5.743), but the paired 95% interval for the R2→R3 relative change includes zero; the observed decrease therefore does not establish a systematic benefit from circuit response. A near-zero median per-token ΔNLL is compatible with an increase in pooled perplexity because positive-tail errors contribute to the mean NLL.

**Audit:** The actual source is `results/quality/absolute_quality.csv` and `paired_effects.csv`. No new Llama inference or ADC calibration is performed. PG-19 already appears in Fig. 5 and remains unchanged; do not invent its absolute PPL values from the WikiText-2 archive.

### Patch 6 — Conclusion (replace the existing paragraph spanning pp. 7–8)

> This work identifies the workload-dependent selection boundary between conventional SRAM key residency and SRAM-CIM computation for shared-prefix QKᵀ under equal added published-macro area. Across 728 modeled workload–area configurations, S2 has lower QKᵀ latency than the pointwise conventional reference in 65.9% of sampled (B,Lp) cases at 8 and 12 mm²; the largest individual modeled stage speedup, 1.97×, occurs at 16 mm². These results follow from competing changes in shared-key residency, discrete CIM engine parallelism, and conventional GPU matrix efficiency. The macro-level energy comparison provides a separate batch-dependent access/computation tradeoff. Llama-3.1-8B numerical evaluation quantifies the Q4/K4, ADC, and published circuit-response contributions using calibration frozen before WikiText-2 and PG-19 tests. Together, the results characterize when added published-macro area is more effective for SRAM locality or near-key attention-score computation. The timing conclusions apply to the modeled shared-prefix QKᵀ stage.

This is a replacement, **not** a second conclusion paragraph; the existing original results and precision remain unchanged.

## D. Figure and citation audit (actual PDF layout)

- **Fig. 1, page 2 — must clean up.** Current text includes an ornamental wall graphic, mixed labels and small red/green/blue boxes, crowded 3-way arrows and a pseudo-roofline that is not based on the measured/model roofline shown in Fig. 6. Replace with a clean *architecture-level* schematic: `S0: HBM/L2 → A100 QKᵀ`; `S1: added SRAM stores prefix K → conventional GPU QKᵀ`; `S2: added SRAM-CIM computes prefix scores → GPU d64 reduction`, with common GPU suffix/remainder as a shared lower label. **No decorative axis, fictitious performance curves or bandwidth arrow labels.** Existing figure source is not present among the supplied artifacts; do not regenerate quantitative Fig. 6 from the conceptual sketch.
- **Fig. 2, page 5 — difficult to read**, especially at two-column print width because of 3-D bars. Preserving original data is higher priority than redrawing from an image. Improve label/axis size and raster quality *from the underlying original plot if available*; otherwise preserve figure, and use Table 5 for exact numerical comparison. The four panels represent only selected area budgets; Table 5 covers all eight.
- **Fig. 3, page 6 — coherent** with Equation (8), but the legend/markers are small. No new analysis needed; 27.844 MiB at A8 and 5.4% at P32k are source-supported.
- **Fig. 4, page 6 — keep unchanged**. Its caption correctly states partial macro energy for one 128-token K tile. Do not relabel ratios as tokens/J or system energy.
- **Fig. 5, page 7 — keep original 4 panels**. Add absolute PPL in accompanying text/table. The boxplot token ΔNLL median is not a summary of mean NLL or relative PPL alone.
- **Fig. 6, page 7 — keep original curves**. The legend identifies `S1` as the faster S1-HD/S1-DP alternative; that is not always identical to the conventional `min(S0,S1-HD,S1-DP)` used for Table 5. The P2k batch crossover and P32k dominance described in the text apply to the plotted series; avoid generalizing to all unplotted coordinates.
- **References:** original [4] Hydragen, [6] PAT, [14] Ouroboros, [15] CIM-Tuner, [16] CHIPSMORE already included; add verified S²-CIM only. Do not insert citation numbers by hand if the author uses BibTeX. Update citation ordering on compile.

## E. Explicitly rejected edits and remaining boundaries

1. Do **not** redefine S2 as a 256-macro signed-bank engine or multiply original macro energy by four. Signedness is already assumed externally in the submitted study, and optional alternate embodiments are separate research.
2. Do **not** replace S0 INT4 with A100 FP16 source-only comparisons or H100 acquisitions. The original 1,248 INT4 TOPS is a legitimate dense-INT4 hardware specification, although model predictions remain different from measured attention kernels.
3. Do **not** add a digital-implementation comparator or describe speculative control-area charges. Present area consistently as **published-macro area**, explicitly limited to this model.
4. Do **not** claim a whole-decoder speedup or infer it from 1.97× QK-stage speedup. An Amdahl upper-envelope expression is optional; if included, label the unknown QK time fraction `f` as not measured.
5. Do **not** present the 117/728 residency-improvement count as S2-versus-S1 success rate or as enlarging the 65.9% region. Intersection of these two sets is not evidenced, and the 117 count only measures three S2 model components. It can remain in the repo as supporting research without entering this lean paper revision.
6. Do **not** claim a calibrated, real optimized Hydragen/PAT GPU head-to-head: the current repo does not demonstrate that. Correct literature positioning suffices for a *source-model architectural* paper but remains a scientific novelty risk.
7. Do **not** treat R0 BF16 numerical quality as a timing-matched A100 INT4 kernel. Numerics and timing have distinct evaluation protocols.
8. Do **not** say source-only stage results imply 'LLM inference acceleration' without the qualifier that only shared-prefix QKᵀ was modeled; remaining decoder steps are conventional.

## F. Final editorial pass after applying the six patches

- Recompile from the actual LaTeX source; verify references and the complete Fig. 1 replacement manually at final IEEE column width.
- Check that Fig. 1 and captions do not make roofline claims inconsistent with Fig. 6.
- Verify the abstract, Results and Conclusion use **"modeled shared-prefix QKᵀ"** and **"equal added published-macro area"** consistently; Table 5's 65.9% means **60/91 evaluated points**.
- Preserve original Figure 2/Table 5 source files and numerical hashes. No chart regeneration from an image.
- Verify R2→R3 at P2k has a 95% interval crossing zero, with exact paired references from archived records.
- Ensure bibliography includes correct S²-CIM authors/title/DOI and separates measured 28-nm chip results from our modeled 7-nm QK endpoint.
- Create a two-column, page-by-page visual diff from the original before freezing the revision.

**Publication assessment:** TCAS-AI's hardware-oriented scope is aligned with the paper, but the high-level novelty objection cannot be eliminated merely by replacing prose. The most defensible minimal contribution is the **explicit workload-/area-dependent cross-architecture selection boundary**. The original manuscript already establishes that argument technically; this revision makes its novelty relative to GPU prefix kernels, general CIM allocation, and new KV-sharing CIM silicon legible, without manufacturing additional evidence.

### External source verification (retrieved 6 November 2026)

- IEEE TCAS-AI scope: https://ieee-cas.org/publication/ieee-transactions-circuits-and-systems-artificial-intelligence
- NVIDIA A100 INT4 architecture: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/ (dense 1248 TOPS, sparse 2496 TOPS); A100 80 GB SXM HBM: https://www.nvidia.com/en-us/data-center/a100/ (2039 GB/s)
- Sinangil et al. 7-nm CIM: https://doi.org/10.1109/JSSC.2020.3031290
- Hydragen: https://arxiv.org/abs/2402.05099
- PAT: https://arxiv.org/abs/2511.22333 ; paper artifacts https://github.com/flashserve/PAT
- CIM-Tuner: https://doi.org/10.23919/DATE69613.2026.11539276
- Ouroboros: https://arxiv.org/abs/2603.02737
- S²-CIM: https://doi.org/10.1109/TCSI.2026.3742618 (IEEE TCSI, early access 19 Oct. 2026)