# TETC manuscript — MODIFY / ADD / KEEP implementation
**8 November 2026 · Target: IEEE Transactions on Emerging Topics in Computing, Track 1 (Emerging Computing Architectures and Paradigms)**

## Source lock and implementation status

The trusted eight-page manuscript, *Where Is the Sweet Spot for SRAM-CIM in Shared-Prefix LLM Attention?*, is the immutable numerical baseline. The original source-model stage results, area accounting and real Llama R0–R3 quality experiments are accepted; no reconstruction or rerun of the 728 points is requested. The integrated GitHub version is `integration/final-framework` at `e93392ecd741`; the prior revision text is in `docs/TCASAI_SOURCE_ANCHORED_MANUSCRIPT_AUDIT_AND_PATCH_20261106.md`.

**Applied to this TETC revision branch:** complete replacement/insert text in `docs/TETC_READY_TO_PASTE_20261108.tex`, verified new bibliography entries in `docs/TETC_NEW_REFERENCES_20261108.bib`, and an editable replacement architectural Figure 1 in `docs/figures/tetc_architecture_comparison.svg`. Original experimental code, data, and manuscript figures have not been altered. The original editable LaTeX project is not part of this repository or the accessible manuscript attachments; therefore, the replacements are source-anchored patches awaiting application to the author's original `.tex`, rather than a fabricated modified main.tex.

### Scientific contract — fixed

- **Four configurations:** S0 LLMCompass-predicted A100 INT4; S1-HD 7-nm 6T SRAM, S1-DP 7-nm 2RW 8T SRAM; S2 fabricated 7-nm SRAM-CIM source macro with **128 macros per engine = 0.4096 mm² published-macro area**. Original external sign handling and 4-bit source timing assumptions remain unchanged.
- **Comparison:** shared-prefix `QK^T` *stage latency*, not complete attention, not decoder TPOT. Same added published-macro area. Conventional comparator is `min(S0,S1-HD,S1-DP)` for each configuration.
- **Grid:** 13 B × 7 prefix lengths × 8 areas = 728. At areas `[1,2,3,4,6,8,12,16]` mm² the original favorable counts per 91 shapes are `[0,11,14,22,52,60,60,57]`. At both **8 and 12 mm², 60/91 = 65.9%**. The 16-mm² count is 57/91 = 62.6%. The **maximum individual modeled QK-stage speedup 1.97×** is at **B=128, prefix=32k, A=16 mm²**. Coverage is a uniformly sampled-grid count, not measured serving frequency or a universally optimal area.
- **Energy:** original tile write+read S1 versus native compute S2 published-macro terms only; nominal crossings B≈41 (HD) / B≈25 (DP) are **not** total-system energy. Do not multiply four native operations into the *original* metric or replace the 128-macro core with a counterfactual 256-macro mapping.
- **Numerics:** original real Llama-3.1-8B R0/R1/R2/R3, disjoint WikiText-2 evaluation and PG-19 held-out transfer; 96 original Llama evaluation units. P512 absolute PPL `6.699/6.721/6.798/6.840`; P2048 `5.481/5.564/5.743/5.666`; P16384 `5.154/5.256/5.425/5.485`. At P2k R2→R3 paired change –1.348%, 95% CI [–3.294%, +0.663%], crosses zero.

## Exact action map

| Action | Original PDF location / anchor | Ready material | Purpose |
|---|---|---|---|
| **KEEP** | Title page and all original quantitative parameters | Existing paper | Retain original question, authors, results |
| **MODIFY** | p.1 ABSTRACT ("Autoregressive decoding…") | Block A in TETC_READY_TO_PASTE.tex | Selection boundary, scope and result qualification |
| **MODIFY** | p.1 Introduction, paragraph starting "At the hardware level..." and ending "...multi-request LLM inference" | Block B | Establish TETC-oriented novelty against the two Sharma studies |
| **ADD** | p.2 just before "This work evaluates that comparison..." | Block C | Precise scientific question; model-only architectural scope |
| **MODIFY** | p.2 "The contributions are:" and its three bullets | Block D | Architectural decision map, mechanism, fidelity |
| **KEEP** | p.2–3 Section II.A/B, Eq. (1)–(2) | Original | Genuine prefix-reuse geometry and I_K=16B op/byte |
| **MODIFY** | p.3 Section II.C, paragraph beginning "Attention-oriented CIM architectures..." | Block E | Compare to Sharma, ISPASS, CIM-Tuner, PAT, S²-CIM, Ouroboros |
| **KEEP** | p.3 Section II.D Eqs. (3)–(4), Table 1 | Original | Pointwise fastest-conventional comparator |
| **ADD** | p.3 immediately after Eq. (4) interpretation | Block F | Define a sampled-grid coverage metric with correct denominator |
| **ADD** | p.4 Section III.A just after the A100/LLMCompass description | Block G | Separate measured macro inputs, predicted A100 timing, numerics |
| **MODIFY/ADD** | p.4 Section III.A/B near "prefix placement is completed before the timed decode interval" | Block H | Specify original S1/S2 placement boundary without costs invented |
| **KEEP** | p.4–5 macro equations, numerical quantization/mapping, Tables 2–4 | Original | No geometry, sign architecture or experimental change |
| **MODIFY** | p.6 Section IV.A, the two paragraphs after Table 5 starting "The complete coverage..." | Block I | Identify sample coverage plateau and different peak-speedup point |
| **KEEP** | p.5–7 original Figs. 2–6 and Table 5 values | Original | The trusted original result set |
| **ADD** | p.7 Section IV.B after first paragraph reporting relative WikiText changes | Block J + compact Table | Present absolute PPL and nonmonotonic R2–R3 CI |
| **MODIFY** | p.7 Section IV.C "The latency results identify three added-macro-area regimes..." through before Fig.6 paragraph | Block K | Publishable architectural decision implications with failure regimes |
| **ADD** | p.7 Fig.6 paragraph after the first interpretation of plotted winners | Block L | S1-only roofline comparison ≠ full S0/S1 pointwise comparator |
| **MODIFY** | pp.7–8 Section V Conclusion | Block M | Scope-limited findings, scientific takeaway |
| **REPLACE FIGURE** | p.2 Fig.1 existing performance/intensity sketch | `docs/figures/tetc_architecture_comparison.svg` | Editable accurate S0/S1/S2 dataflow; no fabricated roofline |
| **ADD CITATIONS** | Section I and II.C | `docs/TETC_NEW_REFERENCES_20261108.bib` | Two Sharma articles and S²-CIM, with publisher DOIs |
| **CHECK** | Page 8 acknowledgement, references, formatting | Below | Follow current IEEE author policies, accurate AI disclosure |

**No numeric changes** to Eqs. (1)–(12), source Tables 1–5, or Figures 2–6.

## What was developed after the original manuscript — scientific disposition

1. **Original S2 placement sensitivity**: 117/728 configurations allow fewer logical K4 tile writes with no increase in modeled native compute waves; separate analysis reports 17.570→17.047 µs in three modeled components for one P2k/B1/A16 point. The intersection with the original S2-winner set is unknown. **Retain in supplementary/internal evidence only.** It cannot support a new global S2 speedup or change Table 5.
2. **S1 cold placement**: 8-mm²/P2k original-geometry reconstruction has 29,163,520 initial K4 bytes for HD and 5,963,776 for DP; over 128 outputs that is 227,840/46,592 bytes per decode step. These are **amortized placement bytes**, not observed service time or new bandwidth. The paper already excludes cold S1 placement and includes S2 active-layer programming. Clarify instead of replacing the original timing.
3. **Separate signed-bank, prefix-context and full-decoder models**: alternate area/digital and source FP16 GPU sensitivities are not the original 128-macro INT4 QK-stage experiment. Preserve in repository, **exclude from the primary TETC performance comparison**.
4. **Real quality evidence**: the archived absolute PPL and paired confidence intervals are immediately publishable as methodological clarification. PG-19 preserves the separate WikiText-2-derived calibration.
5. **Competitive GPU evidence**: no real matched optimized Hydragen/PAT comparison is available. Cite these algorithms as related execution baselines; qualify the submitted S0 as a *predicted model*, never measured CUDA. No synthetic benchmark substituted.
6. **No added digital implementation references, speculative controller-area or control-overhead charges**, or unmeasured TPOT/energy.

## New architecture-specific interpretation of original results

The 8–12-mm² tie maximizes *sampled beneficial coverage*, while the 16-mm² configuration contains the maximum individual modeled acceleration. Increasing area simultaneously enlarges complete S2 engine count and S1 resident K capacity. These benefits have different granularity and depend differently on `B` and `L_p`. At 8 mm², S1-HD's 27.84375 MiB holds approximately 86.9% of the 2k-token 32-MiB all-layer K image, but about 5.4% of the 32k-token 512-MiB image. Increasing `B` raises shared-prefix K reuse at fixed K footprint and also increases the GPU matrix dimension; consequently, neither larger batches nor longer prefixes imply a universal CIM win. This is a conditional *workload-aware design rule*, not a hardware demonstration. The full sampling grid contains 276 favorable cases of 728 (arithmetic aggregate, 37.9%), but **do not use this percentage as an application/serving prevalence estimate**.

## What remains to apply in the original author project

- Insert Blocks A–M at the anchors above, **replacing** corresponding existing paragraphs to avoid repetition. Do not stack overlapping old/new discussions.
- Import the two verified 2025 Sharma BibTeX entries and 2026 S²-CIM; retain original keys for Hydragen, CIM-Tuner, PAT, etc. `TETC_READY_TO_PASTE.tex` marks existing-citation aliases that must be mapped to the author's actual BibTeX keys.
- Convert the paper from IEEE Access format (`VOLUME XX, 2026`) into the IEEE Computer Society / TETC journal template; keep author data, acknowledgement, all figures and references. Page count and publication fees must be checked in the current TETC submission portal.
- Check Fig.2 and Fig.6 axis/legend readability at final two-column width; no regeneration from screenshot or extrapolated pointwise samples.
- Run LaTeX, bibliography and visual page-by-page diff from the original **editable** source after it becomes available. No full-source mutation claimed at this step.

**Acceptance target for this editorial pass:** reader can identify the original experimental evidence, exactly what constitutes a win, why its boundary varies with `B,L_p,A`, how the analysis differs from prior generic CIM mapping studies, and the precise limits of QK-stage timing, macro-only area, partial energy, and held-out numerics.

## Verified external publication details

- IEEE TETC Track 1: https://www.computer.org/digital-library/journals/ec/technical-tracks
- Sharma et al., IEEE TETC 2025 DOI: https://doi.org/10.1109/TETC.2025.3574508
- Sharma et al., ISPASS 2025 DOI: https://doi.org/10.1109/ISPASS64960.2025.00056
- Chen et al., CIM-Tuner DATE 2026 DOI: https://doi.org/10.23919/DATE69613.2026.11539276
- Liu et al., S²-CIM IEEE TCSI 2026 DOI: https://doi.org/10.1109/TCSI.2026.3742618
