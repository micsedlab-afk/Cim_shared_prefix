# TCAS-AI manuscript revision: evidence audit and minimal-change text
**Version: 2026-11-06 | Working source: original eight-page manuscript, "Where Is the Sweet Spot for SRAM-CIM in Shared-Prefix LLM Attention?"**

## Fixed scientific contract

- Preserve **S0 LLMCompass-modeled A100 INT4**, **S1-HD 7-nm 6T SRAM**, **S1-DP 7-nm 2RW 8T SRAM**, and **S2 7-nm SRAM-CIM** exactly as submitted. The S2 engine uses **128 macros / 0.4096 mm²** and the paper's external signed-Q4/K4 processing.
- The comparison endpoint is **shared-prefix QKᵀ latency**, using the **pointwise minimum of S0, S1-HD and S1-DP** as the conventional reference, at equal added **published-macro area**. Preserve 13 B × 7 prefix × 8 area = 728 original design points, original model and Q4/K4 macro energy.
- The original Table 5 favorable counts at 1, 2, 3, 4, 6, 8, 12, 16 mm² are **0, 11, 14, 22, 52, 60, 60, 57 out of 91 each**. Hence 8 and 12 mm² tie on *sampled-grid coverage* (65.9%); the maximum individual **modeled QK-stage** speedup **1.97×** occurs at B128, P32k, 16 mm². Grid coverage is not a serving-workload frequency, and peak speedup is not TPOT.
- Original R0–R3 Llama-3.1-8B real-pretrained, disjoint WikiText-2 / PG-19 quality evaluation is preserved, with **frozen per-layer ADC parameters**. For WikiText-2 P512/P2048/P16384, R0–R3 absolute PPL are (6.699/6.721/6.798/6.840), (5.481/5.564/5.743/5.666), (5.154/5.256/5.425/5.485), respectively. These are **held-out subset** absolute PPL, not whole-corpus numbers.
- The newer 117/728 finite-residency and 0.523 µs component-level placement result uses the same S2 128-macro geometry but is a **secondary sensitivity**, not a replacement for original Table 5. Its 117 points are not proven to coincide with the original S2-favorable 276 points. P2048/B1/A8 and A12 show no modeled stage-component gain; P2048/B1/A16 shows 17.570→17.047 µs/step over *three components only*; the original B128/P32k/A16 maximum-speedup point shows no pinning improvement.
- Independent signed-bank 256-macro scenarios, compressed-key/centroid experiments, SRAM+MAC alternatives, H100 baselines, physical implementation conjectures, and fabricated full-decoder speedups are **outside this manuscript revision**.
- Maintain the original positive performance and energy claims **with their stated endpoints**. Macro-energy Fig. 4 compares S1 tile write+read to S2 macro compute; neither is total-system energy.

## Audited relation to state of the art (as of 6 Nov 2026)

| Reference | Documented primary contribution | Exact overlap | Safe differentiation for this manuscript |
|---|---|---|---|
| **Hydragen** (Juravsky et al. 2024), existing [4] | Exact prefix/suffix decomposition, cross-request query batching, GPU-efficient GEMM, LSE fusion | The shared-prefix reuse mechanism | Treat as a conventional kernel/dataflow foundation; the paper evaluates where **incremental SRAM storage** or CIM arithmetic is preferable under matched macro-area budgets. No claim to invent shared-prefix batching or fusion. |
| **ChunkAttention** (Ye et al. ACL 2024), [5] | Prefix-aware KV chunks and two-phase attention organization | Prefix locality and reuse | No new caching algorithm is claimed. |
| **PAT** (Yi et al. ASPLOS 2026), [6] | Specialized shared-prefix GPU kernel, measured A100/H100 attention and TPOT results | Competitive GPU execution | S0 is a **predicted A100 INT4 operator path**, not a runtime comparison with PAT. State this explicitly. PAT sets an important practical GPU bar. |
| **MoSKA** (Rhee et al. 2025), [7] | Shared-KV GEMM batching, sparse long-context attention and disaggregated infrastructure | Shared-prefix GEMM | Analyze SRAM residency versus CIM allocation on the same modeled GPU path, rather than introduce a new serving backend. |
| **Ouroboros** (Liu et al. ASPLOS 2026), [14] | Wafer-scale SRAM-CIM, token-grained pipelining, KV management and communication-aware mapping | CIM plus finite KV capacity | Report *cross-architecture* published-macro area tradeoffs at a defined QK endpoint. Do not imply token-level scheduling is new. |
| **CIM-Tuner** (Chen et al. DATE 2026), [15] | General area-constrained CIM compute/storage sizing plus mapping search | Storage/compute balance under area constraints | Distinct research variable is the **pointwise conventional GPU/SRAM vs CIM selection boundary** driven by *B, shared P, and Aadd* with a 7-nm published-macro model and numerical-fidelity check. No claim of a first area-constrained CIM exploration. |
| **CHIPSMORE** (Chong et al. Aug 2026), [16] | Multi-mode/multi-request heterogeneous chiplets with hierarchical KV organization | KV hierarchy and multi-request handling | Distinguish macro-level architectural feasibility at shared-prefix QKᵀ from full system implementation. |
| **S²-CIM** (B. Liu et al., IEEE TCSI, **19 Oct 2026**), **new [31]** | Fabricated 28-nm statistics-aware CIM accelerator for KV-sharing-enabled LLMs with page-aware access, specialized arithmetic and system-level measurements | **Closest current direct attention/CIM competitor** | Acknowledge explicitly. It targets accelerator implementation; this paper isolates the *equal published-macro-area choice between conventional SRAM locality and CIM score computation* against a pointwise A100/SRAM reference. **Do not compare raw 28-nm hardware energy/token with 7-nm modeled QK-stage or partial macro energy.** |

**New reference to add (IEEE style):** B. Liu *et al*., “S²-CIM: A 28-nm 68.9-µJ/Token Digital CIM Accelerator With Statistics-Aware Access and Computation for KV Sharing-Enabled LLM Inference,” *IEEE Trans. Circuits Syst. I, Reg. Papers*, early access, Oct. 19, 2026, doi: **10.1109/TCSI.2026.3742618**.

Sources: Hydragen https://arxiv.org/abs/2402.05099 ; PAT https://arxiv.org/abs/2511.22333 ; Ouroboros https://arxiv.org/abs/2603.02737 ; CIM-Tuner https://doi.org/10.23919/DATE69613.2026.11539276 ; CHIPSMORE https://arxiv.org/abs/2608.30509 ; S²-CIM https://doi.org/10.1109/TCSI.2026.3742618 .

## Minimal surgical editing schedule (8-page source)

| Where | Action | Original content retained | Approximate footprint |
|---|---|---|---|
| Page 1 abstract + page 2 contribution bullets | Emphasize **selection boundary** and its underlying mechanisms, explicitly describe model/endpoint | Numerical values, source models, comparative configuration definitions | Replace 1 abstract and 3 bullets |
| Section I, first 2 pages | Add precise gap after existing source discussion | Existing citations [1]–[16] | Replace 1 paragraph |
| Section II.C | Replace enumerative closing of related-work discussion with the comparative paragraph below and add [31] | Existing citations and technology taxonomy | ~120–160 words |
| Section II.D / III.A | Explicit **source-model / new model-component attribution** sentence | LLMCompass A100 INT4 and published macro source | 2–3 sentences |
| Page 5–6 Table 5 / Results / Fig. 6 | Lead with crossover **8–12-mm² coverage plateau / 16-mm² contraction**; distinguish maximum speedup operating point | Figure 2, Figure 3, Figure 4, Figure 6, Table 5 unchanged | 1 Results paragraph |
| Figure 5 or text immediately following | Give original **absolute PPL** next to relative PPL and existing confidence intervals; mark evaluated subsets | WikiText-2/PG-19 plotted differences and frozen calibration | 3-row compact table or 2–3 sentences |
| Section IV.C / Design Implications | Integrate **117/728** placement sensitivity in separate paragraph, with **no performance overlap claim** | Original QK-stage speedup and macro-energy results | ~100 words |
| Section IV.C / Conclusion | One **Amdahl expression**, explicitly an *analytical limit* with unknown measured QK fraction | All original modeled S2 timing claims | 2 sentences + inline formula |
| Figure 1 | Improve manual layout/typography; verify labels (S0/S1/S2, HBM/L2/A100, shared K, QK) before exporting | Concept only; no changes to numeric plots | Replace artwork only if source available |
| Section V Conclusion | Tighten to storage/computation crossover and qualified limits | Source results and fidelity numbers | Replace 1 paragraph |

**Do not add a second experimental architecture, cross-process hardware score comparison, new FPGA/GPU model, or extensive output-length sweep to this paper.** These would dilute a focused low-effort revision.

## Insertion-ready replacement text

### Abstract (replace the current Abstract, keeping the same parameters and values)

> The area assigned to shared-prefix attention can improve conventional SRAM residency or support SRAM compute-in-memory (CIM) score computation. Identifying the preferable allocation requires accounting for shared-key reuse, finite storage capacity, and the conventional GPU execution path. This work analyzes that tradeoff for the shared-prefix QKᵀ operation using an LLMCompass-modeled NVIDIA A100 baseline (S0), high-density SRAM (S1-HD), dual-port SRAM (S1-DP), and a published 7-nm SRAM-CIM macro (S2). The alternatives are evaluated under equal added published-macro area across 728 configurations of batch size, prefix length, and area budget. At each point, S2 is compared with the fastest modeled conventional configuration. S2 achieves lower modeled QKᵀ latency for 65.9% of the sampled workload configurations at 8 and 12 mm², with a maximum modeled speedup of 1.97× at 16 mm². The favorable-region boundary follows from the competing scaling of SRAM key residency, discrete CIM engine parallelism, and GPU matrix efficiency. A macro-level energy analysis identifies additional batch-dependent crossovers. Numerical evaluation on Llama-3.1-8B separates Q4/K4 quantization and the published CIM macro's ADC and circuit response; ADC calibration is fixed before disjoint WikiText-2 and PG-19 evaluation. These results characterize the conditions under which storage-oriented SRAM or compute-enabled SRAM is preferable for shared-prefix score generation.

### Gap + research question (append after current introductory discussion of SOTA, replacing the generic contribution-motivation paragraph)

> Shared-prefix attention implementations exploit query batching and KV reuse to improve conventional GPU execution, whereas CIM accelerators optimize the local execution of selected transformer operations. Area-constrained CIM mapping studies additionally examine the internal allocation of storage and computation. For a shared-prefix workload, the architectural decision also involves the conventional alternatives: increasing SRAM capacity can eliminate repeated key transfers, while dedicating the same macro area to CIM increases local arithmetic parallelism. The relative advantage varies with batch size, prefix length, and discrete engine capacity. We therefore ask: for a fixed published-macro-area budget and a common shared-prefix QKᵀ endpoint, which allocation minimizes modeled latency relative to the fastest available GPU/SRAM configuration, and where does the preferred allocation change?

### Contribution bullets (replace only existing three bullets)

> • A pointwise, equal-published-macro-area comparison of GPU/HBM, high-density SRAM, dual-port SRAM, and 7-nm SRAM-CIM for shared-prefix QKᵀ, spanning 728 combinations of batch size, prefix length, and added macro area.
>
> • A quantitative characterization of the SRAM–CIM selection boundary, explaining its dependence on shared-prefix K residency, grouped-query reuse, GPU utilization, and the discrete number of complete CIM engines. The analysis includes QKᵀ latency, key-traffic reduction, and macro-level access/compute energy.
>
> • A Llama-3.1-8B numerical-fidelity evaluation that separates Q4/K4 quantization, ADC quantization, and the published circuit response, using frozen per-layer calibration and disjoint WikiText-2 and PG-19 evaluation.

### Closing comparison paragraph for Section II.C (after existing paragraphs, add [31])

> Hydragen [4] and MoSKA [7] establish efficient conventional shared-prefix execution through cross-request query batching, while PAT [6] develops an optimized GPU decoding kernel. SRAM-CIM research addresses a different aspect of the memory–compute tradeoff: Ouroboros [14] and CHIPSMORE [16] propose broader inference architectures with KV management, whereas CIM-Tuner [15] jointly optimizes CIM resources and mapping under area constraints. The recently published S²-CIM [31] demonstrates KV-sharing-aware CIM through a fabricated accelerator. These works motivate examining the allocation decision at a common operation endpoint. Here, the compared alternatives share a modeled A100 execution environment and equal added published-macro area; the analysis identifies the workload and capacity ranges in which conventional SRAM locality or CIM score computation provides lower modeled QKᵀ latency. The numerical-fidelity study evaluates the selected macro's quantized execution separately.

### Source/model attribution for Section III.A (insert after current first paragraph)

> The A100 conventional path uses the LLMCompass performance-model methodology [17] with the explicitly stated 4-bit operand, accumulator, and data-transfer assumptions. The SRAM configurations extend the same conventional score-computation path with capacity and locality derived from published 7-nm SRAM macros [20], [21]. The S2 path combines the published 7-nm CIM access characteristics [22], [23] with the defined prefix tiling and active-layer transfer model. Thus the compared results are source-based predictions at a matched shared-prefix QKᵀ endpoint, with equal added published-macro area.

### Sweet-spot interpretation for the Results (after Table 5)

> Table 5 identifies a nonmonotonic sweet spot in the sampled design space. S2-favorable coverage grows from 24.2% at 4 mm² to 65.9% at 8 mm² and remains at 65.9% at 12 mm², before decreasing to 62.6% at 16 mm². Increasing area provides additional complete S2 engines, reducing the number of score-computation waves, and simultaneously increases S1 key residency, reducing conventional GPU-side memory traffic. Their different scaling and the batch-dependent efficiency of GPU GEMM determine the pointwise crossover. The peak modeled QKᵀ speedup, 1.97×, occurs at B=128 and Lp=32k with 16 mm². Consequently, the area budgets with the widest favorable sampled region and the design point with the largest individual speedup are distinct. These results are defined over the evaluated workload grid; the reported coverage does not imply a request-frequency-weighted serving benefit.

### Short supplement for Figure 5 (leave original four panels unchanged)

> Absolute WikiText-2 perplexities for P = 512, 2048 and 16384 are, respectively, R0: 6.699, 5.481, 5.154; R1: 6.721, 5.564, 5.256; R2: 6.798, 5.743, 5.425; and R3: 6.840, 5.666, 5.485. Values are obtained from the same eight held-out sequences at each prefix length. At P=2048, R3 is numerically below R2; the original paired uncertainty intervals do not establish a systematic R3-versus-R2 improvement. The PG-19 evaluation uses the unchanged calibration.

### Optional placement interpretation for Section IV.C (one paragraph)

> An additional finite-residency analysis retains the original 128-macro S2 engine and allocates some engines to complete shared-prefix layers while using the remainder for streamed layers. In 117 of the 728 original workload–area coordinates, this allocation reduces modeled K4 programming with no increase in the native compute-wave count. Using the submitted 2,039-GB/s A100 HBM and published macro timing inputs, the P=2048, B=1, 16-mm² configuration reduces the sum of modeled K4 supply, row programming, and native macro compute components from 17.570 to 17.047 µs per decode batch step for 128 output steps. The corresponding component model shows no change at 8 or 12 mm². This result characterizes a secondary placement boundary; it is separate from the original S2-versus-conventional QKᵀ speedup coverage.

### Scope for the final Results / Conclusion (two sentences, not a new experimental claim)

> The reported S2 speedups apply to the modeled shared-prefix QKᵀ stage. If that stage contributes a fraction f of conventional decoder latency and achieves a stage speedup s, with other operations unchanged, the full-decoder speedup has the analytical upper envelope 1/[(1−f)+f/s]; the decoder-specific fraction f is not measured in this study.

### Conclusion (compact replacement)

> This study identifies the equal-published-macro-area selection boundary between conventional SRAM locality and SRAM-CIM score computation for shared-prefix LLM attention. The 728-point modeled evaluation shows that the preferred allocation depends jointly on shared-prefix length, batch reuse, and the discrete number of available CIM engines. S2 is favorable in 65.9% of the sampled configurations at 8 and 12 mm², while the maximum modeled shared-prefix QKᵀ speedup of 1.97× occurs at 16 mm². The associated SRAM-residency, shared-key traffic, and macro-energy analyses explain the opposing effects of increasing area. A separate Llama-3.1-8B numerical-fidelity evaluation quantifies the effect of Q4/K4 arithmetic and the selected 7-nm CIM macro response under fixed ADC calibration. These findings provide workload-dependent selection criteria for assigning incremental published-macro area to key storage or near-data attention-score computation.

## Figure and evidence audit

1. **Fig. 1**: current conceptual illustration has distracting text/typography and ambiguous axis labels. Redraw for publication, preserving S0/S1/S2 and storage-vs-compute conceptual meaning. No quantitative results are derived from this figure.
2. **Original performance plot / Table 5**: keep both original; add a thin text callout identifying **8–12 mm² as maximum sampled coverage** and **16 mm² as best individual peak**. Do not say 8 mm² is globally optimal.
3. **Fig. 3**: one usable key-capacity plot; explain 87% vs 5.4% at 8 mm² S1-HD for P2k vs P32k.
4. **Fig. 4**: ensure caption consistently reads “macro-level S1 read+write versus S2 compute energy,” retaining the B≈41/25 crossings; do not add total-chip energy.
5. **Fig. 5**: add absolute PPL values; retain original paired bootstrap CI and restricted held-out-subset description. Distinguish original R0–R3 from later SmolLM2 prefix compression.
6. **Fig. 6**: retain modeled roofline; make the interpretation quantitative using the existing P2k and P32k observations. Differentiate conventional S1 branch winner from full S0/S1 pointwise baseline where relevant.
7. **117/728 placement**: keep as secondary; the original 276 S2-favorable coordinates and 117 placement-beneficial coordinates are different events, with overlap **unquantified**.

## Explicit claim checks before submission

- All “faster than conventional” claims specify **modeled shared-prefix QKᵀ** and reference **min(S0, S1-HD, S1-DP)**.
- “Equal-area” means **equal added published-macro area**, across source technology records. There is no 7-nm synthesis claim.
- Favorable *coverage* is count over 91 B×Lp points per area, **not statistical serving throughput or workload probability**.
- Full-decoder speedup is only a conditional analytic formula, **no measured TPOT**.
- R2→R3 nonmonotonicity is an observed NLL/PPL difference; avoid attributing its sign to an established physical mechanism without evidence.
- All included new related work predates 6 November 2026; S²-CIM is **a direct, recently published competitor** that should not be overlooked.
- Keep numerical protocols separate: A100 original INT4 model, A100 FP16 supplemental source model, original Llama BF16 R0–R3 quality study.
- The existing original paper's 7-nm macro figures are historical published macro data, not a new IC tapeout.
- No paper-facing language about external correspondence or the prior submission history.
