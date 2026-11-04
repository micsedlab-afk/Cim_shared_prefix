# Final simulation results and reviewer-evidence response — 4 November 2026

> **Evidence revision (4 Nov 2026).** Read [verified source-only capacity, fused GPU and B1 weight-traffic bounds](VERIFIED_CAPACITY_GPU_TRAFFIC_BOUNDS_20261104.md) and [reviewer addendum](REVIEWER_ADDENDUM_VERIFIED_CAPACITY_20261104.md) **before** interpreting the conditional 0.929 µs result below. That point depends on unverified 16TB/s V-SRAM bandwidth, 64TMAC/s digital PV and other nonphysical support-rate assumptions. The new rate-free study shows that 16mm² can permanently retain at most 143 of 4096 prefix token-layer tiles at P2048; in a source-modeled B1 FP16 all-HBM decoder, even free offload removes ≤0.061342% of the required weight+KV bytes per token. No demonstrated end-to-end speedup follows from the older favorable source comparisons.


**Repository:** `micsedlab-afk/Cim_shared_prefix`  
**Scope:** 7-nm published-macro architectural simulation of dense shared-prefix attention, native Llama-3.1-8B GQA; source-modeled A100 FP16 GPU comparator.  
**Qualification:** `research/prefix-context-offload-20261104` has **156 discovered tests; 155 passed, one skipped** and the full **576-point** prefix-context study executed [in Actions run 41868345731](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/41868345731). MAGMA source-native full QK–softmax–PV and grouped-QK execution [passed in run 41868354178](https://github.com/micsedlab-afk/MAGMA-gpt/actions/runs/41868354178). All source-model results are simulation, not physical GPU timings.

## Scientific result and its exact limits

**Constructive architecture:** Keep one layer's native signed Q4×K4 K+ and K− banks in published 7-nm Sinangil SRAM-CIM macros; store the same layer's **FP16 V** in published 7-nm Yokoyama conventional SRAM; apply bounded-wave score reconstruction, stable streaming softmax, and **digital** PV with FP32 accumulation near memory; return an FP16 prefix-context vector and one FP32 logsumexp per query head. Remaining decoder layers and request-specific suffix attention stay on the modeled GPU. Accepted four-unsigned-evaluation signed mapping is preserved. The previously rejected affine-offset circuit mapping remains excluded.

### R1. Exact physical storage and communication accounting

Reference coordinate: `Llama-3.1-8B`, native GQA 32Q/8KV/D128, P=2048, suffix=128, B=1, O=128, area budget 16 mm² including 0.5 mm² assumed external support.

| Physical element | Count | Area |
|---|---:|---:|
| K+ and K− native Sinangil 7-nm CIM banks | 4,096 complete macros | 13.107200 mm² |
| FP16 V resident SRAM (4 MiB) | 128 complete 512×512 Yokoyama 7-nm macros | 1.149124 mm² |
| Finite wave buffering and FP32 online state | 27 complete SRAM macros | 0.242393 mm² |
| Controller/clock/digital/periphery **sensitivity allowance** | Unknown circuit implementation | 0.500000 mm² assumed |
| **Combined scenario budget consumed** | | **14.998718 mm²** |

Conventional QK-only CIM returns 262,144 bytes of completed FP32 prefix scores per generated token and selected layer. The new prefix-attention endpoint returns **8,320 bytes** (32 query heads × [128 FP16 context entries + 1 FP32 LSE])—a **31.5077× reduction in link traffic**. Locally resident V still incurs **4,194,304 bytes of SRAM-read traffic/token/layer**, included separately in the timing model.

### R2. Executed causal timing and bandwidth sensitivity

Physical inputs: published complete CIM macro area **0.0032 mm²**, published native unsigned service **5.5 ns**, complete signed two-plane storage, four unsigned evaluations per signed Q4×K4, all cold K and V supply/programming, wave-by-wave digital stages and **two-wave** physical SRAM staging.

**Reference conditional rates:** 256 usable CIM ports; 1 TB/s each assumed key supply/programming, V supply/programming, query and context link; **16 TB/s local V SRAM reads**; **64 TMAC/s digital PV**; 1 Texp/s and 1 Tscalar/s stable softmax; 1 Tscale/s and 1 Tadd/s signed reconstruction; zero controller/setup time. These are explicit assumed resources, not fabricated 7-nm digital measurements.

| Executed prefix-context quantity | Model |
|---|---:|
| One-time supply and programming of resident K and V | 12.582912 µs |
| Warm token, selected layer, complete prefix attention | 0.830632 µs |
| Amortized cold cost across 128 generated tokens | 0.098304 µs/token |
| **128-output mean prefix-context service** | **0.928936 µs/token** |

Digital support density **at least** `64 TMAC/s / 0.5 mm² = 128 TMAC/s/mm²` is necessary to realize the reference PV assumption, **before reserving any support area for controllers/softmax/reduction**. The assumed 16 TB/s local V read means 125 GB/s per installed V SRAM macro on average. Physical achievement of neither rate is verified.

### R3. Corrected source GPU baseline, including source-unit error

The original LLMCompass `Matmul.__call__` computes `io_count` in *elements*; `Matmul.roofline_model` divides it by interface capacity in *bytes/s*. The FP16 Matmul memory component is therefore 2× too optimistic. We preserved the historical source points and recomputed **FP16 bytes = 2 × element count** while maintaining the original LLMCompass device FLOPs/bandwidth. Actual A100 hardware is not available; the A100 is a **7-nm GPU architectural model**.

[Native source execution](https://github.com/micsedlab-afk/MAGMA-gpt/actions/runs/41868354178) reproduces the source Matmul/Softmax operators. B=1, P=2048:

| Source A100 FP16 model | Grouped prefix QK | Complete *unfused* prefix QK + softmax + PV |
|---|---:|---:|
| Original Matmul unit-mismatched HBM model | 1.062670 µs | — |
| **Byte-corrected, HBM bandwidth** | **2.125340 µs** | **4.443527 µs** |
| **Byte-corrected, L2 bandwidth** | **0.600284 µs** | **1.294710 µs** |

GPU suffix attention and decoder remainder remain common in these isolated comparisons, **not** an executed complete S0/S2 decoder. The complete-prefix GPU row is **unfused serial source operators**, while Hydragen and FlashAttention can fuse attention and reduce memory traffic. Source HBM/L2 residency alternatives are **modeled placements**, not observed cache behavior.

Against the same reference context cost **0.928936 µs**, the modeled CIM advantage is **+1.196404 µs/layer** relative to source HBM QK **alone**; the GPU L2 QK source is **0.328652 µs/layer faster**. Relative to the full *unfused* prefix models, CIM is quicker by **3.514591 µs** (HBM) or **0.365774 µs** (L2). No result here qualifies advantage over optimized fused Hydragen or full-model TPOT.

### R4. Feasibility frontier and dispatch

The **576-case** CI study varies B={1,2,4,8}, total area={15,16} mm², ports={128,256}, local V bandwidth={1,2,4,8,16,32} TB/s, digital PV={4,16,64} TMAC/s and output={32,128}. All 576 coordinates use the stated source macro and explicitly scenario-defined rates.

| Conditional comparison classification | Coordinates |
|---|---:|
| CIM complete-prefix time below even corrected **HBM GPU QK-only** roofline proxy | **92** |
| CIM below source **HBM unfused full-prefix** model only | **244** |
| CIM not below even source **HBM unfused full-prefix** model | **240** |
| **Total** | **576** |

These categories are model-relative, not physical speedup claims. A separately executed **72-case** B={1,2,4,8}, V bandwidth={1,2,4,8,16,32} TB/s, PV={4,16,64} TMAC/s slice at 16 mm², 256 ports, O128 yielded: **16/72** favorable against corrected HBM QK-only; **0/72** against corrected L2 QK-only; **58/72** against corrected HBM unfused full-prefix; **8/72** against corrected L2 unfused full-prefix. No extrapolation of these 72-case L2 counts to the entire 576-case space.

At B1, O128 and the other reference rates, the independent rate inversion obtains a digital PV crossover of approximately **4.392 TMAC/s at 4 TB/s local V bandwidth** relative to the HBM QK-only source proxy; **4.364 TMAC/s at 8 TB/s**. The L2 unfused complete-prefix source requires approximately **7.946 TMAC/s at 8 TB/s**. These are **required rates under the model**, not a synthesized circuit result. Stable prefix/suffix fusion correctness is independently tested on supplied scores.

### R5. Recovered, genuinely evaluated numerical quality

Preserve `results/quality/quality.json`, `paired_effects.csv` and `source_evidence.json` as distinct **archived inference evaluations**. They cover eight held-out samples, Llama-3.1-8B at B1, suffix128 and P={512,2048,16384}, R0–R3, 96/96 original quality units. Circuit response is emulated from published characterized data. They do **not** directly evaluate this new selective-one-layer digital PV architecture.

| Prefix | R0 PPL | R3 PPL | Paired R3 vs R0 PPL effect, 95% CI |
|---|---:|---:|---|
| 512 | 6.699 | 6.840 | +2.11% [0.46%, 3.59%] |
| 2048 | 5.481 | 5.666 | +3.38% [0.99%, 5.83%] |
| 16384 | 5.154 | 5.485 | +6.43% [3.11%, 9.39%] |

For P2048, R3–R2 = **−1.35% PPL**, CI [−3.29%, +0.66%]; no statistically established R3 improvement. OPT-2.7B and Mistral-7B R0–R3 remain **0/96 each**. The new model's numerical oracle checks stable streaming softmax/PV for provided inputs and the prefix/suffix LSE algebra; it does not prove end-to-end PPL of the selective execution policy.

## Direct response to the existing reviewer-concern matrix

| Original concern | Consolidated evidence and proposed response | Status |
|---|---|---|
| **Hydragen: novelty and competitive baseline** | Adds a physically budgeted Q4K4 signed CIM + SRAM V + online digital context endpoint. Compares source grouped shared-K GPU models, with corrected HBM/L2. CPU functional fusion tested. **No direct competitive fused Hydragen timing**. | Partial; performance unresolved |
| **MoSKA/shared KV and sparsity** | Dense prefix, explicit GQA and complete signed mapping. No sparse/FP8 MoSKA speedup claim; shared-prefix query batching is acknowledged prior art. | Scope clarified |
| **S²-CIM, digital/sparse 28-nm comparison** | Separate published macro metrics/algorithm. No node-scaled iso-area or directly comparable system speedup. | Scope clarified |
| **CIM-Tuner mapping/co-exploration novelty** | Do **not** claim generic area/scheduling optimizer novelty. New contribution candidate: signed physical/traffic-aware feasibility for the selected attention endpoint. | Claim corrected |
| **Ouroboros dynamic KV scheduling** | Dynamic residency and wave pipelining acknowledged as prior art; scope is this finite 7-nm macro/Q4K4-O4 configuration and derived constraints. | Claim corrected |
| **Signed 4-bit I/O and ADC effects** | Two K sign planes, four unsigned evaluations, scaling/reduction and O4 emulation retained. The output FP32 representation is a **transfer format**, not arithmetic accuracy. | Physical mapping modeled; device-numerical scope limited |
| **Physical area and MAC integration** | 4,096 K CIM + 128 V SRAM + 27 buffer SRAM macros all counted once. Total **14.998718 mm²** including **assumed** 0.5 mm² support. Required digital area density derived, **not physically implemented**. | Conditional area, not fabricated iso-area |
| **Proof of lower bound / scheduling** | Existing necessary floors combine resource maxima and causal stages; new finite two-wave schedule is constructive, conditional. Serial sum is **not** a universal lower bound. | Methodology tested |
| **Programming and communication** | Charges cold K+V program (12.583 µs once), V local read (4 MiB/token), query, intermediate stage, FP16 context and LSE output. Context link **31.51× smaller** than FP32-score return. | Model addressed |
| **Fair SRAM+digital MAC** | Conventional SRAM capacity taken from the published 7-nm macro; digital PV is specified by explicit TMAC/s and required TMAC/s/mm². No simulated 7-nm MAC area fabricated. | Thresholds addressed; physical iso-area unqualified |
| **GPU GQA/suffix and full decoder** | Uses native GQA 32Q/8KV, source FP16 A100 grouped QK, corrected HBM/L2 and source full *unfused* prefix model; suffix LSE composition algebra tested. Full decoder TPOT remains null. | Partial |
| **Numerical degradation / circuit nonlinearities** | Retains recovered 96 Llama R0–R3 inference units, paired 95% CI and nonmonotonic R2/R3 behavior; new selective policy still needs end-to-end quality evaluation. | Historical quality supported; new policy unqualified |
| **Workload generalization OPT/Mistral** | Native model geometries have separate profiles; no OPT/Mistral inferred PPL or cross-precision matching. | Open for genuine inference |
| **Reproducibility** | Llama 96 archived inference units; 7-nm causal tests, 576 model coordinates, independent local 72-case replication and source-native MAGMA run. Latest research-branch CI **155 passed, one skipped**, all study artifacts uploaded. | Source-model simulation qualified |
| **Full-system latency, speedup, energy** | Reports **no full decoder TPOT or physical device energy**. Isolated modeled per-layer results cannot establish system speedup; no 7-nm analog-digital integrated silicon. | Not claimed |

## Defensible paper statement

> We present a source-calibrated architectural analysis of signed Q4×K4 shared-prefix LLM attention using a fabricated 7-nm SRAM-CIM macro and a separately published 7-nm SRAM macro. Extending the offload boundary through digital streaming softmax and PV reduces selected-layer output traffic by 31.51× relative to completed-score offload. A finite-residency causal schedule identifies conditional operating regions and their local V-read, digital-PV and bandwidth requirements. A byte-corrected native A100 FP16 source model shows that feasibility depends strongly on GPU KV-cache residency and peripheral throughput. We preserve measured-macro provenance, genuine Llama numerical evaluations and explicit limitations of unqualified integration circuitry.

**Do not claim:** measured GPU/CIM speedup; fabricated full-chip 7-nm physical implementation; matched fused Hydragen advantage; full-model TPOT or energy; established iso-area 7-nm MAC efficiency; new numerical quality on one-layer selective execution; generalized OPT/Mistral inference.

## Reproducible assets

- [Research branch CI — 576-case run, source tests, original comparisons](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/41868345731)
- [Source MAGMA GPU model execution](https://github.com/micsedlab-afk/MAGMA-gpt/actions/runs/41868354178)
- `src/cim_shared_prefix/prefix_context_engine.py`: causal QK/softmax/PV pipeline and numerical oracle.
- `src/cim_shared_prefix/source_roofline_corrected.py`: byte-corrected GPU source estimate, with original values preserved.
- `scripts/run_prefix_context_sweep.py`: 576 scenario execution, complete CSV/JSON uploaded as a workflow artifact.
- `results/integration/prefix_context_independent_local_20261104.json`: independent 72-case execution and reference.
- `results/quality/quality.json`, `paired_effects.csv`, `source_evidence.json`: preserved Llama numerical inference records.

**Evidence classification:** The source-backed architectural simulation and finite-resource response are reproducible. Hardware-validated full-system performance, actual synthesized periphery area and selective-policy inference quality are outside the supported evidence.
