# Independent scientific consistency audit — 4 November 2026

**Audited baseline:** `integration/final-framework@104debe57702fd12c8e17882588ef725d414b32c`. No numerical rows or results are altered here. This audit branch is isolated from the active integration branch.

## Executive decision

The framework is an **executed, reproducible macro-calibrated sensitivity and bounded scheduling study**, not a qualified GPU-vs-CIM architectural speedup or physically complete integrated 7-nm accelerator. A successful CI run is evidence for code execution and tested algebra; it does not validate unavailable silicon-level interface parameters.

Continue development only around the corrections and evidence gates below. Do not run more arbitrary deadlines or claim generic allocation optimization as original.

## 1. Physical technology and evidence provenance

| Element | Verified source or current artifact | Permitted statement | Prohibited extrapolation |
|---|---|---|---|
| Sinangil SRAM-CIM | JSSC 2021 DOI:10.1109/JSSC.2020.3031290, ISSCC DOI:10.1109/ISSCC19947.2020.9062985 | Fabricated 7-nm **8T charge-domain/analog MAV** macro; 64x64 cells, 4,096 physical bits, complete macro 0.0032 mm2, 5.5 ns, 1,024 unsigned I4xW4 native MAV terms, four-bit Flash ADC | Lossless 32b signed QK, a clocked multi-thousand-macro accelerator, 256 proved independent ports, 1 TB/s physically delivered programming, or use of its 372.4 GOPS as signed completed-score GOPS |
| Conventional SRAM | Yokoyama et al. VLSI 2020 DOI:10.1109/VLSICircuits18222.2020.9162985 | Published fabricated 7-nm 512x512 SRAM macro and reported macro density 29.2 Mb/mm2; capacity-area calculation | Equal CIM port bandwidth, 2RW capability, analog compute, or implemented SRAM+MAC block |
| GPU reference | NVIDIA Ampere architecture whitepaper/blog | A100 is TSMC N7; H100 uses TSMC 4N. A100 fits a 7-nm-generation *GPU reference*. | A100 and CIM macro are same PDK, can be integrated on one die, or H100 measurements are A100 measurements |
| Native macro arithmetic | `results/native_sinangil_arithmetic_receipt.json` | Source software arithmetic executed and matched ideal integer dot products | Hardware analog ADC equivalence or device/array-level execution timings |
| A100 LLMCompass | `MAGMA-gpt/src/magma/llmcompass_native_a100.py`, `MAGMA-gpt/docs/LLMCOMPASS_SCALESIM_SCIENTIFIC_CONTRACT.md` | A100 HBM2e **MHA FP16 heuristic model**, with generic A100 real-hardware microbenchmark files in vendored LLMCompass | Measured Llama GQA/BF16 INT4 QK, full exact-attention Hydragen A100 performance, or independent physical validation of injected MAGMA operator times |
| H100 acquisition | Archived HBM4PiM Mistral measurements | H100 evidence for the actual measured workload, precision and intervals only | 7-nm A100 proxy, matched Llama GQA GPU comparator, or fabricated SRAM-CIM measurements |

The Sinangil macro is **charge-domain/analog** with digital input/control and ADC; do not describe it as a fully digital 7-nm CIM datapath. S²-CIM is a separate fabricated **28-nm digital** accelerator.

## 2. Physical resource accounting (valid within its stated mapping)

For Llama-3.1-8B, L=32, H_KV=8, d_h=128, prefix P=2048, Q4/K4 mapped with 2 K sign planes:

- `2 * H_KV * ceil(d_h/64) * ceil(P/16) = 4096` complete CIM macros **per active layer**.
- Single layer: `4096 * 0.0032 = 13.1072 mm2` macro area.
- Simultaneous all 32 layers: `32 * 13.1072 = 419.4304 mm2`. This is **not** the minimum area for sequential execution.
- At a 16-mm2 budget with 0.5-mm2 *hypothetical* external support, a maximum of 4843 individual macros fit by raw area, or 4842 instantiated paired-sign macros; the serial study reports 1 pinned full layer + 373 signed tile-pair scratch units.
- Four unsigned array evaluations per signed logical Q4xK4 operation and 2 physical K sign planes are assumptions fixed by the pinned **project mapping**, not arbitrary capabilities measured for the native unsigned macro.
- After local signed partial reductions, one 32-bit transfer word per score gives `4*32*1*32*2048=8,388,608 bytes/token = 8 MiB/token`. Four bytes are the **transport word width**, not device arithmetic precision.
- For comparison, the 32-layer attention *context outputs* have `32 layers * 32 heads * 128 dimensions * 2 bytes = 262,144 bytes/token` at BF16, excluding Hydragen's LSE terms and any internal buffers. The QK-only CIM path returns **32 times more data** than final BF16 context outputs for this example. The actual GPU baseline does not necessarily transfer those context vectors over the same physical link.

These counts establish capacity and compulsory transfer conditions for the specified mapped QK-only architecture. The support area and interface topology, active ports, clocking, broadcast and achieved bandwidth are unverified.

## 3. Serious numerical/fairness correction

The macro ADC has four-bit output, and `R2/R3` numerical arms explicitly quantize and model a characterized/digitized circuit response. The resulting LLM attention is **dense and non-sparse** but **numerically approximate relative to BF16**.

Never call S2 'exact attention' in a precision sense. Precise wording: **dense shared-prefix Q4/K4 offload with evaluated numerical-quality deviation**.

Hydragen is algorithmically exact attention up to floating-point implementation effects. Compare **identical final attention-context output endpoints** (and then full decoder TPOT) while labeling precision/quality differences. The CIM design performs only prefix QK, returns completed score *words*, and still depends on GPU suffix QK, softmax, PV, projections and the rest of decoding. Comparing its QK score time to the Hydragen fused attention time alone does not establish competitive performance.

Current evaluated accuracy is only Llama 96/96 units (eight paired windows, three prefixes, four arms). `R3-R0` PPL increases are +2.11%, +3.38%, +6.43% at 512/2048/16384. This is model evaluation with digitized *simulated* device response, not measured model quality on fabricated CIM. OPT/Mistral have 0/96 full-model quality each; adapters and CPU arithmetic tests do not fill the gap.

## 4. Lower-bound mathematics: precise proof scope

`feasibility.py` uses optimistic **necessary** resource times (per-layer source macro-service cycles, amortized minimum cold/recurring key-program bytes, one query-source transfer and 32-bit score output) and combines independent resource floors by `max`. This max is a conservative lower bound only for the stated fixed interface, fixed mapping, cold-start protocol and specified maximum resource rates.

Conditional exclusion is justified only if a valid lower bound for the *same output endpoint and workload* exceeds a *qualified matching competing runtime*. The current denominator is the arbitrary `2 ms` completed-score deadline, not an observed A100 Hydragen time. Therefore the reported 128 'excluded' configurations mean **excluded from the hypothetical 2-ms deadline under chosen sensitivities**, not physically impossible CIM configurations and not slower than a GPU.

`integrated_schedule.best_serial` enumerates **whole-layer pin counts with symmetric scratch allocation**, not every legal cache placement, partial-tile pinning, independent port assignment or interstage overlap. Its result is the best computed **within that restricted serial policy family**. Do not claim globally optimal scheduling.

`tests/test_integrated_schedule.py` independently checks slot arithmetic and enumerates **that policy family** for small cases. It does not exhaust all general schedules or prove a universal global optimum. Its lower-bound comparison is a useful regression, not an independent mathematical theorem.

`integrated_schedule` uses provided 1e11/1e12 B/s links, 1e12 additions/s/scalings, and 256 active macro ports, plus a *zero controller delay* in the published sweep. These are explicit sensitivities, not characterized 7-nm multi-macro accelerator parameters. A numerical serial schedule under idealized sources should be called **conditional serial model time**, not hardware-achieved latency.

**Proof requirement:** formalize a complete bound with named compulsory transfers and ports for the QK-only interface. For each small representative case, independently enumerate arbitrary legal resident tiles and eviction schedules, not only the pinned-layer policy. Keep a lower bound separate from constructive policy and from physical rate qualification. Beware of adding overlapping stages when presenting an unavoidable lower bound.

## 5. Integrated data classification

Verified CI: GitHub Actions run `41812268605` succeeded (tests, source reproducibility and result publication); the final dataset-only commit `104debe5` follows that execution. Source project describes 127 passed, one skipped. Do not extrapolate to unexecuted native MAGMA/HBM4PiM, A100 kernels or full-model OPT/Mistral inference.

At HEAD `104debe5`, `results/integration/manifest.json` reports:
- 2,304 native-geometry architectural sensitivity rows.
- 128 conditional resource-floor failures versus 2-ms **hypothetical** deadline.
- 1,792 conditional serial-policy cases within that deadline.
- 384 bracketed by the floor and conditional serial model.
- **0 measured/qualified GPU TPOT; 0 measured speedup; 0 physical digital iso-area.**

Do not call 1,792 cases 'demonstrated achievable on silicon' or 128 'excluded vs optimized A100'. Those columns are conditional classifications.

For Mistral SWA, the existing integration takes *each visible prefix independently cold* in the constructive schedule, while omitting programming from the per-token necessary exclusion bound. These choices make conservative upper-model and optimistic lower-model asymmetry explicit; do not use the same interpretation of persistence as for fixed-prefix Llama.

## 6. Novelty relative to verified SOTA

- Hydragen (arXiv:2402.05099): optimized fused shared-prefix GPU attention with LSE recombination; requires a **real GPU performance baseline**, not merely functional PyTorch equivalence.
- CIM-Tuner (DATE 2026 DOI:10.23919/DATE69613.2026.11539276): already co-explores SRAM-CIM compute/storage capacity, area-constrained hardware mapping, two-level scheduling/tiling. **Generic area-constrained mapping is prior art**.
- Ouroboros (ASPLOS 2026 DOI:10.1145/3779212.3790197): already proposes dynamic KV allocation and communication-aware SRAM-CIM mapping for full-model wafer-scale inference.
- S²-CIM (IEEE TCSI 2026 DOI:10.1109/TCSI.2026.3742618): fabricated 28-nm digital KV-sharing CIM with sparse access and nonzero approximation; system claims use its own methodology.
- MoSKA (arXiv:2511.06010): shared-KV routing and specialized execution; distinct from our fixed-interface dense QK-only offload.

The only defensible **candidate** original contribution now is a formally justified *architecture-specific feasibility and no-go frontier*, calibrated to published 7-nm macro constraints, with full-costed output transport, bandwidth and measured **7-nm-generation A100** shared-prefix baseline. Elementary macro-capacity counting and the generic max-resource bound alone do not establish strong journal-level novelty. Novel findings must be supported by an actual GPU comparison and system-level conclusions within an explicit architectural interface.

If the feasible region disappears, that is a scientifically useful negative systems result *provided* the upper/lower bounds, competing GPU evidence and scope are rigorous; do not construct an artificial winning scenario.

## 7. Frozen next acceptance ladder

**Gate A (immediate):** lock the paper scope as **7-nm published macro calibrated, dense Q4/K4, QK-only GPU offload**. Document all extra assumptions (support-area, 256 ports, bandwidth, sign mapping, score word width, available A100 evidence), rename 'exact' and 'achievable' claims to respect numerical/physical status. No 7-nm synthesis.

**Gate B (mathematics):** independently validate lower-bound correctness and scheduler schedules under arbitrarily placed pinned macro tiles on small cases; prove what is mandatory for this *particular* QK-only interface; bound cold versus warm prefill placement; preserve Mistral sliding-window time variation.

**Gate C (GPU):** implement or obtain actual competitive Hydragen/FlashAttention-style GPU shared-prefix A100 timing with native GQA/MHA, correct precision and full attention-context endpoint. Existing MAGMA native A100 is MHA/FP16 heuristic timing, **not** measured native Llama GQA or signed INT4. H100 data stay outside primary 7-nm comparison. Until this is available, GPU-vs-CIM TPOT and speedup stay null.

**Gate D (physical costs):** retain controller, signed reduce, interface, macro write and support area as bounded parametric variables unless physical evidence exists. Calibrate only truly supported quantities; no arbitrary area allocation advertised as a feasible integrated device. No digital 7-nm iso-area claim without a complete comparable digital block.

**Gate E (quality):** preserve Llama R0–R3 and qualify OPT/Mistral inference on GPU when available; do not transfer Llama numerical quality across models.

**Gate F (conclusions):** regenerate only the matched, scenario-labeled exclusion/frontier results. Report bottleneck dominance, lower and serial conditional times, score-output bandwidth requirements and area caps. Novelty evaluated on the demonstrated new physical frontiers and GPU comparison, not on an unvalidated optimizer.

**Do not** use 2ms deadline as a GPU timing. **Do not** derive a 7nm integrated silicon design or frequency from a fabricated macro. **Do not** call the R2/R3 QK path numerically exact. **Do not** report energy advantage from unsupported GPU/CIM system power.

## 8. Audit addendum — live integration at `67ed4b4fa628`

The integration branch advanced by four commits after the `104debe5` snapshot. The audit's claims about **missing qualified GPU timing** remain valid, but implementation status must be updated:

- `src/cim_shared_prefix/gpu_attention.py` now implements grouped-prefix GPU GEMM, full SDPA and GPU-side post-CIM score remainder, with a common *attention-context output* endpoint.
- `scripts/benchmark_gpu_attention.py` now instruments actual CUDA kernel candidates via CUDA events and profiler, enforces FP16/BF16 and GQA geometry, and records GPU identifiers. **No A100 measurement file was found in the integration repository; the latest CI is CPU-only and tests its fail-closed behavior.**
- `scripts/benchmark_real_decoder.py` profiles real pretrained-weight decode only on a CUDA-enabled machine. It explicitly labels the ordinary Hugging Face SDPA baseline as **not Hydragen optimized**.
- `tests/test_gpu_attention.py` verifies MHA/GQA numerical functional agreement and absence of fabricated records on CPU.
- Latest full CI `41818658065` at `67ed4b4fa628` succeeded. This is **CPU regression**, not GPU measurement.

**Remaining GPU qualification**: execute on verified A100; confirm installed kernel identity and fair preassembly/cache assumptions; time optimized Hydragen, native full-SDPA, real decoder; compare the same output endpoints, precision, batch, prefix and suffix. If run on H100, retain a distinct hardware class and exclude the record from the primary A100 N7 comparison.

**Numerical terminology correction still required**: `gpu_attention.py` may legitimately call its floating-point full attention reference exact (mathematically up to numeric rounding); the complete CIM path must remain labeled *approximate Q4/K4/O4 physical-response emulation*. Its GPU-side remainder currently accepts arbitrary FP32 prefix score tensors; its equality tests with BF16 prefix scores do **not** establish the actual Q4/K4/O4 ADC response equivalence.

**Operator profiling semantics**: layer-level per-kernel CUDA timing must not be naively added to uninstrumented full decoder TPOT. End-to-end output timings need a complete causal model and actual measured hardware state.

