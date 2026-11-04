# Source-grounded S0/S1/S2 system execution: evidence and boundary

**Branch:** `work/system-path`. The original `system.py`, its unit tests, the existing runner and frozen contracts are preserved. New implementation: `source_grounded.py`; exact GQA operator geometry and conditional native integrations: `gqa_native.py` and `scripts/run_{gqa_llmcompass,hbm4pim_gqa_local,matched_system,source_sweep}.py`.

## Evidence recovered and executed

The pinned original `micsedlab-afk/shared_prefix_cim@a8a3f7238c471da2f3b7e6fe5647f8f25b8d84c4` contains:

- `experiments/sram2llm_shared_prefix/sinangil_ideal_macro.py`, Git blob `6e968ef3b4cd2b09529930ad24cce66425ff93e0`;
- `experiments/sram2llm_shared_prefix/sinangil_mappings.py`, Git blob `78c30ae3996f1cd830d2a6d08ebb1685cedbe645`;
- `inputs/sinangil7nm/target_descriptor.json`, Git blob `4516856f81fc383a64f946cda1e3682e36bac321`.

The two source files were recovered byte-for-byte and their Git blob identities independently verified. They remain uncommitted in `_upstream` as the public repository source policy requires. `scripts/run_native_sinangil_check.py` **executed the actual unmodified upstream Python functions**: 96 ideal macro calls and 64 positive/negative signed-mapping calls, checking respectively 192 and 128 complete integer Q4/K4 scores against direct 128-dimensional dots with zero exact-integer discrepancy. Source evidence: `results/native_sinangil_arithmetic_receipt.json`. This is an ideal digital arithmetic execution and **not a device-timing, ADC or real-silicon experiment**. The upstream signed mapping identifies K-bank replication as paper-supported and the Q+/Q- split as a project assumption. Real LLM quality is outside this module.

MAGMA-gpt, SRAM2LLM and HBM4PiM source trees were inspected through GitHub. `git clone` to the local runtime was unavailable (DNS/network restriction), and workstation archives were absent. Consequently the native integrated MAGMA/LLMCompass and HBM4PiM runs were **not executed**. Exact status is recorded in `results/upstream_runtime_availability.json`. These upstream implementations are invoked on an appropriately equipped checkout through `scripts/acquire_upstreams.sh`; no incompatible surrogate timing is generated here.

## Source geometries and physical schedule

Llama-3.1-8B: `d_model=4096`, `d_ff=14336`, 32 layers, 32 Q heads, 8 K/V heads, head dimension 128, GQA=4, vocab=128,256, BF16 primary reference, SwiGLU/SiLU, RMSNorm, RoPE. Source: pinned `shared_prefix_cim/docs/sram2llm/SCIENTIFIC_CONTRACT.md` and model configuration. The original paired numerical axes are `Lp={1024,2048,4096,8192,16384}`, `Lu={128,256}`; serving B/O/memory/link axes are **added sensitivities**, not mislabeled original measurements.

The Sinangil 7-nm **complete 8T SRAM-CIM macro** stores 4,096 bits, 64 reduction rows × 16 independent I4×W4 outputs, area 0.0032 mm², 5.5-ns nominal native unsigned operation, 1,024 MAC per native access, 372.4 GOPS when MAC is counted as 2 ops. DOI [10.1109/JSSC.2020.3031290](https://doi.org/10.1109/JSSC.2020.3031290). Physical-area accounting follows `work/area-digital/docs/area_method.md`: charge complete macro once, never derive logic area by node scaling, arbitrary density or divided whole-chip area. **Off-macro signed reductions, reprogramming infrastructure, controller, network and interface area are unknown.**

For Llama Q4/K4, per layer and per KV head:

- physical K macros = `2 × ceil(head_dim/64) × ceil(Lp/16)` (2 K sign banks);
- total full-residency macro count = the above times `layers × KV_heads`;
- native unsigned macro calls/token = total macro count times `2 × B × GQA_ratio` (Q+/Q- for each K sign bank); Q8/K4 additionally doubles input nibble passes;
- completed score words/token = `layers × B × Q_heads × Lp`, always 32-bit scores across the accelerator–GPU boundary;
- off-macro signed partials per completed score = `4 × ceil(head_dim/64)` (eight for D=128); `partials−1` signed combines and `partials` signed/scaled partial multiplications must be externally timed, independently of the on-macro 4-bit ADC.

**Finite occupancy:** `PhysicalCIM.macro_budget` provisions an explicit count of entire 512-byte CIM macros. When all source K banks fit, each is loaded/programmed once and reused over `O` output steps. Otherwise, one macro is reserved as a scratch bank, fixed macros are divided deterministically across layers, and unpinned banks are loaded and programmed once **per generated token**. Requests sharing the prefix reuse the same K contents; K is not multiplied by B or by 4 Q heads per KV group. A layer completes before the next layer begins. Native independent active-macro count is an **explicit physical concurrency sensitivity**, bounded by the provisioned macros. The source-derived 5.5-ns macro-only lower bound assumes ideal independent simultaneous macro ports; the separately reported serialized scratch schedule assumes one reusable streaming bank. Neither includes unknown setup or non-macro timing. Explicit zero controller/setup costs remain optimistic sensitivity assumptions.

**S1 SRAM physical reference:** Yokoyama et al., VLSI 2020, DOI [10.1109/VLSICircuits18222.2020.9162985](https://doi.org/10.1109/VLSICircuits18222.2020.9162985), complete 512×512-bit, 7-nm SRAM macro density 29.2 Mbit/mm². Whole 32-KiB macro occupancy is charged per KV head per layer; optional `sram_macro_budget` likewise enforces one scratch macro if full K does not fit. Logical signed Q4/K4 keys are stored once, transferred to the GPU per token, and computed only if an **actual signed Q4/K4 GPU kernel profile** is available. The S1 write/reload service requires its own explicitly sourced bandwidth (not CIM program bandwidth). The SRAM macro publication does not specify the required integrated read, write, or GPU link throughput.

## Strict end-to-end endpoints

For each generated token, with exact context `C=Lp+Lu+token_index+1`:

| Path | Completed-score source | Additional data/timing |
| --- | --- | --- |
| **S0 GPU** | Complete **full-context** GQA QK on GPU, including both prefix and suffix | All Q/K/V projections, RoPE, RMSNorm, softmax, grouped A×V, output projection, 14,336-unit SwiGLU gate/up/down/multiply, KV update, output LM head and sampling; profile must match GPU precision and memory system. |
| **S1 SRAM** | Signed Q4/K4 **GPU** QK over resident/refilled K; native GPU suffix QK; merge to full scores | Initial physical SRAM placement/write, SRAM refills, SRAM→GPU effective read+link transfer once per token, exact INT4 kernel and quantization, plus identical remaining decoder stages. |
| **S2 CIM** | Four native unsigned sign paths over the prefix, scaled/digitally merged to **completed 32-bit scores** and delivered to GPU; native GPU suffix score | Initial macro loading/programming amortized by O, recurring streamed fills/programming, native access, query transfer/quantization, signed combine and scale, score transfer, explicit controller, plus identical remaining decoder stages. |

The causal **default is serial**; no unproven prefetch/compute/score overlap is credited. Non-QK GPU stages are taken once per complete decode block. The full-context QK stage is used by S0; it is **excluded from S1/S2** and replaced by separately measured or source-native prefix/suffix stages. The previous `magma_a100_block()` MHA/FP16 adapter remains supported for its exact legacy workload, but cannot be relabeled Llama GQA/BF16/INT4. The strict `GQAGPUPoint` contract rejects MHA, wrong FFN size, mixed score precision, wrong context per token and absent stages. LM head and sampling are charged **once per generated token**, not once per layer.

The original LLMCompass transformer block represents **MHA/GeLU/LayerNorm**. `gqa_native.py` invokes original `Matmul`, grouped `BatchedMatmul` (Q `[B,8,4,128]`, K `[B,8,128,C]`) and `Softmax` at exact Llama dimensions; K/V projections are `[4096,1024]` and SwiGLU gate/up/down GEMMs use 14,336. Source A100 device provides **FP16/HBM2e only**. Running it returns **partial FP16-GQA architectural operator models**, with RMSNorm, RoPE, SiLU/gate multiply, structural service and signed INT4 QK explicitly missing. This is not the source-primary BF16 model or a complete decoder. `run_hbm4pim_gqa_local.py` optionally requests source-bound **H100 BF16 local operator cycles**, whose external HBM4 and host service are excluded and therefore cannot become full TPOT. Native run requires actual pinned private checkouts and model-specific operator calibration.

## Quantities generated and claim status

The deterministic `scripts/run_source_sweep.py` exports `results/source_grounded_sensitivity.csv` (720 coordinates), `results/source_grounded_manifest.json` (ten original `Lp,Lu` shapes and provenance), and native source receipts. Axes: original Lp/Lu, added `B={1,2,4,8}`, `O={1,32,128}`, CIM capacity 4,096 / 32,768 / full-residency macros, and assumed common bandwidth `1e11 / 1e12 bytes/s`. Active macro ports=256; reduction/partial-scaling rates=1e12 op/s; periphery controller and query quantization cost=0 **as explicit optimistic sensitivity only**. No arbitrary value is cited as an actual measured link rate, controller cost, silicon power or GPU time.

For `B=1,Lp=2048,Lu=128,O=128`: complete CIM K macro area is 419.4304 mm² and complete SRAM key-storage footprint is 9.192995 mm² (both excluding external support area). At **4,096 CIM macros**, 1 TB/s assumed link/program rate, 256 ideally independent macro ports and a single serial streaming scratch, the conditional **non-GPU** S2 contribution is 5.758 ms/token, of which macro service is 5.588 ms/token. With **131,072 macros** fully resident and 256 active ports, the same non-GPU sensitivity decreases to 0.06349 ms/token, of which macro service is 0.02253 ms/token. These figures are **non-GPU hypothetical sensitivity outcomes**, not complete TPOT or measured speedups, and the full-resident physical macro footprint remains large. A complete digital iso-area result is unsupported under `docs/AREA_EVIDENCE_POLICY.md`.

**Currently qualified end-to-end TPOT / speedup coordinates: zero.** `S0_full_TPOT_s`, `S1_full_TPOT_s`, `S2_full_TPOT_s`, `S0_S2_speedup` and `S1_S2_speedup` remain null until compatible native operators, exact quantization, physical access/programming and support costs have all been sourced. Even a user-provided complete stage profile with declared rates is labeled a conditional model, not a silicon measurement. There is no undocumented overlap or interpolation of native geometry.

## Commands and acquisition

```sh
bash scripts/acquire_upstreams.sh                     # credentialed workstation with network
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/run_native_sinangil_check.py --root . --output results/native_sinangil_arithmetic_receipt.json
python scripts/audit_upstream_runtime.py --root . --output results/upstream_runtime_availability.json
python scripts/run_source_sweep.py results
python scripts/run_gqa_llmcompass.py --magma-root _upstream/MAGMA-gpt --batch 1 --prefix 1024 --suffix 128 --output-tokens 1 --output /tmp/gqa-a100.json
python scripts/run_hbm4pim_gqa_local.py --hbm4pim-root _upstream/HBM4PiM --work-dir /tmp/h100gqa --output /tmp/h100gqa.json
python scripts/run_matched_system.py --manifest source-verified-scenario.json --output result.json
```

Upstream original files remain in `_upstream/` and **must not be committed**. Real upstream vendor simulations require the correct dependency versions and hardware-specific source revisions, plus explicit full operator and system evidence; this sandbox lacks these upstream installations. No 7-nm synthesis or hardware-model modification was performed.
