# Quality and workload-profile evidence — `work/quality-workloads`

## Recovered evaluated numerical results

Source: private `micsedlab-afk/shared_prefix_cim`, branch
`results/final-quality-scalar-n8-20260923`, directory
`results/sram2llm/final_quality_scalar/`, and upstream implementation
`task/sram2llm-sinangil7nm-20260919` at `a8a3f7238c471da2f3b7e6fe5647f8f25b8d84c4`.
Original files: `sample_arm_n8.csv`, `sample_effects_n8.csv`,
`summary_n8.json`, `protocol.json`, `provenance_n8.json`,
`scalar_qualification.json`, `execution_n8.json`.

Model: **base** `meta-llama/Llama-3.1-8B` at
`d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`; dataset:
`Salesforce/wikitext` **validation**, `wikitext-2-raw-v1`, dataset revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. Frozen held-out
subset: eight disjoint samples, with 128 scored suffix tokens per sample,
for Lp in {512,2048,16384}, batch 1. All four arms share sample
identities and target tokens. The original source archive's 96 numerical
units were reported complete. `scalar_qualification.json` marks scalar
state reuse PASS. Publication export ingests exact upstream summary and
reconciles against original per-window NLL sums; original private raw
archives and model weights are **not** copied to this public repository.

**R0:** reference BF16 prefix attention QK. **R1:** Q4/K4 exact
mathematical signed accumulation. **R2:** exact same codes with ideal
linear macro response, source-constrained signed split, physical tiling,
ideal 4-bit ADC and reconstruction. **R3:** mapped R2 plus
source-digitized Sinangil Fig.16 simulated-circuit response. The ADC
policy in these recovered results is
`sinangil_static_per_layer_r3aware_nearest_o4_v2`, **not** the earlier
maximum-R2 calibration defined in `docs/sram2llm/SCIENTIFIC_CONTRACT.md`.
Do not mix numbers from the two protocols. The source-digitized response,
quantizer and ADC assumptions are part of the evaluated software model.
These are language-model evaluations under an emulated hardware response;
no silicon LLM evaluation, measured PIM power or hardware-area result is
implied.

Global NLL = sum of token NLL / number of scored tokens, and
PPL = exp(global NLL). Source confidence intervals are paired
nonparametric bootstrap over eight held-out samples (10,000
resamples, seed 20260917, percentile 95% CI); resampling tokens
independently would understate dependence. PPL CIs refer to absolute
PPL; paired effect CIs come from the joint resamples and cannot be
estimated by subtracting marginal CI endpoints. Per-sample PPL
averages must not replace global PPL. `quality.py` rejects missing arms,
missing paired windows, malformed intervals, inconsistent NLL/PPL, and
inconsistent window sums.

## Source-grounded model geometry and sweep classification

| Model | Config source | Layers | Q heads | KV heads | Head dim | Native GQA | Context / caveat |
|---|---|---:|---:|---:|---:|---:|---|
| Llama-3.1-8B base | [pinned config](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/config.json) and archived numerical manifest | 32 | 32 | 8 | 128 | 4 | 131072 max positions |
| Mistral-7B-v0.1 | [pinned 1768803 config](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/17688039ea7b7001c702797eed2ab7716a0cc3c2/config.json) | 32 | 32 | 8 | 128 | 4 | max positions 32768, sliding attention window 4096 |
| OPT-2.7B | [model config](https://huggingface.co/facebook/opt-2.7b/blob/main/config.json) (mutable main; pin before any future numerical run) | 32 | 32 | 32 | 80 | 1 (MHA) | 2048 maximum positions |

The `geometry_sweep.csv` covers model-appropriate prefix lengths,
suffix {32,128,256}, batch {1,8,128}, output {16,128,256}, and
GQA ratios {1,2,4,8}. At non-native ratios, KV head count is varied
while Q heads remain fixed: **counterfactual architecture-only geometry**.
No such ratio change is attributed to the original trained weights.
Prefix K is unique and shared across the batch; query scores scale
with batch and Q heads. Completed QK scores cross at 32 bits each.
Signed Q4/K4 path factors (2x K storage, 4x array evaluations) are
explicit project architectural assumptions from `SignedMapping`.

For a sliding window of W including the query position, generated
output step j (zero-indexed) sees
`min(prefix, max(0, W-1-suffix-j))` original prefix keys. The
Mistral source config is revision-dependent; changing revision or
attention implementation requires revisiting this calculation.
SWA eviction makes prefix visibility decrease with suffix and output.
All modeled operations and traffic are summed over generated positions;
original R0--R3 absolute PPL only scores teacher-forced suffix positions
and **cannot** be assigned to a sweep row or extrapolated over generated
output lengths.

Only source-checked geometry is emitted for Mistral and OPT. They have
**no evaluated R0--R3 NLL/PPL** in the recovered archive. The geometry
sweep does not claim model quality, throughput, latency, or energy.

## Reproduction / local archive recovery

```bash
python -m unittest discover -s tests -v
python scripts/export_quality_workloads.py --profile-only --out results/quality
python scripts/export_quality_workloads.py \
  --archive-dir /path/to/shared_prefix_cim/results/sram2llm/final_quality_scalar \
  --out results/quality
```

The exporter verifies byte hashes for `protocol.json` and the result
artifacts listed in `provenance_n8.json`, reconciles NLL with the 96
window-level units, and writes CSV/JSON and two vector SVG figures.
Requires the actual local archived files. **No GPU inference is run**
when the archive is intact. New model-specific numerical evaluation
requires separate adapted attention injection, pinned model/tokenizer
and dataset revisions, and available model-weight/GPU execution.

The October batch-2 helper trial in
`analysis/sram2llm/FINAL_QUALITY_BATCH2_QUALIFICATION_BLOCKER.json`
remains blocked: R0 scalar-vs-batch per-token NLL failed the fixed
2e-5 equivalence criterion. Its development sample numbers must not
be mixed into this n=8 scalar result or advertised as batch-2 quality.

## Extended quality implementation

See [extended_quality_method.md](extended_quality_method.md) for exact pinned OPT/Mistral revisions, BF16 R0 reference, separate model-specific ADC calibration, native attention semantics, and GPU execution gates. The recovered 96-unit Llama R0–R3 archive, original NLL/PPL and bootstrap intervals remain unchanged. Model-specific quality results are pending full-model GPU execution; geometry sweeps remain geometry-only.
