# Extended numerical-quality inference: OPT-2.7B and Mistral-7B-v0.1

Branch: `work/quality-workloads`. Source code: `src/cim_shared_prefix/quality_inference.py`, `quality_evaluation.py`, `scripts/run_extended_quality.py`. This supplements, without changing, the recovered Llama-3.1-8B 96-unit archive, its `sinangil_static_per_layer_r3aware_nearest_o4_v2` calibration, the original `quality.json`, or its confidence intervals.

## Status and evidence boundary

**Full-model OPT/Mistral inference is not completed in this repository.** The development environment used to prepare this branch has CPU-only PyTorch, no installed Transformers 4.46.2, no pinned full-model weight snapshots, and no CUDA device. The new model-quality rows are deliberately absent. Geometry-only data are retained and labeled. Full-model qualifications, new-model ADC calibration, and evaluation must execute on a BF16-capable GPU with the original pinned SRAM2LLM source and the exact pretrained model snapshots. CPU tests use *synthetic input tensors exclusively to verify arithmetic*, never to report model perplexity.

## Source identities

| Model | Model snapshot | Native heads | Context | Precision / caveat |
|---|---|---:|---|---|
| [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/17688039ea7b7001c702797eed2ab7716a0cc3c2/config.json) | `17688039ea7b7001c702797eed2ab7716a0cc3c2` | 32Q/8KV, d=128, 32 layers | 32,768 positions, sliding attention W=4,096 | BF16 checkpoint |
| [OPT-2.7B](https://huggingface.co/facebook/opt-2.7b/blob/905a4b602cda5c501f1b3a2650a4152680238254/config.json) | `905a4b602cda5c501f1b3a2650a4152680238254` | 32Q/32KV, d=80, 32 layers | 2,048 positions | Original checkpoint FP16; evaluation loads BF16 copy, explicitly recorded |

Transformers source is frozen to 4.46.2. Mistral source Git blob `321d3dc0daf378b18e9bde140205c0e2fb2c9e7b`, OPT source Git blob `e4ef510f099d665e2a69e1b45c2f7c0dfd8cf4b2`, shared `cache_utils.py` Git blob `0f696cc3ac6a4d63ac3cf071ef620a9a083b5a05`. The execution backend must match SRAM2LLM Sinangil source SHA-256 `11d8d252b0bbe1f3ed558609a7a0de188b317b9b4875a29cd6d34925f800ba36`. Full-run provenance stores checkpoint revision, dataset/token stream hashes, calibration digest, and a protocol hash. Model snapshots supplied locally must be exact-HF-revision snapshot directories. Pinning repository revisions does not independently constitute execution of inference.

## Numerical intervention and model-specific qualification

The original forward implementation performs Q/K/V projection, OPT positional embeddings or Mistral RoPE, cache update, and GQA head repetition. The adapter only intercepts the source SDPA Q/K-score boundary during a teacher-forced suffix call; controller-disabled prefill remains the unmodified source model. R0-native is executed as a strict equivalence gate against an unwrapped original model call using cloned prefix caches (per-token NLL absolute tolerance 2e-5). The actual reported R0 uses **eager BF16 QK, scaling, FP32 softmax, PV**, so the reference arithmetic matches the original four-arm Llama study; its kernel choice is explicitly recorded.

R1 is symmetric signed Q4/K4 exact accumulation using the original `SinangilQKExecutor`; R2 adds its two 64-row groups, 16-key mapping and nearest four-bit ADC; R3 replaces each weight-bit-plane response with the source-digitized reported *simulated* Sinangil Fig.16 transfer. The same Q4/K4 state, signed paths, physical schedule and calibrated policy must be used for a given arm pair. R3 support violations are fatal. Model-specific project ADC calibration uses held-out-independent development windows and 17 candidate full-scale ratios 0.20–1.00, selecting the ratio minimizing R3-vs-R1 normalized MSE. A selected boundary ratio or missing per-layer policy blocks held-out evaluation. There is no transfer of Llama ADC thresholds to OPT/Mistral.

**OPT d=80:** the backend requires two 64-row tiles. The active 80 rows are zero-extended to 128, preserving 64+16 active dimensions and quantizer scales. R0 remains native d=80. The source OPT SDPA Q already carries a 1/sqrt(80) factor; for Q4/K4 the adapter obtains unscaled `q_proj(hidden_states)` before invoking the backend and scales the returned scores once. OPT is native MHA; no fictitious GQA ratio enters numerical inference.

**Mistral W=4096:** query at absolute position `Lp+j` sees original prefix keys `k` satisfying `max(0,Lp+j-W+1) <= k < Lp`. The adapter intersects source causal/padding masks with explicit SWA semantics. For every suffix position, it dispatches only that visible prefix interval to the backend; keys evicted before that query do not contribute to CIM calls or `visible_prefix_cim_scores_by_layer`. This covers a 16,384-token prefill with only up to 4,095 currently visible prefix tokens. Mistral native Q:KV ratio is four; counterfactual ratios in `geometry_sweep.csv` have **no numerical-quality data**. With `DynamicCache`, the source model retains full physical K while its causal/sliding attention mask excludes evicted keys; the adapter independently enforces the same visibility.

## Evaluation design and statistics

Dataset: exact `Salesforce/wikitext`, `wikitext-2-raw-v1`, **validation**, pinned revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Full text is joined with `"\n\n"` and tokenized without special tokens. Each model uses **its own pinned tokenizer**. Nonoverlapping blocks of `max(Lp)+128+1` tokens preserve identical 128 targets across prefix lengths within a model. The first two blocks are development-only; eight different blocks are selected by fixed SHA ranking before inspection of any R1–R3 numerical results. Test windows and calibration windows are disjoint, and all four arms share tokens, cached prefix state, and target IDs.

- Mistral Lp = {512, 2048, 16384}; OPT Lp = {256, 512, 1024}; suffix = 128; batch = 1; 8 paired blocks × 3 prefixes × 4 arms = **96 evaluated units required per model**.
- NLL is negative FP32 log probability of the scored next token; PPL = exp(token-weighted mean NLL). Every sample/arm stores 128 token NLL values plus the canonical target hash. Output-token generation length is not extrapolated into quality claims.
- For each prefix, the bootstrap draws the *same eight sample indices for every arm*, 10,000 replicates, RNG seed `20260917`; report 95% percentile CI for absolute NLL/PPL and paired ΔNLL/relative ΔPPL. Never bootstrap individual tokens or average per-window perplexity.
- Causal components: `R1−R0` quantization; `R2−R1` mapping/ADC; `R3−R2` characterized Fig.16 response; `R3−R0` overall. Negative differences are permitted. Source circuit response is **simulated/digitized** and ADC is a **project assumption**; full-model NLL is an **executed model/software numerical result** only after all units complete.

## GPU execution instructions

From a workspace containing the exact upstream `shared_prefix_cim` branch and a pinned checked-out `sources/SRAM2LLM` Sinangil backend:

```bash
python -m pip install 'transformers==4.46.2' datasets accelerate sentencepiece safetensors numpy torch
export PYTHONPATH="$PWD/src:$PYTHONPATH"
# Rerun with --model mistral for the GQA model.
python scripts/run_extended_quality.py --model opt --mode qualify \
  --upstream-root /path/to/shared_prefix_cim --allow-download
python scripts/run_extended_quality.py --model opt --mode calibrate \
  --upstream-root /path/to/shared_prefix_cim --allow-download
python scripts/run_extended_quality.py --model opt --mode evaluate \
  --upstream-root /path/to/shared_prefix_cim --allow-download
# Reaggregate already evaluated complete units without loading the model:
python scripts/run_extended_quality.py --model opt --mode aggregate
```

The calibration and evaluation commands require their prior PASS artifacts. By default `--allow-download` is off; cached model/dataset inputs are required. `--model-root` may point to an existing exact-revision Hugging Face snapshot. By default outputs go to `results/quality/extended/{opt,mistral}/`, a distinct namespace from the protected Llama archive. **Do not commit incomplete raw unit files as publication data.** A complete campaign writes `absolute_quality.csv`, `paired_effects.csv`, `quality.json`, `fig_quality_effects.svg`, and validated unit/protocol/calibration manifests. If the CUDA/model gates fail, it writes no model-quality numbers.

## Existing archive: nonmonotonic R2/R3

The independently recovered Llama-3.1-8B scalar n=8 archive, `shared_prefix_cim/results/final-quality-scalar-n8-20260923`, reports `R3−R2` ΔNLL of +0.006189 (Lp512), −0.013573 (Lp2048), +0.011106 (Lp16384). Paired 95% CIs include zero in **all three** cases; the negative 2048 result is not a resolved quality improvement. Per-token cross-entropy is not monotonic in the magnitude of QK-score perturbation. R3 response can change direction of some score errors relative to R2, sometimes partially compensating mapping/ADC bias after softmax. Whether this specifically explains Lp2048 requires source-level per-layer score and attention diagnostics; the archive's aggregate NLL alone cannot determine mechanism. The stored `R3−R2` sign must never be overridden with a monotonic degradation assumption. See source-traceable `results/quality/llama_causal_nll.csv` and `fig_llama_causal_nll.svg`.

## Validation boundary

Local CPU tests cover complete pair cardinality, deterministic bootstrap including negative R3 effects, source R0 delegation, eager-score arithmetic, exact OPT scaling and 80→128 zero-extension, repeat-KV head equivalence, and excluded Mistral evicted-prefix scoring. **These do not substitute for model-specific CUDA integration tests**, source-eager full-model R0 equivalence, development ADC qualification, or any new PPL measurement. Batch-2 remains blocked and outside scope.
