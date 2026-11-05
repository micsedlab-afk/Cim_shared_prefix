# GPU publication evidence recovery — 2026-11-04

> **A100-primary correction (5 Nov 2026).** The 7-nm OJCAS architectural comparison remains A100 80GB FP16 (MAGMA/LLMCompass) versus the published 7-nm SRAM-CIM system. A100 timings in this repository are source/operator models, not physical CUDA data. Historical H100 acquisitions cannot replace A100. The following H100-first experiment instructions concern separate GPU evidence acquisition and **do not define the primary A100 scientific comparator**. See [A100_PRIMARY_REVIEWER_REAUDIT_20261105.md](A100_PRIMARY_REVIEWER_REAUDIT_20261105.md). The competing compressed A100 GPU source model and the selected-layer-specific KV footprint must be included when interpreting full-prefix results. FP16-centroid GPU and ideal Q4K4-CIM are not numerical-equivalent by assumption.\n\n## Immediate conclusion

A **competitive region against optimized Hydragen has not been demonstrated**.
The original 2 ms architectural target was only a hypothetical deadline and
must not be used to claim GPU speedup or impossibility against a measured
GPU. There are **zero** exact matched, end-to-end GPU/CIM coordinates in the
current public repository. The independent physical necessary floors and
published macro area remain valid *conditional* information.

No GPU is connected to the execution environment used for this work:
PyTorch 2.10.0+cpu, `torch.cuda.is_available()==False`, no `nvidia-smi`,
and no reachable SSH/Slurm integration. Connected GitHub access reads sources
and commits code, but cannot execute on the UNISA machines. The available
GitHub Actions runner is CPU-only. **No CUDA benchmark, quality inference or
model evaluation has been fabricated to fill this gap.**

## Specific source recovery

| Asset | Current direct accessibility | Scientific use |
|---|---|---|
| `micsedlab-afk/HBM4PiM` research branch `research/h100-mistral-acquisition-20261026` | GitHub files readable | H100 experiment source code and isolated-GEMM calibration |
| `calibration/native/H100_MEASURED_REFERENCES_465.csv` | Read from GitHub; 465 records | Two exact `1x4096x14336` H100 GEMMs: BF16 39.44025 µs, FP16 39.699251 µs, each `PASS_REUSED_FROZEN`, **no inter-round spread**. These are isolated GEMM measurements only. |
| H100 source branch, `experiments/h100_cuda_acquisition` and `experiments/h100_comprehensive_real` | Source README and Python tools readable | Measurement protocol and archive indexing; **raw on-host acquisitions not mounted** |
| UNISA `~/H100calib/acquisition_20261026` and `~/H100calib/comprehensive_v2_20261026` | Historical filesystem paths, no remote shell | Potential original H100 pretrained Mistral model/serving measurements; must audit raw files before reuse |
| UNISA `~/HBM4PiM` | Historical path, no remote shell | Original HBM4PiM/H100 local worktree |
| `micsedlab-afk/MAGMA-gpt`, `SRAM2LLM`, `shared_prefix_cim` | Source repositories readable; full local checked-out dependencies absent | Native LLMCompass and source Sinangil arithmetic adapters; **not** a complete optimized GPU benchmark |

The historical H100 model snapshot was
`mistralai/Mistral-7B-v0.1` revision
`27d67f1b5f57dc0953326b2601d68371d40ea8da`.
The integration's Mistral source profile is
`17688039ea7b7001c702797eed2ab7716a0cc3c2`.
These **must remain different revision evidence** until weight/config hashes
prove equivalence. The H100 acquisition is FP16; the original OPT/Mistral
R0-R3 extension is an explicit **BF16** evaluation protocol. No conversion
from a measured FP16 kernel time to BF16 is valid.

For reference Llama-3.1-8B, P=2048, suffix=128, B=1, O=128, BF16:
the H100 calibration has no exact full prefix-attention GEMM
`(M,K,N)=(4,128,2048)`, nor the required exact QKV/down/LM-head
projection shapes at M=1. Matching one FFN up-projection GEMM does not fill
the missing operators.

## Actual GPU execution now implemented

1. `scripts/benchmark_gpu_attention.py` tests one native model geometry and
   GPU dtype at individually specified decode steps. It benchmarks a
   **native Flash-SDPA** full attention reference with preconstructed
   contiguous full K/V, a genuine shared-prefix batched GEMM (one K/V per
   KV head reused across B x GQA query rows), a `torch.compile`/Inductor
   variant and the GPU remainder after **32-bit completed prefix QK scores**
   have reached GPU memory. Suffix QK/PV and two separate logsumexp
   normalizations are fused to the complete attention output. Explicit
   CUDA events, CUDA graph replays and Kineto kernel names accompany each
   result. The existing PyTorch reference validates numerical identity.
   Only direct GPU measurements can establish which candidate wins.
   Default input tensors are deterministic **synthetic activations**; the
   resulting event times, when run on CUDA, are real kernel measurements,
   not pretrained-activation or complete-model measurements.
2. `scripts/benchmark_real_decoder.py` measures real pretrained
   Llama/Mistral/OPT model decode on a pinned revision, with real causal
   KV-cache advance, greedy output sampling, all decoder layers, LM head
   and explicit per-generated-token CUDA events. A separate hooked probe
   records real leaf-operator GPU service and kernel names; its
   instrumentation overhead is never mixed with uninstrumented TPOT.
   Its native HF SDPA attention is **not** claimed optimized Hydragen.
   Only a local exact-revision snapshot with real safetensors is accepted.
   Mistral SWA is checked against the source config.
3. `scripts/compare_measured_attn.py` requires **every one of the 128
   output-step measurements**, same native revision, batch and precision,
   real CUDA kernel evidence, full Flash/optimized shared-prefix output,
   and an actual GPU score-to-attention remainder measurement. It
   automatically rejects CPU absence, missing steps and geometry
   substitution. The necessary CIM-QK resource floor alone is compared
   against real measured full GPU attention; overlapping GPU remainder
   is **not added to a lower bound**. A serial conditional feasible
   schedule adds the measured GPU remainder after the QK+score transport
   stages. Unknown physical rates and external support remain named
   sensitivity assumptions. The generator exports a JSON and CSV only
   after these gates pass.
4. `scripts/plot_measured_competition.py` plots the GPU full-attention
   measurement against the conditional CIM QK floor and complete-attention
   serial candidate. The plot refuses non-measurement input. No artificial
   baseline figure is emitted without real GPU evidence.

### Exact UNISA-first workload

On the provisioned UNISA H100 node, with a credentialed repository checkout
and a CUDA Python environment:

```bash
export CIM_REPO_ROOT="$HOME/Cim_shared_prefix"    # actual checkout path
export CIM_PYTHON="/path/to/cuda/python"          # audited GPU environment
export CIM_MODEL=llama
export CIM_PRECISION=bf16
export CIM_PREFIX=2048 CIM_SUFFIX=128
export CIM_BATCH=1 CIM_OUTPUT_TOKENS=128

# Real GPU attention and GPU score-to-full-attention remainder (all tokens).
CIM_TASK=attention sbatch scripts/unisa_gpu_publication.sbatch

# Separate real-weight full decoder (the pinned checkpoint MUST exist).
export CIM_MODEL_SNAPSHOT="/path/to/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
CIM_TASK=decoder sbatch scripts/unisa_gpu_publication.sbatch
```

The script is committed as an executable entrypoint; the above commands have
**not** been submitted by this session. Capture Slurm job ID, PCI identity,
GPU driver, Torch/CUDA version and all event rounds. Then compare only
identical batch/precision/model/length endpoints. Extend to B={2,4,8}
and P={512,4096} after the exact B=1 reference has qualified, retaining
Mistral's visible SWA window and OPT's 2048-position context ceiling.

To compute a measured attention-domain GPU/CIM boundary, the all-step
benchmark JSON must be provided to:

```bash
python scripts/compare_measured_attn.py \
  --gpu-json PATH/attention_cuda_events.json \
  --out PATH/measured_attention_boundary.json \
  --programming-Bps 1e12 --score-output-Bps 1e12 \
  --key-supply-Bps 1e12 --query-Bps 1e12 \
  --reduction-adds-per-s 1e12 --partial-scalings-per-s 1e12 \
  --port-cap 256
python scripts/plot_measured_competition.py \
  --comparison-json PATH/measured_attention_boundary.json \
  --svg PATH/measured_attention_boundary.svg
```

Every interface rate in this **example invocation is an explicit,
unqualified sensitivity**, not a physical H100/CIM measurement. Change the
values only when actual interface caps become available. The function
requires real GPU baseline and remainder times and will never derive a
full-decoder speedup from a single attention stage.

## Numerical inference gate

The existing `scripts/run_extended_quality.py` implements model-specific
R0-R3, numerical-equivalence test, held-out paired WikiText windows,
static per-layer R3-aware O4 ADC calibration and exact 96-unit result
aggregation for each OPT/Mistral model. Run `quality_qualify`,
`quality_calibrate` and `quality_evaluate` **in that order** on a
BF16-capable CUDA GPU using `scripts/unisa_gpu_publication.sbatch`.
The script checks original pinned SRAM2LLM SHA, Transformers 4.46.2,
model revision, identical token stream, and calibrates using development
windows before any held-out quality evaluation. The original Llama 96/96
units and paired confidence intervals are immutable and remain untouched.

`OPT` BF16 numerical-quality measurements are a distinct evaluated copy
of the original FP16-preferred model; they must not be relabeled source-native
FP16 latency. Mistral requires true 4096-token SWA: evicted original prefix
scores are not attributed to CIM. Without the exact upstream source archive
and model snapshot, the R0-R3 extension must remain **0/96** per model.

## Publication claim boundary

A defensible positive offload region must simultaneously satisfy:
(a) measured, competitive complete GPU attention baseline,
(b) GPU score-to-attention remainder at the same endpoint,
(c) actual physical signed macro layout, active ports, programming/score
link/reduction capacities that meet the conditional schedule,
(d) quality at the adopted O4 quantized mapped score implementation,
and (e) exact-model full-decoder operators for TPOT/energy claims.

Currently (a)--(e) are **not simultaneously satisfied**. The only
publication-level performance conclusion is that competition against
optimized Hydragen remains unmeasured and unproven. The prior 1--16 mm²
area feasibility dataset and Llama R0-R3 empirical quality remain useful
as separately qualified components. No new 7nm silicon, digital MAC
iso-area, GPU-CIM speedup or TPOT is reported.
