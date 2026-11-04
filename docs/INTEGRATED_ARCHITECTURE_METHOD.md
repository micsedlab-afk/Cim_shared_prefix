# Integrated physical feasibility: reproducibility and interpretation

## Execution

On Python >=3.10 with NumPy, matplotlib and CPU PyTorch installed:

```sh
python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/run_integrated_architecture.py results/integration
PYTHONPATH=src python scripts/run_residency_sweep.py results/area
PYTHONPATH=src python scripts/run_source_sweep.py results/system
```

The GitHub Actions workflow runs the complete suite, the legacy source and
area studies, the Hydragen-style numerical reference tests, and the integrated
1--16 mm² model sweep. Archives not in this public repository and native
Mistral/OPT GPU inference are **not** silently substituted with fixtures.

## Physical contract

The source macro is the published Dong/Sinangil 7-nm 64x64 8T 4-bit CIM
(0.0032 mm², 5.5 ns per native unsigned operation, O4 ADC). A signed Q4/K4
key tile requires **two distinct physical K-plane macros** and **four unsigned
native evaluations** for each Q sign pair. The native macro area is counted
once for each physically provisioned macro, whether reused across layers or
permanently pinned. One layer of shared-prefix K requires

```
tile_units = kv_heads * ceil(head_dim/64) * ceil(visible_prefix/16)
full_layer_macros = 2 * tile_units
full_model_macros = layers * full_layer_macros
max_signed_tile_units = floor((area_budget - external_support)/(.0032 * 2))
```

External integration blocks (controller, staging, signed reduction, score
interface, interconnect, clock/misc) remain explicit sensitivity areas. They
cannot be assumed to have zero physical area by labeling their scenario 0 mm².

A plan pins R *complete* layers. Every other layer uses disjoint scratch tiles
allocated once in silicon and sequentially overwritten. The plan charges
cold K provisioning and all warm-layer refills, with valid-bit versus
full-macro-image transfer explicitly selected. No macro is double counted.

## Constructive schedule and certified floor

`integrated_schedule.serial_schedule` executes ordered logical stages:
source-key supply, macro programming, native unsigned compute, query input,
signed partial scaling, partial reduction, completed 32-bit score output, and
per-layer controller service. For each layer, load the next scratch wave,
execute all B x GQA distinct queries using the two signed K planes, reduce,
and transfer completed scores before the next dependent layer. A single active
macro port executes all four signed products serially; two or more ports
execute K-plane pairs in disjoint tile groups. Query replicas are provisioned
and charged but are **not** used to claim a compute acceleration here.

Rates and port count are unqualified system sensitivities; stage times are
serially summed to obtain a *conditional feasible execution time* for this
schedule. The serial time is a conservative upper schedule, conditional on
actually delivering the specified interfaces, reductions and control.

`feasibility.lower_bound` independently computes an *optimistic necessary*
resource lower bound: max of a causally summed per-layer macro service floor
and compulsory programming, query and score-link demand divided by their
upper rate caps. Overlappable resource times are **never summed** to exclude
an architecture. A lower bound exceeding a qualified comparable GPU
full-decoder time would conditionally exclude the target at the assumed caps.
The study has **no such qualified GPU reference**, and reports instead a
2 ms *hypothetical completed-score* target and interface-rate thresholds.
Both bounds use the same physical area, signed tile mapping and port cap.

Small geometry tests enumerate feasible whole-layer pin counts, scratch
capacities and port limits. An independently constructed per-wave
operation dispatch verifies macro access slots. Every candidate asserts
lower_bound <= serial schedule, with exact signed call and traffic equality.

## Modeling breadth and limitations

All native model geometries use source profiles: Llama-3.1-8B and Mistral-7B
are GQA; OPT-2.7B is MHA with 80 active dimensions padded within the native
64-row tiles, and the 2048-position context ceiling enforced. Mistral's
4096-window erases evicted shared-prefix keys. Where the visible-prefix
length changes across generated tokens, the constructive model cold-loads
every token-specific active prefix; this is a deliberately conservative
valid serial policy with no presumed inter-token reuse. The necessary floor
for these sequences discards token-by-token programming to avoid making an
invalid exclusion claim. When the visible prefix is unchanged, cold-start
programming is amortized across the full output length.

A PyTorch CPU-capable Hydragen-style reference implements shared-prefix
batched QK/PV and independent suffix QK/PV, fused through logsumexp; unit tests
check complete attention against an unpartitioned exact reference, including
GQA and sliding-window visibility. This is a **functional reference only**.
Real GPU kernel performance and the complete remaining decoder operators
(QKV/MLP/projection/norm/RoPE/KV update/LM head/sampling) are not available
for matched GPU baseline timing. All complete decoder TPOT and speedups remain
null until direct qualified evidence is supplied.

Preserved evaluated quality: 96 Llama R0--R3 units (8 samples x 3 prefix
lengths x 4 arms), with absolute NLL/PPL, paired confidence intervals and
native source provenance in `results/quality`. OPT and Mistral have zero
qualified numerical evaluation units; their geometry/timing sensitivity
results carry no numerical-quality or GPU performance claim.

The 28-nm measured S²-CIM design and node-scaled ASAP7/CACTI estimates from
Ouroboros are not reinterpreted as 7-nm digital MAC physical area.
SRAM+digital MAC is expressed only through required throughput density and
minimum SRAM/interface bandwidth, not an iso-area fabricated implementation.

## Publication gate

The framework provides a replicable conditional area/rate/latency envelope
and a constructive scheduling counterpoint to optimistic resource floors.
Publication-level claims of GPU-CIM speedup, iso-area digital superiority,
completed numerical quality on OPT/Mistral, or full-system physical feasibility
are **not qualified** by these results. See
`docs/INTEGRATION_SCIENCE_GATES.md` and `docs/REVIEWER_RESPONSE_MATRIX.md`.
