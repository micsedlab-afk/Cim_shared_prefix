# Executed 7-nm macro-level causal results — 4 November 2026

Source: `scripts/run_causal_7nm_sim.py` with the original Sinangil published
7-nm 64x64 0.0032-mm², 5.5-ns unsigned 4-bit macro. Executed in GitHub
Actions run [41844270695](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/41844270695),
which passed 138 tests and skipped one optional source-native test.
These are **source-macro-calibrated scheduling simulation results**, not a
physically integrated 7-nm accelerator or measured GPU timings.

## Reference execution

Model: Llama-3.1-8B native GQA 32Q/8KV/128D, 32 decoder layers.
Prefix P=2,048, suffix=128, batch=1, O=128 output steps, signed Q4xK4
with two physical K-sign planes and four unsigned evaluations.
Budget: 16 mm² including **0.5 mm² assumed external support**.
Cap: 256 concurrently usable macro ports (sensitivity).
Supplied key, programming, query, completed score links: each 1 TB/s;
signed reductions and partial scaling: each 1e12 operations/s;
controller overhead: zero (optimistic sensitivity).
All seven resource rates are **unqualified integration assumptions**, not
fabricated physical-interface measurements.

| Quantity | Executed output |
|---|---:|
| Instantiated 7-nm CIM macros | 4,842 |
| Macro silicon footprint | 15.4944 mm² |
| Pinned complete layers | 1 |
| Programmable signed-tile scratch units | 373 |
| First generated-token prefix-score service | 198.021152 µs |
| Subsequent generated-token prefix-score service | 193.826848 µs |
| 128-token average prefix-score service | 193.859616 µs |
| Necessary optimistic macro-resource floor | 64.648620 µs/token |
| Conditional serial-to-floor ratio | 2.9986 (approximately) |

### Mean modeled prefix-score service components

| Service | µs/token | Fraction of total |
|---|---:|---:|
| Physical key source supply | 65.028096 | 33.54% |
| Physical K programming | 65.028096 | 33.54% |
| Unsigned native macro evaluation | 23.892000 | 12.32% |
| Signed partial scaling | 16.777216 | 8.65% |
| Signed partial reductions | 14.680064 | 7.57% |
| Completed 32-bit score output | 8.388608 | 4.33% |
| Query input | 0.065536 | 0.034% |
| Controller | 0 | optimistic assumption |

Key-source supply plus key programming together account for approximately
67.1% of this modeled serial schedule. This result is conditional on the
assumed throughput of those independent physical services and the chosen
pinned-layer/scratch policy. It must not be interpreted as an experimentally
measured DRAM or macro programming bottleneck.

## Direct GPU QK-saving threshold, no arbitrary deadline

`scripts/derive_gpu_saving_thresholds.py` transformed all **2,304 already
executed** integration coordinates into a matching attention-output
break-even requirement. In the reference configuration above, the
full-GPU attention time per layer minus GPU time to produce the same
attention-context output *after completed CIM prefix scores have arrived*
must exceed:

- **6.058113 µs/layer** to beat the modeled complete serial CIM QK path.
- **2.020269375 µs/layer** as the optimistic CIM-only physical
  resource-service lower floor (not a realized implementation time).

These criteria compare the same completed attention-context output, under
the explicitly serial, non-overlapped S2 policy. Any GPU suffix processing
overlap changes the candidate schedule and must be represented explicitly.
The 2,304-case derived CSV and reference JSON are CI artifacts from
[run 41844270695](https://github.com/micsedlab-afk/Cim_shared_prefix/actions/runs/41844270695).
The original hypothetical 2-ms target is not used here.

## Full decoder model contract

`src/cim_shared_prefix/causal_timeline.py` composes S0 GPU-only,
S1 conventional SRAM + exact signed GPU Q4/K4 prefix-QK,
and S2 7-nm SRAM-CIM QK + GPU remainder, preserving common decoder
operators and terminal LM-head/sampling exactly once.
It accepts explicit per-token source-labeled GPU model/measurement inputs
and **does not synthesize absent stage timings**. Thus S0/S1/S2 numerical
TPOT entries in the executed macro-only reference remain `null`.
The original 96 Llama numerical evaluation units remain separate
and their R2/R3 Q4K4O4 quality results are approximate relative to
the native BF16 GPU arithmetic.

This simulation adds an independently testable temporal/cold-warm
execution trace and a physically interpretable necessary performance
condition, without introducing 7-nm synthesis, H100-to-A100 substitutions,
new GPU hardware requirements or speculative digital-MAC area.
