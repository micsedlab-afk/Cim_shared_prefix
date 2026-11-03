# L1 system latency implementation

Implemented in `cim_shared_prefix.system`.

Per generated token:

`T = T_QK,physical + T_merge + T_score_link + T_GPU,other + T_refill + T_program,refill + (T_initial_transfer + T_initial_program)/O`.

- `T_QK,physical` includes the signed decomposition multiplier and every decoder layer.
- The initial transfer scales with *shared* K once, not batch B.
- Recurring refill is separately parameterized by a resident/miss schedule.
- A full score endpoint uses one 32-bit score per head/prefix token; passing two partials costs twice the link bytes and explicit merging.
- `GPU,other` must be sourced from a decoder-stage provider, including suffix score work and all other decoder operators, using aligned geometry and precision.
- All timing sources must be supplied; tests use clearly named synthetic fixtures.
- The default calculation is fully serial and thus a conservative upper timing bound. No overlap claim is made.

Follow-up needed for this branch: implement actual workload/residency scheduler and MAGMA/LLMCompass stage adapter; retain optimistic overlap only as a clearly labeled separately computed sensitivity.
