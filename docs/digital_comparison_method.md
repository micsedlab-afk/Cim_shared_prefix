# Digital-MAC comparison — implemented contract

The module `digital_threshold.py` implements a **necessary** MAC
throughput-per-added-core-area condition for the SRAM+MAC design to equal
the CIM stage at the *completed-score output* endpoint.

Let W be the workload's exact logical Q4xK4 MACs per generated token;
A_b the extra on-chip area budget; A_s SRAM physical macro area; A_o
controller, buffering, scratch, interface and external support area;
T_c the complete CIM QK-score stage; and T_o the digital score stage
excluding MAC arithmetic (including reads, staging, reductions, traffic).

The digital implementation can be competitive only if

  rho_MAC >= W / ((T_c - T_o) * (A_b - A_s - A_o)).

If either denominator factor is nonpositive the candidate is infeasible.

This gives a **break-even requirement**, not a measured or synthesized
digital MAC area, an architecture speedup, or a proof of feasibility.
All A and T inputs require provenance and are marked
measured/published/modeled/sensitivity.

The module never sets `qualified_isoarea=True`. Such qualification
would require direct, complete physical area evidence under a comparable
process, equivalent Q4xK4 signed MAC arithmetic, accumulation,
memory organization and GPU score endpoint.

No 7-nm synthesis is part of the project. See
`docs/AREA_EVIDENCE_POLICY.md`.
