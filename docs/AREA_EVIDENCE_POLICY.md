# AREA EVIDENCE POLICY — binding on every branch

## Nonnegotiable scientific scope

An SRAM-buffered digital MAC baseline cannot be declared **iso-area**
using a guessed area per multiplier/MAC, borrowed density, unspecified
standard-cell utilization, node-scaled area, or the area of a full published
DNN processor divided by its MAC count.

Do not run 7-nm RTL synthesis; no licensed 7-nm PDK is available.
SCALE-Sim provides timing/mapping, **not physically qualified digital area**.
Accelergy requires characterized component models and does not establish area.

## Admissible reported outcomes

**A. Matched-area comparison (qualified).** Only when a complete digital
design's actual physical area is documented, the same technology/process
basis as the SRAM/CIM reference is established, and the entire comparator
periphery, memory, MAC array, control, I/O and required buffering are included
without double-counting. Report process differences and block scope.

**B. Conditional threshold (always possible without inventing area).**
Derive the digital MAC throughput/area needed to match or beat the
source-grounded CIM configuration. Express threshold as a function of
documented SRAM, controller and transfer area assumptions; expose
unknowns as symbols or sensitivity axes. No claim that any digital
design attains the threshold.

**C. Published discrete chip comparator (context only).** For published
real-silicon digital accelerators, report their whole-chip area,
full precision and local-memory capability. Do not strip chip overhead,
rescale or call them implementation-equivalent absent evidence.

## Physical source anchors

- Dong, Sinangil et al., ISSCC 2020, DOI 10.1109/ISSCC19947.2020.9062985;
  7-nm foundry 8T, 64x64 macro, 0.0032 mm2 reported macro area, 4b x 4b,
  5.5 ns nominal access and 372.4 GOPS macro headline; inspect source
  counting conventions and distinguish measured silicon from paper estimates.
- Sumbul et al., CICC 2022, fabricated 7-nm digital NN test-chip,
  whole chip 2.56 mm2, 1,024 MACs, 2MB embedded SRAM and 32-bit RISC-V CPU.
  This is **not** a derivation of single-MAC area or matching INT4 design.

Every numeric source must cite title, DOI/URL, technology, scope,
precision and whether the reported quantity is measured or analytical.

## Review acceptance

1. A `source` string in a generic AreaLedger is bookkeeping, not qualification.
2. Reject missing digital physical area, unspecified area components or
   a mismatch of complete-score endpoint, precision or process when asked
   to output 'iso-area digital vs CIM'.
3. Where area is missing, output **only** digital break-even requirement.
4. Propagate ranges for uncertain integration-area and bandwidth quantities.
5. Keep the GPU's existing die area out of **incremental** accelerator-area
   accounting for both paths.
