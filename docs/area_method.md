# Physical 7-nm macro evidence and digital architectural boundary

**Branch:** `work/area-digital`, based on `369f9472`. **Status:** conditional digital break-even; no qualified digital iso-area datapoint. The acceptance gate is `docs/AREA_EVIDENCE_POLICY.md`.

## 1. Sources, manufacturing scope, and complete macro area

| Reference | Silicon/process | Reported scope | Exact implemented value | Qualification |
| --- | --- | --- | --- | --- |
| Y. Yokoyama *et al.*, “A 29.2 Mb/mm² Ultra High Density SRAM Macro using 7nm FinFET Technology with Dual-Edge Driven Wordline/Bitline and Write/Read-Assist Circuit,” VLSI Circuits 2020, [DOI](https://doi.org/10.1109/VLSICircuits18222.2020.9162985) | Fabricated 7-nm SRAM; 512×512 bit macro | **29.2 Mbit/mm² is published macro density**, including macro circuit overhead. Therefore physical **macro** footprint = 262,144/(29.2×10⁶) = **0.008977534 mm²** (derived from *macro* density). | 262,144 storage bits per complete macro. No MAC, reduction, score transport, or architecture-wide ports credited. | Physical SRAM storage-footprint reference; no inferred port throughput. High-density SRAM differs in porting from the Dong/Sinangil 8T dual-port macro. |
| Q. Dong, M. E. Sinangil *et al.*, “A 351TOPS/W and 372.4GOPS Compute-in-Memory SRAM Macro in 7nm FinFET CMOS,” ISSCC 2020, [DOI](https://doi.org/10.1109/ISSCC19947.2020.9062985); extended JSSC 2021 [DOI](https://doi.org/10.1109/JSSC.2020.3031290) | Fabricated 7-nm foundry 8T two-port 64×64 | Complete **4-Kb, 0.0032-mm² CIM macro** includes internal SRAM peripheral circuits, row counters, sense amplifiers, charge-sharing/4-bit Flash ADC, test/reconfiguration. | 1024 4b×4b unsigned MAVs per nominal 5.5-ns access; reported **372.4 GOPS = 2×1024/5.5ns**, i.e., **186.2 billion native MAV/s**, NOT 372.4 billion signed Q4×K4 MAC/s. | Real-silicon *macro* throughput/area, not an end-to-end signed-attention score engine. |
| H. E. Sumbul *et al.*, “System-Level Design and Integration of a Prototype AR/VR Hardware Featuring a Custom Low-Power DNN Accelerator Chip in 7nm Technology for Codec Avatars,” CICC 2022, [DOI](https://doi.org/10.1109/CICC53496.2022.9772810) | 7-nm 2.56-mm² **whole chip** | 1024 MACs, 2 MB SRAM, signed INT8 input/weights, 32-bit RISC-V and chip interfaces | No digital MAC block area extracted; MAC precision/porting and complete QK score endpoint are not matched. | Context only; **not** an iso-area digital comparison. |

This physical SRAM record uses the published *complete-macro density*. It is never derived from a 6T bitcell area or nominal 7-nm logic density. Multi-bank floorplanning, physical power delivery, clock trees, routing, buffering and external controllers remain **unmeasured**.

## 2. Physical signed mapping and storage replication

The existing `SignedMapping()` contract fixes **2× physical K storage** and **4× native unsigned-array evaluations** per signed Q4×K4 logical MAC. The Sinangil macro is native unsigned 4b×4b MAV: the four-phase decomposition, signs, and wide accumulation are implemented at the architectural boundary, not claimed to be native macro features.

Per KV head per layer, CIM mapping uses `ceil(head_dim/64) × ceil(prefix_tokens/16) × 2 × copies` physical macros. Multiplying by `layers × kv_heads` preserves integer tiling, layer/head separation and explicit replicas. This mapping is valid for 64 reduction rows × 16 4-bit output weights per macro. It charges complete 0.0032-mm² macro area **once** per physical macro, never a second ADC/SA/counter charge. SRAM reference uses `ceil(prefix_tokens × head_dim × 4 / 262144) × layers × kv_heads × copies` whole 512×512 macros; this is a capacity bound, and additional banks/ports may be needed to deliver specified SRAM bandwidth. Both paths store the shared K prefix once per KV head, with explicitly selected additional copies; neither replicates K by batch implicitly.

For the illustrative `B=1, prefix=2048, QH=32, KVH=8, D=128, L=32, Q4/K4` workload:

| Physical macro quantity | SRAM-only key storage | Signed CIM key storage |
| --- | ---: | ---: |
| Macro capacity | 262,144 bit | 4,096 bit |
| Macro count, one copy | 1,024 | 131,072 |
| Complete macro area | **9.192995 mm²** | **419.4304 mm²** |
| Signed storage representation | one 4-bit two's-complement K | 2× positive/negative K arrays |
| Native arithmetic | external signed digital Q4×K4 MAC needed | 4 unsigned MAV evaluations per logical signed operation |
| Off-macro elements | controller, Q/operand staging, reduction, score interface, network, clock | same categories, including signed-multiplex / partial-merge logic |

The area ratio is **45.625×** at this shape and one copy, for *storage macros alone*; neither value includes a fabricated complete digital or CIM accelerator. With `prefix=2048`, score output at the required completed-score interface is **8 MiB/token** (2,097,152 32-bit score words) and the native signed mapping requires **1,073,741,824 unsigned MAV evaluations/token**, before considering physical occupancy or fill/eviction.

## 3. Full incremental area coverage

`SupportArea` requires **all six categories** in each architecture: `controller`, `operand_staging`, `signed_reduction`, `score_interface`, `interconnect`, `clock_and_misc`. Additional scratch/buffer area belongs in `operand_staging` and must be externally characterized. A component equal to zero is an explicitly labeled optimistic sensitivity, never assumed to be free. The existing GPU die area is excluded from both incremental-area ledgers.

For a CIM budget `A_C = N_CIM A_macro,CIM + A_ext,CIM`, digital SRAM cost `A_S = N_SRAM A_macro,SRAM`, and complete digital support `A_D`, the **maximum remaining digital MAC-array area** is `A_core = A_C - A_S - A_D`. If this is nonpositive, even a zero-area MAC array cannot fit. No physical digital MAC area has been sourced, so there is no matching implemented digital design and **no digital iso-area result**.

The 7-nm source macros have different native port organization; matching *technology node alone* does not establish equal read bandwidth or physical integration feasibility. Every externally assumed bandwidth is kept in a sensitivity field.

## 4. Equal completed-score endpoint and digital execution

One full token performs `W = L × B × QH × prefix × D` **logical signed** Q4×K4 MACs, and produces `S = L × B × QH × prefix` complete 32-bit scores for GPU scale/softmax. Digital key SRAM-read traffic, parameterized by actual scheduled KV reuse `R`, is `K_bytes × ceil((B × QH/KVH)/R)`, where `1 ≤ R ≤ B × QH/KVH`. The query-read traffic is `ceil(L × B × QH × D × 4/8)` bytes. Digital partial reductions are `S × (ceil(D / tile_K) − 1)` additions; `tile_K` must match the mapped MAC accumulation. Output bytes are `4S` when all signed partials are merged before the GPU link. `digital_architecture.py` rejects incompatible precision/endpoints.

For **serial stage accounting**, digital non-MAC latency is

`T_nonMAC = (B_Kread + B_Qread)/BW_SRAM + N_reductions/R_reduction + B_scores/BW_score_link + T_controller`.

For the *optimistic pipelined* alternative it is `max(T_SRAM,T_reduce,T_link)+T_controller` (the overlap is **not** proven by any trace). Optional prefix placement/refill amortization is a separate system-path input and must be added equally before any end-to-end claim. Score link is kept explicit on *both* paths; digital internal SRAM bandwidth, output-link bandwidth, reduction rate and controller work cannot be hidden in nominal MAC FLOPS. `BW_SRAM` is aggregate physically delivered SRAM bandwidth, which requires banking/port proof beyond the high-density macro-area record.

When `T_C` is a *completed-score* CIM time at this identical endpoint, the required digital MAC compute performance density is:

`rho_MAC >= W / ((T_C - T_nonMAC) × (A_C - A_S - A_D))`  [signed logical MAC/s/mm² of remaining digital core].

A nonpositive parenthesis means *no feasible core-area* or *no available MAC time* under the supplied sensitivities. For a serial schedule, holding all other rates fixed, the SRAM bandwidth required even before digital MAC execution is

`BW_SRAM > (B_Kread+B_Qread) / (T_C - T_controller - T_reduce - T_link)`.

Corresponding link and reduction thresholds are computed in `architecture_boundary`. These conditions are necessary, **not sufficient**: real compute time must also fit, and actual signed digital physical logic area, clock, memory ports, reductions and IO must be documented to claim parity. The optimization cannot infer a multiplier area from SRAM density, whole-chip divided MAC count, Accelergy, SCALE-Sim, technology scaling, or unspecified utilization.

## 5. SCALE-Sim evidence boundary

`scalesim_bridge.gemm_topology()` produces the single-KV-head per-layer GEMM `Q[(B×GQA), D] @ Kᵀ[D, prefix]` in `Layer,M,N,K` form; repeat across `L × KVH`. Run v3 with the **actual** systolic array configuration, SRAM banks/capacities, port/bandwidth configuration and `-i gemm`. Pin the revision and archive `COMPUTE_REPORT.csv`, `BANDWIDTH_REPORT.csv`, `DETAILED_ACCESS_REPORT.csv` and access traces. `read_scalesim_compute_report()` accepts observed compute cycles and mapping efficiency with an explicitly sourced clock period. These cycles already contain the digital MAC work; do not add them to the non-MAC roofline (double counting). SCALE-Sim does not characterize 7-nm cell area or produce a complete signed Q4×K4 standard-cell physical implementation. The current environment has no SCALE-Sim installation or archived corresponding run; no SCALE-Sim performance data are claimed in the generated results.

## 6. Generated bounds and reproducibility

Run `PYTHONPATH=src python -m unittest discover -s tests -v` then `PYTHONPATH=src python scripts/generate_area_boundaries.py <output-dir>`. This deterministically generates `digital_boundary_sensitivity.csv` and `physical_area_provenance.json`. The sweep explicitly enumerates macro copies (1, 2), batch (1, 8), prefix (512, 2048, 8192), CIM support area (0, 100 mm²), digital support area (0, 200 mm²), SRAM bandwidth (1, 8, 32 TB/s), reuse (1 or theoretical maximum), reduction tile (64, 128), and assumed CIM completed-score target (2× or 4× the maximum of ideal native macro compute floor and completed-score link floor). Link rate 1 TB/s, reduction rate 10¹² add/s, controller latency zero and serial stages are also explicit **sensitivities**, not measurements.

Generated 2,304 coordinates: **1,472 conditional MAC-density thresholds**, **736 insufficient-time points**, **96 insufficient-area points**, **0 qualified digital iso-area points**. These classify the declared parametric design space only. They do not establish that a physical digital implementation exists, that the score link can actually be provisioned, or that the source 7-nm CIM macro reaches the derived full-system completion time.

## 7. Remaining evidence required for a qualified iso-area speed claim

Obtain a *complete, actual physically implemented* digital Q4×K4 signed MAC engine under a comparable process, including its signed multiplication/accumulation, full SRAM banking/ports, local staging, controller, output reduction, score interface, interconnect and clock. Collect the equivalent complete CIM system support area and completed-score timing (including 4× native calls and signed merging), with matching bandwidth and refill policy. Only then can the project present an actual matched-area latency or throughput comparison; neither 7-nm synthesis nor an assumed area-per-MAC is part of this branch.
