# Physical macro allocation, causal reuse and bandwidth boundaries

Status: **conditional architectural bounds**. On branch `work/area-digital`, the previous `docs/area_method.md`, 2304-row historical sensitivity CSV and related implementations remain unchanged. This supplement resolves the ambiguity between 32-layer simultaneous storage and sequential decoder execution. `docs/AREA_EVIDENCE_POLICY.md` remains binding.

## Evidence anchors and scope

| Anchor | Technology; physical scope | What the model uses | Limitation |
|---|---|---|---|
| [Dong et al., ISSCC 2020](https://doi.org/10.1109/ISSCC19947.2020.9062985); [Sinangil et al., JSSC 2021](https://doi.org/10.1109/JSSC.2020.3031290) | Fabricated 7-nm 8T 64×64 CIM; **0.0032 mm² complete macro** includes SRAM, ADC, sense amplifiers, counters and on-macro test/reconfiguration | 4096 physical bits per macro; 64 reduction rows × 16 four-bit weights; 1024 *unsigned* 4b×4b native MAV terms/access; 5.5-ns native access; 372.4 GOPS counts multiplication and addition as **two** operations | Macro *averages* with 4-bit Flash ADC output; exact 32-bit signed QK scores and off-macro accumulation, banking, controller, communication and programming bandwidth are **not silicon-qualified** |
| [Yokoyama et al., VLSI 2020](https://doi.org/10.1109/VLSICircuits18222.2020.9162985) | Fabricated 7-nm 512×512 SRAM, density **29.2 Mbit/mm² for the macro** | Complete storage macro area 262144/(29.2×10^6)=0.008977534 mm²; 262144 bits each | High-density SRAM's physical read/write ports are not matched to CIM macro or derived from the paper; digital signed-MAC physical area is unknown |
| [Sumbul et al., CICC 2022](https://doi.org/10.1109/CICC53496.2022.9772810) | 2.56-mm² 7-nm digital test chip, 1024 MACs, 2MB SRAM, CPU | Context only | No MAC area extracted by chip-area division; not an equivalent signed Q4×K4 physical comparator |

**Integration area.** All six external blocks must be separately accounted: controller, operand staging, signed reduction, score interface, interconnect, clock/misc. The generated rows use 0 and 0.5 mm² **total** split evenly among six blocks *solely as sensitivity assumptions*. No digital physical iso-area result is qualified. The preexisting GPU die area is excluded on both sides.

## 1. Spatial layout and physically persistent layers

For prefix length \(P\), \(H_{KV}\) KV heads, and head dimension \(D\), a complete logical layer has

- \(U_C=H_{KV}\lceil D/64\rceil\lceil P/16\rceil\) unsigned native CIM tiles; each signed tile needs **two independently instantiated** CIM macros for positive and negative K storage. With \(c\) query-replicated key copies, \(M_C=2cU_C\) macros/layer.
- \(U_S=H_{KV}\lceil 4PD/262144\rceil\) whole SRAM storage macros/layer and \(M_S=cU_S\). One head's storage cannot be fractionally shared among macro rows/bit cells in the floor model.

Total **full residency** requires \(L M_C\) CIM or \(L M_S\) SRAM macros. It is the former that produces 419.4304 mm² at \(P=2048,D=128,H_{KV}=8,L=32\). A **single active layer** uses 13.1072 mm² of CIM macros; an SRAM layer uses 0.2872811 mm². Source macro area is charged once per physically instantiated macro. No multiplier/ADC/decoder is counted twice outside the complete CIM macro.

For a budget \(A\), explicit external support \(A_{ext}\), and macro-unit cost \(A_U\), define \(C=\lfloor(A-A_{ext})/A_U\rfloor\) available signed CIM tile pairs (or whole SRAM tile units). A schedule pins \(r\) complete resident layers, consuming \(rU\) units. If \(r<L\), it **must** reserve \(s\ge1\) additional tile units as reprogrammable scratch, with \(s=\min(U,C-rU)\). One nonresident layer needs \(\lceil U/s\rceil\) sequential tile-loading/compute waves, while a pinned layer needs one wave. A fully resident design \(r=L\) needs no scratch. The optimizer enumerates all integer-feasible \(r\) and selects the smallest serial CIM bound or smallest required digital MAC density. It may choose a full active-layer scratch at \(r=0\), even when one complete layer fits, because pinning that sole layer would leave no space to access others.

**Attention causal order:** each generated token traverses all \(L\) layers. For \(O\) generated tokens, cold-start physical key programming visits are

\[V_{cold}=L+(O-1)(L-r).\]

Warm-start with persistent pinned keys installed before timing is \(V_{warm}=O(L-r)\). For each visit, program either only valid physical signed K bits (**lower bound**) or all physical macro storage bits (**full-macro upper traffic sensitivity**). Neither convention assumes a bank-write rate. Refills and remapping are charged every visit; pinning does not make streamed layers persistent. Programming bandwidth is an explicit external architectural parameter. Output length is an amortization axis; first-token programming spikes are not represented by average TPOT.

## 2. Native unsigned evaluation versus useful signed QK score

An unsigned macro has 64×16 weights organized as sixteen four-bit columns and sixty-four reduction rows; it computes 1024 native unsigned MAV terms per 5.5-ns access. The literature's **372.4 GOPS** headline = \(2\times1024/(5.5\,ns)\), i.e. ~186.2 billion native unsigned MAV terms per second, with an operation-counting convention of 2 ops/MAV. It cannot be used directly as signed Q4×K4 32-bit completed-score throughput.

In the existing numerical mapping, signed Q4×K4 needs four native unsigned array evaluations and two physically stored signed-K planes. With two truly independent sign-bank accesses, the **optimistic** minimum is two sequential 5.5-ns accesses per query/tile; a serialized single-bank schedule takes four. Per-KV-head batch/GQA query count is \(Q=B(H_Q/H_{KV})\). With \(c\) physically replicated key copies, query rounds \(=\lceil Q/c\rceil\). The macro-only sequential decoder floor is

\[T_{C,compute}^{min}=5.5\,ns\times(4/p)\times\lceil Q/c\rceil\times\left[r+(L-r)\lceil U_C/s\rceil\right],\]

for sign-bank parallelism \(p\in\{1,2\}\); for \(r=L\), the bracket is \(L\). This bound assumes all scratch tiles can be exercised in parallel with perfect control, bandwidth, and Q broadcast; its high implied aggregate throughput is **not validated** by the single-macro paper. Only the active layer contributes in one decoder step; macros holding other layers never enter the active-layer compute denominator. The number of **useful** Q4×K4 signed MACs/token remains \(W=LBH_QPD\). Physical native work is \(4W\), but padded native issued work is \(4LQ U_C\times1024\). Physical row/column occupancy is \(PD/(64\lceil D/64\rceil\,16\lceil P/16\rceil)\).

Each completed 32-bit score combines \(4\lceil D/64\rceil\) signed partial values, requiring at least \(4\lceil D/64\rceil-1\) additions/sign combinations. The 4-bit macro ADC produces quantized averages; there is no proof that postprocessing reconstructs *exact* high-width integer QK scores. Additional numerical characterization is required before any precision-matched speedup can be claimed.

## 3. Identical score endpoint, SRAM/digital requirements, no fabricated physical area

All paths use **32-bit completed scores after local partial reduction**, before GPU softmax: \(B_{scores}=4 LBH_QP\) bytes per generated token. Input query traffic is at least \(B_Q=\lceil 4LBH_QD/8\rceil\) bytes. Both analog CIM and digital models charge the same source-side Q delivery and score output bandwidth assumptions. Conventional SRAM storage alone and SRAM+digital MAC use the same SRAM residency/reprogramming plan; only the digital engine has an unqualified core-area requirement.

For K reuse \(R\) with \(1\le R\le B(H_Q/H_{KV})\), required digital SRAM read bytes/token are \(B_{K,read}=LB_{K,layer}\lceil Q/R\rceil\). The digital MAC array must additionally provide correctly signed 4b×4b operations, adequate integer accumulation, query staging, local K reductions, controller, SRAM ports, and a score interface. Its possible physical area is **unknown**; the budget merely leaves available capacity \(A_{core}=A-A_{SRAM,inst}-A_{ext,digital}\). No claims are made about a realized core or actual 7-nm digital speed.

A strict **serial** conditional CIM lower time is

\[T_C^{bound}=T_{compute}^{min}+B_{program,C}/BW_{program}+N_{reduce,C}/R_{reduce}+B_Q/BW_Q+B_{scores}/BW_{out}+T_{ctrl}.\]

Digital non-MAC time (without fabricated digital compute cycles) is

\[T_{D,other}=B_{program,D}/BW_{program}+(B_{K,read}+B_Q)/BW_{SRAM}+N_{reduce,D}/R_{reduce}+B_Q/BW_Q+B_{scores}/BW_{out}+T_{ctrl}.\]

Given a source-labeled external deadline \(T\) or the modeled complete-score CIM time at the **same** physical budget and bandwidth sensitivities, a necessary digital density is

\[\rho_{MAC}^{req}=\frac{W}{(T-T_{D,other})A_{core}}\quad\text{signed Q4×K4 MAC/s/mm²}.\]

If \(T-T_{D,other}\le0\) or \(A_{core}\le0\), that condition is infeasible. Necessary one-at-a-time bandwidth thresholds follow \(BW_i^{min}=B_i/[T-\sum_{j\ne i}T_j]\), holding all other rates fixed and assuming zero unmodeled digital MAC time. When no positive time remains, output `unattainable`. These are *lower bounds*. Simultaneously satisfying isolated minima does not establish joint feasibility; nonzero digital compute time, routing, arbitration, clock and physical SRAM ports increase requirements. No optimistic pipeline overlap is assumed.

## 4. Reproducible sweep and figures

`PYTHONPATH=src python scripts/run_residency_sweep.py results/area_final` deterministically enumerates **6,144** conditions: 1–16 mm² in 1-mm² steps, prefix P∈{512,2048,8192}, B∈{1,8}, output O∈{32,128}, support 0/0.5 mm² **for each path**, programming 0.1/1 TB/s, score link 0.25/1 TB/s, SRAM read 0.25/1 TB/s, and K reuse 1 or B×GQA. Reduction throughput 10¹² additions/s, Q transfer bandwidth equal to output bandwidth, zero controller latency and a 2-ms target are **sensitivity assumptions**. The optimizer enumerates all feasible pinned-layer counts and independently selects each CIM/digital spatial schedule. Every row includes source macro area, instantiated macro count, refill payload, score traffic, native occupancy, digital density necessary to beat the modeled CIM schedule, bandwidth minima, and false digital-iso-area qualification.

The `irregular_tile_occupancy.csv` separately tests P=513 and D=96 alongside aligned geometry and full-macro-vs-valid-bit reprogramming. Saved figures: scratch waves vs area, modeled CIM latency vs programming bandwidth, conditional digital signed-MAC density vs area and output bandwidth. The previous 2,304-row sensitivity result and all its tests remain untouched. All values involving a full system or non-macro blocks remain **architectural sensitivities**, never physical measurements.

SCALE-Sim is not present in the current execution environment; **no real SCALE-Sim run is claimed**. `scripts/run_real_scalesim_qk.py` runs v3 directly using user-supplied real architecture config and exports a command receipt when installed. It performs one-KV-head GEMM per layer and explicitly repeats observed cycles across L×H_KV; it is mapping/timing evidence only, never a 7-nm MAC physical area estimate or proof of signed arithmetic.

## 5. Qualification boundary

Source-grounded: physical 7-nm macro footprint, unsigned macro nominal operation count and access, integer tiling, signed-storage/evaluation multiplicities from the project contract, causal layer visits and required byte counts. Conditional modeled: simultaneous sign-plane execution, controller, aggregate bank bandwidth, scratch exchange, programming ports, signed partial reconstruction, precision retention, actual CIM system latency. **Unqualified:** physically complete signed CIM engine area/performance, fully implemented digital MAC area, physical iso-area speedup, SCALE-Sim digital performance if no real run is present. No 7-nm synthesis, invented MAC density, whole-chip divided-MAC area or precision-conflating comparator is used.