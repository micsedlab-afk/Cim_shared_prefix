"""Independent small-case consistency checks for integrated physical schedules."""
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from math import ceil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cim_shared_prefix import Workload
from cim_shared_prefix.area_optimizer import CIM_7NM, CIM_PUBLISHED_ACCESS_S
from cim_shared_prefix.feasibility import lower_bound, PhysicalBudget
from cim_shared_prefix.residency_area import (
    ExternalArea, residency_plan, enumerate_plans, integrated_serial_stages,
)
from cim_shared_prefix.source_grounded import s2_components_residency
from cim_shared_prefix.integrated_schedule import (
    SerialRates, serial_schedule, best_serial,
)

Z = ExternalArea(0, 0, 0, 0, 0, 0)
R = SerialRates(1e11, 1e11, 1e11, 1e11, 1e11, 1e11)


def oracle_slots(w, plan, ports):
    """Independent wave-level event count; each signed tile needs Q± x K±."""
    slots = 0
    q = w.batch * (w.query_heads // w.kv_heads)
    for layer in range(w.layers):
        remaining = plan.layout.tile_units_per_layer
        allocation = remaining if layer < plan.pinned_layers else plan.scratch_tile_units
        while remaining:
            tile_indices = list(range(min(allocation, remaining)))
            remaining -= len(tile_indices)
            # Serial, executable dispatch: complete all Q signs on each tile
            # group before loading the next set of tiles into the same ports.
            if ports == 1:
                for _ in tile_indices:
                    for _ in range(q):
                        for _ in range(4):
                            slots += 1
            else:
                for start in range(0, len(tile_indices), ports // 2):
                    for _ in range(q):
                        # Paired K+ and K- read for each Q sign.
                        for _ in range(2):
                            slots += 1
    return slots


class IntegratedScheduleTests(unittest.TestCase):
    def test_cross_module_exact_same_allocation_and_time(self):
        w = Workload(1, 64, 3, 2, 1, 64, 3, 4)
        for n in (2, 4, 6, 8, 12, 20):
            budget = n * CIM_7NM.area_mm2 + 1e-8
            for pins in range(w.layers + 1):
                plan = residency_plan(w, kind="cim", budget_mm2=budget,
                                      support=Z, pinned_layers=pins)
                if plan is None:
                    continue
                for ports in (1, 2, 3, n):
                    trace = serial_schedule(w, plan, R, active_macro_ports=ports)
                    self.assertEqual(oracle_slots(w, plan, min(ports, plan.instantiated_macros)),
                                     trace.native_access_slots_per_token)
                    from_area = integrated_serial_stages(w, plan, R, active_macro_ports=ports)
                    from_system = s2_components_residency(w, plan, R,
                                                          active_macro_ports=ports)
                    self.assertEqual(asdict(trace), asdict(from_area))
                    self.assertEqual(asdict(trace), asdict(from_system))
                    self.assertEqual(plan.physical_macro_mm2,
                                     plan.instantiated_macros * CIM_7NM.area_mm2)
                    self.assertLessEqual(plan.complete_mm2, budget + 1e-10)
                    self.assertEqual(trace.native_unsigned_calls_per_token,
                                     4 * w.layers * w.batch * w.query_heads *
                                     ceil(w.head_dim / 64) * ceil(w.prefix_tokens / 16))
                    self.assertGreaterEqual(trace.serial_completed_score_s + 1e-14,
                                            trace.optimistic_resource_floor_s)

    def test_per_layer_causal_bound_bruteforce_small_port_cases(self):
        for layers in (1, 2, 3):
            for prefix in (16, 32, 48):
                w = Workload(1, prefix, 0, 1, 1, 64, layers, 3)
                for macros in range(2, 15, 2):
                    budget = macros * CIM_7NM.area_mm2 + 1e-8
                    for ports in (1, 2, 4, macros):
                        opts = enumerate_plans(w, kind="cim", budget_mm2=budget, support=Z)
                        conditional = [serial_schedule(w, p, R, active_macro_ports=ports)
                                       for p in opts]
                        floor = lower_bound(w, PhysicalBudget(budget, 0), R.ceilings(),
                                            active_macro_port_cap=min(ports, macros))
                        if conditional:
                            self.assertIsNotNone(floor)
                            self.assertLessEqual(floor.elapsed_floor_s,
                                                 min(t.serial_completed_score_s for t in conditional) + 1e-14)
                            for p, trace in zip(opts, conditional):
                                self.assertEqual(trace.native_access_slots_per_token,
                                                 oracle_slots(w, p, min(ports, p.instantiated_macros)))
                                self.assertEqual(trace.stages_s["signed_unsigned_macro_evaluations"],
                                                 oracle_slots(w, p, min(ports, p.instantiated_macros)) *
                                                 CIM_PUBLISHED_ACCESS_S)

    def test_cold_start_amortization_and_persistent_pins(self):
        w1 = Workload(1, 32, 0, 1, 1, 64, 3, 1)
        w4 = Workload(1, 32, 0, 1, 1, 64, 3, 4)
        for w in (w1, w4):
            plan = residency_plan(w, kind="cim", budget_mm2=0.04, support=Z,
                                  pinned_layers=1)
            self.assertIsNotNone(plan)
            trace = serial_schedule(w, plan, R)
            self.assertEqual(trace.programming_bytes_total, plan.programming_bytes_total)
            self.assertEqual(trace.completed_score_bytes_per_token, 4 * w.layers *
                             w.query_heads * w.batch * w.prefix_tokens)
        first = residency_plan(w1, kind="cim", budget_mm2=0.04, support=Z,
                               pinned_layers=1)
        multi = residency_plan(w4, kind="cim", budget_mm2=0.04, support=Z,
                               pinned_layers=1)
        self.assertEqual(first.programming_bytes_total, first.valid_key_layer_bytes * w1.layers)
        self.assertEqual(multi.programming_bytes_total,
                         multi.valid_key_layer_bytes * (w4.layers + (w4.output_tokens-1) *
                                                        (w4.layers-1)))

    def test_explicit_port_cap_and_budget(self):
        w = Workload(1, 2048, 128, 32, 8, 128, 32, 128)
        for b in (1, 2, 4, 8, 16):
            best = best_serial(w, budget_mm2=b, support=Z, rates=R,
                               active_macro_ports=256)
            self.assertIsNotNone(best)
            plan, trace = best
            self.assertLessEqual(plan.complete_mm2, b + 1e-10)
            self.assertLessEqual(trace.active_macro_ports, 256)
            self.assertGreaterEqual(trace.serial_completed_score_s,
                                    trace.optimistic_resource_floor_s)

    def test_reject_warm_start_for_cold_floor(self):
        w = Workload(1, 16, 0, 1, 1, 64, 2, 3)
        p = residency_plan(w, kind="cim", budget_mm2=.02, support=Z,
                           pinned_layers=0, cold_start=False)
        with self.assertRaises(ValueError):
            serial_schedule(w, p, R)


if __name__ == "__main__":
    unittest.main()
