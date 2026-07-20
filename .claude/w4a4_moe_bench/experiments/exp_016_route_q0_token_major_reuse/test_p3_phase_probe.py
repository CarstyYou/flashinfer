#!/usr/bin/env python3
"""CPU-only gates for the exp_016 narrow P3 phase probe."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import unittest

import build_p3_probe_overlays as probe_builder
from exp016_p3_probe_common import (
    CONTROL,
    GRID_CTAS,
    PROBE,
    ProbeContractError,
    TICKS_PER_CTA,
    barrier_fingerprint,
    capture_summary,
    normalize_dispatch_flag,
    validate_ticks,
)


ROOT = Path(__file__).resolve().parent
MEMORY_ROOT = ROOT.parents[1]
BASE_SOURCE = MEMORY_ROOT / "moe_dynamic_kernel_opt.py"
DISPATCH = ROOT.parents[3] / probe_builder.DISPATCH_RELATIVE_PATH


def load_exp016_builder():
    path = ROOT / "build_overlays.py"
    spec = importlib.util.spec_from_file_location("exp016_base_builder_for_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TickContractTest(unittest.TestCase):
    def test_control_is_exact_sentinel(self):
        result = validate_ticks([-1] * (GRID_CTAS * TICKS_PER_CTA), mode=CONTROL)
        self.assertTrue(result["gate_pass"])
        self.assertTrue(result["all_sentinel"])
        self.assertIsNone(result["grid_critical_wall_ns"])
        self.assertIsNone(result["grid_critical_wall_us"])
        summary = capture_summary([{"p3_timing": result}])
        self.assertIsNone(summary["grid_critical_wall_us"])

    def test_probe_uses_grid_critical_wall(self):
        ticks = []
        for cta in range(GRID_CTAS):
            ticks.extend((1000 + cta * 3, 4000 + cta * 5))
        result = validate_ticks(ticks, mode=PROBE)
        self.assertEqual(result["grid_start_ns"], 1000)
        self.assertEqual(result["grid_end_ns"], 4000 + (GRID_CTAS - 1) * 5)
        self.assertEqual(
            result["grid_critical_wall_ns"],
            result["grid_end_ns"] - result["grid_start_ns"],
        )
        self.assertFalse(result["additive_estimate_reported"])

    def test_missing_or_reversed_edges_fail(self):
        missing = [100] * (GRID_CTAS * TICKS_PER_CTA)
        missing[7] = -1
        with self.assertRaises(ProbeContractError):
            validate_ticks(missing, mode=PROBE)
        reversed_ticks = []
        for _ in range(GRID_CTAS):
            reversed_ticks.extend((200, 100))
        with self.assertRaises(ProbeContractError):
            validate_ticks(reversed_ticks, mode=PROBE)


class SourceTransformTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_builder = load_exp016_builder()
        source = BASE_SOURCE.read_text(encoding="utf-8")
        cls.baseline, cls.candidate, _ = cls.base_builder.resolve_arms(source)
        cls.dispatch = DISPATCH.read_text(encoding="utf-8")

    def test_both_arms_get_identical_boundary_abi(self):
        for source in (self.baseline, self.candidate):
            instrumented = probe_builder.instrument_kernel(source)
            ast.parse(instrumented)
            self.assertEqual(
                barrier_fingerprint(source), barrier_fingerprint(instrumented)
            )
            self.assertEqual(instrumented.count("_exp016_read_globaltimer()"), 2)
            helper_start = instrumented.index(
                "    def initialize_route_q0_and_publish("
            )
            helper_body = instrumented.index(
                "        tidx, bidz, gdim_z, warp_idx, is_cta_leader = thread_info",
                helper_start,
            )
            self.assertIn(
                "        exp016_p3_ticks: cute.Tensor,",
                instrumented[helper_start:helper_body],
            )
            helper_call = instrumented.index(
                "        self.initialize_route_q0_and_publish("
            )
            helper_call_end = instrumented.index("\n\n        gA =", helper_call)
            self.assertIn(
                "            exp016_p3_ticks,",
                instrumented[helper_call:helper_call_end],
            )
            start = instrumented.index("exp016_p3_tick = _exp016_read_globaltimer()")
            phase = instrumented.index(
                "# Phase 2: warp-private route/pack producers", start
            )
            claim = instrumented.index("while produce_active > Int32(0):", phase)
            first_claim = instrumented.index("atomic_add_global_i32(", claim)
            self.assertLess(start, phase)
            self.assertLess(start, first_claim)

            second = instrumented.index(
                "exp016_p3_tick = _exp016_read_globaltimer()", start + 1
            )
            fence = instrumented.index(
                "# Conservative publish fence before the last-producer CTA", second
            )
            deferred = instrumented.index(
                "if full_tile_publish_enabled == Int32(0):", fence
            )
            self.assertLess(second, fence)
            self.assertLess(second, deferred)

    def test_control_probe_kernel_source_is_shared_and_dispatch_only_flag_differs(self):
        kernel = probe_builder.instrument_kernel(self.baseline)
        self.assertEqual(kernel, probe_builder.instrument_kernel(self.baseline))
        control = probe_builder.instrument_dispatch(self.dispatch, enabled=False)
        probe = probe_builder.instrument_dispatch(self.dispatch, enabled=True)
        self.assertNotEqual(control, probe)
        self.assertEqual(
            normalize_dispatch_flag(control), normalize_dispatch_flag(probe)
        )
        self.assertIn("_EXP016_P3_PHASE_PROBE_ENABLED = False", control)
        self.assertIn("_EXP016_P3_PHASE_PROBE_ENABLED = True", probe)


if __name__ == "__main__":
    unittest.main()
