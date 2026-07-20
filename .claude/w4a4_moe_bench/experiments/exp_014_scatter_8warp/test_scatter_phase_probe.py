#!/usr/bin/env python3
"""CPU-only structure and ABI gates for the exp_014 Scatter phase probe."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

import build_scatter_phase_probe as builder
from exp014_scatter_probe_common import (
    ARMS,
    BASELINE,
    BASE_OVERLAY_ROOT,
    CANDIDATE,
    COMPUTE_WARPS,
    EDGE_NAMES,
    OUTPUT_TILES,
    SAMPLED_TASK_SLOTS,
    SENTINEL,
    TASK_TICKS,
    TICKS_PER_TILE,
    ProbeContractError,
    barrier_fingerprint,
    event_index,
    interval_rows,
    summarize_intervals,
    validate_probe_ticks,
)
import run_exp014_scatter_probe as runner


class ScatterPhaseProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[4]
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "probe_overlays"
        cls.identity = builder.build(cls.repo, cls.output)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_event_abi_is_16_tiles_by_3_edges_by_8_warps(self) -> None:
        self.assertEqual(COMPUTE_WARPS, 8)
        self.assertEqual(OUTPUT_TILES, 16)
        self.assertEqual(SAMPLED_TASK_SLOTS, (0,))
        self.assertEqual(EDGE_NAMES, ("D", "E", "F"))
        self.assertEqual(TICKS_PER_TILE, 24)
        self.assertEqual(TASK_TICKS, 384)
        self.assertEqual(event_index(15, "F", 7), TASK_TICKS - 1)

    def test_probe_boundaries_are_ordered_and_lane0_per_warp(self) -> None:
        for arm in ARMS:
            source = (self.output / arm / "moe_dynamic_kernel.py").read_text(
                encoding="utf-8"
            )
            base = (BASE_OVERLAY_ROOT / arm / "moe_dynamic_kernel.py").read_text(
                encoding="utf-8"
            )
            self.assertEqual(barrier_fingerprint(source), barrier_fingerprint(base))
            self.assertEqual(source.count("mov.u64 $0, %globaltimer;"), 1)
            self.assertEqual(source.count("_exp014_read_globaltimer()"), 3)
            self.assertEqual(
                source.count(
                    "if task_slot_probe == Int32(_EXP014_SCATTER_SAMPLE_TASK):"
                ),
                3,
            )
            self.assertEqual(source.count("def _exp014_ld_shared_volatile_i32("), 1)
            self.assertEqual(
                source.count("task_slot_probe = _exp014_ld_shared_volatile_i32("),
                1,
            )
            self.assertEqual(
                source.count("if lane_id == Int32(0):")
                - base.count("if lane_id == Int32(0):"),
                3,
            )
            for offset in (0, 8, 16):
                self.assertEqual(source.count(f"+ Int32({offset}) + warp_idx,"), 1)
            self.assertIn("self.num_mma_warps = 8", source)

            d = source.index("+ Int32(0) + warp_idx,")
            scatter = source.index(
                "                        self.scatter_sC_to_gmem(", d
            )
            e = source.index("+ Int32(8) + warp_idx,", scatter)
            post_sync = source.index(
                "                        self.epilog_sync_barrier.arrive_and_wait()",
                e,
            )
            f = source.index("+ Int32(16) + warp_idx,", post_sync)
            self.assertLess(d, scatter)
            self.assertLess(scatter, e)
            self.assertLess(e, post_sync)
            self.assertLess(post_sync, f)

    def test_cross_arm_probe_is_matched_except_scatter_mapping(self) -> None:
        self.assertTrue(self.identity["cross_arm"]["gate_pass"])
        baseline = (self.output / BASELINE / "moe_dynamic_kernel.py").read_text(
            encoding="utf-8"
        )
        candidate = (self.output / CANDIDATE / "moe_dynamic_kernel.py").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            baseline,
            builder.normalize_scatter_mapping(candidate),
        )
        self.assertEqual(
            (self.output / BASELINE / "moe_dispatch.py").read_bytes(),
            (self.output / CANDIDATE / "moe_dispatch.py").read_bytes(),
        )

    def test_storage_gate_and_interval_formulas(self) -> None:
        task_capacity = 2
        ticks = [[SENTINEL] * TASK_TICKS for _ in range(task_capacity)]
        for tile in range(OUTPUT_TILES):
            for warp in range(COMPUTE_WARPS):
                base = 1000 + tile * 100
                ticks[0][event_index(tile, "D", warp)] = base + warp
                ticks[0][event_index(tile, "E", warp)] = base + 20 + warp
                ticks[0][event_index(tile, "F", warp)] = base + 30 + warp
        gate = validate_probe_ticks(
            ticks,
            task_tail=1,
            task_capacity=task_capacity,
            task_slice_count=[1],
            task_valid_rows=[128],
        )
        self.assertTrue(gate["gate_pass"])
        rows = interval_rows(ticks, task_tail=1, task_capacity=task_capacity)
        self.assertEqual(len(rows), OUTPUT_TILES)
        self.assertTrue(all(row["body_ns"] == 27 for row in rows))
        self.assertTrue(all(row["including_sync_ns"] == 37 for row in rows))
        summary = summarize_intervals(rows)
        self.assertEqual(summary["body_ns"]["median"], 27)
        self.assertEqual(summary["including_sync_ns"]["median"], 37)

    def test_storage_gate_fails_closed(self) -> None:
        incomplete = [[SENTINEL] * TASK_TICKS]
        with self.assertRaises(ProbeContractError):
            validate_probe_ticks(
                incomplete,
                task_tail=1,
                task_capacity=1,
                task_slice_count=[1],
                task_valid_rows=[128],
            )
        full = [[0] * TASK_TICKS]
        with self.assertRaises(ProbeContractError):
            validate_probe_ticks(
                full,
                task_tail=1,
                task_capacity=1,
                task_slice_count=[2],
                task_valid_rows=[128],
            )
        with self.assertRaises(ProbeContractError):
            validate_probe_ticks(
                full,
                task_tail=1,
                task_capacity=1,
                task_slice_count=[1],
                task_valid_rows=[127],
            )

    def test_runner_overlay_identity_gate(self) -> None:
        for arm in ARMS:
            args = argparse.Namespace(
                flashinfer_root=self.repo,
                overlay_root=self.output,
                arm=arm,
            )
            gate = runner.overlay_identity_gate(args)
            self.assertTrue(gate["gate_pass"], gate["errors"])


if __name__ == "__main__":
    unittest.main()
