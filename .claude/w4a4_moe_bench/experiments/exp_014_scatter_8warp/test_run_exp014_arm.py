#!/usr/bin/env python3
"""CPU-only contract checks for the exp_014 runtime harness."""

from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")

import run_exp014_arm as harness


class Exp014HarnessTest(unittest.TestCase):
    def test_registered_overlays_match_generated_files(self) -> None:
        for arm in harness.ARMS:
            overlay = (
                harness.ROOT / "results" / "overlays" / arm / "moe_dynamic_kernel.py"
            )
            self.assertEqual(
                harness.common.file_sha256(overlay),
                harness.EXPECTED_OVERLAY_SHA256[arm],
            )
        self.assertNotEqual(
            harness.EXPECTED_OVERLAY_SHA256[harness.BASELINE],
            harness.EXPECTED_OVERLAY_SHA256[harness.CANDIDATE],
        )

    def test_validation_matrix_covers_canonical_sweep_and_directed_m256(self) -> None:
        canonical = {
            m for m, fixture in harness.VALIDATION_CASES if fixture == harness.CANONICAL
        }
        self.assertEqual(canonical, set(harness.M_VALUES))
        directed_m256 = {
            fixture
            for m, fixture in harness.VALIDATION_CASES
            if m == 256 and fixture != harness.CANONICAL
        }
        self.assertEqual(directed_m256, set(harness.DIRECTED + harness.CANARIES))
        self.assertEqual(len(harness.VALIDATION_CASES), 12)

    def test_abba_and_measurement_protocol_are_immutable(self) -> None:
        for position, arm in enumerate(harness.ABBA):
            harness.require_abba_position(arm, position)
        with self.assertRaises(RuntimeError):
            harness.require_abba_position(harness.CANDIDATE, 0)
        self.assertEqual(harness.WARMUP, 5)
        self.assertEqual(harness.ITERS, 50)
        self.assertEqual(harness.L2_FLUSH_BYTES, 192 << 20)

    def test_benchmark_path_has_one_arm_position(self) -> None:
        args = type(
            "Args",
            (),
            {
                "results": Path("/tmp/exp014-results"),
                "m": 8192,
                "group": 3,
                "position": 2,
                "arm": harness.CANDIDATE,
            },
        )()
        self.assertEqual(
            harness.benchmark_output_path(args),
            Path(
                "/tmp/exp014-results/raw/benchmark/m8192/"
                "group_3_position_2_candidate_8warp_scatter.json"
            ),
        )


if __name__ == "__main__":
    unittest.main()
