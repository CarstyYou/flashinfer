#!/usr/bin/env python3
"""CPU-only contract checks for the exp_015 runtime harness."""

from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    # The frontend's Python 3.12 is intentionally CPU/tooling-only.  Neither
    # the harness nor the reused exp_005 worker touches a torch attribute at
    # import time, so a module sentinel keeps these contract tests GPU-free.
    sys.modules["torch"] = types.ModuleType("torch")

import run_exp015_arm as harness


class Exp015HarnessTest(unittest.TestCase):
    def test_registered_overlay_hashes_match_locked_sources(self) -> None:
        repo = harness.ROOT.parents[3]
        baseline = (
            harness.ROOT.parent
            / "exp_008_branch_paired_n64_reuse/results/overlays/branch_paired_n64_v1/moe_dynamic_kernel.py"
        )
        candidate = repo / ".claude/w4a4_moe_bench/moe_dynamic_kernel_opt.py"
        self.assertEqual(
            harness.common.file_sha256(baseline),
            harness.EXPECTED_OVERLAY_SHA256[harness.BASELINE],
        )
        self.assertEqual(
            harness.common.file_sha256(candidate),
            harness.EXPECTED_OVERLAY_SHA256[harness.CANDIDATE],
        )
        self.assertNotEqual(
            harness.EXPECTED_OVERLAY_SHA256[harness.BASELINE],
            harness.EXPECTED_OVERLAY_SHA256[harness.CANDIDATE],
        )

    def test_validation_matrix_is_exact_and_candidate_compiles_first_case(self) -> None:
        self.assertEqual(harness.VALIDATION_CASES[0], (256, "canonical"))
        self.assertEqual(harness.VALIDATION_CASES[-1], (8192, "canonical"))
        self.assertEqual(len(harness.VALIDATION_CASES), 8)
        self.assertEqual(
            {fixture for m, fixture in harness.VALIDATION_CASES if m == 256},
            set(harness.M256_FIXTURES),
        )

    def test_abba_position_contract(self) -> None:
        for position, arm in enumerate(harness.ABBA):
            harness.require_abba_position(arm, position)
        with self.assertRaises(RuntimeError):
            harness.require_abba_position(harness.CANDIDATE, 0)

    def test_benchmark_path_is_one_immutable_position(self) -> None:
        args = type(
            "Args",
            (),
            {
                "results": Path("/tmp/exp015-results"),
                "m": 8192,
                "group": 3,
                "position": 2,
                "arm": harness.CANDIDATE,
            },
        )()
        self.assertEqual(
            harness.benchmark_output_path(args),
            Path(
                "/tmp/exp015-results/raw/benchmark/m8192/"
                "group_3_position_2_candidate_v2.json"
            ),
        )

    def test_canary_markers_are_nonzero_and_unique(self) -> None:
        self.assertEqual(len(harness.MARKER_CODES), 8)
        self.assertEqual(len(set(harness.MARKER_CODES)), 8)
        for code in harness.MARKER_CODES:
            packed = harness.packed_byte(code)
            self.assertEqual(packed & 0xF, code)
            self.assertEqual(packed >> 4, code)
        with self.assertRaises(ValueError):
            harness.packed_byte(16)

    def test_protocol_is_locked(self) -> None:
        self.assertEqual(harness.EXPECTED_BLOCK, (288, 1, 1))
        self.assertEqual(harness.NUM_CTA_WARPS, 9)
        self.assertEqual(harness.WARMUP, 5)
        self.assertEqual(harness.ITERS, 50)
        self.assertEqual(harness.L2_FLUSH_BYTES, 192 << 20)

    def test_measure_cli_defaults_to_locked_sample_counts(self) -> None:
        args = harness.parse_args(
            [
                "--flashinfer-root",
                "/workspace/flashinfer",
                "--arm",
                harness.BASELINE,
                "--overlay",
                "/evidence/baseline.py",
                "--jit-root",
                "/tmp/jit",
                "--expected-gpu-uuid",
                "GPU-test",
                "measure",
                "--m",
                "256",
                "--group",
                "0",
                "--position",
                "0",
                "--expected-app-clock-mhz",
                "2377",
                "--expected-jit-artifact-set-sha256",
                "artifact-set",
                "--expected-cubin-sha256",
                "cubin",
            ]
        )
        self.assertEqual(args.command, "measure")
        self.assertEqual(args.warmup, 5)
        self.assertEqual(args.iters, 50)
        self.assertEqual(args.expected_app_clock_mhz, 2377)


if __name__ == "__main__":
    unittest.main()
