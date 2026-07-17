#!/usr/bin/env python3
"""CPU-only unit tests for exp_005 experiment contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from exp005_common import (
    ABBA_ORDER,
    BASELINE,
    BENCHMARK_GROUPS,
    CANDIDATE,
    E,
    evaluate_cross_arm_correctness,
    expected_expert_tile_base,
    expected_terminal_pair_head,
    expected_task_records,
    summarize_paired_abba,
    verify_workspace_evidence,
)


class TaskOracleTest(unittest.TestCase):
    def test_expected_tasks_include_exact_and_tail_tiles(self) -> None:
        row_counts = [0, 128, 129] + [0] * (E - 3)
        self.assertEqual(expected_expert_tile_base(row_counts)[:4], [0, 0, 1, 3])
        records = expected_task_records(row_counts)
        self.assertEqual(len(records), 12)
        self.assertEqual(
            sorted({record[4] for record in records}),
            [1, 128],
        )
        self.assertIn((2, 2, 3, 1, 1), records)

    def test_workspace_contract_passes_and_detects_corruption(self) -> None:
        row_counts = [0, 128, 129] + [0] * (E - 3)
        records = expected_task_records(row_counts)
        snapshot = {
            "row_counts": row_counts,
            "expert_write_rows": row_counts,
            "expert_tile_base": expected_expert_tile_base(row_counts),
            "task_tail": len(records),
            "task_head": len(records) + 110,
            "task_expert": [item[0] for item in records],
            "task_m_tile": [item[1] for item in records],
            "task_slice_begin": [item[2] for item in records],
            "task_slice_count": [item[3] for item in records],
            "task_valid_rows": [item[4] for item in records],
            "routed_rows": sum(row_counts),
            "pair_head": expected_terminal_pair_head(sum(row_counts), num_cta_warps=5),
            "all_work_published": 1,
        }
        verdict = verify_workspace_evidence(snapshot, expected_row_counts=row_counts)
        self.assertTrue(verdict["gate_pass"])
        snapshot["task_valid_rows"][0] = 127
        verdict = verify_workspace_evidence(snapshot, expected_row_counts=row_counts)
        self.assertFalse(verdict["gate_pass"])
        self.assertFalse(verdict["checks"]["task_descriptor_multiset"])


class CorrectnessContractTest(unittest.TestCase):
    def test_self_drift_relative_gate(self) -> None:
        baseline = {
            "cosine_loss": 0.0,
            "relative_l2": 0.006,
            "max_abs": 0.015,
            "token_rel_l2_p99": 0.007,
        }
        candidate_self = dict(baseline)
        candidate_worst = {
            "cosine_loss": 0.0,
            "relative_l2": 0.008,
            "max_abs": 0.025,
            "token_rel_l2_p99": 0.009,
        }
        verdict = evaluate_cross_arm_correctness(
            baseline, candidate_self, candidate_worst
        )
        self.assertTrue(verdict["gate_pass"])
        candidate_worst["relative_l2"] = 0.025
        verdict = evaluate_cross_arm_correctness(
            baseline, candidate_self, candidate_worst
        )
        self.assertFalse(verdict["gate_pass"])


class BenchmarkContractTest(unittest.TestCase):
    def rows(self, baseline: float, candidate: float):
        result = []
        for group in range(BENCHMARK_GROUPS):
            for position, arm in enumerate(ABBA_ORDER):
                result.append(
                    {
                        "group": group,
                        "position": position,
                        "arm": arm,
                        "sample_us": baseline if arm == BASELINE else candidate,
                    }
                )
        return result

    def test_faster_locked_classification(self) -> None:
        summary = summarize_paired_abba(
            self.rows(100.0, 80.0), clock_policy="locked", bootstrap_samples=200
        )
        self.assertEqual(summary["statistical_classification"], "faster")
        self.assertEqual(summary["verdict"], "faster")
        self.assertAlmostEqual(summary["median_ratio_baseline_over_candidate"], 1.25)

    def test_unlocked_clock_forces_advisory_verdict(self) -> None:
        summary = summarize_paired_abba(
            self.rows(100.0, 80.0), clock_policy="unlocked", bootstrap_samples=200
        )
        self.assertEqual(summary["statistical_classification"], "faster")
        self.assertEqual(summary["verdict"], "advisory_inconclusive")

    def test_order_drift_fails_closed(self) -> None:
        rows = self.rows(100.0, 80.0)
        rows[0]["arm"] = CANDIDATE
        with self.assertRaisesRegex(ValueError, "ABBA order drift"):
            summarize_paired_abba(rows, clock_policy="locked", bootstrap_samples=20)


class PathSmokeTest(unittest.TestCase):
    def test_temporary_path_is_available_without_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(Path(directory).is_dir())


if __name__ == "__main__":
    unittest.main()
