import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from diagnose_failure import (
    _validate_args,
    gate_classification,
    summarize_cross_replay,
)


def _replay(
    output_hash: str,
    *,
    formal_pass: bool = True,
    nan_remaining: int = 0,
    inf_count: int = 0,
    workspace_pass: bool = True,
    workspace_hash: str = "workspace",
    zero_rows: int = 0,
):
    value = {
        "formal_metrics": {"formal_pass": formal_pass, "finite": True},
        "sentinel": {"nan_remaining": nan_remaining, "inf_count": inf_count},
        "workspace": {
            "gate_pass": workspace_pass,
            "task_descriptor_multiset_sha256": workspace_hash,
        },
        "output_sha256": output_hash,
        "zero_rows": zero_rows,
    }
    value["gates"] = gate_classification(value)
    return value


class FailureDiagnosticsTest(unittest.TestCase):
    def test_gate_classification_keeps_failure_causes_independent(self):
        value = _replay("a", formal_pass=False, workspace_pass=True)
        self.assertEqual(
            value["gates"],
            {
                "formal_pass": False,
                "sentinel_pass": True,
                "workspace_pass": True,
                "overall_pass": False,
            },
        )

    def test_cross_replay_summary_preserves_raw_stability_evidence(self):
        replays = [_replay("a"), _replay("b", zero_rows=2), _replay("b")]
        pairwise = [
            {
                "first_replay": 0,
                "second_replay": 1,
                "metrics": {
                    "cosine_loss": 0.1,
                    "relative_l2": 0.2,
                    "max_abs": 0.3,
                    "token_rel_l2_p99": 0.4,
                },
            },
            {
                "first_replay": 0,
                "second_replay": 2,
                "metrics": {
                    "cosine_loss": 0.01,
                    "relative_l2": 0.02,
                    "max_abs": 0.03,
                    "token_rel_l2_p99": 0.04,
                },
            },
        ]
        summary = summarize_cross_replay(replays, pairwise)
        self.assertEqual(summary["replay_count"], 3)
        self.assertFalse(summary["exact_output_hash_stable"])
        self.assertEqual(summary["zero_row_counts"], [0, 2, 0])
        self.assertTrue(summary["workspace_descriptor_multiset_stable"])
        self.assertEqual(summary["worst_pairwise_output_drift"]["relative_l2"], 0.2)

    def test_requires_at_least_one_replay(self):
        with self.assertRaisesRegex(ValueError, "at least one replay"):
            summarize_cross_replay([], [])

    def test_accepts_nonhistorical_positive_m_and_protects_output(self):
        worker = SimpleNamespace(KNOWN_ARMS=("arm",), ALL_FIXTURES=("canonical",))
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "diagnostics.json"
            args = SimpleNamespace(
                m=512,
                arm="arm",
                fixture="canonical",
                output=output,
            )
            _validate_args(args, worker)
            output.write_text("existing evidence\n")
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                _validate_args(args, worker)


if __name__ == "__main__":
    unittest.main()
