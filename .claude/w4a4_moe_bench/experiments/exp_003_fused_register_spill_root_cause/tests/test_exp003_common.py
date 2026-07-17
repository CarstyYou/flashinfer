from __future__ import annotations

import pytest

from exp003_common import (
    EXPECTED_BLOCK,
    EXPECTED_FP4_TENSOR_OPS,
    EXPECTED_GRID,
    EXPECTED_TAIL_SECTOR_DELTA,
    EXPECTED_TASK_COUNT,
    EXPECTED_TENSOR_INSTRUCTIONS,
    build_empty_manifest,
    correctness_thresholds,
    evaluate_correctness_gate,
    qualify_spill_candidate,
    summarize_paired_benchmark,
)


def test_runner_initializes_exp003_root_cause_contract() -> None:
    manifest = build_empty_manifest()
    assert manifest["schema"] == "exp003.spill-root-cause.run-manifest.v1"
    assert manifest["status"] == "not_started"
    assert "root_cause_evidence" not in manifest


def work_payload(**extra):
    return {
        "tensor_instructions": EXPECTED_TENSOR_INSTRUCTIONS,
        "fp4_tensor_ops": EXPECTED_FP4_TENSOR_OPS,
        "grid": list(EXPECTED_GRID),
        "block": list(EXPECTED_BLOCK),
        "task_count": EXPECTED_TASK_COUNT,
        **extra,
    }


def test_correctness_gate_uses_floor_and_cap() -> None:
    self_drift = {
        "cosine_loss": 0.0,
        "relative_l2": 1e-5,
        "max_abs": 0.001,
        "token_rel_l2_p99": 0.0001,
    }
    assert correctness_thresholds(self_drift) == {
        "cosine_loss": 1e-6,
        "relative_l2": 1e-4,
        "max_abs": 0.003,
        "token_rel_l2_p99": 0.001,
    }
    result = evaluate_correctness_gate(self_drift, dict(self_drift))
    assert result["gate_pass"] is True


def test_self_drift_over_hard_cap_invalidates_comparison() -> None:
    self_drift = {
        "cosine_loss": 2e-5,
        "relative_l2": 0.0,
        "max_abs": 0.0,
        "token_rel_l2_p99": 0.0,
    }
    result = evaluate_correctness_gate(self_drift, {key: 0.0 for key in self_drift})
    assert result["baseline_self_drift_within_hard_caps"] is False
    assert result["gate_pass"] is False


def test_qualify_exact_14_word_static_dynamic_closure() -> None:
    baseline = work_payload(
        stored_words=122,
        local_load_sectors=4_950_272,
        local_store_sectors=4_950_272,
    )
    candidate = work_payload(
        stored_words=108,
        replacement_words=0,
        local_load_sectors=4_950_272 - EXPECTED_TAIL_SECTOR_DELTA,
        local_store_sectors=4_950_272 - EXPECTED_TAIL_SECTOR_DELTA,
    )
    result = qualify_spill_candidate(baseline=baseline, candidate=candidate)
    assert result["formal_timing_eligible"] is True


def test_replacement_bundle_blocks_timing() -> None:
    baseline = work_payload(
        stored_words=122,
        local_load_sectors=4_950_272,
        local_store_sectors=4_950_272,
    )
    candidate = work_payload(
        stored_words=108,
        replacement_words=14,
        local_load_sectors=4_382_208,
        local_store_sectors=4_382_208,
    )
    assert not qualify_spill_candidate(baseline=baseline, candidate=candidate)[
        "formal_timing_eligible"
    ]


def benchmark_rows(candidate_factor: float = 0.99):
    rows = []
    for repeat in range(5):
        order = "baseline>candidate" if repeat % 2 == 0 else "candidate>baseline"
        baseline = 100.0 + repeat * 0.01
        rows.extend(
            [
                {
                    "repeat": repeat,
                    "order": order,
                    "arm": "baseline",
                    "sample_us": baseline,
                },
                {
                    "repeat": repeat,
                    "order": order,
                    "arm": "candidate",
                    "sample_us": baseline * candidate_factor,
                },
            ]
        )
    return rows


def test_paired_benchmark_materiality() -> None:
    summary = summarize_paired_benchmark(benchmark_rows())
    assert summary["candidate_faster_pair_count"] == 5
    assert summary["material_improvement"] is True


def test_paired_benchmark_rejects_order_drift() -> None:
    rows = benchmark_rows()
    rows[0]["order"] = "candidate>baseline"
    with pytest.raises(ValueError, match="order"):
        summarize_paired_benchmark(rows)
