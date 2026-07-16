from __future__ import annotations

import build_result

IDENTITY = {
    "comparison_group_id": "exp002-pair",
    "rerun_id": "rerun-20260715-a",
    "environment_lock_digest": "env-a",
    "protocol_lock_digest": "protocol-a",
    "per_arm_artifact_fingerprint_sha256": {
        "cutedsl_bf16_fused": "artifact-cutedsl",
        "cutlass_bf16_chain": "artifact-cutlass",
    },
}


def _summary_row(
    m: int,
    arm: str,
    median_us: float,
    stable: bool = True,
    environment_digest: str = "env-a",
):
    return {
        "m": str(m),
        "arm": arm,
        "median_us": str(median_us),
        "stable_le_5_percent": str(stable),
        "comparison_group_id": IDENTITY["comparison_group_id"],
        "rerun_id": IDENTITY["rerun_id"],
        "environment_lock_digest": environment_digest,
        "protocol_lock_digest": IDENTITY["protocol_lock_digest"],
        "artifact_fingerprint_sha256": IDENTITY["per_arm_artifact_fingerprint_sha256"][
            arm
        ],
    }


def _correctness(m: int, gate: bool = True):
    return {
        "evidence_identity": IDENTITY,
        "cases": {
            str(m): {
                "paired_gate_pass": gate,
                "arms": {"cutedsl_bf16_fused": {"formal_pass": gate}},
            }
        },
    }


def test_speedup_uses_baseline_over_cutedsl():
    assert build_result.speedup_percent(baseline_us=200, candidate_us=100) == 100
    assert (
        abs(build_result.speedup_percent(baseline_us=80, candidate_us=100) + 20) < 1e-9
    )


def test_paired_result_uses_only_bf16_boundaries():
    summary = [
        _summary_row(256, "cutedsl_bf16_fused", 100),
        _summary_row(256, "cutlass_bf16_chain", 150),
    ]
    correctness = _correctness(256)
    row = build_result.comparison_rows(summary, correctness)[0]
    assert row["cutedsl_speedup_vs_matched_percent"] == 50
    assert row["formal_comparison"] is True


def test_failed_gate_invalidates_formal_comparison():
    summary = [
        _summary_row(8192, "cutedsl_bf16_fused", 110),
        _summary_row(8192, "cutlass_bf16_chain", 100),
    ]
    correctness = _correctness(8192, gate=False)
    row = build_result.comparison_rows(summary, correctness)[0]
    assert row["formal_comparison"] is False
    assert row["cutedsl_speedup_vs_matched_percent"] is None
    assert "invalid" in build_result.render_markdown([row])


def test_environment_mismatch_is_rejected():
    summary = [
        _summary_row(256, "cutedsl_bf16_fused", 100, environment_digest="env-a"),
        _summary_row(256, "cutlass_bf16_chain", 150, environment_digest="env-b"),
    ]
    correctness = _correctness(256)
    try:
        build_result.comparison_rows(summary, correctness)
    except ValueError as error:
        assert "environment_lock_digest mismatch" in str(error)
    else:
        raise AssertionError("mixed compiler/JIT environments were accepted")


def test_raw_repeat_requires_complete_arm_set():
    raw = [
        {
            **_summary_row(256, "cutedsl_bf16_fused", 100),
            "repeat": "0",
            "order": "cutedsl_bf16_fused>cutlass_bf16_chain",
            "order_index": "0",
        }
    ]
    try:
        build_result.validate_raw_rerun(raw, IDENTITY)
    except ValueError as error:
        assert "complete arm set" in str(error)
    else:
        raise AssertionError("incomplete paired repeat was accepted")
