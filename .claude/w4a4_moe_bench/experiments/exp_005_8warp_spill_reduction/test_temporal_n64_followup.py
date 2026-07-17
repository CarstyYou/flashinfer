import ast

import pytest

from build_temporal_n64_overlay import ANCHOR_PATH, transform
from exp005_common import TEMPORAL_N64, expected_block, require_arm_m
from run_temporal_n64_followup import (
    ABBA_ORDER,
    ANCHOR,
    SUBJECT,
    require_registry,
    summarize_abba,
)


def test_temporal_overlay_has_real_half_accumulator_and_replay_contract():
    source, records = transform(ANCHOR_PATH.read_text())
    ast.parse(source)
    assert "tiled_mma.partition_shape_C(" in source
    assert "self.tile_shape_mnk[1] // 2" in source
    assert "for fc1_half in cutlass.range_constexpr(2):" in source
    assert "self.pass_subtile_barrier.arrive_unaligned()" in source
    assert "self.pass_subtile_barrier.wait_unaligned()" in source
    assert "the temporal-N64 experiment overlay is gated-activation only" in source
    assert len(records) == 8


def test_temporal_arm_contract_is_known_and_uses_288_threads():
    require_arm_m(TEMPORAL_N64, 256)
    assert expected_block(TEMPORAL_N64) == (288, 1, 1)


def test_r2_registry_is_hash_locked_and_scoped_to_candidate_a():
    registry = require_registry()
    relationship = registry["relationships"][0]
    assert relationship["anchor"] == ANCHOR
    assert relationship["subjects"] == [SUBJECT]
    assert "pure causal latency" in relationship["prohibited_claims"][0]


def test_r2_abba_summary_uses_anchor_over_subject_ratio():
    rows = []
    for group in range(5):
        for position, arm in enumerate(ABBA_ORDER):
            rows.append(
                {
                    "group": group,
                    "position": position,
                    "arm": arm,
                    "sample_us": 100.0 if arm == ANCHOR else 125.0,
                }
            )
    summary = summarize_abba(rows, clock_policy="locked")
    assert summary["median_ratio_anchor_over_subject"] == pytest.approx(0.8)
    assert summary["median_speedup_percent"] == pytest.approx(-20.0)
    assert summary["statistical_classification"] == "slower"
