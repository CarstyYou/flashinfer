from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_completion_timing import (
    CTA_EVENTS,
    FC2_PHASES,
    OUTPUT_TILES,
    SENTINEL,
    TASK_EVENTS,
    TILE_EVENT_BASE,
    TILE_EVENTS,
    W4_EVENT_BASE,
    W4_EVENTS,
    CompletionTimingError,
    aggregate_replays,
    analyze_files,
    analyze_replay,
    linear_regression,
    warp_rows,
)
from exp006_common import (
    CTA_TICKS,
    TASK_TICKS,
    TILE_BASE,
    TILE_STRIDE,
    W4_BASE,
    W4_TICKS,
    descriptor_order_sha256,
)


def _cta_row(offset: int) -> list[int]:
    return [
        100 + offset,
        110 + offset,
        120 + offset,
        130 + offset,
        140 + offset,
        150 + offset,
        160 + offset,
        180 + offset,
        3600 + offset,
        3610 + offset,
        3620 + offset,
        3630 + offset,
        3500 + offset,
        3640 + offset,
    ]


def _task_row(task: int) -> list[int]:
    offset = 100 * task
    row = [SENTINEL] * TASK_EVENTS
    row[0:6] = [200 + offset + 10 * index for index in range(6)]
    row[6] = 3500 + offset
    row[7] = 260 + offset
    row[8] = 262 + offset

    scatter_offsets = (60, 70, 80, 75) if task == 0 else (30, 35, 8, 4)
    post_offsets = (38, 29, 20, 23) if task == 0 else (24, 20, 12, 10)
    for tile in range(OUTPUT_TILES):
        base = TILE_EVENT_BASE + tile * TILE_EVENTS
        a = 300 + offset + 200 * tile
        starts = [a + warp for warp in range(4)]
        issues = [a + 20 + warp for warp in range(4)]
        scatter_starts = [a + 53 + warp for warp in range(4)]
        arrivals = [
            d + delta for d, delta in zip(scatter_starts, scatter_offsets, strict=True)
        ]
        completions = [
            e + delta for e, delta in zip(arrivals, post_offsets, strict=True)
        ]
        row[base : base + TILE_EVENTS] = [
            *starts,
            *issues,
            *scatter_starts,
            *arrivals,
            *completions,
        ]

    row[W4_EVENT_BASE : W4_EVENT_BASE + W4_EVENTS] = [
        270 + offset + 2 * index for index in range(W4_EVENTS)
    ]
    return row


def synthetic_payload() -> dict[str, object]:
    capacity = 3
    descriptor_tensors = {
        "task_expert": [1, 2, SENTINEL],
        "task_m_tile": [10, 11, SENTINEL],
        "task_slice_begin": [0, 1, SENTINEL],
        "task_slice_count": [1, 1, SENTINEL],
        "task_valid_rows": [128, 32, SENTINEL],
    }
    used_descriptors = {name: values[:2] for name, values in descriptor_tensors.items()}
    return {
        "task_tail": 2,
        "task_ticks": [_task_row(0), _task_row(1), [SENTINEL] * TASK_EVENTS],
        "task_cta_z": [0, 1, SENTINEL],
        "cta_ticks": [_cta_row(0), _cta_row(100)],
        "descriptor_tensors": descriptor_tensors,
        "task_capacity": capacity,
        "descriptor_order_sha256": descriptor_order_sha256(used_descriptors),
    }


def test_analyzer_reuses_the_common_locked_abi() -> None:
    assert TASK_EVENTS == TASK_TICKS == 339
    assert CTA_EVENTS == CTA_TICKS == 14
    assert TILE_EVENT_BASE == TILE_BASE == 9
    assert TILE_EVENTS == TILE_STRIDE == 20
    assert W4_EVENT_BASE == W4_BASE == 329
    assert W4_EVENTS == W4_TICKS == 10


def test_completion_anchored_reduction_and_both_closures() -> None:
    result = analyze_replay(synthetic_payload(), replay_id="r0")

    assert result["phase_totals_ns"] == {
        "FC2_issue_path": 640,
        "FC2_completion_materialize_pre_sync": 960,
        "FC2_atomic_scatter_body": 1888,
        "FC2_post_scatter_sync": 640,
    }
    assert result["derived"]["FC2_produce_scatter_ready_ns"] == 1600
    assert (
        result["task_records"][0]["phase_totals_ns"]["FC2_atomic_scatter_body"] == 1312
    )
    assert (
        result["task_records"][1]["phase_totals_ns"]["FC2_atomic_scatter_body"] == 576
    )

    # The collective phase uses min(D0..D3) -> max(E0..E3), while the
    # per-warp regression keeps the same-warp D0 -> E0 duration.
    assert result["task_records"][0]["warp_scatter_tile_ns"] == [
        [60] * 16,
        [70] * 16,
        [80] * 16,
        [75] * 16,
    ]
    assert result["task_records"][0]["warp_rows"] == [64, 64, 64, 64]
    assert result["task_records"][1]["warp_rows"] == [32, 32, 0, 0]
    assert result["task_records"][1]["warp_arrival_offset_from_d_sum_ns"] == [
        480,
        560,
        128,
        64,
    ]
    assert result["task_records"][0]["warp_e_to_f_sum_ns"] == [608, 464, 320, 368]
    first_tile = result["task_records"][0]["tile_boundaries_ns"][0]
    assert list(first_tile) == [
        "A0",
        "A1",
        "A2",
        "A3",
        "C0",
        "C1",
        "C2",
        "C3",
        "D0",
        "D1",
        "D2",
        "D3",
        "E0",
        "E1",
        "E2",
        "E3",
        "F0",
        "F1",
        "F2",
        "F3",
    ]
    assert first_tile["F2"] == first_tile["D2"] + 100
    assert first_tile["F0"] == first_tile["D0"] + 98
    assert result["max_e_arrival"] == {
        "warp_counts_including_ties": {"W0": 0, "W1": 16, "W2": 16, "W3": 0},
        "tie_tile_count": 0,
        "output_tile_samples": 32,
    }
    assert result["max_f_completion"] == {
        "warp_counts_including_ties": {"W0": 0, "W1": 16, "W2": 16, "W3": 0},
        "tie_tile_count": 0,
        "output_tile_samples": 32,
    }
    assert result["warp_actual_rows_distributions"]["W2"]["mean"] == 32.0

    assert result["fc2_envelope_closure"] == {
        "envelope_sum_ns": 6258,
        "additive_phase_sum_ns": 4128,
        "intertile_residual_ns": 2130,
        "delta_ns": 0,
        "pass": True,
    }
    assert result["whole_kernel_closure"]["fc2_intertile_residual_ns"] == 2130
    assert (
        result["whole_kernel_closure"]["non_fc2_compute_and_control_residual_ns"] == 662
    )
    assert result["whole_kernel_closure"]["phase_sum_ns"] == 7280
    assert result["whole_kernel_closure"]["pass"]

    descriptors = {
        name: values[:2]
        for name, values in synthetic_payload()["descriptor_tensors"].items()
    }
    assert result["descriptor_order_sha256"] == descriptor_order_sha256(descriptors)
    assert result["descriptor_tensors"] == descriptors
    assert all(
        gate["calibration_gate_pass"] for gate in result["marker_cost_gate"].values()
    )


def test_same_warp_scatter_edge_and_collective_handoff_are_hard_gates() -> None:
    payload = synthetic_payload()
    base = TILE_EVENT_BASE
    payload["task_ticks"][0][base + 14] = payload["task_ticks"][0][base + 10] - 1
    with pytest.raises(CompletionTimingError, match="A/C/D/E/F W2"):
        analyze_replay(payload)

    payload = synthetic_payload()
    base = TILE_EVENT_BASE
    payload["task_ticks"][0][base + 8] = payload["task_ticks"][0][base + 4]
    with pytest.raises(CompletionTimingError, match="pre-scatter collective boundary"):
        analyze_replay(payload)


def test_each_f_must_follow_its_same_warp_e() -> None:
    payload = synthetic_payload()
    base = TILE_EVENT_BASE
    payload["task_ticks"][0][base + 16] = payload["task_ticks"][0][base + 12] - 1
    with pytest.raises(CompletionTimingError, match="A/C/D/E/F W0"):
        analyze_replay(payload)

    payload = synthetic_payload()
    next_a = payload["task_ticks"][0][base + TILE_EVENTS]
    payload["task_ticks"][0][base + 18] = next_a + 4  # F2 is collective max(F).
    with pytest.raises(CompletionTimingError, match="A precedes prior boundary"):
        analyze_replay(payload)


def test_exact_fill_and_unused_sentinel_are_hard_gates() -> None:
    payload = synthetic_payload()
    payload["task_ticks"][0][TILE_EVENT_BASE] = SENTINEL
    with pytest.raises(CompletionTimingError, match="missing task event|exact-fill"):
        analyze_replay(payload)

    payload = synthetic_payload()
    payload["task_ticks"][2][0] = 123
    with pytest.raises(CompletionTimingError, match="beyond task_tail|exact sentinel"):
        analyze_replay(payload)

    payload = synthetic_payload()
    payload["task_cta_z"][2] = 0
    with pytest.raises(CompletionTimingError, match="beyond task_tail|exact sentinel"):
        analyze_replay(payload)


def test_descriptor_contract_and_unused_descriptor_sentinel() -> None:
    payload = synthetic_payload()
    payload["descriptor_tensors"]["task_slice_count"][0] = 2
    with pytest.raises(CompletionTimingError, match="locked ABI requires 1"):
        analyze_replay(payload)

    payload = synthetic_payload()
    payload["descriptor_tensors"]["task_valid_rows"][2] = 0
    with pytest.raises(CompletionTimingError, match="beyond task_tail"):
        analyze_replay(payload)

    payload = synthetic_payload()
    payload["descriptor_order_sha256"] = "stale-hash"
    with pytest.raises(CompletionTimingError, match="descriptor_order_sha256"):
        analyze_replay(payload)

    payload = synthetic_payload()
    payload["task_capacity"] = 4
    with pytest.raises(CompletionTimingError, match="task_capacity 4"):
        analyze_replay(payload)

    with pytest.raises(CompletionTimingError, match="override 1 disagrees"):
        analyze_replay(synthetic_payload(), task_tail=1)


@pytest.mark.parametrize(
    "key", ("task_tail", "task_capacity", "descriptor_order_sha256")
)
def test_formal_capture_identity_fields_are_mandatory(key: str) -> None:
    payload = synthetic_payload()
    del payload[key]
    with pytest.raises(CompletionTimingError, match=f"missing payload keys: {key}"):
        analyze_replay(payload, task_tail=2)


def test_task_envelopes_cannot_overlap_on_one_cta() -> None:
    payload = synthetic_payload()
    payload["task_cta_z"][1] = 0
    with pytest.raises(CompletionTimingError, match="consumer envelopes overlap"):
        analyze_replay(payload)


def test_regressions_use_cta_max_tail_and_all_same_warp_durations() -> None:
    result = analyze_replay(synthetic_payload())
    regressions = result["regressions"]
    cta_fit = regressions["cta_scatter_task_total_vs_valid_rows"]
    assert cta_fit["slope_ns_per_row"] == pytest.approx(23.0 / 3.0)
    assert cta_fit["intercept_ns"] == pytest.approx(992.0 / 3.0)

    per_warp = regressions["per_warp_scatter_tile_vs_actual_rows"]
    assert per_warp["sample_count"] == 128
    assert per_warp["warps"] == ["W0", "W1", "W2", "W3"]
    assert per_warp["same_warp_edge"] == "Ei-Di"
    assert per_warp["fit"]["slope_ns_per_row"] == pytest.approx(1.0369318181818181)
    assert "warp_scatter_task_total_vs_actual_rows" not in regressions

    fit = linear_regression([1, 2, 3], [3, 5, 7])
    assert fit["slope_ns_per_row"] == pytest.approx(2.0)
    assert fit["intercept_ns"] == pytest.approx(1.0)
    assert fit["r_squared"] == pytest.approx(1.0)
    constant_x = linear_regression([4, 4], [1, 2])
    assert constant_x["degenerate"] == "constant_x"
    assert list(constant_x).count("degenerate") == 1


def test_per_warp_regression_rejects_a_negative_same_warp_scatter_interval() -> None:
    payload = synthetic_payload()
    base = TILE_EVENT_BASE
    payload["task_ticks"][0][base + 12] = payload["task_ticks"][0][base + 8] - 1
    with pytest.raises(CompletionTimingError, match="A/C/D/E/F W0"):
        analyze_replay(payload)


def test_marker_gate_reports_raw_upper_bound_without_subtraction() -> None:
    payload = synthetic_payload()
    for row in payload["task_ticks"][:2]:
        row[8] = row[7] + 3
    result = analyze_replay(payload)
    assert not result["marker_cost_gate"]["FC2_issue_path"]["calibration_gate_pass"]
    assert (
        result["marker_cost_gate"]["FC2_issue_path"]["reporting_class"]
        == "raw_upper_bound_inconclusive"
    )
    assert result["marker_cost_gate"]["FC2_completion_materialize_pre_sync"][
        "calibration_gate_pass"
    ]
    assert result["phase_totals_ns"]["FC2_issue_path"] == 640


def test_five_replay_aggregation_and_drift_gates() -> None:
    replays = [
        analyze_replay(synthetic_payload(), replay_id=f"r{index}") for index in range(5)
    ]
    aggregate = aggregate_replays(replays)
    assert aggregate["replays"] == 5
    assert aggregate["phase_totals_ns"]["FC2_atomic_scatter_body"] == 9440
    for phase in FC2_PHASES:
        stats = aggregate["phase_replay_statistics"][phase]
        assert stats["duration_ns"]["cv"] == 0.0
        assert stats["sm_equivalent_share_pct"]["p50"] == pytest.approx(
            stats["sm_equivalent_share_pct"]["mean"]
        )
        assert stats["replay_share_cv_gate_pass"]
        assert stats["reporting_class"] == "diagnostic_estimate"
    assert (
        aggregate["regressions"]["per_warp_scatter_tile_vs_actual_rows"]["sample_count"]
        == 640
    )
    assert aggregate["max_e_arrival"]["warp_counts_including_ties"] == {
        "W0": 0,
        "W1": 80,
        "W2": 80,
        "W3": 0,
    }
    assert aggregate["max_f_completion"]["warp_counts_including_ties"] == {
        "W0": 0,
        "W1": 80,
        "W2": 80,
        "W3": 0,
    }
    assert aggregate["fc2_envelope_closure"]["envelope_sum_ns"] == 31290
    assert aggregate["fc2_envelope_closure"]["intertile_residual_ns"] == 10650
    assert aggregate["whole_kernel_closure"]["denominator_ns"] == 36400
    assert aggregate["whole_kernel_closure"]["phase_sum_ns"] == 36400
    assert aggregate["fc2_envelope_closure"]["pass"]
    assert aggregate["whole_kernel_closure"]["pass"]

    with pytest.raises(CompletionTimingError, match="expected 5 replays"):
        aggregate_replays(replays[:4])

    drifted = deepcopy(replays)
    drifted[-1]["descriptor_order_sha256"] = "descriptor-drift"
    with pytest.raises(CompletionTimingError, match="descriptor order drift"):
        aggregate_replays(drifted)

    inconsistent = deepcopy(replays)
    inconsistent[-1]["fc2_envelope_closure"]["envelope_sum_ns"] += 1
    with pytest.raises(CompletionTimingError, match="components are inconsistent"):
        aggregate_replays(inconsistent)


def test_five_json_capture_files_are_self_contained(tmp_path: Path) -> None:
    paths = []
    for replay in range(5):
        path = tmp_path / f"timing_{replay}.json"
        path.write_text(json.dumps(synthetic_payload()))
        paths.append(path)
    result = analyze_files(paths)
    assert result["schema"] == "exp006.completion-timing.v2"
    assert result["event_abi"]["task_ticks"] == 339
    assert result["event_abi"]["tile_stride"] == 20
    assert len(result["inputs"]) == 5
    assert len({item["sha256"] for item in result["inputs"]}) == 1
    assert result["aggregate"]["replays"] == 5


def test_warp_rows_matches_the_locked_two_by_two_layout() -> None:
    assert warp_rows(128) == [64, 64, 64, 64]
    assert warp_rows(65) == [64, 64, 1, 1]
    assert warp_rows(32) == [32, 32, 0, 0]
