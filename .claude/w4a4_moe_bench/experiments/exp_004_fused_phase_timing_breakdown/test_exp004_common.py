from __future__ import annotations

import pytest

from exp004_common import (
    CONSUMER_INTERVAL_NAMES,
    CONSUMER_WARPS,
    NUM_SMS,
    SENTINEL,
    TICKS_PER_TASK,
    W4_INTERVAL_NAMES,
    EventContractError,
    consumer_tick_index,
    decode_probe_buffer,
    iter_probe_rows,
    summarize_phase_rows,
    timing_ticks_capacity,
    validate_hardware_identity,
    validate_no_marker_buffer,
    w4_tick_index,
)


def synthetic_buffer(task_tail: int = 2, task_capacity: int = 3):
    ticks = [SENTINEL] * timing_ticks_capacity(task_capacity)
    cta = [SENTINEL] * task_capacity
    for task in range(task_tail):
        cta[task] = task % NUM_SMS
        for warp in range(CONSUMER_WARPS):
            base = 100_000 * (task + 1) + 10_000 * warp
            envelope = consumer_tick_index(
                task, warp, "task_envelope", 0, task_capacity
            )
            ticks[envelope] = base
            ticks[envelope + 1] = base + 10_000
            cursor = base + 100
            ordered = [
                "fc1_gate",
                "fc1_up",
                "swiglu_q1",
                "fc2_setup",
                *(
                    name
                    for tile in range(16)
                    for name in (f"fc2_gemm_{tile}", f"fc2_epilogue_scatter_{tile}")
                ),
            ]
            for name in ordered:
                index = consumer_tick_index(task, warp, name, 0, task_capacity)
                ticks[index] = cursor
                ticks[index + 1] = cursor + 10
                cursor += 20
        cursor = 100_000 * (task + 1) + 50
        for name in W4_INTERVAL_NAMES:
            index = w4_tick_index(task, name, 0, task_capacity)
            ticks[index] = cursor
            ticks[index + 1] = cursor + 7
            cursor += 11
    return ticks, cta


def test_slot_layout_is_dense_and_collision_free():
    capacity = 7
    indices = []
    for task in range(capacity):
        for warp in range(CONSUMER_WARPS):
            for name in CONSUMER_INTERVAL_NAMES:
                indices += [
                    consumer_tick_index(task, warp, name, edge, capacity)
                    for edge in (0, 1)
                ]
        for name in W4_INTERVAL_NAMES:
            indices += [w4_tick_index(task, name, edge, capacity) for edge in (0, 1)]
    assert len(indices) == capacity * TICKS_PER_TASK
    assert sorted(indices) == list(range(timing_ticks_capacity(capacity)))


def test_no_marker_requires_all_sentinel():
    capacity = 3
    ticks = [SENTINEL] * timing_ticks_capacity(capacity)
    cta = [SENTINEL] * capacity
    assert validate_no_marker_buffer(ticks, cta, task_capacity=capacity)["gate_pass"]
    ticks[0] = 1
    assert not validate_no_marker_buffer(ticks, cta, task_capacity=capacity)[
        "gate_pass"
    ]


def test_probe_decode_and_additive_summary():
    ticks, cta = synthetic_buffer()
    rows, gate = decode_probe_buffer(
        ticks,
        cta,
        run_id="run_0",
        task_tail=2,
        task_capacity=3,
        task_descriptors=[
            {"expert": 1, "m_tile": 10, "slice": 0, "valid_rows": 128},
            {"expert": 2, "m_tile": 11, "slice": 1, "valid_rows": 17},
        ],
    )
    assert gate["gate_pass"], gate
    assert gate["observed_tick_writes"] == 2 * TICKS_PER_TASK
    summary = summarize_phase_rows(rows)
    assert summary["consumer"]["complete_warp_tasks"] == 8
    assert summary["consumer"]["additivity_gate_pass"]
    assert 0 <= summary["consumer"]["residual_pct"] <= 100
    streamed = list(
        iter_probe_rows(
            ticks,
            cta,
            run_id="run_0",
            task_tail=2,
            task_capacity=3,
            task_descriptors=[
                {"expert": 1, "m_tile": 10, "slice": 0, "valid_rows": 128},
                {"expert": 2, "m_tile": 11, "slice": 1, "valid_rows": 17},
            ],
        )
    )
    assert streamed == rows


def test_probe_decode_fails_on_missing_and_tail_write():
    ticks, cta = synthetic_buffer()
    ticks[consumer_tick_index(0, 0, "fc1_gate", 0, 3)] = SENTINEL
    cta[2] = 0
    _, gate = decode_probe_buffer(
        ticks, cta, run_id="bad", task_tail=2, task_capacity=3
    )
    assert not gate["gate_pass"]
    assert gate["missing_tick_indices"]
    assert gate["tail_cta_writes"] == [2]


def test_probe_decode_rejects_overlap():
    ticks, cta = synthetic_buffer()
    gate_end = consumer_tick_index(0, 0, "fc1_gate", 1, 3)
    up_start = consumer_tick_index(0, 0, "fc1_up", 0, 3)
    ticks[up_start] = ticks[gate_end] - 1
    _, gate = decode_probe_buffer(
        ticks, cta, run_id="bad_overlap", task_tail=2, task_capacity=3
    )
    assert not gate["gate_pass"]
    assert any("overlaps prior interval" in error for error in gate["errors"])


def test_probe_decode_rejects_w4_overlap():
    ticks, cta = synthetic_buffer()
    prior_end = w4_tick_index(0, "gate_tma", 1, 3)
    next_start = w4_tick_index(0, "gate_pass_wait", 0, 3)
    ticks[next_start] = ticks[prior_end] - 1
    _, gate = decode_probe_buffer(
        ticks, cta, run_id="bad_w4_overlap", task_tail=2, task_capacity=3
    )
    assert not gate["gate_pass"]
    assert any("W4 gate_pass_wait: overlaps" in error for error in gate["errors"])


def test_hardware_identity_is_5kp_specific():
    accepted = {
        "name": "NVIDIA Graphics Device",
        "compute_capability": [12, 0],
        "sm_count": 110,
        "uuid": "GPU-test",
        "pci_bus_id": "0000:79:00.0",
    }
    assert validate_hardware_identity(accepted)["gate_pass"]
    rejected = dict(accepted, name="NVIDIA RTX PRO 6000 Blackwell Workstation Edition")
    assert not validate_hardware_identity(rejected)["gate_pass"]


def test_decoder_rejects_capacity_drift():
    with pytest.raises(EventContractError):
        decode_probe_buffer(
            [SENTINEL], [SENTINEL], run_id="bad", task_tail=1, task_capacity=2
        )
