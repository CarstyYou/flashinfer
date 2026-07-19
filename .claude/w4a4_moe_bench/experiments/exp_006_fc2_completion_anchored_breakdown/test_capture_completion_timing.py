from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import capture_completion_timing as capture
from exp006_common import (
    CTA_TICKS,
    DESCRIPTOR_NAMES,
    SENTINEL,
    TASK_TICKS,
    W4_BASE,
    descriptor_order_sha256,
    tile_event,
    validate_control_events,
    validate_descriptors,
    validate_probe_events,
)


def _valid_probe_fixture():
    task_capacity = 2
    row = [0] * TASK_TICKS
    row[:6] = [100, 101, 102, 103, 104, 105]
    row[7], row[8] = 106, 107
    tick = 108
    for tile in range(16):
        base = tile_event(tile, 0)
        row[base : base + 4] = [tick + i for i in range(4)]
        row[base + 4 : base + 8] = [tick + 4 + i for i in range(4)]
        row[base + 8 : base + 12] = [tick + 8 + i for i in range(4)]
        row[base + 12 : base + 16] = [tick + 12 + i for i in range(4)]
        row[base + 16 : base + 20] = [tick + 16 + i for i in range(4)]
        tick += 20
    row[6] = tick
    row[W4_BASE:] = list(range(120, 130))

    task_ticks = [row, [SENTINEL] * TASK_TICKS]
    task_cta = [0, SENTINEL]
    cta = [
        [10, 20, 30, 40, 50, 60, 70, 80, tick + 10, tick + 11, tick + 12, tick + 13, tick + 14, tick + 15]
    ]
    descriptors = {
        "task_expert": [3],
        "task_m_tile": [1],
        "task_slice_begin": [2],
        "task_slice_count": [1],
        "task_valid_rows": [97],
    }
    return task_ticks, task_cta, cta, descriptors, task_capacity


def test_probe_event_and_descriptor_gates_accept_exact_abi() -> None:
    task_ticks, task_cta, cta, descriptors, capacity = _valid_probe_fixture()
    event_gate = validate_probe_events(
        task_ticks,
        task_cta,
        cta,
        task_tail=1,
        task_capacity=capacity,
        grid_z=1,
    )
    descriptor_gate = validate_descriptors(descriptors, task_tail=1)
    assert event_gate["gate_pass"], event_gate["errors"]
    assert descriptor_gate["gate_pass"], descriptor_gate["errors"]
    assert descriptor_gate["descriptor_order_sha256"] == descriptor_order_sha256(
        descriptors
    )
    descriptors["task_slice_begin"] = [4]
    assert not validate_descriptors(descriptors, task_tail=1)["gate_pass"]


def test_probe_gate_rejects_missing_warp_edge_and_bad_same_warp_f() -> None:
    task_ticks, task_cta, cta, _, capacity = _valid_probe_fixture()
    task_ticks[0][tile_event(4, 5)] = SENTINEL
    gate = validate_probe_events(
        task_ticks,
        task_cta,
        cta,
        task_tail=1,
        task_capacity=capacity,
        grid_z=1,
    )
    assert not gate["gate_pass"]
    assert any("missing task event" in error for error in gate["errors"])

    task_ticks, task_cta, cta, _, capacity = _valid_probe_fixture()
    task_ticks[0][tile_event(2, 18)] = task_ticks[0][tile_event(2, 14)] - 1
    gate = validate_probe_events(
        task_ticks,
        task_cta,
        cta,
        task_tail=1,
        task_capacity=capacity,
        grid_z=1,
    )
    assert not gate["gate_pass"]
    assert any("W2" in error for error in gate["errors"])


def test_probe_gate_uses_max_per_warp_f_as_collective_completion() -> None:
    task_ticks, task_cta, cta, _, capacity = _valid_probe_fixture()
    base = tile_event(3, 0)
    e_values = task_ticks[0][base + 12 : base + 16]
    # Reproduce the legal ordering that invalidated the original single-W0 F:
    # F0 can precede another warp's E, while every same-warp Fi still follows Ei.
    task_ticks[0][base + 16 : base + 20] = e_values
    gate = validate_probe_events(
        task_ticks,
        task_cta,
        cta,
        task_tail=1,
        task_capacity=capacity,
        grid_z=1,
    )
    assert gate["gate_pass"], gate["errors"]


def test_probe_gate_requires_same_warp_cross_tile_f_to_a_order() -> None:
    task_ticks, task_cta, cta, _, capacity = _valid_probe_fixture()
    previous = tile_event(2, 0)
    current = tile_event(3, 0)
    task_ticks[0][previous + 16] = task_ticks[0][current] + 1
    gate = validate_probe_events(
        task_ticks,
        task_cta,
        cta,
        task_tail=1,
        task_capacity=capacity,
        grid_z=1,
    )
    assert not gate["gate_pass"]
    assert any("same-warp cross-tile F-to-A" in error for error in gate["errors"])


def test_probe_gate_requires_same_warp_d_to_e_and_collective_handoff() -> None:
    task_ticks, task_cta, cta, _, capacity = _valid_probe_fixture()
    base = tile_event(3, 0)
    task_ticks[0][base + 14] = task_ticks[0][base + 10] - 1
    gate = validate_probe_events(
        task_ticks,
        task_cta,
        cta,
        task_tail=1,
        task_capacity=capacity,
        grid_z=1,
    )
    assert not gate["gate_pass"]
    assert any("same-warp phase edge" in error for error in gate["errors"])

    task_ticks, task_cta, cta, _, capacity = _valid_probe_fixture()
    base = tile_event(3, 0)
    task_ticks[0][base + 8] = task_ticks[0][base + 4]
    gate = validate_probe_events(
        task_ticks,
        task_cta,
        cta,
        task_tail=1,
        task_capacity=capacity,
        grid_z=1,
    )
    assert not gate["gate_pass"]
    assert any("collective boundary" in error for error in gate["errors"])


def test_control_gate_requires_all_marker_buffers_to_remain_sentinel() -> None:
    capacity = 2
    gate = validate_control_events(
        [[SENTINEL] * TASK_TICKS for _ in range(capacity)],
        [SENTINEL] * capacity,
        [[SENTINEL] * CTA_TICKS],
        task_capacity=capacity,
        grid_z=1,
    )
    assert gate["gate_pass"]
    bad = [[SENTINEL] * TASK_TICKS for _ in range(capacity)]
    bad[0][0] = 1
    assert not validate_control_events(
        bad,
        [SENTINEL] * capacity,
        [[SENTINEL] * CTA_TICKS],
        task_capacity=capacity,
        grid_z=1,
    )["gate_pass"]
    assert not validate_control_events(
        [[SENTINEL] * TASK_TICKS],
        [SENTINEL],
        [[SENTINEL] * CTA_TICKS],
        task_capacity=capacity,
        grid_z=1,
    )["gate_pass"]


def test_capture_module_is_cpu_importable_and_declares_saved_descriptors() -> None:
    assert capture.TASK_TICKS == TASK_TICKS
    assert capture.CTA_TICKS == CTA_TICKS
    assert tuple(capture.DESCRIPTOR_NAMES) == tuple(DESCRIPTOR_NAMES)


def test_capture_jit_gate_does_not_assume_cutedsl_static_files_in_jit_root() -> None:
    gate = capture._jit_identity_gate(
        [{"path": "cached/fp4_quantization.so", "sha256": "abc"}]
    )
    assert gate["gate_pass"]
    assert gate["retained_cutedsl_static_artifacts"] == {
        ".cubin": 0,
        ".ptx": 0,
        ".sass": 0,
    }
    assert gate["static_extraction_required_post_capture"]
    assert not capture._jit_identity_gate([])["gate_pass"]
