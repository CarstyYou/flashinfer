#!/usr/bin/env python3
"""CPU-only ABI and event gates for exp_006.

This module deliberately imports neither Torch nor CUDA.  The capture worker
converts device tensors to ordinary Python sequences before invoking these
contracts, which keeps the dangerous slot/order logic unit-testable without a
GPU allocation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
EXP004_ROOT = ROOT.parent / "exp_004_fused_phase_timing_breakdown"

CONTROL = "measurement_no_marker"
PROBE = "completion_anchored_probe"
ARMS = (CONTROL, PROBE)

TASK_TICKS = 339
CTA_TICKS = 14
OUTPUT_TILES = 16
CONSUMER_WARPS = 4
TILE_BASE = 9
TILE_STRIDE = 20
W4_BASE = 329
W4_TICKS = 10
SENTINEL = -1

DESCRIPTOR_NAMES = (
    "task_expert",
    "task_m_tile",
    "task_slice_begin",
    "task_slice_count",
    "task_valid_rows",
)


def tile_event(tile: int, offset: int) -> int:
    if not 0 <= tile < OUTPUT_TILES:
        raise ValueError(f"invalid output tile {tile}")
    if not 0 <= offset < TILE_STRIDE:
        raise ValueError(f"invalid tile-event offset {offset}")
    return TILE_BASE + TILE_STRIDE * tile + offset


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def descriptor_order_sha256(descriptors: Mapping[str, Sequence[int]]) -> str:
    return canonical_sha256(
        [[int(value) for value in descriptors[name]] for name in DESCRIPTOR_NAMES]
    )


def _flat(values: Sequence[Any]) -> list[int]:
    result: list[int] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            result.extend(_flat(value))
        else:
            result.append(int(value))
    return result


def _rows(
    values: Sequence[Any], *, width: int, rows: int, label: str
) -> list[list[int]]:
    flat = _flat(values)
    expected = width * rows
    if len(flat) != expected:
        raise ValueError(f"{label} contains {len(flat)} values, expected {expected}")
    return [flat[index * width : (index + 1) * width] for index in range(rows)]


def validate_descriptors(
    descriptors: Mapping[str, Sequence[int]],
    *,
    task_tail: int,
    num_experts: int = 256,
    tile_rows: int = 128,
    slice_indices: int = 4,
) -> dict[str, Any]:
    errors: list[str] = []
    missing = [name for name in DESCRIPTOR_NAMES if name not in descriptors]
    if missing:
        errors.append(f"missing descriptor tensors: {missing}")
        return {
            "schema": "exp006.descriptor-gate.v1",
            "errors": errors,
            "gate_pass": False,
        }

    normalized = {
        name: [int(value) for value in descriptors[name]] for name in DESCRIPTOR_NAMES
    }
    lengths = {name: len(values) for name, values in normalized.items()}
    if any(length != task_tail for length in lengths.values()):
        errors.append(
            f"descriptor lengths {lengths} do not equal task_tail {task_tail}"
        )

    if not errors:
        for slot in range(task_tail):
            expert = normalized["task_expert"][slot]
            m_tile = normalized["task_m_tile"][slot]
            slice_begin = normalized["task_slice_begin"][slot]
            slice_count = normalized["task_slice_count"][slot]
            valid_rows = normalized["task_valid_rows"][slot]
            if not 0 <= expert < num_experts:
                errors.append(f"task {slot} has invalid expert {expert}")
            if m_tile < 0:
                errors.append(f"task {slot} has negative m_tile {m_tile}")
            if not 0 <= slice_begin < slice_indices:
                errors.append(
                    f"task {slot} has invalid slice_begin {slice_begin}; "
                    f"expected 0..{slice_indices - 1}"
                )
            if slice_count != 1:
                errors.append(
                    f"task {slot} violates _TASK_SLICE_CHUNK=1: slice_count={slice_count}"
                )
            if not 1 <= valid_rows <= tile_rows:
                errors.append(f"task {slot} has invalid valid_rows {valid_rows}")
            if errors:
                break

    return {
        "schema": "exp006.descriptor-gate.v1",
        "task_tail": task_tail,
        "lengths": lengths,
        "descriptor_order_sha256": (
            descriptor_order_sha256(normalized) if not missing else None
        ),
        "errors": errors[:32],
        "gate_pass": not errors,
    }


def validate_control_events(
    task_ticks: Sequence[Any],
    task_cta_z: Sequence[Any],
    cta_ticks: Sequence[Any],
    *,
    task_capacity: int,
    grid_z: int,
) -> dict[str, Any]:
    flat_task_ticks = _flat(task_ticks)
    flat_task_cta = _flat(task_cta_z)
    flat_cta_ticks = _flat(cta_ticks)
    sizes = {
        "task_ticks": len(flat_task_ticks),
        "task_cta_z": len(flat_task_cta),
        "cta_ticks": len(flat_cta_ticks),
    }
    expected_sizes = {
        "task_ticks": task_capacity * TASK_TICKS,
        "task_cta_z": task_capacity,
        "cta_ticks": grid_z * CTA_TICKS,
    }
    writes = {
        "task_ticks": sum(value != SENTINEL for value in flat_task_ticks),
        "task_cta_z": sum(value != SENTINEL for value in flat_task_cta),
        "cta_ticks": sum(value != SENTINEL for value in flat_cta_ticks),
    }
    errors = [
        f"{name} size {sizes[name]} != {expected}"
        for name, expected in expected_sizes.items()
        if sizes[name] != expected
    ]
    errors.extend(
        f"{name} has {count} unexpected writes"
        for name, count in writes.items()
        if count != 0
    )
    return {
        "schema": "exp006.control-event-gate.v1",
        "sizes": sizes,
        "expected_sizes": expected_sizes,
        "writes": writes,
        "errors": errors,
        "gate_pass": not errors,
    }


def validate_probe_events(
    task_ticks: Sequence[Any],
    task_cta_z: Sequence[Any],
    cta_ticks: Sequence[Any],
    *,
    task_tail: int,
    task_capacity: int,
    grid_z: int,
) -> dict[str, Any]:
    task_rows = _rows(
        task_ticks, width=TASK_TICKS, rows=task_capacity, label="task_ticks"
    )
    cta_rows = _rows(cta_ticks, width=CTA_TICKS, rows=grid_z, label="cta_ticks")
    task_cta = _flat(task_cta_z)
    if len(task_cta) != task_capacity:
        raise ValueError(
            f"task_cta_z contains {len(task_cta)} values, expected {task_capacity}"
        )

    errors: list[str] = []
    expected_task_writes = task_tail * TASK_TICKS
    actual_task_writes = sum(value != SENTINEL for row in task_rows for value in row)
    expected_cta_writes = grid_z * CTA_TICKS
    actual_cta_writes = sum(value != SENTINEL for row in cta_rows for value in row)

    if actual_task_writes != expected_task_writes:
        errors.append(
            f"task tick writes {actual_task_writes} != {expected_task_writes}"
        )
    if any(value == SENTINEL for row in task_rows[:task_tail] for value in row):
        errors.append("missing task event inside task_tail")
    if any(value != SENTINEL for row in task_rows[task_tail:] for value in row):
        errors.append("task event written beyond task_tail")
    if any(not 0 <= value < grid_z for value in task_cta[:task_tail]):
        errors.append("invalid task-to-CTA mapping")
    if any(value != SENTINEL for value in task_cta[task_tail:]):
        errors.append("task-to-CTA mapping written beyond task_tail")
    if actual_cta_writes != expected_cta_writes:
        errors.append(f"CTA tick writes {actual_cta_writes} != {expected_cta_writes}")

    if not errors:
        for cta, row in enumerate(cta_rows):
            if any(
                right < left for left, right in zip(row[:7], row[1:8], strict=False)
            ):
                errors.append(f"CTA {cta} launch timeline is non-monotonic")
                break
            if any(value < row[7] for value in row[8:]):
                errors.append(f"CTA {cta} warp exit precedes compute-loop start")
                break
            if row[13] < row[12]:
                errors.append(f"CTA {cta} W4 tail completion precedes W4 loop exit")
                break

    if not errors:
        for task in range(task_tail):
            row = task_rows[task]
            cta = task_cta[task]
            if any(
                right < left for left, right in zip(row[:5], row[1:6], strict=False)
            ):
                errors.append(f"task {task} base consumer timeline is non-monotonic")
                break
            if not row[5] <= row[7] <= row[8] <= row[tile_event(0, 0)]:
                errors.append(f"task {task} calibration/A ordering is invalid")
                break

            prior_completion = row[8]
            prior_f_values = None
            for tile in range(OUTPUT_TILES):
                base = tile_event(tile, 0)
                a_values = row[base : base + 4]
                c_values = row[base + 4 : base + 8]
                d_values = row[base + 8 : base + 12]
                e_values = row[base + 12 : base + 16]
                f_values = row[base + 16 : base + 20]
                if max(a_values) < prior_completion:
                    errors.append(f"task {task} tile {tile} A precedes prior boundary")
                    break
                if prior_f_values is not None:
                    bad_cross_tile_warps = [
                        warp
                        for warp, (prior_f, current_a) in enumerate(
                            zip(prior_f_values, a_values, strict=True)
                        )
                        if current_a < prior_f
                    ]
                    if bad_cross_tile_warps:
                        bad = ", ".join(f"W{warp}" for warp in bad_cross_tile_warps)
                        errors.append(
                            f"task {task} tile {tile} same-warp cross-tile "
                            f"F-to-A edge invalid: {bad}"
                        )
                        break
                same_warp_edges = zip(
                    a_values,
                    c_values,
                    d_values,
                    e_values,
                    f_values,
                    strict=True,
                )
                bad_warps = [
                    warp
                    for warp, values in enumerate(same_warp_edges)
                    if any(
                        right < left
                        for left, right in zip(values, values[1:], strict=False)
                    )
                ]
                if bad_warps:
                    bad = ", ".join(f"A/C/D/E/F W{warp}" for warp in bad_warps)
                    errors.append(
                        f"task {task} tile {tile} same-warp phase edge invalid: {bad}"
                    )
                    break
                if max(c_values) > min(d_values):
                    errors.append(
                        f"task {task} tile {tile} pre-scatter collective boundary "
                        "overlaps completion/materialization"
                    )
                    break
                if max(f_values) < max(e_values):
                    errors.append(
                        f"task {task} tile {tile} max(F0..F3) precedes max(E0..E3)"
                    )
                    break
                prior_completion = max(f_values)
                prior_f_values = f_values
            if errors:
                break
            if row[6] < prior_completion:
                errors.append(f"task {task} task end precedes final tile max(F0..F3)")
                break
            if any(
                right < left
                for left, right in zip(
                    row[W4_BASE:-1], row[W4_BASE + 1 :], strict=False
                )
            ):
                errors.append(f"task {task} W4 timeline is non-monotonic")
                break
            cta_row = cta_rows[cta]
            if row[0] < cta_row[7] or row[6] > cta_row[8]:
                errors.append(f"task {task} consumer envelope escapes CTA W0 envelope")
                break
            if row[W4_BASE + W4_TICKS - 1] > cta_row[13]:
                errors.append(f"task {task} W4 envelope escapes CTA W4 envelope")
                break

    return {
        "schema": "exp006.probe-event-gate.v1",
        "task_tail": task_tail,
        "task_capacity": task_capacity,
        "grid_z": grid_z,
        "expected_task_writes": expected_task_writes,
        "actual_task_writes": actual_task_writes,
        "expected_cta_writes": expected_cta_writes,
        "actual_cta_writes": actual_cta_writes,
        "mapped_tasks": sum(value != SENTINEL for value in task_cta),
        "errors": errors[:32],
        "gate_pass": not errors,
    }


EVENT_ABI = {
    "task_ticks": TASK_TICKS,
    "base_consumer": [0, 6],
    "calibration": [7, 8],
    "tile_base": TILE_BASE,
    "tile_stride": TILE_STRIDE,
    "tile_offsets": {
        "A0": 0,
        "A1": 1,
        "A2": 2,
        "A3": 3,
        "C0": 4,
        "C1": 5,
        "C2": 6,
        "C3": 7,
        "D0": 8,
        "D1": 9,
        "D2": 10,
        "D3": 11,
        "E0": 12,
        "E1": 13,
        "E2": 14,
        "E3": 15,
        "F0": 16,
        "F1": 17,
        "F2": 18,
        "F3": 19,
    },
    "collective_boundaries": {
        "tile_start": "max(A0..A3)",
        "issue_end": "max(C0..C3)",
        "scatter_start": "min(D0..D3)",
        "scatter_end": "max(E0..E3)",
        "tile_completion": "max(F0..F3)",
    },
    "tile_completion": "max(F0..F3)",
    "output_tiles": OUTPUT_TILES,
    "w4": [W4_BASE, W4_BASE + W4_TICKS - 1],
    "cta_ticks": CTA_TICKS,
}
