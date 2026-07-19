#!/usr/bin/env python3
"""Validate and reduce exp_006 completion-anchored FC2 timing captures.

The locked task ABI has 339 ``%globaltimer`` event slots::

    0..6       task/base consumer events
    7..8       adjacent marker-pair calibration
    9..328     16 * (A0..A3, C0..C3, D0..D3, E0..E3, F0..F3)
    329..338   five W4 producer intervals

The additive collective timeline uses ``max(A)``, ``max(C)``, ``min(D)``,
``max(E)``, and ``max(F)``.  Every A/C/D/E/F edge is also validated within
the same warp; no representative warp stands in for another warp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from exp006_common import (
    CTA_TICKS as CTA_EVENTS,
    DESCRIPTOR_NAMES as DESCRIPTOR_FIELDS,
    EVENT_ABI,
    OUTPUT_TILES,
    SENTINEL,
    TASK_TICKS as TASK_EVENTS,
    TILE_BASE as TILE_EVENT_BASE,
    TILE_STRIDE as TILE_EVENTS,
    W4_BASE as W4_EVENT_BASE,
    W4_TICKS as W4_EVENTS,
    descriptor_order_sha256,
    validate_probe_events,
)

EXPECTED_REPLAYS = 5
MAX_VALID_ROWS = 128
MAX_EXPERTS = 256
SLICE_COUNT = 1
SLICE_INDICES = 4

TILE_EVENT_OFFSETS = {
    **{f"A{warp}": warp for warp in range(4)},
    **{f"C{warp}": 4 + warp for warp in range(4)},
    **{f"D{warp}": 8 + warp for warp in range(4)},
    **{f"E{warp}": 12 + warp for warp in range(4)},
    **{f"F{warp}": 16 + warp for warp in range(4)},
}
FC2_PHASES = (
    "FC2_issue_path",
    "FC2_completion_materialize_pre_sync",
    "FC2_atomic_scatter_body",
    "FC2_post_scatter_sync",
)


class CompletionTimingError(ValueError):
    """The capture violates the exp_006 timing or descriptor contract."""


def _as_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _matrix(value: Any, *, columns: int, name: str) -> list[list[int]]:
    raw = _as_list(value)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise CompletionTimingError(f"{name} must be a non-empty 2-D matrix")
    rows: list[list[int]] = []
    for row_id, row_value in enumerate(raw):
        row = _as_list(row_value)
        if not isinstance(row, (list, tuple)) or len(row) != columns:
            raise CompletionTimingError(
                f"{name}[{row_id}] must contain exactly {columns} values"
            )
        rows.append([int(item) for item in row])
    return rows


def _vector(value: Any, *, name: str) -> list[int]:
    raw = _as_list(value)
    if not isinstance(raw, (list, tuple)):
        raise CompletionTimingError(f"{name} must be a 1-D vector")
    if any(isinstance(_as_list(item), (list, tuple)) for item in raw):
        raise CompletionTimingError(f"{name} must be a 1-D vector")
    return [int(item) for item in raw]


def _scalar(value: Any, *, name: str) -> int:
    if hasattr(value, "item"):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise CompletionTimingError(f"{name} must be an integer scalar") from error


def _require_monotonic(
    values: Sequence[int], *, label: str, strict: bool = False
) -> None:
    for edge, (left, right) in enumerate(zip(values, values[1:], strict=False)):
        invalid = right <= left if strict else right < left
        if invalid:
            relation = "strictly increasing" if strict else "monotonic"
            raise CompletionTimingError(
                f"{label} is not {relation} at edge {edge}: {left} -> {right}"
            )


def _mean(values: Sequence[float | int]) -> float:
    if not values:
        raise CompletionTimingError("cannot summarize an empty sample")
    return float(sum(values)) / len(values)


def _percentile(sorted_values: Sequence[float | int], percentile: float) -> float:
    if not sorted_values:
        raise CompletionTimingError("cannot summarize an empty sample")
    position = (len(sorted_values) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return (
        float(sorted_values[lower]) * (1.0 - fraction)
        + float(sorted_values[upper]) * fraction
    )


def distribution(values: Sequence[float | int]) -> dict[str, Any]:
    """Return the pre-registered mean/p50/p95/CV descriptive statistics."""

    ordered = sorted(float(value) for value in values)
    mean = _mean(ordered)
    variance = _mean([(value - mean) ** 2 for value in ordered])
    stddev = math.sqrt(variance)
    return {
        "count": len(ordered),
        "mean": mean,
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "stddev": stddev,
        "cv": stddev / mean if mean else None,
        "min": ordered[0],
        "max": ordered[-1],
    }


def linear_regression(
    x_values: Sequence[float | int], y_values: Sequence[float | int]
) -> dict[str, Any]:
    """Fit a simple OLS line and retain degeneracy explicitly."""

    if len(x_values) != len(y_values) or not x_values:
        raise CompletionTimingError("regression inputs must be non-empty and aligned")
    x = [float(value) for value in x_values]
    y = [float(value) for value in y_values]
    x_mean = _mean(x)
    y_mean = _mean(y)
    ss_x = sum((value - x_mean) ** 2 for value in x)
    ss_y = sum((value - y_mean) ** 2 for value in y)
    covariance = sum(
        (left - x_mean) * (right - y_mean) for left, right in zip(x, y, strict=True)
    )
    if ss_x == 0.0:
        return {
            "count": len(x),
            "x_mean": x_mean,
            "y_mean_ns": y_mean,
            "slope_ns_per_row": None,
            "intercept_ns": None,
            "pearson_r": None,
            "r_squared": None,
            "degenerate": "constant_x",
        }
    slope = covariance / ss_x
    intercept = y_mean - slope * x_mean
    if ss_y == 0.0:
        pearson = None
        r_squared = None
        degenerate = "constant_y"
    else:
        pearson = covariance / math.sqrt(ss_x * ss_y)
        r_squared = pearson * pearson
        degenerate = None
    return {
        "count": len(x),
        "x_mean": x_mean,
        "y_mean_ns": y_mean,
        "slope_ns_per_row": slope,
        "intercept_ns": intercept,
        "pearson_r": pearson,
        "r_squared": r_squared,
        "degenerate": degenerate,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_tail(payload: Mapping[str, Any], override: int | None) -> int:
    captured: list[tuple[str, int]] = []
    if "task_tail" in payload:
        captured.append(("task_tail", _scalar(payload["task_tail"], name="task_tail")))
    scalars = payload.get("scalars")
    if isinstance(scalars, Mapping) and "task_tail" in scalars:
        captured.append(
            (
                "scalars.task_tail",
                _scalar(scalars["task_tail"], name="scalars.task_tail"),
            )
        )
    captured_values = {value for _, value in captured}
    if len(captured_values) > 1:
        raise CompletionTimingError(
            "conflicting task_tail values: "
            + ", ".join(f"{name}={value}" for name, value in captured)
        )
    if not captured_values:
        raise CompletionTimingError(
            "task_tail must be present in the capture; an override cannot replace evidence"
        )
    if override is not None:
        override_value = int(override)
        if override_value != next(iter(captured_values)):
            raise CompletionTimingError(
                f"task_tail override {override_value} disagrees with capture"
            )
        return override_value
    return next(iter(captured_values))


def _descriptor_source(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("descriptor_tensors")
    if isinstance(nested, Mapping):
        return nested
    nested = payload.get("descriptors")
    if isinstance(nested, Mapping):
        return nested
    return payload


def _descriptors(
    payload: Mapping[str, Any], *, task_tail: int, capacity: int
) -> dict[str, list[int]]:
    source = _descriptor_source(payload)
    result: dict[str, list[int]] = {}
    for field in DESCRIPTOR_FIELDS:
        if field not in source:
            raise CompletionTimingError(f"missing descriptor tensor: {field}")
        vector = _vector(source[field], name=field)
        if len(vector) == task_tail:
            used = vector
        elif len(vector) == capacity:
            if any(value != SENTINEL for value in vector[task_tail:]):
                raise CompletionTimingError(
                    f"{field} entries beyond task_tail must be sentinel"
                )
            used = vector[:task_tail]
        else:
            raise CompletionTimingError(
                f"{field} length must equal task_tail ({task_tail}) or capacity ({capacity})"
            )
        result[field] = used

    for task in range(task_tail):
        valid_rows = result["task_valid_rows"][task]
        expert = result["task_expert"][task]
        m_tile = result["task_m_tile"][task]
        slice_begin = result["task_slice_begin"][task]
        slice_count = result["task_slice_count"][task]
        if not 1 <= valid_rows <= MAX_VALID_ROWS:
            raise CompletionTimingError(
                f"task_valid_rows[{task}]={valid_rows} outside 1..{MAX_VALID_ROWS}"
            )
        if not 0 <= expert < MAX_EXPERTS:
            raise CompletionTimingError(
                f"task_expert[{task}]={expert} outside 0..{MAX_EXPERTS - 1}"
            )
        if m_tile < 0:
            raise CompletionTimingError(f"task_m_tile[{task}] must be non-negative")
        if not 0 <= slice_begin < SLICE_INDICES:
            raise CompletionTimingError(
                f"task_slice_begin[{task}]={slice_begin} outside 0..{SLICE_INDICES - 1}"
            )
        if slice_count != SLICE_COUNT:
            raise CompletionTimingError(
                f"task_slice_count[{task}]={slice_count}; locked ABI requires 1"
            )
    return result


def _cta_finals(cta_ticks: Sequence[Sequence[int]]) -> list[int]:
    finals: list[int] = []
    for cta_z, row in enumerate(cta_ticks):
        if any(value == SENTINEL for value in row):
            raise CompletionTimingError(f"cta_ticks[{cta_z}] is not exact-fill")
        _require_monotonic(row[:8], label=f"cta_ticks[{cta_z}] prefix")
        compute_start = row[7]
        if any(value < compute_start for value in row[8:14]):
            raise CompletionTimingError(
                f"cta_ticks[{cta_z}] exit precedes compute-loop start"
            )
        if row[13] < row[12]:
            raise CompletionTimingError(
                f"cta_ticks[{cta_z}] W4 final precedes producer-tail start"
            )
        finals.append(max(*row[8:12], row[13]))
    return finals


def _tile_events(row: Sequence[int], tile: int) -> dict[str, int]:
    base = TILE_EVENT_BASE + TILE_EVENTS * tile
    return {
        name: int(row[base + offset]) for name, offset in TILE_EVENT_OFFSETS.items()
    }


def _validate_task_row(
    row: Sequence[int],
    *,
    task: int,
    cta_z: int,
    cta_row: Sequence[int],
) -> None:
    if any(value == SENTINEL for value in row):
        raise CompletionTimingError(f"task_ticks[{task}] is not exact-fill")
    _require_monotonic(row[:6], label=f"task_ticks[{task}] base prefix")
    if row[5] > row[7]:
        raise CompletionTimingError(
            f"task_ticks[{task}] calibration precedes base event 5"
        )
    _require_monotonic(row[7:9], label=f"task_ticks[{task}] calibration")

    previous_f = row[8]
    for tile in range(OUTPUT_TILES):
        events = _tile_events(row, tile)
        starts = [events[f"A{warp}"] for warp in range(4)]
        issues = [events[f"C{warp}"] for warp in range(4)]
        scatter_starts = [events[f"D{warp}"] for warp in range(4)]
        arrivals = [events[f"E{warp}"] for warp in range(4)]
        completions = [events[f"F{warp}"] for warp in range(4)]
        if max(starts) < previous_f:
            raise CompletionTimingError(
                f"task_ticks[{task}] tile {tile} A precedes prior boundary"
            )
        for warp, edges in enumerate(
            zip(starts, issues, scatter_starts, arrivals, completions, strict=True)
        ):
            _require_monotonic(
                edges,
                label=f"task_ticks[{task}] tile {tile} A/C/D/E/F W{warp}",
            )
        if max(issues) > min(scatter_starts):
            raise CompletionTimingError(
                f"task_ticks[{task}] tile {tile} completion/materialization "
                "overlaps the pre-scatter collective boundary"
            )
        if max(completions) < max(arrivals):
            raise CompletionTimingError(
                f"task_ticks[{task}] tile {tile} max(F0..F3) precedes max(E0..E3)"
            )
        previous_f = max(completions)

    if previous_f > row[6]:
        raise CompletionTimingError(
            f"task_ticks[{task}] final FC2 event exceeds task envelope"
        )
    _require_monotonic(
        row[W4_EVENT_BASE : W4_EVENT_BASE + W4_EVENTS],
        label=f"task_ticks[{task}] W4 sequence",
    )
    if row[0] < cta_row[7] or row[6] > cta_row[8]:
        raise CompletionTimingError(
            f"task_ticks[{task}] consumer envelope escapes CTA {cta_z} W0 span"
        )
    if any(
        value < cta_row[7] or value > cta_row[13]
        for value in row[W4_EVENT_BASE : W4_EVENT_BASE + W4_EVENTS]
    ):
        raise CompletionTimingError(
            f"task_ticks[{task}] W4 sequence escapes CTA {cta_z} W4 span"
        )


def warp_rows(valid_rows: int) -> list[int]:
    """Return actual scatter rows for W0..W3 in the locked 2x2 warp layout."""

    return [max(0, min(64, valid_rows - warp_m_base)) for warp_m_base in (0, 0, 64, 64)]


def _phase_status(
    calibration_median_ns: float,
    tile_phase_distributions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase in FC2_PHASES:
        mean_ns = float(tile_phase_distributions[phase]["mean"])
        pct = 100.0 * calibration_median_ns / mean_ns if mean_ns else None
        passes = pct is not None and pct <= 10.0
        result[phase] = {
            "calibration_median_ns": calibration_median_ns,
            "tile_phase_mean_ns": mean_ns,
            "calibration_pct_of_mean": pct,
            "calibration_gate_pass": passes,
            "reporting_class": (
                "diagnostic_estimate" if passes else "raw_upper_bound_inconclusive"
            ),
        }
    return result


def _regressions(task_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [int(record["valid_rows"]) for record in task_records]
    scatter = [
        int(record["phase_totals_ns"]["FC2_atomic_scatter_body"])
        for record in task_records
    ]
    warp_rows_samples: list[int] = []
    warp_duration_samples: list[int] = []
    for record in task_records:
        for warp in range(4):
            actual_rows = int(record["warp_rows"][warp])
            for duration in record["warp_scatter_tile_ns"][warp]:
                warp_rows_samples.append(actual_rows)
                warp_duration_samples.append(int(duration))
    per_warp_regression: dict[str, Any] = {
        "sample_count": len(warp_duration_samples),
        "warps": ["W0", "W1", "W2", "W3"],
        "same_warp_edge": "Ei-Di",
        "fit": (
            linear_regression(warp_rows_samples, warp_duration_samples)
            if warp_duration_samples
            else None
        ),
    }
    return {
        "cta_scatter_task_total_vs_valid_rows": linear_regression(valid, scatter),
        "per_warp_scatter_tile_vs_actual_rows": per_warp_regression,
    }


def analyze_replay(
    payload: Mapping[str, Any],
    *,
    replay_id: str = "replay_0",
    task_tail: int | None = None,
) -> dict[str, Any]:
    """Validate and reduce one exp_006 replay without requiring CUDA or Torch."""

    required = (
        "task_ticks",
        "task_cta_z",
        "cta_ticks",
        "task_tail",
        "task_capacity",
        "descriptor_order_sha256",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise CompletionTimingError(f"missing payload keys: {', '.join(missing)}")
    task_ticks = _matrix(payload["task_ticks"], columns=TASK_EVENTS, name="task_ticks")
    task_cta_z = _vector(payload["task_cta_z"], name="task_cta_z")
    cta_ticks = _matrix(payload["cta_ticks"], columns=CTA_EVENTS, name="cta_ticks")
    capacity = len(task_ticks)
    declared_capacity = _scalar(payload["task_capacity"], name="task_capacity")
    if declared_capacity != capacity:
        raise CompletionTimingError(
            f"task_capacity {declared_capacity} does not match task_ticks rows {capacity}"
        )
    tail = _task_tail(payload, task_tail)
    if len(task_cta_z) != capacity:
        raise CompletionTimingError("task_cta_z length must equal task capacity")
    if not 1 <= tail <= capacity:
        raise CompletionTimingError(
            f"task_tail {tail} outside task capacity {capacity}"
        )
    descriptors = _descriptors(payload, task_tail=tail, capacity=capacity)
    descriptor_hash = descriptor_order_sha256(descriptors)
    captured_hash = str(payload["descriptor_order_sha256"])
    if captured_hash != descriptor_hash:
        raise CompletionTimingError(
            "descriptor_order_sha256 does not match captured descriptor tensors"
        )
    finals = _cta_finals(cta_ticks)

    common_gate = validate_probe_events(
        task_ticks,
        task_cta_z,
        cta_ticks,
        task_tail=tail,
        task_capacity=capacity,
        grid_z=len(cta_ticks),
    )
    if not common_gate["gate_pass"]:
        raise CompletionTimingError(
            "common exp006 event gate failed: " + "; ".join(common_gate["errors"])
        )

    by_cta: dict[int, list[tuple[int, int, int]]] = {
        cta_z: [] for cta_z in range(len(cta_ticks))
    }
    for task, row in enumerate(task_ticks):
        if task >= tail:
            if task_cta_z[task] != SENTINEL or any(value != SENTINEL for value in row):
                raise CompletionTimingError(
                    f"task slot {task} beyond task_tail is not exact sentinel"
                )
            continue
        cta_z = task_cta_z[task]
        if not 0 <= cta_z < len(cta_ticks):
            raise CompletionTimingError(
                f"task_cta_z[{task}]={cta_z} outside grid_z={len(cta_ticks)}"
            )
        _validate_task_row(
            row,
            task=task,
            cta_z=cta_z,
            cta_row=cta_ticks[cta_z],
        )
        by_cta[cta_z].append((row[0], row[6], task))
    for cta_z, envelopes in by_cta.items():
        ordered = sorted(envelopes)
        for left, right in zip(ordered, ordered[1:], strict=False):
            if right[0] < left[1]:
                raise CompletionTimingError(
                    f"task consumer envelopes overlap on CTA {cta_z}: "
                    f"task {left[2]} and task {right[2]}"
                )

    phase_totals = {phase: 0 for phase in FC2_PHASES}
    task_phase_samples = {phase: [] for phase in FC2_PHASES}
    tile_phase_samples = {phase: [] for phase in FC2_PHASES}
    by_tile_samples = [{phase: [] for phase in FC2_PHASES} for _ in range(OUTPUT_TILES)]
    calibration_values: list[int] = []
    task_records: list[dict[str, Any]] = []
    max_e_arrival_warp_counts = [0, 0, 0, 0]
    max_e_arrival_tie_tiles = 0
    max_f_completion_warp_counts = [0, 0, 0, 0]
    max_f_completion_tie_tiles = 0
    fc2_envelope_total = 0
    fc2_intertile_residual = 0
    task_envelope_total = 0

    for task in range(tail):
        row = task_ticks[task]
        valid_rows = descriptors["task_valid_rows"][task]
        actual_rows = warp_rows(valid_rows)
        per_task = {phase: 0 for phase in FC2_PHASES}
        per_warp_arrival_offset = [0, 0, 0, 0]
        per_warp_post_barrier = [0, 0, 0, 0]
        per_task_max_e_warps = [0, 0, 0, 0]
        per_task_max_e_ties = 0
        per_task_max_f_warps = [0, 0, 0, 0]
        per_task_max_f_ties = 0
        warp_scatter_tiles: list[list[int]] = [[], [], [], []]
        tile_boundaries: list[dict[str, int]] = []
        calibration_ns = row[8] - row[7]
        calibration_values.append(calibration_ns)
        first_tile = _tile_events(row, 0)
        first_a = max(first_tile[f"A{warp}"] for warp in range(4))
        last_tile = _tile_events(row, OUTPUT_TILES - 1)
        last_f = max(last_tile[f"F{warp}"] for warp in range(4))
        intertile = 0
        previous_f: int | None = None

        for tile in range(OUTPUT_TILES):
            events = _tile_events(row, tile)
            tile_boundaries.append(dict(events))
            max_a = max(events[f"A{warp}"] for warp in range(4))
            max_c = max(events[f"C{warp}"] for warp in range(4))
            min_d = min(events[f"D{warp}"] for warp in range(4))
            max_e = max(events[f"E{warp}"] for warp in range(4))
            max_f = max(events[f"F{warp}"] for warp in range(4))
            max_e_warps = [warp for warp in range(4) if events[f"E{warp}"] == max_e]
            for warp in max_e_warps:
                max_e_arrival_warp_counts[warp] += 1
                per_task_max_e_warps[warp] += 1
            if len(max_e_warps) > 1:
                max_e_arrival_tie_tiles += 1
                per_task_max_e_ties += 1
            max_f_warps = [warp for warp in range(4) if events[f"F{warp}"] == max_f]
            for warp in max_f_warps:
                max_f_completion_warp_counts[warp] += 1
                per_task_max_f_warps[warp] += 1
            if len(max_f_warps) > 1:
                max_f_completion_tie_tiles += 1
                per_task_max_f_ties += 1
            durations = {
                "FC2_issue_path": max_c - max_a,
                "FC2_completion_materialize_pre_sync": min_d - max_c,
                "FC2_atomic_scatter_body": max_e - min_d,
                "FC2_post_scatter_sync": max_f - max_e,
            }
            if previous_f is not None:
                intertile += max_a - previous_f
            previous_f = max_f
            for warp in range(4):
                same_warp_scatter = events[f"E{warp}"] - events[f"D{warp}"]
                per_warp_arrival_offset[warp] += same_warp_scatter
                per_warp_post_barrier[warp] += events[f"F{warp}"] - events[f"E{warp}"]
                warp_scatter_tiles[warp].append(same_warp_scatter)
            for phase, duration_ns in durations.items():
                phase_totals[phase] += duration_ns
                per_task[phase] += duration_ns
                tile_phase_samples[phase].append(duration_ns)
                by_tile_samples[tile][phase].append(duration_ns)

        additive_task_sum = sum(per_task.values())
        fc2_envelope_ns = last_f - first_a
        if additive_task_sum + intertile != fc2_envelope_ns:
            raise CompletionTimingError(f"task {task} FC2 envelope closure failed")
        fc2_envelope_total += fc2_envelope_ns
        fc2_intertile_residual += intertile
        task_envelope_total += row[6] - row[0]
        for phase in FC2_PHASES:
            task_phase_samples[phase].append(per_task[phase])
        task_records.append(
            {
                "task_slot": task,
                "cta_z": task_cta_z[task],
                "valid_rows": valid_rows,
                "expert": descriptors["task_expert"][task],
                "m_tile": descriptors["task_m_tile"][task],
                "slice_begin": descriptors["task_slice_begin"][task],
                "slice_count": descriptors["task_slice_count"][task],
                "warp_rows": actual_rows,
                "calibration_ns": calibration_ns,
                "phase_totals_ns": per_task,
                "warp_arrival_offset_from_d_sum_ns": per_warp_arrival_offset,
                "warp_e_to_f_sum_ns": per_warp_post_barrier,
                "warp_scatter_tile_ns": warp_scatter_tiles,
                "max_e_arrival_warp_counts": per_task_max_e_warps,
                "max_e_arrival_tie_tiles": per_task_max_e_ties,
                "max_f_completion_warp_counts": per_task_max_f_warps,
                "max_f_completion_tie_tiles": per_task_max_f_ties,
                "tile_boundaries_ns": tile_boundaries,
                "fc2_envelope_ns": fc2_envelope_ns,
                "fc2_intertile_residual_ns": intertile,
            }
        )

    fc2_additive_sum = sum(phase_totals.values())
    fc2_closure_delta = fc2_envelope_total - fc2_additive_sum - fc2_intertile_residual
    if fc2_closure_delta != 0:
        raise CompletionTimingError(
            f"aggregate FC2 envelope closure failed: delta={fc2_closure_delta} ns"
        )

    grid_z = len(cta_ticks)
    global_start = min(row[0] for row in cta_ticks)
    global_end = max(finals)
    global_wall = global_end - global_start
    if global_wall <= 0:
        raise CompletionTimingError("global wall time must be positive")
    denominator = grid_z * global_wall
    cta_span_sum = sum(
        final - row[0] for row, final in zip(cta_ticks, finals, strict=True)
    )
    launch_idle = denominator - cta_span_sum
    cta_prefix_sum = sum(row[7] - row[0] for row in cta_ticks)
    compute_span_sum = sum(
        final - row[7] for row, final in zip(cta_ticks, finals, strict=True)
    )
    non_fc2_compute_residual = compute_span_sum - fc2_envelope_total
    if non_fc2_compute_residual < 0:
        raise CompletionTimingError(
            "FC2 envelope sum exceeds aggregate CTA compute-loop span"
        )
    whole_phase_sum = (
        launch_idle
        + cta_prefix_sum
        + fc2_additive_sum
        + fc2_intertile_residual
        + non_fc2_compute_residual
    )
    whole_delta = denominator - whole_phase_sum
    if whole_delta != 0:
        raise CompletionTimingError(
            f"whole-kernel SM-equivalent closure failed: delta={whole_delta} ns"
        )

    calibration_summary = distribution(calibration_values)
    task_distributions = {
        phase: distribution(values) for phase, values in task_phase_samples.items()
    }
    tile_distributions = {
        phase: distribution(values) for phase, values in tile_phase_samples.items()
    }
    by_output_tile = [
        {
            "output_tile": tile,
            "phases": {phase: distribution(samples[phase]) for phase in FC2_PHASES},
        }
        for tile, samples in enumerate(by_tile_samples)
    ]
    return {
        "replay_id": replay_id,
        "schema": "exp006.completion-replay.v1",
        "timestamp_unit": "globaltimer_ns",
        "grid_z": grid_z,
        "task_tail": tail,
        "task_capacity": capacity,
        "global_wall_ns": global_wall,
        "sm_equivalent_denominator_ns": denominator,
        "phase_totals_ns": phase_totals,
        "phases": [
            {
                "phase": phase,
                "duration_ns": phase_totals[phase],
                "sm_equivalent_share_pct": 100.0 * phase_totals[phase] / denominator,
                "fc2_additive_share_pct": 100.0
                * phase_totals[phase]
                / fc2_additive_sum,
            }
            for phase in FC2_PHASES
        ],
        "derived": {
            "FC2_produce_scatter_ready_ns": phase_totals["FC2_issue_path"]
            + phase_totals["FC2_completion_materialize_pre_sync"]
        },
        "calibration": calibration_summary,
        "marker_cost_gate": _phase_status(
            float(calibration_summary["p50"]), tile_distributions
        ),
        "task_distributions_ns": task_distributions,
        "tile_distributions_ns": tile_distributions,
        "by_output_tile_distributions_ns": by_output_tile,
        "regressions": _regressions(task_records),
        "warp_actual_rows_distributions": {
            f"W{warp}": distribution(
                [int(record["warp_rows"][warp]) for record in task_records]
            )
            for warp in range(4)
        },
        "max_e_arrival": {
            "warp_counts_including_ties": {
                f"W{warp}": max_e_arrival_warp_counts[warp] for warp in range(4)
            },
            "tie_tile_count": max_e_arrival_tie_tiles,
            "output_tile_samples": tail * OUTPUT_TILES,
        },
        "max_f_completion": {
            "warp_counts_including_ties": {
                f"W{warp}": max_f_completion_warp_counts[warp] for warp in range(4)
            },
            "tie_tile_count": max_f_completion_tie_tiles,
            "output_tile_samples": tail * OUTPUT_TILES,
        },
        "descriptor_tensors": descriptors,
        "descriptor_order_sha256": descriptor_hash,
        "task_records": task_records,
        "fc2_envelope_closure": {
            "envelope_sum_ns": fc2_envelope_total,
            "additive_phase_sum_ns": fc2_additive_sum,
            "intertile_residual_ns": fc2_intertile_residual,
            "delta_ns": fc2_closure_delta,
            "pass": fc2_closure_delta == 0,
        },
        "whole_kernel_closure": {
            "denominator_ns": denominator,
            "launch_skew_early_finish_idle_ns": launch_idle,
            "entry_through_compute_setup_ns": cta_prefix_sum,
            "fc2_additive_phase_sum_ns": fc2_additive_sum,
            "fc2_intertile_residual_ns": fc2_intertile_residual,
            "non_fc2_compute_and_control_residual_ns": non_fc2_compute_residual,
            "phase_sum_ns": whole_phase_sum,
            "delta_ns": whole_delta,
            "pass": whole_delta == 0,
        },
        "envelopes": {
            "cta_span_sum_ns": cta_span_sum,
            "compute_loop_span_sum_ns": compute_span_sum,
            "task_envelope_sum_ns": task_envelope_total,
        },
        "validation": {
            "pass": True,
            "formal_capture_fields_verified": True,
            "declared_task_capacity_verified": True,
            "descriptor_order_sha256_verified": True,
            "common_event_gate_schema": common_gate["schema"],
            "task_exact_fill_rows": tail,
            "task_exact_sentinel_rows": capacity - tail,
            "cta_exact_fill_rows": grid_z,
            "mapped_tasks": tail,
            "descriptor_rows": tail,
            "locked_output_tiles": OUTPUT_TILES,
            "locked_slice_count": SLICE_COUNT,
        },
    }


def _aggregate_regressions(replays: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = [record for replay in replays for record in replay["task_records"]]
    return _regressions(records)


def aggregate_replays(
    replays: Sequence[Mapping[str, Any]], *, expected_replays: int = EXPECTED_REPLAYS
) -> dict[str, Any]:
    """Aggregate the locked five-replay protocol and apply stability gates."""

    if len(replays) != expected_replays:
        raise CompletionTimingError(
            f"expected {expected_replays} replays, received {len(replays)}"
        )
    replay_ids = [str(replay["replay_id"]) for replay in replays]
    if len(set(replay_ids)) != len(replay_ids):
        raise CompletionTimingError("replay_id values must be unique")
    descriptor_hashes = {str(replay["descriptor_order_sha256"]) for replay in replays}
    if len(descriptor_hashes) != 1:
        raise CompletionTimingError("descriptor order drift across replays")
    for field in ("task_tail", "grid_z"):
        if len({int(replay[field]) for replay in replays}) != 1:
            raise CompletionTimingError(f"{field} drift across replays")
    for replay in replays:
        replay_id = str(replay["replay_id"])
        for closure_name in ("fc2_envelope_closure", "whole_kernel_closure"):
            closure = replay[closure_name]
            if not bool(closure["pass"]) or int(closure["delta_ns"]) != 0:
                raise CompletionTimingError(f"{replay_id} has a failed {closure_name}")
        fc2_closure = replay["fc2_envelope_closure"]
        if int(fc2_closure["envelope_sum_ns"]) - int(
            fc2_closure["additive_phase_sum_ns"]
        ) - int(fc2_closure["intertile_residual_ns"]) != int(fc2_closure["delta_ns"]):
            raise CompletionTimingError(
                f"{replay_id} FC2 closure components are inconsistent"
            )
        whole_closure = replay["whole_kernel_closure"]
        if int(whole_closure["denominator_ns"]) - int(
            whole_closure["phase_sum_ns"]
        ) != int(whole_closure["delta_ns"]):
            raise CompletionTimingError(
                f"{replay_id} whole-kernel closure components are inconsistent"
            )
        replay_additive = sum(
            int(replay["phase_totals_ns"][phase]) for phase in FC2_PHASES
        )
        if replay_additive != int(
            replay["fc2_envelope_closure"]["additive_phase_sum_ns"]
        ):
            raise CompletionTimingError(
                f"{replay_id} phase totals disagree with FC2 closure"
            )

    denominator = sum(int(replay["sm_equivalent_denominator_ns"]) for replay in replays)
    totals = {
        phase: sum(int(replay["phase_totals_ns"][phase]) for replay in replays)
        for phase in FC2_PHASES
    }
    additive_sum = sum(totals.values())
    phase_statistics: dict[str, Any] = {}
    for phase in FC2_PHASES:
        duration_stats = distribution(
            [int(replay["phase_totals_ns"][phase]) for replay in replays]
        )
        share_stats = distribution(
            [
                100.0
                * int(replay["phase_totals_ns"][phase])
                / int(replay["sm_equivalent_denominator_ns"])
                for replay in replays
            ]
        )
        fc2_share_stats = distribution(
            [
                100.0
                * int(replay["phase_totals_ns"][phase])
                / sum(int(replay["phase_totals_ns"][item]) for item in FC2_PHASES)
                for replay in replays
            ]
        )
        cv = share_stats["cv"]
        stability_pass = cv is not None and float(cv) <= 0.05
        marker_pass = all(
            bool(replay["marker_cost_gate"][phase]["calibration_gate_pass"])
            for replay in replays
        )
        phase_statistics[phase] = {
            "duration_ns": duration_stats,
            "sm_equivalent_share_pct": share_stats,
            "fc2_additive_share_pct": fc2_share_stats,
            "replay_share_cv_gate_pass": stability_pass,
            "marker_cost_gate_pass_all_replays": marker_pass,
            "reporting_class": (
                "diagnostic_estimate"
                if stability_pass and marker_pass
                else "raw_upper_bound_inconclusive"
            ),
        }

    task_records = [record for replay in replays for record in replay["task_records"]]
    task_distributions = {
        phase: distribution(
            [int(record["phase_totals_ns"][phase]) for record in task_records]
        )
        for phase in FC2_PHASES
    }
    tile_mean_replay_statistics = [
        {
            "output_tile": tile,
            "phase_mean_ns_across_replays": {
                phase: distribution(
                    [
                        float(
                            replay["by_output_tile_distributions_ns"][tile]["phases"][
                                phase
                            ]["mean"]
                        )
                        for replay in replays
                    ]
                )
                for phase in FC2_PHASES
            },
        }
        for tile in range(OUTPUT_TILES)
    ]

    fc2_closure_delta = sum(
        int(replay["fc2_envelope_closure"]["delta_ns"]) for replay in replays
    )
    whole_closure_delta = sum(
        int(replay["whole_kernel_closure"]["delta_ns"]) for replay in replays
    )
    fc2_envelope_sum = sum(
        int(replay["fc2_envelope_closure"]["envelope_sum_ns"]) for replay in replays
    )
    fc2_intertile_sum = sum(
        int(replay["fc2_envelope_closure"]["intertile_residual_ns"])
        for replay in replays
    )
    whole_fields = (
        "launch_skew_early_finish_idle_ns",
        "entry_through_compute_setup_ns",
        "fc2_additive_phase_sum_ns",
        "fc2_intertile_residual_ns",
        "non_fc2_compute_and_control_residual_ns",
        "phase_sum_ns",
    )
    whole_sums = {
        field: sum(int(replay["whole_kernel_closure"][field]) for replay in replays)
        for field in whole_fields
    }
    return {
        "replays": len(replays),
        "replay_ids": replay_ids,
        "descriptor_order_sha256": next(iter(descriptor_hashes)),
        "sm_equivalent_denominator_ns": denominator,
        "phase_totals_ns": totals,
        "phases": [
            {
                "phase": phase,
                "duration_ns": totals[phase],
                "sm_equivalent_share_pct": 100.0 * totals[phase] / denominator,
                "fc2_additive_share_pct": 100.0 * totals[phase] / additive_sum,
            }
            for phase in FC2_PHASES
        ],
        "phase_replay_statistics": phase_statistics,
        "task_distributions_ns": task_distributions,
        "tile_mean_replay_statistics_ns": tile_mean_replay_statistics,
        "regressions": _aggregate_regressions(replays),
        "warp_actual_rows_distributions": {
            f"W{warp}": distribution(
                [int(record["warp_rows"][warp]) for record in task_records]
            )
            for warp in range(4)
        },
        "max_e_arrival": {
            "warp_counts_including_ties": {
                f"W{warp}": sum(
                    int(
                        replay["max_e_arrival"]["warp_counts_including_ties"][
                            f"W{warp}"
                        ]
                    )
                    for replay in replays
                )
                for warp in range(4)
            },
            "tie_tile_count": sum(
                int(replay["max_e_arrival"]["tie_tile_count"]) for replay in replays
            ),
            "output_tile_samples": sum(
                int(replay["max_e_arrival"]["output_tile_samples"])
                for replay in replays
            ),
        },
        "max_f_completion": {
            "warp_counts_including_ties": {
                f"W{warp}": sum(
                    int(
                        replay["max_f_completion"]["warp_counts_including_ties"][
                            f"W{warp}"
                        ]
                    )
                    for replay in replays
                )
                for warp in range(4)
            },
            "tie_tile_count": sum(
                int(replay["max_f_completion"]["tie_tile_count"]) for replay in replays
            ),
            "output_tile_samples": sum(
                int(replay["max_f_completion"]["output_tile_samples"])
                for replay in replays
            ),
        },
        "calibration_ns": distribution(
            [int(record["calibration_ns"]) for record in task_records]
        ),
        "fc2_envelope_closure": {
            "envelope_sum_ns": fc2_envelope_sum,
            "additive_phase_sum_ns": additive_sum,
            "intertile_residual_ns": fc2_intertile_sum,
            "delta_ns": fc2_closure_delta,
            "pass": fc2_closure_delta == 0,
        },
        "whole_kernel_closure": {
            "denominator_ns": denominator,
            **whole_sums,
            "delta_ns": whole_closure_delta,
            "pass": whole_closure_delta == 0,
        },
    }


def _load_capture(path: Path) -> Mapping[str, Any]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text())
    else:
        try:
            import torch
        except ImportError as error:
            raise CompletionTimingError(
                "Torch is required to load non-JSON capture files"
            ) from error
        value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise CompletionTimingError(f"{path} does not contain a mapping")
    return value


def analyze_files(
    paths: Sequence[Path], *, task_tail: int | None = None
) -> dict[str, Any]:
    replays = []
    inputs = []
    for index, path in enumerate(paths):
        resolved = path.resolve()
        replay = analyze_replay(
            _load_capture(resolved),
            replay_id=resolved.stem or f"replay_{index}",
            task_tail=task_tail,
        )
        replays.append(replay)
        inputs.append({"path": str(resolved), "sha256": _file_sha256(resolved)})
    return {
        "schema": "exp006.completion-timing.v2",
        "event_abi": EVENT_ABI,
        "timestamp_unit": "globaltimer_ns",
        "inputs": inputs,
        "replays": replays,
        "aggregate": aggregate_replays(replays),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--task-tail", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)
    result = analyze_files(args.captures, task_tail=args.task_tail)
    rendered = json.dumps(result, indent=args.indent, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
