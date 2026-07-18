#!/usr/bin/env python3
"""Reduce exp_004 whole-kernel ``%globaltimer`` captures on the host.

The additive denominator is SM-equivalent wall time, not one selected CTA:

    D = grid_z * (max(CTA final) - min(CTA entry))

Every additive phase is measured on a CTA timeline.  Warp-4 producer/TMA
intervals overlap the consumer timeline and are therefore reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SENTINEL = -1
CTA_EVENT_NAMES = (
    "entry",
    "p0_start",
    "p0_end",
    "p1_end",
    "p2_end",
    "p3_end",
    "p4_end",
    "compute_loop_start",
    "warp0_exit",
    "warp1_exit",
    "warp2_exit",
    "warp3_exit",
    "warp4_pre_tail",
    "warp4_final",
)
TASK_EVENTS = 65
OUTPUT_TILES = 16
W4_INTERVAL_NAMES = (
    "gate_tma",
    "gate_pass_wait",
    "up_tma",
    "down_tma",
    "final_pass_wait",
)
ADDITIVE_PHASES = (
    "launch_skew/early_finish_idle",
    "prologue",
    "P0",
    "P1",
    "P2",
    "P3",
    "P4",
    "compute_setup",
    "T0_claim",
    "T0_cache_setup",
    "Gate",
    "Up",
    "SwiGLU_Q1",
    "FC2_setup",
    "FC2_gemm",
    "FC2_epilogue_scatter",
    "task_control_final_drain",
)
PHASE_MEANINGS = {
    "launch_skew/early_finish_idle": "CTA launch skew plus early-finish idle",
    "prologue": "kernel entry through P0 start",
    "P0": "Clear/init through its grid barrier",
    "P1": "Histogram through its grid barrier",
    "P2": "Prefix through its grid barrier",
    "P3": "Route + Q0 + Pack through its grid barrier",
    "P4": "deferred publish through its grid barrier",
    "compute_setup": "post-publish setup through consumer-loop entry",
    "T0_claim": "task claim/control",
    "T0_cache_setup": "claimed-task cache/setup through Gate start",
    "Gate": "FC1 Gate",
    "Up": "FC1 Up",
    "SwiGLU_Q1": "SwiGLU and Q1",
    "FC2_setup": "T3 end through first FC2 GEMM start",
    "FC2_gemm": "sum of 16 FC2 GEMM intervals",
    "FC2_epilogue_scatter": "sum of 16 FC2 epilogue/scatter intervals",
    "task_control_final_drain": (
        "unattributed compute-loop control, inter-task gaps, and final drain"
    ),
}


class TimingContractError(ValueError):
    """The capture violates the whole-kernel timing ABI."""


def _as_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _matrix(value: Any, *, columns: int, name: str) -> list[list[int]]:
    raw = _as_list(value)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise TimingContractError(f"{name} must be a non-empty 2-D matrix")
    rows: list[list[int]] = []
    for row_id, row_value in enumerate(raw):
        row = _as_list(row_value)
        if not isinstance(row, (list, tuple)) or len(row) != columns:
            raise TimingContractError(
                f"{name}[{row_id}] must contain exactly {columns} values"
            )
        rows.append([int(item) for item in row])
    return rows


def _vector(value: Any, *, name: str) -> list[int]:
    raw = _as_list(value)
    if not isinstance(raw, (list, tuple)):
        raise TimingContractError(f"{name} must be a 1-D vector")
    if any(isinstance(_as_list(item), (list, tuple)) for item in raw):
        raise TimingContractError(f"{name} must be a 1-D vector")
    return [int(item) for item in raw]


def _scalar(value: Any, *, name: str) -> int:
    if hasattr(value, "item"):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise TimingContractError(f"{name} must be an integer scalar") from error


def _require_monotonic(
    values: Sequence[int], *, label: str, strict: bool = False
) -> None:
    for index, (left, right) in enumerate(zip(values, values[1:], strict=False)):
        invalid = right <= left if strict else right < left
        if invalid:
            relation = "strictly increasing" if strict else "monotonic"
            raise TimingContractError(
                f"{label} is not {relation} at edge {index}: {left} -> {right}"
            )


def _merge_intervals(
    intervals: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _interval_sum(intervals: Sequence[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def _union_sum(intervals: Sequence[tuple[int, int]]) -> int:
    return _interval_sum(_merge_intervals(intervals))


def _intersection_sum(
    left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]]
) -> int:
    a = _merge_intervals(left)
    b = _merge_intervals(right)
    total = 0
    i = j = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _task_intervals(row: Sequence[int]) -> dict[str, list[tuple[int, int]]]:
    intervals: dict[str, list[tuple[int, int]]] = {
        "T0_claim": [(row[0], row[1])],
        "T0_cache_setup": [(row[1], row[2])],
        "Gate": [(row[2], row[3])],
        "Up": [(row[3], row[4])],
        "SwiGLU_Q1": [(row[4], row[5])],
        "FC2_setup": [(row[5], row[7])],
        "FC2_gemm": [],
        "FC2_epilogue_scatter": [],
    }
    for tile in range(OUTPUT_TILES):
        start = 7 + 3 * tile
        intervals["FC2_gemm"].append((row[start], row[start + 1]))
        intervals["FC2_epilogue_scatter"].append((row[start + 1], row[start + 2]))
    return intervals


def _validate_cta_rows(cta_ticks: Sequence[Sequence[int]]) -> list[int]:
    finals: list[int] = []
    for cta_z, row in enumerate(cta_ticks):
        if any(value == SENTINEL for value in row):
            raise TimingContractError(f"cta_ticks[{cta_z}] is not exact-fill")
        _require_monotonic(row[:8], label=f"cta_ticks[{cta_z}] prefix")
        compute_start = row[7]
        for event in range(8, 14):
            if row[event] < compute_start:
                raise TimingContractError(
                    f"cta_ticks[{cta_z}].{CTA_EVENT_NAMES[event]} precedes "
                    "compute_loop_start"
                )
        if row[13] < row[12]:
            raise TimingContractError(
                f"cta_ticks[{cta_z}] warp4_final precedes warp4_pre_tail"
            )
        finals.append(max(*row[8:12], row[13]))
    return finals


def _validate_task_row(
    row: Sequence[int], *, task: int, cta_z: int, cta_row: Sequence[int], cta_final: int
) -> None:
    if any(value == SENTINEL for value in row):
        raise TimingContractError(f"task_ticks[{task}] is not exact-fill")

    consumer_chain = list(row[:6])
    for tile in range(OUTPUT_TILES):
        start = 7 + 3 * tile
        consumer_chain.extend(row[start : start + 3])
    consumer_chain.append(row[6])
    _require_monotonic(consumer_chain, label=f"task_ticks[{task}] consumer chain")

    if row[0] < cta_row[7] or row[6] > cta_final:
        raise TimingContractError(
            f"task_ticks[{task}] consumer envelope escapes CTA {cta_z} compute span"
        )
    for interval, name in enumerate(W4_INTERVAL_NAMES):
        start = row[55 + 2 * interval]
        end = row[56 + 2 * interval]
        if end < start:
            raise TimingContractError(
                f"task_ticks[{task}] W4 {name} has negative duration"
            )
        if start < cta_row[7] or end > cta_final:
            raise TimingContractError(
                f"task_ticks[{task}] W4 {name} escapes CTA {cta_z} compute span"
            )


def _validate_tasks(
    task_ticks: Sequence[Sequence[int]],
    task_cta_z: Sequence[int],
    *,
    task_tail: int,
    cta_ticks: Sequence[Sequence[int]],
    cta_finals: Sequence[int],
) -> None:
    capacity = len(task_ticks)
    if len(task_cta_z) != capacity:
        raise TimingContractError("task_cta_z length must equal task_ticks capacity")
    if task_tail < 0 or task_tail > capacity:
        raise TimingContractError(
            f"task_tail {task_tail} outside task capacity {capacity}"
        )

    by_cta: dict[int, list[tuple[int, int, int]]] = {
        cta_z: [] for cta_z in range(len(cta_ticks))
    }
    for task, row in enumerate(task_ticks):
        if task >= task_tail:
            if task_cta_z[task] != SENTINEL or any(value != SENTINEL for value in row):
                raise TimingContractError(
                    f"task slot {task} beyond task_tail is not exact sentinel"
                )
            continue
        cta_z = task_cta_z[task]
        if not 0 <= cta_z < len(cta_ticks):
            raise TimingContractError(
                f"task_cta_z[{task}]={cta_z} outside grid_z={len(cta_ticks)}"
            )
        _validate_task_row(
            row,
            task=task,
            cta_z=cta_z,
            cta_row=cta_ticks[cta_z],
            cta_final=cta_finals[cta_z],
        )
        by_cta[cta_z].append((row[0], row[6], task))

    for cta_z, envelopes in by_cta.items():
        ordered = sorted(envelopes)
        for left, right in zip(ordered, ordered[1:], strict=False):
            if right[0] < left[1]:
                raise TimingContractError(
                    f"task consumer envelopes overlap on CTA {cta_z}: "
                    f"task {left[2]} and task {right[2]}"
                )


def _w4_summary(
    task_ticks: Sequence[Sequence[int]],
    task_cta_z: Sequence[int],
    *,
    task_tail: int,
) -> dict[str, Any]:
    w4_by_cta: dict[int, list[tuple[int, int]]] = {}
    additive_by_cta: dict[int, list[tuple[int, int]]] = {}
    additive_phase_by_cta: dict[str, dict[int, list[tuple[int, int]]]] = {
        phase: {} for phase in ADDITIVE_PHASES[8:-1]
    }
    interval_sums = {name: 0 for name in W4_INTERVAL_NAMES}

    for task in range(task_tail):
        row = task_ticks[task]
        cta_z = task_cta_z[task]
        task_intervals = _task_intervals(row)
        for phase, intervals in task_intervals.items():
            additive_by_cta.setdefault(cta_z, []).extend(intervals)
            additive_phase_by_cta[phase].setdefault(cta_z, []).extend(intervals)
        for interval, name in enumerate(W4_INTERVAL_NAMES):
            pair = (row[55 + 2 * interval], row[56 + 2 * interval])
            interval_sums[name] += pair[1] - pair[0]
            w4_by_cta.setdefault(cta_z, []).append(pair)

    w4_union = sum(_union_sum(items) for items in w4_by_cta.values())
    additive_overlap = sum(
        _intersection_sum(items, additive_by_cta.get(cta_z, []))
        for cta_z, items in w4_by_cta.items()
    )
    overlap_by_phase = {
        phase: sum(
            _intersection_sum(items, additive_phase_by_cta[phase].get(cta_z, []))
            for cta_z, items in w4_by_cta.items()
        )
        for phase in additive_phase_by_cta
    }
    interval_sum = sum(interval_sums.values())
    return {
        "classification": "non-additive overlap track",
        "interval_sums_ns": interval_sums,
        "interval_sum_ns": interval_sum,
        "union_ns": w4_union,
        "self_overlap_ns": interval_sum - w4_union,
        "overlap_with_additive_ns": additive_overlap,
        "overlap_with_additive_pct_of_w4_union": (
            100.0 * additive_overlap / w4_union if w4_union else 0.0
        ),
        "overlap_by_additive_phase_ns": overlap_by_phase,
    }


def analyze_replay(
    payload: Mapping[str, Any],
    *,
    replay_id: str = "replay_0",
    task_tail: int | None = None,
) -> dict[str, Any]:
    """Validate and reduce one replay payload."""

    required = ("cta_ticks", "task_ticks", "task_cta_z")
    missing = [name for name in required if name not in payload]
    if missing:
        raise TimingContractError(f"missing payload keys: {', '.join(missing)}")
    cta_ticks = _matrix(payload["cta_ticks"], columns=14, name="cta_ticks")
    task_ticks = _matrix(payload["task_ticks"], columns=TASK_EVENTS, name="task_ticks")
    task_cta_z = _vector(payload["task_cta_z"], name="task_cta_z")
    if task_tail is None:
        if "task_tail" in payload:
            task_tail = _scalar(payload["task_tail"], name="task_tail")
        elif (
            isinstance(payload.get("scalars"), Mapping)
            and "task_tail" in payload["scalars"]
        ):
            task_tail = _scalar(
                payload["scalars"]["task_tail"], name="scalars.task_tail"
            )
        else:
            raise TimingContractError(
                "task_tail is required in the payload or as a CLI override"
            )

    cta_finals = _validate_cta_rows(cta_ticks)
    _validate_tasks(
        task_ticks,
        task_cta_z,
        task_tail=task_tail,
        cta_ticks=cta_ticks,
        cta_finals=cta_finals,
    )

    grid_z = len(cta_ticks)
    global_start = min(row[0] for row in cta_ticks)
    global_end = max(cta_finals)
    global_wall = global_end - global_start
    if global_wall <= 0:
        raise TimingContractError("global wall time must be positive")
    denominator = grid_z * global_wall
    cta_spans = sum(
        final - row[0] for final, row in zip(cta_finals, cta_ticks, strict=True)
    )
    launch_skew = sum(row[0] - global_start for row in cta_ticks)
    early_finish_idle = sum(global_end - final for final in cta_finals)
    launch_idle = denominator - cta_spans
    if launch_idle != launch_skew + early_finish_idle:
        raise AssertionError("launch idle identity failed")

    totals = {phase: 0 for phase in ADDITIVE_PHASES}
    totals["launch_skew/early_finish_idle"] = launch_idle
    for row in cta_ticks:
        totals["prologue"] += row[1] - row[0]
        totals["P0"] += row[2] - row[1]
        totals["P1"] += row[3] - row[2]
        totals["P2"] += row[4] - row[3]
        totals["P3"] += row[5] - row[4]
        totals["P4"] += row[6] - row[5]
        totals["compute_setup"] += row[7] - row[6]

    task_phase_total = 0
    for task in range(task_tail):
        for phase, intervals in _task_intervals(task_ticks[task]).items():
            duration = _interval_sum(intervals)
            totals[phase] += duration
            task_phase_total += duration

    compute_spans = sum(
        final - row[7] for final, row in zip(cta_finals, cta_ticks, strict=True)
    )
    residual = compute_spans - task_phase_total
    if residual < 0:
        raise TimingContractError(
            "task phase sum exceeds aggregate CTA compute-loop span"
        )
    totals["task_control_final_drain"] = residual

    phase_sum = sum(totals.values())
    closure_delta = denominator - phase_sum
    if closure_delta != 0:
        raise TimingContractError(
            f"additive phases do not close denominator: delta={closure_delta} ns"
        )
    phases = [
        {
            "phase": phase,
            "meaning": PHASE_MEANINGS[phase],
            "duration_ns": totals[phase],
            "share_pct": 100.0 * totals[phase] / denominator,
        }
        for phase in ADDITIVE_PHASES
    ]
    return {
        "replay_id": replay_id,
        "timestamp_unit": "globaltimer_ns",
        "grid_z": grid_z,
        "task_tail": task_tail,
        "task_capacity": len(task_ticks),
        "global_start_ns": global_start,
        "global_end_ns": global_end,
        "global_wall_ns": global_wall,
        "sm_equivalent_denominator_ns": denominator,
        "cta_span_sum_ns": cta_spans,
        "compute_loop_span_sum_ns": compute_spans,
        "idle_detail": {
            "launch_skew_ns": launch_skew,
            "early_finish_idle_ns": early_finish_idle,
            "combined_ns": launch_idle,
        },
        "phases": phases,
        "phase_totals_ns": totals,
        "closure": {
            "phase_sum_ns": phase_sum,
            "denominator_ns": denominator,
            "delta_ns": closure_delta,
            "pass": closure_delta == 0,
        },
        "w4_non_additive": _w4_summary(task_ticks, task_cta_z, task_tail=task_tail),
        "validation": {
            "pass": True,
            "cta_exact_fill_rows": grid_z,
            "task_exact_fill_rows": task_tail,
            "task_exact_sentinel_rows": len(task_ticks) - task_tail,
            "legal_task_cta_rows": task_tail,
            "monotonic_cta_rows": grid_z,
            "monotonic_task_rows": task_tail,
        },
    }


def aggregate_replays(replays: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not replays:
        raise TimingContractError("at least one replay is required")
    denominator = sum(int(replay["sm_equivalent_denominator_ns"]) for replay in replays)
    totals = {
        phase: sum(int(replay["phase_totals_ns"][phase]) for replay in replays)
        for phase in ADDITIVE_PHASES
    }
    phase_sum = sum(totals.values())
    if phase_sum != denominator:
        raise TimingContractError("aggregate replay closure failed")
    return {
        "replays": len(replays),
        "sm_equivalent_denominator_ns": denominator,
        "phases": [
            {
                "phase": phase,
                "meaning": PHASE_MEANINGS[phase],
                "duration_ns": totals[phase],
                "share_pct": 100.0 * totals[phase] / denominator,
            }
            for phase in ADDITIVE_PHASES
        ],
        "phase_totals_ns": totals,
        "closure": {
            "phase_sum_ns": phase_sum,
            "denominator_ns": denominator,
            "delta_ns": denominator - phase_sum,
            "pass": phase_sum == denominator,
        },
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pt(path: Path) -> Mapping[str, Any]:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise TimingContractError(f"{path} does not contain a mapping")
    return value


def analyze_files(
    paths: Sequence[Path], *, task_tail: int | None = None
) -> dict[str, Any]:
    replays = []
    inputs = []
    for index, path in enumerate(paths):
        resolved = path.resolve()
        replay = analyze_replay(
            _load_pt(resolved),
            replay_id=resolved.stem or f"replay_{index}",
            task_tail=task_tail,
        )
        replays.append(replay)
        inputs.append({"path": str(resolved), "sha256": _file_sha256(resolved)})
    return {
        "schema": "exp004.whole-kernel-timing.v1",
        "timestamp_unit": "globaltimer_ns",
        "inputs": inputs,
        "replays": replays,
        "aggregate": aggregate_replays(replays),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument(
        "--task-tail",
        type=int,
        help="override task_tail for every capture",
    )
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
