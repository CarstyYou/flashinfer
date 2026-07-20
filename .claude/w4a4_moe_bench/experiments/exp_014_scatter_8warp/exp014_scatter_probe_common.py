#!/usr/bin/env python3
"""Pure-Python ABI and validators for the exp_014 Scatter phase probe."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BASE_OVERLAY_ROOT = RESULTS / "overlays"
PROBE_OVERLAY_ROOT = RESULTS / "scatter_phase_probe_overlays"

BASELINE = "baseline_4warp_scatter"
CANDIDATE = "candidate_8warp_scatter"
ARMS = (BASELINE, CANDIDATE)
EXPECTED_BASE_KERNEL_SHA256 = {
    BASELINE: "b6e141179794561f2144bdec079b7e109fddfccc2db6ba0f19c22d30ea4b34ca",
    CANDIDATE: "c88cef63492b60c0a77484b50f6400b83a103d168e1535b78972341503810184",
}

KERNEL_MODULE = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dynamic_kernel"
DISPATCH_MODULE = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch"
DISPATCH_RELATIVE_PATH = Path(
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py"
)
WRAPPER_RELATIVE_PATH = Path("flashinfer/fused_moe/cute_dsl/b12x_moe.py")
EXPECTED_DISPATCH_SHA256 = (
    "cba2d0966631a47a576747e8322b57116122f2c8e5e868f8efb3f5ea692391a4"
)
EXPECTED_WRAPPER_SHA256 = (
    "bcac806795c035decd0773f4f801d477e7ebf14c1d67c3e49eee42ee0579c0a4"
)

COMPUTE_WARPS = 8
OUTPUT_TILES = 16
SAMPLED_TASK_SLOTS = (0,)
EDGE_NAMES = ("D", "E", "F")
TICKS_PER_TILE = len(EDGE_NAMES) * COMPUTE_WARPS
TASK_TICKS = OUTPUT_TILES * TICKS_PER_TILE
SENTINEL = -1

EVENT_ABI = {
    "schema": "exp014.scatter-phase-probe-abi.v1",
    "timestamp": "%globaltimer (ns)",
    "compute_warps": COMPUTE_WARPS,
    "output_tiles": OUTPUT_TILES,
    "sampled_task_slots": list(SAMPLED_TASK_SLOTS),
    "ticks_per_tile": TICKS_PER_TILE,
    "ticks_per_task": TASK_TICKS,
    "index": "task*384 + output_tile*24 + edge*8 + warp",
    "edges": {
        "D": "after pre-scatter epilog_sync and before scatter call",
        "E": "after scatter body returns and before post-scatter epilog_sync",
        "F": "after post-scatter epilog_sync",
    },
    "reductions": {
        "body_ns": "max(E0..E7) - min(D0..D7)",
        "including_sync_ns": "max(F0..F7) - min(D0..D7)",
    },
    "scope": "sampled full-M128 task slot 0 across all 16 FC2 output tiles",
    "classification": "diagnostic-only matched probe",
}

BARRIER_PATTERNS = (
    "cute.arch.sync_threads()",
    "self.epilog_sync_barrier.arrive_and_wait()",
    "self.pass_gate_barrier.arrive_unaligned()",
    "self.pass_gate_barrier.wait_unaligned()",
    "self.pass_final_barrier.arrive_unaligned()",
    "self.pass_final_barrier.wait_unaligned()",
)


class ProbeContractError(RuntimeError):
    """Probe source, storage, or captured values violate the locked ABI."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProbeContractError(f"expected JSON object: {path}")
    return value


def barrier_fingerprint(source: str) -> dict[str, int]:
    return {pattern: source.count(pattern) for pattern in BARRIER_PATTERNS}


def event_index(output_tile: int, edge: str, warp: int) -> int:
    if not 0 <= output_tile < OUTPUT_TILES:
        raise ProbeContractError(f"invalid output tile: {output_tile}")
    if edge not in EDGE_NAMES:
        raise ProbeContractError(f"invalid edge: {edge}")
    if not 0 <= warp < COMPUTE_WARPS:
        raise ProbeContractError(f"invalid warp: {warp}")
    return output_tile * TICKS_PER_TILE + EDGE_NAMES.index(edge) * COMPUTE_WARPS + warp


def _plain(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _matrix(value: Any, *, rows: int) -> list[list[int]]:
    raw = _plain(value)
    if not isinstance(raw, (list, tuple)):
        raise ProbeContractError("scatter ticks must be a vector or matrix")
    if raw and not isinstance(_plain(raw[0]), (list, tuple)):
        if len(raw) != rows * TASK_TICKS:
            raise ProbeContractError(
                f"flat tick capacity drift: {len(raw)} != {rows * TASK_TICKS}"
            )
        raw = [
            raw[index * TASK_TICKS : (index + 1) * TASK_TICKS] for index in range(rows)
        ]
    if len(raw) != rows:
        raise ProbeContractError(f"tick row capacity drift: {len(raw)} != {rows}")
    result = []
    for index, row in enumerate(raw):
        row = _plain(row)
        if not isinstance(row, (list, tuple)) or len(row) != TASK_TICKS:
            raise ProbeContractError(
                f"tick row {index} must contain exactly {TASK_TICKS} values"
            )
        result.append([int(item) for item in row])
    return result


def validate_probe_ticks(
    ticks: Any,
    *,
    task_tail: int,
    task_capacity: int,
    task_slice_count: Any,
    task_valid_rows: Any,
) -> dict[str, Any]:
    if not 0 <= task_tail <= task_capacity:
        raise ProbeContractError(
            f"task_tail outside capacity: {task_tail}/{task_capacity}"
        )
    rows = _matrix(ticks, rows=task_capacity)
    slices = [int(value) for value in _plain(task_slice_count)]
    valid_rows = [int(value) for value in _plain(task_valid_rows)]
    if len(slices) < task_tail:
        raise ProbeContractError("task_slice_count does not cover active tasks")
    if len(valid_rows) < task_tail:
        raise ProbeContractError("task_valid_rows does not cover active tasks")
    if any(slot >= task_tail for slot in SAMPLED_TASK_SLOTS):
        raise ProbeContractError("sampled task slot is outside the active task range")
    if any(slices[slot] != 1 for slot in SAMPLED_TASK_SLOTS):
        raise ProbeContractError(
            "Scatter marker ABI requires task_slice_count==1 for sampled tasks"
        )
    if any(valid_rows[slot] != 128 for slot in SAMPLED_TASK_SLOTS):
        raise ProbeContractError(
            "Scatter marker ABI requires valid_rows==128 for sampled tasks"
        )

    filled_tasks = [
        index
        for index, row in enumerate(rows)
        if any(value != SENTINEL for value in row)
    ]
    for task, row in enumerate(rows):
        if task not in SAMPLED_TASK_SLOTS:
            if any(value != SENTINEL for value in row):
                raise ProbeContractError(f"unsampled task {task} is not exact-sentinel")
            continue
        missing = [index for index, value in enumerate(row) if value == SENTINEL]
        if missing:
            raise ProbeContractError(
                f"sampled task {task} is not exact-fill; "
                f"missing={len(missing)}, first_indices={missing[:16]}, "
                f"filled_task_count={len(filled_tasks)}, "
                f"first_filled_tasks={filled_tasks[:16]}"
            )
        for output_tile in range(OUTPUT_TILES):
            for warp in range(COMPUTE_WARPS):
                d = row[event_index(output_tile, "D", warp)]
                e = row[event_index(output_tile, "E", warp)]
                f = row[event_index(output_tile, "F", warp)]
                if not d <= e <= f:
                    raise ProbeContractError(
                        f"task {task} tile {output_tile} W{warp} is non-monotonic: "
                        f"D={d}, E={e}, F={f}"
                    )
    return {
        "schema": "exp014.scatter-phase-buffer-gate.v1",
        "task_tail": task_tail,
        "task_capacity": task_capacity,
        "task_ticks": TASK_TICKS,
        "sampled_task_slots": list(SAMPLED_TASK_SLOTS),
        "exact_fill_sampled": True,
        "exact_sentinel_unsampled": True,
        "sampled_task_slice_count_one": True,
        "sampled_task_valid_rows_128": True,
        "gate_pass": True,
    }


def interval_rows(
    ticks: Any,
    *,
    task_tail: int,
    task_capacity: int,
    task_slots: Sequence[int] = SAMPLED_TASK_SLOTS,
) -> list[dict[str, int]]:
    rows = _matrix(ticks, rows=task_capacity)
    result = []
    for task in task_slots:
        if not 0 <= task < task_tail:
            raise ProbeContractError(f"sampled task outside active range: {task}")
        row = rows[task]
        for output_tile in range(OUTPUT_TILES):
            d = [
                row[event_index(output_tile, "D", warp)]
                for warp in range(COMPUTE_WARPS)
            ]
            e = [
                row[event_index(output_tile, "E", warp)]
                for warp in range(COMPUTE_WARPS)
            ]
            f = [
                row[event_index(output_tile, "F", warp)]
                for warp in range(COMPUTE_WARPS)
            ]
            d_min = min(d)
            e_max = max(e)
            f_max = max(f)
            result.append(
                {
                    "task": task,
                    "output_tile": output_tile,
                    "d_min_ns": d_min,
                    "e_max_ns": e_max,
                    "f_max_ns": f_max,
                    "body_ns": e_max - d_min,
                    "including_sync_ns": f_max - d_min,
                }
            )
    return result


def _percentile(values: Sequence[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_intervals(rows: Sequence[Mapping[str, int]]) -> dict[str, Any]:
    if not rows:
        raise ProbeContractError("cannot summarize an empty Scatter interval set")
    result: dict[str, Any] = {"samples": len(rows)}
    for metric in ("body_ns", "including_sync_ns"):
        values = [int(row[metric]) for row in rows]
        result[metric] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p10": _percentile(values, 0.10),
            "p90": _percentile(values, 0.90),
            "min": min(values),
            "max": max(values),
        }
    return result
