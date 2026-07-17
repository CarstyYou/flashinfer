#!/usr/bin/env python3
"""CPU-only contracts for exp_004 fused phase timing.

This module owns the immutable experiment identity, probe slot ABI, raw event
validation, and derived phase-share math.  It intentionally imports neither
Torch nor CUDA so the dangerous parts of the capture contract can be tested on
the frontend before a 5KP becomes available.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"

KERNEL_RELATIVE_PATH = Path(
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py"
)
DISPATCH_RELATIVE_PATH = Path(
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py"
)
WRAPPER_RELATIVE_PATH = Path("flashinfer/fused_moe/cute_dsl/b12x_moe.py")
KERNEL_MODULE = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dynamic_kernel"
DISPATCH_MODULE = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch"

EXPECTED_FLASHINFER_COMMIT = "748ad45594f5e701cbbdca59c60335f39d1c3b2f"
EXPECTED_CUTLASS_COMMIT = "b46b16d003484063bca4ed365e44095c4c6ed633"
EXPECTED_KERNEL_SHA256 = (
    "94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106"
)
EXPECTED_DISPATCH_SHA256 = (
    "cba2d0966631a47a576747e8322b57116122f2c8e5e868f8efb3f5ea692391a4"
)
EXPECTED_WRAPPER_SHA256 = (
    "bcac806795c035decd0773f4f801d477e7ebf14c1d67c3e49eee42ee0579c0a4"
)
EXPECTED_IMAGE_DIGEST = (
    "sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba"
)
EXPECTED_PYTHON_DEPS_SHA256 = (
    "32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74"
)

NORMAL = "normal_no_marker"
MEASUREMENT_CONTROL = "measurement_no_marker"
PROBE = "probe_candidate"
ALL_ARMS = (NORMAL, MEASUREMENT_CONTROL, PROBE)

M = 8192
E = 256
H = 2048
I = 512
TOPK = 8
NUM_SMS = 110
MAX_ACTIVE_CLUSTERS = 110
EXPECTED_GRID = (1, 1, 110)
EXPECTED_BLOCK = (160, 1, 1)
EXPECTED_TASK_TAIL = 2536
EXPECTED_TASK_HEAD = 2646
EXPECTED_ROUTED_ROWS = M * TOPK
OUTPUT_TILES = 16
CONSUMER_WARPS = 4
W4_WARP = 4
SENTINEL = -1
MEASURED_REPLAYS = 5

CONSUMER_INTERVAL_NAMES = (
    "task_envelope",
    "fc1_gate",
    "fc1_up",
    "swiglu_q1",
    "fc2_setup",
    *(f"fc2_gemm_{tile}" for tile in range(OUTPUT_TILES)),
    *(f"fc2_epilogue_scatter_{tile}" for tile in range(OUTPUT_TILES)),
)
W4_INTERVAL_NAMES = (
    "gate_tma",
    "gate_pass_wait",
    "up_tma",
    "down_tma",
    "final_pass_wait",
)
CONSUMER_INTERVAL_INDEX = {
    name: index for index, name in enumerate(CONSUMER_INTERVAL_NAMES)
}
W4_INTERVAL_INDEX = {name: index for index, name in enumerate(W4_INTERVAL_NAMES)}
CONSUMER_INTERVALS = len(CONSUMER_INTERVAL_NAMES)
W4_INTERVALS = len(W4_INTERVAL_NAMES)
EDGES = 2
TICKS_PER_TASK = CONSUMER_WARPS * CONSUMER_INTERVALS * EDGES + W4_INTERVALS * EDGES
EXPECTED_WRITES = EXPECTED_TASK_TAIL * TICKS_PER_TASK
EXPECTED_WRITE_BYTES = EXPECTED_WRITES * 8

PRIMARY_PHASES = (
    "fc1_gate",
    "fc1_up",
    "swiglu_q1",
    "fc2_setup",
    "fc2_gemm",
    "fc2_epilogue_scatter",
)
# Stores executed after a declared phase's start timestamp and therefore
# included in that phase's duration: Gate 1, Up 2, SwiGLU 2, setup 1,
# 16 * (GEMM 1 + epilogue 2) = 54 per consumer warp-task.
DECLARED_PHASE_START_SIDE_STORES = 54

FORBIDDEN_ENV_KEYS = (
    "CUTE_DSL_COMPILER_OPT",
    "FLASHINFER_CUTEDSL_IKET_OVERLAY",
    "EXP003_IKET_PROVIDER_ROOT",
    "EXP003_RUN_IKET",
    "EXP003_MARKER_OVERLAY",
    "W4A4_EXP003_MARKER_OVERLAY",
)


class EventContractError(ValueError):
    """The raw timing buffer violates the pre-registered event ABI."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError("CSV rows have inconsistent fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def require_clean_compiler_environment(
    environment: Mapping[str, str] | None = None,
) -> None:
    environment = os.environ if environment is None else environment
    enabled = [key for key in FORBIDDEN_ENV_KEYS if environment.get(key, "").strip()]
    if enabled:
        raise RuntimeError(
            "unset conflicting compiler/instrumentation state: " + ", ".join(enabled)
        )


def require_empty_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise RuntimeError(f"fresh output/JIT directory is not empty: {path}")


def artifact_manifest(root: Path) -> list[dict[str, Any]]:
    suffixes = {
        ".so",
        ".cu",
        ".cuh",
        ".cpp",
        ".ptx",
        ".cubin",
        ".sass",
        ".json",
        ".mlir",
    }
    if not root.exists():
        return []
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in suffixes
    ]


def timing_ticks_capacity(task_capacity: int) -> int:
    if task_capacity <= 0:
        raise ValueError("task_capacity must be positive")
    return task_capacity * TICKS_PER_TASK


def expected_expert_tile_base(
    row_counts: Sequence[int], tile_m: int = 128
) -> list[int]:
    result = [0]
    base = 0
    for raw in row_counts:
        count = int(raw)
        if count < 0:
            raise ValueError("negative row count")
        base += (count + tile_m - 1) // tile_m
        result.append(base)
    return result


def expected_task_records(
    row_counts: Sequence[int],
) -> list[tuple[int, int, int, int, int]]:
    """Return the locked (expert, physical tile, slice, count, valid rows) table."""
    bases = expected_expert_tile_base(row_counts)
    records: list[tuple[int, int, int, int, int]] = []
    for expert, raw in enumerate(row_counts):
        remaining = int(raw)
        local_tile = 0
        while remaining > 0:
            valid_rows = min(128, remaining)
            for slice_begin in range(I // 128):
                records.append(
                    (expert, bases[expert] + local_tile, slice_begin, 1, valid_rows)
                )
            remaining -= 128
            local_tile += 1
    return records


def expected_terminal_pair_head(routed_rows: int, *, grid_z: int = NUM_SMS) -> int:
    claim_count = 5 * 2
    productive_claims = (int(routed_rows) + claim_count - 1) // claim_count
    return (productive_claims + grid_z) * claim_count


def verify_workspace_evidence(
    snapshot: Mapping[str, Any], *, expected_row_counts: Sequence[int]
) -> dict[str, Any]:
    rows = [int(value) for value in expected_row_counts]
    observed_rows = [int(value) for value in snapshot["row_counts"]]
    write_rows = [int(value) for value in snapshot["expert_write_rows"]]
    tile_base = [int(value) for value in snapshot["expert_tile_base"]]
    tail = int(snapshot["task_tail"])
    head = int(snapshot["task_head"])
    fields = (
        "task_expert",
        "task_m_tile",
        "task_slice_begin",
        "task_slice_count",
        "task_valid_rows",
    )
    observed = [
        tuple(int(snapshot[field][index]) for field in fields) for index in range(tail)
    ]
    expected = expected_task_records(rows)
    missing = list((Counter(expected) - Counter(observed)).elements())
    unexpected = list((Counter(observed) - Counter(expected)).elements())
    checks = {
        "routed_row_sum": sum(observed_rows) == sum(rows) == EXPECTED_ROUTED_ROWS,
        "row_counts": observed_rows == rows,
        "expert_write_rows": write_rows == rows,
        "expert_tile_base": tile_base == expected_expert_tile_base(rows),
        "task_tail": tail == EXPECTED_TASK_TAIL == len(expected),
        "task_descriptor_multiset": not missing and not unexpected,
        "all_work_published": int(snapshot["all_work_published"]) == 1,
        "task_head": head == EXPECTED_TASK_HEAD,
        "pair_head": int(snapshot["pair_head"])
        == expected_terminal_pair_head(EXPECTED_ROUTED_ROWS),
    }
    return {
        "checks": checks,
        "gate_pass": all(checks.values()),
        "missing_task_descriptors": [list(item) for item in missing[:32]],
        "unexpected_task_descriptors": [list(item) for item in unexpected[:32]],
        "task_descriptor_order_sha256": canonical_sha256(observed),
        "task_descriptor_multiset_sha256": canonical_sha256(sorted(observed)),
    }


def consumer_tick_index(
    task_slot: int,
    warp_id: int,
    interval: int | str,
    edge: int,
    task_capacity: int,
) -> int:
    if not 0 <= task_slot < task_capacity:
        raise ValueError("task_slot is outside task_capacity")
    if not 0 <= warp_id < CONSUMER_WARPS:
        raise ValueError("consumer warp_id must be 0..3")
    if isinstance(interval, str):
        interval = CONSUMER_INTERVAL_INDEX[interval]
    if not 0 <= interval < CONSUMER_INTERVALS or edge not in (0, 1):
        raise ValueError("invalid consumer interval/edge")
    return (
        ((task_slot * CONSUMER_WARPS) + warp_id) * CONSUMER_INTERVALS + interval
    ) * EDGES + edge


def w4_tick_index(
    task_slot: int,
    interval: int | str,
    edge: int,
    task_capacity: int,
) -> int:
    if not 0 <= task_slot < task_capacity:
        raise ValueError("task_slot is outside task_capacity")
    if isinstance(interval, str):
        interval = W4_INTERVAL_INDEX[interval]
    if not 0 <= interval < W4_INTERVALS or edge not in (0, 1):
        raise ValueError("invalid W4 interval/edge")
    base = task_capacity * CONSUMER_WARPS * CONSUMER_INTERVALS * EDGES
    return base + ((task_slot * W4_INTERVALS + interval) * EDGES + edge)


def _tick_pair(ticks: Sequence[int], start_index: int) -> tuple[int, int]:
    return int(ticks[start_index]), int(ticks[start_index + 1])


def _phase_family(name: str) -> tuple[str, int | None]:
    for prefix in ("fc2_epilogue_scatter_", "fc2_gemm_"):
        if name.startswith(prefix):
            return prefix.removesuffix("_"), int(name.removeprefix(prefix))
    return name, None


def validate_no_marker_buffer(
    ticks: Sequence[int], task_cta_z: Sequence[int], *, task_capacity: int
) -> dict[str, Any]:
    if len(ticks) != timing_ticks_capacity(task_capacity):
        raise EventContractError("timing buffer length does not match task_capacity")
    if len(task_cta_z) != task_capacity:
        raise EventContractError("CTA map length does not match task_capacity")
    non_sentinel_ticks = sum(int(value) != SENTINEL for value in ticks)
    non_sentinel_cta = sum(int(value) != SENTINEL for value in task_cta_z)
    return {
        "gate_pass": non_sentinel_ticks == 0 and non_sentinel_cta == 0,
        "non_sentinel_ticks": non_sentinel_ticks,
        "non_sentinel_task_cta": non_sentinel_cta,
    }


def decode_probe_buffer(
    ticks: Sequence[int],
    task_cta_z: Sequence[int],
    *,
    run_id: str,
    task_tail: int,
    task_capacity: int,
    task_descriptors: Sequence[Mapping[str, int]] | None = None,
    emit_rows: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate a complete probe replay and return one row per closed interval."""
    if len(ticks) != timing_ticks_capacity(task_capacity):
        raise EventContractError("timing buffer length does not match slot ABI")
    if len(task_cta_z) != task_capacity:
        raise EventContractError("CTA map length does not match task_capacity")
    if not 0 < task_tail <= task_capacity:
        raise EventContractError("invalid task_tail")
    if task_descriptors is not None and len(task_descriptors) != task_tail:
        raise EventContractError("task descriptor count does not match task_tail")

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    error_count = 0

    def record_error(message: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < 128:
            errors.append(message)

    interval_count = 0
    expected_indices: set[int] = set()
    expected_cta_slots: set[int] = set(range(task_tail))

    for task in range(task_tail):
        cta = int(task_cta_z[task])
        if not 0 <= cta < NUM_SMS:
            record_error(f"task {task}: invalid cta_z={cta}")
        descriptor = task_descriptors[task] if task_descriptors is not None else {}
        for warp in range(CONSUMER_WARPS):
            intervals: dict[str, tuple[int, int]] = {}
            for interval, name in enumerate(CONSUMER_INTERVAL_NAMES):
                index = consumer_tick_index(task, warp, interval, 0, task_capacity)
                expected_indices.update((index, index + 1))
                start, end = _tick_pair(ticks, index)
                if start == SENTINEL or end == SENTINEL:
                    record_error(f"task {task} warp {warp} {name}: missing edge")
                elif end <= start:
                    record_error(
                        f"task {task} warp {warp} {name}: non-positive interval"
                    )
                intervals[name] = (start, end)
                family, output_tile = _phase_family(name)
                interval_count += 1
                if emit_rows:
                    rows.append(
                        {
                            "run_id": run_id,
                            "cta_z": cta,
                            "task_slot": task,
                            "expert": descriptor.get("expert", ""),
                            "m_tile": descriptor.get("m_tile", ""),
                            "slice": descriptor.get("slice", ""),
                            "valid_rows": descriptor.get("valid_rows", ""),
                            "warp_id": warp,
                            "role": "mma_consumer",
                            "phase": family,
                            "subphase": name,
                            "output_tile": "" if output_tile is None else output_tile,
                            "start_tick": start,
                            "end_tick": end,
                            "duration_tick": end - start,
                            "timestamp_unit": "clock64_tick",
                            "complete": start != SENTINEL
                            and end != SENTINEL
                            and end > start,
                        }
                    )

            envelope_start, envelope_end = intervals["task_envelope"]
            ordered_names = [
                "fc1_gate",
                "fc1_up",
                "swiglu_q1",
                "fc2_setup",
                *(
                    name
                    for tile in range(OUTPUT_TILES)
                    for name in (f"fc2_gemm_{tile}", f"fc2_epilogue_scatter_{tile}")
                ),
            ]
            previous_end = envelope_start
            for name in ordered_names:
                start, end = intervals[name]
                if start < envelope_start or end > envelope_end:
                    record_error(f"task {task} warp {warp} {name}: outside envelope")
                if start < previous_end:
                    record_error(
                        f"task {task} warp {warp} {name}: overlaps prior interval"
                    )
                previous_end = end

        w4_previous_end: int | None = None
        for interval, name in enumerate(W4_INTERVAL_NAMES):
            index = w4_tick_index(task, interval, 0, task_capacity)
            expected_indices.update((index, index + 1))
            start, end = _tick_pair(ticks, index)
            if start == SENTINEL or end == SENTINEL:
                record_error(f"task {task} W4 {name}: missing edge")
            elif end <= start:
                record_error(f"task {task} W4 {name}: non-positive interval")
            if w4_previous_end is not None and start < w4_previous_end:
                record_error(f"task {task} W4 {name}: overlaps prior interval")
            w4_previous_end = end
            family, output_tile = _phase_family(name)
            interval_count += 1
            if emit_rows:
                rows.append(
                    {
                        "run_id": run_id,
                        "cta_z": cta,
                        "task_slot": task,
                        "expert": descriptor.get("expert", ""),
                        "m_tile": descriptor.get("m_tile", ""),
                        "slice": descriptor.get("slice", ""),
                        "valid_rows": descriptor.get("valid_rows", ""),
                        "warp_id": W4_WARP,
                        "role": "tma_producer",
                        "phase": family,
                        "subphase": name,
                        "output_tile": "" if output_tile is None else output_tile,
                        "start_tick": start,
                        "end_tick": end,
                        "duration_tick": end - start,
                        "timestamp_unit": "clock64_tick",
                        "complete": start != SENTINEL
                        and end != SENTINEL
                        and end > start,
                    }
                )

    written_indices = {
        index for index, value in enumerate(ticks) if int(value) != SENTINEL
    }
    unexpected_indices = sorted(written_indices - expected_indices)
    missing_indices = sorted(expected_indices - written_indices)
    tail_cta_writes = [
        slot
        for slot in range(task_tail, task_capacity)
        if int(task_cta_z[slot]) != SENTINEL
    ]
    missing_cta = sorted(
        expected_cta_slots
        - {slot for slot in range(task_tail) if int(task_cta_z[slot]) != SENTINEL}
    )
    if unexpected_indices:
        record_error(f"unexpected tick writes: {len(unexpected_indices)}")
    if missing_indices:
        record_error(f"missing tick writes: {len(missing_indices)}")
    if tail_cta_writes:
        record_error(f"tail CTA writes: {len(tail_cta_writes)}")
    if missing_cta:
        record_error(f"missing CTA assignments: {len(missing_cta)}")

    gate = {
        "schema": "exp004.probe-event-gate.v1",
        "run_id": run_id,
        "task_tail": task_tail,
        "task_capacity": task_capacity,
        "expected_tick_writes": task_tail * TICKS_PER_TASK,
        "observed_tick_writes": len(written_indices),
        "expected_interval_rows": task_tail
        * (CONSUMER_WARPS * CONSUMER_INTERVALS + W4_INTERVALS),
        "observed_interval_rows": interval_count,
        "missing_tick_indices": missing_indices[:64],
        "unexpected_tick_indices": unexpected_indices[:64],
        "missing_cta_slots": missing_cta[:64],
        "tail_cta_writes": tail_cta_writes[:64],
        "error_count": error_count,
        "errors": errors,
        "gate_pass": error_count == 0,
    }
    return rows, gate


def iter_probe_rows(
    ticks: Sequence[int],
    task_cta_z: Sequence[int],
    *,
    run_id: str,
    task_tail: int,
    task_capacity: int,
    task_descriptors: Sequence[Mapping[str, int]] | None = None,
):
    """Yield validated-ABI interval rows without materializing 388k dictionaries.

    Call :func:`decode_probe_buffer` with ``emit_rows=False`` first.  Keeping
    validation and streaming separate makes it impossible for a partial CSV to
    be mistaken for a complete capture while bounding host memory use.
    """
    if task_descriptors is not None and len(task_descriptors) != task_tail:
        raise EventContractError("task descriptor count does not match task_tail")
    for task in range(task_tail):
        cta = int(task_cta_z[task])
        descriptor = task_descriptors[task] if task_descriptors is not None else {}
        common = {
            "run_id": run_id,
            "cta_z": cta,
            "task_slot": task,
            "expert": descriptor.get("expert", ""),
            "m_tile": descriptor.get("m_tile", ""),
            "slice": descriptor.get("slice", ""),
            "valid_rows": descriptor.get("valid_rows", ""),
        }
        for warp in range(CONSUMER_WARPS):
            for interval, name in enumerate(CONSUMER_INTERVAL_NAMES):
                index = consumer_tick_index(task, warp, interval, 0, task_capacity)
                start, end = _tick_pair(ticks, index)
                family, output_tile = _phase_family(name)
                yield {
                    **common,
                    "warp_id": warp,
                    "role": "mma_consumer",
                    "phase": family,
                    "subphase": name,
                    "output_tile": "" if output_tile is None else output_tile,
                    "start_tick": start,
                    "end_tick": end,
                    "duration_tick": end - start,
                    "timestamp_unit": "clock64_tick",
                    "complete": start != SENTINEL and end != SENTINEL and end > start,
                }
        for interval, name in enumerate(W4_INTERVAL_NAMES):
            index = w4_tick_index(task, interval, 0, task_capacity)
            start, end = _tick_pair(ticks, index)
            family, output_tile = _phase_family(name)
            yield {
                **common,
                "warp_id": W4_WARP,
                "role": "tma_producer",
                "phase": family,
                "subphase": name,
                "output_tile": "" if output_tile is None else output_tile,
                "start_tick": start,
                "end_tick": end,
                "duration_tick": end - start,
                "timestamp_unit": "clock64_tick",
                "complete": start != SENTINEL and end != SENTINEL and end > start,
            }


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("quantile of empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_phase_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute additive consumer shares and non-additive W4 overlap evidence."""
    if not rows:
        raise ValueError("no phase rows")
    if any(not bool(row["complete"]) for row in rows):
        raise EventContractError("incomplete rows cannot enter phase-share analysis")

    consumer = [row for row in rows if row["role"] == "mma_consumer"]
    envelopes = [row for row in consumer if row["phase"] == "task_envelope"]
    denominator = sum(int(row["duration_tick"]) for row in envelopes)
    if denominator <= 0:
        raise EventContractError("consumer denominator is zero")

    families: dict[str, list[Mapping[str, Any]]] = {name: [] for name in PRIMARY_PHASES}
    for row in consumer:
        phase = str(row["phase"])
        if phase in families:
            families[phase].append(row)
    phase_rows: list[dict[str, Any]] = []
    declared_sum = 0
    for phase in PRIMARY_PHASES:
        selected = families[phase]
        durations = [int(row["duration_tick"]) for row in selected]
        total = sum(durations)
        declared_sum += total
        phase_rows.append(
            {
                "phase": phase,
                "role": "mma_consumer",
                "intervals": len(durations),
                "sum_ticks": total,
                "denominator_ticks": denominator,
                "share_pct": 100.0 * total / denominator,
                "interval_p50": statistics.median(durations),
                "interval_p95": _quantile(durations, 0.95),
            }
        )
    residual = denominator - declared_sum
    if residual < 0:
        raise EventContractError("declared consumer intervals exceed task envelopes")

    return {
        "schema": "exp004.phase-share-summary.v1",
        "run_ids": sorted({str(row["run_id"]) for row in rows}),
        "consumer": {
            "complete_warp_tasks": len(envelopes),
            "denominator_ticks": denominator,
            "phases": phase_rows,
            "declared_sum_ticks": declared_sum,
            "residual_ticks": residual,
            "residual_pct": 100.0 * residual / denominator,
            "additivity_gate_pass": declared_sum <= denominator,
        },
        "w4": {
            "intervals": sum(row["role"] == "tma_producer" for row in rows),
            "interpretation": "separate producer track; not added to consumer denominator",
        },
    }


def validate_hardware_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on the project's registered 5KP identity surface."""
    name = str(identity.get("name", ""))
    capability = tuple(int(value) for value in identity.get("compute_capability", ()))
    sm_count = int(identity.get("sm_count", -1))
    checks = {
        "name": name == "NVIDIA Graphics Device",
        "not_rtx_pro_6000": "RTX PRO 6000" not in name,
        "compute_capability": capability in ((12, 0), (12, 1)),
        "sm_count": sm_count == NUM_SMS,
        "uuid": str(identity.get("uuid", "")).startswith("GPU-"),
        "pci_bus_id": bool(str(identity.get("pci_bus_id", "")).strip()),
    }
    return {"checks": checks, "gate_pass": all(checks.values())}


def require_keys(value: Mapping[str, Any], keys: Iterable[str], *, label: str) -> None:
    missing = sorted(set(keys) - set(value))
    if missing:
        raise ValueError(f"{label} is missing keys: {missing}")
