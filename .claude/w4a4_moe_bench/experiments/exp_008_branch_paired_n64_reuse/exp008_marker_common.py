#!/usr/bin/env python3
"""Shared ABI and CPU validators for exp_008 phase markers.

The public additive ledger uses only W0--W7 collective boundaries.  Pair/half
events are deliberately retained as a separate same-warp diagnostic track.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OVERLAY_ROOT = RESULTS / "marker_overlays"

KERNEL_MODULE = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dynamic_kernel"
DISPATCH_MODULE = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch"
DISPATCH_RELATIVE_PATH = Path(
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py"
)
WRAPPER_RELATIVE_PATH = Path("flashinfer/fused_moe/cute_dsl/b12x_moe.py")

BASE_KERNEL = {
    "v0": RESULTS / "overlays/temporal_n64_v0/moe_dynamic_kernel.py",
    "v1": RESULTS / "overlays/branch_paired_n64_v1/moe_dynamic_kernel.py",
}
EXPECTED_BASE_KERNEL_SHA256 = {
    "v0": "1953cbb7717cda4461a4f199d05f370a4bdb35b4b8ef7556443caf36b0b12ec2",
    "v1": "f3c246817679d962a3f7160dbe8b9e68262c919e26e306f349200961fc4ac971",
}
EXPECTED_DISPATCH_SHA256 = (
    "cba2d0966631a47a576747e8322b57116122f2c8e5e868f8efb3f5ea692391a4"
)
EXPECTED_WRAPPER_SHA256 = (
    "bcac806795c035decd0773f4f801d477e7ebf14c1d67c3e49eee42ee0579c0a4"
)

VERSIONS = ("v0", "v1")
CONTROL = "measurement_no_marker"
PROBE = "phase_probe"
MARKER_ARMS = (CONTROL, PROBE)
COMPUTE_WARPS = 8
SENTINEL = -1

# CTA record: W0..W7 entry, W0..W7 loop-start, W0..W7 loop-exit,
# W8 terminal completion, and one W0 consecutive marker-pair calibration.
CTA_ENTRY = 0
CTA_LOOP_START = 8
CTA_LOOP_EXIT = 16
CTA_W8_FINAL = 24
CTA_CALIBRATION = 25
CTA_TICKS = 27

# Task record: four public W0..W7 boundaries, six non-additive W0..W7
# pair/activation edges, and one W0 leader claim-start diagnostic.
MAIN_EVENTS = (
    "cache_ready",
    "fc1_interleaved_activation_end",
    "q1_end",
    "fc2_scatter_end",
)
PAIR_EVENTS = (
    "half0_fc1_begin",
    "half0_fc1_end",
    "half0_activation_end",
    "half1_fc1_begin",
    "half1_fc1_end",
    "half1_activation_end",
)
TASK_MAIN = 0
TASK_PAIR = len(MAIN_EVENTS) * COMPUTE_WARPS
TASK_CLAIM_START = TASK_PAIR + len(PAIR_EVENTS) * COMPUTE_WARPS
TASK_TICKS = TASK_CLAIM_START + 1

EVENT_ABI = {
    "schema": "exp008.phase-marker-abi.v1",
    "timestamp": "%globaltimer (ns)",
    "compute_warps": COMPUTE_WARPS,
    "cta_ticks": CTA_TICKS,
    "task_ticks": TASK_TICKS,
    "cta": {
        "entry": [CTA_ENTRY, CTA_ENTRY + COMPUTE_WARPS],
        "loop_start": [CTA_LOOP_START, CTA_LOOP_START + COMPUTE_WARPS],
        "loop_exit": [CTA_LOOP_EXIT, CTA_LOOP_EXIT + COMPUTE_WARPS],
        "w8_terminal": CTA_W8_FINAL,
        "calibration": [CTA_CALIBRATION, CTA_CALIBRATION + 2],
    },
    "task": {
        "main_events": list(MAIN_EVENTS),
        "main_base": TASK_MAIN,
        "pair_events": list(PAIR_EVENTS),
        "pair_base": TASK_PAIR,
        "claim_start": TASK_CLAIM_START,
    },
    "classification": {
        "main": "additive W0-W7 collective ledger",
        "pair": "non-additive same-warp diagnostic track",
        "claim_start": "W0 leader diagnostic only",
    },
}

BARRIER_PATTERNS = (
    "cute.arch.sync_threads()",
    "self.epilog_sync_barrier.arrive_and_wait()",
    "self.pass_gate_barrier.arrive_unaligned()",
    "self.pass_gate_barrier.wait_unaligned()",
    "self.pass_final_barrier.arrive_unaligned()",
    "self.pass_final_barrier.wait_unaligned()",
)


class MarkerContractError(ValueError):
    """A marker overlay or captured buffer violates the registered ABI."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise MarkerContractError(f"expected JSON object: {path}")
    return value


def barrier_fingerprint(source: str) -> dict[str, int]:
    return {pattern: source.count(pattern) for pattern in BARRIER_PATTERNS}


def _plain(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _matrix(value: Any, *, columns: int, name: str) -> list[list[int]]:
    raw = _plain(value)
    if not isinstance(raw, (list, tuple)):
        raise MarkerContractError(f"{name} must be a matrix")
    result = []
    for index, row in enumerate(raw):
        row = _plain(row)
        if not isinstance(row, (list, tuple)) or len(row) != columns:
            raise MarkerContractError(
                f"{name}[{index}] must contain exactly {columns} values"
            )
        result.append([int(item) for item in row])
    return result


def _vector(value: Any, *, name: str) -> list[int]:
    raw = _plain(value)
    if not isinstance(raw, (list, tuple)):
        raise MarkerContractError(f"{name} must be a vector")
    if any(isinstance(_plain(item), (list, tuple)) for item in raw):
        raise MarkerContractError(f"{name} must be one-dimensional")
    return [int(item) for item in raw]


def _require_monotonic(values: Sequence[int], *, label: str) -> None:
    for left, right in zip(values, values[1:], strict=False):
        if right < left:
            raise MarkerContractError(f"{label} is not monotonic: {values}")


def validate_control_events(
    task_ticks: Any, task_cta_z: Any, cta_ticks: Any
) -> dict[str, Any]:
    tasks = _matrix(task_ticks, columns=TASK_TICKS, name="task_ticks")
    ctas = _matrix(cta_ticks, columns=CTA_TICKS, name="cta_ticks")
    owners = _vector(task_cta_z, name="task_cta_z")
    if len(tasks) != len(owners):
        raise MarkerContractError("task_ticks/task_cta_z capacity mismatch")
    violations = sum(item != SENTINEL for row in tasks for item in row)
    violations += sum(item != SENTINEL for row in ctas for item in row)
    violations += sum(item != SENTINEL for item in owners)
    return {
        "schema": "exp008.marker-disabled-buffer-gate.v1",
        "sentinel": SENTINEL,
        "violations": violations,
        "gate_pass": violations == 0,
    }


def _main(row: Sequence[int], event: int) -> list[int]:
    start = TASK_MAIN + event * COMPUTE_WARPS
    return list(row[start : start + COMPUTE_WARPS])


def _pair(row: Sequence[int], event: int) -> list[int]:
    start = TASK_PAIR + event * COMPUTE_WARPS
    return list(row[start : start + COMPUTE_WARPS])


def _cta_warps(row: Sequence[int], start: int) -> list[int]:
    return list(row[start : start + COMPUTE_WARPS])


def validate_probe_events(
    task_ticks: Any,
    task_cta_z: Any,
    cta_ticks: Any,
    *,
    task_tail: int,
) -> dict[str, Any]:
    tasks = _matrix(task_ticks, columns=TASK_TICKS, name="task_ticks")
    ctas = _matrix(cta_ticks, columns=CTA_TICKS, name="cta_ticks")
    owners = _vector(task_cta_z, name="task_cta_z")
    if len(tasks) != len(owners):
        raise MarkerContractError("task_ticks/task_cta_z capacity mismatch")
    if not 0 <= int(task_tail) <= len(tasks):
        raise MarkerContractError("task_tail outside task capacity")
    if not ctas:
        raise MarkerContractError("cta_ticks is empty")

    for cta, row in enumerate(ctas):
        if any(value == SENTINEL for value in row):
            raise MarkerContractError(f"cta_ticks[{cta}] is not exact-fill")
        entries = _cta_warps(row, CTA_ENTRY)
        starts = _cta_warps(row, CTA_LOOP_START)
        exits = _cta_warps(row, CTA_LOOP_EXIT)
        for warp in range(COMPUTE_WARPS):
            _require_monotonic(
                (entries[warp], starts[warp], exits[warp]),
                label=f"cta_ticks[{cta}] W{warp}",
            )
        if row[CTA_W8_FINAL] < min(entries):
            raise MarkerContractError(f"cta_ticks[{cta}] W8 final precedes entry")
        _require_monotonic(
            row[CTA_CALIBRATION : CTA_CALIBRATION + 2],
            label=f"cta_ticks[{cta}] calibration",
        )

    by_cta: dict[int, list[tuple[int, int, int]]] = {
        cta: [] for cta in range(len(ctas))
    }
    for task, row in enumerate(tasks):
        if task >= task_tail:
            if owners[task] != SENTINEL or any(value != SENTINEL for value in row):
                raise MarkerContractError(
                    f"unused task slot {task} is not exact-sentinel"
                )
            continue
        if any(value == SENTINEL for value in row):
            raise MarkerContractError(f"task_ticks[{task}] is not exact-fill")
        owner = owners[task]
        if not 0 <= owner < len(ctas):
            raise MarkerContractError(f"task_cta_z[{task}]={owner} is invalid")
        main = [_main(row, event) for event in range(len(MAIN_EVENTS))]
        for warp in range(COMPUTE_WARPS):
            _require_monotonic(
                [event[warp] for event in main],
                label=f"task_ticks[{task}] main W{warp}",
            )
            pair = [_pair(row, event)[warp] for event in range(len(PAIR_EVENTS))]
            _require_monotonic(pair, label=f"task_ticks[{task}] pair W{warp}")
            if pair[0] < main[0][warp] or pair[-1] > main[1][warp]:
                raise MarkerContractError(
                    f"task_ticks[{task}] pair track escapes FC1 envelope on W{warp}"
                )
        cache_ready = max(main[0])
        task_done = max(main[-1])
        if row[TASK_CLAIM_START] > cache_ready:
            raise MarkerContractError(f"task_ticks[{task}] claim starts after cache")
        by_cta[owner].append((cache_ready, task_done, task))

    for cta, records in by_cta.items():
        previous = max(_cta_warps(ctas[cta], CTA_LOOP_START))
        for cache_ready, task_done, task in sorted(records):
            if cache_ready < previous:
                raise MarkerContractError(
                    f"CTA {cta} task {task} cache overlaps previous collective task"
                )
            previous = task_done
        if previous > max(_cta_warps(ctas[cta], CTA_LOOP_EXIT)):
            raise MarkerContractError(f"CTA {cta} task completion exceeds loop exit")

    return {
        "schema": "exp008.marker-enabled-buffer-gate.v1",
        "task_tail": int(task_tail),
        "task_capacity": len(tasks),
        "grid_z": len(ctas),
        "gate_pass": True,
    }


def additive_rollup(
    task_ticks: Any,
    task_cta_z: Any,
    cta_ticks: Any,
    *,
    task_tail: int,
) -> dict[str, Any]:
    """Reduce validated events to a strictly closing collective ledger."""

    validate_probe_events(task_ticks, task_cta_z, cta_ticks, task_tail=task_tail)
    tasks = _matrix(task_ticks, columns=TASK_TICKS, name="task_ticks")
    ctas = _matrix(cta_ticks, columns=CTA_TICKS, name="cta_ticks")
    owners = _vector(task_cta_z, name="task_cta_z")
    phases = {
        "front_end_route_q0": 0,
        "claim_cache_transition": 0,
        "fc1_interleaved_activation_envelope": 0,
        "q1": 0,
        "combined_fc2_scatter": 0,
        "residual": 0,
    }
    by_cta: dict[int, list[int]] = {cta: [] for cta in range(len(ctas))}
    for task in range(task_tail):
        by_cta[owners[task]].append(task)

    entries: list[int] = []
    finals: list[int] = []
    cta_span_sum = 0
    for cta, cta_row in enumerate(ctas):
        entry = min(_cta_warps(cta_row, CTA_ENTRY))
        loop_start = max(_cta_warps(cta_row, CTA_LOOP_START))
        loop_exit = max(_cta_warps(cta_row, CTA_LOOP_EXIT))
        final = max(loop_exit, cta_row[CTA_W8_FINAL])
        entries.append(entry)
        finals.append(final)
        cta_span_sum += final - entry
        phases["front_end_route_q0"] += loop_start - entry
        previous = loop_start
        ordered = sorted(by_cta[cta], key=lambda task: max(_main(tasks[task], 0)))
        for task in ordered:
            row = tasks[task]
            cache = max(_main(row, 0))
            fc1_act = max(_main(row, 1))
            q1 = max(_main(row, 2))
            done = max(_main(row, 3))
            phases["claim_cache_transition"] += cache - previous
            phases["fc1_interleaved_activation_envelope"] += fc1_act - cache
            phases["q1"] += q1 - fc1_act
            phases["combined_fc2_scatter"] += done - q1
            previous = done
        phases["residual"] += final - previous

    global_start = min(entries)
    global_end = max(finals)
    denominator = len(ctas) * (global_end - global_start)
    launch_skew_early_finish = denominator - cta_span_sum
    if launch_skew_early_finish < 0:
        raise MarkerContractError("negative launch-skew/early-finish residual")
    phases["residual"] += launch_skew_early_finish
    phase_sum = sum(phases.values())
    if phase_sum != denominator:
        raise MarkerContractError(
            f"additive ledger does not close: {phase_sum} != {denominator}"
        )
    return {
        "schema": "exp008.additive-phase-rollup.v1",
        "denominator": {
            "kind": "SM-equivalent globaltimer wall",
            "duration_ns": denominator,
            "global_start_ns": global_start,
            "global_end_ns": global_end,
        },
        "phases": {
            name: {
                "duration_ns": duration,
                "share_pct": 100.0 * duration / denominator,
            }
            for name, duration in phases.items()
        },
        "closure": {
            "phase_sum_ns": phase_sum,
            "denominator_ns": denominator,
            "delta_ns": denominator - phase_sum,
            "gate_pass": phase_sum == denominator,
        },
    }
