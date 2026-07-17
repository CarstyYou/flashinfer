#!/usr/bin/env python3
"""Validate and reduce exp_004 full-population clock64 captures.

The reducer is GPU-free.  It revalidates every raw slot before writing the
large event CSV, keeps W4 overlap separate from additive consumer shares, and
refuses formal phase output when the binary/resource gate failed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from exp004_common import (
    CONSUMER_INTERVAL_INDEX,
    CONSUMER_WARPS,
    DECLARED_PHASE_START_SIDE_STORES,
    DEFAULT_RESULTS,
    EXPECTED_TASK_TAIL,
    MEASURED_REPLAYS,
    MEASUREMENT_CONTROL,
    OUTPUT_TILES,
    PRIMARY_PHASES,
    PROBE,
    SENTINEL,
    W4_INTERVAL_NAMES,
    consumer_tick_index,
    decode_probe_buffer,
    iter_probe_rows,
    read_json,
    validate_no_marker_buffer,
    w4_tick_index,
    write_csv,
    write_json,
)


EVENT_FIELDS = (
    "run_id",
    "source_sha256",
    "cubin_sha256",
    "sass_sha256",
    "gpu_uuid",
    "clock_state",
    "kernel",
    "graph_launch_key",
    "grid_id",
    "cta_z",
    "task_slot",
    "expert",
    "m_tile",
    "slice",
    "valid_rows",
    "warp_id",
    "role",
    "phase",
    "subphase",
    "output_tile",
    "start_tick",
    "end_tick",
    "timestamp_unit",
    "complete",
    "duplicate",
    "overflow",
    "provider_scope",
)


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("empty quantile")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((int(start), int(end)) for start, end in intervals)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if end <= start:
            raise ValueError("non-positive interval entered union")
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _union_ticks(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge_intervals(intervals))


def _intersection_ticks(
    left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]
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


def _load_pt(path: Path) -> Mapping[str, Any]:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise ValueError(f"expected tensor mapping: {path}")
    return value


def _plain_timing(path: Path) -> tuple[list[int], list[int]]:
    value = _load_pt(path)
    return (
        [int(item) for item in value["timing_ticks"].tolist()],
        [int(item) for item in value["task_cta_z"].tolist()],
    )


def _descriptors(path: Path) -> tuple[list[dict[str, int]], Mapping[str, Any]]:
    value = _load_pt(path)
    length = int(value["task_expert"].numel())
    rows = [
        {
            "expert": int(value["task_expert"][slot]),
            "m_tile": int(value["task_m_tile"][slot]),
            "slice": int(value["task_slice_begin"][slot]),
            "slice_count": int(value["task_slice_count"][slot]),
            "valid_rows": int(value["task_valid_rows"][slot]),
        }
        for slot in range(length)
    ]
    return rows, value


def _pair(ticks: Sequence[int], index: int) -> tuple[int, int]:
    return int(ticks[index]), int(ticks[index + 1])


def _consumer_phase_intervals(
    ticks: Sequence[int], *, task: int, warp: int, capacity: int, phase: str
) -> list[tuple[int, int]]:
    if phase in ("fc1_gate", "fc1_up", "swiglu_q1", "fc2_setup"):
        names = (phase,)
    elif phase == "fc2_gemm":
        names = tuple(f"fc2_gemm_{tile}" for tile in range(OUTPUT_TILES))
    elif phase == "fc2_epilogue_scatter":
        names = tuple(f"fc2_epilogue_scatter_{tile}" for tile in range(OUTPUT_TILES))
    else:
        raise ValueError(f"unknown consumer phase: {phase}")
    return [
        _pair(
            ticks,
            consumer_tick_index(task, warp, name, 0, capacity),
        )
        for name in names
    ]


def summarize_run(
    ticks: Sequence[int], *, task_tail: int, task_capacity: int
) -> dict[str, Any]:
    totals = {phase: 0 for phase in PRIMARY_PHASES}
    values = {phase: [] for phase in PRIMARY_PHASES}
    counts = {phase: 0 for phase in PRIMARY_PHASES}
    residual_values: list[int] = []
    denominator = 0
    overlap: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "task_count": 0,
            "producer_union_ticks": 0,
            "consumer_union_ticks": 0,
            "intersection_ticks": 0,
            "producer_covered_values": [],
        }
    )

    for task in range(task_tail):
        task_consumer: dict[str, list[tuple[int, int]]] = {
            phase: [] for phase in PRIMARY_PHASES
        }
        for warp in range(CONSUMER_WARPS):
            envelope = _pair(
                ticks,
                consumer_tick_index(
                    task,
                    warp,
                    CONSUMER_INTERVAL_INDEX["task_envelope"],
                    0,
                    task_capacity,
                ),
            )
            envelope_duration = envelope[1] - envelope[0]
            denominator += envelope_duration
            declared = 0
            for phase in PRIMARY_PHASES:
                intervals = _consumer_phase_intervals(
                    ticks,
                    task=task,
                    warp=warp,
                    capacity=task_capacity,
                    phase=phase,
                )
                duration = _union_ticks(intervals)
                totals[phase] += duration
                values[phase].append(duration)
                counts[phase] += len(intervals)
                declared += duration
                task_consumer[phase].extend(intervals)
            residual = envelope_duration - declared
            if residual < 0:
                raise ValueError("declared intervals exceed a consumer task envelope")
            residual_values.append(residual)

        for producer_index, producer_phase in enumerate(W4_INTERVAL_NAMES):
            producer = _pair(
                ticks,
                w4_tick_index(task, producer_index, 0, task_capacity),
            )
            producer_ticks = producer[1] - producer[0]
            for consumer_phase in PRIMARY_PHASES:
                consumer = task_consumer[consumer_phase]
                consumer_ticks = _union_ticks(consumer)
                intersection = _intersection_ticks((producer,), consumer)
                item = overlap[(producer_phase, consumer_phase)]
                item["task_count"] += 1
                item["producer_union_ticks"] += producer_ticks
                item["consumer_union_ticks"] += consumer_ticks
                item["intersection_ticks"] += intersection
                item["producer_covered_values"].append(
                    100.0 * intersection / producer_ticks
                )

    phase_share = {
        phase: 100.0 * totals[phase] / denominator for phase in PRIMARY_PHASES
    }
    residual_total = sum(residual_values)
    return {
        "denominator_ticks": denominator,
        "phase_totals": totals,
        "phase_values": values,
        "interval_counts": counts,
        "phase_share_pct": phase_share,
        "residual_total": residual_total,
        "residual_values": residual_values,
        "residual_pct": 100.0 * residual_total / denominator,
        "complete_warp_tasks": task_tail * CONSUMER_WARPS,
        "overlap": dict(overlap),
    }


def _single_hash(value: Any, *, label: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        unique = sorted({str(item) for item in value})
        if len(unique) == 1:
            return unique[0]
    raise ValueError(f"{label} is not a single exact identity: {value!r}")


def _binary_probe_identity(binary: Mapping[str, Any]) -> dict[str, str]:
    arm = binary["arms"][PROBE]
    identity = arm["identity"]
    return {
        "cubin_sha256": _single_hash(identity["cubin_sha256"], label="cubin"),
        "sass_sha256": _single_hash(identity["sass_sha256"], label="sass"),
    }


def _phase_root(results: Path, arm: str) -> Path:
    return results / "raw" / "phase_capture" / arm


def _capture_runs(results: Path, arm: str) -> list[dict[str, Any]]:
    root = _phase_root(results, arm)
    manifest = read_json(root / "manifest.json")
    if manifest.get("arm") != arm or len(manifest.get("runs", [])) != MEASURED_REPLAYS:
        raise ValueError(f"{arm}: expected exactly {MEASURED_REPLAYS} capture runs")
    runs = []
    for expected, item in enumerate(manifest["runs"]):
        run_id = f"run_{expected}"
        if item.get("run_id") != run_id:
            raise ValueError(f"{arm}: non-canonical replay order")
        run_root = root / run_id
        metadata = read_json(run_root / "metadata.json")
        runs.append(
            {
                "run_id": run_id,
                "root": run_root,
                "metadata": metadata,
            }
        )
    return runs


def _validate_control_runs(results: Path) -> tuple[list[dict[str, Any]], list[float]]:
    runs = _capture_runs(results, MEASUREMENT_CONTROL)
    elapsed = []
    for run in runs:
        ticks, cta = _plain_timing(run["root"] / "timing.pt")
        metadata = run["metadata"]
        workspace = read_json(run["root"] / "workspace.json")
        run_correctness = metadata["correctness"]
        if (
            not run_correctness["gate"]["gate_pass"]
            or not run_correctness["output_contract"]["gate_pass"]
            or not metadata["workspace_gate"]["gate_pass"]
            or not workspace["verification"]["gate_pass"]
        ):
            raise ValueError(
                f"measurement control semantic gate failed: {run['run_id']}"
            )
        gate = validate_no_marker_buffer(
            ticks, cta, task_capacity=int(workspace["task_capacity"])
        )
        if not gate["gate_pass"] or not metadata["timing_gate"]["gate_pass"]:
            raise ValueError(
                f"measurement control wrote timing events: {run['run_id']}"
            )
        elapsed.append(float(metadata["event_elapsed_us"]))
    return runs, elapsed


def _write_phase_events(
    path: Path,
    *,
    run_records: Sequence[Mapping[str, Any]],
    source_sha256: str,
    binary_identity: Mapping[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in run_records:
            metadata = record["metadata"]
            gpu = metadata["gpu_state_after"]
            runtime_gpu = metadata["runtime"]["gpu"]
            prefix = {
                "source_sha256": source_sha256,
                **binary_identity,
                "gpu_uuid": runtime_gpu["uuid"],
                "clock_state": json.dumps(
                    {
                        "graphics_clock_mhz": gpu.get("graphics_clock_mhz"),
                        "applications_graphics_clock_mhz": gpu.get(
                            "applications_graphics_clock_mhz"
                        ),
                        "max_graphics_clock_mhz": gpu.get("max_graphics_clock_mhz"),
                        "power_draw_w": gpu.get("power_draw_w"),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "kernel": "MoEDynamicKernel",
                "graph_launch_key": "single_node_final_cuda_graph_replay",
                "grid_id": "1x1x110",
                "duplicate": False,
                "overflow": False,
                "provider_scope": "full_population_clock64",
            }
            for row in iter_probe_rows(
                record["ticks"],
                record["cta"],
                run_id=record["run_id"],
                task_tail=record["task_tail"],
                task_capacity=record["task_capacity"],
                task_descriptors=record["descriptors"],
            ):
                combined = {**prefix, **row}
                writer.writerow({field: combined[field] for field in EVENT_FIELDS})
    os.replace(temporary, path)


def analyze(results: Path) -> int:
    binary_path = results / "raw" / "binary_identity.json"
    correctness_path = results / "raw" / "correctness.json"
    calibration_path = results / "raw" / "calibration" / "manifest.json"
    binary = read_json(binary_path)
    correctness = read_json(correctness_path)
    calibration = read_json(calibration_path)
    binary_probe = _binary_probe_identity(binary)
    binary_gate = bool(binary["gates"]["formal_gate_pass"])
    correctness_gate = bool(correctness["gate_pass"])

    control_runs, control_elapsed = _validate_control_runs(results)
    probe_runs = _capture_runs(results, PROBE)
    run_records: list[dict[str, Any]] = []
    task_population: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    probe_elapsed: list[float] = []
    source_hashes: set[str] = set()
    gpu_uuids: set[str] = set()
    event_gates = []

    for run in probe_runs:
        ticks, cta = _plain_timing(run["root"] / "timing.pt")
        descriptors, workspace_tensors = _descriptors(run["root"] / "workspace.pt")
        workspace = read_json(run["root"] / "workspace.json")
        task_tail = int(workspace["scalars"]["task_tail"])
        capacity = int(workspace["task_capacity"])
        _, gate = decode_probe_buffer(
            ticks,
            cta,
            run_id=run["run_id"],
            task_tail=task_tail,
            task_capacity=capacity,
            task_descriptors=descriptors,
            emit_rows=False,
        )
        event_gates.append(gate)
        run_correctness = run["metadata"]["correctness"]
        semantic_gate = (
            run_correctness["gate"]["gate_pass"]
            and run_correctness["output_contract"]["gate_pass"]
            and run["metadata"]["workspace_gate"]["gate_pass"]
            and run["metadata"]["timing_gate"]["gate_pass"]
            and workspace["verification"]["gate_pass"]
        )
        if not gate["gate_pass"] or not semantic_gate:
            raise ValueError(f"probe event gate failed: {run['run_id']}")
        source_hashes.add(
            run["metadata"]["runtime"]["source"]["overlays"]["kernel"]["sha256"]
        )
        gpu_uuids.add(run["metadata"]["runtime"]["gpu"]["uuid"])
        probe_elapsed.append(float(run["metadata"]["event_elapsed_us"]))
        summary = summarize_run(ticks, task_tail=task_tail, task_capacity=capacity)
        summary["run_id"] = run["run_id"]
        run_summaries.append(summary)
        run_records.append(
            {
                **run,
                "ticks": ticks,
                "cta": cta,
                "descriptors": descriptors,
                "task_tail": task_tail,
                "task_capacity": capacity,
            }
        )
        for slot, descriptor in enumerate(descriptors):
            task_population.append(
                {
                    "run_id": run["run_id"],
                    "task_slot": slot,
                    **descriptor,
                    "expected": True,
                    "observed": cta[slot] != SENTINEL,
                    "cta_z": cta[slot],
                    "descriptor_order_sha256": workspace["verification"][
                        "task_descriptor_order_sha256"
                    ],
                }
            )

    if len(source_hashes) != 1 or len(gpu_uuids) != 1:
        raise ValueError("probe replays crossed source or GPU identities")
    source_sha256 = next(iter(source_hashes))
    if source_sha256 != binary["arms"][PROBE]["identity"]["kernel_overlay_sha256"]:
        raise ValueError("probe event source does not match binary overlay identity")
    if next(iter(gpu_uuids)) != binary["gpu_identity"][PROBE]["uuid"]:
        raise ValueError("probe events do not match binary GPU identity")
    write_csv(results / "raw" / "task_population.csv", task_population)
    _write_phase_events(
        results / "raw" / "phase_events.csv",
        run_records=run_records,
        source_sha256=source_sha256,
        binary_identity=binary_probe,
    )

    share_by_phase = {
        phase: [run["phase_share_pct"][phase] for run in run_summaries]
        for phase in PRIMARY_PHASES
    }
    share_by_phase["residual_task_control"] = [
        run["residual_pct"] for run in run_summaries
    ]
    spread = {
        phase: max(values) - min(values) for phase, values in share_by_phase.items()
    }
    stability_gate = max(spread.values()) <= 1.0

    control_median = statistics.median(control_elapsed)
    probe_median = statistics.median(probe_elapsed)
    control_spread_pct = (
        100.0 * (max(control_elapsed) - min(control_elapsed)) / control_median
    )
    latency_drift_pct = 100.0 * abs(probe_median - control_median) / control_median
    overhead_gate = (
        latency_drift_pct <= 1.0 and latency_drift_pct <= 5.0 * control_spread_pct
    )
    application_clocks = {
        str(
            run["metadata"]["gpu_state_after"].get(
                "applications_graphics_clock_mhz", ""
            )
        )
        for run in (*control_runs, *probe_runs)
    }
    application_clocks -= {"", "N/A", "[N/A]", "n/a"}
    current_clocks = {
        str(run["metadata"]["gpu_state_after"].get("graphics_clock_mhz", ""))
        for run in (*control_runs, *probe_runs)
    }
    current_clocks -= {"", "N/A", "[N/A]", "n/a"}
    clock_gate = len(application_clocks) == 1 or len(current_clocks) == 1

    calibration_p95 = float(calibration["aggregate"]["delta_tick_p95"])
    declared_start_side_store_count = sum(
        run["complete_warp_tasks"] * DECLARED_PHASE_START_SIDE_STORES
        for run in run_summaries
    )
    denominator_all = sum(run["denominator_ticks"] for run in run_summaries)
    calibration_upper_pct = (
        100.0 * calibration_p95 * declared_start_side_store_count / denominator_all
    )
    calibration_gate = calibration_upper_pct <= 1.0

    gates = {
        "binary_resource_spill_semantic": binary_gate,
        "correctness": correctness_gate,
        "event_contract_all_runs": all(gate["gate_pass"] for gate in event_gates),
        "full_population_coverage": all(
            summary["complete_warp_tasks"] == record["task_tail"] * CONSUMER_WARPS
            and record["task_tail"] == EXPECTED_TASK_TAIL
            for summary, record in zip(run_summaries, run_records, strict=True)
        ),
        "run_share_absolute_drift_le_1pct": stability_gate,
        "runtime_overhead": overhead_gate,
        "stable_clock_state": clock_gate,
        "calibration_upper_bound": calibration_gate,
    }
    formal_gate = all(gates.values())
    diagnostic_allowed = all(
        gates[name]
        for name in (
            "binary_resource_spill_semantic",
            "correctness",
            "event_contract_all_runs",
            "full_population_coverage",
        )
    )
    gate_payload = {
        "schema": "exp004.phase-analysis-gates.v1",
        "gates": gates,
        "formal_gate_pass": formal_gate,
        "diagnostic_share_allowed": diagnostic_allowed,
        "run_share_spread_percentage_points": spread,
        "latency": {
            "measurement_control_us": control_elapsed,
            "probe_candidate_us": probe_elapsed,
            "measurement_control_median_us": control_median,
            "probe_candidate_median_us": probe_median,
            "control_repeat_spread_pct": control_spread_pct,
            "candidate_control_median_drift_pct": latency_drift_pct,
            "applications_graphics_clock_mhz": sorted(application_clocks),
            "observed_graphics_clock_mhz": sorted(current_clocks),
        },
        "calibration": {
            "delta_tick_p95": calibration_p95,
            "declared_phase_start_side_store_count": declared_start_side_store_count,
            "consumer_denominator_ticks": denominator_all,
            "total_p95_upper_bound_pct": calibration_upper_pct,
        },
    }
    write_json(results / "derived" / "analysis_gates.json", gate_payload)

    if not diagnostic_allowed:
        return 2

    verdict = (
        "canonical"
        if formal_gate
        else "instrumented_diagnostic_not_production_representative"
    )
    phase_rows = []
    for phase in (*PRIMARY_PHASES, "residual_task_control"):
        if phase == "residual_task_control":
            totals = [run["residual_total"] for run in run_summaries]
            values = [
                value for run in run_summaries for value in run["residual_values"]
            ]
            intervals = sum(run["complete_warp_tasks"] for run in run_summaries)
        else:
            totals = [run["phase_totals"][phase] for run in run_summaries]
            values = [
                value for run in run_summaries for value in run["phase_values"][phase]
            ]
            intervals = sum(run["interval_counts"][phase] for run in run_summaries)
        total = sum(totals)
        phase_rows.append(
            {
                "phase": phase,
                "role": "mma_consumer",
                "runs": len(run_summaries),
                "complete_warp_tasks": sum(
                    run["complete_warp_tasks"] for run in run_summaries
                ),
                "intervals": intervals,
                "sum_ticks": total,
                "denominator_ticks": denominator_all,
                "share_pct": 100.0 * total / denominator_all,
                "per_warp_task_p50": statistics.median(values),
                "per_warp_task_p95": _quantile(values, 0.95),
                "run_spread_pct": spread[phase],
                "coverage_pct": 100.0,
                "residual_pct": 100.0
                * sum(run["residual_total"] for run in run_summaries)
                / denominator_all,
                "verdict": verdict if stability_gate else "distribution_only_unstable",
            }
        )
    write_csv(results / "derived" / "mma_phase_share.csv", phase_rows)

    overlap_accumulator: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "task_count": 0,
            "producer_union_ticks": 0,
            "consumer_union_ticks": 0,
            "intersection_ticks": 0,
            "producer_covered_values": [],
        }
    )
    for run in run_summaries:
        for key, value in run["overlap"].items():
            target = overlap_accumulator[key]
            for field in (
                "task_count",
                "producer_union_ticks",
                "consumer_union_ticks",
                "intersection_ticks",
            ):
                target[field] += value[field]
            target["producer_covered_values"].extend(value["producer_covered_values"])
    overlap_rows = []
    for (producer, consumer), value in sorted(overlap_accumulator.items()):
        producer_ticks = value["producer_union_ticks"]
        consumer_ticks = value["consumer_union_ticks"]
        intersection = value["intersection_ticks"]
        overlap_rows.append(
            {
                "producer_phase": producer,
                "consumer_phase": consumer,
                "task_count": value["task_count"],
                "producer_union_ticks": producer_ticks,
                "consumer_union_ticks": consumer_ticks,
                "intersection_ticks": intersection,
                "producer_covered_pct": 100.0 * intersection / producer_ticks,
                "consumer_covered_pct": 100.0 * intersection / consumer_ticks,
                "p50": statistics.median(value["producer_covered_values"]),
                "p95": _quantile(value["producer_covered_values"], 0.95),
                "scope": "same_cta_task_clock64_interval_union",
            }
        )
    write_csv(results / "derived" / "w4_overlap.csv", overlap_rows)
    return 0 if formal_gate else 3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return analyze(args.results.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
