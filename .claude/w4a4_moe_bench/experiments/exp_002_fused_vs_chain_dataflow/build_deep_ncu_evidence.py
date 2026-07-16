#!/usr/bin/env python3
"""Validate NSys-to-deep-NCU targets and build launch-local evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"

TARGETS = tuple(
    (m, arm, skip, phase, expected)
    for m in (256, 8192)
    for arm, skip, phase, expected in (
        ("cutedsl_bf16_fused", 1, "fused_main", "MoEDynamicKernel"),
        ("cutlass_bf16_chain", 3, "expand_quant", "expandInputRowsKernel"),
        ("cutlass_bf16_chain", 5, "fc1", "device_kernel"),
        ("cutlass_bf16_chain", 6, "activation_requant", "doActivationKernel"),
        ("cutlass_bf16_chain", 7, "fc2", "device_kernel"),
        ("cutlass_bf16_chain", 8, "finalize", "finalizeMoeRoutingKernel"),
    )
)

METRICS = {
    "ncu_duration_ns": "gpu__time_duration.sum",
    "dram_read_bytes": "dram__bytes_op_read.sum",
    "dram_write_bytes": "dram__bytes_op_write.sum",
    "l2_read_sectors": "lts__t_sectors_op_read.sum",
    "l2_write_sectors": "lts__t_sectors_op_write.sum",
    "lsu_global_load_footprint_bytes": ("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"),
    "lsu_global_store_footprint_bytes": (
        "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"
    ),
    "global_atomic_sectors": ("l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum"),
    "global_reduction_sectors": ("l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum"),
    "local_load_sectors": "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "local_store_sectors": "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "dynamic_warp_instructions": "sm__inst_executed.sum",
    "tensor_hmma_qmma_omma_instructions": (
        "sm__inst_executed_pipe_tensor_subpipe_hmma.sum"
    ),
    "fp4_to_fp32_tensor_ops": "sm__ops_path_tensor_src_fp4_dst_fp32.sum",
    "tensor_active_pct": (
        "sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_active"
    ),
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "eligible_warps_per_cycle": "smsp__warps_eligible.avg.per_cycle_active",
    "active_warps_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "ipc_active": "smsp__inst_executed.avg.per_cycle_active",
    "alu_instructions": "smsp__inst_executed_pipe_alu.sum",
    "fma_instructions": "smsp__inst_executed_pipe_fma.sum",
    "lsu_instructions": "smsp__inst_executed_pipe_lsu.sum",
    "uniform_instructions": "smsp__inst_executed_pipe_uniform.sum",
    "shared_load_wavefronts": ("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"),
    "shared_store_wavefronts": ("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"),
    "shared_load_bank_conflicts": (
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
    ),
    "shared_store_bank_conflicts": (
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
    ),
    "stall_barrier_pct": ("smsp__warp_issue_stalled_barrier_per_warp_active.pct"),
    "stall_long_scoreboard_pct": (
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
    ),
    "stall_short_scoreboard_pct": (
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct"
    ),
    "stall_wait_pct": "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "stall_not_selected_pct": (
        "smsp__warp_issue_stalled_not_selected_per_warp_active.pct"
    ),
    "stall_math_pipe_throttle_pct": (
        "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"
    ),
    "stall_mio_throttle_pct": (
        "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct"
    ),
    "stall_lg_throttle_pct": (
        "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct"
    ),
    "registers_per_thread": "launch__registers_per_thread",
    "allocated_registers_per_thread": "launch__registers_per_thread_allocated",
    "shared_mem_per_block_bytes": "launch__shared_mem_per_block",
    "dynamic_shared_mem_per_block_bytes": "launch__shared_mem_per_block_dynamic",
    "configured_stack_limit_bytes": "launch__stack_size",
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "register_occupancy_limit_blocks": "launch__occupancy_limit_registers",
    "shared_mem_occupancy_limit_blocks": "launch__occupancy_limit_shared_mem",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval_union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + end - start


def metric_map(
    row: dict[str, Any], path: Path
) -> tuple[dict[str, float], dict[str, str | None]]:
    by_name = {item["name"]: item for item in row.get("metrics", [])}
    values: dict[str, float] = {}
    units: dict[str, str | None] = {}
    for output_name, metric_name in METRICS.items():
        item = by_name.get(metric_name)
        if item is None or item.get("value") is None:
            raise ValueError(f"missing metric {metric_name} in {path}")
        value = item["value"]
        if not isinstance(value, (int, float)):
            raise ValueError(f"non-numeric metric {metric_name} in {path}")
        values[output_name] = float(value)
        units[output_name] = item.get("unit")

    values["dram_total_bytes"] = values["dram_read_bytes"] + values["dram_write_bytes"]
    values["l2_read_write_sector_bytes"] = (
        values["l2_read_sectors"] + values["l2_write_sectors"]
    ) * 32.0
    values["global_atomic_footprint_bytes"] = values["global_atomic_sectors"] * 32.0
    values["global_reduction_footprint_bytes"] = (
        values["global_reduction_sectors"] * 32.0
    )
    values["local_load_footprint_bytes"] = values["local_load_sectors"] * 32.0
    values["local_store_footprint_bytes"] = values["local_store_sectors"] * 32.0
    values["local_total_footprint_bytes"] = (
        values["local_load_footprint_bytes"] + values["local_store_footprint_bytes"]
    )
    values["shared_load_conflicts_per_wavefront"] = (
        values["shared_load_bank_conflicts"] / values["shared_load_wavefronts"]
        if values["shared_load_wavefronts"]
        else 0.0
    )
    values["shared_store_conflicts_per_wavefront"] = (
        values["shared_store_bank_conflicts"] / values["shared_store_wavefronts"]
        if values["shared_store_wavefronts"]
        else 0.0
    )
    return values, units


def collect(results: Path) -> dict[str, Any]:
    nsys_cache: dict[tuple[int, str], list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    selected_intervals: dict[tuple[int, str], list[tuple[int, int]]] = {}

    for m, arm, skip, phase, expected in TARGETS:
        case = results / "ncu" / f"m{m}" / arm / f"deep_launch_{skip}"
        report = case / "trace.ncu-rep"
        report_hash = sha256(report)
        recorded_hash = (case / "trace.ncu-rep.sha256").read_text().split()[0]
        if report_hash != recorded_hash:
            raise ValueError(f"NCU report hash mismatch: {report}")

        inspect = read_json(case / "veloq" / "inspect.json")
        inspect_rows = inspect.get("data", {}).get("rows", [])
        if len(inspect_rows) != 1 or inspect_rows[0].get("key") != "launch:0":
            raise ValueError(f"expected exactly one launch:0 in {case}")
        launch = inspect_rows[0]
        if expected not in launch["kernel_demangled"]:
            raise ValueError(f"unexpected captured kernel in {case}")
        for veloq_name in ("info", "summary", "launches", "disasm"):
            envelope = read_json(case / "veloq" / f"{veloq_name}.json")
            if "error" in envelope:
                raise ValueError(
                    f"VeloQ error in {case / 'veloq' / (veloq_name + '.json')}"
                )

        key = (m, arm)
        if key not in nsys_cache:
            nsys = read_json(
                results / "nsys" / f"m{m}" / arm / "veloq" / "kernels.json"
            )
            nsys_cache[key] = nsys.get("data", {}).get("rows", [])
        nsys_rows = nsys_cache[key]
        if skip >= len(nsys_rows):
            raise ValueError(f"NSys launch skip out of range for m={m} arm={arm}")
        nsys_row = nsys_rows[skip]
        if (
            launch["kernel_demangled"] != nsys_row["demangled_name"]
            or launch["grid_size"] != nsys_row["grid"]
            or launch["block_size"] != nsys_row["block"]
        ):
            raise ValueError(f"NSys/NCU dispatch mismatch for {case}")

        profile_manifest = read_json(case / "profile_manifest.json")
        evidence_identity = read_json(results / "evidence.identity.json")
        expected_identity = {
            "comparison_group_id": evidence_identity["comparison_group_id"],
            "rerun_id": evidence_identity["rerun_id"],
            "environment_lock_digest": evidence_identity["environment_lock_digest"],
            "protocol_lock_digest": evidence_identity["protocol_lock_digest"],
            "artifact_fingerprint_sha256": evidence_identity[
                "per_arm_artifact_fingerprint_sha256"
            ][arm],
        }
        for identity_key, identity_value in expected_identity.items():
            if profile_manifest.get(identity_key) != identity_value:
                raise ValueError(
                    f"deep profile identity drift at {identity_key}: {case}"
                )

        target_manifest = {
            "schema": "exp002.ncu-target-manifest.v1",
            "case": {"m": m},
            "arm": arm,
            "phase": phase,
            "original_launch_skip": skip,
            "expected_kernel_substring": expected,
            "kernel_demangled": launch["kernel_demangled"],
            "grid": launch["grid_size"],
            "block": launch["block_size"],
            "report_sha256": report_hash,
            "kernel_match": True,
            "nsys_row_id": nsys_row["row_id"],
            "ncu_row_id": launch["row_id"],
            **expected_identity,
        }
        (case / "target_manifest.json").write_text(
            json.dumps(target_manifest, indent=2, sort_keys=True) + "\n"
        )

        metrics, units = metric_map(launch, case / "veloq" / "inspect.json")
        selected_intervals.setdefault(key, []).append(
            (nsys_row["start_ns"], nsys_row["start_ns"] + nsys_row["duration_ns"])
        )
        rows.append(
            {
                "m": m,
                "arm": arm,
                "phase": phase,
                "relationship": "paired",
                "nsys_original_launch_skip": skip,
                "kernel_demangled": launch["kernel_demangled"],
                "kernel_function": launch["kernel_function"],
                "grid": launch["grid_size"],
                "block": launch["block_size"],
                "nsys_duration_ns": nsys_row["duration_ns"],
                "metrics": metrics,
                "metric_units": units,
                "provenance": {
                    "report": str(report.relative_to(results)),
                    "report_sha256": report_hash,
                    "target_manifest": str(
                        (case / "target_manifest.json").relative_to(results)
                    ),
                    "nsys_row_id": nsys_row["row_id"],
                    "ncu_row_id": launch["row_id"],
                    "capture": "kernel replay / cache-control all / one selected graph node",
                },
            }
        )

    coverage: list[dict[str, Any]] = []
    for (m, arm), intervals in selected_intervals.items():
        nsys_rows = nsys_cache[(m, arm)]
        all_union = interval_union_ns(
            (row["start_ns"], row["start_ns"] + row["duration_ns"]) for row in nsys_rows
        )
        selected_union = interval_union_ns(intervals)
        coverage.append(
            {
                "m": m,
                "arm": arm,
                "selected_active_union_ns": selected_union,
                "all_kernel_active_union_ns": all_union,
                "coverage_percent": selected_union / all_union * 100.0,
                "role": "paired coverage",
            }
        )

    return {
        "schema": "exp002.deep-ncu.v1",
        "scope": {
            "metrics": "launch-local only; ratios are not averaged across launches",
            "duration": "NCU duration is explanatory, not operator timing authority",
            "traffic": (
                "kernel replay traffic is not rolled up into operator totals; "
                "L2 read+write is explicitly not complete L2 total"
            ),
            "stack": (
                "launch__stack_size is the configured CUDA stack limit, not "
                "the cubin static frame size"
            ),
        },
        "targets": rows,
        "coverage": coverage,
    }


def write_flat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        item = {
            "m": row["m"],
            "arm": row["arm"],
            "phase": row["phase"],
            "relationship": row["relationship"],
            "nsys_original_launch_skip": row["nsys_original_launch_skip"],
            "kernel_function": row["kernel_function"],
            "grid": "x".join(map(str, row["grid"])),
            "block": "x".join(map(str, row["block"])),
            "nsys_duration_ns": row["nsys_duration_ns"],
            "report_sha256": row["provenance"]["report_sha256"],
        }
        item.update(row["metrics"])
        flattened.append(item)
    fields = list(flattened[0]) if flattened else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flattened)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    results = args.results.resolve()
    evidence = collect(results)
    output = results / "ncu" / "deep_launch_metrics.json"
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    write_flat_csv(results / "ncu" / "deep_launch_metrics.csv", evidence["targets"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
