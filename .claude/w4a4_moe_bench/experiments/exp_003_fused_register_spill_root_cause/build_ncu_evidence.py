#!/usr/bin/env python3
"""Compact targeted NCU CSVs into exp_003 spill evidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from exp003_common import (
    ALL_ARMS,
    BASELINE_LOCAL_SECTORS_PER_DIRECTION,
    DEFAULT_RESULTS,
    EXPECTED_FP4_TENSOR_OPS,
    EXPECTED_BLOCK,
    EXPECTED_GRID,
    EXPECTED_TAIL_SECTOR_DELTA,
    EXPECTED_TASK_COUNT,
    EXPECTED_TENSOR_INSTRUCTIONS,
    file_sha256,
    read_json,
    write_csv,
    write_json,
)


METRICS = {
    "duration_ns": "gpu__time_duration.sum",
    "local_load_sectors": "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "local_store_sectors": "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "executed_local_load_instructions": "sm__sass_inst_executed_op_local_ld.sum",
    "executed_local_store_instructions": "sm__sass_inst_executed_op_local_st.sum",
    "tensor_instructions": "sm__inst_executed_pipe_tensor_subpipe_hmma.sum",
    "fp4_tensor_ops": "sm__ops_path_tensor_src_fp4_dst_fp32.sum",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "eligible_warps_per_cycle": "smsp__warps_eligible.avg.per_cycle_active",
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "tc_subpipe_active_pct": "sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "stall_long_scoreboard_pct": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "stall_short_scoreboard_pct": "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "stall_wait_pct": "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "stall_barrier_pct": "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "registers_per_thread": "launch__registers_per_thread",
    "shared_bytes_per_cta": "launch__shared_mem_per_block",
    "configured_stack_limit_bytes": "launch__stack_size",
}

UNIT_SCALE = {
    "": 1.0,
    "%": 1.0,
    "byte": 1.0,
    "Kbyte": 1e3,
    "Mbyte": 1e6,
    "Gbyte": 1e9,
    "sector": 1.0,
    "inst": 1.0,
    "warp": 1.0,
    "register/thread": 1.0,
    "byte/block": 1.0,
    "ns": 1.0,
    "us": 1e3,
    "ms": 1e6,
    "s": 1e9,
}


def parse_native_csv(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    header_index = next(
        (index for index, row in enumerate(rows) if row and row[0] == "ID"), None
    )
    if header_index is None or len(rows) < header_index + 3:
        raise ValueError(f"cannot find NCU header/unit/value rows: {path}")
    header, units, values = rows[header_index : header_index + 3]
    if not (len(header) == len(units) == len(values)):
        raise ValueError(f"NCU CSV width mismatch: {path}")
    by_name = dict(zip(header, values, strict=True))
    by_unit = dict(zip(header, units, strict=True))
    if by_name.get("Kernel Name") not in {
        "range",
        None,
    } and "MoEDynamicKernel" not in by_name.get("Kernel Name", ""):
        raise ValueError(f"CSV does not target the fused main kernel/range: {path}")
    parsed: dict[str, float] = {}
    raw_units: dict[str, str] = {}
    for output_name, metric_id in METRICS.items():
        if not by_name.get(metric_id):
            raise ValueError(f"missing NCU metric {metric_id}: {path}")
        unit = by_unit[metric_id]
        if unit not in UNIT_SCALE:
            raise ValueError(f"unsupported NCU unit {unit!r}: {metric_id}")
        parsed[output_name] = (
            float(by_name[metric_id].replace(",", "")) * UNIT_SCALE[unit]
        )
        raw_units[output_name] = unit
    parsed["local_load_footprint_bytes"] = parsed["local_load_sectors"] * 32.0
    parsed["local_store_footprint_bytes"] = parsed["local_store_sectors"] * 32.0
    return parsed, raw_units


def parse_arm_path(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        arm, separator, raw = value.partition("=")
        if not separator or arm not in ALL_ARMS:
            raise ValueError(f"expected ARM=PATH, got {value!r}")
        parsed[arm] = Path(raw).resolve()
    return parsed


def validate_work_identity(
    baseline: Mapping[str, float], arm: Mapping[str, float]
) -> dict[str, bool]:
    return {
        "tensor_instructions_expected": round(arm["tensor_instructions"])
        == EXPECTED_TENSOR_INSTRUCTIONS,
        "fp4_tensor_ops_expected": round(arm["fp4_tensor_ops"])
        == EXPECTED_FP4_TENSOR_OPS,
        "tensor_instructions_equal_baseline": round(arm["tensor_instructions"])
        == round(baseline["tensor_instructions"]),
        "fp4_tensor_ops_equal_baseline": round(arm["fp4_tensor_ops"])
        == round(baseline["fp4_tensor_ops"]),
    }


def build_evidence(
    metrics: Mapping[str, Mapping[str, float]],
    launches: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if "baseline" not in metrics:
        raise ValueError("baseline NCU metrics are required")
    baseline = metrics["baseline"]
    baseline_gate = {
        "local_load_sectors": round(baseline["local_load_sectors"])
        == BASELINE_LOCAL_SECTORS_PER_DIRECTION,
        "local_store_sectors": round(baseline["local_store_sectors"])
        == BASELINE_LOCAL_SECTORS_PER_DIRECTION,
        **validate_work_identity(baseline, baseline),
    }
    if launches:
        base_launch = launches.get("baseline", {})
        baseline_gate.update(
            {
                "grid_expected": base_launch.get("grid") == list(EXPECTED_GRID),
                "block_expected": base_launch.get("block") == list(EXPECTED_BLOCK),
                "single_fused_main_launch": base_launch.get("launch_count") == 1
                and "MoEDynamicKernel" in str(base_launch.get("kernel", "")),
            }
        )
    deltas: dict[str, Any] = {}
    for arm, values in metrics.items():
        if arm == "baseline":
            continue
        load_delta = round(
            baseline["local_load_sectors"] - values["local_load_sectors"]
        )
        store_delta = round(
            baseline["local_store_sectors"] - values["local_store_sectors"]
        )
        work = validate_work_identity(baseline, values)
        if launches:
            work.update(
                {
                    "grid_equal_baseline": launches.get(arm, {}).get("grid")
                    == launches.get("baseline", {}).get("grid"),
                    "block_equal_baseline": launches.get(arm, {}).get("block")
                    == launches.get("baseline", {}).get("block"),
                    "single_fused_main_launch": launches.get(arm, {}).get(
                        "launch_count"
                    )
                    == 1
                    and "MoEDynamicKernel"
                    in str(launches.get(arm, {}).get("kernel", "")),
                }
            )
        deltas[arm] = {
            "local_load_sector_reduction": load_delta,
            "local_store_sector_reduction": store_delta,
            "expected_14_word_sector_reduction": EXPECTED_TAIL_SECTOR_DELTA,
            "dynamic_14_word_closure_pass": (
                load_delta == EXPECTED_TAIL_SECTOR_DELTA
                and store_delta == EXPECTED_TAIL_SECTOR_DELTA
            ),
            "executed_local_load_instruction_reduction": round(
                baseline["executed_local_load_instructions"]
                - values["executed_local_load_instructions"]
            ),
            "executed_local_store_instruction_reduction": round(
                baseline["executed_local_store_instructions"]
                - values["executed_local_store_instructions"]
            ),
            "work_identity_checks": work,
            "work_identity_pass": all(work.values()),
        }
    return {
        "schema": "exp003.spill-root-cause.ncu-spill-evidence.v1",
        "task_count_contract": EXPECTED_TASK_COUNT,
        "baseline_reproduction": baseline_gate,
        "baseline_reproduction_pass": all(baseline_gate.values()),
        "arms": metrics,
        "launches": dict(launches or {}),
        "deltas": deltas,
        "units": {
            "local_sectors": "32-byte local-address-space sectors; not DRAM bytes",
            "executed_local_instructions": (
                "warp instructions; classified as spill/refill only after static "
                "SASS proves the target has no other local operations"
            ),
            "stall": "percent of active-warp cycles",
            "utilization": "percent of active cycles where metric ID says pct_of_peak_sustained_active",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--native-csv", action="append", required=True, help="ARM=ncu_native_raw.csv"
    )
    parser.add_argument(
        "--cubin", action="append", default=[], help="ARM=benchmark_and_profile_cubin"
    )
    parser.add_argument(
        "--launch-json",
        action="append",
        default=[],
        help="ARM=veloq_ncu_launches.json",
    )
    parser.add_argument(
        "--trace-rep", action="append", default=[], help="ARM=trace.ncu-rep"
    )
    parser.add_argument(
        "--command-text", action="append", default=[], help="ARM=ncu_command.txt"
    )
    parser.add_argument(
        "--profile-target",
        action="append",
        default=[],
        help="ARM=profile_target.json",
    )
    args = parser.parse_args(argv)
    csv_paths = parse_arm_path(args.native_csv)
    cubin_paths = parse_arm_path(args.cubin)
    launch_paths = parse_arm_path(args.launch_json)
    trace_paths = parse_arm_path(args.trace_rep)
    command_paths = parse_arm_path(args.command_text)
    profile_paths = parse_arm_path(args.profile_target)
    metrics: dict[str, dict[str, float]] = {}
    launches: dict[str, dict[str, Any]] = {}
    input_identity: dict[str, Any] = {}
    units: dict[str, str] | None = None
    for arm, path in csv_paths.items():
        values, current_units = parse_native_csv(path)
        if units is not None and current_units != units:
            raise RuntimeError("NCU raw-unit identity differs across arms")
        units = current_units
        metrics[arm] = values
        cubin = cubin_paths.get(arm)
        launch_path = launch_paths.get(arm)
        trace_path = trace_paths.get(arm)
        command_path = command_paths.get(arm)
        profile_path = profile_paths.get(arm)
        if launch_path:
            launch_payload = read_json(launch_path)
            launch_rows = launch_payload.get("data", {}).get("rows", [])
            if len(launch_rows) != 1:
                raise RuntimeError(f"{arm}: expected one VeloQ NCU launch row")
            launch = launch_rows[0]
            launches[arm] = {
                "launch_count": launch_payload.get("data", {}).get("count"),
                "row_id": launch.get("row_id"),
                "kernel": launch.get("kernel_demangled"),
                "grid": launch.get("grid_size"),
                "block": launch.get("block_size"),
                "context_id": launch.get("context_id"),
                "stream_id": launch.get("stream_id"),
            }
        preparation_path = args.results.resolve() / "arms" / arm / "preparation.json"
        preparation = read_json(preparation_path) if preparation_path.is_file() else {}
        if profile_path:
            profile = read_json(profile_path)
            if profile.get("arm") != arm or profile.get("status") != "complete":
                raise RuntimeError(f"{arm}: invalid profile-target identity")
            if profile.get("jit_artifact_set_sha256") != preparation.get(
                "jit_artifact_set_sha256"
            ):
                raise RuntimeError(f"{arm}: profile JIT artifact identity drift")
        if trace_path and trace_path.parent != path.parent:
            raise RuntimeError(f"{arm}: trace and native CSV are not canonical siblings")
        if command_path and command_path.parent != path.parent:
            raise RuntimeError(f"{arm}: NCU command and native CSV are not siblings")
        cubin_sha = file_sha256(cubin) if cubin else None
        if cubin_sha and cubin_sha not in {
            item.get("sha256") for item in preparation.get("jit_artifacts", [])
        }:
            raise RuntimeError(
                f"{arm}: NCU cubin is absent from benchmark JIT artifact lock"
            )
        input_identity[arm] = {
            "native_csv": str(path),
            "native_csv_sha256": file_sha256(path),
            "cubin": str(cubin) if cubin else None,
            "cubin_sha256": cubin_sha,
            "benchmark_cubin_identity_gate": cubin is not None,
            "launch_json": str(launch_path) if launch_path else None,
            "launch_json_sha256": file_sha256(launch_path)
            if launch_path
            else None,
            "trace_rep": str(trace_path) if trace_path else None,
            "trace_rep_sha256": file_sha256(trace_path) if trace_path else None,
            "command_text": str(command_path) if command_path else None,
            "command_text_sha256": file_sha256(command_path)
            if command_path
            else None,
            "profile_target": str(profile_path) if profile_path else None,
            "profile_target_sha256": file_sha256(profile_path)
            if profile_path
            else None,
            "profile_jit_artifact_identity_gate": profile_path is not None,
        }
    if launch_paths and set(launch_paths) != set(csv_paths):
        raise RuntimeError("VeloQ launch identity must be supplied for every NCU arm")
    for label, paths in (
        ("trace", trace_paths),
        ("command", command_paths),
        ("profile target", profile_paths),
    ):
        if paths and set(paths) != set(csv_paths):
            raise RuntimeError(f"{label} identity must be supplied for every NCU arm")
    payload = build_evidence(metrics, launches or None)
    payload["inputs"] = input_identity
    payload["raw_units"] = units
    results = args.results.resolve()
    output_dir = results / "ncu"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"arm": arm, **values} for arm, values in metrics.items()]
    write_csv(output_dir / "spill_metrics.csv", rows)
    write_json(output_dir / "spill_evidence.json", payload)
    return 0 if payload["baseline_reproduction_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
