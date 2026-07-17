#!/usr/bin/env python3
"""Validate identity-locked exp_005 NCU captures and emit compact evidence."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from exp005_common import (
    ALL_ARMS,
    BASELINE,
    CANDIDATE,
    CANONICAL_FIXTURE,
    DEFAULT_RESULTS,
    EXPECTED_GRID,
    canonical_sha256,
    expected_block,
    file_sha256,
    read_json,
    write_csv,
    write_json,
)


# Every ID below is present in the SM120 / Nsight Compute 26.05 metric set used
# by the canonical exp_002 deep capture.  The three additive work counters were
# also exercised by its custom operator-ledger capture.
METRICS = {
    "duration_ns": "gpu__time_duration.sum",
    "local_load_footprint_bytes": "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
    "local_store_footprint_bytes": "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
    "dynamic_spill_refill_instructions": (
        "sass__inst_executed_register_spilling_op_read"
    ),
    "dynamic_spill_store_instructions": (
        "sass__inst_executed_register_spilling_op_write"
    ),
    "dynamic_spill_refill_bytes": (
        "sass__inst_executed_register_spilling_mem_local_op_read"
    ),
    "dynamic_spill_store_bytes": (
        "sass__inst_executed_register_spilling_mem_local_op_write"
    ),
    "executed_warp_instructions": "sm__inst_executed.sum",
    "tensor_instructions": "sm__inst_executed_pipe_tensor_subpipe_hmma.sum",
    "fp4_tensor_ops": "sm__ops_path_tensor_src_fp4_dst_fp32.sum",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "eligible_warps_per_cycle": "smsp__warps_eligible.avg.per_cycle_active",
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "tc_subpipe_active_pct": (
        "sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_active"
    ),
    "ipc_active": "sm__inst_executed.avg.per_cycle_active",
    "alu_active_pct": ("sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active"),
    "fma_active_pct": ("sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active"),
    "lsu_active_pct": ("sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active"),
    "uniform_active_pct": (
        "sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_active"
    ),
    "warp_latency_per_issued_instruction": (
        "smsp__average_warp_latency_per_inst_issued.ratio"
    ),
    "stall_wait_ratio": (
        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
    ),
    "stall_long_scoreboard_ratio": (
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
    ),
    "stall_short_scoreboard_ratio": (
        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"
    ),
    "stall_barrier_ratio": (
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"
    ),
    "stall_not_selected_ratio": (
        "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"
    ),
    "stall_math_pipe_throttle_ratio": (
        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"
    ),
    "stall_mio_throttle_ratio": (
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio"
    ),
    "stall_lg_throttle_ratio": (
        "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio"
    ),
    "registers_per_thread": "launch__registers_per_thread",
    "allocated_registers_per_thread": "launch__registers_per_thread_allocated",
    "shared_mem_per_block_bytes": "launch__shared_mem_per_block",
    "dynamic_shared_mem_per_block_bytes": "launch__shared_mem_per_block_dynamic",
    # Runtime configured stack limit.  It is not the cubin static frame.
    "configured_stack_limit_bytes": "launch__stack_size",
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "register_occupancy_limit_blocks": "launch__occupancy_limit_registers",
    "shared_mem_occupancy_limit_blocks": "launch__occupancy_limit_shared_mem",
}

STALL_NAMES = (
    "wait",
    "long_scoreboard",
    "short_scoreboard",
    "barrier",
    "not_selected",
    "math_pipe_throttle",
    "mio_throttle",
    "lg_throttle",
)

REQUIRED_SECTION_IDS = (
    "SpeedOfLight",
    "ComputeWorkloadAnalysis",
    "MemoryWorkloadAnalysis",
    "Occupancy",
    "SchedulerStats",
    "WarpStateStats",
    "LaunchStats",
    "InstructionStats",
    "SourceCounters",
)

SPILL_METRIC_IDS = (
    METRICS["dynamic_spill_refill_instructions"],
    METRICS["dynamic_spill_store_instructions"],
    METRICS["dynamic_spill_refill_bytes"],
    METRICS["dynamic_spill_store_bytes"],
)

CUSTOM_METRIC_IDS = tuple(
    metric for metric in METRICS.values() if metric not in SPILL_METRIC_IDS
)

UNIT_SCALE = {
    "": 1.0,
    "%": 1.0,
    "byte": 1.0,
    "Kbyte": 1e3,
    "Mbyte": 1e6,
    "Gbyte": 1e9,
    "inst": 1.0,
    "inst/cycle": 1.0,
    "warp": 1.0,
    "cycle": 1.0,
    "register/thread": 1.0,
    "byte/block": 1.0,
    "block": 1.0,
    "ns": 1.0,
    "us": 1e3,
    "ms": 1e6,
    "s": 1e9,
}

EXPECTED_UNITS = {
    "duration_ns": {"ns", "us", "ms", "s"},
    "local_load_footprint_bytes": {"byte", "Kbyte", "Mbyte", "Gbyte"},
    "local_store_footprint_bytes": {"byte", "Kbyte", "Mbyte", "Gbyte"},
    "dynamic_spill_refill_instructions": {"inst"},
    "dynamic_spill_store_instructions": {"inst"},
    "dynamic_spill_refill_bytes": {"byte", "Kbyte", "Mbyte", "Gbyte"},
    "dynamic_spill_store_bytes": {"byte", "Kbyte", "Mbyte", "Gbyte"},
    "executed_warp_instructions": {"inst"},
    "tensor_instructions": {"inst"},
    "fp4_tensor_ops": {""},
    "achieved_occupancy_pct": {"%"},
    "eligible_warps_per_cycle": {"warp"},
    "issue_active_pct": {"%"},
    "tc_subpipe_active_pct": {"%"},
    "ipc_active": {"inst/cycle"},
    "alu_active_pct": {"%"},
    "fma_active_pct": {"%"},
    "lsu_active_pct": {"%"},
    "uniform_active_pct": {"%"},
    "warp_latency_per_issued_instruction": {"cycle"},
    **{f"stall_{name}_ratio": {"inst"} for name in STALL_NAMES},
    "registers_per_thread": {"register/thread"},
    "allocated_registers_per_thread": {"register/thread"},
    "shared_mem_per_block_bytes": {"byte/block"},
    "dynamic_shared_mem_per_block_bytes": {"byte/block"},
    "configured_stack_limit_bytes": {""},
    "waves_per_sm": {""},
    "register_occupancy_limit_blocks": {"block"},
    "shared_mem_occupancy_limit_blocks": {"block"},
}


def _parse_number(raw: str, *, field: str) -> float:
    try:
        value = float(raw.replace(",", ""))
    except ValueError as error:
        raise ValueError(f"non-numeric NCU value for {field}: {raw!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite NCU value for {field}: {raw!r}")
    return value


def _parse_integer(raw: str, *, field: str) -> int:
    value = _parse_number(raw, field=field)
    if not value.is_integer():
        raise ValueError(f"non-integral NCU launch identity for {field}: {raw!r}")
    return int(value)


def parse_native_csv(
    path: Path,
) -> tuple[dict[str, float], dict[str, str], dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    header_indices = [index for index, row in enumerate(rows) if row and row[0] == "ID"]
    if len(header_indices) != 1:
        raise ValueError(
            f"expected one NCU header row, got {len(header_indices)}: {path}"
        )
    header_index = header_indices[0]
    if len(rows) < header_index + 3:
        raise ValueError(f"missing NCU unit/value rows: {path}")
    header, units = rows[header_index : header_index + 2]
    if len(header) != len(set(header)):
        raise ValueError(f"duplicate NCU CSV columns: {path}")
    value_rows = [row for row in rows[header_index + 2 :] if any(row)]
    if len(value_rows) != 1:
        raise ValueError(
            f"expected exactly one NCU launch row, got {len(value_rows)}: {path}"
        )
    values = value_rows[0]
    if not (len(header) == len(units) == len(values)):
        raise ValueError(f"NCU CSV width mismatch: {path}")
    by_name = dict(zip(header, values, strict=True))
    by_unit = dict(zip(header, units, strict=True))

    parsed: dict[str, float] = {}
    raw_units: dict[str, str] = {}
    for output_name, metric_id in METRICS.items():
        raw = by_name.get(metric_id)
        if raw is None or raw in ("", "n/a"):
            raise ValueError(f"missing NCU metric {metric_id}: {path}")
        unit = by_unit[metric_id]
        if unit not in EXPECTED_UNITS[output_name]:
            raise ValueError(
                f"NCU unit drift for {metric_id}: {unit!r}, expected "
                f"{sorted(EXPECTED_UNITS[output_name])}"
            )
        parsed[output_name] = _parse_number(raw, field=metric_id) * UNIT_SCALE[unit]
        raw_units[output_name] = unit
    parsed["local_total_footprint_bytes"] = (
        parsed["local_load_footprint_bytes"] + parsed["local_store_footprint_bytes"]
    )
    parsed["dynamic_spill_total_bytes"] = (
        parsed["dynamic_spill_refill_bytes"] + parsed["dynamic_spill_store_bytes"]
    )

    identity_fields = {
        "row_id": "ID",
        "kernel": "Kernel Name",
        "context_id": "Context",
        "stream_id": "Stream",
        "device_id": "Device",
    }
    missing = [column for column in identity_fields.values() if column not in by_name]
    dimension_columns = {
        "block": (
            "launch__block_dim_x",
            "launch__block_dim_y",
            "launch__block_dim_z",
        ),
        "grid": (
            "launch__grid_dim_x",
            "launch__grid_dim_y",
            "launch__grid_dim_z",
        ),
    }
    missing.extend(
        column
        for columns in dimension_columns.values()
        for column in columns
        if column not in by_name
    )
    if missing:
        raise ValueError(
            f"missing NCU launch identity columns {sorted(set(missing))}: {path}"
        )
    identity = {
        "row_id": _parse_integer(by_name["ID"], field="ID"),
        "kernel": by_name["Kernel Name"],
        "context_id": _parse_integer(by_name["Context"], field="Context"),
        "stream_id": _parse_integer(by_name["Stream"], field="Stream"),
        "device_id": _parse_integer(by_name["Device"], field="Device"),
        "block": [
            _parse_integer(by_name[column], field=column)
            for column in dimension_columns["block"]
        ],
        "grid": [
            _parse_integer(by_name[column], field=column)
            for column in dimension_columns["grid"]
        ],
    }
    return parsed, raw_units, identity


def parse_arm_paths(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        arm, separator, raw = value.partition("=")
        if not separator or arm not in ALL_ARMS or not raw:
            raise ValueError(f"expected ARM=PATH, got {value!r}")
        if arm in parsed:
            raise ValueError(f"duplicate path assignment for {arm}")
        parsed[arm] = Path(raw).resolve()
    if set(parsed) != set(ALL_ARMS):
        raise ValueError(f"both arms are required, got {sorted(parsed)}")
    return parsed


def parse_veloq_launch(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    envelope_checks = {
        "schema": payload.get("schema") == "v1",
        "command": payload.get("command") == "ncu.launches",
        "source": payload.get("source") == {"kind": "ncu", "version": "v1"},
        "trace_kind": payload.get("trace", {}).get("kind") == "ncu",
        "trace_path": isinstance(payload.get("trace", {}).get("path"), str),
    }
    failed = sorted(name for name, passed in envelope_checks.items() if not passed)
    if failed:
        raise ValueError(
            f"invalid VeloQ launches envelope ({', '.join(failed)}): {path}"
        )
    rows = payload.get("data", {}).get("rows", [])
    if len(rows) != 1:
        raise ValueError(f"expected exactly one VeloQ launch row: {path}")
    row = rows[0]
    if row.get("key") != "launch:0" or row.get("row_id") != "launch:0":
        raise ValueError(f"unexpected VeloQ launch key/row_id: {path}")
    result = {
        "key": row.get("key"),
        "row_id": row.get("row_id"),
        "kernel": row.get("kernel_demangled"),
        "grid": row.get("grid_size"),
        "block": row.get("block_size"),
        "context_id": row.get("context_id"),
        "stream_id": row.get("stream_id"),
        "device_id": row.get("device_id"),
        "trace_path": payload["trace"]["path"],
    }
    if not isinstance(result["kernel"], str):
        raise ValueError(f"missing VeloQ demangled kernel: {path}")
    for field in ("grid", "block"):
        value = result[field]
        if (
            not isinstance(value, list)
            or len(value) != 3
            or not all(isinstance(item, int) for item in value)
        ):
            raise ValueError(f"invalid VeloQ {field}: {path}")
    for field in ("context_id", "stream_id", "device_id"):
        if not isinstance(result[field], int):
            raise ValueError(f"invalid VeloQ {field}: {path}")
    return result


def _validate_self_hash(payload: Mapping[str, Any], field: str) -> None:
    expected = payload.get(field)
    if not isinstance(expected, str):
        raise ValueError(f"missing evidence self hash {field}")
    body = dict(payload)
    del body[field]
    if canonical_sha256(body) != expected:
        raise ValueError(f"evidence self hash mismatch: {field}")


def _runtime_contract(runtime: Mapping[str, Any]) -> dict[str, Any]:
    gpu = runtime.get("gpu", {})
    source = runtime.get("source", {})
    imports = runtime.get("imports", {})
    return {
        "cuda_runtime": runtime.get("cuda_runtime"),
        "image_digest": runtime.get("image_digest"),
        "nvcc": runtime.get("nvcc"),
        "ptxas": runtime.get("ptxas"),
        "python": runtime.get("python"),
        "python_deps_sha256": runtime.get("python_deps_sha256"),
        "torch": runtime.get("torch"),
        "gpu": {
            key: gpu.get(key)
            for key in (
                "uuid",
                "name",
                "compute_capability",
                "driver",
                "sm_count",
                "pci_bus_id",
            )
        },
        "source": {
            key: source.get(key)
            for key in (
                "checkout_head",
                "cutlass_commit",
                "locked_source_commit",
                "production_kernel_sha256",
            )
        },
        "cutlass_python_version": imports.get("cutlass_python_version"),
    }


def _validate_static_evidence(
    payload: Mapping[str, Any],
    preparations: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    if payload.get("schema") != "exp005.static-resource-spill-evidence.v1":
        raise ValueError("unexpected static spill evidence schema")
    _validate_self_hash(payload, "evidence_sha256")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ALL_ARMS):
        raise ValueError("static spill evidence does not contain exactly both arms")
    compact: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for arm in ALL_ARMS:
        item = arms[arm]
        cubin_sha = item.get("identity", {}).get("cubin_sha256")
        resource = item.get("resource", {})
        gate = item.get("gates", {}).get("zero_spill_static_gate")
        checks[f"{arm}_cubin_matches_m8192_preparation"] = preparations[arm].get(
            "cubin_sha256"
        ) == [cubin_sha]
        frame = resource.get("stack_bytes_per_thread")
        if not isinstance(frame, int) or frame < 0 or not isinstance(gate, bool):
            raise ValueError(f"invalid static frame/gate for {arm}")
        compact[arm] = {
            "cubin_sha256": cubin_sha,
            "static_frame_bytes_per_thread": frame,
            "static_local_bytes_outside_stack": resource.get(
                "static_local_bytes_outside_stack"
            ),
            "compiler_spill_refill_annotation_count": item.get(
                "compiler_spill_refill", {}
            ).get("annotation_count"),
            "zero_spill_static_gate": gate,
        }
    return compact, checks


def validate_identity_bundle(
    *,
    capture_identities: Mapping[str, Mapping[str, Any]],
    preparations: Mapping[str, Mapping[str, Any]],
    profile_targets: Mapping[str, Mapping[str, Any]],
    native_paths: Mapping[str, Path],
    report_paths: Mapping[str, Path],
    launches: Mapping[str, Mapping[str, Any]],
    native_launches: Mapping[str, Mapping[str, Any]],
    static_payload: Mapping[str, Any],
    correctness: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    checks: dict[str, bool] = {}
    for arm in ALL_ARMS:
        capture = capture_identities[arm]
        _validate_self_hash(capture, "identity_sha256")
        prep = preparations[arm]
        target = profile_targets[arm]
        expected_launch = {
            "grid": list(EXPECTED_GRID),
            "block": list(expected_block(arm)),
            "kernel": "MoEDynamicKernel",
        }
        checks.update(
            {
                f"{arm}_capture_schema": capture.get("schema")
                == "exp005.ncu-capture-identity.v3",
                f"{arm}_capture_revision": capture.get("capture_revision")
                == "canonical_v1",
                f"{arm}_capture_case": capture.get("arm") == arm
                and capture.get("m") == 8192
                and capture.get("fixture_kind") == CANONICAL_FIXTURE,
                f"{arm}_capture_geometry": capture.get("expected_grid")
                == list(EXPECTED_GRID)
                and capture.get("expected_block") == list(expected_block(arm)),
                f"{arm}_section_contract": capture.get("section_ids")
                == list(REQUIRED_SECTION_IDS),
                f"{arm}_custom_metric_contract": capture.get("custom_metric_ids")
                == list(CUSTOM_METRIC_IDS),
                f"{arm}_required_metric_contract": capture.get("required_metric_ids")
                == list(METRICS.values()),
                f"{arm}_native_hash": capture.get("native_raw_sha256")
                == file_sha256(native_paths[arm]),
                f"{arm}_report_hash": capture.get("trace_sha256")
                == file_sha256(report_paths[arm]),
                f"{arm}_preparation_hash": capture.get("preparation_sha256")
                == file_sha256(Path(prep["_path"])),
                f"{arm}_target_hash": capture.get("profile_target_sha256")
                == file_sha256(Path(target["_path"])),
                f"{arm}_preparation_case": prep.get("schema")
                == "exp005.arm-preparation.v1"
                and prep.get("status") == "complete"
                and prep.get("arm") == arm
                and prep.get("m") == 8192
                and prep.get("fixture_kind") == CANONICAL_FIXTURE,
                f"{arm}_capture_cubin": capture.get("cubin_sha256")
                == prep.get("cubin_sha256"),
                f"{arm}_capture_jit": capture.get("jit_artifact_set_sha256")
                == prep.get("jit_artifact_set_sha256"),
                f"{arm}_target_case": target.get("schema") == "exp005.profile-target.v1"
                and target.get("status") == "complete"
                and target.get("arm") == arm
                and target.get("m") == 8192
                and target.get("fixture_kind") == CANONICAL_FIXTURE,
                f"{arm}_target_launch": target.get("expected_launch")
                == expected_launch,
                f"{arm}_target_jit": target.get("jit_artifact_set_sha256")
                == prep.get("jit_artifact_set_sha256"),
                f"{arm}_target_runtime": _runtime_contract(target.get("runtime", {}))
                == _runtime_contract(prep.get("runtime", {})),
                f"{arm}_veloq_report_binding": Path(
                    str(launches[arm]["trace_path"])
                ).resolve()
                == report_paths[arm].resolve(),
                f"{arm}_native_veloq_context": native_launches[arm]["context_id"]
                == launches[arm]["context_id"],
                f"{arm}_native_veloq_stream": native_launches[arm]["stream_id"]
                == launches[arm]["stream_id"],
                f"{arm}_native_veloq_device": native_launches[arm]["device_id"]
                == launches[arm]["device_id"],
                f"{arm}_native_veloq_kernel": native_launches[arm]["kernel"]
                == launches[arm]["kernel"],
                f"{arm}_native_veloq_grid": native_launches[arm]["grid"]
                == launches[arm]["grid"],
                f"{arm}_native_veloq_block": native_launches[arm]["block"]
                == launches[arm]["block"],
            }
        )
    for field in ("case", "fixture", "weights", "reference_sha256"):
        checks[f"cross_arm_{field}"] = preparations[BASELINE].get(
            field
        ) == preparations[CANDIDATE].get(field)
    checks["cross_arm_runtime_contract"] = _runtime_contract(
        preparations[BASELINE].get("runtime", {})
    ) == _runtime_contract(preparations[CANDIDATE].get("runtime", {}))
    checks["cross_arm_ncu_version"] = capture_identities[BASELINE].get(
        "ncu_version"
    ) == capture_identities[CANDIDATE].get("ncu_version")
    checks["cross_arm_collection_protocol"] = capture_identities[BASELINE].get(
        "collection_protocol"
    ) == capture_identities[CANDIDATE].get("collection_protocol")
    checks["cross_arm_section_contract"] = capture_identities[BASELINE].get(
        "section_ids"
    ) == capture_identities[CANDIDATE].get("section_ids")
    checks["correctness_gate"] = (
        correctness.get("schema") == "exp005.correctness.v1"
        and correctness.get("m") == 8192
        and correctness.get("gate_pass") is True
    )
    static_compact, static_checks = _validate_static_evidence(
        static_payload, preparations
    )
    checks.update(static_checks)
    return checks, static_compact


def build_evidence(
    metrics: Mapping[str, Mapping[str, float]],
    launches: Mapping[str, Mapping[str, Any]],
    *,
    native_launches: Mapping[str, Mapping[str, Any]],
    identity_checks: Mapping[str, bool],
    static_arms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = metrics[BASELINE]
    candidate = metrics[CANDIDATE]
    launch_checks: dict[str, dict[str, bool]] = {}
    for arm in ALL_ARMS:
        launch = launches[arm]
        native = native_launches[arm]
        launch_checks[arm] = {
            "kernel": "MoEDynamicKernel" in str(launch.get("kernel")),
            "grid": launch.get("grid") == list(EXPECTED_GRID),
            "block": launch.get("block") == list(expected_block(arm)),
            "native_kernel": "MoEDynamicKernel" in str(native.get("kernel")),
            "native_grid": native.get("grid") == list(EXPECTED_GRID),
            "native_block": native.get("block") == list(expected_block(arm)),
        }

    work_checks: dict[str, bool] = {}
    for name in ("tensor_instructions", "fp4_tensor_ops"):
        baseline_value = baseline[name]
        candidate_value = candidate[name]
        work_checks[f"{name}_baseline_integral"] = baseline_value.is_integer()
        work_checks[f"{name}_candidate_integral"] = candidate_value.is_integer()
        work_checks[f"{name}_equal"] = int(candidate_value) == int(baseline_value)

    dynamic_spill_checks = {
        "spill_refill_instructions_zero": candidate["dynamic_spill_refill_instructions"]
        == 0,
        "spill_store_instructions_zero": candidate["dynamic_spill_store_instructions"]
        == 0,
        "spill_refill_bytes_zero": candidate["dynamic_spill_refill_bytes"] == 0,
        "spill_store_bytes_zero": candidate["dynamic_spill_store_bytes"] == 0,
    }
    static_spill_pass = bool(static_arms[CANDIDATE]["zero_spill_static_gate"])

    arms: dict[str, dict[str, float]] = {}
    for arm in ALL_ARMS:
        arm_metrics = dict(metrics[arm])
        warp_latency = arm_metrics["warp_latency_per_issued_instruction"]
        if warp_latency <= 0:
            raise ValueError(f"{arm}: non-positive total warp latency")
        for name in STALL_NAMES:
            arm_metrics[f"stall_{name}_pct"] = (
                100.0 * arm_metrics[f"stall_{name}_ratio"] / warp_latency
            )
        arms[arm] = arm_metrics

    launch_pass = all(all(values.values()) for values in launch_checks.values())
    identity_pass = all(identity_checks.values())
    work_pass = all(work_checks.values())
    dynamic_spill_pass = all(dynamic_spill_checks.values())
    zero_spill_pass = static_spill_pass and dynamic_spill_pass
    payload: dict[str, Any] = {
        "schema": "exp005.ncu-evidence.v2",
        "arms": arms,
        "launches": dict(launches),
        "launch_checks": launch_checks,
        "launch_identity_pass": launch_pass,
        "evidence_identity_checks": dict(identity_checks),
        "evidence_identity_pass": identity_pass,
        "work_identity_checks": work_checks,
        "work_identity_pass": work_pass,
        "static_spill_evidence": dict(static_arms),
        "candidate_static_zero_spill_pass": static_spill_pass,
        "candidate_dynamic_zero_spill_checks": dynamic_spill_checks,
        "candidate_dynamic_zero_spill_pass": dynamic_spill_pass,
        "candidate_zero_spill_pass": zero_spill_pass,
        "overall_gate_pass": (
            launch_pass and identity_pass and work_pass and zero_spill_pass
        ),
        "candidate_vs_baseline": {
            name: candidate[name] / baseline[name] if baseline[name] else None
            for name in metrics[BASELINE]
        },
        "stack_evidence_boundary": {
            arm: {
                "configured_runtime_stack_limit_bytes": arms[arm][
                    "configured_stack_limit_bytes"
                ],
                "cubin_static_frame_bytes_per_thread": static_arms[arm][
                    "static_frame_bytes_per_thread"
                ],
                "directly_comparable": False,
            }
            for arm in ALL_ARMS
        },
        "evidence_boundary": {
            "spill_existence": (
                "Static cubin/ELF/SASS evidence proves spill code exists; the "
                "NCU sass__ register-spilling counters prove dynamic execution."
            ),
            "generic_local_traffic": (
                "L1TEX local bytes are supporting traffic evidence and are not "
                "used alone to classify compiler spill."
            ),
            "configured_stack_limit": (
                "launch__stack_size is the configured CUDA runtime stack limit, "
                "not the cubin static frame or spill bytes."
            ),
            "stall_percentage": (
                "Each displayed stall/throttle percentage is its NCU per-issued-"
                "instruction ratio divided by total warp latency."
            ),
            "attribution": (
                "Whole-kernel counters do not attribute time to Gate, Up, or FC2."
            ),
        },
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--native-csv", action="append", required=True)
    parser.add_argument("--veloq-launches", action="append", required=True)
    parser.add_argument("--capture-identity", action="append", required=True)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--preparation", action="append", required=True)
    parser.add_argument("--profile-target", action="append", required=True)
    parser.add_argument("--static-evidence", type=Path)
    parser.add_argument("--correctness", type=Path)
    args = parser.parse_args(argv)
    results = args.results.resolve()
    native_paths = parse_arm_paths(args.native_csv)
    launch_paths = parse_arm_paths(args.veloq_launches)
    capture_paths = parse_arm_paths(args.capture_identity)
    report_paths = parse_arm_paths(args.report)
    preparation_paths = parse_arm_paths(args.preparation)
    target_paths = parse_arm_paths(args.profile_target)
    static_path = (
        args.static_evidence.resolve()
        if args.static_evidence
        else results / "static_spill_evidence.json"
    )
    correctness_path = (
        args.correctness.resolve()
        if args.correctness
        else results / "correctness" / "m8192.json"
    )

    metrics: dict[str, dict[str, float]] = {}
    launches: dict[str, dict[str, Any]] = {}
    native_launches: dict[str, dict[str, Any]] = {}
    captures: dict[str, dict[str, Any]] = {}
    preparations: dict[str, dict[str, Any]] = {}
    targets: dict[str, dict[str, Any]] = {}
    units: dict[str, str] | None = None
    inputs: dict[str, Any] = {}
    for arm in ALL_ARMS:
        metrics[arm], current_units, native_launches[arm] = parse_native_csv(
            native_paths[arm]
        )
        if units is not None and current_units != units:
            raise RuntimeError("NCU units differ across arms")
        units = current_units
        launches[arm] = parse_veloq_launch(launch_paths[arm])
        captures[arm] = read_json(capture_paths[arm])
        preparations[arm] = read_json(preparation_paths[arm])
        preparations[arm]["_path"] = str(preparation_paths[arm])
        targets[arm] = read_json(target_paths[arm])
        targets[arm]["_path"] = str(target_paths[arm])
        inputs[arm] = {
            "native_csv": str(native_paths[arm]),
            "native_csv_sha256": file_sha256(native_paths[arm]),
            "veloq_launches": str(launch_paths[arm]),
            "veloq_launches_sha256": file_sha256(launch_paths[arm]),
            "capture_identity": str(capture_paths[arm]),
            "capture_identity_sha256": file_sha256(capture_paths[arm]),
            "report": str(report_paths[arm]),
            "report_sha256": file_sha256(report_paths[arm]),
            "preparation": str(preparation_paths[arm]),
            "preparation_sha256": file_sha256(preparation_paths[arm]),
            "profile_target": str(target_paths[arm]),
            "profile_target_sha256": file_sha256(target_paths[arm]),
        }
    static_payload = read_json(static_path)
    correctness = read_json(correctness_path)
    identity_checks, static_arms = validate_identity_bundle(
        capture_identities=captures,
        preparations=preparations,
        profile_targets=targets,
        native_paths=native_paths,
        report_paths=report_paths,
        launches=launches,
        native_launches=native_launches,
        static_payload=static_payload,
        correctness=correctness,
    )
    payload = build_evidence(
        metrics,
        launches,
        native_launches=native_launches,
        identity_checks=identity_checks,
        static_arms=static_arms,
    )
    payload["inputs"] = inputs
    payload["inputs"]["static_evidence"] = {
        "path": str(static_path),
        "sha256": file_sha256(static_path),
    }
    payload["inputs"]["correctness"] = {
        "path": str(correctness_path),
        "sha256": file_sha256(correctness_path),
    }
    payload["raw_units"] = units
    payload["evidence_sha256"] = canonical_sha256(payload)
    output = results / "ncu"
    write_json(output / "evidence.json", payload)
    write_csv(
        output / "metrics.csv",
        [{"arm": arm, **payload["arms"][arm]} for arm in ALL_ARMS],
    )
    return 0 if payload["overall_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
