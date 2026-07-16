#!/usr/bin/env python3
"""Build the v2 whole-operator NCU traffic ledger for exp_002."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"
ARMS = (
    "cutedsl_bf16_fused",
    "cutlass_bf16_chain",
)
PAIRED_ARMS = ARMS
M_VALUES = (256, 8192)
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512
TOPK = 8
FP4_OPS_PER_PHYSICAL_ROW = 3 * 2 * HIDDEN_SIZE * INTERMEDIATE_SIZE

RAW_METRICS = {
    "profiler_range_duration_ns": "gpu__time_duration.sum",
    "dram_total_authority_bytes": "dram__bytes.sum",
    "dram_read_bytes": "dram__bytes_op_read.sum",
    "dram_write_bytes": "dram__bytes_op_write.sum",
    "l2_total_authority_sectors": "lts__t_sectors.sum",
    "l2_read_sectors": "lts__t_sectors_op_read.sum",
    "l2_write_sectors": "lts__t_sectors_op_write.sum",
    "l2_atomic_sectors": "lts__t_sectors_op_atom.sum",
    "l2_reduction_sectors": "lts__t_sectors_op_red.sum",
    "l2_membar_sectors": "lts__t_sectors_op_membar.sum",
    "lsu_global_load_footprint_bytes": ("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"),
    "lsu_global_store_footprint_bytes": (
        "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"
    ),
    "lsu_global_atomic_footprint_bytes": (
        "l1tex__t_bytes_pipe_lsu_mem_global_op_atom.sum"
    ),
    "lsu_global_reduction_footprint_bytes": (
        "l1tex__t_bytes_pipe_lsu_mem_global_op_red.sum"
    ),
    "lsu_global_ldgsts_footprint_bytes": (
        "l1tex__t_bytes_pipe_lsu_mem_global_op_ldgsts_cache_access.sum"
    ),
    "tma_global_load_interface_bytes": (
        "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum"
    ),
    "tma_global_store_interface_bytes": (
        "l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum"
    ),
    "tma_global_reduction_interface_bytes": (
        "l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_red.sum"
    ),
    "local_load_footprint_bytes": ("l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum"),
    "local_store_footprint_bytes": ("l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum"),
    "dynamic_warp_instructions": "sm__inst_executed.sum",
    "tensor_hmma_qmma_omma_instructions": (
        "sm__inst_executed_pipe_tensor_subpipe_hmma.sum"
    ),
    "fp4_to_fp32_tensor_ops": "sm__ops_path_tensor_src_fp4_dst_fp32.sum",
}

UNIT_SCALE = {
    "": 1.0,
    "%": 1.0,
    "byte": 1.0,
    "Kbyte": 1e3,
    "Mbyte": 1e6,
    "Gbyte": 1e9,
    "ns": 1.0,
    "us": 1e3,
    "ms": 1e6,
    "s": 1e9,
    "sector": 1.0,
    "inst": 1.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_native_raw(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    with path.open(newline="") as file:
        rows = list(csv.reader(file))
    header_index = next(
        (index for index, row in enumerate(rows) if row and row[0] == "ID"), None
    )
    if header_index is None or len(rows) != header_index + 3:
        raise ValueError(f"expected one native NCU range row in {path}")
    header, units, values = rows[header_index : header_index + 3]
    if not (len(header) == len(units) == len(values)):
        raise ValueError(f"native NCU CSV width mismatch: {path}")
    value_by_name = dict(zip(header, values, strict=True))
    unit_by_name = dict(zip(header, units, strict=True))
    if value_by_name.get("Kernel Name") != "range":
        raise ValueError(f"expected app-range result in {path}")

    parsed: dict[str, float] = {}
    raw_units: dict[str, str] = {}
    for output_name, metric_name in RAW_METRICS.items():
        if metric_name not in value_by_name or value_by_name[metric_name] == "":
            raise ValueError(f"missing metric {metric_name} in {path}")
        unit = unit_by_name[metric_name]
        if unit not in UNIT_SCALE:
            raise ValueError(f"unsupported NCU unit {unit!r} for {metric_name}")
        parsed[output_name] = float(value_by_name[metric_name]) * UNIT_SCALE[unit]
        raw_units[output_name] = unit

    parsed["dram_component_sum_bytes"] = (
        parsed["dram_read_bytes"] + parsed["dram_write_bytes"]
    )
    parsed["dram_unclassified_residual_bytes"] = (
        parsed["dram_total_authority_bytes"] - parsed["dram_component_sum_bytes"]
    )

    l2_components = (
        "l2_read_sectors",
        "l2_write_sectors",
        "l2_atomic_sectors",
        "l2_reduction_sectors",
        "l2_membar_sectors",
    )
    parsed["l2_classified_component_sectors"] = sum(
        parsed[name] for name in l2_components
    )
    parsed["l2_unclassified_residual_sectors"] = (
        parsed["l2_total_authority_sectors"] - parsed["l2_classified_component_sectors"]
    )
    for name in ("l2_total_authority_sectors", *l2_components):
        parsed[name.removesuffix("_sectors") + "_bytes"] = parsed[name] * 32.0
    parsed["l2_unclassified_residual_bytes"] = (
        parsed["l2_unclassified_residual_sectors"] * 32.0
    )

    parsed["lsu_global_t_stage_footprint_bytes"] = sum(
        parsed[name]
        for name in (
            "lsu_global_load_footprint_bytes",
            "lsu_global_store_footprint_bytes",
            "lsu_global_atomic_footprint_bytes",
            "lsu_global_reduction_footprint_bytes",
            "lsu_global_ldgsts_footprint_bytes",
        )
    )
    parsed["tma_global_interface_bytes"] = sum(
        parsed[name]
        for name in (
            "tma_global_load_interface_bytes",
            "tma_global_store_interface_bytes",
            "tma_global_reduction_interface_bytes",
        )
    )
    parsed["local_total_footprint_bytes"] = (
        parsed["local_load_footprint_bytes"] + parsed["local_store_footprint_bytes"]
    )
    return parsed, raw_units


def validate_range_identity(
    case_dir: Path, results: Path, m: int, arm: str
) -> dict[str, Any]:
    command = (case_dir / "command.txt").read_text().strip()
    for required in (
        "--replay-mode app-range",
        "--cache-control all",
        "single-replay",
        f"--m {m}",
        f"--arm {arm}",
        "dram__bytes.sum",
        "lts__t_sectors.sum",
    ):
        if required not in command:
            raise ValueError(f"capture command missing {required!r}: {case_dir}")

    export_command = (case_dir / "native_raw.command.txt").read_text()
    if "--print-units base" not in export_command:
        raise ValueError(f"native export did not request base units: {case_dir}")
    if (case_dir / "native_raw.stderr.log").read_text().strip():
        raise ValueError(f"native export wrote stderr: {case_dir}")

    summary = read_json(case_dir / "veloq" / "summary.json")
    ranges = read_json(case_dir / "veloq" / "ranges.json")
    if "error" in summary or "error" in ranges:
        raise ValueError(f"VeloQ error for {case_dir}")
    summary_rows = summary.get("data", {}).get("rows", [])
    range_rows = ranges.get("data", {}).get("rows", [])
    if len(summary_rows) != 1 or len(range_rows) != 1:
        raise ValueError(f"expected one range inventory for {case_dir}")
    totals = summary_rows[0]
    if totals.get("range_count") != 1 or totals.get("launch_count") != 0:
        raise ValueError(f"unexpected NCU result topology for {case_dir}")
    if range_rows[0].get("key") != "range:0":
        raise ValueError(f"unexpected range row for {case_dir}")

    manifest_path = case_dir / "profile_manifest.json"
    manifest = read_json(manifest_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("m") != m
        or manifest.get("arm") != arm
    ):
        raise ValueError(f"profile manifest drift for m={m} arm={arm}")
    expected_nvtx = f"exp002_m{m}_{arm}_single_replay"
    if manifest.get("nvtx_range") != expected_nvtx:
        raise ValueError(f"NVTX identity drift for m={m} arm={arm}")
    identity = read_json(results / "evidence.identity.json")
    expected = {
        "comparison_group_id": identity["comparison_group_id"],
        "rerun_id": identity["rerun_id"],
        "environment_lock_digest": identity["environment_lock_digest"],
        "protocol_lock_digest": identity["protocol_lock_digest"],
        "artifact_fingerprint_sha256": identity["per_arm_artifact_fingerprint_sha256"][
            arm
        ],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"evidence identity drift at {key} for m={m} arm={arm}")

    report = case_dir / "trace.ncu-rep"
    recorded_hash = (case_dir / "trace.ncu-rep.sha256").read_text().split()[0]
    actual_hash = sha256(report)
    if recorded_hash != actual_hash:
        raise ValueError(f"NCU report hash mismatch: {report}")
    csv_path = case_dir / "native_raw.csv"
    recorded_csv_hash = (case_dir / "native_raw.csv.sha256").read_text().split()[0]
    actual_csv_hash = sha256(csv_path)
    if recorded_csv_hash != actual_csv_hash:
        raise ValueError(f"native CSV hash mismatch: {csv_path}")
    veloq_version = (case_dir / "veloq" / "version.txt").read_text().strip()
    if not veloq_version:
        raise ValueError(f"missing VeloQ version for {case_dir}")
    return {
        "trace": str(report.relative_to(results)),
        "trace_sha256": actual_hash,
        "native_csv_sha256": actual_csv_hash,
        "profile_manifest_sha256": sha256(manifest_path),
        "ncu_version": summary["data"]["auxiliary"].get("ncu_version"),
        "range_key": range_rows[0]["key"],
        "nvtx_range": expected_nvtx,
        "capture": "app-range / cache-control all / one CUDA Graph replay",
        "metric_export": "official ncu --import --csv --page raw --print-units base",
        "veloq_limit": (
            f"{veloq_version} inventories range identity but does not project "
            "range metrics"
        ),
    }


def collect(results: Path) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for m in M_VALUES:
        for arm in ARMS:
            case_dir = results / "ncu" / f"m{m}" / arm / "operator-ledger-v2"
            provenance = validate_range_identity(case_dir, results, m, arm)
            metrics, raw_units = parse_native_raw(case_dir / "native_raw.csv")
            physical_rows = metrics["fp4_to_fp32_tensor_ops"] / FP4_OPS_PER_PHYSICAL_ROW
            if abs(physical_rows - round(physical_rows)) > 1e-6:
                raise ValueError(
                    f"non-integral physical row count for m={m} arm={arm}: "
                    f"{physical_rows}"
                )
            metrics["logical_routed_rows"] = float(m * TOPK)
            metrics["physical_routed_rows_from_fp4_ops"] = physical_rows
            metrics["physical_padding_factor"] = physical_rows / (m * TOPK)
            row = {
                "m": m,
                "arm": arm,
                "relationship": "paired",
                "metrics": metrics,
                "raw_csv_units": raw_units,
                "provenance": provenance,
            }
            cases.append(row)
            indexed[(m, arm)] = row

    comparable_metrics = (
        "dram_total_authority_bytes",
        "dram_read_bytes",
        "dram_write_bytes",
        "l2_total_authority_bytes",
        "l2_read_bytes",
        "l2_write_bytes",
        "l2_atomic_bytes",
        "l2_reduction_bytes",
        "l2_membar_bytes",
        "lsu_global_load_footprint_bytes",
        "lsu_global_store_footprint_bytes",
        "lsu_global_atomic_footprint_bytes",
        "lsu_global_reduction_footprint_bytes",
        "lsu_global_ldgsts_footprint_bytes",
        "lsu_global_t_stage_footprint_bytes",
        "tma_global_load_interface_bytes",
        "tma_global_store_interface_bytes",
        "tma_global_reduction_interface_bytes",
        "tma_global_interface_bytes",
        "local_load_footprint_bytes",
        "local_store_footprint_bytes",
        "local_total_footprint_bytes",
        "dynamic_warp_instructions",
        "tensor_hmma_qmma_omma_instructions",
        "fp4_to_fp32_tensor_ops",
        "physical_routed_rows_from_fp4_ops",
        "physical_padding_factor",
    )
    comparisons: list[dict[str, Any]] = []
    for m in M_VALUES:
        target = indexed[(m, PAIRED_ARMS[0])]["metrics"]
        baseline = indexed[(m, PAIRED_ARMS[1])]["metrics"]
        for metric in comparable_metrics:
            target_value = float(target[metric])
            baseline_value = float(baseline[metric])
            comparisons.append(
                {
                    "m": m,
                    "metric": metric,
                    "cutedsl_fused": target_value,
                    "cutlass_bf16_chain": baseline_value,
                    "cutedsl_minus_chain": target_value - baseline_value,
                    "cutedsl_vs_chain_percent": (
                        (target_value / baseline_value - 1.0) * 100.0
                        if baseline_value
                        else None
                    ),
                }
            )
    return {
        "schema": "exp002.operator-range-ncu.v2",
        "scope": {
            "operator_boundary": "one correctness-qualified CUDA Graph replay",
            "cache": (
                "NCU-controlled cold cache before each application-replay pass; "
                "not claimed identical to benchmark's 192 MiB software flush"
            ),
            "rollup": "range counter; no per-node DRAM summation",
            "duration_authority": (
                "uninstrumented benchmark, not profiler_range_duration_ns"
            ),
            "hierarchy_rule": "L1TEX, L2, and DRAM are separate observations and are not additive",
            "lsu_scope": (
                "LSU global T-stage sectorized request footprint; excludes "
                "TMA and non-LSU/TEX paths"
            ),
            "tma_scope": ("TMA interface bytes, reported separately from LSU T-stage"),
            "local_scope": (
                "local-address-space sector footprint; not logical payload or "
                "by itself proof of compiler spills"
            ),
        },
        "cases": cases,
        "paired_comparisons": comparisons,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    results = args.results.resolve()
    evidence = collect(results)
    output = results / "ncu" / "operator_traffic_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    write_csv(
        results / "ncu" / "operator_comparison_v2.csv",
        evidence["paired_comparisons"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
