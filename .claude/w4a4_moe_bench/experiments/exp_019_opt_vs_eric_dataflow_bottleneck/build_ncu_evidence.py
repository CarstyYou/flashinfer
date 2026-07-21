#!/usr/bin/env python3
"""Build compact, conclusion-free paired NCU evidence for exp_019."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"
ARMS = ("latest_opt_fp4", "eric_stage4_fp4")
OPT, ERIC = ARMS
M_VALUES = (1024, 8192)

EXPECTED_SOURCE_SHA256 = {
    OPT: "ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19",
    ERIC: "3a5000a990bb978b434f1c7dac621de25112d9f3cec4a5fdfab5f2970b0dc3b8",
}
EXPECTED_OVERLAY_SHA256 = {
    OPT: EXPECTED_SOURCE_SHA256[OPT],
    ERIC: "98adfba7f4e0d00af24383a556e9c93088355539b50dc82480091225e0448120",
}
EXPECTED_CUBIN_SHA256 = {
    OPT: "e9b322e4c978c490adbe0a9bf0f9a183288c0ecddb1fd72e5a904c487be541f3",
    ERIC: "4c728f4ee6115f342f0b32e578ca4901abd7f35ac233035fb8eaf54fce3900b0",
}
EXPECTED_FIXTURE_SHA256 = {
    1024: "0fa7e8a7d8d1d32172971f987d6f55b534aabf8d12a84a910d010cec25ba04a5",
    8192: "c113ecd5ddeff77154ddbd23fc3dc3c83f8ee822e880179ca5c16b1145372438",
}
EXPECTED_SYMBOL = (
    "kernel_cutlass_kernel_flashinferfused_moecute_dslblackwell_sm12x"
    "moe_dynamic_kernelMoEDynamicKernel_object_at__tensorptrbf16gmemalign16"
    "o204820481_tensorptri32gmemo1_tensorptrf32gmemo1_tens_0"
)
EXPECTED_GRID = [1, 1, 110]
EXPECTED_BLOCK = {OPT: [288, 1, 1], ERIC: [160, 1, 1]}
RAW_RETENTION = {
    "capture_host": "10.6.142.16",
    "root": (
        "/home/xiy/workspace/flashinfer_exp001_corrected_074d93e/.claude/"
        "w4a4_moe_bench/experiments/exp_019_opt_vs_eric_dataflow_bottleneck/"
        "results/raw/ncu"
    ),
    "m8192": {
        OPT: {
            "production_ncu_rep_sha256": (
                "883355ca648f9eaefe4e85db1f3cf054c7f49e1194d09313f242115d1a23af2b"
            ),
            "ledger_ncu_rep_sha256": (
                "ed865eb4cac0a398b75cbd2ffdfe61d82991bab4e0a7d79f701191cf2796a504"
            ),
        },
        ERIC: {
            "production_ncu_rep_sha256": (
                "5dce4fefca1158bbcf02ccb2297985795857c77511273d35a5901cd01819a84e"
            ),
            "ledger_ncu_rep_sha256": (
                "aa163191319ff6142523592cb57270a0d97b02c4a46ba68bff309b72a95181da"
            ),
        },
    },
}

# Keep this projection identical to exp_017.  These are normalized/resource
# observations for one launch; no value is partitioned using phase shares.
SELECTED_METRICS = {
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "tc_subpipe_active_pct": (
        "sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_active"
    ),
    "alu_pipe_active_pct": "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
    "fma_pipe_active_pct": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "xu_pipe_active_pct": "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
    "dram_throughput_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_throughput_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_throughput_pct": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lsu_pipe_active_pct": "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "tma_pipe_active_pct": "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_active",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "registers_per_thread": "launch__registers_per_thread",
    "allocated_registers_per_thread": "launch__registers_per_thread_allocated",
    "shared_mem_per_cta_bytes": "launch__shared_mem_per_block",
    "dynamic_shared_mem_per_cta_bytes": "launch__shared_mem_per_block_dynamic",
    "dynamic_spill_load_instructions": "sass__inst_executed_register_spilling_op_read",
    "dynamic_spill_store_instructions": "sass__inst_executed_register_spilling_op_write",
    "occupancy_limit_registers_cta": "launch__occupancy_limit_registers",
    "occupancy_limit_shared_mem_cta": "launch__occupancy_limit_shared_mem",
    "waves_per_sm": "launch__waves_per_multiprocessor",
}

# Exact exp_002 whole-operator additive metric IDs.  The normal section export
# does not necessarily contain them; an optional per-cell
# ``ledger_native_raw.csv`` supplies a small targeted recapture.
ADDITIVE_METRICS = {
    "profiler_duration_ns": "gpu__time_duration.sum",
    "dram_total_bytes": "dram__bytes.sum",
    "dram_read_bytes": "dram__bytes_op_read.sum",
    "dram_write_bytes": "dram__bytes_op_write.sum",
    "l2_total_sectors": "lts__t_sectors.sum",
    "l2_read_sectors": "lts__t_sectors_op_read.sum",
    "l2_write_sectors": "lts__t_sectors_op_write.sum",
    "l2_atomic_sectors": "lts__t_sectors_op_atom.sum",
    "l2_reduction_sectors": "lts__t_sectors_op_red.sum",
    "l2_membar_sectors": "lts__t_sectors_op_membar.sum",
    "lsu_global_load_footprint_bytes": "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "lsu_global_store_footprint_bytes": "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
    "lsu_global_atomic_footprint_bytes": "l1tex__t_bytes_pipe_lsu_mem_global_op_atom.sum",
    "lsu_global_reduction_footprint_bytes": "l1tex__t_bytes_pipe_lsu_mem_global_op_red.sum",
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
    "local_load_footprint_bytes": "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
    "local_store_footprint_bytes": "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
    "executed_warp_instructions": "sm__inst_executed.sum",
    "tensor_instructions": "sm__inst_executed_pipe_tensor_subpipe_hmma.sum",
    "fp4_to_fp32_tensor_ops": "sm__ops_path_tensor_src_fp4_dst_fp32.sum",
}
PC_STALL_PREFIX = "smsp__pcsamp_warps_issue_stalled_"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _relative(path: Path, results: Path) -> str:
    return str(path.resolve().relative_to(results.resolve()))


def _native_dimension(value: str, *, field: str, path: Path) -> list[int]:
    try:
        parsed = list(ast.literal_eval(value))
    except (SyntaxError, ValueError, TypeError) as error:
        raise RuntimeError(f"invalid {field} in {path}: {value!r}") from error
    if len(parsed) != 3 or not all(isinstance(item, int) for item in parsed):
        raise RuntimeError(f"invalid {field} in {path}: {value!r}")
    return parsed


def parse_native_row(
    path: Path, *, symbol: str, grid: Sequence[int], block: Sequence[int]
) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header_index = next(
        (index for index, row in enumerate(rows) if row and row[0] == "ID"), None
    )
    if header_index is None or len(rows) < header_index + 3:
        raise RuntimeError(f"missing native NCU header/value row: {path}")
    header = rows[header_index]
    units = rows[header_index + 1]
    data_rows = [row for row in rows[header_index + 2 :] if any(row)]
    if len(data_rows) != 1:
        raise RuntimeError(f"expected exactly one native NCU launch: {path}")
    values = data_rows[0]
    if not (len(header) == len(units) == len(values)):
        raise RuntimeError(f"native NCU CSV width mismatch: {path}")
    value_by_name = dict(zip(header, values, strict=True))
    unit_by_name = dict(zip(header, units, strict=True))
    _require(value_by_name.get("Kernel Name") == symbol, f"native symbol drift: {path}")
    _require(
        _native_dimension(value_by_name["Grid Size"], field="grid", path=path)
        == list(grid),
        f"native grid drift: {path}",
    )
    _require(
        _native_dimension(value_by_name["Block Size"], field="block", path=path)
        == list(block),
        f"native block drift: {path}",
    )
    metrics: dict[str, dict[str, Any]] = {}
    for name, raw in value_by_name.items():
        if not raw or name in {
            "ID",
            "Process ID",
            "Process Name",
            "Host Name",
            "Kernel Name",
            "Context",
            "Stream",
            "Block Size",
            "Grid Size",
            "Device",
            "CC",
        }:
            continue
        try:
            numeric = float(raw)
        except ValueError:
            continue
        metrics[name] = {
            "metric": name,
            "unit": unit_by_name[name] or None,
            "value": numeric,
        }
    return metrics


def inspect_launch(path: Path) -> dict[str, Any]:
    value = read_json(path)
    _require(
        value.get("command") == "ncu.inspect" and "error" not in value,
        f"invalid VeloQ inspect: {path}",
    )
    rows = value.get("data", {}).get("rows", [])
    _require(
        len(rows) == 1
        and rows[0].get("type") == "launch"
        and rows[0].get("row_id") == "launch:0",
        f"expected exactly one launch: {path}",
    )
    return rows[0]


def selected_metrics(row: Mapping[str, Any], *, path: Path) -> dict[str, Any]:
    by_name = {item.get("name"): item for item in row.get("metrics", [])}
    output: dict[str, Any] = {}
    for alias, metric in SELECTED_METRICS.items():
        item = by_name.get(metric)
        _require(
            item is not None and isinstance(item.get("value"), (int, float)),
            f"missing numeric selected metric {metric}: {path}",
        )
        output[alias] = {
            "metric": metric,
            "unit": item.get("unit"),
            "value": item["value"],
        }
    return output


def pc_sample_stalls(path: Path) -> dict[str, Any]:
    value = read_json(path)
    _require(
        value.get("command") == "ncu.source-metrics" and "error" not in value,
        f"invalid VeloQ PC-sampling evidence: {path}",
    )
    data = value.get("data", {})
    totals: Counter[str] = Counter()
    auxiliary = data.get("auxiliary", {})
    for key in ("unattributed_sass_counter_totals", "out_of_cubin_counter_totals"):
        totals.update(auxiliary.get(key, {}) or {})
    for row in data.get("rows", []):
        totals.update(row.get("counters", {}) or {})
    reasons = {
        metric.removeprefix(PC_STALL_PREFIX): {
            "metric": metric,
            "unit": "sample",
            "value": float(count),
        }
        for metric, count in totals.items()
        if metric.startswith(PC_STALL_PREFIX) and not metric.endswith("_not_issued")
    }
    denominator = sum(item["value"] for item in reasons.values())
    _require(denominator > 0, f"empty PC-sampling denominator: {path}")
    for item in reasons.values():
        item["share_percent"] = 100.0 * item["value"] / denominator
    _require(
        abs(sum(item["share_percent"] for item in reasons.values()) - 100.0) < 1e-9,
        f"PC-sampling shares do not close: {path}",
    )
    return {
        "denominator": "all non-_not_issued PC-sampling reason samples",
        "total_samples": denominator,
        "reasons": dict(sorted(reasons.items())),
    }


def validate_manifest(manifest: Mapping[str, Any], *, arm: str, m: int) -> None:
    label = f"{arm}/m{m}"
    _require(
        manifest.get("schema") == "exp019.production-ncu-target.v1",
        f"schema drift: {label}",
    )
    _require(manifest.get("status") == "complete", f"incomplete target: {label}")
    _require(
        manifest.get("arm") == arm and manifest.get("m") == m,
        f"target identity drift: {label}",
    )
    launch = manifest.get("launch_contract", {})
    _require(launch.get("symbol") == EXPECTED_SYMBOL, f"symbol drift: {label}")
    _require(launch.get("grid") == EXPECTED_GRID, f"grid drift: {label}")
    _require(launch.get("block") == EXPECTED_BLOCK[arm], f"block drift: {label}")
    source = manifest.get("source_identity", {})
    locked_source = source.get("locked_files", {}).get("source", {}).get("sha256")
    runtime_source = source.get("runtime", {})
    _require(locked_source == EXPECTED_SOURCE_SHA256[arm], f"source SHA drift: {label}")
    _require(
        runtime_source.get("source_sha256") == locked_source,
        f"runtime source drift: {label}",
    )
    _require(
        runtime_source.get("overlay_sha256") == EXPECTED_OVERLAY_SHA256[arm],
        f"overlay drift: {label}",
    )
    jit = manifest.get("jit_identity", {})
    _require(
        jit.get("cubin_sha256") == [EXPECTED_CUBIN_SHA256[arm]], f"cubin drift: {label}"
    )
    _require(jit.get("symbols") == [EXPECTED_SYMBOL], f"JIT symbol drift: {label}")
    correctness = manifest.get("correctness", {})
    _require(correctness.get("gate_pass") is True, f"correctness gate failed: {label}")
    for mode in ("eager", "pre_profile_graph", "post_profile_graph"):
        gate = correctness.get(mode, {})
        _require(gate.get("gate_pass") is True, f"{mode} correctness failed: {label}")
        checks = gate.get("checks", {})
        _require(checks and all(checks.values()), f"{mode} check failed: {label}")
    fixture = manifest.get("fixture_manifest", {})
    _require(
        fixture.get("fixture_sha256") == EXPECTED_FIXTURE_SHA256[m],
        f"fixture drift: {label}",
    )
    _require(
        manifest.get("profile", {}).get("profiled_graph_replays") == 1,
        f"replay count drift: {label}",
    )
    _require(
        manifest.get("telemetry_gate", {}).get("pass") is True,
        f"telemetry gate failed: {label}",
    )
    _require(
        bool(manifest.get("weight_identity", {}).get("packed_weights_sha256"))
        and bool(manifest.get("weight_identity", {}).get("scales_sha256")),
        f"weight identity missing: {label}",
    )


def compact_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    source = manifest["source_identity"]
    fixture = manifest["fixture_manifest"]
    correctness = manifest["correctness"]
    return {
        "source_sha256": source["locked_files"]["source"]["sha256"],
        "overlay_sha256": source["runtime"]["overlay_sha256"],
        "cubin_sha256": manifest["jit_identity"]["cubin_sha256"][0],
        "symbol": manifest["launch_contract"]["symbol"],
        "grid": manifest["launch_contract"]["grid"],
        "block": manifest["launch_contract"]["block"],
        "fixture_sha256": fixture["fixture_sha256"],
        "x_sha256": fixture.get("x_sha256"),
        "topk_ids_sha256": fixture.get("topk_ids_sha256"),
        "topk_weights_sha256": fixture.get("topk_weights_sha256"),
        "reference_sha256": correctness["post_profile_graph"]
        .get("oracle", {})
        .get("reference_sha256"),
        "packed_weights_sha256": manifest["weight_identity"]["packed_weights_sha256"],
        "scales_sha256": manifest["weight_identity"]["scales_sha256"],
        "correctness_gate_pass": correctness["gate_pass"],
        "gpu_uuid": manifest.get("runtime", {}).get("gpu_uuid"),
        "nvcc": manifest.get("runtime", {}).get("nvcc"),
    }


def parse_target(results: Path, *, arm: str, m: int) -> dict[str, Any]:
    raw = results / "raw" / "ncu" / arm / f"m{m}"
    veloq = results / "veloq" / "ncu" / arm / f"m{m}"
    paths = {
        "manifest": raw / "target_manifest.json",
        "native_raw": raw / "native_raw.csv",
        "inspect": veloq / "inspect.json",
        "pc_stalls": veloq / "pc_stalls_file.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    _require(not missing, f"incomplete NCU cell {arm}/m{m}: missing {missing}")
    manifest = read_json(paths["manifest"])
    validate_manifest(manifest, arm=arm, m=m)
    launch = inspect_launch(paths["inspect"])
    contract = manifest["launch_contract"]
    _require(
        launch.get("kernel_demangled") == contract["symbol"],
        f"inspect symbol drift: {arm}/m{m}",
    )
    _require(
        launch.get("grid_size") == contract["grid"], f"inspect grid drift: {arm}/m{m}"
    )
    _require(
        launch.get("block_size") == contract["block"],
        f"inspect block drift: {arm}/m{m}",
    )
    _require(
        launch.get("has_disasm") is True, f"inspect lacks cubin disassembly: {arm}/m{m}"
    )

    native = parse_native_row(
        paths["native_raw"],
        symbol=contract["symbol"],
        grid=contract["grid"],
        block=contract["block"],
    )
    ledger_path = raw / "ledger_native_raw.csv"
    ledger = (
        parse_native_row(
            ledger_path,
            symbol=contract["symbol"],
            grid=contract["grid"],
            block=contract["block"],
        )
        if ledger_path.is_file()
        else {}
    )
    additive: dict[str, Any] = {}
    for alias, metric in ADDITIVE_METRICS.items():
        item = ledger.get(metric, native.get(metric))
        additive[alias] = (
            dict(item) if item is not None else {"metric": metric, "available": False}
        )

    return {
        "identity": compact_identity(manifest),
        "artifacts": {name: _relative(path, results) for name, path in paths.items()}
        | ({"ledger_native_raw": _relative(ledger_path, results)} if ledger else {}),
        "selected_metrics": selected_metrics(launch, path=paths["inspect"]),
        "additive_metrics": additive,
        "pc_sample_stalls": pc_sample_stalls(paths["pc_stalls"]),
    }


def _pair_identity(
    opt: Mapping[str, Any], eric: Mapping[str, Any], *, m: int
) -> dict[str, Any]:
    keys = (
        "fixture_sha256",
        "x_sha256",
        "topk_ids_sha256",
        "topk_weights_sha256",
        "reference_sha256",
        "packed_weights_sha256",
        "scales_sha256",
        "gpu_uuid",
        "nvcc",
    )
    checks = {
        key: opt["identity"].get(key) == eric["identity"].get(key) for key in keys
    }
    _require(
        all(checks.values()), f"paired input/runtime identity drift: m{m}: {checks}"
    )
    return {"pass": True, "checks": checks}


def _metric_delta(
    alias: str, opt: Mapping[str, Any], eric: Mapping[str, Any]
) -> dict[str, Any]:
    _require(opt.get("metric") == eric.get("metric"), f"metric ID drift: {alias}")
    _require(opt.get("unit") == eric.get("unit"), f"metric unit drift: {alias}")
    opt_value = float(opt["value"])
    eric_value = float(eric["value"])
    row = {
        "name": alias,
        "metric": opt["metric"],
        "unit": opt.get("unit"),
        "opt": opt_value,
        "eric": eric_value,
    }
    if opt.get("unit") == "%":
        row["delta_kind"] = "percentage_points"
        row["eric_minus_opt_pp"] = eric_value - opt_value
    else:
        row["delta_kind"] = "absolute_and_percent_vs_opt"
        row["eric_minus_opt"] = eric_value - opt_value
        row["eric_vs_opt_percent"] = (
            100.0 * (eric_value / opt_value - 1.0) if opt_value != 0 else None
        )
    return row


def paired_direct(
    opt: Mapping[str, Any], eric: Mapping[str, Any], *, m: int
) -> dict[str, Any]:
    identity = _pair_identity(opt, eric, m=m)
    selected = [
        _metric_delta(
            alias, opt["selected_metrics"][alias], eric["selected_metrics"][alias]
        )
        for alias in SELECTED_METRICS
    ]
    additive: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for alias in ADDITIVE_METRICS:
        opt_item = opt["additive_metrics"][alias]
        eric_item = eric["additive_metrics"][alias]
        if opt_item.get("available") is False or eric_item.get("available") is False:
            unavailable.append(alias)
        else:
            additive.append(_metric_delta(alias, opt_item, eric_item))

    opt_reasons = opt["pc_sample_stalls"]["reasons"]
    eric_reasons = eric["pc_sample_stalls"]["reasons"]
    stall_rows = []
    for reason in sorted(set(opt_reasons) | set(eric_reasons)):
        opt_item = opt_reasons.get(reason)
        eric_item = eric_reasons.get(reason)
        _require(
            opt_item is not None and eric_item is not None,
            f"PC stall reason drift: m{m}/{reason}",
        )
        _require(
            opt_item["metric"] == eric_item["metric"],
            f"PC stall metric drift: m{m}/{reason}",
        )
        stall_rows.append(
            {
                "reason": reason,
                "metric": opt_item["metric"],
                "unit": "% of non-_not_issued PC samples",
                "opt": opt_item["share_percent"],
                "eric": eric_item["share_percent"],
                "delta_kind": "percentage_points",
                "eric_minus_opt_pp": eric_item["share_percent"]
                - opt_item["share_percent"],
            }
        )
    return {
        "identity_gate": identity,
        "selected_metrics": selected,
        "additive_metrics": additive,
        "additive_unavailable": unavailable,
        "pc_stall_sample_shares": stall_rows,
    }


def build_evidence(results: Path, *, require_complete: bool = False) -> dict[str, Any]:
    results = results.resolve()
    cases: dict[str, Any] = {}
    missing_cells: list[str] = []
    for m in M_VALUES:
        arms: dict[str, Any] = {}
        for arm in ARMS:
            manifest = results / "raw" / "ncu" / arm / f"m{m}" / "target_manifest.json"
            if not manifest.is_file():
                missing_cells.append(f"{arm}/m{m}")
                continue
            arms[arm] = parse_target(results, arm=arm, m=m)
        if not arms:
            continue
        case: dict[str, Any] = {"m": m, "arms": arms, "status": "partial"}
        if set(arms) == set(ARMS):
            case["paired_direct"] = paired_direct(arms[OPT], arms[ERIC], m=m)
            case["status"] = "complete"
        cases[f"m{m}"] = case
    _require(cases, f"no complete target manifests below {results / 'raw/ncu'}")
    if require_complete:
        _require(not missing_cells, f"missing required NCU cells: {missing_cells}")
    evidence = {
        "schema": "exp019.paired-production-ncu-evidence.v1",
        "comparison": {
            "baseline": OPT,
            "candidate": ERIC,
            "boundary": "one correctness-qualified complete fused MoE kernel launch",
            "formal_latency_authority": "exp_018 benchmark; NCU duration is diagnostic only",
            "percentage_metric_delta": "percentage points",
            "non_percentage_metric_delta": "absolute plus percent relative to Opt when Opt != 0",
            "pc_stall_denominator": "own-side non-_not_issued PC-sampling reason samples",
            "phase_partition_forbidden": True,
        },
        "requested_cases": list(M_VALUES),
        "raw_retention": RAW_RETENTION,
        "missing_cells": missing_cells,
        "cases": cases,
    }
    encoded = json.dumps(evidence, indent=2, sort_keys=True).encode()
    _require(
        len(encoded) < 200_000, f"compact NCU evidence exceeds 200 KB: {len(encoded)}"
    )
    return evidence


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output or args.results / "ncu_evidence.json"
    evidence = build_evidence(args.results, require_complete=args.require_complete)
    write_json(output, evidence)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
