#!/usr/bin/env python3
"""Validate exp_017 captures and build compact phase/op comparison evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EXP001 = ROOT.parent / "exp_001_backend_case_sweep" / "results"
EXP004 = ROOT.parent / "exp_004_fused_phase_timing_breakdown"
if str(EXP004) not in sys.path:
    sys.path.insert(0, str(EXP004))

from build_binary_identity import (  # noqa: E402
    SPILL_ANNOTATION_RE,
    _selected_projection,
    opcode_projection,
    parse_instructions,
)


ROLES = (
    "route_align",
    "route_count_sort",
    "q0_fill",
    "q0_absmax",
    "q0_quant",
    "fc1",
    "swiglu",
    "q1_fill",
    "q1_absmax",
    "q1_quant",
    "fc2",
    "topk_reduce",
)
ROLE_TOKENS = {
    "route_align": "moe_align_block_size_kernel",
    "route_count_sort": "count_and_sort_expert_tokens_kernel",
    "q0_fill": "vectorized_elementwise_kernel",
    "q0_absmax": "per_tensor_absmax_kernel",
    "q0_quant": "per_tensor_quant_fp8_kernel",
    "fc1": "fused_moe_kernel",
    "swiglu": "act_and_mul_kernel",
    "q1_fill": "vectorized_elementwise_kernel",
    "q1_absmax": "per_tensor_absmax_kernel",
    "q1_quant": "per_tensor_quant_fp8_kernel",
    "fc2": "fused_moe_kernel",
    "topk_reduce": "moe_sum_reduce",
}
OPT_GROUPS = {
    "Routing / scheduler": (
        "clear_init",
        "histogram",
        "prefix",
        "publish_route_tail",
        "claim_cache_control",
    ),
    "Q0 / input pack": ("route_q0_pack",),
    "FC1 + SwiGLU": ("fc1_gate_up_swiglu",),
    "Q1": ("q1",),
    "FC2 + epilogue + R2S": ("fc2_epilogue_r2s",),
    "Output aggregation": ("scatter",),
    "Residual / skew": ("cta_residual", "launch_skew_early_finish"),
}
TRITON_GROUPS = {
    "Routing / scheduler": ("route_align", "route_count_sort"),
    "Q0 / input pack": ("q0_fill", "q0_absmax", "q0_quant"),
    "FC1 + SwiGLU": ("fc1", "swiglu"),
    "Q1": ("q1_fill", "q1_absmax", "q1_quant"),
    "FC2 + epilogue + R2S": ("fc2",),
    "Output aggregation": ("topk_reduce",),
    "Residual / skew": ("graph_node_bubble",),
}
EXPECTED_FIXTURE_SHA256 = (
    "c113ecd5ddeff77154ddbd23fc3dc3c83f8ee822e880179ca5c16b1145372438"
)
EXPECTED_OCCUPANCY_SHA256 = (
    "3e4350788bfcdd1cca175141f0c6626934589b97cb23d74811e4bc2785531a94"
)
BENCHMARK_GPU_UUID = "GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522"
NCU_GPU_UUID = "GPU-c2ac6efb-f30a-c323-6d38-83908adfb14f"
EXPECTED_OPT_SHA256 = (
    "ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19"
)
NCU_ROLES = ("fc1", "swiglu", "fc2", "topk_reduce")
NCU_ROLE_TOKENS = {
    "fc1": "fused_moe_kernel",
    "swiglu": "act_and_mul_kernel",
    "fc2": "fused_moe_kernel",
    "topk_reduce": "moe_sum_reduce",
}
NCU_METRICS = {
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
    "dynamic_spill_load_instructions": (
        "sass__inst_executed_register_spilling_op_read"
    ),
    "dynamic_spill_store_instructions": (
        "sass__inst_executed_register_spilling_op_write"
    ),
    "occupancy_limit_registers_cta": "launch__occupancy_limit_registers",
    "occupancy_limit_shared_mem_cta": "launch__occupancy_limit_shared_mem",
    "waves_per_sm": "launch__waves_per_multiprocessor",
}
PC_STALL_PREFIX = "smsp__pcsamp_warps_issue_stalled_"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def median_range(values: Iterable[float]) -> dict[str, float]:
    samples = list(values)
    if len(samples) != 5:
        raise RuntimeError(f"expected five samples, found {len(samples)}")
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
    }


def validate_fingerprint(value: Mapping[str, Any], *, label: str) -> None:
    fingerprint = value.get("fingerprint_sha256")
    payload = {key: item for key, item in value.items() if key != "fingerprint_sha256"}
    if fingerprint != canonical_sha256(payload):
        raise RuntimeError(f"{label} fingerprint drift")


def validate_correct_capture(value: Mapping[str, Any], *, mode: str) -> None:
    if value.get("schema") != "exp017.opt-phase-capture.v1":
        raise RuntimeError(f"{mode} schema drift")
    if value.get("mode") != mode or len(value.get("runs", [])) != 5:
        raise RuntimeError(f"{mode} replay contract drift")
    if not value["eager"]["gate_pass"] or not all(
        run["gate_pass"] for run in value["runs"]
    ):
        raise RuntimeError(f"{mode} correctness/route gate failed")
    if value.get("foreign_processes_after"):
        raise RuntimeError(f"{mode} observed a foreign GPU process")
    fixture = value["fixture_identity"]
    if (
        fixture.get("fixture_sha256") != EXPECTED_FIXTURE_SHA256
        or fixture.get("occupancy_sha256") != EXPECTED_OCCUPANCY_SHA256
    ):
        raise RuntimeError(f"{mode} did not use the canonical exp_001 fixture")


def ncu_inspect_row(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("command") != "ncu.inspect" or "error" in value:
        raise RuntimeError(f"invalid VeloQ NCU inspect: {path}")
    rows = value.get("data", {}).get("rows", [])
    if len(rows) != 1 or rows[0].get("type") != "launch":
        raise RuntimeError(f"expected one NCU launch row: {path}")
    return rows[0]


def selected_ncu_metrics(row: Mapping[str, Any], *, path: Path) -> dict[str, Any]:
    by_name = {item["name"]: item for item in row.get("metrics", [])}
    selected: dict[str, Any] = {}
    for output_name, metric_name in NCU_METRICS.items():
        item = by_name.get(metric_name)
        if item is None or not isinstance(item.get("value"), (int, float)):
            raise RuntimeError(f"missing numeric NCU metric {metric_name}: {path}")
        selected[output_name] = {
            "metric": metric_name,
            "value": item["value"],
            "unit": item.get("unit"),
        }
    return selected


def pc_sample_stall_evidence(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("command") != "ncu.source-metrics" or "error" in value:
        raise RuntimeError(f"invalid VeloQ PC-sampling evidence: {path}")
    data = value.get("data", {})
    totals: Counter[str] = Counter()
    auxiliary = data.get("auxiliary", {})
    for key in (
        "unattributed_sass_counter_totals",
        "out_of_cubin_counter_totals",
    ):
        totals.update(auxiliary.get(key, {}) or {})
    for row in data.get("rows", []):
        totals.update(row.get("counters", {}))
    reasons = {
        name.removeprefix(PC_STALL_PREFIX): float(count)
        for name, count in totals.items()
        if name.startswith(PC_STALL_PREFIX)
        and not name.endswith("_not_issued")
    }
    denominator = sum(reasons.values())
    if denominator <= 0:
        raise RuntimeError(f"empty PC-sampling denominator: {path}")
    shares = {
        name: 100.0 * count / denominator for name, count in reasons.items()
    }
    if abs(sum(shares.values()) - 100.0) > 1e-9:
        raise RuntimeError(f"PC-sampling shares do not close: {path}")
    return {
        "denominator": "all non-_not_issued PC-sampling reason counts",
        "total_samples": denominator,
        "reason_counts": dict(sorted(reasons.items())),
        "reason_share_percent": dict(sorted(shares.items())),
    }


def pc_counter_row(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        name.removeprefix(PC_STALL_PREFIX): float(count)
        for name, count in row.get("counters", {}).items()
        if name.startswith(PC_STALL_PREFIX)
        and not name.endswith("_not_issued")
        and float(count) != 0.0
    }


def scatter_pc_evidence(
    *, disasm_path: Path, pc_path: Path, launch_sample_total: float
) -> dict[str, Any]:
    disasm = read_json(disasm_path)
    if disasm.get("command") != "ncu.disasm" or "error" in disasm:
        raise RuntimeError("invalid Fused disassembly evidence")
    kernels = disasm.get("data", {}).get("rows", [])
    if len(kernels) != 1:
        raise RuntimeError("expected one Fused kernel disassembly")
    instructions = kernels[0].get("instructions", [])
    by_address = {int(row["address"]): index for index, row in enumerate(instructions)}

    pc_value = read_json(pc_path)
    if pc_value.get("command") != "ncu.source-metrics" or "error" in pc_value:
        raise RuntimeError("invalid Fused SASS PC-sampling evidence")
    pc_rows = {
        int(row["address"]): row for row in pc_value.get("data", {}).get("rows", [])
    }
    if pc_value.get("data", {}).get("total_matched") != len(pc_rows):
        raise RuntimeError("full Fused SASS PC inventory is incomplete")

    redg = [
        row
        for row in instructions
        if str(row.get("opcode", "")).startswith("REDG.E.ADD.BF16x8")
    ]
    if len(redg) != 4:
        raise RuntimeError(f"expected four unrolled Scatter REDG PCs, found {len(redg)}")

    bundles: list[dict[str, Any]] = []
    for ordinal, redg_row in enumerate(redg):
        redg_address = int(redg_row["address"])
        index = by_address[redg_address]
        before = instructions[max(0, index - 140) : index]
        scale_pack = instructions[max(0, index - 32) : index]
        if (
            sum(str(row.get("opcode", "")).startswith("FMUL") for row in scale_pack)
            < 4
            or sum(
                str(row.get("opcode", "")).startswith("F2FP")
                for row in scale_pack
            )
            < 4
        ):
            raise RuntimeError(f"Scatter scale/pack bundle drift at 0x{redg_address:x}")
        smem_loads = [
            row
            for row in before
            if str(row.get("opcode", "")).startswith(("LDS", "LDSM"))
        ]
        if not smem_loads:
            raise RuntimeError(f"Scatter SMEM loads missing before 0x{redg_address:x}")
        tail_smem_load = smem_loads[-1]

        after = instructions[index + 1 : index + 12]
        barriers = [
            row
            for row in after
            if str(row.get("opcode", "")).startswith("BAR.SYNC")
        ]
        if len(barriers) != 1:
            raise RuntimeError(f"post-Scatter barrier drift at 0x{redg_address:x}")
        barrier = barriers[0]
        barrier_index = by_address[int(barrier["address"])]
        sample_candidates = instructions[barrier_index : barrier_index + 5]
        sampled_sync = max(
            sample_candidates,
            key=lambda row: pc_counter_row(pc_rows.get(int(row["address"]), {})).get(
                "barrier", 0.0
            ),
        )

        def compact_pc(row: Mapping[str, Any]) -> dict[str, Any]:
            address = int(row["address"])
            counters = pc_counter_row(pc_rows.get(address, {}))
            total = sum(counters.values())
            return {
                "pc": f"0x{address:x}",
                "opcode": row.get("opcode"),
                "operands": row.get("operands"),
                "sample_counts": dict(sorted(counters.items())),
                "sample_count": total,
                "share_of_launch_pc_samples_percent": (
                    100.0 * total / launch_sample_total
                ),
            }

        bundles.append(
            {
                "output_tile_ordinal": ordinal,
                "smem_tail_load": compact_pc(tail_smem_load),
                "redg": compact_pc(redg_row),
                "post_scatter_barrier_pc": f"0x{int(barrier['address']):x}",
                "post_scatter_sync_sample": compact_pc(sampled_sync),
            }
        )

    def aggregate(key: str) -> dict[str, Any]:
        totals: Counter[str] = Counter()
        sample_count = 0.0
        for bundle in bundles:
            row = bundle[key]
            totals.update(row["sample_counts"])
            sample_count += float(row["sample_count"])
        return {
            "sample_counts": dict(sorted(totals.items())),
            "sample_count": sample_count,
            "share_of_launch_pc_samples_percent": (
                100.0 * sample_count / launch_sample_total
            ),
        }

    return {
        "classification": "semantic-localized; causal mechanism remains open",
        "source_phase": "scatter_sC_to_gmem",
        "producer_consumer_bridge": (
            "FC2 epilogue R2S -> sC BF16 SMEM plus cached token/weight SMEM -> "
            "LDS -> FP32 scale -> BF16x8 pack -> REDG -> post-Scatter CTA sync"
        ),
        "bundles": bundles,
        "aggregate": {
            "smem_tail_load": aggregate("smem_tail_load"),
            "redg": aggregate("redg"),
            "post_scatter_sync_sample": aggregate("post_scatter_sync_sample"),
        },
        "open_question": (
            "PC samples show work/stalls on SMEM load, REDG, and the following "
            "CTA sync, but do not isolate layout pressure, REDG completion, or "
            "warp/task imbalance as the causal critical-path mechanism."
        ),
    }


def build_ncu_evidence(results: Path) -> dict[str, Any]:
    root = results / "veloq" / "ncu"
    opt_manifest_path = results / "raw" / "ncu" / "opt_fused_deep" / "target_manifest.json"
    opt_command_path = results / "raw" / "ncu" / "opt_fused_deep" / "command.txt"
    triton_manifest_path = (
        results
        / "raw"
        / "ncu"
        / "triton_material_deep"
        / "target_results"
        / "triton_capture_manifest.json"
    )
    triton_command_path = (
        results / "raw" / "ncu" / "triton_material_deep" / "command.txt"
    )
    opt_manifest = read_json(opt_manifest_path)
    triton_manifest = read_json(triton_manifest_path)
    capture_semantics = {
        "replay_mode": "kernel",
        "cache_control": "all",
        "graph_profiling": "node",
        "clock_control": "none",
    }
    command_tokens = {
        "--replay-mode": capture_semantics["replay_mode"],
        "--cache-control": capture_semantics["cache_control"],
        "--graph-profiling": capture_semantics["graph_profiling"],
        "--clock-control": capture_semantics["clock_control"],
    }
    for command_path in (opt_command_path, triton_command_path):
        command = command_path.read_text(encoding="utf-8")
        for option, value in command_tokens.items():
            if f"{option} {value}" not in command:
                raise RuntimeError(
                    f"NCU capture semantics drift for {option}: {command_path}"
                )

    if (
        opt_manifest.get("schema") != "exp017.latest-opt-ncu-target.v1"
        or opt_manifest.get("status") != "complete"
    ):
        raise RuntimeError("Latest-opt NCU target manifest is incomplete")
    opt_fixture = opt_manifest["case"]["fixture"]
    if (
        opt_fixture.get("fixture_sha256") != EXPECTED_FIXTURE_SHA256
        or opt_fixture.get("occupancy_sha256") != EXPECTED_OCCUPANCY_SHA256
    ):
        raise RuntimeError("Latest-opt NCU fixture identity drift")
    opt_gpu = opt_manifest["runtime"]["gpu"]
    if (
        opt_gpu.get("uuid") != NCU_GPU_UUID
        or int(opt_gpu.get("applications_graphics_clock_mhz")) != 2377
        or opt_gpu.get("foreign_processes_before_cuda_context")
    ):
        raise RuntimeError("Latest-opt NCU GPU/clock/exclusivity drift")
    if opt_manifest["source"].get("current_opt_sha256") != EXPECTED_OPT_SHA256:
        raise RuntimeError("Latest-opt NCU source identity drift")
    if not all(
        opt_manifest["correctness"][key].get("gate_pass")
        for key in ("eager", "pre_profile_graph", "post_profile_graph")
    ):
        raise RuntimeError("Latest-opt NCU correctness gate failed")
    static_records = opt_manifest["static_resource_usage"].get("records", [])
    if len(static_records) != 1:
        raise RuntimeError("Latest-opt NCU static resource record drift")
    static_resource = static_records[0]
    if (
        static_resource.get("cubin_sha256") != opt_manifest.get("cubin_sha256")
        or static_resource.get("stack_bytes_per_thread") != 0
    ):
        raise RuntimeError("Latest-opt NCU cubin/stack identity drift")

    opt_path = root / "opt_fused" / "inspect.json"
    opt_row = ncu_inspect_row(opt_path)
    if (
        "MoEDynamicKernel" not in opt_row["kernel_demangled"]
        or opt_row["grid_size"] != [1, 1, 110]
        or opt_row["block_size"] != [288, 1, 1]
    ):
        raise RuntimeError("Latest-opt NCU launch identity drift")
    opt_metrics = selected_ncu_metrics(opt_row, path=opt_path)
    if (
        opt_metrics["registers_per_thread"]["value"]
        != static_resource["registers_per_thread"]
        or opt_metrics["dynamic_spill_load_instructions"]["value"] != 0
        or opt_metrics["dynamic_spill_store_instructions"]["value"] != 0
    ):
        raise RuntimeError("Latest-opt NCU resource/spill crosscheck failed")
    opt_stalls = pc_sample_stall_evidence(
        root / "opt_fused" / "pc_stalls_file.json"
    )
    opt_scatter = scatter_pc_evidence(
        disasm_path=root / "opt_fused" / "disasm.json",
        pc_path=root / "opt_fused" / "pc_stalls_sass_full.json",
        launch_sample_total=opt_stalls["total_samples"],
    )

    validate_fingerprint(triton_manifest, label="Triton NCU target manifest")
    if triton_manifest.get("status") != "capture_complete_topology_pending_veloq":
        raise RuntimeError("Triton NCU target manifest is incomplete")
    if (
        triton_manifest["fixture"].get("fixture_sha256")
        != EXPECTED_FIXTURE_SHA256
        or triton_manifest["fixture"].get("occupancy_sha256")
        != EXPECTED_OCCUPANCY_SHA256
    ):
        raise RuntimeError("Triton NCU fixture identity drift")
    triton_runtime = triton_manifest["runtime_identity"]
    if (
        triton_runtime.get("gpu_uuid") != NCU_GPU_UUID
        or triton_runtime.get("application_clock_mhz") != 2377
        or triton_runtime.get("foreign_compute_process_query")
    ):
        raise RuntimeError("Triton NCU GPU/clock/exclusivity drift")
    if not triton_manifest["eager_correctness"].get("formal_pass") or not all(
        row.get("formal_pass")
        for row in triton_manifest.get("replay_correctness", {}).values()
    ):
        raise RuntimeError("Triton NCU correctness gate failed")
    topology_roles = [
        row.get("role")
        for row in triton_manifest["expected_graph_topology"].get("nodes", [])
    ]
    if topology_roles != list(ROLES):
        raise RuntimeError("Triton NCU 12-node topology drift")

    launches_value = read_json(root / "triton_material" / "launches.json")
    if launches_value.get("command") != "ncu.launches" or "error" in launches_value:
        raise RuntimeError("invalid Triton VeloQ NCU launch inventory")
    launches = launches_value.get("data", {}).get("rows", [])
    if len(launches) != 4:
        raise RuntimeError("expected four Triton material NCU launches")
    triton_launches: dict[str, Any] = {}
    for index, role in enumerate(NCU_ROLES):
        launch = launches[index]
        if NCU_ROLE_TOKENS[role] not in launch["kernel_demangled"]:
            raise RuntimeError(f"Triton NCU launch order drift at {role}")
        inspect_path = root / "triton_material" / f"inspect_{index}.json"
        inspect = ncu_inspect_row(inspect_path)
        for field in ("kernel_demangled", "grid_size", "block_size"):
            if inspect[field] != launch[field]:
                raise RuntimeError(f"Triton NCU inspect/inventory drift: {role}")
        metrics = selected_ncu_metrics(inspect, path=inspect_path)
        if (
            metrics["dynamic_spill_load_instructions"]["value"] != 0
            or metrics["dynamic_spill_store_instructions"]["value"] != 0
        ):
            raise RuntimeError(f"unexpected dynamic spill in Triton {role}")
        triton_launches[role] = {
            "kernel_demangled": launch["kernel_demangled"],
            "grid_size": launch["grid_size"],
            "block_size": launch["block_size"],
            "metrics": metrics,
            "pc_sample_stalls": pc_sample_stall_evidence(
                root / "triton_material" / f"pc_stalls_file_{index}.json"
            ),
        }

    capture_version = (results / "runtime" / "ncu.capture_version.txt").read_text()
    if "Version 2025.3.1.0" not in capture_version:
        raise RuntimeError("NCU capture version drift")
    veloq_version = (root / "version.txt").read_text().strip()
    if veloq_version != "veloq 0.2.2":
        raise RuntimeError("VeloQ version drift")
    opt_summary = read_json(root / "opt_fused" / "summary.json")
    triton_summary = read_json(root / "triton_material" / "summary.json")
    for label, summary, count in (
        ("opt", opt_summary, 1),
        ("triton", triton_summary, 4),
    ):
        if (
            summary.get("command") != "ncu.summary"
            or "error" in summary
            or summary.get("data", {}).get("rows", [{}])[0].get("launch_count")
            != count
        ):
            raise RuntimeError(f"{label} VeloQ NCU summary drift")

    payload = {
        "schema": "exp017.ncu-launch-evidence.v1",
        "evidence_identity": {
            "benchmark_and_phase_gpu_uuid": BENCHMARK_GPU_UUID,
            "ncu_gpu_uuid": NCU_GPU_UUID,
            "application_clock_mhz": 2377,
            "ncu_capture_version": "2025.3.1.0",
            "ncu_reader_version": opt_summary["data"]["auxiliary"]["ncu_version"],
            "veloq_version": veloq_version,
            "capture_semantics": capture_semantics,
            "duration_authority": False,
        },
        "comparison_contract": {
            "mode": "component-reference / launch-local",
            "allowed": "per-launch normalized utilization, stall samples, and Resource",
            "forbidden": "cross-launch ratio rollup or NCU duration as benchmark",
            "whole_operator_additive_counts": {
                "status": "missing",
                "reason": (
                    "VeloQ 0.2.2 does not project complete graph-range metrics; "
                    "per-launch rows are not summed into operator traffic."
                ),
            },
        },
        "opt_fused": {
            "kernel_demangled": opt_row["kernel_demangled"],
            "grid_size": opt_row["grid_size"],
            "block_size": opt_row["block_size"],
            "cubin_sha256": opt_manifest["cubin_sha256"],
            "static_compiler_stack_bytes_per_thread": static_resource[
                "stack_bytes_per_thread"
            ],
            "metrics": opt_metrics,
            "pc_sample_stalls": opt_stalls,
            "scatter_pc": opt_scatter,
        },
        "triton_material_launches": triton_launches,
        "source_files": {
            "opt_manifest": str(opt_manifest_path.relative_to(results)),
            "opt_command": str(opt_command_path.relative_to(results)),
            "triton_manifest": str(triton_manifest_path.relative_to(results)),
            "triton_command": str(triton_command_path.relative_to(results)),
            "veloq_root": "veloq/ncu",
            "opt_report_sha256": file_sha256(
                results / "raw" / "ncu" / "opt_fused_deep" / "trace.ncu-rep"
            ),
            "triton_report_sha256": file_sha256(
                results
                / "raw"
                / "ncu"
                / "triton_material_deep"
                / "trace.ncu-rep"
            ),
        },
    }
    payload["fingerprint_sha256"] = canonical_sha256(payload)
    return payload


def load_or_build_ncu_evidence(results: Path) -> dict[str, Any]:
    if (results / "veloq" / "ncu" / "opt_fused" / "inspect.json").is_file():
        return build_ncu_evidence(results)
    path = results / "ncu_evidence.json"
    value = read_json(path)
    validate_fingerprint(value, label="compact NCU evidence")
    if value.get("schema") != "exp017.ncu-launch-evidence.v1":
        raise RuntimeError("compact NCU evidence schema drift")
    return value


def opt_evidence(results: Path) -> dict[str, Any]:
    root = results / "raw" / "opt_phase"
    control = read_json(root / "control.json")
    probe = read_json(root / "probe.json")
    validate_correct_capture(control, mode="control_no_marker")
    validate_correct_capture(probe, mode="probe")
    identity_fields = ("fixture_identity", "weight_identity")
    if any(control[field] != probe[field] for field in identity_fields):
        raise RuntimeError("opt control/probe fixture or weights differ")
    if control["case"]["scale_kind"] != "equal":
        raise RuntimeError("opt capture is not the canonical exp_001 scale contract")

    control_us = float(control["summary"]["event_elapsed_us"]["median"])
    probe_us = float(probe["summary"]["event_elapsed_us"]["median"])
    overhead_percent = (probe_us / control_us - 1.0) * 100.0
    if overhead_percent > 5.0:
        raise RuntimeError(f"phase probe overhead exceeds 5%: {overhead_percent}")

    def resource(value: Mapping[str, Any]) -> dict[str, Any]:
        rows = value["static_resource_usage"]["records"]
        if len(rows) != 1:
            raise RuntimeError("expected one opt resource record")
        return {
            key: rows[0][key]
            for key in (
                "kernel_symbol",
                "registers_per_thread",
                "stack_bytes_per_thread",
                "static_shared_bytes_per_cta",
                "static_local_bytes_outside_stack",
            )
        }

    resources = {"control": resource(control), "probe": resource(probe)}
    if resources["control"] != resources["probe"]:
        raise RuntimeError("opt control/probe resource drift")
    if resources["probe"]["stack_bytes_per_thread"] != 0:
        raise RuntimeError("opt phase probe introduced stack allocation")

    replay_rows: list[dict[str, dict[str, float]]] = []
    for run in probe["runs"]:
        timing = run["phase_timing"]
        if timing["closure_error_ns"] != 0 or timing["share_sum_percent"] != 100.0:
            raise RuntimeError("opt phase closure failed")
        replay_rows.append({row["name"]: row for row in timing["phase_rows"]})

    phase_stats = {}
    for name in replay_rows[0]:
        phase_stats[name] = {
            "time_us": median_range(
                rows[name]["equivalent_wall_us"] for rows in replay_rows
            ),
            "own_share_percent": median_range(
                rows[name]["share_percent"] for rows in replay_rows
            ),
        }

    groups: dict[str, Any] = {}
    for group, members in OPT_GROUPS.items():
        times = [
            sum(rows[name]["equivalent_wall_us"] for name in members)
            for rows in replay_rows
        ]
        shares = [
            sum(rows[name]["share_percent"] for name in members)
            for rows in replay_rows
        ]
        groups[group] = {
            "members": list(members),
            "time_us": median_range(times),
            "own_share_percent": median_range(shares),
        }

    binary_root = root / "binary"
    parsed: dict[str, list[dict[str, Any]]] = {}
    projections: dict[str, Any] = {}
    for mode in ("control_no_marker", "probe"):
        sass_path = binary_root / mode / "nvdisasm.sass"
        elf_path = binary_root / mode / "elf.txt"
        instructions = parse_instructions(sass_path.read_text(errors="replace"))
        counts = Counter(str(row["opcode"]) for row in instructions)
        local = {
            opcode: count
            for opcode, count in sorted(counts.items())
            if opcode.startswith(("LDL", "STL"))
        }
        annotations = SPILL_ANNOTATION_RE.findall(elf_path.read_text(errors="replace"))
        if local or annotations:
            raise RuntimeError(f"{mode} contains static spill/refill evidence")
        parsed[mode] = instructions
        projections[mode] = _selected_projection(counts)
    sequence_projection = opcode_projection(
        parsed["control_no_marker"], parsed["probe"]
    )
    semantic_fields = ("omma", "utmaldg", "ldsm", "bar", "atomg", "redg")
    semantic_work_equal = all(
        projections["control_no_marker"][field] == projections["probe"][field]
        for field in semantic_fields
    )
    if not semantic_work_equal:
        raise RuntimeError("opt probe changed selected semantic-work counts")

    return {
        "classification": "diagnostic projection; not production-exact phase latency",
        "fixture": control["fixture_identity"],
        "weights": control["weight_identity"],
        "control_event_us": control["summary"]["event_elapsed_us"],
        "probe_event_us": probe["summary"]["event_elapsed_us"],
        "probe_overhead_percent": overhead_percent,
        "probe_grid_critical_us": median_range(
            float(run["phase_timing"]["grid_critical_wall_us"])
            for run in probe["runs"]
        ),
        "phases": phase_stats,
        "groups": groups,
        "resources": resources,
        "static_spill": {
            "control_local_sass": 0,
            "probe_local_sass": 0,
            "control_compiler_annotations": 0,
            "probe_compiler_annotations": 0,
        },
        "codegen_audit": {
            "selected_semantic_work_equal": semantic_work_equal,
            "selected_projection": projections,
            "insertion_only_opcode_projection": sequence_projection[
                "insertion_only_opcode_projection"
            ],
            "branch_target_projection_pass": sequence_projection[
                "branch_target_projection_pass"
            ],
            "production_exact_gate": False,
            "reason": (
                "marker specialization preserves selected OMMA/TMA/LDSM/barrier/"
                "atomic/reduction counts and resources, but causes broader SASS "
                "scheduling/control-flow reordering"
            ),
        },
        "source_files": {
            "control": str((root / "control.json").relative_to(results)),
            "probe": str((root / "probe.json").relative_to(results)),
        },
    }


def triton_evidence(results: Path) -> dict[str, Any]:
    manifest = read_json(results / "triton_capture_manifest.json")
    preflight = read_json(results / "triton_topology_preflight.json")
    validate_fingerprint(manifest, label="Triton capture manifest")
    validate_fingerprint(preflight, label="Triton topology preflight")
    if manifest.get("artifact_stable_during_capture") is not True:
        raise RuntimeError("Triton JIT artifacts changed during capture")
    if manifest["fixture"]["fixture_sha256"] != EXPECTED_FIXTURE_SHA256:
        raise RuntimeError("Triton fixture identity drift")
    if manifest["fixture"]["occupancy_sha256"] != EXPECTED_OCCUPANCY_SHA256:
        raise RuntimeError("Triton occupancy identity drift")
    if not manifest["eager_correctness"]["formal_pass"] or not all(
        value["formal_pass"] for value in manifest["replay_correctness"].values()
    ):
        raise RuntimeError("Triton correctness gate failed")

    graph = read_json(results / "veloq" / "triton_graph_replays.json")
    if graph.get("command") != "nsys.graph-replays":
        raise RuntimeError("wrong VeloQ evidence command")
    rows = graph["data"]["rows"]
    if len(rows) != 5:
        raise RuntimeError(f"expected five Triton replays, found {len(rows)}")
    replay_rows: list[dict[str, float]] = []
    node_ids: list[list[int]] = []
    for replay in rows:
        if (
            replay["kernel_count"] != 12
            or replay["event_count"] != 12
            or replay["stream_count"] != 1
            or not replay["decomposition_available"]
        ):
            raise RuntimeError("Triton replay topology/count drift")
        nodes = sorted(replay["top_nodes"], key=lambda row: row["start_ns"])
        if len(nodes) != 12:
            raise RuntimeError("VeloQ output did not retain all 12 graph nodes")
        values: dict[str, float] = {}
        previous_end = replay["start_ns"]
        current_ids = []
        for role, node in zip(ROLES, nodes, strict=True):
            if ROLE_TOKENS[role] not in node["name"]:
                raise RuntimeError(f"Triton role drift at {role}: {node['name']}")
            if node["start_ns"] < previous_end:
                raise RuntimeError("Triton graph nodes overlap unexpectedly")
            previous_end = node["end_ns"]
            current_ids.append(int(node["graph_node_id"]))
            values[role] = float(node["sum_ns"]) / 1000.0
        if previous_end != replay["end_ns"]:
            raise RuntimeError("last Triton node does not close replay span")
        bubble_ns = int(replay["wall_ns"]) - sum(int(node["sum_ns"]) for node in nodes)
        if bubble_ns != int(replay["idle_inside_replay_ns"]):
            raise RuntimeError("Triton bubble closure mismatch")
        values["graph_node_bubble"] = bubble_ns / 1000.0
        values["graph_wall"] = float(replay["wall_ns"]) / 1000.0
        replay_rows.append(values)
        node_ids.append(current_ids)
    if any(ids != node_ids[0] for ids in node_ids[1:]):
        raise RuntimeError("Triton graph node identity changed across replays")

    groups: dict[str, Any] = {}
    for group, roles in TRITON_GROUPS.items():
        times = [sum(row[role] for role in roles) for row in replay_rows]
        shares = [100.0 * time / row["graph_wall"] for time, row in zip(times, replay_rows, strict=True)]
        groups[group] = {
            "members": list(roles),
            "time_us": median_range(times),
            "own_share_percent": median_range(shares),
        }
    ops = {}
    for role in (*ROLES, "graph_node_bubble"):
        times = [row[role] for row in replay_rows]
        shares = [
            100.0 * row[role] / row["graph_wall"] for row in replay_rows
        ]
        ops[role] = {
            "time_us": median_range(times),
            "own_share_percent": median_range(shares),
        }
    return {
        "classification": "NSys CUDA Graph node elapsed",
        "fixture": manifest["fixture"],
        "runtime": manifest["runtime_identity"],
        "launch_contract": manifest["launch_contract"],
        "graph_wall_us": median_range(row["graph_wall"] for row in replay_rows),
        "ops": ops,
        "groups": groups,
        "topology": {
            "replays": 5,
            "nodes_per_replay": 12,
            "node_ids": node_ids[0],
            "gate_pass": True,
        },
        "source_files": {
            "manifest": "triton_capture_manifest.json",
            "veloq": "veloq/triton_graph_replays.json",
        },
    }


def csv_row(path: Path, *, arm: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["m"] == "8192" and row["arm"] == arm
        ]
    if len(rows) != 1 or rows[0]["stable_le_5_percent"].lower() != "true":
        raise RuntimeError(f"missing stable exp_001 row for {arm}")
    return rows[0]


def whole_op_context() -> dict[str, Any]:
    opt = csv_row(EXP001 / "pair" / "benchmark_summary.csv", arm="cutedsl_bf16_fused")
    triton = csv_row(
        EXP001 / "sglang_triton" / "benchmark_summary.csv",
        arm="sglang_triton_fp8",
    )
    for field, expected in (
        ("fixture_sha256", EXPECTED_FIXTURE_SHA256),
        ("occupancy_sha256", EXPECTED_OCCUPANCY_SHA256),
    ):
        if opt[field] != expected or triton[field] != expected:
            raise RuntimeError(f"exp_001 {field} drift")
    if opt["rerun_id"] != triton["rerun_id"]:
        raise RuntimeError("exp_001 opt/Triton rerun identity differs")
    opt_us = float(opt["median_us"])
    triton_us = float(triton["median_us"])
    target_ratio = 2.0
    target_opt_us = triton_us / target_ratio
    latency_gap_us = max(0.0, opt_us - target_opt_us)
    return {
        "opt_us": opt_us,
        "triton_us": triton_us,
        "opt_speedup_percent": (triton_us / opt_us - 1.0) * 100.0,
        "latency_ratio": triton_us / opt_us,
        "customer_2x_target": {
            "target_ratio": target_ratio,
            "target_opt_us": target_opt_us,
            "latency_gap_us": latency_gap_us,
            "required_reduction_percent_of_current": latency_gap_us / opt_us * 100.0,
            "target_met": opt_us <= target_opt_us,
        },
        "rerun_id": opt["rerun_id"],
        "fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "occupancy_sha256": EXPECTED_OCCUPANCY_SHA256,
        "source_files": {
            "opt": str(
                (EXP001 / "pair" / "benchmark_summary.csv").relative_to(ROOT.parents[3])
            ),
            "triton": str(
                (
                    EXP001
                    / "sglang_triton"
                    / "benchmark_summary.csv"
                ).relative_to(ROOT.parents[3])
            ),
        },
    }


def write_outputs(results: Path, payload: Mapping[str, Any]) -> None:
    (results / "ncu_evidence.json").write_text(
        json.dumps(payload["ncu"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evidence_path = results / "evidence.json"
    evidence_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (results / "phase_op.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "logical_region",
                "opt_time_us_median",
                "opt_own_share_percent_median",
                "triton_time_us_median",
                "triton_own_share_percent_median",
            )
        )
        for region in OPT_GROUPS:
            opt = payload["opt"]["groups"][region]
            triton = payload["triton"]["groups"][region]
            writer.writerow(
                (
                    region,
                    opt["time_us"]["median"],
                    opt["own_share_percent"]["median"],
                    triton["time_us"]["median"],
                    triton["own_share_percent"]["median"],
                )
            )


def build(results: Path) -> dict[str, Any]:
    opt = opt_evidence(results)
    triton = triton_evidence(results)
    ncu = load_or_build_ncu_evidence(results)
    if opt["fixture"]["fixture_sha256"] != triton["fixture"]["fixture_sha256"]:
        raise RuntimeError("cross-runtime fixture mismatch")
    whole = whole_op_context()
    crosscheck = {
        "opt_control_vs_exp001_percent": (
            opt["control_event_us"]["median"] / whole["opt_us"] - 1.0
        )
        * 100.0,
        "triton_nsys_vs_exp001_percent": (
            triton["graph_wall_us"]["median"] / whole["triton_us"] - 1.0
        )
        * 100.0,
    }
    payload = {
        "schema": "exp017.phase-op-evidence.v1",
        "case": {
            "m": 8192,
            "experts": 256,
            "hidden": 2048,
            "intermediate_tp": 512,
            "topk": 8,
        },
        "whole_op_context": whole,
        "opt": opt,
        "triton": triton,
        "ncu": ncu,
        "crosscheck": crosscheck,
        "comparison_contract": {
            "row_values": "side-specific time and own-side share only",
            "forbidden": "row-level speedup, causal equivalence, production-exact opt phase claim",
            "precision": "CuteDSL NVFP4 fused vs SGLang E4M3 FP8 chain",
        },
        "gates": {
            "same_fixture_and_routing": True,
            "correctness": True,
            "five_replays_each": True,
            "opt_probe_overhead_le_5_percent": True,
            "opt_resource_identity_and_no_static_spill": True,
            "triton_ordered_12_node_topology": True,
            "ncu_same_sibling_gpu_and_clock": True,
            "ncu_correctness_and_launch_identity": True,
            "ncu_launch_local_only": True,
            "production_exact_opt_phase": False,
            "reader_classification": "diagnostic projection",
        },
    }
    payload["evidence_sha256"] = canonical_sha256(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument(
        "--ncu-only",
        action="store_true",
        help="build the compact NCU evidence card from VeloQ/raw NCU inputs",
    )
    args = parser.parse_args(argv)
    results = args.results.resolve()
    if args.ncu_only:
        ncu = build_ncu_evidence(results)
        (results / "ncu_evidence.json").write_text(
            json.dumps(ncu, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "schema": ncu["schema"],
                    "fingerprint_sha256": ncu["fingerprint_sha256"],
                }
            )
        )
        return 0
    payload = build(results)
    write_outputs(results, payload)
    print(json.dumps({"schema": payload["schema"], "gates": payload["gates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
