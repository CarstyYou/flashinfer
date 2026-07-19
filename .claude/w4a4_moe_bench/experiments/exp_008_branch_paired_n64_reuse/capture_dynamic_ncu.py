#!/usr/bin/env python3
"""Capture one identity-locked M8192 dynamic-work NCU report for exp_008.

``--dry-run`` performs only CPU-side preparation/JIT/static-evidence checks and
prints the exact native NCU command.  Capture mode profiles one final CUDA Graph
replay node selected by the exact generated MoEDynamicKernel symbol.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Sequence


ARM_CONFIG = {
    "n128": {
        "internal": "candidate_8warp_serial_v0",
        "overlay": "anchor_8warp_n128",
    },
    "v0": {
        "internal": "candidate_8warp_n64_temporal_replay_v0",
        "overlay": "temporal_n64_v0",
    },
    "v1": {
        "internal": "candidate_8warp_n64_temporal_replay_v0",
        "overlay": "branch_paired_n64_v1",
    },
}

# InstructionStats + SourceCounters produce the four compiler-classified
# SpillRefill metrics.  LaunchStats carries the launch resources.  The two
# tensor-work counters are requested explicitly.
SECTION_IDS = ("InstructionStats", "SourceCounters", "LaunchStats")
SPILL_METRICS = {
    "spill_refill_instructions": "sass__inst_executed_register_spilling_op_read",
    "spill_store_instructions": "sass__inst_executed_register_spilling_op_write",
    "spill_refill_bytes": "sass__inst_executed_register_spilling_mem_local_op_read",
    "spill_store_bytes": "sass__inst_executed_register_spilling_mem_local_op_write",
}
CUSTOM_METRICS = {
    "tensor_instructions": "sm__inst_executed_pipe_tensor_subpipe_hmma.sum",
    "fp4_tensor_ops": "sm__ops_path_tensor_src_fp4_dst_fp32.sum",
    "registers_per_thread": "launch__registers_per_thread",
    "shared_mem_per_block_bytes": "launch__shared_mem_per_block",
}
ALL_REQUIRED_METRICS = {**SPILL_METRICS, **CUSTOM_METRICS}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def tool_version(executable: str) -> str:
    return subprocess.check_output(
        [executable, "--version"], text=True, stderr=subprocess.STDOUT
    ).strip()


def parse_number(value: str, *, metric: str) -> float:
    normalized = value.strip().lower()
    if normalized in ("", "n/a", "na", "not available"):
        raise ValueError(f"missing/N-A required metric {metric}")
    parsed = float(value.replace(",", ""))
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite required metric {metric}: {value}")
    return parsed


def parse_dim(value: str, *, field: str) -> list[int]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, tuple) or len(parsed) != 3:
        raise ValueError(f"invalid {field}: {value!r}")
    result = [int(item) for item in parsed]
    if any(item <= 0 for item in result):
        raise ValueError(f"invalid {field}: {result}")
    return result


def parse_native_raw(
    path: Path, *, expected_kernel_symbol: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    header_rows = [index for index, row in enumerate(rows) if row and row[0] == "ID"]
    if len(header_rows) != 1:
        raise ValueError(f"expected one NCU header, got {header_rows}")
    index = header_rows[0]
    if len(rows) < index + 3:
        raise ValueError("NCU raw export lacks unit/value rows")
    header, units = rows[index : index + 2]
    value_rows = [row for row in rows[index + 2 :] if any(row)]
    if len(value_rows) != 1:
        raise ValueError(f"expected one profiled launch, got {len(value_rows)}")
    values = value_rows[0]
    if not (len(header) == len(units) == len(values)):
        raise ValueError("NCU raw CSV width mismatch")
    by_name = dict(zip(header, values, strict=True))
    by_unit = dict(zip(header, units, strict=True))

    metrics: dict[str, Any] = {}
    for label, metric_id in ALL_REQUIRED_METRICS.items():
        if metric_id not in by_name:
            raise ValueError(f"required NCU metric absent: {metric_id}")
        metrics[label] = {
            "metric_id": metric_id,
            "value": parse_number(by_name[metric_id], metric=metric_id),
            "unit": by_unit[metric_id],
        }

    kernel = by_name.get("Kernel Name", "")
    if kernel != expected_kernel_symbol:
        raise ValueError(
            f"profiled kernel symbol drift: {kernel!r} != {expected_kernel_symbol!r}"
        )
    identity = {
        "row_id": int(by_name["ID"]),
        "kernel_symbol": kernel,
        "context_id": int(by_name["Context"]),
        "stream_id": int(by_name["Stream"]),
        "device_id": int(by_name["Device"]),
        "block": parse_dim(by_name["Block Size"], field="Block Size"),
        "grid": parse_dim(by_name["Grid Size"], field="Grid Size"),
    }
    return metrics, identity


def find_exact_cubin(jit_root: Path, expected_sha256: str) -> Path:
    cubins = sorted(jit_root.rglob("*.cubin"))
    matches = [path for path in cubins if sha256_file(path) == expected_sha256]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one preparation cubin, got {len(matches)} from {cubins}"
        )
    return matches[0]


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    arm = ARM_CONFIG[args.arm]
    root = args.flashinfer_root.resolve()
    exp = root / ".claude/w4a4_moe_bench/experiments/exp_008_branch_paired_n64_reuse"
    results = exp / "results"
    internal = str(arm["internal"])
    canonical = results / "canonical" / args.arm
    overlay = results / "overlays" / str(arm["overlay"]) / "moe_dynamic_kernel.py"
    preparation_path = canonical / "raw" / internal / "m8192/canonical/preparation.json"
    preparation = read_json(preparation_path)
    if not (
        preparation.get("status") == "complete"
        and preparation.get("arm") == internal
        and preparation.get("m") == 8192
        and preparation.get("case", {}).get("m") == 8192
        and preparation.get("fixture_kind") == "canonical"
    ):
        raise RuntimeError(f"invalid M8192 preparation: {preparation_path}")
    source_identity = preparation.get("runtime", {}).get("source", {})
    overlay_sha256 = sha256_file(overlay)
    if overlay_sha256 != source_identity.get("overlay_sha256"):
        raise RuntimeError("overlay hash drift from preparation")
    expected_cubin_hashes = preparation.get("cubin_sha256", [])
    if len(expected_cubin_hashes) != 1:
        raise RuntimeError(
            f"expected one preparation cubin SHA: {expected_cubin_hashes}"
        )
    cubin = find_exact_cubin(args.jit_root.resolve(), expected_cubin_hashes[0])

    static_path = results / "static_spill_evidence.json"
    static = read_json(static_path)
    if not static.get("gate_pass"):
        raise RuntimeError("static spill evidence gate is not PASS")
    static_case = static.get("cases", {}).get(f"{args.arm}_m8192")
    if not isinstance(static_case, dict) or not static_case.get("evidence_gate"):
        raise RuntimeError(f"missing/invalid static evidence for {args.arm}_m8192")
    if static_case.get("identity", {}).get("cubin_sha256") != expected_cubin_hashes[0]:
        raise RuntimeError("static/preparation cubin identity drift")
    expected_kernel_symbol = str(static_case["identity"]["kernel_symbol"])
    expected_resource = static_case["resource"]

    preparation_gpu = preparation.get("runtime", {}).get("gpu", {})
    if preparation_gpu.get("uuid") != args.expected_gpu_uuid:
        raise RuntimeError("preparation GPU UUID drift")
    if int(preparation_gpu.get("applications_graphics_clock_mhz", -1)) != int(
        args.expected_app_clock_mhz
    ):
        raise RuntimeError("preparation application graphics clock drift")

    output = results / "ncu" / args.arm / "m8192/canonical_v0"
    temporary_display = output.with_name(f".{output.name}.in-progress.PID")
    report_base = temporary_display / "trace"
    worker = (
        root
        / ".claude/w4a4_moe_bench/experiments/exp_005_8warp_spill_reduction/run_exp005_arm.py"
    )
    target = [
        sys.executable,
        str(worker),
        "--flashinfer-root",
        str(root),
        "--results",
        str(canonical),
        "--arm",
        internal,
        "--m",
        "8192",
        "--fixture",
        "canonical",
        "--overlay",
        str(overlay),
        "--jit-root",
        str(args.jit_root.resolve()),
        "--expected-gpu-uuid",
        args.expected_gpu_uuid,
        "profile",
        "--warmup",
        str(args.warmup),
    ]
    selection = [item for section in SECTION_IDS for item in ("--section", section)]
    selection.extend(("--metrics", ",".join(CUSTOM_METRICS.values())))
    command = [
        args.ncu,
        "--force-overwrite",
        "--profile-from-start",
        "off",
        "--target-processes",
        "all",
        "--graph-profiling",
        "node",
        "--replay-mode",
        "kernel",
        "--cache-control",
        "all",
        "--clock-control",
        "none",
        "--kernel-name",
        f"regex:^{expected_kernel_symbol}$",
        "--kernel-name-base",
        "demangled",
        "--launch-count",
        "1",
        *selection,
        "--export",
        str(report_base),
        *target,
    ]
    return {
        "arm": args.arm,
        "internal_arm": internal,
        "root": root,
        "exp": exp,
        "results": results,
        "canonical": canonical,
        "overlay": overlay,
        "overlay_sha256": overlay_sha256,
        "preparation": preparation,
        "preparation_path": preparation_path,
        "static_path": static_path,
        "static_case": static_case,
        "cubin": cubin,
        "cubin_sha256": expected_cubin_hashes[0],
        "expected_kernel_symbol": expected_kernel_symbol,
        "expected_resource": expected_resource,
        "output": output,
        "command": command,
    }


def validate_profile_target(
    *,
    contract: dict[str, Any],
    target_payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, bool]:
    preparation = contract["preparation"]
    runtime = target_payload.get("runtime", {})
    gpu = runtime.get("gpu", {})
    source = runtime.get("source", {})
    expected_nvtx = f"exp005_{contract['internal_arm']}_m8192_final_replay"
    return {
        "status_complete": target_payload.get("status") == "complete",
        "arm": target_payload.get("arm") == contract["internal_arm"],
        "m": target_payload.get("m") == 8192,
        "canonical_fixture": target_payload.get("fixture_kind") == "canonical",
        "final_replay_nvtx": target_payload.get("nvtx_range") == expected_nvtx,
        "expected_launch": target_payload.get("expected_launch")
        == {"grid": [1, 1, 110], "block": [288, 1, 1], "kernel": "MoEDynamicKernel"},
        "jit_artifact_set_sha256": target_payload.get("jit_artifact_set_sha256")
        == preparation.get("jit_artifact_set_sha256"),
        "overlay_sha256": source.get("overlay_sha256") == contract["overlay_sha256"],
        "gpu_uuid": gpu.get("uuid") == args.expected_gpu_uuid,
        "application_graphics_clock_mhz": int(
            gpu.get("applications_graphics_clock_mhz", -1)
        )
        == args.expected_app_clock_mhz,
    }


def capture(args: argparse.Namespace) -> None:
    contract = build_contract(args)
    output: Path = contract["output"]
    if output.exists():
        raise FileExistsError(f"immutable NCU capture already exists: {output}")
    temporary = output.with_name(f".{output.name}.in-progress.{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    report_base = temporary / "trace"
    command = list(contract["command"])
    planned_report_base = str(
        output.with_name(f".{output.name}.in-progress.PID") / "trace"
    )
    if command.count(planned_report_base) != 1:
        raise RuntimeError("NCU export placeholder identity drift")
    command[command.index(planned_report_base)] = str(report_base)

    try:
        (temporary / "command.txt").write_text(shlex.join(command) + "\n")
        with (
            (temporary / "stdout.log").open("w") as stdout,
            (temporary / "stderr.log").open("w") as stderr,
        ):
            subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
        report = report_base.with_suffix(".ncu-rep")
        if not report.is_file() or report.stat().st_size == 0:
            raise RuntimeError("NCU produced no report")

        native_csv = temporary / "native_raw.csv"
        export = [
            args.ncu,
            "--import",
            str(report),
            "--csv",
            "--page",
            "raw",
            "--print-units",
            "base",
        ]
        (temporary / "native_raw.command.txt").write_text(shlex.join(export) + "\n")
        with (
            native_csv.open("w") as stdout,
            (temporary / "native_raw.stderr.log").open("w") as stderr,
        ):
            subprocess.run(export, stdout=stdout, stderr=stderr, check=True)
        metrics, observed = parse_native_raw(
            native_csv, expected_kernel_symbol=contract["expected_kernel_symbol"]
        )
        if observed["grid"] != [1, 1, 110] or observed["block"] != [288, 1, 1]:
            raise RuntimeError(f"profiled launch geometry drift: {observed}")
        expected_resource = contract["expected_resource"]
        if metrics["registers_per_thread"]["value"] != float(
            expected_resource["registers_per_thread"]
        ):
            raise RuntimeError("NCU/static registers-per-thread drift")
        if metrics["shared_mem_per_block_bytes"]["value"] != float(
            expected_resource["total_shared_bytes_per_cta"]
        ):
            raise RuntimeError("NCU/static shared-memory drift")
        if metrics["tensor_instructions"]["value"] <= 0:
            raise RuntimeError("executed Tensor instructions must be positive")
        if metrics["fp4_tensor_ops"]["value"] <= 0:
            raise RuntimeError("executed FP4 Tensor ops must be positive")

        profile_target = (
            contract["canonical"]
            / "profile_targets"
            / contract["internal_arm"]
            / "m8192/target.json"
        )
        target_payload = read_json(profile_target)
        target_checks = validate_profile_target(
            contract=contract, target_payload=target_payload, args=args
        )
        if not all(target_checks.values()):
            raise RuntimeError(f"profile target identity drift: {target_checks}")

        spill_values = {name: metrics[name]["value"] for name in SPILL_METRICS}
        dynamic_zero_spill = all(value == 0 for value in spill_values.values())
        evidence = {
            "schema": "exp008.dynamic-ncu.v1",
            "arm": args.arm,
            "internal_arm": contract["internal_arm"],
            "m": 8192,
            "fixture": "canonical",
            "metrics": metrics,
            "observed_launch": observed,
            "profile_target_checks": target_checks,
            "gates": {
                "all_required_metrics_present_and_numeric": True,
                "exact_kernel_symbol": True,
                "launch_geometry": True,
                "launch_resources_match_static": True,
                "tensor_work_positive": True,
                "dynamic_zero_spill": dynamic_zero_spill,
            },
            "evidence_boundary": (
                "Dynamic counters apply only to one identity-locked final CUDA "
                "Graph replay node. Dynamic spill is an observed count; static "
                "STACK/local SASS remain separate compiler evidence."
            ),
        }
        write_json(temporary / "dynamic_ncu.json", evidence)
        identity = {
            "schema": "exp008.ncu-capture-identity.v1",
            "arm": args.arm,
            "internal_arm": contract["internal_arm"],
            "overlay_sha256": contract["overlay_sha256"],
            "preparation_path": str(contract["preparation_path"]),
            "preparation_sha256": sha256_file(contract["preparation_path"]),
            "jit_artifact_set_sha256": contract["preparation"][
                "jit_artifact_set_sha256"
            ],
            "cubin_path": str(contract["cubin"]),
            "cubin_sha256": contract["cubin_sha256"],
            "expected_kernel_symbol": contract["expected_kernel_symbol"],
            "profile_target_path": str(profile_target),
            "profile_target_sha256": sha256_file(profile_target),
            "expected_gpu_uuid": args.expected_gpu_uuid,
            "expected_application_graphics_clock_mhz": args.expected_app_clock_mhz,
            "trace_sha256": sha256_file(report),
            "native_raw_sha256": sha256_file(native_csv),
            "ncu_version": tool_version(args.ncu),
            "clock_control": "none",
            "section_ids": list(SECTION_IDS),
            "explicit_metric_ids": list(CUSTOM_METRICS.values()),
            "required_spill_metric_ids": list(SPILL_METRICS.values()),
            "all_required_metric_ids": list(ALL_REQUIRED_METRICS.values()),
            "profile_target_checks": target_checks,
            "observed_launch": observed,
        }
        write_json(temporary / "capture_identity.json", identity)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def dry_run(args: argparse.Namespace) -> None:
    contract = build_contract(args)
    if contract["output"].exists():
        raise FileExistsError(
            f"immutable NCU capture already exists: {contract['output']}"
        )
    payload = {
        "schema": "exp008.ncu-capture-plan.v1",
        "mode": "dry-run-no-gpu",
        "arm": args.arm,
        "internal_arm": contract["internal_arm"],
        "overlay_sha256": contract["overlay_sha256"],
        "cubin_sha256": contract["cubin_sha256"],
        "kernel_symbol": contract["expected_kernel_symbol"],
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "expected_application_graphics_clock_mhz": args.expected_app_clock_mhz,
        "expected_grid": [1, 1, 110],
        "expected_block": [288, 1, 1],
        "section_ids": list(SECTION_IDS),
        "required_metric_ids": list(ALL_REQUIRED_METRICS.values()),
        "output": str(contract["output"]),
        "command": contract["command"],
        "command_shell": shlex.join(contract["command"]),
        "gpu_or_ncu_invoked": False,
    }
    print(json.dumps(payload, sort_keys=True))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-app-clock-mhz", type=int, required=True)
    parser.add_argument("--arm", choices=tuple(ARM_CONFIG), required=True)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        dry_run(args)
        return 0
    capture(args)
    print(json.dumps({"status": "complete", "arm": args.arm, "m": 8192}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
