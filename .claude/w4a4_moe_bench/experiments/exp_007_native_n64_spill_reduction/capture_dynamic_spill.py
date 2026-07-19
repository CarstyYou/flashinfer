#!/usr/bin/env python3
"""Capture one identity-locked M8192 dynamic-spill NCU report for exp_007."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


ARM_CONFIG = {
    "anchor": {
        "internal": "candidate_8warp_serial_v0",
        "overlay": "anchor_8warp_n128",
    },
    "candidate": {
        "internal": "candidate_8warp_n64_temporal_replay_v0",
        "overlay": "candidate_8warp_native_n64_v0",
    },
}

SECTION_IDS = (
    "InstructionStats",
    "SourceCounters",
    "LaunchStats",
    "Occupancy",
)

SPILL_METRICS = {
    "spill_refill_instructions": "sass__inst_executed_register_spilling_op_read",
    "spill_store_instructions": "sass__inst_executed_register_spilling_op_write",
    "spill_refill_bytes": ("sass__inst_executed_register_spilling_mem_local_op_read"),
    "spill_store_bytes": ("sass__inst_executed_register_spilling_mem_local_op_write"),
}

CUSTOM_METRICS = {
    "local_load_bytes": "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
    "local_store_bytes": "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
    "tensor_instructions": "sm__inst_executed_pipe_tensor_subpipe_hmma.sum",
    "fp4_tensor_ops": "sm__ops_path_tensor_src_fp4_dst_fp32.sum",
    "registers_per_thread": "launch__registers_per_thread",
    "allocated_registers_per_thread": "launch__registers_per_thread_allocated",
    "shared_mem_per_block_bytes": "launch__shared_mem_per_block",
    # Runtime configured limit; this is deliberately not used as static STACK.
    "configured_stack_limit_bytes": "launch__stack_size",
    "achieved_occupancy_pct": ("sm__warps_active.avg.pct_of_peak_sustained_active"),
}

ALL_METRICS = {**SPILL_METRICS, **CUSTOM_METRICS}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def tool_version(executable: str) -> str:
    return subprocess.check_output(
        [executable, "--version"], text=True, stderr=subprocess.STDOUT
    ).strip()


def parse_number(value: str, *, metric: str) -> float:
    if value in ("", "n/a"):
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


def parse_native_raw(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
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
    for label, metric_id in ALL_METRICS.items():
        if metric_id not in by_name:
            raise ValueError(f"required NCU metric absent: {metric_id}")
        metrics[label] = {
            "metric_id": metric_id,
            "value": parse_number(by_name[metric_id], metric=metric_id),
            "unit": by_unit[metric_id],
        }

    kernel = by_name.get("Kernel Name", "")
    if "MoEDynamicKernel" not in kernel:
        raise ValueError(f"unexpected kernel: {kernel!r}")
    identity = {
        "row_id": int(by_name["ID"]),
        "kernel": kernel,
        "context_id": int(by_name["Context"]),
        "stream_id": int(by_name["Stream"]),
        "device_id": int(by_name["Device"]),
        "block": parse_dim(by_name["Block Size"], field="Block Size"),
        "grid": parse_dim(by_name["Grid Size"], field="Grid Size"),
    }
    return metrics, identity


def find_exact_cubin(jit_root: Path, expected_hashes: list[str]) -> Path:
    cubins = sorted(jit_root.rglob("*.cubin"))
    matches = [path for path in cubins if sha256_file(path) in expected_hashes]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one preparation cubin, got {len(matches)} from {cubins}"
        )
    return matches[0]


def capture(args: argparse.Namespace) -> None:
    arm = ARM_CONFIG[args.arm]
    root = args.flashinfer_root.resolve()
    exp = root / ".claude/w4a4_moe_bench/experiments/exp_007_native_n64_spill_reduction"
    results = exp / "results"
    canonical = results / "canonical"
    internal = str(arm["internal"])
    overlay = results / "overlays" / str(arm["overlay"]) / "moe_dynamic_kernel.py"
    preparation_path = (
        canonical / "raw" / internal / "m8192" / "canonical" / "preparation.json"
    )
    preparation = read_json(preparation_path)
    if not (
        preparation.get("status") == "complete"
        and preparation.get("arm") == internal
        and preparation.get("m") == 8192
        and preparation.get("fixture_kind") == "canonical"
    ):
        raise RuntimeError(f"invalid M8192 preparation: {preparation_path}")
    source_identity = preparation["runtime"]["source"]
    if sha256_file(overlay) != source_identity["overlay_sha256"]:
        raise RuntimeError("overlay hash drift from preparation")
    expected_cubin_hashes = list(preparation["cubin_sha256"])
    cubin = find_exact_cubin(args.jit_root.resolve(), expected_cubin_hashes)

    output = results / "ncu" / args.arm / "m8192" / "canonical_v0"
    if output.exists():
        raise FileExistsError(f"immutable NCU capture already exists: {output}")
    temporary = output.with_name(f".{output.name}.in-progress.{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    report_base = temporary / "trace"

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
        "--kernel-name",
        "regex:MoEDynamicKernel",
        "--kernel-name-base",
        "demangled",
        "--launch-count",
        "1",
        *selection,
        "--export",
        str(report_base),
        *target,
    ]

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
        metrics, observed = parse_native_raw(native_csv)
        if observed["grid"] != [1, 1, 110] or observed["block"] != [288, 1, 1]:
            raise RuntimeError(f"profiled launch geometry drift: {observed}")

        spill_values = {name: metrics[name]["value"] for name in SPILL_METRICS}
        dynamic_zero_spill = all(value == 0 for value in spill_values.values())
        profile_target = (
            canonical / "profile_targets" / internal / "m8192" / "target.json"
        )
        target_payload = read_json(profile_target)
        if target_payload.get("jit_artifact_set_sha256") != preparation.get(
            "jit_artifact_set_sha256"
        ):
            raise RuntimeError("profile target JIT identity drift")

        evidence = {
            "schema": "exp007.dynamic-spill.v1",
            "arm": args.arm,
            "internal_arm": internal,
            "m": 8192,
            "fixture": "canonical",
            "metrics": metrics,
            "observed_launch": observed,
            "gates": {
                "all_required_metrics_present_and_numeric": True,
                "dynamic_zero_spill": dynamic_zero_spill,
            },
            "evidence_boundary": (
                "Dynamic counts apply only to the one identity-locked final CUDA "
                "Graph replay node; configured stack limit is not static STACK."
            ),
        }
        write_json(temporary / "dynamic_spill.json", evidence)
        identity = {
            "schema": "exp007.ncu-capture-identity.v1",
            "arm": args.arm,
            "internal_arm": internal,
            "overlay_sha256": sha256_file(overlay),
            "preparation_path": str(preparation_path),
            "preparation_sha256": sha256_file(preparation_path),
            "jit_artifact_set_sha256": preparation["jit_artifact_set_sha256"],
            "cubin_path": str(cubin),
            "cubin_sha256": sha256_file(cubin),
            "profile_target_path": str(profile_target),
            "profile_target_sha256": sha256_file(profile_target),
            "trace_sha256": sha256_file(report),
            "native_raw_sha256": sha256_file(native_csv),
            "ncu_version": tool_version(args.ncu),
            "section_ids": list(SECTION_IDS),
            "custom_metric_ids": list(CUSTOM_METRICS.values()),
            "required_spill_metric_ids": list(SPILL_METRICS.values()),
            "observed_launch": observed,
        }
        write_json(temporary / "capture_identity.json", identity)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--arm", choices=tuple(ARM_CONFIG), required=True)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    capture(args)
    print(json.dumps({"status": "complete", "arm": args.arm, "m": 8192}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
