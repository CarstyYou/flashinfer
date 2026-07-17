#!/usr/bin/env python3
"""Capture one identity-locked NCU report for an exp_004 arm."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from exp004_common import (
    ALL_ARMS,
    DEFAULT_RESULTS,
    EXPECTED_BLOCK,
    EXPECTED_GRID,
    canonical_sha256,
    file_sha256,
    read_json,
    write_json,
)
from run_exp004 import preparation_path, worker_command, worker_environment


METRIC_IDS = (
    "gpu__time_duration.sum",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
    "sass__inst_executed_register_spilling_op_read",
    "sass__inst_executed_register_spilling_op_write",
    "sass__inst_executed_register_spilling_mem_local_op_read",
    "sass__inst_executed_register_spilling_mem_local_op_write",
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_tensor_subpipe_hmma.sum",
    "sm__ops_path_tensor_src_fp4_dst_fp32.sum",
    "launch__registers_per_thread",
    "launch__registers_per_thread_allocated",
    "launch__shared_mem_per_block",
    "launch__shared_mem_per_block_dynamic",
    "launch__stack_size",
    "launch__waves_per_multiprocessor",
)


def _tool_version(executable: str) -> str:
    completed = subprocess.run(
        [executable, "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if not completed.stdout.strip():
        raise RuntimeError(f"{executable} --version returned no identity")
    return completed.stdout.strip()


def capture(args: argparse.Namespace) -> None:
    results = args.results.resolve()
    prerequisite_path = preparation_path(results, args.arm)
    prerequisite = read_json(prerequisite_path)
    if prerequisite.get("status") != "complete":
        raise RuntimeError(f"incomplete preparation: {prerequisite_path}")

    environment = worker_environment(args, args.arm)
    output = results / "raw" / "ncu" / args.arm
    if output.exists():
        raise FileExistsError(f"immutable NCU capture already exists: {output}")
    temporary = output.with_name(f".{output.name}.in-progress.{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    report_base = temporary / "trace"
    target = worker_command(args, args.arm, ["profile", "--warmup", str(args.warmup)])
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
        "--metrics",
        ",".join(METRIC_IDS),
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
            subprocess.run(
                command,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=True,
            )
        report = report_base.with_suffix(".ncu-rep")
        if not report.is_file() or report.stat().st_size == 0:
            raise RuntimeError("NCU completed without a non-empty report")
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
        native = temporary / "native_raw.csv"
        with (
            native.open("w") as stdout,
            (temporary / "native_raw.stderr.log").open("w") as stderr,
        ):
            subprocess.run(export, stdout=stdout, stderr=stderr, check=True)
        if native.stat().st_size == 0:
            raise RuntimeError("NCU raw export is empty")

        profile_target = results / "profile_targets" / args.arm / "target.json"
        target_payload = read_json(profile_target)
        checks = {
            "schema": target_payload.get("schema") == "exp004.profile-target.v1",
            "status": target_payload.get("status") == "complete",
            "arm": target_payload.get("arm") == args.arm,
            "launch": target_payload.get("expected_launch")
            == {
                "grid": list(EXPECTED_GRID),
                "block": list(EXPECTED_BLOCK),
                "kernel": "MoEDynamicKernel",
            },
            "jit": target_payload.get("jit_artifact_set_sha256")
            == prerequisite.get("jit_artifact_set_sha256"),
        }
        if not all(checks.values()):
            raise RuntimeError(f"profile target identity failed: {checks}")
        identity = {
            "schema": "exp004.ncu-capture-identity.v1",
            "arm": args.arm,
            "expected_grid": list(EXPECTED_GRID),
            "expected_block": list(EXPECTED_BLOCK),
            "preparation": str(prerequisite_path),
            "preparation_sha256": file_sha256(prerequisite_path),
            "cubin_sha256": prerequisite["cubin_sha256"],
            "jit_artifact_set_sha256": prerequisite["jit_artifact_set_sha256"],
            "profile_target": str(profile_target),
            "profile_target_sha256": file_sha256(profile_target),
            "trace_sha256": file_sha256(report),
            "native_raw_sha256": file_sha256(native),
            "ncu_version": _tool_version(args.ncu),
            "collection_protocol": {
                "profile_from_start": "off",
                "graph_profiling": "node",
                "replay_mode": "kernel",
                "cache_control": "all",
                "kernel_filter": "regex:MoEDynamicKernel",
                "launch_count": 1,
            },
            "metric_ids": list(METRIC_IDS),
        }
        identity["identity_sha256"] = canonical_sha256(identity)
        write_json(temporary / "capture_identity.json", identity)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, default=Path.cwd())
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--arm", choices=ALL_ARMS, required=True)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.results = args.results.resolve()
    capture(args)
    print(json.dumps({"status": "complete", "arm": args.arm}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
