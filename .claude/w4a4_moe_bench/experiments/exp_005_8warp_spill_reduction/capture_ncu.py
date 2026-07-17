#!/usr/bin/env python3
"""Capture one identity-locked M=8192 NCU report for an exp_005 arm."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from build_ncu_evidence import METRICS as EVIDENCE_METRICS
from build_ncu_evidence import parse_native_csv
from exp005_common import (
    ALL_ARMS,
    CANONICAL_FIXTURE,
    DEFAULT_RESULTS,
    canonical_sha256,
    expected_block,
    file_sha256,
    read_json,
    write_json,
)
from run_exp005 import preparation_path, worker_command, worker_environment


# The four ``sass__...spilling...`` metrics are NCU source-derived metrics.
# canonical_v0 proved that naming them only via ``--metrics`` is insufficient:
# NCU silently omitted them.  InstructionStats + SourceCounters activate their
# derivation.  The remaining sections lock the compute, memory, occupancy,
# scheduler, warp-state, and launch evidence used by the report.
SECTION_IDS = (
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
    "sass__inst_executed_register_spilling_op_read",
    "sass__inst_executed_register_spilling_op_write",
    "sass__inst_executed_register_spilling_mem_local_op_read",
    "sass__inst_executed_register_spilling_mem_local_op_write",
)

REQUIRED_METRIC_IDS = tuple(EVIDENCE_METRICS.values())
CUSTOM_METRIC_IDS = tuple(
    metric for metric in REQUIRED_METRIC_IDS if metric not in SPILL_METRIC_IDS
)


def metric_selection_args() -> list[str]:
    """Return the NCU section + custom-metric union for canonical_v1."""
    arguments = [item for section in SECTION_IDS for item in ("--section", section)]
    arguments.extend(("--metrics", ",".join(CUSTOM_METRIC_IDS)))
    return arguments


def _tool_version(executable: str) -> str:
    completed = subprocess.run(
        [executable, "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    value = completed.stdout.strip()
    if not value:
        raise RuntimeError(f"{executable} --version returned no identity")
    return value


def _validate_profile_target(
    target: dict[str, object],
    *,
    arm: str,
    prerequisite: dict[str, object],
) -> None:
    expected_launch = {
        "grid": [1, 1, 110],
        "block": list(expected_block(arm)),
        "kernel": "MoEDynamicKernel",
    }
    checks = {
        "schema": target.get("schema") == "exp005.profile-target.v1",
        "status": target.get("status") == "complete",
        "arm": target.get("arm") == arm,
        "m": target.get("m") == 8192,
        "fixture": target.get("fixture_kind") == CANONICAL_FIXTURE,
        "launch": target.get("expected_launch") == expected_launch,
        "jit": target.get("jit_artifact_set_sha256")
        == prerequisite.get("jit_artifact_set_sha256"),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"profile target identity failed: {', '.join(failed)}")


def capture(args: argparse.Namespace) -> None:
    m = 8192
    results = args.results.resolve()
    prerequisite_path = preparation_path(results, args.arm, m, CANONICAL_FIXTURE)
    prerequisite = read_json(prerequisite_path)
    if prerequisite.get("status") != "complete":
        raise RuntimeError(f"incomplete preparation: {prerequisite_path}")

    environment, _, _ = worker_environment(args, args.arm, m, CANONICAL_FIXTURE)
    output = results / "ncu" / args.arm / "m8192" / "canonical_v1"
    if output.exists():
        raise FileExistsError(f"immutable NCU capture already exists: {output}")
    temporary = output.with_name(f".{output.name}.in-progress.{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    output_for_capture = temporary
    report_base = output_for_capture / "trace"
    target = worker_command(
        args,
        args.arm,
        m,
        CANONICAL_FIXTURE,
        ["profile", "--warmup", str(args.warmup)],
    )
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
        *metric_selection_args(),
        "--export",
        str(report_base),
        *target,
    ]
    try:
        (output_for_capture / "command.txt").write_text(shlex.join(command) + "\n")
        with (
            (output_for_capture / "stdout.log").open("w") as stdout,
            (output_for_capture / "stderr.log").open("w") as stderr,
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
        (output_for_capture / "native_raw.command.txt").write_text(
            shlex.join(export) + "\n"
        )
        native_csv = output_for_capture / "native_raw.csv"
        with (
            native_csv.open("w") as stdout,
            (output_for_capture / "native_raw.stderr.log").open("w") as stderr,
        ):
            subprocess.run(export, stdout=stdout, stderr=stderr, check=True)
        if native_csv.stat().st_size == 0:
            raise RuntimeError("NCU raw export is empty")
        # Fail the capture before publishing canonical_v1 if a requested
        # section silently failed to materialize a required metric.  This is
        # deliberately the same strict parser used by the evidence builder.
        parse_native_csv(native_csv)

        profile_target = (
            results / "profile_targets" / args.arm / "m8192" / "target.json"
        )
        target_payload = read_json(profile_target)
        _validate_profile_target(
            target_payload, arm=args.arm, prerequisite=prerequisite
        )
        identity = {
            "schema": "exp005.ncu-capture-identity.v3",
            "capture_revision": "canonical_v1",
            "arm": args.arm,
            "m": m,
            "fixture_kind": CANONICAL_FIXTURE,
            "expected_grid": [1, 1, 110],
            "expected_block": list(expected_block(args.arm)),
            "preparation": str(prerequisite_path),
            "preparation_sha256": file_sha256(prerequisite_path),
            "cubin_sha256": prerequisite["cubin_sha256"],
            "jit_artifact_set_sha256": prerequisite["jit_artifact_set_sha256"],
            "profile_target": str(profile_target),
            "profile_target_sha256": file_sha256(profile_target),
            "trace_sha256": file_sha256(report),
            "native_raw_sha256": file_sha256(native_csv),
            "ncu_version": _tool_version(args.ncu),
            "collection_protocol": {
                "profile_from_start": "off",
                "target_processes": "all",
                "graph_profiling": "node",
                "replay_mode": "kernel",
                "cache_control": "all",
                "kernel_filter": "regex:MoEDynamicKernel",
                "launch_count": 1,
            },
            "section_ids": list(SECTION_IDS),
            "custom_metric_ids": list(CUSTOM_METRIC_IDS),
            "required_metric_ids": list(REQUIRED_METRIC_IDS),
        }
        identity["identity_sha256"] = canonical_sha256(identity)
        write_json(output_for_capture / "capture_identity.json", identity)
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
    print(json.dumps({"status": "complete", "arm": args.arm, "m": 8192}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
