#!/usr/bin/env python3
"""Run one targeted NCU capture per qualified exp_003 arm."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Sequence

from exp003_common import ALL_ARMS, DEFAULT_RESULTS, FORBIDDEN_ENV_KEYS
from run_exp003 import worker_command, worker_environment


METRIC_IDS = (
    "gpu__time_duration.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    # SM120 does not expose the older register-spilling derived metrics.  The
    # target SASS has no non-spill local operations, so use the native executed
    # local-load/store instruction counters and close their identity against
    # the static LDL/STL inventory.
    "sm__sass_inst_executed_op_local_ld.sum",
    "sm__sass_inst_executed_op_local_st.sum",
    "sm__inst_executed_pipe_tensor_subpipe_hmma.sum",
    "sm__ops_path_tensor_src_fp4_dst_fp32.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__stack_size",
)


def capture(args: argparse.Namespace, arm: str) -> None:
    environment, _, _ = worker_environment(args, arm)
    for key in FORBIDDEN_ENV_KEYS:
        environment.pop(key, None)
    raw = args.results.resolve() / "raw" / "ncu" / arm
    raw.mkdir(parents=True, exist_ok=False)
    report = raw / "trace"
    target = worker_command(args, arm, ["profile", "--warmup", "5"])
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
        str(report),
        *target,
    ]
    (raw / "command.txt").write_text(" ".join(command) + "\n")
    if args.dry_run:
        return
    with (
        (raw / "stdout.log").open("w") as stdout,
        (raw / "stderr.log").open("w") as stderr,
    ):
        subprocess.run(
            command, env=environment, stdout=stdout, stderr=stderr, check=True
        )
    export = [
        args.ncu,
        "--import",
        str(report.with_suffix(".ncu-rep")),
        "--csv",
        "--page",
        "raw",
        "--print-units",
        "base",
    ]
    with (
        (raw / "native_raw.csv").open("w") as stdout,
        (raw / "native_raw.stderr.log").open("w") as stderr,
    ):
        subprocess.run(export, stdout=stdout, stderr=stderr, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, default=Path.cwd())
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--arms", nargs="+", choices=ALL_ARMS, required=True)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for arm in args.arms:
        capture(args, arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
