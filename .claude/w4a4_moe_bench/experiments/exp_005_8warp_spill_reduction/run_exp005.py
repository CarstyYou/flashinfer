#!/usr/bin/env python3
"""Coordinator for exp_005 preparation, correctness, ABBA timing, and profiles."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from exp005_common import (
    ABBA_ORDER,
    ALL_ARMS,
    BASELINE,
    BENCHMARK_GROUPS,
    CANDIDATE,
    CANONICAL_FIXTURE,
    DEFAULT_RESULTS,
    DIRECTED_FIXTURES,
    FORBIDDEN_ENV_KEYS,
    M_VALUES,
    TARGET_RELATIVE_PATH,
    case_directory,
    evaluate_cross_arm_correctness,
    read_json,
    summarize_paired_abba,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "run_exp005_arm.py"


def overlay_path(results: Path, arm: str) -> Path:
    return results / "overlays" / arm / "moe_dynamic_kernel.py"


def worker_environment(
    args: argparse.Namespace, arm: str, m: int, fixture: str
) -> tuple[dict[str, str], Path, Path]:
    results = args.results.resolve()
    overlay = overlay_path(results, arm)
    jit_root = case_directory(results, arm, m, fixture) / "jit"
    environment = dict(os.environ)
    for key in FORBIDDEN_ENV_KEYS:
        environment.pop(key, None)
    environment.update(
        {
            "FLASHINFER_WORKSPACE_BASE": str(jit_root),
            "CUTE_DSL_CACHE_DIR": str(jit_root / "cache"),
            "CUTE_DSL_DUMP_DIR": str(jit_root / "dump"),
            "CUTE_DSL_KEEP": "ir,ptx,cubin,sass",
        }
    )
    return environment, overlay, jit_root


def worker_command(
    args: argparse.Namespace,
    arm: str,
    m: int,
    fixture: str,
    command: Sequence[str],
) -> list[str]:
    _, overlay, jit_root = worker_environment(args, arm, m, fixture)
    if not overlay.is_file():
        raise RuntimeError(
            f"missing immutable overlay {overlay}; run build_overlays.py first"
        )
    return [
        sys.executable,
        str(WORKER),
        "--flashinfer-root",
        str(args.flashinfer_root.resolve()),
        "--results",
        str(args.results.resolve()),
        "--arm",
        arm,
        "--m",
        str(m),
        "--fixture",
        fixture,
        "--overlay",
        str(overlay),
        "--jit-root",
        str(jit_root),
        "--expected-gpu-uuid",
        args.expected_gpu_uuid,
        *command,
    ]


def run_worker(
    args: argparse.Namespace,
    arm: str,
    m: int,
    fixture: str,
    command: Sequence[str],
) -> None:
    environment, _, _ = worker_environment(args, arm, m, fixture)
    subprocess.run(
        worker_command(args, arm, m, fixture, command),
        env=environment,
        check=True,
    )


def preparation_path(results: Path, arm: str, m: int, fixture: str) -> Path:
    return case_directory(results, arm, m, fixture) / "preparation.json"


def refresh_manifest(results: Path) -> None:
    arms: dict[str, Any] = {}
    for arm in ALL_ARMS:
        arms[arm] = {}
        for m in M_VALUES:
            path = preparation_path(results, arm, m, CANONICAL_FIXTURE)
            if path.is_file():
                value = read_json(path)
                arms[arm][str(m)] = {
                    "preparation": str(path.relative_to(results)),
                    "overlay_sha256": value["runtime"]["source"]["overlay_sha256"],
                    "jit_root": value["runtime"]["jit_root"],
                    "jit_artifact_set_sha256": value["jit_artifact_set_sha256"],
                    "cubin_sha256": value["cubin_sha256"],
                    "launch_contract": value["launch_contract"],
                }
    directed: dict[str, Any] = {}
    for fixture in DIRECTED_FIXTURES:
        directed[fixture] = {}
        for arm in ALL_ARMS:
            path = preparation_path(results, arm, 256, fixture)
            if path.is_file():
                value = read_json(path)
                directed[fixture][arm] = {
                    "preparation": str(path.relative_to(results)),
                    "reference_gate_pass": all(
                        output["formal_pass"] for output in value["outputs"]
                    ),
                    "route_task_gate_pass": all(
                        replay["verification"]["gate_pass"]
                        for replay in value["route_task_evidence"]
                    ),
                }
    manifest = {
        "schema": "exp005.run-manifest.v1",
        "status": "in_progress",
        "source": {
            "kernel": TARGET_RELATIVE_PATH.as_posix(),
            "overlay_identity": "overlays/identity.json",
        },
        "cases": list(M_VALUES),
        "arms": arms,
        "directed_fixtures": directed,
        "correctness": {},
        "benchmark": {},
        "profiles": {},
    }
    for m in M_VALUES:
        path = results / "correctness" / f"m{m}.json"
        if path.is_file():
            manifest["correctness"][str(m)] = {
                "path": str(path.relative_to(results)),
                "gate_pass": read_json(path)["gate_pass"],
            }
        path = results / "benchmark" / f"m{m}_summary.json"
        if path.is_file():
            summary = read_json(path)
            manifest["benchmark"][str(m)] = {
                "path": str(path.relative_to(results)),
                "verdict": summary["verdict"],
            }
    for arm in ALL_ARMS:
        path = results / "profile_targets" / arm / "m8192" / "target.json"
        if path.is_file():
            manifest["profiles"][arm] = {
                "path": str(path.relative_to(results)),
                "profiler_verification": read_json(path)["profiler_observed_launch"][
                    "verification"
                ],
            }
    write_json(results / "manifest.json", manifest)


def prepare(args: argparse.Namespace) -> int:
    for m in args.ms:
        for arm in args.arms:
            run_worker(args, arm, m, CANONICAL_FIXTURE, ["prepare"])
            refresh_manifest(args.results.resolve())
    return 0


def prepare_directed(args: argparse.Namespace) -> int:
    for fixture in args.fixtures:
        for arm in args.arms:
            run_worker(args, arm, 256, fixture, ["prepare"])
            refresh_manifest(args.results.resolve())
    return 0


def _tensor_error(actual, expected) -> dict[str, float]:
    import torch

    actual_f = actual.float()
    expected_f = expected.float()
    error = actual_f - expected_f
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.flatten(), expected_f.flatten(), dim=0
    )
    denominator = torch.linalg.vector_norm(expected_f, dim=1).clamp_min(1e-12)
    token_relative = torch.linalg.vector_norm(error, dim=1) / denominator
    return {
        "cosine_loss": max(0.0, 1.0 - float(cosine.item())),
        "relative_l2": float(
            (
                torch.linalg.vector_norm(error)
                / torch.linalg.vector_norm(expected_f).clamp_min(1e-12)
            ).item()
        ),
        "max_abs": float(error.abs().max().item()),
        "token_rel_l2_p99": float(torch.quantile(token_relative, 0.99).item()),
    }


def correctness(args: argparse.Namespace) -> int:
    import torch

    results = args.results.resolve()
    overall = True
    for m in args.ms:
        preparations = {
            arm: read_json(preparation_path(results, arm, m, CANONICAL_FIXTURE))
            for arm in ALL_ARMS
        }
        for identity in ("fixture", "weights", "reference_sha256"):
            if preparations[BASELINE][identity] != preparations[CANDIDATE][identity]:
                raise RuntimeError(f"arm identity drift at M={m}: {identity}")
        baseline_outputs = [
            torch.load(
                case_directory(results, BASELINE, m, CANONICAL_FIXTURE)
                / f"output_{replay}.pt",
                map_location="cpu",
                weights_only=True,
            )
            for replay in range(2)
        ]
        candidate_outputs = [
            torch.load(
                case_directory(results, CANDIDATE, m, CANONICAL_FIXTURE)
                / f"output_{replay}.pt",
                map_location="cpu",
                weights_only=True,
            )
            for replay in range(2)
        ]
        comparisons = [
            _tensor_error(candidate, baseline)
            for candidate in candidate_outputs
            for baseline in baseline_outputs
        ]
        baseline_self_drift = _tensor_error(baseline_outputs[1], baseline_outputs[0])
        candidate_self_drift = _tensor_error(candidate_outputs[1], candidate_outputs[0])
        candidate_worst = {
            name: max(comparison[name] for comparison in comparisons)
            for name in baseline_self_drift
        }
        strict_cross_arm = evaluate_cross_arm_correctness(
            baseline_self_drift, candidate_self_drift, candidate_worst
        )
        formal = all(
            output["formal_pass"]
            for preparation in preparations.values()
            for output in preparation["outputs"]
        )
        route = all(
            replay["verification"]["gate_pass"]
            for preparation in preparations.values()
            for replay in preparation["route_task_evidence"]
        )
        payload = {
            "schema": "exp005.correctness.v1",
            "m": m,
            "fixture_identity_pass": True,
            "independent_reference_gate_pass": formal,
            "route_task_gate_pass": route,
            "baseline_self_drift": baseline_self_drift,
            "candidate_self_drift": candidate_self_drift,
            "candidate_vs_baseline": comparisons,
            "strict_cross_arm_gate": strict_cross_arm,
            "gate_pass": formal and route and bool(strict_cross_arm["gate_pass"]),
        }
        write_json(results / "correctness" / f"m{m}.json", payload)
        overall = overall and payload["gate_pass"]
    refresh_manifest(results)
    return 0 if overall else 2


def benchmark(args: argparse.Namespace) -> int:
    results = args.results.resolve()
    for m in args.ms:
        gate = results / "correctness" / f"m{m}.json"
        if not gate.is_file() or not read_json(gate).get("gate_pass"):
            raise RuntimeError(f"M={m} did not pass correctness before timing")
        for group in range(BENCHMARK_GROUPS):
            for position, arm in enumerate(ABBA_ORDER):
                run_worker(
                    args,
                    arm,
                    m,
                    CANONICAL_FIXTURE,
                    [
                        "measure",
                        "--group",
                        str(group),
                        "--position",
                        str(position),
                        "--warmup",
                        str(args.warmup),
                        "--iters",
                        str(args.iters),
                        "--clock-policy",
                        args.clock_policy,
                    ],
                )
        collect_benchmark(results, m, args.clock_policy)
        refresh_manifest(results)
    return 0


def collect_benchmark(results: Path, m: int, clock_policy: str) -> int:
    rows: list[dict[str, Any]] = []
    for group in range(BENCHMARK_GROUPS):
        for position, arm in enumerate(ABBA_ORDER):
            path = (
                results
                / "raw"
                / "benchmark"
                / f"m{m}"
                / f"group_{group}_position_{position}_{arm}.json"
            )
            value = read_json(path)
            if value.get("status") != "complete":
                raise RuntimeError(f"incomplete benchmark sample: {path}")
            gpu = value["runtime"]["gpu"]
            rows.append(
                {
                    "m": m,
                    "group": group,
                    "position": position,
                    "arm": arm,
                    "sample_us": float(value["sample_us"]),
                    "iters": int(value["iters"]),
                    "l2_flush_bytes": int(value["l2_flush_bytes"]),
                    "jit_artifact_set_sha256": value["jit_artifact_set_sha256"],
                    "gpu_uuid": gpu["uuid"],
                    "graphics_clock_mhz": gpu["graphics_clock_mhz"],
                    "applications_graphics_clock_mhz": gpu[
                        "applications_graphics_clock_mhz"
                    ],
                    "power_draw_w": gpu["power_draw_w"],
                }
            )
    uuids = {row["gpu_uuid"] for row in rows}
    if len(uuids) != 1:
        raise RuntimeError(f"paired benchmark crossed GPUs: {sorted(uuids)}")
    observed_clocks = {
        row["graphics_clock_mhz"]
        for row in rows
        if row["graphics_clock_mhz"] not in ("", "N/A", "[N/A]")
    }
    application_clocks = {
        row["applications_graphics_clock_mhz"]
        for row in rows
        if row["applications_graphics_clock_mhz"] not in ("", "N/A", "[N/A]")
    }
    effective_clock_policy = clock_policy
    clock_gate = {
        "declared": clock_policy,
        "observed_graphics_clock_mhz": sorted(observed_clocks),
        "observed_applications_graphics_clock_mhz": sorted(application_clocks),
        "stable_single_application_clock": len(application_clocks) == 1,
        "current_clock_note": (
            "current clocks may enter the idle P-state between paired processes; "
            "the gate is the fixed NVIDIA application clock setting"
        ),
    }
    if clock_policy == "locked" and len(application_clocks) != 1:
        effective_clock_policy = "unlocked"
        clock_gate["downgrade_reason"] = (
            "declared locked, but nvidia-smi application clocks were absent or changed"
        )
    summary = summarize_paired_abba(rows, clock_policy=effective_clock_policy)
    summary["clock_gate"] = clock_gate
    summary["m"] = m
    write_csv(results / "benchmark" / f"m{m}_raw.csv", rows)
    write_json(results / "benchmark" / f"m{m}_summary.json", summary)
    return 0


def profile(args: argparse.Namespace) -> int:
    for arm in args.arms:
        run_worker(
            args,
            arm,
            args.m,
            CANONICAL_FIXTURE,
            ["profile", "--warmup", str(args.warmup)],
        )
    refresh_manifest(args.results.resolve())
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, default=Path.cwd())
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--expected-gpu-uuid", required=True)
    prepare_parser.add_argument(
        "--arms", nargs="+", choices=ALL_ARMS, default=list(ALL_ARMS)
    )
    prepare_parser.add_argument(
        "--ms", nargs="+", type=int, choices=M_VALUES, default=list(M_VALUES)
    )

    directed_parser = subparsers.add_parser("prepare-directed")
    directed_parser.add_argument("--expected-gpu-uuid", required=True)
    directed_parser.add_argument(
        "--arms", nargs="+", choices=ALL_ARMS, default=list(ALL_ARMS)
    )
    directed_parser.add_argument(
        "--fixtures",
        nargs="+",
        choices=DIRECTED_FIXTURES,
        default=list(DIRECTED_FIXTURES),
    )

    correctness_parser = subparsers.add_parser("correctness")
    correctness_parser.add_argument(
        "--ms", nargs="+", type=int, choices=M_VALUES, default=list(M_VALUES)
    )

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--expected-gpu-uuid", required=True)
    benchmark_parser.add_argument(
        "--ms", nargs="+", type=int, choices=M_VALUES, default=list(M_VALUES)
    )
    benchmark_parser.add_argument("--warmup", type=int, default=5)
    benchmark_parser.add_argument("--iters", type=int, default=50)
    benchmark_parser.add_argument(
        "--clock-policy", choices=("locked", "unlocked"), required=True
    )

    collect_parser = subparsers.add_parser("collect-benchmark")
    collect_parser.add_argument("--m", type=int, choices=M_VALUES, required=True)
    collect_parser.add_argument(
        "--clock-policy", choices=("locked", "unlocked"), required=True
    )

    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--expected-gpu-uuid", required=True)
    profile_parser.add_argument("--m", type=int, choices=M_VALUES, default=8192)
    profile_parser.add_argument(
        "--arms", nargs="+", choices=ALL_ARMS, default=list(ALL_ARMS)
    )
    profile_parser.add_argument("--warmup", type=int, default=5)

    subparsers.add_parser("refresh-manifest")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.results = args.results.resolve()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "prepare-directed":
        return prepare_directed(args)
    if args.command == "correctness":
        return correctness(args)
    if args.command == "benchmark":
        return benchmark(args)
    if args.command == "collect-benchmark":
        return collect_benchmark(args.results, args.m, args.clock_policy)
    if args.command == "profile":
        return profile(args)
    if args.command == "refresh-manifest":
        refresh_manifest(args.results)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
