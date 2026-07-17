#!/usr/bin/env python3
"""Relationship-scoped coordinator for exp_005 R2 temporal-N64 follow-up."""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from exp005_common import (
    BENCHMARK_GROUPS,
    CANDIDATE,
    CANONICAL_FIXTURE,
    DIRECTED_FIXTURES,
    FORBIDDEN_ENV_KEYS,
    M_VALUES,
    RATIO_BAND,
    TEMPORAL_N64,
    case_directory,
    evaluate_cross_arm_correctness,
    file_sha256,
    read_json,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "run_exp005_arm.py"
REGISTRY = ROOT / "comparison_registry.r2.json"
REGISTRY_SHA256 = "7531a9e41560de8f91d3eae3fd1c75044851f12566c9056009fd91ebf8e7b04d"
RESULTS = ROOT / "results" / "n64_temporal_replay" / "canonical_r2"
OVERLAYS = ROOT / "results" / "n64_temporal_replay" / "overlays"
ANCHOR = CANDIDATE
SUBJECT = TEMPORAL_N64
ARMS = (ANCHOR, SUBJECT)
ABBA_ORDER = (ANCHOR, SUBJECT, SUBJECT, ANCHOR)


def require_registry() -> dict[str, Any]:
    if file_sha256(REGISTRY) != REGISTRY_SHA256:
        raise RuntimeError("accepted r2 comparison registry hash drift")
    registry = read_json(REGISTRY)
    relationship = registry["relationships"][0]
    if (
        relationship["id"] != "R2_temporal_n64_replay_vs_candidateA"
        or relationship["anchor"] != ANCHOR
        or relationship["subjects"] != [SUBJECT]
        or relationship["fresh_evidence_namespace"]
        != "results/n64_temporal_replay/canonical_r2"
    ):
        raise RuntimeError("r2 relationship contract drift")
    return registry


def overlay_path(arm: str) -> Path:
    path = OVERLAYS / arm / "moe_dynamic_kernel.py"
    if not path.is_file():
        raise RuntimeError(f"missing immutable R2 overlay: {path}")
    return path


def preparation_path(results: Path, arm: str, m: int, fixture: str) -> Path:
    return case_directory(results, arm, m, fixture) / "preparation.json"


def worker_environment(
    results: Path, arm: str, m: int, fixture: str
) -> tuple[dict[str, str], Path]:
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
    return environment, jit_root


def run_worker(
    args: argparse.Namespace,
    arm: str,
    m: int,
    fixture: str,
    command: Sequence[str],
) -> None:
    results = args.results.resolve()
    environment, jit_root = worker_environment(results, arm, m, fixture)
    argv = [
        sys.executable,
        str(WORKER),
        "--flashinfer-root",
        str(args.flashinfer_root.resolve()),
        "--results",
        str(results),
        "--arm",
        arm,
        "--m",
        str(m),
        "--fixture",
        fixture,
        "--overlay",
        str(overlay_path(arm).resolve()),
        "--jit-root",
        str(jit_root),
        "--expected-gpu-uuid",
        args.expected_gpu_uuid,
        "--comparison-anchor",
        ANCHOR,
        "--comparison-subject",
        SUBJECT,
        *command,
    ]
    subprocess.run(argv, env=environment, check=True)


def refresh_manifest(results: Path) -> None:
    require_registry()
    arms: dict[str, Any] = {}
    for arm in ARMS:
        arms[arm] = {}
        for m in M_VALUES:
            path = preparation_path(results, arm, m, CANONICAL_FIXTURE)
            if path.is_file():
                value = read_json(path)
                arms[arm][str(m)] = {
                    "preparation": str(path.relative_to(results)),
                    "overlay_sha256": value["runtime"]["source"]["overlay_sha256"],
                    "cubin_sha256": value["cubin_sha256"],
                    "jit_artifact_set_sha256": value["jit_artifact_set_sha256"],
                    "launch_contract": value["launch_contract"],
                }
    correctness = {}
    benchmark = {}
    for m in M_VALUES:
        path = results / "correctness" / f"m{m}.json"
        if path.is_file():
            correctness[str(m)] = {
                "path": str(path.relative_to(results)),
                "gate_pass": read_json(path)["gate_pass"],
            }
        path = results / "benchmark" / f"m{m}_summary.json"
        if path.is_file():
            benchmark[str(m)] = {
                "path": str(path.relative_to(results)),
                "verdict": read_json(path)["verdict"],
            }
    static_path = results / "static" / "summary.json"
    static = (
        {
            "path": str(static_path.relative_to(results)),
            "subject_zero_spill_gate": read_json(static_path)[
                "subject_zero_spill_gate"
            ],
        }
        if static_path.is_file()
        else {}
    )
    write_json(
        results / "manifest.json",
        {
            "schema": "exp005.temporal-n64-r2-manifest.v1",
            "status": "in_progress",
            "relationship": "R2_temporal_n64_replay_vs_candidateA",
            "comparison_registry": {
                "path": str(REGISTRY.relative_to(ROOT)),
                "sha256": REGISTRY_SHA256,
            },
            "arms": arms,
            "correctness": correctness,
            "static_spill": static,
            "benchmark": benchmark,
        },
    )


def prepare(args: argparse.Namespace) -> int:
    require_registry()
    for m in args.ms:
        for arm in args.arms:
            run_worker(args, arm, m, CANONICAL_FIXTURE, ["prepare"])
            refresh_manifest(args.results.resolve())
    return 0


def prepare_directed(args: argparse.Namespace) -> int:
    require_registry()
    for fixture in args.fixtures:
        for arm in args.arms:
            run_worker(args, arm, 256, fixture, ["prepare"])
    refresh_manifest(args.results.resolve())
    return 0


def tensor_error(actual, expected) -> dict[str, float]:
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

    require_registry()
    results = args.results.resolve()
    overall = True
    for m in args.ms:
        preparations = {
            arm: read_json(preparation_path(results, arm, m, CANONICAL_FIXTURE))
            for arm in ARMS
        }
        for identity in ("fixture", "weights", "reference_sha256"):
            if preparations[ANCHOR][identity] != preparations[SUBJECT][identity]:
                raise RuntimeError(f"R2 arm identity drift at M={m}: {identity}")
        outputs = {
            arm: [
                torch.load(
                    case_directory(results, arm, m, CANONICAL_FIXTURE)
                    / f"output_{replay}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                for replay in range(2)
            ]
            for arm in ARMS
        }
        comparisons = [
            tensor_error(subject, anchor)
            for subject in outputs[SUBJECT]
            for anchor in outputs[ANCHOR]
        ]
        anchor_self = tensor_error(outputs[ANCHOR][1], outputs[ANCHOR][0])
        subject_self = tensor_error(outputs[SUBJECT][1], outputs[SUBJECT][0])
        subject_worst = {
            name: max(comparison[name] for comparison in comparisons)
            for name in anchor_self
        }
        strict = evaluate_cross_arm_correctness(
            anchor_self, subject_self, subject_worst
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
            "schema": "exp005.temporal-n64-r2-correctness.v1",
            "relationship": "R2_temporal_n64_replay_vs_candidateA",
            "m": m,
            "fixture_identity_pass": True,
            "independent_reference_gate_pass": formal,
            "route_task_gate_pass": route,
            "anchor_self_drift": anchor_self,
            "subject_self_drift": subject_self,
            "subject_vs_anchor": comparisons,
            "strict_cross_arm_gate": strict,
            "gate_pass": formal and route and bool(strict["gate_pass"]),
        }
        write_json(results / "correctness" / f"m{m}.json", payload)
        overall = overall and payload["gate_pass"]
    refresh_manifest(results)
    return 0 if overall else 2


def static_gate(args: argparse.Namespace) -> int:
    require_registry()
    results = args.results.resolve()
    records: dict[str, Any] = {}
    for arm in ARMS:
        records[arm] = {}
        for m in args.ms:
            evidence_path = results / "static" / arm / f"m{m}" / "evidence.json"
            evidence = read_json(evidence_path)
            preparation = read_json(
                preparation_path(results, arm, m, CANONICAL_FIXTURE)
            )
            cubin_hash = evidence["identity"]["cubin_sha256"]
            if cubin_hash not in preparation["cubin_sha256"]:
                raise RuntimeError(f"{arm} M={m}: static evidence cubin mismatch")
            records[arm][str(m)] = {
                "path": str(evidence_path.relative_to(results)),
                "cubin_sha256": cubin_hash,
                "registers_per_thread": evidence["resource"]["registers_per_thread"],
                "stack_bytes_per_thread": evidence["resource"][
                    "stack_bytes_per_thread"
                ],
                "spill_refill_instruction_count": evidence["compiler_spill_refill"][
                    "static_local_instruction_count"
                ],
                "zero_spill_static_gate": evidence["zero_spill_static_gate"],
            }
    subject_gate = all(
        value["zero_spill_static_gate"] for value in records[SUBJECT].values()
    )
    write_json(
        results / "static" / "summary.json",
        {
            "schema": "exp005.temporal-n64-r2-static-summary.v1",
            "relationship": "R2_temporal_n64_replay_vs_candidateA",
            "records": records,
            "subject_zero_spill_gate": subject_gate,
        },
    )
    refresh_manifest(results)
    return 0 if subject_gate else 2


def quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def arm_statistics(values: Sequence[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "median_us": statistics.median(values),
        "p10_us": quantile(values, 0.10),
        "p90_us": quantile(values, 0.90),
        "mean_us": mean,
        "cv": statistics.pstdev(values) / mean,
    }


def summarize_abba(
    rows: Sequence[Mapping[str, Any]], *, clock_policy: str
) -> dict[str, Any]:
    expected = {
        (group, position): arm
        for group in range(BENCHMARK_GROUPS)
        for position, arm in enumerate(ABBA_ORDER)
    }
    seen: dict[tuple[int, int], Mapping[str, Any]] = {}
    by_arm: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for row in rows:
        key = (int(row["group"]), int(row["position"]))
        arm = str(row["arm"])
        if key in seen or expected.get(key) != arm:
            raise ValueError(f"R2 ABBA order drift at {key}: {arm}")
        sample = float(row["sample_us"])
        if not math.isfinite(sample) or sample <= 0:
            raise ValueError(f"invalid sample_us: {sample}")
        seen[key] = row
        by_arm[arm].append(sample)
    if set(seen) != set(expected):
        raise ValueError("R2 benchmark requires five complete ABBA groups")
    paired = []
    for group in range(BENCHMARK_GROUPS):
        anchor_us = statistics.fmean(
            float(seen[(group, p)]["sample_us"]) for p in (0, 3)
        )
        subject_us = statistics.fmean(
            float(seen[(group, p)]["sample_us"]) for p in (1, 2)
        )
        paired.append(
            {
                "group": group,
                "anchor_us": anchor_us,
                "subject_us": subject_us,
                "ratio": anchor_us / subject_us,
            }
        )
    rng = random.Random(20260717)
    bootstrap = []
    for _ in range(10_000):
        sample = [paired[rng.randrange(len(paired))] for _ in paired]
        bootstrap.append(
            statistics.fmean(float(item["anchor_us"]) for item in sample)
            / statistics.fmean(float(item["subject_us"]) for item in sample)
        )
    ci_low, ci_high = quantile(bootstrap, 0.025), quantile(bootstrap, 0.975)
    if ci_low > RATIO_BAND[1]:
        statistical = "faster"
    elif ci_high < RATIO_BAND[0]:
        statistical = "slower"
    elif ci_low >= RATIO_BAND[0] and ci_high <= RATIO_BAND[1]:
        statistical = "equivalent"
    else:
        statistical = "inconclusive"
    verdict = statistical if clock_policy == "locked" else "advisory_inconclusive"
    ratio = statistics.median(by_arm[ANCHOR]) / statistics.median(by_arm[SUBJECT])
    return {
        "schema": "exp005.temporal-n64-r2-paired-abba.v1",
        "relationship": "R2_temporal_n64_replay_vs_candidateA",
        "groups": paired,
        "arms": {arm: arm_statistics(values) for arm, values in by_arm.items()},
        "median_ratio_anchor_over_subject": ratio,
        "median_speedup_percent": (ratio - 1.0) * 100.0,
        "paired_bootstrap": {
            "samples": 10_000,
            "seed": 20260717,
            "ratio_ci95": [ci_low, ci_high],
        },
        "predeclared_ratio_band": list(RATIO_BAND),
        "clock_policy": clock_policy,
        "statistical_classification": statistical,
        "verdict": verdict,
    }


def collect_benchmark(results: Path, m: int, clock_policy: str) -> None:
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
    if len({row["gpu_uuid"] for row in rows}) != 1:
        raise RuntimeError("R2 paired benchmark crossed GPUs")
    application_clocks = {
        row["applications_graphics_clock_mhz"]
        for row in rows
        if row["applications_graphics_clock_mhz"] not in ("", "N/A", "[N/A]")
    }
    effective_clock_policy = clock_policy
    if clock_policy == "locked" and len(application_clocks) != 1:
        effective_clock_policy = "unlocked"
    summary = summarize_abba(rows, clock_policy=effective_clock_policy)
    summary["declared_clock_policy"] = clock_policy
    summary["observed_applications_graphics_clock_mhz"] = sorted(application_clocks)
    summary["m"] = m
    write_csv(results / "benchmark" / f"m{m}_raw.csv", rows)
    write_json(results / "benchmark" / f"m{m}_summary.json", summary)


def benchmark(args: argparse.Namespace) -> int:
    require_registry()
    results = args.results.resolve()
    static = read_json(results / "static" / "summary.json")
    if not static["subject_zero_spill_gate"]:
        raise RuntimeError("subject did not pass static zero-spill gate")
    for m in args.ms:
        correctness_gate = read_json(results / "correctness" / f"m{m}.json")
        if not correctness_gate["gate_pass"]:
            raise RuntimeError(f"M={m} did not pass R2 correctness")
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, default=Path.cwd())
    parser.add_argument("--results", type=Path, default=RESULTS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--expected-gpu-uuid", required=True)
    prepare_parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    prepare_parser.add_argument(
        "--ms", nargs="+", type=int, choices=M_VALUES, default=list(M_VALUES)
    )

    directed_parser = subparsers.add_parser("prepare-directed")
    directed_parser.add_argument("--expected-gpu-uuid", required=True)
    directed_parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
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

    static_parser = subparsers.add_parser("static-gate")
    static_parser.add_argument(
        "--ms", nargs="+", type=int, choices=M_VALUES, default=list(M_VALUES)
    )

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--expected-gpu-uuid", required=True)
    benchmark_parser.add_argument(
        "--ms", nargs="+", type=int, choices=M_VALUES, default=[256, 8192]
    )
    benchmark_parser.add_argument("--warmup", type=int, default=5)
    benchmark_parser.add_argument("--iters", type=int, default=50)
    benchmark_parser.add_argument(
        "--clock-policy", choices=("locked", "unlocked"), required=True
    )

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
    if args.command == "static-gate":
        return static_gate(args)
    if args.command == "benchmark":
        return benchmark(args)
    if args.command == "refresh-manifest":
        refresh_manifest(args.results)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
