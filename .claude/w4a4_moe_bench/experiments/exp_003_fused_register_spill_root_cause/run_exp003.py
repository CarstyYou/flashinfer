#!/usr/bin/env python3
"""Coordinator for exp_003 historical arms and spill root-cause analysis."""

from __future__ import annotations

import argparse
import difflib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from exp003_common import (
    ALL_ARMS,
    DEFAULT_RESULTS,
    FALLBACK_CANDIDATE,
    FORBIDDEN_ENV_KEYS,
    M,
    PRIMARY_CANDIDATE,
    TARGET_RELATIVE_PATH,
    build_empty_manifest,
    file_sha256,
    read_json,
    summarize_paired_benchmark,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "run_exp003_arm.py"


def parse_arm_path(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        arm, separator, raw_path = value.partition("=")
        if not separator or arm not in ALL_ARMS or arm == "baseline":
            raise ValueError(f"expected candidate ARM=PATH, got {value!r}")
        if arm in parsed:
            raise ValueError(f"duplicate arm overlay: {arm}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"overlay does not exist: {path}")
        parsed[arm] = path
    return parsed


def init_overlays(args: argparse.Namespace) -> int:
    source = args.flashinfer_root.resolve() / TARGET_RELATIVE_PATH
    if not source.is_file():
        raise RuntimeError(f"production source is missing: {source}")
    candidates = parse_arm_path(args.candidate_overlay)
    overlays = {"baseline": source, **candidates}
    destination = args.results.resolve() / "overlays"
    destination.mkdir(parents=True, exist_ok=True)
    identities: dict[str, Any] = {}
    baseline_text = source.read_text()
    for arm, path in overlays.items():
        target = destination / f"{arm}.py"
        if target.exists():
            raise RuntimeError(f"immutable overlay already exists: {target}")
        shutil.copyfile(path, target)
        overlay_text = target.read_text()
        diff = "".join(
            difflib.unified_diff(
                baseline_text.splitlines(keepends=True),
                overlay_text.splitlines(keepends=True),
                fromfile="production/moe_dynamic_kernel.py",
                tofile=f"overlays/{arm}.py",
            )
        )
        diff_path = destination / f"{arm}.diff"
        diff_path.write_text(diff)
        if arm == "baseline" and diff:
            raise RuntimeError("baseline overlay unexpectedly differs from production")
        if arm != "baseline" and not diff:
            raise RuntimeError(f"candidate {arm} does not change production source")
        identities[arm] = {
            "input_path": str(path),
            "overlay_path": str(target),
            "overlay_sha256": file_sha256(target),
            "production_sha256": file_sha256(source),
            "diff_path": str(diff_path),
            "diff_sha256": file_sha256(diff_path),
            "reverse_patch_gate": (
                "not_applicable_byte_identical_baseline"
                if arm == "baseline"
                else "unified_diff_exact_endpoints_recorded"
            ),
        }
    write_json(
        destination / "identity.json",
        {
            "schema": "exp003.spill-root-cause.overlays.v1",
            "arms": identities,
        },
    )
    manifest_path = args.results.resolve() / "validation.manifest.json"
    manifest = build_empty_manifest()
    manifest["source"] = {
        "production_kernel": str(source),
        "production_kernel_sha256": file_sha256(source),
        "overlay_identity": str(
            (destination / "identity.json").relative_to(args.results.resolve())
        ),
    }
    write_json(manifest_path, manifest)
    return 0


def worker_environment(
    args: argparse.Namespace, arm: str
) -> tuple[dict[str, str], Path, Path]:
    results = args.results.resolve()
    overlay = results / "overlays" / f"{arm}.py"
    jit_root = results / "raw" / "jit" / arm
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
    args: argparse.Namespace, arm: str, command: Sequence[str]
) -> list[str]:
    _, overlay, jit_root = worker_environment(args, arm)
    return [
        sys.executable,
        str(WORKER),
        "--flashinfer-root",
        str(args.flashinfer_root.resolve()),
        "--results",
        str(args.results.resolve()),
        "--arm",
        arm,
        "--overlay",
        str(overlay),
        "--jit-root",
        str(jit_root),
        "--expected-gpu-uuid",
        args.expected_gpu_uuid,
        *command,
    ]


def run_worker(args: argparse.Namespace, arm: str, command: Sequence[str]) -> None:
    environment, _, _ = worker_environment(args, arm)
    invocation = worker_command(args, arm, command)
    subprocess.run(invocation, env=environment, check=True)


def prepare_arms(args: argparse.Namespace) -> int:
    for arm in args.arms:
        run_worker(args, arm, ["prepare"])
    return 0


def benchmark(args: argparse.Namespace) -> int:
    if args.candidate not in (PRIMARY_CANDIDATE, FALLBACK_CANDIDATE):
        raise RuntimeError("only the selected in-place arm may enter formal timing")
    static = read_json(args.results.resolve() / "static_spill_evidence.json")
    delta = static.get("deltas", {}).get(args.candidate, {})
    if not (
        static.get("baseline_reproduction_pass")
        and delta.get("complete_tail_removal_no_replacement")
        and delta.get("selected_opcode_projection_equal_except_local")
    ):
        raise RuntimeError(
            f"{args.candidate} did not pass the pre-timing static qualification gate"
        )
    correctness_evidence = read_json(args.results.resolve() / "correctness.json")
    if correctness_evidence.get(
        "candidate"
    ) != args.candidate or not correctness_evidence.get("gate_pass"):
        raise RuntimeError(
            f"{args.candidate} did not pass the quant-aware + strict correctness gate"
        )
    for repeat in range(5):
        order = (
            ("baseline", args.candidate)
            if repeat % 2 == 0
            else (args.candidate, "baseline")
        )
        order_label = "baseline>candidate" if repeat % 2 == 0 else "candidate>baseline"
        for arm in order:
            run_worker(
                args,
                arm,
                [
                    "measure",
                    "--repeat",
                    str(repeat),
                    "--order",
                    order_label,
                    "--warmup",
                    str(args.warmup),
                    "--iters",
                    str(args.iters),
                ],
            )
    return collect_benchmark(args.results.resolve(), args.candidate)


def collect_benchmark(results: Path, candidate: str) -> int:
    rows: list[dict[str, Any]] = []
    for repeat in range(5):
        order = "baseline>candidate" if repeat % 2 == 0 else "candidate>baseline"
        for arm in ("baseline", candidate):
            path = results / "raw" / "benchmark" / f"repeat_{repeat}_{arm}.json"
            value = read_json(path)
            if value.get("status") != "complete" or value.get("repeat") != repeat:
                raise RuntimeError(f"invalid benchmark sample: {path}")
            rows.append(
                {
                    "m": M,
                    "repeat": repeat,
                    "order": order,
                    "arm": "baseline" if arm == "baseline" else "candidate",
                    "resolved_arm": arm,
                    "sample_us": float(value["sample_us"]),
                    "iters": int(value["iters"]),
                    "l2_flush_bytes": int(value["l2_flush_bytes"]),
                    "jit_artifact_set_sha256": value["jit_artifact_set_sha256"],
                }
            )
    summary = summarize_paired_benchmark(rows)
    write_csv(results / "benchmark_raw.csv", rows)
    summary_rows = [
        {
            "arm": arm,
            **values,
            "speedup_percent": summary["speedup_percent"],
            "materiality_T_percent": summary["materiality_T_fraction"] * 100.0,
            "candidate_faster_pair_count": summary["candidate_faster_pair_count"],
            "material_improvement": summary["material_improvement"],
        }
        for arm, values in summary["arms"].items()
    ]
    write_csv(results / "benchmark_summary.csv", summary_rows)
    write_json(results / "benchmark_summary.json", {"candidate": candidate, **summary})
    return 0


def tensor_error(actual, expected) -> dict[str, float]:
    import torch

    actual = actual.float()
    expected = expected.float()
    error = actual - expected
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), expected.flatten(), dim=0
    )
    token_denominator = torch.linalg.vector_norm(expected, dim=1).clamp_min(1e-12)
    token_relative = torch.linalg.vector_norm(error, dim=1) / token_denominator
    return {
        "cosine_loss": max(0.0, 1.0 - float(cosine.item())),
        "relative_l2": float(
            (
                torch.linalg.vector_norm(error)
                / torch.linalg.vector_norm(expected).clamp_min(1e-12)
            ).item()
        ),
        "max_abs": float(error.abs().max().item()),
        "token_rel_l2_p99": float(torch.quantile(token_relative, 0.99).item()),
    }


def correctness(args: argparse.Namespace) -> int:
    import torch

    results = args.results.resolve()
    baseline_preparation = read_json(results / "arms" / "baseline" / "preparation.json")
    candidate_preparation = read_json(
        results / "arms" / args.candidate / "preparation.json"
    )
    for identity in ("fixture", "weights", "reference_sha256"):
        if baseline_preparation.get(identity) != candidate_preparation.get(identity):
            raise RuntimeError(f"candidate identity drift at {identity}")
    baseline_outputs = [
        torch.load(
            results / "raw" / "baseline" / f"output_{index}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for index in range(2)
    ]
    candidate = torch.load(
        results / "raw" / args.candidate / "output_0.pt",
        map_location="cpu",
        weights_only=True,
    )
    self_drift = tensor_error(baseline_outputs[1], baseline_outputs[0])
    comparisons = [tensor_error(candidate, baseline) for baseline in baseline_outputs]
    worst = {key: max(value[key] for value in comparisons) for key in self_drift}
    from exp003_common import evaluate_correctness_gate

    strict = evaluate_correctness_gate(self_drift, worst)
    oracle_pass = all(
        bool(item.get("formal_pass"))
        for item in candidate_preparation.get("outputs", [])
    )
    payload = {
        "schema": "exp003.spill-root-cause.correctness.v1",
        "candidate": args.candidate,
        "fixture_identity_pass": True,
        "quant_aware_oracle_pass": oracle_pass,
        "baseline_self_drift": self_drift,
        "candidate_vs_each_baseline": comparisons,
        "strict_candidate_gate": strict,
        "gate_pass": oracle_pass and bool(strict["gate_pass"]),
    }
    write_json(results / "correctness" / f"{args.candidate}.json", payload)
    # Preserve the original formal-candidate contract used by the benchmark;
    # attribution-only arms get a per-arm record but never replace it.
    if args.candidate in (PRIMARY_CANDIDATE, FALLBACK_CANDIDATE):
        static_path = results / "static_spill_evidence.json"
        static_delta = (
            read_json(static_path).get("deltas", {}).get(args.candidate, {})
            if static_path.is_file()
            else {}
        )
        if static_delta.get(
            "complete_tail_removal_no_replacement", False
        ) and static_delta.get("selected_opcode_projection_equal_except_local", False):
            write_json(results / "correctness.json", payload)
    return 0 if payload["gate_pass"] else 2


def analyze_root_cause(args: argparse.Namespace) -> int:
    """Rebuild root-cause evidence, canonical manifest, and reader report."""
    from build_result import main as build_result_main
    from build_spill_root_cause_evidence import main as build_evidence_main

    results = args.results.resolve()
    source = (
        args.source.resolve()
        if args.source is not None
        else args.flashinfer_root.resolve() / TARGET_RELATIVE_PATH
    )
    evidence_argv = [
        "--baseline-sass",
        str(args.baseline_sass.resolve()),
        "--up-first-sass",
        str(args.up_first_sass.resolve()),
        "--baseline-mlir",
        str(args.baseline_mlir.resolve()),
        "--baseline-ptx",
        str(args.baseline_ptx.resolve()),
        "--up-first-ptx",
        str(args.up_first_ptx.resolve()),
        "--source",
        str(source),
        "--static-evidence",
        str(results / "static_spill_evidence.json"),
        "--ncu-evidence",
        str(results / "ncu" / "spill_evidence.json"),
        "--output",
        str(results / "spill_root_cause_evidence.json"),
    ]
    evidence_status = build_evidence_main(evidence_argv)
    if evidence_status != 0:
        return evidence_status
    return build_result_main(["--results", str(results)])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, default=Path.cwd())
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-overlays")
    init_parser.add_argument("--candidate-overlay", action="append", default=[])

    prepare_parser = subparsers.add_parser("prepare-arms")
    prepare_parser.add_argument("--expected-gpu-uuid", required=True)
    prepare_parser.add_argument("--arms", nargs="+", choices=ALL_ARMS, required=True)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--expected-gpu-uuid", required=True)
    benchmark_parser.add_argument(
        "--candidate", choices=(PRIMARY_CANDIDATE, FALLBACK_CANDIDATE), required=True
    )
    benchmark_parser.add_argument("--warmup", type=int, default=5, choices=[5])
    benchmark_parser.add_argument("--iters", type=int, default=50, choices=[50])

    collect_parser = subparsers.add_parser("collect-benchmark")
    collect_parser.add_argument(
        "--candidate", choices=(PRIMARY_CANDIDATE, FALLBACK_CANDIDATE), required=True
    )

    correctness_parser = subparsers.add_parser("correctness")
    correctness_parser.add_argument("--candidate", choices=ALL_ARMS[1:], required=True)

    root_cause_parser = subparsers.add_parser(
        "analyze-root-cause",
        help="rebuild exact spill root-cause evidence and report",
    )
    root_cause_parser.add_argument("--baseline-sass", type=Path, required=True)
    root_cause_parser.add_argument("--up-first-sass", type=Path, required=True)
    root_cause_parser.add_argument("--baseline-mlir", type=Path, required=True)
    root_cause_parser.add_argument("--baseline-ptx", type=Path, required=True)
    root_cause_parser.add_argument("--up-first-ptx", type=Path, required=True)
    root_cause_parser.add_argument("--source", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "init-overlays":
        return init_overlays(args)
    if args.command == "prepare-arms":
        return prepare_arms(args)
    if args.command == "benchmark":
        return benchmark(args)
    if args.command == "collect-benchmark":
        return collect_benchmark(args.results.resolve(), args.candidate)
    if args.command == "correctness":
        return correctness(args)
    if args.command == "analyze-root-cause":
        return analyze_root_cause(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
