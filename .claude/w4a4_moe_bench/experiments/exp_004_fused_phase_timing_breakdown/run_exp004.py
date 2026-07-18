#!/usr/bin/env python3
"""Coordinate the three identity-locked exp_004 GPU arms.

This file does not import Torch.  Each arm runs in a fresh process so the
exact-module overlays and CuteDSL JIT roots cannot leak across identities.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from exp004_common import (
    ALL_ARMS,
    DEFAULT_RESULTS,
    FORBIDDEN_ENV_KEYS,
    MEASUREMENT_CONTROL,
    NORMAL,
    PROBE,
    canonical_sha256,
    file_sha256,
    read_json,
    write_json,
)


ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "run_exp004_arm.py"


def arm_overlay_paths(results: Path, arm: str) -> tuple[Path, Path]:
    root = results / "overlays" / arm
    return root / "moe_dynamic_kernel.py", root / "moe_dispatch.py"


def jit_root(results: Path, arm: str) -> Path:
    return results / "raw" / "preparation" / arm / "jit"


def preparation_path(results: Path, arm: str) -> Path:
    return results / "raw" / "preparation" / arm / "preparation.json"


def worker_environment(args: argparse.Namespace, arm: str) -> dict[str, str]:
    root = jit_root(args.results.resolve(), arm)
    environment = dict(os.environ)
    for key in FORBIDDEN_ENV_KEYS:
        environment.pop(key, None)
    environment.update(
        {
            "FLASHINFER_WORKSPACE_BASE": str(root),
            "CUTE_DSL_CACHE_DIR": str(root / "cache"),
            "CUTE_DSL_DUMP_DIR": str(root / "dump"),
            "CUTE_DSL_KEEP": "ir,ptx,cubin,sass",
        }
    )
    return environment


def worker_command(
    args: argparse.Namespace, arm: str, command: Sequence[str]
) -> list[str]:
    kernel, dispatch = arm_overlay_paths(args.results.resolve(), arm)
    for path in (kernel, dispatch):
        if not path.is_file():
            raise RuntimeError(
                f"missing immutable overlay {path}; run build_overlays.py first"
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
        "--kernel-overlay",
        str(kernel),
        "--dispatch-overlay",
        str(dispatch),
        "--jit-root",
        str(jit_root(args.results.resolve(), arm)),
        "--expected-gpu-uuid",
        args.expected_gpu_uuid,
        *command,
    ]


def run_worker(args: argparse.Namespace, arm: str, command: Sequence[str]) -> None:
    subprocess.run(
        worker_command(args, arm, command),
        env=worker_environment(args, arm),
        check=True,
    )


def _tensor_error(actual: Any, expected: Any) -> dict[str, float]:
    import torch

    actual_f = actual.float()
    expected_f = expected.float()
    error = actual_f - expected_f
    actual_d = actual_f.double()
    expected_d = expected_f.double()
    dot = torch.sum(actual_d * expected_d)
    norms = torch.linalg.vector_norm(actual_d) * torch.linalg.vector_norm(expected_d)
    cosine = float((dot / norms.clamp_min(1e-30)).item())
    token_denominator = torch.linalg.vector_norm(expected_f, dim=1).clamp_min(1e-12)
    token_relative = torch.linalg.vector_norm(error, dim=1) / token_denominator
    return {
        "cosine": cosine,
        "cosine_loss": max(0.0, 1.0 - cosine),
        "relative_l2": float(
            (
                torch.linalg.vector_norm(error)
                / torch.linalg.vector_norm(expected_f).clamp_min(1e-12)
            ).item()
        ),
        "max_abs": float(error.abs().max().item()),
        "token_rel_l2_p99": float(torch.quantile(token_relative, 0.99).item()),
    }


def _formal_pair_gate(metrics: Mapping[str, float]) -> bool:
    return (
        float(metrics["cosine"]) >= 0.999
        and float(metrics["relative_l2"]) <= 0.02
        and float(metrics["max_abs"]) <= 0.08
    )


def refresh_manifest(results: Path) -> None:
    whole_timing = results / "whole_kernel_timing.json"
    whole_summary = results / "derived" / "whole_kernel_capture_summary.json"
    if whole_timing.is_file() and whole_summary.is_file():
        from finalize_whole_kernel import RESULTS as WHOLE_RESULTS
        from finalize_whole_kernel import finalize as finalize_whole_kernel

        if results.resolve() != WHOLE_RESULTS.resolve():
            raise RuntimeError(
                "whole-kernel manifest finalization only accepts the canonical "
                f"results root: {WHOLE_RESULTS}"
            )
        finalize_whole_kernel(check=False)
        return

    blocked_gate = results / "derived" / "blocked_gate.json"
    blocked_result = results / "result.md"
    if blocked_gate.is_file() and blocked_result.is_file():
        from finalize_blocked import build_blocked_manifest

        manifest = build_blocked_manifest(
            results=results,
            gate_path=blocked_gate,
            result_path=blocked_result,
        )
        write_json(results / "manifest.json", manifest)
        return

    arms: dict[str, Any] = {}
    for arm in ALL_ARMS:
        path = preparation_path(results, arm)
        if path.is_file():
            value = read_json(path)
            arms[arm] = {
                "preparation": str(path.relative_to(results)),
                "jit_root": value["runtime"]["jit_root"],
                "jit_artifact_set_sha256": value["jit_artifact_set_sha256"],
                "cubin_sha256": value["cubin_sha256"],
                "kernel_overlay_sha256": value["runtime"]["source"]["overlays"][
                    "kernel"
                ]["sha256"],
                "dispatch_overlay_sha256": value["runtime"]["source"]["overlays"][
                    "dispatch"
                ]["sha256"],
            }
    manifest: dict[str, Any] = {
        "schema": "exp004.run-manifest.v1",
        "status": "in_progress",
        "overlay_identity": "overlays/identity.json",
        "arms": arms,
        "correctness": {},
        "phase_captures": {},
        "profiles": {},
    }
    correctness = results / "raw" / "correctness.json"
    if correctness.is_file():
        value = read_json(correctness)
        manifest["correctness"] = {
            "path": str(correctness.relative_to(results)),
            "gate_pass": value["gate_pass"],
        }
    for arm in (MEASUREMENT_CONTROL, PROBE):
        path = results / "raw" / "phase_capture" / arm / "manifest.json"
        if path.is_file():
            manifest["phase_captures"][arm] = {
                "path": str(path.relative_to(results)),
                "sha256": file_sha256(path),
            }
    calibration = results / "raw" / "calibration" / "manifest.json"
    if calibration.is_file():
        manifest["calibration"] = {
            "path": str(calibration.relative_to(results)),
            "sha256": file_sha256(calibration),
        }
    for arm in ALL_ARMS:
        path = results / "profile_targets" / arm / "target.json"
        if path.is_file():
            manifest["profiles"][arm] = {
                "path": str(path.relative_to(results)),
                "verification": read_json(path)["profiler_observed_launch"][
                    "verification"
                ],
            }
    analysis_gates = results / "derived" / "analysis_gates.json"
    result = results / "result.md"
    if analysis_gates.is_file() and result.is_file():
        gate = read_json(analysis_gates)
        manifest["analysis"] = {
            "path": str(analysis_gates.relative_to(results)),
            "formal_gate_pass": gate["formal_gate_pass"],
            "diagnostic_share_allowed": gate["diagnostic_share_allowed"],
        }
        manifest["result"] = {
            "path": str(result.relative_to(results)),
            "sha256": file_sha256(result),
        }
        manifest["status"] = (
            "canonical_complete"
            if gate["formal_gate_pass"]
            else (
                "diagnostic_complete"
                if gate["diagnostic_share_allowed"]
                else "blocked_by_measurement_gate"
            )
        )
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    write_json(results / "manifest.json", manifest)


def prepare(args: argparse.Namespace) -> int:
    for arm in args.arms:
        run_worker(args, arm, ["prepare"])
        refresh_manifest(args.results.resolve())
    return 0


def correctness(args: argparse.Namespace) -> int:
    import torch

    results = args.results.resolve()
    preparations = {arm: read_json(preparation_path(results, arm)) for arm in ALL_ARMS}
    for field in ("fixture", "weights", "reference_sha256", "case"):
        values = {
            canonical_sha256(preparation[field])
            for preparation in preparations.values()
        }
        if len(values) != 1:
            raise RuntimeError(f"cross-arm identity drift at {field}")

    outputs = {
        arm: [
            torch.load(
                preparation_path(results, arm).parent / f"output_{replay}.pt",
                map_location="cpu",
                weights_only=True,
            )
            for replay in range(2)
        ]
        for arm in ALL_ARMS
    }
    self_drift = {
        arm: _tensor_error(values[1], values[0]) for arm, values in outputs.items()
    }
    comparisons: dict[str, list[dict[str, float]]] = {}
    for left, right in (
        (NORMAL, MEASUREMENT_CONTROL),
        (MEASUREMENT_CONTROL, PROBE),
        (NORMAL, PROBE),
    ):
        key = f"{left}_vs_{right}"
        comparisons[key] = [
            _tensor_error(right_value, left_value)
            for right_value in outputs[right]
            for left_value in outputs[left]
        ]

    candidate_control = comparisons[f"{MEASUREMENT_CONTROL}_vs_{PROBE}"]
    p99_limit = self_drift[MEASUREMENT_CONTROL]["token_rel_l2_p99"] + 0.005
    gates = {
        "all_reference_oracles": all(
            output["gate"]["gate_pass"] and output["output_contract"]["gate_pass"]
            for preparation in preparations.values()
            for output in preparation["outputs"]
        ),
        "all_workspace_gates": all(
            gate["gate_pass"]
            for preparation in preparations.values()
            for gate in preparation["workspace_gates"]
        ),
        "all_pairwise_formal": all(
            _formal_pair_gate(metrics)
            for values in comparisons.values()
            for metrics in values
        ),
        "probe_vs_control_token_p99": max(
            value["token_rel_l2_p99"] for value in candidate_control
        )
        <= p99_limit,
        "measurement_buffers_remain_sentinel": all(
            gate["gate_pass"]
            for gate in preparations[MEASUREMENT_CONTROL]["timing_gates"]
        ),
        "probe_event_contract": all(
            gate["gate_pass"] for gate in preparations[PROBE]["timing_gates"]
        ),
    }
    payload = {
        "schema": "exp004.cross-arm-correctness.v1",
        "identity_pass": True,
        "self_drift": self_drift,
        "comparisons": comparisons,
        "probe_vs_control_token_rel_l2_p99_limit": p99_limit,
        "gates": gates,
        "gate_pass": all(gates.values()),
    }
    write_json(results / "raw" / "correctness.json", payload)
    refresh_manifest(results)
    return 0 if payload["gate_pass"] else 2


def capture_phases(args: argparse.Namespace) -> int:
    results = args.results.resolve()
    correctness_path = results / "raw" / "correctness.json"
    if not correctness_path.is_file() or not read_json(correctness_path).get(
        "gate_pass"
    ):
        raise RuntimeError("cross-arm correctness must pass before phase capture")
    for arm in (MEASUREMENT_CONTROL, PROBE):
        run_worker(args, arm, ["capture-phases", "--warmup", str(args.warmup)])
        refresh_manifest(results)
    return 0


def profile(args: argparse.Namespace) -> int:
    for arm in args.arms:
        run_worker(args, arm, ["profile", "--warmup", str(args.warmup)])
        refresh_manifest(args.results.resolve())
    return 0


def calibrate(args: argparse.Namespace) -> int:
    run_worker(
        args,
        PROBE,
        [
            "capture-calibration",
            "--warmup",
            str(args.warmup),
            "--samples",
            str(args.samples),
        ],
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

    subparsers.add_parser("correctness")

    capture_parser = subparsers.add_parser("capture-phases")
    capture_parser.add_argument("--expected-gpu-uuid", required=True)
    capture_parser.add_argument("--warmup", type=int, default=5)

    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--expected-gpu-uuid", required=True)
    profile_parser.add_argument(
        "--arms", nargs="+", choices=ALL_ARMS, default=list(ALL_ARMS)
    )
    profile_parser.add_argument("--warmup", type=int, default=5)

    calibration_parser = subparsers.add_parser("calibrate")
    calibration_parser.add_argument("--expected-gpu-uuid", required=True)
    calibration_parser.add_argument("--warmup", type=int, default=3)
    calibration_parser.add_argument("--samples", type=int, default=4096)

    subparsers.add_parser("refresh-manifest")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.results = args.results.resolve()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "correctness":
        return correctness(args)
    if args.command == "capture-phases":
        return capture_phases(args)
    if args.command == "profile":
        return profile(args)
    if args.command == "calibrate":
        return calibrate(args)
    if args.command == "refresh-manifest":
        refresh_manifest(args.results)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
