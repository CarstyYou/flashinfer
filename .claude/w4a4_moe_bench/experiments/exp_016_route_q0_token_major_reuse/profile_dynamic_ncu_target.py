#!/usr/bin/env python3
"""Run one identity-locked exp_016 Candidate M8192 graph node under NCU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_dynamic_spill_evidence as evidence  # noqa: E402
import run_exp016_arm as harness  # noqa: E402


WARMUP = 5


def atomic_write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise RuntimeError(f"immutable profile-target output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def configure_harness_globals(args: argparse.Namespace) -> None:
    harness.reused.BASELINE = harness.BASELINE
    harness.reused.CANDIDATE = harness.CANDIDATE
    harness.reused.ARMS = harness.ARMS
    harness.reused.EXPECTED_OVERLAY_SHA256 = {
        harness.BASELINE: args.expected_baseline_sha256,
        harness.CANDIDATE: args.expected_source_sha256,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.results = args.results.resolve()
    args.validation = args.validation.resolve()
    args.overlay = args.overlay.resolve()
    args.jit_root = args.jit_root.resolve()
    args.output = args.output.resolve()
    args.arm = evidence.CANDIDATE
    args.m = evidence.M
    args.fixture = evidence.FIXTURE
    args.scale_kind = evidence.SCALE_KIND
    args.seed = 2026
    args.device_index = 0
    args.jit_policy = "reuse"

    prerequisite = evidence.validated_candidate_identity(args.results, args.validation)
    checks = {
        "validation_sha256": prerequisite["sha256"] == args.expected_validation_sha256,
        "source_sha256": prerequisite["source_sha256"] == args.expected_source_sha256,
        "cubin_sha256": prerequisite["cubin_sha256"] == args.expected_cubin_sha256,
        "artifact_set": prerequisite["jit_artifact_set_sha256"]
        == args.expected_jit_artifact_set_sha256,
        "gpu_uuid": prerequisite["gpu_uuid"] == args.expected_gpu_uuid,
        "clock": prerequisite["application_graphics_clock_mhz"]
        == args.expected_app_clock_mhz,
    }
    if not all(checks.values()):
        raise RuntimeError(f"profile prerequisite drift: {checks}")

    harness._validate_registered_jit(args)
    configure_harness_globals(args)
    source = harness.reused.validate_source(
        args.flashinfer_root, args.overlay, args.arm
    )
    if harness.reused.TARGET_MODULE in sys.modules:
        raise RuntimeError("target module imported before exp_016 overlay installation")
    harness.worker.install_overlay(args.overlay)
    imports = harness.reused.configure_source_checkout(
        args.flashinfer_root, args.jit_root
    )
    if Path(imports["target_module"]) != args.overlay:
        raise RuntimeError("target module did not resolve to selected overlay")
    runtime = harness.reused.runtime_identity(args, source)
    if evidence.stable_runtime_identity(runtime) != prerequisite["runtime_identity"]:
        raise RuntimeError("profile runtime/toolchain drift from validated Candidate")

    _, fixture, weights = harness.make_case(args)
    arm = harness.worker.build_arm(args, fixture, weights)
    arm.eager()
    arm.capture()
    specialization = harness.specialization_contract(args.overlay)
    for _ in range(args.warmup):
        arm.replay()

    nvtx = "exp016_candidate_m8192_dynamic_spill_ncu"
    torch.cuda.nvtx.range_push(nvtx)
    cudart = torch.cuda.cudart()
    profiler_started = False
    replay_error: BaseException | None = None
    output = None
    event_elapsed_ms = 0.0
    try:
        if int(cudart.cudaProfilerStart()) != 0:
            raise RuntimeError("cudaProfilerStart failed")
        profiler_started = True
        output, event_elapsed_ms = arm.replay()
    except BaseException as error:
        replay_error = error
    finally:
        stop_error = None
        if profiler_started and int(cudart.cudaProfilerStop()) != 0:
            stop_error = RuntimeError("cudaProfilerStop failed")
        torch.cuda.nvtx.range_pop()
        if replay_error is not None:
            raise replay_error
        if stop_error is not None:
            raise stop_error

    assert output is not None
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("profile-target output contains non-finite values")
    artifacts, artifact_set, cubins = harness._artifacts(args.jit_root)
    if artifact_set != args.expected_jit_artifact_set_sha256:
        raise RuntimeError("profile target mutated the registered JIT artifact set")
    if cubins != [args.expected_cubin_sha256]:
        raise RuntimeError(
            f"profile target cubin drift: {cubins} != {[args.expected_cubin_sha256]}"
        )
    if source["overlay_sha256"] != args.expected_source_sha256:
        raise RuntimeError("profile target source hash drift")

    payload = {
        "schema": "exp016.dynamic-spill-profile-target.v1",
        "status": "complete",
        "arm": evidence.CANDIDATE,
        "m": evidence.M,
        "fixture": evidence.FIXTURE,
        "scale_kind": evidence.SCALE_KIND,
        "source_sha256": args.expected_source_sha256,
        "cubin_sha256": args.expected_cubin_sha256,
        "jit_artifact_set_sha256": artifact_set,
        "validation_sha256": args.expected_validation_sha256,
        "gpu_uuid": args.expected_gpu_uuid,
        "nvtx_range": nvtx,
        "warmup": args.warmup,
        "event_elapsed_us": event_elapsed_ms * 1000.0,
        "output_sha256": harness.worker.tensor_sha256(output),
        "expected_launch": {
            "grid": evidence.EXPECTED_GRID,
            "block": evidence.EXPECTED_BLOCK,
            "kernel": "MoEDynamicKernel",
        },
        "runtime": runtime,
        "imports": imports,
        "specialization": specialization,
        "compile_identity": harness.worker._compile_identity(),
        "evidence_boundary": (
            "The CUDA profiler API brackets exactly one final CUDA Graph replay. "
            "Native NCU supplies observed kernel/grid/block identity and executed "
            "spill/refill counters; static STACK/LOCAL is not used as this gate."
        ),
    }
    atomic_write_json(args.output, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument(
        "--expected-baseline-sha256",
        default=harness.EXPECTED_OVERLAY_SHA256[harness.BASELINE],
        choices=(harness.EXPECTED_OVERLAY_SHA256[harness.BASELINE],),
    )
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-cubin-sha256", required=True)
    parser.add_argument("--expected-jit-artifact-set-sha256", required=True)
    parser.add_argument("--expected-validation-sha256", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-app-clock-mhz", type=int, required=True)
    parser.add_argument("--warmup", type=int, choices=(WARMUP,), default=WARMUP)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = run(parse_args(argv))
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
