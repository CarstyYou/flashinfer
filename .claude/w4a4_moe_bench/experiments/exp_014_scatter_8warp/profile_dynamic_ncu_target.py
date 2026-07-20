#!/usr/bin/env python3
"""Run one identity-locked exp_014 M8192 graph node under native NCU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import build_dynamic_spill_evidence as evidence  # noqa: E402
import run_exp014_arm as harness  # noqa: E402


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.results = args.results.resolve()
    args.overlay = args.overlay.resolve()
    args.jit_root = args.jit_root.resolve()
    args.output = args.output.resolve()
    args.m = evidence.M
    args.fixture = evidence.FIXTURE
    args.seed = 2026
    args.device_index = 0

    registered_artifacts = harness.require_registered_measurement_jit(args)
    source = harness.validate_source(args.flashinfer_root, args.overlay, args.arm)
    if harness.TARGET_MODULE in sys.modules:
        raise RuntimeError("target module imported before exp_014 overlay installation")
    harness.worker.install_overlay(args.overlay)
    imports = harness.configure_source_checkout(args.flashinfer_root, args.jit_root)
    if Path(imports["target_module"]) != args.overlay:
        raise RuntimeError("target module did not resolve to selected overlay")
    runtime = harness.runtime_identity(args, source)

    fixture_module = harness.worker.load_fixture_module()
    device = torch.device("cuda", args.device_index)
    fixture = fixture_module.make_routed_fixture(
        evidence.M, device=device, seed=args.seed
    )
    weights = fixture_module.make_canonical_weights(device=device, seed=args.seed)
    arm = harness.worker.build_arm(args, fixture, weights)
    arm.eager()
    arm.capture()
    for _ in range(args.warmup):
        arm.replay()

    nvtx = f"exp014_{args.arm}_m8192_dynamic_spill_ncu"
    torch.cuda.nvtx.range_push(nvtx)
    cudart = torch.cuda.cudart()
    try:
        if int(cudart.cudaProfilerStart()) != 0:
            raise RuntimeError("cudaProfilerStart failed")
        output, event_elapsed_ms = arm.replay()
        if int(cudart.cudaProfilerStop()) != 0:
            raise RuntimeError("cudaProfilerStop failed")
    finally:
        torch.cuda.nvtx.range_pop()

    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("profile-target output contains non-finite values")
    artifacts = harness.common.artifact_manifest(args.jit_root)
    artifact_set = harness.common.canonical_sha256(artifacts)
    if artifact_set != registered_artifacts:
        raise RuntimeError("profile target mutated the registered JIT artifact set")
    cubins = sorted(
        {
            str(item["sha256"])
            for item in artifacts
            if str(item["path"]).endswith(".cubin")
        }
    )
    if cubins != [args.expected_cubin_sha256]:
        raise RuntimeError(
            f"profile target cubin drift: {cubins} != {[args.expected_cubin_sha256]}"
        )
    if source["overlay_sha256"] != args.expected_source_sha256:
        raise RuntimeError("profile target source hash drift")

    payload = {
        "schema": "exp014.dynamic-spill-profile-target.v1",
        "status": "complete",
        "arm": args.arm,
        "m": evidence.M,
        "fixture_kind": evidence.FIXTURE,
        "source_sha256": args.expected_source_sha256,
        "cubin_sha256": args.expected_cubin_sha256,
        "jit_artifact_set_sha256": artifact_set,
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
        "compile_identity": harness.worker._compile_identity(),
        "evidence_boundary": (
            "The CUDA profiler API brackets exactly one final CUDA Graph "
            "replay. Native NCU supplies observed kernel/grid/block identity "
            "and dynamic spill/refill counters."
        ),
    }
    atomic_write_json(args.output, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--arm", choices=evidence.ARMS, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-cubin-sha256", required=True)
    parser.add_argument("--expected-jit-artifact-set-sha256", required=True)
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
