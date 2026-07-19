#!/usr/bin/env python3
"""Expose exactly one uninstrumented exp_011 full-kernel replay to NCU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
EXP004 = ROOT.parent / "exp_004_fused_phase_timing_breakdown"
if str(EXP004) not in sys.path:
    sys.path.insert(0, str(EXP004))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import capture_arm  # noqa: E402
import run_exp004_arm as worker  # noqa: E402
from exp004_common import MEASUREMENT_CONTROL, file_sha256, write_json  # noqa: E402


def profile(args: argparse.Namespace) -> dict[str, object]:
    overlay = args.overlay_root.resolve() / args.variant / "no_marker"
    kernel_overlay = overlay / "moe_dynamic_kernel.py"
    dispatch_overlay = overlay / "moe_dispatch.py"
    delegated = argparse.Namespace(
        flashinfer_root=args.flashinfer_root.resolve(),
        arm=MEASUREMENT_CONTROL,
        kernel_overlay=kernel_overlay,
        dispatch_overlay=dispatch_overlay,
        jit_root=args.jit_root.resolve(),
        output=args.output.resolve().parent,
        expected_gpu_uuid=args.expected_gpu_uuid,
    )
    worker.require_empty_directory(delegated.jit_root)
    source = worker.validate_source(delegated)
    worker.install_overlays(kernel_overlay, dispatch_overlay)
    imports = worker.configure_source_checkout(delegated.flashinfer_root)
    runtime = worker.runtime_identity(delegated, source)
    runtime["imports"] = imports

    capture_arm._install_variant_hooks(args.variant)
    fixture_module, fixture, weights = worker.make_case()
    reference = fixture_module.reference_moe_nvfp4(fixture, weights)
    arm = worker.build_arm(fixture, weights)
    eager = arm.eager()
    eager_gate = worker.correctness_gate(
        worker.tensor_error(eager.detach().cpu(), reference.cpu())
    )
    if not eager_gate["gate_pass"]:
        raise RuntimeError(f"eager correctness failed: {eager_gate}")
    arm.capture()
    for _ in range(args.warmup):
        arm.replay(output_sentinel=False, reset_probe=False)

    nvtx = f"exp011_{args.variant}_m8192_ncu_replay"
    torch.cuda.nvtx.range_push(nvtx)
    cudart = torch.cuda.cudart()
    try:
        if int(cudart.cudaProfilerStart()) != 0:
            raise RuntimeError("cudaProfilerStart failed")
        output, elapsed_ms = arm.replay(
            output_sentinel=False,
            reset_probe=False,
        )
        if int(cudart.cudaProfilerStop()) != 0:
            raise RuntimeError("cudaProfilerStop failed")
    finally:
        torch.cuda.nvtx.range_pop()

    correctness = worker.correctness_gate(
        worker.tensor_error(output.detach().cpu(), reference.cpu())
    )
    if not correctness["gate_pass"]:
        raise RuntimeError(f"profile replay correctness failed: {correctness}")
    payload: dict[str, object] = {
        "schema": "exp011.ncu-profile-target.v1",
        "variant": args.variant,
        "nvtx_range": nvtx,
        "event_elapsed_us": elapsed_ms * 1000.0,
        "correctness_gate": correctness,
        "arm_contract": dict(capture_arm._ARM_CONTRACT),
        "overlay": {
            "kernel_sha256": file_sha256(kernel_overlay),
            "dispatch_sha256": file_sha256(dispatch_overlay),
        },
        "runtime": runtime,
        "foreign_processes_after": worker.require_no_foreign_process(runtime),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--variant", choices=capture_arm.VARIANTS, required=True)
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    payload = profile(args)
    print(
        json.dumps(
            {
                "variant": args.variant,
                "status": "complete",
                "event_elapsed_us": payload["event_elapsed_us"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
