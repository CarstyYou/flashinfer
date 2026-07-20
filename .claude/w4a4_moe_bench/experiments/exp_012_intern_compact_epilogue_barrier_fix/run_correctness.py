#!/usr/bin/env python3
"""Correctness-only multi-M runner for the exp_012 synchronization discriminator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
EXP009 = ROOT.parent / "exp_009_intern_stage4_compact_lightcheck"
sys.path.insert(0, str(EXP009))
from run_exp009_arm import ARM_NAME, load_worker  # noqa: E402


def run_case(worker, args, m: int) -> dict[str, Any]:
    args.m = m
    fixture_module, fixture, weights = worker.make_case(args)
    reference = fixture_module.reference_moe_nvfp4(fixture, weights)
    arm = worker.build_arm(args, fixture, weights)
    arm.eager()
    arm.capture()
    replays = []
    for replay_id in range(args.replays):
        output, _ = arm.replay(sentinel=True)
        output_cpu = output.detach().cpu().clone()
        diagnostics = fixture_module.output_diagnostics(output, reference)
        finite = bool(worker.torch.isfinite(output).all().item())
        nan_remaining = int(worker.torch.isnan(output).sum().item())
        inf_count = int(worker.torch.isinf(output).sum().item())
        _, workspace = worker._workspace_snapshot(
            arm.wrapper,
            fixture,
            num_cta_warps=worker.expected_block(args.arm)[0] // 32,
        )
        zero_rows = int((output_cpu.float().abs().sum(dim=1) == 0).sum().item())
        gate_pass = bool(
            diagnostics["formal_pass"]
            and finite
            and nan_remaining == 0
            and inf_count == 0
            and zero_rows == 0
            and workspace["verification"]["gate_pass"]
        )
        replays.append(
            {
                "replay": replay_id,
                "formal_metrics": diagnostics,
                "finite": finite,
                "nan_remaining": nan_remaining,
                "inf_count": inf_count,
                "zero_rows": zero_rows,
                "output_sha256": worker.tensor_sha256(output_cpu),
                "workspace_gate_pass": workspace["verification"]["gate_pass"],
                "workspace_checks": workspace["verification"].get("checks", {}),
                "gate_pass": gate_pass,
            }
        )
    artifacts = worker.artifact_manifest(args.jit_root)
    cubins = sorted(
        item["sha256"] for item in artifacts if item["path"].endswith(".cubin")
    )
    if not cubins:
        raise RuntimeError("candidate JIT produced no cubin")
    compile_identity = worker._compile_identity()
    return {
        "m": m,
        "replays": replays,
        "all_replays_pass": all(row["gate_pass"] for row in replays),
        "compile_identity": compile_identity,
        "expected_launch": {
            "kernel": "MoEDynamicKernel",
            "grid": list(worker.EXPECTED_GRID),
            "block": list(worker.expected_block(args.arm)),
        },
        "cubin_sha256": cubins,
        "jit_artifact_set_sha256": worker.canonical_sha256(artifacts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--expected-overlay-sha256", required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--m", type=int, nargs="+", required=True)
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--arm", default=ARM_NAME)
    parser.add_argument("--fixture", default="canonical")
    parser.add_argument("--device-index", type=int, default=0, choices=[0])
    parser.add_argument("--seed", type=int, default=2026, choices=[2026])
    args = parser.parse_args()
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.overlay = args.overlay.resolve()
    args.jit_root = args.jit_root.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite immutable result: {args.output}")
    if args.replays < 1:
        raise ValueError("replays must be positive")

    worker = load_worker()
    source = worker.validate_source(args.flashinfer_root, args.overlay, args.arm)
    if source["overlay_sha256"] != args.expected_overlay_sha256:
        raise RuntimeError("overlay hash does not match the registered experiment arm")
    if str(args.flashinfer_root) not in sys.path:
        sys.path.insert(0, str(args.flashinfer_root))
    worker.install_overlay(args.overlay)
    imports = worker.configure_source_checkout(args.flashinfer_root)
    if Path(imports["target_module"]) != args.overlay:
        raise RuntimeError("target module did not resolve to selected overlay")
    runtime = worker.runtime_identity(args, source)
    runtime["imports"] = imports

    payload: dict[str, Any] = {
        "schema": "exp012.correctness-only.v1",
        "status": "running",
        "runtime": runtime,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker.write_json(args.output, payload)
    try:
        for m in args.m:
            case = run_case(worker, args, m)
            payload["cases"].append(case)
            worker.write_json(args.output, payload)
        payload["all_cases_pass"] = all(
            case["all_replays_pass"] for case in payload["cases"]
        )
        payload["status"] = "complete"
        worker.write_json(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload["all_cases_pass"] else 1
    except Exception as error:
        payload["status"] = "failed"
        payload["error"] = f"{type(error).__name__}: {error}"
        worker.write_json(args.output, payload)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
