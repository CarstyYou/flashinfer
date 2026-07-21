#!/usr/bin/env python3
"""Run one correctness-qualified current Latest-opt M8192 graph replay under NCU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
OPT_RELATIVE = Path(".claude/w4a4_moe_bench/moe_dynamic_kernel_opt.py")
M = 8192
ARM = "candidate_token_major_reuse"
WARMUP = 5


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise RuntimeError(f"immutable NCU target manifest exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def gate_output(
    fixture_module: Any,
    output: Any,
    reference_cpu: Any,
    core: Any,
    fixture: Any,
    arm: Any,
) -> dict[str, Any]:
    output_cpu = output.detach().cpu().clone()
    diagnostics = fixture_module.output_diagnostics(output_cpu, reference_cpu)
    zero_rows = int((output_cpu == 0).all(dim=-1).sum().item())
    _, route = core.corrected_workspace_snapshot(ARM, arm.wrapper, fixture)
    checks = {
        "formal_pass": bool(diagnostics.get("formal_pass")),
        "finite": bool(diagnostics.get("finite")),
        "no_full_zero_rows": zero_rows == 0,
        "route_task_gate": bool(route["verification"]["gate_pass"]),
    }
    return {
        "diagnostics": diagnostics,
        "full_zero_rows": zero_rows,
        "route_task": route["verification"],
        "checks": checks,
        "gate_pass": all(checks.values()),
    }


def selected_resource(
    core: Any, jit_root: Path, artifacts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return core_capture_resource(core, jit_root, artifacts)


def core_capture_resource(
    core: Any, jit_root: Path, artifacts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    # Reuse exp_017's already-tested exact-cubin resource parser.
    import capture_opt_phase

    return capture_opt_phase.resource_usage(jit_root, artifacts)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.fixture_dir = args.fixture_dir.resolve()
    args.jit_root = args.jit_root.resolve()
    args.output = args.output.resolve()
    args.m = M
    args.fixture = "canonical"
    args.scale_kind = "equal"
    args.seed = 2026
    args.device_index = 0
    args.arm = ARM
    args.overlay = args.flashinfer_root / OPT_RELATIVE

    import torch

    import capture_opt_phase
    import exp017_opt_phase_common as common

    if common.file_sha256(args.overlay) != common.EXPECTED_OPT_SHA256:
        raise RuntimeError("current Latest-opt source identity drift")
    if args.jit_root.exists() and any(args.jit_root.iterdir()):
        raise RuntimeError(f"dedicated NCU JIT root is not empty: {args.jit_root}")
    args.jit_root.mkdir(parents=True, exist_ok=True)

    _, core = capture_opt_phase.load_reused()
    core.reused.BASELINE = core.BASELINE
    core.reused.CANDIDATE = core.CANDIDATE
    core.reused.ARMS = core.ARMS
    core.reused.EXPECTED_OVERLAY_SHA256 = dict(core.EXPECTED_OVERLAY_SHA256)
    source = core.reused.validate_source(args.flashinfer_root, args.overlay, ARM)
    if core.reused.TARGET_MODULE in sys.modules:
        raise RuntimeError(
            "Latest-opt module imported before exact overlay installation"
        )
    core.worker.install_overlay(args.overlay)
    imports = core.reused.configure_source_checkout(args.flashinfer_root, args.jit_root)
    if Path(imports["target_module"]).resolve() != args.overlay:
        raise RuntimeError("current Latest-opt module did not resolve to exact source")
    runtime = core.reused.runtime_identity(args, source)
    runtime["imports"] = imports

    fixture_module, fixture, weights = capture_opt_phase.make_exp001_case(args, core)
    reference = fixture_module.reference_moe_nvfp4(fixture, weights).detach().cpu()
    arm = core.worker.build_arm(args, fixture, weights)
    eager = gate_output(fixture_module, arm.eager(), reference, core, fixture, arm)
    if not eager["gate_pass"]:
        raise RuntimeError(f"Latest-opt eager correctness failed: {eager}")
    arm.capture()
    for _ in range(args.warmup):
        arm.replay()
    pre_profile_output, _ = arm.replay(sentinel=True)
    pre_profile = gate_output(
        fixture_module, pre_profile_output, reference, core, fixture, arm
    )
    if not pre_profile["gate_pass"]:
        raise RuntimeError(
            f"Latest-opt pre-profile graph correctness failed: {pre_profile}"
        )

    specialization = core.specialization_contract(args.overlay)
    nvtx_range = "exp017_latest_opt_m8192_ncu"
    cudart = torch.cuda.cudart()
    started = False
    body_error: BaseException | None = None
    output = None
    elapsed_ms = 0.0
    try:
        status = int(cudart.cudaProfilerStart())
        if status != 0:
            raise RuntimeError(f"cudaProfilerStart failed: {status}")
        started = True
        torch.cuda.nvtx.range_push(nvtx_range)
        try:
            output, elapsed_ms = arm.replay(sentinel=True)
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
    except BaseException as error:
        body_error = error
        raise
    finally:
        if started:
            torch.cuda.synchronize()
            status = int(cudart.cudaProfilerStop())
            if status != 0 and body_error is None:
                raise RuntimeError(f"cudaProfilerStop failed: {status}")

    assert output is not None
    post_profile = gate_output(fixture_module, output, reference, core, fixture, arm)
    if not post_profile["gate_pass"]:
        raise RuntimeError(
            f"Latest-opt post-profile correctness failed: {post_profile}"
        )

    artifacts, artifact_set, cubins = core._artifacts(args.jit_root)
    if len(cubins) != 1:
        raise RuntimeError(f"expected one current Latest-opt cubin: {cubins}")
    resources = selected_resource(core, args.jit_root, artifacts)
    payload = {
        "schema": "exp017.latest-opt-ncu-target.v1",
        "status": "complete",
        "case": {
            "m": M,
            "fixture": fixture.manifest,
            "weights": weights.manifest,
            "scale_kind": "equal",
        },
        "source": {
            **source,
            "current_opt_sha256": common.EXPECTED_OPT_SHA256,
            "dispatch_sha256": common.file_sha256(
                args.flashinfer_root / common.DISPATCH_RELATIVE_PATH
            ),
            "wrapper_sha256": common.file_sha256(
                args.flashinfer_root / common.WRAPPER_RELATIVE_PATH
            ),
        },
        "runtime": runtime,
        "correctness": {
            "eager": eager,
            "pre_profile_graph": pre_profile,
            "post_profile_graph": post_profile,
        },
        "specialization": specialization,
        "profile": {
            "nvtx_range": nvtx_range,
            "warmup_graph_replays": args.warmup,
            "event_elapsed_us": elapsed_ms * 1000.0,
            "boundary": "one current Latest-opt CUDA Graph replay",
        },
        "jit_artifacts": artifacts,
        "jit_artifact_set_sha256": artifact_set,
        "cubin_sha256": cubins[0],
        "static_resource_usage": resources,
    }
    write_json(args.output, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-app-clock-mhz", type=int, default=2377)
    parser.add_argument("--warmup", type=int, default=WARMUP, choices=(WARMUP,))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = run(parse_args(argv))
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "status": payload["status"],
                "cubin_sha256": payload["cubin_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
