#!/usr/bin/env python3
"""Run one identity-locked marker arm as the target of a native NCU capture.

This is not a benchmark.  It rebuilds the same marker-disabled control or
marker-enabled probe in an independent fresh JIT namespace, proves that its
cubin matches the standalone timing capture, and exposes exactly one final
CUDA Graph replay between cudaProfilerStart/Stop.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capture_phase_timing import (  # noqa: E402
    M,
    _artifact_gate,
    _correctness_gate,
    _event_gate,
    _load_gpu_modules,
    _overlay_paths,
    _reset,
    _snapshot,
    install_overlays,
    overlay_identity_gate,
)
from exp008_marker_common import (  # noqa: E402
    DISPATCH_MODULE,
    EVENT_ABI,
    MARKER_ARMS,
    VERSIONS,
    canonical_sha256,
    read_json,
    sha256_file,
    write_json,
)


WARMUPS = 5


def _artifact_hashes(
    artifacts: Sequence[Mapping[str, Any]], suffix: str
) -> list[str]:
    values = [
        str(item.get("sha256"))
        for item in artifacts
        if str(item.get("path", "")).endswith(suffix)
    ]
    return sorted(values)


def _timing_capture_gate(
    capture: Mapping[str, Any], *, version: str, arm: str
) -> dict[str, Any]:
    runs = capture.get("runs", [])
    checks = {
        "schema": capture.get("schema") == "exp008.phase-marker-capture.v1",
        "version": capture.get("version") == version,
        "arm": capture.get("arm") == arm,
        "event_abi": capture.get("event_abi") == EVENT_ABI,
        "overlay_gate": capture.get("overlay_gate", {}).get("gate_pass") is True,
        "jit_gate": capture.get("jit_identity_gate", {}).get("gate_pass") is True,
        "measured_replays": len(runs) == 5,
        "all_replays_pass": bool(runs)
        and all(run.get("gate_pass") is True for run in runs),
        "artifact_manifest": canonical_sha256(capture.get("jit_artifacts", []))
        == capture.get("jit_identity_gate", {}).get("artifact_set_sha256"),
    }
    return {"checks": checks, "gate_pass": all(checks.values())}


def profile(args: argparse.Namespace) -> dict[str, Any]:
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.overlay_root = args.overlay_root.resolve()
    args.jit_root = args.jit_root.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"immutable profile target exists: {args.output}")
    timing_capture = read_json(args.timing_capture.resolve())
    timing_gate = _timing_capture_gate(
        timing_capture, version=args.version, arm=args.arm
    )
    if not timing_gate["gate_pass"]:
        raise RuntimeError(f"timing capture prerequisite failed: {timing_gate}")
    overlay_gate = overlay_identity_gate(args)
    if not overlay_gate["gate_pass"]:
        raise RuntimeError(f"overlay identity gate failed: {overlay_gate['errors']}")

    torch, worker = _load_gpu_modules()
    worker.require_empty_directory(args.jit_root)
    _, kernel, dispatch = _overlay_paths(args)
    install_overlays(kernel, dispatch)
    imports = worker.configure_source_checkout(args.flashinfer_root)
    if Path(imports["target_module"]) != kernel:
        raise RuntimeError("kernel module did not resolve to selected overlay")
    imported_dispatch = importlib.import_module(DISPATCH_MODULE)
    if Path(imported_dispatch.__file__).resolve() != dispatch:
        raise RuntimeError("dispatch module did not resolve to selected overlay")

    source = {
        "version": args.version,
        "arm": args.arm,
        "kernel_overlay": str(kernel),
        "kernel_sha256": sha256_file(kernel),
        "dispatch_overlay": str(dispatch),
        "dispatch_sha256": sha256_file(dispatch),
        "overlay_identity": overlay_gate["manifest"],
    }
    timing_source = timing_capture.get("source", {})
    if any(
        source[field] != timing_source.get(field)
        for field in ("version", "arm", "kernel_sha256", "dispatch_sha256")
    ):
        raise RuntimeError("profile source identity != standalone timing source")
    runtime = worker.runtime_identity(args, source)
    runtime["imports"] = imports
    if int(runtime.get("gpu", {}).get("applications_graphics_clock_mhz", -1)) != int(
        args.expected_app_clock_mhz
    ):
        raise RuntimeError("profile application graphics clock drift")
    if runtime.get("gpu", {}).get("uuid") != timing_capture.get("runtime", {}).get(
        "gpu", {}
    ).get("uuid"):
        raise RuntimeError("profile GPU UUID != standalone timing GPU UUID")

    fixture_module, fixture, weights = worker.make_case(args)
    reference = fixture_module.reference_moe_nvfp4(fixture, weights)
    if fixture.manifest != timing_capture.get("fixture"):
        raise RuntimeError("profile fixture != standalone timing fixture")
    if weights.manifest != timing_capture.get("weights"):
        raise RuntimeError("profile weights != standalone timing weights")
    if worker.tensor_sha256(reference) != timing_capture.get("reference_sha256"):
        raise RuntimeError("profile reference != standalone timing reference")

    captured = worker.build_arm(args, fixture, weights)
    eager = captured.eager()
    eager_error = worker.tensor_error(eager.detach().cpu(), reference.cpu())
    workspace = captured.wrapper._dynamic_workspace
    eager_timing = _snapshot(workspace)
    eager_event = _event_gate(eager_timing, arm=args.arm)
    _, eager_workspace = worker._workspace_snapshot(
        captured.wrapper, fixture, num_cta_warps=9
    )
    eager_gate = {
        "correctness": _correctness_gate(eager_error),
        "event": eager_event,
        "workspace": eager_workspace["verification"],
    }
    if not all(value.get("gate_pass", False) for value in eager_gate.values()):
        raise RuntimeError(f"profile eager gate failed: {eager_gate}")

    captured.capture()
    for _ in range(args.warmup):
        _reset(workspace)
        captured.replay(sentinel=False)

    nvtx = f"exp008_marker_{args.version}_{args.arm}_m8192_final_replay"
    torch.cuda.nvtx.range_push(nvtx)
    cudart = torch.cuda.cudart()
    try:
        if int(cudart.cudaProfilerStart()) != 0:
            raise RuntimeError("cudaProfilerStart failed")
        _reset(workspace)
        output, elapsed_ms = captured.replay(sentinel=True)
        if int(cudart.cudaProfilerStop()) != 0:
            raise RuntimeError("cudaProfilerStop failed")
    finally:
        torch.cuda.nvtx.range_pop()

    error = worker.tensor_error(output.detach().cpu(), reference.cpu())
    timing = _snapshot(workspace)
    event_gate = _event_gate(timing, arm=args.arm)
    _, workspace_payload = worker._workspace_snapshot(
        captured.wrapper, fixture, num_cta_warps=9
    )
    final_gates = {
        "correctness": _correctness_gate(error),
        "event": event_gate,
        "workspace": workspace_payload["verification"],
    }
    if not all(value.get("gate_pass", False) for value in final_gates.values()):
        raise RuntimeError(f"profile final-replay gate failed: {final_gates}")

    artifacts, artifact_gate = _artifact_gate(worker, args.jit_root)
    if not artifact_gate["gate_pass"]:
        raise RuntimeError(f"profile JIT artifact gate failed: {artifact_gate}")
    timing_artifacts = timing_capture["jit_artifacts"]
    binary_checks = {
        suffix: _artifact_hashes(artifacts, suffix)
        == _artifact_hashes(timing_artifacts, suffix)
        and bool(_artifact_hashes(artifacts, suffix))
        for suffix in (".cubin", ".ptx")
    }
    if not all(binary_checks.values()):
        raise RuntimeError(
            "profile binary/PTX identity != standalone timing capture: "
            f"{binary_checks}"
        )

    payload = {
        "schema": "exp008.phase-marker-profile-target.v1",
        "status": "complete",
        "version": args.version,
        "arm": args.arm,
        "m": M,
        "fixture": "canonical",
        "nvtx_range": nvtx,
        "event_elapsed_us_diagnostic_only": elapsed_ms * 1000.0,
        "output_sha256": worker.tensor_sha256(output),
        "source": source,
        "runtime": runtime,
        "expected_launch": {
            "grid": [1, 1, 110],
            "block": [288, 1, 1],
            "kernel": "MoEDynamicKernel",
        },
        "standalone_timing_capture": str(args.timing_capture.resolve()),
        "standalone_timing_artifact_set_sha256": timing_capture[
            "jit_identity_gate"
        ]["artifact_set_sha256"],
        "profile_jit_artifact_set_sha256": artifact_gate["artifact_set_sha256"],
        "profile_jit_artifacts": artifacts,
        "binary_identity_checks": binary_checks,
        "eager_gates": eager_gate,
        "final_replay_gates": final_gates,
        "gate_pass": all(binary_checks.values())
        and all(value.get("gate_pass", False) for value in final_gates.values()),
    }
    write_json(args.output, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--version", choices=VERSIONS, required=True)
    parser.add_argument("--arm", choices=MARKER_ARMS, required=True)
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--timing-capture", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-app-clock-mhz", type=int, default=2377)
    parser.add_argument("--m", type=int, default=M, choices=[M])
    parser.add_argument("--fixture", default="canonical", choices=["canonical"])
    parser.add_argument("--device-index", type=int, default=0, choices=[0])
    parser.add_argument("--seed", type=int, default=2026, choices=[2026])
    parser.add_argument("--warmup", type=int, default=WARMUPS, choices=[WARMUPS])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = profile(args)
    print(
        json.dumps(
            {
                "version": args.version,
                "arm": args.arm,
                "gate_pass": payload["gate_pass"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
