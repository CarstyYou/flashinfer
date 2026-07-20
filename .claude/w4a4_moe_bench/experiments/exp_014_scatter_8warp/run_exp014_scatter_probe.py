#!/usr/bin/env python3
"""Capture one identity-locked exp_014 Scatter phase probe arm."""

from __future__ import annotations

import argparse
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
from typing import Any, Mapping, Sequence

from exp014_scatter_probe_common import (
    ARMS,
    BASE_OVERLAY_ROOT,
    DISPATCH_MODULE,
    DISPATCH_RELATIVE_PATH,
    EVENT_ABI,
    EXPECTED_BASE_KERNEL_SHA256,
    EXPECTED_DISPATCH_SHA256,
    EXPECTED_WRAPPER_SHA256,
    KERNEL_MODULE,
    PROBE_OVERLAY_ROOT,
    RESULTS,
    SENTINEL,
    TASK_TICKS,
    WRAPPER_RELATIVE_PATH,
    barrier_fingerprint,
    canonical_sha256,
    file_sha256,
    interval_rows,
    read_json,
    summarize_intervals,
    validate_probe_ticks,
    write_json,
)


ROOT = Path(__file__).resolve().parent
M_VALUES = (256, 512, 1024, 2048, 4096, 8192)
WARMUPS = 2
REPLAYS = 5


class ExactModuleOverlayFinder(importlib.abc.MetaPathFinder):
    def __init__(self, mapping: Mapping[str, Path]):
        self.mapping = {name: path.resolve() for name, path in mapping.items()}

    def find_spec(self, fullname: str, path=None, target=None):
        del path, target
        overlay = self.mapping.get(fullname)
        if overlay is None:
            return None
        return importlib.util.spec_from_file_location(fullname, overlay)


def overlay_paths(root: Path, arm: str) -> tuple[Path, Path, Path]:
    arm_root = root.resolve() / arm
    return (
        arm_root,
        arm_root / "moe_dynamic_kernel.py",
        arm_root / "moe_dispatch.py",
    )


def install_overlays(kernel: Path, dispatch: Path) -> None:
    imported = [
        module for module in (KERNEL_MODULE, DISPATCH_MODULE) if module in sys.modules
    ]
    if imported:
        raise RuntimeError(f"target modules imported before probe overlays: {imported}")
    sys.meta_path.insert(
        0,
        ExactModuleOverlayFinder({KERNEL_MODULE: kernel, DISPATCH_MODULE: dispatch}),
    )


def overlay_identity_gate(args: argparse.Namespace) -> dict[str, Any]:
    arm_root, kernel, dispatch = overlay_paths(args.overlay_root, args.arm)
    repo = args.flashinfer_root.resolve()
    errors = []
    root_identity_path = args.overlay_root.resolve() / "identity.json"
    arm_identity_path = arm_root / "identity.json"
    root_identity: Mapping[str, Any] = {}
    arm_identity: Mapping[str, Any] = {}
    if not root_identity_path.is_file():
        errors.append("root probe identity missing")
    else:
        root_identity = read_json(root_identity_path)
    if not arm_identity_path.is_file():
        errors.append("arm probe identity missing")
    else:
        arm_identity = read_json(arm_identity_path)

    if root_identity:
        if root_identity.get("schema") != "exp014.scatter-phase-probe-overlays.v1":
            errors.append("root probe identity schema drift")
        if root_identity.get("event_abi") != EVENT_ABI:
            errors.append("root event ABI drift")
        if not root_identity.get("cross_arm", {}).get("gate_pass"):
            errors.append("cross-arm matched-probe gate is not closed")
    if arm_identity:
        if arm_identity.get("schema") != "exp014.scatter-phase-probe-overlay.v1":
            errors.append("arm probe identity schema drift")
        if arm_identity.get("arm") != args.arm:
            errors.append("arm identity drift")
        if arm_identity.get("event_abi") != EVENT_ABI:
            errors.append("arm event ABI drift")
        if not arm_identity.get("probe_enabled"):
            errors.append("probe-enabled identity is false")
        overlay = arm_identity.get("overlay", {})
        if not kernel.is_file() or file_sha256(kernel) != overlay.get("kernel_sha256"):
            errors.append("kernel probe overlay hash drift")
        if not dispatch.is_file() or file_sha256(dispatch) != overlay.get(
            "dispatch_sha256"
        ):
            errors.append("dispatch probe overlay hash drift")
        base = arm_identity.get("base", {})
        base_kernel = BASE_OVERLAY_ROOT / args.arm / "moe_dynamic_kernel.py"
        if (
            not base_kernel.is_file()
            or file_sha256(base_kernel) != EXPECTED_BASE_KERNEL_SHA256[args.arm]
            or base.get("kernel_sha256") != EXPECTED_BASE_KERNEL_SHA256[args.arm]
        ):
            errors.append("base kernel identity drift")
        if kernel.is_file() and barrier_fingerprint(
            kernel.read_text(encoding="utf-8")
        ) != base.get("barrier_fingerprint"):
            errors.append("probe changed the base barrier fingerprint")

    live_sources = (
        (repo / DISPATCH_RELATIVE_PATH, EXPECTED_DISPATCH_SHA256, "dispatch"),
        (repo / WRAPPER_RELATIVE_PATH, EXPECTED_WRAPPER_SHA256, "wrapper"),
    )
    for path, expected, label in live_sources:
        if not path.is_file() or file_sha256(path) != expected:
            errors.append(f"live {label} identity drift")
    return {
        "schema": "exp014.scatter-phase-overlay-gate.v1",
        "arm": args.arm,
        "kernel": str(kernel),
        "dispatch": str(dispatch),
        "root_identity": dict(root_identity),
        "arm_identity": dict(arm_identity),
        "errors": errors,
        "gate_pass": not errors,
    }


def load_gpu_modules():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - GPU image only
        raise RuntimeError("Scatter probe capture requires Torch/CUDA") from error
    import run_exp014_arm as core

    return torch, core


def reset_probe(workspace: Any) -> None:
    workspace.exp014_scatter_ticks.fill_(SENTINEL)


def snapshot_probe(workspace: Any) -> dict[str, Any]:
    ticks = workspace.exp014_scatter_ticks.detach().cpu().clone()
    task_capacity = int(workspace.task_capacity)
    task_tail = int(workspace.task_tail.item())
    expected_numel = task_capacity * TASK_TICKS
    if ticks.dtype.__str__() != "torch.int64":
        raise RuntimeError(f"probe storage dtype drift: {ticks.dtype}")
    if ticks.numel() != expected_numel:
        raise RuntimeError(
            f"probe storage capacity drift: {ticks.numel()} != {expected_numel}"
        )
    slices = workspace.task_slice_count.detach().cpu().clone()[:task_tail]
    valid_rows = workspace.task_valid_rows.detach().cpu().clone()[:task_tail]
    gate = validate_probe_ticks(
        ticks,
        task_tail=task_tail,
        task_capacity=task_capacity,
        task_slice_count=slices,
        task_valid_rows=valid_rows,
    )
    rows = interval_rows(ticks, task_tail=task_tail, task_capacity=task_capacity)
    return {
        "ticks": ticks,
        "task_tail": task_tail,
        "task_capacity": task_capacity,
        "task_slice_count": slices,
        "task_valid_rows": valid_rows,
        "gate": gate,
        "summary": summarize_intervals(rows),
    }


def correctness_gate(diagnostics: Mapping[str, Any], output: Any) -> dict[str, Any]:
    full_zero_rows = int((output == 0).all(dim=-1).sum().item())
    checks = {
        "formal_pass": bool(diagnostics.get("formal_pass")),
        "finite": bool(diagnostics.get("finite")),
        "no_full_zero_rows": full_zero_rows == 0,
    }
    return {
        "checks": checks,
        "full_zero_rows": full_zero_rows,
        "gate_pass": all(checks.values()),
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    if args.warmup != WARMUPS or args.replays != REPLAYS:
        raise RuntimeError("probe protocol requires warmup=2 and replays=5")
    overlay_gate = overlay_identity_gate(args)
    if not overlay_gate["gate_pass"]:
        raise RuntimeError(f"probe overlay gate failed: {overlay_gate['errors']}")

    torch, core = load_gpu_modules()
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.jit_root = args.jit_root.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"immutable probe capture exists: {args.output}")
    staging = args.output.with_name(f".{args.output.name}.in-progress.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale probe staging exists: {staging}")
    core.common.require_empty_directory(args.jit_root)
    arm_root, kernel, dispatch = overlay_paths(args.overlay_root, args.arm)
    del arm_root
    install_overlays(kernel, dispatch)
    imports = core.configure_source_checkout(args.flashinfer_root, args.jit_root)
    if Path(imports["target_module"]) != kernel:
        raise RuntimeError("kernel module did not resolve to the probe overlay")
    imported_dispatch = importlib.import_module(DISPATCH_MODULE)
    if Path(imported_dispatch.__file__).resolve() != dispatch:
        raise RuntimeError("dispatch module did not resolve to the probe overlay")

    source = {
        "arm": args.arm,
        "kernel_overlay": str(kernel),
        "kernel_sha256": file_sha256(kernel),
        "dispatch_overlay": str(dispatch),
        "dispatch_sha256": file_sha256(dispatch),
        "overlay_identity": overlay_gate["arm_identity"],
        "event_abi": EVENT_ABI,
    }
    runtime = core.runtime_identity(args, source)
    runtime["imports"] = imports
    staging.mkdir(parents=True)
    try:
        fixture_module = core.worker.load_fixture_module()
        device = torch.device("cuda", args.device_index)
        fixture = fixture_module.make_routed_fixture(
            args.m, device=device, seed=args.seed
        )
        weights = fixture_module.make_canonical_weights(device=device, seed=args.seed)
        reference = fixture_module.reference_moe_nvfp4(fixture, weights)
        captured = core.worker.build_arm(args, fixture, weights)

        eager_output = captured.eager().detach().cpu().clone()
        workspace = captured.wrapper._dynamic_workspace
        if workspace is None or not hasattr(workspace, "exp014_scatter_ticks"):
            raise RuntimeError("probe workspace storage is missing")
        eager_timing = snapshot_probe(workspace)
        eager_diagnostics = fixture_module.output_diagnostics(
            eager_output, reference.cpu()
        )
        _, eager_workspace = core.worker._workspace_snapshot(
            captured.wrapper, fixture, num_cta_warps=9
        )
        eager = {
            "correctness": eager_diagnostics,
            "correctness_gate": correctness_gate(eager_diagnostics, eager_output),
            "route_task_gate": eager_workspace["verification"],
            "buffer_gate": eager_timing["gate"],
            "interval_summary": eager_timing["summary"],
            "ticks_sha256": core.worker.tensor_sha256(eager_timing["ticks"]),
        }
        torch.save(eager_timing["ticks"], staging / "eager_ticks.pt")
        write_json(staging / "eager.json", eager)
        if not (
            eager["correctness_gate"]["gate_pass"]
            and eager["route_task_gate"]["gate_pass"]
            and eager["buffer_gate"]["gate_pass"]
        ):
            raise RuntimeError(
                "eager correctness/route/probe gate failed: "
                f"correctness={eager['correctness_gate']}, "
                f"diagnostics={eager_diagnostics}, "
                f"route={eager['route_task_gate']}, "
                f"probe={eager['buffer_gate']}"
            )

        captured.capture()
        for _ in range(args.warmup):
            reset_probe(workspace)
            captured.replay(sentinel=False)

        runs = []
        for replay in range(args.replays):
            reset_probe(workspace)
            output, elapsed_ms = captured.replay(sentinel=True)
            output_cpu = output.detach().cpu().clone()
            timing = snapshot_probe(workspace)
            diagnostics = fixture_module.output_diagnostics(output_cpu, reference.cpu())
            _, route = core.worker._workspace_snapshot(
                captured.wrapper, fixture, num_cta_warps=9
            )
            run = {
                "replay": replay,
                "event_elapsed_us": elapsed_ms * 1000.0,
                "output_sha256": core.worker.tensor_sha256(output_cpu),
                "correctness": diagnostics,
                "correctness_gate": correctness_gate(diagnostics, output_cpu),
                "route_task_gate": route["verification"],
                "buffer_gate": timing["gate"],
                "interval_summary": timing["summary"],
                "ticks_sha256": core.worker.tensor_sha256(timing["ticks"]),
            }
            run["gate_pass"] = bool(
                run["correctness_gate"]["gate_pass"]
                and run["route_task_gate"]["gate_pass"]
                and run["buffer_gate"]["gate_pass"]
            )
            torch.save(timing["ticks"], staging / f"ticks_{replay}.pt")
            write_json(staging / f"run_{replay}.json", run)
            if not run["gate_pass"]:
                raise RuntimeError(f"probe replay {replay} gate failed")
            runs.append(run)

        artifacts = core.common.artifact_manifest(args.jit_root)
        cubins = [item for item in artifacts if item["path"].endswith(".cubin")]
        if not cubins:
            raise RuntimeError("fresh probe JIT retained no cubin")
        elapsed = [float(run["event_elapsed_us"]) for run in runs]
        payload = {
            "schema": "exp014.scatter-phase-probe-capture.v1",
            "classification": "diagnostic-only",
            "arm": args.arm,
            "m": args.m,
            "fixture_kind": "canonical",
            "event_abi": EVENT_ABI,
            "source": source,
            "overlay_gate": overlay_gate,
            "runtime": runtime,
            "fixture": fixture.manifest,
            "weights": weights.manifest,
            "reference_sha256": core.worker.tensor_sha256(reference),
            "eager": eager,
            "runs": runs,
            "probe_e2e_us": {
                "median": statistics.median(elapsed),
                "min": min(elapsed),
                "max": max(elapsed),
                "samples": len(elapsed),
            },
            "jit_artifacts": artifacts,
            "jit_artifact_set_sha256": canonical_sha256(artifacts),
            "cubin_sha256": sorted({str(item["sha256"]) for item in cubins}),
            "compile_identity": core.worker._compile_identity(),
            "evidence_boundary": (
                "phase intervals are diagnostic matched-probe evidence; probe E2E "
                "does not replace the uninstrumented exp014 ABBA verdict"
            ),
        }
        write_json(staging / "capture.json", payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(args.output)
        return payload
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--overlay-root", type=Path, default=PROBE_OVERLAY_ROOT)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-app-clock-mhz", type=int, required=True)
    parser.add_argument("--m", type=int, choices=M_VALUES, default=8192)
    parser.add_argument("--device-index", type=int, choices=(0,), default=0)
    parser.add_argument("--seed", type=int, choices=(2026,), default=2026)
    parser.add_argument("--warmup", type=int, choices=(WARMUPS,), default=WARMUPS)
    parser.add_argument("--replays", type=int, choices=(REPLAYS,), default=REPLAYS)
    parser.add_argument("--check-overlays-only", action="store_true")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = RESULTS / "raw/scatter_phase_probe" / args.arm / f"m{args.m}"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    gate = overlay_identity_gate(args)
    if args.check_overlays_only:
        print(json.dumps(gate, sort_keys=True))
        return 0 if gate["gate_pass"] else 1
    payload = capture(args)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
