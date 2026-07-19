#!/usr/bin/env python3
"""Capture one matched exp_006 control or completion-anchored probe arm.

GPU imports are intentionally lazy so the ABI/event helpers remain importable
and testable on a CPU-only frontend.  A capture is fail-closed on source,
overlay, environment, correctness, workspace/descriptor, or event-contract
drift.  Every ``timing_*.pt`` retains the five workspace task descriptors.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
EXP004_ROOT = ROOT.parent / "exp_004_fused_phase_timing_breakdown"
if str(EXP004_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP004_ROOT))

from exp004_common import (  # noqa: E402
    DISPATCH_MODULE,
    EXPECTED_BLOCK,
    EXPECTED_GRID,
    EXPECTED_TASK_TAIL,
    MEASURED_REPLAYS,
    artifact_manifest,
    canonical_sha256,
    file_sha256,
    read_json,
    require_empty_directory,
    write_json,
)

from exp006_common import (  # noqa: E402
    ARMS,
    CONTROL,
    CTA_TICKS,
    DESCRIPTOR_NAMES,
    EVENT_ABI,
    OUTPUT_TILES,
    PROBE,
    SENTINEL,
    TASK_TICKS,
    descriptor_order_sha256,
    validate_control_events,
    validate_descriptors,
    validate_probe_events,
)


def _load_gpu_modules():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised only on bad GPU image
        raise RuntimeError(
            "capture requires the locked Torch/CUDA environment"
        ) from error
    import run_exp004_arm as worker

    return torch, worker


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _descriptor_payload(
    workspace_tensors: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[int]], str]:
    tensors = {
        name: workspace_tensors[name].detach().cpu().clone()
        for name in DESCRIPTOR_NAMES
    }
    plain = {
        name: [int(value) for value in _as_list(tensor)]
        for name, tensor in tensors.items()
    }
    digest = descriptor_order_sha256(plain)
    return tensors, plain, digest


def _snapshot(
    arm: Any,
    workspace_tensors: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[int]], str]:
    workspace = arm.wrapper._dynamic_workspace
    task_capacity = int(workspace.task_capacity)
    task_tail = int(workspace.task_tail.item())
    descriptor_tensors, descriptors, descriptor_hash = _descriptor_payload(
        workspace_tensors
    )
    timing = {
        "task_ticks": workspace.exp004_timing_ticks.detach()
        .reshape(task_capacity, TASK_TICKS)
        .cpu()
        .clone(),
        "task_cta_z": workspace.exp004_task_cta_z.detach().cpu().clone(),
        "cta_ticks": workspace.exp004_cta_ticks.detach()
        .reshape(-1, CTA_TICKS)
        .cpu()
        .clone(),
        **descriptor_tensors,
        "task_tail": task_tail,
        "task_capacity": task_capacity,
        "descriptor_order_sha256": descriptor_hash,
    }
    return timing, descriptors, descriptor_hash


def _reset(arm: Any) -> None:
    workspace = arm.wrapper._dynamic_workspace
    workspace.exp004_timing_ticks.fill_(SENTINEL)
    workspace.exp004_task_cta_z.fill_(SENTINEL)
    workspace.exp004_cta_ticks.fill_(SENTINEL)


def _event_gate(
    timing: Mapping[str, Any],
    *,
    arm_name: str,
    task_tail: int,
    task_capacity: int,
    grid_z: int,
) -> dict[str, Any]:
    args = (
        _as_list(timing["task_ticks"]),
        _as_list(timing["task_cta_z"]),
        _as_list(timing["cta_ticks"]),
    )
    if arm_name == PROBE:
        return validate_probe_events(
            *args,
            task_tail=task_tail,
            task_capacity=task_capacity,
            grid_z=grid_z,
        )
    if arm_name == CONTROL:
        return validate_control_events(
            *args, task_capacity=task_capacity, grid_z=grid_z
        )
    raise AssertionError(arm_name)


def _overlay_identity_gate(args: argparse.Namespace) -> dict[str, Any]:
    kernel_dir = args.kernel_overlay.parent.resolve()
    dispatch_dir = args.dispatch_overlay.parent.resolve()
    errors: list[str] = []
    if kernel_dir != dispatch_dir:
        errors.append("kernel and dispatch overlays do not share one immutable root")
        manifest: Mapping[str, Any] = {}
    else:
        identity_path = kernel_dir / "identity.json"
        if not identity_path.is_file():
            errors.append("overlay identity.json is missing")
            manifest = {}
        else:
            manifest = read_json(identity_path)

    expected_enabled = args.arm == PROBE
    if manifest:
        if manifest.get("schema") != "exp006.completion-overlay.v1":
            errors.append("overlay schema drift")
        if manifest.get("arm") != args.arm:
            errors.append("overlay arm does not match requested arm")
        if bool(manifest.get("probe_enabled")) != expected_enabled:
            errors.append("overlay probe-enable flag drift")
        if manifest.get("event_abi") != EVENT_ABI:
            errors.append("overlay event ABI drift")
        if not manifest.get("static_contract", {}).get("gate_pass", False):
            errors.append("overlay static epi/task-slice contract did not pass")
        hashes = manifest.get("overlay", {})
        if hashes.get("kernel_sha256") != file_sha256(args.kernel_overlay):
            errors.append("kernel overlay hash drift")
        if hashes.get("dispatch_sha256") != file_sha256(args.dispatch_overlay):
            errors.append("dispatch overlay hash drift")
    return {
        "schema": "exp006.overlay-identity-gate.v1",
        "overlay_root": str(kernel_dir),
        "arm": args.arm,
        "manifest": dict(manifest),
        "errors": errors,
        "gate_pass": not errors,
    }


def _jit_identity_gate(artifacts: list[Mapping[str, Any]]) -> dict[str, Any]:
    suffix_counts = {
        suffix: sum(str(item.get("path", "")).endswith(suffix) for item in artifacts)
        for suffix in (".so", ".cubin", ".ptx", ".sass")
    }
    errors: list[str] = []
    if not artifacts:
        errors.append("fresh JIT artifact set is empty")
    if suffix_counts[".so"] == 0:
        errors.append("fresh JIT retained no loadable .so")
    return {
        "schema": "exp006.jit-identity-gate.v1",
        "suffix_counts": suffix_counts,
        "retained_cutedsl_static_artifacts": {
            suffix: suffix_counts[suffix] for suffix in (".cubin", ".ptx", ".sass")
        },
        "static_extraction_required_post_capture": True,
        "artifact_set_sha256": canonical_sha256(artifacts),
        "errors": errors,
        "gate_pass": not errors,
    }


def _runtime_case_gate(
    worker: Any,
    workspace: Any,
    descriptor_gate: Mapping[str, Any],
) -> dict[str, Any]:
    dispatch = importlib.import_module(DISPATCH_MODULE)
    cache_keys = list(dispatch._DYNAMIC_KERNEL_CACHE.keys())
    compile_identity = worker.compile_identity()
    tilers = sorted(
        {
            tuple(value)
            for key in cache_keys
            for value in key
            if isinstance(value, tuple) and value == (128, 128)
        }
    )
    checks = {
        "grid_z_110": int(workspace.exp004_cta_ticks.numel() // CTA_TICKS)
        == EXPECTED_GRID[2],
        "task_tail": int(workspace.task_tail.item()) == EXPECTED_TASK_TAIL,
        "task_slice_count_one": bool(descriptor_gate.get("gate_pass")),
        "dynamic_tiler_128x128": tilers == [(128, 128)],
        "output_tiles_16": OUTPUT_TILES == 16,
        "max_active_clusters_110": compile_identity.get("max_active_clusters")
        == [EXPECTED_GRID[2]],
    }
    return {
        "schema": "exp006.runtime-case-gate.v1",
        "checks": checks,
        "dynamic_cache_entries": len(cache_keys),
        "dynamic_tilers": [list(value) for value in tilers],
        "expected_launch": {
            "grid": list(EXPECTED_GRID),
            "block": list(EXPECTED_BLOCK),
        },
        "compile_identity": compile_identity,
        "gate_pass": all(checks.values()),
    }


def _tensor_hashes(worker: Any, timing: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: worker.tensor_sha256(value)
        for name, value in timing.items()
        if hasattr(value, "detach")
    }


def _gate_failures(payload: Mapping[str, Any]) -> list[str]:
    names = (
        "correctness_gate",
        "output_contract",
        "workspace_gate",
        "descriptor_gate",
        "event_gate",
        "runtime_case_gate",
    )
    return [name for name in names if not payload.get(name, {}).get("gate_pass", False)]


def capture(args: argparse.Namespace) -> dict[str, Any]:
    if args.warmup != 2 or args.replays != MEASURED_REPLAYS:
        raise RuntimeError(
            "exp006 locked protocol requires exactly 2 warmups and 5 measured replays"
        )
    torch, worker = _load_gpu_modules()
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.kernel_overlay = args.kernel_overlay.resolve()
    args.dispatch_overlay = args.dispatch_overlay.resolve()
    args.jit_root = args.jit_root.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"immutable capture output exists: {args.output}")
    require_empty_directory(args.jit_root)

    overlay_gate = _overlay_identity_gate(args)
    if not overlay_gate["gate_pass"]:
        raise RuntimeError(f"overlay identity gate failed: {overlay_gate['errors']}")
    source = worker.validate_source(args)
    worker.install_overlays(args.kernel_overlay, args.dispatch_overlay)
    imports = worker.configure_source_checkout(args.flashinfer_root)
    runtime = worker.runtime_identity(args, source)
    runtime["imports"] = imports
    args.output.mkdir(parents=True)

    fixture_module, fixture, weights = worker.make_case()
    reference = fixture_module.reference_moe_nvfp4(fixture, weights)
    arm = worker.build_arm(fixture, weights)

    eager_output = arm.eager().detach().cpu().clone()
    eager_workspace_tensors, eager_workspace = worker.workspace_snapshot(
        arm.wrapper, fixture
    )
    eager_timing, eager_descriptors, eager_descriptor_hash = _snapshot(
        arm, eager_workspace_tensors
    )
    eager_descriptor_gate = validate_descriptors(
        eager_descriptors,
        task_tail=int(arm.wrapper._dynamic_workspace.task_tail.item()),
    )
    eager_event_gate = _event_gate(
        eager_timing,
        arm_name=args.arm,
        task_tail=int(arm.wrapper._dynamic_workspace.task_tail.item()),
        task_capacity=int(arm.wrapper._dynamic_workspace.task_capacity),
        grid_z=int(eager_timing["cta_ticks"].shape[0]),
    )
    eager_correctness = worker.tensor_error(eager_output, reference.cpu())
    eager_payload: dict[str, Any] = {
        "correctness": eager_correctness,
        "correctness_gate": worker.correctness_gate(eager_correctness),
        "output_contract": worker.output_contract(eager_output, reference.cpu()),
        "workspace_gate": eager_workspace["verification"],
        "descriptor_gate": eager_descriptor_gate,
        "event_gate": eager_event_gate,
    }
    eager_payload["runtime_case_gate"] = _runtime_case_gate(
        worker, arm.wrapper._dynamic_workspace, eager_descriptor_gate
    )
    eager_payload["failed_gates"] = _gate_failures(eager_payload)
    torch.save(eager_timing, args.output / "eager_timing.pt")
    write_json(args.output / "eager.json", eager_payload)
    if eager_payload["failed_gates"]:
        write_json(
            args.output / "capture_failure.json",
            {
                "schema": "exp006.capture-failure.v1",
                "stage": "eager",
                "failed_gates": eager_payload["failed_gates"],
                "overlay_gate": overlay_gate,
                "runtime": runtime,
            },
        )
        raise RuntimeError(f"eager gates failed: {eager_payload['failed_gates']}")

    arm.capture()
    for _ in range(args.warmup):
        _reset(arm)
        arm.replay(output_sentinel=False, reset_probe=False)

    runs: list[dict[str, Any]] = []
    for replay in range(args.replays):
        _reset(arm)
        output, elapsed_ms = arm.replay(output_sentinel=True, reset_probe=False)
        output_cpu = output.detach().cpu().clone()
        workspace_tensors, workspace = worker.workspace_snapshot(arm.wrapper, fixture)
        timing, descriptors, descriptor_hash = _snapshot(arm, workspace_tensors)
        task_tail = int(arm.wrapper._dynamic_workspace.task_tail.item())
        descriptor_gate = validate_descriptors(descriptors, task_tail=task_tail)
        if descriptor_hash != eager_descriptor_hash:
            descriptor_gate = dict(descriptor_gate)
            descriptor_gate["errors"] = list(descriptor_gate.get("errors", [])) + [
                "descriptor order differs from eager locked fixture"
            ]
            descriptor_gate["gate_pass"] = False
        event_gate = _event_gate(
            timing,
            arm_name=args.arm,
            task_tail=task_tail,
            task_capacity=int(arm.wrapper._dynamic_workspace.task_capacity),
            grid_z=int(timing["cta_ticks"].shape[0]),
        )
        correctness = worker.tensor_error(output_cpu, reference.cpu())
        run: dict[str, Any] = {
            "run_id": f"run_{replay}",
            "event_elapsed_us": elapsed_ms * 1000.0,
            "output_sha256": worker.tensor_sha256(output_cpu),
            "correctness": correctness,
            "correctness_gate": worker.correctness_gate(correctness),
            "output_contract": worker.output_contract(output, reference),
            "workspace_gate": workspace["verification"],
            "descriptor_gate": descriptor_gate,
            "event_gate": event_gate,
            "runtime_case_gate": _runtime_case_gate(
                worker, arm.wrapper._dynamic_workspace, descriptor_gate
            ),
            "descriptor_order_sha256": descriptor_hash,
            "timing_sha256": _tensor_hashes(worker, timing),
        }
        run["failed_gates"] = _gate_failures(run)
        torch.save(timing, args.output / f"timing_{replay}.pt")
        write_json(args.output / f"run_{replay}.json", run)
        runs.append(run)
        if run["failed_gates"]:
            write_json(
                args.output / "capture_failure.json",
                {
                    "schema": "exp006.capture-failure.v1",
                    "stage": f"run_{replay}",
                    "failed_gates": run["failed_gates"],
                    "overlay_gate": overlay_gate,
                    "runtime": runtime,
                    "jit_artifacts": artifact_manifest(args.jit_root),
                },
            )
            raise RuntimeError(f"run_{replay} gates failed: {run['failed_gates']}")

    artifacts = artifact_manifest(args.jit_root)
    jit_gate = _jit_identity_gate(artifacts)
    if not jit_gate["gate_pass"]:
        write_json(
            args.output / "capture_failure.json",
            {
                "schema": "exp006.capture-failure.v1",
                "stage": "jit_identity",
                "failed_gates": ["jit_identity_gate"],
                "jit_identity_gate": jit_gate,
                "jit_artifacts": artifacts,
                "overlay_gate": overlay_gate,
                "runtime": runtime,
                "runs": runs,
            },
        )
        raise RuntimeError(f"JIT identity gate failed: {jit_gate['errors']}")
    latencies = [float(run["event_elapsed_us"]) for run in runs]
    payload: dict[str, Any] = {
        "schema": "exp006.completion-capture.v1",
        "classification": "diagnostic-only" if args.arm == PROBE else "control",
        "arm": args.arm,
        "event_abi": EVENT_ABI,
        "source": source,
        "overlay_gate": overlay_gate,
        "runtime": runtime,
        "fixture": fixture.manifest,
        "weights": weights.manifest,
        "reference_sha256": worker.tensor_sha256(reference),
        "eager": eager_payload,
        "runs": runs,
        "latency_us": {
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "samples": len(latencies),
        },
        "descriptor_order_sha256": eager_descriptor_hash,
        "jit_identity_gate": jit_gate,
        "jit_artifacts": artifacts,
        "jit_artifact_set_sha256": canonical_sha256(artifacts),
        "foreign_processes_after": worker.require_no_foreign_process(runtime),
    }
    write_json(args.output / "capture.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--kernel-overlay", type=Path, required=True)
    parser.add_argument("--dispatch-overlay", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--replays", type=int, default=MEASURED_REPLAYS)
    args = parser.parse_args()
    payload = capture(args)
    print(json.dumps({"arm": args.arm, "latency_us": payload["latency_us"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
