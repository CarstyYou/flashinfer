#!/usr/bin/env python3
"""Capture the matched control or whole-kernel diagnostic exp_004 probe."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping

import torch

import run_exp004_arm as worker
from exp004_common import (
    MEASURED_REPLAYS,
    MEASUREMENT_CONTROL,
    PROBE,
    SENTINEL,
    artifact_manifest,
    canonical_sha256,
    require_empty_directory,
    write_json,
)


TASK_TICKS = 65
CTA_TICKS = 14
OUTPUT_TILES = 16


def _snapshot(arm: worker.CapturedArm) -> dict[str, torch.Tensor]:
    workspace = arm.wrapper._dynamic_workspace
    task_capacity = int(workspace.task_capacity)
    return {
        "task_ticks": workspace.exp004_timing_ticks.detach()
        .reshape(task_capacity, TASK_TICKS)
        .cpu()
        .clone(),
        "task_cta_z": workspace.exp004_task_cta_z.detach().cpu().clone(),
        "cta_ticks": workspace.exp004_cta_ticks.detach()
        .reshape(-1, CTA_TICKS)
        .cpu()
        .clone(),
    }


def _reset(arm: worker.CapturedArm) -> None:
    workspace = arm.wrapper._dynamic_workspace
    workspace.exp004_timing_ticks.fill_(SENTINEL)
    workspace.exp004_task_cta_z.fill_(SENTINEL)
    workspace.exp004_cta_ticks.fill_(SENTINEL)


def _probe_gate(
    timing: Mapping[str, torch.Tensor],
    *,
    task_tail: int,
    task_capacity: int,
    grid_z: int,
) -> dict[str, Any]:
    task_ticks = timing["task_ticks"].reshape(task_capacity, TASK_TICKS)
    task_cta = timing["task_cta_z"]
    cta_ticks = timing["cta_ticks"].reshape(grid_z, CTA_TICKS)
    errors: list[str] = []
    expected_task_writes = task_tail * TASK_TICKS
    actual_task_writes = int((task_ticks != SENTINEL).sum().item())
    expected_cta_writes = grid_z * CTA_TICKS
    actual_cta_writes = int((cta_ticks != SENTINEL).sum().item())
    if actual_task_writes != expected_task_writes:
        errors.append(
            f"task tick writes {actual_task_writes} != {expected_task_writes}"
        )
    if bool((task_ticks[:task_tail] == SENTINEL).any().item()):
        errors.append("missing task event inside task_tail")
    if bool((task_ticks[task_tail:] != SENTINEL).any().item()):
        errors.append("task event written beyond task_tail")
    if bool((task_cta[:task_tail] < 0).any().item()) or bool(
        (task_cta[:task_tail] >= grid_z).any().item()
    ):
        errors.append("invalid task-to-CTA mapping")
    if bool((task_cta[task_tail:] != SENTINEL).any().item()):
        errors.append("task-to-CTA mapping written beyond task_tail")
    if actual_cta_writes != expected_cta_writes:
        errors.append(f"CTA tick writes {actual_cta_writes} != {expected_cta_writes}")

    if not errors:
        for task in range(task_tail):
            row = [int(v) for v in task_ticks[task].tolist()]
            consumer_order = row[:6] + row[7:55] + [row[6]]
            if any(
                right < left
                for left, right in zip(consumer_order, consumer_order[1:], strict=False)
            ):
                errors.append(f"task {task} consumer timeline is non-monotonic")
                break
            if any(
                right < left
                for left, right in zip(row[55:64], row[56:65], strict=False)
            ):
                errors.append(f"task {task} W4 timeline is non-monotonic")
                break
        for cta, row_tensor in enumerate(cta_ticks):
            row = [int(v) for v in row_tensor.tolist()]
            if any(
                right < left for left, right in zip(row[:7], row[1:8], strict=False)
            ):
                errors.append(f"CTA {cta} launch timeline is non-monotonic")
                break
            if any(value < row[7] for value in row[8:]):
                errors.append(f"CTA {cta} exit precedes compute-loop start")
                break
            if row[13] < row[12]:
                errors.append(f"CTA {cta} W4 final precedes producer-tail start")
                break

    return {
        "schema": "exp004.whole-kernel-event-gate.v1",
        "task_tail": task_tail,
        "task_capacity": task_capacity,
        "grid_z": grid_z,
        "expected_task_writes": expected_task_writes,
        "actual_task_writes": actual_task_writes,
        "expected_cta_writes": expected_cta_writes,
        "actual_cta_writes": actual_cta_writes,
        "mapped_tasks": int((task_cta != SENTINEL).sum().item()),
        "errors": errors[:32],
        "gate_pass": not errors,
    }


def _control_gate(
    timing: Mapping[str, torch.Tensor],
    *,
    task_tail: int,
    task_capacity: int,
    grid_z: int,
) -> dict[str, Any]:
    del task_tail, task_capacity, grid_z
    writes = {
        name: int((tensor != SENTINEL).sum().item()) for name, tensor in timing.items()
    }
    return {
        "schema": "exp004.whole-kernel-control-gate.v1",
        "writes": writes,
        "gate_pass": all(value == 0 for value in writes.values()),
    }


def _capture_timing(
    arm: worker.CapturedArm, arm_name: str
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    timing = _snapshot(arm)
    workspace = arm.wrapper._dynamic_workspace
    task_capacity = int(workspace.task_capacity)
    task_tail = int(workspace.task_tail.item())
    grid_z = int(timing["cta_ticks"].numel() // CTA_TICKS)
    if timing["task_ticks"].numel() != task_capacity * TASK_TICKS:
        raise RuntimeError("whole-kernel task timing capacity drift")
    if timing["task_cta_z"].numel() != task_capacity:
        raise RuntimeError("whole-kernel task-to-CTA capacity drift")
    if arm_name == PROBE:
        gate = _probe_gate(
            timing,
            task_tail=task_tail,
            task_capacity=task_capacity,
            grid_z=grid_z,
        )
    else:
        gate = _control_gate(
            timing,
            task_tail=task_tail,
            task_capacity=task_capacity,
            grid_z=grid_z,
        )
    return timing, gate


def capture(args: argparse.Namespace) -> dict[str, Any]:
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.kernel_overlay = args.kernel_overlay.resolve()
    args.dispatch_overlay = args.dispatch_overlay.resolve()
    args.jit_root = args.jit_root.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"immutable capture output exists: {args.output}")
    require_empty_directory(args.jit_root)

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
    eager_correctness = worker.tensor_error(eager_output, reference.cpu())
    eager_timing, eager_gate = _capture_timing(arm, args.arm)
    torch.save(eager_timing, args.output / "eager_timing.pt")
    eager_payload = {
        "correctness": eager_correctness,
        "correctness_gate": worker.correctness_gate(eager_correctness),
        "event_gate": eager_gate,
    }
    write_json(args.output / "eager.json", eager_payload)

    arm.capture()
    for _ in range(args.warmup):
        _reset(arm)
        arm.replay(output_sentinel=False, reset_probe=False)

    runs = []
    for replay in range(args.replays):
        _reset(arm)
        output, elapsed_ms = arm.replay(output_sentinel=True, reset_probe=False)
        output_cpu = output.detach().cpu().clone()
        workspace_tensors, workspace = worker.workspace_snapshot(arm.wrapper, fixture)
        del workspace_tensors
        timing, event_gate = _capture_timing(arm, args.arm)
        correctness = worker.tensor_error(output_cpu, reference.cpu())
        run: dict[str, Any] = {
            "run_id": f"run_{replay}",
            "event_elapsed_us": elapsed_ms * 1000.0,
            "output_sha256": worker.tensor_sha256(output_cpu),
            "correctness": correctness,
            "correctness_gate": worker.correctness_gate(correctness),
            "output_contract": worker.output_contract(output, reference),
            "workspace_gate": workspace["verification"],
            "event_gate": event_gate,
            "timing_sha256": {
                name: worker.tensor_sha256(tensor) for name, tensor in timing.items()
            },
        }
        torch.save(timing, args.output / f"timing_{replay}.pt")
        write_json(args.output / f"run_{replay}.json", run)
        runs.append(run)

    latencies = [float(run["event_elapsed_us"]) for run in runs]
    payload: dict[str, Any] = {
        "schema": "exp004.whole-kernel-capture.v1",
        "classification": "diagnostic-only" if args.arm == PROBE else "control",
        "arm": args.arm,
        "source": source,
        "runtime": runtime,
        "eager": eager_payload,
        "runs": runs,
        "latency_us": {
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "samples": len(latencies),
        },
        "jit_artifacts": artifact_manifest(args.jit_root),
        "foreign_processes_after": worker.require_no_foreign_process(runtime),
    }
    payload["jit_artifact_set_sha256"] = canonical_sha256(payload["jit_artifacts"])
    write_json(args.output / "capture.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--arm", choices=(MEASUREMENT_CONTROL, PROBE), required=True)
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
