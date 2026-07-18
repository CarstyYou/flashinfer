#!/usr/bin/env python3
"""Capture diagnostic-only exp_004 phase shares or a no-marker latency control."""

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
    NORMAL,
    PROBE,
    artifact_manifest,
    canonical_sha256,
    iter_probe_rows,
    require_empty_directory,
    summarize_phase_rows,
    write_json,
)


def _aggregate_phase_summaries(summaries: list[Mapping[str, Any]]) -> dict[str, Any]:
    denominator = sum(int(item["consumer"]["denominator_ticks"]) for item in summaries)
    totals: dict[str, int] = {}
    per_run: dict[str, list[float]] = {}
    for item in summaries:
        for phase in item["consumer"]["phases"]:
            name = str(phase["phase"])
            totals[name] = totals.get(name, 0) + int(phase["sum_ticks"])
            per_run.setdefault(name, []).append(float(phase["share_pct"]))
    phases = []
    for name, ticks in totals.items():
        values = per_run[name]
        phases.append(
            {
                "phase": name,
                "sum_ticks": ticks,
                "denominator_ticks": denominator,
                "share_pct": 100.0 * ticks / denominator,
                "run_min_pct": min(values),
                "run_max_pct": max(values),
            }
        )
    return {
        "denominator_ticks": denominator,
        "phases": phases,
        "replays": len(summaries),
    }


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
    eager_probe: dict[str, Any] | None = None
    eager_phase_summary: Mapping[str, Any] | None = None
    eager_timing_payload: dict[str, torch.Tensor] | None = None
    pointer_identity: dict[str, Any] = {}
    if args.arm == PROBE:
        eager_workspace_tensors, eager_workspace = worker.workspace_snapshot(
            arm.wrapper, fixture
        )
        eager_timing_payload, eager_event_gate = worker.timing_snapshot(
            arm,
            PROBE,
            eager_workspace_tensors,
            "eager",
        )
        workspace = arm.wrapper._dynamic_workspace
        pointer_identity["after_eager"] = {
            "timing_ticks": int(workspace.exp004_timing_ticks.data_ptr()),
            "task_cta_z": int(workspace.exp004_task_cta_z.data_ptr()),
        }
        eager_probe = {
            "event_gate": eager_event_gate,
            "workspace_gate": eager_workspace["verification"],
            "timing_sha256": worker.tensor_sha256(eager_timing_payload["timing_ticks"]),
            "task_cta_sha256": worker.tensor_sha256(eager_timing_payload["task_cta_z"]),
        }
        if eager_event_gate["gate_pass"]:
            eager_rows = list(
                iter_probe_rows(
                    eager_timing_payload["timing_ticks"].tolist(),
                    eager_timing_payload["task_cta_z"].tolist(),
                    run_id="eager",
                    task_tail=int(workspace.task_tail.item()),
                    task_capacity=int(workspace.task_capacity),
                    task_descriptors=worker.task_descriptors(eager_workspace_tensors),
                )
            )
            eager_phase_summary = summarize_phase_rows(eager_rows)
            eager_probe["phase_summary"] = eager_phase_summary
        torch.save(eager_timing_payload, args.output / "eager_timing.pt")
        write_json(args.output / "eager_probe.json", eager_probe)
    arm.capture()
    if args.arm == PROBE:
        workspace = arm.wrapper._dynamic_workspace
        pointer_identity["after_graph_capture"] = {
            "timing_ticks": int(workspace.exp004_timing_ticks.data_ptr()),
            "task_cta_z": int(workspace.exp004_task_cta_z.data_ptr()),
        }
    for _ in range(args.warmup):
        arm.replay(output_sentinel=False, reset_probe=args.arm == PROBE)

    runs = []
    phase_summaries: list[Mapping[str, Any]] = []
    for replay in range(args.replays):
        output, elapsed_ms = arm.replay(
            output_sentinel=True, reset_probe=args.arm == PROBE
        )
        output_cpu = output.detach().cpu().clone()
        workspace_tensors, workspace = worker.workspace_snapshot(arm.wrapper, fixture)
        correctness = worker.tensor_error(output_cpu, reference.cpu())
        run: dict[str, Any] = {
            "run_id": f"run_{replay}",
            "event_elapsed_us": elapsed_ms * 1000.0,
            "output_sha256": worker.tensor_sha256(output_cpu),
            "correctness": correctness,
            "correctness_gate": worker.correctness_gate(correctness),
            "output_contract": worker.output_contract(output, reference),
            "workspace_gate": workspace["verification"],
        }
        if args.arm == PROBE:
            workspace_device = arm.wrapper._dynamic_workspace
            run["pointer_identity"] = {
                "timing_ticks": int(workspace_device.exp004_timing_ticks.data_ptr()),
                "task_cta_z": int(workspace_device.exp004_task_cta_z.data_ptr()),
            }
            timing, event_gate = worker.timing_snapshot(
                arm,
                PROBE,
                workspace_tensors,
                run["run_id"],
            )
            run["event_gate"] = event_gate
            torch.save(timing, args.output / f"timing_{replay}.pt")
            run["timing_sha256"] = worker.tensor_sha256(timing["timing_ticks"])
            run["task_cta_sha256"] = worker.tensor_sha256(timing["task_cta_z"])
            if event_gate["gate_pass"]:
                rows = list(
                    iter_probe_rows(
                        timing["timing_ticks"].tolist(),
                        timing["task_cta_z"].tolist(),
                        run_id=run["run_id"],
                        task_tail=int(arm.wrapper._dynamic_workspace.task_tail.item()),
                        task_capacity=int(arm.wrapper._dynamic_workspace.task_capacity),
                        task_descriptors=worker.task_descriptors(workspace_tensors),
                    )
                )
                summary = summarize_phase_rows(rows)
                phase_summaries.append(summary)
                run["phase_summary"] = summary
        write_json(args.output / f"run_{replay}.json", run)
        runs.append(run)

    latencies = [float(run["event_elapsed_us"]) for run in runs]
    payload: dict[str, Any] = {
        "schema": "exp004.diagnostic-phase-capture.v1",
        "classification": "diagnostic-only" if args.arm == PROBE else "control",
        "arm": args.arm,
        "source": source,
        "runtime": runtime,
        "eager_correctness": eager_correctness,
        "eager_probe": eager_probe,
        "pointer_identity": pointer_identity,
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
    if args.arm == PROBE:
        payload["phase_event_complete_replays"] = len(phase_summaries)
        if phase_summaries:
            payload["phase_summary"] = _aggregate_phase_summaries(phase_summaries)
            payload["phase_summary_source"] = "cuda_graph_replays"
        elif eager_phase_summary is not None:
            payload["phase_summary"] = _aggregate_phase_summaries([eager_phase_summary])
            payload["phase_summary_source"] = "eager_diagnostic_fallback"
    write_json(args.output / "capture.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--arm", choices=(NORMAL, PROBE), required=True)
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
