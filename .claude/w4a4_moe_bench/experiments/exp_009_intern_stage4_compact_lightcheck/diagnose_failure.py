#!/usr/bin/env python3
"""Diagnose a failed Intern correctness prepare with three unprofiled replays."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from run_exp009_arm import ARM_NAME, load_worker


def gate_classification(replay: dict[str, Any]) -> dict[str, bool]:
    formal = replay.get("formal_metrics", {})
    sentinel = replay.get("sentinel", {})
    workspace = replay.get("workspace", {})
    formal_pass = bool(formal.get("formal_pass")) and bool(formal.get("finite"))
    sentinel_pass = (
        int(sentinel.get("nan_remaining", -1)) == 0
        and int(sentinel.get("inf_count", -1)) == 0
        and bool(formal.get("finite"))
    )
    workspace_pass = workspace.get("gate_pass") is True
    return {
        "formal_pass": formal_pass,
        "sentinel_pass": sentinel_pass,
        "workspace_pass": workspace_pass,
        "overall_pass": formal_pass and sentinel_pass and workspace_pass,
    }


def summarize_cross_replay(
    replays: Sequence[dict[str, Any]], pairwise: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if not replays:
        raise ValueError("at least one replay is required")
    output_hashes = [str(replay["output_sha256"]) for replay in replays]
    workspace_hashes = [
        str(replay["workspace"]["task_descriptor_multiset_sha256"])
        for replay in replays
    ]
    metric_names = ("cosine_loss", "relative_l2", "max_abs", "token_rel_l2_p99")
    worst = None
    if pairwise:
        worst = {
            name: max(float(row["metrics"][name]) for row in pairwise)
            for name in metric_names
        }
    return {
        "replay_count": len(replays),
        "all_replays_overall_pass": all(
            replay["gates"]["overall_pass"] for replay in replays
        ),
        "exact_output_hash_stable": len(set(output_hashes)) == 1,
        "output_hashes": output_hashes,
        "unique_output_hashes": sorted(set(output_hashes)),
        "zero_row_counts": [int(replay["zero_rows"]) for replay in replays],
        "workspace_descriptor_multiset_stable": len(set(workspace_hashes)) == 1,
        "workspace_descriptor_multiset_hashes": workspace_hashes,
        "all_replays_formal_pass": all(
            replay["gates"]["formal_pass"] for replay in replays
        ),
        "all_replays_sentinel_pass": all(
            replay["gates"]["sentinel_pass"] for replay in replays
        ),
        "all_replays_workspace_pass": all(
            replay["gates"]["workspace_pass"] for replay in replays
        ),
        "worst_pairwise_output_drift": worst,
        "pairwise_output_drift": list(pairwise),
    }


def _validate_args(args, worker) -> None:
    if not isinstance(args.m, int) or isinstance(args.m, bool) or args.m <= 0:
        raise ValueError("M must be a positive integer")
    if args.arm not in worker.KNOWN_ARMS:
        raise ValueError(f"unknown arm: {args.arm}")
    if args.fixture not in worker.ALL_FIXTURES:
        raise ValueError(f"unknown fixture: {args.fixture}")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite immutable diagnostic: {args.output}")


def observed_classification(replays: Sequence[dict[str, Any]]) -> str:
    if any(not replay["gates"]["sentinel_pass"] for replay in replays):
        return "sentinel_or_nonfinite_output"
    if any(not replay["gates"]["formal_pass"] for replay in replays):
        return "numerical_accuracy_failure"
    if any(not replay["gates"]["workspace_pass"] for replay in replays):
        return "workspace_route_task_failure"
    return "not_reproduced_in_diagnostic_replays"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--arm", default=ARM_NAME)
    parser.add_argument("--fixture", default="canonical")
    parser.add_argument("--device-index", type=int, default=0, choices=[0])
    parser.add_argument("--seed", type=int, default=2026, choices=[2026])
    args = parser.parse_args()
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.overlay = args.overlay.resolve()
    args.jit_root = args.jit_root.resolve()
    args.failure = args.failure.resolve()
    args.output = args.output.resolve()
    args.results = args.failure.parents[5]

    worker = load_worker()
    _validate_args(args, worker)
    failure = json.loads(args.failure.read_text())
    if (
        failure.get("status") != "failed"
        or failure.get("arm") != args.arm
        or failure.get("m") != args.m
    ):
        raise RuntimeError("failure evidence identity drift")

    payload = {
        "schema": "exp009.intern-failure-diagnostic.v1",
        "status": "running",
        "arm": args.arm,
        "m": args.m,
        "fixture_kind": args.fixture,
        "original_failure_classification": failure.get("classification"),
        "failure_evidence_sha256": worker.file_sha256(args.failure),
        "replays": [],
    }
    worker.write_json(args.output, payload)
    try:
        source = worker.validate_source(args.flashinfer_root, args.overlay, args.arm)
        if str(args.flashinfer_root) not in sys.path:
            sys.path.insert(0, str(args.flashinfer_root))
        worker.install_overlay(args.overlay)
        imports = worker.configure_source_checkout(args.flashinfer_root)
        if Path(imports["target_module"]) != args.overlay:
            raise RuntimeError("target module did not resolve to Intern overlay")
        runtime = worker.runtime_identity(args, source)
        runtime["imports"] = imports
        payload["runtime"] = runtime
        worker.write_json(args.output, payload)

        fixture_module, fixture, weights = worker.make_case(args)
        reference = fixture_module.reference_moe_nvfp4(fixture, weights)
        arm = worker.build_arm(args, fixture, weights)
        arm.eager()
        arm.capture()
        output_tensors = []
        replays = []
        for replay_index in range(3):
            output, elapsed_ms = arm.replay(sentinel=True)
            finite = bool(worker.torch.isfinite(output).all().item())
            nan_remaining = int(worker.torch.isnan(output).sum().item())
            inf_count = int(worker.torch.isinf(output).sum().item())
            try:
                formal = fixture_module.output_diagnostics(output, reference)
            except Exception as error:
                formal = {
                    "formal_pass": False,
                    "capture_error": f"{type(error).__name__}: {error}",
                }
            formal["finite"] = finite
            try:
                _, workspace_summary = worker._workspace_snapshot(
                    arm.wrapper,
                    fixture,
                    num_cta_warps=worker.expected_block(args.arm)[0] // 32,
                )
                task_hashes = {
                    key: value
                    for key, value in workspace_summary["tensor_sha256"].items()
                    if key.startswith("task_")
                }
                workspace = {
                    "gate_pass": workspace_summary["verification"]["gate_pass"],
                    "checks": workspace_summary["verification"].get("checks", {}),
                    "verification": workspace_summary["verification"],
                    "task_descriptor_multiset_sha256": worker.canonical_sha256(
                        task_hashes
                    ),
                    "tensor_sha256": workspace_summary["tensor_sha256"],
                }
            except Exception as error:
                capture_error = f"{type(error).__name__}: {error}"
                workspace = {
                    "gate_pass": False,
                    "checks": {},
                    "capture_error": capture_error,
                    "task_descriptor_multiset_sha256": worker.canonical_sha256(
                        {"capture_error": capture_error}
                    ),
                }
            # Retain only CPU copies across replays: M8192 outputs are large.
            output_cpu = output.detach().cpu().clone()
            row_abs_sum = output_cpu.float().abs().sum(dim=1)
            replay = {
                "replay": replay_index,
                "formal_metrics": formal,
                "sentinel": {
                    "nan_remaining": nan_remaining,
                    "inf_count": inf_count,
                },
                "workspace": workspace,
                "output_sha256": worker.tensor_sha256(output_cpu),
                "zero_rows": int((row_abs_sum == 0).sum().item()),
                "zero_elements": int((output_cpu == 0).sum().item()),
                "event_elapsed_us_diagnostic_only": elapsed_ms * 1000.0,
            }
            replay["gates"] = gate_classification(replay)
            replays.append(replay)
            output_tensors.append(output_cpu)
            payload["replays"] = replays
            worker.write_json(args.output, payload)

        pairwise = [
            {
                "first_replay": first,
                "second_replay": second,
                "metrics": worker.tensor_error(
                    output_tensors[second], output_tensors[first]
                ),
            }
            for first, second in combinations(range(len(output_tensors)), 2)
        ]
        artifacts = worker.artifact_manifest(args.jit_root)
        cubins = sorted(
            item["sha256"] for item in artifacts if item["path"].endswith(".cubin")
        )
        if not cubins:
            raise RuntimeError("failed prepare JIT contains no cubin")
        payload.update(
            {
                "status": "complete",
                "replay_count": len(replays),
                "observed_classification": observed_classification(replays),
                "cross_replay_stability": summarize_cross_replay(replays, pairwise),
                "cubin_sha256": cubins,
                "jit_artifact_set_sha256": worker.canonical_sha256(artifacts),
            }
        )
        worker.write_json(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 0
    except Exception as error:
        payload["status"] = "failed"
        payload["diagnostic_error"] = f"{type(error).__name__}: {error}"
        worker.write_json(args.output, payload)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
