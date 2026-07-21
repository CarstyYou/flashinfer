#!/usr/bin/env python3
"""Capture one identity-locked exp_019 arm/mode/case (default: M8192)."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from build_phase_overlays import OVERLAY_ROOT, overlay_paths, verify_existing
from phase_common import (
    ARMS,
    BLOCK_THREADS,
    DISPATCH_MODULE,
    DISPATCH_RELATIVE_PATH,
    EVENT_ABI,
    EXPECTED_DISPATCH_SHA256,
    EXPECTED_FIXTURE_MANIFEST_SHA256,
    EXPECTED_FIXTURE_SHA256,
    EXPECTED_SOURCE_SHA256,
    EXPECTED_WRAPPER_SHA256,
    GRID_CTAS,
    L2_FLUSH_BYTES,
    M_VALUES,
    MODES,
    OCCURRENCE_ABI,
    PROBE,
    REPLAYS,
    SOURCE_RELATIVE_PATH,
    STORAGE_VALUES,
    WARMUPS,
    WRAPPER_RELATIVE_PATH,
    canonical_sha256,
    cyclic_process_order,
    expected_occurrences,
    file_sha256,
    occurrence_gate,
    summarize_replays,
    validate_phase_storage,
    write_json,
)


ROOT = Path(__file__).resolve().parent
EXP001 = ROOT.parent / "exp_001_backend_case_sweep"
EXP016 = ROOT.parent / "exp_016_route_q0_token_major_reuse"
EXP017 = ROOT.parent / "exp_017_opt_vs_triton_phase_share"
EXP018 = ROOT.parent / "exp_018_triton_opt_eric_benchmark"
DEFAULT_FIXTURES = EXP001 / "results" / "fixtures"


def load_runtime() -> tuple[Any, Any, Any, Any]:
    """Lazy imports keep all CPU contract tests CUDA-independent."""
    for path in (EXP017, EXP018):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import capture_opt_phase as exp017_capture
    import run_arm as exp018_runner

    overlay_runtime, exp016_core = exp017_capture.load_reused()
    return exp017_capture, exp018_runner, overlay_runtime, exp016_core


def reset_phase_storage(workspace: Any) -> None:
    tensor = getattr(workspace, "exp017_phase_events", None)
    if tensor is None:
        raise RuntimeError("instrumented workspace has no phase storage")
    if int(tensor.numel()) != STORAGE_VALUES:
        raise RuntimeError(f"phase storage capacity drift: {tensor.numel()}")
    tensor.zero_()


def snapshot_phase_storage(workspace: Any, *, mode: str) -> dict[str, Any]:
    tensor = workspace.exp017_phase_events.detach().cpu().clone()
    if str(tensor.dtype) != "torch.int64":
        raise RuntimeError(f"phase storage dtype drift: {tensor.dtype}")
    raw = [int(item) for item in tensor.tolist()]
    parsed = validate_phase_storage(raw, mode=mode)
    return {
        "storage_sha256": canonical_sha256(raw),
        "phase_timing": parsed["timing"],
        "occurrence_totals": parsed["occurrence_totals"],
        "storage_all_zero": parsed["storage_all_zero"],
    }


def descriptor_manifest(workspace: Any, route: Mapping[str, Any]) -> dict[str, Any]:
    verification = route["verification"]
    task_count = int(verification["expected_task_count"])
    observed_tail = int(workspace.task_tail.item())
    if observed_tail != task_count:
        raise RuntimeError(f"task tail drift: {observed_tail} != {task_count}")
    slice_count = int(workspace.task_slice_count[:task_count].sum().item())
    return {
        "task_count": task_count,
        "slice_count": slice_count,
        "task_descriptor_multiset_sha256": verification[
            "task_descriptor_multiset_sha256"
        ],
    }


def specialization_gate(kernel: Path, *, mode: str) -> dict[str, Any]:
    dispatch = importlib.import_module(DISPATCH_MODULE)
    keys = [key for key in dispatch._DYNAMIC_KERNEL_CACHE if key[0] == "dynamic"]
    records = [
        {
            "input_scales_are_reciprocal": bool(key[9]),
            "fast_math": bool(key[10]),
            "share_input_across_experts": bool(key[-2]),
            "phase_probe_enabled": bool(key[-1]),
        }
        for key in keys
    ]
    source = kernel.read_text(encoding="utf-8")
    checks = {
        "dynamic_cache_nonempty": bool(records),
        "locked_math_path": all(
            not row["input_scales_are_reciprocal"]
            and row["fast_math"]
            and not row["share_input_across_experts"]
            for row in records
        ),
        "marker_specialization": all(
            row["phase_probe_enabled"] == (mode == PROBE) for row in records
        ),
        "deferred_publish_only": source.count("full_tile_publish_enabled = Int32(0)")
        == 1,
    }
    return {"records": records, "checks": checks, "gate_pass": all(checks.values())}


def correctness_gate(
    runner: Any, output: Any, reference: Any, route: Mapping[str, Any], m: int
) -> dict[str, Any]:
    basic = runner.sanity(output, m, runner.fp4_worker.tensor_sha256)
    oracle = runner.nvfp4.output_diagnostics(output, reference)
    checks = {
        "basic": bool(basic["pass"]),
        "formal_oracle": bool(oracle["formal_pass"]),
        "route_task": bool(route["verification"]["gate_pass"]),
    }
    return {
        "basic": basic,
        "oracle": oracle,
        "checks": checks,
        "gate_pass": all(checks.values()),
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    if args.m not in M_VALUES or args.warmup != WARMUPS or args.replays != REPLAYS:
        raise RuntimeError("capture protocol drift")
    for name in (
        "flashinfer_root",
        "overlay_root",
        "jit_root",
        "fixture_dir",
        "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        raise FileExistsError(f"immutable capture exists: {args.output}")
    verify_existing(args.flashinfer_root, args.overlay_root)
    _, kernel, dispatch = overlay_paths(args.overlay_root, args.arm, args.mode)

    exp017_capture, runner, overlay_runtime, core = load_runtime()
    overlay_runtime.install_overlays(kernel, dispatch)
    core.common.require_empty_directory(args.jit_root)
    imports = core.reused.configure_source_checkout(args.flashinfer_root, args.jit_root)
    if Path(imports["target_module"]).resolve() != kernel:
        raise RuntimeError("kernel module did not resolve to the selected overlay")
    imported_dispatch = importlib.import_module(DISPATCH_MODULE)
    if Path(imported_dispatch.__file__).resolve() != dispatch:
        raise RuntimeError("dispatch module did not resolve to the selected overlay")

    source = {
        "arm": args.arm,
        "mode": args.mode,
        "base_path": str(args.flashinfer_root / SOURCE_RELATIVE_PATH[args.arm]),
        "base_sha256": EXPECTED_SOURCE_SHA256[args.arm],
        "kernel_overlay": str(kernel),
        "kernel_overlay_sha256": file_sha256(kernel),
        "dispatch_overlay": str(dispatch),
        "dispatch_overlay_sha256": file_sha256(dispatch),
        "live_dispatch_sha256": file_sha256(
            args.flashinfer_root / DISPATCH_RELATIVE_PATH
        ),
        "expected_live_dispatch_sha256": EXPECTED_DISPATCH_SHA256,
        "live_wrapper_sha256": file_sha256(
            args.flashinfer_root / WRAPPER_RELATIVE_PATH
        ),
        "expected_live_wrapper_sha256": EXPECTED_WRAPPER_SHA256,
        "event_abi": EVENT_ABI,
        "occurrence_abi": OCCURRENCE_ABI,
        "classification": "independent diagnostic cubin; not production-exact",
    }
    runtime = core.reused.runtime_identity(args, source)
    runtime["imports"] = imports

    fixture_identity = runner.fixture_identity(args.fixture_dir)
    if fixture_identity["manifest_sha256"] != EXPECTED_FIXTURE_MANIFEST_SHA256:
        raise RuntimeError("fixture manifest drift")
    if fixture_identity["npz_sha256"][str(args.m)] != EXPECTED_FIXTURE_SHA256[args.m]:
        raise RuntimeError(f"M{args.m} fixture drift")
    device = runner.torch.device("cuda", args.device_index)
    x, ids, routing, fixture_manifest = runner.persisted.load_fixture(
        args.fixture_dir, args.m, device
    )
    fixture = runner.nvfp4.RoutedFixture(args.m, x, ids, routing, fixture_manifest)
    weights = runner.nvfp4.make_canonical_weights(device=device, seed=runner.SEED)
    reference = runner.nvfp4.reference_moe_nvfp4(fixture, weights).detach()
    captured = runner.fp4_worker.build_arm(
        argparse.Namespace(m=args.m, device_index=args.device_index), fixture, weights
    )

    eager_output = captured.eager().detach()
    workspace = captured.wrapper._dynamic_workspace
    if workspace is None:
        raise RuntimeError("dynamic workspace missing after eager launch")
    eager_phase = snapshot_phase_storage(workspace, mode=args.mode)
    eager_route = runner.workspace_gate(args, captured, fixture)
    eager_correctness = correctness_gate(
        runner, eager_output, reference, eager_route, args.m
    )
    specialization = specialization_gate(kernel, mode=args.mode)
    manifest = descriptor_manifest(workspace, eager_route)
    eager_occurrence = (
        occurrence_gate(
            eager_phase["occurrence_totals"],
            expected_occurrences(
                args.arm,
                task_count=manifest["task_count"],
                slice_count=manifest["slice_count"],
            ),
        )
        if args.mode == PROBE
        else {"gate_pass": eager_phase["storage_all_zero"]}
    )
    if (
        not eager_correctness["gate_pass"]
        or not eager_occurrence["gate_pass"]
        or not specialization["gate_pass"]
    ):
        raise RuntimeError("eager correctness/occurrence gate failed")

    captured.capture()
    flush, flush_bytes = runner.fp4_worker.make_flusher(device, L2_FLUSH_BYTES)
    for _ in range(args.warmup):
        reset_phase_storage(workspace)
        flush()
        captured.replay(sentinel=False)

    runs = []
    for replay in range(args.replays):
        reset_phase_storage(workspace)
        flush()
        output, elapsed_ms = captured.replay(sentinel=True)
        output = output.detach()
        phase = snapshot_phase_storage(workspace, mode=args.mode)
        route = runner.workspace_gate(args, captured, fixture)
        correctness = correctness_gate(runner, output, reference, route, args.m)
        current_manifest = descriptor_manifest(workspace, route)
        if current_manifest != manifest:
            raise RuntimeError("task/slice manifest changed across graph replays")
        occurrence = (
            occurrence_gate(
                phase["occurrence_totals"],
                expected_occurrences(
                    args.arm,
                    task_count=manifest["task_count"],
                    slice_count=manifest["slice_count"],
                ),
            )
            if args.mode == PROBE
            else {"gate_pass": phase["storage_all_zero"]}
        )
        run = {
            "replay": replay,
            "mode": args.mode,
            "event_elapsed_us": elapsed_ms * 1000.0,
            "output_sha256": runner.fp4_worker.tensor_sha256(output),
            "correctness": correctness,
            "phase_timing": phase["phase_timing"],
            "occurrence_totals": phase["occurrence_totals"],
            "occurrence_gate": occurrence,
            "phase_storage_sha256": phase["storage_sha256"],
        }
        run["gate_pass"] = bool(correctness["gate_pass"] and occurrence["gate_pass"])
        if not run["gate_pass"]:
            raise RuntimeError(f"replay {replay} gate failed")
        runs.append(run)

    artifacts = core.common.artifact_manifest(args.jit_root)
    if not artifacts:
        raise RuntimeError("fresh JIT artifact set is empty")
    resources = exp017_capture.resource_usage(args.jit_root, artifacts)
    visible = core.reused.foreign_processes(runtime["gpu"]["uuid"])
    capture_process, foreign = exp017_capture.partition_visible_processes(
        visible, own_pid=os.getpid()
    )
    if foreign:
        raise RuntimeError(f"foreign process appeared during capture: {foreign}")

    payload = {
        "schema": "exp019.phase-capture.v1",
        "classification": "diagnostic matched probe/control; uninstrumented exp018 E2E is authority",
        "arm": args.arm,
        "mode": args.mode,
        "case": {"m": args.m, "E": 256, "H": 2048, "I_tp": 512, "topk": 8},
        "cyclic_block": args.cyclic_block,
        "cyclic_process_order": [
            list(item) for item in cyclic_process_order(args.cyclic_block)
        ],
        "protocol": {
            "warmup_replays": args.warmup,
            "formal_replays": args.replays,
            "l2_flush_bytes": flush_bytes,
            "flush_order": "reset phase storage -> L2 flush+sync -> graph replay",
            "timing": "external CUDA events plus exp017 %globaltimer ABI",
        },
        "source": source,
        "runtime": runtime,
        "fixture_identity": fixture_identity,
        "fixture_case_manifest": fixture_manifest,
        "weight_identity": runner.normalized_weight_identity(args.arm, weights),
        "reference_sha256": runner.fp4_worker.tensor_sha256(reference),
        "task_slice_manifest": manifest,
        "specialization": specialization,
        "eager": {
            "correctness": eager_correctness,
            "phase": eager_phase,
            "occurrence_gate": eager_occurrence,
        },
        "runs": runs,
        "summary": summarize_replays(runs),
        "jit_artifacts": artifacts,
        "jit_artifact_set_sha256": canonical_sha256(artifacts),
        "static_resource_usage": resources,
        "capture_process_after": capture_process,
        "foreign_processes_after": foreign,
        "launch_identity": {
            "grid": [1, 1, GRID_CTAS],
            "block": [BLOCK_THREADS[args.arm], 1, 1],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--overlay-root", type=Path, default=OVERLAY_ROOT)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--m", type=int, default=8192, choices=M_VALUES)
    parser.add_argument("--cyclic-block", type=int, default=0)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument(
        "--expected-app-clock-mhz", type=int, default=2377, choices=(2377,)
    )
    parser.add_argument("--device-index", type=int, default=0, choices=(0,))
    parser.add_argument("--warmup", type=int, default=WARMUPS, choices=(WARMUPS,))
    parser.add_argument("--replays", type=int, default=REPLAYS, choices=(REPLAYS,))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = capture(parse_args(argv))
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "arm": payload["arm"],
                "mode": payload["mode"],
                "summary": payload["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
