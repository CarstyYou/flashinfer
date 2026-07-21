#!/usr/bin/env python3
"""Capture five warmed latest-opt control/probe graph replays for exp_017."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

from build_opt_phase_overlays import overlay_paths, verify_existing
from exp017_opt_phase_common import (
    DISPATCH_MODULE,
    DISPATCH_RELATIVE_PATH,
    EVENT_ABI,
    EVENTS_PER_CTA,
    EXPECTED_DISPATCH_SHA256,
    EXPECTED_OPT_SHA256,
    EXPECTED_WRAPPER_SHA256,
    GRID_CTAS,
    MODES,
    OPT_RELATIVE_PATH,
    OVERLAY_ROOT,
    PROBE,
    WRAPPER_RELATIVE_PATH,
    canonical_sha256,
    file_sha256,
    summarize_replays,
    validate_events,
    write_json,
)


ROOT = Path(__file__).resolve().parent
EXP001 = ROOT.parent / "exp_001_backend_case_sweep"
EXP016 = ROOT.parent / "exp_016_route_q0_token_major_reuse"
DEFAULT_FIXTURES = EXP001 / "results" / "fixtures"
EXPECTED_FIXTURE_SHA256 = (
    "c113ecd5ddeff77154ddbd23fc3dc3c83f8ee822e880179ca5c16b1145372438"
)
EXPECTED_OCCUPANCY_SHA256 = (
    "3e4350788bfcdd1cca175141f0c6626934589b97cb23d74811e4bc2785531a94"
)
M = 8192
WARMUPS = 2
REPLAYS = 5
L2_FLUSH_BYTES = 192 << 20
ARM = "candidate_token_major_reuse"

RESOURCE_RE = re.compile(
    r"Function\s+(\S+):\s*\n\s*"
    r"REG\s*:\s*(\d+)\s+STACK\s*:\s*(\d+)\s+"
    r"SHARED\s*:\s*(\d+)\s+LOCAL\s*:\s*(\d+)",
    re.I,
)


def load_reused():
    if str(EXP016) not in sys.path:
        sys.path.insert(0, str(EXP016))
    import capture_p3_phase as overlay_runtime
    import run_exp016_arm as core

    return overlay_runtime, core


def load_exp001_fixture_module() -> Any:
    path = EXP001 / "fixture.py"
    spec = importlib.util.spec_from_file_location("exp017_exp001_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load persisted fixture helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_exp001_case(args: argparse.Namespace, core: Any) -> tuple[Any, Any, Any]:
    """Load the exact persisted exp_001 routing and canonical NVFP4 weights."""
    fixture_module = core.worker.load_fixture_module()
    persisted = load_exp001_fixture_module()
    device = core.torch.device("cuda", args.device_index)
    x, topk_ids, topk_weights, manifest = persisted.load_fixture(
        args.fixture_dir, args.m, device
    )
    identity = {
        "fixture_sha256": manifest.get("fixture_sha256"),
        "occupancy_sha256": manifest.get("occupancy_sha256"),
    }
    expected = {
        "fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "occupancy_sha256": EXPECTED_OCCUPANCY_SHA256,
    }
    if identity != expected:
        raise RuntimeError(f"exp_001 M8192 fixture identity drift: {identity}")
    fixture = fixture_module.RoutedFixture(
        args.m, x, topk_ids, topk_weights, manifest
    )
    weights = fixture_module.make_canonical_weights(device=device, seed=args.seed)
    return fixture_module, fixture, weights


def reset_events(workspace: Any) -> None:
    if not hasattr(workspace, "exp017_phase_events"):
        raise RuntimeError("instrumented workspace has no exp_017 event buffer")
    workspace.exp017_phase_events.zero_()


def partition_visible_processes(
    processes: Sequence[Mapping[str, Any]], *, own_pid: int
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Separate this capture's CUDA context from foreign GPU processes."""
    capture = [row for row in processes if str(row.get("pid")) == str(own_pid)]
    foreign = [row for row in processes if str(row.get("pid")) != str(own_pid)]
    return capture, foreign


def snapshot_events(workspace: Any, *, mode: str) -> tuple[list[int], dict[str, Any]]:
    tensor = workspace.exp017_phase_events.detach().cpu().clone()
    if str(tensor.dtype) != "torch.int64":
        raise RuntimeError(f"event dtype drift: {tensor.dtype}")
    expected = GRID_CTAS * EVENTS_PER_CTA
    if tensor.numel() != expected:
        raise RuntimeError(f"event capacity drift: {tensor.numel()} != {expected}")
    raw = [int(value) for value in tensor.tolist()]
    return raw, validate_events(raw, mode=mode)


def correctness_gate(diagnostics: Mapping[str, Any], output: Any) -> dict[str, Any]:
    zero_rows = int((output == 0).all(dim=-1).sum().item())
    checks = {
        "formal_pass": bool(diagnostics.get("formal_pass")),
        "finite": bool(diagnostics.get("finite")),
        "no_full_zero_rows": zero_rows == 0,
    }
    return {
        "checks": checks,
        "full_zero_rows": zero_rows,
        "gate_pass": all(checks.values()),
    }


def specialization_gate(overlay: Path, *, mode: str) -> dict[str, Any]:
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
    source = overlay.read_text(encoding="utf-8")
    zero_publish = re.findall(
        r"^\s*full_tile_publish_enabled\s*=\s*Int32\(0\)\s*$",
        source,
        flags=re.MULTILINE,
    )
    all_publish = re.findall(
        r"^\s*full_tile_publish_enabled\s*=.*$", source, flags=re.MULTILINE
    )
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
        "deferred_publish_only": len(zero_publish) == 1 and len(all_publish) == 1,
    }
    result = {"records": records, "checks": checks, "gate_pass": all(checks.values())}
    if not result["gate_pass"]:
        raise RuntimeError(f"specialization gate failed: {result}")
    return result


def resource_usage(
    jit_root: Path, artifacts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    tool = shutil.which("cuobjdump")
    if not tool:
        raise RuntimeError("cuobjdump is required for resource identity")
    records = []
    raw_hashes = []
    for artifact in artifacts:
        raw_path = str(artifact.get("path", ""))
        if not raw_path.endswith(".cubin"):
            continue
        cubin = Path(raw_path)
        if not cubin.is_absolute():
            cubin = jit_root / cubin
        completed = subprocess.run(
            [tool, "--dump-resource-usage", str(cubin)],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"resource query failed: {completed.stderr}")
        raw_hashes.append(canonical_sha256(completed.stdout))
        for symbol, registers, stack, shared, local in RESOURCE_RE.findall(
            completed.stdout
        ):
            if "MoEDynamicKernel" not in symbol:
                continue
            records.append(
                {
                    "cubin_sha256": artifact["sha256"],
                    "kernel_symbol": symbol,
                    "registers_per_thread": int(registers),
                    "stack_bytes_per_thread": int(stack),
                    "static_shared_bytes_per_cta": int(shared),
                    "static_local_bytes_outside_stack": int(local),
                }
            )
    if len(records) != 1:
        raise RuntimeError(
            f"expected one dynamic-kernel resource record, found {len(records)}"
        )
    return {
        "tool": tool,
        "records": records,
        "raw_output_sha256": raw_hashes,
        "comparison_contract": "compare matched control/probe before interpreting phase rows",
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    if args.m != M or args.warmup != WARMUPS or args.replays != REPLAYS:
        raise RuntimeError("capture is locked to M8192, warmup=2, replays=5")
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.overlay_root = args.overlay_root.resolve()
    args.jit_root = args.jit_root.resolve()
    args.output = args.output.resolve()
    args.fixture_dir = args.fixture_dir.resolve()
    if args.output.exists():
        raise FileExistsError(f"immutable capture exists: {args.output}")
    verify_existing(args.flashinfer_root, args.overlay_root)
    _, kernel, dispatch = overlay_paths(args.overlay_root, args.mode)

    overlay_runtime, core = load_reused()
    overlay_runtime.install_overlays(kernel, dispatch)
    core.common.require_empty_directory(args.jit_root)
    imports = core.reused.configure_source_checkout(args.flashinfer_root, args.jit_root)
    if Path(imports["target_module"]).resolve() != kernel:
        raise RuntimeError("kernel module did not resolve to selected overlay")
    imported_dispatch = importlib.import_module(DISPATCH_MODULE)
    if Path(imported_dispatch.__file__).resolve() != dispatch:
        raise RuntimeError("dispatch module did not resolve to selected overlay")

    checkout_head = core.reused.git(args.flashinfer_root, "rev-parse", "HEAD")
    core.reused.git(
        args.flashinfer_root,
        "merge-base",
        "--is-ancestor",
        core.reused.EXPECTED_FLASHINFER_COMMIT,
        checkout_head,
    )
    cutlass_commit = core.reused.git(
        args.flashinfer_root / "3rdparty/cutlass", "rev-parse", "HEAD"
    )
    if cutlass_commit != core.reused.EXPECTED_CUTLASS_COMMIT:
        raise RuntimeError(
            f"CUTLASS commit drift: {cutlass_commit} != "
            f"{core.reused.EXPECTED_CUTLASS_COMMIT}"
        )

    source = {
        "mode": args.mode,
        "locked_flashinfer_ancestor": core.reused.EXPECTED_FLASHINFER_COMMIT,
        "checkout_head": checkout_head,
        "cutlass_commit": cutlass_commit,
        "base_kernel_path": str(args.flashinfer_root / OPT_RELATIVE_PATH),
        "base_kernel_sha256": EXPECTED_OPT_SHA256,
        "kernel_overlay": str(kernel),
        "kernel_overlay_sha256": file_sha256(kernel),
        "dispatch_overlay": str(dispatch),
        "dispatch_overlay_sha256": file_sha256(dispatch),
        "live_dispatch_sha256": file_sha256(
            args.flashinfer_root / DISPATCH_RELATIVE_PATH
        ),
        "live_wrapper_sha256": file_sha256(
            args.flashinfer_root / WRAPPER_RELATIVE_PATH
        ),
        "expected_live_dispatch_sha256": EXPECTED_DISPATCH_SHA256,
        "expected_live_wrapper_sha256": EXPECTED_WRAPPER_SHA256,
        "event_abi": EVENT_ABI,
    }
    runtime = core.reused.runtime_identity(args, source)
    runtime["imports"] = imports
    runtime["harness"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": file_sha256(Path(__file__).resolve()),
        "reused_exp016": str(EXP016 / "run_exp016_arm.py"),
        "reused_exp016_sha256": file_sha256(EXP016 / "run_exp016_arm.py"),
    }

    # Reuse the canonical exp_001/016 fixture and opt dispatch plumbing.
    fixture_module, fixture, weights = make_exp001_case(args, core)
    oracle_weights = core.reference_weights_for_input_scale(weights, args.scale_kind)
    reference = fixture_module.reference_moe_nvfp4(fixture, oracle_weights)
    captured = core.worker.build_arm(args, fixture, weights)

    eager_output = captured.eager().detach().cpu().clone()
    workspace = captured.wrapper._dynamic_workspace
    if workspace is None:
        raise RuntimeError("dynamic workspace missing after eager launch")
    eager_events, eager_timing = snapshot_events(workspace, mode=args.mode)
    eager_diagnostics = fixture_module.output_diagnostics(
        eager_output, reference.detach().cpu()
    )
    _, route = core.corrected_workspace_snapshot(ARM, captured.wrapper, fixture)
    specialization = specialization_gate(kernel, mode=args.mode)
    eager = {
        "correctness": eager_diagnostics,
        "correctness_gate": correctness_gate(eager_diagnostics, eager_output),
        "route_task_gate": route["verification"],
        "phase_timing": eager_timing,
        "events_sha256": canonical_sha256(eager_events),
    }
    eager["gate_pass"] = bool(
        eager["correctness_gate"]["gate_pass"]
        and eager["route_task_gate"]["gate_pass"]
        and eager["phase_timing"]["gate_pass"]
        and specialization["gate_pass"]
    )
    if not eager["gate_pass"]:
        raise RuntimeError(f"eager gate failed: {eager}")

    captured.capture()
    flush, flush_bytes = core.worker.make_flusher(fixture.x.device, L2_FLUSH_BYTES)
    for _ in range(args.warmup):
        reset_events(workspace)
        flush()
        captured.replay(sentinel=False)

    runs = []
    for replay in range(args.replays):
        reset_events(workspace)
        flush()
        output, elapsed_ms = captured.replay(sentinel=True)
        output_cpu = output.detach().cpu().clone()
        events, timing = snapshot_events(workspace, mode=args.mode)
        diagnostics = fixture_module.output_diagnostics(
            output_cpu, reference.detach().cpu()
        )
        _, route = core.corrected_workspace_snapshot(ARM, captured.wrapper, fixture)
        run = {
            "replay": replay,
            "mode": args.mode,
            "event_elapsed_us": elapsed_ms * 1000.0,
            "output_sha256": core.tensor_sha256(output_cpu),
            "correctness": diagnostics,
            "correctness_gate": correctness_gate(diagnostics, output_cpu),
            "route_task_gate": route["verification"],
            "phase_timing": timing,
            "events": events,
        }
        run["gate_pass"] = bool(
            run["correctness_gate"]["gate_pass"]
            and run["route_task_gate"]["gate_pass"]
            and timing["gate_pass"]
        )
        if not run["gate_pass"]:
            raise RuntimeError(f"replay {replay} gate failed")
        runs.append(run)

    artifacts = core.common.artifact_manifest(args.jit_root)
    if not artifacts:
        raise RuntimeError("fresh JIT artifact set is empty")
    cubins = sorted(
        {
            str(item["sha256"])
            for item in artifacts
            if str(item.get("path", "")).endswith(".cubin")
        }
    )
    if len(cubins) != 1:
        raise RuntimeError(f"expected one retained cubin, found {cubins}")
    visible_processes_after = core.reused.foreign_processes(runtime["gpu"]["uuid"])
    # After this process creates its CUDA context, nvidia-smi reports it as a
    # compute process. Inside the container PID namespace that process is PID 1;
    # it is the capture itself, not a foreign tenant on the leased GPU.
    capture_process_after, foreign_after = partition_visible_processes(
        visible_processes_after, own_pid=os.getpid()
    )
    if foreign_after:
        raise RuntimeError(f"foreign process appeared during capture: {foreign_after}")

    payload = {
        "schema": "exp017.opt-phase-capture.v1",
        "classification": (
            "diagnostic matched probe"
            if args.mode == PROBE
            else "marker-disabled ABI control"
        ),
        "mode": args.mode,
        "case": {
            "m": args.m,
            "fixture": args.fixture,
            "fixture_dir": str(args.fixture_dir),
            "scale_kind": args.scale_kind,
        },
        "protocol": {
            "warmup_replays": args.warmup,
            "formal_replays": args.replays,
            "l2_flush_bytes": flush_bytes,
            "flush_order": "reset events -> L2 flush+sync -> graph replay",
            "timing": "external CUDA events plus in-kernel %globaltimer",
        },
        "source": source,
        "runtime": runtime,
        "fixture_identity": fixture.manifest,
        "weight_identity": weights.manifest,
        "reference_sha256": core.tensor_sha256(reference),
        "specialization": specialization,
        "eager": eager,
        "runs": runs,
        "summary": summarize_replays(runs),
        "jit_artifacts": artifacts,
        "jit_artifact_set_sha256": canonical_sha256(artifacts),
        "cubin_sha256": cubins,
        "static_resource_usage": resource_usage(args.jit_root, artifacts),
        "capture_process_after": capture_process_after,
        "foreign_processes_after": foreign_after,
        "evidence_boundary": (
            "Named phase rows are mutually exclusive CTA-leader intervals. W8 TMA "
            "overlap is not added; uncovered work, zero-task loops, and producer tail "
            "close through CTA residual and launch-skew rows."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--overlay-root", type=Path, default=OVERLAY_ROOT)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument(
        "--expected-app-clock-mhz", type=int, default=2377, choices=(2377,)
    )
    parser.add_argument("--device-index", type=int, default=0, choices=(0,))
    parser.add_argument("--m", type=int, default=M, choices=(M,))
    parser.add_argument("--fixture", default="canonical", choices=("canonical",))
    parser.add_argument("--scale-kind", default="equal", choices=("equal",))
    parser.add_argument("--seed", type=int, default=2026, choices=(2026,))
    parser.add_argument("--warmup", type=int, default=WARMUPS, choices=(WARMUPS,))
    parser.add_argument("--replays", type=int, default=REPLAYS, choices=(REPLAYS,))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = capture(parse_args(argv))
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "mode": payload["mode"],
                "summary": payload["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
