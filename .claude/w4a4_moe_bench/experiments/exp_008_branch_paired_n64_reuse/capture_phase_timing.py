#!/usr/bin/env python3
"""Capture one exp_008 marker-disabled control or marker-enabled probe.

GPU imports are lazy.  Each invocation consumes one immutable overlay root and
one empty JIT namespace; control and probe must therefore run in independent
processes and independent JIT roots.  No captured phase latency replaces the
unmodified Gate-B E2E result.
"""

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


ROOT = Path(__file__).resolve().parent
EXP005 = ROOT.parent / "exp_005_8warp_spill_reduction"
if str(EXP005) not in sys.path:
    sys.path.insert(0, str(EXP005))

from exp008_marker_common import (  # noqa: E402
    BASE_KERNEL,
    CONTROL,
    CTA_CALIBRATION,
    CTA_TICKS,
    DISPATCH_MODULE,
    DISPATCH_RELATIVE_PATH,
    EVENT_ABI,
    KERNEL_MODULE,
    MARKER_ARMS,
    OVERLAY_ROOT,
    PROBE,
    SENTINEL,
    TASK_TICKS,
    VERSIONS,
    WRAPPER_RELATIVE_PATH,
    additive_rollup,
    canonical_sha256,
    read_json,
    sha256_file,
    validate_control_events,
    validate_probe_events,
    write_json,
)


M = 8192
WARMUPS = 2
REPLAYS = 5


class ExactModuleOverlayFinder(importlib.abc.MetaPathFinder):
    def __init__(self, mapping: Mapping[str, Path]):
        self.mapping = dict(mapping)

    def find_spec(self, fullname: str, path=None, target=None):
        del path, target
        overlay = self.mapping.get(fullname)
        if overlay is None:
            return None
        return importlib.util.spec_from_file_location(fullname, overlay)


def install_overlays(kernel: Path, dispatch: Path) -> None:
    imported = [
        module for module in (KERNEL_MODULE, DISPATCH_MODULE) if module in sys.modules
    ]
    if imported:
        raise RuntimeError(f"target modules imported before overlays: {imported}")
    sys.meta_path.insert(
        0,
        ExactModuleOverlayFinder(
            {KERNEL_MODULE: kernel.resolve(), DISPATCH_MODULE: dispatch.resolve()}
        ),
    )


def _load_gpu_modules():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - locked GPU image only
        raise RuntimeError("capture requires the locked Torch/CUDA image") from error
    import run_exp005_arm as worker

    return torch, worker


def _overlay_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = args.overlay_root.resolve() / args.version / args.arm
    return root, root / "moe_dynamic_kernel.py", root / "moe_dispatch.py"


def overlay_identity_gate(args: argparse.Namespace) -> dict[str, Any]:
    root, kernel, dispatch = _overlay_paths(args)
    repo = args.flashinfer_root.resolve()
    errors: list[str] = []
    manifest: Mapping[str, Any] = {}
    identity = root / "identity.json"
    if not identity.is_file():
        errors.append("overlay identity.json missing")
    else:
        manifest = read_json(identity)
    if manifest:
        if manifest.get("schema") != "exp008.phase-marker-overlay.v1":
            errors.append("overlay schema drift")
        if manifest.get("version") != args.version:
            errors.append("overlay version drift")
        if manifest.get("arm") != args.arm:
            errors.append("overlay arm drift")
        if bool(manifest.get("probe_enabled")) != (args.arm == PROBE):
            errors.append("overlay marker-enable flag drift")
        if manifest.get("event_abi") != EVENT_ABI:
            errors.append("overlay event ABI drift")
        hashes = manifest.get("overlay", {})
        if not kernel.is_file() or hashes.get("kernel_sha256") != sha256_file(kernel):
            errors.append("kernel overlay hash drift")
        if not dispatch.is_file() or hashes.get("dispatch_sha256") != sha256_file(
            dispatch
        ):
            errors.append("dispatch overlay hash drift")
        if hashes.get("barrier_fingerprint") != manifest.get("base", {}).get(
            "barrier_fingerprint"
        ):
            errors.append("source barrier fingerprint changed")
        base = manifest.get("base", {})
        live_sources = (
            ("base kernel", BASE_KERNEL[args.version], base.get("kernel_sha256")),
            (
                "production dispatch",
                repo / DISPATCH_RELATIVE_PATH,
                base.get("dispatch_sha256"),
            ),
            (
                "production wrapper",
                repo / WRAPPER_RELATIVE_PATH,
                base.get("wrapper_sha256"),
            ),
        )
        for label, path, expected in live_sources:
            if not path.is_file() or expected != sha256_file(path):
                errors.append(f"{label} live-source hash drift")
    return {
        "schema": "exp008.phase-marker-overlay-gate.v1",
        "root": str(root),
        "kernel": str(kernel),
        "dispatch": str(dispatch),
        "manifest": dict(manifest),
        "errors": errors,
        "gate_pass": not errors,
    }


def _reset(workspace: Any) -> None:
    workspace.exp008_timing_ticks.fill_(SENTINEL)
    workspace.exp008_task_cta_z.fill_(SENTINEL)
    workspace.exp008_cta_ticks.fill_(SENTINEL)


def _snapshot(workspace: Any) -> dict[str, Any]:
    task_capacity = int(workspace.task_capacity)
    return {
        "task_ticks": workspace.exp008_timing_ticks.detach()
        .reshape(task_capacity, TASK_TICKS)
        .cpu()
        .clone(),
        "task_cta_z": workspace.exp008_task_cta_z.detach().cpu().clone(),
        "cta_ticks": workspace.exp008_cta_ticks.detach()
        .reshape(-1, CTA_TICKS)
        .cpu()
        .clone(),
        "task_tail": int(workspace.task_tail.item()),
        "task_capacity": task_capacity,
    }


def _event_gate(timing: Mapping[str, Any], *, arm: str) -> dict[str, Any]:
    if arm == CONTROL:
        return validate_control_events(
            timing["task_ticks"], timing["task_cta_z"], timing["cta_ticks"]
        )
    if arm == PROBE:
        return validate_probe_events(
            timing["task_ticks"],
            timing["task_cta_z"],
            timing["cta_ticks"],
            task_tail=int(timing["task_tail"]),
        )
    raise AssertionError(arm)


def _calibration_summary(timing: Mapping[str, Any]) -> dict[str, Any]:
    rows = timing["cta_ticks"].tolist()
    deltas = [int(row[CTA_CALIBRATION + 1]) - int(row[CTA_CALIBRATION]) for row in rows]
    return {
        "kind": "consecutive %globaltimer + st.global.u64 events",
        "unit": "ns",
        "median": statistics.median(deltas),
        "min": min(deltas),
        "max": max(deltas),
        "samples": len(deltas),
    }


def _correctness_gate(metrics: Mapping[str, float]) -> dict[str, Any]:
    checks = {
        "cosine_loss": float(metrics["cosine_loss"]) <= 0.001,
        "relative_l2": float(metrics["relative_l2"]) <= 0.02,
        "max_abs": float(metrics["max_abs"]) <= 0.08,
    }
    return {"checks": checks, "gate_pass": all(checks.values())}


def _tensor_hashes(worker: Any, timing: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: worker.tensor_sha256(value)
        for name, value in timing.items()
        if hasattr(value, "detach")
    }


def _artifact_gate(
    worker: Any, jit_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifacts = worker.artifact_manifest(jit_root)
    suffix_counts = {
        suffix: sum(str(item.get("path", "")).endswith(suffix) for item in artifacts)
        for suffix in (".so", ".cubin", ".ptx", ".sass")
    }
    errors = []
    if not artifacts:
        errors.append("fresh JIT artifact set is empty")
    # CuteDSL's KEEP=sass setting does not guarantee that a standalone .sass
    # file is retained.  The exact static SASS is derived from the retained
    # cubin with nvdisasm after capture, so requiring .sass here would reject a
    # valid, identity-locked JIT artifact set before that extraction can run.
    for suffix in (".so", ".cubin", ".ptx"):
        if suffix_counts[suffix] == 0:
            errors.append(f"fresh JIT retained no {suffix} artifact")
    return artifacts, {
        "schema": "exp008.phase-marker-jit-gate.v1",
        "jit_root": str(jit_root),
        "suffix_counts": suffix_counts,
        "required_suffixes": [".so", ".cubin", ".ptx"],
        "static_sass_extraction_required_post_capture": True,
        "artifact_set_sha256": canonical_sha256(artifacts),
        "errors": errors,
        "gate_pass": not errors,
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    if args.m != M or args.fixture != "canonical":
        raise RuntimeError("first exp008 phase probe is locked to canonical M8192")
    if args.warmup != WARMUPS or args.replays != REPLAYS:
        raise RuntimeError("locked protocol requires 2 warmups and 5 replays")
    overlay_gate = overlay_identity_gate(args)
    if not overlay_gate["gate_pass"]:
        raise RuntimeError(f"overlay identity gate failed: {overlay_gate['errors']}")

    torch, worker = _load_gpu_modules()
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.jit_root = args.jit_root.resolve()
    final_output = args.output.resolve()
    if final_output.exists():
        raise FileExistsError(f"immutable capture output exists: {final_output}")
    temporary_output = final_output.with_name(
        f".{final_output.name}.in-progress.{os.getpid()}"
    )
    if temporary_output.exists():
        raise FileExistsError(
            f"stale capture staging output exists: {temporary_output}"
        )
    args.output = temporary_output
    worker.require_empty_directory(args.jit_root)
    root, kernel, dispatch = _overlay_paths(args)
    del root
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
    runtime = worker.runtime_identity(args, source)
    runtime["imports"] = imports
    args.output.mkdir(parents=True)

    try:
        payload = _capture_into_staging(
            args, torch, worker, source, runtime, overlay_gate
        )
        final_output.parent.mkdir(parents=True, exist_ok=True)
        args.output.rename(final_output)
        return payload
    except BaseException:
        shutil.rmtree(args.output, ignore_errors=True)
        raise


def _capture_into_staging(
    args: argparse.Namespace,
    torch: Any,
    worker: Any,
    source: Mapping[str, Any],
    runtime: Mapping[str, Any],
    overlay_gate: Mapping[str, Any],
) -> dict[str, Any]:
    fixture_module, fixture, weights = worker.make_case(args)
    reference = fixture_module.reference_moe_nvfp4(fixture, weights)
    captured = worker.build_arm(args, fixture, weights)

    eager = captured.eager()
    eager_cpu = eager.detach().cpu().clone()
    eager_error = worker.tensor_error(eager_cpu, reference.cpu())
    workspace = captured.wrapper._dynamic_workspace
    timing = _snapshot(workspace)
    eager_event_gate = _event_gate(timing, arm=args.arm)
    _, workspace_gate = worker._workspace_snapshot(
        captured.wrapper, fixture, num_cta_warps=9
    )
    eager_payload = {
        "correctness": eager_error,
        "correctness_gate": _correctness_gate(eager_error),
        "workspace_gate": workspace_gate["verification"],
        "event_gate": eager_event_gate,
        "timing_sha256": _tensor_hashes(worker, timing),
    }
    torch.save(timing, args.output / "eager_timing.pt")
    write_json(args.output / "eager.json", eager_payload)
    if not all(
        eager_payload[name].get("gate_pass", False)
        for name in ("correctness_gate", "workspace_gate", "event_gate")
    ):
        raise RuntimeError("eager correctness/workspace/event gate failed")

    captured.capture()
    for _ in range(args.warmup):
        _reset(workspace)
        captured.replay(sentinel=False)

    runs = []
    for replay in range(args.replays):
        _reset(workspace)
        output, elapsed_ms = captured.replay(sentinel=True)
        output_cpu = output.detach().cpu().clone()
        error = worker.tensor_error(output_cpu, reference.cpu())
        timing = _snapshot(workspace)
        event_gate = _event_gate(timing, arm=args.arm)
        _, workspace_payload = worker._workspace_snapshot(
            captured.wrapper, fixture, num_cta_warps=9
        )
        run: dict[str, Any] = {
            "run_id": f"run_{replay}",
            "event_elapsed_us": elapsed_ms * 1000.0,
            "output_sha256": worker.tensor_sha256(output_cpu),
            "correctness": error,
            "correctness_gate": _correctness_gate(error),
            "workspace_gate": workspace_payload["verification"],
            "event_gate": event_gate,
            "task_tail": int(timing["task_tail"]),
            "timing_sha256": _tensor_hashes(worker, timing),
        }
        if args.arm == PROBE:
            run["marker_calibration"] = _calibration_summary(timing)
            run["additive_rollup"] = additive_rollup(
                timing["task_ticks"],
                timing["task_cta_z"],
                timing["cta_ticks"],
                task_tail=int(timing["task_tail"]),
            )
        run["gate_pass"] = all(
            run[name].get("gate_pass", False)
            for name in ("correctness_gate", "workspace_gate", "event_gate")
        )
        torch.save(timing, args.output / f"timing_{replay}.pt")
        write_json(args.output / f"run_{replay}.json", run)
        if not run["gate_pass"]:
            raise RuntimeError(f"run_{replay} gate failed")
        runs.append(run)

    artifacts, artifact_gate = _artifact_gate(worker, args.jit_root)
    if not artifact_gate["gate_pass"]:
        raise RuntimeError(f"JIT artifact gate failed: {artifact_gate['errors']}")
    samples = [float(run["event_elapsed_us"]) for run in runs]
    payload = {
        "schema": "exp008.phase-marker-capture.v1",
        "classification": "diagnostic-only" if args.arm == PROBE else "control",
        "version": args.version,
        "arm": args.arm,
        "event_abi": EVENT_ABI,
        "unmodified_gate_b_remains_performance_authority": True,
        "source": source,
        "overlay_gate": overlay_gate,
        "runtime": runtime,
        "fixture": fixture.manifest,
        "weights": weights.manifest,
        "reference_sha256": worker.tensor_sha256(reference),
        "eager": eager_payload,
        "runs": runs,
        "latency_us": {
            "median": statistics.median(samples),
            "min": min(samples),
            "max": max(samples),
            "samples": len(samples),
        },
        "jit_identity_gate": artifact_gate,
        "jit_artifacts": artifacts,
    }
    write_json(args.output / "capture.json", payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--version", choices=VERSIONS, required=True)
    parser.add_argument("--arm", choices=MARKER_ARMS, required=True)
    parser.add_argument("--overlay-root", type=Path, default=OVERLAY_ROOT)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--m", type=int, default=M, choices=[M])
    parser.add_argument("--fixture", default="canonical", choices=["canonical"])
    parser.add_argument("--device-index", type=int, default=0, choices=[0])
    parser.add_argument("--seed", type=int, default=2026, choices=[2026])
    parser.add_argument("--warmup", type=int, default=WARMUPS)
    parser.add_argument("--replays", type=int, default=REPLAYS)
    parser.add_argument(
        "--check-overlays-only",
        action="store_true",
        help="run only the CPU overlay/source identity gate",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    gate = overlay_identity_gate(args)
    if args.check_overlays_only:
        print(json.dumps(gate, sort_keys=True))
        return 0 if gate["gate_pass"] else 1
    payload = capture(args)
    print(
        json.dumps(
            {
                "version": args.version,
                "arm": args.arm,
                "latency_us": payload["latency_us"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
