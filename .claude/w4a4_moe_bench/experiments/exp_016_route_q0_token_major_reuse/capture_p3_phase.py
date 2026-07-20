#!/usr/bin/env python3
"""Capture one exp_016 arm/mode for the narrow P3 `%globaltimer` probe."""

from __future__ import annotations

import argparse
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence

from build_p3_probe_overlays import overlay_paths
from exp016_p3_probe_common import (
    ARMS,
    BASE_OVERLAY_ROOT,
    CONTROL,
    DISPATCH_MODULE,
    DISPATCH_RELATIVE_PATH,
    EVENT_ABI,
    EXPECTED_BASE_KERNEL_SHA256,
    EXPECTED_DISPATCH_SHA256,
    EXPECTED_WRAPPER_SHA256,
    GRID_CTAS,
    KERNEL_MODULE,
    MODES,
    PROBE,
    PROBE_OVERLAY_ROOT,
    RESULTS,
    SENTINEL,
    TICKS_PER_CTA,
    WRAPPER_RELATIVE_PATH,
    barrier_fingerprint,
    canonical_sha256,
    capture_summary,
    file_sha256,
    normalize_dispatch_flag,
    read_json,
    validate_ticks,
    write_json,
)


ROOT = Path(__file__).resolve().parent
M = 8192
WARMUPS = 2
REPLAYS = 5

RESOURCE_RE = re.compile(
    r"Function\s+(\S+):\s*\n\s*"
    r"REG\s*:\s*(\d+)\s+STACK\s*:\s*(\d+)\s+"
    r"SHARED\s*:\s*(\d+)\s+LOCAL\s*:\s*(\d+)",
    re.I,
)


class ExactModuleOverlayFinder(importlib.abc.MetaPathFinder):
    def __init__(self, mapping: Mapping[str, Path]):
        self.mapping = {name: path.resolve() for name, path in mapping.items()}

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
        raise RuntimeError(f"target modules imported before P3 overlays: {imported}")
    sys.meta_path.insert(
        0,
        ExactModuleOverlayFinder(
            {KERNEL_MODULE: kernel.resolve(), DISPATCH_MODULE: dispatch.resolve()}
        ),
    )


def overlay_identity_gate(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.flashinfer_root.resolve()
    output = args.overlay_root.resolve()
    root, kernel, dispatch = overlay_paths(output, args.arm, args.mode)
    errors: list[str] = []
    root_identity: Mapping[str, Any] = {}
    arm_identity: Mapping[str, Any] = {}
    try:
        root_identity = read_json(output / "identity.json")
    except (OSError, ValueError) as error:
        errors.append(f"root identity unavailable: {error}")
    try:
        arm_identity = read_json(root / "identity.json")
    except (OSError, ValueError) as error:
        errors.append(f"arm/mode identity unavailable: {error}")

    if root_identity:
        if root_identity.get("schema") != "exp016.p3-phase-probe-overlays.v1":
            errors.append("root schema drift")
        if root_identity.get("event_abi") != EVENT_ABI:
            errors.append("root event ABI drift")
    if arm_identity:
        if arm_identity.get("schema") != "exp016.p3-phase-probe-overlay.v1":
            errors.append("arm/mode schema drift")
        if arm_identity.get("arm") != args.arm:
            errors.append("arm identity drift")
        if arm_identity.get("mode") != args.mode:
            errors.append("marker mode identity drift")
        if bool(arm_identity.get("probe_enabled")) != (args.mode == PROBE):
            errors.append("probe-enable identity drift")
        overlay = arm_identity.get("overlay", {})
        if not kernel.is_file() or file_sha256(kernel) != overlay.get("kernel_sha256"):
            errors.append("kernel overlay hash drift")
        if not dispatch.is_file() or file_sha256(dispatch) != overlay.get(
            "dispatch_sha256"
        ):
            errors.append("dispatch overlay hash drift")
        base = BASE_OVERLAY_ROOT / args.arm / "moe_dynamic_kernel.py"
        if (
            not base.is_file()
            or file_sha256(base) != EXPECTED_BASE_KERNEL_SHA256[args.arm]
        ):
            errors.append("base arm overlay drift")
        if kernel.is_file() and barrier_fingerprint(
            kernel.read_text(encoding="utf-8")
        ) != arm_identity.get("base", {}).get("barrier_fingerprint"):
            errors.append("probe changed barrier fingerprint")

    peer_mode = PROBE if args.mode == CONTROL else CONTROL
    _, peer_kernel, peer_dispatch = overlay_paths(output, args.arm, peer_mode)
    if kernel.is_file() and peer_kernel.is_file():
        if kernel.read_bytes() != peer_kernel.read_bytes():
            errors.append("control/probe kernel source is not byte-identical")
    else:
        errors.append("control/probe peer kernel missing")
    if dispatch.is_file() and peer_dispatch.is_file():
        if normalize_dispatch_flag(
            dispatch.read_text(encoding="utf-8")
        ) != normalize_dispatch_flag(peer_dispatch.read_text(encoding="utf-8")):
            errors.append("control/probe dispatch differs beyond enable flag")
    else:
        errors.append("control/probe peer dispatch missing")

    live = (
        (repo / DISPATCH_RELATIVE_PATH, EXPECTED_DISPATCH_SHA256, "dispatch"),
        (repo / WRAPPER_RELATIVE_PATH, EXPECTED_WRAPPER_SHA256, "wrapper"),
    )
    for path, expected, label in live:
        if not path.is_file() or file_sha256(path) != expected:
            errors.append(f"live {label} identity drift")
    return {
        "schema": "exp016.p3-phase-overlay-gate.v1",
        "arm": args.arm,
        "mode": args.mode,
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
    except ImportError as error:  # pragma: no cover - locked GPU image only
        raise RuntimeError("P3 capture requires Torch/CUDA") from error
    import run_exp016_arm as core

    return torch, core


def reset_ticks(workspace: Any) -> None:
    workspace.exp016_p3_ticks.fill_(SENTINEL)


def snapshot_ticks(workspace: Any, *, mode: str) -> tuple[Any, dict[str, Any]]:
    if not hasattr(workspace, "exp016_p3_ticks"):
        raise RuntimeError("instrumented dynamic workspace has no P3 tick buffer")
    ticks = workspace.exp016_p3_ticks.detach().cpu().clone()
    if str(ticks.dtype) != "torch.int64":
        raise RuntimeError(f"P3 tick dtype drift: {ticks.dtype}")
    if ticks.numel() != GRID_CTAS * TICKS_PER_CTA:
        raise RuntimeError(
            f"P3 tick capacity drift: {ticks.numel()} != {GRID_CTAS * TICKS_PER_CTA}"
        )
    return ticks, validate_ticks(ticks, mode=mode)


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


def specialization_contract(core: Any, overlay: Path, *, mode: str) -> dict[str, Any]:
    dispatch = importlib.import_module(DISPATCH_MODULE)
    keys = [key for key in dispatch._DYNAMIC_KERNEL_CACHE if key[0] == "dynamic"]
    records = [
        {
            "input_scales_are_reciprocal": bool(key[9]),
            "fast_math": bool(key[10]),
            "share_input_across_experts": bool(key[-2]),
            "p3_phase_probe_enabled": bool(key[-1]),
        }
        for key in keys
    ]
    source = overlay.read_text(encoding="utf-8")
    zero_assignments = re.findall(
        r"^\s*full_tile_publish_enabled\s*=\s*Int32\(0\)\s*$",
        source,
        flags=re.MULTILINE,
    )
    all_assignments = re.findall(
        r"^\s*full_tile_publish_enabled\s*=.*$", source, flags=re.MULTILINE
    )
    expected_enabled = mode == PROBE
    checks = {
        "dynamic_cache_nonempty": bool(records),
        "unequal_scale_path": all(
            not record["share_input_across_experts"] for record in records
        ),
        "nonreciprocal_fast_math": all(
            not record["input_scales_are_reciprocal"] and record["fast_math"]
            for record in records
        ),
        "marker_specialization": all(
            record["p3_phase_probe_enabled"] == expected_enabled for record in records
        ),
        "deferred_publish_only": len(zero_assignments) == 1
        and len(all_assignments) == 1,
    }
    result = {"records": records, "checks": checks, "gate_pass": all(checks.values())}
    if not result["gate_pass"]:
        raise RuntimeError(f"P3 specialization contract failed: {result}")
    return result


def _resolve_artifact(jit_root: Path, artifact: Mapping[str, Any]) -> Path:
    path = Path(str(artifact["path"]))
    return path if path.is_absolute() else jit_root / path


def collect_resource_usage(
    *, jit_root: Path, artifacts: Sequence[Mapping[str, Any]], output: Path
) -> dict[str, Any]:
    cuobjdump = shutil.which("cuobjdump")
    if not cuobjdump:
        raise RuntimeError("cuobjdump is required for control/probe resource audit")
    cubins = [
        item for item in artifacts if str(item.get("path", "")).endswith(".cubin")
    ]
    if not cubins:
        raise RuntimeError("fresh P3 JIT retained no cubin")
    raw_sections = []
    records = []
    for artifact in cubins:
        cubin = _resolve_artifact(jit_root, artifact)
        completed = subprocess.run(
            [cuobjdump, "--dump-resource-usage", str(cubin)],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"cuobjdump resource query failed for {cubin}: {completed.stderr}"
            )
        raw_sections.append(f"===== {cubin} =====\n{completed.stdout}")
        for symbol, registers, stack, shared, local in RESOURCE_RE.findall(
            completed.stdout
        ):
            if "MoEDynamicKernel" not in symbol:
                continue
            records.append(
                {
                    "cubin_path": str(cubin),
                    "cubin_sha256": artifact["sha256"],
                    "kernel_symbol": symbol,
                    "registers_per_thread": int(registers),
                    "stack_bytes_per_thread": int(stack),
                    "static_shared_bytes_per_cta": int(shared),
                    "static_local_bytes_outside_stack": int(local),
                }
            )
    output.write_text("\n".join(raw_sections), encoding="utf-8")
    if len(records) != 1:
        raise RuntimeError(
            f"expected one MoEDynamicKernel resource record, found {len(records)}"
        )
    return {
        "schema": "exp016.p3-phase-resource-usage.v1",
        "tool": cuobjdump,
        "raw_path": output.name,
        "raw_sha256": file_sha256(output),
        "records": records,
        "comparison_contract": (
            "compare control_no_marker vs probe for the same arm; unequal resources "
            "quantify marker perturbation and may make phase evidence unresolved"
        ),
        "gate_pass": True,
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    if args.m != M or args.fixture != "canonical" or args.scale_kind != "unequal":
        raise RuntimeError("P3 probe is locked to canonical M8192 unequal [E] scale")
    if args.warmup != WARMUPS or args.replays != REPLAYS:
        raise RuntimeError("P3 probe requires warmup=2 and replays=5")
    gate = overlay_identity_gate(args)
    if not gate["gate_pass"]:
        raise RuntimeError(f"P3 overlay gate failed: {gate['errors']}")

    args.flashinfer_root = args.flashinfer_root.resolve()
    args.jit_root = args.jit_root.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"immutable P3 capture exists: {args.output}")
    staging = args.output.with_name(f".{args.output.name}.in-progress.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale P3 capture staging exists: {staging}")

    root, kernel, dispatch = overlay_paths(
        args.overlay_root.resolve(), args.arm, args.mode
    )
    del root
    install_overlays(kernel, dispatch)
    torch, core = load_gpu_modules()
    core.common.require_empty_directory(args.jit_root)
    imports = core.reused.configure_source_checkout(args.flashinfer_root, args.jit_root)
    if Path(imports["target_module"]).resolve() != kernel:
        raise RuntimeError("kernel module did not resolve to the P3 overlay")
    imported_dispatch = importlib.import_module(DISPATCH_MODULE)
    if Path(imported_dispatch.__file__).resolve() != dispatch:
        raise RuntimeError("dispatch module did not resolve to the P3 overlay")

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
            "CUTLASS commit drift: "
            f"{cutlass_commit} != {core.reused.EXPECTED_CUTLASS_COMMIT}"
        )
    source = {
        "arm": args.arm,
        "mode": args.mode,
        "locked_flashinfer_commit": core.reused.EXPECTED_FLASHINFER_COMMIT,
        "checkout_head": checkout_head,
        "cutlass_commit": cutlass_commit,
        "kernel_overlay": str(kernel),
        "kernel_sha256": file_sha256(kernel),
        "dispatch_overlay": str(dispatch),
        "dispatch_sha256": file_sha256(dispatch),
        "base_kernel_sha256": EXPECTED_BASE_KERNEL_SHA256[args.arm],
        "live_dispatch_sha256": file_sha256(
            args.flashinfer_root / DISPATCH_RELATIVE_PATH
        ),
        "live_wrapper_sha256": file_sha256(
            args.flashinfer_root / WRAPPER_RELATIVE_PATH
        ),
        "event_abi": EVENT_ABI,
    }
    runtime = core.reused.runtime_identity(args, source)
    runtime["imports"] = imports
    runtime["harness"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": file_sha256(Path(__file__).resolve()),
        "run_exp016_arm_sha256": file_sha256(ROOT / "run_exp016_arm.py"),
    }

    staging.mkdir(parents=True)
    try:
        fixture_module, fixture, weights = core.make_case(args)
        oracle_weights = core.reference_weights_for_input_scale(
            weights, args.scale_kind
        )
        reference = fixture_module.reference_moe_nvfp4(fixture, oracle_weights)
        captured = core.worker.build_arm(args, fixture, weights)

        eager_output = captured.eager().detach().cpu().clone()
        workspace = captured.wrapper._dynamic_workspace
        if workspace is None:
            raise RuntimeError("dynamic workspace missing after eager launch")
        eager_ticks, eager_timing = snapshot_ticks(workspace, mode=args.mode)
        diagnostics = fixture_module.output_diagnostics(eager_output, reference.cpu())
        _, route = core.corrected_workspace_snapshot(
            args.arm, captured.wrapper, fixture
        )
        specialization = specialization_contract(core, kernel, mode=args.mode)
        eager = {
            "correctness": diagnostics,
            "correctness_gate": correctness_gate(diagnostics, eager_output),
            "route_task_gate": route["verification"],
            "specialization_gate": specialization,
            "p3_timing": eager_timing,
            "ticks_sha256": core.tensor_sha256(eager_ticks),
        }
        eager["gate_pass"] = bool(
            eager["correctness_gate"]["gate_pass"]
            and eager["route_task_gate"]["gate_pass"]
            and eager["specialization_gate"]["gate_pass"]
            and eager["p3_timing"]["gate_pass"]
        )
        torch.save(eager_ticks, staging / "eager_ticks.pt")
        write_json(staging / "eager.json", eager)
        if not eager["gate_pass"]:
            raise RuntimeError(f"P3 eager gate failed: {eager}")

        captured.capture()
        for _ in range(args.warmup):
            reset_ticks(workspace)
            captured.replay(sentinel=False)

        runs = []
        for replay in range(args.replays):
            reset_ticks(workspace)
            output, elapsed_ms = captured.replay(sentinel=True)
            output_cpu = output.detach().cpu().clone()
            ticks, timing = snapshot_ticks(workspace, mode=args.mode)
            diagnostics = fixture_module.output_diagnostics(output_cpu, reference.cpu())
            _, route = core.corrected_workspace_snapshot(
                args.arm, captured.wrapper, fixture
            )
            run = {
                "replay": replay,
                "event_elapsed_us": elapsed_ms * 1000.0,
                "output_sha256": core.tensor_sha256(output_cpu),
                "correctness": diagnostics,
                "correctness_gate": correctness_gate(diagnostics, output_cpu),
                "route_task_gate": route["verification"],
                "p3_timing": timing,
                "ticks_sha256": core.tensor_sha256(ticks),
            }
            run["gate_pass"] = bool(
                run["correctness_gate"]["gate_pass"]
                and run["route_task_gate"]["gate_pass"]
                and run["p3_timing"]["gate_pass"]
            )
            torch.save(ticks, staging / f"ticks_{replay}.pt")
            write_json(staging / f"run_{replay}.json", run)
            if not run["gate_pass"]:
                raise RuntimeError(f"P3 replay {replay} gate failed")
            runs.append(run)

        artifacts = core.common.artifact_manifest(args.jit_root)
        if not artifacts:
            raise RuntimeError("fresh P3 JIT artifact set is empty")
        resources = collect_resource_usage(
            jit_root=args.jit_root,
            artifacts=artifacts,
            output=staging / "resource_usage.txt",
        )
        event_samples = [float(run["event_elapsed_us"]) for run in runs]
        payload = {
            "schema": "exp016.p3-phase-capture.v1",
            "classification": (
                "diagnostic matched probe"
                if args.mode == PROBE
                else "marker-disabled ABI control"
            ),
            "arm": args.arm,
            "mode": args.mode,
            "m": args.m,
            "fixture": args.fixture,
            "scale_kind": args.scale_kind,
            "event_abi": EVENT_ABI,
            "source": source,
            "overlay_gate": gate,
            "runtime": runtime,
            "fixture_identity": fixture.manifest,
            "weight_identity": weights.manifest,
            "reference_sha256": core.tensor_sha256(reference),
            "eager": eager,
            "runs": runs,
            "p3_summary": capture_summary(runs),
            "probe_e2e_us": {
                "median": statistics.median(event_samples),
                "min": min(event_samples),
                "max": max(event_samples),
                "samples": len(event_samples),
            },
            "jit_artifacts": artifacts,
            "jit_artifact_set_sha256": canonical_sha256(artifacts),
            "cubin_sha256": sorted(
                {
                    str(item["sha256"])
                    for item in artifacts
                    if str(item.get("path", "")).endswith(".cubin")
                }
            ),
            "static_resource_usage": resources,
            "evidence_boundary": (
                "P3 is reported only as max(CTA end)-min(CTA start); no additive "
                "SM estimate is produced. Compare same-arm control/probe resources "
                "and E2E perturbation before interpreting the diagnostic wall. "
                "Uninstrumented exp016 E2E remains authoritative."
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
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--overlay-root", type=Path, default=PROBE_OVERLAY_ROOT)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-app-clock-mhz", type=int, required=True)
    parser.add_argument("--m", type=int, choices=(M,), default=M)
    parser.add_argument("--fixture", choices=("canonical",), default="canonical")
    parser.add_argument("--scale-kind", choices=("unequal",), default="unequal")
    parser.add_argument("--device-index", type=int, choices=(0,), default=0)
    parser.add_argument("--seed", type=int, choices=(2026,), default=2026)
    parser.add_argument("--warmup", type=int, choices=(WARMUPS,), default=WARMUPS)
    parser.add_argument("--replays", type=int, choices=(REPLAYS,), default=REPLAYS)
    parser.add_argument("--check-overlays-only", action="store_true")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = RESULTS / "raw/p3_phase" / args.arm / args.mode
    return args


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
                "arm": args.arm,
                "mode": args.mode,
                "p3_summary": payload["p3_summary"],
                "probe_e2e_us": payload["probe_e2e_us"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
