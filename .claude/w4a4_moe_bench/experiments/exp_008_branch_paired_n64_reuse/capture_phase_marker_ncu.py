#!/usr/bin/env python3
"""Capture matched NCU resource/spill evidence for one exp_008 marker arm.

``--dry-run`` is CPU-only: it validates the standalone timing capture, overlay,
retained cubin and exact NCU command without opening a CUDA context.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Sequence


from build_phase_marker_resource import (
    NCU_METRICS,
    _artifact_path,
    _unique_capture_artifact,
    build as build_resource,
    parse_native_raw,
)
from build_static_spill_evidence import parse_binary, run_checked
from capture_phase_timing import overlay_identity_gate
from exp008_marker_common import (
    MARKER_ARMS,
    VERSIONS,
    read_json,
    sha256_file,
    write_json,
)
from profile_phase_marker_target import _timing_capture_gate


SECTION_IDS = ("InstructionStats", "SourceCounters", "LaunchStats")


def _tool_version(executable: str) -> str:
    return subprocess.check_output(
        [executable, "--version"], text=True, stderr=subprocess.STDOUT
    ).strip()


def _empty_or_absent(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.overlay_root = args.overlay_root.resolve()
    args.timing_capture = args.timing_capture.resolve()
    args.timing_jit_root = args.timing_jit_root.resolve()
    args.profile_jit_root = args.profile_jit_root.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"immutable marker NCU output exists: {args.output}")
    if not _empty_or_absent(args.profile_jit_root):
        raise RuntimeError(
            f"profile JIT root must be fresh/empty: {args.profile_jit_root}"
        )

    capture = read_json(args.timing_capture)
    timing_gate = _timing_capture_gate(
        capture, version=args.version, arm=args.arm
    )
    if not timing_gate["gate_pass"]:
        raise RuntimeError(f"timing capture prerequisite failed: {timing_gate}")
    if capture.get("runtime", {}).get("gpu", {}).get("uuid") != args.expected_gpu_uuid:
        raise RuntimeError("timing capture GPU UUID drift")
    if int(
        capture.get("runtime", {})
        .get("gpu", {})
        .get("applications_graphics_clock_mhz", -1)
    ) != int(args.expected_app_clock_mhz):
        raise RuntimeError("timing capture application graphics clock drift")
    overlay_gate = overlay_identity_gate(args)
    if not overlay_gate["gate_pass"]:
        raise RuntimeError(f"overlay identity gate failed: {overlay_gate['errors']}")
    if capture.get("source", {}).get("kernel_sha256") != overlay_gate.get(
        "manifest", {}
    ).get("overlay", {}).get("kernel_sha256"):
        raise RuntimeError("timing capture/overlay kernel identity drift")
    if capture.get("source", {}).get("dispatch_sha256") != overlay_gate.get(
        "manifest", {}
    ).get("overlay", {}).get("dispatch_sha256"):
        raise RuntimeError("timing capture/overlay dispatch identity drift")

    cubin_artifact = _unique_capture_artifact(capture, suffix=".cubin")
    timing_cubin = _artifact_path(args.timing_jit_root, cubin_artifact)
    binary = parse_binary(
        cubin_path=timing_cubin,
        cuobjdump=args.cuobjdump,
        nvdisasm=args.nvdisasm,
    )
    kernel_symbol = str(binary["kernel_symbol"])

    planned = args.output.with_name(f".{args.output.name}.in-progress.PID")
    report_base = planned / "trace"
    target_path = planned / "target.json"
    target = [
        sys.executable,
        str(Path(__file__).resolve().parent / "profile_phase_marker_target.py"),
        "--flashinfer-root",
        str(args.flashinfer_root),
        "--version",
        args.version,
        "--arm",
        args.arm,
        "--overlay-root",
        str(args.overlay_root),
        "--timing-capture",
        str(args.timing_capture),
        "--jit-root",
        str(args.profile_jit_root),
        "--output",
        str(target_path),
        "--expected-gpu-uuid",
        args.expected_gpu_uuid,
        "--expected-app-clock-mhz",
        str(args.expected_app_clock_mhz),
    ]
    sections = [item for section in SECTION_IDS for item in ("--section", section)]
    command = [
        args.ncu,
        "--force-overwrite",
        "--profile-from-start",
        "off",
        "--target-processes",
        "all",
        "--graph-profiling",
        "node",
        "--replay-mode",
        "kernel",
        "--cache-control",
        "all",
        "--clock-control",
        "none",
        "--kernel-name",
        f"regex:^{kernel_symbol}$",
        "--kernel-name-base",
        "demangled",
        "--launch-count",
        "1",
        *sections,
        "--metrics",
        ",".join(
            (
                NCU_METRICS["registers_per_thread"],
                NCU_METRICS["smem_bytes"],
                NCU_METRICS["achieved_occupancy_pct"],
            )
        ),
        "--export",
        str(report_base),
        *target,
    ]
    return {
        "capture": capture,
        "timing_gate": timing_gate,
        "overlay_gate": overlay_gate,
        "cubin_artifact": cubin_artifact,
        "timing_cubin": timing_cubin,
        "binary": binary,
        "kernel_symbol": kernel_symbol,
        "planned_output": planned,
        "planned_report_base": report_base,
        "planned_target_path": target_path,
        "command": command,
    }


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    contract = build_contract(args)
    payload = {
        "schema": "exp008.phase-marker-ncu-plan.v1",
        "mode": "dry-run-no-gpu",
        "version": args.version,
        "arm": args.arm,
        "timing_capture": str(args.timing_capture),
        "timing_cubin_sha256": contract["cubin_artifact"]["sha256"],
        "kernel_symbol": contract["kernel_symbol"],
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "expected_application_graphics_clock_mhz": args.expected_app_clock_mhz,
        "expected_grid": [1, 1, 110],
        "expected_block": [288, 1, 1],
        "section_ids": list(SECTION_IDS),
        "required_metric_ids": list(NCU_METRICS.values()),
        "profile_jit_root": str(args.profile_jit_root),
        "output": str(args.output),
        "command": contract["command"],
        "command_shell": shlex.join(contract["command"]),
        "gpu_or_ncu_invoked": False,
    }
    print(json.dumps(payload, sort_keys=True))
    return payload


def _replace_once(command: list[str], old: str, new: str) -> None:
    if command.count(old) != 1:
        raise RuntimeError(f"NCU command placeholder drift: {old}")
    command[command.index(old)] = new


def capture(args: argparse.Namespace) -> dict[str, Any]:
    contract = build_contract(args)
    temporary = args.output.with_name(f".{args.output.name}.in-progress.{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    args.profile_jit_root.mkdir(parents=True, exist_ok=True)
    report_base = temporary / "trace"
    target_path = temporary / "target.json"
    command = list(contract["command"])
    _replace_once(
        command, str(contract["planned_report_base"]), str(report_base)
    )
    _replace_once(
        command, str(contract["planned_target_path"]), str(target_path)
    )
    try:
        (temporary / "command.txt").write_text(shlex.join(command) + "\n")
        with (
            (temporary / "stdout.log").open("w") as stdout,
            (temporary / "stderr.log").open("w") as stderr,
        ):
            subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
        report = report_base.with_suffix(".ncu-rep")
        if not report.is_file() or report.stat().st_size == 0:
            raise RuntimeError("NCU produced no non-empty report")
        if not target_path.is_file():
            raise RuntimeError("profile target produced no target.json")

        native = temporary / "native_raw.csv"
        export = [
            args.ncu,
            "--import",
            str(report),
            "--csv",
            "--page",
            "raw",
            "--print-units",
            "base",
        ]
        (temporary / "native_raw.command.txt").write_text(shlex.join(export) + "\n")
        with (
            native.open("w") as stdout,
            (temporary / "native_raw.stderr.log").open("w") as stderr,
        ):
            subprocess.run(export, stdout=stdout, stderr=stderr, check=True)
        _, launch = parse_native_raw(
            native, expected_kernel_symbol=contract["kernel_symbol"]
        )
        if launch["grid"] != [1, 1, 110] or launch["block"] != [288, 1, 1]:
            raise RuntimeError(f"profiled launch geometry drift: {launch}")

        target = read_json(target_path)
        if not target.get("gate_pass", False):
            raise RuntimeError("profile target gate failed")
        profile_cubins = [
            item
            for item in target.get("profile_jit_artifacts", [])
            if str(item.get("path", "")).endswith(".cubin")
        ]
        if len(profile_cubins) != 1:
            raise RuntimeError(f"expected one profile cubin: {profile_cubins}")
        timing_hash = str(contract["cubin_artifact"]["sha256"])
        if profile_cubins[0].get("sha256") != timing_hash:
            raise RuntimeError("profile cubin != standalone timing cubin")

        source = contract["capture"]["source"]
        identity = {
            "schema": "exp008.phase-marker-ncu-identity.v1",
            "version": args.version,
            "arm": args.arm,
            "timing_capture_path": str(args.timing_capture),
            "timing_capture_sha256": sha256_file(args.timing_capture),
            "timing_jit_artifact_set_sha256": contract["capture"][
                "jit_identity_gate"
            ]["artifact_set_sha256"],
            "timing_cubin_path": str(contract["timing_cubin"]),
            "timing_cubin_sha256": timing_hash,
            "profile_jit_root": str(args.profile_jit_root),
            "profile_jit_artifact_set_sha256": target[
                "profile_jit_artifact_set_sha256"
            ],
            "profile_cubin_sha256": profile_cubins[0]["sha256"],
            "kernel_source_sha256": source["kernel_sha256"],
            "dispatch_source_sha256": source["dispatch_sha256"],
            "expected_kernel_symbol": contract["kernel_symbol"],
            "expected_gpu_uuid": args.expected_gpu_uuid,
            "expected_application_graphics_clock_mhz": args.expected_app_clock_mhz,
            "target_sha256": sha256_file(target_path),
            "trace_sha256": sha256_file(report),
            "native_raw_sha256": sha256_file(native),
            "ncu_version": _tool_version(args.ncu),
            "cuobjdump_version": _tool_version(args.cuobjdump),
            "nvdisasm_version": _tool_version(args.nvdisasm),
            "clock_control": "none",
            "section_ids": list(SECTION_IDS),
            "required_metric_ids": list(NCU_METRICS.values()),
            "observed_launch": launch,
        }
        write_json(temporary / "capture_identity.json", identity)
        resource = build_resource(
            timing_capture_path=args.timing_capture,
            timing_jit_root=args.timing_jit_root,
            ncu_dir=temporary,
            output=temporary / "resource.json",
            cuobjdump=args.cuobjdump,
            nvdisasm=args.nvdisasm,
        )
        if not resource["gate_pass"]:
            raise RuntimeError(f"marker resource evidence gate failed: {resource}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(args.output)
        return resource
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--version", choices=VERSIONS, required=True)
    parser.add_argument("--arm", choices=MARKER_ARMS, required=True)
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--timing-capture", type=Path, required=True)
    parser.add_argument("--timing-jit-root", type=Path, required=True)
    parser.add_argument("--profile-jit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-app-clock-mhz", type=int, required=True)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--cuobjdump", default="/usr/local/cuda/bin/cuobjdump")
    parser.add_argument("--nvdisasm", default="/usr/local/cuda/bin/nvdisasm")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_checked([args.cuobjdump, "--version"])
    run_checked([args.nvdisasm, "--version"])
    if args.dry_run:
        dry_run(args)
        return 0
    result = capture(args)
    print(
        json.dumps(
            {
                "version": args.version,
                "arm": args.arm,
                "gate_pass": result["gate_pass"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
