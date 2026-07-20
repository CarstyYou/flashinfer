#!/usr/bin/env python3
"""Capture one identity-locked exp_014 M8192 CUDA Graph node with NCU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import build_dynamic_spill_evidence as evidence  # noqa: E402


SECTION_IDS = ("InstructionStats", "SourceCounters", "LaunchStats")
EXPLICIT_METRIC_IDS = tuple(evidence.METRIC_IDS[name] for name in evidence.WORK_METRICS)
KERNEL_REGEX = ".*MoEDynamicKernel.*"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tool_version(executable: str) -> str:
    return subprocess.check_output(
        [executable, "--version"], text=True, stderr=subprocess.STDOUT
    ).strip()


def find_exact_cubin(jit_root: Path, expected_sha256: str) -> Path:
    cubins = sorted(jit_root.rglob("*.cubin"))
    matches = [path for path in cubins if sha256_file(path) == expected_sha256]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one matching cubin {expected_sha256}, got {matches}"
        )
    return matches[0]


def validate_prerequisite(*, results: Path, arm: str, jit_root: Path) -> dict[str, Any]:
    identity = evidence.validated_arm_identity(results, arm)
    cubin = find_exact_cubin(jit_root, identity["cubin_sha256"])
    return {**identity, "cubin": cubin}


def metric_selection_args() -> list[str]:
    arguments = [item for section in SECTION_IDS for item in ("--section", section)]
    arguments.extend(("--metrics", ",".join(EXPLICIT_METRIC_IDS)))
    return arguments


def build_ncu_command(
    args: argparse.Namespace,
    prerequisite: Mapping[str, Any],
    *,
    report_base: Path,
    target_output: Path,
) -> list[str]:
    target = ROOT / "profile_dynamic_ncu_target.py"
    target_command = [
        sys.executable,
        str(target),
        "--flashinfer-root",
        str(args.flashinfer_root),
        "--results",
        str(args.results),
        "--arm",
        args.arm,
        "--overlay",
        str(prerequisite["overlay"]),
        "--jit-root",
        str(args.jit_root),
        "--expected-source-sha256",
        str(prerequisite["source_sha256"]),
        "--expected-cubin-sha256",
        str(prerequisite["cubin_sha256"]),
        "--expected-jit-artifact-set-sha256",
        str(prerequisite["jit_artifact_set_sha256"]),
        "--expected-gpu-uuid",
        args.expected_gpu_uuid,
        "--expected-app-clock-mhz",
        str(args.expected_app_clock_mhz),
        "--output",
        str(target_output),
    ]
    return [
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
        f"regex:{KERNEL_REGEX}",
        "--kernel-name-base",
        "demangled",
        "--launch-count",
        "1",
        *metric_selection_args(),
        "--export",
        str(report_base),
        *target_command,
    ]


def validate_target(
    target: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    prerequisite: Mapping[str, Any],
) -> None:
    checks = {
        "schema": target.get("schema") == "exp014.dynamic-spill-profile-target.v1",
        "complete": target.get("status") == "complete",
        "arm": target.get("arm") == args.arm,
        "m": target.get("m") == evidence.M,
        "fixture": target.get("fixture_kind") == evidence.FIXTURE,
        "source": target.get("source_sha256") == prerequisite["source_sha256"],
        "cubin": target.get("cubin_sha256") == prerequisite["cubin_sha256"],
        "artifacts": target.get("jit_artifact_set_sha256")
        == prerequisite["jit_artifact_set_sha256"],
        "gpu": target.get("gpu_uuid") == args.expected_gpu_uuid,
        "launch": target.get("expected_launch")
        == {
            "grid": evidence.EXPECTED_GRID,
            "block": evidence.EXPECTED_BLOCK,
            "kernel": "MoEDynamicKernel",
        },
    }
    if not all(checks.values()):
        raise RuntimeError(f"profile-target identity drift: {checks}")


def capture(args: argparse.Namespace) -> None:
    output = evidence.capture_root(args.results, args.arm)
    if output.exists():
        raise FileExistsError(f"immutable dynamic NCU capture exists: {output}")
    prerequisite = validate_prerequisite(
        results=args.results, arm=args.arm, jit_root=args.jit_root
    )
    temporary = output.with_name(f".{output.name}.in-progress.{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    report_base = temporary / "trace"
    target_path = temporary / "profile_target.json"
    command = build_ncu_command(
        args, prerequisite, report_base=report_base, target_output=target_path
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
        native_path = temporary / "native_raw.csv"
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
            native_path.open("w") as stdout,
            (temporary / "native_raw.stderr.log").open("w") as stderr,
        ):
            subprocess.run(export, stdout=stdout, stderr=stderr, check=True)

        metrics, observed = evidence.parse_native_raw(native_path)
        if observed["grid"] != evidence.EXPECTED_GRID:
            raise RuntimeError(f"profiled grid drift: {observed['grid']}")
        if observed["block"] != evidence.EXPECTED_BLOCK:
            raise RuntimeError(f"profiled block drift: {observed['block']}")
        target = read_json(target_path)
        validate_target(target, args=args, prerequisite=prerequisite)

        identity = {
            "schema": "exp014.dynamic-spill-capture-identity.v1",
            "arm": args.arm,
            "m": evidence.M,
            "fixture": evidence.FIXTURE,
            "source_sha256": prerequisite["source_sha256"],
            "cubin_sha256": prerequisite["cubin_sha256"],
            "jit_artifact_set_sha256": prerequisite["jit_artifact_set_sha256"],
            "gpu_uuid": args.expected_gpu_uuid,
            "expected_application_graphics_clock_mhz": args.expected_app_clock_mhz,
            "expected_grid": evidence.EXPECTED_GRID,
            "expected_block": evidence.EXPECTED_BLOCK,
            "kernel_filter": KERNEL_REGEX,
            "kernel_symbol": observed["kernel_symbol"],
            "observed_launch": observed,
            "metrics": metrics,
            "section_ids": list(SECTION_IDS),
            "explicit_metric_ids": list(EXPLICIT_METRIC_IDS),
            "required_metric_ids": list(evidence.METRIC_IDS.values()),
            "validation_manifest_sha256": prerequisite["sha256"],
            "trace_sha256": sha256_file(report),
            "native_raw_sha256": sha256_file(native_path),
            "profile_target_sha256": sha256_file(target_path),
            "ncu_version": tool_version(args.ncu),
            "clock_control": "none",
            "capture_protocol": (
                "profile-from-start off; CUDA profiler API brackets one final "
                "graph replay; graph node mode; kernel replay; launch-count=1"
            ),
        }
        write_json(temporary / "capture_identity.json", identity)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    output = evidence.capture_root(args.results, args.arm)
    if output.exists():
        raise FileExistsError(f"immutable dynamic NCU capture exists: {output}")
    prerequisite = validate_prerequisite(
        results=args.results, arm=args.arm, jit_root=args.jit_root
    )
    placeholder = output.with_name(f".{output.name}.in-progress.PID")
    command = build_ncu_command(
        args,
        prerequisite,
        report_base=placeholder / "trace",
        target_output=placeholder / "profile_target.json",
    )
    return {
        "schema": "exp014.dynamic-spill-capture-plan.v1",
        "mode": "dry-run-no-ncu-no-gpu",
        "arm": args.arm,
        "m": evidence.M,
        "fixture": evidence.FIXTURE,
        "source_sha256": prerequisite["source_sha256"],
        "cubin_sha256": prerequisite["cubin_sha256"],
        "jit_artifact_set_sha256": prerequisite["jit_artifact_set_sha256"],
        "gpu_uuid": args.expected_gpu_uuid,
        "grid": evidence.EXPECTED_GRID,
        "block": evidence.EXPECTED_BLOCK,
        "section_ids": list(SECTION_IDS),
        "required_metric_ids": list(evidence.METRIC_IDS.values()),
        "output": str(output),
        "command": command,
        "command_shell": shlex.join(command),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--arm", choices=evidence.ARMS, required=True)
    parser.add_argument(
        "--expected-gpu-uuid",
        choices=(evidence.EXPECTED_GPU_UUID,),
        default=evidence.EXPECTED_GPU_UUID,
    )
    parser.add_argument(
        "--expected-app-clock-mhz",
        type=int,
        choices=(evidence.EXPECTED_APPLICATION_CLOCK_MHZ,),
        default=evidence.EXPECTED_APPLICATION_CLOCK_MHZ,
    )
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.results = args.results.resolve()
    args.jit_root = args.jit_root.resolve()
    if args.dry_run:
        print(json.dumps(dry_run(args), sort_keys=True))
        return 0
    capture(args)
    print(json.dumps({"status": "complete", "arm": args.arm, "m": evidence.M}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
