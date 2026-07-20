#!/usr/bin/env python3
"""Capture one identity-locked exp_016 Candidate graph node with native NCU."""

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
EXP005 = ROOT.parent / "exp_005_8warp_spill_reduction"
for dependency in (ROOT, EXP005):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

import build_dynamic_spill_evidence as evidence  # noqa: E402
import exp005_common as common  # noqa: E402


SECTION_IDS = ("InstructionStats", "SourceCounters", "LaunchStats")
EXPLICIT_METRIC_IDS = tuple(evidence.METRIC_IDS.values())
KERNEL_REGEX = ".*MoEDynamicKernel.*"


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


def validate_prerequisite(
    *, results: Path, validation: Path, jit_root: Path
) -> dict[str, Any]:
    identity = evidence.validated_candidate_identity(results, validation)
    if (
        not identity["jit_root"]
        or Path(identity["jit_root"]).resolve() != jit_root.resolve()
    ):
        raise RuntimeError(
            f"Candidate JIT root drift: {jit_root.resolve()} != {identity['jit_root']}"
        )
    artifacts = common.artifact_manifest(jit_root)
    artifact_set = common.canonical_sha256(artifacts)
    if artifact_set != identity["jit_artifact_set_sha256"]:
        raise RuntimeError(
            "registered Candidate JIT artifact drift: "
            f"{artifact_set} != {identity['jit_artifact_set_sha256']}"
        )
    cubins = sorted(
        {
            str(item["sha256"])
            for item in artifacts
            if str(item["path"]).endswith(".cubin")
        }
    )
    if cubins != [identity["cubin_sha256"]]:
        raise RuntimeError(
            f"registered Candidate cubin drift: {cubins} != "
            f"{[identity['cubin_sha256']]}"
        )
    matching = [
        path
        for path in sorted(jit_root.rglob("*.cubin"))
        if sha256_file(path) == identity["cubin_sha256"]
    ]
    if len(matching) != 1:
        raise RuntimeError(
            f"expected exactly one registered Candidate cubin, got {matching}"
        )
    return {**identity, "cubin": matching[0], "artifacts": artifacts}


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
        "--validation",
        str(args.validation),
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
        str(prerequisite["gpu_uuid"]),
        "--expected-app-clock-mhz",
        str(prerequisite["application_graphics_clock_mhz"]),
        "--expected-validation-sha256",
        str(prerequisite["sha256"]),
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


def validate_target(target: Mapping[str, Any], prerequisite: Mapping[str, Any]) -> None:
    runtime = target.get("runtime")
    checks = {
        "schema": target.get("schema") == "exp016.dynamic-spill-profile-target.v1",
        "status": target.get("status") == "complete",
        "arm": target.get("arm") == evidence.CANDIDATE,
        "m": target.get("m") == evidence.M,
        "fixture": target.get("fixture") == evidence.FIXTURE,
        "scale": target.get("scale_kind") == evidence.SCALE_KIND,
        "source": target.get("source_sha256") == prerequisite["source_sha256"],
        "cubin": target.get("cubin_sha256") == prerequisite["cubin_sha256"],
        "artifacts": target.get("jit_artifact_set_sha256")
        == prerequisite["jit_artifact_set_sha256"],
        "gpu": target.get("gpu_uuid") == prerequisite["gpu_uuid"],
        "validation": target.get("validation_sha256") == prerequisite["sha256"],
        "launch": target.get("expected_launch")
        == {
            "grid": evidence.EXPECTED_GRID,
            "block": evidence.EXPECTED_BLOCK,
            "kernel": "MoEDynamicKernel",
        },
        "runtime": isinstance(runtime, Mapping)
        and evidence.stable_runtime_identity(runtime)
        == prerequisite["runtime_identity"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"profile-target identity drift: {checks}")


def capture(args: argparse.Namespace) -> None:
    output = evidence.capture_root(args.results)
    if output.exists():
        raise FileExistsError(f"immutable dynamic NCU capture exists: {output}")
    prerequisite = validate_prerequisite(
        results=args.results, validation=args.validation, jit_root=args.jit_root
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
        target = evidence.read_json(target_path)
        validate_target(target, prerequisite)

        identity = {
            "schema": "exp016.dynamic-spill-capture-identity.v1",
            "arm": evidence.CANDIDATE,
            "m": evidence.M,
            "fixture": evidence.FIXTURE,
            "scale_kind": evidence.SCALE_KIND,
            "source_sha256": prerequisite["source_sha256"],
            "cubin_sha256": prerequisite["cubin_sha256"],
            "jit_artifact_set_sha256": prerequisite["jit_artifact_set_sha256"],
            "gpu_uuid": prerequisite["gpu_uuid"],
            "expected_application_graphics_clock_mhz": prerequisite[
                "application_graphics_clock_mhz"
            ],
            "expected_grid": evidence.EXPECTED_GRID,
            "expected_block": evidence.EXPECTED_BLOCK,
            "kernel_filter": KERNEL_REGEX,
            "kernel_symbol": observed["kernel_symbol"],
            "observed_launch": observed,
            "metrics": metrics,
            "section_ids": list(SECTION_IDS),
            "explicit_metric_ids": list(EXPLICIT_METRIC_IDS),
            "required_metric_ids": list(evidence.METRIC_IDS.values()),
            "validation_sha256": prerequisite["sha256"],
            "trace_sha256": sha256_file(report),
            "native_raw_sha256": sha256_file(native_path),
            "profile_target_sha256": sha256_file(target_path),
            "ncu_version": tool_version(args.ncu),
            "clock_control": "none",
            "capture_protocol": (
                "profile-from-start off; CUDA profiler API brackets exactly one "
                "final graph replay; graph node mode; kernel replay; launch-count=1"
            ),
        }
        write_json(temporary / "capture_identity.json", identity)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    output = evidence.capture_root(args.results)
    if output.exists():
        raise FileExistsError(f"immutable dynamic NCU capture exists: {output}")
    prerequisite = validate_prerequisite(
        results=args.results, validation=args.validation, jit_root=args.jit_root
    )
    placeholder = output.with_name(f".{output.name}.in-progress.PID")
    command = build_ncu_command(
        args,
        prerequisite,
        report_base=placeholder / "trace",
        target_output=placeholder / "profile_target.json",
    )
    return {
        "schema": "exp016.dynamic-spill-capture-plan.v1",
        "mode": "dry-run-no-ncu-no-gpu",
        "arm": evidence.CANDIDATE,
        "m": evidence.M,
        "fixture": evidence.FIXTURE,
        "scale_kind": evidence.SCALE_KIND,
        "source_sha256": prerequisite["source_sha256"],
        "cubin_sha256": prerequisite["cubin_sha256"],
        "jit_artifact_set_sha256": prerequisite["jit_artifact_set_sha256"],
        "gpu_uuid": prerequisite["gpu_uuid"],
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
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.results = args.results.resolve()
    args.validation = args.validation.resolve()
    args.jit_root = args.jit_root.resolve()
    if args.dry_run:
        print(json.dumps(dry_run(args), sort_keys=True))
        return 0
    capture(args)
    print(
        json.dumps({"status": "complete", "arm": evidence.CANDIDATE, "m": evidence.M})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
