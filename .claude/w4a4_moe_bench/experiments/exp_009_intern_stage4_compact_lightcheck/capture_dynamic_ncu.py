#!/usr/bin/env python3
"""Capture one minimal, identity-locked candidate M256 spill report.

The preparation artifact is also the correctness gate: it contains two
successful canonical replays.  The capture profiles only the final CUDA Graph
replay node selected by the exact symbol parsed from that preparation's cubin.
No throughput, phase, or source/SASS breakdown metrics are collected.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any, Sequence


ARM = "candidate_4warp_stage4_compact"
M = 256
EXPECTED_GRID = [1, 1, 110]
EXPECTED_BLOCK = [160, 1, 1]
SECTION_IDS = ("InstructionStats", "SourceCounters", "LaunchStats")
SPILL_METRICS = {
    "spill_refill_instructions": "sass__inst_executed_register_spilling_op_read",
    "spill_store_instructions": "sass__inst_executed_register_spilling_op_write",
    "spill_refill_bytes": "sass__inst_executed_register_spilling_mem_local_op_read",
    "spill_store_bytes": "sass__inst_executed_register_spilling_mem_local_op_write",
}
LAUNCH_METRICS = {
    "registers_per_thread": "launch__registers_per_thread",
    "shared_mem_per_block_bytes": "launch__shared_mem_per_block",
}
REQUIRED_METRICS = {**SPILL_METRICS, **LAUNCH_METRICS}
RESOURCE_RE = re.compile(
    r"Function\s+(\S+):\s*\n\s*"
    r"REG\s*:\s*(\d+)\s+STACK\s*:\s*(\d+)\s+"
    r"SHARED\s*:\s*(\d+)\s+LOCAL\s*:\s*(\d+)",
    re.I,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_number(value: str, *, metric: str) -> float:
    normalized = value.strip().lower()
    if normalized in {"", "n/a", "na", "not available"}:
        raise ValueError(f"missing/N-A required metric {metric}")
    parsed = float(value.replace(",", ""))
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite required metric {metric}: {value}")
    return parsed


def parse_dim(value: str, *, field: str) -> list[int]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, tuple) or len(parsed) != 3:
        raise ValueError(f"invalid {field}: {value!r}")
    result = [int(item) for item in parsed]
    if any(item <= 0 for item in result):
        raise ValueError(f"invalid {field}: {result}")
    return result


def parse_native_raw(
    path: Path, *, expected_kernel_symbol: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse exactly one NCU raw-page launch and all required metrics."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header_rows = [index for index, row in enumerate(rows) if row and row[0] == "ID"]
    if len(header_rows) != 1:
        raise ValueError(f"expected one NCU header, got {header_rows}")
    index = header_rows[0]
    if len(rows) < index + 3:
        raise ValueError("NCU raw export lacks unit/value rows")
    header, units = rows[index : index + 2]
    value_rows = [row for row in rows[index + 2 :] if any(row)]
    if len(value_rows) != 1:
        raise ValueError(f"expected one profiled graph node, got {len(value_rows)}")
    values = value_rows[0]
    if not (len(header) == len(units) == len(values)):
        raise ValueError("NCU raw CSV width mismatch")
    by_name = dict(zip(header, values, strict=True))
    by_unit = dict(zip(header, units, strict=True))

    metrics: dict[str, Any] = {}
    for label, metric_id in REQUIRED_METRICS.items():
        if metric_id not in by_name:
            raise ValueError(f"required NCU metric absent: {metric_id}")
        metrics[label] = {
            "metric_id": metric_id,
            "value": parse_number(by_name[metric_id], metric=metric_id),
            "unit": by_unit[metric_id],
        }

    kernel_symbol = by_name.get("Kernel Name", "")
    if kernel_symbol != expected_kernel_symbol:
        raise ValueError(
            "profiled kernel symbol drift: "
            f"{kernel_symbol!r} != {expected_kernel_symbol!r}"
        )
    observed = {
        "row_id": int(by_name["ID"]),
        "kernel_symbol": kernel_symbol,
        "context_id": int(by_name["Context"]),
        "stream_id": int(by_name["Stream"]),
        "device_id": int(by_name["Device"]),
        "block": parse_dim(by_name["Block Size"], field="Block Size"),
        "grid": parse_dim(by_name["Grid Size"], field="Grid Size"),
    }
    return metrics, observed


def parse_cubin_symbol(resource_text: str) -> str:
    """Return the sole cubin entry symbol from cuobjdump resource output."""

    records = RESOURCE_RE.findall(resource_text)
    if len(records) != 1:
        raise ValueError(
            f"expected one cubin kernel resource record, got {len(records)}"
        )
    symbol = records[0][0]
    if "MoEDynamicKernel" not in symbol:
        raise ValueError(f"unexpected cubin kernel entry: {symbol}")
    return symbol


def correctness_checks(preparation: dict[str, Any]) -> dict[str, bool]:
    outputs = preparation.get("outputs", [])
    route_evidence = preparation.get("route_task_evidence", [])
    launch = preparation.get("launch_contract", {})
    case = preparation.get("case", {})
    return {
        "status_complete": preparation.get("status") == "complete",
        "candidate_arm": preparation.get("arm") == ARM,
        "m256": preparation.get("m") == M and case.get("m") == M,
        "canonical_fixture": preparation.get("fixture_kind") == "canonical",
        "canonical_shape": case
        == {
            "m": M,
            "experts": 256,
            "hidden": 2048,
            "intermediate_tp": 512,
            "topk": 8,
        },
        "two_correct_replays": len(outputs) == 2
        and all(
            item.get("formal_pass") is True
            and item.get("finite") is True
            and item.get("nonzero") is True
            and item.get("sentinel_nan_remaining") == 0
            for item in outputs
        ),
        "two_route_task_gates": len(route_evidence) == 2
        and all(
            item.get("verification", {}).get("gate_pass") is True
            for item in route_evidence
        ),
        "expected_launch_contract": launch.get("expected_grid") == EXPECTED_GRID
        and launch.get("expected_block") == EXPECTED_BLOCK
        and launch.get("expected_final_replay_kernel") == "MoEDynamicKernel",
    }


def require_all(checks: dict[str, bool], *, label: str) -> None:
    if not all(checks.values()):
        raise RuntimeError(f"{label} gate failed: {checks}")


def unique_cubin(preparation: dict[str, Any], jit_root: Path) -> tuple[Path, str]:
    artifacts = [
        item
        for item in preparation.get("jit_artifacts", [])
        if str(item.get("path", "")).endswith(".cubin")
    ]
    expected_hashes = preparation.get("cubin_sha256", [])
    if len(artifacts) != 1 or len(expected_hashes) != 1:
        raise RuntimeError(
            "expected exactly one preparation cubin: "
            f"artifacts={artifacts}, hashes={expected_hashes}"
        )
    artifact = artifacts[0]
    cubin = jit_root / str(artifact["path"])
    if not cubin.is_file():
        raise RuntimeError(f"preparation cubin is missing from candidate JIT: {cubin}")
    cubin_sha256 = sha256_file(cubin)
    checks = {
        "artifact_hash": cubin_sha256 == artifact.get("sha256"),
        "preparation_hash": cubin_sha256 == expected_hashes[0],
        "artifact_size": cubin.stat().st_size == artifact.get("size"),
    }
    require_all(checks, label="cubin identity")
    return cubin, cubin_sha256


def run_text(command: Sequence[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {shlex.join(command)}\n"
            f"{completed.stderr}"
        )
    return completed.stdout


def tool_version(executable: str) -> str:
    return run_text([executable, "--version"]).strip()


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    root = args.flashinfer_root.resolve()
    exp = (
        root / ".claude/w4a4_moe_bench/experiments/"
        "exp_009_intern_stage4_compact_lightcheck"
    )
    results = exp / "results"
    canonical = results / "canonical"
    preparation_path = canonical / "raw" / ARM / "m256/canonical/preparation.json"
    preparation = read_json(preparation_path)
    checks = correctness_checks(preparation)
    require_all(checks, label="candidate M256 correctness/preparation")

    overlay = results / "overlays/intern_stage4_compact/moe_dynamic_kernel.py"
    overlay_identity_path = overlay.with_name("identity.json")
    overlay_identity = read_json(overlay_identity_path)
    overlay_sha256 = sha256_file(overlay)
    runtime = preparation.get("runtime", {})
    source = runtime.get("source", {})
    gpu = runtime.get("gpu", {})
    identity_checks = {
        "overlay_identity_schema": overlay_identity.get("schema")
        == "exp009.intern-stage4-adapter-identity.v1",
        "overlay_hash": overlay_sha256
        == overlay_identity.get("adapter", {}).get("sha256")
        == source.get("overlay_sha256"),
        "gpu_uuid": gpu.get("uuid") == args.expected_gpu_uuid,
        "application_graphics_clock_mhz": int(
            gpu.get("applications_graphics_clock_mhz", -1)
        )
        == args.expected_app_clock_mhz,
    }
    require_all(identity_checks, label="preparation source/GPU identity")

    jit_root = args.jit_root.resolve()
    cubin, cubin_sha256 = unique_cubin(preparation, jit_root)
    resource_text = run_text([args.cuobjdump, "--dump-resource-usage", str(cubin)])
    kernel_symbol = parse_cubin_symbol(resource_text)

    output = results / "ncu/candidate_m256"
    report_base = output / "candidate_m256"
    worker = exp / "run_exp009_arm.py"
    target = [
        sys.executable,
        str(worker),
        "--flashinfer-root",
        str(root),
        "--results",
        str(canonical),
        "--arm",
        ARM,
        "--m",
        str(M),
        "--fixture",
        "canonical",
        "--overlay",
        str(overlay),
        "--jit-root",
        str(jit_root),
        "--expected-gpu-uuid",
        args.expected_gpu_uuid,
        "--comparison-anchor",
        "baseline_4warp",
        "--comparison-subject",
        ARM,
        "profile",
        "--warmup",
        str(args.warmup),
    ]
    selection = [item for section in SECTION_IDS for item in ("--section", section)]
    selection.extend(("--metrics", ",".join(LAUNCH_METRICS.values())))
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
        f"regex:^{re.escape(kernel_symbol)}$",
        "--kernel-name-base",
        "demangled",
        "--launch-count",
        "1",
        *selection,
        "--export",
        str(report_base),
        *target,
    ]
    return {
        "root": root,
        "exp": exp,
        "results": results,
        "canonical": canonical,
        "preparation_path": preparation_path,
        "preparation": preparation,
        "correctness_checks": checks,
        "identity_checks": identity_checks,
        "overlay": overlay,
        "overlay_identity_path": overlay_identity_path,
        "overlay_sha256": overlay_sha256,
        "jit_root": jit_root,
        "cubin": cubin,
        "cubin_sha256": cubin_sha256,
        "cuobjdump_resource_sha256": hashlib.sha256(resource_text.encode()).hexdigest(),
        "kernel_symbol": kernel_symbol,
        "output": output,
        "report_base": report_base,
        "command": command,
    }


def validate_profile_target(
    contract: dict[str, Any], args: argparse.Namespace
) -> tuple[Path, dict[str, bool]]:
    target_path = contract["canonical"] / "profile_targets" / ARM / "m256/target.json"
    target = read_json(target_path)
    runtime = target.get("runtime", {})
    checks = {
        "status_complete": target.get("status") == "complete",
        "candidate_arm": target.get("arm") == ARM,
        "m256": target.get("m") == M,
        "canonical_fixture": target.get("fixture_kind") == "canonical",
        "final_replay_nvtx": target.get("nvtx_range")
        == f"exp005_{ARM}_m256_final_replay",
        "expected_launch": target.get("expected_launch")
        == {
            "grid": EXPECTED_GRID,
            "block": EXPECTED_BLOCK,
            "kernel": "MoEDynamicKernel",
        },
        "jit_artifact_set": target.get("jit_artifact_set_sha256")
        == contract["preparation"].get("jit_artifact_set_sha256"),
        "overlay_hash": runtime.get("source", {}).get("overlay_sha256")
        == contract["overlay_sha256"],
        "gpu_uuid": runtime.get("gpu", {}).get("uuid") == args.expected_gpu_uuid,
        "application_graphics_clock_mhz": int(
            runtime.get("gpu", {}).get("applications_graphics_clock_mhz", -1)
        )
        == args.expected_app_clock_mhz,
    }
    require_all(checks, label="profile target identity")
    return target_path, checks


def capture(args: argparse.Namespace) -> None:
    contract = build_contract(args)
    output: Path = contract["output"]
    if output.exists():
        raise FileExistsError(f"immutable NCU capture already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    try:
        command_path = output / "command.txt"
        command_path.write_text(
            shlex.join(contract["command"]) + "\n", encoding="utf-8"
        )
        run_text(contract["command"])
        report = contract["report_base"].with_suffix(".ncu-rep")
        if not report.is_file() or report.stat().st_size == 0:
            raise RuntimeError("NCU produced no report")

        native_csv = output / "native_raw.csv"
        export_command = [
            args.ncu,
            "--import",
            str(report),
            "--csv",
            "--page",
            "raw",
            "--print-units",
            "base",
        ]
        native_csv.write_text(run_text(export_command), encoding="utf-8")
        metrics, observed = parse_native_raw(
            native_csv, expected_kernel_symbol=contract["kernel_symbol"]
        )
        if observed["grid"] != EXPECTED_GRID or observed["block"] != EXPECTED_BLOCK:
            raise RuntimeError(f"profiled launch geometry drift: {observed}")
        target_path, target_checks = validate_profile_target(contract, args)

        spill_values = {name: metrics[name]["value"] for name in SPILL_METRICS}
        dynamic_zero_spill = all(value == 0 for value in spill_values.values())
        evidence = {
            "schema": "exp009.candidate-m256-minimal-dynamic-spill.v1",
            "status": "complete",
            "arm": ARM,
            "m": M,
            "fixture": "canonical",
            "correctness_checks": contract["correctness_checks"],
            "identity_checks": {
                **contract["identity_checks"],
                **{
                    f"profile_target_{key}": value
                    for key, value in target_checks.items()
                },
                "exact_cubin_kernel_symbol": True,
                "single_profiled_graph_node": True,
                "observed_launch_geometry": True,
                "all_required_metrics_present_and_numeric": True,
            },
            "identity": {
                "preparation_path": str(contract["preparation_path"]),
                "preparation_sha256": sha256_file(contract["preparation_path"]),
                "overlay_path": str(contract["overlay"]),
                "overlay_sha256": contract["overlay_sha256"],
                "overlay_identity_path": str(contract["overlay_identity_path"]),
                "cubin_path": str(contract["cubin"]),
                "cubin_sha256": contract["cubin_sha256"],
                "kernel_symbol": contract["kernel_symbol"],
                "cuobjdump_resource_sha256": contract["cuobjdump_resource_sha256"],
                "profile_target_path": str(target_path),
                "profile_target_sha256": sha256_file(target_path),
                "gpu_uuid": args.expected_gpu_uuid,
                "expected_application_graphics_clock_mhz": (
                    args.expected_app_clock_mhz
                ),
                "observed_launch": observed,
            },
            "collection": {
                "section_ids": list(SECTION_IDS),
                "spill_metric_ids": list(SPILL_METRICS.values()),
                "explicit_launch_metric_ids": list(LAUNCH_METRICS.values()),
                "ncu_version": tool_version(args.ncu),
                "cuobjdump_version": tool_version(args.cuobjdump),
                "graph_profiling": "node",
                "launch_count": 1,
                "clock_control": "none",
            },
            "metrics": metrics,
            "dynamic_zero_spill": dynamic_zero_spill,
            "artifacts": {
                "command_sha256": sha256_file(command_path),
                "native_raw_sha256": sha256_file(native_csv),
                "ncu_report_sha256": sha256_file(report),
            },
            "evidence_boundary": (
                "Dynamic spill/refill counts and launch resources apply only to "
                "one correctness-gated candidate M256 CUDA Graph node."
            ),
        }
        require_all(evidence["identity_checks"], label="final capture identity")
        write_json(output / "evidence.json", evidence)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def dry_run(args: argparse.Namespace) -> None:
    contract = build_contract(args)
    if contract["output"].exists():
        raise FileExistsError(
            f"immutable NCU capture already exists: {contract['output']}"
        )
    payload = {
        "schema": "exp009.candidate-m256-minimal-dynamic-spill-plan.v1",
        "mode": "dry-run-no-GPU",
        "arm": ARM,
        "m": M,
        "correctness_checks": contract["correctness_checks"],
        "identity_checks": contract["identity_checks"],
        "overlay_sha256": contract["overlay_sha256"],
        "cubin_sha256": contract["cubin_sha256"],
        "kernel_symbol": contract["kernel_symbol"],
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "expected_grid": EXPECTED_GRID,
        "expected_block": EXPECTED_BLOCK,
        "section_ids": list(SECTION_IDS),
        "required_metric_ids": list(REQUIRED_METRICS.values()),
        "output": str(contract["output"]),
        "command_shell": shlex.join(contract["command"]),
        "gpu_or_ncu_invoked": False,
    }
    print(json.dumps(payload, sort_keys=True))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-app-clock-mhz", type=int, required=True)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        dry_run(args)
        return 0
    capture(args)
    print(json.dumps({"status": "complete", "arm": ARM, "m": M}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
