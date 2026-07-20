#!/usr/bin/env python3
"""Build the candidate-only exp_016 M8192 dynamic-spill evidence.

This module is intentionally CPU-only.  It accepts exactly one native NCU
``raw`` row for the correctness-validated Candidate CUDA Graph node.  Missing
or non-integral dynamic counters fail closed.  Static STACK/LOCAL declarations
are outside this evidence contract and can never satisfy this gate.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"

CANDIDATE = "candidate_token_major_reuse"
M = 8192
FIXTURE = "canonical"
SCALE_KIND = "unequal"
EXPECTED_SOURCE_SHA256 = (
    "ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19"
)
EXPECTED_GRID = [1, 1, 110]
EXPECTED_BLOCK = [288, 1, 1]

# These are executed counters.  In particular, neither the cubin's static
# STACK/LOCAL declaration nor source annotations are accepted as a substitute.
METRIC_IDS = {
    "spill_register_read_instructions": (
        "sass__inst_executed_register_spilling_op_read"
    ),
    "spill_register_write_instructions": (
        "sass__inst_executed_register_spilling_op_write"
    ),
    "spill_local_load_bytes": (
        "sass__inst_executed_register_spilling_mem_local_op_read"
    ),
    "spill_local_store_bytes": (
        "sass__inst_executed_register_spilling_mem_local_op_write"
    ),
}
EXPECTED_UNITS = {
    "spill_register_read_instructions": "inst",
    "spill_register_write_instructions": "inst",
    "spill_local_load_bytes": "byte",
    "spill_local_store_bytes": "byte",
}
DYNAMIC_SPILL_METRICS = tuple(METRIC_IDS)


class EvidenceError(RuntimeError):
    """Malformed, missing, or identity-incompatible profiler evidence."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def file_sha256(path: Path) -> str:
    require(path.is_file(), f"missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"invalid JSON {path}: {error}") from error
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evidence_path(path: Path, results: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(results.resolve()))
    except ValueError:
        return str(resolved)


def parse_exact_nonnegative_int(raw: str, *, label: str) -> int:
    normalized = raw.strip().lower()
    require(
        normalized not in ("", "n/a", "na", "not available", "nan"),
        f"missing/N-A required dynamic metric: {label}",
    )
    try:
        value = float(raw.replace(",", ""))
    except ValueError as error:
        raise EvidenceError(f"non-numeric dynamic metric {label}: {raw!r}") from error
    require(math.isfinite(value), f"non-finite dynamic metric {label}: {raw!r}")
    require(value >= 0 and value.is_integer(), f"non-integral metric {label}: {raw!r}")
    return int(value)


def parse_dimension(raw: str, *, label: str) -> list[int]:
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as error:
        raise EvidenceError(f"invalid {label}: {raw!r}") from error
    require(isinstance(value, tuple) and len(value) == 3, f"invalid {label}: {raw!r}")
    result = []
    for index, item in enumerate(value):
        require(
            isinstance(item, int) and not isinstance(item, bool) and item > 0,
            f"invalid {label}[{index}]: {item!r}",
        )
        result.append(item)
    return result


def parse_native_raw(path: Path) -> tuple[dict[str, int], dict[str, Any]]:
    """Parse exactly one launch and the four dynamic spill/refill counters."""
    require(path.is_file(), f"missing native NCU CSV: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    header_indices = [index for index, row in enumerate(rows) if row and row[0] == "ID"]
    require(
        len(header_indices) == 1,
        f"expected exactly one NCU header, got {header_indices}: {path}",
    )
    index = header_indices[0]
    require(len(rows) >= index + 3, f"NCU raw export lacks units/data: {path}")
    header, units = rows[index : index + 2]
    require(len(header) == len(set(header)), f"duplicate NCU columns: {path}")
    value_rows = [row for row in rows[index + 2 :] if any(cell for cell in row)]
    require(
        len(value_rows) == 1,
        f"expected exactly one profiled launch, got {len(value_rows)}: {path}",
    )
    values = value_rows[0]
    require(
        len(header) == len(units) == len(values),
        f"NCU raw CSV width mismatch: {path}",
    )
    by_name = dict(zip(header, values, strict=True))
    by_unit = dict(zip(header, units, strict=True))

    metrics: dict[str, int] = {}
    for name, metric_id in METRIC_IDS.items():
        require(
            metric_id in by_name, f"required dynamic NCU metric absent: {metric_id}"
        )
        require(
            by_unit[metric_id] == EXPECTED_UNITS[name],
            f"unit drift for {metric_id}: {by_unit[metric_id]!r} != "
            f"{EXPECTED_UNITS[name]!r}",
        )
        metrics[name] = parse_exact_nonnegative_int(by_name[metric_id], label=metric_id)

    for column in ("ID", "Kernel Name", "Context", "Stream", "Device"):
        require(column in by_name, f"missing NCU identity column: {column}")
    kernel = by_name["Kernel Name"]
    require("MoEDynamicKernel" in kernel, f"unexpected profiled kernel: {kernel!r}")
    launch = {
        "row_id": parse_exact_nonnegative_int(by_name["ID"], label="ID"),
        "kernel_symbol": kernel,
        "context_id": parse_exact_nonnegative_int(by_name["Context"], label="Context"),
        "stream_id": parse_exact_nonnegative_int(by_name["Stream"], label="Stream"),
        "device_id": parse_exact_nonnegative_int(by_name["Device"], label="Device"),
        "block": parse_dimension(by_name.get("Block Size", ""), label="Block Size"),
        "grid": parse_dimension(by_name.get("Grid Size", ""), label="Grid Size"),
    }
    return metrics, launch


def stable_runtime_identity(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Project only stable compile/runtime identity across validation/capture."""
    return {
        key: runtime.get(key)
        for key in (
            "cuda_runtime",
            "image_id",
            "image_digest",
            "python_deps_sha256",
            "nvcc",
            "ptxas",
            "torch",
            "packages",
        )
    }


def validated_candidate_identity(
    results: Path, validation_path: Path
) -> dict[str, Any]:
    validation = read_json(validation_path)
    runtime = validation.get("runtime")
    require(isinstance(runtime, Mapping), "validation runtime is missing")
    source = runtime.get("source")
    gpu = runtime.get("gpu")
    require(isinstance(source, Mapping), "validation source identity is missing")
    require(isinstance(gpu, Mapping), "validation GPU identity is missing")
    cubins = validation.get("cubin_sha256")
    replays = validation.get("replays")
    specialization = validation.get("specialization")
    overlay = results / "overlays" / CANDIDATE / "moe_dynamic_kernel.py"

    checks = {
        "schema": validation.get("schema") == "exp016.validation-case.v1",
        "status": validation.get("status") == "complete",
        "gate": validation.get("gate_pass") is True,
        "arm": validation.get("arm") == CANDIDATE,
        "m": validation.get("m") == M,
        "fixture": validation.get("fixture") == FIXTURE,
        "scale_kind": validation.get("scale_kind") == SCALE_KIND,
        "replays": isinstance(replays, list)
        and bool(replays)
        and all(
            isinstance(item, Mapping) and item.get("gate_pass") is True
            for item in replays
        ),
        "specialization": isinstance(specialization, Mapping)
        and specialization.get("gate_pass") is True,
        "source": source.get("overlay_sha256") == EXPECTED_SOURCE_SHA256,
        "overlay": overlay.is_file() and file_sha256(overlay) == EXPECTED_SOURCE_SHA256,
        "one_cubin": isinstance(cubins, list)
        and len(cubins) == 1
        and is_sha256(cubins[0]),
        "artifact_set": is_sha256(validation.get("jit_artifact_set_sha256")),
        "gpu_uuid": isinstance(gpu.get("uuid"), str)
        and str(gpu.get("uuid")).startswith("GPU-"),
        "gpu_clock": str(gpu.get("applications_graphics_clock_mhz", "")).isdigit(),
        "toolchain": all(
            runtime.get(key) not in (None, "") for key in ("nvcc", "ptxas")
        ),
    }
    require(all(checks.values()), f"Candidate validation identity drift: {checks}")
    return {
        "path": validation_path.resolve(),
        "sha256": file_sha256(validation_path),
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "overlay": overlay.resolve(),
        "cubin_sha256": cubins[0],
        "jit_artifact_set_sha256": validation["jit_artifact_set_sha256"],
        "jit_root": str(runtime.get("jit_root", "")),
        "gpu_uuid": str(gpu["uuid"]),
        "application_graphics_clock_mhz": int(gpu["applications_graphics_clock_mhz"]),
        "runtime_identity": stable_runtime_identity(runtime),
    }


def capture_root(results: Path) -> Path:
    return results / "ncu" / CANDIDATE


def validate_capture(results: Path, validation_path: Path) -> dict[str, Any]:
    prerequisite = validated_candidate_identity(results, validation_path)
    root = capture_root(results)
    native = root / "native_raw.csv"
    report = root / "trace.ncu-rep"
    target_path = root / "profile_target.json"
    identity_path = root / "capture_identity.json"
    metrics, observed = parse_native_raw(native)
    target = read_json(target_path)
    identity = read_json(identity_path)

    identity_checks = {
        "schema": identity.get("schema") == "exp016.dynamic-spill-capture-identity.v1",
        "arm": identity.get("arm") == CANDIDATE,
        "m": identity.get("m") == M,
        "fixture": identity.get("fixture") == FIXTURE,
        "scale": identity.get("scale_kind") == SCALE_KIND,
        "source": identity.get("source_sha256") == prerequisite["source_sha256"],
        "cubin": identity.get("cubin_sha256") == prerequisite["cubin_sha256"],
        "artifacts": identity.get("jit_artifact_set_sha256")
        == prerequisite["jit_artifact_set_sha256"],
        "gpu": identity.get("gpu_uuid") == prerequisite["gpu_uuid"],
        "clock": identity.get("expected_application_graphics_clock_mhz")
        == prerequisite["application_graphics_clock_mhz"],
        "grid": identity.get("expected_grid") == EXPECTED_GRID,
        "block": identity.get("expected_block") == EXPECTED_BLOCK,
        "metric_ids": identity.get("required_metric_ids") == list(METRIC_IDS.values()),
        "observed": identity.get("observed_launch") == observed,
        "values": identity.get("metrics") == metrics,
        "report_hash": identity.get("trace_sha256") == file_sha256(report),
        "csv_hash": identity.get("native_raw_sha256") == file_sha256(native),
        "target_hash": identity.get("profile_target_sha256")
        == file_sha256(target_path),
        "validation_hash": identity.get("validation_sha256") == prerequisite["sha256"],
        "observed_grid": observed["grid"] == EXPECTED_GRID,
        "observed_block": observed["block"] == EXPECTED_BLOCK,
    }
    require(
        all(identity_checks.values()),
        f"dynamic capture identity drift: {identity_checks}",
    )

    runtime = target.get("runtime")
    require(isinstance(runtime, Mapping), "profile-target runtime is missing")
    target_checks = {
        "schema": target.get("schema") == "exp016.dynamic-spill-profile-target.v1",
        "status": target.get("status") == "complete",
        "arm": target.get("arm") == CANDIDATE,
        "m": target.get("m") == M,
        "fixture": target.get("fixture") == FIXTURE,
        "scale": target.get("scale_kind") == SCALE_KIND,
        "source": target.get("source_sha256") == prerequisite["source_sha256"],
        "cubin": target.get("cubin_sha256") == prerequisite["cubin_sha256"],
        "artifacts": target.get("jit_artifact_set_sha256")
        == prerequisite["jit_artifact_set_sha256"],
        "gpu": target.get("gpu_uuid") == prerequisite["gpu_uuid"],
        "launch": target.get("expected_launch")
        == {
            "grid": EXPECTED_GRID,
            "block": EXPECTED_BLOCK,
            "kernel": "MoEDynamicKernel",
        },
        "runtime": stable_runtime_identity(runtime) == prerequisite["runtime_identity"],
        "validation": target.get("validation_sha256") == prerequisite["sha256"],
    }
    require(
        all(target_checks.values()), f"profile-target identity drift: {target_checks}"
    )

    return {
        "arm": CANDIDATE,
        "m": M,
        "fixture": FIXTURE,
        "scale_kind": SCALE_KIND,
        "source_sha256": prerequisite["source_sha256"],
        "cubin_sha256": prerequisite["cubin_sha256"],
        "jit_artifact_set_sha256": prerequisite["jit_artifact_set_sha256"],
        "gpu_uuid": prerequisite["gpu_uuid"],
        "observed_launch": observed,
        "metrics": metrics,
        "artifacts": {
            "capture_identity": evidence_path(identity_path, results),
            "profile_target": evidence_path(target_path, results),
            "ncu_report": evidence_path(report, results),
            "native_raw": evidence_path(native, results),
            "validation": evidence_path(validation_path, results),
        },
    }


def build_evidence(results: Path, validation_path: Path) -> dict[str, Any]:
    record = validate_capture(results, validation_path)
    metrics = record["metrics"]
    zero_by_metric = {name: metrics[name] == 0 for name in DYNAMIC_SPILL_METRICS}
    zero_dynamic_spill = all(zero_by_metric.values())
    return {
        "schema": "exp016.dynamic-spill-evidence.v1",
        "status": "pass" if zero_dynamic_spill else "reject",
        "scope": {
            "arm": CANDIDATE,
            "m": M,
            "fixture": FIXTURE,
            "scale_kind": SCALE_KIND,
            "execution": "one correctness-validated CUDA Graph replay node",
            "grid": EXPECTED_GRID,
            "block": EXPECTED_BLOCK,
        },
        "candidate": record,
        "checks": {
            "zero_by_dynamic_metric": zero_by_metric,
            "zero_dynamic_spill": zero_dynamic_spill,
            "dynamic_local_load_store_bytes": (
                metrics["spill_local_load_bytes"] + metrics["spill_local_store_bytes"]
            ),
        },
        "gate_pass": zero_dynamic_spill,
        "evidence_boundary": (
            "This gate uses executed NCU spill/refill instruction and local-memory byte "
            "counters from one M8192 Candidate graph node. Static STACK/LOCAL allocation "
            "or compiler annotations cannot substitute for these dynamic counters."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--validation",
        type=Path,
        required=True,
        help="completed exp016.validation-case.v1 Candidate M8192 case",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = args.results.resolve()
    output = (
        args.output.resolve()
        if args.output
        else results / "dynamic_spill_evidence.json"
    )
    value = build_evidence(results, args.validation.resolve())
    write_json(output, value)
    print(json.dumps(value, sort_keys=True))
    return 0 if value["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
