#!/usr/bin/env python3
"""Parse and gate the matched M8192 NCU captures for exp_015.

This is deliberately GPU-free.  Each arm contributes exactly one native NCU
``raw`` CSV row for the final, uninstrumented CUDA Graph replay node.  Missing
metrics, ``n/a`` values, non-integral counters, identity drift, or incomplete
arm coverage fail closed.
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
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"

BASELINE = "baseline"
CANDIDATE = "candidate_v2"
ARMS = (BASELINE, CANDIDATE)
M = 8192
FIXTURE = "canonical"

EXPECTED_GPU_UUID = "GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522"
EXPECTED_APPLICATION_CLOCK_MHZ = 2377
EXPECTED_GRID = [1, 1, 110]
EXPECTED_BLOCK = [288, 1, 1]
EXPECTED_SOURCE_SHA256 = {
    BASELINE: "f3c246817679d962a3f7160dbe8b9e68262c919e26e306f349200961fc4ac971",
    CANDIDATE: "b6e141179794561f2144bdec079b7e109fddfccc2db6ba0f19c22d30ea4b34ca",
}
EXPECTED_CUBIN_SHA256 = {
    BASELINE: "4b835aa8ce91a4dd12b4dc4f43508c205c117aaeb193995fff57dd3ddbeb7725",
    CANDIDATE: "fee96b35d9b2c83e354504774fba2e2bc10e54f0316ade18f8adbdabb2ecbada",
}
EXPECTED_JIT_ARTIFACT_SET_SHA256 = {
    BASELINE: "286ee3e50518e70a343758174c6aa95db76038b5bdc0aa529f2208e45cc9ef3b",
    CANDIDATE: "4358e71894a2602030e32e33b13298a3b739ff4c554aaead82b4ef9c0373d3cc",
}

# exp_008 accepted M8192 canonical ledger.  Pairwise equality alone is not
# enough: both arms must retain this independently established work identity.
EXPECTED_WORK = {
    "executed_tensor_instructions": 31_162_368,
    "fp4_tensor_ops": 510_564_237_312,
}

METRIC_IDS = {
    "executed_tensor_instructions": ("sm__inst_executed_pipe_tensor_subpipe_hmma.sum"),
    "fp4_tensor_ops": "sm__ops_path_tensor_src_fp4_dst_fp32.sum",
    "spill_register_read": "sass__inst_executed_register_spilling_op_read",
    "spill_register_write": "sass__inst_executed_register_spilling_op_write",
    "spill_local_read": ("sass__inst_executed_register_spilling_mem_local_op_read"),
    "spill_local_write": ("sass__inst_executed_register_spilling_mem_local_op_write"),
}

EXPECTED_UNITS = {
    "executed_tensor_instructions": "inst",
    "fp4_tensor_ops": "",
    "spill_register_read": "inst",
    "spill_register_write": "inst",
    "spill_local_read": "byte",
    "spill_local_write": "byte",
}

SPILL_METRICS = (
    "spill_register_read",
    "spill_register_write",
    "spill_local_read",
    "spill_local_write",
)
WORK_METRICS = tuple(EXPECTED_WORK)


class EvidenceError(RuntimeError):
    """Malformed, missing, or identity-incompatible profiler evidence."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


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
        f"missing/N-A required metric: {label}",
    )
    try:
        value = float(raw.replace(",", ""))
    except ValueError as error:
        raise EvidenceError(f"non-numeric required metric {label}: {raw!r}") from error
    require(math.isfinite(value), f"non-finite required metric {label}: {raw!r}")
    require(value >= 0 and value.is_integer(), f"non-integral metric {label}: {raw!r}")
    return int(value)


def parse_dimension(raw: str, *, label: str) -> list[int]:
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as error:
        raise EvidenceError(f"invalid {label}: {raw!r}") from error
    require(
        isinstance(value, tuple) and len(value) == 3,
        f"invalid {label}: {raw!r}",
    )
    result = []
    for index, item in enumerate(value):
        require(
            isinstance(item, int) and not isinstance(item, bool) and item > 0,
            f"invalid {label}[{index}]: {item!r}",
        )
        result.append(item)
    return result


def parse_native_raw(path: Path) -> tuple[dict[str, int], dict[str, Any]]:
    """Parse exactly one NCU raw-page launch and the six registered metrics."""

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
        require(metric_id in by_name, f"required NCU metric absent: {metric_id}")
        require(
            by_unit[metric_id] == EXPECTED_UNITS[name],
            f"unit drift for {metric_id}: {by_unit[metric_id]!r} != "
            f"{EXPECTED_UNITS[name]!r}",
        )
        metrics[name] = parse_exact_nonnegative_int(by_name[metric_id], label=metric_id)

    identity_columns = ("ID", "Kernel Name", "Context", "Stream", "Device")
    for column in identity_columns:
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


def capture_root(results: Path, arm: str) -> Path:
    return results / "ncu" / arm


def validate_capture(results: Path, arm: str) -> dict[str, Any]:
    root = capture_root(results, arm)
    native_path = root / "native_raw.csv"
    report_path = root / "trace.ncu-rep"
    target_path = root / "profile_target.json"
    identity_path = root / "capture_identity.json"
    validation_path = results / "raw" / "validation" / arm / "validation.json"
    metrics, observed = parse_native_raw(native_path)
    identity = read_json(identity_path)
    target = read_json(target_path)
    validation = read_json(validation_path)

    expected_source = EXPECTED_SOURCE_SHA256[arm]
    expected_cubin = EXPECTED_CUBIN_SHA256[arm]
    expected_artifacts = EXPECTED_JIT_ARTIFACT_SET_SHA256[arm]
    overlay = results / "overlays" / arm / "moe_dynamic_kernel.py"

    require(
        identity.get("schema") == "exp015.matched-ncu-capture-identity.v1",
        f"{arm} capture identity schema drift",
    )
    require(identity.get("arm") == arm, f"{arm} capture arm drift")
    require(identity.get("m") == M, f"{arm} capture M drift")
    require(identity.get("fixture") == FIXTURE, f"{arm} capture fixture drift")
    require(
        identity.get("source_sha256") == expected_source,
        f"{arm} capture source hash drift",
    )
    require(
        identity.get("cubin_sha256") == expected_cubin,
        f"{arm} capture cubin hash drift",
    )
    require(
        identity.get("jit_artifact_set_sha256") == expected_artifacts,
        f"{arm} JIT artifact-set drift",
    )
    require(
        identity.get("gpu_uuid") == EXPECTED_GPU_UUID,
        f"{arm} GPU UUID drift",
    )
    require(
        identity.get("expected_application_graphics_clock_mhz")
        == EXPECTED_APPLICATION_CLOCK_MHZ,
        f"{arm} expected application clock drift",
    )
    require(
        identity.get("expected_grid") == EXPECTED_GRID, f"{arm} expected grid drift"
    )
    require(
        identity.get("expected_block") == EXPECTED_BLOCK,
        f"{arm} expected block drift",
    )
    require(
        identity.get("required_metric_ids") == list(METRIC_IDS.values()),
        f"{arm} required metric-ID drift",
    )
    require(
        identity.get("trace_sha256") == file_sha256(report_path),
        f"{arm} report hash drift",
    )
    require(
        identity.get("native_raw_sha256") == file_sha256(native_path),
        f"{arm} native CSV hash drift",
    )
    require(
        identity.get("profile_target_sha256") == file_sha256(target_path),
        f"{arm} profile-target hash drift",
    )
    require(
        identity.get("validation_manifest_sha256") == file_sha256(validation_path),
        f"{arm} validation-manifest hash drift",
    )
    require(file_sha256(overlay) == expected_source, f"{arm} frozen overlay drift")
    require(
        identity.get("observed_launch") == observed,
        f"{arm} observed launch/CSV mismatch",
    )
    require(identity.get("metrics") == metrics, f"{arm} metric/CSV mismatch")
    require(observed["grid"] == EXPECTED_GRID, f"{arm} grid drift")
    require(observed["block"] == EXPECTED_BLOCK, f"{arm} block drift")

    require(
        target.get("schema") == "exp015.matched-ncu-profile-target.v1",
        f"{arm} profile-target schema drift",
    )
    require(target.get("status") == "complete", f"{arm} profile target incomplete")
    require(target.get("arm") == arm, f"{arm} profile-target arm drift")
    require(target.get("m") == M, f"{arm} profile-target M drift")
    require(target.get("fixture_kind") == FIXTURE, f"{arm} target fixture drift")
    require(
        target.get("source_sha256") == expected_source,
        f"{arm} target source drift",
    )
    require(
        target.get("cubin_sha256") == expected_cubin,
        f"{arm} target cubin drift",
    )
    require(
        target.get("jit_artifact_set_sha256") == expected_artifacts,
        f"{arm} target artifact-set drift",
    )
    require(
        target.get("gpu_uuid") == EXPECTED_GPU_UUID,
        f"{arm} target GPU UUID drift",
    )
    require(
        target.get("expected_launch")
        == {
            "grid": EXPECTED_GRID,
            "block": EXPECTED_BLOCK,
            "kernel": "MoEDynamicKernel",
        },
        f"{arm} target launch contract drift",
    )
    target_runtime = target.get("runtime")
    require(isinstance(target_runtime, dict), f"{arm} target runtime is missing")
    target_gpu = target_runtime.get("gpu")
    target_source = target_runtime.get("source")
    require(isinstance(target_gpu, dict), f"{arm} target GPU identity is missing")
    require(isinstance(target_source, dict), f"{arm} target source identity is missing")
    require(target_gpu.get("uuid") == EXPECTED_GPU_UUID, f"{arm} runtime GPU drift")
    require(
        int(float(target_gpu.get("applications_graphics_clock_mhz", -1)))
        == EXPECTED_APPLICATION_CLOCK_MHZ,
        f"{arm} runtime application clock drift",
    )
    require(
        target_source.get("overlay_sha256") == expected_source,
        f"{arm} runtime source drift",
    )
    validation_runtime = validation.get("runtime")
    require(
        validation.get("schema") == "exp015.arm-validation.v1"
        and validation.get("status") == "complete"
        and validation.get("gate_pass") is True
        and validation.get("arm") == arm,
        f"{arm} validation manifest is incomplete or incompatible",
    )
    require(
        validation.get("cubin_sha256") == [expected_cubin],
        f"{arm} validation cubin drift",
    )
    require(
        validation.get("jit_artifact_set_sha256") == expected_artifacts,
        f"{arm} validation JIT artifact-set drift",
    )
    require(
        isinstance(validation_runtime, dict)
        and isinstance(validation_runtime.get("source"), dict)
        and validation_runtime["source"].get("overlay_sha256") == expected_source,
        f"{arm} validation source drift",
    )
    require(
        identity.get("kernel_symbol") == observed["kernel_symbol"],
        f"{arm} kernel symbol drift",
    )

    return {
        "arm": arm,
        "m": M,
        "fixture": FIXTURE,
        "source_sha256": expected_source,
        "cubin_sha256": expected_cubin,
        "jit_artifact_set_sha256": expected_artifacts,
        "gpu_uuid": EXPECTED_GPU_UUID,
        "observed_launch": observed,
        "metrics": metrics,
        "source_record": evidence_path(native_path, results),
        "artifacts": {
            "capture_identity": evidence_path(identity_path, results),
            "profile_target": evidence_path(target_path, results),
            "ncu_report": evidence_path(report_path, results),
            "native_raw": evidence_path(native_path, results),
        },
    }


def build_evidence(results: Path) -> dict[str, Any]:
    records = [validate_capture(results, arm) for arm in ARMS]
    by_arm = {record["arm"]: record for record in records}
    zero_spill = {
        arm: all(by_arm[arm]["metrics"][metric] == 0 for metric in SPILL_METRICS)
        for arm in ARMS
    }
    pairwise_work_identity = {
        metric: by_arm[BASELINE]["metrics"][metric]
        == by_arm[CANDIDATE]["metrics"][metric]
        for metric in WORK_METRICS
    }
    ledger_work_identity = {
        f"{arm}:{metric}": by_arm[arm]["metrics"][metric] == expected
        for arm in ARMS
        for metric, expected in EXPECTED_WORK.items()
    }
    gate = (
        all(zero_spill.values())
        and all(pairwise_work_identity.values())
        and all(ledger_work_identity.values())
    )
    return {
        "schema": "exp015.matched-dynamic-ncu-evidence.v1",
        "status": "pass" if gate else "reject",
        "scope": {
            "m": M,
            "fixture": FIXTURE,
            "execution": "one uninstrumented final CUDA Graph replay node per arm",
            "gpu_uuid": EXPECTED_GPU_UUID,
            "grid": EXPECTED_GRID,
            "block": EXPECTED_BLOCK,
            "metric_ids": METRIC_IDS,
            "expected_exp008_work_ledger": EXPECTED_WORK,
        },
        "records": records,
        "checks": {
            "zero_dynamic_spill": zero_spill,
            "pairwise_tensor_work_identity": pairwise_work_identity,
            "exp008_tensor_work_identity": ledger_work_identity,
        },
        "gate_pass": gate,
        "evidence_boundary": (
            "Dynamic counts cover one identity-locked M8192 canonical graph node "
            "per arm. They do not replace static cubin STACK/LOCAL/LDL/STL checks."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--output",
        type=Path,
        help="default: <results>/matched_dynamic_ncu_evidence.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = args.results.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else results / "matched_dynamic_ncu_evidence.json"
    )
    evidence = build_evidence(results)
    write_json(output, evidence)
    print(json.dumps(evidence, sort_keys=True))
    return 0 if evidence["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
