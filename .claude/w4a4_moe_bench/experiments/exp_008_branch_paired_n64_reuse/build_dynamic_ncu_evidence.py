#!/usr/bin/env python3
"""Validate and summarize exp_008 native-NCU and VeloQ API evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ARMS = ("n128", "v0", "v1")
HARNESS_ARM = {
    "n128": "candidate_8warp_serial_v0",
    "v0": "candidate_8warp_n64_temporal_replay_v0",
    "v1": "candidate_8warp_n64_temporal_replay_v0",
}
SPILL_METRICS = {
    "spill_refill_instructions": "sass__inst_executed_register_spilling_op_read",
    "spill_store_instructions": "sass__inst_executed_register_spilling_op_write",
    "spill_refill_bytes": "sass__inst_executed_register_spilling_mem_local_op_read",
    "spill_store_bytes": "sass__inst_executed_register_spilling_mem_local_op_write",
}
CUSTOM_METRICS = {
    "tensor_instructions": "sm__inst_executed_pipe_tensor_subpipe_hmma.sum",
    "fp4_tensor_ops": "sm__ops_path_tensor_src_fp4_dst_fp32.sum",
    "registers_per_thread": "launch__registers_per_thread",
    "shared_mem_per_block_bytes": "launch__shared_mem_per_block",
}
ALL_REQUIRED_METRICS = {**SPILL_METRICS, **CUSTOM_METRICS}
VELOQ_FILES = (
    "info.json",
    "summary.json",
    "launches.json",
    "inspect_launch0.json",
    "metrics_spill_refill_instructions.json",
    "metrics_spill_store_instructions.json",
    "metrics_spill_refill_bytes.json",
    "metrics_spill_store_bytes.json",
    "metrics_tensor_instructions.json",
    "metrics_fp4_tensor_ops.json",
    "metrics_launch_registers.json",
    "metrics_launch_shared.json",
    "disasm_launch0.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def evidence_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def numeric(value: Any, *, label: str) -> float:
    if value is None or str(value).strip().lower() in (
        "",
        "n/a",
        "na",
        "not available",
    ):
        raise ValueError(f"missing/N-A required value: {label}")
    parsed = float(str(value).replace(",", ""))
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite required value: {label}={value!r}")
    return parsed


def preparation_path(arm: str) -> Path:
    return (
        RESULTS
        / "canonical"
        / arm
        / "raw"
        / HARNESS_ARM[arm]
        / "m8192/canonical/preparation.json"
    )


def capture_root(arm: str) -> Path:
    return RESULTS / "ncu" / arm / "m8192/canonical_v0"


def validate_veloq_envelope(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema") != "v1" or "error" in payload or "data" not in payload:
        raise ValueError(f"invalid VeloQ API response: {path}")
    return payload


def veloq_metric_values(
    payload: dict[str, Any], *, expected_metric_ids: set[str]
) -> dict[str, float]:
    data = payload.get("data", {})
    if data.get("format") != "long":
        raise ValueError(f"expected VeloQ long metric format, got {data.get('format')}")
    rows = data.get("rows", [])
    values: dict[str, float] = {}
    for row in rows:
        if row.get("launch_row_id") != "launch:0":
            raise ValueError(f"unexpected VeloQ launch row: {row}")
        name = str(row.get("counter_name", ""))
        if name in values:
            raise ValueError(f"duplicate VeloQ counter: {name}")
        values[name] = numeric(row.get("value"), label=name)
    if set(values) != expected_metric_ids:
        raise ValueError(
            f"VeloQ metric identity mismatch: {set(values)} != {expected_metric_ids}"
        )
    if int(data.get("total_matched", -1)) != len(expected_metric_ids):
        raise ValueError("VeloQ total_matched does not match required metrics")
    return values


def validate_prerequisites() -> dict[str, Any]:
    static_path = RESULTS / "static_spill_evidence.json"
    static = read_json(static_path)
    if not static.get("gate_pass"):
        raise RuntimeError("static spill evidence gate is not PASS")
    arms: dict[str, Any] = {}
    for arm in ARMS:
        prep_path = preparation_path(arm)
        preparation = read_json(prep_path)
        static_case = static.get("cases", {}).get(f"{arm}_m8192")
        if not isinstance(static_case, dict) or not static_case.get("evidence_gate"):
            raise RuntimeError(f"missing static M8192 case for {arm}")
        source = preparation.get("runtime", {}).get("source", {})
        gpu = preparation.get("runtime", {}).get("gpu", {})
        if not (
            preparation.get("status") == "complete"
            and preparation.get("arm") == HARNESS_ARM[arm]
            and preparation.get("m") == 8192
            and preparation.get("fixture_kind") == "canonical"
            and source.get("overlay_sha256")
            == static_case.get("identity", {}).get("source_sha256")
            and preparation.get("cubin_sha256")
            == [static_case.get("identity", {}).get("cubin_sha256")]
        ):
            raise RuntimeError(f"preparation/static identity drift for {arm}")
        arms[arm] = {
            "preparation_path": evidence_path(prep_path),
            "preparation_sha256": sha256_file(prep_path),
            "source_sha256": source["overlay_sha256"],
            "cubin_sha256": preparation["cubin_sha256"][0],
            "kernel_symbol": static_case["identity"]["kernel_symbol"],
            "registers_per_thread": static_case["resource"][
                "registers_per_thread"
            ],
            "total_shared_bytes_per_cta": static_case["resource"][
                "total_shared_bytes_per_cta"
            ],
            "gpu_uuid": gpu.get("uuid"),
            "application_graphics_clock_mhz": int(
                gpu.get("applications_graphics_clock_mhz", -1)
            ),
            "expected_capture_root": evidence_path(capture_root(arm)),
            "expected_veloq_api_files": [
                evidence_path(capture_root(arm) / "veloq_api" / name)
                for name in VELOQ_FILES
            ],
        }
    return {
        "static_evidence_path": evidence_path(static_path),
        "static_evidence_sha256": sha256_file(static_path),
        "arms": arms,
    }


def analyze_arm(arm: str, prerequisite: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    root = capture_root(arm)
    report = root / "trace.ncu-rep"
    native_csv = root / "native_raw.csv"
    dynamic_path = root / "dynamic_ncu.json"
    identity_path = root / "capture_identity.json"
    identity = read_json(identity_path)
    dynamic = read_json(dynamic_path)
    api = {
        name: validate_veloq_envelope(root / "veloq_api" / name)
        for name in VELOQ_FILES
    }

    metrics = dynamic.get("metrics", {})
    if set(metrics) != set(ALL_REQUIRED_METRICS):
        raise ValueError(f"native required metric set drift for {arm}: {set(metrics)}")
    native_values = {
        label: numeric(value.get("value"), label=f"{arm}:{label}")
        for label, value in metrics.items()
    }
    metric_id_checks = {
        label: metrics[label].get("metric_id") == metric_id
        for label, metric_id in ALL_REQUIRED_METRICS.items()
    }
    observed = dynamic.get("observed_launch", {})
    native_checks = {
        "schema": dynamic.get("schema") == "exp008.dynamic-ncu.v1",
        "arm": dynamic.get("arm") == arm,
        "internal_arm": dynamic.get("internal_arm") == HARNESS_ARM[arm],
        "m8192": dynamic.get("m") == 8192,
        "canonical_fixture": dynamic.get("fixture") == "canonical",
        "metric_ids": all(metric_id_checks.values()),
        "kernel_symbol": observed.get("kernel_symbol")
        == prerequisite["kernel_symbol"],
        "grid": observed.get("grid") == [1, 1, 110],
        "block": observed.get("block") == [288, 1, 1],
        "registers_match_static": native_values["registers_per_thread"]
        == float(prerequisite["registers_per_thread"]),
        "shared_matches_static": native_values["shared_mem_per_block_bytes"]
        == float(prerequisite["total_shared_bytes_per_cta"]),
        "profile_target_checks": all(
            bool(value) for value in dynamic.get("profile_target_checks", {}).values()
        ),
        "nonspill_capture_gates": all(
            bool(value)
            for name, value in dynamic.get("gates", {}).items()
            if name != "dynamic_zero_spill"
        ),
    }

    identity_checks = {
        "schema": identity.get("schema") == "exp008.ncu-capture-identity.v1",
        "arm": identity.get("arm") == arm,
        "internal_arm": identity.get("internal_arm") == HARNESS_ARM[arm],
        "overlay_sha256": identity.get("overlay_sha256")
        == prerequisite["source_sha256"],
        "cubin_sha256": identity.get("cubin_sha256")
        == prerequisite["cubin_sha256"],
        "kernel_symbol": identity.get("expected_kernel_symbol")
        == prerequisite["kernel_symbol"],
        "gpu_uuid": identity.get("expected_gpu_uuid") == prerequisite["gpu_uuid"],
        "clock": identity.get("expected_application_graphics_clock_mhz")
        == prerequisite["application_graphics_clock_mhz"],
        "clock_control_none": identity.get("clock_control") == "none",
        "report_sha256": identity.get("trace_sha256") == sha256_file(report),
        "native_csv_sha256": identity.get("native_raw_sha256")
        == sha256_file(native_csv),
        "required_metric_ids": set(identity.get("all_required_metric_ids", []))
        == set(ALL_REQUIRED_METRICS.values()),
        "profile_target_checks": all(
            bool(value) for value in identity.get("profile_target_checks", {}).values()
        ),
    }

    launches = api["launches.json"].get("data", {}).get("rows", [])
    if len(launches) != 1:
        raise ValueError(f"VeloQ expected exactly one launch for {arm}")
    launch = launches[0]
    disasm_cubin = (
        api["disasm_launch0.json"].get("data", {}).get("auxiliary", {}).get("cubin_sha")
    )
    veloq_checks = {
        "launch_row_id": launch.get("row_id") == "launch:0"
        and launch.get("key") == "launch:0",
        "kernel_symbol": launch.get("kernel_demangled")
        == prerequisite["kernel_symbol"],
        "grid": launch.get("grid_size") == [1, 1, 110],
        "block": launch.get("block_size") == [288, 1, 1],
        "disasm_cubin_sha256": disasm_cubin == prerequisite["cubin_sha256"],
    }
    veloq_values: dict[str, float] = {}
    metric_files = (
        (
            "metrics_spill_refill_instructions.json",
            {SPILL_METRICS["spill_refill_instructions"]},
        ),
        (
            "metrics_spill_store_instructions.json",
            {SPILL_METRICS["spill_store_instructions"]},
        ),
        (
            "metrics_spill_refill_bytes.json",
            {SPILL_METRICS["spill_refill_bytes"]},
        ),
        (
            "metrics_spill_store_bytes.json",
            {SPILL_METRICS["spill_store_bytes"]},
        ),
        (
            "metrics_tensor_instructions.json",
            {CUSTOM_METRICS["tensor_instructions"]},
        ),
        (
            "metrics_fp4_tensor_ops.json",
            {CUSTOM_METRICS["fp4_tensor_ops"]},
        ),
        (
            "metrics_launch_registers.json",
            {CUSTOM_METRICS["registers_per_thread"]},
        ),
        (
            "metrics_launch_shared.json",
            {CUSTOM_METRICS["shared_mem_per_block_bytes"]},
        ),
    )
    for name, expected in metric_files:
        veloq_values.update(
            veloq_metric_values(api[name], expected_metric_ids=expected)
        )
    native_by_metric_id = {
        metrics[label]["metric_id"]: native_values[label] for label in metrics
    }
    veloq_checks["metric_values_match_native"] = veloq_values == native_by_metric_id

    gate = all(native_checks.values()) and all(identity_checks.values()) and all(
        veloq_checks.values()
    )
    spill_values = {label: native_values[label] for label in SPILL_METRICS}
    payload = {
        "arm": arm,
        "paths": {
            "report": evidence_path(report),
            "native_csv": evidence_path(native_csv),
            "dynamic_ncu": evidence_path(dynamic_path),
            "capture_identity": evidence_path(identity_path),
            "veloq_api": evidence_path(root / "veloq_api"),
        },
        "source_sha256": prerequisite["source_sha256"],
        "cubin_sha256": prerequisite["cubin_sha256"],
        "kernel_symbol": prerequisite["kernel_symbol"],
        "gpu_uuid": prerequisite["gpu_uuid"],
        "application_graphics_clock_mhz": prerequisite[
            "application_graphics_clock_mhz"
        ],
        "metrics": metrics,
        "spill_values": spill_values,
        "dynamic_zero_spill": all(value == 0 for value in spill_values.values()),
        "observed_launch": observed,
        "native_checks": native_checks,
        "capture_identity_checks": identity_checks,
        "veloq_api_checks": veloq_checks,
        "gate_pass": gate,
    }
    row = {
        "arm": arm,
        "source_sha256": prerequisite["source_sha256"],
        "cubin_sha256": prerequisite["cubin_sha256"],
        "gpu_uuid": prerequisite["gpu_uuid"],
        "application_graphics_clock_mhz": prerequisite[
            "application_graphics_clock_mhz"
        ],
        **native_values,
        "grid": json.dumps(observed.get("grid")),
        "block": json.dumps(observed.get("block")),
        "dynamic_zero_spill": payload["dynamic_zero_spill"],
        "gate_pass": gate,
    }
    return payload, row


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=RESULTS / "dynamic_ncu_evidence.json"
    )
    parser.add_argument(
        "--csv", type=Path, default=RESULTS / "dynamic_ncu_summary.csv"
    )
    args = parser.parse_args(argv)

    prerequisites = validate_prerequisites()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema": "exp008.dynamic-ncu-summary-plan.v1",
                    "mode": "dry-run-no-gpu",
                    "required_metric_ids": list(ALL_REQUIRED_METRICS.values()),
                    "veloq_api_only": True,
                    "veloq_raw_sidecar_parsed": False,
                    **prerequisites,
                },
                sort_keys=True,
            )
        )
        return 0

    cases: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        payload, row = analyze_arm(arm, prerequisites["arms"][arm])
        cases[arm] = payload
        rows.append(row)
    tensor_equal = len(
        {
            cases[arm]["metrics"]["tensor_instructions"]["value"] for arm in ARMS
        }
    ) == 1
    fp4_equal = len(
        {cases[arm]["metrics"]["fp4_tensor_ops"]["value"] for arm in ARMS}
    ) == 1
    cross_checks = {
        "all_arm_evidence_gates": all(cases[arm]["gate_pass"] for arm in ARMS),
        "all_arms_same_executed_tensor_instructions": tensor_equal,
        "all_arms_same_fp4_tensor_ops": fp4_equal,
        "n128_dynamic_spill_observed": not cases["n128"]["dynamic_zero_spill"],
        "v0_dynamic_zero_spill": cases["v0"]["dynamic_zero_spill"],
        "v1_dynamic_zero_spill": cases["v1"]["dynamic_zero_spill"],
    }
    gate = all(cross_checks.values())
    output = {
        "schema": "exp008.dynamic-ncu-evidence.v1",
        "scope": {
            "arms": list(ARMS),
            "m": 8192,
            "fixture": "canonical",
            "profiled_launches_per_arm": 1,
            "veloq_interface": "supported CLI JSON envelopes only",
            "veloq_raw_sidecar_parsed": False,
        },
        "required_metric_ids": ALL_REQUIRED_METRICS,
        "prerequisites": prerequisites,
        "cases": cases,
        "cross_arm_checks": cross_checks,
        "gate_pass": gate,
        "evidence_boundary": (
            "NCU dynamic metrics describe one final graph-replay MoEDynamicKernel "
            "per arm. They do not establish end-to-end latency or run-to-run variance."
        ),
    }
    args.output.resolve().write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    write_summary(rows, args.csv.resolve())
    print(json.dumps({"arm_count": len(cases), "gate_pass": gate}, sort_keys=True))
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
