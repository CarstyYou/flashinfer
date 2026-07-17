from __future__ import annotations

import csv
import json

import pytest

from build_ncu_evidence import (
    CUSTOM_METRIC_IDS as BUILDER_CUSTOM_METRIC_IDS,
    EXPECTED_UNITS,
    METRICS,
    REQUIRED_SECTION_IDS,
    build_evidence,
    parse_arm_paths,
    parse_native_csv,
    parse_veloq_launch,
    validate_identity_bundle,
)
from capture_ncu import (
    CUSTOM_METRIC_IDS,
    REQUIRED_METRIC_IDS,
    SECTION_IDS,
    SPILL_METRIC_IDS,
    _validate_profile_target,
    metric_selection_args,
)
from exp005_common import (
    ALL_ARMS,
    BASELINE,
    CANDIDATE,
    CANONICAL_FIXTURE,
    EXPECTED_GRID,
    canonical_sha256,
    expected_block,
    file_sha256,
)


PREFERRED_UNIT = {
    name: sorted(units, key=lambda item: (item == "", len(item), item))[0]
    for name, units in EXPECTED_UNITS.items()
}
PREFERRED_UNIT["fp4_tensor_ops"] = ""
PREFERRED_UNIT["configured_stack_limit_bytes"] = ""
PREFERRED_UNIT["waves_per_sm"] = ""
PREFERRED_UNIT["duration_ns"] = "ns"


def _write_native_csv(path, *, extra_row=False, unit_override=None) -> None:
    identity = [
        "ID",
        "Kernel Name",
        "Context",
        "Stream",
        "Device",
        "launch__block_dim_x",
        "launch__block_dim_y",
        "launch__block_dim_z",
        "launch__grid_dim_x",
        "launch__grid_dim_y",
        "launch__grid_dim_z",
    ]
    header = [*identity, *METRICS.values()]
    units = ["" for _ in identity] + [
        (unit_override or {}).get(name, PREFERRED_UNIT[name]) for name in METRICS
    ]
    values = [
        "0",
        "prefix_MoEDynamicKernel_suffix",
        "1",
        "7",
        "0",
        "160",
        "1",
        "1",
        "1",
        "1",
        "110",
        *["1" for _ in METRICS],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["==PROF== Connected to process"])
        writer.writerow(header)
        writer.writerow(units)
        writer.writerow(values)
        if extra_row:
            writer.writerow(values)


def _metrics(spill: float) -> dict[str, float]:
    values = {name: 1.0 for name in METRICS}
    values.update(
        {
            "dynamic_spill_refill_instructions": spill,
            "dynamic_spill_store_instructions": spill,
            "dynamic_spill_refill_bytes": spill,
            "dynamic_spill_store_bytes": spill,
            "tensor_instructions": 100.0,
            "fp4_tensor_ops": 200.0,
            "warp_latency_per_issued_instruction": 10.0,
            "stall_wait_ratio": 2.0,
            "local_total_footprint_bytes": spill * 2.0,
            "dynamic_spill_total_bytes": spill * 2.0,
        }
    )
    return values


def _launch(arm: str) -> dict[str, object]:
    return {
        "key": "launch:0",
        "row_id": "launch:0",
        "kernel": "prefix_MoEDynamicKernel_suffix",
        "grid": list(EXPECTED_GRID),
        "block": list(expected_block(arm)),
        "context_id": 1,
        "stream_id": 7,
        "device_id": 0,
        "trace_path": "trace.ncu-rep",
    }


def _static(candidate_pass: bool) -> dict[str, dict[str, object]]:
    return {
        BASELINE: {
            "static_frame_bytes_per_thread": 488,
            "zero_spill_static_gate": False,
        },
        CANDIDATE: {
            "static_frame_bytes_per_thread": 224,
            "zero_spill_static_gate": candidate_pass,
        },
    }


def _build(*, spill: float, static_pass: bool = True):
    launches = {arm: _launch(arm) for arm in ALL_ARMS}
    return build_evidence(
        {BASELINE: _metrics(10), CANDIDATE: _metrics(spill)},
        launches,
        native_launches=launches,
        identity_checks={"all_inputs_locked": True},
        static_arms=_static(static_pass),
    )


def test_capture_and_builder_section_metric_contracts_are_identical() -> None:
    assert SECTION_IDS == REQUIRED_SECTION_IDS
    assert tuple(METRICS.values()) == REQUIRED_METRIC_IDS
    assert CUSTOM_METRIC_IDS == BUILDER_CUSTOM_METRIC_IDS
    assert not set(SPILL_METRIC_IDS).intersection(CUSTOM_METRIC_IDS)
    assert {"InstructionStats", "SourceCounters"}.issubset(SECTION_IDS)


def test_canonical_v1_collects_section_and_custom_metric_union() -> None:
    arguments = metric_selection_args()
    assert arguments.count("--section") == len(SECTION_IDS)
    for section in SECTION_IDS:
        index = arguments.index(section)
        assert arguments[index - 1] == "--section"
    metrics_index = arguments.index("--metrics")
    assert tuple(arguments[metrics_index + 1].split(",")) == CUSTOM_METRIC_IDS


def test_parse_native_csv_validates_units_launch_and_derives_totals(tmp_path) -> None:
    path = tmp_path / "raw.csv"
    _write_native_csv(path)
    parsed, units, launch = parse_native_csv(path)
    assert parsed["duration_ns"] == 1.0
    assert parsed["local_total_footprint_bytes"] == 2.0
    assert parsed["dynamic_spill_total_bytes"] == 2.0
    assert units["configured_stack_limit_bytes"] == ""
    assert launch["kernel"] == "prefix_MoEDynamicKernel_suffix"
    assert launch["grid"] == [1, 1, 110]
    assert launch["block"] == [160, 1, 1]


def test_parse_native_csv_rejects_multiple_launch_rows(tmp_path) -> None:
    path = tmp_path / "raw.csv"
    _write_native_csv(path, extra_row=True)
    with pytest.raises(ValueError, match="exactly one NCU launch"):
        parse_native_csv(path)


def test_v0_style_csv_without_section_derived_spill_metrics_fails_closed(
    tmp_path,
) -> None:
    path = tmp_path / "v0_raw.csv"
    _write_native_csv(path)
    rows = list(csv.reader(path.open()))
    header = rows[1]
    drop = {header.index(metric) for metric in SPILL_METRIC_IDS}
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows:
            if len(row) == len(header):
                writer.writerow(
                    [value for index, value in enumerate(row) if index not in drop]
                )
            else:
                writer.writerow(row)
    with pytest.raises(ValueError, match="missing NCU metric"):
        parse_native_csv(path)


def test_parse_native_csv_rejects_unit_drift(tmp_path) -> None:
    path = tmp_path / "raw.csv"
    _write_native_csv(path, unit_override={"duration_ns": "cycle"})
    with pytest.raises(ValueError, match="unit drift"):
        parse_native_csv(path)


def test_parse_arm_paths_rejects_duplicate_assignment(tmp_path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_arm_paths(
            [
                f"{BASELINE}={tmp_path / 'a'}",
                f"{BASELINE}={tmp_path / 'b'}",
                f"{CANDIDATE}={tmp_path / 'c'}",
            ]
        )


def test_veloq_launch_requires_v1_ncu_launches_envelope(tmp_path) -> None:
    path = tmp_path / "launches.json"
    payload = {
        "schema": "v1",
        "command": "ncu.inspect",
        "source": {"kind": "ncu", "version": "v1"},
        "trace": {"kind": "ncu", "path": "trace.ncu-rep"},
        "data": {"rows": []},
    }
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid VeloQ launches envelope"):
        parse_veloq_launch(path)


def test_static_frame_and_configured_stack_limit_are_not_conflated() -> None:
    payload = _build(spill=0, static_pass=False)
    boundary = payload["stack_evidence_boundary"][CANDIDATE]
    assert boundary["configured_runtime_stack_limit_bytes"] == 1.0
    assert boundary["cubin_static_frame_bytes_per_thread"] == 224
    assert boundary["directly_comparable"] is False
    assert payload["candidate_dynamic_zero_spill_pass"] is True
    assert payload["candidate_zero_spill_pass"] is False
    assert payload["overall_gate_pass"] is False


def test_dynamic_spill_fails_closed() -> None:
    payload = _build(spill=1, static_pass=True)
    assert payload["candidate_dynamic_zero_spill_pass"] is False
    assert payload["candidate_zero_spill_pass"] is False
    assert payload["overall_gate_pass"] is False


def test_stall_percentage_uses_total_warp_latency_denominator() -> None:
    payload = _build(spill=0, static_pass=True)
    assert payload["arms"][CANDIDATE]["stall_wait_pct"] == 20.0
    assert payload["overall_gate_pass"] is True


def test_non_integral_work_counter_fails_identity() -> None:
    launches = {arm: _launch(arm) for arm in ALL_ARMS}
    baseline = _metrics(1)
    candidate = _metrics(0)
    candidate["tensor_instructions"] = 100.25
    payload = build_evidence(
        {BASELINE: baseline, CANDIDATE: candidate},
        launches,
        native_launches=launches,
        identity_checks={"all_inputs_locked": True},
        static_arms=_static(True),
    )
    assert payload["work_identity_pass"] is False
    assert payload["overall_gate_pass"] is False


def test_profile_target_validation_rejects_wrong_jit_identity() -> None:
    target = {
        "schema": "exp005.profile-target.v1",
        "status": "complete",
        "arm": BASELINE,
        "m": 8192,
        "fixture_kind": CANONICAL_FIXTURE,
        "expected_launch": {
            "grid": [1, 1, 110],
            "block": list(expected_block(BASELINE)),
            "kernel": "MoEDynamicKernel",
        },
        "jit_artifact_set_sha256": "wrong",
    }
    with pytest.raises(RuntimeError, match="jit"):
        _validate_profile_target(
            target,
            arm=BASELINE,
            prerequisite={"jit_artifact_set_sha256": "expected"},
        )


def test_identity_bundle_binds_reports_cubins_and_work_contract(tmp_path) -> None:
    common_runtime = {
        "cuda_runtime": "13.2",
        "image_digest": "sha256:image",
        "nvcc": "nvcc 13.2",
        "ptxas": "ptxas 13.2",
        "python": "3.12",
        "python_deps_sha256": "deps",
        "torch": "2.12",
        "gpu": {
            "uuid": "GPU-test",
            "name": "NVIDIA Graphics Device",
            "compute_capability": [12, 0],
            "driver": "580",
            "sm_count": 110,
            "pci_bus_id": "0000:01:00.0",
        },
        "source": {
            "checkout_head": "head",
            "cutlass_commit": "cutlass",
            "locked_source_commit": "head",
            "production_kernel_sha256": "production",
        },
        "imports": {"cutlass_python_version": "4.6.0"},
    }
    reports = {}
    natives = {}
    preparations = {}
    targets = {}
    captures = {}
    launches = {}
    static_arms = {}
    for index, arm in enumerate(ALL_ARMS):
        report = tmp_path / f"{arm}.ncu-rep"
        report.write_bytes(f"report-{arm}".encode())
        native = tmp_path / f"{arm}.csv"
        native.write_text(f"native-{arm}")
        prep_path = tmp_path / f"{arm}.preparation.json"
        target_path = tmp_path / f"{arm}.target.json"
        cubin = f"cubin-{index}"
        jit = f"jit-{index}"
        prep = {
            "schema": "exp005.arm-preparation.v1",
            "status": "complete",
            "arm": arm,
            "m": 8192,
            "fixture_kind": CANONICAL_FIXTURE,
            "case": {"m": 8192, "topk": 8},
            "fixture": {"sha": "same-fixture"},
            "weights": {"sha": "same-weights"},
            "reference_sha256": "same-reference",
            "cubin_sha256": [cubin],
            "jit_artifact_set_sha256": jit,
            "runtime": common_runtime,
        }
        prep_path.write_text(json.dumps(prep))
        prep["_path"] = str(prep_path)
        target = {
            "schema": "exp005.profile-target.v1",
            "status": "complete",
            "arm": arm,
            "m": 8192,
            "fixture_kind": CANONICAL_FIXTURE,
            "expected_launch": {
                "grid": list(EXPECTED_GRID),
                "block": list(expected_block(arm)),
                "kernel": "MoEDynamicKernel",
            },
            "jit_artifact_set_sha256": jit,
            "runtime": common_runtime,
        }
        target_path.write_text(json.dumps(target))
        target["_path"] = str(target_path)
        launch = _launch(arm)
        launch["trace_path"] = str(report)
        capture = {
            "schema": "exp005.ncu-capture-identity.v3",
            "capture_revision": "canonical_v1",
            "arm": arm,
            "m": 8192,
            "fixture_kind": CANONICAL_FIXTURE,
            "expected_grid": list(EXPECTED_GRID),
            "expected_block": list(expected_block(arm)),
            "section_ids": list(SECTION_IDS),
            "custom_metric_ids": list(CUSTOM_METRIC_IDS),
            "required_metric_ids": list(METRICS.values()),
            "native_raw_sha256": file_sha256(native),
            "trace_sha256": file_sha256(report),
            "preparation_sha256": file_sha256(prep_path),
            "profile_target_sha256": file_sha256(target_path),
            "cubin_sha256": [cubin],
            "jit_artifact_set_sha256": jit,
            "ncu_version": "Nsight Compute 26.05",
            "collection_protocol": {"replay": "kernel"},
        }
        capture["identity_sha256"] = canonical_sha256(capture)
        reports[arm] = report
        natives[arm] = native
        preparations[arm] = prep
        targets[arm] = target
        captures[arm] = capture
        launches[arm] = launch
        static_arms[arm] = {
            "identity": {"cubin_sha256": cubin},
            "resource": {
                "stack_bytes_per_thread": 0 if arm == CANDIDATE else 488,
                "static_local_bytes_outside_stack": 0,
            },
            "compiler_spill_refill": {"annotation_count": 0},
            "gates": {"zero_spill_static_gate": arm == CANDIDATE},
        }
    static_payload = {
        "schema": "exp005.static-resource-spill-evidence.v1",
        "arms": static_arms,
    }
    static_payload["evidence_sha256"] = canonical_sha256(static_payload)
    checks, compact = validate_identity_bundle(
        capture_identities=captures,
        preparations=preparations,
        profile_targets=targets,
        native_paths=natives,
        report_paths=reports,
        launches=launches,
        native_launches=launches,
        static_payload=static_payload,
        correctness={
            "schema": "exp005.correctness.v1",
            "m": 8192,
            "gate_pass": True,
        },
    )
    assert all(checks.values())
    assert compact[CANDIDATE]["zero_spill_static_gate"] is True
