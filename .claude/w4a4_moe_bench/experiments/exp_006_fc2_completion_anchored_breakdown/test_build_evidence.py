from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_evidence import READER_PHASE_SPECS, build_phase_ownership_audit


def _fixture():
    durations = {
        "FC2_issue_path": 10,
        "FC2_completion_materialize_pre_sync": 20,
        "FC2_atomic_scatter_body": 30,
        "FC2_post_scatter_sync": 40,
    }
    marker_rows = [
        {
            "phase": phase,
            "aggregate_duration_ns": duration,
            "marker_cost_gate_pass": True,
            "replay_share_cv_gate_pass": True,
        }
        for phase, duration in durations.items()
    ]
    reader_rows = []
    for spec in READER_PHASE_SPECS:
        reader_rows.append(
            {
                "phase": spec["phase"],
                "boundary": spec["boundary"],
                "components": list(spec["components"]),
                "aggregate_duration_ns": sum(
                    durations[component] for component in spec["components"]
                ),
                "sm_equivalent_share_pct": sum(
                    durations[component] for component in spec["components"]
                )
                / 200
                * 100,
                "component_gates_pass": True,
            }
        )
    all_required_checks = {
        name: True
        for spec in READER_PHASE_SPECS
        for name in spec["required_sass_checks"]
    }
    proof = {
        "schema": "exp006.sass-boundary-proof.v2",
        "gate_pass": True,
        "records": [
            {"checks": dict(all_required_checks), "gate_pass": True} for _ in range(4)
        ],
    }
    aggregate = {
        "phase_totals_ns": durations,
        "sm_equivalent_denominator_ns": 200,
        "fc2_envelope_closure": {
            "additive_phase_sum_ns": 100,
            "intertile_residual_ns": 5,
            "envelope_sum_ns": 105,
            "pass": True,
        },
    }
    source = {
        "production": {"kernel": {"path": "kernel.py", "sha256": "source-hash"}},
        "overlays": {"kernel": {"path": "probe-kernel.py", "sha256": "overlay-hash"}},
    }
    static_hashes = {
        name: f"{name}-hash"
        for name in ("cubin", "ptx", "sass", "resource", "elf", "provenance")
    }
    timing_authority = {
        "sm_equivalent_denominator_ns": 200,
        "gate_pass": True,
    }
    return (
        reader_rows,
        marker_rows,
        aggregate,
        proof,
        source,
        static_hashes,
        timing_authority,
    )


def _audit(fixture):
    return build_phase_ownership_audit(*fixture)


def test_phase_ownership_audit_accepts_closed_data_ready_boundaries() -> None:
    audit = _audit(_fixture())
    assert audit["verdict"] == "PASS"
    assert all(phase["verdict"] == "PASS" for phase in audit["phases"])
    assert audit["closure"]["reader_additive_sum_ns"] == 100
    assert audit["closure"]["envelope_sum_ns"] == 105


def test_phase_ownership_audit_rejects_missing_r2s_boundary_proof() -> None:
    fixture = list(_fixture())
    proof = deepcopy(fixture[3])
    proof["records"][2]["checks"]["materialization_before_d"] = False
    fixture[3] = proof
    audit = _audit(tuple(fixture))
    assert audit["verdict"] == "REJECT"
    fc2 = next(phase for phase in audit["phases"] if phase["phase"] == "FC2_GEMM")
    assert fc2["verdict"] == "REJECT"
    assert not fc2["source_sass_coverage"]["required_sass_checks"][
        "materialization_before_d"
    ]


def test_phase_ownership_audit_rejects_gap_or_double_count() -> None:
    fixture = list(_fixture())
    reader_rows = deepcopy(fixture[0])
    reader_rows[0]["aggregate_duration_ns"] += 1
    fixture[0] = reader_rows
    audit = _audit(tuple(fixture))
    assert audit["verdict"] == "REJECT"
    assert not audit["checks"]["reader_additive_sum_closes"]


def test_phase_ownership_audit_rejects_unbound_probe_overlay() -> None:
    fixture = list(_fixture())
    source = deepcopy(fixture[4])
    source["overlays"]["kernel"].pop("sha256")
    fixture[4] = source
    audit = _audit(tuple(fixture))
    assert audit["verdict"] == "REJECT"
    assert not audit["checks"]["source_sass_coverage_pass"]


def test_phase_ownership_audit_rejects_unbound_share_denominator() -> None:
    fixture = list(_fixture())
    reader_rows = deepcopy(fixture[0])
    reader_rows[1]["sm_equivalent_share_pct"] = 99.0
    fixture[0] = reader_rows
    audit = _audit(tuple(fixture))
    assert audit["verdict"] == "REJECT"
    scatter = next(
        phase for phase in audit["phases"] if phase["phase"] == "FC2_atomic_scatter"
    )
    assert not scatter["checks"]["share_denominator_recomputes"]
