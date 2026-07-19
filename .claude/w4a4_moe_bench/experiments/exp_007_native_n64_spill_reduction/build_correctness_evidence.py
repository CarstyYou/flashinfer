#!/usr/bin/env python3
"""Build strict cross-arm correctness evidence for every exp_007 case."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parent
EXP005 = ROOT.parent / "exp_005_8warp_spill_reduction"
sys.path.insert(0, str(EXP005))

from exp005_common import evaluate_cross_arm_correctness  # noqa: E402


ANCHOR = "candidate_8warp_serial_v0"
CANDIDATE = "candidate_8warp_n64_temporal_replay_v0"
EXPECTED_SOURCE = {
    ANCHOR: "3cd9e6a26056d9221f59ea6749cd601c25cbef017cf6e7349efe0925180407c1",
    CANDIDATE: "1953cbb7717cda4461a4f199d05f370a4bdb35b4b8ef7556443caf36b0b12ec2",
}
EXPECTED_CUBIN = {
    ANCHOR: "b2bc3c4c229ebee967a6b0d3c5649bc06e3629d46793a19af845665f93683f17",
    CANDIDATE: "a9557634cf3d1bff59ca93739e75a1acd1187707222255fead78e2e6e8a73af9",
}


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    error = actual_f - expected_f
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.flatten(), expected_f.flatten(), dim=0
    )
    denominator = torch.linalg.vector_norm(expected_f, dim=1).clamp_min(1.0e-12)
    token_relative = torch.linalg.vector_norm(error, dim=1) / denominator
    return {
        "cosine_loss": max(0.0, 1.0 - float(cosine.item())),
        "relative_l2": float(
            (
                torch.linalg.vector_norm(error)
                / torch.linalg.vector_norm(expected_f).clamp_min(1.0e-12)
            ).item()
        ),
        "max_abs": float(error.abs().max().item()),
        "token_rel_l2_p99": float(torch.quantile(token_relative, 0.99).item()),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def case_path(root: Path, arm: str, m: int, fixture: str) -> Path:
    return root / "raw" / arm / f"m{m}" / fixture


def load_case(
    root: Path, arm: str, m: int, fixture: str
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    directory = case_path(root, arm, m, fixture)
    preparation = read_json(directory / "preparation.json")
    outputs = [
        torch.load(directory / f"output_{replay}.pt", map_location="cpu", weights_only=True)
        for replay in range(2)
    ]
    return preparation, outputs


def validate_preparation(
    preparation: dict[str, Any], *, arm: str, m: int, fixture: str
) -> dict[str, bool]:
    source = preparation.get("runtime", {}).get("source", {})
    checks = {
        "status": preparation.get("status") == "complete",
        "arm": preparation.get("arm") == arm,
        "m": preparation.get("m") == m,
        "harness_fixture": preparation.get("fixture_kind") == fixture,
        "source": source.get("overlay_sha256") == EXPECTED_SOURCE[arm],
        "cubin": preparation.get("cubin_sha256") == [EXPECTED_CUBIN[arm]],
        "independent_reference": all(
            bool(output.get("formal_pass")) for output in preparation.get("outputs", [])
        ),
        "route_task": all(
            bool(item.get("verification", {}).get("gate_pass"))
            for item in preparation.get("route_task_evidence", [])
        ),
    }
    fixture_kind = str(preparation.get("fixture", {}).get("fixture_kind", ""))
    if fixture_kind.startswith("branch_half_slice_canary_"):
        checks["canary_all_n64_blocks"] = all(
            bool(output.get("canary_all_blocks_pass"))
            for output in preparation.get("outputs", [])
        )
    return checks


def build_case(
    *, name: str, root: Path, m: int, fixture: str, required_for_final_gate: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    preparations: dict[str, dict[str, Any]] = {}
    outputs: dict[str, list[torch.Tensor]] = {}
    preparation_checks: dict[str, dict[str, bool]] = {}
    for arm in (ANCHOR, CANDIDATE):
        preparations[arm], outputs[arm] = load_case(root, arm, m, fixture)
        preparation_checks[arm] = validate_preparation(
            preparations[arm], arm=arm, m=m, fixture=fixture
        )

    identity_checks = {
        field: preparations[ANCHOR].get(field) == preparations[CANDIDATE].get(field)
        for field in ("fixture", "weights", "reference_sha256")
    }
    anchor_drift = tensor_error(outputs[ANCHOR][1], outputs[ANCHOR][0])
    candidate_drift = tensor_error(outputs[CANDIDATE][1], outputs[CANDIDATE][0])
    comparisons = [
        tensor_error(candidate, anchor)
        for candidate in outputs[CANDIDATE]
        for anchor in outputs[ANCHOR]
    ]
    candidate_worst = {
        metric: max(value[metric] for value in comparisons)
        for metric in anchor_drift
    }
    strict = evaluate_cross_arm_correctness(
        anchor_drift, candidate_drift, candidate_worst
    )
    gate = (
        all(all(checks.values()) for checks in preparation_checks.values())
        and all(identity_checks.values())
        and bool(strict["gate_pass"])
    )
    payload = {
        "name": name,
        "m": m,
        "harness_fixture": fixture,
        "semantic_fixture": preparations[ANCHOR]["fixture"]["fixture_kind"],
        "preparation_checks": preparation_checks,
        "cross_arm_identity_checks": identity_checks,
        "anchor_self_drift": anchor_drift,
        "candidate_self_drift": candidate_drift,
        "candidate_vs_anchor": comparisons,
        "strict_cross_arm_gate": strict,
        "required_for_final_gate": required_for_final_gate,
        "gate_pass": gate,
    }
    summary = {
        "case": name,
        "m": m,
        "semantic_fixture": payload["semantic_fixture"],
        "anchor_relative_l2_self_drift": anchor_drift["relative_l2"],
        "candidate_relative_l2_self_drift": candidate_drift["relative_l2"],
        "candidate_vs_anchor_worst_relative_l2": candidate_worst["relative_l2"],
        "candidate_vs_anchor_worst_max_abs": candidate_worst["max_abs"],
        "required_for_final_gate": required_for_final_gate,
        "gate_pass": gate,
    }
    return payload, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    results = args.results.resolve()
    cases = [
        ("canonical_m256", results / "canonical", 256, "canonical", True),
        ("canonical_m8192", results / "canonical", 8192, "canonical", True),
        *(
            (fixture, results / "canonical", 256, fixture, True)
            for fixture in ("sparse_empty", "exact_128", "tail_129", "hot_expert")
        ),
        # v0 is intentionally retained as a failed fixture-amplitude diagnostic.
        ("canary_up_v0", results / "canary/canary_up", 256, "canonical", False),
        ("canary_gate_v0", results / "canary/canary_gate", 256, "canonical", False),
        ("canary_up_v2", results / "canary/canary_up_v2", 256, "canonical", True),
        ("canary_gate_v2", results / "canary/canary_gate_v2", 256, "canonical", True),
    ]
    payload_cases: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name, root, m, fixture, required_for_final_gate in cases:
        payload, summary = build_case(
            name=name,
            root=root,
            m=m,
            fixture=fixture,
            required_for_final_gate=required_for_final_gate,
        )
        payload_cases[name] = payload
        rows.append(summary)
    gate = all(
        bool(value["gate_pass"])
        for value in payload_cases.values()
        if value["required_for_final_gate"]
    )
    output = {
        "schema": "exp007.correctness.v1",
        "cases": payload_cases,
        "gate_pass": gate,
        "known_failed_diagnostic_cases": [
            name
            for name, value in payload_cases.items()
            if not value["required_for_final_gate"] and not value["gate_pass"]
        ],
        "known_failed_fixture_attempts": [
            {
                "name": "canary_up_v1",
                "stage": "independent_reference_replay_0",
                "reason": (
                    "Changing w2_global_scale was not a valid final-output-only "
                    "amplitude control for this harness; the fixture was rejected "
                    "without changing thresholds."
                ),
            }
        ],
        "evidence_boundary": (
            "Route exact-once remains descriptor-multiset plus terminal-head inference; "
            "the worker does not carry a per-task consumed bitmap."
        ),
    }
    (results / "correctness_evidence.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    with (results / "correctness_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"gate_pass": gate, "case_count": len(rows)}, sort_keys=True))
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
