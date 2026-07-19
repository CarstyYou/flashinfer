#!/usr/bin/env python3
"""Build fail-closed v0-versus-v1 correctness evidence for exp_008.

This collector is GPU-free.  It compares the immutable exp_007 temporal-N64
baseline copied under ``results/baseline_v0`` with exp_008's branch-paired-N64
candidate.  The numerical gate deliberately reuses exp_005/exp_007's fixed
``evaluate_cross_arm_correctness`` policy without changing its thresholds.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EXP005 = ROOT.parent / "exp_005_8warp_spill_reduction"
sys.path.insert(0, str(EXP005))

from exp005_common import evaluate_cross_arm_correctness  # noqa: E402


HARNESS_ARM = "candidate_8warp_n64_temporal_replay_v0"
EXPECTED_SOURCE = {
    "v0": "1953cbb7717cda4461a4f199d05f370a4bdb35b4b8ef7556443caf36b0b12ec2",
    "v1": "f3c246817679d962a3f7160dbe8b9e68262c919e26e306f349200961fc4ac971",
}
EXPECTED_CUBIN = {
    "v0": "a9557634cf3d1bff59ca93739e75a1acd1187707222255fead78e2e6e8a73af9",
    "v1": "4b835aa8ce91a4dd12b4dc4f43508c205c117aaeb193995fff57dd3ddbeb7725",
}
OVERLAY = {
    "v0": RESULTS / "overlays/temporal_n64_v0/moe_dynamic_kernel.py",
    "v1": RESULTS / "overlays/branch_paired_n64_v1/moe_dynamic_kernel.py",
}


@dataclass(frozen=True)
class CaseSpec:
    name: str
    m: int
    harness_fixture: str
    semantic_fixture: str
    v0_directory: Path
    v1_directory: Path
    canary_branch: str | None = None


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def evidence_path(path: Path) -> str:
    """Return a portable experiment-relative source path when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise ValueError(f"tensor shape mismatch: {actual.shape} != {expected.shape}")
    if actual.dtype != expected.dtype:
        raise ValueError(f"tensor dtype mismatch: {actual.dtype} != {expected.dtype}")
    if not bool(torch.isfinite(actual).all()) or not bool(
        torch.isfinite(expected).all()
    ):
        raise ValueError("non-finite tensor in correctness evidence")

    actual_f = actual.float()
    expected_f = expected.float()
    error = actual_f - expected_f
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.flatten(), expected_f.flatten(), dim=0
    )
    denominator = torch.linalg.vector_norm(expected_f, dim=1).clamp_min(1.0e-12)
    token_relative = torch.linalg.vector_norm(error, dim=1) / denominator
    values = {
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
    if not all(math.isfinite(value) and value >= 0.0 for value in values.values()):
        raise ValueError(f"invalid tensor error values: {values}")
    return values


def load_outputs(directory: Path) -> tuple[list[torch.Tensor], dict[str, str]]:
    expected = [directory / f"output_{replay}.pt" for replay in range(2)]
    observed = sorted(directory.glob("output_*.pt"))
    if observed != expected:
        raise ValueError(
            f"expected exactly output_0.pt/output_1.pt in {directory}, got {observed}"
        )
    outputs = [
        torch.load(path, map_location="cpu", weights_only=True) for path in expected
    ]
    if not all(isinstance(value, torch.Tensor) for value in outputs):
        raise ValueError(f"non-tensor output in {directory}")
    return outputs, {path.name: sha256_file(path) for path in expected}


def validate_preparation(
    preparation: dict[str, Any],
    *,
    logical_arm: str,
    spec: CaseSpec,
) -> dict[str, bool]:
    source = preparation.get("runtime", {}).get("source", {})
    output_records = preparation.get("outputs", [])
    route_records = preparation.get("route_task_evidence", [])
    cubin_artifacts = [
        item
        for item in preparation.get("jit_artifacts", [])
        if str(item.get("path", "")).endswith(".cubin")
        and item.get("sha256") == EXPECTED_CUBIN[logical_arm]
    ]
    checks = {
        "status_complete": preparation.get("status") == "complete",
        "harness_arm": preparation.get("arm") == HARNESS_ARM,
        "m": preparation.get("m") == spec.m
        and preparation.get("case", {}).get("m") == spec.m,
        "harness_fixture": preparation.get("fixture_kind") == spec.harness_fixture,
        "semantic_fixture": preparation.get("fixture", {}).get("fixture_kind")
        == spec.semantic_fixture,
        "source_sha256": source.get("overlay_sha256") == EXPECTED_SOURCE[logical_arm],
        "cubin_sha256": preparation.get("cubin_sha256")
        == [EXPECTED_CUBIN[logical_arm]],
        "cubin_artifact_identity": len(cubin_artifacts) == 1,
        "two_independent_reference_replays": len(output_records) == 2
        and all(bool(output.get("formal_pass")) for output in output_records),
        "reference_record_identity": len(output_records) == 2
        and all(
            output.get("reference_sha256") == preparation.get("reference_sha256")
            for output in output_records
        ),
        "two_route_task_replays": len(route_records) == 2
        and all(
            bool(item.get("verification", {}).get("gate_pass"))
            for item in route_records
        ),
    }
    if spec.canary_branch is not None:
        checks.update(
            {
                "canary_revision_v2": preparation.get("fixture", {}).get(
                    "canary_revision"
                )
                == "v2_final_scatter_scale",
                "canary_branch": len(output_records) == 2
                and all(
                    output.get("canary_branch") == spec.canary_branch
                    for output in output_records
                ),
                "canary_all_n64_blocks": len(output_records) == 2
                and all(
                    bool(output.get("canary_all_blocks_pass"))
                    and len(output.get("canary_blocks", [])) == 8
                    and all(
                        bool(block.get("pass"))
                        for block in output.get("canary_blocks", [])
                    )
                    for output in output_records
                ),
            }
        )
    return checks


def arm_payload(
    *, logical_arm: str, directory: Path, spec: CaseSpec
) -> tuple[dict[str, Any], list[torch.Tensor], dict[str, bool]]:
    preparation_path = directory / "preparation.json"
    preparation = read_json(preparation_path)
    outputs, output_sha256 = load_outputs(directory)
    checks = validate_preparation(preparation, logical_arm=logical_arm, spec=spec)
    return (
        {
            "directory": evidence_path(directory),
            "preparation_path": evidence_path(preparation_path),
            "preparation_sha256": sha256_file(preparation_path),
            "output_sha256": output_sha256,
            "source_sha256": preparation.get("runtime", {})
            .get("source", {})
            .get("overlay_sha256"),
            "cubin_sha256": preparation.get("cubin_sha256"),
            "preparation_checks": checks,
            "preparation": preparation,
        },
        outputs,
        checks,
    )


def build_case(spec: CaseSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    v0, v0_outputs, v0_checks = arm_payload(
        logical_arm="v0", directory=spec.v0_directory, spec=spec
    )
    v1, v1_outputs, v1_checks = arm_payload(
        logical_arm="v1", directory=spec.v1_directory, spec=spec
    )
    p0 = v0["preparation"]
    p1 = v1["preparation"]

    identity = {}
    for field in ("case", "fixture", "weights"):
        value0 = p0.get(field)
        value1 = p1.get(field)
        identity[field] = {
            "v0_sha256": canonical_sha256(value0),
            "v1_sha256": canonical_sha256(value1),
            "equal": value0 == value1,
        }
    identity["reference"] = {
        "v0_sha256": p0.get("reference_sha256"),
        "v1_sha256": p1.get("reference_sha256"),
        "equal": p0.get("reference_sha256") == p1.get("reference_sha256"),
    }
    identity_gate = all(bool(item["equal"]) for item in identity.values())

    output_contract = {
        "replay_0_shape": list(v0_outputs[0].shape) == list(v1_outputs[0].shape),
        "replay_1_shape": list(v0_outputs[1].shape) == list(v1_outputs[1].shape),
        "replay_0_dtype": v0_outputs[0].dtype == v1_outputs[0].dtype,
        "replay_1_dtype": v0_outputs[1].dtype == v1_outputs[1].dtype,
        "all_outputs_finite": all(
            bool(torch.isfinite(value).all()) for value in (*v0_outputs, *v1_outputs)
        ),
    }
    output_contract_gate = all(output_contract.values())

    v0_self_drift = tensor_error(v0_outputs[1], v0_outputs[0])
    v1_self_drift = tensor_error(v1_outputs[1], v1_outputs[0])
    comparisons = []
    for v1_replay, candidate in enumerate(v1_outputs):
        for v0_replay, baseline in enumerate(v0_outputs):
            comparisons.append(
                {
                    "comparison": f"v1_r{v1_replay}_vs_v0_r{v0_replay}",
                    **tensor_error(candidate, baseline),
                }
            )
    candidate_worst = {
        metric: max(value[metric] for value in comparisons) for metric in v0_self_drift
    }
    strict = evaluate_cross_arm_correctness(
        v0_self_drift, v1_self_drift, candidate_worst
    )
    preparation_gate = all(v0_checks.values()) and all(v1_checks.values())
    gate = (
        preparation_gate
        and identity_gate
        and output_contract_gate
        and bool(strict["gate_pass"])
    )

    # The full preparation objects are not duplicated in the durable artifact;
    # their exact paths and hashes above retain source traceability.
    del v0["preparation"]
    del v1["preparation"]
    payload = {
        "name": spec.name,
        "m": spec.m,
        "harness_fixture": spec.harness_fixture,
        "semantic_fixture": spec.semantic_fixture,
        "canary_branch": spec.canary_branch,
        "arms": {"v0": v0, "v1": v1},
        "cross_arm_identity": identity,
        "cross_arm_identity_gate": identity_gate,
        "output_contract": output_contract,
        "output_contract_gate": output_contract_gate,
        "v0_self_drift": v0_self_drift,
        "v1_self_drift": v1_self_drift,
        "v1_vs_v0_four_way": comparisons,
        "v1_vs_v0_worst": candidate_worst,
        "strict_cross_arm_gate": strict,
        "preparation_gate": preparation_gate,
        "gate_pass": gate,
    }
    row = {
        "case": spec.name,
        "m": spec.m,
        "semantic_fixture": spec.semantic_fixture,
        "v0_source_sha256": EXPECTED_SOURCE["v0"],
        "v1_source_sha256": EXPECTED_SOURCE["v1"],
        "v0_cubin_sha256": EXPECTED_CUBIN["v0"],
        "v1_cubin_sha256": EXPECTED_CUBIN["v1"],
        "fixture_identity_sha256": identity["fixture"]["v0_sha256"],
        "weights_identity_sha256": identity["weights"]["v0_sha256"],
        "reference_sha256": identity["reference"]["v0_sha256"],
        "v0_self_drift_relative_l2": v0_self_drift["relative_l2"],
        "v1_self_drift_relative_l2": v1_self_drift["relative_l2"],
        "v1_vs_v0_worst_relative_l2": candidate_worst["relative_l2"],
        "v1_vs_v0_worst_max_abs": candidate_worst["max_abs"],
        "v1_vs_v0_worst_cosine_loss": candidate_worst["cosine_loss"],
        "v1_vs_v0_worst_token_rel_l2_p99": candidate_worst["token_rel_l2_p99"],
        "preparation_gate": preparation_gate,
        "identity_gate": identity_gate,
        "output_contract_gate": output_contract_gate,
        "strict_cross_arm_gate": bool(strict["gate_pass"]),
        "gate_pass": gate,
    }
    return payload, row


def case_specs(results: Path) -> list[CaseSpec]:
    v0_canonical = results / "baseline_v0/canonical/raw" / HARNESS_ARM
    v1_canonical = results / "canonical/v1/raw" / HARNESS_ARM
    specs = [
        CaseSpec(
            name="canonical_m256",
            m=256,
            harness_fixture="canonical",
            semantic_fixture="deterministic_synthetic_random_logits",
            v0_directory=v0_canonical / "m256/canonical",
            v1_directory=v1_canonical / "m256/canonical",
        ),
        CaseSpec(
            name="canonical_m8192",
            m=8192,
            harness_fixture="canonical",
            semantic_fixture="deterministic_synthetic_random_logits",
            v0_directory=v0_canonical / "m8192/canonical",
            v1_directory=v1_canonical / "m8192/canonical",
        ),
    ]
    directed = {
        "sparse_empty": "directed_sparse_empty",
        "exact_128": "directed_exact_128",
        "tail_129": "directed_tail_129",
        "hot_expert": "directed_hot_expert",
    }
    for fixture, semantic in directed.items():
        specs.append(
            CaseSpec(
                name=fixture,
                m=256,
                harness_fixture=fixture,
                semantic_fixture=semantic,
                v0_directory=v0_canonical / f"m256/{fixture}",
                v1_directory=v1_canonical / f"m256/{fixture}",
            )
        )
    for branch in ("up", "gate"):
        fixture = f"canary_{branch}_v2"
        specs.append(
            CaseSpec(
                name=fixture,
                m=256,
                harness_fixture="canonical",
                semantic_fixture=f"branch_half_slice_canary_{branch}",
                v0_directory=(
                    results
                    / "baseline_v0/canary"
                    / fixture
                    / "raw"
                    / HARNESS_ARM
                    / "m256/canonical"
                ),
                v1_directory=(
                    results
                    / "canary"
                    / fixture
                    / "v1/raw"
                    / HARNESS_ARM
                    / "m256/canonical"
                ),
                canary_branch=branch,
            )
        )
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    results = args.results.resolve()

    overlay_identity = {
        logical_arm: {
            "path": evidence_path(path),
            "expected_sha256": EXPECTED_SOURCE[logical_arm],
            "observed_sha256": sha256_file(path),
            "gate_pass": sha256_file(path) == EXPECTED_SOURCE[logical_arm],
        }
        for logical_arm, path in OVERLAY.items()
    }
    overlay_gate = all(value["gate_pass"] for value in overlay_identity.values())

    cases: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for spec in case_specs(results):
        payload, row = build_case(spec)
        cases[spec.name] = payload
        rows.append(row)
    gate = overlay_gate and all(bool(value["gate_pass"]) for value in cases.values())
    output = {
        "schema": "exp008.correctness.v1",
        "comparison": "exp007 temporal-N64 v0 baseline vs exp008 branch-paired-N64 v1",
        "harness_arm": HARNESS_ARM,
        "expected_source_sha256": EXPECTED_SOURCE,
        "expected_cubin_sha256": EXPECTED_CUBIN,
        "overlay_identity": overlay_identity,
        "overlay_gate": overlay_gate,
        "threshold_policy": (
            "exp005_common.evaluate_cross_arm_correctness; "
            "min(cap, max(floor, 3 * v0_self_drift))"
        ),
        "cases": cases,
        "gate_pass": gate,
        "evidence_boundary": (
            "Route exact-once is descriptor-multiset plus terminal-head inference; "
            "the worker does not carry a per-task consumed bitmap. Static identity "
            "and numerical correctness are established here; no performance claim "
            "is made."
        ),
    }
    (results / "correctness_evidence.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    with (results / "correctness_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "gate_pass": gate,
                "case_count": len(rows),
                "output": str((results / "correctness_evidence.json").resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
