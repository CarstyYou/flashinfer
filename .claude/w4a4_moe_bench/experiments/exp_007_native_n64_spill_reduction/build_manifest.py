#!/usr/bin/env python3
"""Assemble the compact identity/gate manifest for exp_007."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_value(payload: dict[str, Any], name: str) -> float:
    return float(payload["metrics"][name]["value"])


def write_diff(anchor: Path, candidate: Path, output: Path) -> None:
    anchor_lines = anchor.read_text().splitlines(keepends=True)
    candidate_lines = candidate.read_text().splitlines(keepends=True)
    diff = difflib.unified_diff(
        anchor_lines,
        candidate_lines,
        fromfile="anchor_8warp_n128/moe_dynamic_kernel.py",
        tofile="candidate_8warp_native_n64_v0/moe_dynamic_kernel.py",
        n=0,
    )
    output.write_text("".join(diff))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path(__file__).parent)
    args = parser.parse_args(argv)
    experiment = args.experiment.resolve()
    results = experiment / "results"

    anchor_source = results / "overlays/anchor_8warp_n128/moe_dynamic_kernel.py"
    candidate_source = (
        results / "overlays/candidate_8warp_native_n64_v0/moe_dynamic_kernel.py"
    )
    source_diff = results / "overlays/anchor_vs_candidate.diff"
    write_diff(anchor_source, candidate_source, source_diff)

    correctness = read_json(results / "correctness_evidence.json")
    work = read_json(results / "work_ledger.json")
    static = read_json(results / "static_spill_evidence.json")
    dynamic = {
        arm: read_json(results / f"ncu/{arm}/m8192/canonical_v0/dynamic_spill.json")
        for arm in ("anchor", "candidate")
    }
    captures = {
        arm: read_json(results / f"ncu/{arm}/m8192/canonical_v0/capture_identity.json")
        for arm in ("anchor", "candidate")
    }
    veloq = read_json(results / "ncu/veloq_identity.json")
    spill_pc = read_json(results / "ncu/spill_pc_identity.json")
    static_anchor = static["cases"]["anchor_m8192"]
    static_candidate = static["cases"]["candidate_m8192"]

    source_identity = {
        "anchor": {
            "path": str(anchor_source.relative_to(experiment)),
            "sha256": sha256_file(anchor_source),
        },
        "candidate": {
            "path": str(candidate_source.relative_to(experiment)),
            "sha256": sha256_file(candidate_source),
        },
        "unified_diff": {
            "path": str(source_diff.relative_to(experiment)),
            "sha256": sha256_file(source_diff),
        },
        "production_kernel_sha256": static_anchor["identity"]["preparation"][
            "source"
        ]["production_kernel_sha256"],
        "flashinfer_commit": static_anchor["identity"]["preparation"]["source"][
            "checkout_head"
        ],
        "cutlass_commit": static_anchor["identity"]["preparation"]["source"][
            "cutlass_commit"
        ],
    }

    useful_work_equal = (
        static["cross_case_checks"]["anchor_candidate_same_static_omma_count"]
        and static["cross_case_checks"][
            "anchor_candidate_same_static_omma_histogram"
        ]
        and metric_value(dynamic["anchor"], "tensor_instructions")
        == metric_value(dynamic["candidate"], "tensor_instructions")
        and metric_value(dynamic["anchor"], "fp4_tensor_ops")
        == metric_value(dynamic["candidate"], "fp4_tensor_ops")
    )
    loaded_cubins_match = all(
        veloq["arms"][arm]["loaded_cubin_sha256"]
        == captures[arm]["cubin_sha256"]
        for arm in ("anchor", "candidate")
    )
    anchor_static_nonzero = not static_anchor["gates"]["zero_spill_static_gate"]
    candidate_static_zero = static_candidate["gates"]["zero_spill_static_gate"]
    anchor_dynamic_nonzero = not dynamic["anchor"]["gates"]["dynamic_zero_spill"]
    candidate_dynamic_zero = dynamic["candidate"]["gates"]["dynamic_zero_spill"]

    gates = {
        "descriptor_work_ledger": bool(work["gate_pass"]),
        "correctness_required_cases": bool(correctness["gate_pass"]),
        "fresh_anchor_residual_spill_reproduced_static": anchor_static_nonzero,
        "fresh_anchor_residual_spill_reproduced_dynamic": anchor_dynamic_nonzero,
        "candidate_static_zero_spill": candidate_static_zero,
        "candidate_dynamic_zero_spill": candidate_dynamic_zero,
        "useful_tensor_work_equal": useful_work_equal,
        "ncu_loaded_cubins_match_static_evidence": loaded_cubins_match,
    }

    environment = static_anchor["identity"]["preparation"]
    manifest = {
        "schema": "exp007.manifest.v1",
        "status": "complete" if all(gates.values()) else "incomplete",
        "objective": (
            "Test whether an 8-warp temporal dual-N64 FC1 bundle, with each "
            "Gate/Up accumulator pair consumed immediately by SwiGLU, can remove "
            "the anchor's residual register spill without changing useful Tensor work."
        ),
        "source_identity": source_identity,
        "environment": {
            "gpu": environment["gpu"],
            "image_digest": environment["image_digest"],
            "nvcc": environment["compile_tools"]["nvcc"],
            "ptxas": environment["compile_tools"]["ptxas"],
            "ncu": captures["candidate"]["ncu_version"],
        },
        "binary_identity": {
            arm: {
                "cubin_sha256": captures[arm]["cubin_sha256"],
                "loaded_cubin_sha256": veloq["arms"][arm][
                    "loaded_cubin_sha256"
                ],
                "kernel": veloq["arms"][arm]["kernel"],
                "grid": veloq["arms"][arm]["grid"],
                "block": veloq["arms"][arm]["block"],
            }
            for arm in ("anchor", "candidate")
        },
        "correctness": {
            "gate_pass": correctness["gate_pass"],
            "required_cases": [
                name
                for name, value in correctness["cases"].items()
                if value["required_for_final_gate"]
            ],
            "known_failed_diagnostic_cases": correctness[
                "known_failed_diagnostic_cases"
            ],
            "known_failed_fixture_attempts": correctness[
                "known_failed_fixture_attempts"
            ],
        },
        "static_spill": {
            "anchor": {
                "registers_per_thread": static_anchor["resource"][
                    "registers_per_thread"
                ],
                "stack_bytes_per_thread": static_anchor["resource"][
                    "stack_bytes_per_thread"
                ],
                "spill_refill_annotations": static_anchor[
                    "compiler_spill_refill"
                ]["annotation_count"],
            },
            "candidate": {
                "registers_per_thread": static_candidate["resource"][
                    "registers_per_thread"
                ],
                "stack_bytes_per_thread": static_candidate["resource"][
                    "stack_bytes_per_thread"
                ],
                "spill_refill_annotations": static_candidate[
                    "compiler_spill_refill"
                ]["annotation_count"],
            },
            "static_omma_count_each": static_anchor["tensor_core_work"][
                "omma_static_instruction_count"
            ],
        },
        "dynamic_spill": {
            arm: {
                name: dynamic[arm]["metrics"][name]
                for name in (
                    "spill_refill_instructions",
                    "spill_store_instructions",
                    "spill_refill_bytes",
                    "spill_store_bytes",
                    "local_load_bytes",
                    "local_store_bytes",
                    "tensor_instructions",
                    "fp4_tensor_ops",
                    "registers_per_thread",
                    "allocated_registers_per_thread",
                    "achieved_occupancy_pct",
                )
            }
            for arm in ("anchor", "candidate")
        },
        "spill_pc_attribution": {
            "verdict": spill_pc["verdict"],
            "evidence_sha256": sha256_file(results / "ncu/spill_pc_identity.json"),
        },
        "gates": gates,
        "overall_gate_pass": all(gates.values()),
        "failed_attempts": [
            {
                "name": "anchor_m256_attempt0_missing_cutlass_gitdir_mount",
                "effect": "failed before JIT; excluded from canonical evidence",
            },
            {
                "name": "canary_v0_output_amplitude_mismatch",
                "effect": (
                    "8-block relative gates passed, but strict max_abs 0.25 exceeded "
                    "the fixed 0.1 cap; retained as a failed diagnostic case."
                ),
            },
            {
                "name": "canary_up_v1_invalid_w2_global_scale_control",
                "effect": (
                    "failed independent-reference replay 0; rejected before the "
                    "final v2 scatter-only scale fixture, with thresholds unchanged."
                ),
            },
        ],
        "evidence_boundaries": [
            "The accepted claim is scoped to the complete immutable candidate bundle, not N64 alone.",
            "Source ordering proves intended immediate consumption; zero static/dynamic spill proves the compiled outcome, but does not isolate one compiler live-range cause.",
            "No latency or performance claim is made in exp_007.",
            "Scheduler exact-once remains descriptor-multiset plus terminal-head inference, not a consumed bitmap.",
            "A/SFA and physical-N128 SFB replay are accepted candidate consequences and are recorded in work_ledger.json.",
            "The spill counters are launch aggregates rather than source counters, so dynamic spill cannot be attributed to individual SASS PCs.",
            "canary_v2 inherited a stale weight_sum_max_abs_error metadata field; that field is rejected from evidence, while the explicit topk_weight_sum, tensor hashes, independent reference, and unchanged thresholds are retained.",
        ],
    }
    manifest_path = results / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"overall_gate_pass": manifest["overall_gate_pass"]}))
    return 0 if manifest["overall_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
