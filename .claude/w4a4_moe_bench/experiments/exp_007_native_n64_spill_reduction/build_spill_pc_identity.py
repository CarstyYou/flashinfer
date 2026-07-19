#!/usr/bin/env python3
"""Build the compact exp_007 static/dynamic spill-PC identity ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_DYNAMIC_METRICS = {
    "spill_refill_instructions": "sass__inst_executed_register_spilling_op_read",
    "spill_store_instructions": "sass__inst_executed_register_spilling_op_write",
    "spill_refill_bytes": ("sass__inst_executed_register_spilling_mem_local_op_read"),
    "spill_store_bytes": ("sass__inst_executed_register_spilling_mem_local_op_write"),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def build_arm(
    results: Path, arm: str, static: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any]:
    capture = results / "ncu" / arm / "m8192" / "canonical_v0"
    dynamic_path = capture / "dynamic_spill.json"
    source_path = capture / "veloq" / "source_spill_by_sass.json"
    dynamic = read_json(dynamic_path)
    source = read_json(source_path)
    static_case = static["cases"][f"{arm}_m8192"]
    ncu_identity = identity["arms"][arm]

    static_cubin = static_case["identity"]["cubin_sha256"]
    loaded_cubin = ncu_identity["loaded_cubin_sha256"]
    if loaded_cubin != static_cubin:
        raise ValueError(f"{arm}: loaded/static cubin mismatch")

    if source.get("schema") != "v1" or source.get("command") != "ncu.source-metrics":
        raise ValueError(f"{arm}: invalid VeloQ source-metrics response")
    source_data = source.get("data")
    if not isinstance(source_data, dict) or source_data.get("axis") != "sass":
        raise ValueError(f"{arm}: invalid VeloQ source-metrics data")
    if source_data.get("auxiliary", {}).get("row_id") != "launch:0":
        raise ValueError(f"{arm}: source-metrics launch identity drift")

    metrics: dict[str, Any] = dynamic["metrics"]
    dynamic_values: dict[str, dict[str, Any]] = {}
    for label, metric_id in REQUIRED_DYNAMIC_METRICS.items():
        metric = metrics.get(label)
        if not isinstance(metric, dict) or metric.get("metric_id") != metric_id:
            raise ValueError(f"{arm}: required dynamic metric absent: {metric_id}")
        value = metric.get("value")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{arm}: required dynamic metric nonnumeric: {metric_id}")
        dynamic_values[label] = {
            "metric_id": metric_id,
            "unit": metric.get("unit"),
            "value": value,
        }

    auxiliary = source_data.get("auxiliary", {})
    matched = auxiliary.get("matched_counters", [])
    skipped = auxiliary.get("skipped_counters", [])
    skipped_by_name = {
        item.get("name"): item.get("reason")
        for item in skipped
        if isinstance(item, dict)
    }
    required_counter_ids = list(REQUIRED_DYNAMIC_METRICS.values())
    required_not_source_attributable = all(
        skipped_by_name.get(name) == "not-a-source-counter"
        for name in required_counter_ids
    )
    rows = source_data.get("rows", [])
    source_attribution_available = bool(matched)
    if source_attribution_available:
        raise ValueError(
            f"{arm}: builder currently expects the observed non-attributable counters"
        )
    if not required_not_source_attributable:
        raise ValueError(f"{arm}: required counter disposition is ambiguous")

    compiler = static_case["compiler_spill_refill"]
    static_pcs = sorted(compiler["annotation_pcs_hex"], key=lambda item: int(item, 16))
    all_dynamic_zero = all(item["value"] == 0 for item in dynamic_values.values())
    static_zero = (
        compiler["annotation_count"] == 0
        and compiler["local_sass_instruction_count"] == 0
        and not static_pcs
    )

    return {
        "identity": {
            "block": ncu_identity["block"],
            "grid": ncu_identity["grid"],
            "kernel": ncu_identity["kernel"],
            "launch_row_id": ncu_identity["launch_row_id"],
            "loaded_cubin_matches_static": True,
            "loaded_cubin_sha256": loaded_cubin,
            "report_sha256": ncu_identity["report_sha256"],
        },
        "static_spill_pcs": {
            "compiler_annotation_count": compiler["annotation_count"],
            "compiler_annotations_equal_local_sass": compiler[
                "annotation_exactly_matches_local_sass"
            ],
            "local_sass_instruction_count": compiler["local_sass_instruction_count"],
            "pc_set_sha256": sha256_strings(static_pcs),
            "static_zero_spill": static_zero,
        },
        "dynamic_spill_totals": {
            "all_four_required_numeric": True,
            "all_four_zero": all_dynamic_zero,
            "metrics": dynamic_values,
        },
        "source_spill_by_sass": {
            "command": (
                "veloq ncu source-metrics <trace.ncu-rep> --row-id launch:0 "
                "--counter 'sass__inst_executed_register_spilling_*' --by sass"
            ),
            "matched_counters": matched,
            "required_counters_skipped_as_not_source_counter": (
                required_not_source_attributable
            ),
            "row_count": len(rows),
            "rows_empty": not rows,
            "source_attribution_available": source_attribution_available,
            "source_rows_are_independent_zero_spill_evidence": False,
            "limitation": (
                "VeloQ reports all required spill counters as not-a-source-counter; "
                "empty rows therefore cannot identify or independently clear spill PCs."
            ),
        },
        "pc_relation_to_static": {
            "status": "not_evaluable",
            "dynamic_pc_subset_of_static_compiler_or_local_sass": None,
            "reason": "NCU report exposes required spill counters only as launch aggregates, not per-SASS instances.",
        },
        "evidence_sha256": {
            "dynamic_spill_json": sha256_file(dynamic_path),
            "source_spill_by_sass_json": sha256_file(source_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    results = args.results.resolve()
    static_path = results / "static_spill_evidence.json"
    identity_path = results / "ncu" / "veloq_identity.json"
    static = read_json(static_path)
    identity = read_json(identity_path)

    arms = {
        arm: build_arm(results, arm, static, identity)
        for arm in ("anchor", "candidate")
    }
    candidate = arms["candidate"]
    candidate_zero_closed_without_pc_attribution = (
        candidate["static_spill_pcs"]["static_zero_spill"]
        and candidate["dynamic_spill_totals"]["all_four_zero"]
    )
    payload = {
        "schema": "exp007.spill-pc-identity.v1",
        "scope": "M8192 canonical_v0, launch:0",
        "arms": arms,
        "verdict": {
            "loaded_cubin_identity_pass": all(
                arm["identity"]["loaded_cubin_matches_static"] for arm in arms.values()
            ),
            "anchor_dynamic_pc_to_static_pc_relation_closed": False,
            "candidate_static_and_aggregate_dynamic_zero_spill": (
                candidate_zero_closed_without_pc_attribution
            ),
            "per_sass_spill_pc_closure": "not_supported_by_capture_counters",
            "statement": (
                "Candidate has zero static spill/local SASS and all four required "
                "dynamic spill aggregates are numeric zero. Per-SASS source attribution "
                "is unavailable for both arms, so anchor dynamic spill PCs cannot be "
                "tested as a subset of the static compiler SpillRefill/local SASS PCs; "
                "empty source rows are not treated as zero evidence."
            ),
        },
        "evidence_sha256": {
            "static_spill_evidence_json": sha256_file(static_path),
            "veloq_identity_json": sha256_file(identity_path),
        },
    }
    output = results / "ncu" / "spill_pc_identity.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
