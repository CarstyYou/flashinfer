#!/usr/bin/env python3
"""Run a profiler-only replay while making post-benchmark overlay drift explicit.

The canonical exp_002 benchmark was captured before its harness files were
finalized.  This wrapper keeps every canonical prerequisite check (fixture,
weights, JIT artifacts, benchmark identity, software, GPU, and output path),
but temporarily supplies the canonical source-overlay identity to that check.
The actual current source identity is restored before the profile manifest is
written and the exact drift is recorded in a separate follow-up manifest.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import run_exp002 as exp002


ORIGINAL_VALIDATE = exp002.validate_profile_prerequisite


def overlay_map(source: dict[str, Any]) -> dict[str, str]:
    return {
        str(item["path"]): str(item["sha256"])
        for item in source.get("experiment_overlays", [])
    }


def validate_profile_prerequisite(
    results: Path,
    fixture: Any,
    weights: Any,
    m: int,
    arm: str,
    runtime: dict[str, Any],
    max_num_tokens: int,
) -> dict[str, Any]:
    correctness = json.loads((results / "correctness.json").read_text())
    expected_source = correctness["runtime"]["source"]
    actual_source = copy.deepcopy(runtime["source"])

    for key in ("flashinfer_commit", "cutlass_commit", "source_status"):
        if actual_source.get(key) != expected_source.get(key):
            raise RuntimeError(f"follow-up source drift exceeds overlay files: {key}")

    expected_overlays = overlay_map(expected_source)
    actual_overlays = overlay_map(actual_source)
    if expected_overlays.keys() != actual_overlays.keys():
        raise RuntimeError("follow-up overlay file set drift")
    changed = {
        path: {"canonical_sha256": expected_overlays[path], "actual_sha256": value}
        for path, value in actual_overlays.items()
        if value != expected_overlays[path]
    }
    if not changed:
        raise RuntimeError("follow-up wrapper used without source-overlay drift")

    runtime["source"] = copy.deepcopy(expected_source)
    try:
        identity = ORIGINAL_VALIDATE(
            results, fixture, weights, m, arm, runtime, max_num_tokens
        )
    finally:
        runtime["source"] = actual_source

    manifest_path = os.environ.get("W4A4_FOLLOWUP_VALIDATION_MANIFEST", "")
    if not manifest_path:
        raise RuntimeError("W4A4_FOLLOWUP_VALIDATION_MANIFEST is required")
    exp002.write_json(
        Path(manifest_path),
        {
            "schema": "exp002.profile-followup-overlay-drift.v1",
            "status": "accepted_for_diagnostic_profile",
            "scope": {"m": m, "arm": arm},
            "changed_overlays": changed,
            "unchanged_source_identity": {
                key: actual_source[key]
                for key in ("flashinfer_commit", "cutlass_commit", "source_status")
            },
            "canonical_prerequisites_revalidated": [
                "paired correctness gate",
                "per-arm oracle gate",
                "fixture manifest",
                "canonical weights/scales",
                "JIT artifact manifest",
                "benchmark row identity and stability",
                "runtime software and GPU identity",
            ],
            "post_capture_gates": [
                "profile output passes shape/dtype/finite validation",
                "exact output SHA-256 stability is audited before it is used as a gate",
                "captured cubin SHA-256 equals canonical deep-profile cubin",
                "kernel name/grid/block equal canonical target",
            ],
        },
    )
    return identity


exp002.validate_profile_prerequisite = validate_profile_prerequisite


if __name__ == "__main__":
    raise SystemExit(exp002.main())
