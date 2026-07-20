#!/usr/bin/env python3
"""Validate and summarize the two-group exp_013 quick ABBA evidence."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
import statistics


ORDER = ("exp008", "exp013_v2", "exp013_v2", "exp008")
EXPECTED_SHA = {
    "exp008": "f3c246817679d962a3f7160dbe8b9e68262c919e26e306f349200961fc4ac971",
    "exp013_v2": "e2fb46e49001b7fe17761fcd9af92b8775d41c6b0c5932172e3c57839d4199e5",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(Path(path).read_text())
        for path in glob.glob(str(args.raw / "m*/*.json"))
    ]
    if len(rows) != 16:
        raise RuntimeError(f"expected 16 immutable ABBA samples, got {len(rows)}")

    stable = {
        "gpu_uuid": {row["runtime"]["gpu"]["uuid"] for row in rows},
        "application_clock_mhz": {
            row["runtime"]["gpu"]["applications_graphics_clock_mhz"] for row in rows
        },
        "image_digest": {row["runtime"]["image_digest"] for row in rows},
        "python_deps_sha256": {row["runtime"]["python_deps_sha256"] for row in rows},
        "nvcc": {row["runtime"]["nvcc"] for row in rows},
        "warmup": {row["warmup"] for row in rows},
        "iters": {row["iters"] for row in rows},
        "l2_flush_bytes": {row["l2_flush_bytes"] for row in rows},
    }
    if any(len(values) != 1 for values in stable.values()):
        raise RuntimeError(f"benchmark identity drift: {stable}")
    for row in rows:
        arm = row["external_arm"]
        if row.get("status") != "complete":
            raise RuntimeError("incomplete benchmark sample")
        if row["runtime"]["source"]["overlay_sha256"] != EXPECTED_SHA[arm]:
            raise RuntimeError("overlay identity drift")
        if ORDER[row["position"]] != arm:
            raise RuntimeError("ABBA order drift")

    cases = []
    for m in (256, 8192):
        case_rows = [row for row in rows if row["m"] == m]
        fixture_ids = {row["fixture"]["topk_ids_sha256"] for row in case_rows}
        fixture_x = {row["fixture"]["x_sha256"] for row in case_rows}
        if len(fixture_ids) != 1 or len(fixture_x) != 1:
            raise RuntimeError(f"fixture drift at M={m}")
        groups = []
        ratios = []
        all_anchor = []
        all_candidate = []
        for group in (0, 1):
            group_rows = sorted(
                (row for row in case_rows if row["group"] == group),
                key=lambda row: row["position"],
            )
            if [row["external_arm"] for row in group_rows] != list(ORDER):
                raise RuntimeError(f"incomplete ABBA group M={m} group={group}")
            anchor = [
                row["sample_us"]
                for row in group_rows
                if row["external_arm"] == "exp008"
            ]
            candidate = [
                row["sample_us"]
                for row in group_rows
                if row["external_arm"] == "exp013_v2"
            ]
            anchor_us = statistics.mean(anchor)
            candidate_us = statistics.mean(candidate)
            ratio = anchor_us / candidate_us
            ratios.append(ratio)
            all_anchor.extend(anchor)
            all_candidate.extend(candidate)
            groups.append(
                {
                    "group": group,
                    "exp008_samples_us": anchor,
                    "exp013_v2_samples_us": candidate,
                    "exp008_mean_us": anchor_us,
                    "exp013_v2_mean_us": candidate_us,
                    "speedup_pct": (ratio - 1.0) * 100.0,
                }
            )
        cases.append(
            {
                "m": m,
                "fixture_topk_ids_sha256": next(iter(fixture_ids)),
                "fixture_x_sha256": next(iter(fixture_x)),
                "exp008_mean_us": statistics.mean(all_anchor),
                "exp013_v2_mean_us": statistics.mean(all_candidate),
                "speedup_pct": (math.prod(ratios) ** (1.0 / len(ratios)) - 1.0) * 100.0,
                "groups": groups,
            }
        )

    payload = {
        "schema": "exp013.quick-abba-summary.v1",
        "status": "complete",
        "decision": "reject_no_speedup",
        "speedup_definition": "(exp008_us / exp013_v2_us - 1) * 100",
        "protocol": {
            "order": list(ORDER),
            "groups": 2,
            "warmup": next(iter(stable["warmup"])),
            "iters": next(iter(stable["iters"])),
            "l2_flush_bytes": next(iter(stable["l2_flush_bytes"])),
        },
        "identity": {key: next(iter(values)) for key, values in stable.items()},
        "overlays": EXPECTED_SHA,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
