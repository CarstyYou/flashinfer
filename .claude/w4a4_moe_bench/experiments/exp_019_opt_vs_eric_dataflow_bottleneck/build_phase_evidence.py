#!/usr/bin/env python3
"""Build the compact paired phase evidence card for exp_019."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

from phase_common import (
    ARMS,
    BLOCK_THREADS,
    CONTROL,
    EXPECTED_SOURCE_SHA256,
    PROBE,
    perturbation_gate,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EXP018_SUMMARY = (
    ROOT.parent / "exp_018_triton_opt_eric_benchmark/results/benchmark_summary.csv"
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def benchmark_anchor(path: Path, m: int) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {
            row["arm"]: row
            for row in csv.DictReader(handle)
            if int(row["m"]) == m
            and row["arm"] in {"latest_opt_fp4", "eric_stage4_fp4"}
        }
    if set(rows) != set(ARMS) or any(row["status"] != "Pass" for row in rows.values()):
        raise RuntimeError(f"missing qualified exp018 benchmark anchor for M{m}")
    opt = float(rows["latest_opt_fp4"]["median_us"])
    eric = float(rows["eric_stage4_fp4"]["median_us"])
    return {
        "latest_opt_us": opt,
        "eric_stage4_us": eric,
        "eric_minus_opt_us": eric - opt,
        "eric_vs_opt_latency_percent": 100.0 * (eric / opt - 1.0),
        "source": str(path.relative_to(ROOT.parents[3])),
    }


def resource_record(capture: dict[str, Any]) -> dict[str, Any]:
    rows = capture["static_resource_usage"]["records"]
    if len(rows) != 1:
        raise RuntimeError("expected one phase cubin resource row")
    row = rows[0]
    return {
        key: row[key]
        for key in (
            "cubin_sha256",
            "registers_per_thread",
            "stack_bytes_per_thread",
            "static_local_bytes_outside_stack",
            "static_shared_bytes_per_cta",
        )
    }


def validate_capture(capture: dict[str, Any], *, arm: str, mode: str, m: int) -> None:
    if (
        capture.get("schema") != "exp019.phase-capture.v1"
        or capture.get("arm") != arm
        or capture.get("mode") != mode
        or capture.get("case", {}).get("m") != m
    ):
        raise RuntimeError(f"capture identity drift: {arm}/{mode}/M{m}")
    if capture["source"]["base_sha256"] != EXPECTED_SOURCE_SHA256[arm]:
        raise RuntimeError(f"source identity drift: {arm}/{mode}/M{m}")
    if capture["launch_identity"] != {
        "grid": [1, 1, 110],
        "block": [BLOCK_THREADS[arm], 1, 1],
    }:
        raise RuntimeError(f"launch identity drift: {arm}/{mode}/M{m}")
    if capture["foreign_processes_after"]:
        raise RuntimeError(f"foreign process in capture: {arm}/{mode}/M{m}")
    if not capture["specialization"]["gate_pass"]:
        raise RuntimeError(f"specialization gate failed: {arm}/{mode}/M{m}")
    for run in capture["runs"]:
        if not run["gate_pass"]:
            raise RuntimeError(f"replay gate failed: {arm}/{mode}/M{m}")
    if mode == PROBE and not capture["eager"]["occurrence_gate"]["gate_pass"]:
        raise RuntimeError(f"occurrence gate failed: {arm}/M{m}")


def build_case(results: Path, m: int) -> dict[str, Any]:
    root = results / "raw/phase" / f"m{m}" / "block0"
    captures: dict[tuple[str, str], dict[str, Any]] = {}
    sources = {}
    resources = {}
    for arm in ARMS:
        resources[arm] = {}
        for mode in (CONTROL, PROBE):
            path = root / f"{arm}_{mode}.json"
            capture = read_json(path)
            validate_capture(capture, arm=arm, mode=mode, m=m)
            captures[(arm, mode)] = capture
            sources[f"{arm}/{mode}"] = str(path.relative_to(results))
            resources[arm][mode] = resource_record(capture)

        control = resources[arm][CONTROL]
        probe = resources[arm][PROBE]
        if (
            control["stack_bytes_per_thread"] != probe["stack_bytes_per_thread"]
            or control["static_local_bytes_outside_stack"]
            != probe["static_local_bytes_outside_stack"]
        ):
            raise RuntimeError(f"probe changed stack/local tier: {arm}/M{m}")

    anchor = benchmark_anchor(EXP018_SUMMARY, m)
    quality = perturbation_gate(
        captures, whole_op_gap_us=abs(anchor["eric_minus_opt_us"])
    )
    opt_rows = {
        row["name"]: row
        for row in captures[("latest_opt_fp4", PROBE)]["summary"]["semantic_rows"]
    }
    eric_rows = {
        row["name"]: row
        for row in captures[("eric_stage4_fp4", PROBE)]["summary"]["semantic_rows"]
    }
    if set(opt_rows) != set(eric_rows):
        raise RuntimeError(f"semantic row vocabulary drift: M{m}")
    comparison = []
    for name in opt_rows:
        opt_us = float(opt_rows[name]["equivalent_wall_us_median"])
        eric_us = float(eric_rows[name]["equivalent_wall_us_median"])
        comparison.append(
            {
                "phase": name,
                "latest_opt_us": opt_us,
                "latest_opt_share_percent": float(
                    opt_rows[name]["share_percent_median"]
                ),
                "eric_stage4_us": eric_us,
                "eric_stage4_share_percent": float(
                    eric_rows[name]["share_percent_median"]
                ),
                "eric_minus_opt_us": eric_us - opt_us,
                "eric_vs_opt_percent": 100.0 * (eric_us / opt_us - 1.0),
            }
        )

    control_gap_us = median(
        run["event_elapsed_us"]
        for run in captures[("eric_stage4_fp4", CONTROL)]["runs"]
    ) - median(
        run["event_elapsed_us"] for run in captures[("latest_opt_fp4", CONTROL)]["runs"]
    )
    probe_gap_us = median(
        run["event_elapsed_us"] for run in captures[("eric_stage4_fp4", PROBE)]["runs"]
    ) - median(
        run["event_elapsed_us"] for run in captures[("latest_opt_fp4", PROBE)]["runs"]
    )
    phase_gap_us = sum(row["eric_minus_opt_us"] for row in comparison)
    benchmark_gap_us = float(anchor["eric_minus_opt_us"])
    same_direction = control_gap_us * benchmark_gap_us > 0.0
    magnitude_ratio = (
        max(abs(control_gap_us), abs(benchmark_gap_us))
        / min(abs(control_gap_us), abs(benchmark_gap_us))
        if control_gap_us != 0.0 and benchmark_gap_us != 0.0
        else None
    )
    phase_probe_error_us = abs(phase_gap_us - probe_gap_us)
    quality["cross_source_consistency"] = {
        "benchmark_gap_us": benchmark_gap_us,
        "fresh_control_gap_us": control_gap_us,
        "probe_event_gap_us": probe_gap_us,
        "phase_sum_gap_us": phase_gap_us,
        "phase_vs_probe_abs_error_us": phase_probe_error_us,
        "benchmark_control_same_direction": same_direction,
        "benchmark_control_magnitude_ratio": magnitude_ratio,
        "benchmark_control_within_2x": magnitude_ratio is not None
        and magnitude_ratio <= 2.0,
        "phase_sum_closes_probe_gap": phase_probe_error_us
        <= max(2.0, 0.01 * abs(probe_gap_us)),
    }
    quality["checks"].update(
        {
            "benchmark_control_same_direction": same_direction,
            "benchmark_control_within_2x": magnitude_ratio is not None
            and magnitude_ratio <= 2.0,
            "phase_sum_closes_probe_gap": quality["cross_source_consistency"][
                "phase_sum_closes_probe_gap"
            ],
        }
    )
    quality["gate_pass"] = all(quality["checks"].values())
    return {
        "m": m,
        "benchmark_anchor": anchor,
        "capture_sources": sources,
        "quality_gate": quality,
        "resources": resources,
        "comparison": comparison,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--m", type=int, action="append", choices=(1024, 8192))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = args.results.resolve()
    selected = sorted(set(args.m or [8192]))
    payload = {
        "schema": "exp019.phase-evidence.v1",
        "classification": "matched diagnostic phase timing; exp018 is E2E authority",
        "cases": [build_case(results, m) for m in selected],
    }
    output = (args.output or results / "phase_evidence.json").resolve()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if len(text.encode()) > 200_000:
        raise RuntimeError("phase evidence card exceeds 200 KB")
    output.write_text(text, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
