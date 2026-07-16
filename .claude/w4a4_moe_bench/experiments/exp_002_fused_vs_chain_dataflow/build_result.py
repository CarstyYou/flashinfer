#!/usr/bin/env python3
"""Build the exp_002 benchmark comparison without profiler attribution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ARMS = (
    "cutedsl_bf16_fused",
    "cutlass_bf16_chain",
)
ROOT = Path(__file__).resolve().parent


def speedup_percent(*, baseline_us: float, candidate_us: float) -> float:
    return (baseline_us / candidate_us - 1.0) * 100.0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def comparison_rows(
    summary: list[dict[str, str]], correctness: dict[str, Any]
) -> list[dict[str, Any]]:
    identity = correctness["evidence_identity"]
    indexed = {(int(row["m"]), row["arm"]): row for row in summary}
    m_values = sorted({m for m, _ in indexed})
    rows = []
    for m in m_values:
        missing = [name for name in ARMS if (m, name) not in indexed]
        if missing:
            raise ValueError(f"m={m} missing arms: {missing}")
        fused = indexed[(m, "cutedsl_bf16_fused")]
        matched = indexed[(m, "cutlass_bf16_chain")]
        fused_us = float(fused["median_us"])
        matched_us = float(matched["median_us"])
        case_gate = correctness["cases"][str(m)]["paired_gate_pass"]
        paired_stable = all(
            indexed[(m, name)]["stable_le_5_percent"].lower() == "true"
            for name in ("cutedsl_bf16_fused", "cutlass_bf16_chain")
        )
        paired_valid = bool(case_gate and paired_stable)
        shared_fields = (
            "comparison_group_id",
            "rerun_id",
            "environment_lock_digest",
            "protocol_lock_digest",
        )
        for key in shared_fields:
            values = {indexed[(m, name)].get(key, "") for name in ARMS}
            if values != {identity[key]}:
                raise ValueError(f"m={m} {key} mismatch")
        for name in ARMS:
            actual = indexed[(m, name)].get("artifact_fingerprint_sha256", "")
            expected = identity["per_arm_artifact_fingerprint_sha256"][name]
            if actual != expected:
                raise ValueError(f"m={m} arm={name} artifact fingerprint mismatch")
        rows.append(
            {
                "m": m,
                "cutedsl_fused_us": fused_us,
                "cutlass_bf16_chain_us": matched_us,
                "cutedsl_speedup_vs_matched_percent": (
                    speedup_percent(baseline_us=matched_us, candidate_us=fused_us)
                    if paired_valid
                    else None
                ),
                "paired_oracle_gate": "pass" if case_gate else "fail",
                "paired_arms_spread_le_5_percent": paired_stable,
                "comparison_group_id": identity["comparison_group_id"],
                "rerun_id": identity["rerun_id"],
                "environment_lock_digest": identity["environment_lock_digest"],
                "protocol_lock_digest": identity["protocol_lock_digest"],
                "formal_comparison": paired_valid,
            }
        )
    return rows


def validate_raw_rerun(raw: list[dict[str, str]], identity: dict[str, Any]) -> None:
    grouped: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in raw:
        grouped.setdefault((int(row["m"]), int(row["repeat"])), []).append(row)
    if not grouped:
        raise ValueError("benchmark raw rows are empty")
    for (m, repeat), rows in grouped.items():
        observed_arms = [row["arm"] for row in rows]
        if len(rows) != len(ARMS) or set(observed_arms) != set(ARMS):
            raise ValueError(
                f"m={m} repeat={repeat} does not contain the complete arm set"
            )
        declared_order = {row["order"] for row in rows}
        if len(declared_order) != 1:
            raise ValueError(f"m={m} repeat={repeat} order declaration mismatch")
        order = next(iter(declared_order)).split(">")
        if order != [
            row["arm"] for row in sorted(rows, key=lambda row: int(row["order_index"]))
        ]:
            raise ValueError(f"m={m} repeat={repeat} observed order mismatch")
        for row in rows:
            for key in (
                "comparison_group_id",
                "rerun_id",
                "environment_lock_digest",
                "protocol_lock_digest",
            ):
                if row.get(key) != identity[key]:
                    raise ValueError(f"m={m} repeat={repeat} {key} mismatch")
            expected_artifact = identity["per_arm_artifact_fingerprint_sha256"][
                row["arm"]
            ]
            if row.get("artifact_fingerprint_sha256") != expected_artifact:
                raise ValueError(
                    f"m={m} repeat={repeat} arm artifact fingerprint mismatch"
                )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Experiment 002 Benchmark Result",
        "",
        "这里仅报告 correctness-qualified、未插桩 CUDA Graph benchmark；机制归因见 "
        "[`operator_dataflow_bottleneck.md`](operator_dataflow_bottleneck.md)。",
        "",
        "| M | CuteDSL fused (us) | CUTLASS BF16 chain (us) | CuteDSL speedup vs matched | Gate | Stable |",
        "|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        formal = row["formal_comparison"]
        matched_speedup = (
            f"{row['cutedsl_speedup_vs_matched_percent']:+.2f}%"
            if formal
            else "invalid"
        )
        lines.append(
            f"| {row['m']} | {row['cutedsl_fused_us']:.3f} | "
            f"{row['cutlass_bf16_chain_us']:.3f} | {matched_speedup} | "
            f"{row['paired_oracle_gate']} | "
            f"{'yes' if row['paired_arms_spread_le_5_percent'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Speedup 定义为 `(baseline_time / cutedsl_time - 1) × 100%`。",
            "两条数据必须来自同一个 unique rerun，并匹配 shared environment、measurement protocol 与各自 artifact fingerprint。",
            "",
            "Profiler duration 不替代本表；NCU 各 launch duration 也不会相加重建 operator time。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    summary = read_rows(args.results / "benchmark_summary.csv")
    raw = read_rows(args.results / "benchmark_raw.csv")
    correctness = json.loads((args.results / "correctness.json").read_text())
    identity = json.loads((args.results / "evidence.identity.json").read_text())
    if correctness.get("evidence_identity") != identity:
        raise ValueError("evidence identity does not match correctness evidence")
    validate_raw_rerun(raw, identity)
    rows = comparison_rows(summary, correctness)
    write_csv(args.results / "comparison.csv", rows)
    (args.results / "result.md").write_text(render_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
