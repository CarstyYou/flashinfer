#!/usr/bin/env python3
"""Render the compact Chinese exp_004 result from validated derived evidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from exp004_common import DEFAULT_RESULTS, read_json


PHASE_LABELS = {
    "fc1_gate": "FC1 Gate",
    "fc1_up": "FC1 Up",
    "swiglu_q1": "SwiGLU + Q1",
    "fc2_setup": "FC2 setup",
    "fc2_gemm": "FC2 GEMM",
    "fc2_epilogue_scatter": "FC2 epilogue + scatter",
    "residual_task_control": "Residual / task control",
}


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def render(results: Path) -> tuple[str, bool]:
    gate_path = results / "derived" / "analysis_gates.json"
    gates = read_json(gate_path)
    formal = bool(gates["formal_gate_pass"])
    diagnostic = bool(gates["diagnostic_share_allowed"])
    lines = [
        "# exp_004：Fused Phase Timing Breakdown",
        "",
        "## 结论",
        "",
    ]
    if formal:
        lines.append(
            "三 binary、resource/spill、语义 work、correctness、完整事件覆盖、扰动与稳定性 gate 全部通过；下表可作为 canonical MMA-consumer phase share。"
        )
    elif diagnostic:
        lines.append(
            "核心 binary/correctness/event gate 通过，但 runtime overhead、calibration 或跨 run 稳定性至少一项未通过；下表仅是 instrumented diagnostic share，不代表 production kernel wall。"
        )
    else:
        lines.append(
            "测量插桩改变了 binary/resource/spill/semantic work，或 correctness/event coverage 未闭合。实验在 fail-closed gate 停止，不发布 phase 百分比。"
        )

    lines.extend(["", "## Gate", "", "| Gate | 结果 |", "|---|---:|"])
    for name, passed in gates["gates"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")

    phase_path = results / "derived" / "mma_phase_share.csv"
    if phase_path.is_file() and diagnostic:
        lines.extend(
            [
                "",
                "## MMA consumer phase share",
                "",
                "分母是完整 task population 中 W0–W3 的 `task_envelope` elapsed cycles；不是 CTA wall 或 kernel wall。",
                "",
                "| Phase | Share | Per warp-task p50 | p95 | Run spread | Verdict |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in _csv(phase_path):
            lines.append(
                "| {phase} | {share:.2f}% | {p50:.0f} ticks | {p95:.0f} ticks | "
                "{spread:.3f} pp | `{verdict}` |".format(
                    phase=PHASE_LABELS.get(row["phase"], row["phase"]),
                    share=float(row["share_pct"]),
                    p50=float(row["per_warp_task_p50"]),
                    p95=float(row["per_warp_task_p95"]),
                    spread=float(row["run_spread_pct"]),
                    verdict=row["verdict"],
                )
            )

        overlap = [
            row
            for row in _csv(results / "derived" / "w4_overlap.csv")
            if row["producer_phase"] == "down_tma"
        ]
        lines.extend(
            [
                "",
                "## Warp 4 Down TMA overlap",
                "",
                "这是同一 CTA/task、同一 clock64 时间轴上的 interval union；不能加到上面的 consumer share。",
                "",
                "| Consumer phase | Down TMA covered | Consumer covered | Per-task p50 | p95 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in overlap:
            lines.append(
                "| {phase} | {producer:.2f}% | {consumer:.2f}% | {p50:.2f}% | {p95:.2f}% |".format(
                    phase=PHASE_LABELS.get(
                        row["consumer_phase"], row["consumer_phase"]
                    ),
                    producer=float(row["producer_covered_pct"]),
                    consumer=float(row["consumer_covered_pct"]),
                    p50=float(row["p50"]),
                    p95=float(row["p95"]),
                )
            )

    latency = gates["latency"]
    calibration = gates["calibration"]
    lines.extend(
        [
            "",
            "## 扰动与边界",
            "",
            f"- Probe/control median latency drift：`{latency['candidate_control_median_drift_pct']:.3f}%`。",
            f"- Clock/store calibration p95 upper bound：`{calibration['total_p95_upper_bound_pct']:.3f}%` of covered consumer ticks。",
            "- Phase share 不能用于按比例拆分 NCU counters，也不直接证明 memory-bound、compute-bound 或 spill 因果。",
            "- 若 static stack/spill 或 non-probe semantic projection 漂移，raw trace 只保留为失败证据。",
            "",
        ]
    )
    return "\n".join(lines), formal


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = args.results.resolve()
    text, formal = render(results)
    output = results / "result.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(text)
    temporary.replace(output)
    return 0 if formal else 2


if __name__ == "__main__":
    raise SystemExit(main())
