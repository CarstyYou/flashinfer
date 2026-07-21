#!/usr/bin/env python3
"""Build the compact, fail-closed exp_018 three-arm benchmark result."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ARMS = ("sglang_triton_fp8", "latest_opt_fp4", "eric_stage4_fp4")
BLOCKS = (0, 1, 2)
M_VALUES = (256, 512, 1024, 2048, 4096, 8192)
SCHEMA = "exp018.arm-block.v1"
EXPECTED_FIXTURE_SHA256 = {
    256: "86b505097acd06bed5a50c3528c78525e6087c07ed69f86606607599ffa21686",
    512: "e6ddb487121a0d681a06bcb453f38623b3d5d8477f2232bbbf78cd2ea4ef23a3",
    1024: "0fa7e8a7d8d1d32172971f987d6f55b534aabf8d12a84a910d010cec25ba04a5",
    2048: "5375fd8b3e5e15f8c956998bfc3e2f3ee59948a2aeaf5ba1294ec6a74092bde3",
    4096: "a1ac93cb8dfb2e81a000476efc36b75588f79f1954b406396b77c172464ce2cc",
    8192: "c113ecd5ddeff77154ddbd23fc3dc3c83f8ee822e880179ca5c16b1145372438",
}
EXPECTED_MANIFEST_SHA256 = (
    "683ec75341e4d8317dfdc5c4b04229f9695f9aa286d575c4f6e1fdef55d90801"
)


class EvidenceError(RuntimeError):
    """The persisted evidence cannot support the requested comparison."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing evidence: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    require(all(set(row) == set(fields) for row in rows), "CSV schema drift")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_blocks(results: Path) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    rerun_ids: set[str] = set()
    protocol_hashes: set[str] = set()
    for arm in ARMS:
        for block in BLOCKS:
            path = results / "raw" / arm / f"block_{block}.json"
            value = read_json(path)
            label = f"{arm}/block{block}"
            require(value.get("schema") == SCHEMA, f"{label} schema drift")
            require(value.get("arm") == arm, f"{label} arm drift")
            require(value.get("block") == block, f"{label} block drift")
            require(value.get("telemetry_gate", {}).get("pass") is True,
                    f"{label} telemetry gate failed")
            require(value.get("block_status") in {"complete", "complete_with_invalid_cells"},
                    f"{label} is not complete")
            rerun_id = value.get("rerun_id")
            protocol_hash = value.get("protocol_sha256")
            require(isinstance(rerun_id, str) and rerun_id, f"{label} missing rerun ID")
            require(isinstance(protocol_hash, str) and len(protocol_hash) == 64,
                    f"{label} missing protocol hash")
            rerun_ids.add(rerun_id)
            protocol_hashes.add(protocol_hash)
            cells = value.get("cells")
            require(isinstance(cells, list) and len(cells) == len(M_VALUES),
                    f"{label} must contain exactly six cells")
            by_m = {int(cell.get("m")): cell for cell in cells}
            require(tuple(sorted(by_m)) == M_VALUES, f"{label} M coverage drift")
            require(len(by_m) == len(cells), f"{label} duplicate M cell")
            value["_cells_by_m"] = by_m
            records[(arm, block)] = value
    require(len(rerun_ids) == 1, f"mixed rerun IDs: {sorted(rerun_ids)}")
    require(len(protocol_hashes) == 1,
            f"mixed measurement protocols: {sorted(protocol_hashes)}")
    return records


def validate_identity(records: dict[tuple[str, int], dict[str, Any]]) -> None:
    for arm in ARMS:
        source = [records[(arm, block)].get("source_identity") for block in BLOCKS]
        weights = [records[(arm, block)].get("weight_identity") for block in BLOCKS]
        jit = [records[(arm, block)].get("jit_identity") for block in BLOCKS]
        require(all(value == source[0] for value in source[1:]),
                f"{arm} source identity drift across blocks")
        require(all(value == weights[0] for value in weights[1:]),
                f"{arm} weight identity drift across blocks")
        require(all(value == jit[0] for value in jit[1:]),
                f"{arm} JIT identity drift across blocks")
        require(isinstance(source[0], dict) and source[0], f"{arm} missing source identity")
        require(isinstance(weights[0], dict) and weights[0], f"{arm} missing weight identity")
        require(isinstance(jit[0], dict) and jit[0], f"{arm} missing JIT identity")

    # The two FP4 implementations must consume the same packed weights/scales.
    opt_weights = records[("latest_opt_fp4", 0)]["weight_identity"]
    eric_weights = records[("eric_stage4_fp4", 0)]["weight_identity"]
    for key in ("seed", "packed_weights_sha256", "scales_sha256"):
        require(key in opt_weights and opt_weights.get(key) == eric_weights.get(key),
                f"FP4 shared weight identity mismatch at {key}")


def validate_cell(cell: dict[str, Any], arm: str, block: int, m: int) -> str:
    label = f"{arm}/block{block}/M{m}"
    require(cell.get("fixture_manifest_sha256") == EXPECTED_MANIFEST_SHA256,
            f"{label} fixture manifest drift")
    require(cell.get("fixture_sha256") == EXPECTED_FIXTURE_SHA256[m],
            f"{label} NPZ fixture drift")
    status = str(cell.get("status", "")).lower()
    require(status in {"pass", "invalid", "inconclusive"},
            f"{label} unknown status: {status}")
    correctness = cell.get("correctness")
    require(isinstance(correctness, dict), f"{label} missing correctness")
    expected_mode = "full_oracle" if block == 0 else "sentinel_sanity"
    require(correctness.get("mode") == expected_mode,
            f"{label} correctness mode drift")
    if status == "pass":
        require(correctness.get("qualification_pass") is True,
                f"{label} pass without correctness qualification")
        sample = cell.get("sample_us")
        require(isinstance(sample, (int, float)) and math.isfinite(sample) and sample > 0,
                f"{label} invalid sample")
    else:
        require(cell.get("sample_us") is None,
                f"{label} non-pass cell must not expose latency")
        require(isinstance(cell.get("reason"), str) and cell["reason"],
                f"{label} non-pass cell missing reason")
    return status


def aggregate(records: dict[tuple[str, int], dict[str, Any]]):
    raw_rows: list[dict[str, Any]] = []
    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for m in M_VALUES:
            cells = [records[(arm, block)]["_cells_by_m"][m] for block in BLOCKS]
            statuses = [validate_cell(cell, arm, block, m)
                        for block, cell in zip(BLOCKS, cells, strict=True)]
            for block, cell, status in zip(BLOCKS, cells, statuses, strict=True):
                raw_rows.append({
                    "rerun_id": records[(arm, block)]["rerun_id"],
                    "protocol_sha256": records[(arm, block)]["protocol_sha256"],
                    "arm": arm,
                    "block": block,
                    "m": m,
                    "status": status,
                    "sample_us": "" if cell.get("sample_us") is None else cell["sample_us"],
                    "reason": cell.get("reason") or "",
                    "fixture_sha256": cell["fixture_sha256"],
                })
            if "invalid" in statuses:
                status = "Invalid"
                reason = next(cell["reason"] for cell in cells
                              if str(cell["status"]).lower() == "invalid")
                samples: list[float] = []
            elif "inconclusive" in statuses:
                status = "Inconclusive"
                reason = next(cell["reason"] for cell in cells
                              if str(cell["status"]).lower() == "inconclusive")
                samples = []
            else:
                status = "Pass"
                reason = ""
                samples = [float(cell["sample_us"]) for cell in cells]
            if samples:
                median_us = statistics.median(samples)
                spread = (max(samples) - min(samples)) / median_us * 100.0
                if spread > 5.0:
                    status = "Inconclusive"
                    reason = f"repeat spread {spread:.2f}% exceeds 5%"
                else:
                    reason = ""
            else:
                median_us = None
                spread = None
            summary = {
                "arm": arm,
                "m": m,
                "status": status,
                "median_us": median_us,
                "min_us": min(samples) if samples else None,
                "max_us": max(samples) if samples else None,
                "spread_percent": spread,
                "reason": reason,
                "samples": samples,
            }
            summaries[(arm, m)] = summary
            summary_rows.append({
                "arm": arm,
                "m": m,
                "status": status,
                "median_us": "" if median_us is None else median_us,
                "min_us": "" if not samples else min(samples),
                "max_us": "" if not samples else max(samples),
                "spread_percent": "" if spread is None else spread,
                "reason": reason,
            })
    return raw_rows, summary_rows, summaries


def speedup(subject: dict[str, Any], baseline: dict[str, Any]) -> float | None:
    if subject["status"] != "Pass" or baseline["status"] != "Pass":
        return None
    return (float(baseline["median_us"]) / float(subject["median_us"]) - 1.0) * 100.0


def latency_cell(row: dict[str, Any]) -> str:
    if row["status"] == "Pass":
        return f"{row['median_us']:.3f} µs"
    reason = str(row["reason"]).replace("|", "/")
    return f"**{row['status']}** ({reason})"


def speedup_cell(value: float | None, *, target_2x: bool = False) -> str:
    if value is None:
        return "—"
    suffix = "；✓ 2×" if target_2x and value >= 100.0 else ""
    if target_2x and value < 100.0:
        suffix = "；未达 2×"
    return f"{value:+.2f}%{suffix}"


def eric_vs_opt_cell(records, summaries, m: int) -> str:
    eric = summaries[("eric_stage4_fp4", m)]
    opt = summaries[("latest_opt_fp4", m)]
    value = speedup(eric, opt)
    if value is None:
        return "—"
    ratios = []
    for block in BLOCKS:
        e = records[("eric_stage4_fp4", block)]["_cells_by_m"][m]
        o = records[("latest_opt_fp4", block)]["_cells_by_m"][m]
        ratios.append((float(o["sample_us"]) / float(e["sample_us"]) - 1.0) * 100.0)
    paired_median = statistics.median(ratios)
    consistent_faster = all(item > 0 for item in ratios) and paired_median > 2.0
    consistent_slower = all(item < 0 for item in ratios) and paired_median < -2.0
    threshold_straddle = (consistent_faster and value <= 2.0) or (
        consistent_slower and value >= -2.0
    )
    if threshold_straddle:
        verdict = "阈值边界/无定论"
    elif consistent_faster:
        verdict = "更快"
    elif consistent_slower:
        verdict = "更慢"
    else:
        verdict = "相当/无定论"
    return f"{value:+.2f}%（{verdict}）"


def render_result(records, summaries) -> str:
    eric_all_pass = all(
        summaries[("eric_stage4_fp4", m)]["status"] == "Pass" for m in M_VALUES
    )
    lines = [
        "# exp_018：Triton FP8 vs Latest Opt FP4 vs Eric Stage4 FP4",
        "",
        f"Eric 六个 prefill case correctness：**{'全部通过' if eric_all_pass else '未全部通过'}**。",
        "",
        "| M | Triton FP8 | Latest Opt FP4 | Eric Stage4 FP4 | Opt vs Triton | Eric vs Triton | Eric vs Opt |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in M_VALUES:
        triton = summaries[("sglang_triton_fp8", m)]
        opt = summaries[("latest_opt_fp4", m)]
        eric = summaries[("eric_stage4_fp4", m)]
        lines.append(
            f"| {m} | {latency_cell(triton)} | {latency_cell(opt)} | "
            f"{latency_cell(eric)} | {speedup_cell(speedup(opt, triton), target_2x=True)} | "
            f"{speedup_cell(speedup(eric, triton), target_2x=True)} | "
            f"{eric_vs_opt_cell(records, summaries, m)} |"
        )
    lines.extend([
        "",
        "`Speedup = baseline_latency / subject_latency - 1`。每格为三个 cyclic process block 的 median；",
        "每个 block 使用 warmup=5、timed=50、192 MiB L2 flush 和 CUDA Graph external-event timing。",
        "Triton FP8 直接调用 SGLang legacy `fused_experts_impl`；本机没有 shape-specific config，",
        "六个 case 的实际 config source 均为 `default_heuristic`。",
        "M512 的 paired-ratio median（+2.006%）与 ratio-of-medians（+1.947%）分居 2% 阈值两侧，",
        "审计后按“阈值边界/无定论”处理。",
        "",
        "原始数据见 [benchmark_raw.csv](benchmark_raw.csv)，聚合数据见 "
        "[benchmark_summary.csv](benchmark_summary.csv)。",
        "",
    ])
    return "\n".join(lines)


def build(results: Path) -> None:
    records = load_blocks(results)
    validate_identity(records)
    raw_rows, summary_rows, summaries = aggregate(records)
    write_csv(results / "benchmark_raw.csv", raw_rows)
    write_csv(results / "benchmark_summary.csv", summary_rows)
    (results / "result.md").write_text(render_result(records, summaries))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    build(args.results.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
