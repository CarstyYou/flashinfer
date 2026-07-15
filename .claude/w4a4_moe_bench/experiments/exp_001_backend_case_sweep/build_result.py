#!/usr/bin/env python3
"""Build the canonical three-backend result with one CuteDSL reference."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
M_VALUES = (256, 512, 1024, 2048, 4096, 8192)
TARGET_PERCENT = 100.0
WITHDRAWN_HISTORICAL_TRITON_US = {
    256: 733.2,
    512: 739.3,
    1024: 763.1,
    2048: 849.5,
    4096: 1102.5,
    8192: 1724.0,
}


class EvidenceError(RuntimeError):
    pass


def speedup_percent(*, baseline_us: float, cutedsl_us: float) -> float:
    return (baseline_us / cutedsl_us - 1.0) * 100.0


def read_backend(
    path: Path, expected_backend: str, *, mixed_backend_file: bool = False
) -> dict[int, dict[str, str]]:
    with path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    by_m: dict[int, dict[str, str]] = {}
    for row in rows:
        if row.get("backend") != expected_backend:
            if mixed_backend_file:
                continue
            raise EvidenceError(
                f"{path}: backend={row.get('backend')!r}, expected {expected_backend!r}"
            )
        if row.get("error"):
            raise EvidenceError(f"{path}: failed row: {row['error']}")
        m = int(row["m"])
        if m not in M_VALUES:
            continue
        if m in by_m:
            raise EvidenceError(f"{path}: duplicate {expected_backend} M={m}")
        by_m[m] = row
    if tuple(sorted(by_m)) != M_VALUES:
        raise EvidenceError(
            f"{path}: {expected_backend} cases={tuple(sorted(by_m))}, "
            f"expected={M_VALUES}"
        )
    return by_m


def validate_contract(rows: dict[int, dict[str, str]]) -> None:
    expected = {
        "hidden": "2048",
        "intermediate_tp": "512",
        "experts": "256",
        "topk": "8",
        "timing": "cuda_graph_events_inside",
        "flush_l2": "1",
        "l2_flush_bytes": "201326592",
        "warmup": "5",
        "iters": "50",
        "repeats": "5",
    }
    for m, row in rows.items():
        for key, value in expected.items():
            if row.get(key) != value:
                raise EvidenceError(
                    f"M={m}: {key}={row.get(key)!r}, expected {value!r}"
                )
        median = float(row["median_us"])
        samples = [float(value) * 1000.0 for value in row["samples_ms"].split(";")]
        if not math.isfinite(median) or median <= 0 or len(samples) != 5:
            raise EvidenceError(f"M={m}: invalid median/samples")
        if any(not math.isfinite(value) or value <= 0 for value in samples):
            raise EvidenceError(f"M={m}: non-finite sample")
        spread = (max(samples) - min(samples)) / median
        if spread > 0.05:
            raise EvidenceError(f"M={m}: unstable sample spread {spread:.2%}")


def compare(
    cutedsl: dict[int, dict[str, str]],
    cutlass: dict[int, dict[str, str]],
    triton: dict[int, dict[str, str]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for m in M_VALUES:
        for key in ("fixture_sha256", "occupancy_sha256", "gpu_uuid"):
            if not triton[m].get(key):
                raise EvidenceError(f"M={m}: Triton {key} missing")
        if triton[m].get("functional_sanity") != "shape_dtype_finite_nonzero":
            raise EvidenceError(f"M={m}: Triton functional sanity missing")

        cutedsl_us = float(cutedsl[m]["median_us"])
        cutlass_us = float(cutlass[m]["median_us"])
        triton_us = float(triton[m]["median_us"])
        cutlass_speedup = speedup_percent(baseline_us=cutlass_us, cutedsl_us=cutedsl_us)
        triton_speedup = speedup_percent(baseline_us=triton_us, cutedsl_us=cutedsl_us)
        output.append(
            {
                "m": m,
                "cutedsl_source_arm": "cutlass_arm",
                "cutedsl_us": cutedsl_us,
                "cutlass_us": cutlass_us,
                "speedup_vs_cutlass_percent": cutlass_speedup,
                "triton_fp8_us": triton_us,
                "speedup_vs_triton_percent": triton_speedup,
                "triton_target_gap_pp": triton_speedup - TARGET_PERCENT,
                "triton_target_met": triton_speedup >= TARGET_PERCENT,
            }
        )
    return output


def write_outputs(output_dir: Path, rows: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "formal.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    all_met = all(bool(row["triton_target_met"]) for row in rows)
    failed_m = [int(row["m"]) for row in rows if not row["triton_target_met"]]
    if all_met:
        verdict_lines = [
            "**Verdict:** the evaluated `100%` CuteDSL speedup threshold versus the vLLM",
            "Triton FP8 reference is **met** for all prefill cases.",
        ]
    else:
        failed_text = ", ".join(f"`M={m}`" for m in failed_m)
        passed_m = [int(row["m"]) for row in rows if row["triton_target_met"]]
        passed_text = ", ".join(f"`M={m}`" for m in passed_m)
        verdict_lines = [
            "**Verdict:** the evaluated `100%` CuteDSL speedup threshold versus the vLLM",
            f"Triton FP8 reference is **not met** for all prefill cases. {failed_text}",
            f"fail; {passed_text} pass.",
        ]
    historical_deltas = [
        f"`M={row['m']}` "
        f"{(float(row['triton_fp8_us']) / WITHDRAWN_HISTORICAL_TRITON_US[int(row['m'])] - 1) * 100:+.2f}%"
        for row in rows
    ]
    lines = [
        "# Experiment 001 Result: CUTLASS vs CuteDSL vs Triton",
        "",
        *verdict_lines,
        "",
        "`Speedup = (baseline time / CuteDSL time - 1) * 100%`.",
        "",
        "| M | CuteDSL (us) | CUTLASS (us) | Speedup vs CUTLASS | vLLM Triton FP8 (us) | Speedup vs Triton | 100% threshold |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['m']} | {row['cutedsl_us']:.3f} | {row['cutlass_us']:.3f} | "
            f"{row['speedup_vs_cutlass_percent']:.2f}% | "
            f"{row['triton_fp8_us']:.3f} | "
            f"{row['speedup_vs_triton_percent']:.2f}% | "
            f"{'PASS' if row['triton_target_met'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Both speedups use the single CuteDSL measurement from the CUTLASS arm.",
            "Positive speedup means the baseline is slower than CuteDSL.",
            "",
            "## Scope",
            "",
            "- Prefill only: `M={256,512,1024,2048,4096,8192}`.",
            "- CUTLASS and Triton were measured in separate arms on the same GPU class.",
            "  Speedup vs Triton is therefore a declared cross-arm ratio, not a paired",
            "  same-host comparison.",
            "- Canonical CuteDSL and Triton do not share recorded fixture/routing",
            "  identity; the CUTLASS-arm CSV does not carry those identity hashes.",
            "- CUTLASS excludes BF16 input quantization from its timed closure; CuteDSL",
            "  includes online input quantization.",
            "- Triton is vLLM `0.11.1rc1` legacy tensor-scaled W8A8 using an untuned",
            "  default heuristic, not a confirmed customer production recipe.",
            "- Measured Triton deltas versus the withdrawn historical W8A8 table",
            "  (not used in speedup calculations):",
            f"  - {', '.join(historical_deltas[:3])}",
            f"  - {', '.join(historical_deltas[3:])}",
            "- This is performance-only evidence and makes no FP4/FP8 numerical-equivalence",
            "  claim.",
            "",
            "## Evidence",
            "",
            "- [`formal.csv`](formal.csv): canonical three-backend result data.",
            "- [`cutlass_arm_raw.csv`](cutlass_arm_raw.csv): CUTLASS/CuteDSL raw arm.",
            "- [`triton_arm_raw.csv`](triton_arm_raw.csv): Triton FP8 raw arm.",
            "- [`manifest.md`](manifest.md): setup, source, stability, and evidence identity.",
            "- [`plan.md`](../plan.md): current experiment contract.",
        ]
    )
    (output_dir / "result.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cutlass-arm-raw", type=Path, default=RESULTS / "cutlass_arm_raw.csv"
    )
    parser.add_argument(
        "--triton-arm-raw", type=Path, default=RESULTS / "triton_arm_raw.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()

    cutedsl = read_backend(
        args.cutlass_arm_raw, "flashinfer_cutedsl", mixed_backend_file=True
    )
    cutlass = read_backend(args.cutlass_arm_raw, "cutlass", mixed_backend_file=True)
    triton = read_backend(args.triton_arm_raw, "triton_fp8")
    for rows in (cutedsl, cutlass, triton):
        validate_contract(rows)
    write_outputs(args.output_dir, compare(cutedsl, cutlass, triton))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
