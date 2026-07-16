#!/usr/bin/env python3
"""Fail-closed canonical join for exp_001's fresh paired and SGLang arms."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

M_VALUES = (256, 512, 1024, 2048, 4096, 8192)
PAIR_ARMS = ("cutedsl_bf16_fused", "cutlass_bf16_chain")
SGLANG_ARM = "sglang_triton_fp8"
ROOT = Path(__file__).resolve().parent


class EvidenceError(RuntimeError):
    pass


def speedup_percent(*, baseline_us: float, cutedsl_us: float) -> float:
    return (baseline_us / cutedsl_us - 1.0) * 100.0


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise EvidenceError(f"missing evidence: {path}")
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise EvidenceError(f"missing evidence: {path}")
    return json.loads(path.read_text())


def truth(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


def float_value(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value) or value <= 0:
        raise EvidenceError(f"invalid {field}={row[field]} in {row}")
    return value


def unique_summary(
    rows: list[dict[str, str]], arms: tuple[str, ...]
) -> dict[tuple[int, str], dict[str, str]]:
    expected = {(m, arm) for m in M_VALUES for arm in arms}
    resolved: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        key = (int(row["m"]), row["arm"])
        if key in resolved:
            raise EvidenceError(f"duplicate summary row: {key}")
        resolved[key] = row
    if set(resolved) != expected:
        raise EvidenceError(
            f"summary case/arm mismatch: missing={expected - set(resolved)}, "
            f"extra={set(resolved) - expected}"
        )
    return resolved


def validate_raw(
    rows: list[dict[str, str]], arms: tuple[str, ...], rerun_id: str
) -> None:
    expected_keys = {
        (m, repeat, arm)
        for m in M_VALUES
        for repeat in range(5)
        for arm in arms
    }
    keys: set[tuple[int, int, str]] = set()
    for row in rows:
        key = (int(row["m"]), int(row["repeat"]), row["arm"])
        if key in keys:
            raise EvidenceError(f"duplicate raw repeat: {key}")
        keys.add(key)
        float_value(row, "sample_us")
        if row["rerun_id"] != rerun_id:
            raise EvidenceError("mixed rerun ID in raw evidence")
    if keys != expected_keys:
        raise EvidenceError(
            f"raw repeat mismatch: missing={expected_keys - keys}, extra={keys - expected_keys}"
        )
    if len(arms) == 2:
        for m in M_VALUES:
            for repeat in range(5):
                paired = [
                    row
                    for row in rows
                    if int(row["m"]) == m and int(row["repeat"]) == repeat
                ]
                if {row["arm"] for row in paired} != set(arms):
                    raise EvidenceError(f"incomplete paired repeat m={m} repeat={repeat}")
                if len({row["order"] for row in paired}) != 1:
                    raise EvidenceError(f"paired order drift m={m} repeat={repeat}")


def validate_correctness(
    pair: dict[str, Any], sglang: dict[str, Any]
) -> None:
    if not pair.get("all_paired_gates_pass"):
        raise EvidenceError("paired FP4 correctness gate did not pass")
    if not sglang.get("all_correctness_and_dispatch_gates_pass"):
        raise EvidenceError("SGLang correctness/dispatch aggregate gate did not pass")
    for m in M_VALUES:
        pair_case = pair.get("cases", {}).get(str(m), {})
        if not pair_case.get("paired_gate_pass"):
            raise EvidenceError(f"paired correctness failed at m={m}")
        for arm in PAIR_ARMS:
            arm_gate = pair_case.get("arms", {}).get(arm, {})
            if not arm_gate.get("formal_pass") or not arm_gate.get("dispatch_pass"):
                raise EvidenceError(f"paired arm gate failed at m={m} arm={arm}")
            if not arm_gate.get("observed_cuda_kernels"):
                raise EvidenceError(f"missing paired dispatch evidence at m={m} arm={arm}")
        sglang_case = sglang.get("cases", {}).get(str(m), {})
        if not sglang_case.get("dispatch_pass"):
            raise EvidenceError(f"SGLang dispatch failed at m={m}")
        if not sglang_case.get("correctness", {}).get("formal_pass"):
            raise EvidenceError(f"SGLang oracle failed at m={m}")
        if not sglang_case.get("observed_cuda_kernels"):
            raise EvidenceError(f"missing SGLang kernel evidence at m={m}")


def build_rows(pair_dir: Path, sglang_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_identity = read_json(pair_dir / "evidence.identity.json")
    sglang_identity = read_json(sglang_dir / "evidence.identity.json")
    rerun_id = pair_identity.get("rerun_id")
    if not rerun_id or sglang_identity.get("rerun_id") != rerun_id:
        raise EvidenceError("pair and SGLang evidence do not share one fresh rerun ID")
    pair_environment = read_json(pair_dir / pair_identity["environment_lock_path"])
    if pair_environment["runtime"]["gpu_uuid"] != sglang_identity.get("gpu_uuid"):
        raise EvidenceError("pair and SGLang arms used different GPU UUIDs")

    pair_summary_rows = read_csv(pair_dir / "benchmark_summary.csv")
    sglang_summary_rows = read_csv(sglang_dir / "benchmark_summary.csv")
    pair_summary = unique_summary(pair_summary_rows, PAIR_ARMS)
    sglang_summary = unique_summary(sglang_summary_rows, (SGLANG_ARM,))
    validate_raw(read_csv(pair_dir / "benchmark_raw.csv"), PAIR_ARMS, rerun_id)
    validate_raw(read_csv(sglang_dir / "benchmark_raw.csv"), (SGLANG_ARM,), rerun_id)
    pair_correctness = read_json(pair_dir / "correctness.json")
    sglang_correctness = read_json(sglang_dir / "correctness.json")
    validate_correctness(pair_correctness, sglang_correctness)

    rows: list[dict[str, Any]] = []
    for m in M_VALUES:
        cute = pair_summary[(m, "cutedsl_bf16_fused")]
        cutlass = pair_summary[(m, "cutlass_bf16_chain")]
        sglang = sglang_summary[(m, SGLANG_ARM)]
        for row in (cute, cutlass, sglang):
            if row["rerun_id"] != rerun_id or not truth(row["stable_le_5_percent"]):
                raise EvidenceError(f"unstable or mixed summary row: {row}")
        fixture_fields = ("fixture_sha256", "occupancy_sha256")
        for field in fixture_fields:
            if len({cute[field], cutlass[field], sglang[field]}) != 1:
                raise EvidenceError(f"{field} mismatch across backends at m={m}")
        if "BF16 input" not in cutlass["boundary"] or "online quant" not in cutlass[
            "boundary"
        ]:
            raise EvidenceError(f"CUTLASS boundary is not BF16 online chain at m={m}")
        cute_us = float_value(cute, "median_us")
        cutlass_us = float_value(cutlass, "median_us")
        sglang_us = float_value(sglang, "median_us")
        rows.append(
            {
                "m": m,
                "cutedsl_fp4_us": cute_us,
                "cutlass_bf16_chain_us": cutlass_us,
                "sglang_triton_fp8_us": sglang_us,
                "speedup_vs_cutlass_percent": speedup_percent(
                    baseline_us=cutlass_us, cutedsl_us=cute_us
                ),
                "speedup_vs_sglang_percent": speedup_percent(
                    baseline_us=sglang_us, cutedsl_us=cute_us
                ),
                "customer_2x_target_met": speedup_percent(
                    baseline_us=sglang_us, cutedsl_us=cute_us
                )
                >= 100.0,
                "fixture_sha256": cute["fixture_sha256"],
                "rerun_id": rerun_id,
            }
        )
    context = {
        "rerun_id": rerun_id,
        "gpu_uuid": sglang_identity["gpu_uuid"],
        "pair_identity": pair_identity,
        "sglang_identity": sglang_identity,
    }
    return rows, context


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_result(rows: list[dict[str, Any]], context: dict[str, Any]) -> str:
    all_met = all(row["customer_2x_target_met"] for row in rows)
    verdict = "TARGET MET" if all_met else "TARGET NOT MET"
    lines = [
        "# Experiment 001 result: backend case sweep",
        "",
        f"**Verdict: {verdict}.** Customer criterion: CuteDSL FP4 must be at least "
        "100% faster (2x throughput by latency ratio) than SGLang Triton FP8 for every M.",
        "",
        "| M | CuteDSL FP4 (us) | CUTLASS BF16 chain (us) | SGLang Triton FP8 (us) | vs CUTLASS | vs SGLang | 2x target |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['m']} | {row['cutedsl_fp4_us']:.3f} | "
            f"{row['cutlass_bf16_chain_us']:.3f} | {row['sglang_triton_fp8_us']:.3f} | "
            f"{row['speedup_vs_cutlass_percent']:+.2f}% | "
            f"{row['speedup_vs_sglang_percent']:+.2f}% | "
            f"{'yes' if row['customer_2x_target_met'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Speedup is `(baseline_time / CuteDSL_time - 1) * 100%`. Both columns use "
            "the single CuteDSL series from the fresh paired CUTLASS rerun.",
            "",
            "CUTLASS is the matched BF16-input online-quantization chain. SGLang is the "
            "direct legacy Triton tensor-scaled W8A8 FP8 chain; its ratio is explicitly "
            "cross-runtime, not a fusion-only causal comparison.",
            "",
            f"Evidence rerun: `{context['rerun_id']}`; GPU: `{context['gpu_uuid']}`. "
            "All six per-arm correctness, dispatch, fixture, identity, and <=5% spread "
            "gates passed before publication.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_manifest(context: dict[str, Any]) -> str:
    pair = context["pair_identity"]
    sglang = context["sglang_identity"]
    return (
        "# Experiment 001 evidence manifest\n\n"
        f"- Fresh rerun ID: `{context['rerun_id']}`\n"
        f"- GPU UUID: `{context['gpu_uuid']}`\n"
        f"- Pair environment lock: `{pair['environment_lock_digest']}`\n"
        f"- Pair protocol lock: `{pair['protocol_lock_digest']}`\n"
        f"- SGLang environment lock: `{sglang['environment_lock_digest']}`\n"
        f"- SGLang protocol lock: `{sglang['protocol_lock_digest']}`\n"
        f"- SGLang artifact lock: `{sglang['artifact_fingerprint_sha256']}`\n\n"
        "Canonical evidence lives only in `results/pair/` and "
        "`results/sglang_triton/`. The old vLLM/prequant evidence is under "
        "`results/superseded_vllm_prequant/` and is not read by the builder.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", type=Path, default=ROOT / "results" / "pair")
    parser.add_argument(
        "--sglang-dir", type=Path, default=ROOT / "results" / "sglang_triton"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    rows, context = build_rows(args.pair_dir, args.sglang_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "formal.csv", rows)
    (args.output_dir / "result.md").write_text(render_result(rows, context))
    (args.output_dir / "manifest.md").write_text(render_manifest(context))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
