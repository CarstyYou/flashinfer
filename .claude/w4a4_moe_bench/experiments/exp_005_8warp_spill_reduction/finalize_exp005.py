#!/usr/bin/env python3
"""Fail-closed finalization of exp_005 evidence and manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

from exp005_common import (
    ALL_ARMS,
    BASELINE,
    CANDIDATE,
    CANONICAL_FIXTURE,
    DIRECTED_FIXTURES,
    M_VALUES,
    canonical_sha256,
    file_sha256,
    read_json,
    summarize_paired_abba,
    write_json,
)


PRODUCTION_SHA256 = "94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106"
CANDIDATE_SHA256 = "3e2bda4e09dc2c67d97abea4d392eb5fa117de9abdd80c49b0e89e1f2dd0b445"
REGISTRY_SHA256 = "d31095521339d30ead54fd9fbe10407d6585728f477b01989cb3c457d2f39c8f"
EXPECTED_CUBINS = {
    BASELINE: "9313fcbc0dd686f0684705e869fdd227608ac83ca43c1dc99d203f8e7143ca79",
    CANDIDATE: "b2bc3c4c229ebee967a6b0d3c5649bc06e3629d46793a19af845665f93683f17",
}
EXPECTED_BLOCKS = {BASELINE: [160, 1, 1], CANDIDATE: [288, 1, 1]}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_self_hash(payload: dict[str, Any], field: str) -> None:
    expected = payload.get(field)
    body = dict(payload)
    body.pop(field, None)
    require(
        isinstance(expected, str) and canonical_sha256(body) == expected,
        f"self hash failed: {field}",
    )


def artifact(results: Path, path: Path) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size > 0, f"missing artifact: {path}")
    return {
        "path": str(path.relative_to(results)),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def validate_correctness(results: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for m in M_VALUES:
        path = results / "correctness" / f"m{m}.json"
        payload = read_json(path)
        require(payload.get("schema") == "exp005.correctness.v1", f"M{m} schema")
        require(payload.get("m") == m, f"M{m} identity")
        require(payload.get("fixture_identity_pass") is True, f"M{m} fixture")
        require(
            payload.get("independent_reference_gate_pass") is True,
            f"M{m} independent reference",
        )
        require(payload.get("route_task_gate_pass") is True, f"M{m} route/task")
        strict = payload["strict_cross_arm_gate"]
        if m in (256, 8192):
            require(payload.get("gate_pass") is True, f"M{m} correctness gate")
            require(strict.get("gate_pass") is True, f"M{m} strict gate")
            classification = "pass"
        else:
            cap = strict["specifications"]["cosine_loss"]["cap"]
            require(payload.get("gate_pass") is False, "M1024 must remain failed")
            require(strict.get("gate_pass") is False, "M1024 strict gate drift")
            require(
                payload["baseline_self_drift"]["cosine_loss"] > cap
                and payload["candidate_self_drift"]["cosine_loss"] > cap
                and strict["candidate_worst_vs_baseline"]["cosine_loss"] > cap,
                "M1024 registered cosine boundary changed",
            )
            classification = "strict_comparison_failed_self_drift_inconclusive"
        output[str(m)] = {
            **artifact(results, path),
            "classification": classification,
            "independent_reference_gate_pass": True,
            "route_task_gate_pass": True,
            "strict_cross_arm_gate_pass": strict["gate_pass"],
        }
    return output


def validate_preparations(results: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    arms: dict[str, Any] = {}
    environment: dict[str, Any] | None = None
    for arm in ALL_ARMS:
        arms[arm] = {}
        cubins: set[str] = set()
        for m in M_VALUES:
            path = (
                results / "raw" / arm / f"m{m}" / CANONICAL_FIXTURE / "preparation.json"
            )
            payload = read_json(path)
            require(payload.get("status") == "complete", f"{arm} M{m} preparation")
            require(
                payload.get("arm") == arm and payload.get("m") == m,
                f"{arm} M{m} identity",
            )
            require(
                payload.get("cubin_sha256") == [EXPECTED_CUBINS[arm]],
                f"{arm} M{m} cubin",
            )
            cubins.add(payload["cubin_sha256"][0])
            launch = payload["launch_contract"]
            require(launch["expected_grid"] == [1, 1, 110], f"{arm} M{m} grid")
            require(
                launch["expected_block"] == EXPECTED_BLOCKS[arm], f"{arm} M{m} block"
            )
            require(
                launch["num_sms"] == 110 and launch["max_active_clusters"] == 110,
                f"{arm} M{m} SMS",
            )
            arms[arm][str(m)] = {
                **artifact(results, path),
                "cubin_sha256": payload["cubin_sha256"][0],
                "jit_artifact_set_sha256": payload["jit_artifact_set_sha256"],
                "expected_grid": launch["expected_grid"],
                "expected_block": launch["expected_block"],
            }
            if environment is None:
                runtime = payload["runtime"]
                environment = {
                    "gpu": runtime["gpu"],
                    "container_image_digest": runtime["image_digest"],
                    "cuda_runtime": runtime["cuda_runtime"],
                    "nvcc": runtime["nvcc"],
                    "ptxas": runtime["ptxas"],
                    "python": runtime["python"],
                    "torch": runtime["torch"],
                    "python_deps_sha256": runtime["python_deps_sha256"],
                    "cutlass_python_version": runtime["imports"][
                        "cutlass_python_version"
                    ],
                    "source": runtime["source"],
                }
        require(cubins == {EXPECTED_CUBINS[arm]}, f"{arm} cubin drift")

    directed: dict[str, Any] = {}
    for fixture in DIRECTED_FIXTURES:
        directed[fixture] = {}
        for arm in ALL_ARMS:
            path = results / "raw" / arm / "m256" / fixture / "preparation.json"
            payload = read_json(path)
            require(
                all(item["formal_pass"] for item in payload["outputs"]),
                f"{fixture} {arm} reference",
            )
            require(
                all(
                    item["verification"]["gate_pass"]
                    for item in payload["route_task_evidence"]
                ),
                f"{fixture} {arm} route oracle",
            )
            directed[fixture][arm] = {**artifact(results, path), "gate_pass": True}
    assert environment is not None
    return arms, {"environment": environment, "directed_fixtures": directed}


def validate_benchmarks(results: Path) -> dict[str, Any]:
    log_path = results / "runtime" / "paired_abba_m256_m8192.stdout.log"
    measurements: list[dict[str, Any]] = []
    for line in log_path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("schema") == "exp005.arm-measurement.v1":
            measurements.append(payload)
    require(len(measurements) == 40, "expected 40 paired benchmark measurements")
    require(
        all(
            row["status"] == "complete"
            and row["runtime"]["gpu"]["uuid"]
            == "GPU-2fdb0b79-0ba7-f356-b714-6c461b71ce12"
            and row["runtime"]["gpu"]["applications_graphics_clock_mhz"] == "2377"
            and not row["runtime"]["gpu"]["foreign_processes_before_cuda_context"]
            for row in measurements
        ),
        "benchmark runtime identity",
    )
    by_key = {
        (row["m"], row["group"], row["position"], row["arm"]): row
        for row in measurements
    }
    require(len(by_key) == 40, "duplicate benchmark sample identity")

    output: dict[str, Any] = {}
    for m in (256, 8192):
        raw_path = results / "benchmark" / f"m{m}_raw.csv"
        summary_path = results / "benchmark" / f"m{m}_summary.json"
        with raw_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == 20, f"M{m} raw sample count")
        for row in rows:
            key = (m, int(row["group"]), int(row["position"]), row["arm"])
            source = by_key[key]
            require(
                math.isclose(
                    float(row["sample_us"]),
                    source["sample_us"],
                    rel_tol=0,
                    abs_tol=1e-12,
                ),
                f"M{m} raw/log timing drift",
            )
        observed = read_json(summary_path)
        recomputed = summarize_paired_abba(rows, clock_policy="locked")
        require(
            all(observed[key] == value for key, value in recomputed.items()),
            f"M{m} benchmark summary drift",
        )
        require(
            observed["clock_gate"]["stable_single_application_clock"] is True,
            f"M{m} clock gate",
        )
        expected = "equivalent" if m == 256 else "faster"
        require(observed["verdict"] == expected, f"M{m} benchmark verdict")
        output[str(m)] = {
            "raw": artifact(results, raw_path),
            "summary": artifact(results, summary_path),
            "verdict": observed["verdict"],
            "baseline_median_us": observed["arms"][BASELINE]["median_us"],
            "candidate_median_us": observed["arms"][CANDIDATE]["median_us"],
            "speedup_percent": observed["median_speedup_percent"],
            "ratio_ci95": observed["paired_bootstrap"]["ratio_ci95"],
        }
    require(
        not (results / "benchmark" / "m1024_summary.json").exists(),
        "M1024 benchmark must remain omitted after strict gate failure",
    )
    output["provenance_log"] = artifact(results, log_path)
    return output


def validate_spill_evidence(results: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    static_path = results / "static_spill_evidence.json"
    static = read_json(static_path)
    validate_self_hash(static, "evidence_sha256")
    require(
        static["comparison"]["candidate_zero_spill_static_gate"] is False,
        "candidate static spill gate unexpectedly passed",
    )

    ncu_path = results / "ncu" / "evidence.json"
    ncu = read_json(ncu_path)
    validate_self_hash(ncu, "evidence_sha256")
    require(ncu["schema"] == "exp005.ncu-evidence.v2", "NCU evidence schema")
    require(ncu["evidence_identity_pass"] is True, "NCU evidence identity")
    require(ncu["launch_identity_pass"] is True, "NCU launch identity")
    require(ncu["work_identity_pass"] is True, "NCU work identity")
    require(ncu["candidate_static_zero_spill_pass"] is False, "NCU static gate")
    require(ncu["candidate_dynamic_zero_spill_pass"] is False, "NCU dynamic gate")
    require(ncu["candidate_zero_spill_pass"] is False, "NCU combined gate")
    require(ncu["overall_gate_pass"] is False, "NCU overall gate")
    require(
        ncu["arms"][BASELINE]["tensor_instructions"]
        == ncu["arms"][CANDIDATE]["tensor_instructions"]
        and ncu["arms"][BASELINE]["fp4_tensor_ops"]
        == ncu["arms"][CANDIDATE]["fp4_tensor_ops"],
        "NCU semantic work mismatch",
    )
    require(
        ncu["arms"][CANDIDATE]["dynamic_spill_refill_instructions"] > 0
        and ncu["arms"][CANDIDATE]["dynamic_spill_store_instructions"] > 0,
        "candidate dynamic spill unexpectedly zero",
    )
    return (
        {
            **artifact(results, static_path),
            "candidate_zero_spill_static_gate": False,
        },
        {
            **artifact(results, ncu_path),
            "canonical_capture_revision": "canonical_v1",
            "identity_gate_pass": True,
            "work_identity_gate_pass": True,
            "candidate_zero_spill_gate_pass": False,
        },
    )


def finalize(args: argparse.Namespace) -> None:
    experiment = args.experiment.resolve()
    results = experiment / "results"
    flashinfer_root = args.flashinfer_root.resolve()

    production = (
        flashinfer_root
        / "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py"
    )
    require(file_sha256(production) == PRODUCTION_SHA256, "production source changed")
    registry = experiment / "comparison_registry.json"
    require(file_sha256(registry) == REGISTRY_SHA256, "comparison registry drift")
    baseline_overlay = results / "overlays/baseline_4warp/moe_dynamic_kernel.py"
    candidate_overlay = (
        results / "overlays/candidate_8warp_serial_v0/moe_dynamic_kernel.py"
    )
    require(
        file_sha256(baseline_overlay) == PRODUCTION_SHA256, "baseline overlay drift"
    )
    require(
        file_sha256(candidate_overlay) == CANDIDATE_SHA256, "candidate overlay drift"
    )

    arms, preparation_evidence = validate_preparations(results)
    correctness = validate_correctness(results)
    benchmark = validate_benchmarks(results)
    static, ncu = validate_spill_evidence(results)
    report = artifact(results, results / "result.md")

    manifest: dict[str, Any] = {
        "schema": "exp005.final-manifest.v1",
        "status": "complete",
        "decision": "candidate_rejected_residual_spill",
        "decision_reason": (
            "Candidate A improves M8192 latency but fails both static and dynamic "
            "zero-spill gates; M1024 strict comparison is inconclusive."
        ),
        "source": {
            "production_kernel_sha256": PRODUCTION_SHA256,
            "candidate_overlay_sha256": CANDIDATE_SHA256,
            "comparison_registry": {
                "path": "../comparison_registry.json",
                "sha256": REGISTRY_SHA256,
            },
        },
        "environment": preparation_evidence["environment"],
        "arms": arms,
        "directed_fixtures": preparation_evidence["directed_fixtures"],
        "correctness": correctness,
        "benchmark": benchmark,
        "static_spill_evidence": static,
        "ncu_evidence": ncu,
        "invalidated_capture": {
            "revision": "canonical_v0",
            "reason": "missing section-derived compiler spill metrics",
            "record": "failed_attempts/attempt_01_ncu_metric_whitelist_missing_spill_derived/README.md",
        },
        "result": report,
        "evidence_boundaries": [
            "The 8-warp/layout change is an inseparable whole-kernel bundle, not a pure spill ablation.",
            "M1024 strict comparison failed because both arms exceed the preregistered cosine self-drift cap.",
            "Static program order supports investigating gate_acc first but does not provide SSA/source-line attribution.",
            "launch__stack_size is a runtime configured limit and is not the cubin static frame.",
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json(results / "manifest.json", manifest)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_experiment = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=default_experiment)
    parser.add_argument(
        "--flashinfer-root", type=Path, default=default_experiment.parents[3]
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    finalize(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
