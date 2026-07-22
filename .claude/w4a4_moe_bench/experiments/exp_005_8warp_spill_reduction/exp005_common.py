#!/usr/bin/env python3
"""CPU-safe contracts for exp_005.

This module deliberately has no Torch/CUDA imports.  It owns the experiment
identity, route/task oracle, artifact hashing, and the pre-registered paired
ABBA statistics so those contracts can be unit tested on the frontend host.
"""

from __future__ import annotations

import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent
BENCH_ROOT = ROOT.parents[1]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from breakdown_harness.artifacts import (
    artifact_manifest,
    canonical_sha256,
    file_sha256,
    read_json,
    write_csv,
    write_json,
)
from breakdown_harness.backends.cutedsl_workspace import (
    expected_expert_tile_base,
    expected_task_records,
    expected_terminal_pair_head,
    verify_workspace_evidence,
)

__all__ = (
    "artifact_manifest",
    "canonical_sha256",
    "expected_expert_tile_base",
    "expected_task_records",
    "expected_terminal_pair_head",
    "file_sha256",
    "read_json",
    "verify_workspace_evidence",
    "write_csv",
    "write_json",
)


DEFAULT_RESULTS = ROOT / "results"
TARGET_RELATIVE_PATH = Path(
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py"
)
TARGET_MODULE = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dynamic_kernel"

EXPECTED_FLASHINFER_COMMIT = "748ad45594f5e701cbbdca59c60335f39d1c3b2f"
EXPECTED_CUTLASS_COMMIT = "b46b16d003484063bca4ed365e44095c4c6ed633"
EXPECTED_KERNEL_SHA256 = (
    "94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106"
)
EXPECTED_IMAGE_DIGEST = (
    "sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba"
)
EXPECTED_PYTHON_DEPS_SHA256 = (
    "32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74"
)

BASELINE = "baseline_4warp"
CANDIDATE = "candidate_8warp_serial_v0"
TEMPORAL_N64 = "candidate_8warp_n64_temporal_replay_v0"
ALL_ARMS = (BASELINE, CANDIDATE)
KNOWN_ARMS = ALL_ARMS + (TEMPORAL_N64,)
M_VALUES = (256, 1024, 8192)
E = 256
H = 2048
I = 512
TOPK = 8
NUM_SMS = 110
MAX_ACTIVE_CLUSTERS = 110
EXPECTED_GRID = (1, 1, 110)
EXPECTED_BLOCKS = {
    BASELINE: (160, 1, 1),
    CANDIDATE: (288, 1, 1),
    TEMPORAL_N64: (288, 1, 1),
}
TILE_M = 128
TILE_N = 128
SLICE_CHUNK = 1
RATIO_BAND = (0.98, 1.02)
CORRECTNESS_SPECS = {
    "cosine_loss": {"floor": 1e-6, "cap": 1e-4},
    "relative_l2": {"floor": 0.005, "cap": 0.03},
    "max_abs": {"floor": 0.02, "cap": 0.10},
    "token_rel_l2_p99": {"floor": 0.01, "cap": 0.05},
}
ABBA_ORDER = (BASELINE, CANDIDATE, CANDIDATE, BASELINE)
BENCHMARK_GROUPS = 5
CANONICAL_FIXTURE = "canonical"
DIRECTED_FIXTURES = ("sparse_empty", "exact_128", "tail_129", "hot_expert")
ALL_FIXTURES = (CANONICAL_FIXTURE,) + DIRECTED_FIXTURES

FORBIDDEN_ENV_KEYS = (
    "CUTE_DSL_COMPILER_OPT",
    "FLASHINFER_CUTEDSL_IKET_OVERLAY",
    "EXP003_IKET_PROVIDER_ROOT",
    "EXP003_RUN_IKET",
    "EXP003_MARKER_OVERLAY",
    "W4A4_EXP003_MARKER_OVERLAY",
)


def require_arm_m(arm: str, m: int) -> None:
    if arm not in KNOWN_ARMS:
        raise ValueError(f"unknown exp_005 arm: {arm}")
    if m not in M_VALUES:
        raise ValueError(f"M must be one of {M_VALUES}, got {m}")


def expected_block(arm: str) -> tuple[int, int, int]:
    if arm not in EXPECTED_BLOCKS:
        raise ValueError(f"unknown exp_005 arm: {arm}")
    return EXPECTED_BLOCKS[arm]


def require_clean_compiler_environment(
    environment: Mapping[str, str] | None = None,
) -> None:
    environment = os.environ if environment is None else environment
    enabled = [key for key in FORBIDDEN_ENV_KEYS if environment.get(key, "").strip()]
    if enabled:
        raise RuntimeError(
            "exp_005 requires the production/default compiler; unset "
            + ", ".join(enabled)
        )


def require_empty_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise RuntimeError(f"fresh per-arm/M JIT root is not empty: {path}")


def case_directory(results: Path, arm: str, m: int, fixture: str) -> Path:
    require_arm_m(arm, m)
    if fixture not in ALL_FIXTURES:
        raise ValueError(f"unknown fixture: {fixture}")
    return results / "raw" / arm / f"m{m}" / fixture


def correctness_thresholds(self_drift: Mapping[str, float]) -> dict[str, float]:
    missing = sorted(set(CORRECTNESS_SPECS) - set(self_drift))
    if missing:
        raise ValueError(f"missing self-drift fields: {missing}")
    thresholds: dict[str, float] = {}
    for name, specification in CORRECTNESS_SPECS.items():
        value = float(self_drift[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid self drift for {name}: {value}")
        thresholds[name] = min(
            float(specification["cap"]),
            max(float(specification["floor"]), 3.0 * value),
        )
    return thresholds


def evaluate_cross_arm_correctness(
    baseline_self_drift: Mapping[str, float],
    candidate_self_drift: Mapping[str, float],
    candidate_worst: Mapping[str, float],
) -> dict[str, Any]:
    thresholds = correctness_thresholds(baseline_self_drift)
    missing = sorted(
        (set(thresholds) - set(candidate_self_drift))
        | (set(thresholds) - set(candidate_worst))
    )
    if missing:
        raise ValueError(f"missing candidate correctness fields: {missing}")
    baseline_valid = all(
        float(baseline_self_drift[name]) <= float(specification["cap"])
        for name, specification in CORRECTNESS_SPECS.items()
    )
    candidate_stability = {
        name: float(candidate_self_drift[name]) <= threshold
        for name, threshold in thresholds.items()
    }
    cross_arm = {
        name: float(candidate_worst[name]) <= threshold
        for name, threshold in thresholds.items()
    }
    return {
        "formula": "min(cap, max(floor, 3 * baseline_self_drift))",
        "specifications": CORRECTNESS_SPECS,
        "thresholds": thresholds,
        "baseline_self_drift_within_caps": baseline_valid,
        "candidate_self_drift": dict(candidate_self_drift),
        "candidate_stability_checks": candidate_stability,
        "candidate_worst_vs_baseline": dict(candidate_worst),
        "cross_arm_checks": cross_arm,
        "gate_pass": baseline_valid
        and all(candidate_stability.values())
        and all(cross_arm.values()),
    }


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile of empty sequence")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _arm_statistics(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"invalid timing values: {values}")
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "median_us": statistics.median(values),
        "p10_us": _quantile(values, 0.10),
        "p90_us": _quantile(values, 0.90),
        "mean_us": mean,
        "cv": statistics.pstdev(values) / mean if len(values) > 1 else 0.0,
    }


def classify_ratio_ci(
    lower: float, upper: float, *, clock_policy: str
) -> tuple[str, str]:
    if clock_policy not in ("locked", "unlocked"):
        raise ValueError("clock_policy must be locked or unlocked")
    band_low, band_high = RATIO_BAND
    if lower > band_high:
        statistical = "faster"
    elif upper < band_low:
        statistical = "slower"
    elif lower >= band_low and upper <= band_high:
        statistical = "equivalent"
    else:
        statistical = "inconclusive"
    verdict = statistical if clock_policy == "locked" else "advisory_inconclusive"
    return statistical, verdict


def summarize_paired_abba(
    rows: Sequence[Mapping[str, Any]],
    *,
    clock_policy: str,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260717,
) -> dict[str, Any]:
    """Summarize five paired ABBA groups and bootstrap the ratio CI."""
    expected_keys = {
        (group, position): arm
        for group in range(BENCHMARK_GROUPS)
        for position, arm in enumerate(ABBA_ORDER)
    }
    seen: dict[tuple[int, int], Mapping[str, Any]] = {}
    by_arm: dict[str, list[float]] = {BASELINE: [], CANDIDATE: []}
    for row in rows:
        key = (int(row["group"]), int(row["position"]))
        if key in seen:
            raise ValueError(f"duplicate benchmark row: {key}")
        expected_arm = expected_keys.get(key)
        arm = str(row["arm"])
        if expected_arm is None or arm != expected_arm:
            raise ValueError(f"ABBA order drift at {key}: {arm} != {expected_arm}")
        sample = float(row["sample_us"])
        if not math.isfinite(sample) or sample <= 0:
            raise ValueError(f"invalid sample_us: {sample}")
        seen[key] = row
        by_arm[arm].append(sample)
    if set(seen) != set(expected_keys):
        missing = sorted(set(expected_keys) - set(seen))
        raise ValueError(
            f"benchmark requires five complete ABBA groups; missing {missing}"
        )

    paired: list[dict[str, float | int]] = []
    for group in range(BENCHMARK_GROUPS):
        baseline = statistics.fmean(
            float(seen[(group, position)]["sample_us"]) for position in (0, 3)
        )
        candidate = statistics.fmean(
            float(seen[(group, position)]["sample_us"]) for position in (1, 2)
        )
        paired.append(
            {
                "group": group,
                "baseline_us": baseline,
                "candidate_us": candidate,
                "ratio": baseline / candidate,
            }
        )

    rng = random.Random(bootstrap_seed)
    bootstrap_ratios: list[float] = []
    for _ in range(bootstrap_samples):
        sample = [paired[rng.randrange(len(paired))] for _ in paired]
        bootstrap_ratios.append(
            statistics.fmean(float(item["baseline_us"]) for item in sample)
            / statistics.fmean(float(item["candidate_us"]) for item in sample)
        )
    ci_low = _quantile(bootstrap_ratios, 0.025)
    ci_high = _quantile(bootstrap_ratios, 0.975)
    ratio = statistics.median(by_arm[BASELINE]) / statistics.median(by_arm[CANDIDATE])
    statistical, verdict = classify_ratio_ci(ci_low, ci_high, clock_policy=clock_policy)
    return {
        "schema": "exp005.paired-abba.v1",
        "groups": paired,
        "arms": {arm: _arm_statistics(values) for arm, values in by_arm.items()},
        "median_ratio_baseline_over_candidate": ratio,
        "median_speedup_percent": (ratio - 1.0) * 100.0,
        "paired_bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "ratio_ci95": [ci_low, ci_high],
        },
        "predeclared_ratio_band": list(RATIO_BAND),
        "clock_policy": clock_policy,
        "statistical_classification": statistical,
        "verdict": verdict,
    }
