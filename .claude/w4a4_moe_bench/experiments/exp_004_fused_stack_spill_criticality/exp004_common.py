#!/usr/bin/env python3
"""Shared, CPU-safe contracts for exp_004.

The GPU runner imports this module, but all validation and aggregation helpers
remain usable in unit tests on a machine without CUDA.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"
TARGET_RELATIVE_PATH = Path(
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py"
)
TARGET_MODULE = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dynamic_kernel"

EXPECTED_FLASHINFER_COMMIT = "074d93e4aa54c75bee1b3dfdb39b7f075a3ff2af"
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

FORMAL_ARMS = ("baseline", "activation_in_place_up", "activation_in_place_gate")
ATTRIBUTION_ARMS = ("up_first_attribution",)
ALL_ARMS = FORMAL_ARMS + ATTRIBUTION_ARMS
PRIMARY_CANDIDATE = "activation_in_place_up"
FALLBACK_CANDIDATE = "activation_in_place_gate"

M = 8192
E = 256
H = 2048
I = 512
TOPK = 8
EXPECTED_GRID = (1, 1, 110)
EXPECTED_BLOCK = (160, 1, 1)
EXPECTED_TASK_COUNT = 2536
EXPECTED_TENSOR_INSTRUCTIONS = 31_162_368
EXPECTED_FP4_TENSOR_OPS = 510_564_237_312
BASELINE_STACK_BYTES = 488
BASELINE_SPILL_WORDS = 122
MAIN_BUNDLE_WORDS = 108
TAIL_BUNDLE_WORDS = 14
SECTORS_PER_WORD = 40_576
BASELINE_LOCAL_SECTORS_PER_DIRECTION = 4_950_272
EXPECTED_TAIL_SECTOR_DELTA = TAIL_BUNDLE_WORDS * SECTORS_PER_WORD

FORBIDDEN_ENV_KEYS = (
    "CUTE_DSL_COMPILER_OPT",
    "EXP003_MARKER_OVERLAY",
    "W4A4_EXP003_MARKER_OVERLAY",
    "FLASHINFER_CUTEDSL_IKET_OVERLAY",
    "EXP003_IKET_PROVIDER_ROOT",
    "EXP003_RUN_IKET",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for row in rows:
        if set(row) != set(fieldnames):
            raise ValueError("CSV rows have inconsistent fields")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def require_clean_compiler_environment(
    environment: Mapping[str, str] | None = None,
) -> None:
    environment = os.environ if environment is None else environment
    enabled = [key for key in FORBIDDEN_ENV_KEYS if environment.get(key, "").strip()]
    if enabled:
        raise RuntimeError(
            "exp_004 requires the production/default compiler; unset "
            + ", ".join(enabled)
        )


def require_empty_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    entries = list(path.iterdir())
    if entries:
        raise RuntimeError(f"fresh per-arm JIT root is not empty: {path}")


def artifact_manifest(root: Path) -> list[dict[str, Any]]:
    suffixes = {
        ".so",
        ".cu",
        ".cuh",
        ".cpp",
        ".ptx",
        ".cubin",
        ".sass",
        ".json",
        ".mlir",
    }
    if not root.exists():
        return []
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in suffixes
    ]


def threshold_from_self_drift(self_value: float, *, floor: float, cap: float) -> float:
    if self_value < 0 or not math.isfinite(self_value):
        raise ValueError(f"invalid self drift: {self_value}")
    return min(cap, max(floor, 3.0 * self_value))


def correctness_thresholds(self_drift: Mapping[str, float]) -> dict[str, float]:
    specifications = {
        "cosine_loss": (1e-6, 1e-5),
        "relative_l2": (1e-4, 0.002),
        "max_abs": (0.002, 0.02),
        "token_rel_l2_p99": (0.001, 0.01),
    }
    missing = sorted(set(specifications) - set(self_drift))
    if missing:
        raise ValueError(f"missing baseline self-drift fields: {missing}")
    return {
        name: threshold_from_self_drift(float(self_drift[name]), floor=floor, cap=cap)
        for name, (floor, cap) in specifications.items()
    }


def evaluate_correctness_gate(
    self_drift: Mapping[str, float], candidate_worst: Mapping[str, float]
) -> dict[str, Any]:
    thresholds = correctness_thresholds(self_drift)
    missing = sorted(set(thresholds) - set(candidate_worst))
    if missing:
        raise ValueError(f"missing candidate correctness fields: {missing}")
    hard_caps = {
        "cosine_loss": 1e-5,
        "relative_l2": 0.002,
        "max_abs": 0.02,
        "token_rel_l2_p99": 0.01,
    }
    baseline_valid = all(
        float(self_drift[name]) <= cap for name, cap in hard_caps.items()
    )
    checks = {
        name: float(candidate_worst[name]) <= threshold
        for name, threshold in thresholds.items()
    }
    return {
        "baseline_self_drift_within_hard_caps": baseline_valid,
        "thresholds": thresholds,
        "candidate_worst": {name: float(candidate_worst[name]) for name in thresholds},
        "checks": checks,
        "gate_pass": baseline_valid and all(checks.values()),
    }


def expected_local_sector_delta(words: int) -> int:
    if words < 0:
        raise ValueError("word delta must be nonnegative")
    return words * SECTORS_PER_WORD


def qualify_spill_candidate(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    target_words: int = TAIL_BUNDLE_WORDS,
) -> dict[str, Any]:
    """Apply the pre-timing static/dynamic gate from the reviewed plan."""
    base_words = int(baseline["stored_words"])
    cand_words = int(candidate["stored_words"])
    removed_words = base_words - cand_words
    replacement_words = int(candidate.get("replacement_words", 0))
    static_pass = removed_words == target_words and replacement_words == 0

    dynamic_available = all(
        key in baseline and key in candidate
        for key in ("local_load_sectors", "local_store_sectors")
    )
    expected = expected_local_sector_delta(removed_words)
    dynamic_delta = None
    dynamic_pass = False
    if dynamic_available:
        dynamic_delta = {
            direction: int(baseline[f"local_{direction}_sectors"])
            - int(candidate[f"local_{direction}_sectors"])
            for direction in ("load", "store")
        }
        dynamic_pass = all(value == expected for value in dynamic_delta.values())

    work_keys = (
        "tensor_instructions",
        "fp4_tensor_ops",
        "grid",
        "block",
        "task_count",
    )
    work_checks = {
        key: baseline.get(key) == candidate.get(key) and baseline.get(key) is not None
        for key in work_keys
    }
    return {
        "target_words": target_words,
        "removed_words": removed_words,
        "replacement_words": replacement_words,
        "expected_sector_delta_per_direction": expected,
        "dynamic_sector_delta": dynamic_delta,
        "static_bundle_gate_pass": static_pass,
        "dynamic_closure_available": dynamic_available,
        "dynamic_closure_gate_pass": dynamic_pass,
        "work_identity_checks": work_checks,
        "work_identity_gate_pass": all(work_checks.values()),
        "formal_timing_eligible": static_pass
        and dynamic_pass
        and all(work_checks.values()),
    }


def summarize_paired_benchmark(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize five repeat means and apply the pre-registered materiality gate."""
    if not rows:
        raise ValueError("benchmark rows are empty")
    by_arm: dict[str, dict[int, float]] = {}
    order_by_repeat: dict[int, str] = {}
    for row in rows:
        arm = str(row["arm"])
        repeat = int(row["repeat"])
        sample_us = float(row["sample_us"])
        if not math.isfinite(sample_us) or sample_us <= 0:
            raise ValueError(f"invalid sample_us: {sample_us}")
        if repeat in by_arm.setdefault(arm, {}):
            raise ValueError(f"duplicate repeat {repeat} for arm {arm}")
        by_arm[arm][repeat] = sample_us
        order = str(row["order"])
        if repeat in order_by_repeat and order_by_repeat[repeat] != order:
            raise ValueError(f"inconsistent order for repeat {repeat}")
        order_by_repeat[repeat] = order
    if set(by_arm) != {"baseline", "candidate"}:
        raise ValueError(f"expected baseline/candidate rows, got {sorted(by_arm)}")
    repeats = sorted(by_arm["baseline"])
    if repeats != list(range(5)) or sorted(by_arm["candidate"]) != repeats:
        raise ValueError("benchmark requires exactly paired repeats 0..4")
    expected_orders = {
        repeat: ("baseline>candidate" if repeat % 2 == 0 else "candidate>baseline")
        for repeat in repeats
    }
    if order_by_repeat != expected_orders:
        raise ValueError(f"paired order drift: {order_by_repeat} != {expected_orders}")

    summaries: dict[str, dict[str, float]] = {}
    for arm, values_by_repeat in by_arm.items():
        values = [values_by_repeat[repeat] for repeat in repeats]
        median = statistics.median(values)
        spread = (max(values) - min(values)) / median
        summaries[arm] = {
            "median_us": median,
            "min_us": min(values),
            "max_us": max(values),
            "spread_fraction": spread,
            "spread_percent": spread * 100.0,
        }
    noise = max(
        summaries["baseline"]["spread_fraction"],
        summaries["candidate"]["spread_fraction"],
    )
    threshold = 3.0 * noise
    speedup = (
        summaries["baseline"]["median_us"] / summaries["candidate"]["median_us"] - 1.0
    )
    same_direction = sum(
        by_arm["candidate"][repeat] < by_arm["baseline"][repeat] for repeat in repeats
    )
    stable = all(value["spread_fraction"] <= 0.05 for value in summaries.values())
    investigate = any(value["spread_fraction"] > 0.01 for value in summaries.values())
    return {
        "arms": summaries,
        "noise_S_fraction": noise,
        "materiality_T_fraction": threshold,
        "speedup_fraction": speedup,
        "speedup_percent": speedup * 100.0,
        "candidate_faster_pair_count": same_direction,
        "stable_le_5_percent": stable,
        "investigate_spread_gt_1_percent": investigate,
        "material_improvement": stable and speedup > threshold and same_direction >= 4,
    }


def validate_manifest_schema(manifest: Mapping[str, Any]) -> list[str]:
    """Return schema violations without requiring the GPU artifacts to exist."""
    required_top = {
        "schema",
        "status",
        "case",
        "environment",
        "source",
        "fixture",
        "arms",
        "correctness",
        "benchmark",
        "static_spill",
        "ncu",
        "decision",
    }
    errors = [
        f"missing top-level field: {key}"
        for key in sorted(required_top - set(manifest))
    ]
    case = manifest.get("case", {})
    expected_case = {
        "m": M,
        "experts": E,
        "hidden": H,
        "intermediate_tp": I,
        "topk": TOPK,
    }
    for key, expected in expected_case.items():
        if case.get(key) != expected:
            errors.append(f"case.{key}: {case.get(key)!r} != {expected!r}")
    arms = manifest.get("arms", {})
    if not isinstance(arms, Mapping) or "baseline" not in arms:
        errors.append("arms.baseline is required")
    for name, arm in arms.items() if isinstance(arms, Mapping) else ():
        for key in (
            "overlay_sha256",
            "jit_root",
            "jit_artifact_set_sha256",
            "cubin_sha256",
        ):
            if key not in arm:
                errors.append(f"arms.{name}.{key} is required")
    return errors


def build_empty_manifest() -> dict[str, Any]:
    return {
        "schema": "exp004.validation-manifest.v1",
        "status": "not_started",
        "case": {
            "m": M,
            "experts": E,
            "hidden": H,
            "intermediate_tp": I,
            "topk": TOPK,
            "activation": "SwiGLU",
            "output_dtype": "bfloat16",
        },
        "environment": {},
        "source": {},
        "fixture": {},
        "arms": {},
        "correctness": {},
        "benchmark": {},
        "static_spill": {},
        "ncu": {},
        "decision": {
            "h108_attribution": "pending",
            "h108_criticality": "out_of_scope_unresolved",
            "h14_attribution": "pending",
            "h14_criticality": "pending",
        },
    }


def select_formal_candidate(
    qualifications: Mapping[str, Mapping[str, Any]],
) -> str | None:
    primary = qualifications.get(PRIMARY_CANDIDATE, {})
    if primary.get("formal_timing_eligible"):
        return PRIMARY_CANDIDATE
    fallback = qualifications.get(FALLBACK_CANDIDATE, {})
    if fallback.get("formal_timing_eligible"):
        return FALLBACK_CANDIDATE
    return None


def require_keys(value: Mapping[str, Any], keys: Iterable[str], *, label: str) -> None:
    missing = sorted(set(keys) - set(value))
    if missing:
        raise ValueError(f"{label} missing fields: {missing}")
