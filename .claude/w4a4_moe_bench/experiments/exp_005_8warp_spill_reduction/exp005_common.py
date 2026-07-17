#!/usr/bin/env python3
"""CPU-safe contracts for exp_005.

This module deliberately has no Torch/CUDA imports.  It owns the experiment
identity, route/task oracle, artifact hashing, and the pre-registered paired
ABBA statistics so those contracts can be unit tested on the frontend host.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
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
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError("CSV rows have inconsistent fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


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


def case_directory(results: Path, arm: str, m: int, fixture: str) -> Path:
    require_arm_m(arm, m)
    if fixture not in ALL_FIXTURES:
        raise ValueError(f"unknown fixture: {fixture}")
    return results / "raw" / arm / f"m{m}" / fixture


def expected_expert_tile_base(
    row_counts: Sequence[int], *, tile_m: int = TILE_M
) -> list[int]:
    base = 0
    result = [0]
    for raw_count in row_counts:
        count = int(raw_count)
        if count < 0:
            raise ValueError("row count cannot be negative")
        base += (count + tile_m - 1) // tile_m
        result.append(base)
    return result


def expected_task_records(
    row_counts: Sequence[int],
    *,
    n: int = I,
    tile_m: int = TILE_M,
    tile_n: int = TILE_N,
    slice_chunk: int = SLICE_CHUNK,
) -> list[tuple[int, int, int, int, int]]:
    """Return (expert, physical_m_tile, slice_begin, slice_count, valid_rows)."""
    if n <= 0 or n % tile_n:
        raise ValueError("n must be a positive multiple of tile_n")
    if slice_chunk <= 0:
        raise ValueError("slice_chunk must be positive")
    bases = expected_expert_tile_base(row_counts, tile_m=tile_m)
    gate_tiles = n // tile_n
    records: list[tuple[int, int, int, int, int]] = []
    for expert, raw_count in enumerate(row_counts):
        count = int(raw_count)
        local_tile = 0
        rows_remaining = count
        while rows_remaining > 0:
            valid_rows = min(tile_m, rows_remaining)
            for slice_begin in range(0, gate_tiles, slice_chunk):
                records.append(
                    (
                        expert,
                        bases[expert] + local_tile,
                        slice_begin,
                        min(slice_chunk, gate_tiles - slice_begin),
                        valid_rows,
                    )
                )
            rows_remaining -= tile_m
            local_tile += 1
    return records


def expected_terminal_pair_head(
    routed_rows: int, *, num_cta_warps: int, grid_z: int = NUM_SMS
) -> int:
    """Expected terminal producer-queue head, including one miss per CTA.

    Each CTA claims ``num_cta_warps * _PRODUCER_PAIRS_PER_WARP`` entries at a
    time.  The locked kernel uses two pairs per warp and every resident CTA
    performs one final claim whose base is already outside the routed range.
    This scheduling counter is arm-dependent and is not a logical work count.
    """
    if routed_rows < 0 or num_cta_warps <= 0 or grid_z <= 0:
        raise ValueError("invalid producer-head geometry")
    claim_count = num_cta_warps * 2
    productive_claims = (routed_rows + claim_count - 1) // claim_count
    return (productive_claims + grid_z) * claim_count


def verify_workspace_evidence(
    snapshot: Mapping[str, Any],
    *,
    expected_row_counts: Sequence[int],
    num_cta_warps: int = 5,
    grid_z: int = NUM_SMS,
) -> dict[str, Any]:
    """Validate runtime route/task arrays without importing Torch.

    The current production scheduler uses an append-only task array followed by
    an atomic task-head claim.  It does not keep a per-task consumed bitmap, so
    exact-once consumption is an inference from the descriptor multiset plus
    the terminal atomic-head state, not a direct per-slot execution counter.
    """
    expected_rows = [int(value) for value in expected_row_counts]
    if len(expected_rows) != E:
        raise ValueError(f"expected {E} row counts, got {len(expected_rows)}")
    row_counts = [int(value) for value in snapshot["row_counts"]]
    write_rows = [int(value) for value in snapshot["expert_write_rows"]]
    tile_base = [int(value) for value in snapshot["expert_tile_base"]]
    tail = int(snapshot["task_tail"])
    head = int(snapshot["task_head"])
    task_fields = (
        "task_expert",
        "task_m_tile",
        "task_slice_begin",
        "task_slice_count",
        "task_valid_rows",
    )
    lengths = {field: len(snapshot[field]) for field in task_fields}
    if any(length < tail for length in lengths.values()):
        raise ValueError(f"task arrays shorter than task_tail={tail}: {lengths}")
    observed = [
        tuple(int(snapshot[field][index]) for field in task_fields)
        for index in range(tail)
    ]
    expected = expected_task_records(expected_rows)
    expected_base = expected_expert_tile_base(expected_rows)
    routed_rows = int(snapshot["routed_rows"])
    expected_pair_head = expected_terminal_pair_head(
        routed_rows, num_cta_warps=num_cta_warps, grid_z=grid_z
    )
    missing = list((Counter(expected) - Counter(observed)).elements())
    unexpected = list((Counter(observed) - Counter(expected)).elements())
    checks = {
        "routed_row_sum": sum(row_counts) == sum(expected_rows) == routed_rows,
        "pair_head_terminal_state": int(snapshot.get("pair_head", -1))
        == expected_pair_head,
        "row_counts": row_counts == expected_rows,
        "expert_write_rows": write_rows == expected_rows,
        "expert_tile_base": tile_base == expected_base,
        "task_tail": tail == len(expected),
        "task_descriptor_multiset": not missing and not unexpected,
        "all_work_published": int(snapshot.get("all_work_published", 0)) == 1,
        # full_tile_publish_enabled is 0 in the locked source.  Every CTA makes
        # one terminal out-of-range atomic claim, hence tail + grid_z.
        "atomic_head_terminal_state": head == tail + grid_z,
    }
    return {
        "checks": checks,
        "gate_pass": all(checks.values()),
        "expected_task_count": len(expected),
        "observed_task_tail": tail,
        "observed_task_head": head,
        "terminal_head_overshoot": head - tail,
        "producer_claim_count": num_cta_warps * 2,
        "expected_pair_head": expected_pair_head,
        "observed_pair_head": int(snapshot.get("pair_head", -1)),
        "missing_task_descriptors": [list(item) for item in missing[:32]],
        "unexpected_task_descriptors": [list(item) for item in unexpected[:32]],
        "task_descriptor_order_sha256": canonical_sha256(observed),
        "task_descriptor_multiset_sha256": canonical_sha256(sorted(observed)),
        "exact_once_evidence": (
            "append-only descriptor multiset + terminal atomic-head inference; "
            "no direct per-task consumed bitmap exists"
        ),
    }


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
