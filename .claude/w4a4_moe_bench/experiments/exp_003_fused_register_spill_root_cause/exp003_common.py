#!/usr/bin/env python3
"""Shared, CPU-safe contracts for exp_003.

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
            "exp_003 requires the production/default compiler; unset "
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
    schema = manifest.get("schema")
    run_schemas = {"exp003.spill-root-cause.run-manifest.v1"}
    closed_schema = "exp003.spill-root-cause.validation-manifest.v1"
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
    if schema == closed_schema:
        required_top.update(
            {
                "capture_provenance",
                "root_cause_evidence",
                "resource_cleanup",
                "retained_attribution_evidence",
            }
        )
    elif schema not in run_schemas:
        return [f"unsupported manifest schema: {schema!r}"]
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
    if not isinstance(arms, Mapping) or (
        manifest.get("status") != "not_started" and "baseline" not in arms
    ):
        errors.append("arms.baseline is required")
    if schema in run_schemas:
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

    if manifest.get("status") != "formation_mechanism_closed":
        errors.append("closed manifest status must be formation_mechanism_closed")
    expected_arms = {"baseline", "up_first_attribution"}
    if not isinstance(arms, Mapping) or set(arms) != expected_arms:
        errors.append(f"closed manifest arms must equal {sorted(expected_arms)}")

    def valid_sha(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    arm_hash_fields = (
        "preparation_sha256",
        "jit_artifact_set_sha256",
        "overlay_sha256",
        "overlay_diff_sha256",
        "cubin_sha256",
        "ptx_sha256",
        "mlir_sha256",
        "ncu_trace_sha256",
    )
    if isinstance(arms, Mapping):
        for name, arm in arms.items():
            if not isinstance(arm, Mapping):
                errors.append(f"arms.{name} must be an object")
                continue
            if not isinstance(arm.get("preparation"), str):
                errors.append(f"arms.{name}.preparation is required")
            for field in arm_hash_fields:
                if not valid_sha(arm.get(field)):
                    errors.append(f"arms.{name}.{field} must be a SHA-256")
            launch = arm.get("launch", {})
            if launch.get("grid") != list(EXPECTED_GRID):
                errors.append(f"arms.{name}.launch.grid drift")
            if launch.get("block") != list(EXPECTED_BLOCK):
                errors.append(f"arms.{name}.launch.block drift")

    for section in (
        "static_spill",
        "ncu",
        "correctness",
        "root_cause_evidence",
        "resource_cleanup",
        "retained_attribution_evidence",
    ):
        value = manifest.get(section, {})
        if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
            errors.append(f"{section}.path is required")
        if not isinstance(value, Mapping) or not valid_sha(value.get("sha256")):
            errors.append(f"{section}.sha256 must be a SHA-256")

    decision = manifest.get("decision", {})
    if decision.get("experiment_goal") != "spill_root_cause_analysis":
        errors.append("decision.experiment_goal drift")
    if decision.get("severity") != "P0_hard_failure_by_project_policy":
        errors.append("decision.severity drift")
    if decision.get("physical_mechanism_status") != "all_bundles_closed":
        errors.append("decision.physical_mechanism_status drift")
    formation_causes = decision.get("formation_causes", {})
    if not isinstance(formation_causes, Mapping) or set(formation_causes) != {
        "main_108_word_bundle",
        "tail_14_word_bundle",
    }:
        errors.append("decision.formation_causes must cover both bundles")
    source_interpretation = decision.get("source_interpretation", {})
    if not isinstance(source_interpretation, Mapping) or set(source_interpretation) != {
        "main_108_word_bundle",
        "tail_14_word_bundle",
    }:
        errors.append("decision.source_interpretation must cover both bundles")
    if decision.get("source_attribution_status") != (
        "high_confidence_program_order_inference_not_compiler_certified"
    ):
        errors.append("decision.source_attribution_status drift")
    if decision.get("main_108_status") != "physical_formation_mechanism_closed":
        errors.append("decision.main_108_status drift")
    if decision.get("tail_14_status") != (
        "physical_formation_mechanism_closed_source_identity_partial"
    ):
        errors.append("decision.tail_14_status drift")
    if decision.get("tail_14_disposition") != (
        "deferred_reprofile_after_main_change"
    ):
        errors.append("decision.tail_14_disposition drift")
    if decision.get("overall_root_cause_status") != (
        "formation_mechanism_closed_source_attribution_inferred"
    ):
        errors.append("decision.overall_root_cause_status drift")
    if decision.get("production_optimization_recommendation_allowed") is not False:
        errors.append("decision must not permit a production optimization")
    if decision.get("followup_experiment_allowed") is not True:
        errors.append("decision must permit the scoped follow-up experiment")
    if decision.get("followup_scope") != "main_108_live_range_mechanism":
        errors.append("decision.followup_scope drift")
    if decision.get("latency_causality") != "not_tested_and_not_required_for_P0":
        errors.append("decision.latency_causality drift")
    if decision.get("tc_cadence_hypothesis") != (
        "unverified_spill_may_be_primary_contributor"
    ):
        errors.append("decision.tc_cadence_hypothesis drift")
    if decision.get("formal_experiment_closed") is not True:
        errors.append("decision.formal_experiment_closed must be true")
    if decision.get("closure_kind") != (
        "physical_formation_mechanisms_closed_with_source_attribution_limits"
    ):
        errors.append("decision.closure_kind drift")

    capture = manifest.get("capture_provenance", {})
    capture_family = capture.get("capture_family")
    schema_families = {
        "renumbered_exp004": {
            "exp004.arm-preparation.v1",
            "exp004.overlays.v1",
            "exp004.static-spill-evidence.v1",
            "exp004.ncu-spill-evidence.v1",
            "exp004.correctness.v1",
            "exp004.attribution-evidence.v1",
        },
        "native_exp003": {
            "exp003.spill-root-cause.arm-preparation.v1",
            "exp003.spill-root-cause.overlays.v1",
            "exp003.spill-root-cause.static-spill-evidence.v1",
            "exp003.spill-root-cause.ncu-spill-evidence.v1",
            "exp003.spill-root-cause.correctness.v1",
            "exp003.spill-root-cause.attribution-evidence.v1",
        },
    }
    consumed_schemas = capture.get("consumed_capture_schemas", [])
    if capture_family not in schema_families:
        errors.append("capture_provenance.capture_family drift")
    elif not isinstance(consumed_schemas, list) or set(consumed_schemas) != schema_families[
        capture_family
    ]:
        errors.append("capture_provenance capture schema family incomplete or mixed")
    dangling = capture.get("legacy_dangling_references", [])
    if capture_family == "renumbered_exp004":
        if capture.get("legacy_experiment_id") != "exp004":
            errors.append("capture_provenance.legacy_experiment_id drift")
        if not isinstance(dangling, list) or not any(
            item.get("historical_target") == "spill_localization_evidence.json"
            and item.get("canonical_replacement") == "spill_root_cause_evidence.json"
            for item in dangling
            if isinstance(item, Mapping)
        ):
            errors.append("capture_provenance legacy dangling-reference mapping missing")
    elif capture_family == "native_exp003":
        if capture.get("legacy_experiment_id") is not None:
            errors.append("native exp003 capture must not claim a legacy experiment ID")
        if dangling != []:
            errors.append("native exp003 capture must not carry legacy dangling references")

    correctness = manifest.get("correctness", {})
    if correctness.get("status") != "diagnostic_only_strict_cross_arm_invalid":
        errors.append("correctness.status drift")
    if correctness.get("strict_cross_arm_gate_pass") is not False:
        errors.append("correctness must not claim strict cross-arm equivalence")
    if correctness.get("baseline_self_drift_within_hard_caps") is not False:
        errors.append("correctness baseline self-drift boundary drift")

    attribution = manifest.get("retained_attribution_evidence", {})
    if attribution.get("formal_verdict") != "source_and_program_order_inference_only":
        errors.append("retained attribution formal verdict drift")
    if attribution.get("gate_pass") is not False:
        errors.append("retained attribution gate_pass must be false")

    cleanup = manifest.get("resource_cleanup", {})
    if cleanup.get("gate_pass") is not True:
        errors.append("resource_cleanup.gate_pass must be true")

    if any(key in manifest for key in ("attribution", "candidate_gates")):
        errors.append("closed manifest contains active legacy attribution fields")
    static_spill = manifest.get("static_spill", {})
    if isinstance(static_spill, Mapping) and "candidate_gates" in static_spill:
        errors.append("closed static_spill contains legacy candidate gates")
    return errors


def build_empty_manifest() -> dict[str, Any]:
    return {
        "schema": "exp003.spill-root-cause.run-manifest.v1",
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
            "experiment_goal": "spill_root_cause_analysis",
            "physical_formation_mechanism": "pending",
            "source_attribution": "pending",
            "latency_causality": "out_of_scope_not_started",
            "tc_cadence_causality": "out_of_scope_not_started",
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
