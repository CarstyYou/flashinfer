#!/usr/bin/env python3
"""Build compact, fail-closed exp_014 M8192 Scatter phase evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from exp014_scatter_probe_common import (
    ARMS,
    BASELINE,
    CANDIDATE,
    EVENT_ABI,
    EXPECTED_BASE_KERNEL_SHA256,
    RESULTS,
    SAMPLED_TASK_SLOTS,
    canonical_sha256,
    file_sha256,
    read_json,
    write_json,
)


ROOT = Path(__file__).resolve().parent
M = 8192
REPLAYS = 5
TILE_SAMPLES_PER_REPLAY = 16 * len(SAMPLED_TASK_SLOTS)
METRICS = ("body_ns", "including_sync_ns")
STATIC_EVIDENCE_NAME = "scatter_phase_static_resource_evidence.json"


class EvidenceError(RuntimeError):
    """Capture evidence is missing, malformed, or identity-inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def finite(value: Any, label: str, *, positive: bool = False) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} is not finite")
    if positive:
        require(result > 0.0, f"{label} must be positive")
    else:
        require(result >= 0.0, f"{label} must be non-negative")
    return result


def capture_path(results: Path, arm: str) -> Path:
    return results / "raw" / "scatter_phase_probe" / arm / f"m{M}" / "capture.json"


def relative_path(path: Path, results: Path) -> str:
    try:
        return path.resolve().relative_to(results.resolve()).as_posix()
    except ValueError as error:
        raise EvidenceError(
            f"artifact is outside the relocatable results root: {path}"
        ) from error


def validate_scatter_ownership(results: Path) -> dict[str, Any]:
    path = results / "ownership_gate.json"
    require(path.is_file(), f"missing Scatter ownership gate: {path}")
    value = read_json(path)
    require(
        value.get("schema") == "exp014.scatter-ownership-gate.v1",
        "Scatter ownership schema drift",
    )
    require(value.get("status") == "pass", "Scatter ownership gate failed")
    require(
        value.get("mapping") == "warp_m=(warp>>1)*32, warp_n=(warp&1)*64",
        "candidate Scatter mapping is not the locked 8-warp mapping",
    )
    require(value.get("vector_width") == 8, "Scatter vector width drift")
    cases = value.get("cases")
    require(isinstance(cases, list) and cases, "Scatter ownership cases are missing")
    full_tiles = [case for case in cases if case.get("valid_rows") == 128]
    require(full_tiles, "Scatter ownership gate has no full M128 tile")
    expected_warps = list(range(8))
    for index, case in enumerate(full_tiles):
        require(
            isinstance(case, Mapping), f"full-tile ownership case {index} malformed"
        )
        require(
            case.get("active_warps") == expected_warps,
            f"full-tile ownership case {index} does not engage W0-W7",
        )
        require(
            case.get("exactly_one_owner") is True,
            f"full-tile ownership case {index} is not exact-once",
        )
        require(
            case.get("invalid_writes") == 0,
            f"full-tile ownership case {index} has invalid writes",
        )
        require(
            case.get("observed_elements") == case.get("expected_elements"),
            f"full-tile ownership case {index} coverage drift",
        )
    return {
        "source": {
            "path": relative_path(path, results),
            "sha256": file_sha256(path),
        },
        "candidate_effective_warps": 8,
        "full_tile_active_warps": expected_warps,
        "full_tile_cases": len(full_tiles),
        "gate_pass": True,
    }


def validate_summary(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    require(value.get("samples") == TILE_SAMPLES_PER_REPLAY, f"{label} sample drift")
    result: dict[str, Any] = {"samples": TILE_SAMPLES_PER_REPLAY}
    for metric in METRICS:
        raw = value.get(metric)
        require(isinstance(raw, Mapping), f"{label}.{metric} must be an object")
        stats = {
            name: finite(raw.get(name), f"{label}.{metric}.{name}")
            for name in ("mean", "median", "p10", "p90", "min", "max")
        }
        require(
            stats["min"]
            <= stats["p10"]
            <= stats["median"]
            <= stats["p90"]
            <= stats["max"],
            f"{label}.{metric} order is inconsistent",
        )
        require(
            stats["min"] <= stats["mean"] <= stats["max"],
            f"{label}.{metric}.mean is outside min/max",
        )
        result[metric] = stats
    require(
        result["including_sync_ns"]["min"] >= result["body_ns"]["min"],
        f"{label} including_sync min is below body min",
    )
    require(
        result["including_sync_ns"]["median"] >= result["body_ns"]["median"],
        f"{label} including_sync median is below body median",
    )
    require(
        result["including_sync_ns"]["mean"] >= result["body_ns"]["mean"],
        f"{label} including_sync mean is below body mean",
    )
    return result


def validate_gate(value: Any, label: str) -> None:
    require(isinstance(value, Mapping), f"{label} is missing")
    require(value.get("gate_pass") is True, f"{label} did not pass")


def runtime_projection(runtime: Mapping[str, Any]) -> dict[str, Any]:
    gpu = runtime.get("gpu")
    require(isinstance(gpu, Mapping), "runtime.gpu is missing")
    projection = {
        "gpu_uuid": gpu.get("uuid"),
        "gpu_name": gpu.get("name"),
        "compute_capability": gpu.get("compute_capability"),
        "sm_count": gpu.get("sm_count"),
        "driver": gpu.get("driver"),
        "applications_graphics_clock_mhz": gpu.get("applications_graphics_clock_mhz"),
        "cuda_runtime": runtime.get("cuda_runtime"),
        "torch": runtime.get("torch"),
        "nvcc": runtime.get("nvcc"),
        "ptxas": runtime.get("ptxas"),
        "image_digest": runtime.get("image_digest"),
        "image_id": runtime.get("image_id"),
        "python_deps_sha256": runtime.get("python_deps_sha256"),
        "packages": runtime.get("packages"),
    }
    require(
        isinstance(projection["gpu_uuid"], str) and projection["gpu_uuid"],
        "runtime GPU UUID is missing",
    )
    require(
        is_sha256(projection["python_deps_sha256"]),
        "runtime python dependency hash is invalid",
    )
    return projection


def validate_capture(
    path: Path,
    *,
    arm: str,
    root_identity: Mapping[str, Any],
    arm_identity: Mapping[str, Any],
) -> dict[str, Any]:
    require(path.is_file(), f"missing capture: {path}")
    capture = read_json(path)
    require(
        capture.get("schema") == "exp014.scatter-phase-probe-capture.v1",
        f"{arm} capture schema drift",
    )
    require(capture.get("classification") == "diagnostic-only", f"{arm} class drift")
    require(capture.get("arm") == arm, f"{arm} capture arm drift")
    require(capture.get("m") == M, f"{arm} capture M drift")
    require(capture.get("fixture_kind") == "canonical", f"{arm} fixture drift")
    require(capture.get("event_abi") == EVENT_ABI, f"{arm} event ABI drift")

    source = capture.get("source")
    require(isinstance(source, Mapping), f"{arm} source identity is missing")
    overlay = arm_identity.get("overlay")
    require(isinstance(overlay, Mapping), f"{arm} overlay identity is missing")
    require(source.get("arm") == arm, f"{arm} source arm drift")
    require(
        source.get("kernel_sha256") == overlay.get("kernel_sha256"),
        f"{arm} kernel source identity drift",
    )
    require(
        source.get("dispatch_sha256") == overlay.get("dispatch_sha256"),
        f"{arm} dispatch source identity drift",
    )
    require(
        source.get("overlay_identity") == arm_identity,
        f"{arm} embedded overlay identity drift",
    )
    require(source.get("event_abi") == EVENT_ABI, f"{arm} source ABI drift")

    overlay_gate = capture.get("overlay_gate")
    validate_gate(overlay_gate, f"{arm}.overlay_gate")
    require(not overlay_gate.get("errors"), f"{arm} overlay gate retained errors")
    require(
        overlay_gate.get("root_identity") == root_identity,
        f"{arm} embedded root identity drift",
    )
    require(
        overlay_gate.get("arm_identity") == arm_identity,
        f"{arm} embedded arm identity drift",
    )

    eager = capture.get("eager")
    require(isinstance(eager, Mapping), f"{arm} eager record is missing")
    for gate in ("correctness_gate", "route_task_gate", "buffer_gate"):
        validate_gate(eager.get(gate), f"{arm}.eager.{gate}")
    validate_summary(eager.get("interval_summary"), f"{arm}.eager.interval_summary")

    raw_runs = capture.get("runs")
    require(isinstance(raw_runs, list), f"{arm} runs must be a list")
    require(len(raw_runs) == REPLAYS, f"{arm} replay count drift")
    runs = []
    for replay, raw in enumerate(raw_runs):
        require(isinstance(raw, Mapping), f"{arm} replay {replay} is not an object")
        require(raw.get("replay") == replay, f"{arm} replay index drift at {replay}")
        require(raw.get("gate_pass") is True, f"{arm} replay {replay} gate failed")
        for gate in ("correctness_gate", "route_task_gate", "buffer_gate"):
            validate_gate(raw.get(gate), f"{arm}.run{replay}.{gate}")
        require(is_sha256(raw.get("output_sha256")), f"{arm} output hash is invalid")
        require(is_sha256(raw.get("ticks_sha256")), f"{arm} tick hash is invalid")
        summary = validate_summary(
            raw.get("interval_summary"), f"{arm}.run{replay}.interval_summary"
        )
        runs.append(
            {
                "replay": replay,
                "event_elapsed_us": finite(
                    raw.get("event_elapsed_us"),
                    f"{arm}.run{replay}.event_elapsed_us",
                    positive=True,
                ),
                "interval_summary": summary,
                "output_sha256": raw["output_sha256"],
                "ticks_sha256": raw["ticks_sha256"],
            }
        )

    elapsed = [run["event_elapsed_us"] for run in runs]
    reported_e2e = capture.get("probe_e2e_us")
    require(isinstance(reported_e2e, Mapping), f"{arm} probe_e2e_us is missing")
    require(reported_e2e.get("samples") == REPLAYS, f"{arm} E2E sample drift")
    expected_e2e = {
        "samples": REPLAYS,
        "median": statistics.median(elapsed),
        "min": min(elapsed),
        "max": max(elapsed),
    }
    for name in ("median", "min", "max"):
        observed = finite(reported_e2e.get(name), f"{arm}.probe_e2e_us.{name}")
        require(
            math.isclose(observed, expected_e2e[name], rel_tol=0.0, abs_tol=1e-9),
            f"{arm} probe E2E {name} does not reproduce from runs",
        )

    runtime = capture.get("runtime")
    require(isinstance(runtime, Mapping), f"{arm} runtime identity is missing")
    require(runtime.get("source") == source, f"{arm} runtime/source identity drift")
    runtime_id = runtime_projection(runtime)
    require(
        is_sha256(capture.get("jit_artifact_set_sha256")), f"{arm} JIT hash invalid"
    )
    cubins = capture.get("cubin_sha256")
    require(
        isinstance(cubins, list) and cubins and all(is_sha256(item) for item in cubins),
        f"{arm} cubin identity is invalid",
    )
    artifacts = capture.get("jit_artifacts")
    require(isinstance(artifacts, list) and artifacts, f"{arm} JIT artifacts missing")
    for index, artifact in enumerate(artifacts):
        require(isinstance(artifact, Mapping), f"{arm} artifact {index} is malformed")
        require(
            isinstance(artifact.get("path"), str) and artifact.get("path"),
            f"{arm} artifact {index} path is invalid",
        )
        require(is_sha256(artifact.get("sha256")), f"{arm} artifact hash is invalid")
    require(
        canonical_sha256(artifacts) == capture.get("jit_artifact_set_sha256"),
        f"{arm} JIT artifact-set hash mismatch",
    )
    artifact_cubins = sorted(
        {
            item.get("sha256")
            for item in artifacts
            if isinstance(item, Mapping)
            and str(item.get("path", "")).endswith(".cubin")
        }
    )
    require(artifact_cubins == sorted(cubins), f"{arm} cubin/artifact mismatch")

    return {
        "path": path,
        "sha256": file_sha256(path),
        "capture": capture,
        "runs": runs,
        "runtime_projection": runtime_id,
        "probe_e2e_us": expected_e2e,
    }


def validate_static_resource_evidence(
    results: Path, captures: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    path = results / STATIC_EVIDENCE_NAME
    require(path.is_file(), f"missing Scatter phase static evidence: {path}")
    value = read_json(path)
    require(
        value.get("schema") == "exp015.static_resource_evidence.v1",
        "Scatter phase static evidence schema drift",
    )
    raw_arms = value.get("arms")
    require(isinstance(raw_arms, Mapping), "static resource arms are missing")
    key_for_arm = {BASELINE: "baseline", CANDIDATE: "candidate"}
    arms = {}
    for arm in ARMS:
        short_name = key_for_arm[arm]
        raw = raw_arms.get(short_name)
        require(isinstance(raw, Mapping), f"static resource arm missing: {short_name}")
        require(raw.get("label") == short_name, f"{arm} static label drift")

        cubin = raw.get("cubin")
        require(isinstance(cubin, Mapping), f"{arm} static cubin identity missing")
        cubin_sha256 = cubin.get("sha256")
        require(is_sha256(cubin_sha256), f"{arm} static cubin hash is invalid")
        require(
            cubin_sha256 in captures[arm]["capture"]["cubin_sha256"],
            f"{arm} capture/static cubin identity mismatch",
        )

        resource = raw.get("resource")
        require(isinstance(resource, Mapping), f"{arm} static resource block missing")
        stack_bytes = resource.get("stack_bytes_per_thread")
        local_bytes = resource.get("local_bytes_outside_stack")
        registers = resource.get("registers_per_thread")
        require(
            all(
                isinstance(item, int) and not isinstance(item, bool) and item >= 0
                for item in (stack_bytes, local_bytes, registers)
            ),
            f"{arm} static resource values are invalid",
        )

        sass = raw.get("sass")
        require(isinstance(sass, Mapping), f"{arm} static SASS block missing")
        selected = sass.get("selected_instruction_counts")
        require(
            isinstance(selected, Mapping),
            f"{arm} selected static instruction counts missing",
        )
        ldl = selected.get("ldl")
        stl = selected.get("stl")
        require(
            all(
                isinstance(item, int) and not isinstance(item, bool) and item >= 0
                for item in (ldl, stl)
            ),
            f"{arm} LDL/STL counts are invalid",
        )

        gates = raw.get("gates")
        require(isinstance(gates, Mapping), f"{arm} static gates missing")
        failed = gates.get("failed", [])
        require(isinstance(failed, list), f"{arm} static failed-gate list malformed")
        require(
            set(failed) <= {"registers_at_most_160"},
            f"{arm} has a non-register static gate failure: {failed}",
        )
        static_zero_spill = stack_bytes == local_bytes == ldl == stl == 0
        require(static_zero_spill, f"{arm} is not static-zero-spill")
        arms[arm] = {
            "cubin_sha256": cubin_sha256,
            "registers_per_thread": registers,
            "stack_bytes_per_thread": stack_bytes,
            "local_bytes_outside_stack": local_bytes,
            "ldl_instructions": ldl,
            "stl_instructions": stl,
            "static_zero_spill": True,
        }

    return {
        "source": {
            "path": relative_path(path, results),
            "sha256": file_sha256(path),
        },
        "arms": arms,
        "static_zero_spill": True,
        "register_count_is_observation_not_acceptance_cap": True,
        "gate_pass": True,
    }


def metric_payload(runs: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    per_replay = []
    for run in runs:
        stats = run["interval_summary"][metric]
        per_replay.append(
            {
                "replay": run["replay"],
                "tile_samples": TILE_SAMPLES_PER_REPLAY,
                **{
                    f"{name}_ns": stats[name]
                    for name in ("mean", "median", "p10", "p90", "min", "max")
                },
            }
        )
    replay_medians = [row["median_ns"] for row in per_replay]
    replay_means = [row["mean_ns"] for row in per_replay]
    return {
        "per_replay": per_replay,
        "aggregate": {
            "primary_estimator": "median_of_replay_medians",
            "value_ns": statistics.median(replay_medians),
            "mean_of_replay_means_ns": statistics.fmean(replay_means),
            "replay_median_min_ns": min(replay_medians),
            "replay_median_max_ns": max(replay_medians),
            "replays": REPLAYS,
            "tile_samples_per_replay": TILE_SAMPLES_PER_REPLAY,
            "pooled_percentiles_available": False,
        },
    }


def e2e_payload(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_replay = [
        {"replay": run["replay"], "elapsed_us": run["event_elapsed_us"]} for run in runs
    ]
    values = [row["elapsed_us"] for row in per_replay]
    return {
        "per_replay": per_replay,
        "aggregate": {
            "primary_estimator": "median",
            "value_us": statistics.median(values),
            "min_us": min(values),
            "max_us": max(values),
            "replays": REPLAYS,
        },
    }


def comparison(baseline: float, candidate: float) -> dict[str, float]:
    require(baseline > 0.0 and candidate > 0.0, "comparison latency must be positive")
    return {
        "baseline": baseline,
        "candidate": candidate,
        "speedup_x": baseline / candidate,
        "speedup_pct": (baseline / candidate - 1.0) * 100.0,
        "latency_reduction_pct": (baseline - candidate) / baseline * 100.0,
    }


def compare_metric(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], *, unit: str
) -> dict[str, Any]:
    value_key = f"value_{unit}"
    per_replay = []
    for left, right in zip(
        baseline["per_replay"], candidate["per_replay"], strict=True
    ):
        require(left["replay"] == right["replay"], "replay alignment drift")
        left_value = left["median_ns"] if unit == "ns" else left["elapsed_us"]
        right_value = right["median_ns"] if unit == "ns" else right["elapsed_us"]
        per_replay.append(
            {
                "replay": left["replay"],
                **comparison(left_value, right_value),
                "unit": unit,
            }
        )
    aggregate = comparison(
        baseline["aggregate"][value_key], candidate["aggregate"][value_key]
    )
    aggregate["unit"] = unit
    return {
        "per_replay_index": per_replay,
        "aggregate": aggregate,
        "pairing_scope": (
            "same replay index only; captures are separate-arm diagnostic runs, not ABBA"
        ),
    }


def build(results: Path) -> dict[str, Any]:
    results = results.resolve()
    root_identity_path = results / "scatter_phase_probe_overlays" / "identity.json"
    require(
        root_identity_path.is_file(), f"missing probe identity: {root_identity_path}"
    )
    root_identity = read_json(root_identity_path)
    require(
        root_identity.get("schema") == "exp014.scatter-phase-probe-overlays.v1",
        "probe root identity schema drift",
    )
    require(root_identity.get("event_abi") == EVENT_ABI, "root event ABI drift")
    require(
        root_identity.get("cross_arm", {}).get("gate_pass") is True,
        "matched probe identity gate failed",
    )

    captures = {}
    for arm in ARMS:
        arm_identity_path = (
            results / "scatter_phase_probe_overlays" / arm / "identity.json"
        )
        require(arm_identity_path.is_file(), f"missing {arm} identity")
        arm_identity = read_json(arm_identity_path)
        require(
            arm_identity == root_identity.get("arms", {}).get(arm),
            f"{arm} root/arm identity mismatch",
        )
        require(
            arm_identity.get("base", {}).get("kernel_sha256")
            == EXPECTED_BASE_KERNEL_SHA256[arm],
            f"{arm} base kernel identity drift",
        )
        overlay = arm_identity.get("overlay", {})
        for name in ("moe_dynamic_kernel.py", "moe_dispatch.py"):
            source_path = results / "scatter_phase_probe_overlays" / arm / name
            expected = overlay.get(
                "kernel_sha256"
                if name == "moe_dynamic_kernel.py"
                else "dispatch_sha256"
            )
            require(
                source_path.is_file() and file_sha256(source_path) == expected,
                f"{arm} live {name} identity drift",
            )
        captures[arm] = validate_capture(
            capture_path(results, arm),
            arm=arm,
            root_identity=root_identity,
            arm_identity=arm_identity,
        )

    scatter_ownership = validate_scatter_ownership(results)
    static_resources = validate_static_resource_evidence(results, captures)

    left = captures[BASELINE]
    right = captures[CANDIDATE]
    require(
        left["runtime_projection"] == right["runtime_projection"],
        "baseline/candidate runtime identity mismatch",
    )
    for field in ("fixture", "weights", "reference_sha256"):
        require(
            left["capture"].get(field) == right["capture"].get(field),
            f"baseline/candidate {field} mismatch",
        )

    arms = {}
    for arm, value in captures.items():
        arms[arm] = {
            "capture": {
                "path": relative_path(value["path"], results),
                "sha256": value["sha256"],
            },
            "body_ns": metric_payload(value["runs"], "body_ns"),
            "including_sync_ns": metric_payload(value["runs"], "including_sync_ns"),
            "probe_e2e_us": e2e_payload(value["runs"]),
            "runtime_identity": value["runtime_projection"],
            "cubin_sha256": value["capture"]["cubin_sha256"],
            "static_resources": static_resources["arms"][arm],
        }

    comparisons = {
        metric: compare_metric(
            arms[BASELINE][metric], arms[CANDIDATE][metric], unit="ns"
        )
        for metric in METRICS
    }
    comparisons["probe_e2e_us"] = compare_metric(
        arms[BASELINE]["probe_e2e_us"],
        arms[CANDIDATE]["probe_e2e_us"],
        unit="us",
    )
    payload = {
        "schema": "exp014.scatter-phase-evidence.v1",
        "status": "complete",
        "case": {
            "m": M,
            "fixture": "canonical",
            "baseline_scatter_warps": 4,
            "candidate_scatter_warps": 8,
            "sampled_task_slots": list(SAMPLED_TASK_SLOTS),
            "output_tiles": 16,
            "replays_per_arm": REPLAYS,
        },
        "event_abi": EVENT_ABI,
        "identity": {
            "probe_overlay_root": {
                "path": relative_path(root_identity_path, results),
                "sha256": file_sha256(root_identity_path),
            },
            "cross_arm_runtime_equal": True,
            "fixture_weights_reference_equal": True,
        },
        "scatter_ownership": scatter_ownership,
        "static_resource_evidence": static_resources,
        "static_zero_spill": True,
        "arms": arms,
        "comparison": comparisons,
        "scope_limits": [
            "Diagnostic `%globaltimer` probe evidence; it does not replace uninstrumented exp014 ABBA E2E.",
            "Only canonical M8192 and sampled full-M128 task slot 0 are represented; tail and other task slots are out of scope.",
            "Each phase sample is one task/output-tile envelope across W0-W7, not total kernel wall time or an additive whole-kernel share.",
            "body includes post-D marker-store overhead and cross-warp release skew; including_sync additionally reaches the post-sync F envelope.",
            "Per-replay-index ratios are diagnostic alignment only; separate-arm captures are not temporally paired ABBA samples.",
            "Capture JSON retains per-replay 16-tile summaries, not pooled raw percentiles; aggregate uses median of replay medians.",
        ],
        "gate_pass": True,
    }
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument(
        "--output", type=Path, default=RESULTS / "scatter_phase_evidence.json"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build(args.results)
    write_json(args.output.resolve(), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
