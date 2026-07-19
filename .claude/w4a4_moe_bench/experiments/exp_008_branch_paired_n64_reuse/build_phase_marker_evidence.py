#!/usr/bin/env python3
"""Apply fail-closed perturbation gates to exp_008 marker captures.

This reducer intentionally requires resource evidence for both disabled control
and enabled probe.  A marker-induced spill/resource transition can never be
used to explain the uninstrumented v0/v1 latency delta.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


from exp008_marker_common import (
    CONTROL,
    EVENT_ABI,
    PROBE,
    VERSIONS,
    canonical_sha256,
    read_json,
    write_json,
)


SPILL_METRICS = (
    "sass__inst_executed_register_spilling_op_read",
    "sass__inst_executed_register_spilling_op_write",
    "sass__inst_executed_register_spilling_mem_local_op_read",
    "sass__inst_executed_register_spilling_mem_local_op_write",
)
RESOURCE_FIELDS = (
    "registers_per_thread",
    "smem_bytes",
    "stack_bytes_per_thread",
    "static_spill_load_bytes",
    "static_spill_store_bytes",
    "compiler_spillrefill_sass",
    "achieved_occupancy_pct",
)
ADDITIVE_PHASES = {
    "front_end_route_q0",
    "claim_cache_transition",
    "fc1_interleaved_activation_envelope",
    "q1",
    "combined_fc2_scatter",
    "residual",
}
RUNTIME_IDENTITY_FIELDS = (
    "python",
    "torch",
    "cuda_runtime",
    "nvcc",
    "ptxas",
    "image_digest",
    "python_deps_sha256",
)
GPU_IDENTITY_FIELDS = (
    "uuid",
    "name",
    "driver",
    "applications_graphics_clock_mhz",
    "compute_capability",
    "sm_count",
)


class PerturbationContractError(ValueError):
    pass


def _require_capture(value: Mapping[str, Any], *, version: str, arm: str) -> None:
    if value.get("schema") != "exp008.phase-marker-capture.v1":
        raise PerturbationContractError("capture schema drift")
    if value.get("version") != version or value.get("arm") != arm:
        raise PerturbationContractError("capture version/arm drift")
    if not value.get("runs"):
        raise PerturbationContractError("capture contains no measured replay")
    if any(not run.get("gate_pass", False) for run in value["runs"]):
        raise PerturbationContractError("capture contains a failed replay")
    if value.get("event_abi") != EVENT_ABI:
        raise PerturbationContractError("capture event ABI drift")
    if not value.get("overlay_gate", {}).get("gate_pass", False):
        raise PerturbationContractError("capture overlay identity gate failed")
    if not value.get("jit_identity_gate", {}).get("gate_pass", False):
        raise PerturbationContractError("capture JIT identity gate failed")
    artifacts = value.get("jit_artifacts")
    if (
        not isinstance(artifacts, list)
        or not artifacts
        or canonical_sha256(artifacts)
        != value.get("jit_identity_gate", {}).get("artifact_set_sha256")
    ):
        raise PerturbationContractError("capture JIT artifact manifest drift")
    source = value.get("source", {})
    manifest = source.get("overlay_identity", {})
    if (
        source.get("version") != version
        or source.get("arm") != arm
        or manifest != value.get("overlay_gate", {}).get("manifest")
        or manifest.get("schema") != "exp008.phase-marker-overlay.v1"
        or manifest.get("version") != version
        or manifest.get("arm") != arm
        or bool(manifest.get("probe_enabled")) != (arm == PROBE)
        or manifest.get("event_abi") != EVENT_ABI
        or manifest.get("overlay", {}).get("kernel_sha256")
        != source.get("kernel_sha256")
        or manifest.get("overlay", {}).get("dispatch_sha256")
        != source.get("dispatch_sha256")
        or manifest.get("contracts", {}).get("dispatch_difference")
        != "_EXP008_PHASE_PROBE_ENABLED only"
        or not manifest.get("contracts", {}).get("jit_cache_key_contains_probe_flag")
        or manifest.get("contracts", {}).get("new_source_barriers") != 0
    ):
        raise PerturbationContractError("capture source/overlay identity drift")
    latency = value.get("latency_us", {}).get("median")
    if (
        isinstance(latency, str)
        or latency is None
        or not math.isfinite(float(latency))
        or float(latency) <= 0.0
    ):
        raise PerturbationContractError("capture latency is non-positive/non-numeric")
    if arm == PROBE:
        for replay, run in enumerate(value["runs"]):
            rollup = run.get("additive_rollup", {})
            phases = rollup.get("phases", {})
            if (
                rollup.get("schema") != "exp008.additive-phase-rollup.v1"
                or set(phases) != ADDITIVE_PHASES
                or not rollup.get("closure", {}).get("gate_pass", False)
            ):
                raise PerturbationContractError(
                    f"probe replay {replay} additive ledger schema/closure drift"
                )
            durations = [payload.get("duration_ns") for payload in phases.values()]
            if any(
                isinstance(item, str)
                or item is None
                or not math.isfinite(float(item))
                or float(item) < 0.0
                for item in durations
            ):
                raise PerturbationContractError(
                    f"probe replay {replay} has invalid phase duration"
                )
            calibration = run.get("marker_calibration", {}).get("median")
            task_tail = run.get("task_tail")
            grid_z = run.get("event_gate", {}).get("grid_z")
            if (
                isinstance(calibration, str)
                or calibration is None
                or not math.isfinite(float(calibration))
                or float(calibration) < 0.0
                or not isinstance(task_tail, int)
                or task_tail <= 0
                or not isinstance(grid_z, int)
                or grid_z <= 0
            ):
                raise PerturbationContractError(
                    f"probe replay {replay} calibration/cardinality drift"
                )


def _paired_capture_identity_gate(
    control: Mapping[str, Any], probe: Mapping[str, Any]
) -> dict[str, Any]:
    control_source = control["source"]
    probe_source = probe["source"]
    control_manifest = control_source["overlay_identity"]
    probe_manifest = probe_source["overlay_identity"]
    control_runtime = control.get("runtime", {})
    probe_runtime = probe.get("runtime", {})

    runtime_equal = all(
        control_runtime.get(field) == probe_runtime.get(field)
        and control_runtime.get(field) is not None
        for field in RUNTIME_IDENTITY_FIELDS
    ) and all(
        control_runtime.get("gpu", {}).get(field)
        == probe_runtime.get("gpu", {}).get(field)
        and control_runtime.get("gpu", {}).get(field) is not None
        for field in GPU_IDENTITY_FIELDS
    )
    normalized_dispatch_equal = (
        control_manifest.get("overlay", {}).get("normalized_dispatch_sha256")
        == probe_manifest.get("overlay", {}).get("normalized_dispatch_sha256")
        and control_manifest.get("overlay", {}).get("normalized_dispatch_sha256")
        is not None
    )
    control_base = control_manifest.get("base", {})
    probe_base = probe_manifest.get("base", {})
    base_keys = ("kernel_sha256", "dispatch_sha256", "wrapper_sha256")
    control_fixture = control.get("fixture")
    probe_fixture = probe.get("fixture")
    control_weights = control.get("weights")
    probe_weights = probe.get("weights")
    control_namespace = (
        control_runtime.get("hostname"),
        control_runtime.get("jit_root"),
    )
    probe_namespace = (
        probe_runtime.get("hostname"),
        probe_runtime.get("jit_root"),
    )
    checks = {
        "kernel_source_equal": control_source.get("kernel_sha256")
        == probe_source.get("kernel_sha256")
        and control_source.get("kernel_sha256") is not None,
        "dispatch_binary_source_distinct": control_source.get("dispatch_sha256")
        != probe_source.get("dispatch_sha256"),
        "normalized_dispatch_equal": normalized_dispatch_equal,
        "base_source_identity_equal": control_base == probe_base
        and all(control_base.get(key) is not None for key in base_keys),
        "event_abi_equal": control.get("event_abi")
        == probe.get("event_abi")
        == EVENT_ABI,
        "fixture_equal": isinstance(control_fixture, Mapping)
        and bool(control_fixture)
        and isinstance(probe_fixture, Mapping)
        and canonical_sha256(control_fixture) == canonical_sha256(probe_fixture),
        "weights_equal": isinstance(control_weights, Mapping)
        and bool(control_weights)
        and isinstance(probe_weights, Mapping)
        and canonical_sha256(control_weights) == canonical_sha256(probe_weights),
        "reference_equal": control.get("reference_sha256")
        == probe.get("reference_sha256")
        and control.get("reference_sha256") is not None,
        "runtime_environment_equal": runtime_equal,
        # Every arm runs in a fresh Docker container.  The physical host roots
        # are mounted at the intentionally stable in-container path
        # /workspace/jit, so namespace identity is (container, path), not the
        # container-local path alone.
        "independent_jit_namespaces": all(control_namespace)
        and all(probe_namespace)
        and control_namespace != probe_namespace,
    }
    return {"checks": checks, "gate_pass": all(checks.values())}


def _cross_version_identity_gate(
    captures: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    v0 = captures["v0"][PROBE]
    v1 = captures["v1"][PROBE]
    v0_runtime = v0["runtime"]
    v1_runtime = v1["runtime"]
    v0_manifest = v0["source"]["overlay_identity"]
    v1_manifest = v1["source"]["overlay_identity"]
    all_jit_namespaces = [
        (
            captures[version][arm]["runtime"].get("hostname"),
            captures[version][arm]["runtime"].get("jit_root"),
        )
        for version in VERSIONS
        for arm in (CONTROL, PROBE)
    ]
    checks = {
        "event_abi": v0.get("event_abi") == v1.get("event_abi") == EVENT_ABI,
        "fixture": canonical_sha256(v0["fixture"])
        == canonical_sha256(v1["fixture"]),
        "weights": canonical_sha256(v0["weights"])
        == canonical_sha256(v1["weights"]),
        "reference": v0.get("reference_sha256") == v1.get("reference_sha256"),
        "runtime": all(
            v0_runtime.get(field) == v1_runtime.get(field)
            and v0_runtime.get(field) is not None
            for field in RUNTIME_IDENTITY_FIELDS
        ),
        "gpu": all(
            v0_runtime.get("gpu", {}).get(field)
            == v1_runtime.get("gpu", {}).get(field)
            and v0_runtime.get("gpu", {}).get(field) is not None
            for field in GPU_IDENTITY_FIELDS
        ),
        "dispatch_base": v0_manifest["base"].get("dispatch_sha256")
        == v1_manifest["base"].get("dispatch_sha256"),
        "wrapper_base": v0_manifest["base"].get("wrapper_sha256")
        == v1_manifest["base"].get("wrapper_sha256"),
        "probe_dispatch": v0["source"].get("dispatch_sha256")
        == v1["source"].get("dispatch_sha256"),
        "normalized_dispatch": v0_manifest["overlay"].get(
            "normalized_dispatch_sha256"
        )
        == v1_manifest["overlay"].get("normalized_dispatch_sha256"),
        "four_independent_jit_namespaces": all(
            all(namespace) for namespace in all_jit_namespaces
        )
        and len(set(all_jit_namespaces)) == 4,
    }
    return {"checks": checks, "gate_pass": all(checks.values())}


def _require_resource(value: Mapping[str, Any], *, version: str, arm: str) -> None:
    if value.get("schema") != "exp008.marker-resource-evidence.v1":
        raise PerturbationContractError("resource evidence schema drift")
    if value.get("version") != version or value.get("arm") != arm:
        raise PerturbationContractError("resource version/arm drift")
    if not value.get("gate_pass", False):
        raise PerturbationContractError("resource evidence gate failed")
    identity = value.get("identity", {})
    identity_fields = (
        "kernel_source_sha256",
        "dispatch_source_sha256",
        "jit_artifact_set_sha256",
        "cubin_sha256",
        "kernel_symbol",
        "static_kernel_symbol",
        "ncu_kernel_symbol",
        "gpu_uuid",
    )
    if any(not identity.get(field) for field in identity_fields):
        raise PerturbationContractError("resource evidence identity is incomplete")
    resource = value.get("resource", {})
    missing = [field for field in RESOURCE_FIELDS if field not in resource]
    if missing:
        raise PerturbationContractError(f"missing resource fields: {missing}")
    dynamic = resource.get("dynamic_spill_metrics", {})
    missing_metrics = [metric for metric in SPILL_METRICS if metric not in dynamic]
    if missing_metrics:
        raise PerturbationContractError(
            f"missing/n-a dynamic spill metrics: {missing_metrics}"
        )
    values = [resource[field] for field in RESOURCE_FIELDS]
    values += [dynamic[metric] for metric in SPILL_METRICS]
    if any(
        isinstance(item, str) or item is None or not math.isfinite(float(item))
        for item in values
    ):
        raise PerturbationContractError("resource evidence contains non-numeric/n-a")


def _resource(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return value["resource"]


def _resource_capture_identity_gate(
    resource: Mapping[str, Any], capture: Mapping[str, Any]
) -> dict[str, Any]:
    identity = resource["identity"]
    source = capture["source"]
    capture_cubins = {
        artifact.get("sha256")
        for artifact in capture["jit_artifacts"]
        if str(artifact.get("path", "")).endswith(".cubin")
    }
    checks = {
        "kernel_source": identity["kernel_source_sha256"]
        == source["kernel_sha256"],
        "dispatch_source": identity["dispatch_source_sha256"]
        == source["dispatch_sha256"],
        "jit_artifact_set": identity["jit_artifact_set_sha256"]
        == capture["jit_identity_gate"].get("artifact_set_sha256"),
        "gpu_uuid": identity["gpu_uuid"]
        == capture["runtime"].get("gpu", {}).get("uuid"),
        "cubin_in_capture_jit_artifacts": identity["cubin_sha256"]
        in capture_cubins,
        "kernel_symbol_static_ncu_exact": identity["kernel_symbol"]
        == identity["static_kernel_symbol"]
        == identity["ncu_kernel_symbol"],
    }
    return {"checks": checks, "gate_pass": all(checks.values())}


def _zero_spill(value: Mapping[str, Any]) -> bool:
    resource = _resource(value)
    static = (
        "static_spill_load_bytes",
        "static_spill_store_bytes",
        "compiler_spillrefill_sass",
    )
    return all(float(resource[field]) == 0.0 for field in static) and all(
        float(resource["dynamic_spill_metrics"][metric]) == 0.0
        for metric in SPILL_METRICS
    )


def _drift_pct(left: float, right: float) -> float:
    if left == right:
        return 0.0
    return 100.0 * abs(right - left) / max(abs(left), 1.0)


def _cv_pct(values: Sequence[float]) -> float:
    if not values:
        return math.inf
    mean = statistics.mean(values)
    if mean == 0.0:
        return 0.0 if all(value == 0.0 for value in values) else math.inf
    return 100.0 * statistics.pstdev(values) / mean


def _phase_gates(probe: Mapping[str, Any]) -> dict[str, Any]:
    samples: dict[str, list[float]] = {}
    shares: dict[str, list[float]] = {}
    marker_cost_pct: dict[str, list[float]] = {}
    denominators: list[float] = []
    grid_sizes: set[int] = set()
    for run in probe["runs"]:
        rollup = run.get("additive_rollup", {})
        phases = rollup.get("phases", {})
        calibration = float(run.get("marker_calibration", {}).get("median", math.inf))
        task_tail = int(run.get("task_tail", 0))
        grid_z = int(run.get("event_gate", {}).get("grid_z", 0))
        grid_sizes.add(grid_z)
        denominator = float(
            rollup.get("denominator", {}).get("duration_ns", 0)
            or sum(float(payload["duration_ns"]) for payload in phases.values())
        )
        if denominator <= 0:
            raise PerturbationContractError("phase replay denominator is not positive")
        denominators.append(denominator)
        for phase, payload in phases.items():
            duration = float(payload["duration_ns"])
            samples.setdefault(phase, []).append(duration)
            shares.setdefault(phase, []).append(
                float(payload.get("share_pct", 100.0 * duration / denominator))
            )
            if phase == "residual":
                continue
            units = grid_z if phase == "front_end_route_q0" else task_tail
            per_unit = duration / units if units > 0 else 0.0
            ratio = math.inf if per_unit <= 0 else 100.0 * calibration / per_unit
            marker_cost_pct.setdefault(phase, []).append(ratio)
    phase_cv = {phase: _cv_pct(values) for phase, values in samples.items()}
    cost_max = {
        phase: max(values) for phase, values in marker_cost_pct.items()
    }
    if len(grid_sizes) != 1 or next(iter(grid_sizes), 0) <= 0:
        raise PerturbationContractError(
            f"phase replay grid_z drift: {sorted(grid_sizes)}"
        )
    grid_z = next(iter(grid_sizes))
    phase_summary = {
        phase: {
            "median_sm_equivalent_us": statistics.median(values) / grid_z / 1000.0,
            "median_share_pct": statistics.median(shares[phase]),
            "replay_cv_pct": phase_cv[phase],
        }
        for phase, values in samples.items()
    }
    return {
        "summary": {
            "phase_time_definition": "median(duration_ns) / grid_z / 1000",
            "grid_z": grid_z,
            "denominator_median_sm_equivalent_us": statistics.median(denominators)
            / grid_z
            / 1000.0,
            "phases": phase_summary,
        },
        "phase_replay_cv_pct": phase_cv,
        "phase_replay_cv_le_5pct": all(value <= 5.0 for value in phase_cv.values()),
        "marker_cost_pct_of_phase_mean_max": cost_max,
        "marker_cost_le_10pct": all(value <= 10.0 for value in cost_max.values()),
    }


def evaluate_version(
    version: str,
    control_capture: Mapping[str, Any],
    probe_capture: Mapping[str, Any],
    control_resource: Mapping[str, Any],
    probe_resource: Mapping[str, Any],
) -> dict[str, Any]:
    _require_capture(control_capture, version=version, arm=CONTROL)
    _require_capture(probe_capture, version=version, arm=PROBE)
    _require_resource(control_resource, version=version, arm=CONTROL)
    _require_resource(probe_resource, version=version, arm=PROBE)
    identity_gate = _paired_capture_identity_gate(control_capture, probe_capture)
    if not identity_gate["gate_pass"]:
        raise PerturbationContractError(
            f"control/probe identity drift: {identity_gate['checks']}"
        )
    resource_identity = {
        CONTROL: _resource_capture_identity_gate(control_resource, control_capture),
        PROBE: _resource_capture_identity_gate(probe_resource, probe_capture),
    }
    if not all(value["gate_pass"] for value in resource_identity.values()):
        raise PerturbationContractError(
            f"resource/capture identity drift: {resource_identity}"
        )

    control_us = float(control_capture["latency_us"]["median"])
    probe_us = float(probe_capture["latency_us"]["median"])
    overhead_pct = 100.0 * (probe_us / control_us - 1.0)
    perturbation_pct = abs(overhead_pct)
    c_resource = _resource(control_resource)
    p_resource = _resource(probe_resource)
    occupancy_same = float(c_resource["achieved_occupancy_pct"]) == float(
        p_resource["achieved_occupancy_pct"]
    )
    resource_drift = {
        field: _drift_pct(float(c_resource[field]), float(p_resource[field]))
        for field in (
            "registers_per_thread",
            "smem_bytes",
            "stack_bytes_per_thread",
            "static_spill_load_bytes",
            "static_spill_store_bytes",
            "compiler_spillrefill_sass",
        )
    }
    max_resource_drift = max(resource_drift.values())
    control_zero = _zero_spill(control_resource)
    probe_zero = _zero_spill(probe_resource)
    exact_spill_identity = {
        metric: float(c_resource["dynamic_spill_metrics"][metric])
        == float(p_resource["dynamic_spill_metrics"][metric])
        for metric in SPILL_METRICS
    }
    phase_gates = _phase_gates(probe_capture)

    hard_stop = (
        perturbation_pct > 10.0
        or not occupancy_same
        or max_resource_drift > 50.0
    )
    spill_transition = control_zero != probe_zero or not all(exact_spill_identity.values())
    quantitative = (
        not hard_stop
        and perturbation_pct <= 5.0
        and occupancy_same
        and control_zero
        and probe_zero
        and max_resource_drift <= 25.0
        and phase_gates["phase_replay_cv_le_5pct"]
        and phase_gates["marker_cost_le_10pct"]
    )
    if hard_stop:
        classification = "stop_attribution"
    elif quantitative:
        classification = "quantitative_probe_level_diagnostic"
    elif spill_transition or not (control_zero and probe_zero):
        classification = "qualitative_only_spill_or_local_resource_changed"
    else:
        classification = "bounded_or_inconclusive_diagnostic"

    return {
        "schema": "exp008.marker-perturbation-version.v1",
        "version": version,
        "control_median_us": control_us,
        "probe_median_us": probe_us,
        "probe_over_control_overhead_pct": overhead_pct,
        "absolute_control_probe_perturbation_pct": perturbation_pct,
        "control_probe_identity": identity_gate,
        "resource_capture_identity": resource_identity,
        "resource": {
            "control_zero_spill": control_zero,
            "probe_zero_spill": probe_zero,
            "spill_transition_or_count_change": spill_transition,
            "dynamic_spill_identity": exact_spill_identity,
            "achieved_occupancy_same": occupancy_same,
            "drift_pct": resource_drift,
            "max_drift_pct": max_resource_drift,
        },
        "phase_stability": phase_gates,
        "classification": classification,
        "quantitative_phase_delta_eligible": quantitative,
        "may_replace_uninstrumented_gate_b_latency": False,
    }


def build(
    capture_root: Path, resource_root: Path, output: Path
) -> dict[str, Any]:
    versions = {}
    all_captures: dict[str, dict[str, Mapping[str, Any]]] = {}
    for version in VERSIONS:
        captures = {
            arm: read_json(capture_root / version / arm / "capture.json")
            for arm in (CONTROL, PROBE)
        }
        all_captures[version] = captures
        resources = {
            arm: read_json(resource_root / version / arm / "resource.json")
            for arm in (CONTROL, PROBE)
        }
        versions[version] = evaluate_version(
            version,
            captures[CONTROL],
            captures[PROBE],
            resources[CONTROL],
            resources[PROBE],
        )
    both_quantitative = all(
        value["quantitative_phase_delta_eligible"] for value in versions.values()
    )
    overhead_asymmetry = abs(
        versions["v1"]["probe_over_control_overhead_pct"]
        - versions["v0"]["probe_over_control_overhead_pct"]
    )
    cross_version_identity = _cross_version_identity_gate(all_captures)
    paired_allowed = (
        both_quantitative
        and overhead_asymmetry <= 2.0
        and cross_version_identity["gate_pass"]
    )
    result = {
        "schema": "exp008.marker-perturbation-evidence.v1",
        "versions": versions,
        "paired_v0_v1_phase_delta": {
            "both_versions_quantitative": both_quantitative,
            "perturbation_overhead_asymmetry_percentage_points": overhead_asymmetry,
            "required_max_asymmetry_percentage_points": 2.0,
            "cross_version_identity": cross_version_identity,
            "allowed": paired_allowed,
        },
        "interpretation_contract": {
            "gate_b_uninstrumented_latency_is_authority": True,
            "marker_phase_share_never_replaces_gate_b": True,
            "spill_change_forbids_precise_uninstrumented_attribution": True,
        },
        "gate_pass": paired_allowed,
    }
    write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.capture_root.resolve(),
        args.resource_root.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(result["paired_v0_v1_phase_delta"], sort_keys=True))
    return 0 if result["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
