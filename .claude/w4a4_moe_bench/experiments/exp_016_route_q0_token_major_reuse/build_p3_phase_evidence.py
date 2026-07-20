#!/usr/bin/env python3
"""Reduce four exp_016 P3 captures into compact, GPU-free evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from exp016_p3_probe_common import (
    ARMS,
    BASELINE,
    CANDIDATE,
    CONTROL,
    EVENT_ABI,
    EXPECTED_BASE_KERNEL_SHA256,
    EXPECTED_DISPATCH_SHA256,
    EXPECTED_WRAPPER_SHA256,
    MODES,
    PROBE,
    RESULTS,
    canonical_sha256,
    file_sha256,
    read_json,
    write_json,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_JSON = RESULTS / "p3_phase_evidence.json"
DEFAULT_MARKDOWN = RESULTS / "p3_phase_evidence.md"

CAPTURE_SCHEMA = "exp016.p3-phase-capture.v1"
SASS_SPILL_SCHEMA = "exp016.p3-sass-spill-evidence.v1"
SASS_SPILL_SIDECAR = "sass_spill_evidence.json"
EXPECTED_REPLAYS = 5
MAX_ABS_PROBE_E2E_PERTURBATION_PERCENT = 2.0


class EvidenceError(RuntimeError):
    """A supplied P3 capture is malformed, inconsistent, or untraceable."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def finite(value: Any, label: str, *, positive: bool = False) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} is not finite")
    if positive:
        require(result > 0.0, f"{label} must be positive")
    return result


def close(actual: Any, expected: float, label: str) -> None:
    observed = finite(actual, label)
    require(
        math.isclose(observed, expected, rel_tol=1.0e-9, abs_tol=1.0e-9),
        f"{label} mismatch: {observed} != {expected}",
    )


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def capture_path(results: Path, arm: str, mode: str) -> Path:
    return results / "raw" / "p3_phase" / arm / mode / "capture.json"


def sass_spill_path(capture: Path) -> Path:
    return capture.parent / SASS_SPILL_SIDECAR


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"missing {label}")
    return value


def _canonical_ascii_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def audit_runtime(
    value: Mapping[str, Any], source: Mapping[str, Any], label: str
) -> dict[str, Any]:
    runtime = _mapping(value.get("runtime"), f"{label}.runtime")
    runtime_source = _mapping(runtime.get("source"), f"{label}.runtime.source")
    require(
        dict(runtime_source) == dict(source), f"{label} runtime/source disagreement"
    )
    gpu = _mapping(runtime.get("gpu"), f"{label}.runtime.gpu")
    imports = _mapping(runtime.get("imports"), f"{label}.runtime.imports")
    harness = _mapping(runtime.get("harness"), f"{label}.runtime.harness")
    for field in (
        "hostname",
        "python",
        "packages",
        "torch",
        "cuda_runtime",
        "nvcc",
        "ptxas",
        "image_digest",
        "image_id",
        "python_deps_sha256",
    ):
        require(
            runtime.get(field) not in (None, ""), f"missing {label}.runtime.{field}"
        )
    require(
        isinstance(runtime["packages"], Mapping), f"{label} package identity missing"
    )
    require(
        gpu.get("compute_capability") in ([12, 0], [12, 1]),
        f"{label} is not SM120/121",
    )
    require(gpu.get("sm_count") == 110, f"{label} SM count drift")
    require(
        gpu.get("foreign_processes_before_cuda_context") == [],
        f"{label} had a foreign GPU process",
    )
    for field in (
        "uuid",
        "name",
        "pci_bus_id",
        "driver",
        "applications_graphics_clock_mhz",
        "max_graphics_clock_mhz",
    ):
        require(gpu.get(field) not in (None, ""), f"missing {label}.gpu.{field}")
    require(
        imports.get("target_module") == source["kernel_overlay"],
        f"{label} imported kernel overlay drift",
    )
    for field in ("cutlass_python", "cutlass_python_version", "flashinfer"):
        require(
            imports.get(field) not in (None, ""), f"missing {label}.imports.{field}"
        )
    require(is_sha256(harness.get("sha256")), f"{label} harness SHA invalid")
    require(
        is_sha256(harness.get("run_exp016_arm_sha256")),
        f"{label} exp016 harness SHA invalid",
    )
    return {
        "hostname": runtime["hostname"],
        "python": runtime["python"],
        "packages": runtime["packages"],
        "torch": runtime["torch"],
        "cuda_runtime": runtime["cuda_runtime"],
        "nvcc": runtime["nvcc"],
        "ptxas": runtime["ptxas"],
        "image_digest": runtime["image_digest"],
        "image_id": runtime["image_id"],
        "python_deps_sha256": runtime["python_deps_sha256"],
        "gpu": {
            field: gpu[field]
            for field in (
                "uuid",
                "name",
                "pci_bus_id",
                "driver",
                "applications_graphics_clock_mhz",
                "max_graphics_clock_mhz",
                "compute_capability",
                "sm_count",
            )
        },
        "imports": {
            field: imports[field]
            for field in ("cutlass_python", "cutlass_python_version", "flashinfer")
        },
        "harness": {
            "sha256": harness["sha256"],
            "run_exp016_arm_sha256": harness["run_exp016_arm_sha256"],
        },
    }


def audit_artifacts(value: Mapping[str, Any], label: str) -> tuple[str, str]:
    artifacts = value.get("jit_artifacts")
    require(isinstance(artifacts, list) and artifacts, f"{label} JIT artifacts missing")
    artifact_hash = value.get("jit_artifact_set_sha256")
    require(is_sha256(artifact_hash), f"{label} artifact-set SHA invalid")
    require(
        artifact_hash == _canonical_ascii_sha256(artifacts),
        f"{label} artifact-set SHA mismatch",
    )
    cubins = value.get("cubin_sha256")
    require(
        isinstance(cubins, list) and len(cubins) == 1 and is_sha256(cubins[0]),
        f"{label} must identify exactly one cubin",
    )
    artifact_cubins = {
        item.get("sha256")
        for item in artifacts
        if isinstance(item, Mapping) and str(item.get("path", "")).endswith(".cubin")
    }
    require(artifact_cubins == {cubins[0]}, f"{label} cubin inventory mismatch")
    return str(artifact_hash), str(cubins[0])


def audit_resource(
    value: Mapping[str, Any], cubin_sha256: str, capture: Path, label: str
) -> dict[str, Any]:
    resource = _mapping(value.get("static_resource_usage"), f"{label}.resources")
    require(
        resource.get("schema") == "exp016.p3-phase-resource-usage.v1",
        f"{label} resource schema drift",
    )
    require(resource.get("gate_pass") is True, f"{label} resource gate failed")
    require(is_sha256(resource.get("raw_sha256")), f"{label} raw resource SHA invalid")
    raw_path = resource.get("raw_path")
    require(
        isinstance(raw_path, str) and Path(raw_path).name == "resource_usage.txt",
        f"{label} raw resource basename drift",
    )
    captured_raw = capture.parent / "resource_usage.txt"
    require(captured_raw.is_file(), f"{label} captured resource artifact missing")
    require(
        file_sha256(captured_raw) == resource["raw_sha256"],
        f"{label} captured resource artifact SHA mismatch",
    )
    records = resource.get("records")
    require(
        isinstance(records, list) and len(records) == 1,
        f"{label} requires one dynamic-kernel resource record",
    )
    record = _mapping(records[0], f"{label}.resource record")
    require(record.get("cubin_sha256") == cubin_sha256, f"{label} resource cubin drift")
    require(
        "MoEDynamicKernel" in str(record.get("kernel_symbol", "")),
        f"{label} kernel symbol drift",
    )
    result = {}
    for field in (
        "registers_per_thread",
        "stack_bytes_per_thread",
        "static_shared_bytes_per_cta",
        "static_local_bytes_outside_stack",
    ):
        value_field = record.get(field)
        require(
            isinstance(value_field, int)
            and not isinstance(value_field, bool)
            and value_field >= 0,
            f"{label} invalid resource field {field}",
        )
        result[field] = value_field
    result["kernel_symbol"] = record["kernel_symbol"]
    result["cubin_sha256"] = cubin_sha256
    return result


def audit_sass_spill(
    capture: Path,
    *,
    arm: str,
    mode: str,
    cubin_sha256: str,
    kernel_symbol: str,
) -> dict[str, Any]:
    sidecar_path = sass_spill_path(capture)
    label = str(sidecar_path)
    sidecar = read_json(sidecar_path)
    require(sidecar.get("schema") == SASS_SPILL_SCHEMA, f"{label} schema drift")
    require(
        sidecar.get("arm") == arm and sidecar.get("mode") == mode,
        f"{label} arm/mode drift",
    )
    capture_identity = _mapping(sidecar.get("capture"), f"{label}.capture")
    require(capture_identity.get("path") == capture.name, f"{label} capture path drift")
    require(
        capture_identity.get("sha256") == file_sha256(capture),
        f"{label} capture SHA drift",
    )
    jit = _mapping(sidecar.get("jit"), f"{label}.jit")
    require(jit.get("root") not in (None, ""), f"{label} JIT root missing")
    require(
        isinstance(jit.get("cubin_inventory_count"), int)
        and not isinstance(jit.get("cubin_inventory_count"), bool)
        and jit["cubin_inventory_count"] >= 1,
        f"{label} cubin inventory invalid",
    )
    require(
        jit.get("matched_cubin_relative_path") not in (None, ""),
        f"{label} matched cubin path missing",
    )
    require(
        jit.get("matched_cubin_sha256") == cubin_sha256,
        f"{label} cubin SHA drift",
    )
    require(sidecar.get("kernel_symbol") == kernel_symbol, f"{label} symbol drift")

    tool = _mapping(sidecar.get("tool"), f"{label}.tool")
    require(tool.get("path") not in (None, ""), f"{label} tool path missing")
    require(is_sha256(tool.get("sha256")), f"{label} tool SHA invalid")
    require(tool.get("version") not in (None, ""), f"{label} tool version missing")
    commands = tool.get("commands")
    require(
        isinstance(commands, list)
        and len(commands) == 2
        and all(isinstance(command, list) and command for command in commands)
        and commands[0][0] == "--dump-sass"
        and commands[1][0] == "--dump-elf",
        f"{label} tool commands drift",
    )

    for field in ("raw_sass", "raw_elf"):
        raw = _mapping(sidecar.get(field), f"{label}.{field}")
        raw_name = raw.get("path")
        require(
            isinstance(raw_name, str) and Path(raw_name).name == raw_name,
            f"{label} {field} path must be a local basename",
        )
        raw_path = sidecar_path.parent / raw_name
        require(raw_path.is_file(), f"{label} {field} artifact missing")
        require(is_sha256(raw.get("sha256")), f"{label} {field} SHA invalid")
        require(
            file_sha256(raw_path) == raw["sha256"],
            f"{label} {field} artifact SHA drift",
        )
        require(
            isinstance(raw.get("size"), int)
            and not isinstance(raw.get("size"), bool)
            and raw["size"] == raw_path.stat().st_size,
            f"{label} {field} size drift",
        )

    counts = _mapping(sidecar.get("counts"), f"{label}.counts")
    count_fields = (
        "sass_instruction_count",
        "spill_refill_annotation_count",
        "spill_refill_annotation_unique_pc_count",
        "ldl_opcode_count",
        "stl_opcode_count",
        "local_sass_opcode_count",
    )
    for field in count_fields:
        count = counts.get(field)
        require(
            isinstance(count, int) and not isinstance(count, bool) and count >= 0,
            f"{label} invalid count {field}",
        )
    require(counts["sass_instruction_count"] > 0, f"{label} parsed no SASS")
    require(
        counts["local_sass_opcode_count"]
        == counts["ldl_opcode_count"] + counts["stl_opcode_count"],
        f"{label} local opcode count disagreement",
    )
    require(
        counts["spill_refill_annotation_unique_pc_count"]
        <= counts["spill_refill_annotation_count"],
        f"{label} annotation count disagreement",
    )
    require(
        counts.get("annotation_pcs_equal_local_sass_pcs") is True,
        f"{label} annotation/local PC closure failed",
    )
    histogram = counts.get("local_sass_opcode_histogram")
    require(isinstance(histogram, Mapping), f"{label} local opcode histogram missing")

    integrity_checks = _mapping(
        sidecar.get("integrity_checks"), f"{label}.integrity checks"
    )
    require(
        bool(integrity_checks)
        and all(value is True for value in integrity_checks.values()),
        f"{label} evidence integrity check failed",
    )
    require(
        sidecar.get("evidence_integrity_gate_pass") is True,
        f"{label} evidence integrity gate failed",
    )
    spill_gate = (
        counts["spill_refill_annotation_count"] == 0
        and counts["ldl_opcode_count"] == 0
        and counts["stl_opcode_count"] == 0
        and counts["local_sass_opcode_count"] == 0
    )
    require(
        sidecar.get("sass_spill_gate_pass") is spill_gate,
        f"{label} spill gate/count disagreement",
    )
    require(
        sidecar.get("gate_pass") is spill_gate,
        f"{label} overall gate/count disagreement",
    )
    return {
        "path": str(sidecar_path),
        "sha256": file_sha256(sidecar_path),
        "cubin_sha256": cubin_sha256,
        "kernel_symbol": kernel_symbol,
        "counts": dict(counts),
        "raw_sass_sha256": sidecar["raw_sass"]["sha256"],
        "raw_elf_sha256": sidecar["raw_elf"]["sha256"],
        "gate_pass": spill_gate,
    }


def _gate(value: Any, label: str) -> None:
    mapping = _mapping(value, label)
    require(mapping.get("gate_pass") is True, f"{label} failed")


def audit_capture(
    value: Mapping[str, Any], path: Path, *, arm: str, mode: str
) -> dict[str, Any]:
    label = str(path)
    require(value.get("schema") == CAPTURE_SCHEMA, f"{label} schema drift")
    require(value.get("arm") == arm, f"{label} arm drift")
    require(value.get("mode") == mode, f"{label} mode drift")
    require(value.get("m") == 8192, f"{label} M drift")
    require(value.get("fixture") == "canonical", f"{label} fixture drift")
    require(value.get("scale_kind") == "unequal", f"{label} scale drift")
    require(value.get("event_abi") == EVENT_ABI, f"{label} event ABI drift")
    expected_class = (
        "diagnostic matched probe" if mode == PROBE else "marker-disabled ABI control"
    )
    require(
        value.get("classification") == expected_class, f"{label} classification drift"
    )

    source = _mapping(value.get("source"), f"{label}.source")
    require(
        source.get("arm") == arm and source.get("mode") == mode,
        f"{label} source arm/mode drift",
    )
    require(
        source.get("base_kernel_sha256") == EXPECTED_BASE_KERNEL_SHA256[arm],
        f"{label} base source drift",
    )
    require(source.get("event_abi") == EVENT_ABI, f"{label} source ABI drift")
    for field in ("kernel_sha256", "dispatch_sha256"):
        require(is_sha256(source.get(field)), f"{label} {field} invalid")

    overlay = _mapping(value.get("overlay_gate"), f"{label}.overlay_gate")
    require(
        overlay.get("gate_pass") is True and overlay.get("errors") == [],
        f"{label} overlay gate failed",
    )
    require(
        overlay.get("arm") == arm and overlay.get("mode") == mode,
        f"{label} overlay arm/mode drift",
    )
    require(
        overlay.get("kernel") == source["kernel_overlay"],
        f"{label} overlay kernel path drift",
    )
    require(
        overlay.get("dispatch") == source["dispatch_overlay"],
        f"{label} overlay dispatch path drift",
    )
    root_identity = _mapping(overlay.get("root_identity"), f"{label}.root identity")
    arm_identity = _mapping(overlay.get("arm_identity"), f"{label}.arm identity")
    require(
        root_identity.get("schema") == "exp016.p3-phase-probe-overlays.v1",
        f"{label} root overlay schema drift",
    )
    require(
        root_identity.get("event_abi") == EVENT_ABI, f"{label} root overlay ABI drift"
    )
    require(
        arm_identity.get("schema") == "exp016.p3-phase-probe-overlay.v1",
        f"{label} arm overlay schema drift",
    )
    require(
        arm_identity.get("arm") == arm and arm_identity.get("mode") == mode,
        f"{label} arm overlay identity drift",
    )
    require(
        bool(arm_identity.get("probe_enabled")) == (mode == PROBE),
        f"{label} marker flag drift",
    )
    require(
        arm_identity.get("event_abi") == EVENT_ABI, f"{label} arm overlay ABI drift"
    )
    expected_overlay_class = (
        "diagnostic" if mode == PROBE else "marker-disabled ABI control"
    )
    require(
        arm_identity.get("classification") == expected_overlay_class,
        f"{label} arm overlay classification drift",
    )
    base_source = _mapping(arm_identity.get("base"), f"{label}.base source")
    base_kernel_sha = base_source.get("kernel_sha256")
    base_dispatch_sha = base_source.get("dispatch_sha256")
    base_wrapper_sha = base_source.get("wrapper_sha256")
    require(
        base_kernel_sha == EXPECTED_BASE_KERNEL_SHA256[arm],
        f"{label} byte-exact base kernel drift",
    )
    require(
        source.get("base_kernel_sha256") == base_kernel_sha,
        f"{label} source/base kernel disagreement",
    )
    require(
        base_dispatch_sha == EXPECTED_DISPATCH_SHA256,
        f"{label} byte-exact base dispatch drift",
    )
    require(
        base_wrapper_sha == EXPECTED_WRAPPER_SHA256,
        f"{label} byte-exact base wrapper drift",
    )
    for field in ("kernel_path", "dispatch_path", "wrapper_path"):
        require(
            base_source.get(field) not in (None, ""), f"missing {label}.base.{field}"
        )
    registered = _mapping(
        _mapping(root_identity.get("arms"), f"{label}.root arms").get(arm),
        f"{label}.root arm",
    ).get(mode)
    require(
        registered == arm_identity, f"{label} root/arm overlay identity disagreement"
    )
    cross_mode = _mapping(
        _mapping(root_identity.get("cross_mode"), f"{label}.cross mode").get(arm),
        f"{label}.cross mode arm",
    )
    require(
        bool(cross_mode) and all(value is True for value in cross_mode.values()),
        f"{label} matched-source gate failed",
    )
    overlay_source = _mapping(arm_identity.get("overlay"), f"{label}.overlay source")
    require(
        overlay_source.get("kernel_sha256") == source["kernel_sha256"],
        f"{label} kernel SHA disagreement",
    )
    require(
        overlay_source.get("dispatch_sha256") == source["dispatch_sha256"],
        f"{label} dispatch SHA disagreement",
    )
    require(
        overlay_source.get("barrier_fingerprint")
        == base_source.get("barrier_fingerprint"),
        f"{label} overlay changed synchronization topology",
    )
    boundary = _mapping(arm_identity.get("boundary"), f"{label}.probe boundary")
    require(
        dict(boundary)
        == {
            "start_anchor_count": 1,
            "end_anchor_count": 1,
            "timer_read_call_sites": 2,
            "new_barriers": 0,
        },
        f"{label} probe boundary drift",
    )

    runtime = audit_runtime(value, source, label)
    fixture_identity = _mapping(
        value.get("fixture_identity"), f"{label}.fixture identity"
    )
    weight_identity = _mapping(value.get("weight_identity"), f"{label}.weight identity")
    require(is_sha256(value.get("reference_sha256")), f"{label} reference SHA invalid")

    eager = _mapping(value.get("eager"), f"{label}.eager")
    require(eager.get("gate_pass") is True, f"{label} eager gate failed")
    for field in (
        "correctness_gate",
        "route_task_gate",
        "specialization_gate",
        "p3_timing",
    ):
        _gate(eager.get(field), f"{label}.eager.{field}")

    runs = value.get("runs")
    require(
        isinstance(runs, list) and len(runs) == EXPECTED_REPLAYS,
        f"{label} replay count drift",
    )
    e2e_samples = []
    wall_samples = []
    for index, run_value in enumerate(runs):
        run = _mapping(run_value, f"{label}.run[{index}]")
        require(run.get("replay") == index, f"{label} replay index drift")
        require(run.get("gate_pass") is True, f"{label} replay gate failed")
        for field in ("correctness_gate", "route_task_gate", "p3_timing"):
            _gate(run.get(field), f"{label}.run[{index}].{field}")
        require(is_sha256(run.get("output_sha256")), f"{label} output SHA invalid")
        require(is_sha256(run.get("ticks_sha256")), f"{label} tick SHA invalid")
        e2e_samples.append(
            finite(run.get("event_elapsed_us"), f"{label} E2E", positive=True)
        )
        timing = _mapping(run.get("p3_timing"), f"{label}.run[{index}].timing")
        wall = timing.get("grid_critical_wall_us")
        if mode == CONTROL:
            require(
                wall is None and timing.get("all_sentinel") is True,
                f"{label} control marker wrote events",
            )
        else:
            wall_samples.append(finite(wall, f"{label} P3 wall", positive=True))
            require(
                timing.get("additive_estimate_reported") is False,
                f"{label} reported additive estimate",
            )

    e2e = _mapping(value.get("probe_e2e_us"), f"{label}.probe E2E")
    close(e2e.get("median"), statistics.median(e2e_samples), f"{label} E2E median")
    close(e2e.get("min"), min(e2e_samples), f"{label} E2E min")
    close(e2e.get("max"), max(e2e_samples), f"{label} E2E max")
    require(e2e.get("samples") == EXPECTED_REPLAYS, f"{label} E2E sample count drift")
    phase = _mapping(value.get("p3_summary"), f"{label}.P3 summary")
    if mode == CONTROL:
        require(
            phase.get("grid_critical_wall_us") is None,
            f"{label} control has phase wall",
        )
        phase_median = None
    else:
        phase_wall = _mapping(phase.get("grid_critical_wall_us"), f"{label}.phase wall")
        close(
            phase_wall.get("median"),
            statistics.median(wall_samples),
            f"{label} P3 median",
        )
        close(phase_wall.get("min"), min(wall_samples), f"{label} P3 min")
        close(phase_wall.get("max"), max(wall_samples), f"{label} P3 max")
        require(
            phase_wall.get("samples") == EXPECTED_REPLAYS,
            f"{label} P3 sample count drift",
        )
        require(
            phase.get("additive_estimate_reported") is False,
            f"{label} phase summary is additive",
        )
        phase_median = float(phase_wall["median"])

    artifact_set, cubin = audit_artifacts(value, label)
    resource = audit_resource(value, cubin, path, label)
    sass_spill = audit_sass_spill(
        path,
        arm=arm,
        mode=mode,
        cubin_sha256=cubin,
        kernel_symbol=str(resource["kernel_symbol"]),
    )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "arm": arm,
        "mode": mode,
        "source": dict(source),
        "base_source": dict(base_source),
        "base_source_identity_sha256": canonical_sha256(base_source),
        "root_overlay_identity_sha256": canonical_sha256(root_identity),
        "arm_overlay_identity_sha256": canonical_sha256(arm_identity),
        "runtime_signature": runtime,
        "fixture_identity_sha256": canonical_sha256(fixture_identity),
        "weight_identity_sha256": canonical_sha256(weight_identity),
        "reference_sha256": value["reference_sha256"],
        "artifact_set_sha256": artifact_set,
        "cubin_sha256": cubin,
        "resource": resource,
        "sass_spill": sass_spill,
        "e2e_median_us": float(e2e["median"]),
        "p3_grid_critical_wall_median_us": phase_median,
        "p3_grid_critical_wall_samples_us": wall_samples,
        "capture_gate_pass": True,
    }


def build_evidence(results: Path) -> dict[str, Any]:
    paths = {
        (arm, mode): capture_path(results, arm, mode) for arm in ARMS for mode in MODES
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        return {
            "schema": "exp016.p3-phase-evidence.v1",
            "status": "Unresolved",
            "gate_pass": False,
            "capture_integrity_gate_pass": False,
            "evidence_classification": "diagnostic/unresolved",
            "missing_captures": missing,
            "required_capture_count": 4,
            "observed_capture_count": 4 - len(missing),
            "reason": "required P3 control/probe captures are incomplete",
        }

    missing_sass_spill = [
        str(sass_spill_path(path))
        for path in paths.values()
        if not sass_spill_path(path).is_file()
    ]
    if missing_sass_spill:
        return {
            "schema": "exp016.p3-phase-evidence.v1",
            "status": "Unresolved",
            "gate_pass": False,
            "capture_integrity_gate_pass": False,
            "instrumentation_gate_pass": False,
            "sass_spill_gate_pass": False,
            "evidence_classification": "diagnostic/unresolved",
            "missing_sass_spill_sidecars": missing_sass_spill,
            "required_sass_spill_sidecar_count": 4,
            "observed_sass_spill_sidecar_count": 4 - len(missing_sass_spill),
            "reason": "required cubin-locked P3 SASS spill sidecars are incomplete",
        }

    audited = {
        key: audit_capture(read_json(path), path, arm=key[0], mode=key[1])
        for key, path in paths.items()
    }
    runtime_hashes = {
        canonical_sha256(record["runtime_signature"]) for record in audited.values()
    }
    require(
        len(runtime_hashes) == 1, "P3 captures differ in GPU/toolchain/runtime identity"
    )
    root_overlay_hashes = {
        record["root_overlay_identity_sha256"] for record in audited.values()
    }
    require(
        len(root_overlay_hashes) == 1,
        "P3 captures differ in root overlay identity",
    )
    fixture_hashes = {record["fixture_identity_sha256"] for record in audited.values()}
    weight_hashes = {record["weight_identity_sha256"] for record in audited.values()}
    reference_hashes = {record["reference_sha256"] for record in audited.values()}
    require(len(fixture_hashes) == 1, "P3 captures differ in fixture identity")
    require(len(weight_hashes) == 1, "P3 captures differ in weight identity")
    require(len(reference_hashes) == 1, "P3 captures differ in reference output")
    for arm in ARMS:
        base_hashes = {
            audited[(arm, mode)]["base_source_identity_sha256"] for mode in MODES
        }
        require(
            len(base_hashes) == 1,
            f"P3 {arm} control/probe base source identity differs",
        )
    for mode in MODES:
        dispatch_hashes = {
            audited[(arm, mode)]["source"]["dispatch_sha256"] for arm in ARMS
        }
        require(
            len(dispatch_hashes) == 1,
            f"P3 {mode} dispatch source differs across arms",
        )

    instrumentation: dict[str, Any] = {}
    for arm in ARMS:
        control = audited[(arm, CONTROL)]
        probe = audited[(arm, PROBE)]
        require(
            control["source"]["kernel_sha256"] == probe["source"]["kernel_sha256"],
            f"{arm} control/probe instrumented kernel source differs",
        )
        require(
            control["source"]["base_kernel_sha256"]
            == probe["source"]["base_kernel_sha256"],
            f"{arm} control/probe base source differs",
        )
        control_resource = control["resource"]
        probe_resource = probe["resource"]
        compared_fields = (
            "registers_per_thread",
            "stack_bytes_per_thread",
            "static_shared_bytes_per_cta",
            "static_local_bytes_outside_stack",
        )
        resource_equal = all(
            control_resource[field] == probe_resource[field]
            for field in compared_fields
        )
        perturbation = 100.0 * (probe["e2e_median_us"] / control["e2e_median_us"] - 1.0)
        e2e_small = abs(perturbation) <= MAX_ABS_PROBE_E2E_PERTURBATION_PERCENT
        control_spill = control["sass_spill"]
        probe_spill = probe["sass_spill"]
        sass_spill_gate = bool(control_spill["gate_pass"] and probe_spill["gate_pass"])
        instrumentation[arm] = {
            "control_e2e_median_us": control["e2e_median_us"],
            "probe_e2e_median_us": probe["e2e_median_us"],
            "probe_e2e_perturbation_percent": perturbation,
            "max_abs_allowed_perturbation_percent": (
                MAX_ABS_PROBE_E2E_PERTURBATION_PERCENT
            ),
            "e2e_perturbation_small": e2e_small,
            "control_resource": {
                field: control_resource[field] for field in compared_fields
            },
            "probe_resource": {
                field: probe_resource[field] for field in compared_fields
            },
            "resource_identity_equal": resource_equal,
            "control_sass_spill": control_spill,
            "probe_sass_spill": probe_spill,
            "sass_spill_gate_pass": sass_spill_gate,
            "gate_pass": resource_equal and e2e_small and sass_spill_gate,
        }

    baseline_wall = audited[(BASELINE, PROBE)]["p3_grid_critical_wall_median_us"]
    candidate_wall = audited[(CANDIDATE, PROBE)]["p3_grid_critical_wall_median_us"]
    assert baseline_wall is not None and candidate_wall is not None
    candidate_minus_baseline = candidate_wall - baseline_wall
    latency_reduction = 100.0 * (baseline_wall - candidate_wall) / baseline_wall
    instrumentation_gate = all(
        record["gate_pass"] for record in instrumentation.values()
    )
    sass_spill_gate = all(
        record["sass_spill_gate_pass"] for record in instrumentation.values()
    )
    baseline_samples = audited[(BASELINE, PROBE)]["p3_grid_critical_wall_samples_us"]
    candidate_samples = audited[(CANDIDATE, PROBE)]["p3_grid_critical_wall_samples_us"]
    candidate_faster = candidate_wall < baseline_wall
    all_candidate_samples_faster = max(candidate_samples) < min(baseline_samples)
    phase_gate = candidate_faster and all_candidate_samples_faster
    overall_gate = instrumentation_gate and phase_gate
    classification = "diagnostic" if overall_gate else "diagnostic/unresolved"
    capture_records = {
        arm: {mode: audited[(arm, mode)] for mode in MODES} for arm in ARMS
    }
    return {
        "schema": "exp016.p3-phase-evidence.v1",
        "status": "Complete",
        "gate_pass": overall_gate,
        "capture_integrity_gate_pass": True,
        "instrumentation_gate_pass": instrumentation_gate,
        "sass_spill_gate_pass": sass_spill_gate,
        "phase_improvement_gate_pass": phase_gate,
        "evidence_classification": classification,
        "performance_authority": "uninstrumented exp016 E2E only",
        "phase_statistic": "max(all CTA end) - min(all CTA start)",
        "additive_sm_estimate_used": False,
        "environment": next(iter(audited.values()))["runtime_signature"],
        "input_identity": {
            "fixture_sha256": next(iter(fixture_hashes)),
            "weight_sha256": next(iter(weight_hashes)),
            "reference_sha256": next(iter(reference_hashes)),
        },
        "instrumentation": instrumentation,
        "phase_comparison": {
            "baseline_grid_critical_wall_median_us": baseline_wall,
            "candidate_grid_critical_wall_median_us": candidate_wall,
            "candidate_minus_baseline_us": candidate_minus_baseline,
            "latency_reduction_percent": latency_reduction,
            "candidate_faster": candidate_faster,
            "all_candidate_samples_faster": all_candidate_samples_faster,
            "interpretation_legal_as_diagnostic": overall_gate,
            "interpretation_legal_as_production_phase_truth": False,
        },
        "captures": capture_records,
        "evidence_boundary": (
            "A resource mismatch or material probe E2E perturbation makes the phase "
            "comparison diagnostic/unresolved. Even a passing matched probe remains "
            "diagnostic and never replaces uninstrumented E2E."
        ),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def render_markdown(evidence: Mapping[str, Any]) -> str:
    lines = ["# exp_016 P3 Phase Evidence", ""]
    if evidence.get("status") != "Complete":
        missing_kind = (
            "SASS spill sidecar"
            if evidence.get("missing_sass_spill_sidecars")
            else "capture"
        )
        missing_count = len(
            evidence.get("missing_sass_spill_sidecars")
            or evidence.get("missing_captures", [])
        )
        lines.extend(
            [
                f"**结论：Unresolved。** 四份必需 {missing_kind} 尚未齐全。",
                "",
                f"缺失：`{missing_count}` 份。",
                "",
            ]
        )
        return "\n".join(lines)

    comparison = _mapping(evidence.get("phase_comparison"), "phase comparison")
    instrumentation = _mapping(evidence.get("instrumentation"), "instrumentation")
    gate = bool(evidence.get("gate_pass"))
    verdict = "可用的 diagnostic 证据" if gate else "diagnostic/unresolved"
    lines.extend(
        [
            f"**结论：{verdict}。** 未插桩 E2E 仍是唯一性能判定依据。",
            "",
            "| P3 grid critical wall | Baseline | Candidate | 差值 | 降幅 |",
            "|---|---:|---:|---:|---:|",
            "| Median | "
            f"{_fmt(comparison['baseline_grid_critical_wall_median_us'])} µs | "
            f"{_fmt(comparison['candidate_grid_critical_wall_median_us'])} µs | "
            f"{_fmt(comparison['candidate_minus_baseline_us'])} µs | "
            f"{_fmt(comparison['latency_reduction_percent'], 2)}% |",
            "",
            "| Arm | REG control→probe | STACK control→probe | SMEM control→probe | LOCAL control→probe | SASS SpillRefill/LDL/STL control→probe | Probe E2E 扰动 | 结论 |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for arm in ARMS:
        record = _mapping(instrumentation[arm], f"instrumentation.{arm}")
        control = _mapping(record["control_resource"], "control resource")
        probe = _mapping(record["probe_resource"], "probe resource")
        control_spill = _mapping(record["control_sass_spill"], "control spill")
        probe_spill = _mapping(record["probe_sass_spill"], "probe spill")
        control_counts = _mapping(control_spill["counts"], "control spill counts")
        probe_counts = _mapping(probe_spill["counts"], "probe spill counts")
        control_spill_text = "/".join(
            str(control_counts[field])
            for field in (
                "spill_refill_annotation_count",
                "ldl_opcode_count",
                "stl_opcode_count",
            )
        )
        probe_spill_text = "/".join(
            str(probe_counts[field])
            for field in (
                "spill_refill_annotation_count",
                "ldl_opcode_count",
                "stl_opcode_count",
            )
        )
        lines.append(
            f"| {arm} | {control['registers_per_thread']}→{probe['registers_per_thread']} | "
            f"{control['stack_bytes_per_thread']}→{probe['stack_bytes_per_thread']} B | "
            f"{control['static_shared_bytes_per_cta']}→{probe['static_shared_bytes_per_cta']} B | "
            f"{control['static_local_bytes_outside_stack']}→{probe['static_local_bytes_outside_stack']} B | "
            f"{control_spill_text}→{probe_spill_text} | "
            f"{_fmt(record['probe_e2e_perturbation_percent'], 2)}% | "
            f"{'通过' if record['gate_pass'] else 'Unresolved'} |"
        )
    lines.extend(
        [
            "",
            "口径：`max(all CTA end) - min(all CTA start)`；未使用 additive SM-equivalent estimate。",
            "",
        ]
    )
    if not gate:
        lines.append(
            "资源身份、SASS zero-spill 或插桩 E2E 扰动门未通过，因此上表 P3 数值不得解释为 production phase 真值。"
        )
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args(argv)
    evidence = build_evidence(args.results.resolve())
    markdown = render_markdown(evidence)
    write_json(args.json_output.resolve(), evidence)
    args.markdown_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.resolve().write_text(markdown, encoding="utf-8")
    print(json.dumps(evidence, sort_keys=True))
    return 0 if evidence.get("gate_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
