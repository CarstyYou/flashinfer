#!/usr/bin/env python3
"""Seal the canonical exp_004 whole-kernel diagnostic evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
TIMING_PATH = RESULTS / "whole_kernel_timing.json"
SUMMARY_PATH = RESULTS / "derived" / "whole_kernel_capture_summary.json"
MANIFEST_PATH = RESULTS / "manifest.json"

PHASE_ORDER = (
    "launch_skew/early_finish_idle",
    "prologue",
    "P0",
    "P1",
    "P2",
    "P3",
    "P4",
    "compute_setup",
    "T0_claim",
    "T0_cache_setup",
    "Gate",
    "Up",
    "SwiGLU_Q1",
    "FC2_setup",
    "FC2_gemm",
    "FC2_epilogue_scatter",
    "task_control_final_drain",
)


class FinalizationError(ValueError):
    """The accepted whole-kernel evidence package is inconsistent."""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalizationError(f"{path} must contain a JSON object")
    return value


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, relative_to: Path = ROOT) -> dict[str, Any]:
    if not path.is_file():
        raise FinalizationError(f"missing required artifact: {path}")
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def rendered_record(
    path: Path, payload: bytes, *, relative_to: Path = ROOT
) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _latencies(capture: Mapping[str, Any]) -> list[float]:
    runs = capture.get("runs")
    if not isinstance(runs, list) or len(runs) != 5:
        raise FinalizationError("each accepted capture must contain five runs")
    values = [float(run["event_elapsed_us"]) for run in runs]
    if any(not bool(run["correctness_gate"]["gate_pass"]) for run in runs):
        raise FinalizationError("all accepted capture runs must pass correctness")
    return values


def normalize_timing(timing: dict[str, Any]) -> dict[str, Any]:
    if timing.get("schema") != "exp004.whole-kernel-timing.v1":
        raise FinalizationError("unexpected whole-kernel timing schema")
    aggregate = timing.get("aggregate")
    if not isinstance(aggregate, dict) or not aggregate.get("closure", {}).get("pass"):
        raise FinalizationError("whole-kernel timing closure did not pass")
    phases = tuple(item["phase"] for item in aggregate.get("phases", []))
    if phases != PHASE_ORDER:
        raise FinalizationError(f"whole-kernel phase coverage mismatch: {phases}")
    if int(aggregate["closure"]["delta_ns"]) != 0:
        raise FinalizationError("whole-kernel timing closure delta is nonzero")
    if len(timing.get("replays", [])) != 5:
        raise FinalizationError("whole-kernel timing requires five replays")

    inputs = timing.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 5:
        raise FinalizationError("whole-kernel timing must bind five raw timing files")
    for entry in inputs:
        basename = Path(str(entry["path"])).name
        stable = Path("raw") / "whole_kernel_probe" / basename
        local = RESULTS / stable
        if file_sha256(local) != entry["sha256"]:
            raise FinalizationError(f"timing input hash mismatch: {local}")
        entry["path"] = stable.as_posix()
    return timing


def build_summary(
    prior: Mapping[str, Any],
    timing: Mapping[str, Any],
    timing_payload: bytes,
    control: Mapping[str, Any],
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    control_values = _latencies(control)
    probe_values = _latencies(probe)
    globaltimer_values = [
        float(replay["global_wall_ns"]) / 1000.0 for replay in timing["replays"]
    ]
    control_mean = statistics.mean(control_values)
    probe_mean = statistics.mean(probe_values)
    globaltimer_mean = statistics.mean(globaltimer_values)
    control_median = statistics.median(control_values)
    probe_median = statistics.median(probe_values)
    aggregate = timing["aggregate"]
    denominator_ns = int(aggregate["sm_equivalent_denominator_ns"])
    largest_span = max(
        max(
            float(replay["phase_totals_ns"][phase])
            / float(replay["sm_equivalent_denominator_ns"])
            * 100.0
            for replay in timing["replays"]
        )
        - min(
            float(replay["phase_totals_ns"][phase])
            / float(replay["sm_equivalent_denominator_ns"])
            * 100.0
            for replay in timing["replays"]
        )
        for phase in PHASE_ORDER
    )

    control_runtime = control["runtime"]
    probe_runtime = probe["runtime"]
    identity_fields = (
        "cuda_runtime",
        "image_digest",
        "nvcc",
        "ptxas",
        "python",
        "python_deps_sha256",
        "torch",
    )
    for field in identity_fields:
        if control_runtime[field] != probe_runtime[field]:
            raise FinalizationError(f"control/probe runtime identity drift: {field}")
    if control_runtime["lease_id"] != probe_runtime["lease_id"]:
        raise FinalizationError("control/probe lease identity drift")
    if control_runtime["gpu"]["uuid"] != probe_runtime["gpu"]["uuid"]:
        raise FinalizationError("control/probe GPU identity drift")

    binary_identity = dict(prior["binary_identity"])
    if binary_identity.get("resource_identity_pass") is not False:
        raise FinalizationError("probe resource drift must remain diagnostic")

    static_records: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ("whole_kernel_control", "whole_kernel_probe"):
        static_records[arm] = {}
        for filename in (
            "kernel.cubin",
            "kernel.ptx",
            "kernel.sass",
            "resource_usage.txt",
        ):
            path = RAW / "whole_kernel_static" / arm / filename
            static_records[arm][filename] = file_record(path, relative_to=RESULTS)
    expected_static = {
        "whole_kernel_control": {
            "kernel.cubin": binary_identity["control"]["cubin_sha256"],
            "kernel.ptx": binary_identity["control"]["ptx_sha256"],
            "kernel.sass": binary_identity["control"]["sass_sha256"],
        },
        "whole_kernel_probe": {
            "kernel.cubin": binary_identity["probe"]["cubin_sha256"],
            "kernel.ptx": binary_identity["probe"]["ptx_sha256"],
            "kernel.sass": binary_identity["probe"]["sass_sha256"],
        },
    }
    for arm, expected in expected_static.items():
        for filename, digest in expected.items():
            if static_records[arm][filename]["sha256"] != digest:
                raise FinalizationError(
                    f"static artifact identity mismatch: {arm}/{filename}"
                )

    timing_inputs = []
    for entry in timing["inputs"]:
        path = RESULTS / entry["path"]
        timing_inputs.append(file_record(path, relative_to=RESULTS))

    gpu = probe_runtime["gpu"]
    imports = probe_runtime["imports"]
    source = probe["source"]
    event_gates = [run["event_gate"] for run in probe["runs"]]
    correctness = [run["correctness"] for run in probe["runs"]]
    workspaces = [run["workspace_gate"] for run in probe["runs"]]
    if any(not gate["gate_pass"] for gate in event_gates):
        raise FinalizationError("all probe event gates must pass")
    if any(not workspace["gate_pass"] for workspace in workspaces):
        raise FinalizationError("all probe workspace gates must pass")
    first_event_gate = event_gates[0]
    if any(gate != first_event_gate for gate in event_gates[1:]):
        raise FinalizationError("probe event-gate population drift across replays")
    summary = {
        "schema": "exp004.whole-kernel-capture-summary.v2",
        "classification": "diagnostic-only",
        "case": prior["case"],
        "measurement_boundary": {
            "coverage": "entry/prologue through final W4 producer tail and return",
            "consumer_marker": (
                "W0 lane0 records representative CTA consumer phase boundaries; "
                "not a W0-W3 instruction-time sum or exact warp-union attribution"
            ),
            "consumer_interval_content": (
                "phase elapsed intervals include synchronization/wait inside their boundaries"
            ),
            "residual": (
                "task_control_final_drain includes inter-task gaps, final no-task "
                "claim/exit, and W4 producer tail"
            ),
            "overlap": "W4 task producer intervals are non-additive",
        },
        "denominator": {
            "name": "SM-equivalent wall",
            "definition": "grid_z * (max CTA final globaltimer - min CTA entry globaltimer)",
            "timestamp_unit": "globaltimer_ns",
            "replays": len(timing["replays"]),
            "aggregate_ns": denominator_ns,
            "average_equivalent_wall_us": globaltimer_mean,
            "closure_delta_ns": int(aggregate["closure"]["delta_ns"]),
            "closure_pass": bool(aggregate["closure"]["pass"]),
        },
        "event_and_correctness_gate": {
            "probe_replays_passed": sum(
                bool(run["correctness_gate"]["gate_pass"]) for run in probe["runs"]
            ),
            "probe_replays_total": len(probe["runs"]),
            "relative_l2_range": [
                min(float(value["relative_l2"]) for value in correctness),
                max(float(value["relative_l2"]) for value in correctness),
            ],
            "max_abs_upper_bound": max(
                float(value["max_abs"]) for value in correctness
            ),
            "workspace_gate_pass": all(
                bool(workspace["gate_pass"]) for workspace in workspaces
            ),
            "task_events_per_replay": {
                "actual": int(first_event_gate["actual_task_writes"]),
                "expected": int(first_event_gate["expected_task_writes"]),
            },
            "cta_events_per_replay": {
                "actual": int(first_event_gate["actual_cta_writes"]),
                "expected": int(first_event_gate["expected_cta_writes"]),
            },
            "mapped_tasks_per_replay": int(first_event_gate["mapped_tasks"]),
            "sentinel_task_slots_per_replay": int(first_event_gate["task_capacity"])
            - int(first_event_gate["task_tail"]),
            "foreign_process_on_target_gpu": bool(probe["foreign_processes_after"]),
        },
        "cross_checks": {
            "control_cuda_event_mean_us": control_mean,
            "control_cuda_event_median_us": control_median,
            "probe_cuda_event_mean_us": probe_mean,
            "probe_cuda_event_median_us": probe_median,
            "probe_globaltimer_mean_us": globaltimer_mean,
            "cuda_event_vs_globaltimer_statistic": "probe mean / globaltimer mean",
            "cuda_event_vs_globaltimer_delta_us": probe_mean - globaltimer_mean,
            "cuda_event_vs_globaltimer_delta_pct": 100.0
            * (probe_mean / globaltimer_mean - 1.0),
            "probe_overhead_statistic": "probe median / control median",
            "probe_overhead_us": probe_median - control_median,
            "probe_overhead_pct": 100.0 * (probe_median / control_median - 1.0),
            "largest_phase_run_to_run_span_pp": largest_span,
        },
        "runtime_identity": {
            "gpu_name": gpu["name"],
            "gpu_uuid": gpu["uuid"],
            "pci_bus_id": gpu["pci_bus_id"],
            "compute_capability": gpu["compute_capability"],
            "sm_count": gpu["sm_count"],
            "driver": gpu["driver"],
            "cuda_runtime": probe_runtime["cuda_runtime"],
            "applications_graphics_clock_mhz": int(
                gpu["applications_graphics_clock_mhz"]
            ),
            "container_image_digest": probe_runtime["image_digest"],
            "python": probe_runtime["python"],
            "python_deps_sha256": probe_runtime["python_deps_sha256"],
            "torch": probe_runtime["torch"],
            "cutlass_dsl": imports["cutlass_python_version"],
            "cutlass_module_path": imports["cutlass_python"],
            "cutlass_commit": source["cutlass_commit"],
            "nvcc": probe_runtime["nvcc"],
            "ptxas": probe_runtime["ptxas"],
            "lease_id": probe_runtime["lease_id"],
            "control_jit_artifact_set_sha256": control["jit_artifact_set_sha256"],
            "probe_jit_artifact_set_sha256": probe["jit_artifact_set_sha256"],
            "capture_checkout_head": source["checkout_head"],
            "locked_source_commit": source["locked_source_commit"],
            "production_kernel_sha256": source["production"]["kernel"]["sha256"],
            "production_dispatch_sha256": source["production"]["dispatch"]["sha256"],
            "production_wrapper_sha256": source["production"]["wrapper"]["sha256"],
        },
        "binary_identity": binary_identity,
        "sources": {
            "evidence_host_results_root": str(RESULTS),
            "availability": "raw evidence is retained on the shared evidence host and ignored by git",
            "control_capture": file_record(
                RAW / "whole_kernel_control" / "capture.json", relative_to=RESULTS
            ),
            "probe_capture": file_record(
                RAW / "whole_kernel_probe" / "capture.json", relative_to=RESULTS
            ),
            "timing_inputs": timing_inputs,
            "whole_kernel_timing": rendered_record(
                TIMING_PATH, timing_payload, relative_to=RESULTS
            ),
            "static_artifacts": static_records,
        },
        "cleanup": prior["cleanup"],
    }
    return summary


def build_manifest(
    timing: Mapping[str, Any],
    timing_payload: bytes,
    summary: Mapping[str, Any],
    summary_payload: bytes,
) -> dict[str, Any]:
    scripts = (
        "analyze_whole_kernel_timing.py",
        "build_whole_kernel_probe.py",
        "finalize_whole_kernel.py",
        "run_exp004.py",
        "run_whole_kernel_capture.py",
        "test_analyze_whole_kernel_timing.py",
    )
    artifacts = {
        "plan": file_record(ROOT / "plan.md"),
        "runbook": file_record(ROOT / "runbook.md"),
        "result": file_record(RESULTS / "result.md"),
        "whole_kernel_timing": rendered_record(TIMING_PATH, timing_payload),
        "capture_summary": rendered_record(SUMMARY_PATH, summary_payload),
        "scripts": {name: file_record(ROOT / name) for name in scripts},
    }
    manifest: dict[str, Any] = {
        "schema": "exp004.run-manifest.v3",
        "status": "closed",
        "classification": "diagnostic-only",
        "verdict": "whole_fused_kernel_phase_breakdown_closed",
        "coverage": {
            "phase_order": list(PHASE_ORDER),
            "complete_entry_to_return": True,
            "p3_interleaved_combined": True,
            "consumer_marker": summary["measurement_boundary"]["consumer_marker"],
            "residual": summary["measurement_boundary"]["residual"],
            "w4_overlap_non_additive": True,
        },
        "denominator": summary["denominator"],
        "validation": {
            "correctness_and_event_gate": "PASS_5_OF_5",
            "denominator_closure": timing["aggregate"]["closure"],
            "binary_resource_identity": "FAIL_EXPECTED_DIAGNOSTIC_ONLY",
            "data_audit": "PASS_WITH_DIAGNOSTIC_BOUNDARY",
            "unsupported_causal_claims": False,
        },
        "identity": {
            "runtime": summary["runtime_identity"],
            "binary": summary["binary_identity"],
        },
        "raw_evidence": {
            "evidence_host_results_root": str(RESULTS),
            "control_capture": summary["sources"]["control_capture"],
            "probe_capture": summary["sources"]["probe_capture"],
            "timing_inputs": summary["sources"]["timing_inputs"],
            "static_artifacts": summary["sources"]["static_artifacts"],
        },
        "accepted_artifacts": artifacts,
        "historical_attempts": {
            "status": "superseded",
            "scope": "consumer-only clock64/306-tick probes and zero-write diagnostics",
            "accepted_for_current_phase_share": False,
        },
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    return manifest


def finalize(*, check: bool) -> None:
    timing = normalize_timing(read_json(TIMING_PATH))
    timing_payload = json_bytes(timing)
    prior_summary = read_json(SUMMARY_PATH)
    control = read_json(RAW / "whole_kernel_control" / "capture.json")
    probe = read_json(RAW / "whole_kernel_probe" / "capture.json")
    summary = build_summary(prior_summary, timing, timing_payload, control, probe)
    summary_payload = json_bytes(summary)
    manifest = build_manifest(timing, timing_payload, summary, summary_payload)

    if check:
        checks = (
            (TIMING_PATH, timing),
            (SUMMARY_PATH, summary),
            (MANIFEST_PATH, manifest),
        )
        for path, expected in checks:
            if read_json(path) != expected:
                raise FinalizationError(f"canonical artifact is stale: {path}")
        stored_manifest = read_json(MANIFEST_PATH)
        payload_sha = stored_manifest.pop("manifest_payload_sha256")
        if canonical_sha256(stored_manifest) != payload_sha:
            raise FinalizationError("manifest payload hash is invalid")
        return

    TIMING_PATH.write_bytes(timing_payload)
    SUMMARY_PATH.write_bytes(summary_payload)
    MANIFEST_PATH.write_bytes(json_bytes(manifest))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    finalize(check=arguments.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
