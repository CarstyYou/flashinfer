#!/usr/bin/env python3
"""Build the compact, machine-checkable exp_011 evidence card."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


VARIANTS = (
    "identity",
    "shared_equal_scale",
    "static_schedule",
    "precomputed_phys_row",
)
SELECTED_COUNTERS = (
    "gpu__time_duration.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
    "smsp__sass_thread_inst_executed_op_conversion_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_memory_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_bit_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_integer_pred_on.sum",
    "sm__inst_executed.sum",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
    "launch__registers_per_thread",
    "launch__stack_size",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values: Iterable[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    rmad = 1.4826 * mad / median if median else 0.0
    return {
        "samples": samples,
        "p50": median,
        "min": min(samples),
        "max": max(samples),
        "rMAD": rmad,
    }


def require_capture_gates(capture: dict[str, Any], *, label: str) -> None:
    runs = capture.get("runs")
    if not isinstance(runs, list) or len(runs) != 5:
        raise ValueError(f"{label}: expected five capture runs")
    for run in runs:
        gates = (
            run["correctness_gate"]["gate_pass"],
            run["workspace_gate"]["gate_pass"],
            run["event_gate"]["gate_pass"],
        )
        if not all(gates):
            raise ValueError(f"{label}: run gate failed: {run['run_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = args.results.resolve()
    raw = results / "raw"
    derived = results / "derived"
    output = args.output.resolve() if args.output else derived / "summary.json"

    arms: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}
    for variant in VARIANTS:
        probe_root = raw / f"{variant}_probe"
        control_root = raw / f"{variant}_no_marker"
        timing_path = probe_root / "timing_summary.json"
        probe_path = probe_root / "capture.json"
        control_path = control_root / "capture.json"
        timing = read_json(timing_path)
        probe = read_json(probe_path)
        control = read_json(control_path)
        require_capture_gates(probe, label=f"{variant}/probe")
        require_capture_gates(control, label=f"{variant}/no_marker")
        phase_samples: dict[str, list[float]] = {
            phase: [
                float(replay["phase_totals_ns"][phase]) / 110.0 / 1000.0
                for replay in timing["replays"]
            ]
            for phase in ("P1", "P2", "P3")
        }
        stage_samples = [
            sum(phase_samples[phase][index] for phase in ("P1", "P2", "P3"))
            for index in range(5)
        ]

        metrics_path = derived / f"ncu_{variant}_metrics.json"
        launches_path = derived / f"ncu_{variant}_launches.json"
        summary_path = derived / f"ncu_{variant}_summary.json"
        metric_payload = read_json(metrics_path)
        launch_payload = read_json(launches_path)
        ncu_summary = read_json(summary_path)
        rows = metric_payload["data"]["rows"]
        launches = launch_payload["data"]["rows"]
        if len(rows) != 1 or len(launches) != 1:
            raise ValueError(f"{variant}: expected exactly one NCU launch")
        launch = launches[0]
        if launch["grid_size"] != [1, 1, 110] or launch["block_size"] != [160, 1, 1]:
            raise ValueError(f"{variant}: NCU launch identity drift")
        counters = rows[0]["counters"]
        compact_counters = {name: counters[name] for name in SELECTED_COUNTERS}
        trace_path = raw / "ncu" / variant / "trace.ncu-rep"
        ncu_cubin_path = raw / "ncu" / variant / "kernel.cubin"
        probe_cubins = [
            artifact
            for artifact in probe["jit_artifacts"]
            if str(artifact["path"]).endswith(".cubin")
        ]
        timing_cubins = [
            artifact
            for artifact in control["jit_artifacts"]
            if str(artifact["path"]).endswith(".cubin")
        ]
        if len(probe_cubins) != 1 or len(timing_cubins) != 1:
            raise ValueError(f"{variant}: expected one probe and one timing cubin")
        ncu_cubin_sha = sha256(ncu_cubin_path)
        if ncu_cubin_sha != timing_cubins[0]["sha256"]:
            raise ValueError(f"{variant}: timing/NCU cubin identity drift")

        arms[variant] = {
            "phase_us": {
                phase: stats(samples) for phase, samples in phase_samples.items()
            },
            "p1_p2_p3_stage_us": stats(stage_samples),
            "probe_event_us": stats(run["event_elapsed_us"] for run in probe["runs"]),
            "no_marker_event_us": stats(
                run["event_elapsed_us"] for run in control["runs"]
            ),
            "correctness_and_workspace_gates": "pass",
            "arm_contract": probe["exp011"]["arm_contract"],
            "binary_identity": {
                "probe_cubin_sha256": probe_cubins[0]["sha256"],
                "no_marker_cubin_sha256": timing_cubins[0]["sha256"],
                "probe_equals_no_marker": (
                    probe_cubins[0]["sha256"] == timing_cubins[0]["sha256"]
                ),
            },
            "ncu": {
                "version": ncu_summary["data"]["auxiliary"]["ncu_version"],
                "trace_sha256": sha256(trace_path),
                "cubin_sha256": ncu_cubin_sha,
                "same_cubin_as_no_marker_timing": True,
                "launch": {
                    "row_id": launch["row_id"],
                    "grid": launch["grid_size"],
                    "block": launch["block_size"],
                },
                "counters": compact_counters,
            },
        }
        for path in (
            timing_path,
            probe_path,
            control_path,
            trace_path,
            ncu_cubin_path,
        ):
            input_hashes[str(path.relative_to(results))] = sha256(path)

    identity = arms["identity"]
    identity_p3 = identity["phase_us"]["P3"]["p50"]
    identity_e2e = identity["no_marker_event_us"]["p50"]
    for arm in arms.values():
        arm["relative_to_identity_pct"] = {
            "P3": (arm["phase_us"]["P3"]["p50"] / identity_p3 - 1.0) * 100.0,
            "no_marker_event": (arm["no_marker_event_us"]["p50"] / identity_e2e - 1.0)
            * 100.0,
        }

    exp010 = results.parent.parent / "exp_010_scatter_vs_chain_finalize" / "results"
    production_summary_path = exp010 / "derived" / "production_phase_summary.json"
    production = read_json(production_summary_path)
    exp004_timing_path = (
        results.parent.parent
        / "exp_004_fused_phase_timing_breakdown"
        / "results"
        / "whole_kernel_timing.json"
    )
    exp004_timing = read_json(exp004_timing_path)
    legacy_p3_samples = [
        float(replay["phase_totals_ns"]["P3"]) / 110.0 / 1000.0
        for replay in exp004_timing["replays"]
    ]
    legacy_p3 = stats(legacy_p3_samples)
    legacy_p3_aggregate_mean = (
        float(exp004_timing["aggregate"]["phase_totals_ns"]["P3"])
        / len(exp004_timing["replays"])
        / 110.0
        / 1000.0
    )
    identity_old_delta = abs(identity_p3 / legacy_p3["p50"] - 1.0)
    production_control = float(production["whole_event_us"]["no_marker_control_median"])
    identity_control_drift = abs(identity_e2e / production_control - 1.0)
    probe_event = identity["probe_event_us"]["p50"]
    probe_perturbation = probe_event / identity_e2e - 1.0
    fidelity = {
        "legacy_capture_p3_replay_us": legacy_p3,
        "legacy_published_p3_aggregate_mean_us": legacy_p3_aggregate_mean,
        "fresh_probe_p3_replay_p50_us": identity_p3,
        "p3_replay_p50_delta_pct": identity_old_delta * 100.0,
        "p3_statistic_comparable": True,
        "p3_classification": "diagnostic phase anchor; not production-exact timing",
        "production_control_event_us": production_control,
        "fresh_identity_no_marker_event_us": identity_e2e,
        "event_drift_pct": identity_control_drift * 100.0,
        "event_acceptance_pct": 1.0,
        "event_gate_pass": identity_control_drift <= 0.01,
        "fresh_probe_event_us": probe_event,
        "probe_vs_no_marker_perturbation_pct": probe_perturbation * 100.0,
        "probe_equals_no_marker_cubin": identity["binary_identity"][
            "probe_equals_no_marker"
        ],
        "phase_estimate_classification": (
            "full-kernel diagnostic probe; no-marker event is the "
            "production-fidelity wall control"
        ),
        "all_correctness_workspace_gates_pass": True,
    }

    chain_path = derived / "actual_chain_anchor.json"
    chain = read_json(chain_path)
    chain_metrics_path = derived / "ncu_chain_expand_metrics.json"
    chain_launches_path = derived / "ncu_chain_expand_launches.json"
    chain_summary_path = derived / "ncu_chain_expand_summary.json"
    chain_metrics = read_json(chain_metrics_path)["data"]["rows"]
    chain_launches = read_json(chain_launches_path)["data"]["rows"]
    chain_ncu_summary = read_json(chain_summary_path)
    if len(chain_metrics) != 1 or len(chain_launches) != 1:
        raise ValueError("actual Chain Expand must have exactly one NCU launch")
    chain_launch = chain_launches[0]
    if chain_launch["grid_size"] != [880, 1, 1] or chain_launch["block_size"] != [
        256,
        1,
        1,
    ]:
        raise ValueError("actual Chain Expand NCU launch identity drift")
    chain_counters = chain_metrics[0]["counters"]
    chain_trace = raw / "ncu" / "chain_expand" / "trace.ncu-rep"
    chain["ncu"] = {
        "version": chain_ncu_summary["data"]["auxiliary"]["ncu_version"],
        "trace_sha256": sha256(chain_trace),
        "launch": {
            "row_id": chain_launch["row_id"],
            "grid": chain_launch["grid_size"],
            "block": chain_launch["block_size"],
        },
        "counters": {name: chain_counters[name] for name in SELECTED_COUNTERS},
        "comparison_note": "independent Expand launch; whole-Fused counters cannot be projected to P3",
    }
    payload = {
        "schema": "exp011.route-q0-pack-summary.v1",
        "case": {"M": 8192, "E": 256, "H": 2048, "topk": 8},
        "fidelity": fidelity,
        "arms": arms,
        "actual_chain": chain,
        "input_hashes": {
            **input_hashes,
            str(production_summary_path.relative_to(results.parent.parent)): sha256(
                production_summary_path
            ),
            str(exp004_timing_path.relative_to(results.parent.parent)): sha256(
                exp004_timing_path
            ),
            str(chain_path.relative_to(results)): sha256(chain_path),
            str(chain_trace.relative_to(results)): sha256(chain_trace),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "fidelity": fidelity}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
