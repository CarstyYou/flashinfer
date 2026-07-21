#!/usr/bin/env python3
"""Shared, CPU-testable contracts for the exp_019 paired phase probe.

The timing prefix is the exp_017 ABI byte-for-byte.  Exp_019 appends a
separate per-CTA occurrence plane to the same allocation so a closed residual
cannot hide a missing interval marker.
"""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
EXP017 = ROOT.parent / "exp_017_opt_vs_triton_phase_share"
EXP018 = ROOT.parent / "exp_018_triton_opt_eric_benchmark"
if str(EXP017) not in sys.path:
    sys.path.insert(0, str(EXP017))

import exp017_opt_phase_common as exp017  # noqa: E402


CONTROL = exp017.CONTROL
PROBE = exp017.PROBE
MODES = exp017.MODES
GRID_CTAS = exp017.GRID_CTAS
PHASE_NAMES = exp017.PHASE_NAMES
EVENTS_PER_CTA = exp017.EVENTS_PER_CTA
EVENT_ABI = exp017.EVENT_ABI

ARMS = ("latest_opt_fp4", "eric_stage4_fp4")
M_VALUES = (1024, 8192)
WARMUPS = 2
REPLAYS = 5
L2_FLUSH_BYTES = 192 << 20
BLOCK_THREADS = {"latest_opt_fp4": 288, "eric_stage4_fp4": 160}

EXPECTED_SOURCE_SHA256 = {
    "latest_opt_fp4": "ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19",
    "eric_stage4_fp4": "3a5000a990bb978b434f1c7dac621de25112d9f3cec4a5fdfab5f2970b0dc3b8",
}
EXPECTED_ERIC_ADAPTER_SHA256 = (
    "98adfba7f4e0d00af24383a556e9c93088355539b50dc82480091225e0448120"
)
EXPECTED_DISPATCH_SHA256 = exp017.EXPECTED_DISPATCH_SHA256
EXPECTED_WRAPPER_SHA256 = exp017.EXPECTED_WRAPPER_SHA256
SOURCE_RELATIVE_PATH = {
    "latest_opt_fp4": Path(".claude/w4a4_moe_bench/moe_dynamic_kernel_opt.py"),
    "eric_stage4_fp4": Path(
        ".claude/w4a4_moe_bench/moe_dyanmice_kernel_ab_stage4_compact.py"
    ),
}
DISPATCH_RELATIVE_PATH = exp017.DISPATCH_RELATIVE_PATH
WRAPPER_RELATIVE_PATH = exp017.WRAPPER_RELATIVE_PATH
DISPATCH_MODULE = exp017.DISPATCH_MODULE

EXPECTED_FIXTURE_MANIFEST_SHA256 = (
    "683ec75341e4d8317dfdc5c4b04229f9695f9aa286d575c4f6e1fdef55d90801"
)
EXPECTED_FIXTURE_SHA256 = {
    1024: "0fa7e8a7d8d1d32172971f987d6f55b534aabf8d12a84a910d010cec25ba04a5",
    8192: "c113ecd5ddeff77154ddbd23fc3dc3c83f8ee822e880179ca5c16b1145372438",
}

# The first GRID_CTAS * EVENTS_PER_CTA values retain the exp_017 timing ABI.
# This independent tail is deliberately not folded into EVENT_ABI.
OCCURRENCES_PER_CTA = len(PHASE_NAMES)
STORAGE_PER_CTA = EVENTS_PER_CTA + OCCURRENCES_PER_CTA
STORAGE_VALUES = GRID_CTAS * STORAGE_PER_CTA
OCCURRENCE_ABI = {
    "schema": "exp019.phase-occurrence-abi.v1",
    "storage": "uint64 tail after the exp017 timing prefix",
    "timing_prefix_values": GRID_CTAS * EVENTS_PER_CTA,
    "grid_ctas": GRID_CTAS,
    "occurrences_per_cta": OCCURRENCES_PER_CTA,
    "phase_names": list(PHASE_NAMES),
    "meaning": "CTA-leader close-marker execution count",
}

SEMANTIC_GROUPS = {
    "clear_histogram_prefix": ("clear_init", "histogram", "prefix"),
    "route_q0_pack_publish": ("route_q0_pack", "publish_route_tail"),
    "claim_cache_control": ("claim_cache_control",),
    "fc1_gate_up_swiglu": ("fc1_gate_up_swiglu",),
    "q1": ("q1",),
    "fc2_epilogue_r2s": ("fc2_epilogue_r2s",),
    "scatter": ("scatter",),
    "cta_residual": ("cta_residual",),
    "launch_skew_early_finish": ("launch_skew_early_finish",),
}

# The compact Eric epilogue has two M64 intervals where Opt has one M128
# interval.  Both arms have 16 N128 output tiles for H=2048.
INTERVAL_FACTORS = {
    "latest_opt_fp4": {
        "swiglu_q1_per_slice": 1,
        "fc2_scatter_per_output_tile": 1,
        "output_tiles_per_slice": 16,
    },
    "eric_stage4_fp4": {
        "swiglu_q1_per_slice": 2,
        "fc2_scatter_per_output_tile": 2,
        "output_tiles_per_slice": 16,
    },
}

# Rotating all four arm/mode cells avoids always placing either arm or marker
# mode at the same thermal/process position.  A launcher may execute any whole
# number of these four blocks; captures record the selected block index.
_BASE_PROCESS_ORDER = (
    ("latest_opt_fp4", CONTROL),
    ("eric_stage4_fp4", CONTROL),
    ("latest_opt_fp4", PROBE),
    ("eric_stage4_fp4", PROBE),
)
CYCLIC_PROCESS_ORDERS = tuple(
    _BASE_PROCESS_ORDER[index:] + _BASE_PROCESS_ORDER[:index]
    for index in range(len(_BASE_PROCESS_ORDER))
)


class PhaseHarnessError(RuntimeError):
    """A source, capture, identity, or paired-noise contract was violated."""


file_sha256 = exp017.file_sha256
canonical_sha256 = exp017.canonical_sha256
text_sha256 = exp017.text_sha256
read_json = exp017.read_json
write_json = exp017.write_json
barrier_fingerprint = exp017.barrier_fingerprint
normalize_dispatch_flag = exp017.normalize_dispatch_flag


def cyclic_process_order(block: int) -> tuple[tuple[str, str], ...]:
    if block < 0:
        raise PhaseHarnessError("cyclic block index must be non-negative")
    return CYCLIC_PROCESS_ORDERS[block % len(CYCLIC_PROCESS_ORDERS)]


def _flat_ints(values: Any) -> list[int]:
    plain = values.tolist() if hasattr(values, "tolist") else values
    if not isinstance(plain, (list, tuple)):
        raise PhaseHarnessError("phase storage must be a flat vector")
    if plain and isinstance(plain[0], (list, tuple)):
        plain = [item for row in plain for item in row]
    return [int(item) for item in plain]


def occurrence_offset(cta: int, phase: str, *, grid_ctas: int = GRID_CTAS) -> int:
    if phase not in PHASE_NAMES:
        raise PhaseHarnessError(f"unknown phase: {phase}")
    if not 0 <= cta < grid_ctas:
        raise PhaseHarnessError(f"CTA out of range: {cta}")
    return (
        grid_ctas * EVENTS_PER_CTA
        + cta * OCCURRENCES_PER_CTA
        + PHASE_NAMES.index(phase)
    )


def validate_phase_storage(
    values: Any, *, mode: str, grid_ctas: int = GRID_CTAS
) -> dict[str, Any]:
    """Validate the reused timing prefix and independent occurrence plane."""
    storage = _flat_ints(values)
    timing_values = grid_ctas * EVENTS_PER_CTA
    expected = timing_values + grid_ctas * OCCURRENCES_PER_CTA
    if len(storage) != expected:
        raise PhaseHarnessError(
            f"phase storage capacity drift: {len(storage)} != {expected}"
        )
    timing = exp017.validate_events(
        storage[:timing_values], mode=mode, grid_ctas=grid_ctas
    )
    tail = storage[timing_values:]
    if any(item < 0 for item in tail):
        raise PhaseHarnessError("negative occurrence count")
    if mode == CONTROL:
        nonzero = [index for index, item in enumerate(tail) if item]
        if nonzero:
            raise PhaseHarnessError(
                f"marker-disabled control wrote occurrences: first={nonzero[:16]}"
            )
        occurrences = {name: 0 for name in PHASE_NAMES}
        rows: list[dict[str, Any]] = []
    else:
        rows = [
            {
                "cta": cta,
                **{
                    name: tail[cta * OCCURRENCES_PER_CTA + index]
                    for index, name in enumerate(PHASE_NAMES)
                },
            }
            for cta in range(grid_ctas)
        ]
        occurrences = {
            name: sum(int(row[name]) for row in rows) for name in PHASE_NAMES
        }
        mandatory = PHASE_NAMES[:6]
        if any(occurrences[name] <= 0 for name in mandatory):
            raise PhaseHarnessError(
                f"missing mandatory occurrence markers: {occurrences}"
            )
    return {
        "mode": mode,
        "timing": timing,
        "occurrence_totals": occurrences,
        "occurrence_rows": rows,
        "storage_all_zero": not any(storage),
        "gate_pass": bool(timing["gate_pass"]),
    }


def expected_occurrences(
    arm: str,
    *,
    task_count: int,
    slice_count: int,
    grid_ctas: int = GRID_CTAS,
) -> dict[str, int]:
    """Expected close-marker counts from the actual task descriptor manifest."""
    if arm not in ARMS:
        raise PhaseHarnessError(f"unknown phase arm: {arm}")
    if task_count < 0 or slice_count < 0:
        raise PhaseHarnessError("negative task/slice manifest")
    factor = INTERVAL_FACTORS[arm]
    swiglu = slice_count * factor["swiglu_q1_per_slice"]
    fc2 = (
        slice_count
        * factor["output_tiles_per_slice"]
        * factor["fc2_scatter_per_output_tile"]
    )
    return {
        "clear_init": grid_ctas,
        "histogram": grid_ctas,
        "prefix": grid_ctas,
        "route_q0_pack": grid_ctas,
        "publish_route_tail": grid_ctas,
        # full_tile_publish_enabled is identity-locked to zero in both arms;
        # each CTA performs exactly one terminal no-task claim.
        "claim_cache_control": task_count + grid_ctas,
        "fc1_gate_up_swiglu": swiglu,
        "q1": swiglu,
        "fc2_epilogue_r2s": fc2,
        "scatter": fc2,
    }


def occurrence_gate(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    checks = {
        name: int(observed.get(name, -1)) == int(expected[name]) for name in PHASE_NAMES
    }
    return {
        "observed": {name: int(observed.get(name, -1)) for name in PHASE_NAMES},
        "expected": {name: int(expected[name]) for name in PHASE_NAMES},
        "checks": checks,
        "gate_pass": all(checks.values()),
    }


def aggregate_semantic_rows(
    phase_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_name = {str(row["name"]): row for row in phase_rows}
    if set(by_name) != {
        *PHASE_NAMES,
        "cta_residual",
        "launch_skew_early_finish",
    }:
        raise PhaseHarnessError(f"phase row vocabulary drift: {sorted(by_name)}")
    result = []
    for group, members in SEMANTIC_GROUPS.items():
        result.append(
            {
                "name": group,
                "members": list(members),
                "sum_cta_ns": sum(int(by_name[name]["sum_cta_ns"]) for name in members),
                "equivalent_wall_us": sum(
                    float(by_name[name]["equivalent_wall_us"]) for name in members
                ),
                "share_percent": sum(
                    float(by_name[name]["share_percent"]) for name in members
                ),
            }
        )
    if not math.isclose(
        sum(row["share_percent"] for row in result), 100.0, abs_tol=1e-9
    ):
        raise PhaseHarnessError("semantic rows do not close to 100%")
    return result


def summarize_replays(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reuse exp_017 replay aggregation and add semantic/occurrence summaries."""
    normalized = []
    for run in runs:
        timing = run.get("phase_timing")
        if not isinstance(timing, Mapping):
            raise PhaseHarnessError("replay is missing phase_timing")
        normalized.append({**run, "phase_timing": timing})
    result = exp017.summarize_replays(normalized)
    if str(result["mode"]) == CONTROL:
        result["semantic_rows"] = None
        result["occurrence_totals"] = None
        return result

    replay_groups = [
        {
            row["name"]: row
            for row in aggregate_semantic_rows(run["phase_timing"]["phase_rows"])
        }
        for run in normalized
    ]
    result["semantic_rows"] = []
    for name in SEMANTIC_GROUPS:
        values = [rows[name] for rows in replay_groups]
        result["semantic_rows"].append(
            {
                "name": name,
                "equivalent_wall_us_median": statistics.median(
                    float(row["equivalent_wall_us"]) for row in values
                ),
                "equivalent_wall_us_range": [
                    min(float(row["equivalent_wall_us"]) for row in values),
                    max(float(row["equivalent_wall_us"]) for row in values),
                ],
                "share_percent_median": statistics.median(
                    float(row["share_percent"]) for row in values
                ),
            }
        )
    occurrence_samples: dict[str, list[int]] = defaultdict(list)
    for run in normalized:
        for name in PHASE_NAMES:
            occurrence_samples[name].append(int(run["occurrence_totals"][name]))
    result["occurrence_totals"] = {
        name: {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stable": len(set(values)) == 1,
        }
        for name, values in occurrence_samples.items()
    }
    return result


def _capture_median(capture: Mapping[str, Any]) -> float:
    value = float(capture["summary"]["event_elapsed_us"]["median"])
    if not math.isfinite(value) or value <= 0:
        raise PhaseHarnessError(f"invalid capture median: {value}")
    return value


def perturbation_gate(
    captures: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    whole_op_gap_us: float,
) -> dict[str, Any]:
    """Apply the per-arm, cross-arm and phase-delta uncertainty gates."""
    missing = {(arm, mode) for arm in ARMS for mode in MODES} - set(captures)
    if missing:
        raise PhaseHarnessError(f"missing paired phase captures: {sorted(missing)}")
    biases = {}
    per_arm = {}
    for arm in ARMS:
        control = _capture_median(captures[(arm, CONTROL)])
        probe = _capture_median(captures[(arm, PROBE)])
        bias = (probe / control - 1.0) * 100.0
        biases[arm] = bias
        per_arm[arm] = abs(bias) <= 5.0
    differential_pp = biases["eric_stage4_fp4"] - biases["latest_opt_fp4"]

    opt_runs = captures[("latest_opt_fp4", PROBE)]["runs"]
    eric_runs = captures[("eric_stage4_fp4", PROBE)]["runs"]
    if len(opt_runs) != REPLAYS or len(eric_runs) != REPLAYS:
        raise PhaseHarnessError("paired probe captures must each contain five replays")
    envelopes = {}
    for group in SEMANTIC_GROUPS:
        if group in {"cta_residual", "launch_skew_early_finish"}:
            continue
        deltas = []
        for opt, eric in zip(opt_runs, eric_runs, strict=True):
            opt_rows = {
                row["name"]: row
                for row in aggregate_semantic_rows(opt["phase_timing"]["phase_rows"])
            }
            eric_rows = {
                row["name"]: row
                for row in aggregate_semantic_rows(eric["phase_timing"]["phase_rows"])
            }
            deltas.append(
                float(eric_rows[group]["equivalent_wall_us"])
                - float(opt_rows[group]["equivalent_wall_us"])
            )
        width = max(deltas) - min(deltas)
        envelopes[group] = {
            "delta_us": deltas,
            "envelope_width_us": width,
            "gate_pass": width < abs(float(whole_op_gap_us)),
        }
    checks = {
        "latest_opt_probe_control_abs_le_5pct": per_arm["latest_opt_fp4"],
        "eric_probe_control_abs_le_5pct": per_arm["eric_stage4_fp4"],
        "cross_arm_differential_abs_le_1pp": abs(differential_pp) <= 1.0,
        "phase_delta_envelopes_lt_whole_op_gap": all(
            row["gate_pass"] for row in envelopes.values()
        ),
    }
    return {
        "probe_control_bias_percent": biases,
        "cross_arm_differential_bias_pp": differential_pp,
        "whole_op_gap_us": abs(float(whole_op_gap_us)),
        "phase_delta_envelopes": envelopes,
        "checks": checks,
        "gate_pass": all(checks.values()),
    }
