#!/usr/bin/env python3
"""Pure-Python contracts for the exp_017 latest-opt phase probe."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OVERLAY_ROOT = RESULTS / "opt_phase_overlays"

CONTROL = "control_no_marker"
PROBE = "probe"
MODES = (CONTROL, PROBE)

EXPECTED_OPT_SHA256 = "ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19"
EXPECTED_DISPATCH_SHA256 = (
    "cba2d0966631a47a576747e8322b57116122f2c8e5e868f8efb3f5ea692391a4"
)
EXPECTED_WRAPPER_SHA256 = (
    "bcac806795c035decd0773f4f801d477e7ebf14c1d67c3e49eee42ee0579c0a4"
)

OPT_RELATIVE_PATH = Path(".claude/w4a4_moe_bench/moe_dynamic_kernel_opt.py")
DISPATCH_RELATIVE_PATH = Path(
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py"
)
WRAPPER_RELATIVE_PATH = Path("flashinfer/fused_moe/cute_dsl/b12x_moe.py")
KERNEL_MODULE = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dynamic_kernel"
DISPATCH_MODULE = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch"

GRID_CTAS = 110
ENTRY_SLOT = 0
CURSOR_SLOT = 1
READER_FINAL_SLOT = 2
TMA_FINAL_SLOT = 3
PHASE_NAMES = (
    "clear_init",
    "histogram",
    "prefix",
    "route_q0_pack",
    "publish_route_tail",
    "claim_cache_control",
    "fc1_gate_up_swiglu",
    "q1",
    "fc2_epilogue_r2s",
    "scatter",
)
PHASE_SLOT_BASE = 4
EVENTS_PER_CTA = PHASE_SLOT_BASE + len(PHASE_NAMES)

EVENT_ABI = {
    "schema": "exp017.opt-full-phase-event-abi.v1",
    "timestamp": "%globaltimer (ns)",
    "grid_ctas": GRID_CTAS,
    "events_per_cta": EVENTS_PER_CTA,
    "layout": {
        "entry": ENTRY_SLOT,
        "cursor": CURSOR_SLOT,
        "reader_final": READER_FINAL_SLOT,
        "tma_final": TMA_FINAL_SLOT,
        "phase_duration_base": PHASE_SLOT_BASE,
        "phase_names": list(PHASE_NAMES),
    },
    "reader_track": (
        "CTA leader intervals only; named intervals are mutually exclusive. "
        "W8 TMA work remains overlapped and is not added as a reader phase."
    ),
    "cta_final": "max(reader_final, tma_final)",
    "denominator": "grid_ctas * (max CTA final - min CTA entry)",
    "phase_equivalent_wall": "sum CTA phase duration / grid_ctas",
    "cta_residual": "sum((CTA final-entry)-sum named durations)",
    "launch_skew": (
        "denominator - sum(CTA final-entry); includes early-finish/launch skew"
    ),
    "closure": "named phases + CTA residual + launch skew == denominator",
    "classification": "diagnostic matched probe; uninstrumented E2E is authority",
}

BARRIER_PATTERNS = (
    "cute.arch.sync_threads()",
    "cute.arch.sync_warp()",
    "self.epilog_sync_barrier.arrive_and_wait()",
    "self.pass_gate_barrier.arrive_unaligned()",
    "self.pass_gate_barrier.wait_unaligned()",
    "self.pass_final_barrier.arrive_unaligned()",
    "self.pass_final_barrier.wait_unaligned()",
    "self.resident_grid_barrier(",
)


class PhaseProbeError(RuntimeError):
    """The source, event storage, or capture violates the probe contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PhaseProbeError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def barrier_fingerprint(source: str) -> dict[str, int]:
    return {pattern: source.count(pattern) for pattern in BARRIER_PATTERNS}


def normalize_dispatch_flag(source: str) -> str:
    return source.replace(
        "_EXP017_OPT_PHASE_PROBE_ENABLED = True",
        "_EXP017_OPT_PHASE_PROBE_ENABLED = <FLAG>",
    ).replace(
        "_EXP017_OPT_PHASE_PROBE_ENABLED = False",
        "_EXP017_OPT_PHASE_PROBE_ENABLED = <FLAG>",
    )


def _flat_ints(values: Any) -> list[int]:
    plain = values.tolist() if hasattr(values, "tolist") else values
    if not isinstance(plain, (list, tuple)):
        raise PhaseProbeError("phase events must be a flat vector")
    if plain and isinstance(plain[0], (list, tuple)):
        plain = [item for row in plain for item in row]
    return [int(value) for value in plain]


def validate_events(
    values: Any, *, mode: str, grid_ctas: int = GRID_CTAS
) -> dict[str, Any]:
    """Validate one replay and reduce it with the exp_017 closure contract."""
    if mode not in MODES:
        raise PhaseProbeError(f"unknown marker mode: {mode}")
    events = _flat_ints(values)
    expected = grid_ctas * EVENTS_PER_CTA
    if len(events) != expected:
        raise PhaseProbeError(f"event capacity drift: {len(events)} != {expected}")
    if mode == CONTROL:
        nonzero = [index for index, value in enumerate(events) if value != 0]
        if nonzero:
            raise PhaseProbeError(
                f"marker-disabled control wrote events: first={nonzero[:16]}"
            )
        return {
            "mode": mode,
            "all_zero": True,
            "observed_ctas": 0,
            "phase_equivalent_wall_us": None,
            "closure_error_ns": 0,
            "gate_pass": True,
        }

    rows = [
        events[index : index + EVENTS_PER_CTA]
        for index in range(0, len(events), EVENTS_PER_CTA)
    ]
    bad_identity = [
        index
        for index, row in enumerate(rows)
        if min(row[ENTRY_SLOT], row[READER_FINAL_SLOT], row[TMA_FINAL_SLOT]) <= 0
        or row[CURSOR_SLOT] <= 0
    ]
    if bad_identity:
        raise PhaseProbeError(f"missing CTA entry/final events: {bad_identity[:16]}")

    entries: list[int] = []
    finals: list[int] = []
    spans: list[int] = []
    residuals: list[int] = []
    phase_sums = {name: 0 for name in PHASE_NAMES}
    for cta, row in enumerate(rows):
        entry = row[ENTRY_SLOT]
        final = max(row[READER_FINAL_SLOT], row[TMA_FINAL_SLOT])
        if final < entry:
            raise PhaseProbeError(f"CTA {cta} final precedes entry")
        durations = row[PHASE_SLOT_BASE:]
        if any(value < 0 for value in durations):
            raise PhaseProbeError(f"CTA {cta} has negative phase duration")
        named = sum(durations)
        span = final - entry
        if named > span:
            raise PhaseProbeError(
                f"CTA {cta} named phase sum exceeds span: {named} > {span}"
            )
        entries.append(entry)
        finals.append(final)
        spans.append(span)
        residuals.append(span - named)
        for name, value in zip(PHASE_NAMES, durations, strict=True):
            phase_sums[name] += value

    grid_wall = max(finals) - min(entries)
    denominator = grid_ctas * grid_wall
    if denominator <= 0:
        raise PhaseProbeError("non-positive grid denominator")
    cta_residual = sum(residuals)
    launch_skew = denominator - sum(spans)
    if launch_skew < 0:
        raise PhaseProbeError("CTA spans exceed grid denominator")
    closure = sum(phase_sums.values()) + cta_residual + launch_skew
    closure_error = closure - denominator
    if closure_error != 0:
        raise PhaseProbeError(f"reader closure error: {closure_error} ns")

    rows_out = []
    for name in PHASE_NAMES:
        total = phase_sums[name]
        rows_out.append(
            {
                "name": name,
                "sum_cta_ns": total,
                "equivalent_wall_us": total / grid_ctas / 1000.0,
                "share_percent": 100.0 * total / denominator,
            }
        )
    for name, total in (
        ("cta_residual", cta_residual),
        ("launch_skew_early_finish", launch_skew),
    ):
        rows_out.append(
            {
                "name": name,
                "sum_cta_ns": total,
                "equivalent_wall_us": total / grid_ctas / 1000.0,
                "share_percent": 100.0 * total / denominator,
            }
        )
    return {
        "mode": mode,
        "all_zero": False,
        "observed_ctas": grid_ctas,
        "grid_start_ns": min(entries),
        "grid_final_ns": max(finals),
        "grid_critical_wall_us": grid_wall / 1000.0,
        "denominator_cta_ns": denominator,
        "phase_rows": rows_out,
        "share_sum_percent": sum(row["share_percent"] for row in rows_out),
        "cta_span_us": {
            "median": statistics.median(spans) / 1000.0,
            "min": min(spans) / 1000.0,
            "max": max(spans) / 1000.0,
        },
        "closure_error_ns": closure_error,
        "gate_pass": True,
    }


def summarize_replays(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise PhaseProbeError("no replay records")
    modes = {str(run["mode"]) for run in runs}
    if len(modes) != 1:
        raise PhaseProbeError(f"mixed capture modes: {modes}")
    mode = modes.pop()
    elapsed = [float(run["event_elapsed_us"]) for run in runs]
    result: dict[str, Any] = {
        "mode": mode,
        "event_elapsed_us": {
            "median": statistics.median(elapsed),
            "min": min(elapsed),
            "max": max(elapsed),
            "samples": len(elapsed),
        },
    }
    if mode == CONTROL:
        result["phase_rows"] = None
        return result
    by_name: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        for row in run["phase_timing"]["phase_rows"]:
            by_name.setdefault(str(row["name"]), []).append(row)
    result["phase_rows"] = [
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
            "share_percent_range": [
                min(float(row["share_percent"]) for row in values),
                max(float(row["share_percent"]) for row in values),
            ],
        }
        for name, values in by_name.items()
    ]
    return result
