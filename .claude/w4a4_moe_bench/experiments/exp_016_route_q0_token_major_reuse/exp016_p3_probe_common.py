#!/usr/bin/env python3
"""Pure-Python contracts for the exp_016 P3 `%globaltimer` probe."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BASE_OVERLAY_ROOT = RESULTS / "overlays"
PROBE_OVERLAY_ROOT = RESULTS / "p3_phase_probe_overlays"

BASELINE = "baseline_pair_major"
CANDIDATE = "candidate_token_major_reuse"
ARMS = (BASELINE, CANDIDATE)
CONTROL = "control_no_marker"
PROBE = "probe"
MODES = (CONTROL, PROBE)

EXPECTED_BASE_KERNEL_SHA256 = {
    BASELINE: "c88cef63492b60c0a77484b50f6400b83a103d168e1535b78972341503810184",
    CANDIDATE: "ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19",
}

KERNEL_MODULE = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dynamic_kernel"
DISPATCH_MODULE = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch"
DISPATCH_RELATIVE_PATH = Path(
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py"
)
WRAPPER_RELATIVE_PATH = Path("flashinfer/fused_moe/cute_dsl/b12x_moe.py")
EXPECTED_DISPATCH_SHA256 = (
    "cba2d0966631a47a576747e8322b57116122f2c8e5e868f8efb3f5ea692391a4"
)
EXPECTED_WRAPPER_SHA256 = (
    "bcac806795c035decd0773f4f801d477e7ebf14c1d67c3e49eee42ee0579c0a4"
)

GRID_CTAS = 110
TICKS_PER_CTA = 2
SENTINEL = -1
EDGE_NAMES = ("start", "end")

EVENT_ABI = {
    "schema": "exp016.p3-phase-probe-abi.v1",
    "timestamp": "%globaltimer (ns)",
    "grid_ctas": GRID_CTAS,
    "ticks_per_cta": TICKS_PER_CTA,
    "index": "cta_z * 2 + edge(start=0,end=1)",
    "start": (
        "CTA leader after the prefix resident-grid barrier returns and before "
        "that CTA's first producer claim"
    ),
    "end": (
        "CTA leader after all Route/Q0 quant/store threads reconverge at the "
        "final CTA sync and before threadfence/deferred publication"
    ),
    "primary_reduction": "max(all CTA end) - min(all CTA start)",
    "primary_name": "grid_critical_wall_ns",
    "forbidden_reduction": "sum/additive SM-equivalent estimate",
    "classification": "diagnostic matched probe; uninstrumented E2E remains authority",
}

BARRIER_PATTERNS = (
    "cute.arch.sync_threads()",
    "self.epilog_sync_barrier.arrive_and_wait()",
    "self.pass_gate_barrier.arrive_unaligned()",
    "self.pass_gate_barrier.wait_unaligned()",
    "self.pass_final_barrier.arrive_unaligned()",
    "self.pass_final_barrier.wait_unaligned()",
    "self.resident_grid_barrier(",
)


class ProbeContractError(RuntimeError):
    """The P3 probe source, storage, or captured values violate the ABI."""


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
        raise ProbeContractError(f"expected a JSON object: {path}")
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
        "_EXP016_P3_PHASE_PROBE_ENABLED = True",
        "_EXP016_P3_PHASE_PROBE_ENABLED = <FLAG>",
    ).replace(
        "_EXP016_P3_PHASE_PROBE_ENABLED = False",
        "_EXP016_P3_PHASE_PROBE_ENABLED = <FLAG>",
    )


def _flat_ints(values: Any) -> list[int]:
    plain = values.tolist() if hasattr(values, "tolist") else values
    if not isinstance(plain, (list, tuple)):
        raise ProbeContractError("P3 ticks must be a flat vector")
    if plain and isinstance(plain[0], (list, tuple)):
        plain = [item for row in plain for item in row]
    return [int(value) for value in plain]


def validate_ticks(
    values: Any,
    *,
    mode: str,
    grid_ctas: int = GRID_CTAS,
) -> dict[str, Any]:
    """Validate one capture and reduce only to the grid critical wall."""
    if mode not in MODES:
        raise ProbeContractError(f"unknown marker mode: {mode}")
    ticks = _flat_ints(values)
    expected = grid_ctas * TICKS_PER_CTA
    if len(ticks) != expected:
        raise ProbeContractError(f"P3 tick capacity drift: {len(ticks)} != {expected}")

    if mode == CONTROL:
        non_sentinel = [index for index, value in enumerate(ticks) if value != SENTINEL]
        if non_sentinel:
            raise ProbeContractError(
                "marker-disabled control wrote P3 events: "
                f"first_indices={non_sentinel[:16]}"
            )
        return {
            "mode": mode,
            "expected_events": 0,
            "observed_events": 0,
            "all_sentinel": True,
            "grid_critical_wall_ns": None,
            "grid_critical_wall_us": None,
            "gate_pass": True,
        }

    missing = [index for index, value in enumerate(ticks) if value == SENTINEL]
    if missing:
        raise ProbeContractError(
            f"probe did not exactly fill all CTA edges: first_missing={missing[:16]}"
        )
    starts = ticks[0::2]
    ends = ticks[1::2]
    invalid = [
        index
        for index, pair in enumerate(zip(starts, ends, strict=True))
        if pair[0] > pair[1]
    ]
    if invalid:
        raise ProbeContractError(f"P3 start follows end for CTA indices {invalid[:16]}")
    if min(starts) <= 0:
        raise ProbeContractError("globaltimer timestamps must be positive")

    durations = [end - start for start, end in zip(starts, ends, strict=True)]
    grid_start = min(starts)
    grid_end = max(ends)
    critical_wall = grid_end - grid_start
    return {
        "mode": mode,
        "expected_events": expected,
        "observed_events": expected,
        "all_sentinel": False,
        "grid_start_ns": grid_start,
        "grid_end_ns": grid_end,
        "grid_critical_wall_ns": critical_wall,
        "grid_critical_wall_us": critical_wall / 1000.0,
        "cta_duration_ns": {
            "median": statistics.median(durations),
            "min": min(durations),
            "max": max(durations),
        },
        "reduction_contract": EVENT_ABI["primary_reduction"],
        "additive_estimate_reported": False,
        "gate_pass": True,
    }


def capture_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    walls = [
        float(run["p3_timing"]["grid_critical_wall_us"])
        for run in runs
        if run["p3_timing"].get("grid_critical_wall_us") is not None
    ]
    if not walls:
        return {
            "grid_critical_wall_us": None,
            "classification": "marker-disabled control",
        }
    return {
        "grid_critical_wall_us": {
            "median": statistics.median(walls),
            "min": min(walls),
            "max": max(walls),
            "samples": len(walls),
        },
        "classification": "diagnostic grid critical wall",
        "additive_estimate_reported": False,
    }
