from __future__ import annotations

import csv

from build_ncu_evidence import METRICS, build_evidence, parse_native_csv
from exp004_common import (
    BASELINE_LOCAL_SECTORS_PER_DIRECTION,
    EXPECTED_FP4_TENSOR_OPS,
    EXPECTED_TAIL_SECTOR_DELTA,
    EXPECTED_TENSOR_INSTRUCTIONS,
)


def metrics(local: int):
    return {
        "duration_ns": 1_000_000.0,
        "local_load_sectors": float(local),
        "local_store_sectors": float(local),
        "executed_local_load_instructions": 1_000_000.0,
        "executed_local_store_instructions": 500_000.0,
        "tensor_instructions": float(EXPECTED_TENSOR_INSTRUCTIONS),
        "fp4_tensor_ops": float(EXPECTED_FP4_TENSOR_OPS),
        "achieved_occupancy_pct": 10.42,
        "eligible_warps_per_cycle": 0.4,
        "issue_active_pct": 20.0,
        "tc_subpipe_active_pct": 25.0,
        "stall_long_scoreboard_pct": 30.0,
        "stall_short_scoreboard_pct": 3.0,
        "stall_wait_pct": 20.0,
        "stall_barrier_pct": 0.0,
        "registers_per_thread": 255.0,
        "shared_bytes_per_cta": 92160.0,
        "configured_stack_limit_bytes": 1024.0,
        "local_load_footprint_bytes": local * 32.0,
        "local_store_footprint_bytes": local * 32.0,
    }


def test_dynamic_14_word_closure() -> None:
    baseline = metrics(BASELINE_LOCAL_SECTORS_PER_DIRECTION)
    candidate = metrics(
        BASELINE_LOCAL_SECTORS_PER_DIRECTION - EXPECTED_TAIL_SECTOR_DELTA
    )
    result = build_evidence({"baseline": baseline, "activation_in_place_up": candidate})
    assert result["baseline_reproduction_pass"] is True
    assert (
        result["deltas"]["activation_in_place_up"]["dynamic_14_word_closure_pass"]
        is True
    )


def test_parse_native_ncu_csv(tmp_path) -> None:
    path = tmp_path / "raw.csv"
    header = ["ID", "Kernel Name", *METRICS.values()]
    units = ["", "", *["" for _ in METRICS]]
    values = ["0", "range", *["1" for _ in METRICS]]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(units)
        writer.writerow(values)
    parsed, _ = parse_native_csv(path)
    assert parsed["duration_ns"] == 1.0
    assert parsed["local_load_footprint_bytes"] == 32.0
