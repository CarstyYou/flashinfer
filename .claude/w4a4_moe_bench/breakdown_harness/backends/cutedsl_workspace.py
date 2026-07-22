"""CPU-safe oracle for the W4A4 dynamic scheduler workspace."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from breakdown_harness.artifacts import canonical_sha256


def expected_expert_tile_base(
    row_counts: Sequence[int], *, tile_m: int = 128
) -> list[int]:
    base = 0
    result = [0]
    for raw_count in row_counts:
        count = int(raw_count)
        if count < 0:
            raise ValueError("row count cannot be negative")
        base += (count + tile_m - 1) // tile_m
        result.append(base)
    return result


def expected_task_records(
    row_counts: Sequence[int],
    *,
    n: int = 512,
    tile_m: int = 128,
    tile_n: int = 128,
    slice_chunk: int = 1,
) -> list[tuple[int, int, int, int, int]]:
    """Return (expert, physical_m_tile, slice_begin, slice_count, valid_rows)."""
    if n <= 0 or n % tile_n:
        raise ValueError("n must be a positive multiple of tile_n")
    if slice_chunk <= 0:
        raise ValueError("slice_chunk must be positive")
    bases = expected_expert_tile_base(row_counts, tile_m=tile_m)
    gate_tiles = n // tile_n
    records: list[tuple[int, int, int, int, int]] = []
    for expert, raw_count in enumerate(row_counts):
        count = int(raw_count)
        local_tile = 0
        rows_remaining = count
        while rows_remaining > 0:
            valid_rows = min(tile_m, rows_remaining)
            for slice_begin in range(0, gate_tiles, slice_chunk):
                records.append(
                    (
                        expert,
                        bases[expert] + local_tile,
                        slice_begin,
                        min(slice_chunk, gate_tiles - slice_begin),
                        valid_rows,
                    )
                )
            rows_remaining -= tile_m
            local_tile += 1
    return records


def expected_terminal_pair_head(
    routed_rows: int, *, num_cta_warps: int, grid_z: int = 110
) -> int:
    """Expected terminal producer-queue head, including one miss per CTA."""
    if routed_rows < 0 or num_cta_warps <= 0 or grid_z <= 0:
        raise ValueError("invalid producer-head geometry")
    claim_count = num_cta_warps * 2
    productive_claims = (routed_rows + claim_count - 1) // claim_count
    return (productive_claims + grid_z) * claim_count


def verify_workspace_evidence(
    snapshot: Mapping[str, Any],
    *,
    expected_row_counts: Sequence[int],
    num_cta_warps: int = 5,
    grid_z: int = 110,
    expected_experts: int = 256,
    intermediate_size: int = 512,
) -> dict[str, Any]:
    """Validate the canonical W4A4 route/task arrays without importing Torch."""
    expected_rows = [int(value) for value in expected_row_counts]
    if len(expected_rows) != expected_experts:
        raise ValueError(
            f"expected {expected_experts} row counts, got {len(expected_rows)}"
        )
    row_counts = [int(value) for value in snapshot["row_counts"]]
    write_rows = [int(value) for value in snapshot["expert_write_rows"]]
    tile_base = [int(value) for value in snapshot["expert_tile_base"]]
    tail = int(snapshot["task_tail"])
    head = int(snapshot["task_head"])
    task_fields = (
        "task_expert",
        "task_m_tile",
        "task_slice_begin",
        "task_slice_count",
        "task_valid_rows",
    )
    lengths = {field: len(snapshot[field]) for field in task_fields}
    if any(length < tail for length in lengths.values()):
        raise ValueError(f"task arrays shorter than task_tail={tail}: {lengths}")
    observed = [
        tuple(int(snapshot[field][index]) for field in task_fields)
        for index in range(tail)
    ]
    expected = expected_task_records(expected_rows, n=intermediate_size)
    expected_base = expected_expert_tile_base(expected_rows)
    routed_rows = int(snapshot["routed_rows"])
    expected_pair_head = expected_terminal_pair_head(
        routed_rows, num_cta_warps=num_cta_warps, grid_z=grid_z
    )
    missing = list((Counter(expected) - Counter(observed)).elements())
    unexpected = list((Counter(observed) - Counter(expected)).elements())
    checks = {
        "routed_row_sum": sum(row_counts) == sum(expected_rows) == routed_rows,
        "pair_head_terminal_state": int(snapshot.get("pair_head", -1))
        == expected_pair_head,
        "row_counts": row_counts == expected_rows,
        "expert_write_rows": write_rows == expected_rows,
        "expert_tile_base": tile_base == expected_base,
        "task_tail": tail == len(expected),
        "task_descriptor_multiset": not missing and not unexpected,
        "all_work_published": int(snapshot.get("all_work_published", 0)) == 1,
        "atomic_head_terminal_state": head == tail + grid_z,
    }
    return {
        "checks": checks,
        "gate_pass": all(checks.values()),
        "expected_task_count": len(expected),
        "observed_task_tail": tail,
        "observed_task_head": head,
        "terminal_head_overshoot": head - tail,
        "producer_claim_count": num_cta_warps * 2,
        "expected_pair_head": expected_pair_head,
        "observed_pair_head": int(snapshot.get("pair_head", -1)),
        "missing_task_descriptors": [list(item) for item in missing[:32]],
        "unexpected_task_descriptors": [list(item) for item in unexpected[:32]],
        "task_descriptor_order_sha256": canonical_sha256(observed),
        "task_descriptor_multiset_sha256": canonical_sha256(sorted(observed)),
        "exact_once_evidence": (
            "append-only descriptor multiset + terminal atomic-head inference; "
            "no direct per-task consumed bitmap exists"
        ),
    }
