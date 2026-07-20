#!/usr/bin/env python3
"""CPU-only exact-once ownership gate for the exp_014 Scatter mapping."""

from __future__ import annotations

from collections import Counter, defaultdict
import json


TILE_M = 128
TILE_N = 128
WARP_ROWS = 32
WARP_COLS = 64
VECTOR_WIDTH = 8
LANES = 32
MATH_WARPS = 8
VALID_ROWS_CASES = (1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128)
OUTPUT_TILE_CASES = (0, 1, 7, 63)


def enumerate_writes(
    valid_rows: int, output_tile_idx: int
) -> list[tuple[tuple[int, int], tuple[int, int, int]]]:
    """Return ``((row, global_col), (warp, lane, vector_iter))`` writes."""
    if not 0 <= valid_rows <= TILE_M:
        raise ValueError(valid_rows)
    writes = []
    tile_n_base = output_tile_idx * TILE_N
    vectors_per_row = WARP_COLS // VECTOR_WIDTH
    for warp in range(MATH_WARPS):
        warp_m_base = (warp >> 1) * WARP_ROWS
        warp_n_base = (warp & 1) * WARP_COLS
        warp_rows = max(0, min(WARP_ROWS, valid_rows - warp_m_base))
        for lane in range(LANES):
            vector_index = lane
            vector_iteration = 0
            while vector_index < warp_rows * vectors_per_row:
                local_row, local_vector_col = divmod(vector_index, vectors_per_row)
                row = warp_m_base + local_row
                first_col = tile_n_base + warp_n_base + local_vector_col * VECTOR_WIDTH
                owner = (warp, lane, vector_iteration)
                for element in range(VECTOR_WIDTH):
                    writes.append(((row, first_col + element), owner))
                vector_index += LANES
                vector_iteration += 1
    return writes


def check_case(valid_rows: int, output_tile_idx: int) -> dict[str, object]:
    writes = enumerate_writes(valid_rows, output_tile_idx)
    counts = Counter(coordinate for coordinate, _ in writes)
    tile_n_base = output_tile_idx * TILE_N
    expected = {
        (row, col)
        for row in range(valid_rows)
        for col in range(tile_n_base, tile_n_base + TILE_N)
    }
    observed = set(counts)
    missing = expected - observed
    invalid = observed - expected
    duplicated = {
        coordinate: count for coordinate, count in counts.items() if count != 1
    }
    if missing or invalid or duplicated:
        raise RuntimeError(
            f"ownership failure for valid_rows={valid_rows}, "
            f"output_tile_idx={output_tile_idx}: missing={len(missing)}, "
            f"invalid={len(invalid)}, duplicated={len(duplicated)}"
        )

    warp_elements: dict[int, int] = defaultdict(int)
    for _, (warp, _, _) in writes:
        warp_elements[warp] += 1
    if valid_rows == TILE_M and set(warp_elements) != set(range(MATH_WARPS)):
        raise RuntimeError(f"full tile did not engage all math warps: {warp_elements}")

    return {
        "valid_rows": valid_rows,
        "output_tile_idx": output_tile_idx,
        "expected_elements": len(expected),
        "observed_elements": len(writes),
        "active_warps": sorted(warp_elements),
        "elements_per_warp": {
            str(warp): warp_elements[warp] for warp in sorted(warp_elements)
        },
        "exactly_one_owner": True,
        "invalid_writes": 0,
    }


def main() -> None:
    cases = [
        check_case(valid_rows, output_tile_idx)
        for valid_rows in VALID_ROWS_CASES
        for output_tile_idx in OUTPUT_TILE_CASES
    ]
    payload = {
        "schema": "exp014.scatter-ownership-gate.v1",
        "status": "pass",
        "mapping": "warp_m=(warp>>1)*32, warp_n=(warp&1)*64",
        "vector_width": VECTOR_WIDTH,
        "cases": cases,
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
