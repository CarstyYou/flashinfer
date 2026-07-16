#!/usr/bin/env python3
"""Build falsifiable source models for fused local traffic and reduction work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"
M_VALUES = (256, 8192)
TOPK = 8
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512
TILE_M = 128
TILE_N = 128
MMA_WARPS = 4
THREADS_PER_WARP = 32
SECTOR_BYTES = 32
BF16_BYTES = 2
FP4_OPS_PER_PHYSICAL_ROW = 3 * 2 * HIDDEN_SIZE * INTERMEDIATE_SIZE


def indexed_targets(evidence: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    return {(int(row["m"]), str(row["phase"])): row for row in evidence["targets"]}


def build_models(
    evidence: dict[str, Any], static_local_sass: dict[str, Any]
) -> dict[str, Any]:
    targets = indexed_targets(evidence)
    slices = INTERMEDIATE_SIZE // TILE_N
    static_stack = static_local_sass["static_instruction_facts"][
        "stack_roundtrip_model"
    ]
    stored_words_per_lane = int(static_stack["total_stored_32bit_words_per_lane"])
    stl64_words_per_lane = int(static_stack["stl64_covered_32bit_words"])
    if not static_stack["all_stored_slots_have_a_later_static_ldl"]:
        raise ValueError("static local-store slots lack complete later reload coverage")
    cases: list[dict[str, Any]] = []
    for m in M_VALUES:
        fused = targets[(m, "fused_main")]["metrics"]
        fc1 = targets[(m, "fc1")]["metrics"]
        fc2 = targets[(m, "fc2")]["metrics"]

        physical_rows = fused["fp4_to_fp32_tensor_ops"] / FP4_OPS_PER_PHYSICAL_ROW
        physical_tiles = physical_rows / TILE_M
        task_count = physical_tiles * slices
        for name, value in (
            ("physical_rows", physical_rows),
            ("physical_tiles", physical_tiles),
            ("task_count", task_count),
        ):
            if abs(value - round(value)) > 1e-6:
                raise ValueError(f"m={m} non-integral {name}: {value}")
        physical_rows = int(round(physical_rows))
        physical_tiles = int(round(physical_tiles))
        task_count = int(round(task_count))

        measured_store_sectors = int(round(fused["local_store_sectors"]))
        measured_load_sectors = int(round(fused["local_load_sectors"]))
        predicted_sectors_per_direction = (
            task_count
            * MMA_WARPS
            * THREADS_PER_WARP
            * stored_words_per_lane
            * 4
            // SECTOR_BYTES
        )
        predicted_stl64_sectors_per_direction = (
            task_count
            * MMA_WARPS
            * THREADS_PER_WARP
            * stl64_words_per_lane
            * 4
            // SECTOR_BYTES
        )
        chain_local_bytes = (
            fc1["local_total_footprint_bytes"] + fc2["local_total_footprint_bytes"]
        )
        fused_local_bytes = fused["local_total_footprint_bytes"]
        local_delta_bytes = fused_local_bytes - chain_local_bytes

        expected_reduction_sectors = (
            m * TOPK * slices * HIDDEN_SIZE * BF16_BYTES // SECTOR_BYTES
        )
        measured_reduction_sectors = int(round(fused["global_reduction_sectors"]))

        cases.append(
            {
                "m": m,
                "physical_routed_rows_from_tensor_ops": physical_rows,
                "physical_expert_m_tiles": physical_tiles,
                "intermediate_slices": slices,
                "compute_task_count": task_count,
                "local_traffic_context": {
                    "classification": "launch-level dynamic NCU facts",
                    "measured_fused_local_store_sectors": measured_store_sectors,
                    "measured_fused_local_load_sectors": measured_load_sectors,
                    "measured_fused_local_footprint_bytes": fused_local_bytes,
                    "measured_chain_fc1_fc2_local_footprint_bytes": chain_local_bytes,
                    "fused_minus_chain_local_footprint_bytes": local_delta_bytes,
                    "static_stack_words_per_lane": stored_words_per_lane,
                    "static_stl64_words_per_lane": stl64_words_per_lane,
                    "predicted_local_sectors_per_direction": (
                        predicted_sectors_per_direction
                    ),
                    "predicted_stl64_sectors_per_direction": (
                        predicted_stl64_sectors_per_direction
                    ),
                    "store_sector_residual": (
                        measured_store_sectors - predicted_sectors_per_direction
                    ),
                    "load_sector_residual": (
                        measured_load_sectors - predicted_sectors_per_direction
                    ),
                    "execution_model_classification": (
                        "exact static-stack-to-dynamic-sector match"
                        if measured_store_sectors == predicted_sectors_per_direction
                        and measured_load_sectors == predicted_sectors_per_direction
                        else "mismatch"
                    ),
                    "source_phase_attribution": (
                        "program-order inference; no source lineinfo"
                    ),
                },
                "partial_output_reduction_model": {
                    "classification": "exact count model for observed source schedule",
                    "logical_routed_rows": m * TOPK,
                    "reductions_per_output_element": slices,
                    "expected_reduction_sectors": expected_reduction_sectors,
                    "measured_fused_reduction_sectors": measured_reduction_sectors,
                    "residual_sectors": (
                        measured_reduction_sectors - expected_reduction_sectors
                    ),
                    "measured_reduction_footprint_bytes": (
                        measured_reduction_sectors * SECTOR_BYTES
                    ),
                },
            }
        )

    return {
        "schema": "exp002.fused-schedule-models.v3",
        "constants": {
            "tile_m": TILE_M,
            "tile_n": TILE_N,
            "intermediate_size": INTERMEDIATE_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "topk": TOPK,
            "mma_warps": MMA_WARPS,
            "threads_per_warp": THREADS_PER_WARP,
            "sector_bytes": SECTOR_BYTES,
            "source_task_slice_chunk": 1,
        },
        "source_evidence": {
            "gate_up_accumulators": (
                "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/"
                "moe_dynamic_kernel.py:1651"
            ),
            "slice_outer_loop": (
                "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/"
                "moe_dynamic_kernel.py:1896"
            ),
            "fc2_output_tile_loop": (
                "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/"
                "moe_dynamic_kernel.py:2352"
            ),
            "per_tile_reduction": (
                "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/"
                "moe_dynamic_kernel.py:2526"
            ),
            "reduction_opcode_source": "flashinfer/cute_dsl/fp4_common.py:1798",
            "static_local_sass": "ncu/static_local_sass.json",
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    results = args.results.resolve()
    evidence = json.loads((results / "ncu" / "deep_launch_metrics.json").read_text())
    static_local_sass = json.loads(
        (results / "ncu" / "static_local_sass.json").read_text()
    )
    models = build_models(evidence, static_local_sass)
    output = results / "ncu" / "fused_schedule_models.json"
    output.write_text(json.dumps(models, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
