#!/usr/bin/env python3
"""Build the CPU-safe descriptor/work ledger for exp_007.

This does not infer runtime behavior from comments.  It checks the registered
N-coordinate map and hashes the scheduler and FC2/scatter source regions that
must remain byte-identical between the immutable overlays.  Runtime Tensor-work
identity is a separate matched-NCU gate.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


M = 128
H = 2048
I = 512
K_TILE = 128
LOGICAL_N = 128
NATIVE_N = 64
SF_VEC = 16


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_region(source: str, start: str, end: str) -> str:
    start_at = source.index(start)
    end_at = source.index(end, start_at)
    return source[start_at:end_at]


def region_identity(anchor: str, candidate: str, start: str, end: str) -> dict[str, Any]:
    anchor_region = source_region(anchor, start, end)
    candidate_region = source_region(candidate, start, end)
    anchor_executable = "\n".join(
        line for line in anchor_region.splitlines() if line.strip() and not line.lstrip().startswith("#")
    )
    candidate_executable = "\n".join(
        line
        for line in candidate_region.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    return {
        "start_marker": start,
        "end_marker": end,
        "anchor_sha256": sha256_bytes(anchor_region.encode()),
        "candidate_sha256": sha256_bytes(candidate_region.encode()),
        "byte_identical": anchor_region == candidate_region,
        "anchor_executable_sha256": sha256_bytes(anchor_executable.encode()),
        "candidate_executable_sha256": sha256_bytes(candidate_executable.encode()),
        "executable_identical": anchor_executable == candidate_executable,
    }


def traffic_bytes(n: int) -> dict[str, int]:
    k_trips = H // K_TILE
    return {
        "k_trips": k_trips,
        "a_per_k": M * K_TILE // 2,
        "b_per_k": n * K_TILE // 2,
        "sfa_per_k": M * (K_TILE // SF_VEC),
        # SM120's current helper physically rounds FC1 SFB N64 to N128.
        "sfb_physical_per_k": max(LOGICAL_N, n) * (K_TILE // SF_VEC),
    }


def build_descriptor_rows() -> list[dict[str, Any]]:
    rows = []
    per_pass = traffic_bytes(NATIVE_N)
    for branch, branch_base in (("up", 0), ("gate", I)):
        for logical_slice in range(I // LOGICAL_N):
            for half in range(LOGICAL_N // NATIVE_N):
                n_begin = branch_base + logical_slice * LOGICAL_N + half * NATIVE_N
                rows.append(
                    {
                        "branch": branch,
                        "logical_slice": logical_slice,
                        "half": half,
                        "global_n_range": [n_begin, n_begin + NATIVE_N],
                        "b_n64_tile_index": n_begin // NATIVE_N,
                        "sfb_n128_tile_index": (
                            branch_base + logical_slice * LOGICAL_N
                        )
                        // LOGICAL_N,
                        "k_trips": per_pass["k_trips"],
                        "bytes": {
                            name.removesuffix("_per_k"): value * per_pass["k_trips"]
                            for name, value in per_pass.items()
                            if name != "k_trips"
                        },
                    }
                )
    return rows


def sum_candidate(rows: list[dict[str, Any]], operand: str) -> int:
    return sum(int(row["bytes"][operand]) for row in rows)


def anchor_totals() -> dict[str, int]:
    per_pass = traffic_bytes(LOGICAL_N)
    passes = 2 * (I // LOGICAL_N)  # Gate + Up for every logical N128 slice.
    return {
        name.removesuffix("_per_k"): value * per_pass["k_trips"] * passes
        for name, value in per_pass.items()
        if name != "k_trips"
    }


def source_temporal_contract(candidate: str) -> dict[str, Any]:
    """Check the registered source ordering without claiming compiler liveness."""
    compact = "".join(candidate.split())
    consumer_start = candidate.index(
        "                    cons_state.reset_count()\n"
        "                    for fc1_half in cutlass.range_constexpr(2):"
    )
    consumer_end = candidate.index(
        '                    cute.arch.fence_proxy("async.shared", space="cta")',
        consumer_start,
    )
    consumer = candidate[consumer_start:consumer_end]
    ordered_markers = (
        "gate_acc.fill(0.0)",
        "up_acc.fill(0.0)",
        "gated_activation_f32(",
        "cute.copy(\n                                    fc1_tiled_copy_r2s,",
    )
    marker_positions = [consumer.index(marker) for marker in ordered_markers]
    checks = {
        "one_gate_n64_accumulator_object": candidate.count(
            "gate_acc = cute.make_rmem_tensor(fc1_acc_shape, self.acc_dtype)"
        )
        == 1,
        "one_up_n64_accumulator_object": candidate.count(
            "up_acc = cute.make_rmem_tensor(fc1_acc_shape, self.acc_dtype)"
        )
        == 1,
        "fc1_tile_is_half_of_logical_n": "mma_tiler_mn[1]//2" in compact,
        "consumer_source_order_gate_up_activation_store": marker_positions
        == sorted(marker_positions),
        "activation_store_is_inside_two_half_loop": (
            "for fc1_half in cutlass.range_constexpr(2):" in consumer
            and "gated_activation_f32(" in consumer
        ),
        "q1_occurs_once_after_both_halves": (
            candidate.count("sA_u8 = cute.recast_tensor(sA[None, None, 0], cutlass.Uint8)")
            == 1
            and candidate.index(
                "sA_u8 = cute.recast_tensor(sA[None, None, 0], cutlass.Uint8)"
            )
            > consumer_end
        ),
        "producer_up_native_n64_coordinate": (
            "intermediate_slice*Int32(2)+Int32(fc1_half)" in compact
        ),
        "producer_gate_native_n64_coordinate": (
            "(intermediate_slice+gate_tile_cnt)*Int32(2)+Int32(fc1_half)"
            in compact
        ),
        "physical_sfb_n128_rounding_explicit": (
            "max(128,self.fc1_tile_shape_mnk[1])" in compact
        ),
        "single_fc1_to_fc2_handoff": (
            candidate.count("self.pass_gate_barrier.arrive_unaligned()") == 1
            and candidate.count("self.pass_gate_barrier.wait_unaligned()") == 1
        ),
    }
    return {
        "consumer_region_sha256": sha256_bytes(consumer.encode()),
        "ordered_markers": list(ordered_markers),
        "checks": checks,
        "gate_pass": all(checks.values()),
        "evidence_boundary": (
            "This proves source structure/order only. Compiler resource/SASS and "
            "NCU evidence separately prove the compiled spill outcome; this check "
            "does not reconstruct exact register live intervals."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    anchor_bytes = args.anchor.read_bytes()
    candidate_bytes = args.candidate.read_bytes()
    anchor = anchor_bytes.decode()
    candidate = candidate_bytes.decode()
    ast.parse(anchor)
    ast.parse(candidate)

    rows = build_descriptor_rows()
    candidate_totals = {
        operand: sum_candidate(rows, operand)
        for operand in ("a", "b", "sfa", "sfb_physical")
    }
    expected_anchor = anchor_totals()

    expected_ranges = {
        "up": [(n, n + NATIVE_N) for n in range(0, I, NATIVE_N)],
        "gate": [(n, n + NATIVE_N) for n in range(I, 2 * I, NATIVE_N)],
    }
    observed_ranges = {
        branch: sorted(
            tuple(row["global_n_range"])
            for row in rows
            if row["branch"] == branch
        )
        for branch in expected_ranges
    }

    checks = {
        "up_n64_ranges_exact_once": observed_ranges["up"] == expected_ranges["up"],
        "gate_n64_ranges_exact_once": observed_ranges["gate"]
        == expected_ranges["gate"],
        "candidate_b_bytes_equal_anchor": candidate_totals["b"]
        == expected_anchor["b"],
        "candidate_a_replay_is_2x_anchor": candidate_totals["a"]
        == 2 * expected_anchor["a"],
        "candidate_sfa_replay_is_2x_anchor": candidate_totals["sfa"]
        == 2 * expected_anchor["sfa"],
        "candidate_physical_sfb_is_2x_anchor": candidate_totals["sfb_physical"]
        == 2 * expected_anchor["sfb_physical"],
    }

    source_identity = {
        "scheduler": region_identity(
            anchor,
            candidate,
            "        # Phase 0: cooperative init",
            "        gA = cute.local_tile(",
        ),
        "fc2_scatter": region_identity(
            anchor,
            candidate,
            "                    # PHASE B: Sweep ALL FC2 output tiles",
            "                    # Signal that FC2/scatter no longer needs sA",
        ),
    }
    checks["scheduler_executable_source_identical"] = source_identity["scheduler"][
        "executable_identical"
    ]
    checks["fc2_scatter_executable_source_identical"] = source_identity[
        "fc2_scatter"
    ][
        "executable_identical"
    ]
    temporal_contract = source_temporal_contract(candidate)
    checks["candidate_source_temporal_contract"] = temporal_contract["gate_pass"]

    payload = {
        "schema": "exp007.work-ledger.v1",
        "identity": {
            "anchor": str(args.anchor),
            "anchor_sha256": sha256_bytes(anchor_bytes),
            "candidate": str(args.candidate),
            "candidate_sha256": sha256_bytes(candidate_bytes),
        },
        "shape": {
            "M": M,
            "H": H,
            "I_tp": I,
            "logical_n": LOGICAL_N,
            "native_fc1_n": NATIVE_N,
            "k_tile": K_TILE,
        },
        "descriptor_rows": rows,
        "traffic_totals": {
            "anchor": expected_anchor,
            "candidate": candidate_totals,
            "boundary": "one complete Gate+Up sweep across I_tp=512",
        },
        "source_identity": source_identity,
        "source_temporal_contract": temporal_contract,
        "checks": checks,
        "gate_pass": all(checks.values()),
        "evidence_boundary": [
            "A/SFA and physical-N128 SFB replay are allowed candidate consequences.",
            "This ledger checks registered coordinates and source identity; matched NCU checks dynamic Tensor work.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate_pass": payload["gate_pass"], "checks": checks}, sort_keys=True))
    return 0 if payload["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
