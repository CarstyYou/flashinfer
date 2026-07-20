#!/usr/bin/env python3
"""Build the locked exp_016 Route/Q0 baseline and token-major overlay."""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MEMORY_ROOT = ROOT.parents[1]
SOURCE = MEMORY_ROOT / "moe_dynamic_kernel_opt.py"
OVERLAY_ROOT = ROOT / "results/overlays"

EXPECTED_BASELINE_SHA256 = (
    "c88cef63492b60c0a77484b50f6400b83a103d168e1535b78972341503810184"
)
EXPECTED_CANDIDATE_SHA256 = (
    "ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19"
)
BASELINE_NAME = "baseline_pair_major"
CANDIDATE_NAME = "candidate_token_major_reuse"
BASELINE_FRAGMENT = ROOT / "fixtures/route_q0_pair_major.pyfrag"

_PRODUCER_DECLARATION = (
    "        producer_batch_pairs = num_cta_warps * Int32(_PRODUCER_PAIRS_PER_WARP)\n"
)
_TOKEN_PRODUCER_DECLARATION = (
    "        # exp_016: pair_head is a token counter in the token-major overlay.\n"
    "        producer_batch_tokens = num_cta_warps\n"
)
_CLAIM_BLOCK = (
    "                claim_count = producer_batch_pairs\n"
    "                if cutlass.const_expr(self.share_input_across_experts):\n"
    "                    claim_count = num_cta_warps\n"
)
_TOKEN_CLAIM_BLOCK = "                claim_count = producer_batch_tokens\n"
_LIMIT_BLOCK = (
    "            producer_limit = total_pairs\n"
    "            if cutlass.const_expr(self.share_input_across_experts):\n"
    "                producer_limit = num_tokens\n"
)
_TOKEN_LIMIT_BLOCK = "            producer_limit = num_tokens\n"

_ORDINARY_START = (
    "                else:\n"
    "                    warp_item = Int32(0)\n"
    "                    while warp_item < Int32(_PRODUCER_PAIRS_PER_WARP):\n"
)
_ORDINARY_END = "                        warp_item += Int32(1)\n"

_TOKEN_MAJOR_BLOCK = """                else:
                    # exp_016 candidate: W0-W8 each own one token.  This
                    # specialization is deliberately locked to topk=8.
                    token_idx = batch_base + warp_idx
                    if token_idx < num_tokens:
                        route_slot_base = warp_idx * Int32(32)
                        if lane_id == Int32(0):
                            topk_slot = Int32(0)
                            while topk_slot < Int32(8):
                                pair_idx = token_idx * num_topk + topk_slot
                                expert_id = topk_ids[pair_idx].to(Int32)
                                weight = topk_weights[pair_idx].to(cutlass.Float32)
                                row = atomic_add_global_i32(
                                    get_ptr_as_int64(expert_write_rows, expert_id),
                                    Int32(1),
                                )
                                phys_tile = expert_tile_base[expert_id] + row // Int32(
                                    self.tile_shape_mnk[0]
                                )
                                phys_row = phys_tile * Int32(
                                    self.tile_shape_mnk[0]
                                ) + row % Int32(self.tile_shape_mnk[0])
                                st_global_i32(
                                    get_ptr_as_int64(token_map, phys_row), token_idx
                                )
                                st_global_f32(
                                    get_ptr_as_int64(token_weights, phys_row), weight
                                )

                                route_slot = route_slot_base + topk_slot
                                _st_shared_i32(
                                    route_phys_rows_addr + route_slot * Int32(4),
                                    phys_row,
                                )
                                _st_shared_i32(
                                    route_expert_ids_addr + route_slot * Int32(4),
                                    expert_id,
                                )

                                topk_slot += Int32(1)
                        cute.arch.sync_warp()

                        # Preserve the baseline's per-lane scale load and
                        # reciprocal work.  Hoist it out of the block loop,
                        # but do not introduce a 32x broadcast optimization.
                        route_gs = cute.make_rmem_tensor((8,), cutlass.Float32)
                        for cache_slot in cutlass.range_constexpr(8):
                            route_slot = route_slot_base + Int32(cache_slot)
                            expert_id = _ld_shared_i32(
                                route_expert_ids_addr + route_slot * Int32(4)
                            )
                            gs_value = input_global_scale[expert_id].to(
                                cutlass.Float32
                            )
                            if (
                                self.input_scales_are_reciprocal
                                and gs_value != cutlass.Float32(0.0)
                            ):
                                if self.fast_math:
                                    gs_value = rcp_approx_ftz(gs_value)
                                else:
                                    gs_value = cutlass.Float32(1.0) / gs_value
                            route_gs[cache_slot] = gs_value

                        sf_idx = lane_id
                        while sf_idx < sf_blocks_per_row:
                            block_start = sf_idx * Int32(16)
                            values = cute.make_rmem_tensor((16,), cutlass.Float32)
                            block_max = cutlass.Float32(0.0)
                            for elem_idx in cutlass.range_constexpr(16):
                                value = cutlass.Float32(
                                    a_input[
                                        token_idx, block_start + Int32(elem_idx)
                                    ]
                                )
                                values[elem_idx] = value
                                block_max = fmax_f32(block_max, fabs_f32(value))

                            # Preserve eight independent quant/store operations;
                            # only the BF16 load and absmax are shared.
                            for cache_slot in cutlass.range_constexpr(8):
                                route_slot = route_slot_base + Int32(cache_slot)
                                phys_row = _ld_shared_i32(
                                    route_phys_rows_addr + route_slot * Int32(4)
                                )
                                phys_tile = phys_row // Int32(
                                    self.tile_shape_mnk[0]
                                )
                                tile_row = phys_row - phys_tile * Int32(
                                    self.tile_shape_mnk[0]
                                )
                                gs_value = route_gs[cache_slot]

                                packed64 = Uint64(0)
                                scale_byte = Uint8(0)
                                if self.fast_math:
                                    packed64, scale_byte = quantize_block_fp4_fast(
                                        values, block_max, gs_value
                                    )
                                else:
                                    packed64, scale_byte = quantize_block_fp4(
                                        values, block_max, gs_value
                                    )

                                output_offset = (
                                    phys_row * output_bytes_per_row
                                    + sf_idx * Int32(8)
                                )
                                st_global_u64(
                                    get_ptr_as_int64(
                                        packed_a_storage, output_offset
                                    ),
                                    packed64,
                                )
                                k_tile_idx = sf_idx // Int32(4)
                                outer_m_idx = tile_row % Int32(32)
                                inner_m_idx = (
                                    tile_row % Int32(32 * 4)
                                ) // Int32(32)
                                inner_k_idx = sf_idx % Int32(4)
                                scale_offset = (
                                    phys_tile
                                    * num_k_tiles
                                    * Int32(32 * 4 * 4)
                                    + k_tile_idx * Int32(32 * 4 * 4)
                                    + outer_m_idx * Int32(4 * 4)
                                    + inner_m_idx * Int32(4)
                                    + inner_k_idx
                                )
                                scale_storage[scale_offset] = scale_byte
                            sf_idx += Int32(32)

                        # Retain the enabled publication protocol even though
                        # exp_016 explicitly runs deferred publication (0).
                        if full_tile_publish_enabled > Int32(0):
                            cute.arch.sync_warp()
                            _threadfence()
                            cute.arch.sync_warp()
                            if lane_id == Int32(0):
                                topk_slot = Int32(0)
                                while topk_slot < Int32(8):
                                    route_slot = route_slot_base + topk_slot
                                    phys_row = _ld_shared_i32(
                                        route_phys_rows_addr
                                        + route_slot * Int32(4)
                                    )
                                    expert_id = _ld_shared_i32(
                                        route_expert_ids_addr
                                        + route_slot * Int32(4)
                                    )
                                    phys_tile = phys_row // Int32(
                                        self.tile_shape_mnk[0]
                                    )
                                    completed = atomic_add_global_i32(
                                        get_ptr_as_int64(
                                            tile_write_count, phys_tile
                                        ),
                                        Int32(1),
                                    ) + Int32(1)
                                    if completed == Int32(
                                        self.tile_shape_mnk[0]
                                    ):
                                        self.publish_ready_tasks(
                                            task_tail,
                                            task_ready,
                                            task_expert,
                                            task_m_tile,
                                            task_slice_begin,
                                            task_slice_count,
                                            task_valid_rows,
                                            route_gate_tile_cnt,
                                            task_slice_chunk,
                                            expert_id,
                                            phys_tile,
                                            Int32(self.tile_shape_mnk[0]),
                                        )
                                    topk_slot += Int32(1)
"""


def pair_major_block() -> str:
    block = BASELINE_FRAGMENT.read_text(encoding="utf-8")
    if not block.endswith("\n"):
        raise RuntimeError("pair-major fixture must end with a newline")
    if not block.startswith(_ORDINARY_START):
        raise RuntimeError("pair-major fixture start anchor drift")
    if not block.endswith(_ORDINARY_END):
        raise RuntimeError("pair-major fixture end anchor drift")
    return block


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def memory_relative(path: Path, memory_root: Path) -> str:
    try:
        return path.resolve().relative_to(memory_root.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"path escapes w4a4_moe_bench root: {path}") from error


def _replace_exact(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return source.replace(old, new, 1)


def _replace_ordinary_branch(source: str) -> str:
    if source.count(_ORDINARY_START) != 1:
        raise RuntimeError("ordinary P3 start: expected exactly one anchor")
    start = source.index(_ORDINARY_START)
    end = source.find(_ORDINARY_END, start)
    if end < 0 or source.find(_ORDINARY_END, end + 1) >= 0:
        raise RuntimeError("ordinary P3 end: expected exactly one anchor")
    end += len(_ORDINARY_END)
    return source[:start] + _TOKEN_MAJOR_BLOCK + source[end:]


def apply_candidate(source: str) -> str:
    candidate = _replace_exact(
        source,
        _PRODUCER_DECLARATION,
        _TOKEN_PRODUCER_DECLARATION,
        label="producer unit",
    )
    candidate = _replace_exact(
        candidate, _CLAIM_BLOCK, _TOKEN_CLAIM_BLOCK, label="token claim"
    )
    candidate = _replace_exact(
        candidate, _LIMIT_BLOCK, _TOKEN_LIMIT_BLOCK, label="token limit"
    )
    return _replace_ordinary_branch(candidate)


def revert_candidate(source: str) -> str:
    baseline = _replace_exact(
        source,
        _TOKEN_PRODUCER_DECLARATION,
        _PRODUCER_DECLARATION,
        label="producer unit reverse",
    )
    baseline = _replace_exact(
        baseline,
        _TOKEN_CLAIM_BLOCK,
        _CLAIM_BLOCK,
        label="token claim reverse",
    )
    baseline = _replace_exact(
        baseline,
        _TOKEN_LIMIT_BLOCK,
        _LIMIT_BLOCK,
        label="token limit reverse",
    )
    baseline = _replace_exact(
        baseline,
        _TOKEN_MAJOR_BLOCK,
        pair_major_block(),
        label="ordinary P3 reverse",
    )
    observed = sha256(baseline.encode())
    if observed != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(
            f"reconstructed baseline drift: {observed} != {EXPECTED_BASELINE_SHA256}"
        )
    return baseline


def resolve_arms(source: str) -> tuple[str, str, str]:
    source_sha = sha256(source.encode())
    if source_sha == EXPECTED_BASELINE_SHA256:
        return source, apply_candidate(source), BASELINE_NAME
    if source_sha == EXPECTED_CANDIDATE_SHA256:
        return revert_candidate(source), source, CANDIDATE_NAME
    raise RuntimeError(
        "locked opt source drift: "
        f"{source_sha} not in "
        f"({EXPECTED_BASELINE_SHA256}, {EXPECTED_CANDIDATE_SHA256})"
    )


def work_ledger(
    *, num_tokens: int = 8192, topk: int = 8, cols: int = 2048, warps: int = 9
) -> dict[str, int]:
    if min(num_tokens, topk, cols, warps) <= 0 or cols % 16:
        raise ValueError("positive dimensions and cols divisible by 16 are required")
    routes = num_tokens * topk
    blocks_per_row = cols // 16
    pair_claim = warps * 2
    return {
        "logical_routes": routes,
        "quant_blocks_baseline": routes * blocks_per_row,
        "quant_blocks_candidate": routes * blocks_per_row,
        "bf16_block_loads_baseline": routes * blocks_per_row,
        "bf16_block_loads_candidate": num_tokens * blocks_per_row,
        "bf16_elements_loaded_baseline": routes * cols,
        "bf16_elements_loaded_candidate": num_tokens * cols,
        "block_absmax_baseline": routes * blocks_per_row,
        "block_absmax_candidate": num_tokens * blocks_per_row,
        "productive_claims_baseline": (routes + pair_claim - 1) // pair_claim,
        "productive_claims_candidate": (num_tokens + warps - 1) // warps,
        "row_allocation_atomics_baseline": routes,
        "row_allocation_atomics_candidate": routes,
        "expert_scale_normalizations_baseline": routes,
        "expert_scale_normalizations_candidate": routes,
        "expert_scale_lane_loads_baseline": routes * 32,
        "expert_scale_lane_loads_candidate": routes * 32,
        "packed_fp4_stores_baseline": routes * blocks_per_row,
        "packed_fp4_stores_candidate": routes * blocks_per_row,
        "sfa_stores_baseline": routes * blocks_per_row,
        "sfa_stores_candidate": routes * blocks_per_row,
    }


def mechanism_gate(baseline: str, candidate: str) -> dict[str, Any]:
    start = candidate.index("                else:\n                    # exp_016")
    end = candidate.index("\n        cute.arch.sync_threads()", start)
    ordinary = candidate[start:end]
    checks = {
        "baseline_identity": sha256(baseline.encode()) == EXPECTED_BASELINE_SHA256,
        "candidate_differs": candidate != baseline,
        "token_counter_claim_9warps": (
            "producer_batch_tokens = num_cta_warps" in candidate
            and "claim_count = producer_batch_tokens" in candidate
        ),
        "token_counter_limit": "producer_limit = num_tokens" in candidate,
        "one_token_per_warp": "token_idx = batch_base + warp_idx" in ordinary,
        "topk8_scope_gate": (
            "specialization is deliberately locked to topk=8" in ordinary
            and "while topk_slot < Int32(8)" in ordinary
        ),
        "warp_private_route_slots": (
            "route_slot_base = warp_idx * Int32(32)" in ordinary
        ),
        "eight_routes_allocated": "while topk_slot < Int32(8)" in ordinary,
        "per_lane_scale_rmem_outside_block_loop": (
            ordinary.index("route_gs = cute.make_rmem_tensor")
            < ordinary.index("while sf_idx < sf_blocks_per_row:")
            and "route_gs[cache_slot] = gs_value" in ordinary
            and "gs_value = route_gs[cache_slot]" in ordinary
        ),
        "no_scale_smem_broadcast": (
            "route_scale_slot" not in ordinary
            and "st_shared_f32(" not in ordinary
            and "ld_shared_f32(" not in ordinary
        ),
        "input_loaded_once_before_expert_loop": (
            ordinary.index("a_input[")
            < ordinary.index("# Preserve eight independent quant/store operations")
            and ordinary.count("a_input[") == 1
        ),
        "eight_independent_quant_stores": (
            "for cache_slot in cutlass.range_constexpr(8):" in ordinary
            and ordinary.count("quantize_block_fp4_fast(") == 1
            and ordinary.count("quantize_block_fp4(") == 1
            and ordinary.count("st_global_u64(") == 1
            and ordinary.count("scale_storage[scale_offset] = scale_byte") == 1
        ),
        "full_tile_publish_retained": (
            "if full_tile_publish_enabled > Int32(0):" in ordinary
            and "self.publish_ready_tasks(" in ordinary
        ),
        "deferred_publish_retained": (
            candidate.count("if full_tile_publish_enabled == Int32(0):")
            == baseline.count("if full_tile_publish_enabled == Int32(0):")
        ),
        "shared_scale_specialization_retained": (
            candidate.count("if cutlass.const_expr(self.share_input_across_experts):")
            == baseline.count("if cutlass.const_expr(self.share_input_across_experts):")
            - 2
            and "shared_input_gs_value = input_global_scale[Int32(0)]" in candidate
            and "route_output_base = cute.make_rmem_tensor((8,), Int32)" in candidate
        ),
        "ordinary_pair_loop_removed": "warp_item" not in ordinary,
    }
    return {"checks": checks, "gate_pass": all(checks.values())}


def _diff(baseline: str, candidate: str) -> str:
    return "".join(
        difflib.unified_diff(
            baseline.splitlines(keepends=True),
            candidate.splitlines(keepends=True),
            fromfile=f"{BASELINE_NAME}/moe_dynamic_kernel.py",
            tofile=f"{CANDIDATE_NAME}/moe_dynamic_kernel.py",
        )
    )


def build_overlays(
    *,
    source: Path = SOURCE,
    overlay_root: Path = OVERLAY_ROOT,
    memory_root: Path = MEMORY_ROOT,
) -> dict[str, Any]:
    source_bytes = source.read_bytes()
    source_sha = sha256(source_bytes)
    baseline, candidate, source_role = resolve_arms(source_bytes.decode("utf-8"))
    baseline_bytes = baseline.encode("utf-8")
    baseline_sha = sha256(baseline_bytes)
    ast.parse(baseline, filename=f"{BASELINE_NAME}/moe_dynamic_kernel.py")
    ast.parse(candidate, filename=f"{CANDIDATE_NAME}/moe_dynamic_kernel.py")

    gate = mechanism_gate(baseline, candidate)
    if not gate["gate_pass"]:
        raise RuntimeError(f"candidate mechanism gate failed: {gate}")
    ledger = work_ledger()
    if not (
        ledger["quant_blocks_baseline"] == ledger["quant_blocks_candidate"]
        and ledger["bf16_block_loads_baseline"]
        == 8 * ledger["bf16_block_loads_candidate"]
    ):
        raise RuntimeError("work-ledger causal gate failed")

    baseline_path = overlay_root / BASELINE_NAME / "moe_dynamic_kernel.py"
    candidate_path = overlay_root / CANDIDATE_NAME / "moe_dynamic_kernel.py"
    diff_path = overlay_root / f"{CANDIDATE_NAME}.diff"
    identity_path = overlay_root / "identity.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_bytes(baseline_bytes)
    candidate_bytes = candidate.encode("utf-8")
    candidate_path.write_bytes(candidate_bytes)
    diff = _diff(baseline, candidate)
    diff_path.write_text(diff, encoding="utf-8")

    identity: dict[str, Any] = {
        "schema": "exp016.route-q0-overlay.v1",
        "path_base": "w4a4_moe_bench_root",
        "source": memory_relative(source, memory_root),
        "source_sha256": source_sha,
        "source_role": source_role,
        "arms": {
            BASELINE_NAME: {
                "path": memory_relative(baseline_path, memory_root),
                "sha256": baseline_sha,
                "byte_identical_to_source": source_role == BASELINE_NAME,
            },
            CANDIDATE_NAME: {
                "path": memory_relative(candidate_path, memory_root),
                "sha256": sha256(candidate_bytes),
                "byte_identical_to_source": source_role == CANDIDATE_NAME,
            },
        },
        "counter_units": {
            BASELINE_NAME: {"unit": "route_pair", "claim_per_cta": 18},
            CANDIDATE_NAME: {"unit": "token", "claim_per_cta": 9},
        },
        "scope": {
            "dispatch": "[E] scale; share_input_across_experts=False",
            "topk": 8,
            "full_tile_publish_enabled": 0,
            "unchanged": "P0/P1/P2/P4, FC1, SwiGLU/Q1, FC2, Scatter, CTA",
        },
        "mechanism_gate": gate,
        "work_ledger_m8192_topk8_h2048": ledger,
        "diff": {
            "path": memory_relative(diff_path, memory_root),
            "sha256": sha256(diff.encode("utf-8")),
        },
    }
    identity_path.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OVERLAY_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = build_overlays(source=args.source, overlay_root=args.output_dir)
    print(json.dumps(identity, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
