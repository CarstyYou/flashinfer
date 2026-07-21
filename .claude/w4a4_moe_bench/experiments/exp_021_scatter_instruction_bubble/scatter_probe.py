#!/usr/bin/env python3
"""Minimal 8-warp Direct-32 Scatter target for instruction/issue diagnosis."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cuda.bindings import runtime as cudart
from cutlass.cutlass_dsl import Int32
from flashinfer.cute_dsl.fp4_common import get_ptr_as_int64, scatter_add_v4_bf16x2


TILE_M = 128
TILE_N = 128
HIDDEN = 2048
OUTPUT_TILES = HIDDEN // TILE_N
SLICES = 4
THREADS = 256
GRID_CTAS = 110
SMEM_PADDING_BYTES = 50 * 1024


@cute.kernel
def scatter_probe_kernel(
    token_map: cute.Tensor,
    route_weights: cute.Tensor,
    valid_rows_by_group: cute.Tensor,
    output: cute.Tensor,
    sC_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim_x, _, _ = cute.arch.grid_dim()
    tidx = Int32(tidx)
    bidx = Int32(bidx)
    gdim_x = Int32(gdim_x)

    smem = utils.SmemAllocator()
    token_cache = smem.allocate_tensor(
        cutlass.Int32, cute.make_layout((TILE_M,)), byte_alignment=16
    )
    weight_cache = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((TILE_M,)), byte_alignment=16
    )
    # Preserve the accepted Opt one-CTA-per-SM shared-memory tier.
    smem.allocate_array(cutlass.Uint8, SMEM_PADDING_BYTES, byte_alignment=1024)
    sC = smem.allocate_tensor(
        cutlass.BFloat16,
        sC_layout.outer,
        byte_alignment=1024,
        swizzle=sC_layout.inner,
    )

    # Runtime dummy values: one negligible initialization per CTA, then all
    # 2,548 slice tasks reuse the same resident K_SW128 sC tile.
    flat_idx = tidx
    while flat_idx < Int32(TILE_M * TILE_N):
        row = flat_idx // Int32(TILE_N)
        col = flat_idx - row * Int32(TILE_N)
        pattern = (row * Int32(3) + col * Int32(5) + bidx) & Int32(15)
        sC[row, col, 0] = cutlass.BFloat16(
            cutlass.Float32(pattern + Int32(1)) * cutlass.Float32(0.0009765625)
        )
        flat_idx += Int32(THREADS)
    cute.arch.sync_threads()

    num_groups = Int32(valid_rows_by_group.shape[0])
    task = bidx
    while task < num_groups * Int32(SLICES):
        group = task // Int32(SLICES)
        if tidx < Int32(TILE_M):
            token_cache[tidx] = Int32(token_map[group, tidx])
            weight_cache[tidx] = cutlass.Float32(route_weights[group, tidx])
        cute.arch.sync_threads()

        valid_rows = Int32(valid_rows_by_group[group])
        if valid_rows < Int32(0):
            valid_rows = Int32(0)
        if valid_rows > Int32(TILE_M):
            valid_rows = Int32(TILE_M)

        lane_id = tidx & Int32(31)
        warp_in_tile = tidx >> Int32(5)
        warp_m_base = (warp_in_tile >> Int32(1)) * Int32(32)
        warp_n_base = (warp_in_tile & Int32(1)) * Int32(64)
        warp_rows = valid_rows - warp_m_base
        if warp_rows < Int32(0):
            warp_rows = Int32(0)
        if warp_rows > Int32(32):
            warp_rows = Int32(32)

        # Match the production output-tile unroll.  One completion barrier is
        # retained per tile; adding an adjacent synthetic pre-barrier would
        # manufacture a bubble because this harness omits the intervening FC2.
        for output_tile in range(0, OUTPUT_TILES, 1, unroll=4):  # type: ignore[call-overload]
            tile_n_base = Int32(output_tile * TILE_N)
            vec_idx = lane_id
            while vec_idx < warp_rows * Int32(8):
                local_row = warp_m_base + vec_idx // Int32(8)
                local_col = warp_n_base + (vec_idx % Int32(8)) * Int32(8)
                global_col = tile_n_base + local_col
                token = Int32(token_cache[local_row])
                weight = cutlass.Float32(weight_cache[local_row])

                scatter_add_v4_bf16x2(
                    get_ptr_as_int64(output, token * Int32(HIDDEN) + global_col),
                    weight * cutlass.Float32(sC[local_row, local_col, 0]),
                    weight * cutlass.Float32(sC[local_row, local_col + 1, 0]),
                    weight * cutlass.Float32(sC[local_row, local_col + 2, 0]),
                    weight * cutlass.Float32(sC[local_row, local_col + 3, 0]),
                    weight * cutlass.Float32(sC[local_row, local_col + 4, 0]),
                    weight * cutlass.Float32(sC[local_row, local_col + 5, 0]),
                    weight * cutlass.Float32(sC[local_row, local_col + 6, 0]),
                    weight * cutlass.Float32(sC[local_row, local_col + 7, 0]),
                )
                vec_idx += Int32(32)
            cute.arch.sync_threads()
        task += gdim_x


@cute.jit
def launch_scatter_probe(
    token_map: cute.Tensor,
    route_weights: cute.Tensor,
    valid_rows: cute.Tensor,
    output: cute.Tensor,
    stream: cuda.CUstream,
):
    layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(
            utils.LayoutEnum.ROW_MAJOR, cutlass.BFloat16, TILE_N
        ),
        cutlass.BFloat16,
    )
    sC_layout = cute.tile_to_shape(
        layout_atom, cute.append((TILE_M, TILE_N), 1), order=(0, 1, 2)
    )
    scatter_probe_kernel(
        token_map, route_weights, valid_rows, output, sC_layout
    ).launch(grid=(GRID_CTAS, 1, 1), block=(THREADS, 1, 1), stream=stream)


def load_exp020_runner(root: Path):
    path = root.parent / "exp_020_dsm_8way_scatter_demo" / "run_demo.py"
    spec = importlib.util.spec_from_file_location("exp020_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import fixture builder: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    import torch
    from cutlass.cute.runtime import from_dlpack

    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if tuple(torch.cuda.get_device_capability()) != (12, 0):
        raise RuntimeError("exp_021 requires SM120")
    exp020 = load_exp020_runner(Path(__file__).resolve().parent)
    case = exp020.build_canonical_groups(args.fixture)
    if case["ledger"]["direct_redg_bf16x8"] != 67_108_864:
        raise RuntimeError("32-way REDG ledger drift")

    token_map = case["token_map"].cuda().contiguous()
    weights = case["route_weights"].cuda().contiguous()
    valid_rows = case["valid_rows"].cuda().contiguous()
    output = torch.zeros((8192, HIDDEN), dtype=torch.bfloat16, device="cuda")
    token_arg = from_dlpack(token_map, assumed_align=16)
    weight_arg = from_dlpack(weights, assumed_align=16)
    valid_arg = from_dlpack(valid_rows, assumed_align=16)
    output_arg = from_dlpack(output, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        launch_scatter_probe,
        token_arg,
        weight_arg,
        valid_arg,
        output_arg,
        stream,
    )

    # Compile/warm outside the profiler range.
    compiled(token_arg, weight_arg, valid_arg, output_arg, stream)
    torch.cuda.synchronize()
    output.zero_()
    torch.cuda.synchronize()

    start_status = cudart.cudaProfilerStart()
    start_status = start_status[0] if isinstance(start_status, tuple) else start_status
    if int(start_status) != 0:
        raise RuntimeError("cudaProfilerStart failed")
    compiled(token_arg, weight_arg, valid_arg, output_arg, stream)
    torch.cuda.synchronize()
    stop_status = cudart.cudaProfilerStop()
    stop_status = stop_status[0] if isinstance(stop_status, tuple) else stop_status
    if int(stop_status) != 0:
        raise RuntimeError("cudaProfilerStop failed")

    payload = {
        "schema": "exp021.target.v1",
        "case": {"m": 8192, "hidden": HIDDEN, "topk": 8, "slices": SLICES},
        "launch": {"grid": [GRID_CTAS, 1, 1], "block": [THREADS, 1, 1]},
        "fixture": {
            "groups": int(token_map.shape[0]),
            "slice_tasks": int(token_map.shape[0]) * SLICES,
            "identity_sha256": case["group_identity_sha256"],
        },
        "work": case["ledger"],
        "path_live": bool(torch.count_nonzero(output).item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
