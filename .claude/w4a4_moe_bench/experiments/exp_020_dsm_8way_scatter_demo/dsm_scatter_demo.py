"""CuTeDSL mechanism demo for Direct-32 versus DSM-8 MoE scatter.

This file intentionally contains only the shared kernel source and its host
launcher.  The harness owns fixture validation, output clearing, timing, and
correctness checks.

Input contract:

* ``token_map`` and ``route_weights`` have shape ``[groups, 128]``;
* ``valid_rows`` has shape ``[groups]``;
* ``output`` is row-major BF16 with shape ``[num_tokens, 2048]``;
* one four-CTA cluster represents one logical group, and CTA rank 0..3
  represents FC2 K-slice 0..3.

``use_dsm=False`` selects Direct-32: every slice CTA scatters its complete
128x128 partial tile.  ``use_dsm=True`` selects DSM-8: the four slice partials
are merged in FP32 through DSM and each output vector has exactly one spatial
owner before the final BF16x8 REDG.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils

from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import Int32, dsl_user_op

from flashinfer.cute_dsl.fp4_common import (
    get_ptr_as_int64,
    scatter_add_v4_bf16x2,
)


_CLUSTER_CTAS = 4
_M_TILE = 128
_N_TILE = 128
_N_TILES = 16
_MATH_WARPS = 8
_THREADS_PER_CTA = 288
_MATH_THREADS = _MATH_WARPS * 32
_VALUES_PER_RED = 8
# token cache (512 B) + weight cache (512 B) + padding (50 KiB) + sC
# (32 KiB) = 84,992 B, matching the accepted Opt kernel's one-CTA-per-SM
# shared-memory envelope.
_SMEM_PADDING_BYTES = 50 * 1024


@dsl_user_op
def remote_smem_ptr_in_cluster(smem_ptr, cta_rank, *, loc=None, ip=None):
    """Backport the CUDA 13.1+ NVVM MAPA wrapper to the locked DSL package.

    The experiment must use the same CuTeDSL package as the accepted Opt
    kernel.  That package exposes the underlying ``nvvm.mapa`` op but predates
    the public ``cute.get_remote_smem_ptr_in_cluster`` helper.
    """

    dsmem_ptr_type = llvm.PointerType.get(7)
    # The locked DSL compiler must keep the mapped pointer generic.  Casting
    # it back to address-space 3 lowers remote loads to local LDS and traps on
    # SM120.  CUDA C++ ``cluster.map_shared_rank`` uses the same
    # shared::cluster -> generic conversion and lowers the load to LD.E.
    generic_ptr_type = llvm.PointerType.get(0)
    remote_dsmem_ptr = nvvm.mapa(
        dsmem_ptr_type,
        smem_ptr.llvm_ptr,
        Int32(cta_rank).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    remote_generic_ptr = llvm.addrspacecast(
        generic_ptr_type, remote_dsmem_ptr, loc=loc, ip=ip
    )
    result = cute.make_ptr(
        smem_ptr.dtype,
        remote_generic_ptr,
        assumed_align=smem_ptr.alignment,
        loc=loc,
        ip=ip,
    )
    if cutlass.const_expr(smem_ptr.value.type.is_swizzled):
        swizzle = cute.Swizzle(cute.static(smem_ptr.value.type.swizzle_type))
        result = cute.recast_ptr(
            result,
            swizzle_=swizzle,
            dtype=smem_ptr.dtype,
            loc=loc,
            ip=ip,
        )
    return result


@cute.kernel
def dsm_scatter_demo_kernel(
    token_map: cute.Tensor,
    route_weights: cute.Tensor,
    valid_rows_by_group: cute.Tensor,
    output: cute.Tensor,
    epi_smem_layout_staged: cute.ComposedLayout,
    use_dsm: cutlass.Constexpr,
):
    """Generate BF16 partials and execute one of the two scatter protocols."""

    tidx, _, _ = cute.arch.thread_idx()
    group_idx, _, _ = cute.arch.cluster_idx()
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

    tidx = Int32(tidx)
    group_idx = Int32(group_idx)
    cta_rank = Int32(cta_rank)

    smem = utils.SmemAllocator()
    token_cache = smem.allocate_tensor(
        cutlass.Int32,
        cute.make_layout((_M_TILE,)),
        byte_alignment=16,
    )
    weight_cache = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((_M_TILE,)),
        byte_alignment=16,
    )
    # Keep the two constexpr arms at an identical dynamic-SMEM footprint.
    _padding = smem.allocate_array(
        cutlass.Uint8,
        _SMEM_PADDING_BYTES,
        byte_alignment=1024,
    )
    sC = smem.allocate_tensor(
        cutlass.BFloat16,
        epi_smem_layout_staged.outer,
        byte_alignment=1024,
        swizzle=epi_smem_layout_staged.inner,
    )

    # Every slice CTA stages the same group metadata.  Only the eight math
    # warps move data; the ninth warp remains control/idle but joins barriers.
    if tidx < Int32(_M_TILE):
        token_cache[tidx] = Int32(token_map[group_idx, tidx])
        weight_cache[tidx] = cutlass.Float32(route_weights[group_idx, tidx])
    cute.arch.sync_threads()

    valid_rows = Int32(valid_rows_by_group[group_idx])
    if valid_rows < Int32(0):
        valid_rows = Int32(0)
    if valid_rows > Int32(_M_TILE):
        valid_rows = Int32(_M_TILE)

    # MAPA is invariant across all sixteen output tiles.  The public wrapper
    # preserves the production swizzle on the remote shared-memory pointers.
    sC_slice0 = sC
    sC_slice1 = sC
    sC_slice2 = sC
    sC_slice3 = sC
    if cutlass.const_expr(use_dsm):
        sC_slice0 = cute.make_tensor(
            remote_smem_ptr_in_cluster(sC.iterator, 0), sC.layout
        )
        sC_slice1 = cute.make_tensor(
            remote_smem_ptr_in_cluster(sC.iterator, 1), sC.layout
        )
        sC_slice2 = cute.make_tensor(
            remote_smem_ptr_in_cluster(sC.iterator, 2), sC.layout
        )
        sC_slice3 = cute.make_tensor(
            remote_smem_ptr_in_cluster(sC.iterator, 3), sC.layout
        )

    output_stride = Int32(output.shape[1])

    for output_tile_idx in cutlass.range_constexpr(_N_TILES):
        # Lightweight deterministic FC2-epilogue stand-in.  Slice rank is
        # encoded in every value so a missing or duplicated DSM slice is
        # observable.  The explicit BF16 store preserves the production
        # FP32->BF16 epilogue boundary before either reduction protocol.
        if tidx < Int32(_MATH_THREADS):
            flat_idx = tidx
            while flat_idx < Int32(_M_TILE * _N_TILE):
                local_row = flat_idx // Int32(_N_TILE)
                local_col = flat_idx - local_row * Int32(_N_TILE)
                pattern = (
                    local_row * Int32(3)
                    + local_col * Int32(5)
                    + Int32(output_tile_idx) * Int32(7)
                ) & Int32(15)
                partial = cutlass.Float32(cta_rank + Int32(1)) * cutlass.Float32(
                    0.03125
                ) + cutlass.Float32(pattern) * cutlass.Float32(0.0009765625)
                sC[local_row, local_col, 0] = cutlass.BFloat16(partial)
                flat_idx += Int32(_MATH_THREADS)

        # Common local completion point.  The proxy fence mirrors the
        # production epilogue-to-scatter handoff and is executed by all 288
        # threads in both arms.
        cute.arch.sync_threads()
        cute.arch.fence_proxy("async.shared", space="cta")

        tile_n_base = Int32(output_tile_idx * _N_TILE)

        if cutlass.const_expr(use_dsm):
            # Publication barrier: all threads in all four CTAs participate.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            if tidx < Int32(_MATH_THREADS):
                # Each CTA rank owns a disjoint 32-row x N128 region.  This
                # gives all four CTAs useful merge work without duplicate
                # output ownership.
                owner_row_base = cta_rank * Int32(32)
                owner_rows = valid_rows - owner_row_base
                if owner_rows < Int32(0):
                    owner_rows = Int32(0)
                if owner_rows > Int32(32):
                    owner_rows = Int32(32)

                vec_idx = tidx
                vecs_per_row = Int32(_N_TILE // _VALUES_PER_RED)
                while vec_idx < owner_rows * vecs_per_row:
                    local_row = owner_row_base + vec_idx // vecs_per_row
                    local_col = (vec_idx % vecs_per_row) * Int32(_VALUES_PER_RED)
                    global_col = tile_n_base + local_col
                    token = Int32(token_cache[local_row])
                    weight = cutlass.Float32(weight_cache[local_row])

                    # Fixed slice0 -> slice1 -> slice2 -> slice3 FP32 order.
                    v0 = cutlass.Float32(sC_slice0[local_row, local_col, 0])
                    v1 = cutlass.Float32(sC_slice0[local_row, local_col + 1, 0])
                    v2 = cutlass.Float32(sC_slice0[local_row, local_col + 2, 0])
                    v3 = cutlass.Float32(sC_slice0[local_row, local_col + 3, 0])
                    v4 = cutlass.Float32(sC_slice0[local_row, local_col + 4, 0])
                    v5 = cutlass.Float32(sC_slice0[local_row, local_col + 5, 0])
                    v6 = cutlass.Float32(sC_slice0[local_row, local_col + 6, 0])
                    v7 = cutlass.Float32(sC_slice0[local_row, local_col + 7, 0])

                    v0 = v0 + cutlass.Float32(sC_slice1[local_row, local_col, 0])
                    v1 = v1 + cutlass.Float32(sC_slice1[local_row, local_col + 1, 0])
                    v2 = v2 + cutlass.Float32(sC_slice1[local_row, local_col + 2, 0])
                    v3 = v3 + cutlass.Float32(sC_slice1[local_row, local_col + 3, 0])
                    v4 = v4 + cutlass.Float32(sC_slice1[local_row, local_col + 4, 0])
                    v5 = v5 + cutlass.Float32(sC_slice1[local_row, local_col + 5, 0])
                    v6 = v6 + cutlass.Float32(sC_slice1[local_row, local_col + 6, 0])
                    v7 = v7 + cutlass.Float32(sC_slice1[local_row, local_col + 7, 0])

                    v0 = v0 + cutlass.Float32(sC_slice2[local_row, local_col, 0])
                    v1 = v1 + cutlass.Float32(sC_slice2[local_row, local_col + 1, 0])
                    v2 = v2 + cutlass.Float32(sC_slice2[local_row, local_col + 2, 0])
                    v3 = v3 + cutlass.Float32(sC_slice2[local_row, local_col + 3, 0])
                    v4 = v4 + cutlass.Float32(sC_slice2[local_row, local_col + 4, 0])
                    v5 = v5 + cutlass.Float32(sC_slice2[local_row, local_col + 5, 0])
                    v6 = v6 + cutlass.Float32(sC_slice2[local_row, local_col + 6, 0])
                    v7 = v7 + cutlass.Float32(sC_slice2[local_row, local_col + 7, 0])

                    v0 = v0 + cutlass.Float32(sC_slice3[local_row, local_col, 0])
                    v1 = v1 + cutlass.Float32(sC_slice3[local_row, local_col + 1, 0])
                    v2 = v2 + cutlass.Float32(sC_slice3[local_row, local_col + 2, 0])
                    v3 = v3 + cutlass.Float32(sC_slice3[local_row, local_col + 3, 0])
                    v4 = v4 + cutlass.Float32(sC_slice3[local_row, local_col + 4, 0])
                    v5 = v5 + cutlass.Float32(sC_slice3[local_row, local_col + 5, 0])
                    v6 = v6 + cutlass.Float32(sC_slice3[local_row, local_col + 6, 0])
                    v7 = v7 + cutlass.Float32(sC_slice3[local_row, local_col + 7, 0])

                    scatter_add_v4_bf16x2(
                        get_ptr_as_int64(output, token * output_stride + global_col),
                        weight * v0,
                        weight * v1,
                        weight * v2,
                        weight * v3,
                        weight * v4,
                        weight * v5,
                        weight * v6,
                        weight * v7,
                    )
                    vec_idx += Int32(_MATH_THREADS)

            # Consumed barrier protects all local and remote sC reads before
            # any CTA overwrites the buffer for the next N128 tile or exits.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        else:
            if tidx < Int32(_MATH_THREADS):
                # Production Direct-32 mapping: each of eight warps owns one
                # disjoint 32-row x 64-column strip of this slice partial.
                lane_id = tidx & Int32(31)
                warp_in_tile = tidx >> Int32(5)
                warp_m_base = (warp_in_tile >> Int32(1)) * Int32(32)
                warp_n_base = (warp_in_tile & Int32(1)) * Int32(64)
                warp_rows = valid_rows - warp_m_base
                if warp_rows < Int32(0):
                    warp_rows = Int32(0)
                if warp_rows > Int32(32):
                    warp_rows = Int32(32)

                vec_idx = lane_id
                vecs_per_strip_row = Int32(64 // _VALUES_PER_RED)
                while vec_idx < warp_rows * vecs_per_strip_row:
                    local_row = warp_m_base + vec_idx // vecs_per_strip_row
                    local_col = warp_n_base + (vec_idx % vecs_per_strip_row) * Int32(
                        _VALUES_PER_RED
                    )
                    global_col = tile_n_base + local_col
                    token = Int32(token_cache[local_row])
                    weight = cutlass.Float32(weight_cache[local_row])

                    scatter_add_v4_bf16x2(
                        get_ptr_as_int64(output, token * output_stride + global_col),
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

            # Local consumed barrier prevents early warps from overwriting sC
            # while another Direct warp still reads the current tile.
            cute.arch.sync_threads()


@cute.jit
def launch_demo(
    token_map: cute.Tensor,
    route_weights: cute.Tensor,
    valid_rows: cute.Tensor,
    output: cute.Tensor,
    use_dsm: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    """Launch Direct-32 or DSM-8 with an identical 4-CTA topology.

    Passing a Python ``bool`` for ``use_dsm`` specializes the single kernel
    source into the two arms while preserving the same SMEM allocations.
    """

    c_layout = utils.LayoutEnum.ROW_MAJOR
    c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(
            c_layout,
            cutlass.BFloat16,
            _N_TILE,
        ),
        cutlass.BFloat16,
    )
    epi_smem_layout_staged = cute.tile_to_shape(
        c_smem_layout_atom,
        cute.append((_M_TILE, _N_TILE), 1),
        order=(0, 1, 2),
    )

    num_groups = valid_rows.shape[0]
    dsm_scatter_demo_kernel(
        token_map,
        route_weights,
        valid_rows,
        output,
        epi_smem_layout_staged,
        use_dsm,
    ).launch(
        grid=(num_groups * _CLUSTER_CTAS, 1, 1),
        block=(_THREADS_PER_CTA, 1, 1),
        cluster=(_CLUSTER_CTAS, 1, 1),
        stream=stream,
    )


__all__ = ["dsm_scatter_demo_kernel", "launch_demo"]
