#!/usr/bin/env python3
"""Build the exp_013 compact-epilogue overlay from locked exp_008 v1."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE = (
    ROOT.parent
    / "exp_008_branch_paired_n64_reuse"
    / "results/overlays/branch_paired_n64_v1/moe_dynamic_kernel.py"
)
OUT_DIR = ROOT / "results/overlays/compact_epi_m64_stage2_v0"
OUT = OUT_DIR / "moe_dynamic_kernel.py"
EXPECTED_BASELINE_SHA256 = (
    "f3c246817679d962a3f7160dbe8b9e68262c919e26e306f349200961fc4ac971"
)


Q1_SETUP_OLD = """                    cons_state.reset_count()
                    for fc1_half in cutlass.range_constexpr(2):
"""

Q1_SETUP_NEW = """                    # sA/sSFA are still FC1 input staging until both N64
                    # halves finish.  Keep half0 Q1 payloads in a small per-thread
                    # register cache and publish them only after half1 FC1 completes.
                    sA_u8 = cute.recast_tensor(sA[None, None, 0], cutlass.Uint8)
                    packed_cols = Int32(self.tile_shape_mnk[2] // 2)
                    sf_blocks_per_half = Int32(
                        self.fc1_tile_shape_mnk[1] // 16
                    )
                    half0_packed_cache = cute.make_rmem_tensor(
                        (epi_rest_m,), Uint64
                    )
                    half0_scale_cache = cute.make_rmem_tensor(
                        (epi_rest_m,), Uint8
                    )
                    gs_value = global_scale[task_expert_idx].to(cutlass.Float32)
                    if self.input_scales_are_reciprocal and gs_value != cutlass.Float32(
                        0.0
                    ):
                        if self.fast_math:
                            gs_value = rcp_approx_ftz(gs_value)
                        else:
                            gs_value = cutlass.Float32(1.0) / gs_value

                    cons_state.reset_count()
                    for fc1_half in cutlass.range_constexpr(2):
"""


Q1_SETUP_V1 = """                    # sA/sSFA remain FC1 input staging until both N64
                    # halves finish. Preserve half0 activation in registers, then
                    # materialize a complete M64xN128 sC tile together with half1.
                    sA_u8 = cute.recast_tensor(sA[None, None, 0], cutlass.Uint8)
                    packed_cols = Int32(self.tile_shape_mnk[2] // 2)
                    sf_blocks_per_row_q1 = Int32(self.tile_shape_mnk[2] // 16)
                    half0_act_acc = cute.make_rmem_tensor(
                        fc1_acc_shape, self.acc_dtype
                    )
                    fc1_tRS_rHalf0Act = fc1_tiled_copy_r2s.retile(half0_act_acc)
                    fc1_m_per_epi = fc1_m_tiles // epi_rest_m
                    gs_value = global_scale[task_expert_idx].to(cutlass.Float32)
                    if self.input_scales_are_reciprocal and gs_value != cutlass.Float32(
                        0.0
                    ):
                        if self.fast_math:
                            gs_value = rcp_approx_ftz(gs_value)
                        else:
                            gs_value = cutlass.Float32(1.0) / gs_value

                    cons_state.reset_count()
                    for fc1_half in cutlass.range_constexpr(2):
"""


Q1_SETUP_SHARED = """                    # sA/sSFA are still FC1 input staging until both N64
                    # halves finish. Cache half0's already-quantized payload in
                    # a compact shared bridge so it does not lengthen register
                    # lifetimes across the complete half1 FC1 GEMM.
                    sA_u8 = cute.recast_tensor(sA[None, None, 0], cutlass.Uint8)
                    packed_cols = Int32(self.tile_shape_mnk[2] // 2)
                    sf_blocks_per_half = Int32(
                        self.fc1_tile_shape_mnk[1] // 16
                    )
                    half0_packed_cache = storage.q1_half0_packed_cache.get_tensor(
                        cute.make_layout((self.tile_shape_mnk[0] * 4,))
                    )
                    half0_scale_cache = storage.q1_half0_scale_cache.get_tensor(
                        cute.make_layout((self.tile_shape_mnk[0] * 4,))
                    )
                    gs_value = global_scale[task_expert_idx].to(cutlass.Float32)
                    if self.input_scales_are_reciprocal and gs_value != cutlass.Float32(
                        0.0
                    ):
                        if self.fast_math:
                            gs_value = rcp_approx_ftz(gs_value)
                        else:
                            gs_value = cutlass.Float32(1.0) / gs_value

                    cons_state.reset_count()
                    for fc1_half in cutlass.range_constexpr(2):
"""


Q1_SETUP_RAW = """                    # sA/sSFA are still FC1 input staging until both N64
                    # halves finish. Cache half0's quantized payload in a raw
                    # shared bridge; explicit PTX avoids extending Python tensor
                    # objects and packed values across the complete half1 GEMM.
                    sA_u8 = cute.recast_tensor(sA[None, None, 0], cutlass.Uint8)
                    packed_cols = Int32(self.tile_shape_mnk[2] // 2)
                    sf_blocks_per_half = Int32(
                        self.fc1_tile_shape_mnk[1] // 16
                    )
                    q1_half0_cache_addr = shared_ptr_to_u32(
                        storage.q1_half0_cache.data_ptr()
                    )
                    gs_value = global_scale[task_expert_idx].to(cutlass.Float32)
                    if self.input_scales_are_reciprocal and gs_value != cutlass.Float32(
                        0.0
                    ):
                        if self.fast_math:
                            gs_value = rcp_approx_ftz(gs_value)
                        else:
                            gs_value = cutlass.Float32(1.0) / gs_value

                    cons_state.reset_count()
                    for fc1_half in cutlass.range_constexpr(2):
"""


COMPACT_FC1_Q1 = """                        # Compact M64 epilogue: consume this M128xN64
                        # accumulator through two M64xN64 rectangles.  The same
                        # sC window is reused only after all 8 math warps finish Q1.
                        for epi_m in cutlass.range_constexpr(epi_rest_m):
                            for mma_m_in_epi in cutlass.range_constexpr(
                                MmaMPerEpiM
                            ):
                                mma_m = epi_m * MmaMPerEpiM + mma_m_in_epi
                                for mma_n in cutlass.range_constexpr(fc1_n_tiles):
                                    gate_slice = fc1_tRS_rGate[
                                        (None, mma_m, mma_n)
                                    ]
                                    up_slice = fc1_tRS_rUp[
                                        (None, mma_m, mma_n)
                                    ]
                                    for elem_idx in cutlass.range_constexpr(
                                        cute.size(fc1_tRS_rAct)
                                    ):
                                        g = alpha_value * gate_slice[elem_idx]
                                        u = alpha_value * up_slice[elem_idx]
                                        fc1_tRS_rAct[elem_idx] = gated_activation_f32(
                                            g,
                                            u,
                                            activation=self.activation,
                                            limit=self.swiglu_limit,
                                            alpha=self.swiglu_alpha,
                                            beta=self.swiglu_beta,
                                            fast_math=self.fast_math,
                                        )
                                    act_vec = fc1_tRS_rAct.load()
                                    act_vec = act_vec.to(cutlass.BFloat16)
                                    fc1_tRS_rAct_out.store(act_vec)
                                    cute.copy(
                                        fc1_tiled_copy_r2s,
                                        fc1_tRS_rAct_out,
                                        fc1_tRS_sD[
                                            (None, mma_m_in_epi, mma_n, 0)
                                        ],
                                    )

                            cute.arch.fence_proxy("async.shared", space="cta")
                            self.epilog_sync_barrier.arrive_and_wait()

                            rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])
                            epi_rows = valid_rows - rows_offset
                            if epi_rows > Int32(self.epi_tile[0]):
                                epi_rows = Int32(self.epi_tile[0])
                            if epi_rows < Int32(0):
                                epi_rows = Int32(0)
                            quant_idx = Int32(tidx)
                            while quant_idx < epi_rows * sf_blocks_per_half:
                                local_row = quant_idx // sf_blocks_per_half
                                local_sf_block = (
                                    quant_idx - local_row * sf_blocks_per_half
                                )
                                row = rows_offset + local_row
                                sf_block = (
                                    Int32(fc1_half) * sf_blocks_per_half
                                    + local_sf_block
                                )
                                block_start = local_sf_block * Int32(16)

                                values = cute.make_rmem_tensor(
                                    (16,), cutlass.Float32
                                )
                                block_max = cutlass.Float32(0.0)
                                for elem_idx in cutlass.range_constexpr(16):
                                    value = cutlass.Float32(
                                        sC[
                                            local_row,
                                            block_start + elem_idx,
                                            0,
                                        ]
                                    )
                                    values[elem_idx] = value
                                    block_max = fmax_f32(
                                        block_max, fabs_f32(value)
                                    )

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

                                if fc1_half == 0:
                                    # With 8 math warps, each M64xN64 rectangle
                                    # has exactly 256 blocks: one per thread.
                                    half0_packed_cache[epi_m] = packed64
                                    half0_scale_cache[epi_m] = scale_byte
                                else:
                                    packed_base = sf_block << Int32(3)
                                    dst_pcol = row & Int32(63)
                                    xor_bits = (
                                        ((dst_pcol >> Int32(1)) & Int32(0x3))
                                        << Int32(4)
                                    )
                                    row_high = row >> Int32(6)
                                    for byte_idx in cutlass.range_constexpr(8):
                                        src_pcol = packed_base + Int32(byte_idx)
                                        dst_row = (
                                            (src_pcol ^ xor_bits) << Int32(1)
                                        ) + row_high
                                        dst_flat = dst_row * packed_cols + dst_pcol
                                        byte_val = Uint8(
                                            (
                                                packed64
                                                >> Uint64(byte_idx * 8)
                                            )
                                            & Uint64(0xFF)
                                        )
                                        sA_u8[dst_flat] = byte_val

                                    outer_m_idx = row % Int32(32)
                                    inner_m_idx = row // Int32(32)
                                    inner_k_idx = sf_block % Int32(4)
                                    k_tile_idx = sf_block // Int32(4)
                                    sf_raw_idx = (
                                        k_tile_idx * Int32(32 * 4 * 4)
                                        + outer_m_idx * Int32(4 * 4)
                                        + inner_m_idx * Int32(4)
                                        + inner_k_idx
                                    )
                                    st_shared_u8(
                                        sfa_base_addr + sf_raw_idx, scale_byte
                                    )
                                quant_idx += Int32(
                                    self.num_mma_warps
                                    * self.num_threads_per_warp
                                )

                            cute.arch.fence_proxy("async.shared", space="cta")
                            self.epilog_sync_barrier.arrive_and_wait()

                        if fc1_half == 1:
                            # FC1 no longer needs sA/sSFA.  Publish the cached
                            # half0 Q1 payload into its original full-tile layout.
                            for cache_m in cutlass.range_constexpr(epi_rest_m):
                                rows_offset = Int32(cache_m) * Int32(
                                    self.epi_tile[0]
                                )
                                epi_rows = valid_rows - rows_offset
                                if epi_rows > Int32(self.epi_tile[0]):
                                    epi_rows = Int32(self.epi_tile[0])
                                if epi_rows < Int32(0):
                                    epi_rows = Int32(0)
                                quant_idx = Int32(tidx)
                                if quant_idx < epi_rows * sf_blocks_per_half:
                                    local_row = quant_idx // sf_blocks_per_half
                                    local_sf_block = (
                                        quant_idx - local_row * sf_blocks_per_half
                                    )
                                    row = rows_offset + local_row
                                    sf_block = local_sf_block
                                    packed64 = half0_packed_cache[cache_m]
                                    scale_byte = half0_scale_cache[cache_m]
                                    packed_base = sf_block << Int32(3)
                                    dst_pcol = row & Int32(63)
                                    xor_bits = (
                                        ((dst_pcol >> Int32(1)) & Int32(0x3))
                                        << Int32(4)
                                    )
                                    row_high = row >> Int32(6)
                                    for byte_idx in cutlass.range_constexpr(8):
                                        src_pcol = packed_base + Int32(byte_idx)
                                        dst_row = (
                                            (src_pcol ^ xor_bits) << Int32(1)
                                        ) + row_high
                                        dst_flat = dst_row * packed_cols + dst_pcol
                                        byte_val = Uint8(
                                            (
                                                packed64
                                                >> Uint64(byte_idx * 8)
                                            )
                                            & Uint64(0xFF)
                                        )
                                        sA_u8[dst_flat] = byte_val

                                    outer_m_idx = row % Int32(32)
                                    inner_m_idx = row // Int32(32)
                                    inner_k_idx = sf_block % Int32(4)
                                    k_tile_idx = sf_block // Int32(4)
                                    sf_raw_idx = (
                                        k_tile_idx * Int32(32 * 4 * 4)
                                        + outer_m_idx * Int32(4 * 4)
                                        + inner_m_idx * Int32(4)
                                        + inner_k_idx
                                    )
                                    st_shared_u8(
                                        sfa_base_addr + sf_raw_idx, scale_byte
                                    )

                            cute.arch.fence_proxy("async.shared", space="cta")
                            self.epilog_sync_barrier.arrive_and_wait()

"""


COMPACT_FC1_Q1_V1 = """                        if fc1_half == 0:
                            # Preserve the first N64 activation while half1 still
                            # consumes the aliased sA/sSFA FC1 input pipeline.
                            for mma_m in cutlass.range_constexpr(fc1_m_tiles):
                                for mma_n in cutlass.range_constexpr(fc1_n_tiles):
                                    gate_slice = fc1_tRS_rGate[
                                        (None, mma_m, mma_n)
                                    ]
                                    up_slice = fc1_tRS_rUp[
                                        (None, mma_m, mma_n)
                                    ]
                                    half0_slice = fc1_tRS_rHalf0Act[
                                        (None, mma_m, mma_n)
                                    ]
                                    for elem_idx in cutlass.range_constexpr(
                                        cute.size(half0_slice)
                                    ):
                                        g = alpha_value * gate_slice[elem_idx]
                                        u = alpha_value * up_slice[elem_idx]
                                        half0_slice[elem_idx] = gated_activation_f32(
                                            g,
                                            u,
                                            activation=self.activation,
                                            limit=self.swiglu_limit,
                                            alpha=self.swiglu_alpha,
                                            beta=self.swiglu_beta,
                                            fast_math=self.fast_math,
                                        )
                        else:
                            # Both N64 branches are now available. Build one full
                            # M64xN128 shared tile per M pass, then run the original
                            # full-N128 Q1 mapping.
                            for epi_m in cutlass.range_constexpr(epi_rest_m):
                                for mma_m_in_epi in cutlass.range_constexpr(
                                    fc1_m_per_epi
                                ):
                                    mma_m = (
                                        epi_m * fc1_m_per_epi + mma_m_in_epi
                                    )
                                    for mma_n in cutlass.range_constexpr(
                                        fc1_n_tiles
                                    ):
                                        half0_slice = fc1_tRS_rHalf0Act[
                                            (None, mma_m, mma_n)
                                        ]
                                        half0_vec = half0_slice.load()
                                        half0_vec = half0_vec.to(cutlass.BFloat16)
                                        fc1_tRS_rAct_out.store(half0_vec)
                                        cute.copy(
                                            fc1_tiled_copy_r2s,
                                            fc1_tRS_rAct_out,
                                            fc1_tRS_sD[
                                                (
                                                    None,
                                                    mma_m_in_epi,
                                                    mma_n,
                                                    0,
                                                )
                                            ],
                                        )

                                        gate_slice = fc1_tRS_rGate[
                                            (None, mma_m, mma_n)
                                        ]
                                        up_slice = fc1_tRS_rUp[
                                            (None, mma_m, mma_n)
                                        ]
                                        for elem_idx in cutlass.range_constexpr(
                                            cute.size(fc1_tRS_rAct)
                                        ):
                                            g = alpha_value * gate_slice[elem_idx]
                                            u = alpha_value * up_slice[elem_idx]
                                            fc1_tRS_rAct[elem_idx] = (
                                                gated_activation_f32(
                                                    g,
                                                    u,
                                                    activation=self.activation,
                                                    limit=self.swiglu_limit,
                                                    alpha=self.swiglu_alpha,
                                                    beta=self.swiglu_beta,
                                                    fast_math=self.fast_math,
                                                )
                                            )
                                        half1_vec = fc1_tRS_rAct.load()
                                        half1_vec = half1_vec.to(cutlass.BFloat16)
                                        fc1_tRS_rAct_out.store(half1_vec)
                                        cute.copy(
                                            fc1_tiled_copy_r2s,
                                            fc1_tRS_rAct_out,
                                            fc1_tRS_sD[
                                                (
                                                    None,
                                                    mma_m_in_epi,
                                                    fc1_n_tiles + mma_n,
                                                    0,
                                                )
                                            ],
                                        )

                                cute.arch.fence_proxy("async.shared", space="cta")
                                self.epilog_sync_barrier.arrive_and_wait()

                                rows_offset = Int32(epi_m) * Int32(
                                    self.epi_tile[0]
                                )
                                epi_rows = valid_rows - rows_offset
                                if epi_rows > Int32(self.epi_tile[0]):
                                    epi_rows = Int32(self.epi_tile[0])
                                if epi_rows < Int32(0):
                                    epi_rows = Int32(0)
                                quant_idx = Int32(tidx)
                                while quant_idx < epi_rows * sf_blocks_per_row_q1:
                                    local_row = quant_idx // sf_blocks_per_row_q1
                                    row = rows_offset + local_row
                                    sf_block = (
                                        quant_idx
                                        - local_row * sf_blocks_per_row_q1
                                    )
                                    block_start = sf_block * Int32(16)

                                    values = cute.make_rmem_tensor(
                                        (16,), cutlass.Float32
                                    )
                                    block_max = cutlass.Float32(0.0)
                                    for elem_idx in cutlass.range_constexpr(16):
                                        value = cutlass.Float32(
                                            sC[
                                                local_row,
                                                block_start + elem_idx,
                                                0,
                                            ]
                                        )
                                        values[elem_idx] = value
                                        block_max = fmax_f32(
                                            block_max, fabs_f32(value)
                                        )

                                    packed64 = Uint64(0)
                                    scale_byte = Uint8(0)
                                    if self.fast_math:
                                        packed64, scale_byte = (
                                            quantize_block_fp4_fast(
                                                values, block_max, gs_value
                                            )
                                        )
                                    else:
                                        packed64, scale_byte = quantize_block_fp4(
                                            values, block_max, gs_value
                                        )
                                    packed_base = sf_block << Int32(3)
                                    dst_pcol = row & Int32(63)
                                    xor_bits = (
                                        ((dst_pcol >> Int32(1)) & Int32(0x3))
                                        << Int32(4)
                                    )
                                    row_high = row >> Int32(6)
                                    for byte_idx in cutlass.range_constexpr(8):
                                        src_pcol = packed_base + Int32(byte_idx)
                                        dst_row = (
                                            (src_pcol ^ xor_bits) << Int32(1)
                                        ) + row_high
                                        dst_flat = dst_row * packed_cols + dst_pcol
                                        byte_val = Uint8(
                                            (
                                                packed64
                                                >> Uint64(byte_idx * 8)
                                            )
                                            & Uint64(0xFF)
                                        )
                                        sA_u8[dst_flat] = byte_val

                                    outer_m_idx = row % Int32(32)
                                    inner_m_idx = row // Int32(32)
                                    inner_k_idx = sf_block % Int32(4)
                                    k_tile_idx = sf_block // Int32(4)
                                    sf_raw_idx = (
                                        k_tile_idx * Int32(32 * 4 * 4)
                                        + outer_m_idx * Int32(4 * 4)
                                        + inner_m_idx * Int32(4)
                                        + inner_k_idx
                                    )
                                    st_shared_u8(
                                        sfa_base_addr + sf_raw_idx, scale_byte
                                    )
                                    quant_idx += Int32(
                                        self.num_mma_warps
                                        * self.num_threads_per_warp
                                    )

                                cute.arch.fence_proxy("async.shared", space="cta")
                                self.epilog_sync_barrier.arrive_and_wait()

"""


COMPACT_FC1_Q1_SHARED = COMPACT_FC1_Q1.replace(
    """                                    half0_packed_cache[epi_m] = packed64
                                    half0_scale_cache[epi_m] = scale_byte
""",
    """                                    cache_idx = (
                                        Int32(epi_m) * Int32(256) + Int32(tidx)
                                    )
                                    half0_packed_cache[cache_idx] = Int64(packed64)
                                    half0_scale_cache[cache_idx] = scale_byte
""",
).replace(
    """                                    packed64 = half0_packed_cache[cache_m]
                                    scale_byte = half0_scale_cache[cache_m]
""",
    """                                    cache_idx = (
                                        Int32(cache_m) * Int32(256) + Int32(tidx)
                                    )
                                    packed64 = Uint64(half0_packed_cache[cache_idx])
                                    scale_byte = half0_scale_cache[cache_idx]
""",
)


COMPACT_FC1_Q1_RAW = COMPACT_FC1_Q1.replace(
    """                                    half0_packed_cache[epi_m] = packed64
                                    half0_scale_cache[epi_m] = scale_byte
""",
    """                                    cache_idx = (
                                        Int32(epi_m) * Int32(256) + Int32(tidx)
                                    )
                                    _exp013_st_shared_u64(
                                        q1_half0_cache_addr + cache_idx * Int32(8),
                                        packed64,
                                    )
                                    st_shared_u8(
                                        q1_half0_cache_addr
                                        + Int32(512 * 8)
                                        + cache_idx,
                                        scale_byte,
                                    )
""",
).replace(
    """                                    packed64 = half0_packed_cache[cache_m]
                                    scale_byte = half0_scale_cache[cache_m]
""",
    """                                    cache_idx = (
                                        Int32(cache_m) * Int32(256) + Int32(tidx)
                                    )
                                    packed64 = _exp013_ld_shared_u64(
                                        q1_half0_cache_addr + cache_idx * Int32(8)
                                    )
                                    scale_byte = _exp013_ld_shared_u8(
                                        q1_half0_cache_addr
                                        + Int32(512 * 8)
                                        + cache_idx
                                    )
""",
)


RAW_SHARED_HELPERS = """_FC2_TILE_RECIP_GS_NUM = 6.0 * 448.0


@dsl_user_op
def _exp013_st_shared_u64(addr, value, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Uint64(value).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.u64 [$0], $1;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _exp013_ld_shared_u64(addr, *, loc=None, ip=None):
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.u64 $0, [$1];",
            "=l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _exp013_ld_shared_u8(addr, *, loc=None, ip=None):
    return Uint8(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.u8 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
"""


COMPACT_SCATTER = """                            # Compact M64 pass, but preserve the baseline's
                            # four-warp scatter participation. W4-W7 skip data
                            # accesses and still join both epilogue barriers.
                            epi_rows = valid_rows - rows_offset
                            if epi_rows > Int32(self.epi_tile[0]):
                                epi_rows = Int32(self.epi_tile[0])
                            if epi_rows < Int32(0):
                                epi_rows = Int32(0)

                            if warp_in_tile < Int32(4):
                                tile_vec_cols = Int32(self.epi_tile[1]) // Int32(8)
                                vec_idx = Int32(tidx)
                                while vec_idx < epi_rows * tile_vec_cols:
                                    local_row = vec_idx // tile_vec_cols
                                    local_vec_col = vec_idx - local_row * tile_vec_cols
                                    local_col = local_vec_col * Int32(8)
                                    global_col = tile_n_base_cur + local_col
                                    cached_row = rows_offset + local_row
                                    tok = ld_shared_i32_relaxed(
                                        scatter_tok_base_addr + cached_row * Int32(4)
                                    )
                                    wv = ld_shared_f32(
                                        scatter_weight_base_addr + cached_row * Int32(4)
                                    )
                                    sc_v0 = cutlass.Float32(
                                        sC[local_row, local_col, epi_buffer]
                                    )
                                    sc_v1 = cutlass.Float32(
                                        sC[local_row, local_col + Int32(1), epi_buffer]
                                    )
                                    sc_v2 = cutlass.Float32(
                                        sC[local_row, local_col + Int32(2), epi_buffer]
                                    )
                                    sc_v3 = cutlass.Float32(
                                        sC[local_row, local_col + Int32(3), epi_buffer]
                                    )
                                    sc_v4 = cutlass.Float32(
                                        sC[local_row, local_col + Int32(4), epi_buffer]
                                    )
                                    sc_v5 = cutlass.Float32(
                                        sC[local_row, local_col + Int32(5), epi_buffer]
                                    )
                                    sc_v6 = cutlass.Float32(
                                        sC[local_row, local_col + Int32(6), epi_buffer]
                                    )
                                    sc_v7 = cutlass.Float32(
                                        sC[local_row, local_col + Int32(7), epi_buffer]
                                    )
                                    scatter_add_v4_bf16x2(
                                        get_ptr_as_int64(
                                            scatter_output,
                                            tok * scatter_N + global_col,
                                        ),
                                        wv * sc_v0,
                                        wv * sc_v1,
                                        wv * sc_v2,
                                        wv * sc_v3,
                                        wv * sc_v4,
                                        wv * sc_v5,
                                        wv * sc_v6,
                                        wv * sc_v7,
                                    )
                                    vec_idx += Int32(4 * self.num_threads_per_warp)

"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, got {count}")
    return text.replace(old, new)


def replace_span(text: str, start: str, end: str, new: str, label: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError(
            f"{label}: non-unique span anchors ({text.count(start)}, {text.count(end)})"
        )
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[:begin] + new + text[finish:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        choices=("v0", "v1", "v2", "v3", "v4", "v5"),
        default="v2",
    )
    args = parser.parse_args()
    version = args.version
    out_dir = ROOT / f"results/overlays/compact_epi_m64_stage2_{version}"
    out = out_dir / "moe_dynamic_kernel.py"
    packed_bridge = version in ("v0", "v3", "v4", "v5")
    if version == "v4":
        q1_setup = Q1_SETUP_SHARED
        compact_fc1_q1 = COMPACT_FC1_Q1_SHARED
    elif version == "v5":
        q1_setup = Q1_SETUP_RAW
        compact_fc1_q1 = COMPACT_FC1_Q1_RAW
    else:
        q1_setup = Q1_SETUP_NEW if packed_bridge else Q1_SETUP_V1
        compact_fc1_q1 = COMPACT_FC1_Q1 if packed_bridge else COMPACT_FC1_Q1_V1

    baseline_bytes = BASELINE.read_bytes()
    observed = sha256(baseline_bytes)
    if observed != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(f"baseline hash drift: {observed}")
    baseline = baseline_bytes.decode()
    candidate = replace_once(
        baseline,
        "        self.epi_tile = (mma_tiler_mn[0], mma_tiler_mn[1])\n",
        "        self.epi_tile = (64, mma_tiler_mn[1])\n",
        "epi_tile",
    )
    if version in ("v2", "v3", "v4", "v5"):
        candidate = replace_once(
            candidate,
            """        tCsC_for_shape = thr_mma.partition_C(sC[None, None, 0])
        epi_m_scale = self.tile_shape_mnk[0] // self.epi_tile[0]
        sub_shape = tCsC_for_shape.shape[:3]
        acc_shape = (sub_shape[0], sub_shape[1] * epi_m_scale, sub_shape[2])
""",
            """        # The accumulator belongs to the full M128xN128 OMMA tile,
        # independent of the smaller M64 epilogue staging window.  Reconstructing
        # it from partition_C(sC[M64]) loses the 8-warp M ownership mapping.
        acc_shape = tiled_mma.partition_shape_C(self.tile_shape_mnk[:2])
""",
            "full FC2 accumulator layout",
        )
    if version == "v4":
        candidate = replace_once(
            candidate,
            """            sSFB_up: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(fc1_sfb_smem_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
""",
            """            sSFB_up: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(fc1_sfb_smem_staged)
                ],
                self.buffer_align_bytes,
            ]
            # 512 half0 FP4 blocks: 8 packed bytes + one scale byte each.
            # This 4.5 KiB bridge replaces a cross-half register lifetime while
            # retaining most of the 16 KiB saved by the compact M64 sC tile.
            q1_half0_packed_cache: cute.struct.MemRange[
                cutlass.Int64, self.tile_shape_mnk[0] * 4
            ]
            q1_half0_scale_cache: cute.struct.MemRange[
                cutlass.Uint8, self.tile_shape_mnk[0] * 4
            ]
            sC: cute.struct.Align[
""",
            "shared Q1 bridge storage",
        )
    if version == "v5":
        candidate = replace_once(
            candidate,
            "_FC2_TILE_RECIP_GS_NUM = 6.0 * 448.0",
            RAW_SHARED_HELPERS,
            "raw shared helpers",
        )
        candidate = replace_once(
            candidate,
            """            sSFB_up: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(fc1_sfb_smem_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
""",
            """            sSFB_up: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(fc1_sfb_smem_staged)
                ],
                self.buffer_align_bytes,
            ]
            q1_half0_cache: cute.struct.MemRange[
                cutlass.Uint8, self.tile_shape_mnk[0] * 4 * 9
            ]
            sC: cute.struct.Align[
""",
            "raw shared Q1 bridge storage",
        )
    candidate = replace_once(candidate, Q1_SETUP_OLD, q1_setup, "Q1 setup")
    candidate = replace_span(
        candidate,
        "                        # Consume this Gate/Up pair immediately.",
        "                    # PHASE B: Sweep ALL FC2 output tiles using cached sA",
        compact_fc1_q1
        + "                    # ============================================================\n",
        "FC1/Q1 compact schedule",
    )
    candidate = replace_span(
        candidate,
        "                            # Per-warp scatter: each warp scatters its own quadrant",
        "                            # Post-scatter barrier: needed to ensure all warps",
        COMPACT_SCATTER,
        "compact scatter",
    )

    invariants = {
        "stage2_cap_unchanged": candidate.count(
            "self.ab_stage = max(1, min(self.ab_stage, 2))"
        )
        == baseline.count("self.ab_stage = max(1, min(self.ab_stage, 2))")
        == 1,
        "eight_math_warps": candidate.count("self.num_mma_warps = 8") == 1,
        "atom_layout_4x2": candidate.count("cute.make_layout((4, 2, 1))") == 1,
        "gemm_call_count_unchanged": candidate.count("cute.gemm(")
        == baseline.count("cute.gemm("),
        "scatter_call_count_unchanged": candidate.count("scatter_add_v4_bf16x2(")
        == baseline.count("scatter_add_v4_bf16x2("),
        "four_warp_scatter_stride": "4 * self.num_threads_per_warp" in candidate,
        "half0_register_cache": (
            (
                "q1_half0_cache" in candidate
                if version == "v5"
                else all(
                    token in candidate
                    for token in ("half0_packed_cache", "half0_scale_cache")
                )
            )
            if packed_bridge
            else all(
                token in candidate for token in ("half0_act_acc", "fc1_tRS_rHalf0Act")
            )
        ),
        "full_fc2_accumulator_layout": (
            "acc_shape = tiled_mma.partition_shape_C(self.tile_shape_mnk[:2])"
            in candidate
            if version in ("v2", "v3", "v4", "v5")
            else True
        ),
        "shared_q1_bridge": (
            all(
                token in candidate
                for token in (
                    "q1_half0_packed_cache",
                    "q1_half0_scale_cache",
                )
            )
            if version == "v4"
            else True
        ),
        "raw_shared_q1_bridge": (
            all(
                token in candidate
                for token in (
                    "q1_half0_cache",
                    "_exp013_st_shared_u64",
                    "_exp013_ld_shared_u64",
                )
            )
            if version == "v5"
            else True
        ),
    }
    if not all(invariants.values()):
        raise RuntimeError(f"source invariants failed: {invariants}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(candidate, encoding="utf-8")
    diff = "".join(
        difflib.unified_diff(
            baseline.splitlines(keepends=True),
            candidate.splitlines(keepends=True),
            fromfile="exp008_v1/moe_dynamic_kernel.py",
            tofile=f"exp013_compact_epi_m64_stage2_{version}/moe_dynamic_kernel.py",
        )
    )
    (out_dir / "candidate.diff").write_text(diff, encoding="utf-8")
    payload = {
        "schema": "exp013.kernel-overlay-identity.v2",
        "version": version,
        "baseline": str(BASELINE),
        "baseline_sha256": observed,
        "candidate": str(out),
        "candidate_sha256": sha256(candidate.encode()),
        "invariants": invariants,
        "changes": [
            "M64xN128 epilogue storage",
            (
                "four M64xN64 Q1 rectangles with half0 packed register bridge"
                if packed_bridge
                else "two full M64xN128 Q1 passes with half0 activation register bridge"
            ),
            "four-warp M64xN128 scatter ownership",
            *(
                ["full M128xN128 tiled-MMA accumulator layout independent of sC"]
                if version in ("v2", "v3", "v4", "v5")
                else []
            ),
            *(
                ["4.5 KiB shared half0 Q1 bridge instead of long-lived registers"]
                if version == "v4"
                else []
            ),
            *(["raw 4.5 KiB shared half0 Q1 bridge"] if version == "v5" else []),
        ],
    }
    (out_dir / "identity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
