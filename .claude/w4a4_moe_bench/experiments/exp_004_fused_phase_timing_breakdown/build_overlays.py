#!/usr/bin/env python3
"""Build immutable kernel + dispatch overlays for exp_004.

The production tree is read only.  Every edit is an exact, counted transform
anchored to the audited production SHA.  ``measurement_no_marker`` and
``probe_candidate`` share identical timing-buffer plumbing; their only source
difference is the compile-time probe flag.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any

from exp004_common import (
    ALL_ARMS,
    CONSUMER_INTERVALS,
    CONSUMER_WARPS,
    DISPATCH_RELATIVE_PATH,
    EDGES,
    EXPECTED_DISPATCH_SHA256,
    EXPECTED_KERNEL_SHA256,
    EXPECTED_WRAPPER_SHA256,
    KERNEL_RELATIVE_PATH,
    MEASUREMENT_CONTROL,
    NORMAL,
    PROBE,
    W4_INTERVALS,
    WRAPPER_RELATIVE_PATH,
    file_sha256,
    write_json,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _replace_exact(
    text: str,
    old: str,
    new: str,
    *,
    label: str,
    expected: int = 1,
) -> str:
    count = text.count(old)
    if count != expected:
        raise ValueError(f"{label}: expected {expected} matches, found {count}")
    return text.replace(old, new)


KERNEL_CONSTANTS = f"""_FC2_TILE_RECIP_GS_NUM = 6.0 * 448.0

# exp_004 experiment-owned phase-timing ABI.  These constants do not exist in
# the production file; build_overlays.py removes the entire patch to recover
# the byte-identical source.
_EXP004_CONSUMER_WARPS = {CONSUMER_WARPS}
_EXP004_CONSUMER_INTERVALS = {CONSUMER_INTERVALS}
_EXP004_W4_INTERVALS = {W4_INTERVALS}
_EXP004_EDGES = {EDGES}


@dsl_user_op
def _exp004_read_clock64(*, loc=None, ip=None):
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %clock64;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
"""


KERNEL_METHODS = """    @cute.jit
    def _exp004_store_consumer_tick(
        self,
        timing_ticks: cute.Tensor,
        task_slot: Int32,
        warp_id: Int32,
        interval: Int32,
        edge: Int32,
        tick: Uint64,
    ):
        index = (
            (
                (task_slot * Int32(_EXP004_CONSUMER_WARPS) + warp_id)
                * Int32(_EXP004_CONSUMER_INTERVALS)
                + interval
            )
            * Int32(_EXP004_EDGES)
            + edge
        )
        timing_ticks[index] = tick

    @cute.jit
    def _exp004_store_w4_tick(
        self,
        timing_ticks: cute.Tensor,
        task_capacity: Int32,
        task_slot: Int32,
        interval: Int32,
        edge: Int32,
        tick: Uint64,
    ):
        base = task_capacity * Int32(
            _EXP004_CONSUMER_WARPS * _EXP004_CONSUMER_INTERVALS * _EXP004_EDGES
        )
        index = base + (
            (task_slot * Int32(_EXP004_W4_INTERVALS) + interval)
            * Int32(_EXP004_EDGES)
            + edge
        )
        timing_ticks[index] = tick

"""


def _consumer_store(interval: str, edge: int, *, indent: int) -> str:
    pad = " " * indent
    return (
        f"{pad}self._exp004_store_consumer_tick(\n"
        f"{pad}    timing_ticks, task_slot_probe, Int32(warp_idx),\n"
        f"{pad}    Int32({interval}), Int32({edge}), exp004_tick,\n"
        f"{pad})\n"
    )


def _w4_store(interval: int, edge: int, *, indent: int) -> str:
    pad = " " * indent
    return (
        f"{pad}self._exp004_store_w4_tick(\n"
        f"{pad}    timing_ticks, exp004_task_capacity, task_slot_probe,\n"
        f"{pad}    Int32({interval}), Int32({edge}), exp004_tick,\n"
        f"{pad})\n"
    )


def _instrument_kernel(source: str) -> str:
    text = _replace_exact(
        source,
        "_FC2_TILE_RECIP_GS_NUM = 6.0 * 448.0\n",
        KERNEL_CONSTANTS,
        label="kernel constants + clock helper",
    )
    text = _replace_exact(
        text,
        "        share_input_across_experts: bool = False,\n    ):\n",
        "        share_input_across_experts: bool = False,\n"
        "        phase_probe_enabled: bool = False,\n"
        "    ):\n",
        label="kernel constructor flag",
    )
    text = _replace_exact(
        text,
        "        self.share_input_across_experts = share_input_across_experts\n",
        "        self.share_input_across_experts = share_input_across_experts\n"
        "        self.phase_probe_enabled = bool(phase_probe_enabled)\n",
        label="kernel constructor flag assignment",
    )
    text = _replace_exact(
        text,
        "    def _thrfrg_SFA(self, sfa_tensor, tiled_mma):\n",
        KERNEL_METHODS + "    def _thrfrg_SFA(self, sfa_tensor, tiled_mma):\n",
        label="kernel probe stores",
    )
    text = _replace_exact(
        text,
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "        max_active_clusters: cutlass.Constexpr,\n",
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "        timing_ticks: cute.Tensor,\n"
        "        task_cta_z: cute.Tensor,\n"
        "        max_active_clusters: cutlass.Constexpr,\n",
        label="kernel host-call ABI",
    )
    text = _replace_exact(
        text,
        "            token_map,\n            token_weights,\n        ).launch(\n",
        "            token_map,\n"
        "            token_weights,\n"
        "            timing_ticks,\n"
        "            task_cta_z,\n"
        "        ).launch(\n",
        label="kernel launch args",
    )
    text = _replace_exact(
        text,
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "    ):\n"
        '        """Kernel entry point."""\n',
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "        timing_ticks: cute.Tensor,\n"
        "        task_cta_z: cute.Tensor,\n"
        "    ):\n"
        '        """Kernel entry point."""\n',
        label="device kernel ABI",
    )
    text = _replace_exact(
        text,
        "        full_tile_publish_enabled = Int32(0)\n",
        "        full_tile_publish_enabled = Int32(0)\n"
        "        exp004_task_capacity = Int32(task_cta_z.shape[0])\n",
        label="runtime probe capacity",
    )

    consumer_entry = """            elif warp_idx < self.num_mma_warps:
                task_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(8))
"""
    consumer_probe = (
        """            elif warp_idx < self.num_mma_warps:
                if cutlass.const_expr(self.phase_probe_enabled):
                    task_slot_probe = _ld_shared_i32(ctrl_base_addr + Int32(28))
                    if lane_id == Int32(0):
                        if warp_idx == Int32(0):
                            task_cta_z[task_slot_probe] = Int32(bidz)
                        exp004_tick = _exp004_read_clock64()
"""
        + _consumer_store("0", 0, indent=24)
        + """                task_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(8))
"""
    )
    text = _replace_exact(
        text, consumer_entry, consumer_probe, label="consumer task-envelope start"
    )

    text = _replace_exact(
        text,
        "                    gate_acc.fill(0.0)\n",
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        "                            exp004_tick = _exp004_read_clock64()\n"
        + _consumer_store("1", 0, indent=28)
        + "                    gate_acc.fill(0.0)\n",
        label="gate start",
    )
    text = _replace_exact(
        text,
        "                    self.pass_gate_barrier.arrive_unaligned()\n\n"
        "                    if cutlass.const_expr(self.is_gated):\n",
        "                    self.pass_gate_barrier.arrive_unaligned()\n"
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        "                            exp004_tick = _exp004_read_clock64()\n"
        + _consumer_store("1", 1, indent=28)
        + _consumer_store("2", 0, indent=28)
        + "\n                    if cutlass.const_expr(self.is_gated):\n",
        label="gate end and up start",
    )
    text = _replace_exact(
        text,
        "                    # Activation + quant into sA\n",
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        "                            exp004_tick = _exp004_read_clock64()\n"
        + _consumer_store("2", 1, indent=28)
        + _consumer_store("3", 0, indent=28)
        + "                    # Activation + quant into sA\n",
        label="up end and activation start",
    )
    text = _replace_exact(
        text,
        "                    self.epilog_sync_barrier.arrive_and_wait()\n\n"
        "                    # ============================================================\n"
        "                    # PHASE B: Sweep ALL FC2 output tiles using cached sA\n",
        "                    self.epilog_sync_barrier.arrive_and_wait()\n"
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        "                            exp004_tick = _exp004_read_clock64()\n"
        + _consumer_store("3", 1, indent=28)
        + "\n                    # ============================================================\n"
        "                    # PHASE B: Sweep ALL FC2 output tiles using cached sA\n",
        label="activation end",
    )
    text = _replace_exact(
        text,
        "                    # Hoist A-side register loads: sA is constant across all\n",
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        "                            exp004_tick = _exp004_read_clock64()\n"
        + _consumer_store("4", 0, indent=28)
        + "                    # Hoist A-side register loads: sA is constant across all\n",
        label="FC2 setup start",
    )
    text = _replace_exact(
        text,
        "                    phase2_cons_state.reset_count()\n"
        "                    for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]\n"
        "                        phase2_peek = phase2_pipeline.consumer_try_wait(\n",
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        "                            exp004_tick = _exp004_read_clock64()\n"
        + _consumer_store("4", 1, indent=28)
        + "                    phase2_cons_state.reset_count()\n"
        "                    for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]\n"
        "                        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                            if lane_id == Int32(0):\n"
        "                                exp004_tick = _exp004_read_clock64()\n"
        + _consumer_store("Int32(5) + output_tile_idx", 0, indent=32)
        + "                        phase2_peek = phase2_pipeline.consumer_try_wait(\n",
        label="FC2 setup end and per-tile GEMM start",
    )
    text = _replace_exact(
        text,
        "                        # Scatter using precomputed metadata (no redundant gmem loads)\n",
        "                        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                            if lane_id == Int32(0):\n"
        "                                exp004_tick = _exp004_read_clock64()\n"
        + _consumer_store("Int32(5) + output_tile_idx", 1, indent=32)
        + _consumer_store("Int32(21) + output_tile_idx", 0, indent=32)
        + "                        # Scatter using precomputed metadata (no redundant gmem loads)\n",
        label="per-tile GEMM end and epilogue start",
    )
    text = _replace_exact(
        text,
        "                            self.epilog_sync_barrier.arrive_and_wait()\n\n"
        "                    # Signal that FC2/scatter no longer needs sA, so the DMA\n",
        "                            self.epilog_sync_barrier.arrive_and_wait()\n"
        "                        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                            if lane_id == Int32(0):\n"
        "                                exp004_tick = _exp004_read_clock64()\n"
        + _consumer_store("Int32(21) + output_tile_idx", 1, indent=32)
        + "\n                    # Signal that FC2/scatter no longer needs sA, so the DMA\n",
        label="per-tile epilogue end",
    )
    text = _replace_exact(
        text,
        "                    self.pass_final_barrier.arrive_unaligned()\n"
        "                    slice_idx += Int32(1)\n",
        "                    self.pass_final_barrier.arrive_unaligned()\n"
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        "                            exp004_tick = _exp004_read_clock64()\n"
        + _consumer_store("0", 1, indent=28)
        + "                    slice_idx += Int32(1)\n",
        label="consumer task-envelope end",
    )

    w4_entry = """            elif warp_idx == self.tma_load_warp_id:
                task_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(8))
"""
    w4_probe = """            elif warp_idx == self.tma_load_warp_id:
                if cutlass.const_expr(self.phase_probe_enabled):
                    task_slot_probe = _ld_shared_i32(ctrl_base_addr + Int32(28))
                task_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(8))
"""
    text = _replace_exact(text, w4_entry, w4_probe, label="W4 task slot")

    text = _replace_exact(
        text,
        "                    # ---- FC1 gate/only pass ----\n"
        "                    prod_state.reset_count()\n",
        "                    # ---- FC1 gate/only pass ----\n"
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        "                            exp004_tick = _exp004_read_clock64()\n"
        + _w4_store(0, 0, indent=28)
        + "                    prod_state.reset_count()\n",
        label="W4 Gate TMA start",
    )
    text = _replace_exact(
        text,
        "                    # Wait for the MMA warps to finish the FC1 gate/only pass\n"
        "                    # before reusing sA/sB/sSFA/sSFB for up/FC2 staging.\n"
        "                    self.pass_gate_barrier.wait_unaligned()\n",
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        "                            exp004_tick = _exp004_read_clock64()\n"
        + _w4_store(0, 1, indent=28)
        + _w4_store(1, 0, indent=28)
        + "                    # Wait for the MMA warps to finish the FC1 gate/only pass\n"
        "                    # before reusing sA/sB/sSFA/sSFB for up/FC2 staging.\n"
        "                    self.pass_gate_barrier.wait_unaligned()\n"
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        "                            exp004_tick = _exp004_read_clock64()\n"
        + _w4_store(1, 1, indent=28),
        label="W4 Gate TMA end and Gate wait",
    )
    text = _replace_exact(
        text,
        "                        # ---- FC1 up pass ----\n"
        "                        up_prod_state.reset_count()\n",
        "                        # ---- FC1 up pass ----\n"
        "                        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                            if lane_id == Int32(0):\n"
        "                                exp004_tick = _exp004_read_clock64()\n"
        + _w4_store(2, 0, indent=32)
        + "                        up_prod_state.reset_count()\n",
        label="W4 Up TMA start",
    )
    text = _replace_exact(
        text,
        "                    # ---- FC2 B_down loads: continuous pipeline ----\n",
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        "                            exp004_tick = _exp004_read_clock64()\n"
        + _w4_store(2, 1, indent=28)
        + "                    # ---- FC2 B_down loads: continuous pipeline ----\n",
        label="W4 Up TMA end",
    )
    text = _replace_exact(
        text,
        "                    phase2_prod_state.reset_count()\n"
        "                    for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]\n",
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        "                            exp004_tick = _exp004_read_clock64()\n"
        + _w4_store(3, 0, indent=28)
        + "                    phase2_prod_state.reset_count()\n"
        "                    for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]\n",
        label="W4 Down TMA start",
    )
    text = _replace_exact(
        text,
        "                    # Ensure MMA warps finish FC2/scatter before DMA starts the\n"
        "                    # next slice/task's FC1 loads into shared A buffers.\n"
        "                    self.pass_final_barrier.wait_unaligned()\n",
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        "                            exp004_tick = _exp004_read_clock64()\n"
        + _w4_store(3, 1, indent=28)
        + _w4_store(4, 0, indent=28)
        + "                    # Ensure MMA warps finish FC2/scatter before DMA starts the\n"
        "                    # next slice/task's FC1 loads into shared A buffers.\n"
        "                    self.pass_final_barrier.wait_unaligned()\n"
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        "                            exp004_tick = _exp004_read_clock64()\n"
        + _w4_store(4, 1, indent=28),
        label="W4 Down TMA end and final wait",
    )
    return text


def _instrument_dispatch(source: str, *, enabled: bool) -> str:
    text = _replace_exact(
        source,
        "_DYNAMIC_SLICE_CHUNK = 1\n",
        "_DYNAMIC_SLICE_CHUNK = 1\n"
        f"_EXP004_PHASE_PROBE_ENABLED = {enabled!r}\n"
        f"_EXP004_TIMING_TICKS_PER_TASK = {CONSUMER_WARPS * CONSUMER_INTERVALS * EDGES + W4_INTERVALS * EDGES}\n",
        label="dispatch probe constants",
    )
    text = _replace_exact(
        text,
        "    task_valid_rows: torch.Tensor\n"
        "    tile_write_count: torch.Tensor\n\n"
        "    # Views\n",
        "    task_valid_rows: torch.Tensor\n"
        "    tile_write_count: torch.Tensor\n"
        "    exp004_timing_ticks: torch.Tensor\n"
        "    exp004_task_cta_z: torch.Tensor\n\n"
        "    # Views\n",
        label="workspace probe fields",
    )
    text = _replace_exact(
        text,
        "        tile_write_count=torch.zeros(physical_tiles, dtype=torch.int32, device=device),\n"
        "    )\n",
        "        tile_write_count=torch.zeros(physical_tiles, dtype=torch.int32, device=device),\n"
        "        exp004_timing_ticks=torch.full(\n"
        "            (max_tasks * _EXP004_TIMING_TICKS_PER_TASK,),\n"
        "            -1, dtype=torch.int64, device=device,\n"
        "        ),\n"
        "        exp004_task_cta_z=torch.full(\n"
        "            (max_tasks,), -1, dtype=torch.int32, device=device\n"
        "        ),\n"
        "    )\n",
        label="workspace probe allocation",
    )
    text = _replace_exact(
        text,
        "        tile_write_count_ptr: cute.Pointer,\n        b_w13: cute.Tensor,\n",
        "        tile_write_count_ptr: cute.Pointer,\n"
        "        exp004_timing_ticks_ptr: cute.Pointer,\n"
        "        exp004_task_cta_z_ptr: cute.Pointer,\n"
        "        b_w13: cute.Tensor,\n",
        label="dynamic launch probe pointers",
    )
    text = _replace_exact(
        text,
        "        tile_write_count = cute.make_tensor(\n"
        "            tile_write_count_ptr,\n"
        "            layout=cute.make_layout((max_phys_tiles,), stride=(1,)),\n"
        "        )\n"
        "        self._kernel(\n",
        "        tile_write_count = cute.make_tensor(\n"
        "            tile_write_count_ptr,\n"
        "            layout=cute.make_layout((max_phys_tiles,), stride=(1,)),\n"
        "        )\n"
        "        exp004_timing_ticks = cute.make_tensor(\n"
        "            exp004_timing_ticks_ptr,\n"
        "            layout=cute.make_layout(\n"
        "                (max_tasks * _EXP004_TIMING_TICKS_PER_TASK,), stride=(1,)\n"
        "            ),\n"
        "        )\n"
        "        exp004_task_cta_z = cute.make_tensor(\n"
        "            exp004_task_cta_z_ptr,\n"
        "            layout=cute.make_layout((max_tasks,), stride=(1,)),\n"
        "        )\n"
        "        self._kernel(\n",
        label="dynamic launch probe tensor views",
    )
    text = _replace_exact(
        text,
        "            token_map,\n"
        "            token_weights_t,\n"
        "            max_active_clusters=max_active_clusters,\n",
        "            token_map,\n"
        "            token_weights_t,\n"
        "            exp004_timing_ticks,\n"
        "            exp004_task_cta_z,\n"
        "            max_active_clusters=max_active_clusters,\n",
        label="dynamic launch kernel probe args",
    )
    text = _replace_exact(
        text,
        "        share_input_across_experts=share_input_across_experts,\n"
        "    )\n"
        "    launch = _DynamicMoELaunch(\n",
        "        share_input_across_experts=share_input_across_experts,\n"
        "        phase_probe_enabled=_EXP004_PHASE_PROBE_ENABLED,\n"
        "    )\n"
        "    launch = _DynamicMoELaunch(\n",
        label="kernel probe flag",
    )
    text = _replace_exact(
        text,
        "    tile_write_count_fake = make_ptr(\n"
        "        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4\n"
        "    )\n\n"
        "    b_w13_fake = cute.runtime.make_fake_compact_tensor(\n",
        "    tile_write_count_fake = make_ptr(\n"
        "        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4\n"
        "    )\n"
        "    exp004_timing_ticks_fake = make_ptr(\n"
        "        cutlass.Uint64, 8, cute.AddressSpace.gmem, assumed_align=8\n"
        "    )\n"
        "    exp004_task_cta_z_fake = make_ptr(\n"
        "        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4\n"
        "    )\n\n"
        "    b_w13_fake = cute.runtime.make_fake_compact_tensor(\n",
        label="compile fake probe pointers",
    )
    text = _replace_exact(
        text,
        "        tile_write_count_fake,\n        b_w13_fake,\n",
        "        tile_write_count_fake,\n"
        "        exp004_timing_ticks_fake,\n"
        "        exp004_task_cta_z_fake,\n"
        "        b_w13_fake,\n",
        label="compile probe args",
    )
    text = _replace_exact(
        text,
        "        workspace.tile_write_count.data_ptr(),\n        weights.w13_fp4,\n",
        "        workspace.tile_write_count.data_ptr(),\n"
        "        workspace.exp004_timing_ticks.data_ptr(),\n"
        "        workspace.exp004_task_cta_z.data_ptr(),\n"
        "        weights.w13_fp4,\n",
        label="runtime probe args",
    )
    text = _replace_exact(
        text,
        "        share_input_across_experts,\n"
        "    )\n"
        "    cached = _DYNAMIC_KERNEL_CACHE.get(cache_key)\n",
        "        share_input_across_experts,\n"
        "        _EXP004_PHASE_PROBE_ENABLED,\n"
        "    )\n"
        "    cached = _DYNAMIC_KERNEL_CACHE.get(cache_key)\n",
        label="cache-key probe identity",
    )
    return text


def _diff(left: str, right: str, *, left_name: str, right_name: str) -> str:
    return "".join(
        difflib.unified_diff(
            left.splitlines(keepends=True),
            right.splitlines(keepends=True),
            fromfile=left_name,
            tofile=right_name,
        )
    )


def build_overlays(repo: Path, output_dir: Path) -> dict[str, Any]:
    repo = repo.resolve()
    output_dir = output_dir.resolve()
    kernel_path = repo / KERNEL_RELATIVE_PATH
    dispatch_path = repo / DISPATCH_RELATIVE_PATH
    wrapper_path = repo / WRAPPER_RELATIVE_PATH
    identities = {
        kernel_path: EXPECTED_KERNEL_SHA256,
        dispatch_path: EXPECTED_DISPATCH_SHA256,
        wrapper_path: EXPECTED_WRAPPER_SHA256,
    }
    for path, expected in identities.items():
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"production source drift: {path} {actual} != {expected}")
    if output_dir.exists():
        raise ValueError(f"immutable overlay output already exists: {output_dir}")

    kernel = kernel_path.read_text()
    dispatch = dispatch_path.read_text()
    instrumented_kernel = _instrument_kernel(kernel)
    arms = {
        NORMAL: (kernel, dispatch),
        MEASUREMENT_CONTROL: (
            instrumented_kernel,
            _instrument_dispatch(dispatch, enabled=False),
        ),
        PROBE: (instrumented_kernel, _instrument_dispatch(dispatch, enabled=True)),
    }
    output_dir.mkdir(parents=True)
    manifest_arms: dict[str, Any] = {}
    for arm in ALL_ARMS:
        arm_dir = output_dir / arm
        arm_dir.mkdir()
        kernel_text, dispatch_text = arms[arm]
        ast.parse(kernel_text, filename=f"{arm}/moe_dynamic_kernel.py")
        ast.parse(dispatch_text, filename=f"{arm}/moe_dispatch.py")
        kernel_out = arm_dir / "moe_dynamic_kernel.py"
        dispatch_out = arm_dir / "moe_dispatch.py"
        kernel_out.write_text(kernel_text)
        dispatch_out.write_text(dispatch_text)
        kernel_diff = _diff(
            kernel,
            kernel_text,
            left_name="production/moe_dynamic_kernel.py",
            right_name=f"{arm}/moe_dynamic_kernel.py",
        )
        dispatch_diff = _diff(
            dispatch,
            dispatch_text,
            left_name="production/moe_dispatch.py",
            right_name=f"{arm}/moe_dispatch.py",
        )
        (arm_dir / "moe_dynamic_kernel.diff").write_text(kernel_diff)
        (arm_dir / "moe_dispatch.diff").write_text(dispatch_diff)
        manifest_arms[arm] = {
            "probe_enabled": arm == PROBE,
            "kernel_sha256": _sha256(kernel_text.encode()),
            "dispatch_sha256": _sha256(dispatch_text.encode()),
            "kernel_diff_sha256": _sha256(kernel_diff.encode()),
            "dispatch_diff_sha256": _sha256(dispatch_diff.encode()),
            "kernel_byte_identical_to_production": kernel_text == kernel,
            "dispatch_byte_identical_to_production": dispatch_text == dispatch,
        }

    # The control/candidate kernel source is intentionally identical; only the
    # dispatch compile-time flag selects whether its probe blocks survive.
    if arms[MEASUREMENT_CONTROL][0] != arms[PROBE][0]:
        raise AssertionError("measurement/probe kernel overlays drifted")
    control_dispatch = arms[MEASUREMENT_CONTROL][1]
    probe_dispatch = arms[PROBE][1]
    expected_delta = control_dispatch.replace(
        "_EXP004_PHASE_PROBE_ENABLED = False",
        "_EXP004_PHASE_PROBE_ENABLED = True",
    )
    if expected_delta != probe_dispatch:
        raise AssertionError("measurement/probe dispatch differs beyond the probe flag")

    manifest = {
        "schema": "exp004.overlays.v1",
        "production": {
            "kernel_sha256": EXPECTED_KERNEL_SHA256,
            "dispatch_sha256": EXPECTED_DISPATCH_SHA256,
            "wrapper_sha256": EXPECTED_WRAPPER_SHA256,
        },
        "probe_abi": {
            "consumer_intervals": CONSUMER_INTERVALS,
            "consumer_warps": CONSUMER_WARPS,
            "w4_intervals": W4_INTERVALS,
            "edges": EDGES,
            "ticks_per_task": CONSUMER_WARPS * CONSUMER_INTERVALS * EDGES
            + W4_INTERVALS * EDGES,
        },
        "arms": manifest_arms,
    }
    write_json(output_dir / "identity.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("results/overlays"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_overlays(args.flashinfer_root, args.output_dir)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
