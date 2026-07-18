#!/usr/bin/env python3
"""Build the diagnostic whole-kernel timing overlay for exp_004.

The overlay uses ``%globaltimer`` so CTA-local intervals share one device-wide
time domain.  W0 records one mutually-exclusive consumer timeline per task;
W4 records its overlapping producer timeline separately.  CTA leaders record
the launch-level P0--P4 boundaries and every warp records its loop exit.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

import build_overlays as base_builder
from exp004_common import (
    DISPATCH_RELATIVE_PATH,
    EXPECTED_DISPATCH_SHA256,
    EXPECTED_KERNEL_SHA256,
    KERNEL_RELATIVE_PATH,
    file_sha256,
    write_json,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "overlays" / "whole_kernel_probe"

TASK_TICKS = 65
CTA_TICKS = 14
W4_BASE = 55


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _replace(text: str, old: str, new: str, *, label: str) -> str:
    return base_builder._replace_exact(text, old, new, label=label)


def _read_and_task_store(event: str, *, indent: int) -> str:
    pad = " " * indent
    return (
        f"{pad}exp004_tick = _exp004_read_globaltimer()\n"
        f"{pad}st_global_u64(\n"
        f"{pad}    get_ptr_as_int64(\n"
        f"{pad}        timing_ticks,\n"
        f"{pad}        task_slot_probe * Int32(_EXP004_TASK_TICKS)\n"
        f"{pad}        + Int32({event}),\n"
        f"{pad}    ),\n"
        f"{pad}    exp004_tick,\n"
        f"{pad})\n"
    )


def _value_task_store(event: str, value: str, *, indent: int) -> str:
    pad = " " * indent
    return (
        f"{pad}st_global_u64(\n"
        f"{pad}    get_ptr_as_int64(\n"
        f"{pad}        timing_ticks,\n"
        f"{pad}        task_slot_probe * Int32(_EXP004_TASK_TICKS)\n"
        f"{pad}        + Int32({event}),\n"
        f"{pad}    ),\n"
        f"{pad}    {value},\n"
        f"{pad})\n"
    )


def _read_and_w4_store(interval: int, edge: int, *, indent: int) -> str:
    return _read_and_task_store(str(W4_BASE + interval * 2 + edge), indent=indent)


def _read_and_cta_store(event: str, *, indent: int) -> str:
    pad = " " * indent
    return (
        f"{pad}exp004_tick = _exp004_read_globaltimer()\n"
        f"{pad}st_global_u64(\n"
        f"{pad}    get_ptr_as_int64(\n"
        f"{pad}        cta_ticks,\n"
        f"{pad}        Int32(bidz) * Int32(_EXP004_CTA_TICKS)\n"
        f"{pad}        + Int32({event}),\n"
        f"{pad}    ),\n"
        f"{pad}    exp004_tick,\n"
        f"{pad})\n"
    )


KERNEL_CONSTANTS = f"""_FC2_TILE_RECIP_GS_NUM = 6.0 * 448.0

# exp_004 diagnostic whole-kernel timing ABI.
_EXP004_TASK_TICKS = {TASK_TICKS}
_EXP004_CTA_TICKS = {CTA_TICKS}


@dsl_user_op
def _exp004_read_globaltimer(*, loc=None, ip=None):
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _exp004_st_shared_u64(addr, value, *, loc=None, ip=None):
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
def _exp004_ld_shared_volatile_u64(addr, *, loc=None, ip=None):
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.volatile.shared.u64 $0, [$1];",
            "=l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _exp004_ld_shared_volatile_i32(addr, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.volatile.shared.s32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


"""


def _add_kernel_abi(source: str) -> str:
    text = _replace(
        source,
        "_FC2_TILE_RECIP_GS_NUM = 6.0 * 448.0\n",
        KERNEL_CONSTANTS,
        label="globaltimer helpers",
    )
    text = _replace(
        text,
        "        share_input_across_experts: bool = False,\n    ):\n",
        "        share_input_across_experts: bool = False,\n"
        "        phase_probe_enabled: bool = False,\n"
        "    ):\n",
        label="kernel constructor flag",
    )
    text = _replace(
        text,
        "        self.share_input_across_experts = share_input_across_experts\n",
        "        self.share_input_across_experts = share_input_across_experts\n"
        "        self.phase_probe_enabled = bool(phase_probe_enabled)\n",
        label="kernel constructor flag assignment",
    )
    text = _replace(
        text,
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "        max_active_clusters: cutlass.Constexpr,\n",
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "        timing_ticks: cute.Tensor,\n"
        "        task_cta_z: cute.Tensor,\n"
        "        cta_ticks: cute.Tensor,\n"
        "        max_active_clusters: cutlass.Constexpr,\n",
        label="kernel host-call ABI",
    )
    text = _replace(
        text,
        "            token_map,\n            token_weights,\n        ).launch(\n",
        "            token_map,\n"
        "            token_weights,\n"
        "            timing_ticks,\n"
        "            task_cta_z,\n"
        "            cta_ticks,\n"
        "        ).launch(\n",
        label="kernel launch args",
    )
    text = _replace(
        text,
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "    ):\n"
        '        """Kernel entry point."""\n',
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "        timing_ticks: cute.Tensor,\n"
        "        task_cta_z: cute.Tensor,\n"
        "        cta_ticks: cute.Tensor,\n"
        "    ):\n"
        '        """Kernel entry point."""\n',
        label="device kernel ABI",
    )
    text = _replace(
        text,
        "        full_tile_publish_enabled = Int32(0)\n",
        "        full_tile_publish_enabled = Int32(0)\n"
        "        exp004_task_capacity = Int32(task_cta_z.shape[0])\n",
        label="runtime probe capacity",
    )
    return text


def _instrument_kernel(source: str) -> str:
    text = _add_kernel_abi(source)

    text = _replace(
        text,
        "        is_cta_leader = Int32(1) if Int32(tidx) == Int32(0) else Int32(0)\n\n",
        "        is_cta_leader = Int32(1) if Int32(tidx) == Int32(0) else Int32(0)\n"
        "        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "            if is_cta_leader > Int32(0):\n"
        + _read_and_cta_store("0", indent=16)
        + "\n",
        label="CTA entry",
    )
    text = _replace(
        text,
        "        # Phase 0: cooperative init — zero routing state, queue state, and output.\n",
        "        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "            if is_cta_leader > Int32(0):\n"
        + _read_and_cta_store("1", indent=16)
        + "        # Phase 0: cooperative init — zero routing state, queue state, and output.\n",
        label="P0 start",
    )
    text = _replace(
        text,
        "        # Phase 1: histogram routed rows per expert.\n",
        "        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "            if is_cta_leader > Int32(0):\n"
        + _read_and_cta_store("2", indent=16)
        + "        # Phase 1: histogram routed rows per expert.\n",
        label="P0 end",
    )
    text = _replace(
        text,
        "        if flat_tid == Int32(0):\n            tile_acc = Int32(0)\n",
        "        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "            if is_cta_leader > Int32(0):\n"
        + _read_and_cta_store("3", indent=16)
        + "        if flat_tid == Int32(0):\n"
        "            tile_acc = Int32(0)\n",
        label="P1 end",
    )
    text = _replace(
        text,
        "        # Phase 2: warp-private route/pack producers into compact physical tiles.\n",
        "        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "            if is_cta_leader > Int32(0):\n"
        + _read_and_cta_store("4", indent=16)
        + "        # Phase 2: warp-private route/pack producers into compact physical tiles.\n",
        label="P2 end",
    )
    p3_end = """            self._resident_grid_barrier(
                barrier_count,
                barrier_epoch,
                Int32(gdim_z),
                is_cta_leader,
            )

            if is_cta_leader > Int32(0):
"""
    p3_probe = (
        """            self._resident_grid_barrier(
                barrier_count,
                barrier_epoch,
                Int32(gdim_z),
                is_cta_leader,
            )
            if cutlass.const_expr(self.phase_probe_enabled):
                if is_cta_leader > Int32(0):
"""
        + _read_and_cta_store("5", indent=20)
        + "\n            if is_cta_leader > Int32(0):\n"
    )
    text = _replace(text, p3_end, p3_probe, label="P3 end")
    text = _replace(
        text,
        "        gA = cute.local_tile(\n",
        "        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "            if is_cta_leader > Int32(0):\n"
        + _read_and_cta_store("6", indent=16)
        + "        gA = cute.local_tile(\n",
        label="P4 end",
    )
    text = _replace(
        text,
        "        consumer_live = Int32(1)\n"
        "        while consumer_live > Int32(0):\n"
        "            if is_cta_leader > Int32(0):\n",
        "        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "            if is_cta_leader > Int32(0):\n"
        + _read_and_cta_store("7", indent=16)
        + "        consumer_live = Int32(1)\n"
        "        while consumer_live > Int32(0):\n"
        "            if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                if is_cta_leader > Int32(0):\n"
        "                    exp004_task_start = _exp004_read_globaltimer()\n"
        "                    _exp004_st_shared_u64(\n"
        "                        route_phys_rows_addr, exp004_task_start\n"
        "                    )\n"
        "            if is_cta_leader > Int32(0):\n",
        label="compute-loop and task start",
    )

    claim_anchor = """            has_task = _ld_shared_i32(ctrl_base_addr + Int32(0))
            is_done = _ld_shared_i32(ctrl_base_addr + Int32(4))
"""
    claim_probe = (
        claim_anchor
        + """            if cutlass.const_expr(self.phase_probe_enabled):
                if warp_idx == Int32(0):
                    if lane_id == Int32(0):
                        if has_task > Int32(0):
                            task_slot_probe = _exp004_ld_shared_volatile_i32(
                                ctrl_base_addr + Int32(28)
                            )
                            exp004_task_start = _exp004_ld_shared_volatile_u64(
                                route_phys_rows_addr
                            )
"""
        + _value_task_store("0", "exp004_task_start", indent=28)
        + """                            st_global_i32(
                                get_ptr_as_int64(task_cta_z, task_slot_probe),
                                Int32(bidz),
                            )
"""
        + _read_and_task_store("1", indent=28)
    )
    text = _replace(text, claim_anchor, claim_probe, label="T0 claim boundary")

    consumer_entry = """            elif warp_idx < self.num_mma_warps:
                task_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(8))
"""
    consumer_probe = """            elif warp_idx < self.num_mma_warps:
                if cutlass.const_expr(self.phase_probe_enabled):
                    task_slot_probe = _exp004_ld_shared_volatile_i32(
                        ctrl_base_addr + Int32(28)
                    )
                task_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(8))
"""
    text = _replace(text, consumer_entry, consumer_probe, label="consumer task slot")

    text = _replace(
        text,
        "                    gate_acc.fill(0.0)\n",
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if warp_idx == Int32(0):\n"
        "                            if lane_id == Int32(0):\n"
        + _read_and_task_store("2", indent=32)
        + "                    gate_acc.fill(0.0)\n",
        label="Gate start",
    )
    text = _replace(
        text,
        "                    self.pass_gate_barrier.arrive_unaligned()\n\n"
        "                    if cutlass.const_expr(self.is_gated):\n",
        "                    self.pass_gate_barrier.arrive_unaligned()\n"
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if warp_idx == Int32(0):\n"
        "                            if lane_id == Int32(0):\n"
        + _read_and_task_store("3", indent=32)
        + "\n                    if cutlass.const_expr(self.is_gated):\n",
        label="Gate end",
    )
    text = _replace(
        text,
        "                    # Activation + quant into sA\n",
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if warp_idx == Int32(0):\n"
        "                            if lane_id == Int32(0):\n"
        + _read_and_task_store("4", indent=32)
        + "                    # Activation + quant into sA\n",
        label="Up end",
    )
    text = _replace(
        text,
        "                    self.epilog_sync_barrier.arrive_and_wait()\n\n"
        "                    # ============================================================\n"
        "                    # PHASE B: Sweep ALL FC2 output tiles using cached sA\n",
        "                    self.epilog_sync_barrier.arrive_and_wait()\n"
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if warp_idx == Int32(0):\n"
        "                            if lane_id == Int32(0):\n"
        + _read_and_task_store("5", indent=32)
        + "\n                    # ============================================================\n"
        "                    # PHASE B: Sweep ALL FC2 output tiles using cached sA\n",
        label="T3 end",
    )
    text = _replace(
        text,
        "                    phase2_cons_state.reset_count()\n"
        "                    for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]\n"
        "                        phase2_peek = phase2_pipeline.consumer_try_wait(\n",
        "                    phase2_cons_state.reset_count()\n"
        "                    for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]\n"
        "                        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                            if warp_idx == Int32(0):\n"
        "                                if lane_id == Int32(0):\n"
        + _read_and_task_store("Int32(7) + output_tile_idx * Int32(3)", indent=36)
        + "                        phase2_peek = phase2_pipeline.consumer_try_wait(\n",
        label="FC2 GEMM starts",
    )
    text = _replace(
        text,
        "                        # Scatter using precomputed metadata (no redundant gmem loads)\n",
        "                        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                            if warp_idx == Int32(0):\n"
        "                                if lane_id == Int32(0):\n"
        + _read_and_task_store("Int32(8) + output_tile_idx * Int32(3)", indent=36)
        + "                        # Scatter using precomputed metadata (no redundant gmem loads)\n",
        label="FC2 GEMM ends",
    )
    text = _replace(
        text,
        "                            self.epilog_sync_barrier.arrive_and_wait()\n\n"
        "                    # Signal that FC2/scatter no longer needs sA, so the DMA\n",
        "                            self.epilog_sync_barrier.arrive_and_wait()\n"
        "                        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                            if warp_idx == Int32(0):\n"
        "                                if lane_id == Int32(0):\n"
        + _read_and_task_store("Int32(9) + output_tile_idx * Int32(3)", indent=36)
        + "\n                    # Signal that FC2/scatter no longer needs sA, so the DMA\n",
        label="FC2 epilogue ends",
    )
    text = _replace(
        text,
        "                    self.pass_final_barrier.arrive_unaligned()\n"
        "                    slice_idx += Int32(1)\n",
        "                    self.pass_final_barrier.arrive_unaligned()\n"
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if warp_idx == Int32(0):\n"
        "                            if lane_id == Int32(0):\n"
        + _read_and_task_store("6", indent=32)
        + "                    slice_idx += Int32(1)\n",
        label="task end",
    )

    w4_entry = """            elif warp_idx == self.tma_load_warp_id:
                task_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(8))
"""
    w4_probe = """            elif warp_idx == self.tma_load_warp_id:
                if cutlass.const_expr(self.phase_probe_enabled):
                    task_slot_probe = _exp004_ld_shared_volatile_i32(
                        ctrl_base_addr + Int32(28)
                    )
                task_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(8))
"""
    text = _replace(text, w4_entry, w4_probe, label="W4 task slot")
    text = _replace(
        text,
        "                    # ---- FC1 gate/only pass ----\n"
        "                    prod_state.reset_count()\n",
        "                    # ---- FC1 gate/only pass ----\n"
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        + _read_and_w4_store(0, 0, indent=28)
        + "                    prod_state.reset_count()\n",
        label="W4 Gate TMA start",
    )
    text = _replace(
        text,
        "                    # Wait for the MMA warps to finish the FC1 gate/only pass\n"
        "                    # before reusing sA/sB/sSFA/sSFB for up/FC2 staging.\n"
        "                    self.pass_gate_barrier.wait_unaligned()\n",
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        + _read_and_w4_store(0, 1, indent=28)
        + _read_and_w4_store(1, 0, indent=28)
        + "                    # Wait for the MMA warps to finish the FC1 gate/only pass\n"
        "                    # before reusing sA/sB/sSFA/sSFB for up/FC2 staging.\n"
        "                    self.pass_gate_barrier.wait_unaligned()\n"
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        + _read_and_w4_store(1, 1, indent=28),
        label="W4 Gate TMA end/wait",
    )
    text = _replace(
        text,
        "                        # ---- FC1 up pass ----\n"
        "                        up_prod_state.reset_count()\n",
        "                        # ---- FC1 up pass ----\n"
        "                        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                            if lane_id == Int32(0):\n"
        + _read_and_w4_store(2, 0, indent=32)
        + "                        up_prod_state.reset_count()\n",
        label="W4 Up TMA start",
    )
    text = _replace(
        text,
        "                    # ---- FC2 B_down loads: continuous pipeline ----\n",
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        + _read_and_w4_store(2, 1, indent=28)
        + "                    # ---- FC2 B_down loads: continuous pipeline ----\n",
        label="W4 Up TMA end",
    )
    text = _replace(
        text,
        "                    phase2_prod_state.reset_count()\n"
        "                    for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]\n",
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        + _read_and_w4_store(3, 0, indent=28)
        + "                    phase2_prod_state.reset_count()\n"
        "                    for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]\n",
        label="W4 Down TMA start",
    )
    text = _replace(
        text,
        "                    # Ensure MMA warps finish FC2/scatter before DMA starts the\n"
        "                    # next slice/task's FC1 loads into shared A buffers.\n"
        "                    self.pass_final_barrier.wait_unaligned()\n",
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        + _read_and_w4_store(3, 1, indent=28)
        + _read_and_w4_store(4, 0, indent=28)
        + "                    # Ensure MMA warps finish FC2/scatter before DMA starts the\n"
        "                    # next slice/task's FC1 loads into shared A buffers.\n"
        "                    self.pass_final_barrier.wait_unaligned()\n"
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if lane_id == Int32(0):\n"
        + _read_and_w4_store(4, 1, indent=28),
        label="W4 Down TMA end/final wait",
    )

    text = _replace(
        text,
        "        if warp_idx == self.tma_load_warp_id:\n"
        "            ml_pipeline.producer_tail(prod_state)\n",
        "        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "            if lane_id == Int32(0):\n"
        + _read_and_cta_store("Int32(8) + Int32(warp_idx)", indent=16)
        + "        if warp_idx == self.tma_load_warp_id:\n"
        "            ml_pipeline.producer_tail(prod_state)\n",
        label="warp loop exits",
    )
    text = _replace(
        text,
        "            phase2_pipeline.producer_tail(phase2_prod_state)\n"
        "        return\n",
        "            phase2_pipeline.producer_tail(phase2_prod_state)\n"
        "            if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                if lane_id == Int32(0):\n"
        + _read_and_cta_store("13", indent=20)
        + "        return\n",
        label="W4 producer-tail end",
    )
    return text


def _instrument_dispatch(source: str, *, enabled: bool) -> str:
    text = base_builder._instrument_dispatch(source, enabled=enabled)
    text = _replace(
        text,
        "_EXP004_TIMING_TICKS_PER_TASK = 306\n",
        f"_EXP004_TIMING_TICKS_PER_TASK = {TASK_TICKS}\n"
        f"_EXP004_CTA_TICKS_PER_CTA = {CTA_TICKS}\n",
        label="whole-kernel dispatch constants",
    )
    text = _replace(
        text,
        "    exp004_task_cta_z: torch.Tensor\n\n    # Views\n",
        "    exp004_task_cta_z: torch.Tensor\n"
        "    exp004_cta_ticks: torch.Tensor\n\n"
        "    # Views\n",
        label="CTA timing workspace field",
    )
    text = _replace(
        text,
        "        exp004_task_cta_z=torch.full(\n"
        "            (max_tasks,), -1, dtype=torch.int32, device=device\n"
        "        ),\n"
        "    )\n",
        "        exp004_task_cta_z=torch.full(\n"
        "            (max_tasks,), -1, dtype=torch.int32, device=device\n"
        "        ),\n"
        "        exp004_cta_ticks=torch.full(\n"
        "            (get_num_sm(device) * _EXP004_CTA_TICKS_PER_CTA,),\n"
        "            -1, dtype=torch.int64, device=device,\n"
        "        ),\n"
        "    )\n",
        label="CTA timing allocation",
    )
    text = _replace(
        text,
        "        exp004_task_cta_z_ptr: cute.Pointer,\n        b_w13: cute.Tensor,\n",
        "        exp004_task_cta_z_ptr: cute.Pointer,\n"
        "        exp004_cta_ticks_ptr: cute.Pointer,\n"
        "        b_w13: cute.Tensor,\n",
        label="CTA timing launch pointer",
    )
    text = _replace(
        text,
        "        exp004_task_cta_z = cute.make_tensor(\n"
        "            exp004_task_cta_z_ptr,\n"
        "            layout=cute.make_layout((max_tasks,), stride=(1,)),\n"
        "        )\n"
        "        self._kernel(\n",
        "        exp004_task_cta_z = cute.make_tensor(\n"
        "            exp004_task_cta_z_ptr,\n"
        "            layout=cute.make_layout((max_tasks,), stride=(1,)),\n"
        "        )\n"
        "        exp004_cta_ticks = cute.make_tensor(\n"
        "            exp004_cta_ticks_ptr,\n"
        "            layout=cute.make_layout(\n"
        "                (max_active_clusters * _EXP004_CTA_TICKS_PER_CTA,), stride=(1,)\n"
        "            ),\n"
        "        )\n"
        "        self._kernel(\n",
        label="CTA timing tensor view",
    )
    text = _replace(
        text,
        "            exp004_timing_ticks,\n"
        "            exp004_task_cta_z,\n"
        "            max_active_clusters=max_active_clusters,\n",
        "            exp004_timing_ticks,\n"
        "            exp004_task_cta_z,\n"
        "            exp004_cta_ticks,\n"
        "            max_active_clusters=max_active_clusters,\n",
        label="CTA timing kernel arg",
    )
    text = _replace(
        text,
        "    exp004_task_cta_z_fake = make_ptr(\n"
        "        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4\n"
        "    )\n\n"
        "    b_w13_fake = cute.runtime.make_fake_compact_tensor(\n",
        "    exp004_task_cta_z_fake = make_ptr(\n"
        "        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4\n"
        "    )\n"
        "    exp004_cta_ticks_fake = make_ptr(\n"
        "        cutlass.Uint64, 8, cute.AddressSpace.gmem, assumed_align=8\n"
        "    )\n\n"
        "    b_w13_fake = cute.runtime.make_fake_compact_tensor(\n",
        label="CTA timing fake pointer",
    )
    text = _replace(
        text,
        "        exp004_timing_ticks_fake,\n"
        "        exp004_task_cta_z_fake,\n"
        "        b_w13_fake,\n",
        "        exp004_timing_ticks_fake,\n"
        "        exp004_task_cta_z_fake,\n"
        "        exp004_cta_ticks_fake,\n"
        "        b_w13_fake,\n",
        label="CTA timing compile arg",
    )
    text = _replace(
        text,
        "        workspace.exp004_timing_ticks.data_ptr(),\n"
        "        workspace.exp004_task_cta_z.data_ptr(),\n"
        "        weights.w13_fp4,\n",
        "        workspace.exp004_timing_ticks.data_ptr(),\n"
        "        workspace.exp004_task_cta_z.data_ptr(),\n"
        "        workspace.exp004_cta_ticks.data_ptr(),\n"
        "        weights.w13_fp4,\n",
        label="CTA timing runtime arg",
    )
    return text


def build(repo: Path, output: Path, *, enabled: bool = True) -> dict[str, object]:
    repo = repo.resolve()
    kernel_path = repo / KERNEL_RELATIVE_PATH
    dispatch_path = repo / DISPATCH_RELATIVE_PATH
    if file_sha256(kernel_path) != EXPECTED_KERNEL_SHA256:
        raise ValueError("production kernel identity drift")
    if file_sha256(dispatch_path) != EXPECTED_DISPATCH_SHA256:
        raise ValueError("production dispatch identity drift")
    if output.exists():
        raise FileExistsError(f"immutable whole-kernel overlay exists: {output}")

    production_kernel = kernel_path.read_text()
    production_dispatch = dispatch_path.read_text()
    kernel = _instrument_kernel(production_kernel)
    dispatch = _instrument_dispatch(production_dispatch, enabled=enabled)
    ast.parse(kernel, filename="whole_kernel_probe/moe_dynamic_kernel.py")
    ast.parse(dispatch, filename="whole_kernel_probe/moe_dispatch.py")

    output.mkdir(parents=True)
    (output / "moe_dynamic_kernel.py").write_text(kernel)
    (output / "moe_dispatch.py").write_text(dispatch)
    kernel_diff = "".join(
        difflib.unified_diff(
            production_kernel.splitlines(keepends=True),
            kernel.splitlines(keepends=True),
            fromfile="production/moe_dynamic_kernel.py",
            tofile="whole_kernel_probe/moe_dynamic_kernel.py",
        )
    )
    dispatch_diff = "".join(
        difflib.unified_diff(
            production_dispatch.splitlines(keepends=True),
            dispatch.splitlines(keepends=True),
            fromfile="production/moe_dispatch.py",
            tofile="whole_kernel_probe/moe_dispatch.py",
        )
    )
    (output / "moe_dynamic_kernel.diff").write_text(kernel_diff)
    (output / "moe_dispatch.diff").write_text(dispatch_diff)
    manifest: dict[str, object] = {
        "schema": "exp004.whole-kernel-probe-overlay.v1",
        "classification": "diagnostic-only" if enabled else "measurement-control",
        "probe_enabled": enabled,
        "timer": "%globaltimer",
        "task_ticks_per_task": TASK_TICKS,
        "cta_ticks_per_cta": CTA_TICKS,
        "production": {
            "kernel_sha256": EXPECTED_KERNEL_SHA256,
            "dispatch_sha256": EXPECTED_DISPATCH_SHA256,
        },
        "kernel_sha256": _sha256_text(kernel),
        "dispatch_sha256": _sha256_text(dispatch),
        "kernel_diff_sha256": _sha256_text(kernel_diff),
        "dispatch_diff_sha256": _sha256_text(dispatch_diff),
    }
    write_json(output / "identity.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--disabled",
        action="store_true",
        help="build the matched plumbing-only measurement control",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.flashinfer_root.resolve(),
                args.output.resolve(),
                enabled=not args.disabled,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
