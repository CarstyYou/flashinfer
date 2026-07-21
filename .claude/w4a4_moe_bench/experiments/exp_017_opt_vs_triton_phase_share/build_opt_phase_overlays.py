#!/usr/bin/env python3
"""Build matched no-marker/probe overlays for exp_017 latest-opt timing."""

from __future__ import annotations

import argparse
import ast
import difflib
import json
from pathlib import Path
import shutil
from typing import Any

from exp017_opt_phase_common import (
    CONTROL,
    CURSOR_SLOT,
    DISPATCH_RELATIVE_PATH,
    ENTRY_SLOT,
    EVENT_ABI,
    EVENTS_PER_CTA,
    EXPECTED_DISPATCH_SHA256,
    EXPECTED_OPT_SHA256,
    EXPECTED_WRAPPER_SHA256,
    MODES,
    OPT_RELATIVE_PATH,
    OVERLAY_ROOT,
    PHASE_NAMES,
    PHASE_SLOT_BASE,
    PROBE,
    READER_FINAL_SLOT,
    TMA_FINAL_SLOT,
    WRAPPER_RELATIVE_PATH,
    barrier_fingerprint,
    canonical_sha256,
    file_sha256,
    normalize_dispatch_flag,
    read_json,
    text_sha256,
    write_json,
)


ROOT = Path(__file__).resolve().parent


KERNEL_CONSTANTS = f"""_FC2_TILE_RECIP_GS_NUM = 6.0 * 448.0

# exp_017 diagnostic full-phase timing ABI. Control and probe compile this
# byte-identical source; the dispatch specializes the constexpr enable flag.
_EXP017_OPT_PHASE_EVENTS_PER_CTA = {EVENTS_PER_CTA}


@dsl_user_op
def _exp017_read_globaltimer(*, loc=None, ip=None):
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


"""


def replace_exact(
    text: str, old: str, new: str, *, label: str, expected: int = 1
) -> str:
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{label}: expected {expected} anchors, found {count}")
    return text.replace(old, new, expected)


def _event_index(slot: int) -> str:
    return f"Int32(bidz) * Int32(_EXP017_OPT_PHASE_EVENTS_PER_CTA) + Int32({slot})"


def mark_entry(*, indent: int) -> str:
    pad = " " * indent
    return (
        f"{pad}if cutlass.const_expr(self.exp017_opt_phase_probe_enabled):\n"
        f"{pad}    if is_cta_leader > Int32(0):\n"
        f"{pad}        exp017_entry_now = _exp017_read_globaltimer()\n"
        f"{pad}        st_global_u64(\n"
        f"{pad}            get_ptr_as_int64(exp017_phase_events, {_event_index(ENTRY_SLOT)}),\n"
        f"{pad}            exp017_entry_now,\n"
        f"{pad}        )\n"
        f"{pad}        st_global_u64(\n"
        f"{pad}            get_ptr_as_int64(exp017_phase_events, {_event_index(CURSOR_SLOT)}),\n"
        f"{pad}            exp017_entry_now,\n"
        f"{pad}        )\n"
    )


def mark_start(tag: str, *, indent: int) -> str:
    pad = " " * indent
    return (
        f"{pad}if cutlass.const_expr(self.exp017_opt_phase_probe_enabled):\n"
        f"{pad}    if is_cta_leader > Int32(0):\n"
        f"{pad}        exp017_{tag}_start = _exp017_read_globaltimer()\n"
        f"{pad}        st_global_u64(\n"
        f"{pad}            get_ptr_as_int64(exp017_phase_events, {_event_index(CURSOR_SLOT)}),\n"
        f"{pad}            exp017_{tag}_start,\n"
        f"{pad}        )\n"
    )


def mark_close(phase: str, tag: str, *, indent: int) -> str:
    slot = PHASE_SLOT_BASE + PHASE_NAMES.index(phase)
    pad = " " * indent
    return (
        f"{pad}if cutlass.const_expr(self.exp017_opt_phase_probe_enabled):\n"
        f"{pad}    if is_cta_leader > Int32(0):\n"
        f"{pad}        exp017_{tag}_end = _exp017_read_globaltimer()\n"
        f"{pad}        exp017_{tag}_cursor = _ld_global_u64(\n"
        f"{pad}            get_ptr_as_int64(exp017_phase_events, {_event_index(CURSOR_SLOT)})\n"
        f"{pad}        )\n"
        f"{pad}        exp017_{tag}_total = _ld_global_u64(\n"
        f"{pad}            get_ptr_as_int64(exp017_phase_events, {_event_index(slot)})\n"
        f"{pad}        )\n"
        f"{pad}        st_global_u64(\n"
        f"{pad}            get_ptr_as_int64(exp017_phase_events, {_event_index(slot)}),\n"
        f"{pad}            exp017_{tag}_total + exp017_{tag}_end - exp017_{tag}_cursor,\n"
        f"{pad}        )\n"
        f"{pad}        st_global_u64(\n"
        f"{pad}            get_ptr_as_int64(exp017_phase_events, {_event_index(CURSOR_SLOT)}),\n"
        f"{pad}            exp017_{tag}_end,\n"
        f"{pad}        )\n"
    )


def mark_final(slot: int, tag: str, condition: str, *, indent: int) -> str:
    pad = " " * indent
    return (
        f"{pad}if cutlass.const_expr(self.exp017_opt_phase_probe_enabled):\n"
        f"{pad}    if {condition}:\n"
        f"{pad}        exp017_{tag}_final = _exp017_read_globaltimer()\n"
        f"{pad}        st_global_u64(\n"
        f"{pad}            get_ptr_as_int64(exp017_phase_events, {_event_index(slot)}),\n"
        f"{pad}            exp017_{tag}_final,\n"
        f"{pad}        )\n"
    )


def instrument_kernel(source: str) -> str:
    """Add the full-phase ABI without introducing synchronization."""
    original_barriers = barrier_fingerprint(source)
    text = replace_exact(
        source,
        "_FC2_TILE_RECIP_GS_NUM = 6.0 * 448.0\n",
        KERNEL_CONSTANTS,
        label="globaltimer helper",
    )
    text = replace_exact(
        text,
        "        share_input_across_experts: bool = False,\n    ):\n",
        "        share_input_across_experts: bool = False,\n"
        "        exp017_opt_phase_probe_enabled: bool = False,\n"
        "    ):\n",
        label="constructor flag",
    )
    text = replace_exact(
        text,
        "        self.share_input_across_experts = share_input_across_experts\n",
        "        self.share_input_across_experts = share_input_across_experts\n"
        "        self.exp017_opt_phase_probe_enabled = bool(\n"
        "            exp017_opt_phase_probe_enabled\n"
        "        )\n",
        label="constructor flag assignment",
    )

    # Host/kernel launch ABI.
    text = replace_exact(
        text,
        "        token_weights: cute.Tensor,\n"
        "        max_active_clusters: cutlass.Constexpr,\n",
        "        token_weights: cute.Tensor,\n"
        "        exp017_phase_events: cute.Tensor,\n"
        "        max_active_clusters: cutlass.Constexpr,\n",
        label="host-call event ABI",
    )
    text = replace_exact(
        text,
        "            token_map,\n            token_weights,\n        ).launch(\n",
        "            token_map,\n"
        "            token_weights,\n"
        "            exp017_phase_events,\n"
        "        ).launch(\n",
        label="device launch event argument",
    )
    text = replace_exact(
        text,
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "    ):\n"
        '        """Kernel entry point."""\n',
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "        exp017_phase_events: cute.Tensor,\n"
        "    ):\n"
        '        """Kernel entry point."""\n',
        label="device kernel event ABI",
    )
    text = replace_exact(
        text,
        "        launch_params: DynamicLaunchParams,\n"
        "        full_tile_publish_enabled: Int32,\n"
        "    ):\n"
        "        tidx, bidz, gdim_z, warp_idx, is_cta_leader = thread_info\n",
        "        launch_params: DynamicLaunchParams,\n"
        "        full_tile_publish_enabled: Int32,\n"
        "        exp017_phase_events: cute.Tensor,\n"
        "    ):\n"
        "        tidx, bidz, gdim_z, warp_idx, is_cta_leader = thread_info\n",
        label="init helper event ABI",
    )
    text = replace_exact(
        text,
        "            launch_params,\n"
        "            full_tile_publish_enabled,\n"
        "        )\n\n"
        "        gA = cute.local_tile(\n",
        "            launch_params,\n"
        "            full_tile_publish_enabled,\n"
        "            exp017_phase_events,\n"
        "        )\n\n"
        "        gA = cute.local_tile(\n",
        label="init helper event argument",
    )

    # CTA entry and initialization phase transitions.
    entry_anchor = (
        "        is_cta_leader = Int32(1) if Int32(tidx) == Int32(0) else Int32(0)\n\n"
        "        if warp_idx == 0:\n"
    )
    text = replace_exact(
        text,
        entry_anchor,
        "        is_cta_leader = Int32(1) if Int32(tidx) == Int32(0) else Int32(0)\n"
        + mark_entry(indent=8)
        + "\n        if warp_idx == 0:\n",
        label="CTA entry marker",
    )
    clear_start_anchor = (
        "        # Phase 0: cooperative init — zero routing state, queue state, and output.\n"
        "        task_capacity = Int32(task_ready.shape[0])\n"
    )
    text = replace_exact(
        text,
        clear_start_anchor,
        "        # Phase 0: cooperative init — zero routing state, queue state, and output.\n"
        + mark_start("clear", indent=8)
        + "        task_capacity = Int32(task_ready.shape[0])\n",
        label="clear start",
    )
    clear_end_anchor = (
        "            is_cta_leader,\n"
        "        )\n\n"
        "        # Phase 1: histogram routed rows per expert.\n"
    )
    text = replace_exact(
        text,
        clear_end_anchor,
        "            is_cta_leader,\n"
        "        )\n"
        + mark_close("clear_init", "clear", indent=8)
        + "\n        # Phase 1: histogram routed rows per expert.\n",
        label="clear/histogram transition",
    )
    histogram_end_anchor = (
        "            is_cta_leader,\n        )\n\n        if flat_tid == Int32(0):\n"
    )
    text = replace_exact(
        text,
        histogram_end_anchor,
        "            is_cta_leader,\n"
        "        )\n"
        + mark_close("histogram", "histogram", indent=8)
        + "\n        if flat_tid == Int32(0):\n",
        label="histogram/prefix transition",
    )
    prefix_end_anchor = (
        "            is_cta_leader,\n"
        "        )\n\n"
        "        # Phase 2: warp-private route/pack producers into compact physical tiles.\n"
    )
    text = replace_exact(
        text,
        prefix_end_anchor,
        "            is_cta_leader,\n"
        "        )\n"
        + mark_close("prefix", "prefix", indent=8)
        + "\n        # Phase 2: warp-private route/pack producers into compact physical tiles.\n",
        label="prefix/route transition",
    )
    route_end_anchor = (
        "        cute.arch.sync_threads()\n"
        "        # Conservative publish fence before the last-producer CTA flushes any\n"
    )
    text = replace_exact(
        text,
        route_end_anchor,
        "        cute.arch.sync_threads()\n"
        + mark_close("route_q0_pack", "route", indent=8)
        + "        # Conservative publish fence before the last-producer CTA flushes any\n",
        label="route/publish transition",
    )
    publish_end_anchor = (
        "                _st_global_release_i32(\n"
        "                    get_ptr_as_int64(all_work_published, Int32(0)),\n"
        "                    Int32(1),\n"
        "                )\n\n\n"
        "    @cute.jit\n"
    )
    text = replace_exact(
        text,
        publish_end_anchor,
        "                _st_global_release_i32(\n"
        "                    get_ptr_as_int64(all_work_published, Int32(0)),\n"
        "                    Int32(1),\n"
        "                )\n"
        + mark_close("publish_route_tail", "publish", indent=8)
        + "\n\n    @cute.jit\n",
        label="publish end",
    )

    # Persistent reader track. Each close updates the cursor, so the named
    # intervals cannot overlap; gaps and W8 overlap fall into residual.
    claim_anchor = (
        "        while consumer_live > Int32(0):\n"
        "            has_task, is_done = self.claim_and_cache_task(\n"
    )
    text = replace_exact(
        text,
        claim_anchor,
        "        while consumer_live > Int32(0):\n"
        + mark_start("claim", indent=12)
        + "            has_task, is_done = self.claim_and_cache_task(\n",
        label="claim start",
    )
    claim_end_anchor = (
        "                scatter_weight_base_addr,\n"
        "            )\n"
        "            if has_task == Int32(0):\n"
    )
    text = replace_exact(
        text,
        claim_end_anchor,
        "                scatter_weight_base_addr,\n"
        "            )\n"
        + mark_close("claim_cache_control", "claim", indent=12)
        + "            if has_task == Int32(0):\n",
        label="claim end",
    )
    fc1_start_anchor = (
        "                while slice_idx < task_slice_count_val:\n"
        "                    cons_state = self.fc1_gate_up_swiglu_to_sC(\n"
    )
    text = replace_exact(
        text,
        fc1_start_anchor,
        "                while slice_idx < task_slice_count_val:\n"
        + mark_start("fc1", indent=20)
        + "                    cons_state = self.fc1_gate_up_swiglu_to_sC(\n",
        label="FC1 start",
    )
    fc1_end_anchor = (
        '                    cute.arch.fence_proxy("async.shared", space="cta")\n'
        "                    self.epilog_sync_barrier.arrive_and_wait()\n\n"
        "                    self.quantize_q1_sC_to_sA_sSFA(\n"
    )
    text = replace_exact(
        text,
        fc1_end_anchor,
        '                    cute.arch.fence_proxy("async.shared", space="cta")\n'
        "                    self.epilog_sync_barrier.arrive_and_wait()\n"
        + mark_close("fc1_gate_up_swiglu", "fc1", indent=20)
        + "\n                    self.quantize_q1_sC_to_sA_sSFA(\n",
        label="FC1/Q1 transition",
    )
    q1_end_anchor = (
        '                    cute.arch.fence_proxy("async.shared", space="cta")\n'
        "                    self.epilog_sync_barrier.arrive_and_wait()\n\n"
        "                    self.load_fc2_a_fragments(\n"
    )
    text = replace_exact(
        text,
        q1_end_anchor,
        '                    cute.arch.fence_proxy("async.shared", space="cta")\n'
        "                    self.epilog_sync_barrier.arrive_and_wait()\n"
        + mark_close("q1", "q1", indent=20)
        + "\n"
        + mark_start("fc2_first", indent=20)
        + "                    self.load_fc2_a_fragments(\n",
        label="Q1/FC2 transition",
    )
    fc2_end_anchor = (
        '                        cute.arch.fence_proxy("async.shared", space="cta")\n'
        "                        self.epilog_sync_barrier.arrive_and_wait()\n\n"
        "                        self.scatter_sC_to_gmem(\n"
    )
    text = replace_exact(
        text,
        fc2_end_anchor,
        '                        cute.arch.fence_proxy("async.shared", space="cta")\n'
        "                        self.epilog_sync_barrier.arrive_and_wait()\n"
        + mark_close("fc2_epilogue_r2s", "fc2", indent=24)
        + "\n                        self.scatter_sC_to_gmem(\n",
        label="FC2/scatter transition",
    )
    scatter_end_anchor = (
        "                        # tile begins collective phase2 pipeline operations.\n"
        "                        self.epilog_sync_barrier.arrive_and_wait()\n\n"
        "                    # Signal that FC2/scatter no longer needs sA, so the DMA\n"
    )
    text = replace_exact(
        text,
        scatter_end_anchor,
        "                        # tile begins collective phase2 pipeline operations.\n"
        "                        self.epilog_sync_barrier.arrive_and_wait()\n"
        + mark_close("scatter", "scatter", indent=24)
        + "                        if output_tile_idx + Int32(1) < output_tile_cnt:\n"
        + mark_start("fc2_next", indent=28)
        + "\n                    # Signal that FC2/scatter no longer needs sA, so the DMA\n",
        label="scatter end/next FC2 start",
    )

    final_anchor = (
        "        if warp_idx == self.tma_load_warp_id:\n"
        "            ml_pipeline.producer_tail(prod_state)\n"
        "            phase2_pipeline.producer_tail(phase2_prod_state)\n"
        "        return\n"
    )
    text = replace_exact(
        text,
        final_anchor,
        "        if warp_idx == self.tma_load_warp_id:\n"
        "            ml_pipeline.producer_tail(prod_state)\n"
        "            phase2_pipeline.producer_tail(phase2_prod_state)\n"
        + mark_final(
            TMA_FINAL_SLOT,
            "tma",
            "Int32(tidx) == Int32(self.tma_load_warp_id * self.num_threads_per_warp)",
            indent=12,
        )
        + mark_final(
            READER_FINAL_SLOT,
            "reader",
            "is_cta_leader > Int32(0)",
            indent=8,
        )
        + "        return\n",
        label="reader/TMA final markers",
    )

    if barrier_fingerprint(text) != original_barriers:
        raise RuntimeError("exp_017 instrumentation changed barrier fingerprint")
    timer_reads = text.count("_exp017_read_globaltimer()")
    if timer_reads != 18:
        raise RuntimeError(f"timer read call-site count drift: {timer_reads} != 18")
    ast.parse(text, filename="exp017_opt_phase/moe_dynamic_kernel.py")
    return text


def instrument_dispatch(source: str, *, enabled: bool) -> str:
    """Plumb the fixed-size event buffer through the dynamic launch ABI."""
    text = replace_exact(
        source,
        "_DYNAMIC_SLICE_CHUNK = 1\n",
        "_DYNAMIC_SLICE_CHUNK = 1\n"
        f"_EXP017_OPT_PHASE_PROBE_ENABLED = {enabled!r}\n"
        f"_EXP017_OPT_PHASE_EVENTS_PER_CTA = {EVENTS_PER_CTA}\n",
        label="dispatch constants",
    )
    text = replace_exact(
        text,
        "    task_valid_rows: torch.Tensor\n"
        "    tile_write_count: torch.Tensor\n\n"
        "    # Views\n",
        "    task_valid_rows: torch.Tensor\n"
        "    tile_write_count: torch.Tensor\n"
        "    exp017_phase_events: torch.Tensor\n\n"
        "    # Views\n",
        label="workspace event field",
    )
    text = replace_exact(
        text,
        "        tile_write_count=torch.zeros(physical_tiles, dtype=torch.int32, device=device),\n"
        "    )\n",
        "        tile_write_count=torch.zeros(physical_tiles, dtype=torch.int32, device=device),\n"
        "        exp017_phase_events=torch.zeros(\n"
        "            get_num_sm(device) * _EXP017_OPT_PHASE_EVENTS_PER_CTA,\n"
        "            dtype=torch.int64, device=device,\n"
        "        ),\n"
        "    )\n",
        label="workspace event allocation",
    )
    text = replace_exact(
        text,
        "        tile_write_count_ptr: cute.Pointer,\n        b_w13: cute.Tensor,\n",
        "        tile_write_count_ptr: cute.Pointer,\n"
        "        exp017_phase_events_ptr: cute.Pointer,\n"
        "        b_w13: cute.Tensor,\n",
        label="dynamic-launch event pointer",
    )
    text = replace_exact(
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
        "        exp017_phase_events = cute.make_tensor(\n"
        "            exp017_phase_events_ptr,\n"
        "            layout=cute.make_layout(\n"
        "                (max_active_clusters * _EXP017_OPT_PHASE_EVENTS_PER_CTA,),\n"
        "                stride=(1,),\n"
        "            ),\n"
        "        )\n"
        "        self._kernel(\n",
        label="dynamic-launch event tensor",
    )
    text = replace_exact(
        text,
        "            token_map,\n"
        "            token_weights_t,\n"
        "            max_active_clusters=max_active_clusters,\n",
        "            token_map,\n"
        "            token_weights_t,\n"
        "            exp017_phase_events,\n"
        "            max_active_clusters=max_active_clusters,\n",
        label="kernel event argument",
    )
    text = replace_exact(
        text,
        "        share_input_across_experts,\n"
        "    )\n"
        "    cached = _DYNAMIC_KERNEL_CACHE.get(cache_key)\n",
        "        share_input_across_experts,\n"
        "        _EXP017_OPT_PHASE_PROBE_ENABLED,\n"
        "    )\n"
        "    cached = _DYNAMIC_KERNEL_CACHE.get(cache_key)\n",
        label="JIT cache event identity",
    )
    text = replace_exact(
        text,
        "        share_input_across_experts=share_input_across_experts,\n"
        "    )\n"
        "    launch = _DynamicMoELaunch(\n",
        "        share_input_across_experts=share_input_across_experts,\n"
        "        exp017_opt_phase_probe_enabled=_EXP017_OPT_PHASE_PROBE_ENABLED,\n"
        "    )\n"
        "    launch = _DynamicMoELaunch(\n",
        label="kernel event specialization",
    )
    text = replace_exact(
        text,
        "    tile_write_count_fake = make_ptr(\n"
        "        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4\n"
        "    )\n\n"
        "    b_w13_fake = cute.runtime.make_fake_compact_tensor(\n",
        "    tile_write_count_fake = make_ptr(\n"
        "        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4\n"
        "    )\n"
        "    exp017_phase_events_fake = make_ptr(\n"
        "        cutlass.Uint64, 8, cute.AddressSpace.gmem, assumed_align=8\n"
        "    )\n\n"
        "    b_w13_fake = cute.runtime.make_fake_compact_tensor(\n",
        label="compile fake event pointer",
    )
    text = replace_exact(
        text,
        "        task_valid_rows_fake,\n"
        "        tile_write_count_fake,\n"
        "        b_w13_fake,\n",
        "        task_valid_rows_fake,\n"
        "        tile_write_count_fake,\n"
        "        exp017_phase_events_fake,\n"
        "        b_w13_fake,\n",
        label="compile event argument",
    )
    text = replace_exact(
        text,
        "        workspace.task_valid_rows.data_ptr(),\n"
        "        workspace.tile_write_count.data_ptr(),\n"
        "        weights.w13_fp4,\n",
        "        workspace.task_valid_rows.data_ptr(),\n"
        "        workspace.tile_write_count.data_ptr(),\n"
        "        workspace.exp017_phase_events.data_ptr(),\n"
        "        weights.w13_fp4,\n",
        label="runtime event argument",
    )
    if text.count("_EXP017_OPT_PHASE_PROBE_ENABLED") != 3:
        raise RuntimeError(
            "dispatch enable flag must appear in constant/cache/constructor"
        )
    ast.parse(text, filename="exp017_opt_phase/moe_dispatch.py")
    return text


def diff(left: str, right: str, *, left_name: str, right_name: str) -> str:
    return "".join(
        difflib.unified_diff(
            left.splitlines(keepends=True),
            right.splitlines(keepends=True),
            fromfile=left_name,
            tofile=right_name,
            n=0,
        )
    )


def overlay_paths(output: Path, mode: str) -> tuple[Path, Path, Path]:
    root = output / mode
    return root, root / "moe_dynamic_kernel.py", root / "moe_dispatch.py"


def verify_existing(repo: Path, output: Path) -> dict[str, Any]:
    identity = read_json(output / "identity.json")
    if identity.get("schema") != "exp017.opt-phase-overlays.v1":
        raise RuntimeError("overlay root schema drift")
    if identity.get("event_abi") != EVENT_ABI:
        raise RuntimeError("event ABI drift")
    live = (
        (repo / OPT_RELATIVE_PATH, EXPECTED_OPT_SHA256, "latest opt"),
        (repo / DISPATCH_RELATIVE_PATH, EXPECTED_DISPATCH_SHA256, "dispatch"),
        (repo / WRAPPER_RELATIVE_PATH, EXPECTED_WRAPPER_SHA256, "wrapper"),
    )
    for path, expected, label in live:
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"live {label} identity drift")

    kernels: dict[str, str] = {}
    dispatches: dict[str, str] = {}
    for mode in MODES:
        root, kernel, dispatch = overlay_paths(output, mode)
        manifest = read_json(root / "identity.json")
        checks = {
            "registered": manifest == identity["modes"][mode],
            "mode": manifest.get("mode") == mode,
            "enabled": bool(manifest.get("probe_enabled")) == (mode == PROBE),
            "kernel_hash": file_sha256(kernel) == manifest["overlay"]["kernel_sha256"],
            "dispatch_hash": file_sha256(dispatch)
            == manifest["overlay"]["dispatch_sha256"],
            "barriers": barrier_fingerprint(kernel.read_text(encoding="utf-8"))
            == manifest["base"]["barrier_fingerprint"],
        }
        if not all(checks.values()):
            raise RuntimeError(f"{mode} overlay drift: {checks}")
        kernels[mode] = kernel.read_text(encoding="utf-8")
        dispatches[mode] = dispatch.read_text(encoding="utf-8")
    if kernels[CONTROL] != kernels[PROBE]:
        raise RuntimeError("control/probe kernel source differs")
    if normalize_dispatch_flag(dispatches[CONTROL]) != normalize_dispatch_flag(
        dispatches[PROBE]
    ):
        raise RuntimeError("control/probe dispatch differs beyond constexpr flag")
    return identity


def build(repo: Path, output: Path) -> dict[str, Any]:
    repo = repo.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable overlay exists: {output}")
    source_paths = {
        "kernel": repo / OPT_RELATIVE_PATH,
        "dispatch": repo / DISPATCH_RELATIVE_PATH,
        "wrapper": repo / WRAPPER_RELATIVE_PATH,
    }
    expected = {
        "kernel": EXPECTED_OPT_SHA256,
        "dispatch": EXPECTED_DISPATCH_SHA256,
        "wrapper": EXPECTED_WRAPPER_SHA256,
    }
    for label, path in source_paths.items():
        observed = file_sha256(path)
        if observed != expected[label]:
            raise RuntimeError(f"{label} source drift: {observed} != {expected[label]}")

    kernel_source = source_paths["kernel"].read_text(encoding="utf-8")
    dispatch_source = source_paths["dispatch"].read_text(encoding="utf-8")
    kernel = instrument_kernel(kernel_source)
    dispatches = {
        CONTROL: instrument_dispatch(dispatch_source, enabled=False),
        PROBE: instrument_dispatch(dispatch_source, enabled=True),
    }
    if normalize_dispatch_flag(dispatches[CONTROL]) != normalize_dispatch_flag(
        dispatches[PROBE]
    ):
        raise RuntimeError("matched dispatch gate failed")

    output.mkdir(parents=True)
    try:
        modes: dict[str, Any] = {}
        for mode in MODES:
            root, kernel_path, dispatch_path = overlay_paths(output, mode)
            root.mkdir(parents=True)
            kernel_path.write_text(kernel, encoding="utf-8")
            dispatch_path.write_text(dispatches[mode], encoding="utf-8")
            kernel_diff = diff(
                kernel_source,
                kernel,
                left_name="latest-opt/moe_dynamic_kernel_opt.py",
                right_name=f"{mode}/moe_dynamic_kernel.py",
            )
            dispatch_diff = diff(
                dispatch_source,
                dispatches[mode],
                left_name="production/moe_dispatch.py",
                right_name=f"{mode}/moe_dispatch.py",
            )
            (root / "moe_dynamic_kernel.diff").write_text(kernel_diff, encoding="utf-8")
            (root / "moe_dispatch.diff").write_text(dispatch_diff, encoding="utf-8")
            manifest = {
                "schema": "exp017.opt-phase-overlay.v1",
                "mode": mode,
                "probe_enabled": mode == PROBE,
                "classification": (
                    "diagnostic probe" if mode == PROBE else "marker-disabled control"
                ),
                "event_abi": EVENT_ABI,
                "base": {
                    "kernel_path": str(source_paths["kernel"]),
                    "kernel_sha256": EXPECTED_OPT_SHA256,
                    "dispatch_path": str(source_paths["dispatch"]),
                    "dispatch_sha256": EXPECTED_DISPATCH_SHA256,
                    "wrapper_path": str(source_paths["wrapper"]),
                    "wrapper_sha256": EXPECTED_WRAPPER_SHA256,
                    "barrier_fingerprint": barrier_fingerprint(kernel_source),
                },
                "overlay": {
                    "kernel_sha256": file_sha256(kernel_path),
                    "dispatch_sha256": file_sha256(dispatch_path),
                    "kernel_diff_sha256": text_sha256(kernel_diff),
                    "dispatch_diff_sha256": text_sha256(dispatch_diff),
                    "barrier_fingerprint": barrier_fingerprint(kernel),
                },
                "boundary": {
                    "timer_read_call_sites": kernel.count("_exp017_read_globaltimer()"),
                    "phase_names": list(PHASE_NAMES),
                    "new_barriers": 0,
                },
            }
            write_json(root / "identity.json", manifest)
            modes[mode] = manifest
        identity = {
            "schema": "exp017.opt-phase-overlays.v1",
            "event_abi": EVENT_ABI,
            "modes": modes,
            "cross_mode": {
                "kernel_source_byte_identical": file_sha256(
                    overlay_paths(output, CONTROL)[1]
                )
                == file_sha256(overlay_paths(output, PROBE)[1]),
                "normalized_dispatch_byte_identical": canonical_sha256(
                    normalize_dispatch_flag(dispatches[CONTROL])
                )
                == canonical_sha256(normalize_dispatch_flag(dispatches[PROBE])),
                "only_compile_time_enable_flag_differs": True,
                "fresh_jit_and_resource_audit_required_per_mode": True,
            },
        }
        if not all(identity["cross_mode"].values()):
            raise RuntimeError("matched control/probe identity gate failed")
        write_json(output / "identity.json", identity)
        return verify_existing(repo, output)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, default=ROOT.parents[3])
    parser.add_argument("--output", type=Path, default=OVERLAY_ROOT)
    parser.add_argument("--check-existing", action="store_true")
    args = parser.parse_args()
    result = (
        verify_existing(args.flashinfer_root.resolve(), args.output.resolve())
        if args.check_existing
        else build(args.flashinfer_root.resolve(), args.output.resolve())
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
