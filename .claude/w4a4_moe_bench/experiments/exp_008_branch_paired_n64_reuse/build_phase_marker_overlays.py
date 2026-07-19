#!/usr/bin/env python3
"""Build immutable exp_008 marker-disabled/control and enabled/probe overlays.

For each v0/v1 kernel, control and probe share byte-identical instrumented
kernel source.  Only one dispatch constexpr differs, and that constexpr is in
the dynamic-kernel cache key.  The transform adds no synchronization call.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import difflib
from pathlib import Path
import shutil
import sys
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parent
EXP004 = ROOT.parent / "exp_004_fused_phase_timing_breakdown"
if str(EXP004) not in sys.path:
    sys.path.insert(0, str(EXP004))

import build_whole_kernel_probe as exp004_whole  # noqa: E402

from exp008_marker_common import (  # noqa: E402
    BASE_KERNEL,
    COMPUTE_WARPS,
    CONTROL,
    CTA_CALIBRATION,
    CTA_ENTRY,
    CTA_LOOP_EXIT,
    CTA_LOOP_START,
    CTA_TICKS,
    CTA_W8_FINAL,
    DISPATCH_RELATIVE_PATH,
    EVENT_ABI,
    EXPECTED_BASE_KERNEL_SHA256,
    EXPECTED_DISPATCH_SHA256,
    EXPECTED_WRAPPER_SHA256,
    MARKER_ARMS,
    OVERLAY_ROOT,
    PROBE,
    TASK_CLAIM_START,
    TASK_MAIN,
    TASK_PAIR,
    TASK_TICKS,
    VERSIONS,
    WRAPPER_RELATIVE_PATH,
    barrier_fingerprint,
    canonical_sha256,
    sha256_file,
    write_json,
)


KERNEL_CONSTANTS = f"""_FC2_TILE_RECIP_GS_NUM = 6.0 * 448.0

# exp_008 experiment-owned phase marker ABI.  The enabled specialization is
# diagnostic only; the disabled specialization retains the same host ABI.
_EXP008_COMPUTE_WARPS = {COMPUTE_WARPS}
_EXP008_TASK_TICKS = {TASK_TICKS}
_EXP008_CTA_TICKS = {CTA_TICKS}


@dsl_user_op
def _exp008_read_globaltimer(*, loc=None, ip=None):
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
def _exp008_st_shared_u64(addr, value, *, loc=None, ip=None):
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
def _exp008_ld_shared_volatile_u64(addr, *, loc=None, ip=None):
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
def _exp008_ld_shared_volatile_i32(addr, *, loc=None, ip=None):
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


def _replace_exact(
    text: str, old: str, new: str, *, label: str, expected: int = 1
) -> str:
    count = text.count(old)
    if count != expected:
        raise ValueError(f"{label}: expected {expected} exact matches, found {count}")
    return text.replace(old, new, expected)


def _task_store(event: str, *, indent: int, tick: str | None = None) -> str:
    pad = " " * indent
    lines = ""
    value = tick
    if value is None:
        value = "exp008_tick"
        lines += f"{pad}{value} = _exp008_read_globaltimer()\n"
    lines += (
        f"{pad}st_global_u64(\n"
        f"{pad}    get_ptr_as_int64(\n"
        f"{pad}        exp008_timing_ticks,\n"
        f"{pad}        task_slot_probe * Int32(_EXP008_TASK_TICKS)\n"
        f"{pad}        + Int32({event}),\n"
        f"{pad}    ),\n"
        f"{pad}    {value},\n"
        f"{pad})\n"
    )
    return lines


def _cta_store(event: str, *, indent: int, tick: str | None = None) -> str:
    pad = " " * indent
    lines = ""
    value = tick
    if value is None:
        value = "exp008_tick"
        lines += f"{pad}{value} = _exp008_read_globaltimer()\n"
    lines += (
        f"{pad}st_global_u64(\n"
        f"{pad}    get_ptr_as_int64(\n"
        f"{pad}        exp008_cta_ticks,\n"
        f"{pad}        Int32(bidz) * Int32(_EXP008_CTA_TICKS)\n"
        f"{pad}        + Int32({event}),\n"
        f"{pad}    ),\n"
        f"{pad}    {value},\n"
        f"{pad})\n"
    )
    return lines


def _reload_task_slot(*, indent: int) -> str:
    pad = " " * indent
    return (
        f"{pad}task_slot_probe = _exp008_ld_shared_volatile_i32(\n"
        f"{pad}    ctrl_base_addr + Int32(28)\n"
        f"{pad})\n"
    )


def _reload_warp_saved_task_slot(*, indent: int) -> str:
    pad = " " * indent
    return (
        f"{pad}task_slot_probe = _exp008_ld_shared_volatile_i32(\n"
        f"{pad}    route_phys_rows_addr + Int32(8)\n"
        f"{pad}    + warp_idx * Int32(4)\n"
        f"{pad})\n"
    )


def _add_kernel_abi(source: str) -> str:
    text = _replace_exact(
        source,
        "_FC2_TILE_RECIP_GS_NUM = 6.0 * 448.0\n",
        KERNEL_CONSTANTS,
        label="globaltimer helpers",
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
    call_abi = (
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "        max_active_clusters: cutlass.Constexpr,\n"
    )
    text = _replace_exact(
        text,
        call_abi,
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "        exp008_timing_ticks: cute.Tensor,\n"
        "        exp008_task_cta_z: cute.Tensor,\n"
        "        exp008_cta_ticks: cute.Tensor,\n"
        "        max_active_clusters: cutlass.Constexpr,\n",
        label="host-call marker ABI",
    )
    text = _replace_exact(
        text,
        "            token_map,\n            token_weights,\n        ).launch(\n",
        "            token_map,\n"
        "            token_weights,\n"
        "            exp008_timing_ticks,\n"
        "            exp008_task_cta_z,\n"
        "            exp008_cta_ticks,\n"
        "        ).launch(\n",
        label="device launch marker args",
    )
    kernel_abi = (
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "    ):\n"
        '        """Kernel entry point."""\n'
    )
    text = _replace_exact(
        text,
        kernel_abi,
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "        exp008_timing_ticks: cute.Tensor,\n"
        "        exp008_task_cta_z: cute.Tensor,\n"
        "        exp008_cta_ticks: cute.Tensor,\n"
        "    ):\n"
        '        """Kernel entry point."""\n',
        label="device kernel marker ABI",
    )
    return text


def _instrument_kernel(source: str) -> str:
    original_barriers = barrier_fingerprint(source)
    text = _add_kernel_abi(source)

    entry_anchor = (
        "        is_cta_leader = Int32(1) if Int32(tidx) == Int32(0) else Int32(0)\n\n"
        "        if warp_idx == 0:\n"
    )
    entry_marker = (
        "        is_cta_leader = Int32(1) if Int32(tidx) == Int32(0) else Int32(0)\n"
        "        exp008_lane_id = Int32(tidx) & Int32(31)\n"
        "        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "            if warp_idx < Int32(_EXP008_COMPUTE_WARPS):\n"
        "                if exp008_lane_id == Int32(0):\n"
        + _cta_store(f"Int32({CTA_ENTRY}) + warp_idx", indent=20)
        + "\n        if warp_idx == 0:\n"
    )
    text = _replace_exact(text, entry_anchor, entry_marker, label="W0-W7 kernel entry")

    loop_anchor = (
        "        consumer_live = Int32(1)\n"
        "        while consumer_live > Int32(0):\n"
        "            if is_cta_leader > Int32(0):\n"
    )
    loop_marker = (
        "        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "            if warp_idx < Int32(_EXP008_COMPUTE_WARPS):\n"
        "                if exp008_lane_id == Int32(0):\n"
        + _cta_store(f"Int32({CTA_LOOP_START}) + warp_idx", indent=20)
        + "            if is_cta_leader > Int32(0):\n"
        + _cta_store(str(CTA_CALIBRATION), indent=16)
        + _cta_store(str(CTA_CALIBRATION + 1), indent=16)
        + "        consumer_live = Int32(1)\n"
        "        while consumer_live > Int32(0):\n"
        "            if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                if is_cta_leader > Int32(0):\n"
        "                    exp008_claim_start = _exp008_read_globaltimer()\n"
        "                    _exp008_st_shared_u64(\n"
        "                        route_phys_rows_addr, exp008_claim_start\n"
        "                    )\n"
        "            if is_cta_leader > Int32(0):\n"
    )
    text = _replace_exact(
        text, loop_anchor, loop_marker, label="loop start/calibration/claim start"
    )

    claim_anchor = (
        "            has_task = _ld_shared_i32(ctrl_base_addr + Int32(0))\n"
        "            is_done = _ld_shared_i32(ctrl_base_addr + Int32(4))\n"
    )
    claim_marker = (
        claim_anchor + "            if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                if has_task > Int32(0):\n"
        "                    if is_cta_leader > Int32(0):\n"
        + _reload_task_slot(indent=24)
        + "                        exp008_claim_start = (\n"
        "                            _exp008_ld_shared_volatile_u64(\n"
        "                                route_phys_rows_addr\n"
        "                            )\n"
        "                        )\n"
        + _task_store(str(TASK_CLAIM_START), indent=24, tick="exp008_claim_start")
        + "                        st_global_i32(\n"
        "                            get_ptr_as_int64(\n"
        "                                exp008_task_cta_z, task_slot_probe\n"
        "                            ),\n"
        "                            Int32(bidz),\n"
        "                        )\n"
    )
    text = _replace_exact(
        text, claim_anchor, claim_marker, label="leader claim diagnostic mapping"
    )

    cache_anchor = (
        "                    cache_row += Int32(self.threads_per_cta)\n"
        "                cute.arch.sync_threads()\n"
    )
    cache_marker = (
        cache_anchor
        + "                if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                    if warp_idx < Int32(_EXP008_COMPUTE_WARPS):\n"
        "                        if exp008_lane_id == Int32(0):\n"
        + _reload_task_slot(indent=28)
        + _task_store(f"Int32({TASK_MAIN}) + warp_idx", indent=28)
    )
    text = _replace_exact(
        text, cache_anchor, cache_marker, label="collective cache-ready boundary"
    )

    pair_begin_anchor = "                        gate_acc.fill(0.0)\n"
    pair_begin_marker = (
        "                        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                            if exp008_lane_id == Int32(0):\n"
        + _reload_task_slot(indent=32)
        + _task_store(
            f"Int32({TASK_PAIR}) + (Int32(fc1_half) * Int32(3)) * "
            "Int32(_EXP008_COMPUTE_WARPS) + warp_idx",
            indent=32,
        )
        + pair_begin_anchor
    )
    text = _replace_exact(
        text, pair_begin_anchor, pair_begin_marker, label="half FC1 begin"
    )

    pair_end_anchor = "                        if fc1_half == 1:\n"
    pair_end_marker = (
        "                        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                            if exp008_lane_id == Int32(0):\n"
        + _reload_task_slot(indent=32)
        + _task_store(
            f"Int32({TASK_PAIR}) + (Int32(fc1_half) * Int32(3) + Int32(1)) * "
            "Int32(_EXP008_COMPUTE_WARPS) + warp_idx",
            indent=32,
        )
        + pair_end_anchor
    )
    text = _replace_exact(text, pair_end_anchor, pair_end_marker, label="half FC1 end")

    activation_end_anchor = (
        "                                    ],\n"
        "                                )\n\n"
        "                    # Both disjoint N64 activations are now durable in the full\n"
    )
    activation_end_marker = (
        "                                    ],\n"
        "                                )\n"
        "                        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                            if exp008_lane_id == Int32(0):\n"
        + _reload_task_slot(indent=32)
        + _task_store(
            f"Int32({TASK_PAIR}) + (Int32(fc1_half) * Int32(3) + Int32(2)) * "
            "Int32(_EXP008_COMPUTE_WARPS) + warp_idx",
            indent=32,
        )
        + "\n                    # Both disjoint N64 activations are now durable in the full\n"
    )
    text = _replace_exact(
        text,
        activation_end_anchor,
        activation_end_marker,
        label="half activation end",
    )

    fc1_collective_anchor = (
        "                    # N128 sC tile before the single Q1 pass reads them.\n"
        '                    cute.arch.fence_proxy("async.shared", space="cta")\n'
        "                    self.epilog_sync_barrier.arrive_and_wait()\n\n"
        "                    # Q1 runs exactly once after both N64 halves.  sA remains\n"
    )
    fc1_collective_marker = (
        "                    # N128 sC tile before the single Q1 pass reads them.\n"
        '                    cute.arch.fence_proxy("async.shared", space="cta")\n'
        "                    self.epilog_sync_barrier.arrive_and_wait()\n"
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if exp008_lane_id == Int32(0):\n"
        + _reload_task_slot(indent=28)
        + _task_store(f"Int32({TASK_MAIN + COMPUTE_WARPS}) + warp_idx", indent=28)
        + "\n                    # Q1 runs exactly once after both N64 halves.  sA remains\n"
    )
    text = _replace_exact(
        text,
        fc1_collective_anchor,
        fc1_collective_marker,
        label="FC1+interleaved activation collective end",
    )

    q1_anchor = (
        '                    cute.arch.fence_proxy("async.shared", space="cta")\n'
        "                    self.epilog_sync_barrier.arrive_and_wait()\n\n"
        "                    # ============================================================\n"
        "                    # PHASE B: Sweep ALL FC2 output tiles using cached sA\n"
    )
    q1_marker = (
        '                    cute.arch.fence_proxy("async.shared", space="cta")\n'
        "                    self.epilog_sync_barrier.arrive_and_wait()\n"
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if exp008_lane_id == Int32(0):\n"
        + _reload_task_slot(indent=28)
        + _task_store(f"Int32({TASK_MAIN + 2 * COMPUTE_WARPS}) + warp_idx", indent=28)
        + "                            _st_shared_i32(\n"
        "                                route_phys_rows_addr + Int32(8)\n"
        "                                + warp_idx * Int32(4),\n"
        "                                task_slot_probe,\n"
        "                            )\n"
        + "\n                    # ============================================================\n"
        "                    # PHASE B: Sweep ALL FC2 output tiles using cached sA\n"
    )
    text = _replace_exact(text, q1_anchor, q1_marker, label="Q1 collective end")

    done_anchor = (
        "                    self.pass_final_barrier.arrive_unaligned()\n"
        "                    slice_idx += Int32(1)\n"
    )
    done_marker = (
        "                    self.pass_final_barrier.arrive_unaligned()\n"
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if exp008_lane_id == Int32(0):\n"
        # Reload the per-warp copy saved at Q1.  A fresh ctrl[28] reload races
        # W0 starting the next scheduler loop; the dead route scratch offsets
        # are warp-private and are not touched by the next task's claim stamp.
        + _reload_warp_saved_task_slot(indent=28)
        + _task_store(f"Int32({TASK_MAIN + 3 * COMPUTE_WARPS}) + warp_idx", indent=28)
        + "                    slice_idx += Int32(1)\n"
    )
    text = _replace_exact(
        text, done_anchor, done_marker, label="combined FC2+scatter end"
    )

    exit_anchor = (
        "        if warp_idx == self.tma_load_warp_id:\n"
        "            ml_pipeline.producer_tail(prod_state)\n"
        "            phase2_pipeline.producer_tail(phase2_prod_state)\n"
        "        return\n"
    )
    exit_marker = (
        "        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "            if warp_idx < Int32(_EXP008_COMPUTE_WARPS):\n"
        "                if exp008_lane_id == Int32(0):\n"
        + _cta_store(f"Int32({CTA_LOOP_EXIT}) + warp_idx", indent=20)
        + "        if warp_idx == self.tma_load_warp_id:\n"
        "            ml_pipeline.producer_tail(prod_state)\n"
        "            phase2_pipeline.producer_tail(phase2_prod_state)\n"
        "            if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                if exp008_lane_id == Int32(0):\n"
        + _cta_store(str(CTA_W8_FINAL), indent=20)
        + "        return\n"
    )
    text = _replace_exact(
        text, exit_anchor, exit_marker, label="W0-W7 exit and W8 terminal"
    )

    if barrier_fingerprint(text) != original_barriers:
        raise ValueError("instrumentation changed the source barrier fingerprint")
    ast.parse(text, filename="exp008_marker/moe_dynamic_kernel.py")
    return text


@contextmanager
def _exp004_dispatch_abi() -> Iterator[None]:
    old_task_ticks = exp004_whole.TASK_TICKS
    old_cta_ticks = exp004_whole.CTA_TICKS
    try:
        exp004_whole.TASK_TICKS = TASK_TICKS
        exp004_whole.CTA_TICKS = CTA_TICKS
        yield
    finally:
        exp004_whole.TASK_TICKS = old_task_ticks
        exp004_whole.CTA_TICKS = old_cta_ticks


def _instrument_dispatch(source: str, *, enabled: bool) -> str:
    # Reuse the exp004/006 audited dynamic-workspace/JIT plumbing, then give
    # the experiment-owned ABI an exp008 namespace.
    with _exp004_dispatch_abi():
        text = exp004_whole._instrument_dispatch(source, enabled=enabled)
    text = text.replace("_EXP004", "_EXP008").replace("exp004_", "exp008_")
    if f"_EXP008_TIMING_TICKS_PER_TASK = {TASK_TICKS}" not in text:
        raise ValueError("exp008 task timing capacity was not installed")
    if f"_EXP008_CTA_TICKS_PER_CTA = {CTA_TICKS}" not in text:
        raise ValueError("exp008 CTA timing capacity was not installed")
    if f"_EXP008_PHASE_PROBE_ENABLED = {enabled!r}" not in text:
        raise ValueError("exp008 compile-time marker flag was not installed")
    ast.parse(text, filename="exp008_marker/moe_dispatch.py")
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


def _normalized_dispatch(text: str) -> str:
    return text.replace(
        "_EXP008_PHASE_PROBE_ENABLED = True",
        "_EXP008_PHASE_PROBE_ENABLED = <FLAG>",
    ).replace(
        "_EXP008_PHASE_PROBE_ENABLED = False",
        "_EXP008_PHASE_PROBE_ENABLED = <FLAG>",
    )


def build_version(repo: Path, output_root: Path, version: str) -> dict[str, Any]:
    if version not in VERSIONS:
        raise ValueError(version)
    base_kernel_path = BASE_KERNEL[version]
    dispatch_path = repo / DISPATCH_RELATIVE_PATH
    wrapper_path = repo / WRAPPER_RELATIVE_PATH
    identities = {
        "base_kernel": (base_kernel_path, EXPECTED_BASE_KERNEL_SHA256[version]),
        "dispatch": (dispatch_path, EXPECTED_DISPATCH_SHA256),
        "wrapper": (wrapper_path, EXPECTED_WRAPPER_SHA256),
    }
    for label, (path, expected) in identities.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{label} source drift: {actual} != {expected}: {path}")

    base_kernel = base_kernel_path.read_text()
    base_dispatch = dispatch_path.read_text()
    kernel = _instrument_kernel(base_kernel)
    dispatches = {
        CONTROL: _instrument_dispatch(base_dispatch, enabled=False),
        PROBE: _instrument_dispatch(base_dispatch, enabled=True),
    }
    if _normalized_dispatch(dispatches[CONTROL]) != _normalized_dispatch(
        dispatches[PROBE]
    ):
        raise ValueError("control/probe dispatches differ beyond the one constexpr")

    version_root = output_root / version
    if version_root.exists():
        raise FileExistsError(f"immutable marker overlay exists: {version_root}")
    manifests: dict[str, Any] = {}
    for arm in MARKER_ARMS:
        arm_root = version_root / arm
        arm_root.mkdir(parents=True)
        kernel_path = arm_root / "moe_dynamic_kernel.py"
        marker_dispatch_path = arm_root / "moe_dispatch.py"
        kernel_path.write_text(kernel)
        marker_dispatch_path.write_text(dispatches[arm])
        (arm_root / "moe_dynamic_kernel.diff").write_text(
            _diff(
                base_kernel,
                kernel,
                left_name=f"{version}/base/moe_dynamic_kernel.py",
                right_name=f"{version}/{arm}/moe_dynamic_kernel.py",
            )
        )
        (arm_root / "moe_dispatch.diff").write_text(
            _diff(
                base_dispatch,
                dispatches[arm],
                left_name="production/moe_dispatch.py",
                right_name=f"{version}/{arm}/moe_dispatch.py",
            )
        )
        manifest = {
            "schema": "exp008.phase-marker-overlay.v1",
            "version": version,
            "arm": arm,
            "probe_enabled": arm == PROBE,
            "classification": "diagnostic-only" if arm == PROBE else "control",
            "event_abi": EVENT_ABI,
            "base": {
                "kernel_path": str(base_kernel_path),
                "kernel_sha256": EXPECTED_BASE_KERNEL_SHA256[version],
                "dispatch_path": str(dispatch_path),
                "dispatch_sha256": EXPECTED_DISPATCH_SHA256,
                "wrapper_sha256": EXPECTED_WRAPPER_SHA256,
                "barrier_fingerprint": barrier_fingerprint(base_kernel),
            },
            "overlay": {
                "kernel_sha256": sha256_file(kernel_path),
                "dispatch_sha256": sha256_file(marker_dispatch_path),
                "normalized_dispatch_sha256": canonical_sha256(
                    _normalized_dispatch(dispatches[arm])
                ),
                "barrier_fingerprint": barrier_fingerprint(kernel),
            },
            "contracts": {
                "kernel_source_shared_between_control_probe": True,
                "dispatch_difference": "_EXP008_PHASE_PROBE_ENABLED only",
                "jit_cache_key_contains_probe_flag": True,
                "new_source_barriers": 0,
                "w8_scope": "terminal completion anchor only; no producer phase track",
            },
        }
        write_json(arm_root / "identity.json", manifest)
        manifests[arm] = manifest

    cross_arm = {
        "kernel_sha256_equal": manifests[CONTROL]["overlay"]["kernel_sha256"]
        == manifests[PROBE]["overlay"]["kernel_sha256"],
        "normalized_dispatch_sha256_equal": canonical_sha256(
            _normalized_dispatch(dispatches[CONTROL])
        )
        == canonical_sha256(_normalized_dispatch(dispatches[PROBE])),
    }
    if not all(cross_arm.values()):
        raise AssertionError(cross_arm)
    summary = {
        "schema": "exp008.phase-marker-version-overlays.v1",
        "version": version,
        "event_abi": EVENT_ABI,
        "arms": manifests,
        "cross_arm": {**cross_arm, "gate_pass": True},
    }
    write_json(version_root / "identity.json", summary)
    return summary


def build_all(repo: Path, output_root: Path) -> dict[str, Any]:
    repo = repo.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"immutable marker output exists: {output_root}")
    output_root.mkdir(parents=True)
    try:
        versions = {
            version: build_version(repo, output_root, version) for version in VERSIONS
        }
    except Exception:
        shutil.rmtree(output_root)
        raise
    summary = {
        "schema": "exp008.phase-marker-overlays.v1",
        "event_abi": EVENT_ABI,
        "versions": versions,
        "gate_pass": all(
            value["cross_arm"]["gate_pass"] for value in versions.values()
        ),
    }
    write_json(output_root / "identity.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=ROOT.parents[3], help="FlashInfer checkout root"
    )
    parser.add_argument("--output", type=Path, default=OVERLAY_ROOT)
    args = parser.parse_args()
    result = build_all(args.repo, args.output)
    print(result["schema"], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
