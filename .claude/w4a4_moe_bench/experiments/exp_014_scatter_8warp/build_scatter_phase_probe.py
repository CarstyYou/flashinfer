#!/usr/bin/env python3
"""Build matched baseline/candidate `%globaltimer` Scatter probe overlays."""

from __future__ import annotations

import argparse
import ast
import difflib
import json
from pathlib import Path
import shutil
from typing import Any

from exp014_scatter_probe_common import (
    ARMS,
    BASELINE,
    BASE_OVERLAY_ROOT,
    CANDIDATE,
    COMPUTE_WARPS,
    DISPATCH_RELATIVE_PATH,
    EVENT_ABI,
    EXPECTED_BASE_KERNEL_SHA256,
    EXPECTED_DISPATCH_SHA256,
    EXPECTED_WRAPPER_SHA256,
    OUTPUT_TILES,
    PROBE_OVERLAY_ROOT,
    SAMPLED_TASK_SLOTS,
    TASK_TICKS,
    TICKS_PER_TILE,
    WRAPPER_RELATIVE_PATH,
    barrier_fingerprint,
    canonical_sha256,
    file_sha256,
    text_sha256,
    write_json,
)


ROOT = Path(__file__).resolve().parent

KERNEL_CONSTANTS = f"""_FC2_TILE_RECIP_GS_NUM = 6.0 * 448.0

# exp_014 diagnostic Scatter phase marker ABI. The probe adds no barrier.
_EXP014_SCATTER_COMPUTE_WARPS = {COMPUTE_WARPS}
_EXP014_SCATTER_OUTPUT_TILES = {OUTPUT_TILES}
_EXP014_SCATTER_TICKS_PER_TILE = {TICKS_PER_TILE}
_EXP014_SCATTER_TICKS_PER_TASK = {TASK_TICKS}
_EXP014_SCATTER_SAMPLE_TASK = {SAMPLED_TASK_SLOTS[0]}


@dsl_user_op
def _exp014_read_globaltimer(*, loc=None, ip=None):
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
def _exp014_ld_shared_volatile_i32(addr, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.s32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


"""

BASELINE_MAPPING = (
    "warp_m_base = (warp_in_tile >> Int32(1)) * Int32(64)",
    "# Per-warp scatter: each warp scatters its own quadrant\n"
    "            # of sC (64 M-rows x 64 N-cols).",
    "if warp_epi_rows > Int32(64):\n                warp_epi_rows = Int32(64)",
)
CANDIDATE_MAPPING = (
    "warp_m_base = (warp_in_tile >> Int32(1)) * Int32(32)",
    "# Per-warp scatter: all eight math warps cover one disjoint\n"
    "            # sC strip (32 M-rows x 64 N-cols).",
    "if warp_epi_rows > Int32(32):\n                warp_epi_rows = Int32(32)",
)


def replace_exact(
    text: str, old: str, new: str, *, label: str, expected: int = 1
) -> str:
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{label}: expected {expected} anchors, found {count}")
    return text.replace(old, new, expected)


def marker_store(edge: int, *, indent: int) -> str:
    pad = " " * indent
    body = " " * (indent + 4)
    return (
        f"{pad}if task_slot_probe == Int32(_EXP014_SCATTER_SAMPLE_TASK):\n"
        f"{body}exp014_scatter_tick = _exp014_read_globaltimer()\n"
        f"{body}st_global_u64(\n"
        f"{body}    get_ptr_as_int64(\n"
        f"{body}        exp014_scatter_ticks,\n"
        f"{body}        task_slot_probe * Int32(_EXP014_SCATTER_TICKS_PER_TASK)\n"
        f"{body}        + output_tile_idx * Int32(_EXP014_SCATTER_TICKS_PER_TILE)\n"
        f"{body}        + Int32({edge * COMPUTE_WARPS}) + warp_idx,\n"
        f"{body}    ),\n"
        f"{body}    exp014_scatter_tick,\n"
        f"{body})\n"
    )


def instrument_kernel(source: str) -> str:
    original_barriers = barrier_fingerprint(source)
    text = replace_exact(
        source,
        "_FC2_TILE_RECIP_GS_NUM = 6.0 * 448.0\n",
        KERNEL_CONSTANTS,
        label="globaltimer helper and ABI constants",
    )
    text = replace_exact(
        text,
        "        share_input_across_experts: bool = False,\n    ):\n",
        "        share_input_across_experts: bool = False,\n"
        "        scatter_phase_probe_enabled: bool = False,\n"
        "    ):\n",
        label="constructor probe flag",
    )
    text = replace_exact(
        text,
        "        self.share_input_across_experts = share_input_across_experts\n",
        "        self.share_input_across_experts = share_input_across_experts\n"
        "        self.scatter_phase_probe_enabled = bool(scatter_phase_probe_enabled)\n",
        label="constructor probe flag assignment",
    )
    text = replace_exact(
        text,
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "        max_active_clusters: cutlass.Constexpr,\n",
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "        exp014_scatter_ticks: cute.Tensor,\n"
        "        max_active_clusters: cutlass.Constexpr,\n",
        label="host-call probe ABI",
    )
    text = replace_exact(
        text,
        "            token_map,\n            token_weights,\n        ).launch(\n",
        "            token_map,\n"
        "            token_weights,\n"
        "            exp014_scatter_ticks,\n"
        "        ).launch(\n",
        label="device launch probe argument",
    )
    text = replace_exact(
        text,
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "    ):\n"
        '        """Kernel entry point."""\n',
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "        exp014_scatter_ticks: cute.Tensor,\n"
        "    ):\n"
        '        """Kernel entry point."""\n',
        label="device kernel probe ABI",
    )
    text = replace_exact(
        text,
        "        warp_idx = cute.arch.make_warp_uniform(warp_idx)\n"
        "        is_cta_leader = Int32(1) if Int32(tidx) == Int32(0) else Int32(0)\n",
        "        warp_idx = cute.arch.make_warp_uniform(warp_idx)\n"
        "        lane_id = Int32(tidx) & Int32(31)\n"
        "        is_cta_leader = Int32(1) if Int32(tidx) == Int32(0) else Int32(0)\n",
        label="lane identity",
    )
    text = replace_exact(
        text,
        "            elif warp_idx < self.num_mma_warps:\n"
        "                task_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(8))\n",
        "            elif warp_idx < self.num_mma_warps:\n"
        "                task_slot_probe = Int32(0)\n"
        "                if cutlass.const_expr(self.scatter_phase_probe_enabled):\n"
        "                    task_slot_probe = _exp014_ld_shared_volatile_i32(\n"
        "                        ctrl_base_addr + Int32(28)\n"
        "                    )\n"
        "                task_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(8))\n",
        label="claimed task slot",
    )

    scatter_anchor = (
        '                        cute.arch.fence_proxy("async.shared", space="cta")\n'
        "                        self.epilog_sync_barrier.arrive_and_wait()\n\n"
        "                        self.scatter_sC_to_gmem(\n"
        "                            tidx,\n"
        "                            output_tile_idx,\n"
        "                            valid_rows,\n"
        "                            sC,\n"
        "                            tRS_sD,\n"
        "                            scatter_output,\n"
        "                            scatter_tok_base_addr,\n"
        "                            scatter_weight_base_addr,\n"
        "                        )\n\n"
        "                        # Finish this tile's scatter before the next output\n"
        "                        # tile begins collective phase2 pipeline operations.\n"
        "                        self.epilog_sync_barrier.arrive_and_wait()\n"
    )
    scatter_probe = (
        '                        cute.arch.fence_proxy("async.shared", space="cta")\n'
        "                        self.epilog_sync_barrier.arrive_and_wait()\n"
        "                        if cutlass.const_expr(\n"
        "                            self.scatter_phase_probe_enabled\n"
        "                        ):\n"
        "                            if lane_id == Int32(0):\n"
        + marker_store(0, indent=32)
        + "\n                        self.scatter_sC_to_gmem(\n"
        "                            tidx,\n"
        "                            output_tile_idx,\n"
        "                            valid_rows,\n"
        "                            sC,\n"
        "                            tRS_sD,\n"
        "                            scatter_output,\n"
        "                            scatter_tok_base_addr,\n"
        "                            scatter_weight_base_addr,\n"
        "                        )\n"
        "                        if cutlass.const_expr(\n"
        "                            self.scatter_phase_probe_enabled\n"
        "                        ):\n"
        "                            if lane_id == Int32(0):\n"
        + marker_store(1, indent=32)
        + "\n                        # Finish this tile's scatter before the next output\n"
        "                        # tile begins collective phase2 pipeline operations.\n"
        "                        self.epilog_sync_barrier.arrive_and_wait()\n"
        "                        if cutlass.const_expr(\n"
        "                            self.scatter_phase_probe_enabled\n"
        "                        ):\n"
        "                            if lane_id == Int32(0):\n"
        + marker_store(2, indent=32)
    )
    text = replace_exact(
        text, scatter_anchor, scatter_probe, label="D/E/F Scatter boundaries"
    )

    if barrier_fingerprint(text) != original_barriers:
        raise RuntimeError("probe instrumentation changed the barrier fingerprint")
    if text.count("_exp014_read_globaltimer()") != 3:
        raise RuntimeError("probe must contain exactly three timer read call sites")
    timing_abi_count = text.count("exp014_scatter_ticks,")
    if timing_abi_count != 4:
        raise RuntimeError(f"kernel timing ABI/store count drift: {timing_abi_count}")
    ast.parse(text, filename="exp014_scatter_probe/moe_dynamic_kernel.py")
    return text


def instrument_dispatch(source: str) -> str:
    text = replace_exact(
        source,
        "_DYNAMIC_SLICE_CHUNK = 1\n",
        "_DYNAMIC_SLICE_CHUNK = 1\n"
        "_EXP014_SCATTER_PHASE_PROBE_ENABLED = True\n"
        f"_EXP014_SCATTER_OUTPUT_TILES = {OUTPUT_TILES}\n"
        f"_EXP014_SCATTER_TICKS_PER_TASK = {TASK_TICKS}\n",
        label="dispatch probe constants",
    )
    text = replace_exact(
        text,
        "    task_valid_rows: torch.Tensor\n"
        "    tile_write_count: torch.Tensor\n\n"
        "    # Views\n",
        "    task_valid_rows: torch.Tensor\n"
        "    tile_write_count: torch.Tensor\n"
        "    exp014_scatter_ticks: torch.Tensor\n\n"
        "    # Views\n",
        label="workspace timing field",
    )
    text = replace_exact(
        text,
        "        tile_write_count=torch.zeros(physical_tiles, dtype=torch.int32, device=device),\n"
        "    )\n",
        "        tile_write_count=torch.zeros(physical_tiles, dtype=torch.int32, device=device),\n"
        "        exp014_scatter_ticks=torch.full(\n"
        "            (max_tasks * _EXP014_SCATTER_TICKS_PER_TASK,),\n"
        "            -1, dtype=torch.int64, device=device,\n"
        "        ),\n"
        "    )\n",
        label="workspace timing allocation",
    )
    text = replace_exact(
        text,
        "        tile_write_count_ptr: cute.Pointer,\n        b_w13: cute.Tensor,\n",
        "        tile_write_count_ptr: cute.Pointer,\n"
        "        exp014_scatter_ticks_ptr: cute.Pointer,\n"
        "        b_w13: cute.Tensor,\n",
        label="dynamic launch timing pointer",
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
        "        exp014_scatter_ticks = cute.make_tensor(\n"
        "            exp014_scatter_ticks_ptr,\n"
        "            layout=cute.make_layout(\n"
        "                (max_tasks * _EXP014_SCATTER_TICKS_PER_TASK,), stride=(1,)\n"
        "            ),\n"
        "        )\n"
        "        self._kernel(\n",
        label="dynamic launch timing tensor",
    )
    text = replace_exact(
        text,
        "            token_map,\n"
        "            token_weights_t,\n"
        "            max_active_clusters=max_active_clusters,\n",
        "            token_map,\n"
        "            token_weights_t,\n"
        "            exp014_scatter_ticks,\n"
        "            max_active_clusters=max_active_clusters,\n",
        label="kernel timing argument",
    )
    text = replace_exact(
        text,
        "        share_input_across_experts=share_input_across_experts,\n"
        "    )\n"
        "    launch = _DynamicMoELaunch(\n",
        "        share_input_across_experts=share_input_across_experts,\n"
        "        scatter_phase_probe_enabled=_EXP014_SCATTER_PHASE_PROBE_ENABLED,\n"
        "    )\n"
        "    launch = _DynamicMoELaunch(\n",
        label="kernel probe flag",
    )
    text = replace_exact(
        text,
        "        share_input_across_experts,\n"
        "    )\n"
        "    cached = _DYNAMIC_KERNEL_CACHE.get(cache_key)\n",
        "        share_input_across_experts,\n"
        "        _EXP014_SCATTER_PHASE_PROBE_ENABLED,\n"
        "    )\n"
        "    cached = _DYNAMIC_KERNEL_CACHE.get(cache_key)\n",
        label="JIT cache probe identity",
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
        "    exp014_scatter_ticks_fake = make_ptr(\n"
        "        cutlass.Uint64, 8, cute.AddressSpace.gmem, assumed_align=8\n"
        "    )\n\n"
        "    b_w13_fake = cute.runtime.make_fake_compact_tensor(\n",
        label="compile fake timing pointer",
    )
    text = replace_exact(
        text,
        "        task_valid_rows_fake,\n"
        "        tile_write_count_fake,\n"
        "        b_w13_fake,\n",
        "        task_valid_rows_fake,\n"
        "        tile_write_count_fake,\n"
        "        exp014_scatter_ticks_fake,\n"
        "        b_w13_fake,\n",
        label="compile timing argument",
    )
    text = replace_exact(
        text,
        "        workspace.task_valid_rows.data_ptr(),\n"
        "        workspace.tile_write_count.data_ptr(),\n"
        "        weights.w13_fp4,\n",
        "        workspace.task_valid_rows.data_ptr(),\n"
        "        workspace.tile_write_count.data_ptr(),\n"
        "        workspace.exp014_scatter_ticks.data_ptr(),\n"
        "        weights.w13_fp4,\n",
        label="runtime timing argument",
    )
    if text.count("_EXP014_SCATTER_PHASE_PROBE_ENABLED") != 3:
        raise RuntimeError(
            "dispatch probe flag is not present in constant/constructor/cache"
        )
    timing_abi_count = text.count("exp014_scatter_ticks")
    if timing_abi_count != 9:
        raise RuntimeError(
            f"dispatch timing ABI/storage count drift: {timing_abi_count}"
        )
    ast.parse(text, filename="exp014_scatter_probe/moe_dispatch.py")
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


def normalize_scatter_mapping(source: str) -> str:
    text = source
    for baseline, candidate in zip(BASELINE_MAPPING, CANDIDATE_MAPPING, strict=True):
        count = text.count(candidate)
        if count not in (0, 1):
            raise RuntimeError(f"candidate mapping anchor count drift: {count}")
        if count == 1:
            text = text.replace(candidate, baseline)
    return text


def verify_existing(repo: Path, output: Path) -> dict[str, Any]:
    identity_path = output / "identity.json"
    if not identity_path.is_file():
        raise RuntimeError(f"probe overlay identity is missing: {identity_path}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("schema") != "exp014.scatter-phase-probe-overlays.v1":
        raise RuntimeError("probe overlay identity schema drift")
    if identity.get("event_abi") != EVENT_ABI:
        raise RuntimeError("probe event ABI drift")
    for arm in ARMS:
        manifest = identity.get("arms", {}).get(arm, {})
        arm_root = output / arm
        kernel = arm_root / "moe_dynamic_kernel.py"
        dispatch = arm_root / "moe_dispatch.py"
        base = BASE_OVERLAY_ROOT / arm / "moe_dynamic_kernel.py"
        checks = {
            "base": file_sha256(base) == EXPECTED_BASE_KERNEL_SHA256[arm],
            "kernel": file_sha256(kernel)
            == manifest.get("overlay", {}).get("kernel_sha256"),
            "dispatch": file_sha256(dispatch)
            == manifest.get("overlay", {}).get("dispatch_sha256"),
            "dispatch_live": file_sha256(repo / DISPATCH_RELATIVE_PATH)
            == EXPECTED_DISPATCH_SHA256,
            "wrapper_live": file_sha256(repo / WRAPPER_RELATIVE_PATH)
            == EXPECTED_WRAPPER_SHA256,
            "barriers": barrier_fingerprint(kernel.read_text(encoding="utf-8"))
            == manifest.get("overlay", {}).get("barrier_fingerprint"),
        }
        if not all(checks.values()):
            raise RuntimeError(f"{arm} existing probe overlay drift: {checks}")
    return identity


def build(repo: Path, output: Path) -> dict[str, Any]:
    repo = repo.resolve()
    output = output.resolve()
    dispatch_path = repo / DISPATCH_RELATIVE_PATH
    wrapper_path = repo / WRAPPER_RELATIVE_PATH
    if file_sha256(dispatch_path) != EXPECTED_DISPATCH_SHA256:
        raise RuntimeError("production dispatch identity drift")
    if file_sha256(wrapper_path) != EXPECTED_WRAPPER_SHA256:
        raise RuntimeError("production wrapper identity drift")
    if output.exists():
        raise FileExistsError(f"immutable probe overlay root exists: {output}")

    base_sources = {}
    for arm in ARMS:
        path = BASE_OVERLAY_ROOT / arm / "moe_dynamic_kernel.py"
        observed = file_sha256(path)
        if observed != EXPECTED_BASE_KERNEL_SHA256[arm]:
            raise RuntimeError(f"{arm} base overlay drift: {observed}")
        base_sources[arm] = path.read_text(encoding="utf-8")
    if normalize_scatter_mapping(base_sources[CANDIDATE]) != base_sources[BASELINE]:
        raise RuntimeError("base arms differ beyond the locked Scatter mapping")

    dispatch_source = dispatch_path.read_text(encoding="utf-8")
    dispatch_overlay = instrument_dispatch(dispatch_source)
    kernels = {arm: instrument_kernel(base_sources[arm]) for arm in ARMS}
    if normalize_scatter_mapping(kernels[CANDIDATE]) != kernels[BASELINE]:
        raise RuntimeError("probe arms differ beyond the locked Scatter mapping")

    output.mkdir(parents=True)
    try:
        manifests: dict[str, Any] = {}
        for arm in ARMS:
            arm_root = output / arm
            arm_root.mkdir()
            kernel_path = arm_root / "moe_dynamic_kernel.py"
            marker_dispatch_path = arm_root / "moe_dispatch.py"
            kernel_path.write_text(kernels[arm], encoding="utf-8")
            marker_dispatch_path.write_text(dispatch_overlay, encoding="utf-8")
            kernel_diff = diff(
                base_sources[arm],
                kernels[arm],
                left_name=f"{arm}/base/moe_dynamic_kernel.py",
                right_name=f"{arm}/scatter_probe/moe_dynamic_kernel.py",
            )
            dispatch_diff = diff(
                dispatch_source,
                dispatch_overlay,
                left_name="production/moe_dispatch.py",
                right_name=f"{arm}/scatter_probe/moe_dispatch.py",
            )
            (arm_root / "moe_dynamic_kernel.diff").write_text(
                kernel_diff, encoding="utf-8"
            )
            (arm_root / "moe_dispatch.diff").write_text(dispatch_diff, encoding="utf-8")
            manifest = {
                "schema": "exp014.scatter-phase-probe-overlay.v1",
                "arm": arm,
                "classification": "diagnostic-only",
                "probe_enabled": True,
                "event_abi": EVENT_ABI,
                "base": {
                    "kernel_path": str(
                        BASE_OVERLAY_ROOT / arm / "moe_dynamic_kernel.py"
                    ),
                    "kernel_sha256": EXPECTED_BASE_KERNEL_SHA256[arm],
                    "dispatch_path": str(dispatch_path),
                    "dispatch_sha256": EXPECTED_DISPATCH_SHA256,
                    "wrapper_path": str(wrapper_path),
                    "wrapper_sha256": EXPECTED_WRAPPER_SHA256,
                    "barrier_fingerprint": barrier_fingerprint(base_sources[arm]),
                },
                "overlay": {
                    "kernel_sha256": file_sha256(kernel_path),
                    "dispatch_sha256": file_sha256(marker_dispatch_path),
                    "kernel_diff_sha256": text_sha256(kernel_diff),
                    "dispatch_diff_sha256": text_sha256(dispatch_diff),
                    "barrier_fingerprint": barrier_fingerprint(kernels[arm]),
                },
                "contracts": {
                    "new_source_barriers": 0,
                    "lane": "lane0 of each W0-W7",
                    "D": "after pre-sync and before scatter call",
                    "E": "after scatter body and before post-sync",
                    "F": "after post-sync",
                    "task_slice_count": 1,
                    "output_tiles": OUTPUT_TILES,
                },
            }
            write_json(arm_root / "identity.json", manifest)
            manifests[arm] = manifest

        cross_arm = {
            "dispatch_sha256_equal": manifests[BASELINE]["overlay"]["dispatch_sha256"]
            == manifests[CANDIDATE]["overlay"]["dispatch_sha256"],
            "normalized_kernel_sha256_equal": canonical_sha256(
                normalize_scatter_mapping(kernels[BASELINE])
            )
            == canonical_sha256(normalize_scatter_mapping(kernels[CANDIDATE])),
            "event_abi_equal": manifests[BASELINE]["event_abi"]
            == manifests[CANDIDATE]["event_abi"],
        }
        if not all(cross_arm.values()):
            raise RuntimeError(f"matched probe gate failed: {cross_arm}")
        identity = {
            "schema": "exp014.scatter-phase-probe-overlays.v1",
            "event_abi": EVENT_ABI,
            "arms": manifests,
            "cross_arm": {**cross_arm, "gate_pass": True},
        }
        write_json(output / "identity.json", identity)
        verify_existing(repo, output)
        return identity
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, default=ROOT.parents[3])
    parser.add_argument("--output", type=Path, default=PROBE_OVERLAY_ROOT)
    parser.add_argument("--check-existing", action="store_true")
    args = parser.parse_args()
    if args.check_existing:
        result = verify_existing(args.flashinfer_root.resolve(), args.output.resolve())
    else:
        result = build(args.flashinfer_root.resolve(), args.output.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
