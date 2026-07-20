#!/usr/bin/env python3
"""Build matched no-marker/probe overlays for exp_016 P3 timing."""

from __future__ import annotations

import argparse
import ast
import difflib
import json
from pathlib import Path
import shutil
from typing import Any

from exp016_p3_probe_common import (
    ARMS,
    BASE_OVERLAY_ROOT,
    CONTROL,
    DISPATCH_RELATIVE_PATH,
    EVENT_ABI,
    EXPECTED_BASE_KERNEL_SHA256,
    EXPECTED_DISPATCH_SHA256,
    EXPECTED_WRAPPER_SHA256,
    MODES,
    PROBE,
    PROBE_OVERLAY_ROOT,
    TICKS_PER_CTA,
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

# exp_016 diagnostic P3 timing ABI.  Marker-disabled and marker-enabled
# specializations share this source and add no synchronization operation.
_EXP016_P3_TICKS_PER_CTA = {TICKS_PER_CTA}


@dsl_user_op
def _exp016_read_globaltimer(*, loc=None, ip=None):
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


def marker_store(edge: int, *, indent: int) -> str:
    pad = " " * indent
    return (
        f"{pad}exp016_p3_tick = _exp016_read_globaltimer()\n"
        f"{pad}st_global_u64(\n"
        f"{pad}    get_ptr_as_int64(\n"
        f"{pad}        exp016_p3_ticks,\n"
        f"{pad}        Int32(bidz) * Int32(_EXP016_P3_TICKS_PER_CTA)\n"
        f"{pad}        + Int32({edge}),\n"
        f"{pad}    ),\n"
        f"{pad}    exp016_p3_tick,\n"
        f"{pad})\n"
    )


def instrument_kernel(source: str) -> str:
    """Add the narrow P3 ABI and two leader-only stores without barriers."""
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
        "        p3_phase_probe_enabled: bool = False,\n"
        "    ):\n",
        label="constructor marker flag",
    )
    text = replace_exact(
        text,
        "        self.share_input_across_experts = share_input_across_experts\n",
        "        self.share_input_across_experts = share_input_across_experts\n"
        "        self.p3_phase_probe_enabled = bool(p3_phase_probe_enabled)\n",
        label="constructor marker flag assignment",
    )
    text = replace_exact(
        text,
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "        max_active_clusters: cutlass.Constexpr,\n",
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "        exp016_p3_ticks: cute.Tensor,\n"
        "        max_active_clusters: cutlass.Constexpr,\n",
        label="host-call marker ABI",
    )
    text = replace_exact(
        text,
        "            token_map,\n            token_weights,\n        ).launch(\n",
        "            token_map,\n"
        "            token_weights,\n"
        "            exp016_p3_ticks,\n"
        "        ).launch(\n",
        label="device launch marker arg",
    )
    text = replace_exact(
        text,
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "    ):\n"
        '        """Kernel entry point."""\n',
        "        token_map: cute.Tensor,\n"
        "        token_weights: cute.Tensor,\n"
        "        exp016_p3_ticks: cute.Tensor,\n"
        "    ):\n"
        '        """Kernel entry point."""\n',
        label="device kernel marker ABI",
    )
    text = replace_exact(
        text,
        "        launch_params: DynamicLaunchParams,\n"
        "        full_tile_publish_enabled: Int32,\n"
        "    ):\n"
        "        tidx, bidz, gdim_z, warp_idx, is_cta_leader = thread_info\n",
        "        launch_params: DynamicLaunchParams,\n"
        "        full_tile_publish_enabled: Int32,\n"
        "        exp016_p3_ticks: cute.Tensor,\n"
        "    ):\n"
        "        tidx, bidz, gdim_z, warp_idx, is_cta_leader = thread_info\n",
        label="Route/Q0 helper marker ABI",
    )
    text = replace_exact(
        text,
        "            launch_params,\n"
        "            full_tile_publish_enabled,\n"
        "        )\n",
        "            launch_params,\n"
        "            full_tile_publish_enabled,\n"
        "            exp016_p3_ticks,\n"
        "        )\n",
        label="Route/Q0 helper marker argument",
    )

    p3_start_anchor = (
        "        self.resident_grid_barrier(\n"
        "            barrier_count,\n"
        "            barrier_epoch,\n"
        "            Int32(gdim_z),\n"
        "            is_cta_leader,\n"
        "        )\n\n"
        "        # Phase 2: warp-private route/pack producers into compact physical tiles.\n"
    )
    p3_start = (
        "        self.resident_grid_barrier(\n"
        "            barrier_count,\n"
        "            barrier_epoch,\n"
        "            Int32(gdim_z),\n"
        "            is_cta_leader,\n"
        "        )\n"
        "        if cutlass.const_expr(self.p3_phase_probe_enabled):\n"
        "            if is_cta_leader > Int32(0):\n"
        + marker_store(0, indent=16)
        + "\n        # Phase 2: warp-private route/pack producers into compact physical tiles.\n"
    )
    text = replace_exact(
        text, p3_start_anchor, p3_start, label="post-prefix P3 start boundary"
    )

    p3_end_anchor = (
        "        cute.arch.sync_threads()\n"
        "        # Conservative publish fence before the last-producer CTA flushes any\n"
    )
    p3_end = (
        "        cute.arch.sync_threads()\n"
        "        if cutlass.const_expr(self.p3_phase_probe_enabled):\n"
        "            if is_cta_leader > Int32(0):\n"
        + marker_store(1, indent=16)
        + "        # Conservative publish fence before the last-producer CTA flushes any\n"
    )
    text = replace_exact(
        text, p3_end_anchor, p3_end, label="post-reconvergence P3 end boundary"
    )

    if barrier_fingerprint(text) != original_barriers:
        raise RuntimeError("P3 instrumentation changed the barrier fingerprint")
    if text.count("_exp016_read_globaltimer()") != 2:
        raise RuntimeError("P3 probe must contain exactly two timer read call sites")
    if text.count("exp016_p3_ticks,") != 4:
        raise RuntimeError("kernel P3 timing ABI/store count drift")
    ast.parse(text, filename="exp016_p3_probe/moe_dynamic_kernel.py")
    return text


def instrument_dispatch(source: str, *, enabled: bool) -> str:
    """Plumb a 2-tick-per-CTA buffer through the dynamic launch ABI."""
    text = replace_exact(
        source,
        "_DYNAMIC_SLICE_CHUNK = 1\n",
        "_DYNAMIC_SLICE_CHUNK = 1\n"
        f"_EXP016_P3_PHASE_PROBE_ENABLED = {enabled!r}\n"
        f"_EXP016_P3_TICKS_PER_CTA = {TICKS_PER_CTA}\n",
        label="dispatch marker constants",
    )
    text = replace_exact(
        text,
        "    task_valid_rows: torch.Tensor\n"
        "    tile_write_count: torch.Tensor\n\n"
        "    # Views\n",
        "    task_valid_rows: torch.Tensor\n"
        "    tile_write_count: torch.Tensor\n"
        "    exp016_p3_ticks: torch.Tensor\n\n"
        "    # Views\n",
        label="workspace P3 field",
    )
    text = replace_exact(
        text,
        "        tile_write_count=torch.zeros(physical_tiles, dtype=torch.int32, device=device),\n"
        "    )\n",
        "        tile_write_count=torch.zeros(physical_tiles, dtype=torch.int32, device=device),\n"
        "        exp016_p3_ticks=torch.full(\n"
        "            (get_num_sm(device) * _EXP016_P3_TICKS_PER_CTA,),\n"
        "            -1, dtype=torch.int64, device=device,\n"
        "        ),\n"
        "    )\n",
        label="workspace P3 allocation",
    )
    text = replace_exact(
        text,
        "        tile_write_count_ptr: cute.Pointer,\n        b_w13: cute.Tensor,\n",
        "        tile_write_count_ptr: cute.Pointer,\n"
        "        exp016_p3_ticks_ptr: cute.Pointer,\n"
        "        b_w13: cute.Tensor,\n",
        label="dynamic-launch P3 pointer",
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
        "        exp016_p3_ticks = cute.make_tensor(\n"
        "            exp016_p3_ticks_ptr,\n"
        "            layout=cute.make_layout(\n"
        "                (max_active_clusters * _EXP016_P3_TICKS_PER_CTA,), stride=(1,)\n"
        "            ),\n"
        "        )\n"
        "        self._kernel(\n",
        label="dynamic-launch P3 tensor",
    )
    text = replace_exact(
        text,
        "            token_map,\n"
        "            token_weights_t,\n"
        "            max_active_clusters=max_active_clusters,\n",
        "            token_map,\n"
        "            token_weights_t,\n"
        "            exp016_p3_ticks,\n"
        "            max_active_clusters=max_active_clusters,\n",
        label="kernel P3 argument",
    )
    text = replace_exact(
        text,
        "        share_input_across_experts=share_input_across_experts,\n"
        "    )\n"
        "    launch = _DynamicMoELaunch(\n",
        "        share_input_across_experts=share_input_across_experts,\n"
        "        p3_phase_probe_enabled=_EXP016_P3_PHASE_PROBE_ENABLED,\n"
        "    )\n"
        "    launch = _DynamicMoELaunch(\n",
        label="kernel P3 marker flag",
    )
    text = replace_exact(
        text,
        "        share_input_across_experts,\n"
        "    )\n"
        "    cached = _DYNAMIC_KERNEL_CACHE.get(cache_key)\n",
        "        share_input_across_experts,\n"
        "        _EXP016_P3_PHASE_PROBE_ENABLED,\n"
        "    )\n"
        "    cached = _DYNAMIC_KERNEL_CACHE.get(cache_key)\n",
        label="JIT cache P3 identity",
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
        "    exp016_p3_ticks_fake = make_ptr(\n"
        "        cutlass.Uint64, 8, cute.AddressSpace.gmem, assumed_align=8\n"
        "    )\n\n"
        "    b_w13_fake = cute.runtime.make_fake_compact_tensor(\n",
        label="compile fake P3 pointer",
    )
    text = replace_exact(
        text,
        "        task_valid_rows_fake,\n"
        "        tile_write_count_fake,\n"
        "        b_w13_fake,\n",
        "        task_valid_rows_fake,\n"
        "        tile_write_count_fake,\n"
        "        exp016_p3_ticks_fake,\n"
        "        b_w13_fake,\n",
        label="compile P3 argument",
    )
    text = replace_exact(
        text,
        "        workspace.task_valid_rows.data_ptr(),\n"
        "        workspace.tile_write_count.data_ptr(),\n"
        "        weights.w13_fp4,\n",
        "        workspace.task_valid_rows.data_ptr(),\n"
        "        workspace.tile_write_count.data_ptr(),\n"
        "        workspace.exp016_p3_ticks.data_ptr(),\n"
        "        weights.w13_fp4,\n",
        label="runtime P3 argument",
    )
    if text.count("_EXP016_P3_PHASE_PROBE_ENABLED") != 3:
        raise RuntimeError("dispatch P3 flag must appear in constant/constructor/cache")
    ast.parse(text, filename="exp016_p3_probe/moe_dispatch.py")
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


def overlay_paths(output: Path, arm: str, mode: str) -> tuple[Path, Path, Path]:
    root = output / arm / mode
    return root, root / "moe_dynamic_kernel.py", root / "moe_dispatch.py"


def verify_existing(repo: Path, output: Path) -> dict[str, Any]:
    identity_path = output / "identity.json"
    if not identity_path.is_file():
        raise RuntimeError(f"P3 probe identity missing: {identity_path}")
    identity = read_json(identity_path)
    if identity.get("schema") != "exp016.p3-phase-probe-overlays.v1":
        raise RuntimeError("P3 probe root schema drift")
    if identity.get("event_abi") != EVENT_ABI:
        raise RuntimeError("P3 event ABI drift")
    if file_sha256(repo / DISPATCH_RELATIVE_PATH) != EXPECTED_DISPATCH_SHA256:
        raise RuntimeError("live dispatch source drift")
    if file_sha256(repo / WRAPPER_RELATIVE_PATH) != EXPECTED_WRAPPER_SHA256:
        raise RuntimeError("live wrapper source drift")

    for arm in ARMS:
        base = BASE_OVERLAY_ROOT / arm / "moe_dynamic_kernel.py"
        if file_sha256(base) != EXPECTED_BASE_KERNEL_SHA256[arm]:
            raise RuntimeError(f"{arm} base overlay drift")
        kernels: dict[str, str] = {}
        dispatches: dict[str, str] = {}
        for mode in MODES:
            root, kernel, dispatch = overlay_paths(output, arm, mode)
            manifest = read_json(root / "identity.json")
            registered = identity["arms"][arm][mode]
            checks = {
                "manifest_matches_root": manifest == registered,
                "kernel_hash": file_sha256(kernel)
                == manifest["overlay"]["kernel_sha256"],
                "dispatch_hash": file_sha256(dispatch)
                == manifest["overlay"]["dispatch_sha256"],
                "base_hash": manifest["base"]["kernel_sha256"]
                == EXPECTED_BASE_KERNEL_SHA256[arm],
                "barriers_unchanged": barrier_fingerprint(
                    kernel.read_text(encoding="utf-8")
                )
                == manifest["base"]["barrier_fingerprint"],
                "mode": manifest["mode"] == mode,
                "enabled": bool(manifest["probe_enabled"]) == (mode == PROBE),
            }
            if not all(checks.values()):
                raise RuntimeError(f"{arm}/{mode} overlay drift: {checks}")
            kernels[mode] = kernel.read_text(encoding="utf-8")
            dispatches[mode] = dispatch.read_text(encoding="utf-8")
        if kernels[CONTROL] != kernels[PROBE]:
            raise RuntimeError(f"{arm} control/probe kernel source differs")
        if normalize_dispatch_flag(dispatches[CONTROL]) != normalize_dispatch_flag(
            dispatches[PROBE]
        ):
            raise RuntimeError(f"{arm} control/probe dispatch differs beyond flag")
    return identity


def build(repo: Path, output: Path) -> dict[str, Any]:
    repo = repo.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable P3 probe overlay exists: {output}")
    dispatch_path = repo / DISPATCH_RELATIVE_PATH
    wrapper_path = repo / WRAPPER_RELATIVE_PATH
    if file_sha256(dispatch_path) != EXPECTED_DISPATCH_SHA256:
        raise RuntimeError("production dispatch identity drift")
    if file_sha256(wrapper_path) != EXPECTED_WRAPPER_SHA256:
        raise RuntimeError("production wrapper identity drift")

    base_identity = read_json(BASE_OVERLAY_ROOT / "identity.json")
    if base_identity.get("schema") != "exp016.route-q0-overlay.v1":
        raise RuntimeError("exp016 base overlay identity schema drift")
    base_sources: dict[str, str] = {}
    for arm in ARMS:
        base = BASE_OVERLAY_ROOT / arm / "moe_dynamic_kernel.py"
        observed = file_sha256(base)
        if observed != EXPECTED_BASE_KERNEL_SHA256[arm]:
            raise RuntimeError(f"{arm} base overlay drift: {observed}")
        if base_identity["arms"][arm]["sha256"] != observed:
            raise RuntimeError(f"{arm} base identity disagreement")
        base_sources[arm] = base.read_text(encoding="utf-8")

    dispatch_source = dispatch_path.read_text(encoding="utf-8")
    dispatches = {
        CONTROL: instrument_dispatch(dispatch_source, enabled=False),
        PROBE: instrument_dispatch(dispatch_source, enabled=True),
    }
    if normalize_dispatch_flag(dispatches[CONTROL]) != normalize_dispatch_flag(
        dispatches[PROBE]
    ):
        raise RuntimeError("control/probe dispatch differs beyond constexpr flag")
    kernels = {arm: instrument_kernel(source) for arm, source in base_sources.items()}

    output.mkdir(parents=True)
    try:
        arms_manifest: dict[str, Any] = {}
        for arm in ARMS:
            arms_manifest[arm] = {}
            for mode in MODES:
                root, kernel_path, dispatch_overlay_path = overlay_paths(
                    output, arm, mode
                )
                root.mkdir(parents=True)
                kernel_path.write_text(kernels[arm], encoding="utf-8")
                dispatch_overlay_path.write_text(dispatches[mode], encoding="utf-8")
                kernel_diff = diff(
                    base_sources[arm],
                    kernels[arm],
                    left_name=f"{arm}/base/moe_dynamic_kernel.py",
                    right_name=f"{arm}/{mode}/moe_dynamic_kernel.py",
                )
                dispatch_diff = diff(
                    dispatch_source,
                    dispatches[mode],
                    left_name="production/moe_dispatch.py",
                    right_name=f"{arm}/{mode}/moe_dispatch.py",
                )
                (root / "moe_dynamic_kernel.diff").write_text(
                    kernel_diff, encoding="utf-8"
                )
                (root / "moe_dispatch.diff").write_text(dispatch_diff, encoding="utf-8")
                manifest = {
                    "schema": "exp016.p3-phase-probe-overlay.v1",
                    "arm": arm,
                    "mode": mode,
                    "probe_enabled": mode == PROBE,
                    "classification": (
                        "diagnostic" if mode == PROBE else "marker-disabled ABI control"
                    ),
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
                        "dispatch_sha256": file_sha256(dispatch_overlay_path),
                        "kernel_diff_sha256": text_sha256(kernel_diff),
                        "dispatch_diff_sha256": text_sha256(dispatch_diff),
                        "barrier_fingerprint": barrier_fingerprint(kernels[arm]),
                    },
                    "boundary": {
                        "start_anchor_count": kernels[arm].count(
                            "# Phase 2: warp-private route/pack producers"
                        ),
                        "end_anchor_count": kernels[arm].count(
                            "# Conservative publish fence before the last-producer CTA"
                        ),
                        "timer_read_call_sites": kernels[arm].count(
                            "_exp016_read_globaltimer()"
                        ),
                        "new_barriers": 0,
                    },
                }
                write_json(root / "identity.json", manifest)
                arms_manifest[arm][mode] = manifest

        cross_mode = {}
        for arm in ARMS:
            cross_mode[arm] = {
                "kernel_source_byte_identical": (
                    arms_manifest[arm][CONTROL]["overlay"]["kernel_sha256"]
                    == arms_manifest[arm][PROBE]["overlay"]["kernel_sha256"]
                ),
                "normalized_dispatch_byte_identical": (
                    canonical_sha256(normalize_dispatch_flag(dispatches[CONTROL]))
                    == canonical_sha256(normalize_dispatch_flag(dispatches[PROBE]))
                ),
                "only_compile_time_enable_flag_differs": True,
                "fresh_jit_and_cubin_required_per_mode": True,
                "resource_usage_comparison_required": True,
            }
            if not all(cross_mode[arm].values()):
                raise RuntimeError(f"{arm} matched control/probe gate failed")
        identity = {
            "schema": "exp016.p3-phase-probe-overlays.v1",
            "event_abi": EVENT_ABI,
            "base_overlay_identity_sha256": canonical_sha256(base_identity),
            "arms": arms_manifest,
            "cross_mode": cross_mode,
            "evidence_contract": {
                "primary_phase_statistic": "grid critical wall",
                "additive_sm_estimate_forbidden": True,
                "control_and_probe_require_separate_fresh_jit_roots": True,
                "control_probe_cubin_resources_must_be_compared_before_use": True,
                "uninstrumented_e2e_remains_performance_authority": True,
            },
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
