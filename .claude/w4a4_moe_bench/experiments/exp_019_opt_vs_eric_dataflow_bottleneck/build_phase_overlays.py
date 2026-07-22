#!/usr/bin/env python3
"""Build matched Opt/Eric control and phase-probe overlays for exp_019."""

from __future__ import annotations

import argparse
import ast
import difflib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from phase_common import (
    ARMS,
    CONTROL,
    DISPATCH_RELATIVE_PATH,
    EVENT_ABI,
    EVENTS_PER_CTA,
    EXPECTED_DISPATCH_SHA256,
    EXPECTED_ERIC_ADAPTER_SHA256,
    EXPECTED_SOURCE_SHA256,
    EXPECTED_WRAPPER_SHA256,
    MODES,
    OCCURRENCE_ABI,
    OCCURRENCES_PER_CTA,
    PHASE_NAMES,
    PROBE,
    SOURCE_RELATIVE_PATH,
    STORAGE_PER_CTA,
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
BENCH_ROOT = ROOT.parents[1]
EXP017 = ROOT.parent / "exp_017_opt_vs_triton_phase_share"
for dependency in (BENCH_ROOT, EXP017):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

import build_opt_phase_overlays as exp017_builder  # noqa: E402
from breakdown_harness.fragments import eric_stage4_adapter as eric_adapter  # noqa: E402


OVERLAY_ROOT = ROOT / "results" / "phase_overlays"


def replace_exact(
    text: str, old: str, new: str, *, label: str, expected: int = 1
) -> str:
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{label}: expected {expected} anchors, found {count}")
    return text.replace(old, new, expected)


def _event_index(slot: int) -> str:
    return f"Int32(bidz) * Int32(_EXP017_OPT_PHASE_EVENTS_PER_CTA) + Int32({slot})"


def _occurrence_index(phase: str) -> str:
    slot = PHASE_NAMES.index(phase)
    return (
        f"Int32(gdim_z) * Int32(_EXP017_OPT_PHASE_EVENTS_PER_CTA) + "
        f"Int32(bidz) * Int32({OCCURRENCES_PER_CTA}) + Int32({slot})"
    )


def mark_close(phase: str, tag: str, *, indent: int) -> str:
    """Close one interval and independently count this close marker."""
    slot = exp017_builder.PHASE_SLOT_BASE + PHASE_NAMES.index(phase)
    pad = " " * indent
    return (
        f"{pad}if cutlass.const_expr(self.exp017_opt_phase_probe_enabled):\n"
        f"{pad}    if is_cta_leader > Int32(0):\n"
        f"{pad}        exp017_{tag}_end = _exp017_read_globaltimer()\n"
        f"{pad}        exp017_{tag}_cursor = _ld_global_u64(\n"
        f"{pad}            get_ptr_as_int64(exp017_phase_events, {_event_index(exp017_builder.CURSOR_SLOT)})\n"
        f"{pad}        )\n"
        f"{pad}        exp017_{tag}_total = _ld_global_u64(\n"
        f"{pad}            get_ptr_as_int64(exp017_phase_events, {_event_index(slot)})\n"
        f"{pad}        )\n"
        f"{pad}        st_global_u64(\n"
        f"{pad}            get_ptr_as_int64(exp017_phase_events, {_event_index(slot)}),\n"
        f"{pad}            exp017_{tag}_total + exp017_{tag}_end - exp017_{tag}_cursor,\n"
        f"{pad}        )\n"
        f"{pad}        exp019_{tag}_count = _ld_global_u64(\n"
        f"{pad}            get_ptr_as_int64(exp017_phase_events, {_occurrence_index(phase)})\n"
        f"{pad}        )\n"
        f"{pad}        st_global_u64(\n"
        f"{pad}            get_ptr_as_int64(exp017_phase_events, {_occurrence_index(phase)}),\n"
        f"{pad}            exp019_{tag}_count + Uint64(1),\n"
        f"{pad}        )\n"
        f"{pad}        st_global_u64(\n"
        f"{pad}            get_ptr_as_int64(exp017_phase_events, {_event_index(exp017_builder.CURSOR_SLOT)}),\n"
        f"{pad}            exp017_{tag}_end,\n"
        f"{pad}        )\n"
    )


OPT_CLOSE_SPECS = (
    ("clear_init", "clear", 8),
    ("histogram", "histogram", 8),
    ("prefix", "prefix", 8),
    ("route_q0_pack", "route", 8),
    ("publish_route_tail", "publish", 8),
    ("claim_cache_control", "claim", 12),
    ("fc1_gate_up_swiglu", "fc1", 20),
    ("q1", "q1", 20),
    ("fc2_epilogue_r2s", "fc2", 24),
    ("scatter", "scatter", 24),
)


def instrument_opt(source: str) -> str:
    """Call the reviewed exp_017 anchor map, adding only occurrence writes."""
    text = exp017_builder.instrument_kernel(source)
    for phase, tag, indent in OPT_CLOSE_SPECS:
        text = replace_exact(
            text,
            exp017_builder.mark_close(phase, tag, indent=indent),
            mark_close(phase, tag, indent=indent),
            label=f"Opt occurrence marker {phase}",
        )
    ast.parse(text, filename="exp019/latest_opt/moe_dynamic_kernel.py")
    return text


def _plumb_eric_abi(source: str) -> str:
    text = replace_exact(
        source,
        "_FC2_TILE_RECIP_GS_NUM = 6.0 * 448.0\n",
        exp017_builder.KERNEL_CONSTANTS,
        label="Eric globaltimer helper",
    )
    text = replace_exact(
        text,
        "        share_input_across_experts: bool = False,\n    ):\n",
        "        share_input_across_experts: bool = False,\n"
        "        exp017_opt_phase_probe_enabled: bool = False,\n"
        "    ):\n",
        label="Eric constructor marker flag",
    )
    text = replace_exact(
        text,
        "        self.share_input_across_experts = share_input_across_experts\n",
        "        self.share_input_across_experts = share_input_across_experts\n"
        "        self.exp017_opt_phase_probe_enabled = bool(\n"
        "            exp017_opt_phase_probe_enabled\n"
        "        )\n",
        label="Eric constructor marker assignment",
    )
    text = replace_exact(
        text,
        "        token_weights: cute.Tensor,\n"
        "        max_active_clusters: cutlass.Constexpr,\n",
        "        token_weights: cute.Tensor,\n"
        "        exp017_phase_events: cute.Tensor,\n"
        "        max_active_clusters: cutlass.Constexpr,\n",
        label="Eric host event ABI",
    )
    text = replace_exact(
        text,
        "            token_map,\n            token_weights,\n        ).launch(\n",
        "            token_map,\n"
        "            token_weights,\n"
        "            exp017_phase_events,\n"
        "        ).launch(\n",
        label="Eric launch event argument",
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
        label="Eric kernel event ABI",
    )
    return text


def instrument_eric(adapter_source: str) -> str:
    """Apply the minimal monolithic Eric anchor map without adding sync."""
    original_barriers = barrier_fingerprint(adapter_source)
    text = _plumb_eric_abi(adapter_source)

    entry = (
        "        is_cta_leader = Int32(1) if Int32(tidx) == Int32(0) else Int32(0)\n\n"
        "        if warp_idx == 0:\n"
    )
    text = replace_exact(
        text,
        entry,
        "        is_cta_leader = Int32(1) if Int32(tidx) == Int32(0) else Int32(0)\n"
        + exp017_builder.mark_entry(indent=8)
        + "\n        if warp_idx == 0:\n",
        label="Eric CTA entry",
    )
    text = replace_exact(
        text,
        "        # Phase 0: cooperative init — zero routing state, queue state, and output.\n"
        "        task_capacity = Int32(task_ready.shape[0])\n",
        "        # Phase 0: cooperative init — zero routing state, queue state, and output.\n"
        + exp017_builder.mark_start("clear", indent=8)
        + "        task_capacity = Int32(task_ready.shape[0])\n",
        label="Eric clear start",
    )
    text = replace_exact(
        text,
        "            is_cta_leader,\n"
        "        )\n\n"
        "        # Phase 1: histogram routed rows per expert.\n",
        "            is_cta_leader,\n"
        "        )\n"
        + mark_close("clear_init", "clear", indent=8)
        + "\n        # Phase 1: histogram routed rows per expert.\n",
        label="Eric clear/histogram",
    )
    text = replace_exact(
        text,
        "            is_cta_leader,\n        )\n\n        if flat_tid == Int32(0):\n",
        "            is_cta_leader,\n"
        "        )\n"
        + mark_close("histogram", "histogram", indent=8)
        + "\n        if flat_tid == Int32(0):\n",
        label="Eric histogram/prefix",
    )
    text = replace_exact(
        text,
        "            is_cta_leader,\n"
        "        )\n\n"
        "        # Phase 2: warp-private route/pack producers into compact physical tiles.\n",
        "            is_cta_leader,\n"
        "        )\n"
        + mark_close("prefix", "prefix", indent=8)
        + "\n        # Phase 2: warp-private route/pack producers into compact physical tiles.\n",
        label="Eric prefix/route",
    )
    text = replace_exact(
        text,
        "        cute.arch.sync_threads()\n"
        "        # Conservative publish fence before the last-producer CTA flushes any\n",
        "        cute.arch.sync_threads()\n"
        + mark_close("route_q0_pack", "route", indent=8)
        + "        # Conservative publish fence before the last-producer CTA flushes any\n",
        label="Eric route/publish",
    )
    text = replace_exact(
        text,
        "                _st_global_release_i32(\n"
        "                    get_ptr_as_int64(all_work_published, Int32(0)),\n"
        "                    Int32(1),\n"
        "                )\n\n"
        "        gA = cute.local_tile(\n",
        "                _st_global_release_i32(\n"
        "                    get_ptr_as_int64(all_work_published, Int32(0)),\n"
        "                    Int32(1),\n"
        "                )\n"
        + mark_close("publish_route_tail", "publish", indent=8)
        + "\n        gA = cute.local_tile(\n",
        label="Eric publish/consumer setup",
    )
    text = replace_exact(
        text,
        "        while consumer_live > Int32(0):\n"
        "            if is_cta_leader > Int32(0):\n",
        "        while consumer_live > Int32(0):\n"
        + exp017_builder.mark_start("claim", indent=12)
        + "            if is_cta_leader > Int32(0):\n",
        label="Eric claim start",
    )
    claim_end = (
        "                    cache_row += Int32(self.threads_per_cta)\n"
        "                cute.arch.sync_threads()\n"
        "            if has_task == Int32(0):\n"
    )
    text = replace_exact(
        text,
        claim_end,
        "                    cache_row += Int32(self.threads_per_cta)\n"
        "                cute.arch.sync_threads()\n"
        + mark_close("claim_cache_control", "claim", indent=12)
        + "            if has_task == Int32(0):\n",
        label="Eric claim/cache end",
    )
    text = replace_exact(
        text,
        "                while slice_idx < task_slice_count_val:\n"
        "                    # ============================================================\n"
        "                    # PHASE A: FC1 for this slice (gate + up)\n",
        "                while slice_idx < task_slice_count_val:\n"
        + exp017_builder.mark_start("fc1", indent=20)
        + "                    # ============================================================\n"
        "                    # PHASE A: FC1 for this slice (gate + up)\n",
        label="Eric FC1 start",
    )
    text = replace_exact(
        text,
        "                        self.epilog_sync_barrier.arrive_and_wait()\n\n"
        "                        rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])\n",
        "                        self.epilog_sync_barrier.arrive_and_wait()\n"
        + mark_close("fc1_gate_up_swiglu", "fc1", indent=24)
        + "\n                        rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])\n",
        label="Eric interleaved SwiGLU/Q1 transition",
    )
    text = replace_exact(
        text,
        '                        cute.arch.fence_proxy("async.shared", space="cta")\n'
        "                        self.epilog_sync_barrier.arrive_and_wait()\n\n"
        "                    # ============================================================\n"
        "                    # PHASE B: Sweep ALL FC2 output tiles using cached sA\n",
        '                        cute.arch.fence_proxy("async.shared", space="cta")\n'
        "                        self.epilog_sync_barrier.arrive_and_wait()\n"
        + mark_close("q1", "q1", indent=24)
        + "\n                    # ============================================================\n"
        "                    # PHASE B: Sweep ALL FC2 output tiles using cached sA\n",
        label="Eric interleaved Q1 close",
    )
    text = replace_exact(
        text,
        "                    # ============================================================\n"
        "                    scatter_N = Int32(scatter_output.shape[1])\n",
        "                    # ============================================================\n"
        + exp017_builder.mark_start("fc2", indent=20)
        + "                    scatter_N = Int32(scatter_output.shape[1])\n",
        label="Eric FC2 start",
    )
    text = replace_exact(
        text,
        "                            self.epilog_sync_barrier.arrive_and_wait()\n"
        "                            rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])\n",
        "                            self.epilog_sync_barrier.arrive_and_wait()\n"
        + mark_close("fc2_epilogue_r2s", "fc2", indent=28)
        + "                            rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])\n",
        label="Eric interleaved FC2/scatter transition",
    )
    text = replace_exact(
        text,
        "                            # (pipeline consumer is collective across all MMA warps).\n"
        "                            self.epilog_sync_barrier.arrive_and_wait()\n\n"
        "                    # Signal that FC2/scatter no longer needs sA, so the DMA\n",
        "                            # (pipeline consumer is collective across all MMA warps).\n"
        "                            self.epilog_sync_barrier.arrive_and_wait()\n"
        + mark_close("scatter", "scatter", indent=28)
        + "\n                    # Signal that FC2/scatter no longer needs sA, so the DMA\n",
        label="Eric scatter close/next FC2",
    )
    final = (
        "        if warp_idx == self.tma_load_warp_id:\n"
        "            ml_pipeline.producer_tail(prod_state)\n"
        "            if cutlass.const_expr(self.is_gated):\n"
        "                up_pipeline.producer_tail(up_prod_state)\n"
        "            phase2_pipeline.producer_tail(phase2_prod_state)\n"
        "        return\n"
    )
    text = replace_exact(
        text,
        final,
        "        if warp_idx == self.tma_load_warp_id:\n"
        "            ml_pipeline.producer_tail(prod_state)\n"
        "            if cutlass.const_expr(self.is_gated):\n"
        "                up_pipeline.producer_tail(up_prod_state)\n"
        "            phase2_pipeline.producer_tail(phase2_prod_state)\n"
        + exp017_builder.mark_final(
            exp017_builder.TMA_FINAL_SLOT,
            "tma",
            "Int32(tidx) == Int32(self.tma_load_warp_id * self.num_threads_per_warp)",
            indent=12,
        )
        + exp017_builder.mark_final(
            exp017_builder.READER_FINAL_SLOT,
            "reader",
            "is_cta_leader > Int32(0)",
            indent=8,
        )
        + "        return\n",
        label="Eric final markers",
    )

    if barrier_fingerprint(text) != original_barriers:
        raise RuntimeError("Eric instrumentation changed barrier fingerprint")
    ast.parse(text, filename="exp019/eric_stage4/moe_dynamic_kernel.py")
    return text


def instrument_dispatch(source: str, *, enabled: bool) -> str:
    """Reuse exp_017 dispatch plumbing, enlarging only its flat allocation."""
    text = exp017_builder.instrument_dispatch(source, enabled=enabled)
    text = replace_exact(
        text,
        f"_EXP017_OPT_PHASE_EVENTS_PER_CTA = {EVENTS_PER_CTA}\n",
        f"_EXP017_OPT_PHASE_EVENTS_PER_CTA = {EVENTS_PER_CTA}\n"
        f"_EXP019_PHASE_STORAGE_PER_CTA = {STORAGE_PER_CTA}\n",
        label="phase storage constant",
    )
    text = replace_exact(
        text,
        "            get_num_sm(device) * _EXP017_OPT_PHASE_EVENTS_PER_CTA,\n",
        "            get_num_sm(device) * _EXP019_PHASE_STORAGE_PER_CTA,\n",
        label="phase storage allocation",
    )
    text = replace_exact(
        text,
        "                (max_active_clusters * _EXP017_OPT_PHASE_EVENTS_PER_CTA,),\n",
        "                (max_active_clusters * _EXP019_PHASE_STORAGE_PER_CTA,),\n",
        label="phase storage tensor layout",
    )
    ast.parse(text, filename="exp019/moe_dispatch.py")
    return text


def overlay_paths(output: Path, arm: str, mode: str) -> tuple[Path, Path, Path]:
    root = output / arm / mode
    return root, root / "moe_dynamic_kernel.py", root / "moe_dispatch.py"


def _adapter_source(source: str) -> tuple[str, dict[str, Any] | None]:
    adapter = eric_adapter.make_adapter(source)
    if text_sha256(adapter) != EXPECTED_ERIC_ADAPTER_SHA256:
        raise RuntimeError("Eric compatibility adapter identity drift")
    validation = eric_adapter.validate_adapter(source, adapter)
    return adapter, validation


def build(repo: Path, output: Path) -> dict[str, Any]:
    repo, output = repo.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable overlay exists: {output}")
    dispatch_path = repo / DISPATCH_RELATIVE_PATH
    wrapper_path = repo / WRAPPER_RELATIVE_PATH
    if file_sha256(dispatch_path) != EXPECTED_DISPATCH_SHA256:
        raise RuntimeError("dispatch source drift")
    if file_sha256(wrapper_path) != EXPECTED_WRAPPER_SHA256:
        raise RuntimeError("wrapper source drift")
    dispatch_source = dispatch_path.read_text(encoding="utf-8")
    dispatches = {
        mode: instrument_dispatch(dispatch_source, enabled=mode == PROBE)
        for mode in MODES
    }
    if normalize_dispatch_flag(dispatches[CONTROL]) != normalize_dispatch_flag(
        dispatches[PROBE]
    ):
        raise RuntimeError("control/probe dispatch differs beyond constexpr flag")

    output.mkdir(parents=True)
    try:
        arms: dict[str, Any] = {}
        for arm in ARMS:
            source_path = repo / SOURCE_RELATIVE_PATH[arm]
            if file_sha256(source_path) != EXPECTED_SOURCE_SHA256[arm]:
                raise RuntimeError(f"{arm} source drift")
            base = source_path.read_text(encoding="utf-8")
            adapter_validation = None
            instrument_input = base
            if arm == "eric_stage4_fp4":
                instrument_input, adapter_validation = _adapter_source(base)
                kernel = instrument_eric(instrument_input)
            else:
                kernel = instrument_opt(instrument_input)
            modes: dict[str, Any] = {}
            for mode in MODES:
                root, kernel_path, selected_dispatch = overlay_paths(output, arm, mode)
                root.mkdir(parents=True)
                kernel_path.write_text(kernel, encoding="utf-8")
                selected_dispatch.write_text(dispatches[mode], encoding="utf-8")
                kernel_diff = "".join(
                    difflib.unified_diff(
                        base.splitlines(keepends=True),
                        kernel.splitlines(keepends=True),
                        fromfile=source_path.name,
                        tofile="moe_dynamic_kernel.py",
                        n=0,
                    )
                )
                dispatch_diff = "".join(
                    difflib.unified_diff(
                        dispatch_source.splitlines(keepends=True),
                        dispatches[mode].splitlines(keepends=True),
                        fromfile="moe_dispatch.py",
                        tofile=f"{mode}/moe_dispatch.py",
                        n=0,
                    )
                )
                (root / "moe_dynamic_kernel.diff").write_text(
                    kernel_diff, encoding="utf-8"
                )
                (root / "moe_dispatch.diff").write_text(dispatch_diff, encoding="utf-8")
                manifest = {
                    "schema": "exp019.phase-overlay.v1",
                    "arm": arm,
                    "mode": mode,
                    "probe_enabled": mode == PROBE,
                    "event_abi": EVENT_ABI,
                    "occurrence_abi": OCCURRENCE_ABI,
                    "base": {
                        "source_path": str(source_path),
                        "source_sha256": EXPECTED_SOURCE_SHA256[arm],
                        "adapter_sha256": (
                            EXPECTED_ERIC_ADAPTER_SHA256
                            if arm == "eric_stage4_fp4"
                            else None
                        ),
                        "adapter_validation": adapter_validation,
                        "dispatch_sha256": EXPECTED_DISPATCH_SHA256,
                        "wrapper_sha256": EXPECTED_WRAPPER_SHA256,
                        "barrier_fingerprint": barrier_fingerprint(base),
                    },
                    "overlay": {
                        "kernel_sha256": file_sha256(kernel_path),
                        "dispatch_sha256": file_sha256(selected_dispatch),
                        "kernel_diff_sha256": text_sha256(kernel_diff),
                        "dispatch_diff_sha256": text_sha256(dispatch_diff),
                        "barrier_fingerprint": barrier_fingerprint(kernel),
                    },
                    "boundary": {
                        "timer_read_call_sites": kernel.count(
                            "_exp017_read_globaltimer()"
                        ),
                        "occurrence_close_call_sites": kernel.count(
                            "_count = _ld_global_u64("
                        ),
                        "phase_names": list(PHASE_NAMES),
                        "new_barriers": 0,
                        "eric_interval_accumulation": (
                            {
                                "swiglu_q1": "each epi_m",
                                "fc2_scatter": "each output_tile x epi_m",
                            }
                            if arm == "eric_stage4_fp4"
                            else None
                        ),
                    },
                }
                write_json(root / "identity.json", manifest)
                modes[mode] = manifest
            control_kernel = overlay_paths(output, arm, CONTROL)[1]
            probe_kernel = overlay_paths(output, arm, PROBE)[1]
            if control_kernel.read_bytes() != probe_kernel.read_bytes():
                raise RuntimeError(f"{arm} control/probe kernel differs")
            arms[arm] = {"modes": modes}
        identity = {
            "schema": "exp019.phase-overlays.v1",
            "event_abi": EVENT_ABI,
            "occurrence_abi": OCCURRENCE_ABI,
            "arms": arms,
            "cross_mode": {
                "kernel_source_byte_identical": True,
                "normalized_dispatch_byte_identical": canonical_sha256(
                    normalize_dispatch_flag(dispatches[CONTROL])
                )
                == canonical_sha256(normalize_dispatch_flag(dispatches[PROBE])),
                "only_compile_time_enable_flag_differs": True,
                "independent_arm_mode_jit_roots_required": True,
                "diagnostic_cubin_not_production_cubin": True,
            },
        }
        if not all(identity["cross_mode"].values()):
            raise RuntimeError("matched overlay identity gate failed")
        write_json(output / "identity.json", identity)
        return verify_existing(repo, output)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def verify_existing(repo: Path, output: Path) -> dict[str, Any]:
    repo, output = repo.resolve(), output.resolve()
    identity = read_json(output / "identity.json")
    if identity.get("schema") != "exp019.phase-overlays.v1":
        raise RuntimeError("overlay root schema drift")
    if (
        identity.get("event_abi") != EVENT_ABI
        or identity.get("occurrence_abi") != OCCURRENCE_ABI
    ):
        raise RuntimeError("phase ABI drift")
    for arm in ARMS:
        source = repo / SOURCE_RELATIVE_PATH[arm]
        if file_sha256(source) != EXPECTED_SOURCE_SHA256[arm]:
            raise RuntimeError(f"live {arm} source drift")
        kernels = []
        dispatches = []
        for mode in MODES:
            root, kernel, dispatch = overlay_paths(output, arm, mode)
            manifest = read_json(root / "identity.json")
            checks = {
                "registered": manifest == identity["arms"][arm]["modes"][mode],
                "arm": manifest.get("arm") == arm,
                "mode": manifest.get("mode") == mode,
                "enabled": bool(manifest.get("probe_enabled")) == (mode == PROBE),
                "kernel_hash": file_sha256(kernel)
                == manifest["overlay"]["kernel_sha256"],
                "dispatch_hash": file_sha256(dispatch)
                == manifest["overlay"]["dispatch_sha256"],
                "barriers": manifest["base"]["barrier_fingerprint"]
                == manifest["overlay"]["barrier_fingerprint"],
            }
            if not all(checks.values()):
                raise RuntimeError(f"{arm}/{mode} overlay drift: {checks}")
            kernels.append(kernel.read_bytes())
            dispatches.append(dispatch.read_text(encoding="utf-8"))
        if kernels[0] != kernels[1]:
            raise RuntimeError(f"{arm} control/probe kernel source differs")
        if normalize_dispatch_flag(dispatches[0]) != normalize_dispatch_flag(
            dispatches[1]
        ):
            raise RuntimeError(f"{arm} dispatch differs beyond marker flag")
    return identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, default=ROOT.parents[3])
    parser.add_argument("--output", type=Path, default=OVERLAY_ROOT)
    parser.add_argument("--check-existing", action="store_true")
    args = parser.parse_args()
    result = (
        verify_existing(args.flashinfer_root, args.output)
        if args.check_existing
        else build(args.flashinfer_root, args.output)
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
