#!/usr/bin/env python3
"""Build matched exp_006 completion-anchored control/probe overlays.

The builder reuses exp_004's audited whole-kernel plumbing (base consumer,
CTA, and W4 markers) and changes only the task ABI plus the FC2 tile markers.
Both arms therefore come from the same production source and this same builder;
``--disabled`` changes only the compile-time marker-enable flag in dispatch.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parent
EXP004_ROOT = ROOT.parent / "exp_004_fused_phase_timing_breakdown"
if str(EXP004_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP004_ROOT))

import build_whole_kernel_probe as exp004_builder  # noqa: E402
from exp004_common import (  # noqa: E402
    DISPATCH_RELATIVE_PATH,
    EXPECTED_DISPATCH_SHA256,
    EXPECTED_KERNEL_SHA256,
    KERNEL_RELATIVE_PATH,
    file_sha256,
    write_json,
)

from exp006_common import (  # noqa: E402
    CONTROL,
    CTA_TICKS,
    EVENT_ABI,
    PROBE,
    TASK_TICKS,
    W4_BASE,
)


DEFAULT_OUTPUT = ROOT / "results" / "overlays" / PROBE


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _replace_exact(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one exact anchor, found {count}")
    return text.replace(old, new, 1)


def _task_store(event: str, *, indent: int) -> str:
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


@contextmanager
def _exp004_abi_override() -> Iterator[None]:
    """Temporarily parameterize exp_004's generator without editing it."""

    old_task_ticks = exp004_builder.TASK_TICKS
    old_w4_base = exp004_builder.W4_BASE
    old_constants = exp004_builder.KERNEL_CONSTANTS
    try:
        exp004_builder.TASK_TICKS = TASK_TICKS
        exp004_builder.W4_BASE = W4_BASE
        exp004_builder.KERNEL_CONSTANTS = old_constants.replace(
            f"_EXP004_TASK_TICKS = {old_task_ticks}",
            f"_EXP004_TASK_TICKS = {TASK_TICKS}",
        )
        yield
    finally:
        exp004_builder.TASK_TICKS = old_task_ticks
        exp004_builder.W4_BASE = old_w4_base
        exp004_builder.KERNEL_CONSTANTS = old_constants


def _validate_static_contract(source: str) -> dict[str, Any]:
    checks = {
        "task_slice_chunk_one": source.count("_TASK_SLICE_CHUNK = 1\n") == 1,
        "epi_tile_matches_mma_tiler": source.count(
            "        self.epi_tile = (mma_tiler_mn[0], mma_tiler_mn[1])\n"
        )
        == 1,
        "epi_rest_definition": source.count(
            "                epi_rest_m = self.tile_shape_mnk[0] // self.epi_tile[0]\n"
        )
        == 1,
        "fc2_output_loop": source.count(
            "                    for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]\n"
        )
        == 2,  # one W0-W3 consumer loop and one W4 producer loop
    }
    if not all(checks.values()):
        raise ValueError(f"production static contract drift: {checks}")
    return {"checks": checks, "gate_pass": True}


def _instrument_kernel(source: str) -> str:
    static_contract = _validate_static_contract(source)
    del static_contract
    with _exp004_abi_override():
        text = exp004_builder._instrument_kernel(source)

    # exp_004 records A/C only on W0.  exp_006 needs same-warp A/C/D/E/F
    # edges so no representative warp is used as another warp's boundary.
    for name, old_event, new_event in (
        ("A", "Int32(7) + output_tile_idx * Int32(3)",
         "Int32(9) + output_tile_idx * Int32(20) + warp_idx"),
        ("C", "Int32(8) + output_tile_idx * Int32(3)",
         "Int32(13) + output_tile_idx * Int32(20) + warp_idx"),
    ):
        old_block = (
            "                        if cutlass.const_expr(self.phase_probe_enabled):\n"
            "                            if warp_idx == Int32(0):\n"
            "                                if lane_id == Int32(0):\n"
            + _task_store(old_event, indent=36)
        )
        new_block = (
            "                        if cutlass.const_expr(self.phase_probe_enabled):\n"
            "                            if lane_id == Int32(0):\n"
            + _task_store(new_event, indent=32)
        )
        text = _replace_exact(
            text,
            old_block,
            new_block,
            label=f"per-warp {name} boundary",
        )

    # exp_004 records one W0-only post-barrier edge.  A single warp cannot be
    # used as a cross-warp timer boundary, so exp_006 records F0..F3: each
    # compute warp's lane0 reads the timer after the same collective barrier.
    old_f_block = (
        "                        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                            if warp_idx == Int32(0):\n"
        "                                if lane_id == Int32(0):\n"
        + _task_store(
            "Int32(9) + output_tile_idx * Int32(3)", indent=36
        )
    )
    new_f_block = (
        "                        if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                            if lane_id == Int32(0):\n"
        + _task_store(
            "Int32(25) + output_tile_idx * Int32(20) + warp_idx", indent=32
        )
    )
    text = _replace_exact(
        text,
        old_f_block,
        new_f_block,
        label="per-warp F after post-scatter barrier",
    )

    # Consecutive marker-pair calibration, once per task on W0 lane0.
    phase_b_header = (
        "                    # ============================================================\n"
        "                    # PHASE B: Sweep ALL FC2 output tiles using cached sA\n"
    )
    calibration = (
        "                    if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                        if warp_idx == Int32(0):\n"
        "                            if lane_id == Int32(0):\n"
        + _task_store("7", indent=32)
        + _task_store("8", indent=32)
        + "\n"
    )
    text = _replace_exact(
        text,
        phase_b_header,
        calibration + phase_b_header,
        label="marker-pair calibration",
    )

    # D0..D3: each compute warp's lane0 immediately after the pre-scatter
    # barrier and before that same warp executes any scatter-path instruction.
    pre_scatter_anchor = (
        "                            self.epilog_sync_barrier.arrive_and_wait()\n"
        "                            rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])\n"
    )
    d_marker = (
        "                            self.epilog_sync_barrier.arrive_and_wait()\n"
        "                            if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                                if lane_id == Int32(0):\n"
        + _task_store(
            "Int32(17) + output_tile_idx * Int32(20) + warp_idx", indent=36
        )
        + "                            rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])\n"
    )
    text = _replace_exact(
        text,
        pre_scatter_anchor,
        d_marker,
        label="per-warp D after pre-scatter barrier",
    )

    # E0..E3: each compute warp's lane0 records its own completion before it
    # reaches the collective post-scatter barrier.
    e_anchor = (
        "                                vec_idx += Int32(self.num_threads_per_warp)\n\n"
        "                            # Post-scatter barrier: needed to ensure all warps\n"
    )
    e_marker = (
        "                                vec_idx += Int32(self.num_threads_per_warp)\n\n"
        "                            if cutlass.const_expr(self.phase_probe_enabled):\n"
        "                                if lane_id == Int32(0):\n"
        + _task_store(
            "Int32(21) + output_tile_idx * Int32(20) + warp_idx", indent=36
        )
        + "\n"
        "                            # Post-scatter barrier: needed to ensure all warps\n"
    )
    text = _replace_exact(
        text,
        e_anchor,
        e_marker,
        label="per-warp E before post-scatter barrier",
    )

    if f"_EXP004_TASK_TICKS = {TASK_TICKS}" not in text:
        raise ValueError("task ABI constant was not updated")
    ast.parse(text, filename="completion_anchored/moe_dynamic_kernel.py")
    return text


def _instrument_dispatch(source: str, *, enabled: bool) -> str:
    with _exp004_abi_override():
        text = exp004_builder._instrument_dispatch(source, enabled=enabled)

    # This overlay is intentionally locked to the only shape for which the
    # 16-tile/epi_rest_m=1 ABI is valid.  The descriptor gate separately checks
    # slice_count==1 for every materialized runtime task.
    tiler_anchor = (
        "    mma_tiler_mn = (\n"
        "        _level_tile_m(activation_precision),\n"
        "        _level_tile_n(activation_precision),\n"
        "    )\n\n"
        "    cache_key = (\n"
    )
    tiler_guard = (
        "    mma_tiler_mn = (\n"
        "        _level_tile_m(activation_precision),\n"
        "        _level_tile_n(activation_precision),\n"
        "    )\n"
        "    exp006_locked_case = (E, m, k, n, num_topk)\n"
        "    if exp006_locked_case != (256, 8192, 2048, 512, 8):\n"
        "        raise RuntimeError(\n"
        "            f\"exp006 locked-case drift: {exp006_locked_case}\"\n"
        "        )\n"
        "    if mma_tiler_mn != (128, 128) or k // mma_tiler_mn[1] != 16:\n"
        "        raise RuntimeError(\n"
        "            f\"exp006 tile ABI drift: tiler={mma_tiler_mn}, hidden={k}\"\n"
        "        )\n\n"
        "    cache_key = (\n"
    )
    text = _replace_exact(
        text,
        tiler_anchor,
        tiler_guard,
        label="locked dynamic dispatch guard",
    )
    ast.parse(text, filename="completion_anchored/moe_dispatch.py")
    return text


def build(repo: Path, output: Path, *, enabled: bool = True) -> dict[str, Any]:
    repo = repo.resolve()
    output = output.resolve()
    kernel_path = repo / KERNEL_RELATIVE_PATH
    dispatch_path = repo / DISPATCH_RELATIVE_PATH
    if file_sha256(kernel_path) != EXPECTED_KERNEL_SHA256:
        raise ValueError("production kernel identity drift")
    if file_sha256(dispatch_path) != EXPECTED_DISPATCH_SHA256:
        raise ValueError("production dispatch identity drift")
    if output.exists():
        raise FileExistsError(f"immutable exp006 overlay exists: {output}")

    production_kernel = kernel_path.read_text()
    production_dispatch = dispatch_path.read_text()
    static_contract = _validate_static_contract(production_kernel)
    kernel = _instrument_kernel(production_kernel)
    dispatch = _instrument_dispatch(production_dispatch, enabled=enabled)

    output.mkdir(parents=True)
    kernel_output = output / "moe_dynamic_kernel.py"
    dispatch_output = output / "moe_dispatch.py"
    kernel_output.write_text(kernel)
    dispatch_output.write_text(dispatch)

    kernel_diff = "".join(
        difflib.unified_diff(
            production_kernel.splitlines(keepends=True),
            kernel.splitlines(keepends=True),
            fromfile="production/moe_dynamic_kernel.py",
            tofile=f"{PROBE}/moe_dynamic_kernel.py",
        )
    )
    dispatch_diff = "".join(
        difflib.unified_diff(
            production_dispatch.splitlines(keepends=True),
            dispatch.splitlines(keepends=True),
            fromfile="production/moe_dispatch.py",
            tofile=f"{PROBE}/moe_dispatch.py",
        )
    )
    (output / "moe_dynamic_kernel.diff").write_text(kernel_diff)
    (output / "moe_dispatch.diff").write_text(dispatch_diff)

    arm = PROBE if enabled else CONTROL
    manifest: dict[str, Any] = {
        "schema": "exp006.completion-overlay.v1",
        "arm": arm,
        "classification": "diagnostic-only" if enabled else "measurement-control",
        "probe_enabled": enabled,
        "timer": "%globaltimer",
        "event_abi": EVENT_ABI,
        "static_contract": static_contract,
        "base_builder": {
            "path": str(EXP004_ROOT / "build_whole_kernel_probe.py"),
            "sha256": file_sha256(EXP004_ROOT / "build_whole_kernel_probe.py"),
        },
        "production": {
            "kernel_sha256": EXPECTED_KERNEL_SHA256,
            "dispatch_sha256": EXPECTED_DISPATCH_SHA256,
        },
        "overlay": {
            "kernel_sha256": _sha256_text(kernel),
            "dispatch_sha256": _sha256_text(dispatch),
            "kernel_diff_sha256": _sha256_text(kernel_diff),
            "dispatch_diff_sha256": _sha256_text(dispatch_diff),
        },
        "anchors": {
            "A0_A3": "each compute warp lane0 before phase2_pipeline.consumer_try_wait",
            "C0_C3": "each compute warp lane0 after final OMMA issue",
            "D0_D3": "each compute warp lane0 after pre-scatter epilog_sync_barrier",
            "E0_E3": "each compute warp lane0 after scatter loop and before post barrier",
            "F0_F3": "each compute warp lane0 after post-scatter epilog_sync_barrier",
        },
        "cta_ticks_per_cta": CTA_TICKS,
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
        help="build matched plumbing with all timestamp writes compiled out",
    )
    args = parser.parse_args()
    manifest = build(
        args.flashinfer_root.resolve(),
        args.output.resolve(),
        enabled=not args.disabled,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
