#!/usr/bin/env python3
"""Build the diagnostic exp_004 probe used for approximate phase shares.

The formal probe remains immutable.  This follow-up derives a separate overlay
directly from production, inlines every timing store, and reloads the claimed
task slot with a side-effecting volatile shared-memory load.  The latter avoids
the stale shared-control value that sent the original probe writes out of
bounds.
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
DEFAULT_OUTPUT = ROOT / "results" / "overlays" / "diagnostic_probe"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _consumer_store(interval: str, edge: int, *, indent: int) -> str:
    pad = " " * indent
    return (
        f"{pad}exp004_event_index = (\n"
        f"{pad}    ((\n"
        f"{pad}        task_slot_probe * Int32(_EXP004_CONSUMER_WARPS)\n"
        f"{pad}        + Int32(warp_idx)\n"
        f"{pad}    ) * Int32(_EXP004_CONSUMER_INTERVALS)\n"
        f"{pad}    + Int32({interval})) * Int32(_EXP004_EDGES)\n"
        f"{pad}    + Int32({edge})\n"
        f"{pad})\n"
        f"{pad}st_global_u64(\n"
        f"{pad}    get_ptr_as_int64(timing_ticks, exp004_event_index),\n"
        f"{pad}    exp004_tick,\n"
        f"{pad})\n"
    )


def _w4_store(interval: int, edge: int, *, indent: int) -> str:
    pad = " " * indent
    return (
        f"{pad}exp004_event_index = (\n"
        f"{pad}    exp004_task_capacity * Int32(\n"
        f"{pad}        _EXP004_CONSUMER_WARPS\n"
        f"{pad}        * _EXP004_CONSUMER_INTERVALS\n"
        f"{pad}        * _EXP004_EDGES\n"
        f"{pad}    )\n"
        f"{pad}    + (\n"
        f"{pad}        task_slot_probe * Int32(_EXP004_W4_INTERVALS)\n"
        f"{pad}        + Int32({interval})\n"
        f"{pad}    ) * Int32(_EXP004_EDGES)\n"
        f"{pad}    + Int32({edge})\n"
        f"{pad})\n"
        f"{pad}st_global_u64(\n"
        f"{pad}    get_ptr_as_int64(timing_ticks, exp004_event_index),\n"
        f"{pad}    exp004_tick,\n"
        f"{pad})\n"
    )


def build(repo: Path, output: Path) -> dict[str, object]:
    repo = repo.resolve()
    kernel_path = repo / KERNEL_RELATIVE_PATH
    dispatch_path = repo / DISPATCH_RELATIVE_PATH
    if file_sha256(kernel_path) != EXPECTED_KERNEL_SHA256:
        raise ValueError("production kernel identity drift")
    if file_sha256(dispatch_path) != EXPECTED_DISPATCH_SHA256:
        raise ValueError("production dispatch identity drift")
    if output.exists():
        raise FileExistsError(f"immutable diagnostic overlay exists: {output}")

    production_kernel = kernel_path.read_text()
    production_dispatch = dispatch_path.read_text()
    original_methods = base_builder.KERNEL_METHODS
    original_consumer_store = base_builder._consumer_store
    original_w4_store = base_builder._w4_store
    try:
        base_builder.KERNEL_METHODS = ""
        base_builder._consumer_store = _consumer_store
        base_builder._w4_store = _w4_store
        kernel = base_builder._instrument_kernel(production_kernel)
    finally:
        base_builder.KERNEL_METHODS = original_methods
        base_builder._consumer_store = original_consumer_store
        base_builder._w4_store = original_w4_store

    kernel = base_builder._replace_exact(
        kernel,
        "                            task_cta_z[task_slot_probe] = Int32(bidz)\n",
        "                            st_global_i32(\n"
        "                                get_ptr_as_int64(task_cta_z, task_slot_probe),\n"
        "                                Int32(bidz),\n"
        "                            )\n",
        label="CTA-map inline store",
    )
    clock_helper = """@dsl_user_op
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
    volatile_helper = (
        clock_helper
        + """@dsl_user_op
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
    )
    kernel = base_builder._replace_exact(
        kernel,
        clock_helper,
        volatile_helper,
        label="volatile claimed-slot helper",
    )
    old_load = "task_slot_probe = _ld_shared_i32(ctrl_base_addr + Int32(28))"
    new_load = (
        "task_slot_probe = _exp004_ld_shared_volatile_i32(\n"
        "                        ctrl_base_addr + Int32(28)\n"
        "                    )"
    )
    if kernel.count(old_load) != 2:
        raise ValueError("claimed-slot load count drift")
    kernel = kernel.replace(old_load, new_load)

    dispatch = base_builder._instrument_dispatch(production_dispatch, enabled=True)
    ast.parse(kernel, filename="diagnostic_probe/moe_dynamic_kernel.py")
    ast.parse(dispatch, filename="diagnostic_probe/moe_dispatch.py")

    output.mkdir(parents=True)
    (output / "moe_dynamic_kernel.py").write_text(kernel)
    (output / "moe_dispatch.py").write_text(dispatch)
    kernel_diff = "".join(
        difflib.unified_diff(
            production_kernel.splitlines(keepends=True),
            kernel.splitlines(keepends=True),
            fromfile="production/moe_dynamic_kernel.py",
            tofile="diagnostic_probe/moe_dynamic_kernel.py",
        )
    )
    dispatch_diff = "".join(
        difflib.unified_diff(
            production_dispatch.splitlines(keepends=True),
            dispatch.splitlines(keepends=True),
            fromfile="production/moe_dispatch.py",
            tofile="diagnostic_probe/moe_dispatch.py",
        )
    )
    (output / "moe_dynamic_kernel.diff").write_text(kernel_diff)
    (output / "moe_dispatch.diff").write_text(dispatch_diff)
    manifest: dict[str, object] = {
        "schema": "exp004.diagnostic-probe-overlay.v1",
        "classification": "diagnostic-only",
        "diagnosis": {
            "fact": "probe indexing path used an out-of-range task-slot value",
            "compiler_mechanism_inference": (
                "has_side_effects=False allowed invalid shared-load reuse or motion"
            ),
            "observed_stale_slot_examples": [65790, 65860],
            "task_capacity": 3068,
            "effect": "computed timing indices were outside the probe buffer",
        },
        "changed_bundle": [
            "inline explicit global phase and CTA-map stores",
            "reload claimed task slot with ld.volatile.shared.s32",
        ],
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
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.flashinfer_root.resolve(), args.output.resolve()),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
