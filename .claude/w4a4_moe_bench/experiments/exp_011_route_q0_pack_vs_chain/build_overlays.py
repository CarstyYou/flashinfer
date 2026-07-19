#!/usr/bin/env python3
"""Build immutable full-kernel overlays for the exp_011 P3 controls.

This builder reuses the audited exp_004 whole-kernel timing ABI.  Every arm
keeps the complete fused kernel and its 110 x 160 launch; only the P3 source
delta named by the arm is applied.  Probe/no-marker selection remains a
compile-time dispatch constant, never a runtime mode branch in the kernel.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
EXP004 = ROOT.parent / "exp_004_fused_phase_timing_breakdown"
if str(EXP004) not in sys.path:
    sys.path.insert(0, str(EXP004))

import build_whole_kernel_probe as whole  # noqa: E402
from exp004_common import (  # noqa: E402
    DISPATCH_RELATIVE_PATH,
    EXPECTED_DISPATCH_SHA256,
    EXPECTED_KERNEL_SHA256,
    KERNEL_RELATIVE_PATH,
    file_sha256,
    write_json,
)


VARIANTS = (
    "identity",
    "shared_equal_scale",
    "static_schedule",
    "precomputed_phys_row",
)
MODES = ("probe", "no_marker")
EXPERT_BITS = 8
EXPERT_MASK = (1 << EXPERT_BITS) - 1


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _replace(text: str, old: str, new: str, *, label: str) -> str:
    return whole._replace(text, old, new, label=label)


def _identity(source: str) -> str:
    return source


def _static_schedule(source: str) -> str:
    """Replace the global P3 batch claim with a fixed CTA-round mapping."""

    text = _replace(
        source,
        "        produce_active = Int32(1)\n        while produce_active > Int32(0):\n",
        "        produce_active = Int32(1)\n"
        "        producer_round = Int32(0)\n"
        "        while produce_active > Int32(0):\n",
        label="static schedule round state",
    )
    text = _replace(
        text,
        "                batch_base = atomic_add_global_i32(\n"
        "                    get_ptr_as_int64(pair_head, Int32(0)),\n"
        "                    claim_count,\n"
        "                )\n",
        "                # exp_011 static schedule: preserve the ten-pair batch and\n"
        "                # CTA barrier, but assign it by deterministic CTA round.\n"
        "                batch_base = (\n"
        "                    producer_round * Int32(gdim_z) + Int32(bidz)\n"
        "                ) * claim_count\n",
        label="static schedule pair claim",
    )
    text = _replace(
        text,
        "            batch_base = _ld_shared_i32(ctrl_base_addr + Int32(28))\n"
        "            producer_limit = total_pairs\n",
        "            batch_base = _ld_shared_i32(ctrl_base_addr + Int32(28))\n"
        "            producer_round += Int32(1)\n"
        "            producer_limit = total_pairs\n",
        label="static schedule round advance",
    )
    return text


def _precomputed_phys_row(source: str) -> str:
    """Decode ``(expert_rank << 8) | expert`` and remove row allocation atomic.

    The encoded route tensor is constructed by ``capture_arm.py`` outside the
    timed boundary.  E=256 is locked by the experiment plan, so the low eight
    bits recover the original expert id and the remaining bits are its stable
    occurrence rank in canonical pair order.
    """

    text = _replace(
        source,
        "            expert_id = topk_ids[hist_idx].to(Int32)\n",
        f"            expert_id = topk_ids[hist_idx].to(Int32) & Int32({EXPERT_MASK})\n",
        label="precomputed histogram expert decode",
    )
    text = _replace(
        text,
        "                                expert_id = topk_ids[pair_idx].to(Int32)\n"
        "                                weight = topk_weights[pair_idx].to(cutlass.Float32)\n",
        f"                                expert_id = (\n"
        f"                                    topk_ids[pair_idx].to(Int32) & Int32({EXPERT_MASK})\n"
        "                                )\n"
        "                                weight = topk_weights[pair_idx].to(cutlass.Float32)\n",
        label="precomputed shared-branch expert decode",
    )
    text = _replace(
        text,
        "                        if pair_idx < total_pairs:\n"
        "                            expert_id = topk_ids[pair_idx].to(Int32)\n"
        "                            token_idx = pair_idx // num_topk\n",
        "                        if pair_idx < total_pairs:\n"
        "                            encoded_route = topk_ids[pair_idx].to(Int32)\n"
        f"                            expert_id = encoded_route & Int32({EXPERT_MASK})\n"
        "                            token_idx = pair_idx // num_topk\n",
        label="precomputed pair expert/rank decode",
    )
    text = _replace(
        text,
        "                            if lane_id == Int32(0):\n"
        "                                row = atomic_add_global_i32(\n"
        "                                    get_ptr_as_int64(expert_write_rows, expert_id),\n"
        "                                    Int32(1),\n"
        "                                )\n"
        "                                phys_tile = expert_tile_base[expert_id] + row // Int32(\n",
        "                            if lane_id == Int32(0):\n"
        f"                                row = encoded_route >> Int32({EXPERT_BITS})\n"
        "                                phys_tile = expert_tile_base[expert_id] + row // Int32(\n",
        label="precomputed row allocation replacement",
    )
    return text


TRANSFORMS: dict[str, Callable[[str], str]] = {
    "identity": _identity,
    "shared_equal_scale": _identity,
    "static_schedule": _static_schedule,
    "precomputed_phys_row": _precomputed_phys_row,
}


def _diff(left: str, right: str, *, left_name: str, right_name: str) -> str:
    return "".join(
        difflib.unified_diff(
            left.splitlines(keepends=True),
            right.splitlines(keepends=True),
            fromfile=left_name,
            tofile=right_name,
        )
    )


def _mechanism_gate(variant: str, source: str) -> dict[str, Any]:
    static_claim = "get_ptr_as_int64(pair_head, Int32(0))"
    row_atomic = "get_ptr_as_int64(expert_write_rows, expert_id)"
    checks: dict[str, bool] = {
        "full_kernel_compute_retained": "Consumer steady state" in source,
        "p3_resident_grid_barrier_retained": source.count(
            "self._resident_grid_barrier("
        )
        >= 4,
        "no_runtime_variant_branch": "exp011_variant" not in source,
    }
    if variant == "static_schedule":
        checks.update(
            {
                "pair_head_claim_removed": static_claim not in source,
                "cta_round_mapping_present": "producer_round * Int32(gdim_z)" in source,
                "expert_row_atomic_retained": row_atomic in source,
            }
        )
    elif variant == "precomputed_phys_row":
        checks.update(
            {
                "pair_head_claim_retained": static_claim in source,
                # The shared-input branch remains in source but is eliminated by
                # the locked False specialization for this arm.  The ordinary
                # routed-pair branch must be the only occurrence removed.
                "ordinary_expert_row_atomic_removed": source.count(row_atomic) == 1,
                "encoded_rank_decode_present": "encoded_route >> Int32(8)" in source,
            }
        )
    else:
        checks.update(
            {
                "pair_head_claim_retained": static_claim in source,
                "expert_row_atomic_retained": row_atomic in source,
            }
        )
    return {"checks": checks, "gate_pass": all(checks.values())}


def build(repo: Path, output: Path) -> dict[str, Any]:
    repo = repo.resolve()
    output = output.resolve()
    kernel_path = repo / KERNEL_RELATIVE_PATH
    dispatch_path = repo / DISPATCH_RELATIVE_PATH
    if file_sha256(kernel_path) != EXPECTED_KERNEL_SHA256:
        raise ValueError("production kernel identity drift")
    if file_sha256(dispatch_path) != EXPECTED_DISPATCH_SHA256:
        raise ValueError("production dispatch identity drift")
    if output.exists():
        raise FileExistsError(f"immutable exp_011 overlay root exists: {output}")

    production_kernel = kernel_path.read_text()
    production_dispatch = dispatch_path.read_text()
    instrumented_kernel = whole._instrument_kernel(production_kernel)
    output.mkdir(parents=True)
    arms: dict[str, Any] = {}

    for variant in VARIANTS:
        variant_kernel = TRANSFORMS[variant](instrumented_kernel)
        mechanism_gate = _mechanism_gate(variant, variant_kernel)
        if not mechanism_gate["gate_pass"]:
            raise ValueError(f"{variant} mechanism gate failed: {mechanism_gate}")
        for mode in MODES:
            enabled = mode == "probe"
            dispatch = whole._instrument_dispatch(production_dispatch, enabled=enabled)
            arm_dir = output / variant / mode
            arm_dir.mkdir(parents=True)
            ast.parse(
                variant_kernel, filename=f"{variant}/{mode}/moe_dynamic_kernel.py"
            )
            ast.parse(dispatch, filename=f"{variant}/{mode}/moe_dispatch.py")
            kernel_out = arm_dir / "moe_dynamic_kernel.py"
            dispatch_out = arm_dir / "moe_dispatch.py"
            kernel_out.write_text(variant_kernel)
            dispatch_out.write_text(dispatch)
            kernel_diff = _diff(
                production_kernel,
                variant_kernel,
                left_name="production/moe_dynamic_kernel.py",
                right_name=f"{variant}/{mode}/moe_dynamic_kernel.py",
            )
            dispatch_diff = _diff(
                production_dispatch,
                dispatch,
                left_name="production/moe_dispatch.py",
                right_name=f"{variant}/{mode}/moe_dispatch.py",
            )
            (arm_dir / "moe_dynamic_kernel.diff").write_text(kernel_diff)
            (arm_dir / "moe_dispatch.diff").write_text(dispatch_diff)
            arms[f"{variant}/{mode}"] = {
                "variant": variant,
                "mode": mode,
                "probe_enabled": enabled,
                "kernel_sha256": _sha256_text(variant_kernel),
                "dispatch_sha256": _sha256_text(dispatch),
                "kernel_diff_sha256": _sha256_text(kernel_diff),
                "dispatch_diff_sha256": _sha256_text(dispatch_diff),
                "mechanism_gate": mechanism_gate,
            }

    # Probe/no-marker pairs must differ only in their dispatch compile flag.
    for variant in VARIANTS:
        probe = (output / variant / "probe" / "moe_dynamic_kernel.py").read_text()
        no_marker = (
            output / variant / "no_marker" / "moe_dynamic_kernel.py"
        ).read_text()
        if probe != no_marker:
            raise AssertionError(f"{variant} probe/no-marker kernel source drift")

    manifest: dict[str, Any] = {
        "schema": "exp011.full-kernel-overlays.v1",
        "production": {
            "kernel_sha256": EXPECTED_KERNEL_SHA256,
            "dispatch_sha256": EXPECTED_DISPATCH_SHA256,
        },
        "launch_contract": {"grid": [1, 1, 110], "block": [160, 1, 1]},
        "route_encoding": {
            "applies_to": "precomputed_phys_row",
            "format": "(expert_occurrence_rank << 8) | expert_id",
            "expert_bits": EXPERT_BITS,
            "expert_mask": EXPERT_MASK,
        },
        "arms": arms,
    }
    write_json(output / "identity.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "overlays",
    )
    args = parser.parse_args()
    manifest = build(args.flashinfer_root, args.output)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
