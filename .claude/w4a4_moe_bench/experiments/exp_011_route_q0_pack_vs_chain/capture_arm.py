#!/usr/bin/env python3
"""Capture one exp_011 full-kernel P3 arm.

The GPU orchestration, fixture, phase ABI, correctness checks, and runtime
identity gates are inherited from exp_004.  This wrapper only supplies the
exp_011 input transform and the variant-aware workspace contract.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
EXP004 = ROOT.parent / "exp_004_fused_phase_timing_breakdown"
if str(EXP004) not in sys.path:
    sys.path.insert(0, str(EXP004))

import run_exp004_arm as worker  # noqa: E402
import run_whole_kernel_capture as whole_capture  # noqa: E402
from exp004_common import (  # noqa: E402
    E,
    EXPECTED_GRID,
    H,
    I,
    M,
    MEASURED_REPLAYS,
    MEASUREMENT_CONTROL,
    PROBE,
    TOPK,
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


_ORIGINAL_BUILD_ARM = worker.build_arm
_ORIGINAL_WORKSPACE_SNAPSHOT = worker.workspace_snapshot
_ARM_CONTRACT: dict[str, Any] = {}


def _terminal_claim_head(limit: int, claim_count: int) -> int:
    productive = (limit + claim_count - 1) // claim_count
    terminal = int(EXPECTED_GRID[2])
    return (productive + terminal) * claim_count


def _encode_precomputed_rows(ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    flat = ids.detach().reshape(-1).cpu().tolist()
    counts = [0] * E
    encoded: list[int] = []
    for raw in flat:
        expert = int(raw)
        if not 0 <= expert <= EXPERT_MASK:
            raise ValueError(f"expert id {expert} does not fit the locked 8-bit field")
        rank = counts[expert]
        counts[expert] += 1
        encoded.append((rank << EXPERT_BITS) | expert)
    encoded_tensor = torch.tensor(
        encoded,
        dtype=torch.int32,
        device=ids.device,
    ).reshape_as(ids)
    decoded = encoded_tensor & EXPERT_MASK
    if not torch.equal(decoded, ids.to(torch.int32)):
        raise RuntimeError("precomputed route encoding is not reversible")
    expected_counts = torch.bincount(ids.reshape(-1).long(), minlength=E).cpu()
    if counts != expected_counts.tolist():
        raise RuntimeError("precomputed occurrence ranks do not close the histogram")
    return encoded_tensor, {
        "encoding": "(expert_occurrence_rank << 8) | expert_id",
        "expert_bits": EXPERT_BITS,
        "encoded_topk_sha256": worker.tensor_sha256(encoded_tensor),
        "decoded_topk_sha256": worker.tensor_sha256(decoded),
        "rank_histogram_sha256": worker.tensor_sha256(expected_counts),
        "rank_max": max(counts) - 1,
        "constructed_outside_timed_boundary": True,
        "gate_pass": True,
    }


def _build_variant_arm(variant: str, fixture: Any, weights: Any) -> worker.CapturedArm:
    global _ARM_CONTRACT

    if variant in {"identity", "static_schedule"}:
        _ARM_CONTRACT = {
            "variant": variant,
            "input_transform": "none",
            "topk_ids_sha256": worker.tensor_sha256(fixture.topk_ids),
            "gate_pass": True,
        }
        return _ORIGINAL_BUILD_ARM(fixture, weights)

    from flashinfer.fused_moe.cute_dsl import B12xMoEWrapper

    values = weights.cutedsl()
    topk_ids = fixture.topk_ids
    w1_alpha = values["w1_alpha"]
    if variant == "shared_equal_scale":
        alpha_bits = w1_alpha.contiguous().view(torch.int32)
        all_equal = bool(torch.all(alpha_bits == alpha_bits[0]).item())
        if not all_equal:
            raise RuntimeError("shared_equal_scale requires bitwise-equal W1 scales")
        w1_alpha = w1_alpha[:1].contiguous()
        _ARM_CONTRACT = {
            "variant": variant,
            "input_transform": "bitwise-equal per-expert W1 alpha normalized to scalar",
            "original_w1_alpha_shape": list(values["w1_alpha"].shape),
            "runtime_w1_alpha_shape": list(w1_alpha.shape),
            "bitwise_equal_across_experts": all_equal,
            "expected_share_input_specialization": True,
            "expected_productive_claims": math.ceil(M / 5),
            "expected_terminal_pair_head": _terminal_claim_head(M, 5),
            "gate_pass": all_equal and w1_alpha.numel() == 1,
        }
    elif variant == "precomputed_phys_row":
        topk_ids, encoding = _encode_precomputed_rows(fixture.topk_ids)
        _ARM_CONTRACT = {
            "variant": variant,
            "input_transform": "precomputed stable per-expert physical-row rank",
            **encoding,
            "expected_expert_write_rows": 0,
        }
    else:
        raise AssertionError(variant)

    wrapper = B12xMoEWrapper(
        num_experts=E,
        top_k=TOPK,
        hidden_size=H,
        intermediate_size=I,
        use_cuda_graph=True,
        max_num_tokens=M,
        output_dtype=torch.bfloat16,
        device=str(fixture.x.device),
        activation="silu",
        quant_mode="w4a4",
        source_format="modelopt",
    )

    if variant == "shared_equal_scale":
        # The public wrapper uses w1_alpha both as the P3 input scale and as
        # the per-expert FC1 alpha retained by _WeightViews.  Keep those roles
        # separate: scalar input_gs selects quantize-once/fanout, while the
        # bitwise-equivalent [E] alpha remains valid for compute indexing.
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            _get_weight_views,
            launch_sm120_moe,
        )

        weight_views = _get_weight_views(
            w1_fp4=values["w1_fp4"],
            w1_blockscale=values["w1_sf"],
            w2_fp4=values["w2_fp4"],
            w2_blockscale=values["w2_sf"],
            w1_alphas=values["w1_alpha"],
            w2_alphas=values["w2_alpha"],
            n=I,
            k=H,
            activation_precision=wrapper.activation_precision,
        )

        def launch() -> torch.Tensor:
            return launch_sm120_moe(
                a=fixture.x,
                topk_ids=topk_ids,
                topk_weights=fixture.topk_weights,
                w1_weight=values["w1_fp4"],
                w1_weight_sf=values["w1_sf"],
                w1_alpha=w1_alpha,
                fc2_input_scale=values["fc2_input_scale"],
                w2_weight=values["w2_fp4"],
                w2_weight_sf=values["w2_sf"],
                w2_alpha=values["w2_alpha"],
                num_experts=E,
                top_k=TOPK,
                num_local_experts=E,
                scatter_output=wrapper._moe_output[: fixture.m],
                activation=wrapper.activation,
                swiglu_alpha=wrapper.swiglu_alpha,
                swiglu_beta=wrapper.swiglu_beta,
                swiglu_limit=wrapper.swiglu_limit,
                activation_precision=wrapper.activation_precision,
                quant_mode=wrapper.quant_mode,
                source_format=wrapper.source_format,
                _workspace=wrapper._dynamic_workspace,
                _weight_views=weight_views,
            )

        _ARM_CONTRACT["compute_w1_alpha_shape"] = list(weight_views.w1_alpha.shape)
        _ARM_CONTRACT["input_and_compute_scale_roles_separated"] = True
    else:

        def launch() -> torch.Tensor:
            return wrapper.run(
                x=fixture.x,
                w1_weight=values["w1_fp4"],
                w1_weight_sf=values["w1_sf"],
                w2_weight=values["w2_fp4"],
                w2_weight_sf=values["w2_sf"],
                token_selected_experts=topk_ids,
                token_final_scales=fixture.topk_weights,
                w1_alpha=w1_alpha,
                w2_alpha=values["w2_alpha"],
                fc2_input_scale=values["fc2_input_scale"],
            )

    return worker.CapturedArm(launch, wrapper)


def _variant_workspace_snapshot(
    variant: str, wrapper: Any, fixture: Any
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors, payload = _ORIGINAL_WORKSPACE_SNAPSHOT(wrapper, fixture)
    workspace = wrapper._dynamic_workspace
    verification = dict(payload["verification"])
    checks = dict(verification["checks"])
    variant_checks: dict[str, bool] = {}

    if variant == "shared_equal_scale":
        observed = int(workspace.pair_head.item())
        expected = _terminal_claim_head(M, 5)
        checks["pair_head"] = observed == expected
        variant_checks["shared_token_claim_head"] = observed == expected
    elif variant == "static_schedule":
        observed = int(workspace.pair_head.item())
        checks["pair_head"] = observed == 0
        variant_checks["pair_head_unused"] = observed == 0
    elif variant == "precomputed_phys_row":
        zero = bool(torch.all(workspace.expert_write_rows == 0).item())
        checks["expert_write_rows"] = zero
        variant_checks["expert_write_rows_unused"] = zero

    verification["checks"] = checks
    verification["gate_pass"] = all(checks.values()) and all(variant_checks.values())
    verification["exp011_variant_checks"] = variant_checks
    payload["verification"] = verification
    payload["exp011_variant"] = variant
    return tensors, payload


def _install_variant_hooks(variant: str) -> None:
    def build_arm(fixture: Any, weights: Any) -> worker.CapturedArm:
        return _build_variant_arm(variant, fixture, weights)

    def workspace_snapshot(
        wrapper: Any, fixture: Any
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return _variant_workspace_snapshot(variant, wrapper, fixture)

    worker.build_arm = build_arm
    worker.workspace_snapshot = workspace_snapshot


def capture(args: argparse.Namespace) -> dict[str, Any]:
    mode = args.mode
    exp004_arm = PROBE if mode == "probe" else MEASUREMENT_CONTROL
    overlay = args.overlay_root.resolve() / args.variant / mode
    kernel_overlay = overlay / "moe_dynamic_kernel.py"
    dispatch_overlay = overlay / "moe_dispatch.py"
    if not kernel_overlay.is_file() or not dispatch_overlay.is_file():
        raise FileNotFoundError(
            f"missing {args.variant}/{mode} overlay; run build_overlays.py first"
        )
    _install_variant_hooks(args.variant)
    delegated = argparse.Namespace(
        flashinfer_root=args.flashinfer_root,
        arm=exp004_arm,
        kernel_overlay=kernel_overlay,
        dispatch_overlay=dispatch_overlay,
        jit_root=args.jit_root,
        output=args.output,
        expected_gpu_uuid=args.expected_gpu_uuid,
        warmup=args.warmup,
        replays=args.replays,
    )
    payload = whole_capture.capture(delegated)
    exp011 = {
        "schema": "exp011.route-q0-pack-arm-contract.v1",
        "variant": args.variant,
        "mode": mode,
        "exp004_arm": exp004_arm,
        "overlay": {
            "kernel": str(kernel_overlay),
            "kernel_sha256": file_sha256(kernel_overlay),
            "dispatch": str(dispatch_overlay),
            "dispatch_sha256": file_sha256(dispatch_overlay),
        },
        "launch_contract": {"grid": list(EXPECTED_GRID), "block": [160, 1, 1]},
        "arm_contract": dict(_ARM_CONTRACT),
    }
    payload["exp011"] = exp011
    write_json(args.output.resolve() / "capture.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--replays", type=int, default=MEASURED_REPLAYS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = capture(args)
    print(
        json.dumps(
            {
                "variant": args.variant,
                "mode": args.mode,
                "latency_us": payload["latency_us"],
                "arm_contract": payload["exp011"]["arm_contract"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
