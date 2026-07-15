#!/usr/bin/env python3
"""Measure the vLLM tensor-scaled FP8 Triton fused-MoE baseline."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

REFERENCE_ROOT = Path(
    os.environ.get("W4A4_MOE_BENCH_ROOT", "/workspace/source/w4a4_moe_bench")
).resolve()
sys.path.insert(0, str(REFERENCE_ROOT / "scripts"))

import bench_qwen35_w4a4_moe_backends as common  # noqa: E402
import torch  # noqa: E402

from fixture import load_fixture, validate_output  # noqa: E402

BACKEND = "triton_fp8"
RESULTS = Path(__file__).resolve().parent / "results"


def make_tensor_scaled_fp8_weights(
    *, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize native BF16 master weights outside the measured region."""
    from vllm import _custom_ops as ops

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    e = common.QWEN35_E
    h = common.QWEN35_HIDDEN
    n = common.QWEN35_I_TP
    w1 = torch.empty((e, 2 * n, h), dtype=torch.float8_e4m3fn, device=device)
    w2 = torch.empty((e, h, n), dtype=torch.float8_e4m3fn, device=device)
    w1_scale = torch.empty((e, 1, 1), dtype=torch.float32, device=device)
    w2_scale = torch.empty((e, 1, 1), dtype=torch.float32, device=device)

    for expert in range(e):
        w1_master = (
            torch.randn((2 * n, h), generator=generator, device=device) / 10
        ).to(torch.bfloat16)
        w2_master = (torch.randn((h, n), generator=generator, device=device) / 10).to(
            torch.bfloat16
        )
        w1[expert], w1_scale[expert] = ops.scaled_fp8_quant(w1_master)
        w2[expert], w2_scale[expert] = ops.scaled_fp8_quant(w2_master)

    return w1, w2, w1_scale, w2_scale


def build_launch(
    *,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
):
    import triton
    import vllm
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.config import (
        fp8_w8a8_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        get_moe_configs,
        try_get_optimal_moe_config,
    )

    # Only the scale is prepared here. fused_experts quantizes BF16 x inside
    # every measured forward, matching vLLM's NVFP4-vs-FP8 reference harness.
    _, a1_scale = ops.scaled_fp8_quant(x)
    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
    )
    tuned_configs = get_moe_configs(
        common.QWEN35_E,
        common.QWEN35_I_TP,
        "fp8_w8a8",
    )
    triton_config = try_get_optimal_moe_config(
        tuple(w1.shape),
        tuple(w2.shape),
        common.QWEN35_TOPK,
        "fp8_w8a8",
        x.shape[0],
    )

    def launch():
        kwargs = {}
        if "allow_deep_gemm" in inspect.signature(fused_experts).parameters:
            kwargs["allow_deep_gemm"] = False
        if (
            "allow_cutlass_block_scaled_grouped_gemm"
            in inspect.signature(fused_experts).parameters
        ):
            kwargs["allow_cutlass_block_scaled_grouped_gemm"] = False
        return fused_experts(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            quant_config=quant_config,
            **kwargs,
        )

    meta = {
        "backend_detail": "vllm.fused_experts",
        "quant_mode": "fp8_w8a8_tensor_scaled",
        "storage": "float8_e4m3fn_per_expert_tensor_scale",
        "implementation": "triton_fused_moe",
        "input_dtype": "bfloat16",
        "output_dtype": "bfloat16",
        "vllm_version": vllm.__version__,
        "triton_version": triton.__version__,
        "alternate_backends_allowed": "false",
        "triton_config": json.dumps(triton_config, sort_keys=True),
        "triton_config_source": "tuned_file" if tuned_configs else "default_heuristic",
    }
    return launch, meta


def run_case(
    *,
    m: int,
    args: argparse.Namespace,
    device: torch.device,
    l2_flush,
    l2_flush_bytes: int,
    weights,
) -> dict[str, object]:
    x, topk_ids, topk_weights, fixture_meta = load_fixture(args.fixture_dir, m, device)
    launch, meta = build_launch(
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w1=weights[0],
        w2=weights[1],
        w1_scale=weights[2],
        w2_scale=weights[3],
    )

    output = launch()
    torch.cuda.synchronize()
    validate_output(output, m)
    samples_ms = common.bench_call(
        launch,
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
        use_cuda_graph=args.cuda_graph,
        l2_flush=l2_flush,
    )
    median_ms = common.statistics.median(samples_ms)
    flops = common.full_fused_moe_flops(m)
    return {
        "backend": BACKEND,
        "m": m,
        "hidden": common.QWEN35_HIDDEN,
        "intermediate_tp": common.QWEN35_I_TP,
        "experts": common.QWEN35_E,
        "topk": common.QWEN35_TOPK,
        "timing": (
            "cuda_graph_events_inside" if args.cuda_graph else "direct_cuda_events"
        ),
        "flush_l2": int(args.flush_l2),
        "l2_flush_bytes": l2_flush_bytes,
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "median_us": median_ms * 1000.0,
        "tflops": flops / (median_ms / 1000.0) / 1.0e12,
        "samples_ms": ";".join(f"{value:.6f}" for value in samples_ms),
        "error": "",
        "gpu_uuid": args.gpu_uuid,
        "functional_sanity": "shape_dtype_finite_nonzero",
        **fixture_meta,
        **meta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-values", nargs="+", required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--cuda-graph", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--flush-l2", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--l2-flush-bytes", type=int, default=201326592)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--csv", type=Path, default=RESULTS / "triton_arm_raw.csv")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--print-traceback", action="store_true")
    args = parser.parse_args()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.m_values = common.parse_m_values(args.m_values)

    torch.cuda.set_device(args.device_index)
    device = torch.device("cuda", args.device_index)
    l2_flush, l2_flush_bytes = common.make_l2_flusher(
        enabled=args.flush_l2,
        requested_bytes=args.l2_flush_bytes,
        device=device,
    )
    print(
        f"device={torch.cuda.get_device_name(device)} "
        f"gpu_uuid={args.gpu_uuid} uuid_visible={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"capability={torch.cuda.get_device_capability(device)}"
    )
    print(
        "contract="
        f"E{common.QWEN35_E}-H{common.QWEN35_HIDDEN}-"
        f"I{common.QWEN35_I_TP}-topk{common.QWEN35_TOPK} "
        "fp8=tensor_scaled topk_outside_timing input_quant_inside_timing"
    )
    weights = make_tensor_scaled_fp8_weights(device=device, seed=args.seed)
    rows: list[dict[str, object]] = []
    for m in args.m_values:
        print(f"running backend={BACKEND} m={m}", flush=True)
        try:
            row = run_case(
                m=m,
                args=args,
                device=device,
                l2_flush=l2_flush,
                l2_flush_bytes=l2_flush_bytes,
                weights=weights,
            )
        except Exception as error:  # preserve the failed row in raw evidence
            if args.print_traceback:
                traceback.print_exc()
            row = common.error_row(
                backend=BACKEND,
                m=m,
                error=error,
                args=args,
                l2_flush_bytes=l2_flush_bytes,
            )
            if args.fail_fast:
                rows.append(row)
                common.write_csv(args.csv, rows)
                raise
        rows.append(row)
        print(row, flush=True)

    common.write_csv(args.csv, rows)
    print(f"wrote_csv={args.csv}")
    return int(any(row["error"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
