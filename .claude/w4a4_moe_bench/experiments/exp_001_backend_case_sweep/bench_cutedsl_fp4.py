#!/usr/bin/env python3
"""Measure a diagnostic CuteDSL repeat on the Triton-arm fixtures."""

from __future__ import annotations

import argparse
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

BACKEND = "flashinfer_cutedsl"
RESULTS = Path(__file__).resolve().parent / "results"


def run_case(
    *, m, args, device, l2_flush, l2_flush_bytes, weights
) -> dict[str, object]:
    x, topk_ids, topk_weights, fixture_meta = load_fixture(args.fixture_dir, m, device)
    launch, meta = common.build_flashinfer_cutedsl_launch(
        weights,
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        max_num_tokens=max(args.m_values),
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
        "timing": "cuda_graph_events_inside",
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
        "iket_overlay": "disabled",
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
    parser.add_argument(
        "--csv", type=Path, default=RESULTS / "triton_arm_cutedsl_diagnostic.csv"
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--print-traceback", action="store_true")
    parser.add_argument("--b12x-root", default=str(REFERENCE_ROOT / "b12x"))
    parser.add_argument("--flashinfer-root", default="/workspace/source/flashinfer")
    args = parser.parse_args()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.m_values = common.parse_m_values(args.m_values)
    if os.environ.get("FLASHINFER_CUTEDSL_IKET_OVERLAY", "0") != "0":
        raise RuntimeError("diagnostic repeat requires IKET overlay disabled")

    common.configure_import_paths(args)
    common.configure_flashinfer_source_checkout()
    torch.cuda.set_device(args.device_index)
    device = torch.device("cuda", args.device_index)
    l2_flush, l2_flush_bytes = common.make_l2_flusher(
        enabled=args.flush_l2,
        requested_bytes=args.l2_flush_bytes,
        device=device,
    )
    print(
        f"device={torch.cuda.get_device_name(device)} gpu_uuid={args.gpu_uuid} "
        f"capability={torch.cuda.get_device_capability(device)}"
    )
    weights = common.make_modelopt_nvfp4_weights(device=device, seed=args.seed)
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
        except Exception as error:
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
