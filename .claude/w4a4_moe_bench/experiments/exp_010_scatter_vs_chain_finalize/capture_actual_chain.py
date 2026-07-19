#!/usr/bin/env python3
"""Fresh actual CUTLASS graph replay for the Finalize fidelity anchor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXP002 = ROOT.parent / "exp_002_fused_vs_chain_dataflow"
if str(EXP002) not in sys.path:
    sys.path.insert(0, str(EXP002))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    import torch
    from run_exp002 import (
        build_arm,
        configure_source_checkout,
        make_canonical_weights,
        make_routed_fixture,
    )

    configure_source_checkout(args.flashinfer_root.resolve())
    device = torch.device("cuda", 0)
    weights = make_canonical_weights(device=device, seed=2026)
    fixture = make_routed_fixture(8192, device=device, seed=2026)
    arm = build_arm(
        "cutlass_bf16_chain",
        fixture=fixture,
        weights=weights,
        max_num_tokens=8192,
    )
    arm.eager()
    arm.capture()
    for _ in range(args.warmup):
        arm.replay_ms()

    torch.cuda.synchronize()
    cudart = torch.cuda.cudart()
    status = int(cudart.cudaProfilerStart())
    if status != 0:
        raise RuntimeError(f"cudaProfilerStart failed: {status}")
    torch.cuda.nvtx.range_push("exp010_actual_cutlass_chain_m8192")
    try:
        elapsed_us = arm.replay_ms() * 1000.0
    finally:
        torch.cuda.nvtx.range_pop()
        status = int(cudart.cudaProfilerStop())
        if status != 0:
            raise RuntimeError(f"cudaProfilerStop failed: {status}")
    payload = {
        "schema": "exp010.actual-chain-anchor.v1",
        "event_elapsed_us": elapsed_us,
        "arm_metadata": arm.metadata,
        "fixture": fixture.manifest,
        "weights": weights.manifest,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
