#!/usr/bin/env python3
"""One immutable quick E2E benchmark sample for an exp_013 arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
EXP005 = ROOT.parent / "exp_005_8warp_spill_reduction"
sys.path.insert(0, str(EXP005))
import run_exp005_arm as worker  # noqa: E402


INTERNAL_ARM = "candidate_8warp_n64_temporal_replay_v0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--expected-overlay-sha256", required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument(
        "--external-arm", choices=("exp008", "exp013_v2"), required=True
    )
    parser.add_argument("--m", type=int, choices=(256, 8192), required=True)
    parser.add_argument("--group", type=int, choices=range(2), required=True)
    parser.add_argument("--position", type=int, choices=range(4), required=True)
    parser.add_argument("--warmup", type=int, default=5, choices=(5,))
    parser.add_argument("--iters", type=int, default=50, choices=(50,))
    parser.add_argument("--device-index", type=int, default=0, choices=(0,))
    parser.add_argument("--seed", type=int, default=2026, choices=(2026,))
    args = parser.parse_args()

    args.flashinfer_root = args.flashinfer_root.resolve()
    args.overlay = args.overlay.resolve()
    args.jit_root = args.jit_root.resolve()
    args.output = args.output.resolve()
    args.arm = INTERNAL_ARM
    args.fixture = "canonical"
    args.results = args.output.parent
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite immutable sample: {args.output}")

    source = worker.validate_source(args.flashinfer_root, args.overlay, args.arm)
    if source["overlay_sha256"] != args.expected_overlay_sha256:
        raise RuntimeError("overlay hash drift")
    artifacts_before = worker.artifact_manifest(args.jit_root)
    if not any(item["path"].endswith(".cubin") for item in artifacts_before):
        raise RuntimeError("benchmark requires a precompiled JIT root")

    if str(args.flashinfer_root) not in sys.path:
        sys.path.insert(0, str(args.flashinfer_root))
    worker.install_overlay(args.overlay)
    imports = worker.configure_source_checkout(args.flashinfer_root)
    runtime = worker.runtime_identity(args, source)
    runtime["imports"] = imports

    _, fixture, weights = worker.make_case(args)
    arm = worker.build_arm(args, fixture, weights)
    arm.eager()
    arm.capture()
    flush, flush_bytes = worker.make_flusher(fixture.x.device)
    for _ in range(args.warmup):
        flush()
        arm.replay()
    total_ms = 0.0
    for _ in range(args.iters):
        flush()
        _, elapsed_ms = arm.replay()
        total_ms += elapsed_ms

    artifacts_after = worker.artifact_manifest(args.jit_root)
    if worker.canonical_sha256(artifacts_after) != worker.canonical_sha256(
        artifacts_before
    ):
        raise RuntimeError("JIT artifact drift during benchmark")
    payload = {
        "schema": "exp013.quick-e2e-sample.v1",
        "status": "complete",
        "external_arm": args.external_arm,
        "m": args.m,
        "group": args.group,
        "position": args.position,
        "sample_us": total_ms * 1000.0 / args.iters,
        "warmup": args.warmup,
        "iters": args.iters,
        "l2_flush_bytes": flush_bytes,
        "timing": "outer CUDA graph with external CUDA events",
        "fixture": fixture.manifest,
        "runtime": runtime,
        "jit_artifact_set_sha256": worker.canonical_sha256(artifacts_after),
    }
    worker.write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
