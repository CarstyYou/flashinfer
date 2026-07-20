#!/usr/bin/env python3
"""Correctness and identity gate for the exp_013 compact candidate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
EXP005 = ROOT.parent / "exp_005_8warp_spill_reduction"
sys.path.insert(0, str(EXP005))
import run_exp005_arm as worker  # noqa: E402


INTERNAL_ARM = "candidate_8warp_n64_temporal_replay_v0"
EXPECTED_BLOCK = (288, 1, 1)
BOUNDARY_ROWS = {
    "rows_63": 63,
    "rows_64": 64,
    "rows_65": 65,
}


def make_case(args: argparse.Namespace):
    fixture_module = worker.load_fixture_module()
    device = worker.torch.device("cuda", args.device_index)
    base = fixture_module.make_routed_fixture(args.m, device=device, seed=args.seed)
    if args.fixture in BOUNDARY_ROWS:
        count = BOUNDARY_ROWS[args.fixture]
        token = worker.torch.arange(args.m, device=device, dtype=worker.torch.int64)[
            :, None
        ]
        slot = worker.torch.arange(
            worker.TOPK, device=device, dtype=worker.torch.int64
        )[None, :]
        ids = ((token * 11 + slot) % (worker.E - 1) + 1).to(worker.torch.int32)
        ids[:count, 0] = 0
        sorted_ids = ids.sort(dim=-1).values
        if bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any().item()):
            raise RuntimeError(f"{args.fixture} contains duplicate per-token experts")
        slot_weights = worker.torch.arange(
            1, worker.TOPK + 1, device=device, dtype=worker.torch.float32
        )
        weights_topk = (
            (slot_weights / slot_weights.sum()).expand(args.m, worker.TOPK).contiguous()
        )
        occupancy = worker.torch.bincount(ids.flatten().long(), minlength=worker.E)
        if int(occupancy[0].item()) != count:
            raise RuntimeError("directed expert row count drift")
        manifest = {
            "fixture_kind": f"directed_{args.fixture}",
            "seed": args.seed,
            "m": args.m,
            "shape": {
                "experts": worker.E,
                "hidden": worker.H,
                "intermediate_tp": worker.I,
                "topk": worker.TOPK,
            },
            "target_expert": 0,
            "target_valid_rows": count,
            "x_sha256": fixture_module.tensor_sha256(base.x),
            "topk_ids_sha256": fixture_module.tensor_sha256(ids),
            "topk_weights_sha256": fixture_module.tensor_sha256(weights_topk),
            "occupancy_sha256": fixture_module.tensor_sha256(occupancy),
            "duplicate_expert_ids": 0,
        }
        fixture = fixture_module.RoutedFixture(
            args.m, base.x, ids, weights_topk, manifest
        )
    else:
        fixture = worker.make_directed_fixture(fixture_module, base, args.fixture)
    weights = fixture_module.make_canonical_weights(device=device, seed=args.seed)
    return fixture_module, fixture, weights


def tensor_sha256(tensor) -> str:
    value = tensor.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(worker.torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def run_case(args: argparse.Namespace, m: int, fixture_kind: str) -> dict[str, Any]:
    args.m = m
    args.fixture = fixture_kind
    fixture_module, fixture, weights = make_case(args)
    reference = fixture_module.reference_moe_nvfp4(fixture, weights)
    arm = worker.build_arm(args, fixture, weights)
    arm.eager()
    arm.capture()
    replays = []
    for replay_id in range(args.replays):
        output, elapsed_ms = arm.replay(sentinel=True)
        output_cpu = output.detach().cpu().clone()
        diagnostics = fixture_module.output_diagnostics(output, reference)
        finite = bool(worker.torch.isfinite(output).all().item())
        nan_remaining = int(worker.torch.isnan(output).sum().item())
        inf_count = int(worker.torch.isinf(output).sum().item())
        _, workspace = worker._workspace_snapshot(
            arm.wrapper, fixture, num_cta_warps=EXPECTED_BLOCK[0] // 32
        )
        zero_rows = int((output_cpu.float().abs().sum(dim=1) == 0).sum().item())
        gate_pass = bool(
            diagnostics["formal_pass"]
            and finite
            and nan_remaining == 0
            and inf_count == 0
            and zero_rows == 0
            and workspace["verification"]["gate_pass"]
        )
        replays.append(
            {
                "replay": replay_id,
                "event_elapsed_us": elapsed_ms * 1000.0,
                "formal_metrics": diagnostics,
                "finite": finite,
                "nan_remaining": nan_remaining,
                "inf_count": inf_count,
                "zero_rows": zero_rows,
                "output_row0_first8": output_cpu[0, :8].float().tolist(),
                "output_sha256": tensor_sha256(output_cpu),
                "workspace_gate_pass": workspace["verification"]["gate_pass"],
                "workspace_checks": workspace["verification"].get("checks", {}),
                "gate_pass": gate_pass,
            }
        )
    case = {
        "m": m,
        "fixture": fixture.manifest,
        "reference_sha256": tensor_sha256(reference),
        "replays": replays,
        "all_replays_pass": all(row["gate_pass"] for row in replays),
    }
    del arm, reference, weights, fixture
    gc.collect()
    worker.torch.cuda.empty_cache()
    return case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--expected-overlay-sha256", required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--replays", type=int, default=2)
    parser.add_argument("--device-index", type=int, default=0, choices=[0])
    parser.add_argument("--seed", type=int, default=2026, choices=[2026])
    args = parser.parse_args()
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.overlay = args.overlay.resolve()
    args.jit_root = args.jit_root.resolve()
    args.output = args.output.resolve()
    args.arm = INTERNAL_ARM
    args.results = args.output.parent
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite immutable result: {args.output}")
    worker.require_empty_directory(args.jit_root)

    source = worker.validate_source(args.flashinfer_root, args.overlay, args.arm)
    if source["overlay_sha256"] != args.expected_overlay_sha256:
        raise RuntimeError("overlay hash drift")
    if str(args.flashinfer_root) not in sys.path:
        sys.path.insert(0, str(args.flashinfer_root))
    worker.install_overlay(args.overlay)
    imports = worker.configure_source_checkout(args.flashinfer_root)
    if Path(imports["target_module"]) != args.overlay:
        raise RuntimeError("target module did not resolve to candidate overlay")
    runtime = worker.runtime_identity(args, source)
    runtime["imports"] = imports

    payload: dict[str, Any] = {
        "schema": "exp013.correctness.v1",
        "status": "running",
        "runtime": runtime,
        "expected_launch": {
            "kernel": "MoEDynamicKernel",
            "grid": list(worker.EXPECTED_GRID),
            "block": list(EXPECTED_BLOCK),
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker.write_json(args.output, payload)
    cases = (
        (256, "canonical"),
        (256, "rows_63"),
        (256, "rows_64"),
        (256, "rows_65"),
        (256, "sparse_empty"),
        (8192, "canonical"),
    )
    try:
        for m, fixture_kind in cases:
            case = run_case(args, m, fixture_kind)
            payload["cases"].append(case)
            worker.write_json(args.output, payload)
            if not case["all_replays_pass"]:
                payload["status"] = "rejected_correctness"
                payload["all_cases_pass"] = False
                worker.write_json(args.output, payload)
                return 1
        artifacts = worker.artifact_manifest(args.jit_root)
        cubins = sorted(
            item["sha256"] for item in artifacts if item["path"].endswith(".cubin")
        )
        if not cubins:
            raise RuntimeError("candidate JIT produced no cubin")
        payload["compile_identity"] = worker._compile_identity()
        payload["jit_artifacts"] = artifacts
        payload["cubin_sha256"] = cubins
        payload["jit_artifact_set_sha256"] = worker.canonical_sha256(artifacts)
        payload["all_cases_pass"] = True
        payload["status"] = "complete"
        worker.write_json(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 0
    except Exception as error:
        payload["status"] = "failed"
        payload["error"] = f"{type(error).__name__}: {error}"
        worker.write_json(args.output, payload)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
