#!/usr/bin/env python3
"""Validate and benchmark the exp_020 Direct-32 and DSM-8 scatter demo."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


M = 8192
E = 256
H = 2048
TOPK = 8
TILE_M = 128
TILE_N = 128
SLICES = 4
EXPECTED_GROUPS = 637
TAIL_ROWS = (1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128)
ARMS = ("direct32", "dsm8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_text(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def load_kernel_module(path: Path):
    spec = importlib.util.spec_from_file_location("exp020_dsm_scatter_demo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_canonical_groups(fixture_path: Path) -> dict[str, Any]:
    import numpy as np
    import torch

    with np.load(fixture_path) as fixture:
        topk_ids = torch.from_numpy(np.array(fixture["topk_ids"], copy=True)).to(
            torch.int64
        )
        topk_weights = torch.from_numpy(
            np.array(fixture["topk_weights"], copy=True)
        ).to(torch.float32)
    if tuple(topk_ids.shape) != (M, TOPK):
        raise RuntimeError(f"topk_ids shape drift: {tuple(topk_ids.shape)}")
    if tuple(topk_weights.shape) != (M, TOPK):
        raise RuntimeError(f"topk_weights shape drift: {tuple(topk_weights.shape)}")
    if bool(((topk_ids < 0) | (topk_ids >= E)).any()):
        raise RuntimeError("topk expert id out of range")
    if not bool(torch.isfinite(topk_weights).all()):
        raise RuntimeError("non-finite route weight")
    if not bool(
        torch.allclose(topk_weights.sum(dim=1), torch.ones(M), atol=1e-6, rtol=1e-6)
    ):
        raise RuntimeError("top-k route weights do not sum to one")

    token_groups: list[Any] = []
    weight_groups: list[Any] = []
    valid_rows: list[int] = []
    experts: list[int] = []
    expert_tiles: list[int] = []
    occupancy: list[int] = []
    for expert in range(E):
        routes = torch.nonzero(topk_ids == expert, as_tuple=False)
        occupancy.append(int(routes.shape[0]))
        for tile, start in enumerate(range(0, int(routes.shape[0]), TILE_M)):
            chunk = routes[start : start + TILE_M]
            valid = int(chunk.shape[0])
            tokens = torch.zeros(TILE_M, dtype=torch.int32)
            weights = torch.zeros(TILE_M, dtype=torch.float32)
            if valid:
                token_index = chunk[:, 0]
                slot_index = chunk[:, 1]
                tokens[:valid] = token_index.to(torch.int32)
                weights[:valid] = topk_weights[token_index, slot_index]
            token_groups.append(tokens)
            weight_groups.append(weights)
            valid_rows.append(valid)
            experts.append(expert)
            expert_tiles.append(tile)

    token_map = torch.stack(token_groups)
    route_weights = torch.stack(weight_groups)
    valid = torch.tensor(valid_rows, dtype=torch.int32)
    group_expert = torch.tensor(experts, dtype=torch.int32)
    group_m_tile = torch.tensor(expert_tiles, dtype=torch.int32)
    if token_map.shape[0] != EXPECTED_GROUPS:
        raise RuntimeError(
            f"logical group drift: {token_map.shape[0]} != {EXPECTED_GROUPS}"
        )
    active_mask = torch.arange(TILE_M).unsqueeze(0) < valid.to(torch.int64).unsqueeze(1)
    active_tokens = token_map[active_mask].to(torch.int64)
    active_weights = route_weights[active_mask]
    if int(active_tokens.numel()) != M * TOPK:
        raise RuntimeError("routed-row ledger does not close")
    collisions = torch.bincount(active_tokens, minlength=M)
    if not bool((collisions == TOPK).all()):
        raise RuntimeError("every token must have exactly top-k routed rows")
    routed_weight_sum = torch.zeros(M, dtype=torch.float32)
    routed_weight_sum.index_add_(0, active_tokens, active_weights)
    if not bool(torch.allclose(routed_weight_sum, torch.ones(M), atol=1e-6, rtol=1e-6)):
        raise RuntimeError("grouped route weights do not preserve token sums")

    identity = b"".join(
        tensor.contiguous().numpy().tobytes()
        for tensor in (token_map, route_weights, valid, group_expert, group_m_tile)
    )
    return {
        "name": "canonical_m8192",
        "num_tokens": M,
        "token_map": token_map,
        "route_weights": route_weights,
        "valid_rows": valid,
        "group_expert": group_expert,
        "group_m_tile": group_m_tile,
        "fixture_sha256": sha256_file(fixture_path),
        "group_identity_sha256": sha256_bytes(identity),
        "ledger": {
            "groups": int(token_map.shape[0]),
            "slice_tasks": int(token_map.shape[0]) * SLICES,
            "routed_rows": int(active_tokens.numel()),
            "occupancy_min": min(occupancy),
            "occupancy_max": max(occupancy),
            "direct_redg_bf16x8": M * TOPK * SLICES * (H // 8),
            "dsm_redg_bf16x8": M * TOPK * (H // 8),
            "direct_reduction_bytes": M * TOPK * SLICES * H * 2,
            "dsm_reduction_bytes": M * TOPK * H * 2,
            "dsm_partial_read_bytes": M * TOPK * SLICES * H * 2,
            "dsm_fp32_adds": M * TOPK * (SLICES - 1) * H,
        },
    }


def build_tail_groups() -> dict[str, Any]:
    import torch

    groups = len(TAIL_ROWS)
    token_map = torch.zeros((groups, TILE_M), dtype=torch.int32)
    weights = torch.zeros((groups, TILE_M), dtype=torch.float32)
    valid = torch.tensor(TAIL_ROWS, dtype=torch.int32)
    token = 0
    for group, rows in enumerate(TAIL_ROWS):
        token_map[group, :rows] = torch.arange(token, token + rows, dtype=torch.int32)
        weights[group, :rows] = 1.0
        token += rows
    return {
        "name": "tail_ownership",
        "num_tokens": token,
        "token_map": token_map,
        "route_weights": weights,
        "valid_rows": valid,
        "group_expert": torch.arange(groups, dtype=torch.int32),
        "group_m_tile": torch.zeros(groups, dtype=torch.int32),
        "fixture_sha256": None,
        "group_identity_sha256": sha256_bytes(
            b"".join(
                tensor.contiguous().numpy().tobytes()
                for tensor in (token_map, weights, valid)
            )
        ),
        "ledger": {"groups": groups, "routed_rows": token},
    }


def semantic_reference(case: dict[str, Any], device: str = "cuda"):
    import torch

    token_map = case["token_map"].to(device=device, dtype=torch.int64)
    weights = case["route_weights"].to(device=device, dtype=torch.float32)
    valid = case["valid_rows"].to(device=device, dtype=torch.int64)
    groups = int(token_map.shape[0])
    row = torch.arange(TILE_M, device=device, dtype=torch.int64).unsqueeze(0)
    active_mask = row < valid.unsqueeze(1)
    active_tokens = token_map[active_mask]
    active_weights = weights[active_mask]
    reference = torch.zeros((case["num_tokens"], H), device=device, dtype=torch.float32)
    for output_tile in range(H // TILE_N):
        local_columns = torch.arange(TILE_N, device=device, dtype=torch.int64)
        local_rows = row.expand(groups, -1)[active_mask]
        merged = torch.zeros(
            (active_tokens.numel(), TILE_N), device=device, dtype=torch.float32
        )
        for slice_index in range(SLICES):
            code = (
                local_rows.unsqueeze(1) * 3
                + local_columns.unsqueeze(0) * 5
                + output_tile * 7
            ) & 15
            partial = (
                (slice_index + 1) * 0.03125 + code.to(torch.float32) * 0.0009765625
            ).to(torch.bfloat16)
            merged.add_(partial.to(torch.float32))
        merged.mul_(active_weights.unsqueeze(1))
        reference[:, output_tile * TILE_N : (output_tile + 1) * TILE_N].index_add_(
            0, active_tokens, merged
        )
    return reference


def make_launcher(module: Any, case: dict[str, Any]):
    import torch
    from cuda.bindings import driver as cuda
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack

    token_map = case["token_map"].cuda().contiguous()
    route_weights = case["route_weights"].cuda().contiguous()
    valid_rows = case["valid_rows"].cuda().contiguous()
    output = torch.empty((case["num_tokens"], H), device="cuda", dtype=torch.bfloat16)
    token_arg = from_dlpack(token_map, assumed_align=16)
    weight_arg = from_dlpack(route_weights, assumed_align=16)
    valid_arg = from_dlpack(valid_rows, assumed_align=16)
    output_arg = from_dlpack(output, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = {
        arm: cute.compile(
            module.launch_demo,
            token_arg,
            weight_arg,
            valid_arg,
            output_arg,
            arm == "dsm8",
            stream,
        )
        for arm in ARMS
    }

    def launch(arm: str) -> None:
        if arm not in ARMS:
            raise ValueError(arm)
        compiled[arm](token_arg, weight_arg, valid_arg, output_arg, stream)

    return launch, output


def tensor_hash(tensor: Any) -> str:
    import torch

    value = tensor.detach().contiguous().cpu().view(-1).view(torch.uint8)
    return sha256_bytes(value.numpy().tobytes())


def error_summary(output: Any, reference: Any) -> dict[str, Any]:
    import torch

    observed = output.to(torch.float32)
    error = (observed - reference).abs()
    tolerance = 0.005 + 0.05 * reference.abs()
    # P99 is diagnostic only.  Sampling avoids sorting all 16.8M canonical
    # elements; max/mean and the tolerance gate still cover the full output.
    p99_sample = error.flatten()[::1024]
    return {
        "finite": bool(torch.isfinite(observed).all()),
        "nonzero": int(torch.count_nonzero(observed).item()),
        "max_abs": float(error.max().item()),
        "mean_abs": float(error.mean().item()),
        "p99_abs": float(torch.quantile(p99_sample, 0.99).item()),
        "within_atol_0p005_rtol_0p05": bool((error <= tolerance).all()),
        "output_sha256": tensor_hash(output),
    }


def validate_case(
    module: Any, case: dict[str, Any], repeats: int = 3
) -> dict[str, Any]:
    import torch

    launch, output = make_launcher(module, case)
    reference = semantic_reference(case)
    arms: dict[str, Any] = {}
    for arm in ARMS:
        runs = []
        for _ in range(repeats):
            output.zero_()
            launch(arm)
            torch.cuda.synchronize()
            runs.append(error_summary(output, reference))
        arms[arm] = {
            "runs": runs,
            "pass": all(
                run["finite"] and run["within_atol_0p005_rtol_0p05"] for run in runs
            ),
            "stable_hash": len({run["output_sha256"] for run in runs}) == 1,
        }
    return {
        "case": case["name"],
        "num_tokens": case["num_tokens"],
        "groups": int(case["token_map"].shape[0]),
        "group_identity_sha256": case["group_identity_sha256"],
        "arms": arms,
        "pass": all(value["pass"] for value in arms.values()),
    }


def distribution(samples: list[float]) -> dict[str, float | int]:
    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples)
    return {
        "count": len(samples),
        "median_us": statistics.median(samples),
        "mean_us": mean,
        "min_us": min(samples),
        "max_us": max(samples),
        "stdev_us": stdev,
        "cv": stdev / mean if mean else 0.0,
    }


def gpu_snapshot() -> dict[str, Any]:
    fields = (
        "uuid,clocks.applications.graphics,clocks.current.graphics,"
        "temperature.gpu,power.draw"
    )
    row = run_text(
        [
            "nvidia-smi",
            "--query-gpu=" + fields,
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()[0]
    values = [part.strip() for part in row.split(",")]
    snapshot = dict(zip(fields.split(","), values, strict=True))
    processes = run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    snapshot["compute_processes"] = [
        line.strip() for line in processes.splitlines() if line.strip()
    ]
    return snapshot


def require_group_gpu_contract(snapshot: dict[str, Any]) -> None:
    expected_clock = os.environ.get("EXP020_EXPECTED_CLOCK_MHZ")
    if expected_clock and snapshot["clocks.applications.graphics"] != expected_clock:
        raise RuntimeError(
            "application clock drift: "
            f"{snapshot['clocks.applications.graphics']} != {expected_clock}"
        )
    # Docker PID namespaces prevent a stable PID equality check, but an extra
    # process on the selected GPU is still unambiguous contamination.
    if len(snapshot["compute_processes"]) != 1:
        raise RuntimeError(
            f"foreign compute process detected: {snapshot['compute_processes']}"
        )


def benchmark(module: Any, case: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    launch, output = make_launcher(module, case)
    eviction = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)

    for arm in ARMS:
        for _ in range(20):
            output.zero_()
            launch(arm)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def sample(arm: str) -> float:
        output.zero_()
        eviction.zero_()
        start.record()
        launch(arm)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end) * 1000.0)

    groups: list[dict[str, Any]] = []
    early_stop_reason: str | None = None
    orders = (("direct32", "dsm8"), ("dsm8", "direct32"))
    for group_index in range(5):
        before = gpu_snapshot()
        require_group_gpu_contract(before)
        order = orders[group_index & 1]
        arm_samples: dict[str, list[float]] = {}
        for arm in order:
            arm_samples[arm] = [sample(arm) for _ in range(100)]
        after = gpu_snapshot()
        require_group_gpu_contract(after)
        summaries = {arm: distribution(arm_samples[arm]) for arm in ARMS}
        baseline = float(summaries["direct32"]["median_us"])
        candidate = float(summaries["dsm8"]["median_us"])
        groups.append(
            {
                "group": group_index,
                "order": list(order),
                "gpu_before": before,
                "gpu_after": after,
                "samples_us": arm_samples,
                "summary": summaries,
                "paired_delta_us": baseline - candidate,
                "speedup_pct": (baseline / candidate - 1.0) * 100.0,
            }
        )
        print(
            json.dumps(
                {
                    "benchmark_group": group_index,
                    "direct32_median_us": baseline,
                    "dsm8_median_us": candidate,
                    "speedup_pct": groups[-1]["speedup_pct"],
                }
            ),
            flush=True,
        )
        observed_groups_stable = all(
            float(group["summary"][arm]["cv"]) <= 0.015
            for group in groups
            for arm in ARMS
        )
        both_orders_nonpositive = len(groups) >= 2 and all(
            group["speedup_pct"] <= 0.0 for group in groups
        )
        if both_orders_nonpositive and observed_groups_stable:
            # Acceptance requires every group to be positive.  Once a stable
            # 100-sample group in both AB and BA order is negative, running the
            # remaining groups cannot change the decision and only burns shared
            # GPU time.
            early_stop_reason = "two_stable_orders_nonpositive_speedup"
            print(
                json.dumps(
                    {
                        "benchmark_early_stop": early_stop_reason,
                        "failed_groups": [group["group"] for group in groups],
                    }
                ),
                flush=True,
            )
            break

    speedups = np.array([group["speedup_pct"] for group in groups], dtype=np.float64)
    rng = np.random.default_rng(20260721)
    resampled = speedups[
        rng.integers(0, speedups.size, size=(20_000, speedups.size))
    ].mean(axis=1)
    lower = float(np.quantile(resampled, 0.05))
    median_speedup = float(np.median(speedups))
    all_positive = bool((speedups > 0).all())
    cv_pass = all(
        float(group["summary"][arm]["cv"]) <= 0.015 for group in groups for arm in ARMS
    )
    all_groups_completed = len(groups) == 5
    preliminary_pass = (
        all_groups_completed
        and all_positive
        and median_speedup >= 2.0
        and lower > 0.0
        and cv_pass
    )
    return {
        "protocol": {
            "warmup_per_arm": 20,
            "planned_groups": 5,
            "launches_per_arm_per_group": 100,
            "order": "AB/BA/AB/BA/AB",
            "early_reject": (
                "stop after stable nonpositive speedup in both AB and BA order; "
                "the all-groups-positive acceptance gate can no longer pass"
            ),
            "output_clear_in_event": False,
            "l2_eviction_in_event": False,
            "l2_eviction_bytes": int(eviction.numel()),
            "bootstrap_seed": 20260721,
            "bootstrap_resamples": 20_000,
            "one_sided_lower_quantile": 0.05,
        },
        "groups": groups,
        "groups_completed": len(groups),
        "all_groups_completed": all_groups_completed,
        "early_stop_reason": early_stop_reason,
        "group_speedup_pct": speedups.tolist(),
        "median_speedup_pct": median_speedup,
        "mean_speedup_pct": float(speedups.mean()),
        "bootstrap_95pct_lower_speedup_pct": lower,
        "all_groups_positive": all_positive,
        "all_group_arm_cv_le_0p015": cv_pass,
        "preliminary_performance_pass": preliminary_pass,
    }


def artifact_inventory(jit_root: Path) -> list[dict[str, Any]]:
    artifacts = []
    for suffix in (".cubin", ".sass", ".ptx", ".mlir"):
        for path in sorted(jit_root.rglob("*" + suffix)):
            artifacts.append(
                {
                    "path": str(path.relative_to(jit_root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return artifacts


def reuse_validation_evidence(
    results: Path,
    kernel_path: Path,
    fixture_path: Path,
    canonical: dict[str, Any],
    observed_uuid: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse correctness only when every relevant identity is unchanged."""
    payload_path = results / "raw" / "demo.json"
    manifest_path = results / "manifest.json"
    if not payload_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            "benchmark mode requires a prior exact-hash validation run; "
            "run validate first"
        )
    payload = json.loads(payload_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    environment = manifest.get("environment", {})
    fixture = manifest.get("fixture", {})
    checks = {
        "schema": payload.get("schema") == "exp020.demo.v1",
        "validation_pass": payload.get("validation_pass") is True,
        "kernel_source_sha256": environment.get("kernel_source_sha256")
        == sha256_file(kernel_path),
        "fixture_sha256": fixture.get("sha256") == sha256_file(fixture_path),
        "group_identity_sha256": fixture.get("group_identity_sha256")
        == canonical["group_identity_sha256"],
        "gpu_uuid": environment.get("gpu", {}).get("uuid") == observed_uuid,
        "image_id": environment.get("image_id") == os.environ.get("EXP020_IMAGE_ID"),
        "image_digest": environment.get("image_digest")
        == os.environ.get("EXP020_IMAGE_DIGEST"),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"prior validation identity mismatch: {failed}")
    return payload["validation"], {
        "reused": True,
        "checks": checks,
        "payload_sha256": sha256_file(payload_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def environment_manifest(
    kernel_path: Path, fixture_path: Path, jit_root: Path
) -> dict[str, Any]:
    import torch

    return {
        "kernel_source_sha256": sha256_file(kernel_path),
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "remote_launcher_sha256": sha256_file(kernel_path.parent / "run_remote.sh"),
        "fixture_sha256": sha256_file(fixture_path),
        "flashinfer_commit": run_text(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"]
        ),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu": gpu_snapshot(),
        "expected_gpu_uuid": os.environ.get("EXP020_EXPECTED_GPU_UUID"),
        "expected_application_clock_mhz": os.environ.get("EXP020_EXPECTED_CLOCK_MHZ"),
        "nvcc": run_text(["nvcc", "--version"]),
        "image_id": os.environ.get("EXP020_IMAGE_ID"),
        "image_digest": os.environ.get("EXP020_IMAGE_DIGEST"),
        "jit_root": str(jit_root),
        "artifacts": artifact_inventory(jit_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "benchmark", "all"))
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--jit-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    root = Path(__file__).resolve().parent
    kernel_path = root / "dsm_scatter_demo.py"
    module = load_kernel_module(kernel_path)
    capability = torch.cuda.get_device_capability(0)
    if capability != (12, 0):
        raise RuntimeError(f"exp_020 requires SM120, observed {capability}")
    observed_uuid = gpu_snapshot()["uuid"]
    expected_uuid = os.environ.get("EXP020_EXPECTED_GPU_UUID")
    if expected_uuid and observed_uuid != expected_uuid:
        raise RuntimeError(f"GPU UUID drift: {observed_uuid} != {expected_uuid}")

    canonical = build_canonical_groups(args.fixture)
    if args.mode == "benchmark":
        validation, validation_evidence = reuse_validation_evidence(
            args.results,
            kernel_path,
            args.fixture,
            canonical,
            observed_uuid,
        )
    else:
        tail = build_tail_groups()
        validation = {
            "canonical": validate_case(module, canonical),
            "tails": validate_case(module, tail),
        }
        validation_evidence = {"reused": False}
    validation_pass = all(value["pass"] for value in validation.values())
    payload: dict[str, Any] = {
        "schema": "exp020.demo.v1",
        "mode": args.mode,
        "validation": validation,
        "validation_evidence": validation_evidence,
        "validation_pass": validation_pass,
        "canonical_ledger": canonical["ledger"],
    }
    if not validation_pass:
        payload["verdict"] = "reject_correctness"
    elif args.mode in {"benchmark", "all"}:
        payload["benchmark"] = benchmark(module, canonical)
        payload["verdict"] = (
            "performance_pass_pending_binary_gates"
            if payload["benchmark"]["preliminary_performance_pass"]
            else "reject_performance"
        )
    else:
        payload["verdict"] = "validation_pass"

    args.results.mkdir(parents=True, exist_ok=True)
    write_json(args.results / "raw" / "demo.json", payload)
    manifest = {
        "schema": "exp020.manifest.v1",
        "experiment": "exp_020_dsm_8way_scatter_demo",
        "status": payload["verdict"],
        "case": {"m": M, "experts": E, "hidden": H, "topk": TOPK},
        "launch_contract": {
            "grid": [EXPECTED_GROUPS * SLICES, 1, 1],
            "block": [288, 1, 1],
            "cluster": [SLICES, 1, 1],
            "dynamic_smem_bytes_per_cta": 84_992,
            "output_tiles_per_cluster": H // TILE_N,
        },
        "environment": environment_manifest(kernel_path, args.fixture, args.jit_root),
        "fixture": {
            "path": str(args.fixture),
            "sha256": canonical["fixture_sha256"],
            "group_identity_sha256": canonical["group_identity_sha256"],
        },
        "ledger": canonical["ledger"],
    }
    write_json(args.results / "manifest.json", manifest)
    print(
        json.dumps({"verdict": payload["verdict"], "validation_pass": validation_pass})
    )


if __name__ == "__main__":
    main()
