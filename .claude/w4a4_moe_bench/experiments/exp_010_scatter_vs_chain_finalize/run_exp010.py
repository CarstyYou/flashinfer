#!/usr/bin/env python3
"""Build and run the exp_010 standalone Scatter and Finalize components."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
EXP006 = ROOT.parent / "exp_006_fc2_completion_anchored_breakdown"
if str(EXP006) not in sys.path:
    sys.path.insert(0, str(EXP006))

M = 8192
H = 2048
TOPK = 8
SLICES = 4
OUTPUT_TILES = 16
GRID = 110
TASK_TAIL = 2536


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    return {
        "count": len(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "mean": statistics.mean(samples),
        "pstdev": statistics.pstdev(samples),
    }


def load_extension(build_dir: Path):
    from torch.utils.cpp_extension import load

    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
    return load(
        name="exp010_scatter_finalize_ext",
        sources=[str(ROOT / "scatter_finalize_ext.cu")],
        build_directory=str(build_dir),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=True,
    )


def build_replay_fixture(timing: dict[str, Any]) -> dict[str, Any]:
    import torch

    task_tail = int(timing["task_tail"])
    if task_tail != TASK_TAIL:
        raise RuntimeError(f"task_tail drift: {task_tail} != {TASK_TAIL}")
    task_cta = timing["task_cta_z"][:task_tail].to(torch.int64)
    if bool(((task_cta < 0) | (task_cta >= GRID)).any()):
        raise RuntimeError("invalid task-to-CTA mapping")

    queues: list[list[int]] = [[] for _ in range(GRID)]
    for task, cta in enumerate(task_cta.tolist()):
        queues[int(cta)].append(task)
    offsets = [0]
    flat_tasks: list[int] = []
    for queue in queues:
        flat_tasks.extend(queue)
        offsets.append(len(flat_tasks))
    if sorted(flat_tasks) != list(range(task_tail)):
        raise RuntimeError("CTA queues do not cover each task exactly once")

    task_expert = timing["task_expert"][:task_tail].to(torch.int64)
    task_m_tile = timing["task_m_tile"][:task_tail].to(torch.int64)
    task_slice = timing["task_slice_begin"][:task_tail].to(torch.int64)
    task_valid = timing["task_valid_rows"][:task_tail].to(torch.int64)
    if not bool((timing["task_slice_count"][:task_tail] == 1).all()):
        raise RuntimeError("task_slice_count drift")

    max_rows = int(timing["token_map"].numel())
    token_map = timing["token_map"].to(torch.int64)
    token_weights = timing["token_weights"].to(torch.float32)
    topk_ids = timing["topk_ids"].to(torch.int64)
    topk_weights = timing["topk_weights"].to(torch.float32)
    if tuple(topk_ids.shape) != (M, TOPK):
        raise RuntimeError(f"topk_ids shape drift: {tuple(topk_ids.shape)}")

    tile_expert: dict[int, int] = {}
    tile_valid: dict[int, int] = {}
    descriptor_multiset: list[tuple[int, int, int, int]] = []
    for expert, tile, slice_index, valid in zip(
        task_expert.tolist(),
        task_m_tile.tolist(),
        task_slice.tolist(),
        task_valid.tolist(),
        strict=True,
    ):
        descriptor_multiset.append((expert, tile, slice_index, valid))
        previous = tile_expert.setdefault(tile, expert)
        if previous != expert:
            raise RuntimeError(f"physical tile {tile} maps to multiple experts")
        previous_valid = tile_valid.setdefault(tile, valid)
        if previous_valid != valid:
            raise RuntimeError(f"physical tile {tile} has inconsistent valid rows")

    route_slot = torch.full((max_rows,), -1, dtype=torch.int32)
    route_rows = torch.full((M, TOPK), -1, dtype=torch.int32)
    active_rows = 0
    for tile, expert in tile_expert.items():
        for local_row in range(tile_valid[tile]):
            physical_row = tile * 128 + local_row
            token = int(token_map[physical_row])
            matches = torch.nonzero(topk_ids[token] == expert).flatten()
            if matches.numel() != 1:
                raise RuntimeError(
                    f"token={token} expert={expert} has {matches.numel()} route matches"
                )
            slot = int(matches[0])
            observed_weight = float(token_weights[physical_row])
            expected_weight = float(topk_weights[token, slot])
            if abs(observed_weight - expected_weight) > 1e-7:
                raise RuntimeError("physical-row route weight mismatch")
            if int(route_rows[token, slot]) != -1:
                raise RuntimeError("duplicate token/route physical-row assignment")
            route_slot[physical_row] = slot
            route_rows[token, slot] = physical_row
            active_rows += 1
    if active_rows != M * TOPK or bool((route_rows < 0).any()):
        raise RuntimeError(
            f"route mapping does not close: active={active_rows}, expected={M * TOPK}"
        )

    ticks = timing["task_ticks"][:task_tail].to(torch.int64)
    cadence = torch.zeros((task_tail, OUTPUT_TILES), dtype=torch.int64)
    for tile in range(OUTPUT_TILES):
        base = 9 + tile * 20
        max_a = ticks[:, base : base + 4].max(dim=1).values
        min_d = ticks[:, base + 8 : base + 12].min(dim=1).values
        cadence[:, tile] = torch.clamp(min_d - max_a, min=0)

    valid_histogram: dict[str, int] = {}
    for value in task_valid.tolist():
        valid_histogram[str(value)] = valid_histogram.get(str(value), 0) + 1
    cta_work = [len(queue) for queue in queues]
    collision_histogram: dict[str, int] = {}
    for token in token_map[route_rows.flatten().to(torch.int64)].tolist():
        collision_histogram[str(token)] = collision_histogram.get(str(token), 0) + 1
    if set(collision_histogram.values()) != {TOPK}:
        raise RuntimeError("each token must have exactly top-k physical rows")

    return {
        "task_tail": task_tail,
        "max_rows": max_rows,
        "token_map": token_map.to(torch.int32),
        "token_weights": token_weights,
        "topk_weights": topk_weights,
        "route_slot": route_slot,
        "route_rows": route_rows,
        "cta_offsets": torch.tensor(offsets, dtype=torch.int32),
        "cta_tasks": torch.tensor(flat_tasks, dtype=torch.int32),
        "task_m_tile": task_m_tile.to(torch.int32),
        "task_slice": task_slice.to(torch.int32),
        "task_valid_rows": task_valid.to(torch.int32),
        "cadence_ns": cadence,
        "ledger": {
            "tasks": task_tail,
            "active_physical_rows": active_rows,
            "route_contributions": active_rows * SLICES,
            "redg_bf16x8": active_rows * SLICES * (H // 8),
            "reduction_payload_bytes": active_rows * SLICES * H * 2,
            "descriptor_multiset_sha256": hashlib.sha256(
                json.dumps(sorted(descriptor_multiset), separators=(",", ":")).encode()
            ).hexdigest(),
            "valid_rows_histogram": valid_histogram,
            "cta_task_count": distribution(cta_work),
            "token_collision_count": distribution(collision_histogram.values()),
        },
    }


def phase_summary(timestamps: Any, *, launch_grid: int) -> dict[str, Any]:
    import torch

    value = timestamps.to(torch.int64)
    if bool((value < 0).any()):
        raise RuntimeError("timestamp buffer contains unwritten entries")
    d = value[..., 0]
    e = value[..., 1]
    f = value[..., 2]
    body = e.max(dim=-1).values - d.min(dim=-1).values
    post = f.max(dim=-1).values - e.max(dim=-1).values
    if bool((body < 0).any()) or bool((post < 0).any()):
        raise RuntimeError("non-monotonic D/E/F timestamp")
    body_total = int(body.sum())
    post_total = int(post.sum())
    return {
        "body_additive_ns": body_total,
        "post_sync_additive_ns": post_total,
        "d_to_f_additive_ns": body_total + post_total,
        # Keep the 110-SM denominator used by the production phase probe so
        # direct-grid arms cannot silently change the comparison statistic.
        "body_normalized_to_110_us": body_total / (GRID * 1000.0),
        "post_sync_normalized_to_110_us": post_total / (GRID * 1000.0),
        "d_to_f_normalized_to_110_us": (body_total + post_total) / (GRID * 1000.0),
        # This is per launched CTA, not an SM-equivalent latency.  It is kept
        # separately for service-time diagnostics only.
        "body_mean_cta_service_us": body_total / (launch_grid * 1000.0),
        "post_sync_mean_cta_service_us": post_total / (launch_grid * 1000.0),
        "d_to_f_mean_cta_service_us": (body_total + post_total)
        / (launch_grid * 1000.0),
        "launch_grid": launch_grid,
    }


def benchmark_call(
    call, *, warmup: int, replays: int, prepare=None, capture_payload=None
):
    import torch

    for _ in range(warmup):
        if prepare is not None:
            prepare()
        call()
    torch.cuda.synchronize()
    elapsed: list[float] = []
    payloads: list[Any] = []
    for _ in range(replays):
        if prepare is not None:
            prepare()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = call()
        end.record()
        end.synchronize()
        elapsed.append(float(start.elapsed_time(end)) * 1000.0)
        payloads.append(capture_payload() if capture_payload is not None else result)
    return elapsed, payloads


def error_summary(observed: Any, reference: Any) -> dict[str, Any]:
    import torch

    delta = observed.float() - reference.float()
    return {
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "finite": bool(torch.isfinite(observed).all()),
        "nonzero": int(torch.count_nonzero(observed)),
    }


def chain_reference(full_rows: Any, route_rows: Any, weights: Any):
    import torch

    output = torch.empty((M, H), dtype=torch.bfloat16, device=full_rows.device)
    chunk = 256
    for begin in range(0, M, chunk):
        end = min(M, begin + chunk)
        selected = full_rows[route_rows[begin:end].long()].float()
        result = (selected * weights[begin:end, :, None]).sum(dim=1)
        output[begin:end] = result.to(torch.bfloat16)
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if tuple(torch.cuda.get_device_capability()) != (12, 0):
        raise RuntimeError(
            f"exp_010 requires SM120, got {torch.cuda.get_device_capability()}"
        )
    timing = torch.load(args.production_timing, map_location="cpu", weights_only=True)
    fixture = build_replay_fixture(timing)
    device = torch.device("cuda", 0)
    ext = load_extension(args.build_dir)

    device_names = (
        "token_map",
        "token_weights",
        "topk_weights",
        "route_slot",
        "route_rows",
        "cta_offsets",
        "cta_tasks",
        "task_m_tile",
        "task_slice",
        "task_valid_rows",
        "cadence_ns",
    )
    gpu = {name: fixture[name].to(device) for name in device_names}
    partials = torch.empty(
        (fixture["max_rows"], SLICES, H), dtype=torch.bfloat16, device=device
    )
    full_rows = torch.empty(
        (fixture["max_rows"], H), dtype=torch.bfloat16, device=device
    )
    ext.fill_partials(partials)
    ext.fill_full_rows(full_rows)
    torch.cuda.synchronize()

    raw_dir = args.results / "raw" / "standalone"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        name: fixture[name]
        for name in (
            "token_map",
            "token_weights",
            "topk_weights",
            "route_slot",
            "route_rows",
            "cta_offsets",
            "cta_tasks",
            "task_m_tile",
            "task_slice",
            "task_valid_rows",
            "cadence_ns",
        )
    }
    torch.save(metadata, raw_dir / "replay_metadata.pt")

    arms: dict[str, Any] = {}
    canonical_output = None
    for name, mode, shard_factor, span_matched, direct_grid, use_cadence in (
        ("b_phase_no_cadence", 0, 1, False, False, False),
        ("b_phase_cadence_replay", 0, 1, False, False, True),
        ("b_component", 1, 1, False, False, False),
        ("d_shard4", 1, 4, False, False, False),
        ("d_shard32", 1, 32, False, False, False),
        ("d_span_matched", 1, 32, True, False, False),
        ("d_direct_grid", 1, 1, False, True, False),
    ):
        output_shards = 32 if span_matched else shard_factor
        output = torch.zeros((output_shards, M, H), dtype=torch.bfloat16, device=device)
        timestamps = torch.full(
            (TASK_TAIL, OUTPUT_TILES, 4, 3),
            -1,
            dtype=torch.int64,
            device=device,
        )
        cadence = (
            gpu["cadence_ns"] if use_cadence else torch.zeros_like(gpu["cadence_ns"])
        )

        def prepare(output=output, timestamps=timestamps):
            output.zero_()
            timestamps.fill_(-1)

        def call(output=output, timestamps=timestamps, cadence=cadence):
            ext.scatter_launch(
                partials,
                gpu["token_map"],
                gpu["token_weights"],
                gpu["route_slot"],
                gpu["cta_offsets"],
                gpu["cta_tasks"],
                gpu["task_m_tile"],
                gpu["task_slice"],
                gpu["task_valid_rows"],
                cadence,
                output,
                timestamps,
                TASK_TAIL,
                M,
                shard_factor,
                mode,
                span_matched,
                direct_grid,
            )
            return None

        elapsed, timestamp_replays = benchmark_call(
            call,
            warmup=args.warmup,
            replays=args.replays,
            prepare=prepare,
            # The device→host copy happens after the end event synchronizes,
            # so it is not included in event_elapsed_us.
            capture_payload=lambda timestamps=timestamps: timestamps.cpu().clone(),
        )
        phase_replays = [
            phase_summary(
                timestamp_cpu,
                launch_grid=(TASK_TAIL if direct_grid else GRID),
            )
            for timestamp_cpu in timestamp_replays
        ]
        for replay, timestamp_cpu in enumerate(timestamp_replays):
            torch.save(
                timestamp_cpu,
                raw_dir / f"{name}_timestamps_replay_{replay}.pt",
            )
        phase_statistics = {
            key: distribution(replay[key] for replay in phase_replays)
            for key in (
                "body_normalized_to_110_us",
                "post_sync_normalized_to_110_us",
                "d_to_f_normalized_to_110_us",
                "body_mean_cta_service_us",
                "post_sync_mean_cta_service_us",
                "d_to_f_mean_cta_service_us",
            )
        }
        effective = output.float().sum(dim=0).to(torch.bfloat16)
        if canonical_output is None and name == "b_phase_no_cadence":
            canonical_output = effective.clone()
        correctness = (
            error_summary(effective, canonical_output)
            if canonical_output is not None
            else {}
        )
        arms[name] = {
            "event_elapsed_us": distribution(elapsed),
            "phase_replays": phase_replays,
            "phase_replay_statistics": phase_statistics,
            "correctness_vs_b_phase": correctness,
            "output_sha256": tensor_sha256(effective),
            "output_shards": output_shards,
            "mode": mode,
            "shard_factor": shard_factor,
            "span_matched": span_matched,
            "direct_grid": direct_grid,
            "cadence_replay": use_cadence,
        }
        del call, prepare, effective, output, timestamps
        torch.cuda.empty_cache()

    chain_output = torch.empty((M, H), dtype=torch.bfloat16, device=device)

    def chain_call():
        ext.chain_finalize(
            full_rows, gpu["route_rows"], gpu["topk_weights"], chain_output
        )
        return None

    chain_elapsed, _ = benchmark_call(
        chain_call, warmup=args.warmup, replays=args.replays
    )
    reference = chain_reference(full_rows, gpu["route_rows"], gpu["topk_weights"])
    chain_correctness = error_summary(chain_output, reference)
    arms["c_source_shape_finalize"] = {
        "event_elapsed_us": distribution(chain_elapsed),
        "correctness": chain_correctness,
        "output_sha256": tensor_sha256(chain_output),
    }

    so_path = Path(ext.__file__).resolve()
    resource_path = args.results / "derived" / "extension_resource.txt"
    sass_path = args.results / "derived" / "extension_sass.txt"
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    resource = subprocess.run(
        ["cuobjdump", "--dump-resource-usage", str(so_path)],
        check=True,
        text=True,
        capture_output=True,
    )
    resource_path.write_text(resource.stdout + resource.stderr)
    sass = subprocess.run(
        ["cuobjdump", "--dump-sass", str(so_path)],
        check=True,
        text=True,
        capture_output=True,
    )
    sass_path.write_text(sass.stdout + sass.stderr)

    payload = {
        "schema": "exp010.standalone-capture.v2",
        "timestamp_unix": time.time(),
        "hardware": {
            "name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "uuid": os.environ.get("KDK_LEASE_GPU_UUID"),
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nvcc": subprocess.run(
                ["nvcc", "--version"], check=True, text=True, capture_output=True
            ).stdout,
            "extension_so": str(so_path),
            "extension_sha256": sha256(so_path),
        },
        "production_timing": {
            "path": str(args.production_timing.resolve()),
            "sha256": sha256(args.production_timing),
        },
        "fixture": fixture["ledger"],
        "arms": arms,
        "artifacts": {
            "metadata": str(raw_dir / "replay_metadata.pt"),
            "resource": str(resource_path),
            "sass": str(sass_path),
        },
    }
    write_json(args.results / "standalone_capture.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-timing", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--replays", type=int, default=5)
    args = parser.parse_args()
    args.results = args.results.resolve()
    args.build_dir = args.build_dir.resolve()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
