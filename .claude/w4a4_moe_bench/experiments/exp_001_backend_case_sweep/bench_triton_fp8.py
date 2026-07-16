#!/usr/bin/env python3
"""Benchmark the pinned SGLang legacy Triton tensor-scaled FP8 MoE chain."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.nn import functional as F

from fixture import E, H, I, TOPK, load_fixture, validate_output

M_VALUES = (256, 512, 1024, 2048, 4096, 8192)
EXPECTED_IMAGE_DIGEST = (
    "sha256:00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2"
)
EXPECTED_IMAGE_ID = "sha256:663867442f321ded36228bafd889fd1db05cbef7a7c8ea6e072df33234dabbfd"
EXPECTED_SGLANG_VERSION = "0.5.15.post1"
EXPECTED_SGLANG_COMMIT = "0b3bb0cbe31873994c9f989fddfe2f87ca839fdd"
COMPARISON_GROUP_ID = "exp001_cutedsl_vs_sglang_triton_fp8_cross_runtime"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = EXPERIMENT_ROOT / "results" / "sglang_triton"
DEFAULT_FIXTURES = EXPERIMENT_ROOT / "results" / "fixtures"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def command_output(command: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"ERROR: {error}"


def query_gpu_uuid() -> str:
    output = command_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"])
    return output.splitlines()[0].strip()


def initialize_sglang() -> dict[str, Any]:
    import sglang
    import triton
    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    server_args = ServerArgs(model_path="dummy", moe_runner_backend="triton")
    set_global_server_args_for_scheduler(server_args)
    version = importlib.metadata.version("sglang")
    if version != EXPECTED_SGLANG_VERSION:
        raise RuntimeError(f"SGLang version drift: {version} != {EXPECTED_SGLANG_VERSION}")
    return {
        "sglang_version": version,
        "sglang_module": str(Path(sglang.__file__).resolve()),
        "triton_version": triton.__version__,
        "server_args": {
            "model_path": "dummy",
            "moe_runner_backend": str(server_args.moe_runner_backend),
            "enable_fused_moe_sum_all_reduce": bool(
                server_args.enable_fused_moe_sum_all_reduce
            ),
            "enable_deterministic_inference": bool(
                server_args.enable_deterministic_inference
            ),
        },
    }


def runtime_manifest(expected_gpu_uuid: str) -> dict[str, Any]:
    image_digest = os.environ.get("W4A4_IMAGE_DIGEST", "")
    if image_digest != EXPECTED_IMAGE_DIGEST:
        raise RuntimeError(
            f"container image drift: {image_digest} != {EXPECTED_IMAGE_DIGEST}"
        )
    image_id = os.environ.get("W4A4_IMAGE_ID", "")
    if image_id != EXPECTED_IMAGE_ID:
        raise RuntimeError(f"container image ID drift: {image_id} != {EXPECTED_IMAGE_ID}")
    commit = os.environ.get("W4A4_SGLANG_COMMIT", "")
    if commit != EXPECTED_SGLANG_COMMIT:
        raise RuntimeError(f"SGLang commit drift: {commit} != {EXPECTED_SGLANG_COMMIT}")
    lease_id = os.environ.get("KDK_LEASE_ID", "")
    rerun_id = os.environ.get("W4A4_RERUN_ID", "")
    if not lease_id or len(rerun_id) < 8:
        raise RuntimeError("KDK_LEASE_ID and a unique W4A4_RERUN_ID are required")
    actual_uuid = query_gpu_uuid()
    if actual_uuid != expected_gpu_uuid:
        raise RuntimeError(f"GPU UUID drift: {actual_uuid} != {expected_gpu_uuid}")
    foreign = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid",
            "--format=csv,noheader",
        ]
    )
    if foreign:
        raise RuntimeError(f"foreign GPU compute process present:\n{foreign}")
    sglang = initialize_sglang()
    return {
        "timestamp_unix": time.time(),
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_uuid": actual_uuid,
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "resource_lease_id": lease_id,
        "measurement_rerun_id": rerun_id,
        "image": "lmsysorg/sglang:latest",
        "image_digest": image_digest,
        "image_id": image_id,
        "sglang_commit": commit,
        "foreign_compute_process_query": foreign,
        **sglang,
    }


@dataclass(frozen=True)
class Fp8Weights:
    w1: torch.Tensor
    w2: torch.Tensor
    w1_scale: torch.Tensor
    w2_scale: torch.Tensor
    manifest: dict[str, Any]


def make_fp8_weights(*, device: torch.device, seed: int) -> Fp8Weights:
    """Create per-expert E4M3 weights/scales outside the measured closure."""
    from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    w1 = torch.empty((E, 2 * I, H), dtype=torch.float8_e4m3fn, device=device)
    w2 = torch.empty((E, H, I), dtype=torch.float8_e4m3fn, device=device)
    w1_scale = torch.empty(E, dtype=torch.float32, device=device)
    w2_scale = torch.empty(E, dtype=torch.float32, device=device)
    for expert in range(E):
        master1 = (
            torch.randn(
                (2 * I, H), generator=generator, device=device, dtype=torch.float32
            )
            / 10.0
        ).to(torch.bfloat16)
        master2 = (
            torch.randn(
                (H, I), generator=generator, device=device, dtype=torch.float32
            )
            / 10.0
        ).to(torch.bfloat16)
        quant1, scale1 = scaled_fp8_quant(master1, None)
        quant2, scale2 = scaled_fp8_quant(master2, None)
        w1[expert].copy_(quant1[..., :H])
        w2[expert].copy_(quant2[..., :I])
        w1_scale[expert] = scale1.reshape(-1)[0]
        w2_scale[expert] = scale2.reshape(-1)[0]
    manifest = {
        "seed": seed,
        "master_distribution": "torch.float32 normal(0,0.1) rounded to BF16",
        "quantizer": "sglang.srt.layers.quantization.fp8_kernel.scaled_fp8_quant",
        "storage_dtype": "float8_e4m3fn",
        "layout": {"w1": [E, 2 * I, H], "w2": [E, H, I]},
        "scale_contract": "FP32 [E], one tensor scale per expert",
        "w1_sha256": tensor_sha256(w1),
        "w2_sha256": tensor_sha256(w2),
        "w1_scale_sha256": tensor_sha256(w1_scale),
        "w2_scale_sha256": tensor_sha256(w2_scale),
    }
    return Fp8Weights(w1, w2, w1_scale, w2_scale, manifest)


def resolved_config(
    x: torch.Tensor, topk_ids: torch.Tensor, weights: Fp8Weights
) -> dict[str, Any]:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        get_moe_configs,
        try_get_optimal_moe_config,
    )

    tuned = get_moe_configs(E, I, "fp8_w8a8", per_channel_quant=False)
    config, (down_config, max_block_m) = try_get_optimal_moe_config(
        tuple(weights.w1.shape),
        tuple(weights.w2.shape),
        TOPK,
        "fp8_w8a8",
        x.shape[0],
        block_shape=None,
        per_channel_quant=False,
        return_down_config=True,
    )
    return {
        "up": config,
        "down": down_config,
        "down_max_block_m": max_block_m,
        "source": "tuned_file" if tuned else "default_heuristic",
        "tuned_grid": sorted(tuned) if tuned else [],
        "topk_ids_shape": list(topk_ids.shape),
    }


def build_launch(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: Fp8Weights,
) -> tuple[Callable[[], torch.Tensor], dict[str, Any]]:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        fused_experts_impl,
    )

    def launch() -> torch.Tensor:
        return fused_experts_impl(
            x,
            weights.w1,
            weights.w2,
            topk_weights,
            topk_ids,
            inplace=False,
            activation="silu",
            is_gated=True,
            use_fp8_w8a8=True,
            per_channel_quant=False,
            w1_scale=weights.w1_scale,
            w2_scale=weights.w2_scale,
            a1_scale=None,
            a2_scale=None,
            block_shape=None,
            no_combine=False,
            filter_expert=True,
        )

    module = inspect.getmodule(fused_experts_impl)
    source = Path(inspect.getsourcefile(fused_experts_impl) or "").resolve()
    return launch, {
        "backend": "sglang_legacy_triton_fp8_chain",
        "callable": f"{module.__name__}.fused_experts_impl",
        "callable_source": str(source),
        "callable_source_sha256": file_sha256(source),
        "config": resolved_config(x, topk_ids, weights),
        "quant_contract": {
            "weight": "E4M3 per-expert tensor scale",
            "input": "BF16 -> dynamic tensor-wise E4M3 inside FC1",
            "intermediate": "BF16 -> dynamic tensor-wise E4M3 inside FC2",
            "per_channel_quant": False,
            "block_shape": None,
        },
        "alternate_dispatch": "direct callable; no runner/DeepGEMM/CUTLASS dispatcher",
    }


@torch.inference_mode()
def fp8_oracle(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: Fp8Weights,
) -> torch.Tensor:
    """Independent expert matmuls using the actual E4M3 tensors and scales."""
    from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant

    input_q, input_scale = scaled_fp8_quant(x, None)
    input_dequant = input_q[..., :H].float() * input_scale.float()
    m = x.shape[0]
    flat_ids = topk_ids.flatten().long()
    activation = torch.empty((m * TOPK, I), device=x.device, dtype=torch.bfloat16)
    for expert in range(E):
        positions = torch.nonzero(flat_ids == expert, as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        tokens = torch.div(positions, TOPK, rounding_mode="floor")
        weight = weights.w1[expert].float() * weights.w1_scale[expert]
        fc1 = input_dequant[tokens] @ weight.transpose(0, 1)
        activation[positions] = (F.silu(fc1[:, :I]) * fc1[:, I:]).to(
            torch.bfloat16
        )
    activation_q, activation_scale = scaled_fp8_quant(activation, None)
    activation_dequant = activation_q[..., :I].float() * activation_scale.float()
    flat_output = torch.empty(
        (m * TOPK, H), device=x.device, dtype=torch.bfloat16
    )
    for expert in range(E):
        positions = torch.nonzero(flat_ids == expert, as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        weight = weights.w2[expert].float() * weights.w2_scale[expert]
        flat_output[positions] = (
            activation_dequant[positions] @ weight.transpose(0, 1)
        ).to(torch.bfloat16)
    return (
        flat_output.float().view(m, TOPK, H) * topk_weights.unsqueeze(-1)
    ).sum(dim=1)


def output_diagnostics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    actual_f = actual.float()
    reference_f = reference.float()
    error = (actual_f - reference_f).abs()
    tolerance = 0.01 + 0.1 * reference_f.abs()
    within = error <= tolerance
    relative_error = error / reference_f.abs().clamp_min(1e-8)
    percent = float(within.float().mean().item() * 100.0)
    return {
        "oracle": "actual E4M3 weights/scales plus two dynamic activation E4M3 round trips",
        "rtol": 0.1,
        "atol": 0.01,
        "formal_percent_threshold": 97.0,
        "formal_percent_within": percent,
        "formal_pass": percent >= 97.0,
        "cosine": float(
            F.cosine_similarity(actual_f.flatten(), reference_f.flatten(), dim=0).item()
        ),
        "relative_l2": float(
            (
                torch.linalg.vector_norm(actual_f - reference_f)
                / torch.linalg.vector_norm(reference_f).clamp_min(1e-12)
            ).item()
        ),
        "max_abs_error": float(error.max().item()),
        "max_relative_error": float(relative_error.max().item()),
        "output_sha256": tensor_sha256(actual),
        "reference_sha256": tensor_sha256(reference),
    }


def observed_cuda_kernels(launch: Callable[[], torch.Tensor]) -> list[str]:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        launch()
        torch.cuda.synchronize()
    names: list[str] = []
    for event in trace.events():
        device_type = str(getattr(event, "device_type", "")).lower()
        if "cuda" in device_type:
            names.append(event.name)
    if not names:
        raise RuntimeError("PyTorch profiler observed no CUDA kernels")
    lowered = [name.lower() for name in names]
    if sum("fused_moe_kernel" in name for name in lowered) < 2:
        raise RuntimeError(f"missing two SGLang Triton fused_moe kernels: {names}")
    forbidden = [name for name in names if "deepgemm" in name.lower() or "cutlass" in name.lower()]
    if forbidden:
        raise RuntimeError(f"alternate backend kernel observed: {forbidden}")
    return names


@dataclass
class CapturedCall:
    launch: Callable[[], torch.Tensor]
    output: torch.Tensor | None = None
    graph: torch.cuda.CUDAGraph | None = None
    start: torch.cuda.Event | None = None
    end: torch.cuda.Event | None = None

    def capture(self) -> None:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.launch()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        self.start = torch.cuda.Event(enable_timing=True, external=True)
        self.end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(self.graph, stream=stream):
            self.start.record()
            self.output = self.launch()
            self.end.record()
        torch.cuda.synchronize()

    def replay_ms(self) -> float:
        if self.graph is None or self.start is None or self.end is None:
            raise RuntimeError("call has not been captured")
        self.graph.replay()
        torch.cuda.synchronize()
        return float(self.start.elapsed_time(self.end))


def make_l2_flusher(device: torch.device, num_bytes: int) -> tuple[Callable[[], None], int]:
    buffer = torch.empty((num_bytes + 3) // 4, device=device, dtype=torch.int32)
    state = 0

    def flush() -> None:
        nonlocal state
        state = (state + 1) & 0x7FFFFFFF
        buffer.fill_(state)
        torch.cuda.synchronize()

    flush()
    return flush, buffer.numel() * buffer.element_size()


def artifact_manifest(weights: Fp8Weights) -> dict[str, Any]:
    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fused_moe
    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config as config
    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels as kernels

    paths = [
        Path(__file__).resolve(),
        Path(inspect.getsourcefile(fused_moe) or "").resolve(),
        Path(inspect.getsourcefile(config) or "").resolve(),
        Path(inspect.getsourcefile(kernels) or "").resolve(),
    ]
    jit_root_value = os.environ.get("W4A4_SGLANG_JIT_DIR", "")
    if not jit_root_value:
        raise RuntimeError("W4A4_SGLANG_JIT_DIR must name the dedicated Triton cache")
    jit_root = Path(jit_root_value).resolve()
    jit_files = [
        {
            "path": str(path.relative_to(jit_root)),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(jit_root.rglob("*"))
        if path.is_file()
    ]
    if not jit_files:
        raise RuntimeError("dedicated SGLang Triton cache contains no JIT artifacts")
    payload = {
        "source_files": [
            {"path": str(path), "sha256": file_sha256(path)} for path in paths
        ],
        "weights": weights.manifest,
        "jit_root": str(jit_root),
        "jit_artifacts": jit_files,
    }
    return {**payload, "fingerprint_sha256": canonical_sha256(payload)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--m-values", nargs="+", type=int, default=list(M_VALUES))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--l2-flush-bytes", type=int, default=192 << 20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.smoke and tuple(args.m_values) != M_VALUES:
        raise RuntimeError(f"canonical M list must be exactly {M_VALUES}")
    if (args.results / "evidence.identity.json").exists():
        raise RuntimeError("refusing to overwrite completed SGLang evidence")
    args.results.mkdir(parents=True, exist_ok=True)
    runtime = runtime_manifest(args.expected_gpu_uuid)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    weights = make_fp8_weights(device=device, seed=args.seed)
    flush_l2, flush_bytes = make_l2_flusher(device, args.l2_flush_bytes)
    protocol = {
        "m_values": list(args.m_values),
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "l2_flush_bytes": flush_bytes,
        "timing": "CUDA graph external events inside; synchronized L2 flush outside",
        "aggregation": "median of five repeat means",
    }
    protocol["fingerprint_sha256"] = canonical_sha256(protocol)
    raw: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    correctness: dict[str, Any] = {
        "runtime": runtime,
        "weights": weights.manifest,
        "protocol": protocol,
        "cases": {},
    }

    for m in args.m_values:
        print(f"exp001 SGLang Triton FP8 prepare m={m}", flush=True)
        x, topk_ids, topk_weights, fixture = load_fixture(args.fixture_dir, m, device)
        launch, contract = build_launch(x, topk_ids, topk_weights, weights)
        eager_output = launch()
        torch.cuda.synchronize()
        validate_output(eager_output, m)
        reference = fp8_oracle(x, topk_ids, topk_weights, weights)
        diagnostics = output_diagnostics(eager_output, reference)
        kernels = observed_cuda_kernels(launch)
        case = {
            "fixture": fixture,
            "contract": contract,
            "correctness": diagnostics,
            "observed_cuda_kernels": kernels,
            "dispatch_pass": True,
        }
        correctness["cases"][str(m)] = case
        write_json(args.results / "correctness.json", correctness)
        if not diagnostics["formal_pass"]:
            raise RuntimeError(f"FP8 correctness gate failed at m={m}: {diagnostics}")

        captured = CapturedCall(launch)
        captured.capture()
        for _ in range(args.warmup):
            flush_l2()
            captured.replay_ms()
        samples: list[float] = []
        for repeat in range(args.repeats):
            elapsed_ms = 0.0
            for _ in range(args.iters):
                flush_l2()
                elapsed_ms += captured.replay_ms()
            sample_us = elapsed_ms / args.iters * 1000.0
            samples.append(sample_us)
            raw.append(
                {
                    "m": m,
                    "repeat": repeat,
                    "arm": "sglang_triton_fp8",
                    "sample_us": sample_us,
                    "iters": args.iters,
                    "fixture_sha256": fixture["fixture_sha256"],
                    "occupancy_sha256": fixture["occupancy_sha256"],
                    "rerun_id": runtime["measurement_rerun_id"],
                    "comparison_group_id": COMPARISON_GROUP_ID,
                }
            )
            write_csv(args.results / "benchmark_raw.csv", raw)
        median_us = statistics.median(samples)
        spread = (max(samples) - min(samples)) / median_us * 100.0
        summary.append(
            {
                "m": m,
                "arm": "sglang_triton_fp8",
                "median_us": median_us,
                "min_us": min(samples),
                "max_us": max(samples),
                "spread_percent": spread,
                "stable_le_5_percent": spread <= 5.0,
                "warmup": args.warmup,
                "iters": args.iters,
                "repeats": args.repeats,
                "timing": "cuda_graph_external_events_inside",
                "boundary": "BF16 input -> SGLang legacy Triton W8A8 FP8 MoE chain -> BF16 output",
                "fixture_sha256": fixture["fixture_sha256"],
                "occupancy_sha256": fixture["occupancy_sha256"],
                "config_source": contract["config"]["source"],
                "rerun_id": runtime["measurement_rerun_id"],
                "comparison_group_id": COMPARISON_GROUP_ID,
            }
        )
        write_csv(args.results / "benchmark_summary.csv", summary)
        del captured, eager_output, reference, x, topk_ids, topk_weights
        torch.cuda.empty_cache()

    artifacts = artifact_manifest(weights)
    environment = {
        "runtime": {key: value for key, value in runtime.items() if key != "timestamp_unix"}
    }
    environment["fingerprint_sha256"] = canonical_sha256(environment)
    write_json(args.results / "environment.lock.json", environment)
    write_json(args.results / "artifact.lock.json", artifacts)
    write_json(args.results / "protocol.lock.json", protocol)
    identity = {
        "schema_version": 1,
        "comparison_group_id": COMPARISON_GROUP_ID,
        "rerun_id": runtime["measurement_rerun_id"],
        "gpu_uuid": runtime["gpu_uuid"],
        "environment_lock_digest": environment["fingerprint_sha256"],
        "artifact_fingerprint_sha256": artifacts["fingerprint_sha256"],
        "protocol_lock_digest": protocol["fingerprint_sha256"],
        "m_values": list(args.m_values),
        "smoke": bool(args.smoke),
    }
    identity["fingerprint_sha256"] = canonical_sha256(identity)
    write_json(args.results / "evidence.identity.json", identity)
    correctness["evidence_identity"] = identity
    correctness["all_correctness_and_dispatch_gates_pass"] = True
    write_json(args.results / "correctness.json", correctness)
    for row in raw + summary:
        row["environment_lock_digest"] = identity["environment_lock_digest"]
        row["artifact_fingerprint_sha256"] = identity[
            "artifact_fingerprint_sha256"
        ]
        row["protocol_lock_digest"] = identity["protocol_lock_digest"]
    write_csv(args.results / "benchmark_raw.csv", raw)
    write_csv(args.results / "benchmark_summary.csv", summary)
    write_json(args.results / "manifests" / "runtime.json", runtime)
    write_json(args.results / "manifests" / "artifacts.json", artifacts)
    return int(any(not row["stable_le_5_percent"] for row in summary))


if __name__ == "__main__":
    raise SystemExit(main())
