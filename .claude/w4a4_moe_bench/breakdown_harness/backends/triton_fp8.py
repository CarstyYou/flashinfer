"""Reusable SGLang Triton FP8 MoE backend primitives.

This module deliberately owns no experiment identity, environment pins, default
paths, CLI, or evidence-output protocol.  Those remain with the experiment that
uses these primitives.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.nn import functional as F

from breakdown_harness.artifacts import canonical_sha256, file_sha256
from breakdown_harness.case import E, H, I, TOPK

__all__ = (
    "CapturedCall",
    "Fp8Weights",
    "build_launch",
    "canonical_sha256",
    "file_sha256",
    "fp8_oracle",
    "initialize_sglang",
    "make_fp8_weights",
    "make_l2_flusher",
    "output_diagnostics",
    "resolved_config",
    "tensor_sha256",
)


def initialize_sglang(*, expected_version: str | None = None) -> dict[str, Any]:
    """Initialize the Triton MoE backend without owning experiment pins."""
    import sglang
    import triton
    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    server_args = ServerArgs(model_path="dummy", moe_runner_backend="triton")
    set_global_server_args_for_scheduler(server_args)
    version = importlib.metadata.version("sglang")
    if expected_version is not None and version != expected_version:
        raise RuntimeError(f"SGLang version drift: {version} != {expected_version}")
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


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


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
            torch.randn((H, I), generator=generator, device=device, dtype=torch.float32)
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
        activation[positions] = (F.silu(fc1[:, :I]) * fc1[:, I:]).to(torch.bfloat16)
    activation_q, activation_scale = scaled_fp8_quant(activation, None)
    activation_dequant = activation_q[..., :I].float() * activation_scale.float()
    flat_output = torch.empty((m * TOPK, H), device=x.device, dtype=torch.bfloat16)
    for expert in range(E):
        positions = torch.nonzero(flat_ids == expert, as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        weight = weights.w2[expert].float() * weights.w2_scale[expert]
        flat_output[positions] = (
            activation_dequant[positions] @ weight.transpose(0, 1)
        ).to(torch.bfloat16)
    return (flat_output.float().view(m, TOPK, H) * topk_weights.unsqueeze(-1)).sum(
        dim=1
    )


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


def make_l2_flusher(
    device: torch.device, num_bytes: int
) -> tuple[Callable[[], None], int]:
    buffer = torch.empty((num_bytes + 3) // 4, device=device, dtype=torch.int32)
    state = 0

    def flush() -> None:
        nonlocal state
        state = (state + 1) & 0x7FFFFFFF
        buffer.fill_(state)
        torch.cuda.synchronize()

    flush()
    return flush, buffer.numel() * buffer.element_size()
