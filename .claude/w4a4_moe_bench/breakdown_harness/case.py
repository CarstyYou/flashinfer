"""Canonical Qwen3.5-shape NVFP4 fixtures and an independent MoE oracle.

Both production arms consume views derived from the same packed weights and
logical scales.  The reference deliberately reconstructs the MoE with PyTorch
expert matmuls; it does not call either fused-MoE implementation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

E = 256
H = 2048
I = 512
TOPK = 8
SF_VEC = 16
CASE_FAMILY = "qwen35-moe-e256-h2048-i512-topk8"


def case_id(m: int) -> str:
    if m <= 0:
        raise ValueError(f"M must be positive, got {m}")
    return f"{CASE_FAMILY}-m{m}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixture_path(root: Path, m: int) -> Path:
    return root / f"m{m}.npz"


def sha256_array(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(value)
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def load_fixture(root: Path, m: int, device: torch.device):
    """Load the persisted BF16 input and top-k routing for one canonical case."""
    path = fixture_path(root, m)
    with np.load(path) as fixture:
        x_bits = fixture["x_bf16_bits"].copy()
        topk_ids_np = fixture["topk_ids"].copy()
        topk_weights_np = fixture["topk_weights"].copy()
    if x_bits.shape != (m, H) or x_bits.dtype != np.uint16:
        raise ValueError(
            f"invalid x fixture: shape={x_bits.shape}, dtype={x_bits.dtype}"
        )
    if topk_ids_np.shape != (m, TOPK) or topk_ids_np.dtype != np.int32:
        raise ValueError("invalid topk_ids fixture")
    if topk_weights_np.shape != (m, TOPK) or topk_weights_np.dtype != np.float32:
        raise ValueError("invalid topk_weights fixture")
    if topk_ids_np.min() < 0 or topk_ids_np.max() >= E:
        raise ValueError("fixture expert id is out of range")
    sorted_ids = np.sort(topk_ids_np, axis=1)
    if np.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
        raise ValueError("fixture contains duplicate expert ids for one token")
    if not np.isfinite(topk_weights_np).all():
        raise ValueError("fixture contains non-finite routing weight")
    if (topk_weights_np < 0).any():
        raise ValueError("fixture contains a negative routing weight")
    if not np.allclose(topk_weights_np.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("fixture routing weights are not normalized")

    occupancy = np.bincount(topk_ids_np.reshape(-1), minlength=E).astype(np.int64)
    x = torch.from_numpy(x_bits).view(torch.bfloat16).to(device=device)
    topk_ids = torch.from_numpy(topk_ids_np).to(device=device)
    topk_weights = torch.from_numpy(topk_weights_np).to(device=device)
    return (
        x,
        topk_ids,
        topk_weights,
        {
            "case_id": case_id(m),
            "fixture_kind": "persisted_numpy_bf16_input_and_topk_routing",
            "m": m,
            "shape": {
                "experts": E,
                "hidden": H,
                "intermediate_tp": I,
                "topk": TOPK,
            },
            "fixture_path": path.name,
            "fixture_sha256": sha256_file(path),
            "x_sha256": sha256_array(x_bits),
            "topk_ids_sha256": sha256_array(topk_ids_np),
            "topk_weights_sha256": sha256_array(topk_weights_np),
            "occupancy_sha256": hashlib.sha256(occupancy.tobytes()).hexdigest(),
            "occupancy_min": int(occupancy.min()),
            "occupancy_max": int(occupancy.max()),
            "duplicate_expert_ids": 0,
            "weight_sum_max_abs_error": float(
                np.max(np.abs(topk_weights_np.sum(axis=1) - 1.0))
            ),
        },
    )


def validate_output(output: torch.Tensor, m: int) -> None:
    if output.shape != (m, H):
        raise ValueError(f"wrong output shape {tuple(output.shape)}")
    if output.dtype != torch.bfloat16:
        raise ValueError(f"wrong output dtype {output.dtype}")
    if not torch.isfinite(output).all().item():
        raise ValueError("output contains non-finite values")
    if not torch.count_nonzero(output).item():
        raise ValueError("output is entirely zero")


def round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash dtype, shape, and logical contiguous bytes of a tensor."""
    value = tensor.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RoutedFixture:
    m: int
    x: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    manifest: dict[str, Any]


@dataclass(frozen=True)
class CanonicalWeights:
    w1_packed: torch.Tensor
    w2_packed: torch.Tensor
    w1_scale_linear: torch.Tensor
    w2_scale_linear: torch.Tensor
    w1_scale_cutedsl: torch.Tensor
    w2_scale_cutedsl: torch.Tensor
    w1_scale_cutlass: torch.Tensor
    w2_scale_cutlass: torch.Tensor
    w1_global_scale: torch.Tensor
    w2_global_scale: torch.Tensor
    fc2_input_scale: torch.Tensor
    manifest: dict[str, Any]

    def cutedsl(self) -> dict[str, torch.Tensor | str]:
        return {
            "w1_fp4": self.w1_packed,
            "w1_sf": self.w1_scale_cutedsl,
            "w1_alpha": self.w1_global_scale,
            "w2_fp4": self.w2_packed,
            "w2_sf": self.w2_scale_cutedsl,
            "w2_alpha": self.w2_global_scale,
            "fc2_input_scale": self.fc2_input_scale,
            "w13_layout": "w31",
        }

    def cutlass(self) -> dict[str, torch.Tensor]:
        return {
            "fc1_weight": self.w1_packed.contiguous().view(torch.long),
            "fc2_weight": self.w2_packed.contiguous().view(torch.long),
            "fc1_blockscale_i32": self.w1_scale_cutlass.view(torch.int32),
            "fc2_blockscale_i32": self.w2_scale_cutlass.view(torch.int32),
            "fc1_gs": self.w1_global_scale,
            "fc2_gs": self.w2_global_scale,
        }


def make_routed_fixture(
    m: int, *, device: torch.device, seed: int = 2026
) -> RoutedFixture:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + m)
    x = (
        torch.randn((m, H), generator=generator, device=device, dtype=torch.float32)
        / 10.0
    ).to(torch.bfloat16)
    logits = torch.randn(
        (m, E), generator=generator, device=device, dtype=torch.float32
    )
    topk_logits, topk_ids = torch.topk(logits, TOPK, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1, dtype=torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    sorted_ids = topk_ids.sort(dim=-1).values
    duplicate_count = int((sorted_ids[:, 1:] == sorted_ids[:, :-1]).sum().item())
    if duplicate_count:
        raise ValueError(f"routing contains {duplicate_count} duplicate expert ids")
    if not torch.isfinite(topk_weights).all().item():
        raise ValueError("routing weights contain non-finite values")
    if (topk_weights < 0).any().item():
        raise ValueError("routing weights contain negative values")
    torch.testing.assert_close(
        topk_weights.sum(dim=-1),
        torch.ones(m, device=device),
        rtol=1e-6,
        atol=1e-6,
    )

    occupancy = torch.bincount(topk_ids.flatten().long(), minlength=E)
    occupancy_f32 = occupancy.float()
    manifest = {
        "fixture_kind": "deterministic_synthetic_random_logits",
        "seed": seed + m,
        "m": m,
        "shape": {"experts": E, "hidden": H, "intermediate_tp": I, "topk": TOPK},
        "x_sha256": tensor_sha256(x),
        "topk_ids_sha256": tensor_sha256(topk_ids),
        "topk_weights_sha256": tensor_sha256(topk_weights),
        "occupancy_sha256": tensor_sha256(occupancy),
        "occupancy_min": int(occupancy.min().item()),
        "occupancy_p50": float(torch.quantile(occupancy_f32, 0.50).item()),
        "occupancy_p95": float(torch.quantile(occupancy_f32, 0.95).item()),
        "occupancy_max": int(occupancy.max().item()),
        "zero_token_experts": int((occupancy == 0).sum().item()),
        "duplicate_expert_ids": duplicate_count,
        "weight_sum_max_abs_error": float(
            (topk_weights.sum(dim=-1) - 1.0).abs().max().item()
        ),
        "predicted_cutlass_route_branch": (
            "fused_route_candidate"
            if m == 256
            else (
                "three_step_large_prefix_candidate"
                if m == 8192
                else "three_step_route_prefix_variant_unresolved"
            )
        ),
    }
    return RoutedFixture(m, x, topk_ids, topk_weights, manifest)


def make_canonical_weights(
    *, device: torch.device, seed: int = 2026
) -> CanonicalWeights:
    """Generate one packed weight payload and derive both scale layouts.

    The logical scale uses a deterministic, nonuniform E4M3 pattern so a bad
    layout transform is numerically observable.  Both backend views still
    derive from the same canonical logical tensor and swizzled storage.
    """
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.quantization.fp4_quantization import block_scale_interleave

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    w1_packed = torch.randint(
        0,
        256,
        (E, 2 * I, H // 2),
        generator=generator,
        device=device,
        dtype=torch.uint8,
    )
    w2_packed = torch.randint(
        0,
        256,
        (E, H, I // 2),
        generator=generator,
        device=device,
        dtype=torch.uint8,
    )
    w1_scale_linear = torch.ones(
        (E, 2 * I, H // SF_VEC), device=device, dtype=torch.float8_e4m3fn
    )
    w2_scale_linear = torch.ones(
        (E, H, I // SF_VEC), device=device, dtype=torch.float8_e4m3fn
    )
    # Nonuniform, exactly representable E4M3 values make layout mistakes
    # observable.  The small powers of two keep uniformly sampled E2M1 codes
    # in the same rough magnitude regime as model weights instead of creating
    # pathological +/-12 weights and unstable activation quantization.
    for scale in (w1_scale_linear, w2_scale_linear):
        scale[..., 0::3] = 1.0 / 64.0
        scale[..., 1::3] = 1.0 / 32.0
        scale[..., 2::3] = 1.0 / 16.0
    w1_swizzled = block_scale_interleave(w1_scale_linear.view(torch.uint8)).view(
        torch.float8_e4m3fn
    )
    w2_swizzled = block_scale_interleave(w2_scale_linear.view(torch.uint8)).view(
        torch.float8_e4m3fn
    )
    w1_swizzled_2d = w1_swizzled.view(E * 2 * I, H // SF_VEC)
    w2_swizzled_2d = w2_swizzled.view(E * H, I // SF_VEC)
    w1_scale_cutedsl = convert_sf_to_mma_layout(
        w1_swizzled_2d,
        m=2 * I,
        k=H,
        num_groups=E,
        sf_vec_size=SF_VEC,
    )
    w2_scale_cutedsl = convert_sf_to_mma_layout(
        w2_swizzled_2d,
        m=H,
        k=I,
        num_groups=E,
        sf_vec_size=SF_VEC,
    )
    w1_scale_cutlass = w1_swizzled.view(
        E, round_up(2 * I, 128), round_up(H // SF_VEC, 4)
    )
    w2_scale_cutlass = w2_swizzled.view(E, round_up(H, 128), round_up(I // SF_VEC, 4))
    w1_global_scale = torch.ones(E, device=device, dtype=torch.float32)
    w2_global_scale = torch.ones(E, device=device, dtype=torch.float32)
    fc2_input_scale = torch.ones(1, device=device, dtype=torch.float32)

    if not torch.equal(
        swizzled_scale_to_linear(w1_scale_cutlass, m=2 * I, k=H),
        w1_scale_linear,
    ):
        raise RuntimeError("W1 logical -> swizzled scale round trip failed")
    if not torch.equal(
        swizzled_scale_to_linear(w2_scale_cutlass, m=H, k=I),
        w2_scale_linear,
    ):
        raise RuntimeError("W2 logical -> swizzled scale round trip failed")
    if (
        w1_scale_cutedsl.untyped_storage().data_ptr()
        != w1_swizzled.untyped_storage().data_ptr()
    ):
        raise RuntimeError(
            "W1 CuteDSL scale view does not alias canonical swizzled storage"
        )
    if (
        w2_scale_cutedsl.untyped_storage().data_ptr()
        != w2_swizzled.untyped_storage().data_ptr()
    ):
        raise RuntimeError(
            "W2 CuteDSL scale view does not alias canonical swizzled storage"
        )

    manifest = {
        "seed": seed,
        "canonical_format": "modelopt_w31_packed_e2m1_nonuniform_e4m3_scale",
        "w1_packed_sha256": tensor_sha256(w1_packed),
        "w2_packed_sha256": tensor_sha256(w2_packed),
        "w1_logical_scale_sha256": tensor_sha256(w1_scale_linear),
        "w2_logical_scale_sha256": tensor_sha256(w2_scale_linear),
        "w1_swizzled_storage_sha256": tensor_sha256(w1_swizzled),
        "w2_swizzled_storage_sha256": tensor_sha256(w2_swizzled),
        "w1_cutedsl_scale_sha256": tensor_sha256(w1_scale_cutedsl),
        "w2_cutedsl_scale_sha256": tensor_sha256(w2_scale_cutedsl),
        "w1_cutlass_scale_sha256": tensor_sha256(w1_scale_cutlass),
        "w2_cutlass_scale_sha256": tensor_sha256(w2_scale_cutlass),
        "w1_global_scale_sha256": tensor_sha256(w1_global_scale),
        "w2_global_scale_sha256": tensor_sha256(w2_global_scale),
        "fc2_input_scale_sha256": tensor_sha256(fc2_input_scale),
        "w1_packed_shape": list(w1_packed.shape),
        "w2_packed_shape": list(w2_packed.shape),
        "logical_scale_pattern": "E4M3 repeating 1/64,1/32,1/16 over K blocks",
        "w1_scale_transform": "logical linear -> block_scale_interleave -> shared swizzled storage -> cutedsl MMA/CUTLASS views",
        "w2_scale_transform": "logical linear -> block_scale_interleave -> shared swizzled storage -> cutedsl MMA/CUTLASS views",
    }
    return CanonicalWeights(
        w1_packed,
        w2_packed,
        w1_scale_linear,
        w2_scale_linear,
        w1_scale_cutedsl,
        w2_scale_cutedsl,
        w1_scale_cutlass,
        w2_scale_cutlass,
        w1_global_scale,
        w2_global_scale,
        fc2_input_scale,
        manifest,
    )


_E2M1_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    values = _E2M1_VALUES.to(packed.device)
    unpacked = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        device=packed.device,
        dtype=torch.uint8,
    )
    unpacked[..., 0::2] = packed & 0x0F
    unpacked[..., 1::2] = packed >> 4
    return values[unpacked.long()]


def dequantize_linear_nvfp4(
    packed: torch.Tensor,
    scale_linear: torch.Tensor,
    *,
    global_scale: float = 1.0,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    values = unpack_e2m1(packed)
    k = values.shape[-1]
    expected = (*values.shape[:-1], k // SF_VEC)
    if tuple(scale_linear.shape) != expected:
        raise ValueError(
            f"linear scale shape {tuple(scale_linear.shape)} != expected {expected}"
        )
    if scale_linear.dtype == torch.uint8:
        scale_values = scale_linear.view(torch.float8_e4m3fn).float()
    else:
        scale_values = scale_linear.float()
    scaled = values.view(*values.shape[:-1], k // SF_VEC, SF_VEC)
    scaled = scaled * (scale_values / global_scale).unsqueeze(-1)
    return scaled.reshape_as(values).to(dtype)


def swizzled_scale_to_linear(
    scale_swizzled: torch.Tensor, *, m: int, k: int
) -> torch.Tensor:
    """Invert FlashInfer's 128x4 NVFP4 scale-factor swizzle."""
    m_tiles = math.ceil(m / 128)
    k_tiles = math.ceil(k / 64)
    group_size = m_tiles * k_tiles * 32 * 4 * 4
    if scale_swizzled.numel() % group_size:
        raise ValueError("swizzled scale storage does not contain whole groups")
    groups = scale_swizzled.numel() // group_size
    temporary = scale_swizzled.reshape(groups, m_tiles, k_tiles, 32, 4, 4)
    linear = temporary.permute(0, 1, 4, 3, 2, 5).reshape(
        groups, m_tiles * 128, k_tiles * 4
    )
    linear = linear[:, :m, : k // SF_VEC]
    return linear[0] if groups == 1 else linear


def dequantize_swizzled_nvfp4(
    packed: torch.Tensor,
    scale_swizzled: torch.Tensor,
    *,
    global_scale: float = 1.0,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    m, packed_k = packed.shape
    k = packed_k * 2
    linear = swizzled_scale_to_linear(scale_swizzled, m=m, k=k)
    return dequantize_linear_nvfp4(
        packed, linear, global_scale=global_scale, dtype=dtype
    )


@torch.inference_mode()
def reference_moe_nvfp4(
    fixture: RoutedFixture, weights: CanonicalWeights
) -> torch.Tensor:
    """Execute the specified quantized MoE as independent expert operations."""
    from flashinfer import fp4_quantize

    scalar_one = torch.ones((), device=fixture.x.device, dtype=torch.float32)
    input_q, input_sf = fp4_quantize(
        fixture.x,
        scalar_one,
        is_sf_swizzled_layout=False,
        enable_pdl=True,
    )
    input_dequant = dequantize_linear_nvfp4(
        input_q, input_sf, global_scale=1.0, dtype=torch.float32
    )

    flat_experts = fixture.topk_ids.flatten().long()
    flat_output = torch.zeros(
        (fixture.m * TOPK, H), device=fixture.x.device, dtype=torch.float32
    )
    for expert in range(E):
        route_positions = torch.nonzero(
            flat_experts == expert, as_tuple=False
        ).flatten()
        if route_positions.numel() == 0:
            continue
        token_positions = torch.div(route_positions, TOPK, rounding_mode="floor")
        expert_input = input_dequant[token_positions]
        w1 = dequantize_linear_nvfp4(
            weights.w1_packed[expert],
            weights.w1_scale_linear[expert],
            global_scale=float(weights.w1_global_scale[expert].item()),
            dtype=torch.float32,
        )
        # ModelOpt w31: first half is up (w3), second half is gate (w1).
        up = expert_input @ w1[:I].transpose(0, 1)
        gate = expert_input @ w1[I:].transpose(0, 1)
        activation = F.silu(gate) * up
        activation_q, activation_sf = fp4_quantize(
            activation.to(torch.bfloat16),
            scalar_one,
            is_sf_swizzled_layout=False,
            enable_pdl=True,
        )
        activation_dequant = dequantize_linear_nvfp4(
            activation_q,
            activation_sf,
            global_scale=1.0,
            dtype=torch.float32,
        )
        w2 = dequantize_linear_nvfp4(
            weights.w2_packed[expert],
            weights.w2_scale_linear[expert],
            global_scale=float(weights.w2_global_scale[expert].item()),
            dtype=torch.float32,
        )
        flat_output[route_positions] = activation_dequant @ w2.transpose(0, 1)

    return (
        flat_output.view(fixture.m, TOPK, H) * fixture.topk_weights.unsqueeze(-1)
    ).sum(dim=1)


def output_diagnostics(
    actual: torch.Tensor, reference: torch.Tensor
) -> dict[str, float | bool | list[int] | str]:
    actual_f32 = actual.float()
    reference_f32 = reference.float()
    error = actual_f32 - reference_f32
    reference_norm = torch.linalg.vector_norm(reference_f32)
    actual_norm = torch.linalg.vector_norm(actual_f32)
    relative_l2 = torch.linalg.vector_norm(error) / reference_norm.clamp_min(1e-12)
    cosine = F.cosine_similarity(actual_f32.flatten(), reference_f32.flatten(), dim=0)
    strict_tolerance = 0.2 + 0.2 * reference_f32.abs()
    output_scale = max(float(reference_f32.std().item()), 0.01)
    magnitude_atol = max(0.05, 1.5 * output_scale)
    relative_error = error.abs() / (reference_f32.abs() + 1e-8)
    magnitude_within = (error.abs() < magnitude_atol) | (relative_error < 0.5)
    formal_pass_ratio = float(magnitude_within.float().mean().item())
    least_squares_gain = torch.sum(actual_f32 * reference_f32) / torch.sum(
        reference_f32 * reference_f32
    ).clamp_min(1e-12)
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype).replace("torch.", ""),
        "finite": bool(torch.isfinite(actual).all().item()),
        "nonzero": bool(torch.count_nonzero(actual).item()),
        "cosine": float(cosine.item()),
        "relative_l2": float(relative_l2.item()),
        "actual_mean": float(actual_f32.mean().item()),
        "actual_std": float(actual_f32.std().item()),
        "reference_mean": float(reference_f32.mean().item()),
        "reference_std": float(reference_f32.std().item()),
        "actual_to_reference_l2_ratio": float(
            (actual_norm / reference_norm.clamp_min(1e-12)).item()
        ),
        "least_squares_actual_gain": float(least_squares_gain.item()),
        "max_abs_error": float(error.abs().max().item()),
        "mean_abs_error": float(error.abs().mean().item()),
        "percent_within_rtol_0.2_atol_0.2": float(
            (error.abs() <= strict_tolerance).float().mean().item() * 100.0
        ),
        "strict_rtol_0.2_atol_0.2_allclose": bool(
            torch.all(error.abs() <= strict_tolerance).item()
        ),
        "formal_magnitude_atol": magnitude_atol,
        "formal_magnitude_rtol": 0.5,
        "formal_percent_threshold": 97.0,
        "formal_percent_within": formal_pass_ratio * 100.0,
        "formal_pass": formal_pass_ratio >= 0.97,
        "output_sha256": tensor_sha256(actual),
        "reference_sha256": tensor_sha256(reference),
    }
