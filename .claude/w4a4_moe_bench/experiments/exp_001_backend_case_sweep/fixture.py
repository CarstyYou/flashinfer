"""Shared, persisted input/routing fixtures for the Triton-arm benchmark."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

E = 256
H = 2048
TOPK = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixture_path(root: Path, m: int) -> Path:
    return root / f"m{m}.npz"


def load_fixture(root: Path, m: int, device: torch.device):
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
    if not np.isfinite(topk_weights_np).all():
        raise ValueError("fixture contains non-finite routing weight")
    if not np.allclose(topk_weights_np.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("fixture routing weights are not normalized")

    occupancy = np.bincount(topk_ids_np.reshape(-1), minlength=E).astype(np.int64)
    occupancy_sha256 = hashlib.sha256(occupancy.tobytes()).hexdigest()
    x = torch.from_numpy(x_bits).view(torch.bfloat16).to(device=device)
    topk_ids = torch.from_numpy(topk_ids_np).to(device=device)
    topk_weights = torch.from_numpy(topk_weights_np).to(device=device)
    return (
        x,
        topk_ids,
        topk_weights,
        {
            "fixture_sha256": sha256_file(path),
            "occupancy_sha256": occupancy_sha256,
            "occupancy_min": int(occupancy.min()),
            "occupancy_max": int(occupancy.max()),
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
