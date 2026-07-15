#!/usr/bin/env python3
"""Create one framework-neutral synthetic input/routing fixture per M."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fixture import E, H, TOPK, fixture_path, sha256_file

M_VALUES = (256, 512, 1024, 2048, 4096, 8192)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "fixtures"


def make_fixture(root: Path, m: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed + m)
    x = (rng.standard_normal((m, H), dtype=np.float32) / 10).astype(np.float32)
    x_bits = torch.from_numpy(x).to(torch.bfloat16).view(torch.uint16).numpy().copy()
    logits = rng.standard_normal((m, E), dtype=np.float32)
    unsorted_ids = np.argpartition(logits, -TOPK, axis=1)[:, -TOPK:]
    unsorted_logits = np.take_along_axis(logits, unsorted_ids, axis=1)
    order = np.argsort(-unsorted_logits, axis=1)
    topk_ids = np.take_along_axis(unsorted_ids, order, axis=1).astype(np.int32)
    topk_logits = np.take_along_axis(unsorted_logits, order, axis=1)
    shifted = topk_logits - topk_logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)

    path = fixture_path(root, m)
    np.savez(path, x_bf16_bits=x_bits, topk_ids=topk_ids, topk_weights=weights)
    occupancy = np.bincount(topk_ids.reshape(-1), minlength=E)
    return {
        "m": m,
        "path": path.name,
        "sha256": sha256_file(path),
        "occupancy_min": int(occupancy.min()),
        "occupancy_max": int(occupancy.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = [make_fixture(args.output_dir, m, args.seed) for m in M_VALUES]
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"seed": args.seed, "fixtures": manifest}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
