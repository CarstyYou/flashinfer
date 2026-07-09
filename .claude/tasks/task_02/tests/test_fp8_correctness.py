"""task_02 FP8 moe_gemm correctness cells.

Mirrors 6KD test/test_fp8.py::test_moe_gemm_fp8_nt_groupwise: bf16-source
per-expert matmul reference, calc_diff < 1e-3. SFA constructed in the
zero-padding [Kb, MpE] layout with per-expert 4-row-aligned offsets.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import torch

FI_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(FI_ROOT))

from flashinfer.grouped_mm import moe_gemm_fp8_nt_groupwise  # noqa: E402
from flashinfer.testing.utils import (  # noqa: E402
    per_block_cast_to_fp8,
    per_token_cast_to_fp8,
)


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum().item()
    if denom == 0:
        return 0.0
    return 1.0 - 2.0 * (x * y).sum().item() / denom


def compute_padded_offset(offset: int, problem_idx: int) -> int:
    return (offset + problem_idx * 3) // 4 * 4


def build_zero_padding_sfa(sf_a: torch.Tensor, offsets) -> torch.Tensor:
    total_rows = sf_a.size(0)
    num_experts = len(offsets) - 1
    scale_k = sf_a.size(1)
    padded_m = compute_padded_offset(total_rows, num_experts)
    padded = torch.zeros((scale_k, padded_m), device=sf_a.device, dtype=sf_a.dtype)
    for i in range(num_experts):
        start, end = offsets[i], offsets[i + 1]
        if start == end:
            continue
        padded_start = compute_padded_offset(start, i)
        padded[:, padded_start : padded_start + end - start] = sf_a[start:end].t()
    return padded


def run_cell(m_per_expert_list, n, k):
    torch.random.manual_seed(0)
    num_experts = len(m_per_expert_list)
    offsets = [0]
    for m_pe in m_per_expert_list:
        offsets.append(offsets[-1] + m_pe)
    total_rows = offsets[-1]

    a = torch.randn((total_rows, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn(
        (num_experts, n, k), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)
    m_indptr = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    ref = torch.zeros((total_rows, n), dtype=torch.bfloat16, device="cuda")
    for i in range(num_experts):
        start, end = offsets[i], offsets[i + 1]
        if start < end:
            ref[start:end] = a[start:end] @ b[i].t()

    a_fp8, sf_a = per_token_cast_to_fp8(a)
    a_scale = build_zero_padding_sfa(sf_a, offsets)
    b_fp8_list, b_sf_list = [], []
    for i in range(num_experts):
        b_i_fp8, b_i_sf = per_block_cast_to_fp8(b[i])
        b_fp8_list.append(b_i_fp8)
        b_sf_list.append(b_i_sf)
    b_fp8 = torch.stack(b_fp8_list, dim=0)
    b_scale = torch.stack(b_sf_list, dim=0).transpose(-1, -2).contiguous()

    out = moe_gemm_fp8_nt_groupwise(a_fp8, b_fp8, a_scale, b_scale, m_indptr)
    torch.cuda.synchronize()
    return calc_diff(out.float(), ref.float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="fp8")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = 0

    def record(e, label, n, k, diff):
        nonlocal failures
        ok = diff < 1e-3
        failures += 0 if ok else 1
        rows.append([e, label, n, k, f"{diff:.6e}", "PASS" if ok else "FAIL"])
        print(rows[-1])

    for n, k in [(4096, 7168), (7168, 4096)]:
        for e in (4, 8):
            for m_pe in (1, 4, 8, 16, 192, 256, 1024):
                record(e, f"uniform_{m_pe}", n, k, run_cell([m_pe] * e, n, k))
        record(8, "uneven", n, k, run_cell([1, 1, 8, 16, 64, 128, 192, 256], n, k))
        record(8, "empty_expert", n, k, run_cell([0, 8, 0, 256, 16, 0, 1, 64], n, k))
        record(16, "routing_512", n, k, run_cell([32] * 16, n, k))

    out_csv = outdir / f"correctness_{args.tag}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["E", "m_pe", "N", "K", "calc_diff", "verdict"])
        w.writerows(rows)
    print(f"[done] {len(rows)} cells, {failures} FAIL -> {out_csv}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
