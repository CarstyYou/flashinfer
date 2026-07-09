"""m_pe=1 cells: cute (unpadded, zero-padding mode) vs cutlass (caller pads each
expert to 4 rows — cutlass's own m_indptr multiple-of-4 contract). Padded
performance is charged to cutlass as-is."""

import math
import sys
from pathlib import Path

import torch

FI_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(FI_ROOT))

from flashinfer.gemm import group_gemm_fp8_nt_groupwise  # noqa: E402
from flashinfer.grouped_mm import moe_gemm_fp8_nt_groupwise  # noqa: E402
from flashinfer.testing.utils import (  # noqa: E402
    per_block_cast_to_fp8,
    per_token_cast_to_fp8,
)

sys.path.insert(0, str(FI_ROOT / "tests" / "grouped_mm"))
from test_cute_sm120_fp8 import per_token_cast_to_fp8_for_moe_gemm  # noqa: E402

WARMUP = 10
ITERS = 50


def median_us(call):
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        call()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3)
    times.sort()
    return times[len(times) // 2]


def bench(e, n, k):
    torch.random.manual_seed(0)
    m_pe = 1
    total = e * m_pe
    a = torch.randn((total, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((e, n, k), dtype=torch.bfloat16, device="cuda") / math.sqrt(k)
    m_indptr = torch.tensor([i * m_pe for i in range(e + 1)], dtype=torch.int32, device="cuda")

    a_fp8, a_scale = per_token_cast_to_fp8_for_moe_gemm(a, m_indptr)
    b_fp8_list, b_sf_list = [], []
    for i in range(e):
        f, s = per_block_cast_to_fp8(b[i])
        b_fp8_list.append(f)
        b_sf_list.append(s)
    b_fp8 = torch.stack(b_fp8_list, dim=0)
    b_scale = torch.stack(b_sf_list, dim=0).transpose(-1, -2).contiguous()

    out_cute = torch.empty((total, n), dtype=torch.bfloat16, device="cuda")
    us_cute = median_us(
        lambda: moe_gemm_fp8_nt_groupwise(a_fp8, b_fp8, a_scale, b_scale, m_indptr, out=out_cute)
    )

    # cutlass side: pad each expert to 4 rows (its m_indptr multiple-of-4 contract).
    pad = 4
    a_padded = torch.zeros((e * pad, k), dtype=torch.bfloat16, device="cuda")
    for i in range(e):
        a_padded[i * pad] = a[i]
    m_indptr_p = torch.tensor([i * pad for i in range(e + 1)], dtype=torch.int32, device="cuda")
    ap_fp8, ap_sf = per_token_cast_to_fp8(a_padded)
    ap_scale_mn = ap_sf.t().contiguous()
    out_base = torch.empty((e * pad, n), dtype=torch.bfloat16, device="cuda")
    us_cutlass = median_us(
        lambda: group_gemm_fp8_nt_groupwise(
            ap_fp8, b_fp8, ap_scale_mn, b_scale, m_indptr_p,
            scale_major_mode="MN", out=out_base, skip_check=True,
        )
    )
    pct_over = (us_cutlass / us_cute - 1.0) * 100.0
    print(f"E={e} N={n} K={k}: cute {us_cute:.3f} us | cutlass(pad4) {us_cutlass:.3f} us (+{pct_over:.1f}%)")


def main():
    for n, k in [(4096, 7168), (7168, 4096)]:
        for e in (4, 8):
            bench(e, n, k)


if __name__ == "__main__":
    main()
