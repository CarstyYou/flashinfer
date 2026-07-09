"""task_02 bench: moe_gemm_fp8_nt_groupwise (cute) vs group_gemm_fp8_nt_groupwise (cutlass sm120).

Same fp8 inputs per cell; the cutlass baseline consumes MN-major SFA [Kb, cum_m]
(contiguous, no zero-padding) and SFB [E, Kb, Nb]. Cells restricted to
m_pe % 4 == 0 (baseline m_indptr alignment contract). warmup 10 + 50 iter median.
"""

import argparse
import csv
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

sys.path.insert(0, str(Path(__file__).parent))
from test_fp8_correctness import build_zero_padding_sfa  # noqa: E402

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


def bench_cell(e, m_pe, n, k):
    torch.random.manual_seed(0)
    offsets = [i * m_pe for i in range(e + 1)]
    total_rows = offsets[-1]
    a = torch.randn((total_rows, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((e, n, k), dtype=torch.bfloat16, device="cuda") / math.sqrt(k)
    m_indptr = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    a_fp8, sf_a = per_token_cast_to_fp8(a)
    a_scale_zp = build_zero_padding_sfa(sf_a, offsets)
    a_scale_mn = sf_a.t().contiguous()
    b_fp8_list, b_sf_list = [], []
    for i in range(e):
        f, s = per_block_cast_to_fp8(b[i])
        b_fp8_list.append(f)
        b_sf_list.append(s)
    b_fp8 = torch.stack(b_fp8_list, dim=0)
    b_scale = torch.stack(b_sf_list, dim=0).transpose(-1, -2).contiguous()

    out_cute = torch.empty((total_rows, n), dtype=torch.bfloat16, device="cuda")
    us_cute = median_us(
        lambda: moe_gemm_fp8_nt_groupwise(
            a_fp8, b_fp8, a_scale_zp, b_scale, m_indptr, out=out_cute
        )
    )
    out_base = torch.empty((total_rows, n), dtype=torch.bfloat16, device="cuda")
    # skip_check: sm120 num_groups > 1 disable is wrapper-level (see task findings);
    # bypassed for perf baseline only, no correctness claim.
    us_base = median_us(
        lambda: group_gemm_fp8_nt_groupwise(
            a_fp8, b_fp8, a_scale_mn, b_scale, m_indptr,
            scale_major_mode="MN", out=out_base, skip_check=True,
        )
    )
    return us_cute, us_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="r1")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n, k in [(4096, 7168), (7168, 4096)]:
        for e in (4, 8):
            for m_pe in (4, 8, 16, 192, 256, 1024):
                us_cute, us_base = bench_cell(e, m_pe, n, k)
                pct = (us_base - us_cute) / us_base * 100.0
                rows.append([e, m_pe, n, k, f"{us_cute:.3f}", f"{us_base:.3f}", f"{pct:+.2f}%"])
                print(rows[-1])

    out_csv = outdir / f"bench_fp8_vs_cutlass_{args.tag}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["E", "m_pe", "N", "K", "cute_us", "cutlass_us", "speedup"])
        w.writerows(rows)
    print(f"[done] {len(rows)} cells -> {out_csv}")


if __name__ == "__main__":
    main()
