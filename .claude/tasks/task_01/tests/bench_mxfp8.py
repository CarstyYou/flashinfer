"""task_01 MXFP8 perf cells (pre/post kernel-tree sync).

Same uniform cells as the correctness matrix; warmup 10 + 50 iter median via
CUDA events. Writes CSV to --outdir/bench_<tag>.csv.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import torch

FI_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(FI_ROOT / "tests" / "grouped_mm"))

import test_cute_sm120_mxfp8 as H  # noqa: E402

from flashinfer.grouped_mm import moe_gemm_mxfp8_nt_groupwise  # noqa: E402

WARMUP = 10
ITERS = 50


def bench_cell(m_per_expert, num_groups, n, k, k_gran):
    torch.random.manual_seed(0)
    token_num = m_per_expert * num_groups
    a = torch.randn((token_num, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((num_groups, n, k), dtype=torch.bfloat16, device="cuda") / math.sqrt(k)
    m_indptr = torch.tensor(
        [i * m_per_expert for i in range(num_groups + 1)], dtype=torch.int32, device="cuda"
    )
    a_fp8, a_sf = H.per_token_cast_to_mxfp8_for_moe_gemm(a, m_indptr, gran_k=k_gran)
    b_fp8_list, b_sf_list = [], []
    for i in range(num_groups):
        b_i_fp8, b_i_sf = H.per_block_cast_to_fp8(b[i], use_ue8m0=True, gran_k=k_gran)
        b_fp8_list.append(b_i_fp8)
        b_sf_list.append(b_i_sf)
    b_fp8 = torch.stack(b_fp8_list, dim=0)
    b_sf = H.transform_sf_into_required_layout(
        torch.stack(b_sf_list, dim=0), mn=n, k=k, recipe=(k_gran, k_gran),
        num_groups=num_groups, is_sfa=False,
    )
    out = torch.empty(token_num, n, dtype=torch.bfloat16, device="cuda")

    def call():
        moe_gemm_mxfp8_nt_groupwise(
            a_fp8, b_fp8, a_sf, b_sf, m_indptr,
            scale_granularity_mnk=(1, 1, k_gran), out=out,
        )

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
        times.append(s.elapsed_time(e) * 1e3)  # us
    times.sort()
    return times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", required=True, help="pre_sync / post_sync")
    ap.add_argument("--only", default=None, help="E,m_pe,N,K single-cell filter")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    only = tuple(int(x) for x in args.only.split(",")) if args.only else None
    rows = []
    for n, k in [(4096, 7168), (7168, 4096)]:
        for e in (4, 8):
            for m_pe in (1, 4, 8, 16, 192, 256, 1024):
                if only and (e, m_pe, n, k) != only:
                    continue
                us = bench_cell(m_pe, e, n, k, 128)
                rows.append([e, m_pe, n, k, 128, f"{us:.3f}"])
                print(rows[-1])

    out_csv = outdir / f"bench_{args.tag}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["E", "m_pe", "N", "K", "granK", "median_us"])
        w.writerows(rows)
    print(f"[done] {len(rows)} cells -> {out_csv}")


if __name__ == "__main__":
    main()
