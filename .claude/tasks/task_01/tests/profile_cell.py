"""Single-cell driver for NCU capture: MXFP8 moe cell (E, m_pe, N, K)."""

import argparse
import math
import sys
from pathlib import Path

import torch

FI_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(FI_ROOT / "tests" / "grouped_mm"))

import test_cute_sm120_mxfp8 as H  # noqa: E402

from flashinfer.grouped_mm import moe_gemm_mxfp8_nt_groupwise  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", type=int, default=4)
    ap.add_argument("--m-pe", type=int, default=8)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=7168)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    torch.random.manual_seed(0)
    e, m_pe, n, k = args.e, args.m_pe, args.n, args.k
    token_num = m_pe * e
    a = torch.randn((token_num, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((e, n, k), dtype=torch.bfloat16, device="cuda") / math.sqrt(k)
    m_indptr = torch.tensor(
        [i * m_pe for i in range(e + 1)], dtype=torch.int32, device="cuda"
    )
    a_fp8, a_sf = H.per_token_cast_to_mxfp8_for_moe_gemm(a, m_indptr, gran_k=128)
    b_fp8_list, b_sf_list = [], []
    for i in range(e):
        b_i_fp8, b_i_sf = H.per_block_cast_to_fp8(b[i], use_ue8m0=True, gran_k=128)
        b_fp8_list.append(b_i_fp8)
        b_sf_list.append(b_i_sf)
    b_fp8 = torch.stack(b_fp8_list, dim=0)
    b_sf = H.transform_sf_into_required_layout(
        torch.stack(b_sf_list, dim=0),
        mn=n,
        k=k,
        recipe=(128, 128),
        num_groups=e,
        is_sfa=False,
    )
    out = torch.empty(token_num, n, dtype=torch.bfloat16, device="cuda")
    for _ in range(args.iters):
        moe_gemm_mxfp8_nt_groupwise(
            a_fp8,
            b_fp8,
            a_sf,
            b_sf,
            m_indptr,
            scale_granularity_mnk=(1, 1, 128),
            out=out,
        )
    torch.cuda.synchronize()
    print("done")


if __name__ == "__main__":
    main()
