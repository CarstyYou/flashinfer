"""task_01 MXFP8 correctness cells (pre/post kernel-tree sync).

Reuses input-construction helpers from the upstream test
tests/grouped_mm/test_cute_sm120_mxfp8.py. Metric: calc_diff < 1e-3 (plus
cos_sim recorded). Writes one CSV row per cell to --outdir.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

FI_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(FI_ROOT / "tests" / "grouped_mm"))

import test_cute_sm120_mxfp8 as H  # noqa: E402

from flashinfer.grouped_mm import moe_gemm_mxfp8_nt_groupwise  # noqa: E402


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum().item()
    if denom == 0:
        return 0.0
    return 1.0 - 2.0 * (x * y).sum().item() / denom


def run_cell(m_per_expert_list, n, k, k_gran):
    torch.random.manual_seed(0)
    num_groups = len(m_per_expert_list)
    offsets = [0]
    for m_pe in m_per_expert_list:
        offsets.append(offsets[-1] + m_pe)
    token_num = offsets[-1]

    a = torch.randn((token_num, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((num_groups, n, k), dtype=torch.bfloat16, device="cuda") / math.sqrt(k)
    m_indptr = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    a_fp8, a_sf = H.per_token_cast_to_mxfp8_for_moe_gemm(a, m_indptr, gran_k=k_gran)

    a_deq = torch.zeros_like(a)
    for j in range(num_groups):
        start, end = offsets[j], offsets[j + 1]
        if start == end:
            continue
        a_j_fp8, a_j_sf = H.per_token_cast_to_fp8(a[start:end], use_ue8m0=True, gran_k=k_gran)
        a_deq[start:end] = H.per_token_dequant_from_fp8(a_j_fp8, a_j_sf, gran_k=k_gran, dtype=a.dtype)

    b_fp8_list, b_sf_list = [], []
    for i in range(num_groups):
        b_i_fp8, b_i_sf = H.per_block_cast_to_fp8(b[i], use_ue8m0=True, gran_k=k_gran)
        b_fp8_list.append(b_i_fp8)
        b_sf_list.append(b_i_sf)
    b_fp8 = torch.stack(b_fp8_list, dim=0)
    b_sf_ue8m0 = torch.stack(b_sf_list, dim=0)
    b_sf = H.transform_sf_into_required_layout(
        b_sf_ue8m0, mn=n, k=k, recipe=(k_gran, k_gran), num_groups=num_groups, is_sfa=False
    )
    b_deq = H.per_block_dequant_from_fp8(b_fp8, b_sf_ue8m0, gran_k=k_gran, dtype=b.dtype)

    ref = torch.zeros(token_num, n, dtype=torch.bfloat16, device="cuda")
    for j in range(num_groups):
        start, end = offsets[j], offsets[j + 1]
        if start == end:
            continue
        ref[start:end] = (a_deq[start:end] @ b_deq[j].t()).to(torch.bfloat16)

    out = moe_gemm_mxfp8_nt_groupwise(
        a_fp8, b_fp8, a_sf, b_sf, m_indptr,
        scale_granularity_mnk=(1, 1, k_gran), out_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    diff = calc_diff(out.reshape(-1).float(), ref.reshape(-1).float())
    cos = F.cosine_similarity(out.reshape(-1).float(), ref.reshape(-1).float(), dim=0).item()
    return diff, cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", required=True, help="pre_sync / post_sync")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    shapes = [(4096, 7168), (7168, 4096)]
    uniform_cells = [
        (e, m_pe) for e in (4, 8) for m_pe in (1, 4, 8, 16, 192, 256, 1024)
    ]
    rows = []
    failures = 0
    for n, k in shapes:
        for e, m_pe in uniform_cells:
            for k_gran in (128, 32) if m_pe in (8, 256) else ((128,)):
                diff, cos = run_cell([m_pe] * e, n, k, k_gran)
                ok = diff < 1e-3
                failures += 0 if ok else 1
                rows.append([e, f"uniform_{m_pe}", n, k, k_gran, f"{diff:.6e}", f"{cos:.6f}", "PASS" if ok else "FAIL"])
                print(rows[-1])
        for label, mlist in [
            ("uneven", [1, 1, 8, 16, 64, 128, 192, 256]),
            ("empty_expert", [0, 8, 0, 256, 16, 0, 1, 64]),
        ]:
            diff, cos = run_cell(mlist, n, k, 128)
            ok = diff < 1e-3
            failures += 0 if ok else 1
            rows.append([len(mlist), label, n, k, 128, f"{diff:.6e}", f"{cos:.6f}", "PASS" if ok else "FAIL"])
            print(rows[-1])

    out_csv = outdir / f"correctness_{args.tag}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["E", "m_pe", "N", "K", "granK", "calc_diff", "cos_sim", "verdict"])
        w.writerows(rows)
    print(f"[done] {len(rows)} cells, {failures} FAIL -> {out_csv}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
