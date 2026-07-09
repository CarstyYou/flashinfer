"""task_04 sub-task 0: grouped_mm_fp8 (cudnn) SM120 smoke.

Checks: (1) basic cell computes correctly vs raw fp8 matmul ref (alpha=None);
(2) small m_pe / m_pe=1 / empty expert acceptance.
"""

import math
import sys
from pathlib import Path

import torch

FI_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(FI_ROOT))

from flashinfer.grouped_mm import grouped_mm_fp8  # noqa: E402


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum().item()
    return 0.0 if denom == 0 else 1.0 - 2.0 * (x * y).sum().item() / denom


def run_cell(m_per_expert_list, n, k):
    torch.random.manual_seed(0)
    e = len(m_per_expert_list)
    offsets = [0]
    for m_pe in m_per_expert_list:
        offsets.append(offsets[-1] + m_pe)
    total = offsets[-1]
    a = (torch.randn(total, k, device="cuda") / math.sqrt(k)).to(torch.float8_e4m3fn)
    b = (torch.randn(e, n, k, device="cuda") / math.sqrt(k)).to(torch.float8_e4m3fn)
    m_indptr = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    out = grouped_mm_fp8(a, b, m_indptr, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()

    ref = torch.zeros(total, n, dtype=torch.bfloat16, device="cuda")
    for i in range(e):
        s, t = offsets[i], offsets[i + 1]
        if s < t:
            ref[s:t] = (a[s:t].float() @ b[i].float().t()).to(torch.bfloat16)
    return calc_diff(out.float(), ref.float())


def main():
    cases = [
        ("basic_4x8", [8] * 4, 256, 512),
        ("m_pe_1", [1] * 4, 256, 512),
        ("uneven", [1, 5, 0, 26], 256, 512),
        ("fc1_small", [8] * 4, 4096, 7168),
    ]
    for label, mlist, n, k in cases:
        try:
            diff = run_cell(mlist, n, k)
            print(f"{label}: calc_diff={diff:.6e} {'PASS' if diff < 1e-3 else 'FAIL'}")
        except Exception as exc:  # noqa: BLE001
            print(f"{label}: REJECTED {type(exc).__name__}: {str(exc)[:150]}")


if __name__ == "__main__":
    main()
