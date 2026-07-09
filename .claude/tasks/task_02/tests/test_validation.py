"""task_02 FP8 moe_gemm validation cells: every illegal input must be rejected."""

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

sys.path.insert(0, str(Path(__file__).parent))
from test_fp8_correctness import build_zero_padding_sfa  # noqa: E402


def make_valid(e=4, m_pe=8, n=256, k=512):
    torch.random.manual_seed(0)
    offsets = [i * m_pe for i in range(e + 1)]
    a = torch.randn((offsets[-1], k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((e, n, k), dtype=torch.bfloat16, device="cuda") / math.sqrt(k)
    m_indptr = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    a_fp8, sf_a = per_token_cast_to_fp8(a)
    a_scale = build_zero_padding_sfa(sf_a, offsets)
    b_fp8_list, b_sf_list = [], []
    for i in range(e):
        f, s = per_block_cast_to_fp8(b[i])
        b_fp8_list.append(f)
        b_sf_list.append(s)
    b_fp8 = torch.stack(b_fp8_list, dim=0)
    b_scale = torch.stack(b_sf_list, dim=0).transpose(-1, -2).contiguous()
    return a_fp8, b_fp8, a_scale, b_scale, m_indptr


def expect_raise(label, exc_types, fn):
    try:
        fn()
    except exc_types as exc:
        print(f"PASS {label}: {type(exc).__name__}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {label}: wrong exception {type(exc).__name__}: {exc}")
        return False
    print(f"FAIL {label}: no exception raised")
    return False


def main():
    a, b, a_scale, b_scale, m_indptr = make_valid()
    ok = True

    out = moe_gemm_fp8_nt_groupwise(a, b, a_scale, b_scale, m_indptr)
    assert out.shape == (a.shape[0], b.shape[1]) and out.dtype == torch.bfloat16
    print("PASS baseline: valid inputs accepted")

    ok &= expect_raise(
        "bad granularity",
        ValueError,
        lambda: moe_gemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, m_indptr, scale_granularity_mnk=(1, 1, 128)
        ),
    )
    ok &= expect_raise(
        "bad scale_major_mode",
        ValueError,
        lambda: moe_gemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, m_indptr, scale_major_mode="K"
        ),
    )
    ok &= expect_raise(
        "bad backend",
        NotImplementedError,
        lambda: moe_gemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, m_indptr, backend="cutlass"
        ),
    )
    ok &= expect_raise(
        "bad out_dtype",
        NotImplementedError,
        lambda: moe_gemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, m_indptr, out_dtype=torch.float16
        ),
    )
    ok &= expect_raise(
        "bad m_indptr size",
        ValueError,
        lambda: moe_gemm_fp8_nt_groupwise(a, b, a_scale, b_scale, m_indptr[:-1]),
    )
    ok &= expect_raise(
        "bad m_indptr dtype",
        ValueError,
        lambda: moe_gemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, m_indptr.to(torch.int64)
        ),
    )
    ok &= expect_raise(
        "SFA missing padding",
        Exception,
        lambda: moe_gemm_fp8_nt_groupwise(
            a, b, a_scale[:, : a.shape[0]].contiguous(), b_scale, m_indptr
        ),
    )
    ok &= expect_raise(
        "SFA non-contiguous",
        Exception,
        lambda: moe_gemm_fp8_nt_groupwise(
            a, b, a_scale.t().contiguous().t(), b_scale, m_indptr
        ),
    )
    ok &= expect_raise(
        "SFA K-major shape",
        Exception,
        lambda: moe_gemm_fp8_nt_groupwise(
            a, b, a_scale.t().contiguous(), b_scale, m_indptr
        ),
    )
    ok &= expect_raise(
        "SFB bad shape",
        Exception,
        lambda: moe_gemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale.transpose(-1, -2).contiguous(), m_indptr
        ),
    )
    ok &= expect_raise(
        "SFB non-contiguous",
        Exception,
        lambda: moe_gemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale.transpose(-1, -2), m_indptr
        ),
    )
    ok &= expect_raise(
        "bad a dtype",
        Exception,
        lambda: moe_gemm_fp8_nt_groupwise(
            a.view(torch.uint8), b, a_scale, b_scale, m_indptr
        ),
    )

    print("[done]", "ALL PASS" if ok else "HAS FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
