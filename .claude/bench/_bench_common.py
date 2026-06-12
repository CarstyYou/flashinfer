"""Bench common utils for grouped GEMM perf sweep.

Shared by:
- bench_grouped_gemm_fixed.py  (PM PR scope: fc1/fc2 + 8 m_pe + 4 backend × 2 granK)
- bench_moe_gemm_routing.py    (future: routing + 3 customer × 4 scenario × 10 batch)

Backends (4) and caller-side padding contract:
- cute_sm120  flashinfer.grouped_mm.moe_gemm_mxfp8_nt_groupwise (ZeroPadding):
              kernel handles internal pad — caller passes m_pe as-is, no caller padding.
- cudnn       flashinfer.grouped_mm.grouped_mm_mxfp8 (granK=32 only):
              caller pads each expert's M to multiple of 16 (kernel TileM atom; m_pe < 16
              triggers cudaErrorIllegalInstruction).
- dg          deep_gemm.m_grouped_fp8_gemm_nt_contiguous:
              caller pads each expert's M to deep_gemm.get_mk_alignment_for_contiguous_layout().
              Padded rows marked via grouped_layout = -1.
- cutlass     custom CUTLASS SM120 grouped GEMM, 1D per-token N=1 scale
              (.claude/bench/cutlass_grouped_gemm_sm120.cu, JIT-compiled via
              torch.utils.cpp_extension; granK ∈ {32, 128} via 2 template variants):
              no caller padding — kernel accepts arbitrary m_pe.

TFLOPS dropped per PM (only t_us reported). CSV column `m_pe_padded` records actual
per-expert M after caller padding (= m_pe for zero-caller-pad backends).

Timing loop mirrors 6KD bench_real_dist.py: l2-flush per iter, 10 warmup + 50 rep median.
"""

import csv
import math
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch

from flashinfer.deep_gemm import get_col_major_tma_aligned_packed_tensor
from flashinfer.grouped_mm import grouped_mm_mxfp8, moe_gemm_mxfp8_nt_groupwise


# =============================================================================
# §1 L2-flush + 50-rep median CUDA-event timing (mirror 6KD bench_real_dist.py)
# =============================================================================

_FLUSH_BUF: Optional[torch.Tensor] = None


def _flush_buf() -> torch.Tensor:
    global _FLUSH_BUF
    if _FLUSH_BUF is None:
        _FLUSH_BUF = torch.empty(int(8e9 // 4), dtype=torch.int, device="cuda")
    return _FLUSH_BUF


def bench_median_us(fn: Callable[[], None], warmup: int = 10, rep: int = 50) -> float:
    buf = _flush_buf()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        buf.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    times.sort()
    return times[len(times) // 2]


# =============================================================================
# §2 MXFP8 quant helpers (Python eager; copied from
#     flashinfer/tests/grouped_mm/test_cute_sm120_mxfp8.py, which itself copied
#     from DeepGEMM test utils).
# =============================================================================


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _align(x: int, y: int) -> int:
    return _ceil_div(x, y) * y


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def _pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float and x.size(-1) % 4 == 0
    return (x.view(torch.int) >> 23).to(torch.uint8).view(torch.int)


def _per_token_cast_to_fp8(
    x: torch.Tensor,
    use_ue8m0: bool,
    gran_k: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    padded_n = _align(n, gran_k)
    x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_fp8 = (
        (x_view * (1.0 / sf.unsqueeze(2)))
        .to(torch.float8_e4m3fn)
        .view(m, padded_n)[:, :n]
        .contiguous()
    )
    return x_fp8, sf


def _per_block_cast_to_fp8(
    x: torch.Tensor, use_ue8m0: bool, gran_k: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (_align(m, gran_k), _align(n, gran_k)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, gran_k, x_padded.size(1) // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return (
        x_scaled.view_as(x_padded)[:m, :n].contiguous(),
        sf.view(x_view.size(0), x_view.size(2)),
    )


def _compute_padded_offset(offset: int, problem_idx: int, alignment: int) -> int:
    return (offset + problem_idx * (alignment - 1)) // alignment * alignment


def _per_token_cast_to_mxfp8_for_moe_gemm(
    x: torch.Tensor, token_offset: torch.Tensor, gran_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    assert token_offset.dtype == torch.int32
    assert token_offset[0].item() == 0

    token_num, k = x.shape
    E = token_offset.numel() - 1
    PACK_NSF = 4
    PACK_NK = gran_k * PACK_NSF
    m_padded = _compute_padded_offset(token_num, E, alignment=PACK_NSF)
    k_align = (k + PACK_NK - 1) // PACK_NK

    fp8_output = torch.empty((token_num, k), dtype=torch.float8_e4m3fn, device=x.device)
    sf_int = torch.zeros((k_align, m_padded), dtype=torch.int32, device=x.device)

    for i in range(E):
        start = token_offset[i].item()
        end = token_offset[i + 1].item()
        if start == end:
            continue
        actual_m = end - start
        expert_fp8, expert_sf_ue8m0 = _per_token_cast_to_fp8(
            x[start:end], use_ue8m0=True, gran_k=gran_k
        )
        n_sf = _ceil_div(k, gran_k)
        n_sf_padded = _align(n_sf, PACK_NSF)
        if n_sf_padded != n_sf:
            pad = torch.zeros(
                (actual_m, n_sf_padded - n_sf), dtype=torch.float32, device=x.device
            )
            expert_sf_ue8m0 = torch.cat([expert_sf_ue8m0, pad], dim=1)
        packed_int = _pack_ue8m0_to_int(expert_sf_ue8m0)
        fp8_output[start:end] = expert_fp8
        padded_offset = _compute_padded_offset(start, i, alignment=PACK_NSF)
        sf_int[:, padded_offset : padded_offset + actual_m] = packed_int.t()

    return fp8_output, sf_int.transpose(0, 1)


def _ue8m0_float_to_uint8(sf_float: torch.Tensor) -> torch.Tensor:
    return ((sf_float.view(torch.int) >> 23) & 0xFF).to(torch.uint8)


# =============================================================================
# §3 Fixed config (PM PR scope)
# =============================================================================

LAYER_SHAPES = {
    "fc1": (4096, 7168),  # (N, K)
    "fc2": (7168, 4096),
}
NUM_EXPERT = 8
M_PE_SWEEP = [1, 4, 8, 16, 192, 256, 1024, 4096]
GRAN_K_SWEEP = [32, 128]


def make_base_inputs(
    num_expert: int, m_pe: int, n: int, k: int, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    cum_m = num_expert * m_pe
    a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
    b_bf16 = torch.randn(
        num_expert, n, k, dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)
    m_indptr = torch.tensor(
        [i * m_pe for i in range(num_expert + 1)],
        dtype=torch.int32,
        device="cuda",
    )
    return a_bf16, b_bf16, m_indptr


def _pad_a_per_expert(
    a_bf16: torch.Tensor, num_expert: int, m_pe: int, m_pe_padded: int
) -> torch.Tensor:
    """Pad each expert's M slice from m_pe to m_pe_padded with zero rows."""
    if m_pe_padded == m_pe:
        return a_bf16
    k = a_bf16.shape[1]
    a_out = torch.zeros(
        num_expert * m_pe_padded, k, dtype=a_bf16.dtype, device=a_bf16.device
    )
    for i in range(num_expert):
        a_out[i * m_pe_padded : i * m_pe_padded + m_pe] = a_bf16[
            i * m_pe : (i + 1) * m_pe
        ]
    return a_out


def _padded_indptr(num_expert: int, m_pe_padded: int, device) -> torch.Tensor:
    return torch.tensor(
        [i * m_pe_padded for i in range(num_expert + 1)],
        dtype=torch.int32,
        device=device,
    )


# =============================================================================
# §4 Per-backend input prep + dispatch
# =============================================================================

# ---- cute SM120 ZeroPadding -------------------------------------------------


def prep_cute_sm120(a_bf16, b_bf16, m_indptr, m_pe: int, gran_k: int) -> dict:
    # No caller padding — kernel handles ZeroPadding internally.
    a_fp8, a_sf = _per_token_cast_to_mxfp8_for_moe_gemm(a_bf16, m_indptr, gran_k=gran_k)
    num_groups, n, k = b_bf16.shape
    b_fp8_list, b_sf_list = [], []
    for i in range(num_groups):
        bi_fp8, bi_sf = _per_block_cast_to_fp8(b_bf16[i], use_ue8m0=True, gran_k=gran_k)
        b_fp8_list.append(bi_fp8)
        b_sf_list.append(bi_sf)
    b_fp8 = torch.stack(b_fp8_list, dim=0)
    b_sf_block = torch.stack(b_sf_list, dim=0)
    b_sf_per_row = b_sf_block.repeat_interleave(gran_k, dim=1)
    b_sf = get_col_major_tma_aligned_packed_tensor(b_sf_per_row)
    return dict(
        a=a_fp8,
        b=b_fp8,
        a_scale=a_sf,
        b_scale=b_sf,
        m_indptr=m_indptr,
        gran_k=gran_k,
        m_pe_padded=m_pe,
    )


def call_cute_sm120(inputs: dict) -> torch.Tensor:
    return moe_gemm_mxfp8_nt_groupwise(
        inputs["a"],
        inputs["b"],
        inputs["a_scale"],
        inputs["b_scale"],
        inputs["m_indptr"],
        scale_granularity_mnk=(1, 1, inputs["gran_k"]),
        out_dtype=torch.bfloat16,
    )


# ---- cudnn MXFP8 (granK=32 only) --------------------------------------------


def prep_cudnn(a_bf16, b_bf16, m_indptr, m_pe: int, gran_k: int) -> dict:
    # cuDNN MoE grouped GEMM requires per-expert M aligned to 16 (kernel TileM atom).
    if gran_k != 32:
        raise ValueError(f"cudnn MXFP8 only supports gran_k=32; got {gran_k}")

    num_groups, n, k = b_bf16.shape
    m_pe_padded = _align(m_pe, 16)
    a_bf16_padded = _pad_a_per_expert(a_bf16, num_groups, m_pe, m_pe_padded)
    m_indptr_padded = _padded_indptr(num_groups, m_pe_padded, a_bf16.device)

    a_fp8, a_sf_float = _per_token_cast_to_fp8(
        a_bf16_padded, use_ue8m0=True, gran_k=gran_k
    )
    a_sf_uint8 = _ue8m0_float_to_uint8(a_sf_float).contiguous()

    b_fp8_list, b_sf_list = [], []
    for i in range(num_groups):
        bi_fp8, bi_sf_block = _per_block_cast_to_fp8(
            b_bf16[i], use_ue8m0=True, gran_k=gran_k
        )
        bi_sf_per_row = bi_sf_block.repeat_interleave(gran_k, dim=0)
        b_fp8_list.append(bi_fp8)
        b_sf_list.append(_ue8m0_float_to_uint8(bi_sf_per_row))
    b_fp8 = torch.stack(b_fp8_list, dim=0)
    b_sf_uint8 = torch.stack(b_sf_list, dim=0).contiguous()

    return dict(
        a=a_fp8,
        b=b_fp8,
        a_scale=a_sf_uint8,
        b_scale=b_sf_uint8,
        m_indptr=m_indptr_padded,
        m_pe_padded=m_pe_padded,
    )


def call_cudnn(inputs: dict) -> torch.Tensor:
    return grouped_mm_mxfp8(
        inputs["a"],
        inputs["b"],
        inputs["a_scale"],
        inputs["b_scale"],
        inputs["m_indptr"],
        out_dtype=torch.bfloat16,
        backend="cudnn",
    )


# ---- DeepGEMM contiguous ----------------------------------------------------


def prep_dg(a_bf16, b_bf16, m_indptr, m_pe: int, gran_k: int) -> dict:
    """DG signature: m_grouped_fp8_gemm_nt_contiguous(a, b, d, grouped_layout, recipe=...)
    where a = (a_fp8, a_sf), b = (b_fp8, b_sf), grouped_layout = per-token expert ID int32.

    Caller padding: each expert's M aligned to deep_gemm.get_mk_alignment_for_contiguous_layout().
    Padded rows marked grouped_layout = -1.
    """
    import deep_gemm

    num_groups, n, k = b_bf16.shape
    out_dtype = torch.bfloat16

    alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    m_pe_padded = _align(m_pe, alignment)

    a_bf16_padded = _pad_a_per_expert(a_bf16, num_groups, m_pe, m_pe_padded)
    cum_m_padded = num_groups * m_pe_padded

    a_fp8, a_sf = _per_token_cast_to_fp8(a_bf16_padded, use_ue8m0=True, gran_k=gran_k)
    b_fp8_list, b_sf_list = [], []
    for i in range(num_groups):
        bi_fp8, bi_sf = _per_token_cast_to_fp8(b_bf16[i], use_ue8m0=True, gran_k=gran_k)
        b_fp8_list.append(bi_fp8)
        b_sf_list.append(bi_sf)
    b_fp8 = torch.stack(b_fp8_list, dim=0)
    b_sf = torch.stack(b_sf_list, dim=0)

    grouped_layout = torch.full((cum_m_padded,), -1, dtype=torch.int32, device="cuda")
    for i in range(num_groups):
        start = i * m_pe_padded
        grouped_layout[start : start + m_pe] = i

    d = torch.empty(cum_m_padded, n, dtype=out_dtype, device="cuda")
    return dict(
        a_pair=(a_fp8, a_sf),
        b_pair=(b_fp8, b_sf),
        d=d,
        grouped_layout=grouped_layout,
        gran_k=gran_k,
        m_pe_padded=m_pe_padded,
    )


def call_dg(inputs: dict) -> torch.Tensor:
    import deep_gemm

    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        inputs["a_pair"],
        inputs["b_pair"],
        inputs["d"],
        inputs["grouped_layout"],
        recipe=(1, 1, inputs["gran_k"]),
    )
    return inputs["d"]


# ---- cutlass FP8 groupwise (m_indptr % 4 == 0) ------------------------------


def prep_cutlass(a_bf16, b_bf16, m_indptr, m_pe: int, gran_k: int) -> dict:
    """Custom CUTLASS SM120 grouped GEMM wrapper (1D per-token N=1 scale).

    No caller padding required — kernel accepts arbitrary m_pe.
    a_scale layout: (E, k_blocks, m_pe) contiguous, within-group M-fastest.
    b_scale layout: (E, k_blocks, n)   contiguous, within-group N-fastest.
    """
    num_groups, n, k = b_bf16.shape
    k_blocks = k // gran_k

    a_fp8, a_sf_fp32 = _per_token_cast_to_fp8(a_bf16, use_ue8m0=False, gran_k=gran_k)
    a_sf_per_group = a_sf_fp32.view(num_groups, m_pe, k_blocks)
    a_scale = a_sf_per_group.transpose(1, 2).contiguous()

    b_fp8_list, b_sf_list = [], []
    for i in range(num_groups):
        bi_fp8, bi_sf_fp32 = _per_token_cast_to_fp8(
            b_bf16[i], use_ue8m0=False, gran_k=gran_k
        )
        b_fp8_list.append(bi_fp8)
        b_sf_list.append(bi_sf_fp32)
    b_fp8 = torch.stack(b_fp8_list, dim=0)
    b_sf_per_row = torch.stack(b_sf_list, dim=0)
    b_scale = b_sf_per_row.transpose(1, 2).contiguous()

    return dict(
        a=a_fp8,
        b=b_fp8,
        a_scale=a_scale,
        b_scale=b_scale,
        m_indptr=m_indptr,
        gran_k=gran_k,
        m_pe_padded=m_pe,
    )


def call_cutlass(inputs: dict) -> torch.Tensor:
    from _cutlass_loader import get_cutlass_module

    mod = get_cutlass_module()
    gran_k = inputs["gran_k"]
    # TileM=128 Cooperative across all m_pe; TileM=64 Pingpong variant compiled but
    # 5K Pro smoke showed it 14% slower at m_pe=1 (launch overhead > tile-waste savings).
    fn = getattr(mod, f"run_m128_k{gran_k}")
    return fn(
        inputs["a"],
        inputs["b"],
        inputs["a_scale"],
        inputs["b_scale"],
        inputs["m_indptr"],
    )


# =============================================================================
# §5 Backend registry + cell-skip rules
# =============================================================================

BACKENDS = {
    "cute_sm120": dict(
        prep=prep_cute_sm120,
        call=call_cute_sm120,
        supports_granK={32, 128},
        display="cute SM120 moeGemm",
    ),
    "cudnn": dict(
        prep=prep_cudnn,
        call=call_cudnn,
        supports_granK={32},
        display="cuDNN MXFP8",
    ),
    "dg": dict(
        prep=prep_dg,
        call=call_dg,
        supports_granK={32, 128},
        display="DeepGEMM contiguous",
    ),
    "cutlass": dict(
        prep=prep_cutlass,
        call=call_cutlass,
        supports_granK={32, 128},
        display="cuTLASS FP8 groupwise",
    ),
}


def cell_supported(backend: str, m_pe: int, gran_k: int) -> bool:
    spec = BACKENDS[backend]
    if gran_k not in spec["supports_granK"]:
        return False
    return True


# =============================================================================
# §6 CSV writer + TFLOPS calculator
# =============================================================================

CSV_COLS = [
    "layer",
    "n",
    "k",
    "num_expert",
    "m_pe",
    "m_pe_padded",
    "total_rows",
    "backend",
    "gran_k",
    "t_us",
]


def write_row(csv_path: Path, row: dict) -> None:
    new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if new:
            w.writeheader()
        w.writerow(row)
