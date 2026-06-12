"""JIT-compile cutlass_grouped_gemm_sm120.cu via torch.utils.cpp_extension.

Exposes:
    get_cutlass_module() -> module with .run_k128 and .run_k32 entries.

First call triggers nvcc compile (~2 min cold); subsequent calls use cached build_dir.
"""

import functools
import os
from pathlib import Path

import torch.utils.cpp_extension

_BENCH_DIR = Path(__file__).parent
_FLASHINFER_ROOT = _BENCH_DIR.parent.parent
_CUTLASS_INCLUDE = _FLASHINFER_ROOT / "3rdparty" / "cutlass" / "include"
_CUTLASS_UTIL_INCLUDE = (
    _FLASHINFER_ROOT / "3rdparty" / "cutlass" / "tools" / "util" / "include"
)
_SOURCE = _BENCH_DIR / "cutlass_grouped_gemm_sm120.cu"

_BUILD_DIR = Path(
    os.environ.get(
        "CUTLASS_BENCH_BUILD_DIR",
        str(_BENCH_DIR / ".build_cutlass_sm120"),
    )
)


@functools.cache
def get_cutlass_module():
    assert _SOURCE.exists(), f"missing source: {_SOURCE}"
    assert _CUTLASS_INCLUDE.exists(), f"missing cutlass include: {_CUTLASS_INCLUDE}"
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)

    return torch.utils.cpp_extension.load(
        name="cutlass_grouped_gemm_sm120",
        sources=[str(_SOURCE)],
        extra_include_paths=[
            str(_CUTLASS_INCLUDE),
            str(_CUTLASS_UTIL_INCLUDE),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--extended-lambda",
            "-gencode=arch=compute_120a,code=sm_120a",
            "-gencode=arch=compute_121a,code=sm_121a",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        ],
        build_directory=str(_BUILD_DIR),
        verbose=True,
    )
