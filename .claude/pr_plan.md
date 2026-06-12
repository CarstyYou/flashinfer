# PR #3562 — MXFP8 MoE GEMM (cute SM120) — Final state

URL: https://github.com/flashinfer-ai/flashinfer/pull/3562
Branch: `xiy/sm120_group_gemm_mxfp8` (HEAD: `4203d856`)
Base: `flashinfer-ai/main`
Status: PM (`@samuellees`) 满意当前改动, **不需要阶段二**, 待 maintainer merge。

## 最终文件 layout

```
csrc/cute_sm120_mxfp8_groupwise/                              [top-level kernel infra, backend-first naming]
├── cute_sm120_mxfp8_runner.h                                 [Runner interface + class declaration]
├── cute_sm120_mxfp8_runner.cu                                [Runner class template impl]
├── cute_sm120_mxfp8_op.cu                                    [op-layer wrapper (CutlassMXFP8GroupwiseMoeGEMMSM120)]
├── cute_sm120_mxfp8_op_jit_binding.cu                        [TVM-FFI export `moe_gemm_mxfp8_nt_groupwise`]
└── sm120_blockscaled/                                        [kernel template helpers, CUTLASS-style naming]
    ├── builder.cuh, kernel_impl.cuh, launch.cuh
    ├── math.cuh, scheduler.cuh, utils.cuh

flashinfer/grouped_mm/cute_sm120_mxfp8_groupwise/             [Python sub-pkg, mirror cudnn/ pattern]
├── __init__.py                                               [re-export `moe_gemm_mxfp8_nt_groupwise`]
└── core.py                                                   [Python entry impl + accessor + _check_* helpers]

flashinfer/jit/cute_sm120_mxfp8_groupwise.py                  [Single-file JIT spec (no jit/grouped_mm/ intermediate dir)]
flashinfer/grouped_mm/__init__.py                             [+1 line: re-export `moe_gemm_mxfp8_nt_groupwise`]
flashinfer/aot.py                                             [+1 line: import path 更新]

docs/api/grouped_mm.rst                                       [+ section "MXFP8 MoE GEMM (SM120 cute backend)"]
tests/grouped_mm/test_cute_sm120_mxfp8.py                     [32 cells, uses DG-copied helpers]
```

## Public API

| Layer | 名字 |
|---|---|
| Python entry | `flashinfer.grouped_mm.moe_gemm_mxfp8_nt_groupwise(a, b, a_scale, b_scale, m_indptr, scale_granularity_mnk, scale_major_mode="MN", backend="cute", out=None, out_dtype=None)` |
| C++ Runner class | `flashinfer::gemm::mxfp8_cute_sm120::CuteSm120Mxfp8GemmRunner{,Interface}` (backend-first ordering) |
| C++ op-layer wrapper | `CutlassMXFP8GroupwiseMoeGEMMSM120` |
| C++ FFI export 名 | `moe_gemm_mxfp8_nt_groupwise` |
| 内部 C++ kernel-level naming (保留, layout-detail) | `quantize_mxfp8_zero_padding{,_impl}` namespace function + `quantize_mxfp8_zero_padding_kernel_sm120` `__global__` |

## 关键 design 决策

| Q# | 决策 |
|---|---|
| Naming | `moe_gemm_*` (MoE-native, no token padding) vs `group_gemm_*` family (pads tokens to grouped shape); 子目录 / 模块名 用 backend-first ordering (`cute_sm120_mxfp8_groupwise`) |
| csrc layout | `csrc/cute_sm120_mxfp8_groupwise/` (top-level under csrc/, **不放 nv_internal/**, kernel infra 通用未来跨 GEMM variants 复用) |
| Include style | 全模块前缀 (`#include "cute_sm120_mxfp8_groupwise/..."`); JIT spec `extra_include_paths=[csrc]` |
| Python sub-pkg | `flashinfer/grouped_mm/cute_sm120_mxfp8_groupwise/{__init__.py, core.py}` mirror `cudnn/` pattern; sample backend 跟 cudnn 同 sub-pkg 命名风格, CODEOWNERS routing 保留给 `flashinfer/grouped_mm/` team |
| JIT 层 | Single-file `flashinfer/jit/cute_sm120_mxfp8_groupwise.py` (drop `jit/grouped_mm/` intermediate dir) |
| C++ class 名 | `CuteSm120Mxfp8GemmRunner` (backend-first 跟 csrc dir 一致) |
| op-layer wrapper | `CutlassMXFP8GroupwiseMoeGEMMSM120` (acronym 全大写 MXFP8/GEMM, scaling scheme + mode + arch) |
| Dedicated quantize CUDA kernel | **删除** — PM 反馈客户 MoE pipeline 通常 own quantize step (custom kernel 或 DG helper chain); 只 expose GEMM entry |
| `flashinfer.quantization.mxfp8_layout.py` | **删除** — 4 helpers 跟 `flashinfer.quantization.mxfp8_quantize` 撞 namespace; test 内 inline 替代 (用 DG verbatim helpers) |
| Test attribution | Reference quantization helpers **`# COPIED FROM DeepGEMM`** verbatim (`per_token_cast_to_fp8`, `per_block_cast_to_fp8`, `pack_ue8m0_to_int`, `transform_sf_into_required_layout`, `ceil_to_ue8m0`, `align`, `ceil_div`); 减 flashinfer 跟 DG 的 code drift |

## Test setup

- 路径: `tests/grouped_mm/test_cute_sm120_mxfp8.py`
- 参数化: 32 cells = `num_groups ∈ {2,4} × rows_per_group ∈ {64,128} × (n,k) ∈ {(4096,7168),(7168,4096)} × k_gran ∈ {32,128} × is_weight_scale_float ∈ {True,False}`
- `is_weight_scale_float`:
  - `True` (默认): weight quantize 用 `per_block_cast_to_fp8(use_ue8m0=True)` → 直接 UE8M0 scale
  - `False` (客户 checkpoint scenario): weight quantize 用 `per_block_cast_to_fp8(use_ue8m0=False)` → FP32 scale → `per_block_resmooth_to_ue8m0` 转 UE8M0 (模拟 model weight load 后一次性 resmooth)
- Reference matmul: 用 fp32 sf 直接 broadcast dequant (skip int32 pack→unpack roundtrip)
- Accuracy threshold: `F.cosine_similarity > 0.99`
- Hardware skip: `is_sm120a_supported()` 用于非 SM120 环境

## Post-`4203d856` fixes (working tree, 待 incremental commit — new rule: 不 amend, fast-forward push)

| # | 改动 | 文件 | Reason |
|---|---|---|---|
| 1 | `compute_padded_offset(offset, problem_idx, alignment)` — 去掉 `alignment` default value (强制 explicit); docstring +1 行说明 `alignment = PACK_NSF` 来源 (col-major sf + 4 UE8M0 pack 1 int32 沿 M) | `tests/grouped_mm/test_cute_sm120_mxfp8.py` | xiy 反馈: `alignment=4` magic 来源不显眼, caller 应 explicit `PACK_NSF` 标 dependency |
| 2 | 2 caller (`m_padded = ...`, `padded_offset = ...`) 显式 pass `alignment=PACK_NSF` | 同上 | 同 #1 |
| 3 | `n_sf = k // gran_k` → `n_sf = ceil_div(k, gran_k)` (DG helper) | 同上 | xiy 反馈: k 非 `gran_k` align case 下 `k // gran_k` floor 跟 `per_token_cast_to_fp8` 内部 `align(k, gran_k) // gran_k` mismatch; test 当前 k 都 % gran_k == 0 不 trigger, 但 robustness fix |

**新 rule (`[[feedback_no_more_amend]]`)**: PR HEAD ≥ `4203d856` 后 **不 amend**, 改动作为 new commit on top, fast-forward push (不 force-push)。 上述 3 处 fix 待 xiy explicit 后 1 个新 commit ship。

## Verification (latest run, HEAD `4203d856`)

| 检查 | 结果 |
|---|---|
| Cold-cache pytest (32/32) | ✓ 154s (cold JIT) / 1.45s (cached) |
| Pre-commit (`clang-format`, `mypy`, `ruff check`, `ruff format`) | ✓ all Passed |
| RTX PRO 6000 Blackwell Server Edition (SM120, GB202) | ✓ |

## 跨 repo 改动 (PR 之外, 同 task 一并 ship)

| Repo | 文件 | 改动 |
|---|---|---|
| `6KD_fp8_block_scale` | `.release/scripts/sync_to_flashinfer.py` | FILE_MAP destinations 更新到新 `csrc/cute_sm120_mxfp8_groupwise/` 路径; SUBS include rewrite 加新模块前缀; 删 `quantize_mxfp8_zero_padding.cuh` entry (PR 内已 N6 删除) |
| `mega_inference` | `Justfile` | `_FI_SYNCED` paths 同步到新 layout; 删 `quantize_mxfp8_zero_padding.cuh` entry |

## 待 PR 流程项

| # | 项 | 状态 |
|---|---|---|
| 1 | Maintainer code review (含 CodeRabbit AI bot 自动 review) | 进行中 — CodeRabbit 一条 false positive 已 reply 解释 (worst-case 一致约定) |
| 2 | CI checks (post-merge style) | 待 maintainer trigger |
| 3 | Merge | 待 maintainer 批准 |
