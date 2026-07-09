# Task 02: FP8 `moe_gemm_fp8_nt_groupwise` 接线 (op / binding / JIT / Python entry)

在 task_01 已落地的 kernel 树上, 接通 FP8 float-scale ZeroPadding MoE 的唯一公开入口
`moe_gemm_fp8_nt_groupwise` (PM 约束: 与 MXFP8 一样只暴露 moe_gemm)。

## 前置状态

| 项 | 值 |
|---|---|
| FI 分支 | `sm120_moe_gemm_fp8_internal` @ `f4e9edf` (task_01: kernel 树已含 `cute_sm120_fp8_runner.{h,cu}` + `sm120_blockscaling/`, 未入 JIT sources) |
| 6KD 参照 | `main` `e583569`; thop 参照 `thop/fp8GroupwiseGemm.cpp` (`moe_gemm_fp8_nt_groupwise` @ L379) |
| MXFP8 counterpart | `csrc/cute_sm120_mxfp8_groupwise/cute_sm120_mxfp8_op{,_jit_binding}.cu` + `flashinfer/grouped_mm/cute_sm120_mxfp8_groupwise/` — 层次/词根/codestyle 基线 (核心原则 8) |

## Scope

- **In**: FP8 op `.cu` + jit binding + JIT module 扩展 (单 .so) + Python 子包/entry/导出 +
  validation/smoke/correctness/bench (task 内 tests)
- **Out**: docs `.rst` / AOT 注册 / upstream `tests/` / lint pass / strip 公开分支 (→ task_03+);
  dense/batched/masked/contiguous/psum 入口; FP8 quant 公开 helper (test 内构造够用则不 ship, sub_task_0 定)

## 集成后文件树 (主 review 对象)

```
csrc/cute_sm120_mxfp8_groupwise/
├── cute_sm120_fp8_op.cu                      NEW  手写: CuteFP8GroupwiseMoeGEMMSM120 launcher
│                                                  (mirror mxfp8_op.cu; 含 check_zero_padding_sfa_layout port)
├── cute_sm120_fp8_op_jit_binding.cu          NEW  手写: 单一 FFI export moe_gemm_fp8_nt_groupwise
└── (其余 task_01 已就位, 不动)

flashinfer/jit/cute_sm120_mxfp8_groupwise.py  MOD  source_paths += {cute_sm120_fp8_runner.cu,
                                                   cute_sm120_fp8_op.cu, cute_sm120_fp8_op_jit_binding.cu}
                                                   (同一 module 单 .so; module 名不变 → MXFP8 调用方无感)
flashinfer/grouped_mm/__init__.py             MOD  re-export moe_gemm_fp8_nt_groupwise
flashinfer/grouped_mm/cute_sm120_fp8_groupwise/   NEW dir (mirror MXFP8 sibling 子包)
├── __init__.py                               NEW  re-export + __all__
└── core.py                                   NEW  moe_gemm_fp8_nt_groupwise + _check_* helpers
                                                   (mirror MXFP8 core.py 结构; SFA layout check Python 侧预检)

.claude/tasks/task_02/                        internal-only
├── plan.md / sub_task_0_findings.md
├── tests/{test_validation.py, smoke.py, test_fp8_correctness.py, bench_fp8_vs_cutlass.py}
└── results/
```

## API 签名 sketch

```python
def moe_gemm_fp8_nt_groupwise(
    a: torch.Tensor,            # (cum_m, k) fp8_e4m3, token-packed, A 不 padding
    b: torch.Tensor,            # (num_experts, n, k) fp8_e4m3
    a_scale: torch.Tensor,      # fp32 padded SFA, contiguous [Kb, MpE]; MpE = (cum_m + 3E)//4*4,
                                #   expert i 起点 = (m_indptr[i] + 3i)//4*4, 16B aligned
                                #   (exact 校验 = 6KD check_zero_padding_sfa_layout, sub_task_0 摘录)
    b_scale: torch.Tensor,      # (num_experts, ceil(k/128), ceil(n/128)) fp32 contiguous
    m_indptr: torch.Tensor,     # (num_experts + 1,) int32 CSR cumsum (== 6KD token_offset)
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),   # 仅接受此值
    scale_major_mode: Literal["MN"] = "MN",
    backend: Literal["cute"] = "cute",
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,   # 默认 bf16
) -> torch.Tensor               # (cum_m, n)
```

- FFI binding: `moe_gemm_fp8_nt_groupwise` → `CuteFP8GroupwiseMoeGEMMSM120(a, b, a_scale, b_scale,
  m_indptr, out, scale_major_mode, gran_m, gran_n, gran_k)` — 与 MXFP8 binding 逐位对齐
- runner 调用: `CuteSm120Fp8GemmRunner<e4m3, bf16, float>::moe_gemm_fp8_nt_groupwise(D, A, B,
  token_offset, num_experts, total_rows, n, k, stream, SFA, SFB, granM, granN, granK)`
  (exact 签名 sub_task_0 从 `cute_sm120_fp8_runner.h` 摘录)
- Dispatch: FI 层零 dispatch; runner 内部 ZeroPadding 四档 selector (≤8 SwapAB / ≤32 M32 / ≤64 M64 / M128)
  + staged R2G epilogue

## Sub-task 表

| # | Sub-task | Gate |
|---|---|---|
| 0 | verify: FP8 runner.h moe 签名 + thop `check_zero_padding_sfa_layout` / `check_sfb_layout` 逐条摘录 + test 侧 SFA 构造方案 (复用 DG `get_col_major_tma_aligned_tensor` + per-expert padded offset) | `sub_task_0_findings.md` + xiy review |
| 1 | `cute_sm120_fp8_op.cu` + `cute_sm120_fp8_op_jit_binding.cu` | 语法/结构 review (Phase 3.5) |
| 2 | JIT module sources 扩展 | module JIT build 过, 记录 compile time delta |
| 3 | Python 子包 + entry + 导出 | `import` OK + validation tests PASS (非法 granularity/layout/dtype 拒绝) |
| 4 | smoke | 单 cell 出 tensor 不 crash, shape/dtype 对 |
| 5 | correctness | cells: E∈{4,8} × m_pe∈{1,4,8,16,192,256,1024} × (N,K)∈{(4096,7168),(7168,4096)} + uneven + empty expert; ref = per-expert dequant bf16 matmul; `calc_diff < 1e-3` |
| 6 | bench vs `group_gemm_fp8_nt_groupwise` (cutlass sm120) | 全 cells 数据落 CSV; 预期 small-M 领先 (6KD exp_32: MoE M_PE≤16 全正), large-M 接近; 负点如实记录 |
| 7 | findings 沉淀 + plan `## Results` | 收口 |

每 sub-task stop unstaged; code 改动过 Phase 3.5 subagent review (逐项对照 MXFP8 counterpart)。

## Risks

| # | Risk | 检测 / 缓解 |
|---|---|---|
| R1 | SFA padded-layout 校验 port 不完整 → kernel 读 OOB scale (静默错) | sub_task_0 逐条摘录 6KD 校验; validation test 覆盖每条拒绝路径; correctness 含 uneven/empty |
| R2 | fp8_runner.cu 入编译后 module 实例化 ~42→~90, compile time / .so size 膨胀 | sub-task 2 记录; 超时则评估 fp8_runner 拆独立 module (需 xiy 决策) |
| R3 | baseline `group_gemm_fp8_nt_groupwise` 的 m_indptr 4-aligned 约束 + 历史 sm120 disabled path | bench cells 取 m_pe%4==0 或 documented; bench 前 grep 当前 main |
| R4 | FP8 correctness ref 的 SFA 构造与 kernel contract 不一致 (padded offset 算错) | 先 bit-level 自检: 构造后按 contract 公式抽查 expert 起点; smoke 用 uniform scale=1 验 |
| R5 | tvm-ffi binding 参数序/类型与 runner 不匹配 (int64_t vs int) | mirror MXFP8 binding 逐位对齐; smoke 立即暴露 |

## Test strategy

- correctness ref: per-expert 独立 quant (`(1,128,128)` per-token-group float scale) + dequant bf16
  matmul; SFA 构造复用 DG `get_col_major_tma_aligned_tensor` + ZeroPadding per-expert offset
  (6KD `test/utils/layout.py` 仅作 cross-check 参照, 不引为 runtime dep)
- bench 协议: warmup 10 + 50 iter median, 每侧 ≥2 轮; 小 kernel delta 按 task_01 教训须隔离进程复核
- 环境: 6K Pro job `3028389` + `xiyTrtllm` 容器复用; `FLASHINFER_JIT_DEBUG=0 MAX_JOBS=8 FLASHINFER_NVCC_THREADS=2`

## Results (2026-07-09, 6K Pro 2u2g-spr-0490)

### Gates

| Gate | 结果 |
|---|---|
| Phase 3.5 code review | 2 blocking (n/k>0 漏检、same-device 漏检) + 2 minor, 修复后复核清零 |
| JIT build | 增量 1m20s; `.so` 1.02 → 1.52 MB; 双 FFI 入口就位 (R2 未触发) |
| validation | **13/13 PASS** (granularity/major-mode/backend/out_dtype/m_indptr/SFA×3/SFB×2/dtype 拒绝路径全覆盖) |
| correctness | **34/34 PASS**, calc_diff 7.11e-4 ~ 7.14e-4 (与 6KD test_fp8 自测范围一致); 含 uneven / empty_expert / routing_512 |
| bench vs cutlass | **24/24 全正**, 两轮无漂移 (>3pp 0 个), 无负点 |

### Bench vs `group_gemm_fp8_nt_groupwise` (cutlass sm120, `skip_check=True`, r1)

| 区间 | speedup 范围 |
|---|---|
| E=4 小 M (m_pe 4-16) | +57.1% ~ +68.8% |
| E=8 小 M (m_pe 4-16) | +15.5% ~ +19.3% |
| 大 M (m_pe 192-256) | +5.0% ~ +17.5% |
| m_pe=1024 | +0.4% ~ +5.1% |

数据: `results/{correctness_fp8.csv, bench_fp8_vs_cutlass_r1.csv, bench_fp8_vs_cutlass_r2.csv}`。

### 观察

- R3 命中: baseline 在当前 main 仍有 sm120 `num_groups > 1` wrapper-level disable
  (`gemm_base.py:7283`), bench 以 `skip_check=True` bypass, 仅作 perf baseline 不作正确性声明。
- validation 首轮 1 FAIL 是 test 用例 bug (n=k=256 方阵下 SFB transpose 后 shape 不变, 合法),
  非 binding 漏检; 改非方阵后 13/13。
- 与 6KD exp_32 报告方向一致 (MoE small-M 大幅领先); 绝对幅度更大, 因 baseline 不同
  (FI cutlass grouped entry vs 6KD 内部 CUTLASS cooperative paired binary), 不可直比。

### Verdict

task_02 完成: FP8 `moe_gemm_fp8_nt_groupwise` 全链路接通 (op/binding/JIT/Python entry),
validation + correctness + bench 三 gate 全绿。剩余 (task_03+): docs .rst / AOT / upstream test /
lint pass / strip 公开分支。
