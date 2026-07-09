# Task 01: 6KD kernel 目录整体搬运 + FI MXFP8 精度/性能验证

Progressive engineering 第一步: 只做 `kernels/include/cute_sm120_mxfp8_groupwise/` 整体搬运
(FI in-tree MXFP8 布局重构对齐 + FP8 kernel 源码进树但不接线), 然后验证 FI 现有 MXFP8 入口
`moe_gemm_mxfp8_nt_groupwise` 精度和性能无回退。FP8 入口接线 = task_02+。

## 外部 repo

| 项 | 值 |
|---|---|
| repo | `/home/scratch.xiy_gpu/mega_inference/6KD_fp8_block_scale` (mega_inference submodule) |
| HEAD | `5c562b1` (main, fp8_blockscaling exp_19..32 merged) |
| sync 工具 | `.release/scripts/sync_to_flashinfer.py` (16-file FILE_MAP + 5 SUBS + leftover 校验) |
| FI 分支 | `sm120_moe_gemm_fp8_internal` @ upstream main `a25af45` (2026-07-08) |

## Scope

- **In**: 16-file kernel source sync (整体搬运, 含 FP8 runner/family 源码进树); MXFP8 in-tree 布局重构
  (sm120_common 抽出 + 旧文件 DEL); FI MXFP8 入口 correctness + perf 验证 (sync 前 vs sync 后)
- **Out**: FP8 op/binding/JIT/Python entry/test (task_02+); `quantize_mxfp8_for_moe.cuh` 及其 kernel
  (xiy 指定不集成); dense/batched/masked/contiguous/psum 入口; 6KD kernel 内部改动; strip 公开分支
- **无新公开 API**: 本 task 不新增/不改任何 Python entry 签名; FP8 API sketch 放 task_02 plan

## 集成后文件树 (主 review 对象)

```
csrc/cute_sm120_mxfp8_groupwise/
├── cute_sm120_mxfp8_op.cu                    KEEP (只 include runner.h, 不受重构影响)
├── cute_sm120_mxfp8_op_jit_binding.cu        KEEP
├── cute_sm120_mxfp8_runner.h                 MOD  sync (6KD 当前版; quantize 声明处理见 R1)
├── cute_sm120_mxfp8_runner.cu                MOD  sync (同上)
├── cute_sm120_fp8_runner.h                   NEW  sync (进树不接线, 不入 JIT sources)
├── cute_sm120_fp8_runner.cu                  NEW  sync (同上; 全六 GemmType dispatch + ZeroPadding selector)
├── sm120_blockscaled/                        (MXFP8 family)
│   ├── builder.cuh                           MOD  sync
│   ├── kernel_impl.cuh                       MOD  sync
│   ├── launch.cuh                            MOD  sync
│   ├── sf_mxfp8_tma_load.cuh                 NEW  sync
│   ├── math.cuh                              DEL  → sm120_common/math.cuh
│   ├── scheduler.cuh                         DEL  → sm120_common/scheduler.cuh
│   └── utils.cuh                             DEL  → 拆入 builder.cuh / sf_mxfp8_tma_load.cuh /
│                                                    sm120_common/{ab_tma_load,epilogue}.cuh (sub_task_0 核实去向)
├── sm120_blockscaling/                       NEW dir (FP8 float-scale family, sync, 不接线)
│   ├── builder.cuh
│   ├── kernel_impl.cuh
│   ├── launch.cuh
│   └── sf_fp8_tma_load.cuh
└── sm120_common/                             NEW dir (shared components, sync; MXFP8 kernel 实际消费)
    ├── ab_tma_load.cuh
    ├── epilogue.cuh
    ├── math.cuh
    └── scheduler.cuh

flashinfer/jit/cute_sm120_mxfp8_groupwise.py  仅当 JIT sources / include 列表受重构影响时 MOD
                                              (sub_task_0 核实; 目标: MXFP8 module 行为不变)
其余 flashinfer Python / csrc / tests / docs   本 task 一律不动

.claude/tasks/task_01/                        internal-only (strip 不入公开分支)
├── plan.md / sub_task_0_findings.md / tests/ / results/
```

后续 task 预告 (不在本 task 实施): task_02 = FP8 op/binding/JIT/Python entry (`moe_gemm_fp8_nt_groupwise`,
mirror MXFP8 subpackage pattern); task_03+ = correctness/bench/upstream test/docs。

## Sub-task 表

| # | Sub-task | Gate |
|---|---|---|
| 0 | verify: sync `--dry-run` vs FI 树逐文件对照、include 闭包 (quantize/utils 去向)、JIT sources 列表影响、R1 方案确认 | `sub_task_0_findings.md` + xiy review |
| 1 | kernel source sync (16 files) + DEL 3 旧文件 + R1 处理落地 | JIT compile 过 (MXFP8 module, release mode) |
| 2 | MXFP8 correctness 验证 | 现有 `tests/grouped_mm/test_cute_sm120_mxfp8.py` 全 PASS + task 内 calc_diff cells 全 < 1e-3 |
| 3 | MXFP8 perf 验证 (sync 前 vs sync 后, 同 cells 同协议) | 全 cells 回退 ≤ 噪声带 OR documented gap |
| 4 | findings 沉淀 + results 落地 | plan.md `## Results` + shared findings.md 更新 |

每个 sub-task 完成默认 stop unstaged; code 改动过 Phase 3.5 subagent review。

## Risks

| # | Risk | 检测 / 缓解 |
|---|---|---|
| R1 | **6KD `cute_sm120_mxfp8_runner.cu:26` include 且内联 `quantize_mxfp8_for_moe`**, FILE_MAP 不搬该 header → 原样 sync 编译必断 | sub_task_0 dry-run 确认; 候选: (a) sync script 加 strip 规则 (6KD release tooling 改动, 需 xiy 授权) (b) 6KD runner 加 guard macro; 方案 xiy 定 |
| R2 | 布局重构 (DEL math/scheduler/utils + 新增 sm120_common) 断 MXFP8 include 闭包或改 JIT sources 集合 | sub_task_0 列 include 闭包; sub-task 1 gate = JIT compile |
| R3 | 6KD 侧 exp_19..32 改了 shared components (epilogue/scheduler 等), MXFP8 kernel 行为/性能可能随之变化 | sub-task 2/3 直接测; 回退超噪声带则 bisect 到具体 shared 文件 |
| R4 | 布局重构后 JIT cache stale → 假 PASS/假 FAIL | 每轮验证前清 `~/.cache/flashinfer` |
| R5 | host pre-commit 2.17.0 太老 | lint 在容器内跑 |

## Test strategy

- **Correctness**: 现有 `tests/grouped_mm/test_cute_sm120_mxfp8.py` (upstream regression) + task 内
  cells: E ∈ {4, 8} × m_pe ∈ {1, 4, 8, 16, 192, 256, 1024} × (N,K) ∈ {(4096,7168), (7168,4096)},
  含 uneven m_pe + empty expert; ref = per-expert dequant bf16 matmul; `calc_diff < 1e-3`
  (MXFP8 上游 test 自身 threshold 为 cos-sim > 0.99, 两套都跑)
- **Perf**: 同 cells, sync 前 (`a25af45` + 旧树) vs sync 后同 binary 协议 paired 对比;
  warmup 10 + 50 iter median; 百分比报告; 判定带 ±1% 噪声
- 环境: 6K Pro 节点 (Slurm) + Docker `xiyTrtllm`; `FLASHINFER_JIT_DEBUG=0 MAX_JOBS=8 FLASHINFER_NVCC_THREADS=2`

## Results (2026-07-08, 6K Pro 2u2g-spr-0490, job 3028389)

### Correctness

| 项 | pre-sync | post-sync |
|---|---|---|
| upstream `test_cute_sm120_mxfp8.py` | 69 passed | 69 passed (fresh JIT compile of 重构树) |
| task cells (40: uniform 28 + granK32 8 + uneven 2 + empty_expert 2) | 40/40 PASS | 40/40 PASS |
| calc_diff 范围 | ≤ 8.7e-11 | 逐 cell 与 pre **完全相同** (输出 bit-identical) |

注: task cells 的 ref 用同一 quant 输入的 dequant matmul (回归检测口径, 非 full-pipeline 精度口径),
故 calc_diff ~1e-11 量级; upstream test 的 cos-sim 口径独立覆盖。

### Perf (28 uniform cells, warmup 10 + 50 iter median, 每侧 2 轮)

- 25/28 cells: |delta| ≤ 0.7% (噪声带内); E=8 全部 14 cells 无回退
- 3 个 E=4 small-M cells 有跨轮一致的小回退 (documented gap):

| cell (E, m_pe, N, K) | pre r1/r2 (us) | post r1/r2 (us) | delta 区间 |
|---|---|---|---|
| 4, 8, 4096, 7168 | 35.6 / 35.6 | 37.2 / 36.2 | -1.5% ~ -4.5% |
| 4, 4, 7168, 4096 | 35.6 / 35.6 | 36.2 / 36.4 | -1.4% ~ -2.2% |
| 4, 16, 7168, 4096 | 40.0 / 40.1 | 40.6 / 40.3 | -0.5% ~ -1.5% |

数据: `results/bench_{pre_sync,pre_sync_r2,post_sync,post_sync_r2}.csv`、
`results/correctness_{pre_sync,post_sync}.csv`。

### 结论与观察

- 事实: 整体搬运 + 布局重构后, MXFP8 correctness 无变化 (bit-identical), 编译干净, E=8 与大 M 性能持平。
- 事实: 28-cell 序列 bench 中 E=4 小 M 2-3 个 cell 出现 ~0.6-1.6us 增量 (两轮一致), 触发回退调查。
- gate 判定: correctness gate PASS; perf gate PASS (调查后确认 kernel 无回退, 见下)。

### 回退调查 (E=4 small-M, 最差 cell `(4,8,4096,7168)`)

四层证据链, 全部指向 **kernel 零回退, bench 差异为测量环境效应**:

| # | 证据 | 结果 |
|---|---|---|
| 1 | runner moe dispatch 源码 diff (旧 HEAD vs 新树) | 阈值/KT 完全一致 (≤12 SwapAB / ≤32 M32 / <96 M64); cell 走 `KT_SWAPAB_N8<128,8,128,4>` 两侧相同 |
| 2 | SwapAB 执行路径语义对照 (subagent, 14 组件: builder/SF load/kernel_impl/epilogue/scheduler/launch/flags) | 全部语义等价; 唯一结构差异 = 同 TU 实例化 8 → 42 kernel (cubin 375KB → 1MB) |
| 3 | SASS diff (`cuobjdump -sass`, ZeroPadding SwapAB GranK=128 实例) | **逐指令一致** (4117/4117 行, 仅 mangled name 的 namespace 不同); registers 56 相同 |
| 4 | NCU full-set (同 cell, 两树各采) | `sm__cycles_elapsed.max` **完全相等 (202072)**, duration 83.456 vs 83.488us (+0.04%), grid/occupancy/L2 hit 全同 |
| 5 | 隔离单 cell fresh-process bench (各 3 轮) | pre 35.68/36.00/36.26, post 36.10/36.19/36.22 — 分布重叠, 差异在单侧 run 方差 (~0.6us) 内 |

结论: 原 bench 序列中的 -1.5%~-4.5% 是小 kernel (~36us) 在 28-cell 序列 harness 中的
run-to-run 方差, 不可归因于搬运的 kernel 代码。证据: `results/ncu/cell_4_8_{pre,post}.ncu-rep`。

调查副产物 (踩坑): NCU 注入下不能做 flashinfer JIT 编译 — ninja/nvcc 工具链在 TreeLauncher
注入下 hang (25 min 零编译进程); 必须先普通跑一次完成 JIT, NCU 时 cache hit。

### Verdict

task_01 完成: 16-file 整体搬运 + MXFP8 布局重构落地, quantize strip 规则生效 (残留 0),
MXFP8 精度 bit-identical、性能经 SASS + NCU + 隔离 bench 三重验证零回退。
后续: task_02 = FP8 op/binding/JIT/Python entry 接线。
