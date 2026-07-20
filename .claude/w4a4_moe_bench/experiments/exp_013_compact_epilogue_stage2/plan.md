# exp_013：Stage2 compact epilogue 快速验证

## 状态

已收口：**Reject**。单变量验证 exp_008 已接受 kernel 是否能从 `M128×N128` epilogue working set
缩小为 `M64×N128`；结果未通过 zero-spill gate，诊断性 ABBA 也显示稳定小幅回退。

## 目标与因果问题

Baseline 锁定 exp_008 `branch_paired_n64_v1`，source SHA-256
`f3c246817679d962a3f7160dbe8b9e68262c919e26e306f349200961fc4ac971`。Candidate 只做：

1. `epi_tile: (128,128) → (64,128)`；
2. 把 FC1 activation/Q1 重排成四个 `(M64,N64)` rectangle：每个 N64 Gate/Up accumulator
   依次消费两个 M64 half并写入 compact `sC`；half0 Q1 的每线程两个 `packed64+scale`
   先留在显式 register cache，half1 FC1 释放输入 staging 后再与 half1 Q1 一起写回原 full
   `sA/sSFA` 坐标；
3. 把 FC2 scatter ownership 改成覆盖当前 M64 pass 的 4-warp cooperative mapping；其余 4 个 math
   warps 继续不参与 scatter。

第 2、3 项是 compact epilogue 的必要消费顺序/索引适配，不作为独立优化。Stage2、8 math warps + 1 DMA warp、
双 N64 Gate/Up、A/SFA 复用、Route/Q0、Q1、FC1/FC2 tensor work 与数学模式均保持不变。
不修改 production、Intern kernel、exp_008 overlay 或 `.claude/w4a4_moe_bench/moe_dynamic_kernel_opt.py`。

## 对比臂与实现约束

| Arm | Source | 唯一差异 |
|---|---|---|
| `exp008_v1` | exp_008 immutable accepted overlay | M128 epilogue；原 8-warp quadrant scatter |
| `compact_epi_m64_stage2_v0` | 从 baseline 生成的 exp_013 immutable overlay | M64 epilogue；每个 pass 仍由 4 个 scatter warps 覆盖 64×128 |

生成器必须 fail closed：核对 baseline hash、每个锚点唯一命中，保存 unified diff，并拒绝 stage cap、
warp/layout、Gate/Up/Q1/FC2 compute body 或其他 executable source 漂移。Candidate block 仍为 `(288,1,1)`；
本实验不得把 scatter 扩为 8 warp。

Compact FC1/Q1 的执行契约为：

```text
for n_half in 0..1:
  FC1 Gate/Up M128×N64                         # OMMA work 不变
  for m_half in 0..1:
    activation[m_half,n_half] → sC[0:64,0:64] # 复用同一 compact window
    fence_proxy → 8-math-warp barrier
    Q1 rectangle(m_half,n_half)
    fence_proxy → 8-math-warp barrier          # 才允许复用 compact sC
```

Q1 坐标必须满足 `logical_row=m_half*64+local_row`、
`logical_sf_block=n_half*4+local_sf_block`；四个 rectangle 的 ledger 必须证明原 `128×128`
activation/FP4 block 恰好覆盖一次。由于 `sA/sSFA` 同时承载 FC1 input staging，half0 Q1 不得提前
覆盖它们；每线程 cache 固定为两个 `Uint64+Uint8`，half1 FC1 完成后才按原 swizzled layout 写回。
barrier 不得放在部分 warp 才进入的 tail 分支内；新增 cache 必须纳入 zero-spill gate。

## Gate A：correctness 与静态资源

1. Fresh JIT；锁 source/PTX/cubin、kernel symbol、grid/block、GPU UUID 和工具链身份。
2. Candidate 跑 canonical M256、canonical M8192，以及 M256 下使同一 expert 的 `valid_rows` 精确为
   `63/64/65` 的三个 directed fixtures与 `sparse_empty`；每个结果必须 finite、通过既有
   数值/workspace/route/task gates，且
   full-zero output rows 为 0。
3. 核对 FC1/Q1、FC2、scatter 的 logical trip count：四个 FC1/Q1 rectangle 无重漏；每个 FC2 M64 pass
   由 W0–W3 以 `tid∈[0,127]、stride=128` 恰好覆盖 `valid_rows×128`，W4–W7 不读写 scatter 数据但
   必须参加 R2S 后和 scatter 后的两道 8-warp barrier；不得减少 Tensor work、输出列或有效 routed rows。
4. exact candidate cubin 必须 `STACK=0`、compiler SpillRefill/local SASS=0；否则直接 reject。

任一 correctness、identity、work 或 zero-spill gate 失败，不运行性能测试。

## Gate B：最小同机 E2E

- 同一块空闲 5KP、固定 2377 MHz、相同容器/checkout/dependencies/toolchain；两臂独立 fresh JIT。
- Cases：canonical M256、M8192；outer fused MoE CUDA Graph node，不含 JIT、capture、host launch。
- 每次 replay 前 192 MiB L2 flush；warmup=5、timed=50；每个 M 做 2 组 A-B-B-A，位置为独立进程。
- 报 arm median、paired-group speedup（`baseline/candidate-1`）、方向与范围；不把 50 replay 当独立样本。
- 本轮是快速筛选，不用 NCU、IKET 或 phase instrumentation。

## 预注册判定

```text
correctness / identity / work / zero-spill fail
  → reject，不测性能

任一 M 回退超过 2%，或 paired groups 方向不一致
  → reject

两个 M 都不回退，且至少一个 M 的两组 ABBA paired speedup 均超过 2%
  → accept 为后续候选

两个 M 都在 ±2% 内
  → no effect，reject
```

混合或噪声结果按“没有可复现效果”处理并 reject，不追加重度 profile。正式 `result.md` 前执行
ex-post data audit；全部生成物只放 `results/`，无关 JIT/cache/raw 临时文件不入库。

## 后续、非本实验变量

基于 fused phase breakdown，另开两个单变量实验：`Route/Q0 → 8 warp` 与 `Scatter → 8 warp`。
两者不得并入 exp_013；先分别验证正确性和 E2E，没效果即 reject。

## 执行后说明

- 初版把 compact `sC[M64]` 用来反推 full `M128×N128` FC2 accumulator layout，导致 FC2 输出全零；
  改为 `tiled_mma.partition_shape_C(full tile)` 后，canonical M256/M8192、63/64/65 行边界与
  sparse-empty 全部通过。
- 正确的 v2 仍为 `STACK=24 B/thread`，因此 Gate A 已判 reject。
- 为决定是否值得继续消除 spill，额外执行了两组诊断性 ABBA；它不改变 Gate A 判定，也不能作为
  acceptance evidence。M256 与 M8192 均稳定小幅慢于 exp_008，故停止该方向。

## Plan Review

**Date**: 2026-07-19
**Reviewer**: isolated subagent
**Verdict**: ⚠️ Gaps — 已一次性修正，不再复审

- M64 `sC` 会覆盖原先两 N64×M128 activation：已定义四个 `(M64,N64)` R2S/Q1 rectangle、前后
  8-warp barrier 与 logical row/SF ledger。
- 4-warp compact scatter 的 ownership 与边界不足：已锁 W0–W3 `tid/stride=128`、W4–W7
  barrier-only，并增加 `valid_rows=63/64/65` directed cases。
- “稳定超过 2%”不够可执行：已要求同一 M 的两组 paired ABBA 都过线。
