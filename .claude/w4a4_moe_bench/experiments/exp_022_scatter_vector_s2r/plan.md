# exp_022：Scatter 向量化 S2R

## 目标

只修复当前 Opt Scatter 从 `sC` 读取 BF16x8 时的 shared-memory wavefront 放大：

```text
当前：每 lane 8 x LDS.U16，实测 actual / ideal = 3.9755x
候选：每 lane 1 x aligned 128-bit S2R，目标接近 1x
```

本实验不改 FC2、Scatter ownership、REDG、warp 数、同步或 `sC` layout；不与
FC2/Scatter overlap 方案混合。

## 两臂与唯一变量

- Baseline：当前 `moe_dynamic_kernel_opt.py`，source SHA256
  `ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19`。
- Candidate：从 Baseline 生成独立 overlay；只把 Scatter 中同一 lane 的连续 8 个
  `sC` BF16 scalar load 改为一次 128-bit BF16x8 tiled copy，再按原顺序转 FP32、乘 route
  weight 并调用原 REDG。
- 固定 `K_SW128` physical layout、8-warp `32M x 64N` ownership、lane-to-output mapping、
  metadata load、tail predication、barrier、block/grid、SMEM 容量和全部 logical output work。
- 不允许把 swizzled `sC` 当 row-major raw pointer；若编译器不能生成合法向量 load，本方案直接
  Reject，不用额外布局改动掩盖失败。

## 机制预检

当前映射中每 lane 的 8 个 BF16 地址连续且 16B 对齐。先只编译/反汇编 Candidate：

- 必须看到 `LDS.128` 或等价的单条 128-bit shared load；
- 若仍为 `8 x LDS.U16`，只是源码包装变化，Reject；
- 若退化为 `4 x LDS.U32`，不视为达到本实验目标；
- cubin 必须保持 zero spill，且 grid/block、warp topology、SMEM 与 Baseline 相同。
- 在编译前枚举全部 `(warp, lane, loop, buffer)` 的 swizzled physical address；每个 vector base
  必须 16B 对齐，8 个元素必须恰为该 lane 原有的连续 16B segment，且不得跨到相邻 row/vector。
  地址枚举、SASS 和 correctness 三者必须同时闭合，不能只依赖逻辑 layout 推断。

机制预检失败即停止，不做完整 benchmark 或 deep profile。

## 正确性与工作量门禁

- 使用当前 canonical fixture，检查 M=256、M=8192，并覆盖
  `valid_rows=0/1/31/32/33/63/64/65/95/96/97/127/128`；`valid_rows=0` 用独立 ownership
  check，其余走完整 fused kernel；
- 加入 duplicate-destination/contention fixture，显式验证同一 token 的 topk=8、4 slices 共 32 个
  contribution 没有因 vector load 改变 ownership 或累加语义；
- 与 Baseline 使用相同 oracle/tolerance，并检查 finite、sentinel、元素次序和 output ownership；
- Scatter 的 REDG 动态次数与 Baseline 相同；M8192 anchor 为 `67,108,864`；
- Tensor/FC2 work、route metadata work 和 output shape 均不得变化；
- 记录 source/JIT/cubin hash、编译环境、GPU UUID、register/SMEM/stack/spill、grid/block。

## 最小性能与 NCU 验证

- 在同一空闲 5KP、同一进程和固定环境做未插桩 paired ABBA；M8192 为主 case，M256 为 guard。
- 每臂 warmup 5 次；按 `ABBA/BAAB/ABBA` 三组顺序，每个位置 50 次 replay。保存全部 raw
  samples、每组 arm median/CV 与 paired speedup。Accept 要求 M8192 三组 speedup 全为正且总体
  median speedup `>=1.0%`；M256 median regression 不得低于 `-1.0%`，任一 arm CV `>1.5%`
  的整组作废重测。
- correctness 与静态机制 gate 通过后，只采最小 NCU：`sC` shared load actual/ideal wavefront、
  shared throughput/stall、REDG 动态工作、register/stack/spill 和 occupancy。
- NCU 的 `sC` actual/ideal 必须只累加两臂各自已审计的 Scatter `sC` LDS SASS PC 集合；metadata
  LDS、FC2 R2S 和其他 phase 必须排除。若改用 standalone，只能作机制证据，且必须先证明其
  Scatter SASS body、动态 vector/REDG 次数和 layout 与 fused arm 等价。
- 目标是 `sC actual / ideal <= 1.5x`，且 latency 与机制证据方向一致。

## 固定比较身份

- Shape：`E=256, H=2048, I_tp=512, topk=8, NVFP4 weight, BF16 input/output`；launch
  baseline 为 grid `(1,1,110)`、block `(288,1,1)`。
- Fixture seed=2026；manifest SHA256
  `683ec75341e4d8317dfdc5c4b04229f9695f9aa286d575c4f6e1fdef55d90801`；M256
  `86b505097acd06bed5a50c3528c78525e6087c07ed69f86606607599ffa21686`；M8192
  `c113ecd5ddeff77154ddbd23fc3dc3c83f8ee822e880179ca5c16b1145372438`。
- Host `10.6.142.16`、GPU UUID `GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522`、application
  clock 2377 MHz、driver 580.95.05；foreign process 或 thermal/clock drift 时整组无效。
- Container `nvcr.io/nvidia/pytorch:26.05-py3`，image ID
  `sha256:a4e056e1d34a5cc9387512ffa3abeed778e3dc7966633c5154d771705d8835ac`，nvcc
  13.2.78，CUTLASS commit `b46b16d003484063bca4ed365e44095c4c6ed633`，python deps SHA256
  `32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74`。
- 两臂使用各自 fresh JIT root；capture 前冻结 source/JIT/cubin hash，capture 中 artifact set 不得漂移。

## 判定

- **Accept**：正确性/身份/zero-spill 全通过；真实生成 128-bit S2R；wavefront 放大降至
  `<=1.5x`；M8192 paired benchmark 可重复正收益且 M256 不回退。
- **Reject**：任一正确性或工作量门禁失败、load 被 scalarize、仍有明显 shared amplification、
  产生 spill，或未插桩性能没有正收益。
- **Unresolved**：性能与 NCU 机制方向冲突；只补能区分冲突的最小证据，不扩大实验。

Accept 后才把单一变更移入 Opt；Reject 时只保留实验 overlay 与结论。

## 产物

```text
exp_022_scatter_vector_s2r/
  plan.md
  comparison_registry.json
  build_candidate.py
  results/
    result.md
    manifest.json
    raw/
```

复用现有 correctness/benchmark/profile runner，不复制 JIT cache、临时 cubin 或重复采样文件进 git。

## Plan Review

**Date**: 2026-07-21
**Reviewer**: subagent `/root/exp022_plan_review`
**Verdict**: `⚠️ Gaps`（以上缺口已一次性修正；按 single-round 规则不复审）

- 冻结 fixture、shape、GPU/clock、container/nvcc/CuteDSL dependency、launch 与采样协议；
- 将 NCU shared wavefront 归因限制为审计过的 Scatter `sC` LDS PC，排除 metadata；
- 预注册三组 paired 判定阈值和 noise/CV gate；
- 补 `valid_rows=0` 与 duplicate-destination/contention corner case；
- 在实现前枚举 swizzled physical address，不能把逻辑连续直接当作物理连续。
