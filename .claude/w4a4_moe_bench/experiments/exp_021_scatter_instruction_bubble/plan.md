# exp_021：Scatter 指令热点与 Bubble 定位

## 目标

不修改 Opt/production，只回答当前 Direct-32 Scatter 为什么慢：热点是否落在 `LDS`、
convert/multiply/pack、`REDG`、barrier/loop 中的某类 SASS PC，以及 scheduler 是否因 eligible
warp 不足产生明显 bubble。

## 最小实验

- 单臂、diagnostic-only standalone Scatter；复用 exp_020 的 M8192 canonical route fixture。
- 8 warps / 256 threads，grid=110；`sC` 保持 BF16 `128x128`、row-major logical、`K_SW128`
  physical layout，SMEM padding 保持一 CTA/SM。
- `sC` 使用 runtime dummy 值并在主循环前初始化；不做数值精度验证，但必须保持路径 live、地址合法。
- 每 CTA 只初始化一次 `sC`，随后重放完整 Scatter；PC 结论只统计 Scatter 源码/SASS 区间，且初始化与
  初始化同步的动态指令/周期占比必须 `<1%`，否则本次 capture 不判定 bubble。
- 保留真实 token/weight、tail、topk=8 与 4-slice，即每个输出元素 32 个 REDG contribution。
- 只采一次 targeted NCU：`InstructionStats`、`SourceCounters`、`SchedulerStats`、
  `WarpStateStats`、memory/shared 与 launch/resource；用 SASS PC/source line 解释。
- NCU 定位指令候选后，按用户 follow-up 使用已有 audited IKET provider：full-grid warp trace
  判断 warp/CTA tail，selected-cluster named range 仅作顺序与方向性诊断。
- 分析前将 standalone 与当前 Opt Scatter 的 `LDS/REDG` 形式、循环展开、动态次数和资源并列审计；
  形态不一致时，结果只描述 standalone，不能外推 Opt。
- PC stall 同时报告原始样本和每条 executed warp instruction 归一化值；等待 PC 与其 producer/consumer
  依赖分开解释。
- 从 fixture 与 task 映射闭合 `8 routes x 4 slices = 32` 个相同 `[token,h]` destination contribution，
  并记录 637 groups、2548 slice tasks、110 CTA 的映射。

## 判定

- `REDG` PC 主导且伴随对应 throttle/scoreboard：定位 reduction service/contention；下一步才做冲突拓扑实验。
- `LDS` PC 主导：检查 shared transaction/bank conflict 和向量化。
- Issue Active 低、Active warps 存在而 Eligible warps/No Eligible 显著：确认 scheduler bubble，再由 PC stall
  区分依赖链与 memory/reduction backpressure。
- warp/barrier tail 若没有 IKET 证据则保留未知，不猜测。

本实验不报告 production speedup，也不据 standalone duration 归因 production latency。

## IKET follow-up

- Provider 固定为 IKET 0.7.10；保留 full-grid warp trace 与 selected-cluster named-range 原始 trace。
- 时间单位只写 `raw timestamp units`，不擅自换算为 ns/µs。
- full-grid trace 可定量判断 kernel-end warp/CTA skew；named overlay 若显著扰动 runtime，则不生成
  phase 占比，也不外推 production fused kernel。
- 本 follow-up 属于同一实验的证据补强；按 single-round 规则不重复 Plan Review。

## Plan Review

**Date**: 2026-07-21
**Reviewer**: subagent

**Verdict**: ⚠️ Gaps（已一次性纳入上述实验 gate，按 single-round 规则不复审）

**Gaps + suggested fix**:

- 排除 `sC` 初始化/同步对 scheduler 统计的污染，并只解释 Scatter PC。
- 对照当前 Opt Scatter 的 SASS、动态工作和资源；漂移则限制结论范围。
- PC stall 按 executed warp instructions 归一化，避免把高频误判为高单次成本。
- 用 canonical fixture 与 task 映射证明实际 32-way destination fan-in。
