# exp_012：Intern compact epilogue 跨 pass 同步修复

## 状态

已收口（2026-07-19）：该单点 barrier 修复被拒绝。Fresh Baseline M8192 复现失败；Candidate
M4096/M8192 仍稳定出现全零行，M2048 回归通过。未运行 benchmark 或 NCU，结论见
`results/result.md`。

## 目标

快速验证 exp_009 在 `M=4096/8192` 出现整行全零与跨 replay 非确定性，是否由 FC1 compact
epilogue 的两个 64-row pass 复用单级 `sC`、但第一轮 Q1 写完后缺少 CTA 同步所致。

本实验只做 correctness，不做 benchmark、NCU 或性能归因。

## 实验臂

| Arm | 定义 |
|---|---|
| Baseline | exp_009 已锁定的 Intern adapter overlay，SHA-256 `42ca8d40e18b5d0f001236b09b85cbc0aa30e6010f0954efd538d8b9a2fb57d2`；保留既有失败证据，并在同一执行会话重跑一次 M8192 作为 fresh failure control |
| Candidate | 从 Baseline 生成独立 overlay；只把 FC1 compact epilogue 循环后的 `fence_proxy + epilog_sync_barrier` 移入每个 `epi_m` pass 的 Q1 末尾 |

Candidate 不修改 production 或原始 Intern 文件。生成器必须校验 baseline hash、唯一替换锚点，并保存候选 hash 与最小 diff。

## 因果模型

当前执行顺序是：

```text
pass 0: R2S(sC) → barrier → Q1(sC→sA/sSFA)
pass 1: R2S(sC) → barrier → Q1(sC→sA/sSFA)
final barrier
```

不同 warp 完成 pass 0 Q1 的时间不同；已完成的 warp 可进入 pass 1 覆写共享 `sC`，而其他 warp
仍在读取 pass 0 `sC`。Candidate 改为：

```text
pass 0: R2S(sC) → barrier → Q1 → fence/barrier
pass 1: R2S(sC) → barrier → Q1 → fence/barrier
```

## 固定条件

- 复用 exp_009 的 canonical fixture、reference、dispatcher adapter、容器、CuteDSL/CUDA 工具链与 correctness harness。
- 使用同一块空闲 5KP，并记录 GPU UUID、source hash、JIT/cubin 身份；candidate 使用 fresh JIT namespace。
- 计划主判别用例：`M=4096`、`M=8192`；每个 case 运行两次独立 preparation，共 4 个 correctness replay；两者都通过后补一次 `M=2048` preparation 回归。
- 每个 case 沿用 harness 的正式数值、finite、workspace/route/task gate；额外要求没有 full-zero output row。
- 不要求跨 replay bitwise hash 一致，因为 atomic scatter 的顺序本身可变；每个 replay 均须独立通过正确性阈值。
- 计划每次 preparation 记录并核对 candidate overlay hash、JIT artifact/cubin hash、目标 MoEDynamicKernel entry 与 launch geometry；未实际命中候选 cubin 时实验无效。

## 判定

- **接受该修复候选**：Candidate 在 M4096、M8192 均通过 correctness、workspace gate 且 full-zero rows 为 0，随后 M2048 回归通过。
- **拒绝该修复候选**：Candidate 编译成功，但 M4096 或 M8192 仍出现 correctness failure/full-zero rows。
- **实验无效**：候选 diff 超界、工具链/源码身份漂移，或 barrier 改法无法编译；不能据此否定根因。

全部脚本、overlay、日志和结果写入本实验 `results/`；正式结论前核对原始 JSON/日志，不从 benchmark 数据推断。

## 实际执行与偏差

- 每个 M 实际只进行一次 fixture/build/capture，随后执行 3 或 4 次 CUDA Graph replay；没有完成两次独立 preparation。
- `expected_launch` 是 harness contract，不是 profiler-observed entry/geometry；本实验没有运行 NCU 或 NSys。
- 候选身份由 exact overlay import、fresh JIT namespace、dynamic cache entry 与唯一 cubin hash 约束。
- 重复失败足以拒绝这个具体 barrier relocation candidate，但不能排除同一区域存在其他竞态。

## Plan Review

**Date**: 2026-07-19
**Reviewer**: isolated subagent
**Verdict**: ⚠️ Gaps — 计划已修订；实际执行偏差已在上节记录

- 随机 race 可能让少量 candidate replay 假通过：加入同会话 baseline M8192 fresh failure control，并要求主 case 各做两次 preparation、共 4 replay；实际只完成一次 capture 后 4 replay。
- 必须证明实际运行候选而非旧缓存：实际完成 overlay/JIT/cubin 身份约束，但没有 profiler-observed entry/launch geometry。
