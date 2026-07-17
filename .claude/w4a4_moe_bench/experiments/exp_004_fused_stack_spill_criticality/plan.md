# Experiment 004 Plan: Fused Stack/Spill Criticality

Status: **executed / closed inconclusive (2026-07-16)**. Primary 与唯一 fallback 均未命中
14-word static gate，因此未进入 paired benchmark；attribution-only arm 的 static + NCU 已闭合。
完整结论见 [results/result.md](results/result.md)。本文底部记录了 exp_004 唯一一次
`Plan Review`；所有重大缺口已一次性修正，不再进行第二轮 review。本实验与 exp_003 分开；任何
IKET marker 都不会进入 production timing 或 NCU binary。

## Goal

在 `M=8192` CuteDSL Fused MoE 上调查以下因果链：

`FC1 accumulator lifetime → static spill bundle → executed local traffic → uninstrumented latency`

Formal scope 明确拆成两部分：

1. 对 108-word main bundle，只验证它是否属于 first-produced accumulator 跨第二个 FC1 GEMM 的
   **结构性 lifetime**；没有 FP32 等价的 clean elimination arm，因此不对它作 formal criticality verdict。
2. 对 14-word activation tail，用语义等价、单变量变体闭合完整因果链，并判断其性能收益是否超过噪声。

本实验不预设全部 122 words 都属于 `gate_acc`，也不把 stack frame、compiler spill annotation、
local-address-space traffic 或 latency 当作同一个量。报告必须把 `H108 attribution-only`、
`H14 criticality` 与仍 unresolved 的 `H108 criticality` 分开。

## Existing Evidence and Source Model

exp_002 的 production M8192 binary 提供动机，不替代本实验的 fresh baseline：

| Evidence layer | Existing observation | Interpretation limit |
|---|---:|---|
| Resource | `REG=255/thread`, `SMEM=92,160 B/CTA`, `STACK=488 B/thread` | `LOCAL=0` 不表示没有 spill；NCU configured stack limit 也不是实际 frame |
| Static SASS | `122×LDL.LU`, `54×STL.64 + 14×STL` | 无 source lineinfo，变量归因仍需实验 |
| Dynamic local | load/store 各 `4,950,272 sectors`，各 `158.409 MB` | local footprint 不是 DRAM traffic |
| Timing | `1782.547 us`, repeat spread 约 `0.14%` | 历史时间只用于量级参考 |

动态 local sectors 与下式精确闭合：

```text
122 words/lane × 2536 tasks × 4 MMA warps × 4 sectors/word
= 4,950,272 sectors per direction
```

production source 中：

- `gate_acc` 在 `moe_dynamic_kernel.py:1651` 创建，在 `:1904–2030` 完成 Gate OMMA；
- `up_acc` 在 `:2038–2166` 完成 Up OMMA；
- `gate_acc` 跨过整段 Up，直到 `:2196–2214` 才与 `up_acc` 一起进入 SwiGLU；
- M8192 使用 128×128 tile 与四个 MMA warps。每 lane 的两个 FP32 accumulator 本身需要
  约 256 registers，尚未计入 A/B/SF/control，因此主 spill 在当前 tile/warp layout 下是结构性的。

SASS program order 进一步给出两个候选 bundle：

- `54×STL.64 = 108 words/lane` 位于 Gate final OMMA drain 与 Up 之间；首次对应 LDL 位于最后一个
  Up OMMA 之后。它高置信对应“first-produced accumulator 跨 second GEMM 停泊”；
- 余下 `14×STL` 在 Gate reload 已开始后的 activation 展开区出现，更像 secondary register eviction。

这是待证模型，不是最终结论。

## Hypotheses and Arms

### Hypotheses

- **H108 attribution**：108-word bundle 属于 first-produced FP32 accumulator 跨第二个 FC1 GEMM 的
  结构性 lifetime，而不是 Gate 数学本身。
- **H14 attribution**：14-word tail 来自 activation 使用独立 FP32 destination `tRS_rD` 造成的
  secondary pressure。
- **H14 criticality**：消除该 14-word tail 后，未插桩 latency 的改善大于 paired noise threshold。
- **H108 criticality**：out of formal scope。在保持 FP32 数学、tile/layout 与 task schedule 不变时，
  目前没有已知 clean arm 可以单独消除主 bundle；本实验必须把该项写成 unresolved。

### Staged arms

| Arm | Change | Role | Formal performance use |
|---|---|---|---|
| `baseline` | exact production kernel | 唯一 causal anchor | Yes |
| `activation_in_place_up` | gated activation 写回已经完成消费的 `up_acc` fragment，并直接由该 fragment 转 BF16/STSM；移除独立 FP32 `tRS_rD` destination | primary，目标 14-word secondary bundle | Yes，前提是全部 identity/work gates 通过 |
| `activation_in_place_gate` | 同一数学与 dataflow，但逐元素写回已经完成消费的 `gate_acc` fragment | primary 未命中时唯一一次 compile-only fallback | Yes；最多只选择一个 in-place arm 进入 benchmark |
| `up_first_attribution` | 交换 FC1 Up/Gate semantic order，并同步交换 weight slice/pipeline ownership；保持 OMMA work、tile/layout、barrier count | 观察 108-word bundle 是否跟随 first-produced accumulator | Attribution only；不用于证明 spill speedup |

in-place arms 不抽成 `@cute.jit` helper，避免 accumulator pass-by-value 改变 codegen。单纯移动
`make_rmem_tensor` 或 lexical scope 不是有效变体；必须由 binary/local-traffic gate 证明真实 lifetime 改变。
primary 若未完整移除目标 14-word bundle，才编译一次 fallback；禁止继续搜索更多 codegen variants。

## Fixed Runtime Contract

| Field | Locked value |
|---|---|
| Case | `M=8192, E=256, H=2048, I_tp=512, topk=8, SwiGLU, BF16 output` |
| Fixture | exp_002 `fixture.py`, routed seed `10218`; identical input/routing/weights/scales for every arm |
| Source | FlashInfer `074d93e4aa54c75bee1b3dfdb39b7f075a3ff2af`; production kernel SHA `94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106` |
| CUTLASS | `b46b16d003484063bca4ed365e44095c4c6ed633` |
| Container | `nvcr.io/nvidia/pytorch:26.05-py3@sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba` |
| Python deps | CUTLASS DSL 4.6.0 tree `32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74` |
| Compiler | production/default CuteDSL compiler mode；unset `CUTE_DSL_COMPILER_OPT` 与所有 exp_003/IKET overlay env |
| JIT | fresh, separate root per arm; keep IR/PTX/cubin/SASS; no cross-arm cache reuse |
| GPU | one leased 5KP UUID for all arms; no foreign process; record clocks/power policy/driver/nvcc/ptxas |
| Launch | outer CUDA Graph; grid/block `1×1×110 / 160×1×1`; same helper topology and one fused main node |

Fresh baseline 与所有 candidates 必须在同一轮、同一 GPU、同一 toolchain、独立 Python process 中重建，
防止 module/JIT cache 串臂；exp_002/exp_003 只作历史证据。每个 overlay 保存 exact diff/reverse-patch gate，
benchmark cubin 与 NCU cubin 必须用 SHA-256 闭合。

## Phase 0: Static Qualification Before Timing

每个 arm 从 fresh JIT dump 提取 resource、SASS 与 CFG。先验证 baseline 重现：

- `STACK=488 B/thread`；
- `122×LDL.LU`, `54×STL.64 + 14×STL`；
- 根据 122-word/task 模型，fresh NCU 应重现 local load/store 各 `4,950,272 sectors`；
- grid/block、task model 与 selected semantic opcode projection 与 production 一致。

Candidate gate：

1. primary `activation_in_place_up` 只有完整移除原 14-word tail、且没有在其他 offset/PC 生成 replacement
   bundle 时才进入 formal benchmark；否则只编译一次 `activation_in_place_gate` fallback。被选中的 arm
   若恰好消除 14 words，预期 `STACK≈432 B/thread`，local sectors 每方向减少
   `14 × 40,576 = 568,064`；实际以 static width 与 dynamic closure 为准。
2. `up_first_attribution` 不只看 PC 相对位置。必须从保留的 compiler IR/PTX/SASS 建立
   `first accumulator SSA/fragment producer → STL offsets → later LDL → activation operand` def-use 链；swap 后
   该链必须从 Gate 变为 Up。无法跨层闭合时，H108 只能保留为 source+program-order inference。
   若它消除或新增大量 non-local instructions/resources，归因控制无效。
3. 不要求 full SASS/CFG identity，因为实验目标就是改变 codegen；但 OMMA/TMA/LDSM/BAR/atomic/global
   selected counts、Tensor work、task schedule、tile/warp layout 与 launch geometry 不得发生未解释漂移。

partial reduction、同宽 replacement bundle、跨层 def-use 不闭合或 static/dynamic delta 不闭合，都不得升级为
attribution；任一 candidate 没有完整改变目标 spill bundle，不得继续用 latency 结果“检验 spill”。

## Correctness and Work-Identity Gate

每个 arm 单独对与 exp_002 相同的 quant-aware dequantized reference 通过：cosine `>=0.999`、
relative-L2 `<=0.02`、max-abs `<=0.08`、所有 token finite/nonzero。

另外先执行两次独立 baseline output 建立 atomic self-drift，再对 formal candidate 施加更严格的
candidate-vs-baseline gate。令 `cosine_loss=max(0, 1-cosine)`，`self_*` 为两次 baseline 的误差；
candidate 分别与两份 baseline output 比较并取最差值，必须同时满足：

```text
cosine_loss <= min(1e-5, max(1e-6, 3 × self_cosine_loss))
relative_L2 <= min(0.002, max(1e-4, 3 × self_relative_L2))
max_abs     <= min(0.02,  max(0.002, 3 × self_max_abs))
token_relL2_p99 <= min(0.01, max(0.001, 3 × self_token_relL2_p99))
```

任一 baseline self-drift 已超过上述 hard cap 时，formal numerical comparison 无效，不能放宽阈值。

还必须保持：

- `task_count=2536`、task table/routing/workspace invariants；
- dynamic Tensor instructions `31,162,368` 与 FP4 Tensor ops `510,564,237,312`；
- grid/block、CUDA Graph boundary、material node count、helper topology；
- input/routing/packed weights/scales 与 output dtype/shape。

使用 BF16 parking、额外 GEMM、改变 tile/task schedule 或改变数学精度的 arm 即使 correctness 通过，
也只能是 diagnostic-only。

## Phase 1: Uninstrumented Paired Benchmark

只有 `baseline` 与通过 static/work gates 的 formal candidate 进入 authoritative timing：

- graph capture、JIT、allocation 与 correctness 在 timing 外；
- graph 内 external CUDA events；每次 replay 前、timed interval 外 flush `192 MiB` L2；
- 5 warmups；每 repeat 50 iterations；共 5 repeats；
- repeat 内交替 `baseline→candidate` / `candidate→baseline`，保留全部 raw samples 与顺序；
- latency 只认 uninstrumented CUDA-event median；NCU duration 不替代 benchmark。

定义：

```text
spread_arm = (max(repeat_mean) - min(repeat_mean)) / median(repeat_mean)
S = max(spread_baseline, spread_candidate)
T = 3 × S
speedup = baseline_median / candidate_median - 1
```

material improvement 必须同时满足 `speedup > T` 且至少 `4/5` paired repeats 同方向。exp_002 的
`S≈0.14%` 只说明预期量级；本轮 T 必须由 fresh samples 计算。

## Phase 2: Targeted NCU and SASS Closure

只 profile benchmark 已锁定的 exact cubin，每个 arm 一个 fused main launch。最小证据：

- binary：`REG / STACK / SHARED / LOCAL`、`LDL/STL/STL.64` count/width/offset/program order、
  compiler `SpillRefill` annotations；
- dynamic local：local load/store sectors、executed spill read/write instructions；
- work gate：Tensor instructions、FP4 Tensor ops；
- resource/schedule：Achieved occupancy、Eligible warps、Issue active、TC subpipe active；
- stalls：Long/Short scoreboard、Wait、Barrier；必要时用 SourceCounters/InstructionStats 对 local PC 做
  scoped diagnostic，不把 PC samples 当成 exact phase time。

若消除 `Δwords`，每方向预期：

```text
Δlocal_sectors = Δwords × 2536 tasks × 4 MMA warps × 4 sectors
               = Δwords × 40,576
```

static width、executed spill instructions 与 dynamic local sectors 必须闭合。只看到 stack frame 变小不足以
接受 attribution；local footprint 也不得写成 DRAM bytes。

## Decision Table

| Observation | Verdict |
|---|---|
| 被选中的 in-place arm 完整消除 14-word bundle、无 replacement、dynamic delta 闭合，speedup `>T` 且 `4/5` 同向 | 接受 H14 attribution 与 H14 material criticality |
| 14-word bundle 消除且无新 resource/sync tax，但 speedup `<=T` | 接受 attribution；结果为 non-material（若明显变慢则同时记录 regression），推翻 secondary spill 的 P0 地位 |
| speedup `>T` 但少于 `4/5` repeats 同向 | 性能不稳定，H14 criticality inconclusive |
| 14-word 只部分减少、在其他 PC/offset 生成 replacement，或 static/dynamic delta 不闭合 | H14 attribution inconclusive；不作 spill latency 归因 |
| local 降低但 occupancy/resource 同时改变 | 只称 spill + resource/latency-hiding 联合效应，不称纯 local-load latency |
| latency 改善但目标 local traffic 不降 | 不得归因于 spill |
| `up_first_attribution` 的跨层 def-use 证明 108-word bundle 跟随 first-produced accumulator | 接受 H108 structural lifetime attribution；不等于接受 criticality |
| formal arms 无法移除 108-word bundle | H108 criticality 保持 unresolved；不得用混杂 diagnostic arm 升级为 formal |
| candidate 未实质改变目标 spill | 该变体没有测试到假设，结果为 inconclusive |

## Stop Conditions

- fresh baseline 不重现 `488 B / 122 words / 4,950,272 sectors per direction`：先修基线，禁止拼接历史数据；
- correctness、task count、Tensor work、grid/block、graph topology 或 fixture identity 任一漂移：formal comparison invalid；
- benchmark 与 NCU cubin hash 不同：丢弃 profiler 证据；
- candidate 出现未解释的 non-local SASS、barrier、SMEM、tile/layout 或 task-schedule drift：降级 diagnostic-only；
- 任一 arm spread `>5%`、foreign GPU process 或 clocks/power drift：停止；spread `>1%` 时先排查并只重跑一次；
- NCU 缺 dynamic local sectors 或 executed spill counters：static SASS 不能单独回答 criticality；
- 不启动 IKET、PerfSim、DRPI、PerfBot 或 BF16/SMEM parking；当前 formal scope 只覆盖 H108
  attribution 与 H14 criticality。

## Required Outputs

所有脚本、manifest、raw samples、binary/NCU evidence 与报告放在本实验 `results/`：

- `results/validation.manifest.json`：环境、source/overlay/JIT/cubin、fixture、correctness 与 dispatch identity；
- `results/benchmark_raw.csv` / `benchmark_summary.csv`：未插桩 paired timing；
- `results/static_spill_evidence.json`：两个 spill bundle 与 candidate delta；
- `results/ncu/spill_metrics.csv`：dynamic local/work/resource/schedule/stall evidence；
- `results/result.md`：按 decision table 给 attribution、criticality 与 unresolved boundary。

Raw profiler artifacts不可覆盖；只有 compact evidence 与读者报告进入 git。

## Plan Review

**Date**: 2026-07-16
**Reviewer**: subagent `/root/exp004_plan_review`

**Verdict**: ✗ Misaligned

**Gaps + suggested fix**:

- Formal arm 只触及 14 words，原 Goal 却暗示要判断 108-word criticality：将 H108 收窄为
  attribution-only，明确其 criticality unresolved，不用 BF16 parking 冒充 formal arm。
- 无 lineinfo 时只看 bundle 位置不能证明 accumulator 身份：加入 IR/PTX/SASS 跨层 def-use 闭合；
  无法闭合则保留 inference。
- 宽松 reference gate 可能放过语义漂移：加入 baseline atomic self-drift 与严格
  candidate-vs-baseline 数值门。
- primary 可能未命中 14 words，decision tree 也遗漏 partial/replacement/mismatch：预注册唯一一个
  同义 in-place fallback，并补齐所有 middle-case 判定，禁止把不闭合结果升级为归因。
