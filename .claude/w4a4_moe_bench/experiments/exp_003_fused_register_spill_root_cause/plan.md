# Experiment 003 Plan: Fused Register-Spill Root Cause

Status: **closed (2026-07-17)**。

本实验由原 `exp_004` 重编号而来。已采 artifact 中的 `exp004.*` schema、绝对路径、lease ID、命令与 hash
都是历史证据身份，保持原样；早先的 latency/criticality 预注册计划保存在
[plan_legacy_exp004_criticality.md](plan_legacy_exp004_criticality.md)。结论见
[results/result.md](results/result.md)。

历史 `static_spill_evidence.json:h108_boundary` 仍写有旧文件名 `spill_localization_evidence.json`；为保持
artifact hash 未改写。当前 canonical replacement 是 `results/spill_root_cause_evidence.json`，映射记录在
validation manifest。

## Goal

解释 fused kernel register spill 为什么形成；源码与 SASS 定位只是建立因果机制的证据手段：

```text
stack/local symptom
  → exact spill/reload PC + stack slot
  → producer + semantic value class
  → physical-register reuse
  → reload + original consumer
  → overlapping live sets / resource-conflict mechanism
```

Register spill 按项目约束属于 P0 hard failure，不要求本实验先证明 latency 影响。

## Non-goals

- 不证明 spill 对 latency 或 TC cadence 的因果影响。
- 不报告 speedup。
- 不把 stack frame、local traffic 或单条 `STL/LDL` 称为源码 root cause。
- 不提出 production optimization。

## Locked Evidence Identity

| Field | Locked value |
|---|---|
| Case | `M=8192, E=256, H=2048, I_tp=512, topk=8, SwiGLU, BF16 output` |
| Kernel / launch | `MoEDynamicKernel`, grid `1×1×110`, block `160×1×1` |
| Production source SHA-256 | `94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106` |
| Baseline cubin | `9313fcbc0dd686f0684705e869fdd227608ac83ca43c1dc99d203f8e7143ca79` |
| Baseline SASS | `34b4c38161642a27ca6b4ec41ffad0bd70f6ff99fd8118997a4b2416c5e3abba` |
| Baseline NCU | `3367df71ef0c3b750c03c60436d100a994863271016995c76f21195dd9eaaea8` |
| Up-first cubin | `691ca03362e2e1efd7a2ad1af2d9074a38e5b809f7f0d99a6239e41f4009fee6` |
| Up-first SASS | `b7b236276e27acb2f7e0908f01df5e864ff2863d9a817aea67720026e55a6f98` |

## Closure Contract

| Scope | Required evidence | Final state |
|---|---|---|
| Main 108 | 27 OMMA vectors、108-word roundtrip、108/108 reload first-use、Python/MLIR/PTX def-use | `physical_formation_mechanism_closed` |
| Tail 14 | 14 条 producer→store→reuse→reload→consumer 物理链与 activation temporary reuse | `physical_formation_mechanism_closed_source_identity_partial` |
| Tail 9 scalars | unique source SSA / full virtual→physical map | `deferred_reprofile_after_main_change` |
| Latency / TC cadence | matched no-spill counterfactual | 本实验不判定 |

Main 108 的物理形成机制：first-pass FP32 accumulator 跨完整 second-pass FC1 保活并与后续 working
set 重叠；在观测到的 `255 registers/thread` allocation 下，compiler 将 108 words/lane 暂存到 local。
Production source/IR program order 将 first pass 解释为 Gate、second pass 解释为 Up，因此 Main 对应
`gate_acc` 跨 Up FC1 保活；这是高置信 program-order inference，不是 compiler-certified slot attribution。

Tail 14 的物理形成机制：activation 入口每个参与计算的 lane 仍有 5 个 second-pass accumulator register
values 和 9 个 index/address/control scalar 存活，activation temporaries 复用其物理寄存器，因此 allocator
执行 save/reuse/restore。Baseline source order 将 second pass 解释为 Up；9 个 scalar 的唯一 source SSA 是
非阻塞 residual。Main 改动后 register allocation 会重新生成，届时重新 profile。

## Outputs

- `results/spill_root_cause_evidence.json`：逐 PC/slot 问题点证据。
- `results/validation.manifest.json`：证据身份与闭合状态。
- `results/result.md`：读者报告。

统一重建入口：

```text
/home/scratch.xiy_gpu/local/bin/python3.10 run_exp003.py \
  --results <results> analyze-root-cause \
  --baseline-sass <baseline.sass> --up-first-sass <up-first.sass> \
  --baseline-mlir <baseline.mlir> --baseline-ptx <baseline.ptx> \
  --up-first-ptx <up-first.ptx> --source <production-kernel.py>
```

入口要求 Python >= 3.10；上面的 interpreter path 已在本次收口验证。入口按
`root-cause evidence → validation manifest → report` fail-closed 生成。

## Follow-up Hypothesis

“Register spill 是 TC cadence 偏低的主要贡献者”目前仅为待验证假设。下一实验必须构造
correctness-equivalent 的 reduced/no-spill arm，锁住 Tensor work、launch topology 与 task schedule，
再比较 latency、TC subpipe active、Issue Active、warp stalls 和 local traffic。

旧 cadence 实验已删除：其 IKET marker 将 stack 从 `488` 改为 `432 B/thread`，测量扰动了目标机制；
后续 cadence 插桩必须先通过完整 resource/SASS identity gate。
