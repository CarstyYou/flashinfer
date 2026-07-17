# Experiment 004 Plan: Fused Spill Problem-Point Localization

Status: **executed / physical mechanism localized / source-value bridge unresolved (2026-07-16)**。

目录名保留历史的 `fused_stack_spill_criticality`，避免破坏已采集 artifact 路径；它不再代表实验问题。
本计划是定位阶段的 superseding addendum；早先预注册的 latency/criticality 计划原文保存在
[plan_legacy_criticality.md](plan_legacy_criticality.md)，不会被事后改写。结论见
[results/result.md](results/result.md)。

## Goal

找到 fused kernel 中 register spill 的具体问题点：

```text
stack/local symptom
  → exact cubin + spill/reload PC + stack slot
  → original producer and semantic value class
  → physical-register temporary reuse
  → reload and original consumer
  → live-range/resource-conflict mechanism
  → source-value boundary
```

本项目将 register spill 定义为 P0 hard failure，不要求先证明 latency 影响。但 P0 严重性不能绕过
问题点定位：在 source value、producer/consumer 与 live interval 闭合前，不给出具体优化方案。

## Non-goals

- 不证明 spill 对 latency 的影响，不报告 speedup。
- 不用 pass-order correlation 代替 accumulator-lifetime 归因。
- 不把 stack frame、local traffic 或某条 STL/LDL 本身称为源码 root cause。
- 不在问题点未闭合时提出 source、schedule、tile 或 pipeline 改法。

## Locked Evidence Identity

| Field | Locked value |
|---|---|
| Case | `M=8192, E=256, H=2048, I_tp=512, topk=8, SwiGLU, BF16 output` |
| Kernel | `MoEDynamicKernel` |
| Launch | grid `1×1×110`, block `160×1×1` |
| Production source SHA-256 | `94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106` |
| Baseline cubin | `9313fcbc0dd686f0684705e869fdd227608ac83ca43c1dc99d203f8e7143ca79` |
| Baseline SASS | `34b4c38161642a27ca6b4ec41ffad0bd70f6ff99fd8118997a4b2416c5e3abba` |
| Baseline NCU | `3367df71ef0c3b750c03c60436d100a994863271016995c76f21195dd9eaaea8` |
| Up-first cubin | `691ca03362e2e1efd7a2ad1af2d9074a38e5b809f7f0d99a6239e41f4009fee6` |
| Up-first SASS | `b7b236276e27acb2f7e0908f01df5e864ff2863d9a817aea67720026e55a6f98` |

## Localization Procedure

1. 从 baseline SASS 枚举全部 `STL/STL.64` 与对应 stack slots。
2. 对每个 slot 闭合 `producer → STL → physical-register reuse → LDL → original consumer`。
3. 按 semantic role 分类每个 live value，禁止把 14-word tail 预设为同质 accumulator。
4. 对 108-word main bundle 验证所有 108 个 reload value 的首个使用。
5. 用 `up_first` 只做交叉验证：确认哪个 bundle 保留、哪个 bundle 消失；它不能单独命名源码原因。
6. 在 Python/MLIR/PTX 各层闭合内部 def-use，并明确 virtual-to-physical mapping 的缺口。

## Acceptance States

| State | Required evidence | Allowed conclusion |
|---|---|---|
| Symptom-confirmed | stack/local footprint | 发生 spill |
| Instruction-localized | exact cubin + PC/opcode/slot | spill 指令在哪里 |
| Semantic-localized | phase/warp role + value class | 哪类数据被 spill |
| Mechanism-localized | producer/reuse/reload/consumer + live-range conflict | 物理问题机制 |
| Source-mechanism-localized | compiler-certified IR/PTX value → physical reg/slot → source live interval | 才能进入具体优化设计 |

本轮已闭合两个 SASS physical mechanism：main 108 是第一段 FC1 收尾时随 producer 逐步 spill、
跨完整第二段 FC1 保活的 accumulator；tail 14 是 activation 入口的 mixed live-set
save/reuse/restore。tail 中 9 个 scalar 的唯一 source SSA 及 virtual-to-physical allocation bridge 尚未闭合。因此整体状态为
`partially_mechanism_localized`，optimization gate 保持关闭。

## Required Outputs

- `results/spill_localization_evidence.json`：机器可审计的逐 PC/slot 物理链与跨层边界。
- `results/validation.manifest.json`：artifact 身份与定位状态。
- `results/result.md`：读者报告，只写已定位问题点、证据边界与下一步取证。

统一重建入口：

```text
run_exp004.py --results <results> localize \
  --baseline-sass <baseline.sass> --up-first-sass <up-first.sass> \
  --baseline-mlir <baseline.mlir> --baseline-ptx <baseline.ptx> \
  --up-first-ptx <up-first.ptx> --source <production-kernel.py>
```

该入口按 `localization evidence → validation manifest → report` 顺序 fail-closed 生成；partial
localization 写入状态字段，生成成功仍返回 `0`。

## Stop Condition

如果缺少 compiler-certified virtual value → physical register/stack slot → source live interval 映射，
报告必须停在“source-value partially localized”，下一步只能请求该证据，不得生成优化建议。
