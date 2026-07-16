# 实验 003：Fused TensorCore Cadence Breakdown

## 1. 问题与证据边界

本实验调查 `M=8192` CuteDSL Fused MoE 的低 TensorCore activity，区分两个候选机制：

- 必要的 planned TC-off：route/pack、activation/quant、epilogue/scatter 等 non-TC 工作；
- `FC1 Gate / FC1 Up / FC2` 内反复出现的 TensorCore starvation。

生产环境锚点来自 exp_002：Fused latency `1782.547 us`，比 CUTLASS Chain 慢 `6.55%`；
Fused main 的 `TC subpipe active=25.73%`、`Issue active=19.95%`。本实验只分析三个
selected-CTA IKET trace（CTA z=`0/55/109`），时间单位是 `raw timestamp units`。IKET duration
不是生产 latency，也不能与 NCU active-cycle 百分比相加、相减或互相换算。

完整环境、正确性、binary identity、capture digest 与 analyzer provenance 见
[`validation.manifest.json`](validation.manifest.json)。

## 2. 判定

**Formal verdict：`inconclusive`。** 有三个独立阻断：

1. marker candidate 改变了 stack resource：control `488 B/thread`，candidate `432 B/thread`；
2. sampling coverage 只有 steady/full 与 steady/partial 两个 stratum 通过，early/tail 不足；
3. diagnostic interval 中 `unclassified` 的 95% 区间为 `73.90%–74.75%`，远高于 formal
   判定允许的 `20%` 上限。

**受限推断：** 在已经显式插桩且通过 PC/SASS 闭合的 above-calibration wait/barrier 范围内，
没有观察到 pervasive repeated starvation；但大量 `unclassified` 时间使本实验不能排除其他
starvation 机制，也不能证明 planned TC-off 主导。

## 3. 关键证据

### 3.1 Binary 与事件闭合

| Gate | 结果 | 说明 |
|---|---|---|
| Correctness / task model | Pass | 三个 profile PID 均通过 quant-aware oracle、workspace 与 task invariants |
| Dynamic ordinal closure | Pass | 每个 warp/slice 的 Gate、Up、FC2 各 `1024` 个 range events |
| Selected semantic opcode projection | Pass | OMMA/TMA/LDSM/BAR/atomic/global opcode counts 在 control/candidate 间相同 |
| PC/SASS wait/barrier closure | Pass | 三个 capture 均有 24 个 static sites 通过；marker 指令不作为语义证据 |
| Resource identity | **Fail** | `REG/SHARED/LOCAL` 相同，`STACK 488→432 B/thread` |

这里只证明 selected exact opcode-token counts 相同；没有证明 full SASS 或 full CFG identity。
详细结果见 [`binary_gate.json`](validation/binary_gate/binary_gate.json) 和
[`pc_sass_gate/`](validation/pc_sass_gate/)。

### 3.2 三个 CTA 的 diagnostic interval

下表是 analyzer 在现有三个 CTA 上输出的诊断区间。由于 coverage fail，
`population_weighted_point=null`；这些数字不是 whole-launch 或 population estimate。

| Bucket | 3-CTA diagnostic 95% interval |
|---|---:|
| Tensor issue | 6.17%–7.41% |
| Planned non-TC | 16.84%–17.73% |
| Explicit SASS-verified starvation | 1.27%–1.35% |
| Orchestration | 0.31%–0.32% |
| Unclassified | **73.90%–74.75%** |

Coverage 明细与 bootstrap 配置见
[`weighted_phase_shares.json`](derived/weighted_phase_shares.json)。

### 3.3 QMMA-to-QMMA gap

所有 gap 都只由同一 warp 内相邻 QMMA range 构造；表中的样本不能跨并行 warp 相加为 elapsed time。
empty-marker calibration p95 在四个 MMA warp 上均为 `32` raw units。

| Phase | Gap count | p50 / p95 / p99 / p99.9 / max | `>calibration` | `>2×` | `>4×` |
|---|---:|---:|---:|---:|---:|
| FC1 Gate | 286,440 | 0 / 32 / 96 / 224 / 928 | 3.199% | 2.275% | 0.305% |
| FC1 Up | 286,440 | 0 / 32 / 128 / 224 / 704 | 3.231% | 2.409% | 0.362% |
| FC2 | 282,240 | 0 / 32 / 32 / 352 / 544 | 0.541% | 0.474% | 0.425% |

三个 phase 的 p95 都没有超过 marker calibration，FC2 的 p99 也仍为 `32`。这支持上面的受限推断：
已闭合的 wait/barrier 不是普遍、反复的 cadence interruption。Gate/Up 的长尾 gap 和 FC2 的极少数
长 gap 仍主要落在 `unclassified`，因此不能从本表判断它们是 planned work、scoreboard、dependency
还是其他机制。完整分布与 bucket raw-duration totals 见
[`cadence_summary.json`](derived/cadence_summary.json)。

## 4. 收口与下一步

计划原本允许在 coverage 不足时继续补采最多九个 CTA。更多 CTA 可能改善 early/tail sampling
coverage，但不能恢复已经被 `STACK 488→432` 永久取消的 binary comparability，也不能把本轮
diagnostic 升级为 formal result。因此本轮选择在三个 CTA 停止补采并明确判为 `inconclusive`；
这是一项收口决策，不表示 coverage 已经闭合或无法改善。

如果未来继续调查 TC cadence，最小下一步应先给 `unclassified` 的 inter-QMMA 与 phase-transition
区间增加可验证的语义边界，再决定是否扩展 CTA population。当前按项目优先级独立建立
`exp_004_fused_stack_spill_criticality`，用未插桩 benchmark + NCU + lifetime/reorder 单变量实验验证
122-word/lane stack roundtrip 是否来自 Gate accumulator 跨 Up 保活，以及它是否位于关键路径。

### Analyzer provenance

三个 capture manifest 记录的 analyzer SHA 是 `7d75fdf848…`；正式 derived artifacts 使用后续
parser-only 修正版，并由当前 `827e87f356…` analyzer 完整重放验证，公共输出逐字节一致。已知的
post-capture 修正包括：只在唯一无歧义时处理 equal-timestamp range boundary，以及把 ephemeral
Docker hostname 排除出 runtime identity；GPU UUID/source/provider/image/toolchain/JIT drift 仍会被拒绝。
capture-time analyzer 源码没有保留，无法对该版本做 cryptographic source diff；raw trace 未发生变化，
该 provenance drift 也不能用于恢复 formal validity。
