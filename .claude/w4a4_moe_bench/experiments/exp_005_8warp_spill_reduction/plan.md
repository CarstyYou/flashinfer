# exp_005：8-Warp Gate/Up Spill Reduction

Status: **completed / Candidate A rejected for residual spill (2026-07-17)**

## 1. Goal

实现并验证 `kernel_design.md` 的 Candidate A：保持 `M128×N128×K128`、Gate→Up 串行顺序和完整
fused MoE semantic work，把 compute role 从 W0–W3 扩为 W0–W7，TMA role 从 W4 移到 W8，目标是消除
production `MoEDynamicKernel` 的 register spill。

成功必须同时满足：

1. candidate 对固定 fixture 正确；
2. candidate binary 为 `9 warps / CTA`、8-warp tiled-MMA，且 Tensor work 没有被删减；
3. `STACK=0 B/thread`，无 compiler spill/refill local SASS，NCU dynamic spill/local traffic 为 0；
4. 保存 baseline/candidate 的 latency 与 NCU 证据，判断去 spill 后的 TC cadence、schedule 与性能变化。

本实验评估的是 **8-warp whole-kernel design 是否解决 spill 并带来净收益**。由于 block geometry、P0–P4
参与线程数和 T3/T4 ownership 都会变化，不把任何 latency/TC 变化单独归因为 spill；若要证明 spill 的
纯因果贡献，需要另做 matched counterfactual。

## 2. Arms and Independent Source Contract

| Arm | Source | Intended change |
|---|---|---|
| `baseline_4warp` | production `moe_dynamic_kernel.py` 的 immutable copy | byte-identical control |
| `candidate_8warp_serial_v0` | experiment-owned full-module overlay | `num_mma_warps: 4→8`；`atom_layout: (2,2,1)→(4,2,1)` |

Production file不得修改。overlay generator 必须锁定 production SHA-256
`94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106`，使用 exact-match transform，
保存 full overlay、unified diff 和 SHA-256。每个 arm 在独立 Python process、独立 import overlay 和 fresh JIT
root 中运行，禁止 module/JIT cache 串臂。

Candidate 的 planned execution contract：

```text
CTA = W0–W7 compute + W8 TMA = 288 threads
MMA API = MmaMXF4NVF4Op
atom_layout = (4, 2, 1)
Gate N128 → pass_gate → Up N128 → SwiGLU/Q1 → FC2/scatter
```

以下 source-derived 自适应行为必须逐项验证，不能只因代码引用 `num_mma_warps` 就假定正确：

- TMA pipeline consumer group 与所有 named barrier participant；
- accumulator/tiled-copy ownership，W0–W7 是否共同写满 `sC`；
- Q1 的 `tidx`/stride-256 覆盖是否无漏写/重复写；
- FC2 epilogue 后 W0–W3 scatter 四个 `64×64` quadrant，W4–W7 `warp_epi_rows=0` 但仍参加 barrier；
- T0 metadata cache 和 P0–P4 route/pack 在 288-thread CTA 下的 coverage；
- grid、task population、task schedule、OMMA work 与 output atomic combine 是否完整。

不得合入 Candidate B、subtile、Gate/Up 并发、SMEM handoff、epilogue compact、activation rewrite、stage count
或 fast-math 改动。

Candidate A 版本规则：`v0` 只能包含上述两行 exact transform。若 correctness/launch gate 暴露 8-warp
plumbing 缺陷，后续版本只允许修 barrier participant、T3 tiled-copy/Q1 ownership、T4 scatter participation/mapping
或 route coverage，且必须保持 8 compute warps + W8 TMA、`M128×N128`、Gate→Up 串行和相同 semantic work。
每版 overlay immutable、独立编号；最终用于结论的版本必须从 Gate A 到 Gate E 全量重跑，禁止跨版本拼证据。

### 2.1 Accepted comparison registry

- Locator: `comparison_registry.json`
- Accepted revision: `r1`
- Accepted content SHA-256: `d31095521339d30ead54fd9fbe10407d6585728f477b01989cb3c457d2f39c8f`
- Relationship: `R1_candidateA_v0_vs_baseline`
- Boundary: paired complete fused-MoE launch；changed variable 是不可拆的 8-warp MMA ownership bundle。

Registry 在任何 GPU capture 前锁定了 logical/timed boundary、controlled invariants、allowed resource/schedule
consequences 和 prohibited claims。若关系边界变化，必须先生成新 accepted revision，再 fresh rerun；不得事后把
controlled-invariant failure 改写为 allowed consequence。

## 3. Fixed Environment and Cases

继承 exp_002/exp_003 的 canonical environment，并 fresh 记录而不是引用历史值：

| Field | Locked value |
|---|---|
| Host class | dedicated SM120 5KP；同一 GPU UUID 跑两臂；无 foreign process |
| Container | `nvcr.io/nvidia/pytorch:26.05-py3`，digest 必须为 `sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba` |
| CUTLASS DSL overlay | canonical 4.6.0 dependency tree，记录 content SHA/module paths |
| Compiler | production/default CuteDSL compiler mode；unset IKET/marker/compiler-opt overlays |
| Shape | `E=256, H=2048, I_tp=512, topk=8, SwiGLU, BF16 output` |
| Precision | BF16 input；FC1/FC2 NVFP4 block-scaled；Q1 NVFP4 |
| Fixture | exp_002 deterministic fixture；base seed `2026`；每个 M 使用既有 routed seed contract |
| Launch | outer CUDA Graph；明确绑定 final replay `MoEDynamicKernel` node；`num_sms=110`、`max_active_clusters=110`、grid.z=110 |

Cases 只覆盖 prefill `M>=256`：

| M | Purpose |
|---:|---|
| 256 | small-prefill correctness、partial-row/tail behavior、latency |
| 1024 | intermediate task population 与调度 sanity |
| 8192 | primary spill/resource/NCU case 与主要 latency 结论 |

另外运行一个 directed route-topology fixture suite，至少覆盖：存在 empty/sparse expert、某 expert 恰好 128
routed rows、某 expert 为 129 rows（跨 tile + tail）以及 hot expert。每个 fixture 由 CPU 独立计算期望 task
multiset（expert、M tile、I slice、valid rows），验证实际每个 task 恰好消费一次；使用 expert-distinct
weight/scale 和 output sentinel 验证 Q1/scatter 没有漏写或 quadrant 丢失。该 suite 是 correctness gate，
不进入性能汇总。

manifest 必须记录 host、GPU UUID/PCI、driver、clocks/power、container digest、`nvcc --version`、
`ptxas --version`、CUDA runtime、Python、Torch、CUTLASS DSL version/path、FlashInfer commit、production/overlay
hash、JIT artifact/cubin/PTX/SASS hash、grid/block、fixture/input/output hash。

## 4. Gates and Measurements

### Gate A：fresh baseline reproduction

- fresh `baseline_4warp` 必须编译为 block `(160,1,1)`；每个 M 必须记录并断言
  `num_sms=110`、`max_active_clusters=110`、grid `(1,1,110)` 和 final replay kernel instance；
- production source hash、resource tier、OMMA family/work、correctness 与 exp_003 canonical baseline 相容；
- 若 cubin 不同，先解释 toolchain/environment drift，不把历史 NCU 与 fresh candidate 拼接。

### Gate B：candidate compile and structural identity

- block 必须为 `(288,1,1)`；TMA warp id 为 8；每个 M 必须与 baseline 使用相同
  `num_sms/max_active_clusters/grid.z=110`，并记录 final replay kernel instance；若任一项不相同则停止性能比较；
- generated MLIR/PTX/SASS 必须证明 8-warp tiled-MMA specialization 已生效；实际 SASS family 从 disassembly
  报告，不从 API 推断；
- 对比静态与动态 Tensor instruction/work、task_tail/row_counts/task_head，禁止通过少算工作获得低 stack；
- 对每个 M 保存 JIT cubin/SASS hash；若同一 arm 的 M=256/1024/8192 不是同一 binary，Gate D 必须对
  每个 distinct binary 重复 resource、static spill/refill 与 dynamic local-traffic 证明；
- compile、launch、barrier hang、illegal access 或 task population mismatch 均 fail closed。

### Gate C：correctness

每个 M 对 baseline/candidate 使用相同输入与 reference。至少两次 replay，要求：

- finite、nonzero、shape/dtype 正确；
- candidate 与 reference 通过 exp_002 canonical formal gate；同时报告 candidate↔baseline 的
  `max_abs_error / relative_l2 / cosine / percent_within`；
- cross-arm gate 使用 baseline 两次 replay 的 atomic-order self drift：每个 metric threshold 固定为
  `min(cap, max(floor, 3×baseline_self_drift))`。`cosine_loss/relative_l2/max_abs/token_rel_l2_p99`
  的 `(floor, cap)` 分别为 `(1e-6,1e-4)/(0.005,0.03)/(0.02,0.10)/(0.01,0.05)`；baseline
  self drift 超 cap、candidate self drift 或 candidate↔baseline 超 threshold 均 fail closed；
- task population、routed row sum `M×topk`、duplicate/missing task 和 output stability 通过；
- directed route-topology suite 的期望 task multiset、exact-once consumption、Q1/scatter sentinel 全部通过；
- 任一 correctness 失败时停止 performance/NCU 结论，先定位 ownership、barrier 或 scatter 问题。

### Gate D：zero-spill resource proof

Primary `M=8192` candidate 必须同时满足；若 Gate B 发现多份 distinct binary，则每份 binary 都必须满足：

- `STACK=0 B/thread`；
- static SASS 没有 compiler-annotated spill/refill local instructions；
- NCU dynamic local load/store spill traffic 为 0；
- register、SMEM、achieved occupancy 与 launch geometry 完整记录。

任何非零 spill 都表示“8 warp 方向尚未解决 P0 问题”，必须继续定位剩余 live set，不能用 latency 改善替代。

### Gate E：performance and cadence

- benchmark：同一进程外的 paired ABBA（或 seeded randomized interleave）至少 5 个 sample block；每组固定
  replay 次数，报告 median、p10/p90、CV、paired bootstrap 95% CI 和
  `speedup = baseline_us / candidate_us`；M=256/1024/8192 均测；
- 使用独占 GPU 并锁定 application clocks；若权限不允许锁钟，记录每组 clocks/power，性能 verdict 只能为
  advisory/inconclusive。预先定义 ratio band `[0.98, 1.02]`：95% CI lower > 1.02 为 faster，upper < 0.98
  为 slower，整个 CI 位于 band 内为 equivalent，其余为 inconclusive；
- NCU：M=8192 final replay node，baseline/candidate fresh matched capture；至少采集 LaunchStats、Occupancy、
  SchedulerStats、WarpStateStats、ComputeWorkloadAnalysis、MemoryWorkloadAnalysis、InstructionStats、
  SourceCounters；
- reader metrics：kernel duration、TC subpipe active、Issue Active、Achieved occupancy、register/stack、
  local traffic、Warp stalls（Wait/Long scoreboard/Short scoreboard/Barrier）与 throttle；
- NSys 只用于确认 launch/node、CUDA Graph 与 whole-kernel wall，不做 phase attribution。

若 candidate zero-spill 且正确但变慢，实验仍是有效结果：结合 occupancy、issue、stall、control-plane 与
T3/T4 extra-warp work判断 8-warp design 的负收益；不得凭 whole-kernel counter 猜具体 phase。

## 5. Decision Tree

```text
compile/layout fail
  → 修 8-warp layout/ownership；不采性能

correctness/task-population fail
  → 定位 tiled-copy / Q1 / scatter / barrier；不采性能

correct + spill remains
  → 8 warp 未解决 P0；定位剩余 spill PC/live set

correct + zero spill
  → 比较 latency + TC cadence + stalls
      ├─ faster: Candidate A 可进入后续 productionization 讨论
      ├─ flat: spill 已解决，但被额外 warp/control/schedule 成本抵消
      └─ slower: 保留证据，判断是否继续 subtile/lifetime/SMEM 方向
```

## 6. Outputs and Closure

所有生成物放在 `exp_005_8warp_spill_reduction/results/`：

- `results/overlays/`：immutable baseline/candidate source、patch、identity；
- `results/raw/<arm>/<M>/`：fixture、outputs、benchmark、JIT artifacts；
- `results/ncu/<arm>/m8192/` 与 `results/nsys/`：profiler artifacts；
- `results/manifest.json`：环境、binary、fixture、correctness、resource 与 capture identity；
- `results/result.md`：是否 zero-spill、性能/TC cadence 变化、证据边界和下一步。

正式收口前对 report 执行一次 ex-post data audit。只有 correctness、work identity、zero-spill proof 与结果
manifest 闭合后，才能把 exp_005 标为完成。

### 6.1 Execution closure

- M256、M8192 correctness 通过；M1024 independent reference 与 route/task 通过，但两臂 self-drift 和
  cross-arm cosine 均超过预注册 cap，因此 strict comparison protocol 失败，未采 M1024 性能。
- Candidate static frame 为 224 B/thread，compiler SpillRefill SASS 与 NCU dynamic spill 均非零，
  zero-spill gate 失败。
- 诊断性能：M256 按预注册 ±2% band 为 equivalent；M8192 speedup 为 9.75%。
- 正式 NCU 证据为 `canonical_v1`；缺少 section-derived spill metrics 的 `canonical_v0` 已隔离。
- ex-post data audit 已完成；`results/manifest.json` 与 `results/result.md` 已闭合，production kernel 未修改。

## Plan Review

- Date: 2026-07-17
- Reviewer: isolated `experiment-plan-review` subagent
- Verdict: **⚠️ Gaps → addressed before execution**
- Scope: single required pre-execution review；本实验不再重复 Plan Review。

审计发现并已修正：

1. 原计划只记录 `max_active_clusters/num_sms/grid.z`，未锁定两臂相同；现锁定并逐 M 断言 grid 与 final replay instance。
2. 原计划只在 M8192 证明 zero-spill，未处理按 M 产生不同 JIT binary；现按 cubin/SASS hash 去重并覆盖每份 binary。
3. 原 correctness fixture 不足以覆盖 empty、exact-128、129-tail、hot-expert route topology；现增加 task multiset
   exact-once 与 Q1/scatter sentinel suite。
4. 原性能协议缺少 order、clock、noise band 和 inconclusive verdict；现使用 paired interleave、clock gate、95% CI
   与预声明 equivalence band。
5. 原 decision tree 允许未定义的 candidate 修补并可能跨版本拼证据；现限定 Candidate A plumbing 修复范围，
   overlay immutable/versioned，final version 必须重跑 Gates A–E。
