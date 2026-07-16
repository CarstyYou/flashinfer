# Exp 002: Fused-vs-Chain Dataflow Bottleneck

## 0. 判定

在 `M=8192` primary prefill case，CuteDSL fused 比 matched CUTLASS BF16-input chain
慢 **6.55%**，尽管 fused 的完整 operator DRAM traffic 少 **35.79%**。因此本 case 已经证明
fusion 的 GMEM-elimination 收益存在，但该收益被 fused kernel 内的 resource/cadence tax 抵消。

当前最强优化点是：**缩短 FC1 Gate/Up accumulator 的同时存活范围，降低 local-memory traffic，
改善 fused kernel 的 dependency/cadence**。这是由动态 NCU 与当前 binary 的
静态 SASS 共同指向的下一实验，不是已经闭合的源码因果结论；binary 没有 lineinfo，具体 local
访问属于哪个 source object 仍需 counterfactual 验证。

本轮不启动 IKET。现有证据已经足以选择一个可证伪动作；只有该动作显著降低 local traffic、但
TensorCore cadence/latency 不改善时，才需要 IKET 区分 barrier、角色等待与 tail。

## 1. Comparison contract

| 项目 | 固定值 |
|---|---|
| Operator boundary | precomputed routing + BF16 input → online input quant/pack → FC1 → SwiGLU/requant → FC2/reduce → BF16 output |
| Shape | `E=256, H=2048, I_tp=512, topk=8, SwiGLU` |
| Target / baseline | `cutedsl_bf16_fused` / `cutlass_bf16_chain`；不含 prequant arm |
| Hardware | SM120 5KP；GPU UUID `GPU-4a286357-c999-9547-3a04-25961b1ffd08` |
| Source | FlashInfer `074d93e4aa54c75bee1b3dfdb39b7f075a3ff2af`；CUTLASS `b46b16d003484063bca4ed365e44095c4c6ed633` |
| Compiler/runtime | CUTLASS DSL `4.6.0`；container/image、Python tree 与 JIT hashes 见 [`environment.lock.json`](environment.lock.json) |
| Evidence identity | comparison group `exp002_cutedsl_vs_cutlass_bf16`；rerun `exp002-dsl460-20260716T0310Z-r1`；完整 fingerprints 见 [`evidence.identity.json`](evidence.identity.json) |
| Correctness | 两个 arms 在三个 M 上分别通过同一 quant-aware oracle gate；formal percent-within 均为 100%，见 [`correctness.json`](correctness.json) |

这是 implementation-confounded 的 backend 对比：可以判断哪种机制成为收益或税项，但不能把全部
时延差直接命名为“fusion 本身”的单一因果效应。

结论边界仅覆盖 deterministic synthetic random router、单张 5KP、上述一个 MoE shape；oracle
是 quant-aware tolerance gate 而非 bitwise equality，不能直接外推真实 Qwen routing 分布或其他 shape。

### 未插桩 benchmark authority

| M | CuteDSL fused | CUTLASS BF16 chain | CuteDSL speedup | 最大 repeat spread | 判定 |
|---:|---:|---:|---:|---:|---|
| 256 | 538.956 us | 550.739 us | +2.19% | 0.34% | control win；本轮不继续归因 |
| 1024 | 615.582 us | 577.220 us | -6.23% | 0.34% | benchmark-only loss |
| 8192 | 1782.547 us | 1665.779 us | -6.55% | 0.14% | primary loss |

原始样本与 order/interleave 在 [`benchmark_raw.csv`](benchmark_raw.csv)，汇总在
[`benchmark_summary.csv`](benchmark_summary.csv)。NSys 单 replay 的方向独立一致：M256 为
497.694 vs 509.695 us，M8192 为 1760.827 vs 1662.108 us；profiler duration 只用于解释，
不替代上表。

## 2. Observed launch topology

两边都已使用 CUDA Graph，因此不能借用 eager Python launch overhead。

| Case | CuteDSL fused | CUTLASS chain | Graph 内部空隙 |
|---|---|---|---:|
| M256 | 2 nodes：1.088 us helper + 496.414 us fused main | 9 nodes；wall 509.695 us | fused 192 ns；chain 2 ns |
| M8192 | 2 nodes：1.120 us helper + 1759.483 us fused main | 9 nodes；wall 1662.108 us | fused 224 ns；chain 0 ns |

M8192 chain 的实际顺序为：

```text
prefix(block 32.543 us → global 1.952 us → merge 12.288 us)
  → expand/online-quant 245.600 us
  → stride metadata 2.048 us
  → FC1 510.175 us
  → SwiGLU/requant 231.647 us
  → FC2 408.255 us
  → finalize 222.944 us
```

这些是 observed node intervals，不是可相加的因果 phase decomposition；PDL 造成少量 overlap，
operator wall 使用 interval union。Deep NCU targets 覆盖 chain active-time union 的 97.10%
（M8192）/98.84%（M256），未 profile 的 prefix/stride 仍保留在 NSys topology 中。
证据位于 [`nsys/`](nsys/) 与 [`ncu/deep_launch_metrics.json`](ncu/deep_launch_metrics.json)。

## 3. Logical dataflow and storage edges

| Logical phase | CuteDSL fused | CUTLASS chain | 可比较证据 |
|---|---|---|---|
| clear / histogram / prefix | fused main 内，含 grid barriers | 3 个 prefix nodes | topology；不能拆 fused phase time |
| route + input quant + pack | 仍先物化 packed A/SFA、token map/weight 到 GMEM；当前 `full_tile_publish_enabled=0`，route 与 compute 无 overlap | `expandInputRowsKernel` + metadata node，物化到 GMEM | complete-range traffic + NSys |
| FC1 → SwiGLU/requant | Gate/Up accumulators 在 RMEM/local address space；activation 经现有 SMEM `sC` 写入 FP4 `sA/sSFA` | FC1 output 写 GMEM，再由 activation node 读写 | range traffic；launch-local resource/cadence |
| requant → FC2 | `sA/sSFA` 留在 SMEM/RMEM 并跨 16 个 FC2 output tiles 复用 | activation workspace 由 FC2 node 从 GMEM 读取 | range traffic + equal TensorCore work |
| FC2 → reduce/output | 每个 I=128 slice 直接 route-weighted global reduction；4 个 slice 累加同一 output | FC2 partial workspace → finalize node | reduction request count；不能当 DRAM bytes |

当前 fused schedule 的 4-way partial-output reduction 与 NCU 精确一致：M8192 为
33,554,432 个 32-B reduction sectors，即 1,073,741,824 B request footprint。它是 measured schedule tax，
但尚无 phase-critical-time 证据，而且减少它需要增加 on-chip activation/output state；本轮只把它
保留为 measured tax，不把它排成下一优化动作。source/count model 见
[`ncu/fused_schedule_models.json`](ncu/fused_schedule_models.json)。

## 4. Fusion mechanism ledger

完整 operator-range counters 见 [`ncu/operator_comparison_v2.csv`](ncu/operator_comparison_v2.csv)。
不同 cache level 的 bytes 语义不同，不能跨行相加。

| Mechanism / tax | M256 fused vs chain | M8192 fused vs chain | 判定 |
|---|---:|---:|---|
| DRAM read+write | 503.277 vs 480.082 MB（+4.83%） | 933.777 vs 1454.170 MB（**-35.79%**） | 大 M 的 GMEM elimination 成立；小 M 不成立 |
| L2 total | +14.87% | +1.61% | DRAM 下降没有同比降低全部 cache traffic |
| LSU global T-stage footprint | +4.45% | +38.32% | fused 的 request-side work 仍重 |
| local load+store footprint | 127.926 vs 64.175 MB（+99.34%） | 316.817 vs 154.460 MB（**+105.11%**） | 明确的 fused tax；不能仅凭 counter 命名具体 source object |
| dynamic warp instructions | -15.60% | -29.00% | fused 执行更少动态指令，但仍更慢 |
| Tensor instructions / FP4 ops | 完全相同 | 完全相同 | Tensor work equal；不表示 scalar/metadata work 全部相同 |
| Graph nodes | 2 vs 9 | 2 vs 9 | boundary elimination 成立；当前 graph 内无可量化 idle-gap 收益 |
| Useful in-kernel overlap | 未证明 | 未证明 | source 中“融合”不等于 overlap evidence |

M256 fused 的 +2.19% control win 不能归因为 DRAM elimination，因为该 case 的 DRAM traffic
反而高 4.83%；本轮按约定不继续分析 fused 更快的 regime。

## 5. Primary bottleneck

M8192 launch-local NCU 显示：

| Metric | Fused main | CUTLASS FC1 | CUTLASS FC2 |
|---|---:|---:|---:|
| registers/thread (allocated) | 255 (256) | 168 (168) | 168 (168) |
| shared memory/CTA | 92,160 B | 90,112 B | 90,112 B |
| CTA warps / active-warps metric | 5 warps；10.42% | 12 warps；22.92% | 12 warps；22.92% |
| Tensor pipe active | 25.72% | 62.69% | 38.30% |
| eligible warps/cycle | 0.202 | 0.217 | 0.284 |
| warp stall wait | 29.25% | 19.22% | 17.88% |
| warp stall long scoreboard | 19.01% | 12.32% | 19.67% |

Fused 使用 1 CTA/SM，但一个 CTA 只有 4 个 MMA warps + 1 个 TMA warp；255-register tier 和
92 KB SMEM 都各自把 residency 限在 1 CTA/SM。它完成与 chain FC1+FC2 完全相同的
31,162,368 条 Tensor instructions。Fused 的 Tensor-active 比例包含 route/quant/reduce 等非 TC
阶段，而 chain 的两个比例各自是 GEMM launch scope；这里只把差异作为 cadence signal，不能跨
launch 平均或视为纯 apples-to-apples SOL。

源码中一个 `128×128` FP32 accumulator 分布到 128 个 MMA threads，平均就是每 thread 128 个
FP32 values；Gate 与 Up 同时 live 的 payload 已达到 256 registers/thread，尚未计入 operands 与
control state，结构上超过 255-register ceiling。

本次 DSL 4.6 binary 的 captured static SASS 包含 54 条 `STL.64`、14 条 `STL` 与 122 条
`LDL.LU`。54 条 `STL.64` 覆盖 108 个 32-bit stack slots，另 14 条 `STL` 覆盖 14 个 slots；
122 个 stored slots 都有后续 static reload。按“每 compute task 由 128 个 MMA threads 执行一次”
建立的当前-binary execution model，在 M256 和 M8192 都**精确重建** NCU 的 local load/store
sectors（残差均为 0）。因此动态 local traffic 可绑定到这组 compiled stack roundtrip；根据源码
program order，把其中 108 个 slots 映射为 Gate 跨 Up 的状态仍是无 lineinfo 的高置信 inference，
不是直接 source attribution。证据见 [`ncu/static_local_sass.json`](ncu/static_local_sass.json) 与
[`ncu/fused_schedule_models.json`](ncu/fused_schedule_models.json)。

```text
M256  : 122 words × 4 B × 128 threads × 1024 tasks = 63,963,136 B / direction
M8192 : 122 words × 4 B × 128 threads × 2536 tasks = 158,408,704 B / direction
```

综合判定：**大 M 的主矛盾不是 DRAM volume——这项收益已经兑现；但 global request/cache/reduction
work 与 fused schedule tax 仍重。当前最强的可执行信号，是 Gate→Up→SwiGLU live range 产生的
local stack roundtrip，以及 5-warp resource-heavy CTA 的弱 cadence。**

## 6. 下一实验

先做最小 diagnostic arm `gate_smem_bf16`：Gate 完成后把 `128×128` tile 转成 BF16 写入当前
尚未用于 T1/T2 的 `sC`，让 Up 复用 accumulator registers；SwiGLU 时从 `sC` 读取 Gate，随后
覆盖写回现有 activation，再沿用当前 Q1/FC2。它不新增 SMEM 容量或 GMEM edge，task queue、FC2
与 4-way reduction 不变；代价是新增 shared roundtrip/barrier，且 Gate 提前转为 BF16，因此只作
`diagnostic-only`，不能直接称为纯 spill cost 或最终实现。

固定 synthetic fixture、单 GPU、graph boundary、task queue、route、FC2 schedule、weights/scales
与 formal oracle。variant 是新 artifact，必须与原 fused arm 做一次 fresh paired rerun，repeat 内
交替次序；不能与当前旧 baseline 单边拼接。

接受该方向需要同时满足：

1. correctness gate 通过，Tensor instructions/FP4 ops、physical rows、task count 与 reduction
   sectors 保持当前 work contract；
2. 54 条 `STL.64` Gate block 消失，dynamic local footprint 至少下降 80%；若只剩 14-word
   transition block，M8192 local total 的模型值约为 36.356 MB；同时记录新增 shared traffic，
   并确认 DRAM-elimination 收益保留；
3. M8192 未插桩 latency 至少改善 1%，才支持“该 handoff bundle 可回收关键路径”；改善至少 3% 且
   M256/M1024 regression 不超过 1%，才把该方向列为优先实现。

如果 local/resource 指标没有改变，则说明实现没有改变实际 live range；如果 local 至少下降 80%
但 M8192 latency 改善不足 1%，则拒绝这个 BF16-SMEM handoff 作为首选实现。若同时观察到 shared/
barrier tax 上升，它仍不足以单独推翻 spill criticality；下一轮再用 semantic-preserving N64
subfragment 或 IKET 区分 barrier/role idle。4-way global reduction schedule 仍留在独立实验。

## 7. KDK shadow validation

| Assertion | Result |
|---|---|
| 拒绝混用 compiler/JIT 或单边重测 evidence | pass：所有 raw/derived evidence 绑定同一 rerun、environment、protocol 与 per-arm artifact fingerprint |
| 从实际 timeline 发现 chain | pass：NSys 观测到 2-vs-9 nodes；未照抄 source prediction |
| 保留 many-to-many phase、storage edge 与 overlap 边界 | pass：未给 fused 伪造 additive phase time |
| 只 roll up additive counts | pass：bytes/instructions 使用 complete range；SOL/stall/occupancy 保持 launch-local |
| 输出优化动作与 discriminator | pass：选择 accumulator-lifetime 方向，并给出接受/推翻指标 |

本实验新增的一条 KDK 规则：**compiler/JIT identity 改变时，静态 SASS 模型必须从新 binary 重建；
禁止沿用旧 binary 的 opcode 常数或 source attribution。** 已沉淀并 push 到 KDK commit
`7ef0bad`。
