# 实验 002：融合算子与算子链的数据流分析

本文展示实验约束、Chain 各 op 与 Fused 各 phase 的耗时分布、NSys/NCU 证据，以及 M8192
Fused 主 kernel 的瓶颈判断。Fused phase 占比来自 diagnostic probe，只用于内部时间归因和调查排序，
不替代未插桩 benchmark/NSys 时间。

## 1. 实验设置与对比约束

### 1.1 对比对象

| 项目 | 本实验固定值 |
|---|---|
| 算子边界 | `X BF16 + topk_ids + topk_weights → W4A4 MoE → Y BF16` |
| 边界外操作 | router logits、softmax 和 top-k 选择；两边接收相同的 `topk_ids/topk_weights` |
| 形状 | `E=256, H=2048, I_tp=512, topk=8, activation=SwiGLU` |
| CuteDSL Fused | `cutedsl_bf16_fused`：BF16 输入，由 `B12xMoEWrapper` 在线量化并在主 kernel 内完成 MoE |
| CUTLASS Chain | `cutlass_bf16_chain`：BF16 输入，`input_sf=None`，由原生 CUTLASS chain 在线量化；`enable_pdl=True`、`use_fused_finalize=True` |
| 输入/输出 | 两边均为 BF16；FC1、FC2 权重以及内部 GEMM activation 为 NVFP4 |
| 用例 | `M=256`：小 M 对照；`M=1024`：仅 benchmark；`M=8192`：主要分析用例 |

这是**边界对齐、但实现差异混杂**的 backend 对比。两边除了是否跨 phase 融合，还使用不同的
kernel 实现、tile、warp layout、调度和 codegen。因此可以比较同一算子边界下的时间与硬件证据，
但不能把全部差值直接解释成“fusion 本身”的纯因果效应。

### 1.2 未插桩时间锚点

下表只定义性能现象和主分析用例。加速比为 `CUTLASS_time / CuteDSL_time - 1`。

| M | CuteDSL Fused latency | CUTLASS Chain latency | CuteDSL speedup | Max repeat spread | 本报告用途 |
|---:|---:|---:|---:|---:|---|
| 256 | 538.956 us | 550.739 us | +2.19% | 0.34% | 小 M 对照 |
| 1024 | 615.582 us | 577.220 us | -6.23% | 0.34% | 仅 benchmark，未采集 NSys/NCU |
| 8192 | 1782.547 us | 1665.779 us | -6.55% | 0.14% | 主要分析用例 |

原始样本见 [`benchmark_raw.csv`](benchmark_raw.csv)，汇总见
[`benchmark_summary.csv`](benchmark_summary.csv)。Profiler 时间只用于解释执行机制，不替代本表。

## 2. 实际执行时间与耗时分布

### 2.1 CUDA Graph replay 概览

`Kernel wall` 是首个 kernel 开始到最后一个 kernel 结束的区间；`Kernel sum` 是各节点时长之和。
当 PDL 使相邻节点重叠时，sum 可以大于 wall。

| M | 实现 | Material kernel count | Kernel wall | Kernel sum | Same-stream PDL overlap | Kernel idle gap |
|---:|---|---:|---:|---:|---:|---:|
| 256 | CuteDSL Fused | 2 | 497.694 us | 497.502 us | 0 | 0.192 us |
| 256 | CUTLASS Chain | 9 | 509.695 us | 511.293 us | 1.600 us | 0.002 us |
| 8192 | CuteDSL Fused | 2 | 1760.827 us | 1760.603 us | 0 | 0.224 us |
| 8192 | CUTLASS Chain | 9 | 1662.108 us | 1667.452 us | 5.344 us | 0 |

这里的 `2 vs 9` 是 **material kernel 节点数**，不是 CUDA Graph 的全部节点数；两边 graph 还各有
两个用于计时的 external CUDA event 节点。Chain 的 kernel 边界之间没有明显 GPU idle bubble，
反而存在少量 PDL overlap。

### 2.2 Chain op 耗时与占比分布

本节用例为 M8192。占比分母是 Chain 的 `1662.108 us` kernel wall。相邻节点存在 PDL overlap，因此各 op 占比合计为
`100.31%`，不重新归一化。

| 逻辑位置 | CUTLASS Chain op | Duration | Arm wall share | 与前一节点关系 |
|---|---|---:|---:|---|
| Prefix | block → global → merge | 46.655 us（区间并集） | 2.81% | 内部 overlap 0.128 us |
| Route + Q0 + Pack | `expandInputRowsKernel` | 245.600 us | 14.78% | overlap 0.480 us |
| GEMM metadata | stride metadata | 2.048 us | 0.12% | overlap 0.032 us |
| FC1 | grouped GEMM1 | 510.175 us | 30.69% | overlap 0.032 us |
| SwiGLU + Q1 | activation/requant | 231.647 us | 13.94% | overlap 0.064 us |
| FC2 | grouped GEMM2 | 408.255 us | 24.56% | overlap 4.512 us |
| 最终输出就绪 | finalize | 222.944 us | 13.41% | overlap 0.096 us |
| **完整 kernel 区间** | **9 个 material kernels** | **1662.108 us** | **100%** | **总 overlap 5.344 us** |

Prefix 的三个节点分别为 32.543、1.952 和 12.288 us；完整原始 trace 位于 [`nsys/`](nsys/)。

Chain 的耗时重心很清楚：FC1（30.69%）与 FC2（24.56%）合计占 55.26%，是第一优先级；其次是
Route + Q0 + Pack（14.78%）、SwiGLU + Q1（13.94%）和 Finalize（13.41%）；Prefix 与 GEMM metadata
合计仅 2.93%。该排序用于决定后续优先分析哪些算子；占比高本身不等于已经证明其存在可优化瓶颈。

### 2.3 Fused 主 kernel phase 耗时与占比分布

本节用例为 M8192，证据等级为 `diagnostic estimate`：probe 相对 matched no-marker control 的 median
latency 扰动为 `+1.74%`，仅接受用于 phase 排序；表中时间为 raw、unadjusted probe 值，未按 control
delta 校正。Fused 主 kernel 没有可供 NSys 独立计时的内部 kernel 边界，因此这里补充 exp_004 的
`%globaltimer` 诊断插桩。分母是 5 次 replay 的 SM-equivalent wall：
`Σ 110 × (max CTA final - min CTA entry) = 1,004,280,640 ns`；等效 wall 是
`Σ CTA phase time / (5 × 110)`。下表各行互斥，严格合计为 100%。

| Fused 主 kernel phase | 占 SM-equivalent wall | 等效 wall |
|---|---:|---:|
| Launch skew / early-finish idle | 1.83% | 33.33 us |
| Entry / prologue | 0.01% | 0.19 us |
| P0 Clear / init | 0.68% | 12.34 us |
| P1 Histogram | 0.39% | 7.16 us |
| P2 Prefix | 0.75% | 13.64 us |
| P3 Route + Q0 + Pack | **19.10%** | **348.80 us** |
| P4 Publish | 0.19% | 3.42 us |
| Compute setup | <0.01% | 0.01 us |
| T0 Claim / control | 0.98% | 17.96 us |
| T0 Cache / task setup | 0.76% | 13.96 us |
| FC1 Gate | **12.87%** | **234.95 us** |
| FC1 Up | **12.68%** | **231.57 us** |
| SwiGLU + Q1 | **6.17%** | **112.61 us** |
| FC2 setup | 0.12% | 2.19 us |
| FC2 GEMM | **10.20%** | **186.34 us** |
| FC2 epilogue + scatter | **32.43%** | **592.11 us** |
| Task control / final drain / producer tail | 0.84% | 15.41 us |
| **合计** | **100.00%** | **1825.96 us** |

### 2.4 按 Chain op 对齐的横向语义对比

本节用例为 M8192。Speedup 统一按 `Chain / Fused - 1` 计算，正值表示 Fused 更快。Phase 行使用
raw diagnostic phase time 与 NSys op time，`≈` 只描述时间差，不作纯 fusion 因果归因；完整算子行
使用未插桩 benchmark 作为性能真值。组合值按未四舍五入的原始值计算。

| 逻辑位置 | CuteDSL Fused phase（等效 wall / phase share） | CUTLASS Chain op（NSys duration / arm wall share） | Fused 相对 Chain |
|---|---:|---:|---:|
| FC2 scale 准备 | scale 广播 helper：1.120 us（0.06% arm wall，主 kernel 外） |  | — |
| Launch / entry | Launch skew + Entry：33.52 us（1.84%） |  | — |
| Prefix / routing offsets | P0 Clear + P1 Histogram + P2 Prefix：33.135 us（1.81%） | Prefix block → global → merge：46.655 us（2.81%，区间并集） | ≈ 快 13.520 us；**+40.8%** |
| Route + Q0 + Pack | P3 + P4 Publish：352.216 us（19.29%） | `expandInputRowsKernel`：245.600 us（14.78%） | ≈ 慢 106.616 us；**−30.3%** |
| GEMM metadata / task setup | Compute setup + T0 Claim + T0 Cache：31.919 us（1.75%） | stride metadata：2.048 us（0.12%） | ≈ 慢 29.871 us；**−93.6%** |
| FC1 | Gate + Up：466.513 us（25.55%） | grouped GEMM1：510.175 us（30.69%） | ≈ 快 43.662 us；**+9.4%** |
| SwiGLU + Q1 | 112.612 us（6.17%） | activation/requant：231.647 us（13.94%） | ≈ 快 119.035 us；**+105.7%** |
| FC2 compute | FC2 setup + GEMM：188.528 us（10.32%） | grouped GEMM2：408.255 us（24.56%，含中间输出 epilogue） | ≈ 快 219.727 us；**+116.5%** |
| 最终输出就绪 | FC2 epilogue + scatter：592.108 us（32.43%） | finalize：222.944 us（13.41%） | ≈ 慢 369.164 us；**−62.3%** |
| Task drain / producer tail | 15.41 us（0.84%） |  | — |
| **完整算子（未插桩 benchmark）** | **1782.547 us** | **1665.779 us** | **慢 116.767 us；−6.55%** |

## 3. NCU 证据展示

本章先展示 M8192 各 launch 的 NCU metrics，再给出完整算子的可加计数。Occupancy、SOL、Warp stall
和 cadence 只按各自 launch 展示，不能跨 kernel 加总或平均；完整算子的 bytes/request/instruction 计数
来自同口径 operator range。DRAM、L2、LSU global、TMA 和 local 是不同观测点，不能相加成总流量；
性能时间仍以第 1、2 章为准，不使用 NCU duration。

### 3.1 Fused main 与 Chain 各 op 的 NCU metrics

每张表对应一个独立 deep-profiled launch。Fused main 覆盖从 Clear 到 FC2 scatter 的全部融合 phase；
Chain 每张表只覆盖标题所示 op。因此这些表可用于观察各 launch 的资源压力与执行节奏，但不能把
Fused main 与任一 Chain op 当作 phase-matched 对照，也不能把 Chain 的比例类 metrics 加总或平均。

#### 3.1.1 Fused

##### Fused main：`MoEDynamicKernel`

| 类别 | Metric | Value |
|---|---|---:|
| Launch | NSys duration (% of arm wall) | 1759.483 us（99.92%） |
| Launch | Grid / block | `1x1x110 / 160x1x1` |
| Resource | Registers / thread | 255（allocated 256） |
| Resource | SMEM / CTA | 92,160 B（dynamic 91,136 B） |
| Resource | Actual compiler stack frame | 488 B / thread |
| Resource | Compiler-annotated static spill/refill SASS | 122 refill / 68 store instructions |
| Occupancy | Achieved occupancy | 10.42% |
| Schedule | Eligible warps / active cycle | 0.202 |
| Schedule | Issue active (active cycles) | 19.95% |
| Compute utilization | TC subpipe active (active cycles) | 25.73% |
| Compute utilization | ALU pipe active (active cycles) | 8.44% |
| Compute utilization | FMA pipe active (active cycles) | 5.31% |
| Compute utilization | XU executed-pipe utilization (active cycles) | 1.64% |
| Memory utilization | DRAM throughput (elapsed cycles) | 38.46% |
| Memory utilization | L2 throughput (elapsed cycles) | 41.35% |
| Memory utilization | L1TEX throughput (active cycles) | 36.32% |
| Memory utilization | TMA pipe active (active cycles) | 0.25% |
| Stall | Warp stalls (Wait / Long scoreboard / Sleeping / Short scoreboard / Barrier) | 29.22% / 18.98% / 14.77% / 9.76% / 5.69% |
| Throttle | Warp throttles (MIO / LG / Math pipe / TEX) | 2.20% / 0.47% / 0.23% / 0% |

#### 3.1.2 Chain

##### Route + Q0 + Pack：`expandInputRowsKernel`

| 类别 | Metric | Value |
|---|---|---:|
| Launch | NSys duration (% of arm wall) | 245.600 us（14.78%） |
| Launch | Grid / block | `880x1x1 / 256x1x1` |
| Resource | Registers / thread | 48 |
| Resource | SMEM / CTA | 1,024 B（dynamic 0 B） |
| Occupancy | Achieved occupancy | 69.52% |
| Schedule | Eligible warps / active cycle | 2.244 |
| Schedule | Issue active (active cycles) | 61.05% |
| Compute utilization | TC subpipe active (active cycles) | 0% |
| Memory traffic | Local load + store footprint | 0 B |
| Stall | Warp stalls (Wait / Long scoreboard / Sleeping / Short scoreboard / Barrier) | 19.60% / 31.75% / — / 3.58% / 0% |
| Throttle | Warp throttles (MIO / LG / Math pipe / TEX) | 0.03% / 0.01% / 13.18% / — |

##### FC1：`device_kernel`

| 类别 | Metric | Value |
|---|---|---:|
| Launch | NSys duration (% of arm wall) | 510.175 us（30.69%） |
| Launch | Grid / block | `1x110x1 / 384x1x1` |
| Resource | Registers / thread | 168 |
| Resource | SMEM / CTA | 90,112 B（dynamic 89,088 B） |
| Occupancy | Achieved occupancy | 22.92% |
| Schedule | Eligible warps / active cycle | 0.217 |
| Schedule | Issue active (active cycles) | 15.96% |
| Compute utilization | TC subpipe active (active cycles) | 62.69% |
| Memory traffic | Local load + store footprint | 53.296 MB |
| Stall | Warp stalls (Wait / Long scoreboard / Sleeping / Short scoreboard / Barrier) | 19.22% / 12.32% / — / 1.21% / 7.06% |
| Throttle | Warp throttles (MIO / LG / Math pipe / TEX) | 2.21% / <0.01% / 24.06% / — |

##### SwiGLU + Q1：`doActivationKernel`

| 类别 | Metric | Value |
|---|---|---:|
| Launch | NSys duration (% of arm wall) | 231.647 us（13.94%） |
| Launch | Grid / block | `880x1x1 / 256x1x1` |
| Resource | Registers / thread | 57（allocated 64） |
| Resource | SMEM / CTA | 1,024 B（dynamic 0 B） |
| Occupancy | Achieved occupancy | 44.11% |
| Schedule | Eligible warps / active cycle | 1.507 |
| Schedule | Issue active (active cycles) | 53.49% |
| Compute utilization | TC subpipe active (active cycles) | 0% |
| Memory traffic | Local load + store footprint | 0 B |
| Stall | Warp stalls (Wait / Long scoreboard / Sleeping / Short scoreboard / Barrier) | 27.26% / 26.87% / — / 3.35% / 0% |
| Throttle | Warp throttles (MIO / LG / Math pipe / TEX) | 0.41% / 0.01% / 9.99% / — |

##### FC2：`device_kernel`

| 类别 | Metric | Value |
|---|---|---:|
| Launch | NSys duration (% of arm wall) | 408.255 us（24.56%） |
| Launch | Grid / block | `1x110x1 / 384x1x1` |
| Resource | Registers / thread | 168 |
| Resource | SMEM / CTA | 90,112 B（dynamic 89,088 B） |
| Occupancy | Achieved occupancy | 22.92% |
| Schedule | Eligible warps / active cycle | 0.284 |
| Schedule | Issue active (active cycles) | 20.96% |
| Compute utilization | TC subpipe active (active cycles) | 38.30% |
| Memory traffic | Local load + store footprint | 101.165 MB |
| Stall | Warp stalls (Wait / Long scoreboard / Sleeping / Short scoreboard / Barrier) | 17.88% / 19.67% / — / 2.90% / 12.61% |
| Throttle | Warp throttles (MIO / LG / Math pipe / TEX) | 1.56% / <0.01% / 14.46% / — |

##### Finalize：`finalizeMoeRoutingKernel`

| 类别 | Metric | Value |
|---|---|---:|
| Launch | NSys duration (% of arm wall) | 222.944 us（13.41%） |
| Launch | Grid / block | `8192x1x1 / 256x1x1` |
| Resource | Registers / thread | 48 |
| Resource | SMEM / CTA | 1,024 B（dynamic 0 B） |
| Occupancy | Achieved occupancy | 79.01% |
| Schedule | Eligible warps / active cycle | 0.403 |
| Schedule | Issue active (active cycles) | 25.61% |
| Compute utilization | TC subpipe active (active cycles) | 0% |
| Memory traffic | Local load + store footprint | 0 B |
| Stall | Warp stalls (Wait / Long scoreboard / Sleeping / Short scoreboard / Barrier) | 3.64% / 89.67% / — / 0.66% / 0% |
| Throttle | Warp throttles (MIO / LG / Math pipe / TEX) | 0.04% / 0.66% / 0.62% / — |

### 3.2 完整算子的可加计数

以下全部来自完整 app-range，而不是把 deep launches 相加。流量单位为十进制 MB（`10^6 B`）；
差异为 `Fused / Chain - 1`。

M8192（主要分析用例）：

| Operator metric | CuteDSL Fused | CUTLASS Chain | Fused − Chain (relative delta) |
|---|---:|---:|---:|
| DRAM read | 689.325 MB | 953.981 MB | -264.656 MB（-27.74%） |
| DRAM write | 244.452 MB | 500.189 MB | -255.737 MB（-51.13%） |
| **DRAM read + write** | **933.777 MB** | **1454.170 MB** | **-520.393 MB（-35.79%）** |
| L2 traffic | 3753.999 MB | 3694.407 MB | +59.592 MB（+1.61%） |
| LSU global request footprint | 5557.711 MB | 4018.004 MB | +1539.707 MB（+38.32%） |
| └ LSU global load footprint | 4307.634 MB | 3697.935 MB | +609.699 MB（+16.49%） |
| └ LSU global reduction footprint | 1073.742 MB | 0 MB | +1073.742 MB（Chain=0） |
| TMA interface traffic | 1869.742 MB | 2646.344 MB | -776.602 MB（-29.35%） |
| Local load + store footprint | 316.817 MB | 154.460 MB | +162.357 MB（+105.11%） |
| Executed warp instructions | 386.620 M inst | 544.532 M inst | -157.911 M inst（-29.00%） |
| Tensor instructions | 31.162 M inst | 31.162 M inst | 0 |
| FP4→FP32 Tensor ops | 510.564 G op | 510.564 G op | 0 |

M256（小 M 对照）：

| Operator metric | CuteDSL Fused | CUTLASS Chain | Fused − Chain (relative delta) |
|---|---:|---:|---:|
| DRAM read | 494.066 MB | 466.241 MB | +27.825 MB（+5.97%） |
| DRAM write | 9.212 MB | 13.841 MB | -4.630 MB（-33.45%） |
| **DRAM read + write** | **503.277 MB** | **480.082 MB** | **+23.195 MB（+4.83%）** |
| L2 traffic | 1143.424 MB | 995.431 MB | +147.994 MB（+14.87%） |
| LSU global request footprint | 174.912 MB | 167.465 MB | +7.447 MB（+4.45%） |
| TMA interface traffic | 754.975 MB | 918.553 MB | -163.578 MB（-17.81%） |
| Local load + store footprint | 127.926 MB | 64.175 MB | +63.751 MB（+99.34%） |
| Executed warp instructions | 71.011 M inst | 84.137 M inst | -13.126 M inst（-15.60%） |
| Tensor instructions | 12.583 M inst | 12.583 M inst | 0 |
| FP4→FP32 Tensor ops | 206.158 G op | 206.158 G op | 0 |

两个实现的 Tensor instructions 与 FP4 ops 在同一 M 上完全一致，构成 TensorCore 工作量等价门禁；
由 FP4 ops 推导的 physical routed rows 和 padding factor 也一致。与此同时，M8192 的 DRAM、L2、
LSU 与 TMA 指标并不同向；因此不能用单个“global traffic”数字代替不同 memory interface 的证据。

原始汇总见 [`operator_comparison_v2.csv`](ncu/operator_comparison_v2.csv)，完整 range counter
来自各 case 的 `native_raw.csv`。

### 3.3 Fused 主 kernel 瓶颈分析

| 优先级 | Fused bottleneck / 疑点 | 直接证据 | 优化方向 |
|---|---|---|---|
| P0 | Resource pressure 与 register spill | 255 Registers/thread、92,160 B SMEM/CTA、488 B/thread stack frame；122 words/lane 的静态 stack roundtrip 与动态 local sectors 精确闭合。exp_003 已闭合物理形成机制：108-word Main 是 first-pass accumulator 跨完整 second pass 保活；14-word Tail 是 activation temporary reuse 时保存的 5 个 second-pass accumulator register values 与 9 个 index/address/control scalar。Baseline source order 将 first/second pass 高置信解释为 Gate/Up，但无 compiler-certified SSA→physical-slot map | 构造 correctness-equivalent 的 reduced/no-spill arm，先验证 spill 对 latency 与 TC cadence 的因果影响；暂不下 production optimization 结论 |
| P1 | Whole-kernel TC activity 低，原因未闭合 | Fused TC subpipe active `25.73%`；内部时间图同时显示 P3 占 19.10%、FC2 epilogue + scatter 占 32.43%，证明 whole-launch average 混入大量 non-TC phase，但不能直接解释成 TC starvation | 在高时间权重 phase 内进一步区分 planned TC-off 与 T1/T2/T4 的 producer、barrier 或同 warp critical-path starvation |
| P1 | FC2 epilogue + scatter 是最大 phase，内部机制未拆分 | Diagnostic phase share 为 `32.43%`；Fused 完整 range 独有 `1073.742 MB` LSU global reduction footprint；源码显示 epilogue、同步和 route-weighted atomic scatter，但现有证据不能把整段时间归因给 atomic | 定位每个 FC2 tile 的 GEMM→epilogue→barrier→atomic scatter→下一次 GEMM 边界，再用受门禁的 counterfactual 界定各部分成本 |
| P1 | P3 Route + Q0 + Pack 时间权重高且与 compute 串行 | Diagnostic phase share 为 `19.10%`；`full_tile_publish_enabled=0`，P3/P4 完成并经过 resident-grid barrier 后 compute 才开始；P3 内三个操作交错，当前不能继续拆分 | 先定位 P3 内部重心，再验证 ready-task/full-tile publish 是否产生真实 route/compute overlap；收益只看未插桩 latency |
| 排除项 | Launch/bubble 不是当前主要瓶颈 | Material kernel count 已从 `9→2`；Chain 内部没有明显 idle gap，而 Fused 主 kernel 占自身 wall 的 `99.92%`，M8192 仍慢 `6.55%` | 优先优化 Fused 主 kernel 内部，不继续压缩 graph launch 数量 |

当前第一优先级是围绕 exp_003 已闭合的形成机制构造 reduced/no-spill counterfactual；P0 来自项目的
no-spill 工程约束，不依赖 latency criticality 证明。“spill 是 TC cadence 偏低的主要贡献者”仍是假设。

## 4. 下一步调查与优化（Next To Do）

| 优先级 | 可证伪问题 | 最小调查 | 必须保持 | 接受 / 推翻条件 |
|---|---|---|---|---|
| P0 | Register spill 是否是 TC cadence 偏低的主要贡献者？ | 基于 exp_003 Main live-range 机制构造 correctness-equivalent 的 reduced/no-spill arm；比较 stack/local traffic、latency、TC subpipe active、Issue Active 与 warp stalls | Tensor work、launch topology、task schedule、输出正确性与 timing protocol；禁止 measurement-only instrumentation 改变 resource/SASS | 只有 spill 明确减少/消除，且 TC cadence 与 latency 在 matched counterfactual 中同向改善，才接受主要贡献；否则推翻或降级 |
| P1 | 控制 spill 后，剩余 TC inactive time 中多少是必要 non-TC phase，多少是真实 TC starvation？ | 若 reduced/no-spill arm 仍有明显 TC cadence 问题，再用通过 resource/SASS identity gate 的 IKET phase/warp-role timeline | 相同 case、task/dispatch 与 Tensor work；instrumented duration 不作性能时间；marker 不得改变 stack/register/SASS work | 只有 T1/T2/T4 内 gap 与 wait/scoreboard/barrier 对齐才接受 starvation；否则归为 planned TC-off |
| P1 | FC2 epilogue + scatter 的 32.43% 主要花在哪里？ | 展开每个 T4 tile 的 GEMM→epilogue→barrier→atomic scatter→barrier→next GEMM；必要时加入 diagnostic-only 的受控 bundle 消融 | Tensor work、task schedule 与 phase coverage；数值无效变体必须标记 diagnostic-only | 只有 phase-local timing 与 counterfactual 同时支持，才把成本归因给具体 epilogue/scatter 子机制 |
| P1 | P3 Route + Q0 + Pack 的 19.10% 哪里可优化，能否与 compute overlap？ | 先为交错的 Route/Q0/Pack 建立不伪造独立 wall 的内部证据，再只改变 publish/consume 时序验证 overlap | Logical work、正确性、task count、Tensor instructions 与 timing boundary | 定位到稳定内部重心，并且真实 overlap 使未插桩 latency 改善超过 repeat spread，才接受对应优化方向 |
