# 实验 002：融合算子与算子链的数据流分析

本文展示实验约束、NSys/NCU 证据和 M8192 Fused 主 kernel 的瓶颈判断；phase 级归因与优化实验尚未完成。

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

## 2. NSys 实际执行时间

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

### 2.2 M8192：末端对齐的非对称对照

表中各行按语义执行顺序排列；它同时给出 Chain 的 op/kernel 构成以及两边可由 NSys 独立计时的节点。
空白表示 Fused 没有独立 kernel 边界，**不表示没有执行这项工作，也不表示耗时为零**。占比的分母是
各自 arm 的完整 kernel wall。

| 逻辑位置 | CUTLASS Chain 节点 | Chain duration (% of wall) | 与前一 Chain 节点关系 | CuteDSL Fused 节点 | Fused duration (% of wall) |
|---|---|---:|---:|---|---:|
| FC2 scale 准备 |  |  |  | scale 广播 helper | 1.120 us（0.06%） |
| Prefix | block → global → merge | 46.655 us（2.81%，区间并集） | 内部 overlap 0.128 us |  |  |
| Route + Q0 + Pack | `expandInputRowsKernel` | 245.600 us（14.78%） | overlap 0.480 us |  |  |
| GEMM metadata | stride metadata | 2.048 us（0.12%） | overlap 0.032 us |  |  |
| FC1 | grouped GEMM1 | 510.175 us（30.69%） | overlap 0.032 us |  |  |
| SwiGLU + Q1 | activation/requant | 231.647 us（13.94%） | overlap 0.064 us |  |  |
| FC2 | grouped GEMM2 | 408.255 us（24.56%） | overlap 4.512 us |  |  |
| 最终输出就绪 | finalize | 222.944 us（13.41%） | overlap 0.096 us | `MoEDynamicKernel` | 1759.483 us（99.92%）† |
| **完整 kernel 区间** | **9 个 material kernels** | **1662.108 us（100% wall）** |  | **2 个 material kernels** | **1760.827 us（100% wall）** |

† Fused 主 kernel 从 Clear 一直覆盖到 FC2 scatter；把它放在“最终输出就绪”一行只是末端对齐，
不能拿 `1759.483 us` 与 Chain 的 `finalize 222.944 us` 作逐阶段比较。

Chain 各行占比合计为 100.31%，因为相邻节点存在 PDL overlap；Fused 两个 material kernel 合计为
99.99%，其余是 0.224 us kernel gap。上述占比不能当作互斥 phase 分解后强制加总为 100%。Prefix 的
三个节点分别为 32.543、1.952 和 12.288 us。完整原始 trace 位于 [`nsys/`](nsys/)。

Chain 的耗时重心很清楚：FC1（30.69%）与 FC2（24.56%）合计占 55.26%，是第一优先级；其次是
Route + Q0 + Pack（14.78%）、SwiGLU + Q1（13.94%）和 Finalize（13.41%）；Prefix 与 GEMM metadata
合计仅 2.93%。该排序用于决定后续优先分析哪些算子；占比高本身不等于已经证明其存在可优化瓶颈。

### 2.3 Host launch 与 correlation ID

每个 trace 只有一次 `cudaGraphLaunch_v10000`；同一个 correlation ID `4` 关联该 replay 内全部
2/9 个 kernel。CUDA Graph 不会为每个内部 kernel 暴露一笔可单独归属的 CPU launch overhead。

| M | 实现 | `cudaGraphLaunch` API duration | API enter → first kernel start | First kernel vs API return |
|---:|---|---:|---:|---:|
| 256 | CuteDSL Fused | 78.319 us | 63.319 us | 提前 15.000 us 开始 |
| 256 | CUTLASS Chain | 64.789 us | 54.231 us | 提前 10.558 us 开始 |
| 8192 | CuteDSL Fused | 59.459 us | 52.173 us | 提前 7.286 us 开始 |
| 8192 | CUTLASS Chain | 70.339 us | 59.695 us | 提前 10.644 us 开始 |

API 返回前 GPU 已经开始执行，因此 API 区间不能直接当作额外加在 GPU wall 上的 launch cost。
本实验能直接观测的是 graph replay 的入口延迟、kernel 区间、相邻 overlap/gap；不能把 CPU 时间
按 correlation ID 分摊到每个 graph 内部 kernel。

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
| P1 | Whole-kernel TC activity 低，原因未闭合 | Fused TC subpipe active `25.73%`，但该 launch 同时包含 P0–P4、T3 和 scatter 等必要 non-TC 工作；不能直接解释成 TC starvation | 用 phase/warp-role breakdown 区分 planned TC-off 与 T1/T2/T4 内真实 producer、barrier 或同 warp critical-path starvation |
| P1 | FC2 atomic scatter / barrier 候选 | Fused 独有 `1073.742 MB` LSU global reduction footprint；每个 I128 slice 的 FC2 tile 都执行 epilogue、同步和 route-weighted atomic scatter | 定位 QMMA 到下一次 QMMA 之间的 scatter/barrier gap，再用受门禁的 diagnostic-only 消融界定成本上界 |
| P1 | P0–P4 串行化，未利用 route/compute overlap | `full_tile_publish_enabled=0`；route/pack 与 task publication 完成并经过 5 个 resident-grid barrier 后，P5 compute 才开始 | 验证 ready-task/full-tile publish 是否产生真实 overlap，并以未插桩 latency 判断收益 |
| 排除项 | Launch/bubble 不是当前主要瓶颈 | Material kernel count 已从 `9→2`；Chain 内部没有明显 idle gap，而 Fused 主 kernel 占自身 wall 的 `99.92%`，M8192 仍慢 `6.55%` | 优先优化 Fused 主 kernel 内部，不继续压缩 graph launch 数量 |

当前第一优先级是围绕 exp_003 已闭合的形成机制构造 reduced/no-spill counterfactual；P0 来自项目的
no-spill 工程约束，不依赖 latency criticality 证明。“spill 是 TC cadence 偏低的主要贡献者”仍是假设。

## 4. 下一步调查与优化（Next To Do）

| 优先级 | 可证伪问题 | 最小调查 | 必须保持 | 接受 / 推翻条件 |
|---|---|---|---|---|
| P0 | Register spill 是否是 TC cadence 偏低的主要贡献者？ | 基于 exp_003 Main live-range 机制构造 correctness-equivalent 的 reduced/no-spill arm；比较 stack/local traffic、latency、TC subpipe active、Issue Active 与 warp stalls | Tensor work、launch topology、task schedule、输出正确性与 timing protocol；禁止 measurement-only instrumentation 改变 resource/SASS | 只有 spill 明确减少/消除，且 TC cadence 与 latency 在 matched counterfactual 中同向改善，才接受主要贡献；否则推翻或降级 |
| P1 | 控制 spill 后，剩余 TC inactive time 中多少是必要 non-TC phase，多少是真实 TC starvation？ | 若 reduced/no-spill arm 仍有明显 TC cadence 问题，再用通过 resource/SASS identity gate 的 IKET phase/warp-role timeline | 相同 case、task/dispatch 与 Tensor work；instrumented duration 不作性能时间；marker 不得改变 stack/register/SASS work | 只有 T1/T2/T4 内 gap 与 wait/scoreboard/barrier 对齐才接受 starvation；否则归为 planned TC-off |
| P1 | FC2 atomic scatter + epilog barrier 是否打断 TC cadence？ | IKET 展开每个 T4 tile 的 QMMA→epilogue→barrier→atomic scatter→barrier→next QMMA；必要时加入 diagnostic-only scatter bundle 消融 | Tensor work、task schedule与phase coverage；数值无效变体必须标记 diagnostic-only | Gap 集中在 scatter/barrier，且消融给出稳定 latency 上界才接受 |
| P1 | P0–P4 串行化是否能通过 ready-task/full-tile publish 转化为 route/compute overlap？ | 只改变 publish/consume 时序；用 NSys/IKET 验证 overlap，并用未插桩 benchmark 验证性能 | Logical work、正确性、task count、Tensor instructions 与 timing boundary | Trace 中出现真实 overlap且 latency 改善超过 repeat spread，同时无 work/resource regression 才接受 |
