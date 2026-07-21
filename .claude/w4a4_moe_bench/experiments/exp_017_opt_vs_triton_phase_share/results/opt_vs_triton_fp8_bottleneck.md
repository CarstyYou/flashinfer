# exp_017：Latest opt vs Triton FP8 的 2× 目标瓶颈分析

本报告只分析 M8192。未插桩性能、Triton 12-node timeline、Latest opt 完整 phase 分布和当前
binary 的 launch-local NCU 已闭合；NCU 用来识别 Resource / Schedule / utilization / stall 症状，
不以 profile duration 代替 benchmark，也不把 Fused whole-launch metrics 伪装成单个 phase metrics。

## 1. 实验设置与对比约束

### 1.1 对比对象

| 项目 | 本实验固定值 |
|---|---|
| 算子边界 | `X BF16 + topk_ids + topk_weights → MoE → Y BF16`；router logits、softmax 和 top-k 选择在边界外 |
| 形状 | `M=8192, E=256, H=2048, I_tp=512, topk=8, SwiGLU` |
| Latest opt | BF16 输入，NVFP4 weight/activation，一次 persistent fused kernel；8 math warps + 1 TMA warp |
| Triton FP8 | SGLang legacy Triton chain；BF16 输入，动态 tensor-wise E4M3 activation、E4M3 weight，12 个 CUDA Graph nodes |
| 性能与 phase-time 证据 | 同一输入与 routing fixture、同一 5KP `GPU-ab3d...9522`、2377 MHz；各 backend 使用自身锁定 runtime 和数值格式 |
| NCU 补采证据 | 两侧均在同一 sibling 5KP `GPU-c2ac...b14f`、2377 MHz、独占 lease 上采集；只比较规范化 launch-local metrics |
| 性能目标 | `Latest opt latency ≤ Triton FP8 latency / 2` |

这是客户目标对照，不是纯 fusion 因果实验。两侧 precision、weight/scale 表示、kernel、tile、schedule、
runtime 和 codegen 都不同；完整 latency ratio 可以判断是否达标，但 phase/op 横表只能观察时间落点。

### 1.2 未插桩性能锚点与 2× 缺口

| Case | Latest opt | Triton FP8 | 当前 ratio / speedup | 2× 目标 | 尚需减少 |
|---:|---:|---:|---:|---:|---:|
| M8192 | **1354.733 µs**（spread 2.01%） | **2213.402 µs**（spread 3.47%） | **1.6338× / +63.38%** | **≤1106.701 µs** | **248.032 µs / 18.31%** |

Speedup 定义为 `Triton / Latest opt - 1`。因此 2× latency ratio 对应 `+100%` speedup；最终是否
收口只看 correctness-qualified、未插桩 benchmark，不能把 diagnostic phase time 直接当作可回收预算。

## 2. 实际执行时间与耗时分布

### 2.1 Replay 概览

| 实现 | 时间证据 | Material execution | 完整观测区间 | Gap / residual |
|---|---|---:|---:|---:|
| Latest opt | `%globaltimer` phase probe | 1 个 fused main kernel | **1331.968 µs** SM-equivalent denominator | CTA residual 31.066 µs；launch skew 25.659 µs，均已计入 100% |
| Triton FP8 | NSys CUDA Graph | 12 个串行 kernel nodes | **2156.058 µs** graph wall；node sum 2153.818 µs | **2.336 µs** bubble；无 overlap |

Latest opt probe 相对 matched no-marker control 扰动 `+1.92%`，REG/STACK/SMEM 保持
`165 / 0 B / 1024 B static`（另有 83,968 B dynamic SMEM），且无 static spill；但 marker 引起了更广泛的 SASS scheduling/control-flow
重排，所以以下 Latest opt phase 是 `diagnostic estimate`，不是 production-exact latency。

### 2.2 Triton FP8 Chain op 耗时与占比分布

百分比分母为 Triton graph wall `2156.058 µs`；以下均为 5 replay 中位数。

| 逻辑位置 | 独立 graph op | Duration | Arm wall share |
|---|---|---:|---:|
| Routing | Align 26.912 µs；Count/sort 13.472 µs | **40.384 µs** | **1.87%** |
| Q0 | Fill 0.480 µs；Absmax 38.400 µs；Quant 10.176 µs | **49.056 µs** | **2.28%** |
| FC1 | `fused_moe_kernel` | **1004.669 µs** | **46.61%** |
| SwiGLU | `act_and_mul` | **126.144 µs** | **5.86%** |
| Q1 | Fill 0.640 µs；Absmax 38.720 µs；Quant 18.592 µs | **57.920 µs** | **2.69%** |
| FC2 | `fused_moe_kernel` | **650.142 µs** | **30.15%** |
| 输出归并 | TopK reduce/combine | **225.696 µs** | **10.46%** |
| Graph bubble | 相邻 nodes 间 gap | **2.336 µs** | **0.11%** |
| **完整 graph 区间** | **12 个 material nodes** | **2156.058 µs** | **100%** |

Q0 是唯一 replay range 超过 5% 的关键 group：`[45.568, 50.560] µs`。各行独立取中位数，
因此四舍五入后的行和不要求与 graph-wall 中位数逐位相等。

### 2.3 Latest opt Fused phase 耗时与占比分布

| Fused phase | Diagnostic time | Phase share |
|---|---:|---:|
| Clear / init | 12.359 µs | 0.93% |
| Histogram | 8.922 µs | 0.67% |
| Prefix | 13.910 µs | 1.04% |
| Route + Q0 + Pack | **94.038 µs** | **7.06%** |
| Publish / route tail | 13.144 µs | 0.99% |
| Claim / cache / control | 27.043 µs | 2.03% |
| FC1 Gate/Up + SwiGLU | **451.763 µs** | **33.96%** |
| Q1 | 33.859 µs | 2.54% |
| FC2 GEMM + epilogue + R2S | **248.184 µs** | **18.63%** |
| Atomic Scatter | **371.964 µs** | **27.91%** |
| CTA residual | 31.066 µs | 2.34% |
| Launch skew / early finish | 25.659 µs | 1.93% |
| **完整 phase denominator** | **1331.968 µs** | **100%** |

每个 replay 内部严格闭合；表中逐 phase 中位数独立计算，因此显示值合计存在约 `0.03` percentage-point
的 median/rounding 差异。FC2 行已经包含 epilogue、scale/cast、R2S 和 pre-scatter sync，未转嫁给 Scatter。

### 2.4 Phase/Op 横向语义对照

| 逻辑位置 | Latest opt phase（time / own share） | Triton op（time / own share） | 当前观察 |
|---|---:|---:|---|
| Routing / scheduler | 75.402 µs / 5.65% | 40.384 µs / 1.87% | Opt 投影时间较长；边界不同，不计算 speedup |
| Q0 / input pack | 94.038 µs / 7.06% | 49.056 µs / 2.28% | Opt 投影时间较长；Q0 仍有剩余调查空间 |
| FC1 + SwiGLU | 451.763 µs / 33.96% | 1130.877 µs / 52.45% | Opt 投影时间较短；不同 precision/实现，不作因果归因 |
| Q1 | 33.859 µs / 2.54% | 57.920 µs / 2.69% | Opt 投影时间较短 |
| FC2 + epilogue + R2S | 248.184 µs / 18.63% | 650.142 µs / 30.15% | Opt 投影时间较短 |
| 输出归并 | Scatter 371.964 µs / 27.91% | TopK reduce 225.696 µs / 10.46% | Opt 投影时间较长；算法和完成边界不等价 |
| Residual / skew | 56.646 µs / 4.25% | Graph bubble 2.336 µs / 0.11% | 非同类边界，不作横向归因 |
| **完整算子（未插桩）** | **1354.733 µs** | **2213.402 µs** | **快 858.669 µs；+63.38%，距 2× 仍差 248.032 µs** |

横表只决定调查权重。Latest opt phase 使用 SM-equivalent diagnostic denominator，Triton op 使用
NSys graph elapsed；除完整算子行外，所有行都不是正式性能 speedup。

## 3. NCU 证据展示

本章不比较 NCU duration，只用 launch-local metrics 回答四个问题：

| 要回答的问题 | 主要看什么 | 正确读法 |
|---|---|---|
| 有多少工作能同时驻留？ | Resource、Achieved occupancy | 低 occupancy 只说明 resident warps 少；必须结合 grid、CTA 大小和 Resource limit，不能单独判为瓶颈 |
| scheduler 实际发射是否持续？ | Issue active | 数值低表示 issue activity 不连续；它不等于 Eligible warps，不能单独区分“没有 ready instruction”还是“有指令但被 dependency / barrier / throttle 阻塞” |
| 哪类硬件路径最忙？ | TC / ALU / FMA / XU、DRAM / L2 / L1 / LSU | 都是相对各自 peak 的 utilization，但 TC/ALU/FMA 看 pipe-active cycles，XU/LSU 看 executed-instruction rate，memory hierarchy 也有 active/elapsed 两种分母；可判断压力落点，不能相加，也不是 phase 耗时占比 |
| PC sampling 时 warp 的 issue state 落在哪里？ | Stall / Throttle sample share | 是该 launch 全部 non-`_not_issued` PC-sampling reason counts 的样本构成，不是 elapsed-time 百分比；分母还包含未展示的 selected / not-selected 等 reason，所以表内几项不要求加总为 100% |

本次用 NCU 2025.3.1 以 `kernel replay + cache-control all + graph-profiling node` 采集，VeloQ 0.2.2
配合 NCU 2026.1.1 reader 提取。因此 utilization / stall 只代表该 capture scope 下的硬件 signature，
不代表 production cache residency；完整 Triton Graph range 的 additive metrics 仍缺失，因此不把多个
launch 的 traffic / instruction counts 相加。

### 3.1 Latest opt Fused main

| 观察问题 | 关键指标 | 数值 | 这说明什么 |
|---|---|---:|---|
| 驻留并行度 | 110 CTA；9 warps/CTA；Registers used / allocated；total SMEM；Achieved occupancy | 165 / 168 reg/thread；84,992 B/CTA；**18.75%** | 每个 SM 恰好驻留 1 个 9-warp CTA；grid 本身也只有 110 CTA。单独降低 Registers 或 SMEM 不会自动产生第 2 个 CTA，除非同时改变 persistent topology / task ownership |
| 发射连续性 | Issue active | **27.81%** | whole launch 仍有大量 issue capacity 未被使用，但该平均值混合了 GEMM、routing、quant 和 Scatter，不能指出是哪一 phase |
| Compute pressure | TC / ALU / FMA active；XU instruction-rate utilization | **35.34% / 11.75% / 7.10%；2.04%** | 在同为 cycles-active 的 TC/ALU/FMA 中，TC 最高；XU 是另一种 instruction-rate 口径，不能加入排序。whole launch 没有持续压满 TC，但不能把 35.34% 归给 FC1 或 FC2 |
| Memory pressure | DRAM / L2 / L1 throughput；LSU instruction-rate utilization；TMA active | **52.70% / 52.39% / 48.25%；33.29%；0.48%** | 已展示的 whole-launch memory paths 没有持续压满；这是多 phase 混合后的平均压力，不代表 Scatter 或 Q0 单独的带宽利用率 |
| 等待落点 | Wait / Sleeping / Long scoreboard / Barrier / Short scoreboard sample share | **21.93% / 12.14% / 9.92% / 9.84% / 7.85%** | 样本分散在 dependency、persistent wait 和 CTA sync；没有一个 launch-wide reason 足以单独解释总耗时 |
| 发射受限线索 | Math pipe / MIO / LG throttle sample share | **14.02% / 5.31% / 0.04%** | Math pipe 与 MIO 是主要 throttle 线索，但仍需 exact-PC / phase 证据才能落到源码 |
| Spill 排除 | compiler stack；dynamic spill load/store | 0 B/thread；**0 / 0** | 当前 exact cubin 的旧 spill 问题已排除，不是本轮 blocker |

**本表结论：** Latest opt 是一个混合 TC、memory、同步和 persistent scheduling 的 launch；NCU 只证明
whole launch 在已展示路径上没有持续压满单一硬件路径，以及 1 CTA/SM 没有跨 CTA latency hiding。它尚未告诉我们应改
FC1、FC2、Q0 还是 Scatter，所以优化优先级必须继续由第 2 章的 phase time 和 exact-PC 证据决定。

### 3.2 Triton FP8 FC1

| 观察问题 | 关键指标 | 数值 | 这说明什么 |
|---|---|---:|---|
| 驻留并行度 | Registers used / allocated；total SMEM；Achieved occupancy | 178 / 184 reg/thread；50,176 B/CTA；**16.57%** | 128-thread CTA 受 Resource 限制最多 2 CTA/SM，约 8 resident warps；低 occupancy 是该 tile 的结构事实，不可直接与 Fused 数值比快慢 |
| 发射连续性 | Issue active | **13.01%** | scheduler 发射不连续，存在明显 dependency / supply gap |
| Compute pressure | TC / ALU / FMA active；XU instruction-rate utilization | **57.05% / 4.62% / 1.00%；1.11%** | 在同为 cycles-active 的 TC/ALU/FMA 中，TC 最高，FC1 具有明显 Tensor Core 特征；XU 口径不同，不加入排序 |
| Memory pressure | DRAM / L2 / L1 throughput；LSU instruction-rate utilization | **54.44% / 54.22% / 41.27%；19.02%** | DRAM/L2 压力中等，FC1 同时存在 memory 活动 |
| 等待落点 | Long scoreboard / Wait / Barrier / Short scoreboard sample share | **43.55% / 29.03% / 4.24% / 3.47%** | memory/dependency wait 是主要样本，和低 Issue active 一致 |
| 发射受限线索 | Math pipe / MIO / LG throttle sample share | **8.86% / 0.86% / 0.46%** | 有一定 math-pipe pressure，但小于 dependency wait |
| Spill 排除 | dynamic spill load/store | **0 / 0** | 无动态 spill/refill |

**本表结论：** Triton FC1 呈现 **TC + memory/dependency mixed signature**；Tensor Core 活跃的同时，
memory/dependency wait 仍明显。本轮没有 roofline / operator-range traffic，不能 formal 判为 compute-bound 或 memory-bound。

### 3.3 Triton FP8 SwiGLU

| 观察问题 | 关键指标 | 数值 | 这说明什么 |
|---|---|---:|---|
| 驻留并行度 | Registers used / allocated；total SMEM；Achieved occupancy | 34 / 40 reg/thread；1,024 B/CTA；**88.91%** | resident warps 已很多，问题不是缺 occupancy |
| 发射连续性 | Issue active | **24.64%** | 即使 occupancy 高，scheduler 仍不能持续发射，说明更多 resident warps 也未消除主要等待 |
| Compute pressure | TC / ALU / FMA active；XU instruction-rate utilization | **0 / 6.55% / 12.66%；21.64%** | 没有 TC；观察到 XU、FMA、ALU 活动，但不同指标不能换算为工作量构成 |
| Memory pressure | DRAM / L2 / L1 throughput；LSU instruction-rate utilization | **94.60% / 32.39% / 15.83%；2.08%** | DRAM 已接近峰值，是最清晰的硬件压力 |
| 等待落点 | Long scoreboard / Wait / Short scoreboard / Barrier sample share | **84.64% / 5.21% / 3.62% / 0** | 样本几乎由 long scoreboard 主导，与外存依赖一致 |
| 发射受限线索 | Math pipe / MIO / LG throttle sample share | **0.15% / 0.35% / 0.01%** | compute / MIO throttle 都不是主要矛盾 |
| Spill 排除 | dynamic spill load/store | **0 / 0** | 无动态 spill/refill |

**本表结论：** Triton SwiGLU 是明确的 **DRAM / memory-dependency dominated** kernel；高 occupancy
没有改变这一点。

### 3.4 Triton FP8 FC2

| 观察问题 | 关键指标 | 数值 | 这说明什么 |
|---|---|---:|---|
| 驻留并行度 | Registers used / allocated；total SMEM；Achieved occupancy | 182 / 184 reg/thread；50,176 B/CTA；**16.44%** | 与 FC1 类似，128-thread CTA 最多约 2 CTA/SM；这是 tile/Resource 共同形成的低 resident-warp 状态 |
| 发射连续性 | Issue active | **20.17%** | 发射连续性高于 FC1，但仍有明显空档 |
| Compute pressure | TC / ALU / FMA active；XU instruction-rate utilization | **42.85% / 10.11% / 3.60%；3.32%** | 在同为 cycles-active 的 TC/ALU/FMA 中，TC 最高；ALU/FMA 高于 FC1，XU 只与同口径 XU 比较，均不能据此计算 absolute work |
| Memory pressure | DRAM / L2 / L1 throughput；LSU instruction-rate utilization | **63.56% / 46.23% / 31.67%；19.81%** | DRAM 相对利用率高于 FC1，compute 与 memory 指标同时活跃 |
| 等待落点 | Wait / Long scoreboard / Short scoreboard / Barrier sample share | **33.43% / 31.05% / 9.52% / 3.13%** | dependency wait 仍主导，且短依赖样本高于 FC1 |
| 发射受限线索 | Math pipe / MIO / LG throttle sample share | **4.82% / 0.87% / 0.95%** | throttle 不是首要样本来源 |
| Spill 排除 | dynamic spill load/store | **0 / 0** | 无动态 spill/refill |

**本表结论：** Triton FC2 呈现 **TC + memory mixed signature**；不能只按 GEMM 名称把它归为纯 compute-bound。

### 3.5 Triton FP8 TopK reduce

| 观察问题 | 关键指标 | 数值 | 这说明什么 |
|---|---|---:|---|
| 驻留并行度 | Registers used / allocated；total SMEM；Achieved occupancy | 54 / 56 reg/thread；1,024 B/CTA；**61.49%** | occupancy 不低，不是首要限制 |
| 发射连续性 | Issue active | **6.29%** | scheduler 绝大多数时间无法持续发射 |
| Compute pressure | TC / ALU / FMA active；XU instruction-rate utilization | **0 / 2.83% / 2.74%；0** | 已展示 compute-path utilization 很低 |
| Memory pressure | DRAM / L2 / L1 throughput；LSU instruction-rate utilization | **95.67% / 33.27% / 16.18%；2.06%** | DRAM 已接近峰值，是绝对主压力 |
| 等待落点 | Long scoreboard / Wait / Short scoreboard / Barrier sample share | **97.34% / 0.83% / 0.24% / 0** | 几乎全部样本都在等待长延迟 memory dependency |
| 发射受限线索 | Math pipe / MIO / LG throttle sample share | **0.04% / 0.01% / 0** | compute 与 shared-memory throttle 均可排除为主要矛盾 |
| Spill 排除 | dynamic spill load/store | **0 / 0** | 无动态 spill/refill |

**本表结论：** Triton TopK reduce 是明确的 **DRAM / long-scoreboard dominated** kernel。

### 3.6 NCU 横向结论

| Launch | 在自身实现中的耗时权重 | NCU signature | 对当前 2× 目标的含义 |
|---|---:|---|---|
| Latest opt Fused main | Opt 的完整 main kernel | TC 35.34%、DRAM 52.70%、Issue active 27.81%；等待原因分散 | 混合 launch，没有单一 launch-wide ceiling；必须用 phase time 决定深挖顺序 |
| Triton FC1 | Triton wall 的 46.61% | TC + memory/dependency mixed，同时有 long-scoreboard / Wait | reference 的最大时间池，但 Fused 的 whole-launch TC 数值不能与它做 FC1 speedup |
| Triton SwiGLU | 5.86% | DRAM 94.60%、long scoreboard 84.64% | 独立 materialization 明显受 memory 限制；fusion 具备消除该 GMEM 边界的机制机会，但本轮没有 operator-range traffic 来量化收益 |
| Triton FC2 | 30.15% | TC + DRAM mixed，dependency wait 明显 | reference 的第二大时间池；当前证据不支持把它视为单一 TC ceiling |
| Triton TopK reduce | 10.46% | DRAM 95.67%、long scoreboard 97.34% | reference 输出归并本身已经接近 DRAM ceiling；不能把它与 Fused atomic Scatter 当成同一个 kernel 实现 |

因此，**不能**用 `Fused TC 35.34% < Triton FC1 TC 57.05%` 推导“Fused FC1 更差”：前者平均了
完整 fused kernel，后者只覆盖 FC1。NCU 在这里提供硬件性质，phase/op 时间才提供优化预算。

### 3.7 Scatter exact-PC 证据

源码与 SASS 已闭合以下 producer-consumer 链：
`FC2 epilogue R2S → sC BF16 SMEM + cached token/weight SMEM → LDS → FP32 scale → BF16x8 pack → REDG → CTA sync`。
四个 PC bundle 对应静态展开的四个 FC2 output tiles。

| Edge | Exact PC anchors | Dynamic PC samples（四个 bundle 合计） | 当前能说明什么 |
|---|---|---:|---|
| SMEM tail load | `0x107e0 / 0x11a40 / 0x12cc0 / 0x13f30` | bundle 全部 samples 1,244（0.72%）；其中 MIO throttle 793、Wait 280 | Scatter 的 sC/metadata SMEM load 路径存在 MIO / Wait 症状 |
| Scale + pack | 每个 REDG 前的 `FMUL ×8 + F2FP ×4` | 静态相邻关系已确认；没有独立 phase time | 能确认数据变换位置，不能单独量化耗时 |
| REDG BF16x8 | `0x10930 / 0x11bc0 / 0x12e60 / 0x140d0` | bundle 全部 samples 917（0.53%）；其中 Wait 477、MIO throttle 256 | REDG bundle 有动态等待，但不是 launch-wide 唯一热点 |
| Post-Scatter sync | barrier `0x10960 / 0x11bf0 / 0x12e90 / 0x14110`；sample `0x10970 / 0x11c00 / 0x12ea0 / 0x14130`（末项为紧邻 `BRA`） | bundle 全部 samples 4,717（2.72%）；其中 Barrier 4,629 | 等待位置在每个 output tile 的 Scatter 后；barrier 可能是上游不均衡或 REDG completion 的结果，尚不能视作根因 |

定位层级为 **semantic-localized，尚未 mechanism-localized**：当前证据无法区分 SMEM layout、REDG
完成延迟和 warp/task imbalance 谁是 post-sync 等待的根因，因此不把任一项直接写成优化方案。

### 3.8 当前瓶颈判断

| 优先级 | 当前判断 | 证据闭合程度 |
|---|---|---|
| P0 | Scatter 仍是首要深挖对象 | 371.964 µs / 27.91%；SMEM load、REDG、post-sync 均已有 exact-PC 动态样本，但根因仍未分离 |
| P1 | Route/Q0 是已验证杠杆，当前 residual 原因未知 | 94.038 µs / 7.06%；本轮 whole-launch NCU 不能提供 phase-local root cause |
| P1 | FC1/FC2 是大时间池，但当前没有单 phase cadence 证据 | 合计 699.948 µs / 52.59%；Fused whole-launch TC active 不能拆给两个 GEMM |
| 结构约束，非根因 | Fused main 只有 1 persistent CTA/SM | 110 CTA × 9 warps，Achieved occupancy 18.75%；grid、Register、SMEM 都不会自然提供第 2 CTA。它解释 latency-hiding 条件，但尚未定位哪个 phase 因此变慢 |
| 排除项 | 旧 Production spill 不是当前 blocker | current exact cubin compiler stack 0 B/thread，dynamic spill/refill 0 / 0 |

当前证据仍不能把 **248.032 µs** 缺口分摊给上述行，也不能假设 Scatter 的全部时间都可消除。

## 4. 下一步调查与优化（Next To Do）

| 顺序 | 下一步方向 | 接受条件 |
|---|---|---|
| 主线 1（现在） | 用 matched、低扰动 edge probe 把 Scatter 分成 `SMEM LDS → scale/pack → REDG → post-sync`，判断 barrier 是 REDG completion 还是 warp/task imbalance 的结果 | 使用 matched control；保持 Registers/stack/SMEM/spill 身份，审计 SASS drift 与 timing perturbation；得到可复算的 dominant edge，否则只保留 semantic-localized 结论 |
| 主线 2（条件触发） | 只有主线 1 证明具体机制后，做 production-embedded 单变量或明确 bundle counterfactual | correctness、zero-spill 通过，目标 edge 与 fresh 未插桩 M8192 E2E 同向下降；不得把组合变化纯归因于 REDG 数量 |
| 主线 3（条件触发） | 若 Scatter 仍不足以收口，测试 persistent topology / Resource bundle 是否能增加有效 latency hiding | 必须同时说明 grid、Register、SMEM 和 task ownership；只降低单一 Resource 但仍是 1 CTA/SM 不算有效实验 |
| 主线 4（条件触发） | 再定位当前 token-major Route/Q0 residual；最后才做 FC1/FC2 phase-local cadence 调查 | 每次先得到 exact phase/instruction/value → producer/consumer 证据，再进入源码优化 |
| 收口门禁 | 每个 accepted change 后重跑未插桩 M8192 target benchmark | `Latest opt ≤1106.701 µs`、正确性通过且保持 zero-spill |

直接整体 Stage4 已因 SMEM 超限拒绝，compact epilogue 已因 stack 与性能回退拒绝；在出现新的受控证据前，
不重新投入这两个方向，也不把跨 worktile overlap 写成已确认瓶颈。

原始汇总见 [result.md](result.md)，统一证据见 [evidence.json](evidence.json)，NCU 证据卡见
[ncu_evidence.json](ncu_evidence.json)，横表数据见
[phase_op.csv](phase_op.csv)。既有受控依据见 [exp_010 Scatter fidelity](../../exp_010_scatter_vs_chain_finalize/results/result.md)、
[exp_014 Scatter 8-warp](../../exp_014_scatter_8warp/results/result.md) 和
[exp_016 Route/Q0 token-major](../../exp_016_route_q0_token_major_reuse/results/result.md)。
