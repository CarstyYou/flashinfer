# exp_019：Latest Opt vs Eric Stage4 Dataflow Bottleneck

结论先行：**继续以 Latest Opt 为主干，不整体合并 Eric kernel。** M8192 时 Eric 慢
`352.340 µs / 26.35%`；主要回退位于 Route/Q0、Scatter 和 FC2。Eric 唯一明确领先的主要阶段是
FC1 Gate+Up+SwiGLU，快 `23.513 µs / 5.29%`，值得把其 pipeline 深度作为单变量候选移植到 Opt，
但当前证据不能把这项收益单独归因于 Stage4。

## 1. 对比约束与性能锚点

| 项目 | Latest Opt | Eric Stage4 |
|---|---|---|
| 完整边界 | BF16 input → NVFP4 fused MoE → BF16 output | 相同 |
| 形状 | `E=256, H=2048, I_tp=512, topk=8` | 相同 |
| CTA | 8 math warps + 1 TMA warp，288 threads | 4 math warps + 1 TMA warp，160 threads |
| FC1 / pipeline | paired N64 Gate/Up，AB stage ≤2 | N128 Gate 后 N128 Up，AB stage=4 |
| Epilogue | M128 | 两个 M64 pass |
| 正式 latency | exp_018 correctness-qualified、未插桩 benchmark | 相同协议 |

| Case | Latest Opt | Eric Stage4 | Eric 相对 Opt | 判定 |
|---:|---:|---:|---:|---|
| M1024 | 571.110 µs | 549.762 µs | 快 21.348 µs；`+3.88%` speedup | 正式 benchmark 有效；本轮 phase 归因未闭合 |
| M8192 | **1337.271 µs** | **1689.611 µs** | **慢 352.340 µs / 26.35%** | 本报告主 case |

M8192 的 matched phase probe 扰动为 Opt `+0.482%`、Eric `+0.439%`，两臂差分仅
`0.043 percentage point`。fresh no-marker control gap 为 `369.408 µs`，与正式 gap 的量级比为
`1.048×`；phase 差值合计 `369.466 µs`，与 probe event gap `370.464 µs` 仅差 `0.998 µs`。
因此 M8192 可以做 phase 归因，但 phase 数值仍是 diagnostic estimate，不替代正式 latency。

M1024 的 fresh control 仅显示 Eric 快 `7.168 µs`，而正式 benchmark 是 `21.348 µs`，幅度相差
`2.98×`。方向一致但跨来源幅度未过门禁，因此不使用本轮 M1024 phase 数据解释正式 crossover。

## 2. Phase 耗时与优劣

两侧都按同一语义边界计时：

`Clear/Histogram/Prefix → Route/Q0/Pack → Claim → FC1 Gate+Up+SwiGLU → Q1 → FC2+epilogue+R2S → Scatter`

### 2.1 M8192 横向时间对照

`Delta = Eric - Opt`；正值表示 Eric 更慢。FC2 行包含 epilogue、R2S 与 pre-scatter sync，
没有把 GEMM 尾部转嫁给 Scatter。Triton FP8 列复用 exp_017 的 12-node SGLang chain timeline；
`—` 表示没有独立 kernel 边界。它与两个 NVFP4 fused kernel 的 precision、实现和计时方式不同，
仅用于观察时间分布，不参与 `Delta`，也不计算 phase speedup。

| Phase / logical stage | Latest Opt（time / own share） | Eric Stage4（time / own share） | Triton FP8 Chain（time / own share） | Delta |
|---|---:|---:|---:|---:|
| Clear + Histogram + Prefix / Routing | 35.199 µs / 2.68% | 34.638 µs / 2.06% | 40.384 µs / 1.87% | -0.561 µs |
| Route + Q0 + Pack + publish / Q0 | **107.567 µs / 8.20%** | **348.508 µs / 20.71%** | 49.056 µs / 2.28% | **+240.941 µs** |
| Claim + cache + control | 27.158 µs / 2.07% | 26.210 µs / 1.55% | — | -0.948 µs |
| FC1 Gate + Up + SwiGLU | **444.413 µs / 33.84%** | **420.900 µs / 25.03%** | 1130.877 µs / 52.45% | **-23.513 µs** |
| Q1 | 34.022 µs / 2.59% | 55.118 µs / 3.28% | 57.920 µs / 2.69% | +21.096 µs |
| FC2 GEMM + epilogue + R2S | **233.033 µs / 17.74%** | **289.432 µs / 17.20%** | 650.142 µs / 30.15% | **+56.399 µs** |
| Scatter / TopK reduce | **372.003 µs / 28.34%** | **464.452 µs / 27.59%** | 225.696 µs / 10.46% | **+92.449 µs** |
| CTA residual | 35.192 µs / 2.68% | 14.421 µs / 0.86% | — | -20.771 µs |
| Launch skew / early finish / Graph bubble | 24.327 µs / 1.85% | 28.701 µs / 1.71% | 2.336 µs / 0.11% | +4.374 µs |
| **Diagnostic/timeline denominator** | **1312.913 µs / 100%** | **1682.379 µs / 100%** | **2156.058 µs / 100%** | **+369.466 µs** |

Triton FP8 的原始 node 组成与计时口径见 [exp_017 bottleneck](../../exp_017_opt_vs_triton_phase_share/results/opt_vs_triton_fp8_bottleneck.md)。

### 2.2 当前 phase 判断

- Eric 回退的第一来源是 Route/Q0，占 diagnostic gap 的约 `65%`；其次是 Scatter、FC2 和 Q1。
- Eric 的 FC1 bundle 确实更快，但 Stage、warp layout、Gate/Up tile、epilogue 和 accumulator
  lifetime 同时变化，不能把 `23.513 µs` 归给其中任一项。
- M1024 只保留“Eric 优势可能来自 FC1/Scatter 对其他回退的抵消”这一待验证方向，不作为事实结论。

## 3. M8192 成对 Production NCU

两侧都是一个完整 fused MoE kernel，因此 NCU 可以直接做 kernel-to-kernel 对比。NCU duration 只作
诊断；正式性能仍取第 1 章 benchmark。

### 3.1 Resource、Schedule 与 Utilization

| 观察项 | Latest Opt | Eric Stage4 | 直接读法 |
|---|---:|---:|---|
| Registers / thread | 165 | **254** | Eric 高 89 regs/thread |
| Total SMEM / CTA | 84,992 B | 94,208 B | 两侧都只能驻留 1 CTA/SM |
| Achieved occupancy | **18.76%** | **10.41%** | Eric resident warps 更少 |
| Dynamic spill load / store | **0 / 0** | **947,200 / 509,600** | Eric production cubin 有动态 spill/refill |
| Issue active | **27.81%** | **18.99%** | Eric 发射更不连续 |
| TC / ALU / FMA / XU | 35.37% / 11.76% / 7.11% / 2.05% | 27.05% / 7.74% / 4.66% / 1.61% | Eric 各 compute path 利用率均更低 |
| DRAM / L2 / L1 / LSU | 52.32% / 52.53% / 48.29% / 33.31% | 42.79% / 42.73% / 38.20% / 23.02% | Eric 更慢但没有把 memory paths 压满 |
| Warp stalls：Wait / Long / Short / Barrier | 22.02% / 9.86% / 8.04% / 9.84% | **33.18% / 19.65% / 11.26%** / 6.49% | Eric dependency-wait 样本更多 |
| Warp throttles：Math / MIO / LG | 14.06% / 5.31% / 0.04% | 0.26% / 2.84% / 0.41% | Eric 不是 math-pipe 饱和问题 |

Stall/Throttle 是各自 launch 中 non-`_not_issued` PC-sampling reason 的样本占比，不是 elapsed-time 百分比。

### 3.2 完整算子的可加计数

| Metric | Latest Opt | Eric Stage4 | Eric - Opt |
|---|---:|---:|---:|
| DRAM total bytes | 884.791 MB | 918.458 MB | +3.81% |
| DRAM read bytes | 682.019 MB | 688.003 MB | +0.88% |
| L2 total sectors | 109.239 M | 115.331 M | +5.58% |
| L2 write sectors | 5.439 M | 9.515 M | +74.94% |
| Global load request footprint | 550.565 MB | **4307.657 MB** | **+682.41% / 7.82×** |
| Local load / store footprint | **0 / 0** | **121.242 / 130.458 MB** | Eric 独有 |
| Executed warp instructions | 393.174 M | 351.080 M | -10.71% |
| Tensor instructions | 31.310 M | 31.310 M | 相同 |

`Global load request footprint` 是 LSU/L1 request 侧工作量，不等于实际 DRAM traffic。Eric 的 request
高 `7.82×`，但 DRAM read 只高 `0.88%`，说明大部分重复请求被 cache 承接；它仍会消耗 load、地址计算
与 cache hierarchy 资源。源码也确认当前 benchmark 的 `share_input_across_experts=false` 路径中，Opt
按 token 复用一次 BF16 load/absmax 给 topk=8，而 Eric 按 routed pair 重做。

### 3.3 瓶颈判断

1. **Route/Q0 是双方最大、证据最完整的差异。** Phase 回退 `240.941 µs`，源码 work mapping 和
   `7.82×` Global load request footprint 同向；“重复 load/absmax 是主要原因”仍标为 inference，
   因为尚无单变量 Eric ablation。
2. **Eric 存在明确 Resource / Schedule 代价。** 254 regs/thread、动态 spill、10.41% occupancy、较低
   Issue active 与更多 Wait/scoreboard 样本同时出现；但 whole-kernel NCU 不能计算 spill 的独立 latency。
3. **数学工作不是差异来源。** 两侧 Tensor instructions 完全相同，Eric 甚至执行更少 warp instructions；
   回退来自 data movement、resource lifetime 与 schedule，而不是额外 Tensor Core 数学量。
4. **FC1 优势是真实的 bundle 结果，不是已定位机制。** 即使 Eric 整体有 spill，其 FC1 phase 仍快
   `23.513 µs`；因此不能写成“spill 使 FC1 变慢”，也不能写成“Stage4 已被证明有效”。

## 4. 可整合机制与 Next To Do

| 判定 | 设计选择 | 依据 / 下一步 |
|---|---|---|
| 保留 Opt | token-major Route/Q0 | Eric 在 M8192 的该 phase 慢 240.941 µs；Opt 避免 topk=8 重复 input load/absmax |
| 保留 Opt | 8-warp Scatter | Eric 4-warp compact 路径慢 92.449 µs；当前没有可借鉴的正收益 |
| 保留 Opt | paired N64、zero-spill 主干 | Opt production NCU 为 0 spill/refill，Eric 有明确 local traffic |
| 候选实验 | 只移植 FC1 更深的 pipeline / buffer schedule | Eric FC1 bundle 快 23.513 µs；保持 Opt 的 8 warps、paired N64、epilogue 和其他 phase 不变，先尝试 Stage3/4 或生命周期互斥 buffer 复用 |
| 暂不移植 | Eric 整体 4-warp + compact M64 bundle | Q1、FC2、Scatter 均回退，且无法拆分各变量贡献 |

最小整合实验的门禁是：`只改 FC1 pipeline → compile zero-spill → correctness → M1024/M8192 fresh paired benchmark`。
只要出现 spill 或 M8192 无收益就 reject。若小 M crossover 对产品路径重要，应先重做一次同机 production
pair benchmark 稳定信号，再决定是否补 M1024 NCU；当前不值得直接投入深度 profile。

统一证据见 [phase_evidence.json](phase_evidence.json) 与 [ncu_evidence.json](ncu_evidence.json)；正式
benchmark 来源见 [exp_018 result](../../exp_018_triton_opt_eric_benchmark/results/result.md)。
