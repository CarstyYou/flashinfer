# exp_019：Current Opt vs Eric Stage4 Dataflow Bottleneck

结论先行：**继续以 Current Opt 为主干，不整体合并 Eric kernel。** exp_022 的 Scatter 向量化
S2R 已使 M8192 Current Opt 降至 `1224.544 µs`；对照 Eric 的 `1689.611 µs`，Eric 慢
`465.067 µs / 37.98%`。Current Opt 的 Scatter 已略快于 Triton，当前最明确的相对短板转为 Q0。
Eric 的 FC1 Gate+Up+SwiGLU 仍显示局部优势，但跨 capture 幅度不能用于单变量归因，更不能证明
Stage4 是原因。

## 1. 对比约束与性能锚点

| 项目 | Current Opt | Eric Stage4 |
|---|---|---|
| 完整边界 | BF16 input → NVFP4 fused MoE → BF16 output | 相同 |
| 形状 | `E=256, H=2048, I_tp=512, topk=8` | 相同 |
| CTA | 8 math warps + 1 TMA warp，288 threads | 4 math warps + 1 TMA warp，160 threads |
| FC1 / pipeline | paired N64 Gate/Up，AB stage ≤2 | N128 Gate 后 N128 Up，AB stage=4 |
| Epilogue | M128 | 两个 M64 pass |
| latency 来源 | M1024：exp_018；M8192：exp_022 correctness-qualified、未插桩 benchmark | exp_018 correctness-qualified、未插桩 benchmark |

| Case | Current Opt | Eric Stage4 | Eric 相对 Opt | 判定 |
|---:|---:|---:|---:|---|
| M1024 | 571.110 µs | 549.762 µs | 快 21.348 µs；`+3.88%` speedup | exp_022 未补测；保留 exp_018 锚点 |
| M8192 | **1224.544 µs** | **1689.611 µs** | **慢 465.067 µs / 37.98%** | 本报告主 case |

M8192 Current Opt 的新 phase probe 来自 exp_022，probe/control 扰动为 `-0.521%`；Eric phase
沿用 exp_019 的 matched capture。两者不是同轮 paired capture，因此 Current Opt 与 Eric 的 phase
差值只用于横向观察，不作为单变量因果证据。两侧 diagnostic denominator 相差 `499.086 µs`，与
benchmark gap `465.067 µs` 方向一致、量级接近。

M1024 的 fresh control 仅显示 Eric 快 `7.168 µs`，而正式 benchmark 是 `21.348 µs`，幅度相差
`2.98×`。方向一致但跨来源幅度未过门禁，因此不使用本轮 M1024 phase 数据解释正式 crossover。

## 2. Phase 耗时与优劣

两侧都按同一语义边界计时：

`Clear/Histogram/Prefix → Route/Q0/Pack → Claim → FC1 Gate+Up+SwiGLU → Q1 → FC2+epilogue+R2S → Scatter`

### 2.1 M8192 横向时间对照

FC2 行包含 epilogue、R2S 与 pre-scatter sync，没有把 GEMM 尾部转嫁给 Scatter。Triton FP8
列复用 exp_017 的 12-node SGLang chain timeline。`Speedup = Triton time / FP4 time - 1`；正值表示
FP4 更快，负值表示 FP4 仍慢于 Triton，`+100%` 对应 `2×` 性能。由于 precision、实现和计时方式
不同，speedup 用于定位各 logical phase 距目标的差距，不作为同精度、单变量的因果 speedup；没有
可对齐边界的行不计算。

| Phase / logical stage | Current Opt（time / own share） | Eric Stage4（time / own share） | Triton FP8 Chain（time / own share） | Opt vs Triton speedup | Eric vs Triton speedup |
|---|---:|---:|---:|---:|---:|
| Clear + Histogram + Prefix / Routing | 35.055 µs / 2.96% | 34.638 µs / 2.06% | 40.384 µs / 1.87% | +15.203% | +16.59% |
| Route + Q0 + Pack + publish / Q0 | **109.520 µs / 9.26%** | **348.508 µs / 20.71%** | 49.056 µs / 2.27% | **-55.208%** | **-85.92%** |
| Claim + cache + control | 29.741 µs / 2.51% | 26.210 µs / 1.55% | — | — | — |
| FC1 Gate + Up + SwiGLU | **465.720 µs / 39.36%** | **420.900 µs / 25.03%** | 1130.877 µs / 52.44% | **+142.823%** | **+168.68%** |
| Q1 | 35.450 µs / 3.00% | 55.118 µs / 3.28% | 57.920 µs / 2.69% | +63.383% | +5.08% |
| FC2 GEMM + epilogue + R2S | **232.799 µs / 19.67%** | **289.432 µs / 17.20%** | 650.142 µs / 30.15% | **+179.272%** | **+124.63%** |
| Scatter / TopK reduce | **216.753 µs / 18.32%** | **464.452 µs / 27.59%** | 225.696 µs / 10.47% | **+4.126%** | **-51.41%** |
| CTA residual + launch skew / Graph bubble | 58.255 µs / 4.92% | 43.122 µs / 2.56% | 2.336 µs / 0.11% | — | — |
| **Diagnostic/timeline denominator** | **1183.293 µs / 100%** | **1682.379 µs / 100%** | **2156.411 µs / 100%** | **+82.238%** | **+28.18%** |

Current Opt 的新 phase 数据与计时口径见 [exp_022 bottleneck](../../exp_022_scatter_vector_s2r/results/bottleneck.md)；
Triton FP8 的原始 node 组成见 [exp_017 bottleneck](../../exp_017_opt_vs_triton_phase_share/results/opt_vs_triton_fp8_bottleneck.md)。

### 2.2 当前 phase 判断

- 相对 Triton FP8，Current Opt 的 FC1、FC2 已超过 `+100%` 目标，Scatter 也由负收益转为
  `+4.126%`；Q0 `-55.208%` 是当前最明确的相对短板。
- Eric 的 FC1、FC2 同样超过 `+100%`，但 Q0 `-85.92%`、Scatter `-51.41%`，Q1 也仅 `+5.08%`。
- 跨 capture 横向观察中，Eric 相对 Current Opt 的最大差距是 Scatter `+247.699 µs` 和 Q0
  `+238.988 µs`；FC2 与 Q1 分别多 `56.633 µs`、`19.668 µs`。
- Eric 的 FC1 bundle 仍显示更快，但新横向差值 `44.820 µs` 不是 matched capture；原 matched
  证据为 `23.513 µs`。Stage、warp layout、tile、epilogue 和 accumulator lifetime 同时变化，
  不能把收益归给其中任一项。
- M1024 只保留“Eric 优势可能来自 FC1/Scatter 对其他回退的抵消”这一待验证方向，不作为事实结论。

## 3. Pre-exp022 Opt vs Eric 成对 Production NCU（历史机制证据）

本章是 Scatter 向量化之前的 matched whole-kernel NCU，Opt cubin 不代表第 1、2 章的 Current Opt。
它只保留用于解释 Eric 的 Route/Q0、spill 与资源调度差异；Current Opt 的新 Scatter 机制使用
exp_022 的 PC-scoped 证据。

### 3.1 Resource、Schedule 与 Utilization

| 观察项 | Pre-exp022 Opt | Eric Stage4 | 直接读法 |
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

| Metric | Pre-exp022 Opt | Eric Stage4 | Eric - Opt |
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

1. **Route/Q0 差异跨 capture 稳定。** 原 matched phase 回退为 `240.941 µs`，更新后的横向观察为
   `238.988 µs`，与源码 work mapping 和 `7.82×` Global load request footprint 同向；“重复
   load/absmax 是主要原因”仍标为 inference，因为尚无单变量 Eric ablation。
2. **Eric 存在明确 Resource / Schedule 代价。** 254 regs/thread、动态 spill、10.41% occupancy、较低
   Issue active 与更多 Wait/scoreboard 样本同时出现；但 whole-kernel NCU 不能计算 spill 的独立 latency。
3. **数学工作不是差异来源。** 两侧 Tensor instructions 完全相同，Eric 甚至执行更少 warp instructions；
   回退来自 data movement、resource lifetime 与 schedule，而不是额外 Tensor Core 数学量。
4. **Scatter 优化已有单变量闭环。** exp_022 将 `sC` load 从 `32 × LDS.U16` 改为
   `4 × LDS.128`，shared wavefront amplification 从 `3.9755×` 降至 `1.0000×`，M8192 benchmark
   提升 `+9.183%`；Candidate 为 146 regs/thread、0 spill。
5. **FC1 优势是 bundle 结果，不是已定位机制。** 原 matched evidence 中 Eric FC1 快
   `23.513 µs`；不能写成“spill 使 FC1 变慢”，也不能写成“Stage4 已被证明有效”。

## 4. 可整合机制与 Next To Do

| 判定 | 设计选择 | 依据 / 下一步 |
|---|---|---|
| 保留 Opt | token-major Route/Q0 | 原 matched gap `240.941 µs`、新横向 gap `238.988 µs`；Opt 避免 topk=8 重复 input load/absmax |
| 保留 Opt | 8-warp + vectorized S2R Scatter | exp_022 单变量实验 `+9.183%`，Current Opt Scatter 已比 Triton 快 `4.126%` |
| 保留 Opt | paired N64、zero-spill 主干 | Current Opt Candidate 为 146 regs/thread、0 spill；Eric 有明确 local traffic |
| 候选实验 | 只移植 FC1 更深的 pipeline / buffer schedule | 原 matched evidence 中 Eric FC1 快 `23.513 µs`；保持 Opt 的 8 warps、paired N64、epilogue 和其他 phase 不变，先尝试 Stage3/4 或生命周期互斥 buffer 复用 |
| 暂不移植 | Eric 整体 4-warp + compact M64 bundle | Q1、FC2、Scatter 均回退，且无法拆分各变量贡献 |

最小整合实验的门禁是：`只改 FC1 pipeline → compile zero-spill → correctness → M1024/M8192 fresh paired benchmark`。
只要出现 spill 或 M8192 无收益就 reject。若小 M crossover 对产品路径重要，应先重做一次同机 production
pair benchmark 稳定信号，再决定是否补 M1024 NCU；当前不值得直接投入深度 profile。

历史 matched 证据见 [phase_evidence.json](phase_evidence.json) 与 [ncu_evidence.json](ncu_evidence.json)；
Current Opt 的新 benchmark、phase 与 Scatter PC-scoped 证据见
[exp_022 result](../../exp_022_scatter_vector_s2r/results/result.md) 和
[exp_022 bottleneck](../../exp_022_scatter_vector_s2r/results/bottleneck.md)；Eric 与 M1024 benchmark
来源见 [exp_018 result](../../exp_018_triton_opt_eric_benchmark/results/result.md)。
