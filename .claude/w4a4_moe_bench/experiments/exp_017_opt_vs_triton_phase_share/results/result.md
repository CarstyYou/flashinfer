# exp_017：Latest opt FP4 vs SGLang Triton FP8 Phase/Op 对照

面向客户 2× 目标的正式分析见 [opt_vs_triton_fp8_bottleneck.md](opt_vs_triton_fp8_bottleneck.md)。

## 结论

M=8192 的完整 benchmark 中，Latest opt 为 **1354.73 µs**，SGLang Triton FP8 为
**2213.40 µs**；Latest opt 快 **63.38%**（latency ratio 1.63×），还未达到 2× 目标。

横向分布揭示了两种实现不同的时间结构：

- Triton 的 `FC1 + SwiGLU + FC2` 占 **82.60%**；主要时间仍在两个 GEMM。
- Latest opt 的同区域占 **52.59%**；`Scatter` 单独占 **27.91%**，已成为第二大 phase。
- Latest opt 在 FC1/FC2 区域的观测时间明显更短，但 `Scatter` 比 Triton 的末端
  `TopK reduce` 更长。两者边界和算法不同，这里只定位时间去向，不计算逐行 speedup。

## Phase/Op 横向对照

以下均为 5 次 replay 的中位数；百分比只使用各自一侧的总时间作分母。
Latest opt 的分母是插桩 phase 投影闭合得到的 **1331.97 µs**，Triton 的分母是 CUDA Graph
首末 node 间的实际 GPU elapsed **2156.06 µs**。

| Logical region | Latest opt CuteDSL FP4：phase 投影 µs / 本侧占比 | SGLang Triton FP8：graph elapsed µs / 本侧占比 |
|---|---:|---:|
| Routing / scheduler | **75.40 / 5.65%**<br>Clear 12.36 / 0.93%<br>Histogram 8.92 / 0.67%<br>Prefix 13.91 / 1.04%<br>Publish 13.14 / 0.99%<br>Claim/cache/control 27.04 / 2.03% | **40.38 / 1.87%**<br>Align 26.91 / 1.25%<br>Count/sort 13.47 / 0.63% |
| Q0 / input pack | **94.04 / 7.06%**<br>Route + Q0 + Pack | **49.06 / 2.28%** `[45.57, 50.56]`<br>Fill 0.48 / 0.02%<br>Absmax 38.40 / 1.78%<br>Quant 10.18 / 0.47% |
| FC1 + SwiGLU | **451.76 / 33.96%**<br>Fused FC1 Gate/Up + SwiGLU | **1130.88 / 52.45%**<br>FC1 1004.67 / 46.61%<br>SwiGLU 126.14 / 5.86% |
| Q1 | **33.86 / 2.54%** | **57.92 / 2.69%**<br>Fill 0.64 / 0.03%<br>Absmax 38.72 / 1.80%<br>Quant 18.59 / 0.86% |
| FC2 + epilogue + R2S | **248.18 / 18.63%** | **650.14 / 30.15%**<br>FC2 `fused_moe_kernel` |
| Output aggregation | **371.96 / 27.91%**<br>Atomic Scatter | **225.70 / 10.46%**<br>TopK reduce/combine |
| Residual / skew | **56.65 / 4.25%**<br>CTA residual 31.07 / 2.34%<br>Launch skew 25.66 / 1.93% | **2.34 / 0.11%**<br>Graph-node bubble |

## 证据边界

- 两侧使用同一份 exp_001 M8192 fixture、完全相同的 routing/occupancy，以及同一块 5KP。
- Triton 数据来自 NSys 中 5 次完整 CUDA Graph replay；每次都通过有序 12-node 拓扑检查。
- Latest opt 数据来自 `%globaltimer` matched probe。Probe 相对 no-marker control 的扰动为
  **+1.92%**；两者均为 165 REG/thread、0 STACK、1024 B static SMEM，且没有静态
  spill/refill。
- Marker 保留了 OMMA/TMA/LDSM/barrier/atomic/reduction 数量，但引起了更广泛的 SASS
  scheduling/control-flow 重排。因此 Latest opt 行是可信的**诊断投影**，不是 production-exact
  phase latency。
- FP4 fused 与 FP8 chain 的精度、物化边界和末端聚合方式不同；表格用于观察热点，不能据此做
  逐行因果归因。

完整可审计数据见 [evidence.json](evidence.json)，横表数据见 [phase_op.csv](phase_op.csv)。
