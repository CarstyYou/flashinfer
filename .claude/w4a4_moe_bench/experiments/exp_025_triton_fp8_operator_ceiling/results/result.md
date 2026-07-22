# exp_025：SGLang Triton FP8 逐算子性能上界

## 结论

M8192 的 graph wall 为 **2156.058 μs**。FC1 / FC2 分别占 **46.61% / 30.15%**，TopK reduce 占 **10.46%**；三者合计 **87.22%**，是当前 chain 的主耗时。

相对同一 5KP GPU UUID 上实测、source-contract-compatible 的 full-card calibrated TC roof，FC1 的 Useful / Executed efficiency 为 **47.74% / 53.57%**，FC2 为 **36.89% / 41.39%**；padding efficiency 为 **89.12%**。由于 target SASS binding 缺失，这些值保持 diagnostic estimate。

第 3 节覆盖 FC1、SwiGLU、FC2、TopK reduce 四个主要 op；Routing、Q0、Q1 只保留在时间 accounting。SwiGLU / TopK reduce 的 NCU DRAM throughput 为 **94.60% / 95.67%**，只作为 sibling-GPU launch-local diagnostic。所有 op 的独立同契约 SOTA anchor 仍 unavailable。

当前 verdict：**accounting=vetted，GEMM percentage=diagnostic（same UUID measured roof；target SASS binding unavailable），coverage=partial**。M256/M1024 只有 E2E benchmark，不能插值出逐 op 占比。

## 1. Scope 与整体测量

```text
BF16 input → Routing → Q0 → FC1 → SwiGLU → Q1 → FC2 → TopK reduce → BF16 output
```

| M | exp_001 E2E | exp_018 独立 sanity | 差异 | per-op timeline |
|---:|---:|---:|---:|---|
| 256 | 774.032 μs | 768.593 μs | -0.70% | unavailable |
| 1024 | 765.011 μs | 760.301 μs | -0.62% | unavailable |
| 8192 | 2213.402 μs | 2150.661 μs | -2.83% | available |

M8192 NSys graph wall / exp_001 E2E = **97.41%**；工具与采样协议不同，因此只作一致性检查。

## 2. M8192 各 op 时间与占比

| Op | Median time | Graph share |
|---|---:|---:|
| Routing / scheduler | 40.384 μs | 1.87% |
| Q0（input quant） | 49.056 μs | 2.28% |
| FC1 | 1004.669 μs | 46.61% |
| SwiGLU | 126.144 μs | 5.86% |
| Q1（intermediate quant） | 57.920 μs | 2.69% |
| FC2 | 650.142 μs | 30.15% |
| TopK reduce / finalize | 225.696 μs | 10.46% |
| Graph bubble | 2.336 μs | 0.11% |

时间取五次 replay 的 median-of-sums，占比取每次 `op time / replay wall` 后的 median。component median sum 与 graph-wall median 相差 **+0.289 μs**，各项 median share 之和偏离 100% **+0.0195 pp**；两者都是 median non-additivity。

## 3. M8192 主要 op 的资源达成率

| Op | Graph share | 已观测资源达成率 | 对优化的含义 |
|---|---:|---|---|
| FC1 | 46.61% | TC Useful **47.74%** / Executed **53.57%**（source-contract diagnostic）；NCU DRAM throughput **54.44%**（sibling-GPU diagnostic）；Padding **89.12%** | 主耗时；现有 TC/DRAM diagnostic 均未逼近 ceiling。Achieved occupancy 16.57%，优先调查调度与访存等待。 |
| SwiGLU | 5.86% | NCU DRAM throughput **94.60%**（sibling-GPU diagnostic） | DRAM throughput 接近上限；优先减少 traffic、改善 locality 或融合。 |
| FC2 | 30.15% | TC Useful **36.89%** / Executed **41.39%**（source-contract diagnostic）；NCU DRAM throughput **63.56%**（sibling-GPU diagnostic）；Padding **89.12%** | DRAM diagnostic 比 TC 更接近上限，但仍未封顶；Achieved occupancy 16.44%，继续区分 memory wait 与 compute schedule。 |
| TopK reduce / finalize | 10.46% | NCU DRAM throughput **95.67%**（sibling-GPU diagnostic） | DRAM throughput 接近上限；优先减少读流量、优化 reduction/locality 或与前级融合。 |

TC diagnostic 的分母来自 exp_026 中同一 GPU UUID 的 `QMMA.16832.F32.E4M3.E4M3` full-card measured roof。目标 Triton 的 FP8 instruction 目前由 source/dispatch contract 推导，缺少可追溯 target SASS binding，因此不能称 exact MFU。NCU DRAM throughput 来自 sibling GPU，只允许 launch-local diagnostic，不能与 NSys 时间混算或称 complete-op efficiency。

## 4. 优化优先级与最小下一步

1. **FC1**：占比最高，优先调查低 occupancy、访存等待与 grouped-GEMM 调度。
2. **TopK reduce + SwiGLU**：DRAM throughput 已接近上限，优先减少 traffic、改善 locality 或融合。
3. **FC2**：继续区分 memory wait 与 TC schedule；现有 diagnostic 不能单独决定优化顺序。
4. 只有结论需要升级时，再补 target SASS binding、同责任 standalone 或独立 SOTA anchor。

Raw NCU metrics、TFLOP/s、cycle proxy、公式、input digest 与 identity status 见 [model.json](model.json)。
