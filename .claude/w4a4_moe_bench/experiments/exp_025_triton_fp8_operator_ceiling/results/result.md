# exp_025：SGLang Triton FP8 逐算子性能上界

## 结论

M8192 的 graph wall 为 **2156.058 μs**。FC1 / FC2 分别占 **46.61% / 30.15%**，TopK reduce 占 **10.46%**；三者合计 **87.22%**，是当前 chain 的主耗时。

相对同一 5KP GPU UUID 上实测、source-contract-compatible 的 full-card calibrated TC roof，FC1 的 Useful / Executed efficiency 为 **47.74% / 53.57%**，FC2 为 **36.89% / 41.39%**；padding efficiency 为 **89.12%**。这是本报告的主 ceiling 百分比估计。

Routing、Q0/Q1、SwiGLU、TopK reduce 没有完整且合法的 operator roof，保持 unavailable；不会用 logical payload rate 或单项 NCU utilization 冒充 hardware-ceiling efficiency。所有 op 仍缺少独立同契约 SOTA anchor，因此 SOTA distance 也保持 unavailable。

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

## 3. M8192 逐算子 ceiling card

| Op | Graph share | Hardware ceiling efficiency | Padding efficiency | SOTA distance |
|---|---:|---|---:|---|
| Routing / scheduler | 1.87% | unavailable（no legal complete-op roof） | — | unavailable |
| Q0（input quant） | 2.28% | unavailable（no legal complete-op roof） | — | unavailable |
| FC1 | 46.61% | 47.74% Useful / 53.57% Executed（same-UUID calibrated TC estimate） | 89.12% | unavailable |
| SwiGLU | 5.86% | unavailable（no legal complete-op roof） | — | unavailable |
| Q1（intermediate quant） | 2.69% | unavailable（no legal complete-op roof） | — | unavailable |
| FC2 | 30.15% | 36.89% Useful / 41.39% Executed（same-UUID calibrated TC estimate） | 89.12% | unavailable |
| TopK reduce / finalize | 10.46% | unavailable（no legal complete-op roof） | — | unavailable |

主表只呈现对用户有意义的 ceiling 百分比。Raw TFLOP/s、logical payload rate、cycle proxy 计数与 nominal diagnostic 均下沉到 [model.json](model.json)。

## 4. GEMM 与 NCU 交叉证据

| GEMM | Calibrated Useful ceiling efficiency | Calibrated Executed ceiling efficiency | Padding efficiency | TC active（诊断） |
|---|---:|---:|---:|---:|
| FC1 | 47.74% | 53.57% | 89.12% | 57.05% |
| FC2 | 36.89% | 41.39% | 89.12% | 42.85% |

Calibrated efficiency 的分母来自 exp_026 中同一 GPU UUID 的 `QMMA.16832.F32.E4M3.E4M3` full-card measured roof。目标 Triton 的 FP8 instruction 目前由 source/dispatch contract 推导，缺少可追溯 target SASS binding，因此百分比保持 diagnostic。Sibling-GPU NCU 的 `l1tex__cycles_elapsed.sum` 只保留在 model 中作交叉诊断，不参与主百分比。

Dispatch/source-derived physical routed rows = **73536**，logical rows = 65536；由 fixture per-expert occupancy 按 `BLOCK_SIZE_M=64` 逐 expert 向上取整得到，不是 NCU dynamic counter。

`TC active` 与 calibrated efficiency 的分母、采集语义不同，只并列展示，不作差也不互证。

| Op | TC active | DRAM throughput | Issue active | Achieved occupancy | PC-sampled stall reason share (Wait / Long / Short / Barrier) |
|---|---:|---:|---:|---:|---|
| FC1 | 57.05% | 54.44% | 13.01% | 16.57% | 29.03% / 43.55% / 3.47% / 4.24% |
| FC2 | 42.85% | 63.56% | 20.17% | 16.44% | 33.43% / 31.05% / 9.52% / 3.13% |
| SwiGLU | 0.00% | 94.60% | 24.64% | 88.91% | 5.21% / 84.64% / 3.62% / 0.00% |
| TopK reduce / finalize | 0.00% | 95.67% | 6.29% | 61.49% | 0.83% / 97.34% / 0.24% / 0.00% |

NCU 来自同配置 sibling GPU，只允许 normalized launch-local 诊断；NCU duration、跨 launch 可加 traffic 与 NSys 时间均不混算。Stall 列的分母是全部非 `_not_issued` PC samples，只展示四类，合计不要求 100%。

## 5. 下一步最小补证

1. 若要判断 Q0/Q1、SwiGLU 或 TopK reduce 的硬件余量，分别补同责任 standalone calibration；在此之前保持 unavailable。
2. 用独立、同 shape/precision/layout/protocol 的 grouped-GEMM 实现补 FC1/FC2 SOTA anchor；否则 SOTA gap 保持 unavailable。
3. 只有需要验证跨 M 稳定性时，再为 M256 补一次轻量 NSys timeline；M1024 不做插值。

公式、input digest 与 identity status 见 [model.json](model.json)。
