# exp_024：CUTLASS Chain 逐算子性能上界

## 结论

M8192 下，相对 5KP 实测 NVFP4 Tensor Core ceiling，FC1 的 Useful / Executed efficiency 为 **47.11% / 58.34%**，FC2 为 **29.44% / 36.45%**；两者 padding efficiency 均为 **80.76%**。

非 GEMM op 不能套 MFU。基于 NCU physical bytes 与 exp_026 directional DRAM roof 的投影只作为资源诊断：Route/Q0/Pack 为 **25.63%（diagnostic，不是完整 op ceiling）**，SwiGLU/Q1 为 **54.13%（diagnostic，不是完整 op ceiling）**，Finalize 为 **unavailable（投影为 114.16% > 100%，scope 不闭合）**。

硬件 ceiling verdict 为 **accept**；operator SOTA distance 仍为 unavailable。计算分母来自同一 RTX 5KP SKU、不同 GPU UUID 的实测迁移，不能表述为同卡同窗测量。

## 1. Scope 与证据边界

```text
BF16 input → Prefix → Route/Q0/Pack → GEMM metadata → FC1 → SwiGLU/Q1 → FC2 → Finalize → BF16 output
```

逐 op 时间来自 exp_002 canonical NSys；work counter 来自同 rerun 的独立 NCU replay。计算 ceiling 来自 exp_026 vetted `nvfp4-e2m1-vs16` record。consumer GPU `GPU-4a286357-c999-9547-3a04-25961b1ffd08` 与 calibration GPU `GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522` UUID 不同；二者同为 RTX 5KP、110 SM，因此这里只接受 same-SKU transfer。

| M | E2E benchmark | NSys active union | NSys / benchmark |
|---:|---:|---:|---:|
| 256 | 550.739 μs | 509.693 μs | 92.55% |
| 1024 | 577.220 μs | unavailable | unavailable |
| 8192 | 1665.779 μs | 1662.108 μs | 99.78% |

## 2. 各 op 时间与占比

| Op | M256 time / share | M8192 time / share |
|---|---:|---:|
| Prefix | 4.351 μs / 0.85% | 46.655 μs / 2.81% |
| Route/Q0/Pack | 11.008 μs / 2.16% | 245.600 μs / 14.78% |
| GEMM metadata | 1.983 μs / 0.39% | 2.048 μs / 0.12% |
| FC1 | 273.248 μs / 53.61% | 510.175 μs / 30.69% |
| SwiGLU/Q1 | 9.600 μs / 1.88% | 231.647 μs / 13.94% |
| FC2 | 204.031 μs / 40.03% | 408.255 μs / 24.56% |
| Finalize | 7.040 μs / 1.38% | 222.944 μs / 13.41% |

占比分母是全部 kernel interval 的 active union。相邻 launch 存在 PDL overlap，所以各行 duration/share 不是互斥分区；M256 / M8192 的 share 合计为 **100.3076% / 100.3138%**，cross-category overlap 为 **1.568 / 5.216 μs**，不重新归一化。

## 3. M8192 逐算子 ceiling 百分比

| Op | Hardware ceiling efficiency | Padding | SOTA distance |
|---|---|---:|---|
| Route/Q0/Pack | complete op ceiling unavailable; DRAM 25.63%（diagnostic，不是完整 op ceiling） | — | unavailable |
| FC1 | 47.11% Useful / 58.34% Executed（calibrated TC） | 80.76% | unavailable |
| SwiGLU/Q1 | complete op ceiling unavailable; DRAM 54.13%（diagnostic，不是完整 op ceiling） | — | unavailable |
| FC2 | 29.44% Useful / 36.45% Executed（calibrated TC） | 80.76% | unavailable |
| Finalize | complete op ceiling unavailable; DRAM unavailable（投影为 114.16% > 100%，scope 不闭合） | — | unavailable |

DRAM resource efficiency = `max(R/BWread, W/BWwrite, (R+W)/BWcopy) / NCU duration`。它只说明 physical DRAM 资源投影；大于 100% 直接标 invalid，不截断，也不称完整 op ceiling。

## 4. GEMM ceiling 达成率

| M | GEMM | Calibrated Useful | Calibrated Executed | Padding | Nominal Useful / Executed（次级） | TC active（诊断） |
|---:|---|---:|---:|---:|---:|---:|
| 256 | FC1 | 2.75% | 43.98% | 6.25% | 2.94% / 46.96% | 47.71% |
| 256 | FC2 | 1.84% | 29.45% | 6.25% | 1.97% / 31.45% | 33.85% |
| 8192 | FC1 | 47.11% | 58.34% | 80.76% | 50.31% / 62.30% | 62.69% |
| 8192 | FC2 | 29.44% | 36.45% | 80.76% | 31.43% / 38.92% | 38.30% |

主分母是 exp_026 对 exact `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` 指令的 full-card ~100 ms calibrated window；由于 exp_002 没有同 launch cycle counter，本报告没有使用 per-cycle normalization。Nominal 百分比只作架构次级参照；`TC active` 分母不同，也只并列诊断。

M256 的 6.25% padding efficiency 使 Nominal Useful MFU 为 1.97%–2.94%，而 Nominal Executed MFU 为 31.45%–46.96%；padding 是 Useful MFU 与 Executed MFU 差距的主要来源，但这不等同于证明它是相对 SOTA latency gap 的首要原因。

## 5. 未闭合项

1. 用 contract-equivalent、独立实现测 FC1/FC2 grouped-GEMM SOTA；在此之前不报告 SOTA gap。
2. Route/Q0/Pack、SwiGLU/Q1、Finalize 仍需同 layout/责任边界的 standalone calibration；DRAM resource diagnostic 不能替代它。

原始 throughput、公式输入、digest 与逐 op ceiling status 见 [model.json](model.json)。
