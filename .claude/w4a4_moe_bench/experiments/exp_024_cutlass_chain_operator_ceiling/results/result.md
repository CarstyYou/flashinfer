# exp_024：CUTLASS Chain 逐算子性能上界

## 结论

M8192 下，相对 5KP 实测 NVFP4 Tensor Core ceiling，FC1 的 Useful / Executed efficiency 为 **47.11% / 58.34%**，FC2 为 **29.44% / 36.45%**；两者 padding efficiency 均为 **80.76%**。

第 3 节覆盖 5 个主要 op；Prefix 与 GEMM metadata 只保留在时间 accounting。非 GEMM 不套 MFU：Route/Q0/Pack 为 DRAM Read **10.50%** / Write **12.94%**；1:1 Copy reference 25.63%（diagnostic），SwiGLU/Q1 为 DRAM Read **44.62%**（Write 3.50%），Finalize 为 DRAM Read **94.89%**（Write 6.55%）。

硬件 ceiling verdict 为 **accept**；operator SOTA distance 仍为 unavailable。各资源百分比不能相加或合成一个总分；DRAM 结论是 NCU replay 上的 scoped diagnostic。计算分母来自同一 RTX 5KP SKU、不同 GPU UUID 的实测迁移，不能表述为同卡同窗测量。

## 1. Scope 与证据边界

```text
BF16 input → Prefix → Route/Q0/Pack → GEMM metadata → FC1 → SwiGLU/Q1 → FC2 → Finalize → BF16 output
```

逐 op 时间来自 exp_002 canonical NSys；work counter 来自同 rerun 的独立 NCU replay。计算 ceiling 来自 exp_026 vetted `nvfp4-e2m1-vs16` record。consumer GPU `GPU-4a286357-c999-9547-3a04-25961b1ffd08` 与 calibration GPU `GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522` UUID 不同；二者同为 RTX 5KP、110 SM，因此这里只接受 same-SKU transfer。

| M | 在模型中的作用 | 逐 op 证据 |
|---:|---|---|
| 256 | 小规模 / padding 压力场景 | 可用 |
| 1024 | 整体 benchmark sanity | 不可用；禁止插值 |
| 8192 | Prefill 主优化场景 | 可用；第 3 节的主判定 case |

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

## 3. M8192 主要 op 的资源 ceiling 达成率

| Op | 时间占比 | 已校准资源达成率 | 对优化的含义 |
|---|---:|---|---|
| Route/Q0/Pack | 14.78% | DRAM Read **10.50%** / Write **12.94%**；1:1 Copy reference 25.63%（diagnostic） | 未接近 streaming DRAM roof；拆开 Route 与 Quant/Pack，检查 irregular access、量化计算和并行度。 |
| FC1 | 30.69% | TC Useful **47.11%** / Executed **58.34%**；DRAM Read **58.65%**（Write 18.35%）；Padding **80.76%** | 现有证据未见 TC 或 DRAM 单项逼近 ceiling；下一步区分计算调度与权重读取。 |
| SwiGLU/Q1 | 13.94% | DRAM Read **44.62%**（Write 3.50%） | 未接近 DRAM Read roof；优先检查 ALU/SFU、量化长指令、局部性与并行度。 |
| FC2 | 24.56% | TC Useful **29.44%** / Executed **36.45%**；DRAM Read **32.35%** / Write **47.32%**；1:1 Copy reference 86.87%（diagnostic）；Padding **80.76%** | 1:1 copy reference 提示 mixed-R/W traffic 值得调查；需 ratio-matched standalone 才能与 TC 优化排序。 |
| Finalize | 13.41% | DRAM Read **94.89%**（Write 6.55%） | 已接近 DRAM Read roof；优先减少读取量、改善 locality 或与前级融合。 |

这里没有把异构资源压成一个总分：Tensor Core、DRAM Read、DRAM Write 使用各自分母。read/write mix 在 40%–60% 时只显示 1:1 copy diagnostic reference；没有 ratio-matched calibration 时不能把它称为 ceiling。因此 read-heavy Finalize 使用 DRAM Read 达成率，而不是错误的 copy roof。

TC 主分母是 exp_026 对 exact `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` 指令的 full-card calibrated window。DRAM 百分比来自 NCU physical bytes / NCU duration 与 exp_026 directional roof，只用于定位接近哪个资源 ceiling，不等同于完整 mixed-op efficiency。

## 4. 优化优先级与最小下一步

1. **Finalize**：DRAM Read 已达 94.89%，优先减少读取量、改善 reuse/layout。
2. **FC2**：做 ratio-matched mixed-R/W standalone，确认 traffic 与 TC 哪个更值得先优化。
3. **FC1**：用最小 standalone 对照区分 TC schedule 与权重读取，不能仅凭当前表选择其中一个。
4. **Route/Q0/Pack + SwiGLU/Q1**：分别抽取 standalone，检查量化/ALU/SFU/irregular access 与 latency。

原始 throughput、公式输入、digest 与逐 op ceiling status 见 [model.json](model.json)。
