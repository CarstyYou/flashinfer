# exp_022：Current Opt FP4 vs Triton FP8 耗时分布

M8192 下，Current Opt 的 FC1/FC2 计算链明显更短；当前横向最突出的差距是 Q0，Scatter 的绝对耗时则已略低于 Triton 的输出归并。

## M8192 Phase / Op 耗时与占比

| 逻辑终点 | Current Opt FP4（time / own share） | SGLang Triton FP8（time / own share） | Opt - Triton | Speedup |
|---|---:|---:|---:|---:|
| Routing / scheduler | Clear + Histogram + Prefix：35.055 µs / 2.96% | Align + Count/sort：40.384 µs / 1.87% | -5.329 µs | +15.203% |
| Q0 / input pack ready | Route + Q0 + Pack + publish：109.520 µs / 9.26% | Fill + Absmax + Quant：49.056 µs / 2.27% | **+60.464 µs** | **-55.208%** |
| Fused-only control | Claim + cache + control：29.741 µs / 2.51% | — | — | — |
| FC1 + SwiGLU ready | 465.720 µs / 39.36% | 1130.877 µs / 52.44% | -665.157 µs | +142.823% |
| Q1 ready | 35.450 µs / 3.00% | 57.920 µs / 2.69% | -22.470 µs | +63.383% |
| FC2 tile ready | FC2 + epilogue + R2S：232.799 µs / 19.67% | FC2：650.142 µs / 30.15% | -417.343 µs | +179.272% |
| Output Y ready | Scatter：216.753 µs / 18.32% | TopK reduce / combine：225.696 µs / 10.47% | -8.943 µs | +4.126% |
| 不可对齐开销 | CTA residual + launch skew：58.255 µs / 4.92% | Graph bubble：2.336 µs / 0.11% | — | — |
| **分解行合计** | **1183.293 µs / 100%** | **2156.411 µs / 100%** | **-973.118 µs** | **+82.238%** |

每格为 5 次 replay 的行耗时中位数及本侧归一化占比：Opt 是 `%globaltimer` equivalent-wall 投影，Triton 是 NSys graph node elapsed；`Speedup = Triton time / Opt time - 1`，正数表示 Opt 更快。Current Opt 仅补采 matched control/probe，插桩扰动为 `-0.521%`；Triton 复用 exp_017 的已锁定 12-node CUDA Graph timeline。两侧按逻辑终点对齐，只用于耗时分布观察，不表示内部算法等价或因果归因。证据身份见 [manifest.json](manifest.json) 与 [exp_017 evidence](../../exp_017_opt_vs_triton_phase_share/results/evidence.json)。

## 横向观察

- FC1 + FC2 合计：Current Opt 为 `698.519 µs / 59.03%`，Triton 为 `1781.019 µs / 82.59%`，是总耗时差中最大的观测项。
- Q0：Current Opt 比 Triton 多 `60.464 µs`，占比 `9.26% vs 2.27%`，是最明确的相对短板。
- Scatter：Current Opt 绝对耗时比 Triton 少 `8.943 µs`，但自身占比更高；高占比来自整体执行区间缩短，不能解读为 Scatter 比 Triton 更慢。
