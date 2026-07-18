# exp_004：整个 Fused Kernel 的 Phase 耗时占比

## 结论

本次已经覆盖完整 fused kernel，不再只统计 Gate/Up/SwiGLU/FC2：

```text
entry/prologue → Clear → Histogram → Prefix → Route+Q0+Pack → Publish
→ compute setup → [Claim/cache → Gate → Up → SwiGLU/Q1
→ FC2 setup → FC2 GEMM → FC2 epilogue/scatter] × task
→ task control/final drain → producer tail
```

主要结果：

- `FC2 epilogue + scatter` 最大，占 **32.43%**；
- Gate 前的 `Route + Q0 + Pack` 占 **19.10%**，是不可忽略的第二大阶段；
- `Gate` 与 `Up` 分别占 **12.87% / 12.68%**；
- `FC2 GEMM` 占 **10.20%**，`SwiGLU + Q1` 占 **6.17%**；
- `Clear + Histogram + Prefix + Publish` 合计约 **2.00%**；
- 每个 task 的 `Claim + cache/setup` 合计约 **1.75%**。

因此，Gate 前的 route/quant/pack 已经纳入结果，并且占比很高。高占比表示后续调查优先级，不能单凭本实验称为已证实 bottleneck。

## 完整可加占比

主分母是 `%globaltimer` 同一时间域下的 SM-equivalent wall：

```text
D = 110 × (max CTA final - min CTA entry)
```

`等效 wall` 是 `Σ CTA phase time / 110`，用于直观展示；它不是各 phase 对单次 kernel wall 的独立因果贡献。Gate 到 FC2 的边界由 **W0 lane0** 记录，表示 CTA 的代表性 consumer timeline，不是 W0–W3 四个 warp 的指令时间求和或精确 phase union。下表所有行互斥，合计严格为 100%。

| Phase | 占比 | 等效 wall | 5 次 replay 范围 |
|---|---:|---:|---:|
| Launch skew / early-finish idle | 1.83% | 33.33 µs | 1.79%–1.85% |
| Entry / prologue | 0.01% | 0.19 µs | 0.01%–0.01% |
| P0 Clear / init | 0.68% | 12.34 µs | 0.66%–0.68% |
| P1 Histogram | 0.39% | 7.16 µs | 0.39%–0.40% |
| P2 Prefix | 0.75% | 13.64 µs | 0.74%–0.75% |
| P3 Route + Q0 + Pack | **19.10%** | **348.80 µs** | 19.05%–19.16% |
| P4 Publish | 0.19% | 3.42 µs | 0.18%–0.19% |
| Compute setup | <0.01% | 0.01 µs | <0.01% |
| T0 Claim / control | 0.98% | 17.96 µs | 0.97%–1.02% |
| T0 Cache / task setup | 0.76% | 13.96 µs | 0.76%–0.77% |
| FC1 Gate | **12.87%** | **234.95 µs** | 12.85%–12.91% |
| FC1 Up | **12.68%** | **231.57 µs** | 12.66%–12.69% |
| SwiGLU + Q1 | **6.17%** | **112.61 µs** | 6.16%–6.18% |
| FC2 setup | 0.12% | 2.19 µs | 0.12%–0.12% |
| FC2 GEMM | **10.20%** | **186.34 µs** | 10.19%–10.22% |
| FC2 epilogue + scatter | **32.43%** | **592.11 µs** | 32.39%–32.47% |
| Task control / final drain / producer tail | 0.84% | 15.41 µs | 0.84%–0.85% |
| **合计** | **100.00%** | **1825.96 µs** | — |

P3 内 Route、Q0 与 Pack 按 routed pair 交错执行，不存在三个共同的全局 phase boundary，因此不伪造三个独立 wall-time 数字。

最后一行是合法 residual，包含 inter-task gap、最终 no-task claim/exit 和 W4 producer tail。Gate/Up/SwiGLU/FC2 interval 也包含边界内的同步等待，因此不能解释为对应数学或内存指令的纯执行时间。

## W4 Producer / TMA overlap

W4 与 W0–W3 并行，以下 interval 不进入上面的 100%：

| W4 interval | 等效 service interval |
|---|---:|
| Gate TMA producer | 194.98 µs |
| Gate pass wait | 0.29 µs |
| Up TMA producer | 200.82 µs |
| Down TMA producer | 799.84 µs |
| Final pass wait | 0.30 µs |

W4 interval union 中 **99.09%** 与 consumer phase 重叠。这里测到的是 producer 代码区间，包含 pipeline acquire/backpressure，不等于 TMA engine 独占 active time。

## 测量有效性与边界

| Gate | 结果 |
|---|---:|
| Correctness | 5/5 PASS；relative-L2 `0.013039–0.013043` |
| Task events | 每次 `164840/164840`，2536 task 全覆盖 |
| CTA events | 每次 `1540/1540`，110 CTA 全覆盖 |
| Denominator closure | 5/5 PASS；aggregate delta `0 ns` |
| Phase 稳定性 | 最大 run-to-run span `0.113` 个百分点 |
| CUDA Event / globaltimer（5-run mean） | `1834.09 / 1825.96 µs`，差 `0.45%` |
| No-marker / probe latency（median） | `1802.94 / 1834.40 µs`，扰动 `+1.74%` |

该结果仍是 **diagnostic estimate**：matched control 为 `REG=255 / STACK=488 B/thread / LDL=122 / STL=68`，probe 为 `REG=255 / STACK=464 B/thread / LDL=135 / STL=64`，SASS identity 发生变化。它适合回答完整 phase 分布和确定下一步调查顺序，不升级为 production-exact timing。

完整 5 次 replay 数据见 [whole_kernel_timing.json](whole_kernel_timing.json)，capture、toolchain、binary 与 source hash 见 [whole_kernel_capture_summary.json](derived/whole_kernel_capture_summary.json)，canonical closure 见 [manifest.json](manifest.json)。
