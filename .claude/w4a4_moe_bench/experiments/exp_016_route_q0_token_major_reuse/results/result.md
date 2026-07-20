# exp_016：Route/Q0 Token-major 输入复用

> **整体判定：Accept** — Validation/E2E、P3 phase 与 dynamic spill 门禁均通过。

> **Validation + E2E scoped 判定：Accept** — 正确性、身份与完整三组 ABBA sweep 均通过。

## 正确性与身份

| 项目 | 状态 |
|---|---:|
| Source / cubin / GPU / toolchain identity | pass |
| Paired FP4 / SFA / metadata digest | pass |
| P3 phase gate | pass |
| P3 capture / instrumentation / SASS spill / SMEM / phase | pass / pass / pass / pass / pass |
| Dynamic spill gate | pass |

每个 paired case 均按逻辑 route 顺序比较；physical row 顺序差异不会造成误判。

## 未插桩 E2E ABBA

| M | Baseline median (us) | Candidate median (us) | Median improvement | Gate |
|---:|---:|---:|---:|---:|
| 256 | 519.792 | 514.128 | +1.08% | pass |
| 512 | 535.648 | 526.624 | +1.70% | pass |
| 1024 | 578.944 | 572.080 | +1.20% | pass |
| 2048 | 642.608 | 618.224 | +3.95% | pass |
| 4096 | 867.616 | 813.840 | +6.61% | pass |
| 8192 | 1463.296 | 1333.728 | +9.71% | pass |

### 每组 ABBA

| M | Group | Baseline median (us) | Candidate median (us) | Improvement | CV (A / B) | Gate |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0 | 519.792 | 514.320 | +1.06% | 0.30% / 0.18% | pass |
| 256 | 1 | 519.808 | 514.048 | +1.12% | 0.31% / 0.20% | pass |
| 256 | 2 | 519.680 | 514.128 | +1.08% | 0.21% / 0.27% | pass |
| 512 | 0 | 535.648 | 526.624 | +1.71% | 0.51% / 0.22% | pass |
| 512 | 1 | 534.960 | 526.512 | +1.60% | 0.42% / 0.25% | pass |
| 512 | 2 | 535.696 | 526.720 | +1.70% | 0.49% / 0.50% | pass |
| 1024 | 0 | 578.896 | 572.192 | +1.17% | 0.23% / 0.54% | pass |
| 1024 | 1 | 578.944 | 572.080 | +1.20% | 0.36% / 0.51% | pass |
| 1024 | 2 | 579.296 | 572.000 | +1.28% | 0.64% / 0.34% | pass |
| 2048 | 0 | 642.592 | 618.416 | +3.91% | 0.32% / 0.46% | pass |
| 2048 | 1 | 642.608 | 618.192 | +3.95% | 0.28% / 0.39% | pass |
| 2048 | 2 | 642.832 | 618.224 | +3.98% | 0.32% / 0.51% | pass |
| 4096 | 0 | 867.744 | 813.808 | +6.63% | 0.20% / 0.28% | pass |
| 4096 | 1 | 867.600 | 813.840 | +6.61% | 0.24% / 0.41% | pass |
| 4096 | 2 | 867.616 | 813.968 | +6.59% | 0.29% / 0.46% | pass |
| 8192 | 0 | 1463.968 | 1333.728 | +9.77% | 0.26% / 0.22% | pass |
| 8192 | 1 | 1463.296 | 1333.792 | +9.71% | 0.21% / 0.24% | pass |
| 8192 | 2 | 1461.536 | 1333.632 | +9.59% | 0.18% / 0.23% | pass |

Improvement = `(baseline median / candidate median - 1) × 100%`。每组把两个 A position 与两个 B position 各自合并为 100 个 replay 后计算 median/CV。

## Route/Q0 机制证据

Candidate 是整体 token-major P3 重构：同一 token 的 BF16 input load 与 block absmax 只做一次，再按 8 个 expert 分别 quant/store；M8192 BF16 block load 从 8,388,608 降至 1,048,576，productive producer claim 从 3,641 降至 911，route metadata ownership/shuffle 也随之改变。row-allocation atomic 与 FP4/SFA store 数量不变，因此本实验接受的是组合机制，不把收益拆给单一子变化。

| P3 grid critical wall | Baseline | Candidate | 降低 |
|---|---:|---:|---:|
| M8192 matched probe | 241.344 µs | 100.768 µs | 58.25% |

| Arm | REG control→probe | SMEM | STACK | LOCAL | SASS zero-spill | Probe E2E 扰动 |
|---|---:|---:|---:|---:|---:|---:|
| baseline_pair_major | 165→165 | 1024→1024 B | 0→0 B | 0→0 B | pass | -0.32% |
| candidate_token_major_reuse | 165→165 | 1024→1024 B | 0→0 B | 0→0 B | pass | -0.15% |

P3 口径为 `max(all CTA end) - min(all CTA start)`；它只用于定位收益，不替代未插桩 E2E。

P3 provenance: [p3_phase_evidence.json](p3_phase_evidence.json), SHA256 `5711c42651fb2690aade9f94a306ee11ef83a9945db357b253928cb6f6b8f4ca`。SASS sidecar 对 control/probe 四个 cubin 均验证为 zero-spill；SMEM gate 验证每个 arm 的 control/probe 静态 SMEM 相同。

## Spill

Candidate M8192 的 executed register spill/refill instruction 与 local load/store byte counters 全为 0（4 项动态指标）。

Dynamic spill provenance: [dynamic_spill_evidence.json](dynamic_spill_evidence.json), SHA256 `6583e1951f10f60affcf54f4307bf1e92952f1063f92a49cbde3ea73689a009e`。

> 性能判定以未插桩 E2E 为准；P3 matched probe 与 dynamic spill 作为独立、identity-locked 的支持证据。
