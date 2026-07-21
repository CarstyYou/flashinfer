# exp_022：Scatter 向量化 S2R

结论：**Accept，已进入 `moe_dynamic_kernel_opt.py`**。Live source SHA256 为
`db2ea34071b5dddd8ae6a366a600e412ffa7b872438b2e55f09b037e2764977a`，与全部验证使用的
Candidate 字节一致。

| M | Baseline | Candidate | Speedup |
|---:|---:|---:|---:|
| 256 | 514.176 µs | 514.208 µs | -0.006% |
| 8192 | 1336.992 µs | 1224.544 µs | **+9.183%** |

M8192 三组 paired speedup 为 `+9.215% / +9.221% / +9.035%`，最大 position CV 为
`0.278%`；M256 最大 position CV 为 `0.297%`。完整 ABBA samples 见
[benchmark_raw.json](raw/benchmark_raw.json)。

| M8192 PC-scoped 证据 | Baseline | Candidate |
|---|---:|---:|
| Scatter `sC` load | 32 × `LDS.U16` | 4 × `LDS.128` |
| Shared wavefront actual / ideal | 67,108,864 / 16,880,640 | 8,388,608 / 8,388,608 |
| Amplification | 3.9755× | **1.0000×** |
| REDG thread work | 67,108,864 | 67,108,864 |
| Registers / spill | 165 / 0 | 146 / 0 |

15 cases × 2 replay 全部通过，包括 M256/M8192 canonical、`valid_rows=0/1/31/32/33/63/64/65/95/96/97/127/128`、sentinel、ownership 与 contention 门禁；摘要见
[correctness_summary.json](raw/correctness_summary.json)。PC、SASS 与报告身份见
[ncu_pc_evidence.json](raw/ncu_pc_evidence.json)、[static_sass_gate.json](raw/static_sass_gate.json) 和
[manifest.json](manifest.json)。
