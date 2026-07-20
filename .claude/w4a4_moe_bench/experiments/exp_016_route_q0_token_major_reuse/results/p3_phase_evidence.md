# exp_016 P3 Phase Evidence

**结论：可用的 diagnostic 证据。** 未插桩 E2E 仍是唯一性能判定依据。

| P3 grid critical wall | Baseline | Candidate | 差值 | 降幅 |
|---|---:|---:|---:|---:|
| Median | 241.344 µs | 100.768 µs | -140.576 µs | 58.25% |

| Arm | REG control→probe | STACK control→probe | SMEM control→probe | LOCAL control→probe | SASS SpillRefill/LDL/STL control→probe | Probe E2E 扰动 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline_pair_major | 165→165 | 0→0 B | 1024→1024 B | 0→0 B | 0/0/0→0/0/0 | -0.32% | 通过 |
| candidate_token_major_reuse | 165→165 | 0→0 B | 1024→1024 B | 0→0 B | 0/0/0→0/0/0 | -0.15% | 通过 |

口径：`max(all CTA end) - min(all CTA start)`；未使用 additive SM-equivalent estimate。
