# Experiment 002 Benchmark Result

这里仅报告 correctness-qualified、未插桩 CUDA Graph benchmark；机制归因见 [`fusedop_dataflow_bottleneck.md`](fusedop_dataflow_bottleneck.md)。

| M | CuteDSL fused (us) | CUTLASS BF16 chain (us) | CuteDSL speedup vs matched | Gate | Stable |
|---:|---:|---:|---:|:---:|:---:|
| 256 | 538.956 | 550.739 | +2.19% | pass | yes |
| 1024 | 615.582 | 577.220 | -6.23% | pass | yes |
| 8192 | 1782.547 | 1665.779 | -6.55% | pass | yes |

Speedup 定义为 `(baseline_time / cutedsl_time - 1) × 100%`。
两条数据必须来自同一个 unique rerun，并匹配 shared environment、measurement protocol 与各自 artifact fingerprint。

Profiler duration 不替代本表；NCU 各 launch duration 也不会相加重建 operator time。
