# exp_018：Triton FP8 vs Latest Opt FP4 vs Eric Stage4 FP4

Eric 六个 prefill case correctness：**全部通过**。

| M | Triton FP8 | Latest Opt FP4 | Eric Stage4 FP4 | Opt vs Triton | Eric vs Triton | Eric vs Opt |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 768.593 µs | 513.958 µs | 512.454 µs | +49.54%；未达 2× | +49.98%；未达 2× | +0.29%（相当/无定论） |
| 512 | 745.142 µs | 523.503 µs | 513.505 µs | +42.34%；未达 2× | +45.11%；未达 2× | +1.95%（阈值边界/无定论） |
| 1024 | 760.301 µs | 571.110 µs | 549.762 µs | +33.13%；未达 2× | +38.30%；未达 2× | +3.88%（更快） |
| 2048 | 876.565 µs | 617.022 µs | 646.644 µs | +42.06%；未达 2× | +35.56%；未达 2× | -4.58%（更慢） |
| 4096 | 1154.598 µs | 808.790 µs | 956.845 µs | +42.76%；未达 2× | +20.67%；未达 2× | -15.47%（更慢） |
| 8192 | 2150.661 µs | 1337.271 µs | 1689.611 µs | +60.82%；未达 2× | +27.29%；未达 2× | -20.85%（更慢） |

`Speedup = baseline_latency / subject_latency - 1`。每格为三个 cyclic process block 的 median；
每个 block 使用 warmup=5、timed=50、192 MiB L2 flush 和 CUDA Graph external-event timing。
Triton FP8 直接调用 SGLang legacy `fused_experts_impl`；本机没有 shape-specific config，
六个 case 的实际 config source 均为 `default_heuristic`。
M512 的 paired-ratio median（+2.006%）与 ratio-of-medians（+1.947%）分居 2% 阈值两侧，
审计后按“阈值边界/无定论”处理。

原始数据见 [benchmark_raw.csv](benchmark_raw.csv)，聚合数据见 [benchmark_summary.csv](benchmark_summary.csv)。
