# Experiment 001 result: backend case sweep

**Verdict: TARGET NOT MET.** Customer criterion: CuteDSL FP4 must be at least 100% faster (2x throughput by latency ratio) than SGLang Triton FP8 for every M.

| M | CuteDSL FP4 (us) | CUTLASS BF16 chain (us) | SGLang Triton FP8 (us) | vs CUTLASS | vs SGLang | 2x target |
|---:|---:|---:|---:|---:|---:|:---:|
| 256 | 541.118 | 547.422 | 773.806 | +1.16% | +43.00% | no |
| 512 | 562.901 | 551.336 | 745.975 | -2.05% | +32.52% | no |
| 1024 | 626.057 | 582.214 | 765.606 | -7.00% | +22.29% | no |
| 2048 | 751.308 | 723.201 | 875.833 | -3.74% | +16.57% | no |
| 4096 | 1025.632 | 1005.594 | 1156.068 | -1.95% | +12.72% | no |
| 8192 | 1787.921 | 1676.457 | 2166.001 | -6.23% | +21.15% | no |

Speedup is `(baseline_time / CuteDSL_time - 1) * 100%`. Both columns use the single CuteDSL series from the fresh paired CUTLASS rerun.

CUTLASS is the matched BF16-input online-quantization chain. SGLang is the direct legacy Triton tensor-scaled W8A8 FP8 chain; its ratio is explicitly cross-runtime, not a fusion-only causal comparison.

Evidence rerun: `exp001-corrected-20260716T0443Z-r1`; GPU: `GPU-4a286357-c999-9547-3a04-25961b1ffd08`. All six per-arm correctness, dispatch, fixture, identity, and <=5% spread gates passed before publication.
