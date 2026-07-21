# Experiment 001 result: backend case sweep

**Verdict: TARGET NOT MET.** Customer criterion: Latest opt CuteDSL FP4 must be at least 100% faster (2x throughput by latency ratio) than SGLang Triton FP8 for every M.

| M | Production CuteDSL FP4 (us) | Latest opt CuteDSL FP4 (us) | Opt vs Production* | CUTLASS BF16 chain (us) | SGLang Triton FP8 (us) | Opt vs CUTLASS | Opt vs SGLang | 2x target |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 256 | 541.118 | 515.667 | +4.94% | 552.696 | 774.032 | +7.18% | +50.10% | no |
| 512 | 562.901 | 527.430 | +6.73% | 545.881 | 747.169 | +3.50% | +41.66% | no |
| 1024 | 626.057 | 576.386 | +8.62% | 598.717 | 765.011 | +3.87% | +32.73% | no |
| 2048 | 751.308 | 622.972 | +20.60% | 728.206 | 879.615 | +16.89% | +41.20% | no |
| 4096 | 1025.632 | 812.124 | +26.29% | 1011.524 | 1170.760 | +24.55% | +44.16% | no |
| 8192 | 1787.921 | 1354.733 | +31.98% | 1701.766 | 2213.402 | +25.62% | +63.38% | no |

Speedup is `(baseline_time / latest_opt_time - 1) * 100%`. Opt vs CUTLASS and Opt vs SGLang use Latest opt as the denominator; all three current arms share the fresh rerun.

`Opt vs Production*` is cross-rerun context only: Production retains the original `exp001-corrected-20260716T0443Z-r1` evidence on `GPU-4a286357-c999-9547-3a04-25961b1ffd08`, while Latest opt uses `exp001-latest-opt-20260721T042841Z-r1` on `GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522`. It is not a paired performance claim.

CUTLASS is the matched BF16-input online-quantization chain. SGLang is the direct legacy Triton tensor-scaled W8A8 FP8 chain; its ratio is explicitly cross-runtime, not a fusion-only causal comparison.

CuteDSL source: `.claude/w4a4_moe_bench/moe_dynamic_kernel_opt.py`; SHA256 `ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19`; FlashInfer `996c3622cd3ce8603a0bd217545a9afe5516f6aa`.

Evidence rerun: `exp001-latest-opt-20260721T042841Z-r1`; GPU: `GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522`. All six per-arm correctness, dispatch, fixture, identity, and <=5% spread gates passed before publication.
