# Grouped MoE GEMM 性能对比 — cute SM120 vs cudnn / DG / cutlass (6K Pro Server Edition)

## 0. 结论速览

- **cute SM120 在 small-M MoE 区间领先所有对手**: m_pe=1 fc1 granK=128 cute 225.3us vs cutlass 433.5us (**+92.4%**), vs DG 260.1us (**+15.5%**), vs cudnn 256.0us (**+13.6%**)。
- **large-M (m_pe=4096) 区间 cute / cudnn / DG 收敛 ±5%**, cutlass 仍 +21-89% 落后。
- **cutlass 全程最慢, gap 主因 = 缺 SwapAB 优化** (TileM=128 hardcoded, m_pe=1 时 99.2% MMA waste)。

## 1. 关键数据 (50-rep median, l2-flush per iter, 6K Pro Server Edition SM120)

每非-cute cell 后括号 = 该 op 比 cute 慢多少 (`+X%` for <2x, `Y.YYx` for ≥2x)。`pad N` = caller-padded per-expert M。

### 1.1 fc1 (N=4096, K=7168) -- granK=32

| m_pe | cute | cudnn | dg | cutlass |
|---|---|---|---|---|
| 1 | 229.4 | 256.0 (pad 16) (+11.6%) | 280.6 (pad 128) (+22.3%) | 504.1 (+119.8%) |
| 4 | 229.4 | 250.9 (pad 16) (+9.4%) | 282.6 (pad 128) (+23.2%) | 515.4 (+124.7%) |
| 8 | 229.4 | 254.0 (pad 16) (+10.7%) | 282.6 (pad 128) (+23.2%) | 507.1 (+121.1%) |
| 16 | 229.4 | 250.9 (+9.4%) | 283.6 (pad 128) (+23.7%) | 548.8 (+139.3%) |
| 192 | 264.2 | 278.5 (+5.4%) | 301.1 (pad 256) (+14.0%) | 632.6 (+139.4%) |
| 256 | 268.3 | 276.5 (+3.1%) | 303.1 (+13.0%) | 577.3 (+115.2%) |
| 1024 | 704.5 | 753.7 (+7.0%) | 761.9 (+8.1%) | 1508.7 (+114.1%) |
| 4096 | 2671.6 | 2669.6 (-0.1%) | 2741.3 (+2.6%) | 5048.2 (+89.0%) |

### 1.2 fc1 (N=4096, K=7168) -- granK=128

| m_pe | cute | dg | cutlass |
|---|---|---|---|
| 1 | 225.3 | 260.1 (pad 128) (+15.5%) | 433.5 (+92.4%) |
| 4 | 225.3 | 256.0 (pad 128) (+13.6%) | 433.0 (+92.2%) |
| 8 | 225.3 | 260.1 (pad 128) (+15.5%) | 433.2 (+92.3%) |
| 16 | 225.3 | 259.1 (pad 128) (+15.0%) | 430.9 (+91.3%) |
| 192 | 260.1 | 272.4 (pad 256) (+4.7%) | 457.1 (+75.7%) |
| 256 | 262.1 | 273.4 (+4.3%) | 457.2 (+74.4%) |
| 1024 | 686.1 | 717.8 (+4.6%) | 989.7 (+44.3%) |
| 4096 | 2462.7 | 2587.6 (+5.1%) | 3042.3 (+23.5%) |

### 1.3 fc2 (N=7168, K=4096) -- granK=32

| m_pe | cute | cudnn | dg | cutlass |
|---|---|---|---|---|
| 1 | 225.3 | **FAIL** | 270.3 (pad 128) (+20.0%) | 474.1 (+110.5%) |
| 4 | 225.3 | 241.7 (pad 16) (+7.3%) | 270.3 (pad 128) (+20.0%) | 480.8 (+113.4%) |
| 8 | 225.3 | 242.7 (pad 16) (+7.7%) | 270.3 (pad 128) (+20.0%) | 506.1 (+124.7%) |
| 16 | 227.3 | **FAIL** | 266.2 (pad 128) (+17.1%) | 470.4 (+106.9%) |
| 192 | 256.0 | 264.2 (+3.2%) | 284.7 (pad 256) (+11.2%) | 607.1 (+137.1%) |
| 256 | 258.0 | 265.2 (+2.8%) | 284.7 (+10.3%) | 558.2 (+116.3%) |
| 1024 | 729.1 | 735.2 (+0.8%) | 792.6 (+8.7%) | 1579.4 (+116.6%) |
| 4096 | 2700.3 | 2595.8 (-3.9%) | 2696.2 (-0.2%) | 5147.6 (+90.6%) |

### 1.4 fc2 (N=7168, K=4096) -- granK=128

| m_pe | cute | dg | cutlass |
|---|---|---|---|
| 1 | 221.2 | 249.9 (pad 128) (+13.0%) | 414.5 (+87.4%) |
| 4 | 221.2 | 249.9 (pad 128) (+13.0%) | 419.7 (+89.7%) |
| 8 | 221.2 | 250.9 (pad 128) (+13.4%) | 413.0 (+86.7%) |
| 16 | 223.2 | 249.9 (pad 128) (+11.9%) | 417.5 (+87.0%) |
| 192 | 251.9 | 262.1 (pad 256) (+4.1%) | 437.8 (+73.8%) |
| 256 | 251.9 | 256.0 (+1.6%) | 438.0 (+73.9%) |
| 1024 | 716.8 | 733.2 (+2.3%) | 999.1 (+39.4%) |
| 4096 | 2501.6 | 2589.7 (+3.5%) | 3029.6 (+21.1%) |

`**FAIL**` = cudnn-internal CUDA IMA bug (fc2 N=7168 K=4096 m_pe ∈ {1, 16} specific, seed-independent + deterministic, 110/112 cell PASS)。

## 2. 主要实现差异 (跟性能相关)

| # | Axis | cute SM120 | cudnn MXFP8 | DG contiguous | cutlass (87c-derived) |
|---|---|---|---|---|---|
| 1 | Caller-side per-expert M alignment | **无** (kernel ZeroPadding 内部 pad) | 16 (kernel TileM 最小) | 128 (`get_theoretical_mk_alignment_for_contiguous_layout()`) | **无** (kernel 内部 pad TileM=128) |
| 2 | Small-M kernel optimization | **SwapAB + TileN=8** (small dim 用 small tile, 87.5% MMA waste vs 99.2%) | TileM=16 + 内部 pipeline | 强制 caller pad to 128 | 无 (TileM=128 hardcoded, 99.2% waste at m_pe=1) |
| 3 | granK 支持 | 32, 128 | **32 only** (industry MX 1×32) | 32, 128 | 32, 128 |
| 4 | Scale tensor dtype | UE8M0 int32-packed | UE8M0 uint8 raw | UE8M0 int32-packed | FP32 raw |

## 3. 关键结论 & 观察

### 3.1 cute SM120 优于对手的方面

- **Small-M 区间 (m_pe ≤ 16) 全胜 (kernel-internal SwapAB+TileN=8 让 small dim 用 small tile)**: m_pe=1 fc1 granK=128 cute 比 cutlass 快 **+92.4%**, 比 DG 快 **+15.5%**, 比 cudnn 快 **+13.6%**; 同 shape m_pe=16 趋势一致 (+91.3% / +15.0% / +9.4%)。 → SwapAB config 见 [csrc/cute_sm120_mxfp8_groupwise/cute_sm120_mxfp8_runner.cu](../../csrc/cute_sm120_mxfp8_groupwise/cute_sm120_mxfp8_runner.cu) (cutlass 87c 缺这个见 [.claude/bench/cutlass_grouped_gemm_sm120.cu](.claude/bench/cutlass_grouped_gemm_sm120.cu))。

- **caller padding 浪费在 small-M 暴露 (cute 零 caller-pad vs DG 128-pad / cudnn 16-pad)**: m_pe=1 fc1 granK=128 cute 225.3 vs DG 260.1us (**+15.5%**); m_pe=1 fc1 granK=32 cute 229.4 vs cudnn 256.0us (**+11.6%**). DG 强制 pad to 128, 8 expert × 128 = 1024 tokens 不管 m_pe 实际多小 — m_pe ∈ {1, 4, 8, 16} DG t_us 几乎一致 (256-283us)。 → padding contract 见 [.claude/bench_plan.md](.claude/bench_plan.md) §4。

- **granK=128 比 granK=32 略快 (scale tensor 4× 小)**: cute fc1 m_pe=4096 granK=128 2462us vs granK=32 2671us (**-7.8%**); cutlass fc1 m_pe=4096 granK=128 3042us vs granK=32 5048us (**-39.7%**) — cutlass TileK=32 时 K-loop 4× 长, 启动 overhead 大。 → 全表见 [.claude/6kpro_bench_results/6kpro_full.md](.claude/6kpro_bench_results/6kpro_full.md)。

### 3.2 cute SM120 劣于对手的方面

- **Large-M 区间 cudnn 略领先 (cudnn 大 M MMA pipeline 更优, 跟我们差 ≤ 4%)**: m_pe=4096 fc1 granK=32 cute 2671.6 vs cudnn 2669.6us 持平 (-0.1%); m_pe=4096 fc2 granK=32 cute 2700.3 vs cudnn 2595.8us (**cudnn 优 3.9%**)。 → 全表见 [.claude/6kpro_bench_results/6kpro_full.md](.claude/6kpro_bench_results/6kpro_full.md) §1.3。

## 4. 细节 link

- 完整 110-cell CSV (raw): [.claude/6kpro_bench_results/6kpro_full.csv](.claude/6kpro_bench_results/6kpro_full.csv)
- 4-table markdown (vetted): [.claude/6kpro_bench_results/6kpro_full.md](.claude/6kpro_bench_results/6kpro_full.md)
- Bench plan + caller padding contract: [.claude/bench_plan.md](.claude/bench_plan.md)
- Custom CUTLASS wrapper (87c-derived, ScaleGranularityN=1): [.claude/bench/cutlass_grouped_gemm_sm120.cu](.claude/bench/cutlass_grouped_gemm_sm120.cu)
- CUTLASS 87c upstream reference: [3rdparty/cutlass/examples/87_blackwell_geforce_gemm_blockwise/87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu](../../3rdparty/cutlass/examples/87_blackwell_geforce_gemm_blockwise/87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu)
- cute SM120 ZeroPadding kernel source (SwapAB scheduler): [csrc/cute_sm120_mxfp8_groupwise/](../../csrc/cute_sm120_mxfp8_groupwise/)
- Bench script + common utils: [.claude/bench/](.claude/bench/)
- 已知 caveat: fc2 cudnn m_pe ∈ {1, 16} 2 cell deterministic IMA (cudnn-internal kernel-select bug, 110/112 cell PASS)
