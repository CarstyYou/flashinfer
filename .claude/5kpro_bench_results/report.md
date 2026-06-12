# Grouped MoE GEMM 性能对比 — cute SM120 vs cudnn / DG / cutlass (5K Pro Workstation Edition)

## 0. 结论速览

- **cute SM120 在 small-M MoE 区间领先所有对手, 优势比 6K Pro 更大**: m_pe=1 fc1 granK=128 cute 210.9us vs cutlass 524.4us (**+148.6%**), vs DG 290.8us (**+37.9%**), vs cudnn 268.3us (granK=32, **+24.8%**)。
- **large-M (m_pe=4096) 区间 cute / cudnn / DG 收敛 ±2%**, cutlass 仍 +20-85% 落后。
- **cute 优势在 5K Pro 比 6K Pro 更显著** (5K Pro SM 数少 → small-M 区间 kernel-internal overhead 占比更高, SwapAB 收益放大)。

## 1. 关键数据 (50-rep median, l2-flush per iter, 5K Pro Workstation Edition SM120)

每非-cute cell 后括号 = 该 op 比 cute 慢多少 (`+X%` 为对手慢, `-X%` 为对手快)。`pad N` = caller-padded per-expert M。全 112/112 cell PASS (无 cudnn FAIL, 6K Pro 的 2 fc2 cudnn fail 在 5K Pro 不复现 — kernel-select 跟硬件相关)。

### 1.1 fc1 (N=4096, K=7168) -- granK=32

| m_pe | cute | cudnn | dg | cutlass |
|---|---|---|---|---|
| 1 | 215.0 | 268.3 (pad 16) (+24.8%) | 321.5 (pad 128) (+49.5%) | 605.2 (+181.4%) |
| 4 | 217.1 | 266.2 (pad 16) (+22.6%) | 321.5 (pad 128) (+48.1%) | 631.2 (+190.7%) |
| 8 | 221.2 | 266.2 (pad 16) (+20.4%) | 319.5 (pad 128) (+44.4%) | 634.9 (+187.0%) |
| 16 | 223.2 | 264.2 (+18.3%) | 323.6 (pad 128) (+45.0%) | 628.7 (+181.6%) |
| 192 | 344.1 | 348.2 (+1.2%) | 385.0 (pad 256) (+11.9%) | 829.0 (+140.9%) |
| 256 | 348.2 | 354.3 (+1.8%) | 387.1 (+11.2%) | 799.6 (+129.7%) |
| 1024 | 1052.7 | 1046.5 (-0.6%) | 1196.0 (+13.6%) | 2127.3 (+102.1%) |
| 4096 | 4044.8 | 4005.9 (-1.0%) | 4483.1 (+10.8%) | 7490.1 (+85.2%) |

### 1.2 fc1 (N=4096, K=7168) -- granK=128

| m_pe | cute | dg | cutlass |
|---|---|---|---|
| 1 | 210.9 | 290.8 (pad 128) (+37.9%) | 524.4 (+148.6%) |
| 4 | 213.0 | 288.8 (pad 128) (+35.6%) | 532.2 (+149.8%) |
| 8 | 215.0 | 290.8 (pad 128) (+35.2%) | 520.6 (+142.1%) |
| 16 | 219.1 | 290.8 (pad 128) (+32.7%) | 525.5 (+139.8%) |
| 192 | 337.9 | 340.0 (pad 256) (+0.6%) | 650.2 (+92.4%) |
| 256 | 342.0 | 342.0 | 635.4 (+85.8%) |
| 1024 | 1024.0 | 1058.8 (+3.4%) | 1427.6 (+39.4%) |
| 4096 | 3937.3 | 4022.3 (+2.2%) | 4743.6 (+20.5%) |

### 1.3 fc2 (N=7168, K=4096) -- granK=32

| m_pe | cute | cudnn | dg | cutlass |
|---|---|---|---|---|
| 1 | 215.0 | 258.0 (pad 16) (+20.0%) | 299.0 (pad 128) (+39.0%) | 593.4 (+175.9%) |
| 4 | 221.2 | 256.0 (pad 16) (+15.7%) | 299.0 (pad 128) (+35.2%) | 627.6 (+183.7%) |
| 8 | 225.3 | 256.0 (pad 16) (+13.6%) | 301.1 (pad 128) (+33.6%) | 736.2 (+226.8%) |
| 16 | 225.3 | 258.0 (+14.5%) | 301.1 (pad 128) (+33.6%) | 597.2 (+165.1%) |
| 192 | 335.9 | 338.0 (+0.6%) | 370.7 (pad 256) (+10.4%) | 1406.6 (+318.8%) |
| 256 | 325.7 | 342.0 (+5.0%) | 372.7 (+14.5%) | 818.5 (+151.3%) |
| 1024 | 1077.3 | 1071.1 (-0.6%) | 1146.9 (+6.5%) | 2300.6 (+113.6%) |
| 4096 | 4100.1 | 4024.3 (-1.8%) | 4331.5 (+5.6%) | 7596.0 (+85.3%) |

### 1.4 fc2 (N=7168, K=4096) -- granK=128

| m_pe | cute | dg | cutlass |
|---|---|---|---|
| 1 | 210.9 | 276.5 (pad 128) (+31.1%) | 552.8 (+162.0%) |
| 4 | 217.1 | 276.5 (pad 128) (+27.4%) | 558.2 (+157.1%) |
| 8 | 219.1 | 276.5 (pad 128) (+26.2%) | 548.2 (+150.1%) |
| 16 | 223.2 | 276.5 (pad 128) (+23.9%) | 544.4 (+143.9%) |
| 192 | 329.7 | 333.8 (pad 256) (+1.2%) | 626.7 (+90.1%) |
| 256 | 321.6 | 337.9 (+5.1%) | 613.1 (+90.6%) |
| 1024 | 1054.7 | 1062.9 (+0.8%) | 1451.6 (+37.6%) |
| 4096 | 4005.9 | 4051.0 (+1.1%) | 4802.4 (+19.9%) |

## 2. 主要实现差异 (跟性能相关)

| # | Axis | cute SM120 | cudnn MXFP8 | DG contiguous | cutlass (87c-derived) |
|---|---|---|---|---|---|
| 1 | Caller-side per-expert M alignment | **无** (kernel ZeroPadding 内部 pad) | 16 (kernel TileM 最小) | 128 (`get_theoretical_mk_alignment_for_contiguous_layout()`) | **无** (kernel 内部 pad TileM=128) |
| 2 | Small-M kernel optimization | **SwapAB + TileN=8** (small dim 用 small tile, 87.5% MMA waste vs 99.2%) | TileM=16 + 内部 pipeline | 强制 caller pad to 128 | 无 (TileM=128 hardcoded, 99.2% waste at m_pe=1) |
| 3 | granK 支持 | 32, 128 | **32 only** (industry MX 1×32) | 32, 128 | 32, 128 |
| 4 | Scale tensor dtype | UE8M0 int32-packed | UE8M0 uint8 raw | UE8M0 int32-packed | FP32 raw |

## 3. 关键结论 & 观察

### 3.1 cute SM120 优于对手的方面

- **Small-M 区间 (m_pe ≤ 16) 全胜 (kernel-internal SwapAB+TileN=8 让 small dim 用 small tile)**: m_pe=1 fc1 granK=128 cute 比 cutlass 快 **+148.6%**, 比 DG 快 **+37.9%**, 比 cudnn 快 **+24.8%** (granK=32); 同 shape m_pe=16 趋势一致 (+139.8% / +32.7% / +18.3%)。 → SwapAB config 见 [csrc/cute_sm120_mxfp8_groupwise/cute_sm120_mxfp8_runner.cu](../../csrc/cute_sm120_mxfp8_groupwise/cute_sm120_mxfp8_runner.cu) (cutlass 87c 缺这个见 [.claude/bench/cutlass_grouped_gemm_sm120.cu](.claude/bench/cutlass_grouped_gemm_sm120.cu))。

- **caller padding 浪费在 small-M 暴露**: m_pe=1 fc1 granK=128 cute 210.9 vs DG 290.8us (**+37.9%**), m_pe=16 同 shape +32.7%。DG 强制 pad to 128, 8 expert × 128 = 1024 tokens 不管 m_pe 实际多小 — m_pe ∈ {1, 4, 8, 16} DG t_us 几乎一致 (288-291us)。 → padding contract 见 [.claude/bench_plan.md](.claude/bench_plan.md) §4。

- **5K Pro 比 6K Pro cute 优势更大 (small-M SM-bound overhead 更显著)**: 5K Pro 物理 SM 数少于 6K Pro Server Edition, small-M 区间 kernel launch + pipeline overhead 占比更高, SwapAB+TileN=8 的省 8× MMA waste 收益放大。 → 跟 6K Pro 数据对比见 [../6kpro_bench_results/report.md](../6kpro_bench_results/report.md)。

- **granK=128 比 granK=32 略快 (scale tensor 4× 小)**: cute fc1 m_pe=4096 granK=128 3937us vs granK=32 4045us (**-2.7%**); cutlass fc1 m_pe=4096 granK=128 4744us vs granK=32 7490us (**-36.7%**) — cutlass TileK=32 时 K-loop 4× 长, 启动 overhead 大。 → 全表见 [.claude/5kpro_bench_results/5kpro_full.md](.claude/5kpro_bench_results/5kpro_full.md)。

### 3.2 cute SM120 劣于对手的方面

- **Large-M 区间 cudnn 略领先, gap ≤ 2%**: m_pe=1024 fc1 granK=32 cute 1052.7 vs cudnn 1046.5us (**cudnn 优 0.6%**); m_pe=4096 fc1 granK=32 cute 4044.8 vs cudnn 4005.9us (**cudnn 优 1.0%**); m_pe=4096 fc2 granK=32 cute 4100.1 vs cudnn 4024.3us (**cudnn 优 1.8%**)。 → 全表见 [.claude/5kpro_bench_results/5kpro_full.md](.claude/5kpro_bench_results/5kpro_full.md)。

## 4. 细节 link

- 完整 112-cell CSV (raw, 全 PASS): [.claude/5kpro_bench_results/5kpro_full.csv](.claude/5kpro_bench_results/5kpro_full.csv)
- 4-table markdown (vetted): [.claude/5kpro_bench_results/5kpro_full.md](.claude/5kpro_bench_results/5kpro_full.md)
- 6K Pro Phase A 报告: [.claude/6kpro_bench_results/report.md](.claude/6kpro_bench_results/report.md)
- Bench plan + caller padding contract: [.claude/bench_plan.md](.claude/bench_plan.md)
- Custom CUTLASS wrapper (87c-derived, ScaleGranularityN=1): [.claude/bench/cutlass_grouped_gemm_sm120.cu](.claude/bench/cutlass_grouped_gemm_sm120.cu)
- CUTLASS 87c upstream reference: [3rdparty/cutlass/examples/87_blackwell_geforce_gemm_blockwise/87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu](../../3rdparty/cutlass/examples/87_blackwell_geforce_gemm_blockwise/87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu)
- cute SM120 ZeroPadding kernel source (SwapAB scheduler): [csrc/cute_sm120_mxfp8_groupwise/](../../csrc/cute_sm120_mxfp8_groupwise/)
- Bench script + common utils: [.claude/bench/](.claude/bench/)
