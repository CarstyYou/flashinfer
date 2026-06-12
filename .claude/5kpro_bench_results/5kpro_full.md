# Grouped MoE GEMM perf comparison -- `5kpro_full.csv`

All t_us = 50-rep median (l2-flush per iter). `pad N` = caller-padded per-expert M (kernel sees padded rows; TFLOPS uses logical m_pe).

## Backend / API mapping

| Backend | Library API | Notes |
|---|---|---|
| **cute** | `flashinfer.grouped_mm.moe_gemm_mxfp8_nt_groupwise` (backend=`"cute"`, scale_granularity_mnk=(1, 1, granK)) | PR #3562 cute SM120 ZeroPadding mode (kernel handles all padding internally) |
| **cudnn** | `flashinfer.grouped_mm.grouped_mm_mxfp8` (backend=`"cudnn"`) | cuDNN 9.23 grouped MoE GEMM, granK=32 only (industry MX 1x32 spec) |
| **dg** | `deep_gemm.m_grouped_fp8_gemm_nt_contiguous` (recipe=(1, 1, granK)) | DeepGEMM upstream leavelet/sm120 branch HEAD 76e93aa (NOT a flashinfer API; caller pads M to 128 per `get_theoretical_mk_alignment_for_contiguous_layout()`) |
| **cutlass** | custom `.cu` wrapper around CUTLASS example `87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu` (`ScaleGranularityN=1` -- 1D per-token scale matching cute) | flashinfer `group_gemm_fp8_nt_groupwise` has upstream guard rejecting num_groups>1 on SM120; bench uses custom `.cu` bypass; TileM=128 Cooperative, no SwapAB |


### fc1 (N=4096, K=7168) -- granK=32

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


### fc1 (N=4096, K=7168) -- granK=128

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


### fc2 (N=7168, K=4096) -- granK=32

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


### fc2 (N=7168, K=4096) -- granK=128

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
