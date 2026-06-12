# Grouped MoE GEMM perf comparison -- `6kpro_full.csv`

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
| 1 | 229.4 | 256.0 (pad 16) (+11.6%) | 280.6 (pad 128) (+22.3%) | 504.1 (+119.8%) |
| 4 | 229.4 | 250.9 (pad 16) (+9.4%) | 282.6 (pad 128) (+23.2%) | 515.4 (+124.7%) |
| 8 | 229.4 | 254.0 (pad 16) (+10.7%) | 282.6 (pad 128) (+23.2%) | 507.1 (+121.1%) |
| 16 | 229.4 | 250.9 (+9.4%) | 283.6 (pad 128) (+23.7%) | 548.8 (+139.3%) |
| 192 | 264.2 | 278.5 (+5.4%) | 301.1 (pad 256) (+14.0%) | 632.6 (+139.4%) |
| 256 | 268.3 | 276.5 (+3.1%) | 303.1 (+13.0%) | 577.3 (+115.2%) |
| 1024 | 704.5 | 753.7 (+7.0%) | 761.9 (+8.1%) | 1508.7 (+114.1%) |
| 4096 | 2671.6 | 2669.6 (-0.1%) | 2741.3 (+2.6%) | 5048.2 (+89.0%) |


### fc1 (N=4096, K=7168) -- granK=128

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


### fc2 (N=7168, K=4096) -- granK=32

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


### fc2 (N=7168, K=4096) -- granK=128

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
