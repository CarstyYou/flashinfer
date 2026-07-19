# exp_006：FC2 GEMM 与 Atomic Scatter 边界验证

本实验只验证一个问题：原报告中“atomic scatter 占 Fused whole-wall 超过 30%”是否成立。
按新边界，epilogue、scale/cast、R2S 和 pre-scatter sync 全部归入 `FC2 GEMM`；scatter loop
与 post-scatter sync 归入 `Atomic scatter`。

| FC2 阶段 | Marker 边界 | 等效 wall | FC2 additive 占比 | SM-equivalent denominator 占比 |
|---|---|---:|---:|---:|
| FC2 GEMM | A→D | **273.653 us** | **34.744%** | **14.905%** |
| Atomic scatter | D→F | **513.971 us** | **65.256%** | **27.995%** |
| **两阶段合计** | — | **787.624 us** | **100.000%** | **42.900%** |

Inter-tile residual 为 9.709 us（SM-equivalent denominator 的 0.529%），只用于闭合；加上它后
`A→F` FC2 tile-sweep envelope 为 797.332 us（43.429%）。

结论：原“超过 30%”的数字包含了 FC2 epilogue/R2S，不能称为 Atomic scatter。重新划分后，
Atomic scatter 阶段占该 denominator 的 **27.995%**；其中不含 post-sync 的 `D→E` scatter body
为 506.585 us（**27.593%**）。因此“scatter 阶段确实很重”成立，但“atomic 指令本身超过
30%”不成立；`D→F` 还包含 SMEM load、route-weight multiply、地址/循环控制、warp 工作不均衡与
post-scatter sync，不是纯 REDG 指令 latency。

结果等级为 `diagnostic estimate`。Matched no-marker control 与 probe median 分别为 1806.336 us 和
1844.864 us，插桩扰动为 `+2.133%`。占比分母是 5 个 replay、每个 110 CTA 的 `%globaltimer`
SM-equivalent denominator（1,009,768,320 ns），不是上述 CUDA-event median。表中时间是跨 CTA
累积后折算的 additive 时间，不是单个 CTA 的串行 wall latency。Canonical evidence 见 [evidence.json](evidence.json)，数据审计见
[data_audit.json](data_audit.json)。
