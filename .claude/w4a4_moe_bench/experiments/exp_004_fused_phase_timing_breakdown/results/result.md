# exp_004：Fused Phase Timing Breakdown

## 结论

Formal probe 仍因 binary resource/spill identity drift 判为失败；在用户接受“近似占比”的边界后，diagnostic follow-up 已成功得到稳定的 phase 分布：

- `FC2 epilogue + scatter` 最大，约 **41.09%**；
- `FC1 Gate + Up` 合计约 **34.05%**；
- `FC2 GEMM` 约 **13.44%**；
- 全部 GEMM phase 合计约 **47.49%**。

这些数是 W0–W3 MMA-consumer warp-task cycles 的占比，用于决定后续调查优先级；不是各 phase 对 production kernel wall time 的可加贡献。Gate 前的 Clear / Histogram / Prefix / Route-Q0-Pack / Publish，以及 W4 producer timeline，不在这个分母中。

## Diagnostic phase 占比

固定 case：`M=8192, E=256, H=2048, I_tp=512, topk=8`，5 次 CUDA Graph replay 加权汇总。

| Phase | 占比 | 5 次 replay 范围 |
|---|---:|---:|
| FC1 Gate | 17.11% | 17.10%–17.13% |
| FC1 Up | 16.93% | 16.91%–16.96% |
| SwiGLU + Q1 | 8.20% | 8.17%–8.22% |
| FC2 setup | 0.06% | 0.05%–0.06% |
| FC2 GEMM | 13.44% | 13.43%–13.45% |
| FC2 epilogue + scatter | 41.09% | 41.06%–41.12% |
| Residual / task control | 3.18% | 3.17%–3.18% |

FC2 整段约占 **54.58%**；其中 epilogue + scatter 占 FC2 tracked cycles 的 **75.27%**。因此下一步不能只盯 FC2 OMMA，FC2 后处理与 scatter 才是本次 breakdown 暴露出的最大时间区间。

## 测量有效性

| 检查 | 结果 |
|---|---:|
| Eager event gate | PASS，`776016/776016` writes |
| CUDA Graph replay event gate | 5/5 PASS；每次 `776016/776016` writes，0 error |
| Reference correctness | 5/5 PASS；relative-L2 `0.013040±0.000001` |
| No-marker latency median | `1798.98 us` |
| Diagnostic probe latency median | `1832.96 us` |
| Probe latency perturbation | `+33.98 us` / `+1.89%` |

原 formal probe 的 spill 并未消失，而是发生重排：`STACK 488→456 B/thread`、`STL 68→64`、`LDL 122→132`。成功的 diagnostic overlay 还加入了 inline stores 与 volatile task-slot reload，且没有保留该最终 cubin 的 resource audit，所以这里只发布 diagnostic estimate，不恢复 formal timing 资格。

## 0-write 定位与证据更正

消融证据确认：original、direct-store helper 和 inline-store 三种实现均为 0-write；clock canary 随后读到 `65790/65860`，超过合法 `task_capacity=3068`，对应 event index 已落到 buffer 之外。在保持 inline stores 不变、仅把 claimed-slot reload 改为 `ld.volatile.shared.s32` 后，eager 与 5 次 graph replay 全部 exact-fill。因此可以把直接根因收敛到 **probe indexing path 使用了越界 task-slot 值**。

`[inference]` 更底层的 compiler mechanism 是：原 shared-load helper 的 `has_side_effects=False` 允许了不合法的 load reuse/motion；当前证据没有定位到具体 compiler pass，不把这一层写成事实。

此前报告把唯一的 `st.global.u64` 误认成 probe store。PTX differential 显示：probe 相对 no-marker 新增 `36` 个 `st.global.b64` timing stores 与 `1` 个 `st.global.b32` CTA-map store；唯一 `st.global.u64` 两边都有，是原 kernel 的 FP4 pack store。

完整数值和 capture identity 见 [diagnostic_phase_summary.json](derived/diagnostic_phase_summary.json)；0-write 消融链见 [probe_zero_write_diagnosis.json](derived/probe_zero_write_diagnosis.json)。
