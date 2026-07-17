# exp_004：Fused Stack/Spill Criticality

## 结论

- **H14 attribution：inconclusive；H14 criticality：not tested。** Primary 与唯一 fallback 都仍为 `488 B/thread、122 words/lane`，没有改变目标 14-word bundle，因此按预注册 stop condition 不跑 paired benchmark。
- `up_first_attribution` 将 stack 从 `488 B` 降到 `432 B`，local bundle 净减少 14 words；NCU 每方向精确减少 `568,064` local sectors 和 `142,016` executed local instructions，selected non-local projection 与 measured Tensor work 保持不变。这个结果使下一步应优先调查 **FC1 pass order / accumulator lifetime 的 codegen**；现有证据不能排除 activation destination 参与。
- **H108 attribution 仍是 source + program-order inference，未达到 formal acceptance。** 108-word main bundle 在交换前后都保留，但 compiler artifacts 没有提供从 MLIR SSA 到 ptxas physical spill registers 的跨层映射。H108 criticality 仍为 unresolved / out of scope。
- 所有 arm 均通过独立 quant-aware reference；但 baseline replay self-drift 的 relative-L2 为 `0.006521`，超过预注册 hard cap `0.002`，所以 strict cross-arm correctness gate 无效，不能事后放宽。

## 1. Static 与正确性资格门

| Arm | Stack B/thread | Main words | Tail words | Quant-aware oracle | Strict cross-arm gate |
|---|---:|---:|---:|---:|---:|
| baseline | 488 | 108 | 14 | True | baseline anchor |
| activation_in_place_up | 488 | 108 | 14 | True | False |
| activation_in_place_gate | 488 | 108 | 14 | True | False |
| up_first_attribution | 432 | 108 | 0 | True | False |

两个 in-place arm 的 selected non-local opcode projection 均与 baseline 一致，但目标 14-word scalar-STL bundle 在相同 PC/stack offsets 原位保留、没有净减少，因此没有测试到 H14 criticality。

## 2. Attribution-only 的动态闭合

| Metric | Baseline | Up-first | Delta |
|---|---:|---:|---:|
| Local load sectors | 4,950,272 | 4,382,208 | -568,064 |
| Local store sectors | 4,950,272 | 4,382,208 | -568,064 |
| Executed local load instructions | 1,237,568 | 1,095,552 | -142,016 |
| Executed local store instructions | 689,792 | 547,776 | -142,016 |
| Tensor instructions | 31,162,368 | 31,162,368 | 0 |
| FP4 Tensor ops | 510,564,237,312 | 510,564,237,312 | 0 |

Local sectors 是 local-address-space footprint，不是 DRAM bytes；NCU duration 也不替代未插桩 benchmark。

## 3. 收口与下一步

- 本轮停止，不继续搜索额外 codegen variants，也不报告 speedup。
- 下一轮若继续，应预注册一个保持 FP32 数学与 work identity、但直接缩短 first accumulator 跨 second GEMM lifetime 的 clean arm；不应只围绕 activation destination 搜索。
- 若要 formal 接受 H108 attribution，需要 compiler liveness/register-allocation 映射，或能把 semantic accumulator SSA 与 physical STL/LDL chain 闭合的证据工具。
