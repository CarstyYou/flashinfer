# exp_015：Phase Skeleton 重构结果

## 结论

**接受当前重构。** `FC2 + Scatter` 已拆成两个独立 phase helper，完整 kernel 的
correctness、工作量、zero-spill 和资源均保持，性能通过预注册 no-regression gate。

```text
fc2_to_sC()                 # wait + B load + OMMA + down-alpha + R2S
→ fence + CTA sync            # sC 可见，依赖显式化
→ scatter_sC_to_gmem()        # 只读 sC，写 GMEM
→ post-scatter CTA sync
```

`down_acc` 只存活于 `fc2_to_sC()` 内；Scatter 不再持有 FC2 accumulator 或 epilogue 逻辑。
本次没有增加同步，只将原有 phase 依赖留在 caller 中显式展示。

## 性能

| Case | exp_008 baseline | 重构后 | Speedup | 95% CI | 判定 |
|---:|---:|---:|---:|---:|---|
| M256 | 518.958 µs | 519.033 µs | -0.0145% | [-0.0527%, +0.0267%] | Pass |
| M8192 | 1567.245 µs | 1568.049 µs | -0.0513% | [-0.0648%, -0.0359%] | Pass |

Speedup = `(baseline / candidate - 1) × 100%`；预注册 no-regression 边界为 `-1.5%`。

## 验证

| 检查 | 结果 |
|---|---|
| Correctness / route identity | 8 / 8 cases 通过 |
| Static resource | 两臂均为 160 registers/thread；STACK/LOCAL/LDL/STL = 0 |
| Helper inline | 两臂 CALL/RET = 0 |
| Tensor work | 两臂 OMMA = 448；matched NCU Tensor/FP4 work 完全一致 |
| Dynamic spill | 两臂 spill read/write 四项均为 0 |
| Launch / SMEM | 两臂 grid/block = `(1,1,110)/(288,1,1)`；total/dynamic SMEM = 84,992/83,968 B |
| Performance protocol | M256/M8192 各 5 组 A-B-B-A，共 40 positions |

## 适用边界

- 结论只适用于当前 full-N128 epilogue，即 `epi_rest_m == 1`。
- 如果恢复 compact-M64 / `epi_rest_m > 1`，必须改为逐 epi 的 `R2S → sync → Scatter`，
  不能直接复用当前边界。
- 本实验证明的是“结构拆分低于预注册回归边界”，不是 FC2 或 Scatter 单独的性能改善。

机器证据见 [evidence.json](evidence.json)、[static_resource_evidence.json](static_resource_evidence.json)
和 [matched_dynamic_ncu_evidence.json](matched_dynamic_ncu_evidence.json)。
