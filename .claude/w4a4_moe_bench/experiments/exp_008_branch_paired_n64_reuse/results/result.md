# exp_008：双 N64 Gate/Up 配对复用结果

## 结论

**接受当前版本。** 在同一块 5KP、同一锁频环境的直接 ABBA 对比中，当前版本相对 Production：

- M256 快 **4.34%**；
- M8192 快 **15.06%**。

当前版本同时保持 8/8 correctness、static zero-spill 和 dynamic zero-spill。

| 直接同机对比 | Case | Baseline | 当前版本 | Speedup | 95% CI |
|---|---:|---:|---:|---:|---:|
| Production → 当前 | M256 | 541.625 µs | **519.074 µs** | **+4.34%** | [+4.33%, +4.36%] |
| Production → 当前 | M8192 | 1804.155 µs | **1567.950 µs** | **+15.06%** | [+15.03%, +15.10%] |
| exp_005 N128 8-warp → 当前 | M256 | 533.983 µs | **519.056 µs** | **+2.88%** | [+2.86%, +2.88%] |
| exp_005 N128 8-warp → 当前 | M8192 | 1642.558 µs | **1567.696 µs** | **+4.78%** | [+4.74%, +4.81%] |
| exp_007 双 N64及时消费 → 当前 | M256 | 571.745 µs | **519.235 µs** | **+10.11%** | [+10.09%, +10.13%] |
| exp_007 双 N64及时消费 → 当前 | M8192 | 1807.423 µs | **1567.819 µs** | **+15.28%** | [+15.23%, +15.32%] |

Speedup = `(baseline latency / current latency - 1) × 100%`。三组对比分别独立执行 ABBA，因此“当前版本”的 latency 有轻微采样差异。

## exp_007 与 exp_008 的关系

```text
exp_005：N128 8-warp（较快，但有 spill）
  → exp_007：双 N64 及时消费（清零 spill，但 A/SFA 重复加载；不测性能）
  → exp_008：Gate/Up 配对复用 A/SFA（保持 zero-spill，并验证性能）
```

两个实验没有重复：exp_007 隔离回答“缩短 accumulator live range 能否清掉 spill”；exp_008 修改了
Gate/Up pipeline，回答“能否消除额外 replay 并转化为性能收益”。新实现生成了不同 cubin，因此 correctness
与 spill gates 也必须重新验证，不能直接继承 exp_007 的结论。

## 验证状态

| 检查 | 结果 |
|---|---|
| Correctness | 8 / 8 cases 通过 |
| 当前版本 spill | Static 0；dynamic 0 |
| E2E 样本 | 3 组直接对比 × 2 cases × 20 positions = 120 |
| 测量协议 | 同 GPU；2377 MHz；5 组 ABBA/case；每 position 独立进程；warmup=5；timed=50；192 MiB L2 flush |
| 身份校验 | Fixture、weights、reference、source、cubin、JIT、容器、依赖与编译工具链全部通过 |
| Phase marker | 只保留定性诊断；插桩扰动 gate 未通过，不用于精确归因 E2E speedup |

机器证据见 [manifest.json](manifest.json)、[e2e/summary.json](e2e/summary.json)、
[correctness_evidence.json](correctness_evidence.json)、[work_ledger.json](work_ledger.json)、
[static_spill_evidence.json](static_spill_evidence.json) 和 [dynamic_ncu_evidence.json](dynamic_ncu_evidence.json)。
