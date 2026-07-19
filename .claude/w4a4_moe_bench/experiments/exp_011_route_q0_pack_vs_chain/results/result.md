# exp_011：Route/Q0/Pack 瓶颈分析

## 结论

在 `M=8192, E=256, H=2048, topk=8` 上，Fused `P3 Route + Q0 + Pack` 的
full-kernel diagnostic probe p50 为 **348.816 μs**；actual CUTLASS `Expand` 为 **250.847 μs**。
两者责任和并行拓扑不完全相同，只能并列观察量级，**不计算 ratio**；现有证据尚未解释
Fused 与 Chain 的这段 baseline 差距。

当前已测 controls 中，收益最大的优化机会是 **equal-scale shared-input composite**：主用例的 256 个
per-expert input scales 逐 bit 相同，但 production 因 shape 为 `[E]` 没进入已有的
quantize-once/fanout 路径。规范化为 scalar 后：

- P3：`348.816 → 105.357 μs`，**降低 69.80%**；
- 未插桩完整 Fused kernel：`1799.040 → 1565.824 μs`，**降低 12.96%**；
- correctness、workspace、工作量与 255 registers/thread resource gate 全部通过。

该 control 同时改变 quant work、input access 与 token/pair scheduling，尚不能把收益拆成纯
Q0、load 或 schedule。完全删除 `pair_head` claim 没有收益，证据支持将其列为低优先级；
precomputed-row control 删除 65,536 次 `expert_write_rows` atomic，但同时改变 row order 并增加
decode，因此只能把 row allocation 列为**相对低优先级**，不能作纯 atomic 因果结论。

## 1. 比较约定与 fidelity

| Gate | Reference | Fresh identity | 差异 | 判定 |
|---|---:|---:|---:|---:|
| P3 diagnostic replay p50 | 348.531 μs（exp_004） | **348.816 μs** | +0.082% | ✓ diagnostic anchor |
| 未插桩完整 Fused event p50 | 1803.840 μs | **1799.040 μs** | -0.266% | ✓（阈值 1%） |
| Correctness / workspace | pass | 8 个 probe/control arm 全 pass | — | ✓ |

P3 使用 5 replay 的 `%globaltimer` additive ticks `/110`，是 SM-equivalent phase estimate；
完整 Fused 使用 CUDA-event kernel wall；CUTLASS 使用 NSys/NCU 的独立 kernel wall。三种口径分列，
禁止相加。identity overlay 保留完整 P0→Scatter kernel，launch 仍为 `110 CTA × 160 threads`。
exp_004 报告中的 348.800 μs 是 5 replay aggregate mean；本表从原始 evidence 重算同口径 replay p50。
probe 与 no-marker cubin 不同，probe event 比 no-marker 高约 1.99%，因此 P3 只能用于诊断阶段分布；
**no-marker event 才是 production-fidelity wall control**。

## 2. 实际执行拓扑与耗时

```text
Fused one-kernel
P1 Histogram → P2 Prefix → P3 [claim → row allocation → BF16 load → Q0/Pack → route stores]
                                      110 CTA × 5 producer warps               → grid barrier

CUTLASS chain
Block Prefix → Global Prefix → Merge Prefix → Expand BF16→NVFP4
                                              880 CTA × 256 threads
```

| 路径 | 阶段 | 时间 | 口径 |
|---|---|---:|---|
| Fused | P1 Histogram | 7.091 μs | phase p50 |
| Fused | P2 Prefix | 13.644 μs | phase p50 |
| Fused | **P3 Route/Q0/Pack** | **348.816 μs** | phase p50 |
| Fused | P1+P2+P3 | **369.567 μs** | 逐 replay 求和后的 p50 |
| Chain | Block / Global / Merge Prefix | 33.728 / 1.984 / 12.544 μs | actual NSys kernel wall |
| Chain | **Expand BF16→NVFP4** | **250.847 μs** | actual NSys kernel wall |
| Chain | Prefix3→Expand interval | **298.623 μs** | first start→Expand completion；PDL overlap 0.480 μs |

Fused `P1+P2+P3` 与 Chain `Prefix3→Expand` 只并列 raw time：Fused 还在 P3 做动态 row
allocation 与 route metadata，Chain Expand 已消费完整 row map；physical layout 和同步责任也不同。

Baseline P3 的 logical work 为 65,536 routes、8,388,608 FP4 blocks、134,217,728 BF16
values（268.435 MB input payload）。shared-input arm 将量化降为 1,048,576 blocks、16,777,216
values（33.554 MB），但 65,536 routed rows 的 packed FP4、scale 和 route stores 不变。

## 3. 最小 controls 与 NCU 证据

| Full-kernel arm | P3 p50 | 相对 | 未插桩 event p50 | 相对 | Global atomic requests | Conversion thread inst. | Memory thread inst. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Identity | 348.816 μs | — | 1799.040 μs | — | 77,444 | 741.182 M | 1,783.835 M |
| Shared equal scale | **105.357 μs** | **-69.80%** | **1565.824 μs** | **-12.96%** | 72,529 | 656.247 M | 1,644.485 M |
| Static batch schedule | 349.808 μs | +0.28% | 1802.464 μs | +0.19% | 70,780 | 741.182 M | 1,784.230 M |
| Precomputed physical row | 343.489 μs | -1.53% | 1801.920 μs | +0.16% | 11,908 | 741.182 M | 1,783.699 M |

NCU 对每个 arm 只采一个未插桩完整 Fused launch。上述指令与 atomic 是 whole-kernel 可加计数，
不能按 P3 时间占比投影；它们只用于 matched-arm delta：

- Static arm 恰好减少 **6,664** 个 global atomic requests，对应删除 P3 `pair_head` claims，
  但 P3/完整 kernel 都未改善，故 claim 不是当前主要瓶颈。
- Precomputed-row arm 减少 **65,536** 个 global atomic requests，对应删除每 route 的 row
  allocation；P3 仅降 1.53%，完整 kernel +0.16%。但该 control 同时改变 row order，并使 P1
  从 7.091 μs 升至 13.574 μs，故只能判为相对低优先级，不能单独归因给 row atomic。
- Shared arm 的 atomic 只少 **4,915**（claim 数降低），但同时少 84.935 M conversion、
  139.350 M memory thread instructions 和 14.680 M bit instructions；这与 P3 和未插桩 event
  的大幅同向改善一致，但仍不能区分 quant、input access 与 schedule 的独立贡献。
- 四个 Fused arm 均为 255 registers/thread、1024 B stack；local load+store 约 317 MB，
  没有出现能解释 243 μs P3 改善的 resource-envelope 变化。

Actual Chain Expand 的 fresh NCU identity 为 `grid=880, block=256, 48 registers/thread`，
kernel duration 271.840 μs，global atomic=0、local load/store=0。它是独立 kernel，不能与
whole-Fused counters横向相减；这里仅确认 Chain 的量化/pack 路径没有承担上述两类 atomic。
Chain Expand 也执行 134,217,728 次 conversion，因此 equal-scale fast path 的收益**不能解释**
baseline Fused-vs-Chain 差距。

## 4. 已定位优化机会、未闭合问题与下一步

**已定位：** canonical equal-scale case 中，shared-input composite 是当前 controls 里收益最大的
优化机会；`pair_head` 是低优先级，`expert_write_rows` 是相对低优先级。shared arm 的 P3 phase
减少 **243.459 μs**，未插桩完整 kernel wall 减少 **233.216 μs**；两者分母不同，只作为
同向证据，不作收益闭合。

**未闭合：** shared 收益中 quant reuse、input access 与 scheduling 各占多少，以及一般
unequal-scale case 的 Fused-vs-Chain baseline 差距，当前都没有因果证据。

下一步按优先级：

1. 先增加“token schedule + 仍量化 8 次”的 companion control，把 quant reuse 与 schedule 改变的
   独立贡献分开；同时补 P3-scoped DRAM/L2 与 PC-level 证据。
2. 若进入 production，只允许标量化 P3 input scale；FC1 compute alpha 必须继续保留 `[E]`，并覆盖
   unequal-scale、hot-expert 与 `M=256..8192` correctness/benchmark。
3. 对一般 unequal-scale case，验证 `110×5 warps` 与 Chain `880×256 threads` 的并行映射差异。
   这是解释 baseline 差距的高优先级假设，当前尚未用受控实验确认。

因果审计结论：shared composite benefit 与 `pair_head` 低优先级为 `✓ vetted`；纯 Q0/load/schedule
拆分、row atomic 独立贡献及 baseline Fused-vs-Chain 原因为 `⚠ unresolved`。
