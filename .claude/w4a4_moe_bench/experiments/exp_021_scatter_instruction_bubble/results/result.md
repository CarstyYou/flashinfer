# exp_021：Scatter 指令热点与 Bubble

结论：IKET 排除了 standalone Scatter 的明显 CTA 内 warp 收尾问题，并发现 CTA runtime 与路由
工作量强相关；NCU/SASS 进一步把最清晰的指令级瓶颈候选定位到 **`sC` 的标量 `LDS.U16`
读取链**。因此第一优化方向仍是 `sC` layout/load，而不是 barrier；`REDG` 的独立代价尚未闭合。

## 1. IKET：时间线与 Tail

M8192 full-grid warp trace 覆盖 110 个 CTA、每 CTA 8 个 warp。以下均为 IKET
`raw timestamp units`，不能换算为 ns/µs。

| 观测 | Canonical replay | 结论 |
|---|---:|---|
| 同一 CTA 最大 warp end skew | **64** | 未观察到明显 kernel-end warp tail |
| CTA lifetime | 329,984–485,024 | CTA 之间存在明显长尾 |
| 每 CTA valid-row work | 1,833–2,783 | 路由工作量不均 |
| work 与 lifetime Pearson `r` | **0.783** | 两者强相关，但不是单独的因果证明 |

另一轮 launch 的同 CTA 最大 end skew 为 96，方向一致。这个结论只描述 standalone Scatter，
不能直接外推为 production fused kernel 的所有 tile 内 barrier 行为。

Named-range capture 没有进入定量结论：同一 CTA0 的 lifetime 相比无 named overlay 增加约
**26.6%**，且 timestamp 以 32 units 量化。两者资源均为 40 regs/thread、0 stack、0 local，
但运行时扰动已经足以拒绝 production phase 占比。该 trace 只保留作执行顺序诊断。

## 2. Production 精确指令证据

分析对象是 Opt M8192 的四段 Scatter SASS；以下占比是 **not-issued PC samples**，不是耗时占比。

| Stall reason | Samples | 占比 |
|---|---:|---:|
| Short scoreboard | 7,500 | 37.29% |
| Wait | 5,883 | 29.25% |
| MIO throttle | 2,783 | 13.84% |
| Long scoreboard | 2,433 | 12.10% |
| Math pipe throttle | 708 | 3.52% |
| 其余 | 804 | 4.00% |

`Barrier` 在这些 Production Scatter PC 区间内为 0，因此没有证据把当前问题归因于 barrier。

每个静态展开 body 都是 `8×LDS.U16(sC) + 2×LDS(metadata) + convert/FMul/pack + 1×REDG`。
源码对应 [`sC` 读取与 `REDG`](../../../moe_dynamic_kernel_opt.py#L1405-L1473)。

| Shared 读取 | Actual wavefronts | Ideal wavefronts | 放大 |
|---|---:|---:|---:|
| `sC`: 32 个 `LDS.U16` PC | 67,108,864 | 16,880,640 | **3.975×** |
| metadata: 8 个 `LDS` PC | 4,220,160 | 4,220,160 | 1.000× |

`sC` 读取中有 50,228,224 个 excessive wavefront，占实际值的 74.85%；metadata 没有
excessive wavefront。这个对照把问题收敛到了 `sC` layout/access，而不是笼统的 shared memory。

## 3. 哪条指令链在等

| SASS 类别 | PC samples / 1M executed warp instructions | 读法 |
|---|---:|---|
| `LDS.U16(sC)` | 163.7 | load 本身出现 MIO/Wait |
| `IMAD.U32` consumer | 280.2 | 多为等待前序 `LDS` 数据，不代表 IMAD 慢 |
| `IMAD.WIDE` consumer | 948.8 | 同上，是依赖落点 |
| `REDG.BF16x8` | 254.5 | 当前 PC 样本不能单独衡量 reduction service |
| `FMUL` | 53.8 | 不是当前首要矛盾 |

Standalone 保持相同 `LDS/REDG` body 与动态次数时，每 scheduler 有 `2.00` active warps、仅
`0.48` eligible warps，Issue Active 为 `38.70%`，说明这条指令流存在明显 eligibility bubble。
但 standalone 只有 40 regs/thread、没有 FC2 producer，因此这些比例不外推为 Production phase
的定量结果。初始化只占 0.54% 动态指令和 0.36% not-issued samples，不影响该观察。

## 4. 判定与下一步

1. **先验证 `sC` 读路径。** 做一个单变量 demo，只改变 `sC` 的 shared layout 或向量 load，保持
   8-warps mapping、REDG 数量和地址完全不变。接受条件是 wavefront amplification 从 `3.975×`
   明显下降且 standalone latency 同向改善，再集成回 Opt。
2. **REDG 暂不下结论。** Production 有 1 GiB global reduction footprint，但现有证据不能区分
   destination contention、L2 reduction service throughput 与 strong-ordering；REDG PC 样本少也
   不表示它便宜。
3. **区分证据职责。** IKET 用于 phase/warp/CTA 时间线，NCU/SASS 用于精确 transaction、stall
   和 PC 归因；两者不能互相替代。

完整数字、身份与证据边界见 [evidence.json](evidence.json) 和 [manifest.json](manifest.json)。
