# exp_021：Scatter 指令热点与 Bubble

结论：当前最清晰、已定位的瓶颈候选是 **`sC` 的标量 `LDS.U16` 读取链**。Production
每次 `BF16x8` Scatter 先执行 8 次 `LDS.U16`；在当前 `K_SW128` layout 与 lane mapping 下，
这些读取产生约 **3.98× shared wavefront amplification**。等待主要落在 shared-memory 发射和
后继依赖链，不能解释为 `IMAD/SHF` 本身慢。

## 1. Production 精确证据

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

## 2. 哪条指令链在等

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

## 3. 判定与下一步

1. **先验证 `sC` 读路径。** 做一个单变量 demo，只改变 `sC` 的 shared layout 或向量 load，保持
   8-warps mapping、REDG 数量和地址完全不变。接受条件是 wavefront amplification 从 `3.975×`
   明显下降且 standalone latency 同向改善，再集成回 Opt。
2. **REDG 暂不下结论。** Production 有 1 GiB global reduction footprint，但现有证据不能区分
   destination contention、L2 reduction service throughput 与 strong-ordering；REDG PC 样本少也
   不表示它便宜。
3. **不需要 IKET。** 当前第一问题点已由 Production exact SASS/source counters 定位；只有后续要
   区分 warp skew 或 tail 时才需要 trace。

完整数字、身份与证据边界见 [evidence.json](evidence.json) 和 [manifest.json](manifest.json)。
