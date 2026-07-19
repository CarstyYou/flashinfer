# exp_010：Scatter vs Chain Finalize

## 结论

剥离后的 Scatter **没有复现 full-kernel diagnostic probe 的约 514 μs**：同为 5 replay、`%globaltimer`
additive ticks `/110` 的 SM-equivalent diagnostic estimate，fresh embedded `D→F` p50 为
`512.155 μs`，standalone p50 为 `408.582 μs`，低 `20.223%`。因此 standalone fidelity
gate 失败，不能用它解释完整 fused kernel 中全部 Scatter 成本，也不能据此给出严格 speedup。
embedded probe 的 CUDA-event wall 比 no-marker control 高 2.12%，且 cubin/resource identity 不同；
`512.155 μs` 只用于诊断阶段分布，不是 production-exact wall time。

真实 CUTLASS `finalizeMoeRoutingKernel` 的 NSys kernel wall 为 `223.072 μs`。自写的
source-shape proxy CUDA-event wall 为 `234.464 μs`，相差 `+5.11%`，可用于确认 Chain
finalize 的量级，但它不是同一 cubin。

| 观测 | 数值 | 统计口径 | 相对说明 |
|---|---:|---|---|
| Embedded Fused Scatter `D→F` | **512.155 μs** | 逐 replay 求 `D→F` additive `/110`，再取 5 replay p50 | diagnostic fidelity anchor |
| Standalone logical Scatter replay | **408.582 μs** | 5 replay phase p50，additive `/110` | 低 20.22%，fidelity fail |
| Standalone + 简化 cadence replay | 409.530 μs | 5 replay phase p50，additive `/110` | 比 standalone 高 0.23% |
| Actual CUTLASS Finalize | **223.072 μs** | fresh NSys kernel wall | 真实 graph/PDL launch |
| Source-shape Finalize proxy | 234.464 μs | CUDA-event kernel wall | 比 actual 高 5.11%，descriptive |

前三行是 SM-equivalent phase estimate，后两行是 kernel wall；两类统计量不可相减或计算
speedup。即使忽略口径，contribution representation、rounding boundary 和并行模型也不同。

## 工作量与 dataflow

Standalone 已复放 production 导出的 task/route metadata，逻辑工作量闭合：

| 项目 | Fused Scatter | Chain Finalize |
|---|---:|---:|
| Routed rows | 65,536 | 65,536 |
| 每个 output element 的 contribution | 4 slices × 8 routes = **32** | 8 full-K rows |
| Reduction mechanism | 67,108,864 次 BF16x8 `REDG` | FP32 register accumulation + 1 次 BF16 store |
| Global reduction footprint | **1,073,741,824 B** | 0 B |
| Global load / store request footprint | whole-Fused 无法按 phase拆分 | 1,124.073 MB / 134.218 MB |

这证明两条路径的核心算法差异：Fused 将每个 FC2 K-slice 立即写入最终 Y，形成 32-way
global reduction；Chain 先形成 full-K expert row，再按 top-k 在寄存器中归并并只写一次 Y。
它能解释“为什么值得调查 reduction ownership”，但尚不能量化该机制对端到端时间的贡献。

## 为什么 standalone 没对应上 514 μs

| Fidelity 项 | Embedded Fused probe | Standalone | 判定 |
|---|---|---|---|
| task / routed row / REDG 数 | 2536 / 65536 / 67,108,864 | 相同 | ✓ |
| SMEM `sC` layout / load | swizzled，`8×LDS.U16 + 2×LDS` | row-major，`1×LDS.128 + 2×LDS` | ✗ |
| multiply SASS | `FMUL` | `FMUL.FTZ` | ✗ |
| Registers / thread | 255（完整 fused kernel） | 40 | ✗ |
| Dynamic SMEM / CTA | 91,136 B | 92,160 B launch allocation | △ |
| L2 初态策略 | production 自然执行上下文 | 未做 matched eviction | ✗ |
| `D→F` phase p50 | 512.155 μs | 408.582 μs | ✗ |

同口径 phase estimate 相差约 `103.573 μs`，但它混合了 layout、resource envelope、SASS、
cache 初态及完整 kernel embedding 等多个未闭合维度，当前无法分解。demo 不是 production
Scatter 的 machine replica。

## NCU 证据

本实验不新增 standalone NCU 因果结论：fidelity 已失败，采到的只会解释 row-major demo。
以下复用 exp_002 同源码/软件身份、同型号但**不同物理 5KP GPU**的 silicon evidence。Fused
数字覆盖完整 MoEDynamicKernel，不能按 phase 投影；Finalize 数字对应独立真实 launch。

| Metric | Fused whole launch | Chain Finalize |
|---|---:|---:|
| Registers / thread | 255 | 48 |
| Achieved occupancy | 10.42% | 79.01% |
| Issue active | 19.98% | 25.61% |
| Global reduction footprint | **1,073.742 MB** | 0 |
| Local load + store footprint | 316.817 MB | 0 |
| Long scoreboard stall | 19.01% | 89.67% |

Finalize 观测到较高 long-scoreboard stall，同时具有大 grid、高 occupancy、无 global reduction；
它在约 223 μs 完成。Fused whole-launch 的低 occupancy 与 local traffic 是重要上下文，
但不能在没有 phase-local counterfactual 时直接归因到 Scatter。

## 诊断 controls

- 当前 cadence replay 对 `D→F` 改变 `+0.23%`；它只重放局部等待，不能代表真实跨 CTA
  到达时序，因此不支持“cadence 无关”的结论。
- Direct-grid 与 component baseline 的 phase p50 分别为 `411.324/411.332 μs`（`−0.002%`）；
  CUDA-event wall 分别为 `4468.512/4642.976 μs`（`−3.76%`）。phase service未改善，但整体
  调度区间有变化；两种口径必须同时保留。旧 `/2536=17.853 μs` 不是可比 latency。
- Shard 4/32 同时扩大 output footprint 至 4×/32×，明显变慢不能单独证明 contention。
- Span-matched phase p50 为 `322.450 μs`，但 event wall 只改善 `1.10%`，且 arm 非正交；
  原因保持 unresolved，不作为 cache/locality 优化结论。

## 判定与下一步

当前能确认的瓶颈线索是 **32-way global-reduction work 与完整 fused kernel 的资源/布局上下文**；
尚不能确认 atomic contention、SMEM layout 或低 occupancy 各自贡献多少。

下一步不继续用普通 standalone 猜 production：应在 production 内做最小 counterfactual，保持
原 SMEM layout、255-register envelope、task cadence 和其余 phase 不变，只改变 Scatter 的
reduction ownership/multiplicity，再看 `D→F` 与未插桩 e2e 是否同步改善。

数据审计：fresh embedded diagnostic anchor、actual CUTLASS anchor 和 work ledger 为 `✓ vetted`；
standalone 的 machine fidelity 与独立 correctness reference 为 `⚠/✗`；所有严格 B/C speedup、
sharding 因果和 direct-grid 17.853 μs 对比均已拒绝。
