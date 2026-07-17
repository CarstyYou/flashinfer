# exp_005：8-Warp Gate/Up Spill Reduction

## 结论

**Candidate A 不通过。** 将 compute warps 从 4 增加到 8 后，M8192 整体快
**9.75%**，但 candidate 仍有 **224 B/thread static frame、84 条 compiler
SpillRefill SASS**，且 NCU 动态 spill 计数非零。按本实验“kernel 基本不允许
spill”的最高优先级约束，不能进入 production。

这是一个有性能收益但没有完成目标的设计实验：8-warp/layout 是不可拆的整体改动，
不能把加速纯归因于 spill 减少。

## 1. 对比约束与正确性

两臂使用同一 source commit、fixture、toolchain、GPU UUID 和 grid
`(1,1,110)`。production 未修改；candidate 只有两处变化：

```text
num_mma_warps: 4 → 8
atom_layout:   (2,2,1) → (4,2,1)
```

| M | Independent reference | Route/task | Strict cross-arm | 性能处理 |
|---:|---|---|---|---|
| 256 | Pass | Pass | Pass | 已测 |
| 1024 | Pass | Pass | **Fail / inconclusive** | 未测 |
| 8192 | Pass | Pass | Pass | 已测、NCU primary case |

M1024 的 baseline/candidate self cosine drift 分别为 `1.4377e-4` / `1.4436e-4`，
cross-arm worst 为 `1.4639e-4`，均超过预注册 cap `1e-4`。因此只能判定
strict comparison protocol 未通过，不能写成 candidate 特有的 correctness failure，
也不事后放宽阈值。

`sparse_empty / exact_128 / tail_129 / hot_expert` 四个 directed fixtures 的
formal reference、sentinel 和 route/task gate 在两臂均通过。

## 2. Spill 与资源证据

| M8192 指标 | Baseline 4-warp | Candidate 8-warp | 变化 |
|---|---:|---:|---:|
| Block threads | 160 | 288 | +128 |
| Registers/thread | 255 | 168 | -34.1% |
| Static frame/thread | 488 B | 224 B | **-54.1%** |
| Spill words/lane | 122 | 56 | -54.1% |
| Compiler SpillRefill SASS | 190（122 load + 68 store） | 84（56 load + 28 store） | -55.8% |
| Dynamic spill refill instructions | 1,237,568 | 1,136,128 | -8.2% |
| Dynamic spill store instructions | 689,792 | 568,064 | -17.6% |
| Generic local load+store footprint | 316.82 MB | 290.86 MB | -8.2% |
| Zero-spill gate | — | **Fail** | residual spill |

Tensor work identity 通过：两臂均为 `31,162,368` Tensor instructions 和
`510,564,237,312` FP4 Tensor ops，没有通过少算工作获得资源或性能收益。

## 3. 性能与 schedule 变化

`speedup = baseline_time / candidate_time - 1`；5 组 ABBA、每臂 10 个样本，
application clock 固定为 2377 MHz。

| M | Baseline median | Candidate median | Speedup | Paired bootstrap ratio 95% CI | 预注册判定 |
|---:|---:|---:|---:|---:|---|
| 256 | 539.655 us | 532.348 us | +1.37% | [1.0134, 1.0140] | Equivalent（±2% band） |
| 8192 | 1787.283 us | 1628.537 us | **+9.75%** | [1.0971, 1.0978] | Faster |

M8192 NCU whole-kernel duration 为 `1891.360 → 1721.216 us`，speedup
`+9.89%`，与 benchmark 的方向和幅度一致。

| M8192 NCU 指标 | Baseline | Candidate |
|---|---:|---:|
| Achieved occupancy | 10.42% | 18.74% |
| Eligible warps/cycle | 0.20 | 0.27 |
| Issue active | 19.97% | 23.14% |
| TC subpipe active | 25.74% | 28.35% |
| Executed warp instructions | 386.62 M | 406.17 M（+5.1%） |
| Warp stalls：Wait / Long / Short / Barrier | 29.2% / 18.8% / 9.7% / 5.6% | 18.4% / 15.5% / 5.5% / 20.3% |
| Math pipe throttle | 0.2% | 10.1% |

观测到的正向变化包括 local/spill traffic 减少、可调度 warp 增加以及
Issue/TC active 提高；代价是更多 executed warp instructions，并显著增加 Barrier
stall 与 Math pipe throttle。它们都是 8-warp/layout bundle 的伴随证据，不是
“spill 减少导致 9.75% 加速”的纯因果分解。

## 4. 判定与下一步

1. 不修改 production，不采用 Candidate A。
2. 下一候选应优先验证“缩短 Gate accumulator 跨完整 Up GEMM 的生命周期”，例如
   smaller sub-tile/interleave 或受控 SMEM handoff；这是下一实验假设，不是本实验已证明的优化结论。
3. 若保留 8-warp 作为基础，应以本 candidate 为 matched control，只增加一个
   lifetime 变量，并先过 zero-spill gate，再比较性能。
4. M1024 若要进入 acceptance matrix，必须前置修订并锁定数值稳定性协议后 fresh rerun，
   不能沿用本次数据后调阈值。

证据边界：

- 正式 NCU 证据只使用 `canonical_v1`；缺少 spill-derived metrics 的
  `canonical_v0` 已隔离为失败 capture。
- `launch__stack_size=1024` 是 runtime configured limit，不是 488/224 B static frame。
- `[inference]` 物理 SASS 程序序支持优先调查 `gate_acc` live range，但 cubin 无 source lineinfo 和
  SSA→physical map，不能把 56 words 全部严格命名为 `gate_acc`。
- Directed exact-once 由 descriptor multiset 与 terminal atomic-head 推断，没有独立
  consumed bitmap。

主要证据：[comparison registry](../comparison_registry.json)、
[correctness](correctness/)、[benchmark](benchmark/)、
[static spill evidence](static_spill_evidence.json)、
[NCU evidence](ncu/evidence.json)。
