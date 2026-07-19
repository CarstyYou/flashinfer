# exp_007：双 N64 及时消费 accumulator 的 spill 验证

## 结论

**接受该 candidate bundle：spill 已清零。**

Candidate 将 FC1 改为：

```text
half 0: Gate N64 → Up N64 → 立即 SwiGLU → 写 sC
half 1: Gate N64 → Up N64 → 立即 SwiGLU → 写 sC
两半完成后：Q1 N128 → FC2 → scatter
```

它在保持正确性和 Tensor work 等价时，把当前 8-warp N128 anchor 的静态与动态 spill 都降为 0。这支持
“缩小 accumulator ownership，并及时消费以缩短 live range”这一设计方向；但没有 compiler live-range / 动态
spill-PC 证据，因此不能把它表述为唯一因果证明。

## 核心证据

| 证据 | Anchor：8-warp N128 | Candidate：temporal 双 N64 |
|---|---:|---:|
| Correctness | 8 个 required case 全部通过 | 8 个 required case 全部通过 |
| Registers / thread | 168 | 146 |
| Static STACK / thread | 224 B | **0 B** |
| Compiler SpillRefill local SASS | 84 | **0** |
| Dynamic spill refill / store instructions | 1,136,128 / 568,064 | **0 / 0** |
| Dynamic local load / store footprint | 145,424,384 B / 145,424,384 B | **0 B / 0 B** |
| Static OMMA | 448 | 448 |
| Executed Tensor instructions | 31,162,368 | 31,162,368 |
| FP4 Tensor ops | 510,564,237,312 | 510,564,237,312 |
| Achieved occupancy | 18.75% | 18.75% |

Correctness 覆盖 M256、M8192、`sparse_empty`、`exact_128`、`tail_129`、`hot_expert`，以及分别隔离
Gate / Up、两个 N64 half 和四个 logical slice 的 canary。最终 canary 未放宽阈值，最差 cross-arm
relative-L2 为 0.001236、max-abs 为 0.0625。

## 工作等价性与代价

- 两个 N64 half 的 B footprint 合计与 anchor 的一个 N128 pass 相同；Q1、FC2 和 scatter 仍各执行一次。
- A、SFA 和物理 N128 SFB 会 replay 2 次，因此这些 FC1 traffic 是 anchor 的 2 倍。这是当前 bundle 的明确代价。
- Candidate 的 loaded cubin 与静态 SASS/resource 证据完全匹配；动态 NCU 采集的是同一 M8192 CUDA Graph node。

## 证据边界

- 本实验只证明 **该完整 immutable candidate bundle** 可以 zero-spill，不证明“任何 N64 实现都必然 zero-spill”。
- NCU spill counters 只有 launch aggregate，不支持 per-SASS source attribution；空 source rows 没有被当作零 spill 证据。
- canary_v0 的 max-abs failure 和 v1 的无效 W2 scale fixture 均保留；最终采用预注册的 v2 scatter-only scale。
  v2 中遗留的 `weight_sum_max_abs_error` 字段已从证据池排除，实际 `topk_weight_sum`、tensor hash、独立 reference
  和固定阈值均已交叉校验。
- **未测 latency。** zero-spill 是否足以抵消额外 A/SFA/SFB traffic，必须由后续性能实验决定；若更慢则 reject。

机器可审计证据见 [manifest.json](manifest.json)、[correctness_evidence.json](correctness_evidence.json)、
[work_ledger.json](work_ledger.json)、[static_spill_evidence.json](static_spill_evidence.json)、
[dynamic NCU evidence](ncu/) 和 [spill PC identity](ncu/spill_pc_identity.json)。
