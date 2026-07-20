# exp_016：Route/Q0 Token-major 输入复用

## 目标

在当前 accepted optimized fused kernel 上，验证锁定的 `topk=8 + [E] unequal-scale` 场景中，
Route/Q0 从 route-pair-major 改为 token-major，并复用同一 token 的 BF16 load 与 block absmax，
能否降低前端 phase latency 和 fused E2E latency。

本实验不减少量化工作：每个 token 的 8 个 expert 仍分别使用各自的 `input_global_scale[e]`
完成 8 份 quant/store。equal-scale 的 quantize-once/fanout 不是本实验变量。

## 对照与单一改动

- Baseline：当前 `moe_dynamic_kernel_opt.py`，source SHA256
  `c88cef63492b60c0a77484b50f6400b83a103d168e1535b78972341503810184`；
  `110 CTA × 288 threads`，P3 由 W0–W8 每轮处理 18 个 routed pairs。
- Candidate：从 Baseline 生成独立 overlay；P3 改为每个 warp 每轮处理 1 个 token：
  lane 0 为 top-8 routes 分配并发布 row/metadata，全 warp 对每个 16-BF16 block 只 load 和求 absmax
  一次，再按 8 个 expert scale 分别 quant/store。
- P3 ownership 锁定为：`pair_head` 计 token、每 CTA claim 9 tokens、W0–W8 各处理 1 token；
  每 token 必须完成 8 次 row allocation、metadata 和 quant/store。验证末轮不足 9 token 时无重复和空洞。
- 保持不变：P0/P1/P2/P4、FC1、SwiGLU/Q1、FC2、Scatter、CTA 配置、输出格式、Graph 模式、
  输入、工具链和 GPU。
- 主测试始终传入 `[E]` input scale（equal-scale sanity 也保持 `[E]`），并断言
  `share_input_across_experts=False`、`input_scales_are_reciprocal=False`、`fast_math=True`；锁定
  dispatch source、JIT artifact 和 cubin 身份，禁止误入已有 scalar equal-scale fanout specialization。
- 锁定 `full_tile_publish_enabled=0` 并保留 deferred-publish 协议；enabled overlap 路径不在本实验范围。
- 不修改 accepted opt；Candidate 仅通过实验 overlay 注入，通过门槛后才提升。

## 工作量账本（M8192、topk=8、H=2048）

| 项目 | Baseline | Candidate | 约束 |
|---|---:|---:|---|
| logical routes | 65,536 | 65,536 | 相同 |
| FP4 quant blocks（128/token-route） | 8,388,608 | 8,388,608 | 必须相同 |
| per-expert quant/store | 8/token | 8/token | 必须相同 |
| BF16 input block loads | 8,388,608 | 1,048,576 | 预期降 8× |
| BF16 input elements loaded | 134,217,728 | 16,777,216 | 预期降 8× |
| block absmax | 8,388,608 | 1,048,576 | 预期降 8× |
| global `pair_head` productive claims | 3,641 | 911 | `ceil(work/claim)`，预期约降 4× |
| row allocation atomics | 65,536 | 65,536 | 必须相同 |
| topk ID/weight logical loads | 65,536 | 65,536 | 必须相同 |
| logical expert-scale normalization | 65,536 | 65,536 | 必须相同且在 block loop 外 |
| lane-level expert-scale loads | 2,097,152 | 2,097,152 | 当前实现保留每 lane load，不归入本次收益 |
| packed FP4/SFA/route metadata outputs | 相同 | 相同 | 必须闭合 |

若 Candidate 的 quant/conversion 工作也减少，则本实验的因果门禁失败，不能将收益归因于
token-major 输入与 absmax 复用。

## 渐进验证

1. **结构与账本**：CPU/静态 gate 证明 token/route ownership、top-8 独立 scale、quant/store 次数和
   输出覆盖闭合。按 `(expert, token, occurrence, weight bits)` canonicalize physical rows后比较
   FP4/SFA payload、metadata、row counts 和 task coverage；不直接逐 physical row 比较。
2. **编译资源**：Baseline/Candidate 均检查静态 zero spill；重点审计 values、8 个 route offset 和
   expert scale 的 live range。静态 spill 立即 Reject。
3. **正确性**：固定测试 `[M256 equal-scale [E], M256 unequal-scale [E], M256 hot-expert,
   M256 tail, M8192 unequal-scale [E]]`；检查 canonicalized FP4/SFA、token/weight metadata、
   route/task gate、最终 Y 和全零无效行。atomic Scatter 加法顺序允许在 baseline-derived tolerance 内变化。
4. **快速性能门**：同机未插桩 M8192，5 warmup、每 position 50 replay，做 3 组 ABBA；每组用两端
   Baseline 中值对同组 Candidate 中值。三组 delta 必须同号、median improvement 至少 2%，且两臂
   replay CV 均不超过 1.5%，否则 Reject/Unresolved，不把噪声解释为收益。
5. **正向补证**：仅在快速门通过后补 M256–8192 sweep、窄 P3 start/end `%globaltimer` phase probe，
   以及 M8192 dynamic zero-spill。P3 marker 从 prefix barrier 之后、首次 claim 之前开始，到全部
   quant/store reconverge 之后、final fence/deferred publish 之前结束；报告 grid critical wall，不把
   SM-equivalent additive estimate当 latency。probe/no-marker 必须检查资源、spill 和 E2E 扰动；marker
   失真只使 phase 证据 Unresolved，不作为 Candidate Reject 条件。
6. **按需 NCU**：只有 phase 与 E2E 同向改善后，才采 scoped memory、atomic、conversion 和 warp
   activity；不重做 fused-vs-chain 全套 breakdown，也不重采 Chain/IKET。

旧 exp_011 的 `348.816 us` 来自 `110×160/5-warp` production，不得作为当前 Baseline；当前
Baseline/Candidate 必须成对重采。exp_008 的 `front_end_route_q0=279.122 us` 包含 P0–P4 且 probe
扰动为 `+3.189%`，只能作为优先级背景，不能进入本实验 speedup 计算。

## 判定

- **Accept**：全部 correctness 通过；静态和动态均 zero spill；合法 P3 probe 时 phase latency 明确下降；
  M8192 未插桩满足上述 3×ABBA 门槛，且完整 `M>=256` sweep 无超过 1.5% 的稳定回退。
- **Reject**：正确性/工作量账本/spill 任一失败，或 M8192 P3/E2E 没有可重复的同向收益。
- phase probe 只作诊断；最终性能判定以未插桩 E2E 为准。

## 输出

- `results/result.md`：只展示 Baseline、Candidate 的 correctness、工作量、spill、P3 latency 与
  fused E2E 对比，并明确 Accept/Reject。
- `results/manifest.json` 与 `results/raw/`：保存 source/cubin、GPU UUID、nvcc、grid/block、输入和
  原始证据；读者报告不展开这些身份细节。

## Plan Review

**Date**: 2026-07-20
**Reviewer**: subagent
**Verdict**: ⚠ Ready with fixes

审查发现 dispatch specialization、P3 ownership、deferred publish、完整工作量账本、physical-row
canonicalization、marker 边界和可执行性能阈值未锁定。以上缺口已一次性并入本 plan；按 single-round
规则不再复审，直接进入实施。
