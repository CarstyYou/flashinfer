# exp_014：Fused Scatter 有效 4 Warp → 8 Warp

## 目标

在 accepted optimized fused kernel 中，只改变 Scatter 的 work mapping：让 W0–W7 共同覆盖同一个
`M128 × N128` output tile，验证它能否降低 Scatter phase latency 和 fused E2E latency。

本实验不是“CTA 配置为 8 个 math warp”——当前 CTA 已有 8 个 math warp；问题是现有 Scatter
`2×2` quadrant mapping 只有 W0–W3 有效工作，W4–W7 因 `warp_m_base >= 128` 空转。

## 对照与单一改动

- Baseline：当前 `moe_dynamic_kernel_opt.py`，Scatter 由 4 个有效 warp 各负责 `64×64`。
- Candidate：从 Baseline 生成独立 overlay；Scatter 由 8 个有效 warp 各负责 `32×64`。
- Candidate mapping 锁定为 `warp_m=(warp>>1)*32`、`warp_n=(warp&1)*64`、row clamp=32；
  对每个有效 `(M,N)` 元素证明恰好写一次，REDG vector width 和次数不变。
- 保持不变：FC1/Q1/FC2、CTA 的 8 math + 1 TMA warp 配置、scheduler、barrier、tile、vector width、
  REDG 次数、输入、工具链和 GPU。
- 不修改 production kernel；Candidate 通过实验 overlay 注入，达到门槛后再提升到 opt baseline。

## 验证

1. 身份：记录 baseline/candidate source hash、JIT/cubin hash、GPU UUID、nvcc、grid/block；禁止缓存串臂。
2. 正确性：覆盖 `M=256, 512, 1024, 2048, 4096, 8192`，并定向构造
   `valid_rows=1/31/32/33/63/64/65/95/96/97/127/128`；检查首、中、末全部 N128 output tile，
   数值、finite、workspace/route/task gate、full-zero rows，以及每个有效元素恰好一个 owner。
3. 资源：编译信息与一次 M8192 dynamic NCU 均须证明 zero spill；同时确认 8 个 math warp 都有
   Scatter work，不以 block size 代替有效 warp 证据。
4. 性能：同机、同 route/task 输入、同 CUDA Graph 模式做 paired fused E2E；Baseline 预先锁定
   source/cubin hash。匹配的 `%globaltimer` probe 采集全部 8 个 warp，以 CTA 最早进入到最晚完成
   定义 Scatter latency；所有 warp 无条件进入前后 barrier，并记录 probe 对未插桩 E2E 的扰动。
5. 插桩版单独检查 zero spill，不能用未插桩 cubin 的资源结果替代。

## 判定

- **Accept**：全部 correctness 通过、zero spill，M8192 Scatter phase 明确下降，且 `M>=256` fused E2E
  无显著回退并在主要 prefill case 有稳定收益。
- **Reject**：correctness/spill 失败，或 Scatter phase / fused E2E 没有可重复收益。
- 若 phase probe 扰动过大，只把它作为诊断证据；最终性能判定以未插桩 fused E2E 为准。

## 输出

- `results/result.md`：只展示 Baseline 4-warp 与 Candidate 8-warp 的 correctness、spill、Scatter phase
  latency 和 fused E2E 对比。
- `results/raw/` 与 `results/manifest.json`：保存可追溯原始证据和完整身份。

旧 standalone Direct-32/Slice-8/Form A harness 与结果不属于本实验；实施前从目录隔离，不能混入
fused 4→8-warp 的 evidence 或 commit。

## Plan Review

**Date**: 2026-07-20
**Reviewer**: subagent

**Verdict**: ✗ Misaligned

**Gaps + suggested fix**:

- 旧 harness/结果仍是 standalone reduction-form 实验：替换为 fused overlay harness，并隔离旧结果。
- Ownership 不可审计：锁定 `32×64` 映射、row clamp=32，并验证每个有效元素恰好一个 owner。
- Tail 与 output-tile 覆盖不足：加入 12 个边界 `valid_rows`，覆盖首、中、末 N128 tile。
- phase probe 边界不合法：采集全部 8 warp，以 CTA 最慢完成时间为终点，barrier 保持无条件参与。
- 身份与资源约束不足：锁定 source/cubin、route/task、Graph 模式，并单独验证插桩 cubin zero spill。

以上缺口已一次性并入本 plan；按 single-round 规则不再复审，直接进入实施。
