# exp_010：Fused Scatter 与 Chain Finalize 瓶颈分析

## 状态

已收口（2026-07-19）：R1 standalone fidelity gate 失败；R2 只保留 descriptive
component reference。Plan Review 已完成且未进行第二轮 review。结论见 `results/result.md`。

## 目标

在 `M=8192, E=256, H=2048, I_tp=512, topk=8, SwiGLU` 主用例上回答：

1. 能否把 production Fused `D→F` Scatter dataflow 剥离为 standalone diagnostic
   kernel，并在相同工作量与计时口径下复现约 `513.971 us`；
2. Fused Scatter 为什么比 CUTLASS Chain 的 `finalizeMoeRoutingKernel` 慢；
3. 将差距至少定位到以下可判别机制之一，而不是直接猜优化：
   - `4 FC2 K-slices × 8 top-k` 带来的 reduction work multiplicity；
   - 同一最终输出地址的 atomic contention；
   - persistent task/warp 并行度、负载不均或 latency-hiding；
   - Scatter body 内的 SMEM load、convert/multiply/address 或 REDG issue/latency。

`M=256` 只作为小 M 控制，不取代 M8192 主结论。

## 已锁定事实与锚点

- Production source：
  `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py`，
  SHA256 `94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106`。
- exp_006 M8192 diagnostic phase anchor：
  - `D→E` Scatter body `506.585 us`；
  - `E→F` post-sync `7.385 us`；
  - `D→F` `513.971 us`。
- 该时间是 5 replay × 110 CTA、`%globaltimer` SM-equivalent additive estimate，
  不是 CUDA-event kernel latency；fresh fidelity capture 必须使用相同口径。
- 每个 task 为 `expert × M-tile<=128 × FC2 K-slice=128`；每 task sweep 16 个
  N=128 output tiles。W0–W3 各负责 `64×64` quadrant，W4 不执行 Scatter。
- 每个最终 output element 有 `4 slices × 8 experts = 32` 个 BF16 reduction
  contribution。理论 payload `1,073,741,824 B` 与既有 Fused operator-range NCU
  global-reduction footprint 精确闭合。

## 对比关系与允许结论

| ID | Subject | Anchor / reference | 类型 | 允许结论 |
|---|---|---|---|---|
| R1 | standalone source-faithful Fused Scatter | fresh production exp_006-style `D→F` | diagnostic-only fidelity | 仅判断剥离是否保留 production Scatter 时间/工作特征 |
| R2 | GMEM-resident Fused-style aggregation | actual-source Chain finalize | component-reference | 从同一个 FP32 slice ledger 构造各自合法输入，在共同的“GMEM contribution representation→最终 Y”边界比较；不称为 whole-op speedup、同输入 kernel speedup 或纯 fusion 因果 |
| R3 | Scatter diagnostic arm | R1 standalone anchor | diagnostic-only | 只归因预先声明的单变量；不进入 production 排名 |

所有 arm 源自同一个 canonical FP32 slice ledger。Fused 路径按 production 语义将每个
slice 独立 FP32→BF16，再乘 route weight 并做 32 次 BF16 reduction；Chain 路径先按
full-K FC2 contract 在 FP32 合并 4 slices、再 BF16 round，最后由 finalize 做 FP32 top-k
归并。两条路径分别校验各自 reference；禁止把 BF16 partial 直接相加冒充 Chain 输入，
也禁止要求两条路径 bitwise 相同。物理 layout 转换、workspace 构造和 output clear 均在
timed boundary 外，并单独记录时间但不回填。

## Arm 与数据流

### A. Fresh production phase anchor

复用 exp_006 已审计的 D/E/F boundary，在同一 GPU/session fresh capture：

```text
RMEM down_acc
  -> down_alpha / BF16 cast / R2S / pre-barrier
  -> D
  -> SMEM token+weight + SMEM sC
  -> FP32 multiply / BF16x8 pack / REDG to final Y
  -> E -> post-barrier -> F
```

Fresh A 必须从实际运行中导出并固化：`token_map/weights`、physical-row allocation、
task descriptor 顺序以及 task→CTA 动态分配。B replay 前校验 descriptor multiset、
token collision histogram、2536 tasks、valid-row histogram 和每 CTA task/work 分布；
不得凭旧 fixture 推断或重建原子分配次序。

同一 session 另采 actual CUTLASS graph 中的 finalize anchor（目标约 `223 us`），锁定其
前驱 launch、PDL signal/wait context、launch flags 与 correlation identity。C 未通过该
fresh anchor 的 fidelity gate 时，只能作为 descriptive source microbenchmark。

### B. Standalone Fused Scatter

实验目录内实现，不修改 production：

- `grid=110 CTA`、`block=160`；W0–W3 执行与 production 相同的 quadrant、
  vector-loop 和 BF16x8 REDG；W4 保持非 Scatter role；
- 保留 production task descriptor 顺序、2536 tasks、每 task valid-row 分布、
  4 slice multiplicity、16 output tiles、token destination 与 route weight；
- 每 tile 的 synthetic/canonical FC2 BF16 partial 先准备到 `sC`，preload/fill 和
  output clear 均位于 D 前；D→F 只包含 production Scatter body 与 post-sync；
- 保留相同的 shared token/weight cache、`64×64` warp ownership、每次 8 BF16、
  4 pack + 1 REDG 的 generated mechanism；
- 输出与 CPU/PyTorch FP32 accumulation reference 及 production fixture 做容差正确性。

B 分成两个明确用途，禁止混用时间：

- `B_phase`：贡献已经驻留 SMEM，从 D 计到 F；只用于 R1 的约 `514 us` fidelity；
- `B_component`：从 GMEM-resident、per-slice BF16 contribution 开始，包含 source-faithful
  GMEM→SMEM staging 再执行相同 Scatter，计到最终 Y；只用于 R2 的共同组件边界。

禁止用一个从 GMEM dense tensor 直接 scatter 的普通 kernel 冒充 source-faithful arm。

### C. Chain Finalize

实验目录内包装实际源逻辑：

- 来源 `csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh` 的
  `finalizeMoeRoutingKernel`；`grid=M`、`block=256`；
- 每 CTA 一个 token，读取 8 个 permuted expert rows，FP32 乘 route scale 并归并，
  最后一次 BF16 vector store；无 global reduction atomic；
- 输入为 B arm 相同 logical slice contributions 在计时外按 Chain FC2 contract 合成的
  full-K expert rows，并验证 permutation map、scale 和最终 Y。

C 的 timed boundary 从 GMEM full-K expert rows/map/scale 开始，到最终 Y，且必须使用
与 actual graph 相同的完整模板实例：BF16 input/output、`ScaleMode::DEFAULT`、bias 配置、
PDL 配置、grid/block 和 predecessor context。它与 `B_component` 共享逻辑 FP32 ledger，
但物理 contribution representation 与合法 rounding contract 不同，所以只报告
component-reference ratio 和绝对时间，不报告严格 paired speedup。

若无法调用 exact compiled launcher，允许实验拥有的 source-exact binding，但必须对模板类型、
launch geometry、compiler flags 和 SASS 机制做 identity gate；否则 R2 降级为 descriptive。

### D. 最小判别 arms（仅在 B 通过 fidelity 后）

按以下顺序执行，不做无界 sweep：

1. `destination_shards={1,4,32}`：REDG 数量、value math、warp mapping 与 task 顺序不变，
   逐步分散 32-way destination collision；后续归并在 timed boundary 外。另加一个
   `span-matched` control：保留 baseline collision group，但把组基址置换到与 shard arm
   相同的分配跨度，用来识别单纯地址跨度/locality 影响。由于“唯一 destination 数量”
   无法在固定 REDG 数与 collision multiplicity 下同时保持不变，剩余 working-set 混杂
   必须结合 dynamic traffic/queue evidence 报告，不能仅凭 latency 归因 contention。
2. `direct_task_grid`：每 CTA 固定一个 task，Scatter body、valid rows、atomic destination
   与 REDG 数量不变；用于判断 persistent task claim/tail/并行映射影响。该 arm 单独报告
   CUDA-event wall 与按实际 grid 计算的 additive service，不沿用 110-CTA denominator。

仅当上述结果仍不能区分问题，才增加下一最小 arm；不得把 no-op、删原子或错误输出当性能候选。

## Fidelity Gate

B 必须同时满足：

1. task count/order、valid rows、slice/output-tile multiplicity、token/weight fixture 相同；
2. 理论与 NCU dynamic REDG/request footprint 闭合；
3. D→E SASS 中保留 source-faithful `LDS + FMUL + pack + REDG`，无意外
   OMMA/TMA/GMEM contribution load、spill/refill；
4. D→F 采用与 A 相同的 `%globaltimer` SM-equivalent denominator，5 replay 稳定；
5. B/A 的差异不超过 fresh paired repeat spread 与 probe perturbation 合成后预先计算的
   acceptance band。禁止先看结果再选择阈值。
6. 锁定 production 的 tile=`128x128x128`、`num_sms=110`、block=160、4+1 warp role、
   dynamic SMEM、register envelope、`max_active_clusters`、fast-math/JIT/compiler flags；
   记录 cubin hash 与关键 SASS。Standalone 的 register/SMEM/occupancy 漂移必须进入 gate，
   不能只按指令形状宣称 source-faithful。
7. output clear 后执行预先固定的 L2 eviction policy，贡献 preload 后不再触碰 output；
   用 A/B 的 L2/DRAM reduction traffic 验证 output/cache 初态是否相容。若不相容则 gate
   失败，不事后挑选 warm/cold policy。

C 也必须通过 actual graph anchor：同 session latency spread、模板/launch/PDL identity、
grid/block、cubin/SASS 与 output correctness 均闭合；否则 R2 只能 descriptive。

Gate 失败时，只报告为何 standalone 不能代表 production；不得用它与 Chain 做 causal ratio。

## 测量与 NCU

- Hardware：同一 5KP GPU UUID、锁定 application clock、无 foreign process。
- Environment：固定 image digest、FlashInfer source、CUTLASS submodule、Python/CUDA/
  CuteDSL/nvcc/ptxas/nvdisasm identity 与 JIT/cubin hash；manifest 同时记录 exact tile、
  `num_sms`、cluster limit、compiler/JIT flags、dynamic SMEM 和 registers/thread。
- Correctness 先于 timing；每次 timed replay 前确保 output 初始状态相同。
- Benchmark：CUDA Graph/event，warmup、repeat、L2 policy 和 order 全部锁定；保存 raw samples。
- Phase：A/B 使用相同 D/E/F `%globaltimer` ABI；报告 body 与 post-sync，禁止将 preload/clear
  算入 Scatter。
- NCU 对 B/C 采最小问题导向集合：
  - dynamic reduction/global load/store requests 与 L2/DRAM traffic；
  - Executed warp instructions、Issue active、Achieved occupancy、eligible/active warps；
  - Long/Short scoreboard、Wait、MIO/LG/Math throttle；
  - LSU/L1TEX/L2 utilization；register/SMEM/stack/spill；
  - SourceCounters/InstructionStats 按 Scatter/Finalize PC 定位。
- 比例类 metrics 保持 launch-local；不把 whole Fused launch NCU 按 phase share 拆分。

## 判定树

1. B fidelity 失败：报告 dataflow/resource/cache 哪项不闭合；在通过前不做 R2 因果结论。
2. C fidelity 失败：R2 降级为 descriptive，不引用约 `223 us` actual finalize 作 paired ratio。
3. B/C fidelity 通过且 B 明显慢于 C：
   - sharding 同时降低 B phase latency 与 atomic stall/queue evidence：contention 支持；
   - direct-grid 降低 latency 且工作/REDG 不变：调度/负载均衡支持；
   - 两个 arm 都改善：结合 span control、dynamic traffic、queue/stall 与 wall/service
     方向判断复合机制；没有正交证据时只写“二者均相关”，不强行选单一根因；
   - 两者无效但 B 的 dynamic reduction work 与 C 的 load/store work差异闭合：
     保留“algorithmic work/memory-interface gap”，继续用 PC-level evidence定位，
     不把全部差值直接称为 REDG latency。
4. B 与 C 接近或 B 更快：不得宣称 atomic 已是 production 主因；优先检查 embedding、
   resource envelope、handoff 和 R2 物理 representation 差异。
5. B 与 C 接近但 A 慢：问题属于 embedding/resource/handoff context，独立 demo不能解释；
   下一步必须是 production 内嵌 counterfactual。
6. latency、traffic、stall/queue 证据互相冲突：结论标记 unresolved，并只提出能区分冲突的
   下一最小实验；不得用解释性文字填平冲突。

任何优化建议必须经过：

```text
observation -> exact PC/data edge -> mechanism -> phase critical-path evidence
            -> controlled counterfactual -> uninstrumented benefit
```

## 产物

```text
exp_010_scatter_vs_chain_finalize/
  plan.md
  <harness/source/build scripts>
  tests/
  results/
    result.md
    manifest.json
    raw/
    derived/
```

脚本输出和报告只放 `results/`。最终 `result.md` 以中文为主，至少展示：

1. A/B fidelity；
2. B/C benchmark 与正确性；
3. 两者 dataflow/work ledger；
4. 精简 NCU 表；
5. 已定位瓶颈、未闭合疑点和下一步。

报告动笔前运行 data-audit；因果瓶颈结论再做一次轻量独立 review。

## Plan Review

- 日期：2026-07-19
- Reviewer：subagent `/root/q0_exact_dataflow`
- Verdict：`✗ Misaligned`（已一次性修正；不进行第二轮 review）
- 已修正的重大 gaps：
  1. 拆开 D→F fidelity 与 GMEM contribution→Y component comparison，取消非法边界比值；
  2. 增加 fresh actual Chain finalize anchor 与 PDL/predecessor identity gate；
  3. 用同一 FP32 slice ledger 分别构造 Fused/Chain 合法 rounding 语义；
  4. 要求 fresh production metadata/task allocation 导出与动态分布 replay；
  5. 锁定 tile、SM 数、warp role、resource/compiler/cubin identity，并审查 cache 初态；
  6. 为 sharding 增加 span control，为 direct-grid 分开 wall/service denominator；
  7. 补齐 B≤C、C fidelity 失败、两 arm 同时改善与证据冲突的判定分支。
