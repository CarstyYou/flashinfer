# exp_017：Latest opt vs SGLang Triton FP8 Phase/Op 耗时与占比

状态：complete — phase/op、current-binary launch-local NCU 与正式 bottleneck report 已闭合；operator-range additive counts 和 phase-local mechanism 明确保留为 missing/open

## 目标

在唯一主 case `M=8192, E=256, H=2048, I_tp=512, topk=8` 上，回答：

> Latest opt CuteDSL fused kernel 距离 SGLang Triton FP8 的 2× 性能目标还差多少，时间分别落在哪些 phase/op，下一步应先闭合哪个问题？

成功门槛是未插桩 Latest opt latency 不超过 `Triton FP8 / 2`。`result.md` 保留紧凑横表，
`opt_vs_triton_fp8_bottleneck.md` 使用现有证据量化目标缺口、区分时间热点与根因，并给出最小下一步。
由于 FP4 fused 与 FP8 chain 的数据类型、物化边界和调度不同，本实验是
`component-reference / descriptive`，不做逐 phase 因果归因，也不把两侧同名行宣称为严格等价工作。

## 锁定对象

| Field | Latest opt CuteDSL | SGLang Triton FP8 |
|---|---|---|
| Entry | `.claude/w4a4_moe_bench/moe_dynamic_kernel_opt.py` | `fused_experts_impl` direct callable |
| Source identity | SHA256 `ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19` | SGLang `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` |
| Precision | BF16 input → NVFP4 fused MoE → BF16 | BF16 input → tensor-scaled E4M3 W8A8 chain → BF16 |
| Execution | one persistent kernel；8 math warps + 1 TMA warp | CUDA Graph chain；两个 Triton GEMM kernel |
| Triton GEMM config | N/A | M8192: `BM64/BN128/BK256, GROUP_SIZE_M=64, 4 warps, 2 stages` |
| Fixture | exp_001 canonical M8192 fixture and routing | 同一 fixture and routing |

两侧必须在同一块 5KP `GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522`、2377 MHz application
clock、无 foreign process的独占 lease 内采集。CuteDSL 与 SGLang 使用各自已锁定容器；跨 runtime
差异保留在 manifest，不假装是相同软件环境。

## 测量方式

### A. Latest opt fused phase

复用 exp_016 已验证的 overlay/dispatch/workspace plumbing，只扩展 `%globaltimer` event ABI；不修改
`moe_dynamic_kernel_opt.py` 本体。建立 matched no-marker control 与 probe specialization。

互斥 reader track：

1. Clear/init
2. Histogram
3. Prefix
4. Route + Q0 + Pack
5. Publish / route tail
6. Claim + cache + task control
7. FC1 Gate/Up + SwiGLU
8. Q1
9. FC2 GEMM + epilogue + R2S
10. Scatter
11. final drain / producer tail / launch skew residual

W8 TMA producer与 consumer 重叠，不强行加入互斥 100%；如保留，只作 non-additive 辅助 track。
主分母沿用 exp_004 的闭合口径：

```text
D = grid_ctas × (max CTA final - min CTA entry)
phase_equivalent_wall = Σ CTA phase interval / grid_ctas
```

对每个 CTA，已记录的有序 interval 先求 union；`CTA residual = CTA entry→final - recorded union`，
显式吸收 zero-task CTA、inter-task gap、最终 no-task claim 与 producer tail。`Launch skew / early-finish`
为 `D - Σ CTA(entry→final)`。二者与已知 phase共同严格闭合到 `D`，禁止丢弃 residual或重复计算。
所有 reader rows必须互斥且严格闭合为 100%。记录 5 个 warmed CUDA Graph replay；每次 replay 前
执行 192 MiB L2 flush，但 flush 不计入 phase 时间。

### B. SGLang Triton FP8 chain

复用 exp_001 的 fixture、weight builder 与 `build_launch()`，新增一个仅负责 capture 的薄脚本。
在 pinned SGLang image 内用：

```text
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
             --cuda-graph-trace=node:host-only
             --capture-range=cudaProfilerApi --capture-range-end=stop
```

capture 5 个 warmed graph replay。每次 replay有唯一 NVTX label；固定执行顺序为
`flush → CUDA sync → NVTX push → graph replay → CUDA sync → NVTX pop`；第 1/5 replay均重新执行
输出 correctness gate，末次 CUDA sync完成后才调用 `cudaProfilerStop`。L2 flush 在 replay range外完成。
只使用 VeloQ `info → summary → graph replay recipe/search/timeline` 读取 `.nsys-rep`，不直接查询
parquet/sqlite。重复出现的两个 `fused_moe_kernel` 依 graph node执行顺序和 node identity 区分 FC1/FC2，
不只依赖相同 kernel name。

Triton reader rows：

1. Route / align / sort
2. Q0：fill + absmax + BF16→E4M3 quant
3. FC1 `fused_moe_kernel`
4. SwiGLU `act_and_mul`
5. Q1：fill + absmax + BF16→E4M3 quant
6. FC2 `fused_moe_kernel`
7. TopK reduce/combine
8. graph-node bubble residual

预期 12-node manifest必须逐 replay按以下顺序闭合：`align → count/sort → Q0 fill → Q0 absmax →
Q0 quant → FC1 → act_and_mul → Q1 fill → Q1 absmax → Q1 quant → FC2 → reduce`。每个 replay的
分母为第一个 MoE node start 到最后一个 MoE node end 的 GPU span。先 gate 同一 CUDA stream且 node
interval无重叠，再以相邻 node gap作为 bubble；若发现重叠，则改用 interval-union归属并禁止
`Σ duration + bubble` 的简单闭合。最终报告 5 replay median及范围。

## 横向展示约定

报告先给两侧各自合法的 phase/op 表，再给一张非对称横向表：

| Logical region | Latest opt phase(s) | Triton op(s) |
|---|---|---|
| Routing/setup | Clear + Histogram + Prefix + Publish | Align + sort |
| Input quant/pack | Route + Q0 + Pack | Q0 quant chain |
| FC1/activation | FC1 Gate/Up + SwiGLU | FC1 + act_and_mul |
| Intermediate quant | Q1 | Q1 quant chain |
| FC2 | FC2 GEMM + epilogue + R2S | FC2 fused_moe |
| Output aggregation | Scatter | TopK reduce/combine |

每格展示 `<side-specific time> / own-side share`，并在列名标出 CuteDSL 是 SM-equivalent、Triton 是
graph elapsed。空缺保持空白，不伪造对齐。行级 raw time只作异构口径并列观察，不计算差值、ratio或
phase speedup；只有双方未插桩 whole-op CUDA-event latency可以并列。唯一允许的正式 whole-op性能
上下文是 exp_001 的同机 benchmark。

## Gates

- 同一 M8192 fixture/occupancy hash、同一 GPU UUID、2377 MHz、无 foreign process。
- 两侧 eager及 graph replay correctness与 dispatch identity通过。
- Capture前锁定 opt/control/probe source、dispatch、wrapper、容器 digest、CUDA/nvcc/ptxas/CuteDSL、
  SGLang/Triton、JIT root、实际 cubin与 graph topology hash；任一 drift fail closed。
- Latest opt control/probe各 5 replay；event coverage、单调性、task/CTA coverage与 denominator closure通过。
- Latest opt probe相对 no-marker CUDA-event median扰动不超过 5%；记录 control/probe REG/SMEM/STACK
  与 exact-cubin static spill。除已知 `%globaltimer`/marker store外，对 normalized SASS opcode/control-flow
  signature、barrier fingerprint和资源做 matched audit；出现额外 codegen/resource drift时降级为
  diagnostic-only，不宣称 production-exact。
- Triton trace包含 5 个完整 replay；每个 replay必须严格出现 12 个预期 CUDA nodes、两个
  `fused_moe_kernel`、且不得出现 CUTLASS/DeepGEMM。
- 两侧各自 5 replay的 phase/op share稳定；任一关键 row范围超过 5% 时报告范围而不隐藏噪声。

## 明确不做

- 不跑六个 M；不做 IKET、PerfSim或全量 Triton 源码分析。NCU 只补采 M8192
  current Latest opt fused main 与 Triton FC1 / FC2 / TopK reduce / SwiGLU 四个 material launches。
- 不修改 Production、Latest opt 或 SGLang实现。
- 不跑无差别 full-set NCU；正式 bottleneck report 对仍缺失的机制 bridge 保持 open，
  不把 whole-launch metric 或 time share 伪装成 phase root cause。
- 只保留用户可读报告、小型 CSV/JSON evidence 和 manifest；原始 `.nsys-rep` 为非提交证据，不复制无关 JIT cache。

## NCU 补采扩展（2026-07-21）

这是同一 exp_017 证据包的扩展，不新建 experiment。一次正确性通过后，只围绕当前
`248.032 us` residual 与 Scatter problem point 补采：

| Relationship | Mode / scope | Subject | Reference | Allowed claim | Prohibited claim |
|---|---|---|---|---|---|
| `exp017-ncu-launch-local` | `component-reference / launch-local` | Latest opt fused main | Triton material launches | 各 launch 自身的 Resource / Schedule / utilization / stall / traffic 现象 | 跨 launch 求和或平均 ratio；把 Fused whole launch 与某个 Triton node 当 phase-matched |
| `exp017-ncu-operator-counts` | `component-reference / complete operator range` | Latest opt one graph replay | Triton one graph replay | 仅对经证明的 compatible additive counts 并列 | 用 range duration 替代 benchmark；从 per-launch replay 相加合成 operator traffic |

固定不变项：exp_001 M8192 fixture/routing/weights，2377 MHz application clock，Latest opt
源码/dispatch/binary，SGLang image/commit/config/topology，数值语义，图重放边界。两侧 NCU 都在同一块
sibling 5KP `GPU-c2ac6efb-f30a-c323-6d38-83908adfb14f` 上补采；未插桩性能与 phase-time 锚点仍来自
`GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522`。NCU duration 不作为 benchmark authority，只比较
规范化 utilization / stall / Resource 等合法 launch-local 指标。跨 backend 差异继续标记
`implementation-confounded`。

### Capture questions and minimum bundles

1. `opt-fused-deep`：Fused main 是否有可观察的 TC/ALU/FMA/XU、memory path、Issue、
   stall/throttle、Achieved occupancy 或 Resource 症状；必须同时采 utilization + stall/throttle + Resource。
2. `scatter-pc`：在 exact loaded cubin 中，Scatter `SMEM/metadata -> scale -> REDG -> sync` 哪个
   PC 具有动态 stall/work 证据；若不能连到 phase/value/producer-consumer，最高只到
   `instruction-localized` 或 `semantic-localized`。
3. `triton-material-deep`：分别保留 FC1、SwiGLU、FC2、TopK reduce 的 launch-local 证据；
   TopK reduce 无适用的 GEMM/activation adapter，保持 descriptive reduction/scatter scope。
4. `operator-additive`：只当 native NCU 返回一个 `range` workload、VeloQ 能合法投影该 range
   的 additive metrics，且每侧恰好一个 complete graph replay 时接受；否则记为 missing，
   不解析 VeloQ sidecar，不从 launch rows 相加。

报告面数据只从 `veloq ncu` JSON 生成；raw `.ncu-rep` 与 JIT/runtime logs 保持 gitignored。
捕获前必须通过 correctness / fixture / source / binary / GPU / clock / foreign-process gates；用完释放
direct-SSH lease。

## 执行顺序与停止条件

1. 生成并 CPU-test opt control/probe overlay。
2. 在租约 GPU 上先做 correctness + 1 replay smoke；失败即停，不继续正式 capture。
3. 完成 opt 5 replay与 Triton NSys 5 replay。
4. VeloQ抽取、闭合审计、生成横向表。
5. 生成面向 2× 目标的 bottleneck report；只把现有证据支持的 hotspot 写入结论，根因不足则转成最小调查。

## 下一步实验约束

### Scatter problem-point localization（现在）

- 问题：current Scatter 的 `SMEM/metadata → scale → REDG → sync` 中，哪个 exact PC/edge 进入关键路径？
- 最小调查：对 current exact binary 做 problem-directed NCU + source/SASS mapping；只有 PC 证据无法区分时才追加 matched diagnostic edge probe。
- 固定项：fixture/routing、source/binary identity、8-warp ownership、数值语义和 benchmark protocol。
- 接受：闭合 `PC/edge → phase/value → producer/consumer mechanism`；未达到 mechanism-localized 不允许 source/schedule 优化。

### Scatter reduction ownership（定位后条件触发）

- 问题：若 REDG ownership/multiplicity 被证明位于关键路径，改变该组合 bundle 是否降低 Scatter 与完整 E2E？
- 最小实验：production-embedded counterfactual；普通 standalone 不作为 fidelity authority。
- 固定项：fixture/routing、FC1/FC2 Tensor work、数值语义、zero-spill 和 paired benchmark protocol；显式记录 unavoidable task/slice/scratch consequences。
- 接受：correctness/zero-spill 通过，Scatter phase 与 fresh paired 未插桩 M8192 E2E 同向改善；结论只归给声明 bundle，不能纯归因于 REDG 数量。

### Route/Q0 residual（Scatter 分支后仍未达标才启动）

- 问题：token-major accepted path 的 `94.038 us` 中，是否存在稳定且可单变量改变的 dominant subphase？
- 最小调查：在 perturbation gate 下细分现有 Route/Q0 reader boundary，不改 production source。
- 接受：dominant subphase 在 5 replay 稳定，并能形成单变量、可保持 correctness/work/resource 的候选；否则停止细分。

### Current-binary NCU（launch-local 已完成；phase-local mechanism 仍 open）

- 已完成：current fused main，以及 Triton FC1、SwiGLU、FC2、TopK reduce 的 problem-directed launch-local NCU；Resource、Schedule、compute/memory utilization、PC-sampling stall/throttle 与 spill evidence 已审计。
- 当前结论：Fused whole launch 呈混合硬件 signature；Scatter 已到 `semantic-localized`，尚未到 `mechanism-localized`。whole-launch utilization 不能拆给 FC1/FC2 phase。
- 明确缺失：complete operator-range additive traffic，以及 Fused FC1/FC2 的 phase-local cadence。不得从 launch-local ratios 或 phase time 伪造这些证据。

最终目标门禁始终是 correctness-qualified 未插桩 M8192 `Latest opt ≤ 1106.701 us`。

## Plan Review

**Date**: 2026-07-20
**Reviewer**: subagent

**Verdict**: ⚠️ Gaps

**Gaps + applied fixes**:

- 两侧 time分母不同：横表明确标注 SM-equivalent vs graph elapsed，仅展示 own-side share；禁止行级
  ratio，只有未插桩 whole-op elapsed可并列。
- phase闭合和 12-node映射不完整：补充 CTA residual/launch skew公式、完整 Triton node manifest及
  overlap fallback。
- graph replay同步与正确性不足：锁定 flush/sync/NVTX/replay/sync顺序，并校验第 1/5 replay输出。
- identity未覆盖实际 binary/topology：补齐 source/runtime/JIT/cubin/graph topology gate，并要求审计
  marker之外的 normalized codegen/resource drift。

## Closure Review

- Data audit：PASS。目标与完整算子数据、Triton op、Opt phase、launch-local NCU 和既有受控实验均可追溯；row-level speedup 与 production-exact phase latency 未被伪造，operator-range traffic 和 phase-local mechanism 明确标为 missing/open。
- Bottleneck conclusion review：PASS。Scatter 371.964 µs / 27.91% 是首要热点；exact-PC 已定位 SMEM load、REDG 与 post-sync 样本，但尚不能区分 REDG completion、SMEM path 或 warp/task imbalance，故只进入下一步证据采集，不直接生成源码优化。
