# Experiment 002 Plan: Fused-vs-Chain Operator Dataflow

Status: **complete (2026-07-16)**. Correctness-qualified benchmark、NSys、operator-range NCU、
selected-launch NCU 与 current-binary SASS audit 已完成；判定和下一实验见
[`results/operator_dataflow_bottleneck.md`](results/operator_dataflow_bottleneck.md)。

## Goal

在 SM120 Qwen3.5 prefill MoE case 中，对比单-launch CuteDSL W4A4 与 CUDA Graph
中的 CUTLASS multi-kernel chain，找到一项有证据、可验证的优化点；同时用真实场景验证
KDK `operator-dataflow-bottleneck` 是否能正确处理 fused-vs-chain，而不是只生成 profiler
摘要。

本实验不重新回答 exp_001 的三 backend 性能排名，也不以拆出每个 phase 的精确耗时为目标。

## Questions

1. CUTLASS 在 CUDA Graph replay 中实际执行哪些 nodes，它们的顺序、gap、overlap、route 与
   finalize branch 是什么？
2. CuteDSL 消除的 launch/GMEM storage edges，是否被 fused node 内的 TensorCore cadence、
   resource pressure、synchronization、producer/consumer idle 或 tail 抵消？
3. loss 与 control case 的差异能否定位到一项具体代码动作和一个可接受/推翻它的指标？

## Fixed Case Contract

- Hardware：一张 SM120 5KP；固定 GPU UUID
  `GPU-4a286357-c999-9547-3a04-25961b1ffd08`。
- Container：`nvcr.io/nvidia/pytorch:26.05-py3`，固定 image digest
  `sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba`。
- Python dependency overlay：由于上述 base image 缺少 `tvm_ffi` 且自带 Cutlass DSL 4.4.2，
  使用只读 CUTLASS DSL 4.6.0 overlay，固定内容树 SHA-256
  `32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74`，并记录实际
  `cutlass`/`tvm_ffi` module path 与 distribution version；benchmark/profile 必须完全一致。
- Source：从 FlashInfer `074d93e4aa54c75bee1b3dfdb39b7f075a3ff2af` 建立独立 clean
  checkout；CUTLASS 固定为 `b46b16d003484063bca4ed365e44095c4c6ed633`。实验脚本作为
  overlay 单独记录 diff/SHA-256，不改 production kernel。
- Shape：`E=256, H=2048, I_tp=512, topk=8, SwiGLU, BF16 output`。
- Cases：`M=8192` 为大prefill primary case，`M=256` 为小prefill control；`M=1024`
  只做 fresh benchmark neutral probe，不强制进入 profiler。
- Router logits、softmax 与 top-k 在 boundary 外；两边接收同一 `topk_ids/topk_weights`。
- 不增加独立 `b12x` backend arm，不测 Triton，不修改 production kernel；
  `cutedsl_bf16_fused` 沿用 exp_001 的 `B12xMoEWrapper` 实现。

使用独立 `FLASHINFER_WORKSPACE_BASE`，不设置 tactic/autotune/blocklist override。固定并记录
`enable_pdl=True`、`use_fused_finalize=True`、显式 graph capture stream、CUDA Graph node mode、
环境变量、JIT `.so`/generated source SHA-256。每个 `arm × M` 在 benchmark、NSys、NCU 三处
必须具有相同的 demangled kernel、grid/block、node multiplicity 与 tactic identity；任一漂移都
停止映射并重新 capture，不能把不同 dispatch 的证据拼在一起。

benchmark结束后分别生成 content-addressed、append-only 的 environment/toolchain、measurement protocol 和
JIT artifact locks，并由 `results/evidence.identity.json` 指向它们。稳定的 `comparison_group_id` 只说明比较
关系；每次完整重测必须使用唯一 `rerun_id`，且每个repeat包含两个arms。raw row同时绑定shared environment、
protocol和per-arm artifact fingerprint。任一身份变化时correctness、paired benchmark、绑定旧binary的NSys/NCU、
derived report与verdict全部stale，禁止单边替换或覆盖旧lock。

## Shared Fixture and Correctness Gate

每个 M 使用同一份 BF16 input、routing ids/weights，以及同一份 packed FP4 W1/W2 和逻辑
scale。两种 backend layout 必须从同一 canonical tensor 派生，不再使用 exp_001 中
CuteDSL `seed` 与 CUTLASS `seed+17` 的独立权重。

fixture 使用确定性的 synthetic random-logit router，与 exp_001 的分布意图一致，但不代表真实
Qwen traffic；结论严格限定为该 fixture。记录：

- input、routing、packed W1/W2、logical scale 和 backend-derived scale layout 的 SHA-256；
- seed、shape、dtype、layout 与转换函数；
- 每 expert token histogram 的 min/p50/p95/max、zero-token expert 数；每 token 的 duplicate
  expert 数必须为 0，top-k weights 必须 finite、nonnegative 且 sum-to-one；
- `M=256` 预期尝试 fused-route branch，`M=8192` 预期 three-step/large-prefix branch；这只是
  source prediction，最终以 NSys observed nodes 为准；
- output shape/dtype/finite/nonzero、cosine、relative-L2、max/mean absolute error 和
  percent-within；
- 用同一 canonical BF16 input/weight/routing 构建独立 dequantized BF16/FP32 reference MoE，
  reference 明确包含 input quantize/dequantize 与 FC2 前 activation quantize/dequantize；
- 每个 paired arm 必须分别通过同一个 formal gate：沿用仓库 Qwen-like CuteDSL/B12x MoE
  quant-aware criterion，`atol=max(0.05, 1.5 × oracle.std())`、`rtol=0.5`，至少 97% 元素满足
  `abs_error < atol OR relative_error < rtol`。CUTLASS 小 shape 使用的严格
  `torch.testing.assert_close(rtol=0.2, atol=0.2)`通过率、cross-backend cosine/L2 和 max/mean
  error 同时记录为诊断，但不能替代共同 oracle gate。

只有同一 fixture 的两个 arm 均独立通过 oracle gate，才允许 BF16 full-operator paired 结论。
atomic/finalize reduction 不要求 bitwise；correctness 失败时只允许分别描述 topology，不得写
speedup 或 fusion tax 归因。这样即使两个 backend 共享同一种错误，也不会互相“证明正确”。

## Arms and Comparison Modes

| Arm | Timed boundary | Role | Gate |
|---|---|---|---|
| `cutedsl_bf16_fused` | BF16 input → one fused launch → BF16 output | target | paired candidate |
| `cutlass_bf16_chain` | BF16 input → native CUTLASS online quant/route/FC1/activation/FC2/finalize graph nodes → BF16 output | matched baseline | paired candidate |

所有 arms 使用 outer CUDA Graph replay。CUDA events 放在 graph 内；192 MiB L2 flush 在每次
replay 前、timed interval 外。graph capture、JIT、tactic preparation、allocation 与 first
warmup 均在 timing 外。

## Phase 1: Fresh Benchmark

- 先 eager warmup/correctness，再 capture 每个 arm 的 graph。
- `M={256,1024,8192}`：5 warmups、50 iterations、5 repeated samples。
- repeat 内交替 `A→B` 与 `B→A`，保留原始 samples 与实际运行顺序。
- uninstrumented CUDA-event median 是 operator performance authority；profiler duration 只用于解释。
- 样本 spread 超过 5%、GPU 有 foreign compute process、dispatch 漂移或 correctness 失败时停止
  formal comparison。

Fresh benchmark 决定 profiler 角色，不强迫复现 exp_001 符号：

- 若 `cutedsl_bf16_fused` 在 `M=8192` 仍慢于 matched chain，作为 primary loss；
- `M=256` 作为 fixed/setup control；`M=1024` 只有在差值落入 paired spread 时才称 neutral。

## Phase 2: NSys Capture and VeloQ Query

`M=256` 与 `M=8192` 的两个 arms 都必须使用同一 harness 的 `single-replay` mode capture；
`M=1024` 仍为 benchmark-only neutral probe。warmup 后只 capture 一个带唯一 NVTX 名称的
CUDA Graph replay，并由紧贴该 range 的 `cudaProfilerStart/Stop` 触发 collection。Capture：

- NSys node mode：`--cuda-graph-trace=node:host-only`；
- 最小 trace：CUDA runtime、NVTX、OS runtime；不默认开启 CPU sampling 或 GPU metrics；
- 每个 capture 写入全新 `results/nsys/<case>/<arm>/`，保留 `.nsys-rep`、capture manifest、
  stdout/stderr 与 SHA-256；
- 首先检查 VeloQ `info`、`summary` capability bits，再使用 graph replay recipe、`stats`、
  `search --with-nvtx`、`timeline`、`gaps`、`concurrency`、`inspect/correlate`；保存 exact command、
  filters、row ids、window 与 JSON 输出。

NSys 要确认：

- fused arm 是否确为一个 material kernel；
- CUTLASS 的 quant、route、pack、GEMM1、activation/requant、GEMM2、memset/finalize nodes 与
  launch multiplicity；
- observed ordering、stream、gap 与 overlap。只有 trace 提供 dependency evidence 时才使用
  `critical path`；否则只写 observed ordered timeline。

Operator wall time 使用 benchmark/NSys declared-window interval union，不使用 kernel duration sum。
由于两边已用 CUDA Graph，不把 eager Python launch overhead 当作 fusion benefit。

## Phase 3: Targeted NCU

NSys 后再选择 NCU targets；NCU 固定 `--graph-profiling=node`：

- CuteDSL：每个 material case 的 fused launch；
- CUTLASS：quant、GEMM1、activation/requant、GEMM2，以及占 chain active-time interval union
  ≥5% 或控制 material storage edge 的 route/finalize node；
- 小 helper 不因名字像 phase 就全部 profile；缺少适配器的 non-GEMM node 保持 descriptive。

两个 M 的 fused launch 都必须 capture。CUTLASS 必须覆盖每个 distinct kernel instance class，
直到已选 nodes 的 active-time interval union 达到 chain kernel-active union 的 ≥90%；quant、
expand/permute、GEMM1、activation/requant、GEMM2、finalize 若实际存在则不受 5% 阈值豁免。
报告必须量化未覆盖 active-time union，不能把未 profile 的 class 归入 residual phase。

每个 target 单独 capture 一个 launch，记录 exact NSys row ↔ NCU report mapping、kernel name、
grid/block、graph/node mode、launch skip/count、tool versions 与 report hash。先取 duration、launch、
occupancy/resource、DRAM/L2/L1/shared traffic、TensorCore issue、Issue Active、IPC、eligible warps、
stalls、spills/local traffic 与 atomics；missing metric 记为 unknown。

Operator-level traffic 另用一次完整 graph range capture：`--replay-mode app-range --cache-control all`，
由既有 `cudaProfilerStart/Stop` 把一次 graph replay 定义为一个 range，直接读取该 range 的 DRAM、L2、
global 与 local totals。不得把默认 cache-control 下逐 node kernel-replay 的 DRAM bytes 相加，因为逐 node
cache reset/replay 会破坏 chain 内真实 cache state。Node report 只用于 launch-local resource、spill、cadence
与 source/SASS 归因。若当前 VeloQ 版本只能枚举 range 而不能投影 range metric，保留其 range identity JSON，
再使用同版本官方 `ncu --import --csv --page raw` 导出 counters，并显式记录该 fallback。

SM120/NCU 2026.1.1 已预先枚举可用 cadence counters。FP4 TensorCore work 的 primary 是
`sm__inst_executed_pipe_tensor_subpipe_hmma`、
`sm__ops_path_tensor_src_fp4_dst_fp32`，cadence/activity 是
`sm__pipe_tensor_subpipe_hmma_cycles_active`，并与 active cycles、Issue Active/eligible warps 一起
归一化。若 selected report 未暴露这些 counters，则 fallback 为
`sm__inst_executed_pipe_tensor` + `sm__pipe_tensor_cycles_active` + correlated SASS 的实际 QMMA
instruction class；若 primary 与 fallback 都不可用，则禁止提出“改善 TC cadence”的优化候选。

NCU durations 不相加重建 operator wall time；SOL、occupancy、IPC、stall% 不跨 launches 平均。
可相加的 bytes/instructions 只有在 scope、multiplicity 和 replay 语义确认一致后才 roll up。

本实验强制输出 fused-vs-chain 的三条 fusion 机制账本：

1. **GMEM elimination**：在相同 operator boundary 下，比较完整 graph range 的
   `DRAM read bytes + DRAM write bytes`，并分层列出 global request、L2 与 local/spill traffic；同时列出
   每个被消除或仍存在的中间 tensor materialize/reload edge。Node-level counters只解释流量来自哪里，
   不再承担 operator DRAM rollup。L2 hit 或 cache traffic不能代替 DRAM byte total，range 不完整时不得
   暗算为 0。
2. **Launch/scheduling elimination**：从 NSys 比较 graph node count、launch/gap interval union、
   dependency handoff 和 operator wall interval。两边都使用 CUDA Graph，因此只报告仍实际存在的
   GPU node/handoff 成本，不借用 eager Python launch overhead。
3. **Latency hiding/overlap**：用 TC/CUDA-core/TMA/LDST issue/activity、eligible warps、barrier/wait
   与 IKET（必要时）判断 fused 内不同工作是否真正重叠或互相填补 latency；source 中“融合”本身
   不是 overlap 证据。

与三项收益并列记录 fusion tax：新增 registers/SMEM、occupancy/residency 下降、TC cadence 被
scalar/memory phase 打断、barrier/role idle、原本可独立调度阶段被串行化，以及 tail 增长。

## Phase 4: Optional IKET

只有 NSys + NCU 无法区分 fused launch 内下列问题时才启动 IKET：

- FC1/activation-requant/FC2 handoff 是否造成 producer/consumer idle；
- fixed route/setup 与 steady compute 的 overlap；
- `M=8192` 相对 `M=256` 的 warp/CTA skew 或 tail。

IKET 必须匹配 source/binary/dispatch 和 case，优先覆盖 `M=8192` 与 `M=256`。instrumented
duration 不替代 production timing，不按 phase share 拆分 whole-launch NCU counters，也不把
selected CTA 推广为 full grid。

## Logical Phase and Storage-Edge Model

统一映射：

`initialize → route/prefix → input quantize/pack → task publish/claim → FC1 gate/up →`
`SwiGLU + requant → FC2/down → scatter/reduce/output → tail/handoff`

每条 phase mapping 标记 `exact / contains-extra / partial / overlapped / unresolved`。每条 edge
记录 producer/consumer、GMEM/SMEM/RMEM、materialization/reload、launch、barrier/atomic 与 ownership。

## Optimization Decision

最终不是选择“最快 backend”，而是选择一项证据最强的优化点：

- edge candidate：减少一个已测得的 GMEM materialization/reload、graph node 或外部同步；
- node candidate：改善 fused kernel 内已定位的 TC cadence、resource tier、barrier/role idle、
  atomic 或 tail；
- 每项 candidate 必须写明 source action、预期 dataflow 变化、接受/推翻指标和仍存替代解释。

优先级先按 `GMEM elimination → launch/scheduling elimination → measured latency hiding` 检查实际
收益，再与 resource/cadence/synchronization/tail tax 对账；不预设 fusion 必然更快。

若现有证据只能定位 phase 而不能支持动作，结论必须是 `尚不能选择优化点`，并只保留能改变
该判断的最小下一实验。不得把更长的 profiler 摘要当作完成。

判定分支固定如下：

- setup identity 或任一 oracle gate 失败：formal paired result invalid；只保留 capability/topology；
- `M=256` 相对 exp_001 符号改变：把它当作 regime-control 结果，不能继续称为固定“win”；
- native BF16 CUTLASS graph 不支持：paired comparison为inconclusive，不用prequantized输入替换；
- route/finalize branch 在不同 M 间变化是 workload-regime 证据；同一 `arm × M` 在 benchmark
  与 profiler 间变化则是 dispatch drift，证据作废并重采；
- fused-vs-native-BF16 是唯一 whole-operator paired comparison。

## KDK Shadow Validation

冻结待验证 KDK snapshot：clean HEAD
`84d72d19d207cf0bebdc70a88b272d2c27d6c5a0`。先用独立手工 oracle
完成 topology/roll-up 判定，再让 KDK `operator-dataflow-bottleneck` 能力处理同一证据，
避免循环自证。pass/fail assertions 是：

1. 拒绝混用不同compiler/JIT fingerprint或单边重测的performance evidence；
2. 从 graph-node timeline 发现实际 chain，而不是照抄源码预期；
3. 保留 phase many-to-many、storage edge 和 overlap，不制造 additive phase time；
4. 只 roll up additive counts，不平均 per-launch ratios；
5. 最终帮助选择优化点或明确指出缺少哪项最小证据。

其中 1–4 的手工 oracle 固定为：environment/JIT fingerprint不一致必须拒绝paired结论；source prediction 与 NSys
不同时必须接受 NSys；operator wall time 必须使用 declared-window/active interval union 而不是
kernel sum；ratio/percent 不得跨 launch 平均。第 5 项必须输出 `source action + discriminator`，
或明确 `insufficient evidence + 最小下一实验`。逐项记录 observed、expected、pass/fail 与偏差。

只把真实使用中减少误判/返工的规则保留回 KDK；项目命令、host、case 数据与结论留在本实验。

## Artifacts

脚本保存在 experiment root；所有生成物与报告保存在 `results/`：

```text
exp_002_fused_vs_chain_dataflow/
├── plan.md
├── fixture.py
├── run_exp002.py
├── build_result.py
└── results/
    ├── fixtures/
    ├── correctness.json
    ├── benchmark_raw.csv
    ├── benchmark_summary.csv
    ├── nsys/
    ├── ncu/
    ├── manifests/
    └── operator_dataflow_bottleneck.md
```

Raw profiler artifacts不可覆盖；派生 VeloQ cache 不是第二 evidence source。

## Stop Conditions

- shared fixture/correctness、GPU identity、source/dispatch 或 graph boundary 任一不成立：停止
  formal paired analysis；
- BF16 CUTLASS path 无法 graph capture：记录capability failure并停止paired comparison，不用其他boundary替换；
- NSys 无 graph-node/kernel/NVTX capability：recapture，不从缺失表推断没有 nodes；
- NCU capture 改变 dispatch、超时或 replay 不可控：保留 NSys + source bounded conclusion，
  不制造 per-launch counter；
- 当前证据已支持一个可验证优化点时停止扩展工具；PerfSim/DRPI/PerfBot 不默认进入本实验。

## Plan Review

- Date：2026-07-15
- Reviewer：独立 subagent `/root/exp002_single_round_review`
- Verdict：⚠️ Gaps
- Single-round findings and resolutions：
  1. setup/dispatch 身份不够严格 → 固定 source/submodule/JIT/tactic/graph stream，并规定
     `arm × M` topology drift 为 stop condition；
  2. routing fixture 缺少分布与 branch contract → 增加 histogram、zero expert、duplicate、
     normalization、预期 branch 与 synthetic-scope 限制；
  3. cross-backend check 可能共享错误 → 两个 arms 分别对独立 dequantized oracle 过同一
     Qwen-like magnitude-scaled gate；
  4. profiler coverage 可能漏 control/instance → NSys 强制两个 M×两个 arms，NCU 覆盖 mandatory
     phases 与 ≥90% active-time union；
  5. sign flip/capability/branch failure 没有判定树 → 增加明确 allowed claim/invalid 分支；
  6. TC cadence 指标未预确认 → 在目标 5KP 用 NCU 2026.1.1 枚举 primary/fallback，并禁止无
     counter 时提出该候选；
  7. KDK 可能循环自证 → 冻结 KDK snapshot，先做独立 topology/roll-up oracle，再逐项 shadow
     validation。

按 `experiment-plan-review` 约束，本实验只做这一轮 plan review；上述 gaps 已一次性回写，不再
重复 review。
