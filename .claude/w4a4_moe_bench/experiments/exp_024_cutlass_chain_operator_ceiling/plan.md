# exp_024：CUTLASS Chain 逐算子性能上界模型

## 目标

按 KDK `operator-performance-ceiling` 独立分析 CUTLASS BF16-input / online-NVFP4 MoE chain，回答每个 material op 的：

1. 实测时间、占比与 achieved useful/physical rate；
2. 距离可信 hardware ceiling 和 empirical SOTA ceiling 多远；
3. 哪些 ceiling 当前没有 authority，下一步最小校准是什么。

本实验不以 Fused vs Chain 为目标，不消费 Fused phase timing，也不把融合流量收益写入模型。

## Scope

- Case：`E=256, H=2048, I_tp=512, topk=8, SwiGLU`；M256 与 M8192 有完整 Chain timeline/NCU，M1024 只有整体 benchmark。
- Subject：`cutlass_bf16_chain`，边界为 BF16 input → native CUTLASS online quant chain → BF16 output。
- Nodes：`Prefix → Route/Q0/Pack → GEMM metadata → FC1 → SwiGLU/Q1 → FC2 → Finalize`；share 分母使用完整 Chain active union。相邻 launch 的 PDL overlap 单独登记，禁止用残差构造 helper 以强行凑 100%。
- 时间 authority：exp_002 canonical NSys replay；完整算子 benchmark 只作整体 sanity，不替代逐 launch 时间。
- Work authority：source contract 给 logical work；validated NCU counter 给 physical FP4 tensor work。两者是同 canonical rerun 的 paired cross-capture，不声称同一次 replay。

## 输入

- `exp_002_fused_vs_chain_dataflow/results/benchmark_summary.csv`
- `exp_002_fused_vs_chain_dataflow/results/ncu/deep_launch_metrics.json`
- `exp_002_fused_vs_chain_dataflow/results/ncu/operator_traffic_v2.json`
- exp_002 对应 benchmark/NSys/NCU manifests、raw report digest 与 source contract
- `exp_026_5kp_ceiling_calibration/results/profile.json`：5KP exact NVFP4
  full-card measured roof 与 directional physical DRAM roof

Builder 只读消费这些 accepted artifacts；不重跑 benchmark、NSys 或 NCU。每个输入保存路径和 SHA256，身份无法闭合的 cell 标 `unavailable`。

## 模型

| Node | 任务类型 | 主模型 |
|---|---|---|
| Route/Q0/Pack | route + quant + data transform | directional DRAM 百分比；1:1 copy 只作 diagnostic reference |
| FC1 | Tensor Core grouped GEMM | calibrated TC Useful/Executed、directional DRAM、padding；raw rate 下沉 model |
| SwiGLU/Q1 | elementwise + quant | DRAM Read/Write 百分比；不强行套 MFU |
| FC2 | Tensor Core grouped GEMM | calibrated TC Useful/Executed、directional DRAM、padding；ratio 不匹配的 copy 只作 diagnostic |
| Finalize | top-k gather/reduction/store | read-heavy，使用 DRAM Read 百分比；不把 1:1 copy roof 冒充分母 |

主要公式：

```text
logical routed rows = M * topk
FC1 useful FLOPs = rows * 4 * H * I
FC2 useful FLOPs = rows * 2 * H * I
padding efficiency = useful FLOPs / executed FP4 tensor FLOPs
hardware efficiency = achieved rate / compatible hardware roof
SOTA efficiency = matched independent SOTA time / measured time
```

`Tensor pipe active` 只作 utilization 证据，不命名为 MFU。主分母使用 exp_026
同 SKU、exact `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` 的 full-card measured
roof；官方架构缩放只作 nominal diagnostic。没有独立 contract-equivalent
implementation 时，SOTA efficiency 标 `unavailable`，禁止从 CUTLASS 自身时间反向拟合。
读者报告以 ceiling 百分比为主，TFLOP/s、GB/s 和公式输入仅保存在 model。
第 3 章必须覆盖完整 op graph；没有合法 ceiling 的 Prefix/metadata 也保留并标
`unavailable`，不能只展开 GEMM。每行同时给出时间占比、资源达成率和它对下一步优化的含义。

## 验证

- 七类 node 的 interval union 必须闭合到 Chain active union；raw duration/share 因 PDL overlap 可以超过 100%，必须显式登记 overlap 且不得重新归一化；
- `useful work <= executed work`，FC1/FC2 executed sum 必须闭合 operator FP4 tensor ops；
- M256 作为 padding/fixed-cost regime，M8192 作为 steady-state regime，公式冻结不调参；
- 所有百分比检查单位、分母与 `>100%` invalid gate；
- DRAM Read/Write 必须匹配 directional roof；read fraction 在 40%–60% 时也只能显示 1:1 copy diagnostic reference，除非 calibration 的读写比在预先声明的 tolerance 内匹配；禁止取多个不兼容 roof 的 `max`；
- 报告动笔前运行 data-audit。

## 输出

```text
exp_024_cutlass_chain_operator_ceiling/
  plan.md
  build_result.py
  results/
    model.json
    result.md
```

不创建新 profiler artifact、第二份叙事报告、通用 schema 或测试框架。

## Decision

- `accept`：至少一个 hardware 或 SOTA ceiling 有独立 authority，且模型跨 M256/M8192 自洽；
- `revise`：work/time accounting 有效，但 ceiling authority 缺失；明确最小 calibration；
- `reject`：边界、身份或 work scope 无法闭合，或模型依赖循环拟合。

## Plan Review

- Date: 2026-07-21
- Reviewer: isolated subagent `/root/exp024_plan_review`
- Single-round status: 已在旧 scope 下消费；按 gate 约束，scope 重写不触发第二轮 review。
- Retained gates: 跨 artifact identity、logical/physical work 分离、禁止 target-self-fit、MFU authority gate、缺失项保持 unavailable。
- Scope correction: owner 已否决旧的 Fused-vs-Chain storage-edge 模型；本计划删除 Fused arm，仅建立 CUTLASS Chain per-op ceiling card。
