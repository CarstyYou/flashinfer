# exp_025：SGLang Triton FP8 Chain 逐算子性能上界模型

## 目标

按 KDK `operator-performance-ceiling` 独立分析 SGLang Triton FP8 MoE chain，给出各 op 的时间/占比、achieved rate、hardware ceiling distance 与 empirical SOTA distance。它不是 Opt-vs-Triton 对比报告。

## Scope

- Case：`M=8192, E=256, H=2048, I_tp=512, topk=8, SwiGLU`；现有 accepted per-op timeline 只覆盖 M8192。
- Subject：`sglang_triton_fp8`，边界从既有 top-k ids/weights + BF16 activation 到 BF16 MoE output；router logits/softmax/top-k selection 不在边界内。
- Material nodes：`Q0 → FC1 → SwiGLU → Q1 → FC2 → TopK reduce/finalize`；routing/scheduler 与 residual/helper 保留在时间闭合中。
- Reader reporting：完整 graph 保留在第 2 章 accounting；第 3 章只展开 M8192 share ≥3% 的 FC1、SwiGLU、FC2、TopK reduce。
- 时间 authority：exp_017 canonical Triton NSys five-replay topology；NCU replay time不参与时间或 MFU分母。
- Work authority：source/shape contract 给 logical work；只有通过 launch identity 的动态 counter 才能给 physical work。

## 输入

- `exp_017_opt_vs_triton_phase_share/results/{evidence.json,phase_op.csv,triton_capture_manifest.json,triton_artifact.lock.json}`
- `exp_017_opt_vs_triton_phase_share/results/veloq/triton_graph_replays.json`：唯一逐 node 时间输入，必须核对 ordinal 与 capture digest；`phase_op.csv` 只作派生横表 sanity
- `exp_017_opt_vs_triton_phase_share/results/ncu_evidence.json` 及其 referenced raw NCU reports
- `exp_018_triton_opt_eric_benchmark/results/benchmark_summary.csv` 仅作整体 uninstrumented sanity
- Triton FP8 backend source与dispatch contract
- `exp_026_5kp_ceiling_calibration/results/profile.json`：同一 GPU UUID 上 exact
  FP8 no-scale full-card measured roof

首轮只复用既有 accepted evidence，不启动新 benchmark/NSys/NCU。若缺少 M256 phase timing或兼容 roof，结果标 `unavailable/validation-limited`，不为填表重采。

## 模型

| Node | 任务类型 | 主模型 |
|---|---|---|
| Q0 | BF16→FP8 quant/pack | logical values/s、payload GB/s |
| FC1 | FP8 Tensor Core grouped GEMM | useful/executed TFLOP/s、padding efficiency、MFU authority gate |
| SwiGLU | elementwise ALU/SFU | elements/s；仅有兼容 ALU/SFU 或 bandwidth roof 时给效率 |
| Q1 | BF16→FP8 quant/pack | logical values/s、payload GB/s |
| FC2 | FP8 Tensor Core grouped GEMM | useful/executed TFLOP/s、padding efficiency、MFU authority gate |
| TopK reduce/finalize | gather/reduction/store | output elements/s、payload GB/s、matched reduction SOTA gate |

通用 shape 公式与 exp_024 相同，但 FP8 precision、kernel padding、physical work 与 hardware peak 独立登记；禁止把 NVFP4 ceiling 或 CUTLASS timing当作 Triton FP8 的 MFU/SOTA authority。

Physical routed rows 允许由 fixture 的 per-expert occupancy、pinned JIT `BLOCK_SIZE_M`、grid/predication contract 独立推导，并以 fixture/source/config/launch identity gate 验证；不能证明时才 unavailable，不为此补 NCU。

实施前冻结 ceiling registry：

| Scope | Ceiling | Authority / status |
|---|---|---|
| FC1/FC2 dense FP8 | exp_026 exact `QMMA.16832.F32.E4M3.E4M3` full-card measured roof | target 与 calibration GPU UUID、application clock 一致；target instruction 仅由 source/dispatch contract 推导，缺 SASS binding，因此百分比标 diagnostic |
| Q0/Q1 | compatible quant/transform roof | unavailable；logical payload 不除 DRAM peak |
| SwiGLU | NCU physical DRAM saturation | launch-local diagnostic；不是完整 elementwise roof |
| TopK reduce | NCU physical DRAM saturation | launch-local diagnostic；不是 reduction SOTA roof |
| all nodes | independent empirical SOTA | unavailable；target 自身和不同 precision backend均不接受 |

## 验证

- timeline material nodes + routing/helper/residual 必须闭合到 Triton graph wall；
- source-derived useful work 与 observed physical work分列，缺 physical work 不推测 Executed MFU；
- exp_026 exact-mode full-card roof 是主分母；目标 instruction 未保存可追溯 SASS
  binding 时只能称 source-contract-compatible estimate；official same-architecture FP8 rate只形成
  `architecture-derived nominal peak`，TC active 不是 MFU；sibling-GPU
  `l1tex__cycles_elapsed.sum` 只作诊断，不参与主百分比；
- NSys/NCU cross-capture 必须核对 fixture SHA、SGLang/container/toolchain、ordered dispatch、逐 kernel JIT/cubin identity、report hash与 stale rule；NCU在 sibling GPU，无法闭合的项只作 normalized launch-local diagnostic；
- SOTA anchor 必须独立且 contract/precision/shape/protocol equivalent；Opt NVFP4不是有效绝对 SOTA anchor；
- 只有一个 M regime 时，模型 verdict 至少降级为 `validation-limited`；
- reader ceiling threshold 固定为 M8192 share ≥3%；低于阈值的真实 op 与 graph bubble 仍保留时间 accounting，不生成 ceiling/优化任务；
- 报告动笔前运行 data-audit。

## 输出

```text
exp_025_triton_fp8_operator_ceiling/
  plan.md
  build_result.py
  results/
    model.json
    result.md
```

不创建新 profiler artifact、第二份 bottleneck 报告、通用 schema 或测试框架。
读者报告以 ceiling 百分比为主；TFLOP/s、cycle proxy 和公式输入下沉 model。
主表按 op 同行展示 share、TC/DRAM diagnostic、padding 与优化含义；不再单设 GEMM-only 或完整 NCU reader table。

## Decision

- `accounting=vetted`：M8192 时间/work/identity闭合；`validation=limited`：只有一个 per-op regime，不阻塞本轮停止；
- `revise`：accounting 有效，但 exact/calibrated ceiling、SOTA或跨 regime验证缺失；
- `reject`：边界/身份/work 无法闭合，或混用不同 precision/backend 作为 ceiling。

## Plan Review

- Date: 2026-07-22
- Reviewer: isolated subagent `/root/exp022_design_probe`（new independent task）
- Verdict: ⚠️ gaps；本轮已一次性修正，禁止重复 review。
- Fixes: 逐 node 时间改由 graph replay ordinal/digest 驱动；physical work允许经身份门禁的 dispatch-derived padding；冻结 ceiling registry；补齐 NSys/NCU/JIT cross-capture identity与 stale gate；拆开 accounting status 和 cross-regime validation status。
