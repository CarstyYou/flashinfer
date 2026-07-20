# exp_015：Phase Skeleton Refactor Regression

## 状态

已收口：一次性 Plan Review、CuteDSL 编译、8/8 correctness、static/dynamic resource gate
与 M256/M8192 ABBA 均通过，接受 Candidate v2 作为后续优化 baseline。

## 目标

验证 `moe_dynamic_kernel_opt.py` 的 phase/device-function 重构是否保持原有 correctness、任务工作量、
zero-spill、资源与性能，同时让 kernel steady-state 显式展示 phase 及同步依赖。验证完成前不集成任何
Scatter 优化。

## 对比对象

| Arm | Source | SHA-256 |
|---|---|---|
| Baseline | exp_008 accepted immutable overlay `branch_paired_n64_v1/moe_dynamic_kernel.py` | `f3c246817679d962a3f7160dbe8b9e68262c919e26e306f349200961fc4ac971` |
| Candidate v1（已冻结、未执行） | `results/overlays/candidate_v1/moe_dynamic_kernel.py` | `561cd40a2f5d5015eb397b3b80c8123549235dfa6473dabdf5a5745886126a01` |
| Candidate v2（最终验证对象） | `.claude/w4a4_moe_bench/moe_dynamic_kernel_opt.py` | `b6e141179794561f2144bdec079b7e109fddfccc2db6ba0f19c22d30ea4b34ca` |

Baseline 与本次修改前的 Git blob byte-identical。Candidate 只允许代码组织变化：不得改变 scheduler、
tile、warp role、pipeline stage、SMEM layout、barrier participant、计算或 Scatter dataflow。

各版本 source 先复制到 `results/overlays/{baseline,candidate_v1,candidate_v2}/` 并冻结 hash/diff。每个 arm 使用独立进程和
fresh JIT root。若 Candidate 编译或验证失败后需要改源码，必须建立 `candidate_v2+`；最终被判定的版本从
compile 到 performance 全量 fresh 重跑，禁止跨版本拼接证据。

## Candidate 结构

```text
initialize_route_q0_and_publish
→ persistent task loop
  → claim_and_cache_task
  → Math warps:
      fc1_gate_up_swiglu_to_sC
      → fence + epilog_sync
      → quantize_q1_sC_to_sA_sSFA
      → fence + epilog_sync
      → load_fc2_a_fragments
      → for each FC2 output tile:
          fc2_to_sC
          → fence + epilog_sync
          → scatter_sC_to_gmem
          → epilog_sync
      → pass_final arrive
  → TMA warp:
      load_fc1_tma_slice
      → pass_gate wait
      → load_fc2_tma_slice
      → pass_final wait
→ producer tail
```

Gate/Up accumulators与即时 SwiGLU 必须在同一 helper 内产生并消亡。最后一轮 FC1 stages release 后的
`pass_gate arrive` 保持在 SwiGLU 之前，以保留 FC2 prefetch overlap。所有会 advance 的 pipeline
state 必须显式 return 并由 caller 重绑定。FC2/Scatter 两段切分锁定 full N128 epilogue，
`epi_rest_m == 1`；未来若恢复 compact M64 epilogue，必须重新设计逐 epi 的 R2S→同步→Scatter 顺序，不能直接复用
当前两段边界。

## 固定环境

- Hardware：5K Pro，`NVIDIA Graphics Device`，SM120，110 SM；优先复用 exp_008 的 GPU UUID
  `GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522`。
- Node/container：`R6KD-CX8aaS-GPU-16` 的 `xiyTrtllm5kp`，镜像
  `nvcr.io/nvidia/pytorch:26.05-py3`，image ID
  `sha256:a4e056e1d34a5cc9387512ffa3abeed778e3dc7966633c5154d771705d8835ac`，registry digest
  `sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba`；所有 Python、编译、
  benchmark 只在容器内运行。
- Baseline/Candidate 使用同一 GPU、container、toolchain、fixture/weight seeds、launch geometry、
  CUDA Graph boundary 和 cache-flush protocol；记录 driver、CUDA、nvcc、PyTorch、CuteDSL/CUTLASS、
  FlashInfer/CUTLASS commits、source/cubin/SASS hashes。
- 编译关闭 IKET、phase marker 与 compiler overlay；`CUTE_DSL_KEEP=ir,ptx,cubin,sass`。

## 验证顺序

### 1. Compile gate

先只编译最终 Candidate v2 的 `M=256 canonical`。编译失败立即停止，不进入 benchmark；修复必须生成新版本。
随后在完全相同环境 fresh 编译 Baseline。完成 correctness cases 后，逐 case 枚举实际 loaded source、完整
kernel symbol、CUDA Graph node、grid=`(1,1,110)`、block=`(288,1,1)` 与 cubin hash；发现 distinct cubin
时，对该 cubin 重复全部 static/dynamic resource gate。

### 2. Correctness 与 work identity

两臂使用相同输入，覆盖：

- `M=256`：`canonical`、`sparse_empty`、`exact_128`、`tail_129`、`hot_expert`、
  `canary_gate_v2`、`canary_up_v2`；
- `M=8192`：`canonical`。

每个 case 校验共同 FP32 reference、两次 replay stability、route/task workspace invariants。数值门禁原样继承
exp_008：`cosine_loss floor/cap=1e-6/1e-4`、`relative_l2=0.005/0.03`、
`max_abs=0.02/0.10`、`token_rel_l2_p99=0.01/0.05`；逐指标阈值为
`min(cap, max(floor, 3 × baseline self-drift))`，同时约束 Candidate self-drift 与 cross-arm error。
Candidate 必须 8/8 通过，且与 Baseline 的 task count、expert/m-tile/slice descriptor multiset、terminal queue
state 一致。output NaN sentinel 不作为写覆盖证据；另用“所有目标位置期望非零且可区分”的 write canary
检查漏写，scheduler exact-once 仍由 exp_008 work ledger 的 descriptor multiset + terminal head 受约束推断。

### 3. Static codegen/resource gate

对 exact loaded cubin 检查：

- `REG <= 160`、`STACK=0`、`LOCAL=0`、SASS `LDL=0`、`STL=0`；
- dynamic SMEM struct extent=`83,968 B`、总 SMEM/CTA=`84,992 B`；
- OMMA family不变，静态 OMMA count=`448`；
- grid/block、pipeline/barrier 数量与 participant、关键 global atomic/load/store families 不缺失。
- machine-check source/AST：所有会 advance 的 pipeline state 均 return+rebind；Gate/Up accumulator不跨 helper；
  SwiGLU 紧随对应 accumulator；`pass_gate arrive`、两个 fence/epilog barrier、`pass_final arrive` 顺序及参与者不变。
- 比较 IR/SASS call graph，helper 必须 inline，Candidate 不得比 Baseline 新增 `CALL/RET` frame。

对每个 distinct cubin 采 matched NCU：executed Tensor instructions、FP4 Tensor ops，以及
`sass__inst_executed_register_spilling_op_{read,write}`、
`sass__inst_executed_register_spilling_mem_local_op_{read,write}`。Tensor work 必须与 Baseline/exp_008 ledger
一致，四项 dynamic spill 必须为 0；missing/`n/a` fail closed。

若资源或工作身份漂移，Candidate reject；不得用 benchmark 结果覆盖 identity failure。

### 4. Performance regression gate

只测 `M=256` 与 `M=8192 canonical` 的未插桩 full fused MoE CUDA Graph latency。锁 graphics clock，
每次 192 MiB L2 flush，warmup=5、timed=50，A-B-B-A × 5 groups，每个 position 独立进程。每个 arm
复用一个已通过 correctness 且 hash 冻结的 JIT root；每个 position 前后必须验证 artifact-set/cubin hash 不变，
不得重新编译或跨 arm 共享 JIT root。
报告 raw samples、p50、p10/p90、paired ratio 与以完整 ABBA group 为单位的 10k bootstrap 95% CI。
预注册 no-regression 边界为 slowdown 1.5%：CI 完全越过 -1.5% 则 reject；CI 跨越 -1.5% 为
inconclusive，不得自动 pass；CI 完全位于边界内才接受 no measurable regression，不宣称重构带来优化。

## 判定

- 所有 correctness/work/static/perf gates 通过：接受重构，作为后续 Scatter 优化的新 baseline。
- correctness 或 work identity 失败：立即 reject 并恢复到 exp_008 baseline。
- zero-spill/SMEM/OMMA identity 失败：reject，定位 helper boundary 对 codegen 的影响。
- 仅性能 gate 失败：保留实验 finding，但不接受重构。

## 产物

所有脚本输出与证据写入本实验 `results/`：`result.md`、`manifest.json`、raw correctness/benchmark、
cubin/SASS resource summary。不得改写 exp_008 历史数据。

## Plan Review

**Date**: 2026-07-20
**Reviewer**: isolated subagent
**Verdict**: ⚠️ Gaps — 已一次性修正；不再复审

- 增加 immutable overlay/version 规则；任何源码修复都新建版本，并从 compile 到 perf 全量 fresh 重跑。
- 对每个 case 枚举 loaded cubin/symbol/launch identity；每个 distinct cubin 都做 static + matched dynamic gate，
  Baseline 也 fresh 重现，container image ID/digest 锁定。
- 锁定 exp_008 FP32 threshold 公式；移除 NaN sentinel 的过度声明，增加独立 nonzero write canary。
- 复用 work ledger，并用 matched NCU executed Tensor/FP4 work 与四项 dynamic spill 指标闭合工作量和 zero-spill。
- 增加 phase helper 的 AST/IR/SASS machine gate，检查 state return/rebind、accumulator lifetime、barrier 顺序和 inline。
- 将噪声自适应性能阈值替换为固定协议与 paired-group bootstrap CI；inconclusive 不得算 pass。
