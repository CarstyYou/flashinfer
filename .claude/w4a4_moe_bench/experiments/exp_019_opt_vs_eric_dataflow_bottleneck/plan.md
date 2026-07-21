# exp_019：Latest Opt vs Eric Stage4 Dataflow Bottleneck

状态：closed；M8192 phase + paired NCU 完成，M1024 phase 归因因跨来源幅度门禁失败而降级为 inconclusive

## 目标

在同一块 5KP、同一输入/权重/路由与完整 MoE 边界下，直接比较两个语义等价的 NVFP4 fused kernel：

1. `M=8192` 时 Eric 比 Opt 慢 `20.85%`，差距主要落在哪些 phase，整 kernel 的硬件代价是什么；
2. `M=1024` 时 Eric 比 Opt 快 `3.88%`，若跨来源幅度门禁通过，再判断优势来自哪个 phase；
3. 哪些 Eric 机制值得以单变量方式移植到 Opt，哪些应保留 Opt 实现。

本实验是 `paired / equivalent` 的 kernel-to-kernel 对比。phase timing 用来定位时间差发生在哪一段，
NCU 用来解释完整 kernel 的硬件行为；不把 whole-kernel NCU ratio 按 phase 占比分摊。

## 锁定对象与比较约束

| Field | Latest Opt | Eric Stage4 |
|---|---|---|
| Source | `.claude/w4a4_moe_bench/moe_dynamic_kernel_opt.py` | `.claude/w4a4_moe_bench/moe_dyanmice_kernel_ab_stage4_compact.py` |
| SHA-256 | `ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19` | `3a5000a990bb978b434f1c7dac621de25112d9f3cec4a5fdfab5f2970b0dc3b8` |
| Boundary | BF16 input → complete NVFP4 fused MoE → BF16 output | 相同 |
| Cases | `M=1024,8192; E=256; H=2048; I_tp=512; topk=8` | 相同 |

- 复用 exp_001 canonical fixture 与 exp_018 的共同 FP4 weight/oracle；两臂 correctness、fixture hash、
  packed weight/scale hash、routing/task count必须一致。
- 复用 exp_018 未插桩 benchmark 作为正式 latency 锚点；源码或 fixture 未漂移时不重跑 benchmark。
- 同 GPU UUID、2377 MHz application clock、同容器/dependency、独立 fresh JIT root、无 foreign process。
- Eric 只允许使用 exp_018 已审计的三 keyword compatibility adapter；不得改 kernel body。
- 任何 source/dispatch/wrapper/cubin/correctness/GPU/clock drift 均 fail closed，禁止拼接不同 rerun。

可执行 identity lock：

| Identity | Expected |
|---|---|
| FP4 image ID / digest | `sha256:a4e056e1d34a5cc9387512ffa3abeed778e3dc7966633c5154d771705d8835ac` / `sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba` |
| Dependency hash | `32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74` |
| Dispatch / wrapper SHA-256 | `cba2d0966631a47a576747e8322b57116122f2c8e5e868f8efb3f5ea692391a4` / `bcac806795c035decd0773f4f801d477e7ebf14c1d67c3e49eee42ee0579c0a4` |
| Eric adapter SHA-256 | `98adfba7f4e0d00af24383a556e9c93088355539b50dc82480091225e0448120` |
| Production cubin SHA-256 | Opt `e9b322e4c978c490adbe0a9bf0f9a183288c0ecddb1fd72e5a904c487be541f3`; Eric `4c728f4ee6115f342f0b32e578ca4901abd7f35ac233035fb8eaf54fce3900b0` |
| Launch | grid `110×1×1`; Opt block 288 / Eric block 160；exact MoE symbol由 exp_018 `jit_identity` 锁定 |
| Resolved design | Opt 8 math + 1 TMA、OMMA `(4,2)`、AB stage≤2；Eric 4+1、OMMA `(2,2)`、AB stage=4、epilogue M64 |

fixture manifest及 M1024/M8192 NPZ hash继续使用 exp_018 plan中的 immutable values。control、probe、NCU
分别使用 `arm × mode` 独立 JIT root。只有未插桩 NCU target必须重现对应 production cubin；phase
control/probe因额外 event ABI 是独立 diagnostic binary，各自锁定 overlay/cubin/resource并做 matched SASS审计。

## 证据 A：同边界 phase timing

复用 exp_017 的 `%globaltimer` helper、事件 ABI、control/probe 配对、闭合公式与结果聚合，不复制
一套新的 runner。Opt 沿用 exp_017 anchor map；Eric 的 monolithic body 使用独立的最小 anchor map，
共同输出同一 phase vocabulary。两臂都生成 matched no-marker control 与 marker probe，production
源文件不修改。

统一的语义边界：

1. Clear + Histogram + Prefix
2. Route + Q0 + Pack + publish tail
3. Claim + cache + task control
4. FC1 Gate + Up + SwiGLU
5. Q1
6. FC2 GEMM + epilogue + R2S + pre-scatter sync
7. Scatter to GMEM
8. CTA residual / producer tail / launch skew

FC2 的 epilogue、R2S 与进入 scatter 前的依赖同步必须归入 FC2，禁止把 GEMM 尾部误算为 Scatter。
Eric compact epilogue 会按 `epi_m` 交错多个 FC2-tail 与 Scatter 区间；probe 必须分别累计所有区间，
SwiGLU/R2S 与 Q1 也按 `epi_m` 交错；两组都必须逐 interval 累计，不能伪装成连续源码段，也不得
为方便计时新增同步。每个 phase另记录 implementation-specific occurrence count，并与实际
`task × slice × epi/output-tile` manifest核对；residual closure不能替代漏 marker 检查。
每臂每 case 采 5 个 warmed CUDA Graph replay；每次前做 192 MiB L2 flush，flush 不计时。使用相同分母：

```text
D = grid_ctas × (max CTA final - min CTA entry)
phase_equivalent_wall = Σ CTA phase interval / grid_ctas
```

reader rows必须互斥并与 residual 严格闭合至 100%。采集使用 cyclic paired process blocks，轮换
Opt/Eric 与 control/probe顺序。probe 相对同臂 control 的 CUDA-event median扰动须 `≤5%`，且两臂
`probe/control` 差分偏差须 `≤1 percentage point`；phase delta 的 replay envelope也必须小于 exp_018
whole-op gap。M1024 任一差分噪声门禁失败则只报告 `Inconclusive`，不解释 3.88% 反转。同时审计
REG/SMEM/STACK/static spill 与 normalized SASS/control-flow drift。
通过扰动门禁后，允许展示同一语义行的诊断性时间、own-side share、Eric-vs-Opt delta；它不替代未插桩 E2E。

## 证据 B：成对 production-kernel NCU

先对主 case `M=8192` 的两个 production exact cubin 各采一个 correctness-qualified 完整 graph replay。
只有 M1024 fresh control 与正式 benchmark 的 crossover 信号通过跨来源幅度门禁时，才扩展 M1024 NCU。
两个实现都只有一个 material fused MoE kernel，因此该 launch 同时就是完整
operator boundary；若 dispatch preflight 发现额外 material kernel，则停止并修改比较契约。

四份 capture 使用同一 NCU 版本、section/metric bundle、cache-control、clock-control 与 replay protocol。
正式 latency 仍取 exp_018；NCU duration只作 profiler 内 paired diagnostic。报告直接横向比较：

- Launch / Resource：grid、block、register、SMEM、stack、dynamic spill/refill、Achieved occupancy；
- Schedule：Issue active、compact stall / throttle sample share；
- Compute：TC subpipe、ALU/FMA/XU utilization 与 Tensor instruction/work counts；
- Memory：DRAM/L2/L1/LSU/TMA utilization，以及可加的 GMEM/L2/local traffic footprint；
- Work：Executed warp instructions 等完整 kernel 可加计数。

不同分母的 utilization 不相加；stall/throttle 百分比必须注明是 PC-sampling sample share，不伪装成
elapsed-time share。由于 Opt 是 288 threads/CTA、Eric 是 160 threads/CTA，warp/instruction/traffic 总量
必须同时报告完整 logical operator count，必要时按 routed row/output归一化；不得直接把 per-warp 数字
当工作等价。只有 metric 语义、单位、实例归并与完整 launch scope都一致时才计算 delta。

## 不做与停止条件

- 不跑六个 M；`M=2048` 仅当 1024/8192 phase 结果无法解释 crossover 时再补，不预采。
- 不跑 NSys：两臂各一个 material kernel，launch topology不是本轮问题；preflight 不满足时才触发。
- 不跑 IKET / PerfSim / 全 metric NCU；先使用复用的 source marker和问题导向 NCU bundle。
- 不修改 Opt 或 Eric production source，不在本实验直接集成优化。
- 只保留共享 runner、compact CSV/JSON、manifest 和用户可读 `results/opt_vs_eric_triton_bottleneck.md`；raw profiler产物 gitignore。

## 报告与判定

`results/opt_vs_eric_triton_bottleneck.md` 使用中文、四个章节：

1. 对比约束与 exp_018 benchmark 锚点；
2. 两臂 phase 耗时/占比及同边界横向差值；
3. `M1024`、`M8192` 的 paired kernel-to-kernel NCU 表与瓶颈解释；
4. 可整合机制与 Next To Do。

机制只有在 `phase delta + paired NCU + source/SASS` 三者形成一致证据链时才写为移植候选；否则只列为
待验证疑点。后续移植必须以 Opt 为 baseline、一次只改变一个机制，并通过 correctness、zero-spill 与
fresh paired benchmark；没有 E2E 收益即 reject。

## 执行顺序

1. CPU 侧锁定 source/fixture/phase boundary，并测试共享 overlay/manifest builder。
2. GPU smoke：两臂 eager + graph correctness，各 1 replay；失败即停。
3. 同机完成两臂 control/probe phase capture与扰动审计。
4. 先完成 M8192 两份 exact production-kernel NCU capture，并用 VeloQ 生成统一证据卡片；第二 case 由门禁触发。
5. 做 data audit 与一次独立 bottleneck conclusion review，再生成最终报告。

## Plan Review

**Date**：2026-07-21
**Reviewer**：isolated subagent
**Verdict**：⚠️ Gaps — 已一次性修正；不再复审

**Gaps + applied fixes**：

- M1024 的 3.88% 信号可能小于允许的 marker 扰动：加入 cyclic paired blocks、`≤1 pp` 跨臂差分
  扰动门禁与 phase-delta uncertainty gate；失败即降级为 `Inconclusive`。
- Eric 不只 FC2/Scatter 交错，SwiGLU/R2S 与 Q1 也按 `epi_m` 交错：改为两组逐 interval 累计并核对
  implementation-specific occurrence count，禁止用 residual 掩盖漏 marker。
- identity gate缺少可执行 reference：补齐 image/dependency、adapter、dispatch/wrapper、production cubin、
  launch、resolved stage/warp/tile与独立 JIT root锁定值。

## 执行后偏差与处理

- M1024 fresh no-marker control gap 为 `-7.168 µs`，正式 exp_018 gap 为 `-21.348 µs`；方向一致但
  幅度比 `2.98×`，超过 data-audit 的 `2×` 门禁。因此 M1024 phase 不作正式 crossover 归因，也未触发
  M1024 deep NCU。
- M8192 fresh control、phase sum、probe event 与正式 benchmark gap 同向且量级闭合，完成 paired
  production NCU；报告只在该 case 给出硬件瓶颈结论。
- 原始 VeloQ `inspect.json` 只用于本地证据构建；后续执行应在 capture host 先生成 compact evidence
  card，再仅回传 card、manifest 与 raw locator/hash。

M8192 canonical NCU raw 保留在 `10.6.142.16`：
`/home/xiy/workspace/flashinfer_exp001_corrected_074d93e/.claude/w4a4_moe_bench/experiments/exp_019_opt_vs_eric_dataflow_bottleneck/results/raw/ncu/`。
production report SHA256：Opt `883355ca648f9eaefe4e85db1f3cf054c7f49e1194d09313f242115d1a23af2b`，
Eric `5dce4fefca1158bbcf02ccb2297985795857c77511273d35a5901cd01819a84e`；additive ledger SHA256：
Opt `ed865eb4cac0a398b75cbd2ffdfe61d82991bab4e0a7d79f701191cf2796a504`，
Eric `aa163191319ff6142523592cb57270a0d97b02c4a46ba68bff309b72a95181da`。
