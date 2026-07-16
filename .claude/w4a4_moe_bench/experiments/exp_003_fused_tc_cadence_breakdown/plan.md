# Experiment 003 Plan: Fused TC Cadence Breakdown

Status: **reviewed / gaps fixed / ready for implementation**. 本文底部已记录唯一一次
`Plan Review`；所有重大缺口已一次性修正，不再进行第二轮 review。

## Goal

回答一个可证伪问题：`M=8192` CuteDSL fused MoE 的 whole-launch
`TC subpipe active = 25.73%`，其低 activity 的主导候选机制是必要的 planned TC-off、
`T1 FC1 Gate / T2 FC1 Up / T4 FC2` 内部真实 TensorCore starvation，还是 task
orchestration。结论来自 assignment-aware sampled timeline 与完整 task population 加权；它不尝试用
IKET 数值重建 NCU 的 `25.73%`。

本实验是 instrumented breakdown，不重新比较 Fused 与 CUTLASS latency，也不把 IKET duration
当作 production latency。`exp_002` 已确认的 stack/spill criticality 不并入本实验；`exp_003`
收口后立即建立独立 `exp_004` 做 lifetime/reorder 单变量消融、未插桩 benchmark 与 NCU。

## Hypothesis and Decision

- **H1：真实 starvation**。同一 MMA warp 的 `T1/T2/T4` 内，QMMA-to-QMMA gap
  反复与 producer/consumer wait、barrier 或其他阻塞 PC/SASS 闭合。
- **H0：主要是 planned TC-off**。低 whole-launch TC activity 主要由 init、route/prefix/pack、
  activation/quant、epilogue/scatter 等必要 non-TC 工作构成；`T1/T2/T4` 内 QMMA cadence
  连续，未发现反复的 wait/barrier 闭合。

判定使用同一 warp、同一 task/slice/block 的事件序列，不跨并行 warp 相加：

1. 每一个 intended `cute.gemm`/QMMA unit 都有一个 range，payload 含 K-block 与 MMA ordinal；
   `QMMA gap = next qmma.start - previous qmma.end`。动态 event count 必须由同一
   task/slice 内连续且无重复的 ordinal 闭合。另以 control/candidate 同口径的 static semantic
   OMMA count 验证 binary 语义未漂移；static instruction count 与 runtime range-event count
   分母不同，禁止互相比较。
2. 每个 complete task/warp envelope 使用不重叠 leaf-range union 分为：`Tensor issue`、
   `planned non-TC`（S2R、activation/quant、epilogue/scatter）、`starvation`（实际等待的
   producer/consumer wait 与 barrier）、`orchestration`（claim/poll、metadata、CTA handoff、tail）和
   `unclassified`。只用 duration-weighted share，频率只作辅助。
3. 从同一 fixture 的完整 task table 构造 population strata：`task slot early/steady/tail ×
   full/partial valid_rows × slice_count`。sample task 必须用 `task_slot` 映射回该 population；每个 stratum
   至少覆盖 8 个 complete tasks，来自至少 3 个 selected-CTA captures，否则不做加权 dominance 判定。
4. 对 capture/CTA 做 cluster bootstrap，报告 population-weighted share 的 95% interval。只有某 bucket
   lower bound `>50%` 且高于其他 bucket upper bound 才称 `planned-dominant`、
   `starvation-dominant` 或 `orchestration-dominant`；两个 bucket lower bound 均 `>=20%` 且无主导者时称
   `mixed`；coverage不足、`unclassified` upper bound `>20%` 或 interval 无法区分时称 `inconclusive`。
5. wait/barrier 只有在 measured interval 明显高于同一 warp 的 empty-marker calibration p95，并由
   同 cubin PC/SASS 证明其边界时才进入 starvation bucket；否则进入 `unclassified`。

加权结果是 instrumented sampled-warp diagnostic estimate，不是 NCU active-cycle denominator；不得把它
与 `25.73%` 相加或相减。whole-launch `25.73%` 仍由 `exp_002` production NCU 负责，estimate 只决定
下一项值得验证的机制。

## Evidence Anchor from exp_002

以下只作为未插桩 production anchor，不与 IKET duration 混算：

- `M=8192` Fused `1782.547 us`，CUTLASS Chain `1665.779 us`，Fused 慢 `6.55%`；
  repeat max spread `0.14%`。
- Fused main `1759.483 us`，占自身 operator wall `99.92%`。
- `TC subpipe active 25.73%`、`Issue active 19.95%`、Eligible warps `0.202`、
  Achieved occupancy `10.42%`。
- grid/block 为 `(1,1,110) / (160,1,1)`；四个 MMA warps + 一个 TMA warp。
- Fused 与 Chain 的 Tensor instructions、FP4 Tensor ops 相同。

权威来源：`../exp_002_fused_vs_chain_dataflow/results/fusedop_dataflow_bottleneck.md`、
对应 NCU report、target manifest 与 source/SASS artifact。

## Fixed Case and Runtime Contract

| Field | Locked value |
|---|---|
| Arm | `cutedsl_bf16_fused` only |
| Shape | `M=8192, E=256, H=2048, I_tp=512, topk=8, SwiGLU, BF16 output` |
| Fixture | 复用 `exp_002/fixture.py`，base seed `2026`；M8192 routed seed `10218` |
| Boundary | BF16 input → one fused W4A4 MoE launch → BF16 output |
| Launch mode | outer CUDA Graph；只分析最后一个明确标识的 graph replay node |
| Production source | FlashInfer `074d93e4aa54c75bee1b3dfdb39b7f075a3ff2af` |
| CUTLASS | `b46b16d003484063bca4ed365e44095c4c6ed633` |
| Kernel source SHA-256 | `94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106` |
| Container | `nvcr.io/nvidia/pytorch:26.05-py3` |
| Image digest | `sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba` |
| Python deps | CUTLASS DSL 4.6.0 read-only overlay tree `32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74` |
| IKET | audited standalone `0.7.10`, Python 3.12；记录 provider tree/file hashes，不使用版本不明的 DSL-bundled CLI |
| GPU | direct-SSH 5KP pool 中一张无 process、无 lease 的 SM120 GPU；以 full UUID 绑定 |
| JIT | 独立 `exp003` workspace；`CUTE_DSL_COMPILER_OPT=iket`；`CUTE_DSL_KEEP=ir,ptx,cubin,sass` 写入独立 dump root；禁止复用 production JIT cache |

同一 source/case 生成三个 binary identity：

1. `production anchor`：`exp_002` normal compiler、no marker；只作历史 production anchor；
2. `iket compiler control`：`CUTE_DSL_COMPILER_OPT=iket`、no marker；
3. `marker candidate`：同一 IKET compiler、marker-only overlay。

control 与 candidate 都用各自 fresh JIT root。只有 candidate 进入 IKET capture；control 用来隔离
compiler mode 与 marker 对 SASS/resource/dispatch 的影响。

manifest 必须记录 host、GPU UUID/index/PCI ID、driver、CUDA runtime、`nvcc --version`、
`ptxas --version`、Python、Torch、CUTLASS DSL/module path、IKET version/provider hash、container digest、
source/overlay/runner/analyzer SHA-256、环境变量、JIT artifacts、kernel name、grid/block 与 graph identity。

## Instrumentation-only Overlay Contract

旧参考工程的 `moe_dynamic_kernel_iket.py` 不能复用：它除 marker 外还改变 activation helper、
SwiGLU 参数和 gated `ab_stage` cap。`exp_003` 从上述 exact production source 重新生成 overlay，
只允许加入 `cute.experimental.iket` range、用于 marker payload 的纯索引别名和注释；不得改变
compute、memory、barrier、pipeline、tile、warp layout、task schedule 或 launch geometry。

overlay gate：

1. 先校验 production source SHA-256；不匹配即停止。
2. 保留 exact instrumentation patch；反向移除该 patch 后必须逐字节恢复 production source SHA。
3. 人工审计 diff，确认 current activation helper、SwiGLU 参数和 `ab_stage <= 2` cap 原样保留。
4. 比较 no-marker control 与 marker candidate 的 grid/block、static registers/SMEM/stack/local、
   normalized non-marker opcode sequence、CFG topology、QMMA/TMA/global reduction/barrier count。marker
   instrumentation site由 instrument config 声明并从 normalized signature 中剔除；若 resource/occupancy
   tier、non-marker CFG 或 semantic opcode count 漂移，timeline只能标 diagnostic，不能作 dominance 判定。
5. instrumented run 必须仍为 `MoEDynamicKernel`、grid `(1,1,110)`、block `(160,1,1)`、
   same task model 与 one material graph node；否则证据作废。
6. prepare 与每个 IKET profile PID 都必须通过与 `exp_002` 相同的独立 quant-aware oracle gate，
   并满足 cosine `>=0.999`、relative-L2 `<=0.02`、max-abs `<=0.08`、每个 token finite/nonzero。
   逐 token relative-L2 p99 保留，但使用 marker candidate 相对同工具链 no-marker control 的上界
   `control + 0.005`，不使用会误杀低范数 token 的绝对阈值；atomic output 不要求 bitwise。
7. final replay 后读取 dynamic workspace，并按本 case 的 deferred publish policy 校验：
   `full_tile_publish_enabled == 0`、`task_tail == expected_task_count`、每个 CTA 在队列耗尽时
   多执行一次 atomic claim，故 `task_head == task_tail + grid_z`（本 case 为 `+110`），且
   `task_ready[:tail] == 0`；同时要求 `row_counts.sum == M*topk`、task descriptor table 与
   source model 逐项一致，`token_map/token_weights` 满足范围、finite 与 routing weight 不变量。
   漏 task 即使 element gate 通过也判失败。

### Named ranges

- all roles：`phase0_init`、`histogram`、`prefix_sum`、`route_pack`、`setup_compute`、
  `marker_calibration`；
- task transition：`task_claim_or_poll`、`task_metadata`、`task_handoff_sync`、`task_tail_exit`；
- MMA warps：`mma_task(task_slot)/mma_slice(slice)`、`fc1_gate`、`fc1_up`、`fc2_block`、
  通用 `wait/s2r/qmma`、`act_quant`、
  `fc2_{epilogue,pre_scatter_barrier,atomic_scatter,post_scatter_barrier}`；
- TMA warp：`tma_task(task_slot)/tma_slice(slice)`、通用 `tma_acquire/tma_issue`、
  `gate_pass_wait`、`final_pass_wait`。

IKET 0.7.10 只有 event ID `1..30` 可供用户 range，`0/31` 保留。overlay 必须静态证明
unique range name `<=30`。通用 range 的 payload 合同为
`phase_id * 1_000_000 + (ordinal + 1)`，`phase_id={gate:1, up:2, fc2:3}`；因此包括
`ordinal=-1` sentinel 在内都可逆，analyzer 不得猜测 phase。

每个 repeated range payload 绑定 task、intermediate slice、K tile 或 FC2 output block；无法唯一绑定的
range 明确标 `partial`，不与另一 payload 强行对齐。

## Capture Protocol

1. 按 `direct-ssh-gpu-pool` 做只读发现；在共享 lease root 原子获取一个 lease，二次核验
   UUID/process/memory；以 foreground SSH 和 full-UUID `CUDA_VISIBLE_DEVICES` 运行，退出后只清理
   本次 owned process/lease 并 recheck。
2. 将 verified IKET 0.7.10 provider 只读挂载到 container；先跑 provider/version/import smoke。
3. 先生成并保存同 fixture 的 oracle reference；执行 control/candidate overlay gate、M8192 correctness、
   workspace/task invariants、kernel/grid/block/JIT identity 与 static SASS/resource smoke。
4. 通过 KDK `iket_safe_capture.py` 先 dry-run，再使用 fresh output、`postprocess=json`、
   `context-buffer-size=1G`、hard cap `4G`、不使用 `--clobber`。
5. selected-CTA capture 使用预声明顺序
   `(0,0,{0,55,109,13,27,41,69,83,96})`，每个坐标独立 fresh capture。先跑前三个；只有
   population strata未达到 `8 complete tasks × 3 captures` 时才按顺序补采，最多九个。固定坐标不被
   当作 early/middle/tail；实际 `task_slot/task descriptor/local ordinal` 决定 stratum。
   runner 必须拒绝预声明集合外或 grid 越界坐标；analyzer 另从 decoded artifact 核对实际 enabled cluster，
   防止 IKET 对非法坐标静默回退到 `(0,0,0)`。
6. IKET tracker/profile 会重复执行 target；target 必须 side-effect safe，并给每个 PID 写独立 manifest。
   setup只含一次 eager launch与 graph capture，warmup为0，之后只 replay一次 graph。decoded trace 中必须
   恰好存在一个 matching graph node，并绑定唯一
   `(pid, contextId, graphLaunchKey, gridId, kernelName)`；regular eager launch明确排除。
7. 任一 capture 若 context overflow、缺少目标 graph node、缺 marker、选中 CTA 没有完整 task/slice，
   先停止解释；只允许增大 buffer 或换一个预先声明的 CTA，不允许改 workload 后拼接。

## Analysis and PC/SASS Closure

- canonical numerical input 是各 run 的 `iket/pid_*/iket.decoded_results.json`；保留 raw JSON、
  tracker、instrument config、target manifest、capture manifest 与 SHA-256。
- KDK generic analyzer先做 launch identity、scope、range/event合法性检查；experiment-specific analyzer
  再按 warp/role/payload输出 top-level phase union、每个 intended QMMA interval/gap、
  wait/barrier/scatter/orchestration leaf union、task assignment与event-count closure。
- inclusive nested range 不相加；只在同一 warp track 内计算 interval union。timestamp unit 未由 artifact
  证明时统一写 `raw timestamp units`。
- PC 只来自同一次 instrumented tracker cubin。用 `instrument.config.json` 的 range offset 对齐
  `nvdisasm`，标注 QMMA、load/S2R、DEPBAR/BAR、atomic reduction 等指令；禁止把 canonical cubin PC
  硬映射到 instrumented cubin。
- `exp_002` whole-launch NCU stall/utilization只能作为全局背景，不能按 IKET phase duration比例拆分。
- population weighting只使用同一 profile PID 写出的完整 task descriptor table；tracker pass、另一次
  capture 或静态期望不能替代它。early/steady/tail分别是 task slot `[0,10%) / [10%,90%) / [90%,100%)`。

## Required Outputs

所有生成物都放在本实验的 `results/`：

- `results/validation.manifest.json`：环境、工具、source/overlay/JIT、fixture/correctness、dispatch、
  selected scope 与 artifact digest；
- `results/raw/iket/<cluster>/...`：provider-native capture（gitignored，但保留并由 manifest引用）；
- `results/derived/`：generic IKET summary、warp/role/payload cadence CSV/JSON、PC/SASS mapping；
- `results/derived/task_population.json` 与 `weighted_phase_shares.json`：完整 task population、sample coverage、
  stratum weights、bootstrap interval和四分支判定；
- `results/result.md`：只报告证据范围、planned TC-off vs starvation 判定、证据不足与 `exp_004` handoff。

## Stop Conditions

出现以下任一项立即停止 formal interpretation：source/overlay reverse gate失败、oracle失败、
kernel/grid/block/graph漂移、GPU UUID/lease漂移、IKET provider identity不明、target重复执行不安全、
trace overflow、目标 ranges 缺失、selected scope未声明、或无法把 PC/SASS 绑定到同一 instrumented cubin。

## Out of Scope and Next Experiment

- 不在本实验修改 production kernel，不做性能优化 verdict。
- 不做 stack/spill lifetime ablation，不用 IKET duration声称 spill cost。
- 不做 FC2 scatter diagnostic-only 消融，也不做 ready-task/full-tile publish overlap 改动。
- 本实验完成后建立 `exp_004_fused_stack_spill_criticality`，验证 122-word/lane stack roundtrip
  是否来自 Gate accumulator 跨 Up 保活并位于关键路径。

## Plan Review

**Date**: 2026-07-16
**Reviewer**: subagent

**Verdict**: ✗ Misaligned

**Gaps + suggested fix**:

- Whole-launch goal 与三个 CTA 的证据范围冲突：加入 task/CTA assignment-aware population weighting、
  coverage gate 与区间；不能满足时只能判 inconclusive。
- Dynamic task claim 使固定 CTA 不代表 early/middle/tail：按 task slot、full/partial、slice 与执行阶段分层，
  用预声明 CTA 序列补足 coverage。
- `80% + p50` 会漏 rare-but-long stall且没有 mixed case：改为 duration-weighted leaf interval union，
  固定 dominant/mixed/inconclusive 分支并单列 orchestration。
- Reverse patch 不能证明 binary identity：加入同工具链 no-marker control，对照 semantic SASS/CFG、资源、
  dispatch与QMMA event-count closure。
- Graph replay/correctness identity不足：绑定完整 launch tuple，只允许一个 matching graph replay，并校验
  full task workspace、routing invariants与逐 token correctness。
