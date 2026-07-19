# exp_009：实习生 stage4 compact 轻量验证

Status: **Complete**

## 目标

仅针对 canonical W4A4 MoE benchmark，回答
`moe_dyanmice_kernel_ab_stage4_compact.py` 相对当前 Production：

1. 能否正确编译并通过 M256、M8192 correctness；
2. 是否仍有 compiler/static spill，以及 M8192 运行时是否执行 spill/refill；
3. M256、M8192 的完整 fused CUDA Graph latency 是更快还是更慢。

本实验不做 phase timing、完整 NCU metrics、source/SASS breakdown 或优化归因。

## 对比臂与实现边界

| Arm | Source |
|---|---|
| Production | 当前锁定 production kernel，SHA-256 `94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106` |
| Intern stage4 compact | 原始实习生文件，SHA-256 `91034c7cd3b3b9fe8cbde6dbf1bb2c8c13e4261ff9e9e7d642f3ce9d83788768` |

原始实习生文件缺少当前 dispatcher 必传的 `swiglu_alpha/beta/limit` 参数，因此不能直接启动。实验生成 immutable
adapter overlay，仅恢复这三个 keyword 参数并锁定 canonical `activation="silu"`；参数不进入 kernel body，不改变
stage4、SMEM、warp、tile、epilogue 或数学实现。报告同时记录 original hash、adapter diff/hash 与最终 cubin hash。

## 固定条件

- GPU：同一块 5KP、同一 application clock，无其他 compute process。
- 软件：复用 exp_008 已锁定的 FlashInfer checkout、CUTLASS、26.05 容器、CuteDSL/CUDA/nvcc/ptxas 环境。
- Shape：`E=256, H=2048, I_tp=512, topk=8`；仅 `M=256, 8192`。
- 输入、weights、reference、routing fixture、CUDA Graph boundary 和 L2 flush 与 exp_008 Production ABBA 相同。
- 两臂各自使用独立 fresh JIT namespace；禁止继承旧 cubin。

## 最小执行流程

1. 从原始实习生文件生成 immutable adapter overlay，并机器校验 adapter diff 只包含构造参数兼容层。
2. Production 和 candidate 分别 prepare M256、M8192 canonical；校验 source/JIT/cubin/GPU/toolchain identity。
3. Candidate 跑 M256、M8192 correctness，沿用既有固定阈值；任一失败即停止，不报性能。
4. 对 candidate 每个 distinct cubin 解析 registers、SMEM、STACK、compiler SpillRefill/local SASS。静态与动态证据
   必须共同锁定 exact mangled/demangled kernel symbol、grid/block、loaded cubin hash、CUDA Graph node/launch ID；
   多匹配或缺失均 fail closed，不能用整 cubin/整进程聚合替代目标 launch。
5. 若 correctness 通过，M256、M8192 各做 3 组 `Production-Candidate-Candidate-Production`；warmup=5、timed=50、
   每次 replay 前 192 MiB L2 flush，每个 position 独立进程。报告 paired speedup，
   `speedup = Production / Candidate - 1`；完整 ABBA group 是 paired statistical unit，预注册 `±2%` 轻量噪声带。
   三组方向一致且 paired estimate 越过 `±2%` 才回答更快/更慢，否则回答“轻量测量无定论/相当”。
   3 组只用于轻量方向判断，不包装成高置信泛化结论。
6. 仅对 candidate M8192 未插桩 cubin 采四项 NCU dynamic spill/refill counters；missing/`n/a` fail closed。
   若 M256 和 M8192 实际是同一 cubin，不重复采集；若不同，只对 M256 补 static，不扩大成完整 NCU。

## 判定

- 编译或 correctness 失败：candidate 不可用，性能不测。
- Static spill 非零或 M8192 dynamic spill/refill 非零：回答“仍有 spill”，并给出精确计数。
- Static 与 dynamic 均为 0：只回答本 specialization 未观察到 spill，不外推其他配置。
- 性能只报告 M256、M8192 的同机 paired 数据；不做因果归因。

## 产物

全部放入 `results/`：original/adapter overlay identity、correctness、static spill、单次 dynamic spill、最终 quick benchmark、
manifest 与简洁 `result.md`。正式汇报前执行 ex-post data audit。

## Plan Review

**Date**: 2026-07-19
**Reviewer**: isolated subagent

**Verdict**: ⚠️ Gaps — 已一次性修正；不再复审

**Gaps + suggested fix**:

- Spill 证据必须绑定 exact kernel entry 与 CUDA Graph launch：已补 symbol、grid/block、cubin、graph node/launch ID
  联合身份及 missing/多匹配 fail-closed 规则。
- 3 组 ABBA 缺少方向判定：已补完整 group 为 paired unit、`±2%` 噪声带与方向冲突时“无定论”规则。

## Follow-up：M sweep

扩展到 `M = 256, 512, 1024, 2048, 4096, 8192`：

1. 三个 arm 每个 M 先独立运行 correctness preparation；Intern 失败点额外记录连续 replay 的数值误差、
   NaN/sentinel、workspace route/task gate 与 cross-replay stability，用于区分精度偏差、漏写和潜在竞态；
2. 只有相应 arm correctness 通过才允许 E2E benchmark；失败单元格写 `Invalid`，禁止运行或展示错误 kernel 的耗时；
3. 本轮只做快速 benchmark：每个正确 arm/M 保留 1 个 timed sample；每个 sample 为 `warmup=5`、`timed=50`、
   replay 前 192 MiB L2 flush，arm 间默认 cooldown 2 秒；不使用 ABBA 重复采样；
4. 最终只展示六个 M 的三臂 latency/speedup 表与 Invalid 边界；不启动 NCU 或 phase breakdown。

本节属于同一 exp_009 的 scope follow-up；根据 single-round 规则，不重复执行 Plan Review。

## Closeout

- 最终结果以 `results/result.md` 与 `results/full_m_sweep/summary.json` 为准；旧 ABBA 重复采样已删除。
- Intern 在 `M256–2048` correctness 通过并快于 Production/exp_008；`M4096/8192` correctness 失败，禁止报告性能。
- 本实验到此关闭；不继续定位、修复或优化 Intern kernel。若未来重启，必须新建实验并重新定义目标。
