# exp_018：Triton FP8 vs Latest Opt FP4 vs Eric Stage4 FP4 Benchmark

Status: **Complete**

## 目标

在同一块 5KP 上，对 `M=256,512,1024,2048,4096,8192` 做一次 fresh 三臂 benchmark，回答：

1. Eric Stage4 FP4 在全部 prefill case 是否正确；
2. Eric、Latest Opt 相对 SGLang Triton FP8 的完整 MoE E2E latency 与 speedup；
3. Eric 相对 Latest Opt 哪些 M 更快或更慢。

本实验只做 correctness + 未插桩 benchmark，不做 NCU、NSys、IKET 或因果 breakdown。

## 三臂身份

| Arm | 实现 | 锁定身份 |
|---|---|---|
| SGLang Triton FP8 | legacy Triton `fused_experts_impl`；BF16 input、动态 tensor-wise E4M3 activation、E4M3 weight | SGLang commit `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`；沿用 exp_001 pinned image |
| Latest Opt FP4 | `.claude/w4a4_moe_bench/moe_dynamic_kernel_opt.py` | SHA-256 `ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19` |
| Eric Stage4 FP4 | `.claude/w4a4_moe_bench/moe_dyanmice_kernel_ab_stage4_compact.py` | SHA-256 `3a5000a990bb978b434f1c7dac621de25112d9f3cec4a5fdfab5f2970b0dc3b8` |

Eric 源码只允许生成已经验证过的 constructor-keyword compatibility adapter；adapter 只能补
`swiglu_alpha/beta/limit`，不得修改 kernel body、stage、warp、tile、epilogue 或数学实现。

## 比较约定与约束

- Case：`E=256, H=2048, I_tp=512, topk=8, SwiGLU, BF16 output`；只测上述六个 M。
- 三臂必须读取 exp_001 的同一组 persisted NPZ fixture；禁止把 exp_009 的旧 fixture 或旧 latency
  拼入本实验。fixture manifest SHA-256 锁定为
  `683ec75341e4d8317dfdc5c4b04229f9695f9aa286d575c4f6e1fdef55d90801`，NPZ hash 为：

  | M | NPZ SHA-256 |
  |---:|---|
  | 256 | `86b505097acd06bed5a50c3528c78525e6087c07ed69f86606607599ffa21686` |
  | 512 | `e6ddb487121a0d681a06bcb453f38623b3d5d8477f2232bbbf78cd2ea4ef23a3` |
  | 1024 | `0fa7e8a7d8d1d32172971f987d6f55b534aabf8d12a84a910d010cec25ba04a5` |
  | 2048 | `5375fd8b3e5e15f8c956998bfc3e2f3ee59948a2aeaf5ba1294ec6a74092bde3` |
  | 4096 | `a1ac93cb8dfb2e81a000476efc36b75588f79f1954b406396b77c172464ce2cc` |
  | 8192 | `c113ecd5ddeff77154ddbd23fc3dc3c83f8ee822e880179ca5c16b1145372438` |
- 三臂逻辑边界统一为 `BF16 input → 完整 MoE → BF16 output`。不同 precision/runtime 是被比较实现的
  固有差异，因此只允许报告完整算子 latency/speedup，禁止归因为 precision、fusion 或某个 phase。
- 两条 FP4 arm 使用相同 seed 派生的同一组 packed NVFP4 weights/scales，并共同对照独立的 dequantized
  NVFP4 PyTorch oracle；FP8 使用实际 E4M3 weights/scales 构建的独立 oracle。FP4 gate 沿用
  `≥97%` 元素满足 `abs_err < max(0.05, 1.5×oracle.std)` 或 `rel_err < 0.5`；FP8 gate 为
  `rtol=0.1, atol=0.01`。每个 arm/M 至少检查两次 graph replay，并在 timed sample 后再检查一次，
  同时校验 finite/nonzero/sentinel 与 workspace reset。任一 arm/M 失败，该单元格写 `Invalid`，不计算 latency。
- 使用同一 GPU UUID、2377 MHz application clock、同一 direct-SSH lease 与共同 rerun ID；每个 arm
  使用独立 fresh JIT/cache。FP4 image ID 锁定为
  `sha256:a4e056e1d34a5cc9387512ffa3abeed778e3dc7966633c5154d771705d8835ac`、digest 为
  `sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba`；SGLang image ID 锁定为
  `sha256:663867442f321ded36228bafd889fd1db05cbef7a7c8ea6e072df33234dabbfd`、digest 为
  `sha256:00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2`。同时记录 FlashInfer/CUTLASS commit、Python dependencies、wrapper/launch
  kwargs、precision-specific weight hashes、环境开关和实际 JIT artifact/symbol。
- Eric adapter SHA-256 必须为 `98adfba7f4e0d00af24383a556e9c93088355539b50dc82480091225e0448120`，
  并保留三行 keyword-only diff 与 AST-equivalence proof。
- 不调用 `torch.profiler`、NCU 或 NSys。dispatch identity 由 exact imported module/source SHA、resolved
  config/constructor kwargs、JIT artifact hash 与目标 kernel symbol 共同闭合。
- 检测到 foreign compute process、GPU/clock/source/hash drift 即停止整轮；禁止用半轮数据拼接。

## 测量协议

- 每次 sample：CUDA Graph 单次完整 operator replay；external CUDA events 在 graph 内。
- 每次 replay 前执行 `192 MiB` L2 flush，flush 与 synchronize 不进入计时。
- 每个 block/arm/M：`warmup=5`、`timed=50`，得到一个 sample；共 3 个 process-level block，顺序为
  `Opt→Eric→Triton`、`Eric→Triton→Opt`、`Triton→Opt→Eric`。报告 3 个 block sample 的 median，
  `spread=(max-min)/median×100%` 必须 `≤5%`。
- arm 间 cooldown 2 秒；这不是 ABBA。每个 block 前后记录 graphics/memory clock、temperature、power、
  P-state 与 throttle counters；application clock/UUID 必须匹配，thermal/hardware slowdown counter 不得增加。
  相对结果属于同机、同 clock、同协议的 cross-process/cross-runtime comparison。
- 只保留可审计的 compact raw CSV、correctness/identity manifest 和最终 `result.md`；不生成 profiler 产物。

## 结果格式与判定

`results/result.md` 只展示一张六行表：

| M | Triton FP8 | Latest Opt FP4 | Eric Stage4 FP4 | Opt vs Triton | Eric vs Triton | Eric vs Opt |
|---:|---:|---:|---:|---:|---:|---:|

- `X vs Y speedup = Y_latency / X_latency - 1`；同时保留原始 latency。
- Latest Opt / Eric 相对 Triton 的 2× 目标为 speedup `≥100%`。
- Eric vs Opt 的“更快/更慢”只在 3 个 matched block ratio 方向一致且 median 幅度越过 `±2%` 时判定；
  否则写“相当/轻量测量无定论”。
- 不平均不同 M，不使用旧实验 latency 补洞，不从 benchmark 推导内部瓶颈。
- correctness cell 失败不重跑，记录原因并继续其他 cell；只有双方均为 `Pass` 才计算对应 speedup。
  identity/protocol/foreign-process 失败或 spread 超限时标为 `Inconclusive`。任何环境级失败只能使用新
  rerun ID 从头重跑，禁止选择性补采。
- Eric“全部正确”要求六格全为 `Pass`；2× 目标逐 M 判定。正式写报告前执行 ex-post data audit；
  无法闭合身份、correctness、稳定性或 lineage 时不发布 speedup。

## 最小实现

1. 复用 exp_001 fixture、oracle、Triton callable 和计时基础函数，但不复用其中的 profiler dispatch gate。
2. 只新增一个参数化 arm runner，同时承载 Triton、Latest Opt 与 Eric；三个 arm 分进程、分 JIT root。
3. 新增一个 compact result builder，拒绝 mixed rerun、fixture/hash、protocol 或未显式标记原因的缺失 case。
4. benchmark 完成后回收容器、JIT 临时目录和 direct-SSH lease。

## Plan Review

**Date**: 2026-07-21
**Reviewer**: isolated subagent

**Verdict**: ⚠️ Gaps — 已一次性修正；不再复审

**Gaps + suggested fix**:

- 旧 runner 的 profiler dispatch 与实验 scope 冲突：改为无 profiler 的 module/config/JIT/symbol identity gate。
- Correctness、source、fixture 与 adapter identity 未完全闭合：已补共享 FP4 oracle、固定 tolerance、重复 replay、
  fixture/image/dependency/JIT/adapter 锁。
- 整臂串行会混入顺序漂移：已改为三个 cyclic process block，并增加 clock/temperature/power/throttle gate。
- Invalid/Inconclusive 与快慢判定不足：已补 cell 隔离、禁止选择性补采、spread 公式和 `±2%` 方向阈值。
