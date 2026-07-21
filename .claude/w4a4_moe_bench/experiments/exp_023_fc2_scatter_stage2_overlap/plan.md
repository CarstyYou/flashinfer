# exp_023：FC2 -> Scatter 双 stage overlap

## 目标

验证将 FC2 输出按 `M64 x N128` 切为两个 `sC` stage，能否形成真实的生产者/消费者流水：

```text
FC2 producer:  produce stage[p] for output tile j
Scatter consumer: consume stage[q] for output tile j-1
```

重点不是把 `128x128` 串行拆成两次 `64x128`，而是证明 FC2 和 Scatter 在不同 output tile
之间实际重叠。仅有 `stages=2` 配置、但时间线上仍串行，判为失败。

## 候选设计束

- Baseline：当前 `moe_dynamic_kernel_opt.py`，source SHA256
  `ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19`。
- Candidate：独立 overlay；warp role 固定为 W0-W7 FC2 producer、W8-W11 Scatter consumer、
  W12 TMA/control，即 `8 + 4 + 1 = 13 warps / 416 threads`。新增 consumer 不得执行
  `tiled_mma.get_slice(tidx)`、FC1/FC2 fragment setup 或其他高寄存器 math setup。
- `sC` 的总 payload 保持 32 KiB：同一 allocation 对 FC1/Q1 保留完整 `M128xN128` view，
  对 FC2/Scatter alias 为两个 `M64xN128` stage；不复制第二份 32 KiB payload。
- 用两个 stage 的 full/empty handoff 协议控制 publish、consume 与 reuse。FC2 producer 可以在
  Scatter consumer 排空 tile j 的 stage 时推进 tile j+1；末尾必须完整 drain。
- 现有 Route/Q0、FC1、Q1 的 logical ownership 与工作量固定在原 288-thread/9-warp virtual
  stride；显式限定只有 W0-W8 执行这些 phase，W9-W12 不得索引 9-warp route cache。新增 warps
  只参加确实定义为 CTA-wide 的同步，不能笼统加入 math/TMA named barrier。

这是不可拆的最小设计束：dedicated Scatter warpgroup、416-thread block、双 stage alias 和 handoff
barrier 是产生 overlap 的共同必要条件。允许它们引起 register allocation、barrier、occupancy 和
schedule 变化，但不允许改变 MoE logical work、FC2 tile/math、Scatter 的 logical output set/BF16x8
REDG 次数或 persistent task semantics；4 consumer warps 的 physical lane ownership允许从 baseline
8-warps remap，但必须逐元素证明唯一覆盖。

## Stage 与同步协议

- 定义 `seq = 2 * output_tile_idx + m_half`，`m_half in {0,1}`；pipeline stage index为
  `seq % 2`。producer/consumer 各自只按 seq 单调前进，不从共享的 mutable output index重建身份。
- `PipelineAsync(num_stages=2)` 使用 4 个 Int64 mbarrier：2 个 full、2 个 empty。按当前
  CuTeDSL per-warp arrival 约定，producer cooperative group size=8，consumer group size=4；
  W0-W7 对每个 seq 都收敛执行 acquire/commit，W8-W11 对同一 seq 收敛执行 wait/release。
- producer acquire 对应 empty epoch 后，只有拥有该 M64 half 的 warp写入 stage；所有 8 producer
  完成 R2S 后执行 `fence_proxy("async.shared", space="cta")` 再 commit full epoch。
- consumer wait full epoch 后读取并 Scatter 完整 M64xN128 logical set；完成最后一条 REDG 后
  release empty epoch，producer 在相同 stage 下次 acquire 前不得复用。
- task/slice metadata、token/weight cache 直到全部 `2 * output_tile_cnt` seq 被 release且 producer
  tail完成后才允许 pass-final；odd/short iteration 必须 drain，不能让上一个 task 的 stage/tag串入下一个。
- diagnostic overlay 可记录 `(task, slice, output_tile, half, stage, epoch)` tag校验协议；tag不进入
  未插桩性能 Candidate。

## 与 exp_013 的边界

exp_013 是同一批 math warps串行执行 `R2S -> barrier -> 4-warp Scatter`，并连带修改 FC1/Q1；
它没有 FC2/Scatter overlap。本实验不得复用该串行结果作为结论，也不得重做其 compact-Q1 变更。

## 分级验证

### Gate A：编译、资源与协议

- Candidate 可编译，完整 `sC` view 与两个 compact stage alias 地址/容量闭合；总 `sC` payload
  仍为 32 KiB，仅允许少量 stage barrier state 增量；
- 8-warp FC2 的 OMMA tile/instruction work 不变；Scatter logical output set和 REDG work不变；
- register redistribution 预注册为 W0-W7=224、W8-W11=48、W12=32 regs/thread，总上界
  64,512/65,536；不在看到结果后试多套预算。静态与动态均须 zero spill；记录 register
  allocation、stack、SMEM、barrier、block/grid 和 max active CTA。416-thread block 若不可 launch、
  产生 stack/local/spill 或超过资源边界则立即 Reject；
- tail、少于两个 output tile、odd tile count、最后 stage drain 和 CTA exit 无死锁。
- 除主 H=2048/16 output tiles 外，protocol smoke 用 H=128/256/384 覆盖 output_tile_cnt=1/2/3，
  zero-iteration由同一状态机的 device micro-smoke覆盖；同时覆盖 4 slices、连续 task切换和 cache
  overwrite canary。它们只验证协议，不进入性能结论。

### Gate B：正确性

- 使用当前 canonical fixture，检查 M=256、M=8192，并覆盖
  `valid_rows=1/31/32/33/63/64/65/95/96/97/127/128`；
- 与 Baseline 使用相同 oracle/tolerance，检查 finite/sentinel、stage ownership、无重复/漏写；
- route/task/source identity、FC2 Tensor work、REDG count 和最终 output shape 必须相同。

### Gate C：实际 overlap

- 只对一个 selected CTA/cluster 做稀疏 phase timestamp/IKET 证据；每个 event绑定同一
  `(CTA, task, slice, output_tile, half, warp role)`，记录 producer publish、consumer begin/end和
  下一 tile producer 区间；
- 必须直接出现 W8-W11 的 `Scatter LDS->REDG(tile j)` 与 W0-W7 的
  `FC2 OMMA(tile j+1)` 时间区间交叠；R2S、TMA preload 或错误 task/slice 的重叠不算通过。
  插桩数据只证明顺序和 overlap，不作为 phase 百分比或性能结论；
- 同时跑未插桩 Candidate，确认 instrumented overlay 没改变 binary/resource 到产生 spill 或改变
  warp roles；若插桩扰动过大，降低采样密度，不从扰动 trace 推 latency。

### Gate D：未插桩性能

- 同一空闲 5KP、同一进程和固定环境做 paired ABBA；M8192 为主 case，M256 为 guard；
- 每臂 warmup 5 次；按 `ABBA/BAAB/ABBA` 三组顺序，每个位置 50 次 replay。保存 raw samples、
  每组 arm median/CV、speedup；Accept 要求 M8192 三组 speedup 全为正且总体 median speedup
  `>=1.0%`；M256 median regression 不得低于 `-1.0%`；任一 arm CV `>1.5%` 的整组作废重测。
  必要时只采一次最小 NCU 验证 FC2/Scatter work、resource、occupancy 和 spill，不做无目标
  deep profile。

## 固定比较身份

- Shape：`E=256, H=2048, I_tp=512, topk=8, NVFP4 weight, BF16 input/output`；baseline
  launch grid `(1,1,110)`、block `(288,1,1)`，Candidate grid固定 `(1,1,110)`、block
  `(416,1,1)`。
- Fixture seed=2026；manifest SHA256
  `683ec75341e4d8317dfdc5c4b04229f9695f9aa286d575c4f6e1fdef55d90801`；M256
  `86b505097acd06bed5a50c3528c78525e6087c07ed69f86606607599ffa21686`；M8192
  `c113ecd5ddeff77154ddbd23fc3dc3c83f8ee822e880179ca5c16b1145372438`。
- Host `10.6.142.16`、GPU UUID `GPU-2fdb0b79-0ba7-f356-b714-6c461b71ce12`、application
  clock 2377 MHz、driver 580.95.05；foreign process 或 thermal/clock drift 时整组无效。
- Container `nvcr.io/nvidia/pytorch:26.05-py3`，image ID
  `sha256:a4e056e1d34a5cc9387512ffa3abeed778e3dc7966633c5154d771705d8835ac`，nvcc
  13.2.78，CUTLASS commit `b46b16d003484063bca4ed365e44095c4c6ed633`，python deps SHA256
  `32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74`。
- 两臂使用各自 fresh JIT root；capture 前冻结 source/JIT/cubin hash，capture中 artifact set不得漂移。

## 判定

- **Accept**：全部 correctness/identity/zero-spill gate 通过；证据显示真实跨 tile overlap；M8192
  paired benchmark 可重复正收益且 M256 不回退。
- **Reject**：资源不可行、deadlock/正确性失败、Route/Q0/FC2/REDG work 漂移、没有真实 overlap、
  产生 spill，或未插桩性能没有正收益。
- **Unresolved**：overlap 与性能方向冲突；只补区分同步开销、consumer starvation 或资源下降所需的
  最小证据，不把实验扩大为完整 bottleneck report。

Accept 后才把设计束移入 Opt；Reject 时不修改当前 Opt。

## 产物

```text
exp_023_fc2_scatter_stage2_overlap/
  plan.md
  comparison_registry.json
  build_candidate.py
  results/
    result.md
    manifest.json
    raw/
```

复用现有 correctness/benchmark/IKET 工具；不为每个 gate 新写 runner，不保存 JIT cache、临时
cubin 或重复采样产物进 git。

## Plan Review

**Date**: 2026-07-21
**Reviewer**: subagent `/root/exp023_plan_review`
**Verdict**: `⚠️ Gaps`（以上缺口已一次性修正；按 single-round 规则不复审）

- 锁定合法的 W0-W7/W8-W11/W12 role、register budget，并隔离 MMA setup；
- Route/Q0 保留 288-thread virtual stride，新增 warps不得访问 route cache；
- 明确 seq/stage/epoch、full/empty participants、proxy fence、reuse与 task-end drain；
- 将比较不变量从 physical warp ownership修正为 logical output/REDG work；
- 增加 output_tile_cnt=0/1/2/3、4-slice、task/cache切换 protocol smoke；
- IKET overlap event绑定同 CTA/task/slice/tile/half，并限定 OMMA 与 LDS->REDG区间；
- 冻结环境、binary identity和三组 paired性能阈值。
