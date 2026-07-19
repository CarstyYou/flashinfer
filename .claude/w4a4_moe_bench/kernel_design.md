# SM120 Dynamic W4A4 Fused MoE Kernel Design

本文描述当前 `MoEDynamicKernel` 的实际实现。逻辑顺序和角色来自源码静态审阅；本文不把 profile 数据解释为 phase 耗时或瓶颈结论。

主要源码：

- `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py`
- `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py`
- Source identity: `flashinfer @ 748ad45594f5e701cbbdca59c60335f39d1c3b2f`
- 当前源码 API：`[API] MmaMXF4NVF4Op`。
- 当前 canonical exp_002 binary 的反汇编确认实际指令为 `[SASS] OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`；证据见 [sass.txt](experiments/exp_002_fused_vs_chain_dataflow/results/ncu/m8192/cutedsl_bf16_fused/deep-resource_launch_1_r2/binary/sass.txt)，artifact 对应 `flashinfer @ 074d93e4aa54c75bee1b3dfdb39b7f075a3ff2af`，SASS SHA-256 为 `2ede628ea1c639e817cf48814a7273386c6b9ba777bd63eceda8d3955b624f7c`。上述两个 commit 之间相关 kernel/dispatch 源码未变化。

CUTLASS 多-kernel baseline 见 [cutlass_fused_moe_pipeline.md](cutlass_fused_moe_pipeline.md)。

## Scope

- 输入：`X BF16 [M, H]`、`topk_ids [M, 8]`、`topk_weights [M, 8]`。
- 权重：FC1 Gate/Up 与 FC2 Down 均为 NVFP4，并带 block scale。
- 输出：`Y BF16 [M, H]`。
- 目标 shape：`H=2048`、`I_tp=512`、`E=256`、`top_k=8`。
- CTA：160 threads，即 4 个 MMA warp + 1 个 TMA warp；目标 occupancy 为 1 CTA/SM。
- MMA tile：`BM × BN × BK = 128 × 128 × 128`。
- Router、Softmax 和 Top-K 不在本 kernel 中；kernel 直接接收路由结果。

当前 benchmark 的 `w1_alpha` 是 `[E]`，因此 `share_input_across_experts=False`。Route/pack 阶段会按 routed pair 分别量化输入。

## 为什么当前是 4 个 MMA warp

### 已确认的事实

- 当前 CTA 使用一组 4 个 MMA warp：Warp 0–3 负责计算，Warp 4 负责 TMA；对应 `num_mma_warps=4`、`atom_layout=(2, 2, 1)`。
- dynamic kernel 由原始 static per-slice microkernel 保守演进而来，沿用了同一套 4-warp Gate/Up/FC2 compute body；源码将其明确标为 first implementation pass，而不是已经完成 warp-layout 调优的最终设计。
- 4 warp 不是 `128 × 128` tile 或 `MmaMXF4NVF4Op` 的硬性要求。同仓库 dense SM120 block-scaled GEMM 对 `128 × 128` tile 使用 8 个 MMA warp 和 `(4, 2, 1)` atom layout。
- 现有源码、提交说明和 PR 说明都没有给出“为何最初选择 4 warp”的直接解释。因此，不能把“为了照顾 GEMM 前后的 memory/latency-bound 操作”当成已证实的设计动机。

### 与前后 phase 的实际耦合

当前 4 个 MMA warp 被复用于整条 task 内数据流，而不只执行 GEMM：

```text
P3 Route/Q0/Pack:  W0–W4 共同分配 routed-pair 工作
T0 Metadata cache: W0–W3 各缓存 32 rows；W4 只参与同步
T1/T2 FC1:         W0–W3 tiled-MMA；W4 TMA
T3 SwiGLU/Q1:      W0–W3 持有 Gate/Up fragments 并完成 activation/requant；W4 预取 Down
T4 FC2/scatter:    W0–W3 tiled-MMA，并各负责一个 64×64 scatter quadrant；W4 TMA
```

因此改成 8-warp tiled-MMA layout 不只是增加 GEMM threads：需要更新 CTA 大小、TMA warp id、MMA atom layout 和 barrier participant count，并重新审计或按需调整 T0 metadata ownership、T3 Q1 分工；其中 T4 硬编码的四象限 epilogue/scatter mapping 必须处理。若选择两个独立的软件子组，还会额外引入 group-local partition、pipeline 和 handoff 重构。这里是**实现耦合**，但不是“前后操作必然要求只能使用 4 warp”。

### 当前判断与代价

最稳妥的解释是：首版优先复用一组 4-warp consumer 贯穿 `FC1 → SwiGLU/Q1 → FC2 → scatter`，以降低 fused control-flow、handoff 和同步设计复杂度；这是基于源码结构的推断，不是作者明确记录的原始动机。

这个选择的直接代价是每线程 accumulator ownership 偏大。一个 `128 × 128` FP32 accumulator 平均对应约 128 values/thread；Gate 完整保活到 Up 完成时，逻辑上约有 256 个 Gate+Up FP32 values/thread，再叠加 fragments、地址和控制状态，很容易超过单线程寄存器预算并触发 spill。增加到 8 个 MMA warp 可把 accumulator ownership 约减半，因此是后续实验的合理方向；但它是否带来净收益仍需同时验证 spill、occupancy、TMA/consumer pipeline，以及前后 phase 的额外线程与同步成本。

相关实验演进：[exp_005 N128 8-warp](experiments/exp_005_8warp_spill_reduction/results/result.md) → [exp_007 双 N64 zero-spill](experiments/exp_007_native_n64_spill_reduction/results/result.md) → [exp_008 Gate/Up 配对复用](experiments/exp_008_branch_paired_n64_reuse/results/result.md)。实验 kernel 不改变本文对 Production source 的描述。

## Task and tile contract

一个 compute task 是：

```text
task = (expert e, physical M tile m, intermediate slice s, valid_rows)

M tile              = 最多 128 个 expert-major routed rows
intermediate slice  = 128 columns
```

对当前 shape：

```text
FC1 reduction tiles       = H / 128 = 16
intermediate slices       = I / 128 = 4
FC2 output tiles per task = H / 128 = 16
```

一个 task 只处理一个 `I=128` slice：先生成 `Gate/Up [M128, I128]`，再用该 slice 扫过 16 个 FC2 output tiles。`I=512` 因此需要 4 个 task；每个 task 产生 FC2 partial，最后通过 route-weighted atomic scatter 汇入同一个 token output。

T4 epilogue/scatter 中，Warp 0–3 共同覆盖一个 `128 × 128` output tile，各自负责一个 `64 × 64` quadrant：

```text
Q00 = Warp 0: M[  0: 64], N[  0: 64]
Q01 = Warp 1: M[  0: 64], N[ 64:128]
Q10 = Warp 2: M[ 64:128], N[  0: 64]
Q11 = Warp 3: M[ 64:128], N[ 64:128]
```

## One-launch phase map

当前实际顺序为：

```text
P0 Clear
  → grid barrier 1
P1 Expert histogram
  → grid barrier 2
P2 Expert tile prefix
  → grid barrier 3
P3 Route + FC1-input quantize + expert-major pack
  → grid barrier 4
P4 Materialize all tasks
  → grid barrier 5
P5 Consume tasks:
     Claim/cache
       → FC1 Gate
       → FC1 Up
       → SwiGLU + FC2-input requant
       → FC2 output-tile sweep + weighted atomic scatter
       → next slice/task
```

`full_tile_publish_enabled` 当前固定为 `0`。因此 P3/P4 完成并经过 grid-wide rendezvous 后，P5 才开始；route/pack 与 compute 当前没有交叠。

## Vertical warp-phase hardware dataflow

![Current source-derived warp-phase overview](diagrams/w4a4_moe_fused_overview.svg)

[可编辑 Draw.io](diagrams/w4a4_moe_fused_overview.drawio)

时间沿表格向下推进，Warp 0–4 固定为列，因此整次 launch 不再拆成两个 panel。每一行表示逻辑 phase，不表示实测耗时。同一行只对齐各 warp 的角色，不表示 lockstep、同一 tile 或必然 overlap；顺序由表中的 barrier/pipeline 建立。

- `[G]`：logical GMEM address space，HBM-backed / L2-coherent；普通 load 还可能使用 L1，但实际 cache residency/hit 必须由 profile 确认。
- `[S]`：CTA shared memory；`[R]`：thread registers / MMA fragments and accumulators。
- `LD/ST/ATOM`：load-store / atomic memory path；`TMA`：异步 `[G] → [S]` 搬运。
- `TC/MMA`：源码为 `[API] MmaMXF4NVF4Op`；canonical exp_002 binary 为 `[SASS] OMMA.SF...`。`CUDA ALU/SFU` 是非 MMA 算术，具体 issue-pipe mix 需由 SASS/profile 确认。
- TMA pipeline：Warp 4 `producer_acquire → TMA → commit/advance`；Warp 0–3 `consumer_wait → [S]→[R] fragment loads → release/advance → MMA from [R]`。它是 stage coordination，不是 CTA-wide barrier。
- `Q0`：FC1-input quantization；`Q1`：SwiGLU 后的 FC2-input requantization。

| Time / phase ↓ | Shared phase dataflow: storage → engine → storage | Warp 0 | Warp 1 | Warp 2 | Warp 3 | Warp 4 |
|---|---|---|---|---|---|---|
| **P0 Clear**<br>→ grid B1 | `0 [R] ─ST→ routing/queue/Y [G]` | grid-stride clear；grid tid 0 另清 queue heads | grid-stride clear | grid-stride clear | grid-stride clear | grid-stride clear |
| **P1 Histogram**<br>→ grid B2 | `topk_ids [G] ─LD→ expert [R] ─ATOM→ row_counts [G]` | count routed pairs | same | same | same | same |
| **P2 Prefix**<br>→ grid B3 | `row_counts [G] ─LD→ tile_acc [R] ─CUDA prefix→ base [R] ─ST→ expert_tile_base [G]` | `*` grid tid 0 serial scan | wait | wait | wait | wait |
| **P3a Claim route batch** | `pair_head [G] ─ATOM(CTA leader)→ batch_base [R] ─ST→ ctrl [S] ─CTA sync/LD(all warps)→ [R]` | up to 2 candidate pairs | up to 2 candidate pairs | up to 2 candidate pairs | up to 2 candidate pairs | up to 2 candidate pairs |
| **P3b Route + Q0 + Pack**<br>repeat P3a/P3b<br>→ fence + grid B4 | metadata: `route tuple [G] ─LD→ [R]`；lane 0 `expert_write_rows [G] ─ATOM→ physical row [R]`，then `token/weight [R] ─ST(indexed by row)→ map/weight [G]`<br>Q0: `X BF16 [G] ─LD→ FP32 block [R] ─CUDA quantize→ FP4/SFA [R] ─ST→ packed A/SFA [G]` | process assigned pairs | same | same | same | same |
| **P4 Publish tasks**<br>→ grid B5 | `row_counts/base [G] ─LD→ [R] ─CUDA build→ descriptors [R] ─ST→ task arrays/tail [G]` | `*` CTA thread 0 publishes owned tasks | wait | wait | wait | wait |
| **T0 Claim + cache** | `task [G] ─LD(W0 lane 0)→ [R] ─ST→ ctrl [S] ─LD(all warps)→ task args [R]`<br>`map/weight [G] ─LD(W0–3)→ [R] ─ST→ scatter cache [S]` | `*` pop task；cache rows 0..31 | cache rows 32..63 | cache rows 64..95 | cache rows 96..127 | no metadata cache row；CTA sync |
| **T1 FC1 Gate**<br>`K128 ×16` | `[G] A/SFA + [G] Gate W/SFB ─TMA(W4)→ [S] ─LD(W0–3)→ fragments [R] ─TC/MMA→ gate_acc FP32 [R]` | tiled-MMA member 0 | member 1 | member 2 | member 3 | TMA A/SFA + Gate W/SFB |
| **pass_gate** | `gate_acc` 保留在 `[R]`；handoff 后允许复用 Gate staging buffers | arrive, no wait | arrive, no wait | arrive, no wait | arrive, no wait | wait |
| **T2 FC1 Up**<br>`K128 ×16` | `[G] A/SFA + [G] Up W/SFB ─TMA(W4)→ sA/sSFA + sB_up/sSFB_up [S] ─LD(W0–3)→ fragments [R] ─TC/MMA→ up_acc FP32 [R]` | tiled-MMA member 0 | member 1 | member 2 | member 3 | TMA Up；随后进入 phase2 producer |
| **T3a SwiGLU + store** | `gate/up acc [R] ─CUDA ALU/SFU→ activation FP32 [R] ─convert→ BF16 [R] ─ST→ sC [S]`<br>可交叠窗口：`Down W/SFB [G] ─TMA(W4)→ reused Gate sB/sSFB [S]` | fragment activation | same | same | same | phase2 producer；ahead ≤ stage capacity / stalled |
| **epilog_sync A** | `sC [S]` producer→consumer handoff | arrive + wait | arrive + wait | arrive + wait | arrive + wait | excluded；phase2 producer ahead/stalled |
| **T3b Q1** | `BF16 sC [S] ─LD→ FP32 block [R] ─CUDA Q1→ FP4/SFA [R] ─ST→ sA/sSFA [S]` | cooperative `qidx 0..31 + 128k` | `32..63 + 128k` | `64..95 + 128k` | `96..127 + 128k` | phase2 producer；ahead ≤ stage capacity / stalled |
| **epilog_sync B** | `sA/sSFA [S]` producer→FC2-consumer handoff | arrive + wait | arrive + wait | arrive + wait | arrive + wait | excluded；phase2 producer ahead/stalled |
| **T4a Hoist FC2 A**<br>once/task | `FP4 sA/sSFA [S] ─LD(W0–3)→ A/SFA fragments [R]`；跨 16 个 output tiles 复用 | hoist member 0 | member 1 | member 2 | member 3 | phase2 producer；ahead ≤ stage capacity / stalled |
| **T4b FC2 tile `j`**<br>`j=0..15` | `Down W/SFB [G] ─TMA(W4)→ sB/sSFB [S] ─LD(W0–3)→ B/SFB fragments [R]`<br>`A/B fragments [R] ─TC/MMA→ down_acc FP32 [R]` | tiled-MMA member 0 | member 1 | member 2 | member 3 | phase2 producer；may be ahead/stalled/done/wait |
| **T4c Scale + store `j`** | `down_acc [R] ─CUDA scale/convert→ BF16 [R] ─ST→ sC [S]` | epilogue member 0 | member 1 | member 2 | member 3 | phase2 producer；may be ahead/stalled/done/wait |
| **epilog_sync pre-scatter** | `sC [S]` producer→scatter-consumer handoff | arrive + wait | arrive + wait | arrive + wait | arrive + wait | excluded；phase2 producer may be ahead/stalled/done/wait |
| **T4d Weighted scatter `j`** | `token id + route weight [S] ─LD→ [R]`；token id selects Y address，`sC [S] ─LD→ values [R] ─CUDA weight multiply→ [R] ─ATOM→ Y [G]` | scatter Q00 | scatter Q01 | scatter Q10 | scatter Q11 | phase2 producer；may be ahead/stalled/done/wait |
| **epilog_sync post-scatter**<br>→ next `j` | 当前 tile 的 scatter 完成；Warp 0–3 collective consumer 可进入下一 tile | arrive + wait | arrive + wait | arrive + wait | arrive + wait | excluded；phase2 producer may be ahead/stalled/done/wait |
| **pass_final**<br>→ next slice/task | on-chip task data lifetime ends；下一轮可覆盖 `sA/sSFA` 等 buffers | arrive, no wait | arrive, no wait | arrive, no wait | arrive, no wait | wait；then next task |

`*` 是 leader-only 工作：P2 只由整个 grid 的 `flat_tid==0` 执行；P4 由各 CTA 的 thread 0（Warp 0 lane 0）写其负责的 tasks，`flat_tid==0` 写 `task_tail`；T0 由 CTA leader claim task。

Warp 0–3 合作完成一个 `128 × 128` tiled-MMA，不是 4 次 GEMM。Q1 是 128 个 MMA threads 按 flattened `(row, 16-value block)` 的线性交错分工；只有 T4 scatter 的 `Q00..Q11` 是固定的 `64 × 64` spatial quadrant。

SMEM buffer lifetime：`sA/sSFA` 先作为 T1/T2 FC1 input staging，T3 被 Q1 覆写成 FC2 activation，T4 再读取并跨 16 个 N tiles 复用；`sB/sSFB` 从 Gate staging 转为 Down staging，Down 不是独立 buffer；`sB_up/sSFB_up` 只服务 Up；`sC` 分别承担 T3 activation→Q1 和 T4 epilogue→scatter 的 handoff。

## Phase mechanics

### P0–P2: initialize routing state

- 全部 threads grid-stride 清零 routing counters、queue state 和最终输出。
- 全部 threads 对 `M × top_k` routed pairs 做 expert histogram。
- 仅 `flat_tid==0` 对 expert row counts 做 padded physical-tile prefix。

### P3: route, quantize, and pack

每个 warp 每批最多处理 2 个 routed pairs：

```text
lane 0:
    expert row allocation
    token_map / token_weight write

all 32 lanes:
    X BF16, 16 values/block
      → block max + input_global_scale[e]
      → NVFP4 values + E4M3 SFA
      → expert-major packed A/SFA workspace
```

这里产生 compute 所需的 packed FC1 input，但它会先完整写入 GMEM workspace。

### P4: materialize the task queue

CTA leaders 按 expert 的 physical M tiles 创建 task descriptors。当前每个 `(expert, M128 tile)` 创建 4 个 `I128` slice tasks，然后写定 `task_tail`，所有 CTA 经过最后一次 resident-grid barrier 后进入 consumer loop。

### T0: claim task and cache scatter metadata

- CTA leader 从全局 queue 获取一个 task，并把 task fields 广播到 shared control block。
- Warp 0–3 各缓存 32 行 `token_map` 和 `route_weight` 到 SMEM。
- Warp 4 不搬运这 128 行 metadata，只参与 CTA synchronization 并读取 task arguments。

### T1: FC1 Gate

对 16 个 `H-K128` tiles：

```text
Warp 4:   A/SFA GMEM + Gate W/SFB GMEM → TMA → SMEM
Warp 0–3: SMEM → register fragments → NVFP4 block-scaled MMA
                                         → gate_acc FP32 RMEM
```

Gate 完成后 Warp 0–3 对 `pass_gate` arrive 而不等待；Warp 4 等待所有 MMA warp 完成后才复用相应 shared buffers。

### T2: FC1 Up

对同样的 16 个 `H-K128` tiles：

```text
Warp 4:   A/SFA GMEM + Up W/SFB GMEM → TMA → SMEM
Warp 0–3: SMEM → register fragments → NVFP4 block-scaled MMA
                                         → up_acc FP32 RMEM
```

Warp 4 随后转入 Down weight producer pipeline。

### T3: SwiGLU and FC2-input requant

Warp 0–3 执行：

```text
gate_acc FP32 RMEM + up_acc FP32 RMEM
    → alpha[e] scaling
    → SwiGLU FP32 RMEM
    → BF16 sC SMEM
    → 16-value block requant using fc2_input_scale[e]
    → FP4 sA + E4M3 sSFA SMEM
```

该阶段没有完整 intermediate tensor 写入 GMEM，但包含一次 `RMEM → BF16 SMEM → RMEM quantize → FP4/SFA SMEM` 的 shared-memory round trip。

### T4: FC2 sweep and route-weighted combine

FC2 activation A/SFA 从 SMEM 提升到 register fragments 一次，并跨 16 个 output-N tiles 复用。每个 output tile：

```text
Warp 4:   Down W/SFB GMEM → TMA → SMEM
Warp 0–3: A/SFA registers + Down W/SFB
            → NVFP4 block-scaled MMA
            → down_acc FP32 RMEM
            → w2_alpha[e] scaling
            → BF16 sC SMEM
            → read one 64×64 quadrant
            → multiply cached route weight
            → BF16 vector atomic-add to Y GMEM
```

Atomic scatter 同时汇合 4 个 intermediate slices 和 top-8 expert contributions，因此没有完整 FC2 partial-output workspace。

## Synchronization boundaries

- 5 个 resident-grid barriers：清零、histogram、prefix、route/pack、task publication 之间的全 grid 边界。
- `pass_gate`：Warp 0–3 arrive，Warp 4 wait。
- `epilog_sync`：仅 Warp 0–3 参与，用于 activation/requant 和每个 FC2 output tile 的 shared-memory handoff。
- `pass_final`：Warp 0–3 完成全部 FC2/scatter 后 arrive，Warp 4 wait；随后 shared A buffers 才能服务下一个 slice/task。

## Source map

| Area | Source lines |
|---|---|
| Tile, warp roles, barriers | `moe_dynamic_kernel.py:269-324` |
| One kernel launch | `moe_dynamic_kernel.py:629-689` |
| P0 clear | `moe_dynamic_kernel.py:963-1022` |
| P1/P2 histogram and prefix | `moe_dynamic_kernel.py:1024-1055` |
| P3 route/quant/pack | `moe_dynamic_kernel.py:1057-1424` |
| P4 deferred task publication | `moe_dynamic_kernel.py:1425-1486` |
| T0 task claim/cache | `moe_dynamic_kernel.py:1751-1840` |
| T1 Gate consumers | `moe_dynamic_kernel.py:1895-2034` |
| T2 Up consumers | `moe_dynamic_kernel.py:2036-2167` |
| T3 SwiGLU/requant | `moe_dynamic_kernel.py:2168-2302` |
| T4 FC2/scatter consumers | `moe_dynamic_kernel.py:2304-2549` |
| Warp 4 Gate/Up/Down TMA | `moe_dynamic_kernel.py:2551-2704` |

源码文件头仍描述 ready-task overlap，但当前行为应以 `full_tile_publish_enabled = 0`（`moe_dynamic_kernel.py:961`）以及 P3/P4 的 resident-grid barriers 为准。FC2 output tile 数由 shape 动态决定；当前 `H=2048` 时是 16。
