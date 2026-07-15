# SM120 Dynamic W4A4 Fused MoE Kernel Design

本文描述当前 `MoEDynamicKernel` 的实际实现。内容来自源码静态审阅，尚未加入运行时 profile 结论。

主要源码：

- `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py`
- `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py`
- Source identity: `flashinfer @ 517cca9c2e7d91f524fcb5f078370c056308d461`

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

时间沿表格向下推进，Warp 0–4 固定为列，因此整次 launch 不再拆成两个 panel。每一行表示逻辑 phase，不表示实测耗时。同一行只对齐各 warp 的角色，不表示 lockstep、同一 tile 或必然 overlap；顺序由表中的 barrier/pipeline 建立。

- `[G]`：logical GMEM address space，HBM-backed / L2-coherent；普通 load 还可能使用 L1，但实际 cache residency/hit 必须由 profile 确认。
- `[S]`：CTA shared memory；`[R]`：thread registers / MMA fragments and accumulators。
- `LD/ST/ATOM`：load-store / atomic memory path；`TMA`：异步 `[G] → [S]` 搬运。
- `TC/QMMA`：Tensor Core block-scaled MMA；`CUDA ALU/SFU`：非 QMMA 算术，具体 issue-pipe mix 需由 SASS/profile 确认。
- TMA pipeline：Warp 4 `producer_acquire → TMA → commit/advance`；Warp 0–3 `consumer_wait → [S]→[R] fragment loads → release/advance → QMMA from [R]`。它是 stage coordination，不是 CTA-wide barrier。
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
| **T1 FC1 Gate**<br>`K128 ×16` | `[G] A/SFA + [G] Gate W/SFB ─TMA(W4)→ [S] ─LD(W0–3)→ fragments [R] ─TC/QMMA→ gate_acc FP32 [R]` | QMMA member 0 | member 1 | member 2 | member 3 | TMA A/SFA + Gate W/SFB |
| **pass_gate** | `gate_acc` 保留在 `[R]`；handoff 后允许复用 Gate staging buffers | arrive, no wait | arrive, no wait | arrive, no wait | arrive, no wait | wait |
| **T2 FC1 Up**<br>`K128 ×16` | `[G] A/SFA + [G] Up W/SFB ─TMA(W4)→ sA/sSFA + sB_up/sSFB_up [S] ─LD(W0–3)→ fragments [R] ─TC/QMMA→ up_acc FP32 [R]` | QMMA member 0 | member 1 | member 2 | member 3 | TMA Up；随后进入 phase2 producer |
| **T3a SwiGLU + store** | `gate/up acc [R] ─CUDA ALU/SFU→ activation FP32 [R] ─convert→ BF16 [R] ─ST→ sC [S]`<br>可交叠窗口：`Down W/SFB [G] ─TMA(W4)→ reused Gate sB/sSFB [S]` | fragment activation | same | same | same | phase2 producer；ahead ≤ stage capacity / stalled |
| **epilog_sync A** | `sC [S]` producer→consumer handoff | arrive + wait | arrive + wait | arrive + wait | arrive + wait | excluded；phase2 producer ahead/stalled |
| **T3b Q1** | `BF16 sC [S] ─LD→ FP32 block [R] ─CUDA Q1→ FP4/SFA [R] ─ST→ sA/sSFA [S]` | cooperative `qidx 0..31 + 128k` | `32..63 + 128k` | `64..95 + 128k` | `96..127 + 128k` | phase2 producer；ahead ≤ stage capacity / stalled |
| **epilog_sync B** | `sA/sSFA [S]` producer→FC2-consumer handoff | arrive + wait | arrive + wait | arrive + wait | arrive + wait | excluded；phase2 producer ahead/stalled |
| **T4a Hoist FC2 A**<br>once/task | `FP4 sA/sSFA [S] ─LD(W0–3)→ A/SFA fragments [R]`；跨 16 个 output tiles 复用 | hoist member 0 | member 1 | member 2 | member 3 | phase2 producer；ahead ≤ stage capacity / stalled |
| **T4b FC2 tile `j`**<br>`j=0..15` | `Down W/SFB [G] ─TMA(W4)→ sB/sSFB [S] ─LD(W0–3)→ B/SFB fragments [R]`<br>`A/B fragments [R] ─TC/QMMA→ down_acc FP32 [R]` | QMMA member 0 | member 1 | member 2 | member 3 | phase2 producer；may be ahead/stalled/done/wait |
| **T4c Scale + store `j`** | `down_acc [R] ─CUDA scale/convert→ BF16 [R] ─ST→ sC [S]` | epilogue member 0 | member 1 | member 2 | member 3 | phase2 producer；may be ahead/stalled/done/wait |
| **epilog_sync pre-scatter** | `sC [S]` producer→scatter-consumer handoff | arrive + wait | arrive + wait | arrive + wait | arrive + wait | excluded；phase2 producer may be ahead/stalled/done/wait |
| **T4d Weighted scatter `j`** | `token id + route weight [S] ─LD→ [R]`；token id selects Y address，`sC [S] ─LD→ values [R] ─CUDA weight multiply→ [R] ─ATOM→ Y [G]` | scatter Q00 | scatter Q01 | scatter Q10 | scatter Q11 | phase2 producer；may be ahead/stalled/done/wait |
| **epilog_sync post-scatter**<br>→ next `j` | 当前 tile 的 scatter 完成；Warp 0–3 collective consumer 可进入下一 tile | arrive + wait | arrive + wait | arrive + wait | arrive + wait | excluded；phase2 producer may be ahead/stalled/done/wait |
| **pass_final**<br>→ next slice/task | on-chip task data lifetime ends；下一轮可覆盖 `sA/sSFA` 等 buffers | arrive, no wait | arrive, no wait | arrive, no wait | arrive, no wait | wait；then next task |

`*` 是 leader-only 工作：P2 只由整个 grid 的 `flat_tid==0` 执行；P4 由各 CTA 的 thread 0（Warp 0 lane 0）写其负责的 tasks，`flat_tid==0` 写 `task_tail`；T0 由 CTA leader claim task。

Warp 0–3 合作完成一个 `128 × 128` tiled-QMMA，不是 4 次 GEMM。Q1 是 128 个 MMA threads 按 flattened `(row, 16-value block)` 的线性交错分工；只有 T4 scatter 的 `Q00..Q11` 是固定的 `64 × 64` spatial quadrant。

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
Warp 0–3: SMEM → register fragments → NVFP4 block-scaled QMMA
                                         → gate_acc FP32 RMEM
```

Gate 完成后 Warp 0–3 对 `pass_gate` arrive 而不等待；Warp 4 等待所有 MMA warp 完成后才复用相应 shared buffers。

### T2: FC1 Up

对同样的 16 个 `H-K128` tiles：

```text
Warp 4:   A/SFA GMEM + Up W/SFB GMEM → TMA → SMEM
Warp 0–3: SMEM → register fragments → NVFP4 block-scaled QMMA
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
            → NVFP4 block-scaled QMMA
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
