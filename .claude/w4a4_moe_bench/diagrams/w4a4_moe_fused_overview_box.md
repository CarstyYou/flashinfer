# W4A4 Fused MoE — Text Box Overview

This is the text-box alternative to the editable Draw.io overview. Time flows downward; box height is not measured duration.

```text
Legend: [G] GMEM   [S] SMEM   [R] RMEM
        ✓SRC = source/API/PTX confirmed   △SASS = emitted machine mnemonic still needs disassembly

                                      logical time ↓

┌─ P0–P2: one-launch setup ────────────────────────────────────────────────────────────────┐
│ P0  [R] zero ──ST.global──▶ [G] routing / queue / Y                  → grid B1          │
│ P1  [G] topk_ids ──LD + ATOM.global──▶ [G] row_counts               → grid B2          │
│ P2  [G] row_counts ──LD──▶ [R] CUDA prefix ──ST.global──▶ [G] base  → grid B3          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─ P3: Route + Q0 + Pack ──────────────────────────────────────────────────────────────────┐
│ [G] X BF16 + route tuple + expert_write_rows                                             │
│     ┌─ repeat until routed pairs exhausted ───────────────────────────────────────────┐  │
│     │ CTA W0/L0: ATOM pair_head → [R] batch_base → ST.shared ctrl[S] → CTA sync       │  │
│     │ Warp 0–4, each ≤2 candidate pairs:                                               │  │
│               [G] X ──global load──▶ [R] 16×FP32                                        │
│                       ──CUDA Q0──▶ [R] packed FP4 u64 + E4M3 byte                        │
│                       ──ST.global──▶ [G] packed A/SFA                                    │
│               lane0: ATOM row allocation + ST.global token_map/route_weight              │
│     └───────────────────────────────────────────────────────────────────────────── ↺ ──┘  │
│ after loop: threadfence → grid B4. Q0 main data path does not stage through SMEM.         │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─ P4 + T0: publish and claim compute work ─────────────────────────────────────────────────┐
│ P4  each CTA W0/L0: [G] counts/base ──CUDA build + ST.global──▶ [G] task arrays          │
│     grid tid 0 ──ST.global──▶ [G] task_tail                         → grid B5              │
│ T0  [G] task ──LD──▶ [R] ──ST.shared──▶ [S] ctrl                                       │
│     ═════════════════ CTA sync 1: task ctrl published ═══════════════════════════════    │
│     [S] ctrl ──LD──▶ [R] has_task/is_done + cache bounds                                │
│     [G] token_map/weight ──LD + ST.shared──▶ [S] scatter cache                          │
│     W0/W1/W2/W3 cache rows 0–31 / 32–63 / 64–95 / 96–127; W4 carries no cache rows       │
│     ═════════════════ CTA sync 2: cache ready; split roles ══════════════════════════    │
│     [S] ctrl ──LD──▶ [R] full task args in W0–3 compute / W4 TMA roles                  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                             │ role split
                                             ▼
┌─ T1: FC1 Gate, K128 ×16 ─────────────────────────────────────────────────────────────────┐
│ Warp 4 producer                              Warp 0–3 cooperative consumers               │
│ ┌────────────────────────────┐              ┌──────────────────────────────────────────┐ │
│ │ [G] packed A/SFA           │              │ [S] sA/sB ──LdMatrix atom ✓SRC────────┐ │ │
│ │ [G] Gate W/SFB             │              │             LDSM variant △SASS         │ │ │
│ │          │                 │              │ [S] SFA/SFB ──CopyUniversal shared LD─┤ │ │
│ │          ▼ TMA ✓SRC        │              │                                         ▼ │ │
│ │ [S] sA/sSFA+sB/sSFB stages├─────────────▶│ [R] A/B/SF fragments                     │ │
│ └────────────────────────────┘              │          │ MmaMXF4NVF4Op / Tensor Core     │ │
│                                             │          ▼                                │ │
│                                             │ [R] gate_acc FP32                         │ │
│                                             └──────────────────────────────────────────┘ │
└──────────────────────────── pass_gate: W0–3 arrive; W4 wait ─────────────────────────────┘
                                             │
                                             ▼
┌─ T2: FC1 Up, K128 ×16 ───────────────────────────────────────────────────────────────────┐
│ Warp 4 producer                              Warp 0–3 cooperative consumers               │
│ ┌────────────────────────────┐              ┌──────────────────────────────────────────┐ │
│ │ [G] packed A/SFA           │              │ [S] sA/sB_up ──LdMatrix atom───────────┐ │ │
│ │ [G] Up W/SFB               │              │ [S] SFA/SFB_up ──CopyUniversal────────┤ │ │
│ │          │                 │              │                                         ▼ │ │
│ │          ▼ TMA ✓SRC        │              │ [R] A/B/SF fragments                     │ │
│ │ [S] sA/sSFA               ├─────────────▶│          │ Tensor Core / QMMA             │ │
│ │ [S] sB_up/sSFB_up          │              │          ▼                                │ │
│ └────────────────────────────┘              │ [R] up_acc FP32                           │ │
│                                             └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                    ┌────────────────────────┴──────────────────────────────┐
                    ▼                                                       ▼
┌─ T3: Warp 0–3 SwiGLU + Q1 ───────────────────────────┐   ┌─ W4 Down producer rail ───────┐
│ [R] gate_acc + up_acc                                │   │ [G] Down W/SFB                │
│          │ CUDA ALU/SFU: alpha + SwiGLU              │   │          │                    │
│          ▼                                            │   │          ▼ TMA ✓SRC           │
│ [R] activation FP32 → BF16                           │   │ [S] reused Gate sB/sSFB      │
│          │ CopyUniversal R→S ✓SRC                    │   │                               │
│          │ StMatrix-derived mapping; STSM? △SASS     │   │ Before FC2 consumption:      │
│          ▼                                            │   │ ahead ≤ stages or stalled    │
│ [S] sC BF16                                           │   │                               │
│ ═══════════ epilog_sync A · W0–3 only ═════════════  │   │ continues through T4 to      │
│          │ LD.shared                                  │   │ pass_final; not lockstep     │
│          ▼                                            │   └───────────────────────────────┘
│ [R] 16-value blocks ──CUDA Q1──▶ [R] FP4/SFA         │
│ ownership: W0 0..31+128k | W1 32..63+128k | W2 64..95+128k | W3 96..127+128k            │
│          │ generic shared byte stores + st.shared.u8  │
│          ▼                                            │
│ [S] FP4 sA + E4M3 sSFA                               │
│ ═══════════ epilog_sync B · W0–3 only ═════════════  │
└───────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─ T4: FC2 + route-weighted scatter ────────────────────────────────────────────────────────┐
│ Hoist once/task:                                                                         │
│   [S] sA ──LdMatrix atom ✓SRC; LDSM? △SASS──▶ [R] A fragments                           │
│   [S] sSFA ──CopyUniversal shared load──────────────▶ [R] SFA fragments                 │
│                                                                                          │
│ ┌─ repeat output tile j = 0..15 ────────────────────────────────────────────────────────┐ │
│ │ W4 producer rail: [G] Down W/SFB[j] ──TMA──▶ [S] sB/sSFB[j]                         │ │
│ │     bounded dependency, not tile-by-tile lockstep; may run ahead, stall, or finish    │ │
│ │ W0–3:                                                                                │ │
│ │   [S] sB ──LdMatrix atom──▶ [R] B fragments                                          │ │
│ │   [S] sSFB ──CopyUniversal──▶ [R] SFB fragments                                     │ │
│ │   [R] A/B/SF ──MmaMXF4NVF4Op / Tensor Core──▶ [R] down_acc FP32                     │ │
│ │   [R] down_acc ──CUDA scale + convert──▶ [R] BF16                                   │ │
│ │         ──CopyUniversal R→S; STSM? △SASS──▶ [S] sC BF16                            │ │
│ │   ═════════════════ epilog_sync pre-scatter · W0–3 only ═════════════════            │ │
│ │   [S] token/weight+sC ──LD.shared──▶ [R] ──CUDA weight multiply──▶ [R] FP32          │ │
│ │         ──4×cvt.rn.satfinite.bf16x2.f32                                              │ │
│ │         ──PTX red.global.add.noftz.v4.bf16x2 ✓SRC──▶ [G] Y                          │ │
│ │   Scatter ownership: W0 Q00 | W1 Q01 | W2 Q10 | W3 Q11                              │ │
│ │   ═════════════════ epilog_sync post-scatter · next j ═══════════════════            │ │
│ └──────────────────────────────────────────────────────────────────────────────────────┘ │
└────────────── pass_final: W0–3 arrive/no-wait; W4 waits → next slice/task ───────────────┘
                                             │
                                             └──▶ next slice/task
```

The Draw.io/SVG version uses the same facts and evidence labels; this file intentionally favors source-level readability over precise visual alignment.
