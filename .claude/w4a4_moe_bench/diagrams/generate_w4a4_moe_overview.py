#!/usr/bin/env python3.10
"""Generate matching Draw.io and SVG artifacts for the W4A4 fused-MoE overview.

The Python node/edge specification is canonical. The generated Draw.io XML is
editable for exploration, and the SVG can be reviewed without Draw.io, but
manual XML edits are overwritten on regeneration.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom


WIDTH = 1800
HEIGHT = 2840
OUT_DIR = Path(__file__).resolve().parent


PALETTE = {
    "phase": ("none", "#334155"),
    "phase_label": ("#F5F5F5", "#444444"),
    "lane": ("none", "#777777"),
    "gmem": ("#E1D5E7", "#9673A6"),
    "smem": ("#DAE8FC", "#6C8EBF"),
    "rmem": ("#D5E8D4", "#82B366"),
    "tma": ("#FFFFFF", "#1D4ED8"),
    "tc": ("#FFFFFF", "#166534"),
    "cuda": ("#FFFFFF", "#92400E"),
    "sync": ("#FFF5F5", "#B85450"),
    "neutral": ("#FFFFFF", "#444444"),
    "note": ("#FFFFFF", "#555555"),
    "rail": ("#1D4ED8", "#1D4ED8"),
    "text": ("none", "none"),
}

EDGE_COLORS = {
    "data": "#475569",
    "tma": "#1D4ED8",
    "tc": "#166534",
    "cuda": "#92400E",
    "atom": "#991B1B",
    "sync": "#B85450",
}


@dataclass(frozen=True)
class Node:
    ident: str
    x: int
    y: int
    w: int
    h: int
    label: str
    kind: str = "neutral"
    font_size: int = 12
    bold: bool = False
    dashed: bool = False
    background: bool = False


@dataclass(frozen=True)
class Edge:
    ident: str
    source: str
    target: str
    label: str = ""
    kind: str = "data"
    direction: str = "tb"
    dashed: bool = False
    width: float = 2.0


NODES: list[Node] = []
EDGES: list[Edge] = []


def add_node(
    ident: str,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    kind: str = "neutral",
    *,
    font_size: int = 12,
    bold: bool = False,
    dashed: bool = False,
    background: bool = False,
) -> None:
    NODES.append(
        Node(
            ident,
            x,
            y,
            w,
            h,
            label,
            kind,
            font_size,
            bold,
            dashed,
            background,
        )
    )


def add_edge(
    ident: str,
    source: str,
    target: str,
    label: str = "",
    kind: str = "data",
    *,
    direction: str = "tb",
    dashed: bool = False,
    width: float = 2.0,
) -> None:
    EDGES.append(Edge(ident, source, target, label, kind, direction, dashed, width))


LANE_X = [190, 500, 810, 1120, 1430]
LANE_W = 300
COMPUTE_X = 190
COMPUTE_W = 1230
W4_X = 1445
W4_W = 270


def phase(ident: str, label: str, y: int, h: int) -> None:
    add_node(
        f"{ident}_outer",
        20,
        y,
        1760,
        h,
        "",
        "phase",
        dashed=False,
        background=True,
    )
    add_node(
        f"{ident}_label",
        32,
        y + 8,
        138,
        38,
        label,
        "phase_label",
        font_size=16,
        bold=True,
        background=True,
    )


def build_spec() -> None:
    def local_row(
        prefix: str,
        y: int,
        h: int,
        labels: list[str],
        kind: str,
        *,
        font_size: int = 13,
    ) -> list[str]:
        result = []
        for warp, (x, label) in enumerate(zip(LANE_X[:4], labels, strict=True)):
            ident = f"{prefix}_w{warp}"
            add_node(
                ident,
                x + 12,
                y,
                LANE_W - 24,
                h,
                label,
                kind,
                font_size=font_size,
                bold=True,
            )
            result.append(ident)
        return result

    def thin_sync(ident: str, y: int, label: str, *, w4: bool = False) -> None:
        width = 1530 if w4 else 1230
        add_node(ident, COMPUTE_X, y, width, 20, label, "sync", font_size=11, bold=True)

    # Title and concise visual legend.
    add_node(
        "title",
        30,
        18,
        1740,
        38,
        "SM120 Dynamic W4A4 Fused MoE — Warp/Hardware Dataflow Overview",
        "text",
        font_size=22,
        bold=True,
        background=True,
    )
    add_node(
        "subtitle",
        30,
        55,
        1740,
        28,
        "Logical order, not measured duration · CTA = 5 warps · tile = 128×128×128",
        "text",
        font_size=13,
        background=True,
    )
    legend = [
        ("leg_g", 55, "[G] GMEM data", "gmem", 170),
        ("leg_s", 245, "[S] SMEM data", "smem", 170),
        ("leg_r", 435, "[R] RMEM data", "rmem", 170),
        ("leg_tma", 625, "TMA", "tma", 150),
        ("leg_tc", 795, "Tensor Core", "tc", 180),
        ("leg_cuda", 995, "CUDA ALU/SFU", "cuda", 200),
        ("leg_sync", 1215, "Barrier", "sync", 160),
        ("leg_cert", 1395, "edges: LDG/LDS/TMA/... · * pending", "note", 250),
    ]
    for ident, x, label, kind, width in legend:
        add_node(
            ident,
            x,
            92,
            width,
            34,
            label,
            kind,
            font_size=12,
            bold=True,
            background=True,
        )

    # Fixed warp lanes. The role labels remain valid in both route and compute phases.
    for warp, x in enumerate(LANE_X):
        add_node(
            f"lane_bg_{warp}",
            x,
            138,
            LANE_W,
            2600,
            "",
            "lane",
            dashed=True,
            background=True,
        )
        role = (
            "route producer / compute member"
            if warp < 4
            else "route producer / T1–T4 TMA"
        )
        add_node(
            f"lane_header_{warp}",
            x + 5,
            142,
            LANE_W - 10,
            48,
            f"Warp {warp}\n{role}",
            "neutral" if warp < 4 else "tma",
            font_size=14,
            bold=True,
            background=True,
        )

    # W4 Down producer rail: one directed dependency with compact stage marks.
    for ident, y, label in [
        ("rail_start", 1433, "start"),
        ("rail_stream", 1745, "stream"),
        ("rail_supply", 2175, "supply j"),
        ("rail_wait", 2677, "wait"),
    ]:
        add_node(ident, 1719, y, 8, 8, "", "rail")
        add_node(f"{ident}_text", 1729, y - 7, 48, 22, label, "text", font_size=9)
    add_edge("e_down_rail", "rail_start", "rail_wait", "", "tma", width=3)

    # P0–P2 are grid-wide prelude phases, not warp-lane ownership.
    phase("setup", "P0–P2", 205, 100)
    add_node(
        "setup_flow",
        190,
        232,
        1530,
        42,
        "P0 clear [R]→[G]  → grid B1 →  P1 histogram [G] IDs→counts  → grid B2 →  P2 CUDA prefix [G] counts→base  → grid B3",
        "neutral",
        font_size=13,
        bold=True,
    )

    # P3: all five warps are route/Q0 producers; loop and B4 are explicit.
    phase("p3", "P3 · Route/Q0", 345, 250)
    add_node(
        "p3_input",
        190,
        365,
        1530,
        34,
        "[G] X BF16 + route tuple + expert_write_rows",
        "gmem",
        font_size=14,
        bold=True,
    )
    add_node(
        "p3_claim", 202, 413, 150, 30, "W0/L0 claim", "cuda", font_size=10, bold=True
    )
    add_node(
        "p3_ctrl", 390, 413, 170, 30, "[S] route ctrl", "smem", font_size=10, bold=True
    )
    add_edge(
        "e_p3_ctrl", "p3_claim", "p3_ctrl", "ATOM + STS", direction="lr", width=1.5
    )
    add_node(
        "p3_loop_note",
        600,
        413,
        680,
        30,
        "↺ CTA sync → pack, until routed pairs exhausted",
        "text",
        font_size=12,
        bold=True,
    )
    p3_r = local_row("p3_r", 448, 28, ["[R] X · ≤2 pairs"] * 4, "rmem", font_size=10)
    add_node(
        "p3_r_w4",
        W4_X,
        448,
        W4_W,
        28,
        "[R] X · ≤2 pairs",
        "rmem",
        font_size=10,
        bold=True,
    )
    q0_nodes = local_row(
        "p3_q0", 484, 34, ["CUDA Q0 + route alloc"] * 4, "cuda", font_size=10
    )
    add_node(
        "p3_q0_w4",
        W4_X,
        484,
        W4_W,
        34,
        "CUDA Q0 + route alloc",
        "cuda",
        font_size=10,
        bold=True,
    )
    for r_ident in p3_r + ["p3_r_w4"]:
        add_edge(f"e_{r_ident}_in", "p3_input", r_ident, "LDG", width=1.3)
    for warp, q_ident in enumerate(q0_nodes + ["p3_q0_w4"]):
        r_ident = p3_r[warp] if warp < 4 else "p3_r_w4"
        add_edge(f"e_{q_ident}_q", r_ident, q_ident, "", "cuda", width=1.5)
    add_node(
        "p3_output",
        190,
        530,
        1530,
        34,
        "[G] packed A/SFA + token_map/route_weight",
        "gmem",
        font_size=14,
        bold=True,
    )
    for ident in q0_nodes + ["p3_q0_w4"]:
        add_edge(f"e_{ident}_out", ident, "p3_output", "STG", width=1.3)
    thin_sync("p3_b4", 569, "threadfence + grid B4 · Q0 payload is in GMEM", w4=True)

    # P4 task materialization.
    phase("p4", "P4 · tasks", 605, 100)
    add_node(
        "p4_build",
        202,
        625,
        280,
        36,
        "each CTA W0/L0 · build descriptors",
        "cuda",
        font_size=11,
        bold=True,
    )
    add_node(
        "p4_tasks",
        520,
        625,
        540,
        36,
        "[G] task arrays",
        "gmem",
        font_size=13,
        bold=True,
    )
    add_node(
        "p4_tail_writer",
        1120,
        625,
        160,
        36,
        "grid tid 0",
        "cuda",
        font_size=11,
        bold=True,
    )
    add_node(
        "p4_tail", 1320, 625, 300, 36, "[G] task_tail", "gmem", font_size=12, bold=True
    )
    add_edge("e_p4", "p4_build", "p4_tasks", "STG", direction="lr")
    add_edge("e_p4_tail", "p4_tail_writer", "p4_tail", "STG", direction="lr")
    thin_sync("p4_b5", 672, "grid B5 · tasks and tail published", w4=True)

    # T0: task-control and scatter-metadata paths are deliberately separate.
    phase("t0", "T0 · claim/cache", 710, 225)
    add_node(
        "t0_task_g",
        202,
        735,
        276,
        30,
        "[G] task descriptor",
        "gmem",
        font_size=12,
        bold=True,
    )
    add_node(
        "t0_ctrl",
        600,
        735,
        600,
        30,
        "[S] task ctrl/status",
        "smem",
        font_size=12,
        bold=True,
    )
    add_edge("e_t0_task", "t0_task_g", "t0_ctrl", "W0/L0 · LDG + STS", direction="lr")
    thin_sync("t0_sync1", 772, "CTA sync 1 · ctrl published", w4=True)
    add_node(
        "t0_meta_g",
        190,
        800,
        1230,
        24,
        "[G] token_map / route_weight",
        "gmem",
        font_size=12,
        bold=True,
    )
    cache = local_row(
        "t0_cache",
        832,
        24,
        [f"[S] cache rows {w * 32}..{w * 32 + 31}" for w in range(4)],
        "smem",
        font_size=10,
    )
    add_node(
        "t0_w4_idle",
        W4_X,
        832,
        W4_W,
        24,
        "no metadata cache rows",
        "neutral",
        font_size=10,
    )
    for ident in cache:
        add_edge(f"e_{ident}", "t0_meta_g", ident, "LDG + STS", width=1.2)
    thin_sync("t0_sync2", 862, "CTA sync 2 · metadata cache ready", w4=True)
    add_node(
        "t0_ctrl_read",
        190,
        887,
        1530,
        16,
        "[S] task ctrl/status",
        "smem",
        font_size=9,
        bold=True,
    )
    add_node(
        "t0_lds_label", 190, 902, 1530, 13, "LDS ↓", "text", font_size=9, bold=True
    )
    args = local_row(
        "t0_args", 916, 16, ["[R] role task args"] * 4, "rmem", font_size=9
    )
    add_node(
        "t0_args_w4",
        W4_X,
        916,
        W4_W,
        16,
        "[R] role task args",
        "rmem",
        font_size=9,
        bold=True,
    )
    for ident in args + ["t0_args_w4"]:
        add_edge(f"e_{ident}_args", "t0_ctrl_read", ident, "", width=1.1)

    # T1 Gate: shared shelf spans W0–W3; register/TC/accumulator boxes are lane-local.
    phase("t1", "T1 · Gate", 945, 260)
    add_node(
        "t1_g",
        W4_X,
        966,
        W4_W,
        42,
        "[G] A/SFA + Gate W/SFB",
        "gmem",
        font_size=12,
        bold=True,
    )
    add_node("t1_tma", W4_X, 1018, W4_W, 34, "TMA · W4", "tma", font_size=13, bold=True)
    add_node(
        "t1_s",
        COMPUTE_X,
        1018,
        COMPUTE_W,
        34,
        "[S] Gate stages · sA/sSFA + sB/sSFB",
        "smem",
        font_size=13,
        bold=True,
    )
    add_edge("e_t1_gt", "t1_g", "t1_tma", "", "tma")
    add_edge("e_t1_ts", "t1_tma", "t1_s", "TMA", "tma", direction="rl", width=3)
    t1_r = local_row("t1_r", 1080, 32, ["[R] A/B/SF frags"] * 4, "rmem", font_size=11)
    t1_tc = local_row(
        "t1_tc",
        1138,
        34,
        [f"MmaMXF4NVF4Op\ntiled-MMA member W{w}" for w in range(4)],
        "tc",
        font_size=9,
    )
    t1_acc = local_row("t1_acc", 1177, 25, ["[R] gate_acc"] * 4, "rmem", font_size=11)
    for warp in range(4):
        add_edge(
            f"e_t1_sr_{warp}", "t1_s", t1_r[warp], "LdMatrix* / LDS(SF)", width=1.3
        )
        add_edge(f"e_t1_rt_{warp}", t1_r[warp], t1_tc[warp], "", "tc", width=2.2)
        add_edge(f"e_t1_ta_{warp}", t1_tc[warp], t1_acc[warp], "", "tc", width=2.2)

    phase("pg", "pass_gate", 1210, 30)
    thin_sync(
        "pg_bar", 1215, "W0–W3 arrive/no-wait  ·  W4 wait before buffer reuse", w4=True
    )

    # T2 Up and start of the independent Down producer rail.
    phase("t2", "T2 · Up", 1250, 260)
    add_node(
        "t2_g",
        W4_X,
        1271,
        W4_W,
        42,
        "[G] A/SFA + Up W/SFB",
        "gmem",
        font_size=12,
        bold=True,
    )
    add_node("t2_tma", W4_X, 1323, W4_W, 34, "TMA · W4", "tma", font_size=13, bold=True)
    add_node(
        "t2_s",
        COMPUTE_X,
        1323,
        COMPUTE_W,
        34,
        "[S] Up stages · sA/sSFA + sB_up/sSFB_up",
        "smem",
        font_size=13,
        bold=True,
    )
    add_edge("e_t2_gt", "t2_g", "t2_tma", "", "tma")
    add_edge("e_t2_ts", "t2_tma", "t2_s", "TMA", "tma", direction="rl", width=3)
    t2_r = local_row("t2_r", 1385, 32, ["[R] A/B/SF frags"] * 4, "rmem", font_size=11)
    t2_tc = local_row(
        "t2_tc",
        1443,
        34,
        [f"MmaMXF4NVF4Op\ntiled-MMA member W{w}" for w in range(4)],
        "tc",
        font_size=9,
    )
    t2_acc = local_row("t2_acc", 1482, 25, ["[R] up_acc"] * 4, "rmem", font_size=11)
    for warp in range(4):
        add_edge(
            f"e_t2_sr_{warp}", "t2_s", t2_r[warp], "LdMatrix* / LDS(SF)", width=1.3
        )
        add_edge(f"e_t2_rt_{warp}", t2_r[warp], t2_tc[warp], "", "tc", width=2.2)
        add_edge(f"e_t2_ta_{warp}", t2_tc[warp], t2_acc[warp], "", "tc", width=2.2)

    # T3 activation/Q1 with lane-local compute and shared shelves.
    phase("t3", "T3 · SwiGLU/Q1", 1520, 430)
    t3_acc = local_row("t3_acc", 1545, 30, ["[R] gate + up"] * 4, "rmem", font_size=11)
    t3_cuda = local_row("t3_cuda", 1592, 40, ["CUDA\nSwiGLU"] * 4, "cuda", font_size=12)
    t3_act = local_row("t3_act", 1648, 28, ["[R] BF16 act"] * 4, "rmem", font_size=11)
    add_node(
        "t3_sc",
        COMPUTE_X,
        1702,
        COMPUTE_W,
        32,
        "[S] sC BF16",
        "smem",
        font_size=13,
        bold=True,
    )
    thin_sync("t3_sync_a", 1740, "epilog_sync A · W0–W3 only")
    qlabels = [
        "[R] 16×FP32 · 0..31+128k",
        "[R] 16×FP32 · 32..63+128k",
        "[R] 16×FP32 · 64..95+128k",
        "[R] 16×FP32 · 96..127+128k",
    ]
    t3_q1_r = local_row("t3_q1_r", 1770, 28, qlabels, "rmem", font_size=9)
    t3_q1 = local_row(
        "t3_q1", 1810, 34, ["CUDA Q1 FP4 quant"] * 4, "cuda", font_size=10
    )
    add_node(
        "t3_sa",
        COMPUTE_X,
        1878,
        COMPUTE_W,
        30,
        "[S] FP4 sA + E4M3 sSFA",
        "smem",
        font_size=13,
        bold=True,
    )
    thin_sync("t3_sync_b", 1914, "epilog_sync B · W0–W3 only")
    for warp in range(4):
        add_edge(f"e_t3_ac_{warp}", t3_acc[warp], t3_cuda[warp], "", "cuda", width=2)
        add_edge(f"e_t3_ca_{warp}", t3_cuda[warp], t3_act[warp], "", "cuda", width=2)
        add_edge(
            f"e_t3_as_{warp}", t3_act[warp], "t3_sc", "R→S CopyUniversal*", width=1.3
        )
        add_edge(f"e_t3_sq_{warp}", "t3_sc", t3_q1_r[warp], "LDS", width=1.3)
        add_edge(f"e_t3_q_{warp}", t3_q1_r[warp], t3_q1[warp], "", "cuda", width=1.5)
        add_edge(
            f"e_t3_qs_{warp}",
            t3_q1[warp],
            "t3_sa",
            "sA: byte store*\nsSFA: PTX st.shared.u8",
            width=1.3,
        )
    add_node(
        "t3_down_g",
        W4_X,
        1550,
        W4_W,
        36,
        "[G] Down W/SFB",
        "gmem",
        font_size=12,
        bold=True,
    )
    add_node(
        "t3_down_tma",
        W4_X,
        1602,
        W4_W,
        34,
        "TMA · bounded stages",
        "tma",
        font_size=12,
        bold=True,
    )
    add_node(
        "t3_down_s",
        W4_X,
        1652,
        W4_W,
        42,
        "[S] reused Gate sB/sSFB",
        "smem",
        font_size=11,
        bold=True,
    )
    add_node(
        "t3_down_note",
        W4_X,
        1760,
        W4_W,
        50,
        "before FC2 consume:\nahead ≤ stages / stalled",
        "note",
        font_size=11,
        dashed=True,
    )
    add_edge("e_t3_dgt", "t3_down_g", "t3_down_tma", "", "tma")
    add_edge("e_t3_dts", "t3_down_tma", "t3_down_s", "TMA", "tma", width=3)

    # T4: A hoist, per-tile B supply, lane-local QMMA/epilogue/scatter.
    phase("t4", "T4 · FC2/scatter", 1960, 730)
    add_node(
        "t4_sa",
        COMPUTE_X,
        1985,
        COMPUTE_W,
        30,
        "[S] Q1 sA/sSFA",
        "smem",
        font_size=13,
        bold=True,
    )
    t4_a = local_row("t4_a", 2042, 30, ["[R] A/SFA held"] * 4, "rmem", font_size=11)
    add_node("t4_loop", 180, 2085, 1550, 600, "", "note", dashed=True, background=True)
    add_node(
        "t4_loop_label",
        195,
        2088,
        520,
        22,
        "repeat output tile j = 0..15",
        "text",
        font_size=13,
        bold=True,
        background=True,
    )
    add_node(
        "t4_bs",
        COMPUTE_X,
        2120,
        COMPUTE_W,
        30,
        "[S] Down sB/sSFB[j]",
        "smem",
        font_size=13,
        bold=True,
    )
    t4_ab = local_row(
        "t4_ab", 2177, 34, ["[R] A(reused)+B/SF"] * 4, "rmem", font_size=11
    )
    t4_tc = local_row(
        "t4_tc",
        2236,
        36,
        [f"MmaMXF4NVF4Op\ntiled-MMA member W{w}" for w in range(4)],
        "tc",
        font_size=9,
    )
    t4_acc = local_row("t4_acc", 2277, 28, ["[R] down_acc"] * 4, "rmem", font_size=11)
    t4_epi = local_row(
        "t4_epi", 2322, 38, ["CUDA scale/convert"] * 4, "cuda", font_size=11
    )
    add_node(
        "t4_sc",
        COMPUTE_X,
        2386,
        COMPUTE_W,
        30,
        "[S] sC BF16",
        "smem",
        font_size=13,
        bold=True,
    )
    thin_sync("t4_sync_pre", 2422, "epilog_sync pre-scatter · W0–W3 only")
    t4_sc_q = []
    t4_meta = []
    for warp, x in enumerate(LANE_X[:4]):
        qname = ["Q00", "Q01", "Q10", "Q11"][warp]
        sc_ident = f"t4_sc_q_w{warp}"
        meta_ident = f"t4_meta_w{warp}"
        add_node(
            sc_ident,
            x + 12,
            2448,
            132,
            26,
            f"[S] sC · {qname}",
            "smem",
            font_size=9,
            bold=True,
        )
        add_node(
            meta_ident,
            x + 152,
            2448,
            136,
            26,
            "[S] token/weight",
            "smem",
            font_size=9,
            bold=True,
        )
        t4_sc_q.append(sc_ident)
        t4_meta.append(meta_ident)
    scatter_labels = [
        "Q00 · [R] token + weighted FP32",
        "Q01 · [R] token + weighted FP32",
        "Q10 · [R] token + weighted FP32",
        "Q11 · [R] token + weighted FP32",
    ]
    t4_scatter = local_row("t4_scatter", 2508, 38, scatter_labels, "rmem", font_size=10)
    t4_cvt = local_row(
        "t4_cvt",
        2560,
        34,
        ["[PTX] 4×cvt.rn.satfinite\nbf16x2.f32"] * 4,
        "cuda",
        font_size=9,
    )
    add_node(
        "t4_y",
        COMPUTE_X,
        2630,
        COMPUTE_W,
        30,
        "[G] Y BF16",
        "gmem",
        font_size=13,
        bold=True,
    )
    thin_sync("t4_sync_post", 2665, "epilog_sync post-scatter → next j")
    for warp in range(4):
        add_edge(
            f"e_t4_a_{warp}",
            "t4_sa",
            t4_a[warp],
            "LdMatrix* / LDS(SF) · once",
            width=1.2,
        )
        add_edge(
            f"e_t4_ah_{warp}",
            t4_a[warp],
            t4_ab[warp],
            "reuse A",
            dashed=True,
            width=1.2,
        )
        add_edge(
            f"e_t4_b_{warp}",
            "t4_bs",
            t4_ab[warp],
            "LdMatrix* / LDS(SF) · per j",
            width=1.2,
        )
        add_edge(f"e_t4_tc_{warp}", t4_ab[warp], t4_tc[warp], "", "tc", width=2.2)
        add_edge(f"e_t4_ta_{warp}", t4_tc[warp], t4_acc[warp], "", "tc", width=2.2)
        add_edge(f"e_t4_ae_{warp}", t4_acc[warp], t4_epi[warp], "", "cuda", width=2)
        add_edge(
            f"e_t4_es_{warp}", t4_epi[warp], "t4_sc", "R→S CopyUniversal*", width=1.2
        )
        add_edge(
            f"e_t4_sc_data_{warp}",
            t4_sc_q[warp],
            t4_scatter[warp],
            "LDS(sC) + CUDA mul",
            width=1.2,
        )
        add_edge(
            f"e_t4_meta_data_{warp}",
            t4_meta[warp],
            t4_scatter[warp],
            "LDS(token/weight)",
            width=1.2,
        )
        add_edge(
            f"e_t4_sc_{warp}", t4_scatter[warp], t4_cvt[warp], "", "cuda", width=1.5
        )
        add_edge(
            f"e_t4_sy_{warp}",
            t4_cvt[warp],
            "t4_y",
            "[PTX] red.global.add.noftz.v4.bf16x2",
            "atom",
            width=2.5,
        )
    add_node(
        "t4_down_g",
        W4_X,
        2115,
        W4_W,
        36,
        "[G] Down W/SFB[j]",
        "gmem",
        font_size=12,
        bold=True,
    )
    add_node(
        "t4_down_tma",
        W4_X,
        2165,
        W4_W,
        34,
        "TMA producer",
        "tma",
        font_size=12,
        bold=True,
    )
    add_edge("e_t4_dgt", "t4_down_g", "t4_down_tma", "", "tma")
    add_edge(
        "e_t4_dts",
        "t4_down_tma",
        "t4_bs",
        "TMA G→S · bounded",
        "tma",
        direction="rl",
        width=3,
    )
    add_node(
        "t4_down_note",
        W4_X,
        2300,
        W4_W,
        60,
        "during FC2/scatter:\nahead / stalled / done /\npass_final wait",
        "note",
        font_size=11,
        dashed=True,
    )

    phase("pf", "pass_final", 2700, 35)
    thin_sync(
        "pf_bar", 2707, "W0–W3 arrive/no-wait  ·  W4 wait → next slice/task", w4=True
    )

    add_node(
        "footer",
        30,
        2750,
        1740,
        65,
        "Arrow labels name source/API movement; LDG/LDS/STG/STS denote logical spaces, not asserted SASS. [PTX] quotes inline PTX; emitted SASS lowering is still pending.\n* Source construct/API and address space confirmed; exact SASS pending. LdMatrix/StMatrix-derived mappings are source-confirmed; LDSM/STSM need disassembly. · Evidence: w4a4_moe_fused_overview_box.md · Source: flashinfer 517cca9c2e7d91f524fcb5f078370c056308d461",
        "text",
        font_size=11,
        background=True,
    )


def drawio_style(node: Node) -> str:
    fill, stroke = PALETTE[node.kind]
    if node.kind == "text":
        style = "text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=0;strokeColor=none;fillColor=none;"
    elif node.kind in {"phase", "lane", "note"}:
        style = (
            f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
        )
    else:
        style = (
            f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
        )
    style += f"fontSize={node.font_size};fontColor=#333333;"
    if node.bold:
        style += "fontStyle=1;"
    if node.dashed:
        style += "dashed=1;dashPattern=6 4;"
    if node.kind in {"phase", "gmem", "smem", "rmem", "tma", "tc", "cuda", "sync"}:
        style += "strokeWidth=2;"
    return style


def edge_style(edge: Edge) -> str:
    color = EDGE_COLORS[edge.kind]
    if edge.direction == "lr":
        anchors = "exitX=1;exitY=0.5;entryX=0;entryY=0.5;"
    elif edge.direction == "rl":
        anchors = "exitX=0;exitY=0.5;entryX=1;entryY=0.5;"
    else:
        anchors = "exitX=0.5;exitY=1;entryX=0.5;entryY=0;"
    style = (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
        f"{anchors}strokeColor={color};strokeWidth={edge.width};"
        "fontSize=11;fontColor=#333333;labelBackgroundColor=#FFFFFF;endArrow=block;endFill=1;"
    )
    if edge.dashed:
        style += "dashed=1;dashPattern=6 4;"
    return style


def write_drawio(path: Path) -> None:
    mxfile = ET.Element(
        "mxfile",
        {"host": "Claude", "type": "device"},
    )
    diagram = ET.SubElement(
        mxfile, "diagram", {"id": "w4a4-moe-overview", "name": "W4A4 MoE Overview"}
    )
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "1600",
            "dy": "1200",
            "grid": "0",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": str(WIDTH),
            "pageHeight": str(HEIGHT),
            "math": "0",
            "shadow": "0",
            "background": "#FFFFFF",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    ordered_nodes = [n for n in NODES if n.background] + [
        n for n in NODES if not n.background
    ]
    for node in ordered_nodes:
        value = node.label.replace("\n", "<br>")
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": node.ident,
                "value": value,
                "style": drawio_style(node),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(node.x),
                "y": str(node.y),
                "width": str(node.w),
                "height": str(node.h),
                "as": "geometry",
            },
        )

    for edge in EDGES:
        value = edge.label.replace("\n", "<br>")
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": edge.ident,
                "value": value,
                "style": edge_style(edge),
                "edge": "1",
                "source": edge.source,
                "target": edge.target,
                "parent": "1",
            },
        )
        ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})

    rough = ET.tostring(mxfile, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="UTF-8")
    path.write_bytes(pretty)


def svg_text(node: Node) -> str:
    if not node.label:
        return ""
    lines = node.label.split("\n")
    line_h = node.font_size * 1.25
    start_y = (
        node.y + node.h / 2 - (len(lines) - 1) * line_h / 2 + node.font_size * 0.35
    )
    weight = "700" if node.bold else "400"
    chunks = [
        f'<text x="{node.x + node.w / 2:.1f}" y="{start_y:.1f}" text-anchor="middle" '
        f'font-family="DejaVu Sans,Arial,sans-serif" font-size="{node.font_size}" font-weight="{weight}" fill="#333333">'
    ]
    for idx, line in enumerate(lines):
        dy = "0" if idx == 0 else f"{line_h:.1f}"
        chunks.append(
            f'<tspan x="{node.x + node.w / 2:.1f}" dy="{dy}">{escape(line)}</tspan>'
        )
    chunks.append("</text>")
    return "".join(chunks)


def node_svg(node: Node) -> str:
    fill, stroke = PALETTE[node.kind]
    if node.kind == "text":
        return svg_text(node)
    dash = ' stroke-dasharray="7 5"' if node.dashed else ""
    fill_attr = "none" if fill == "none" else fill
    stroke_w = (
        2
        if node.kind in {"phase", "gmem", "smem", "rmem", "tma", "tc", "cuda", "sync"}
        else 1.5
    )
    rect = (
        f'<rect x="{node.x}" y="{node.y}" width="{node.w}" height="{node.h}" rx="8" '
        f'fill="{fill_attr}" stroke="{stroke}" stroke-width="{stroke_w}"{dash}/>'
    )
    return rect + svg_text(node)


def edge_points(edge: Edge, nodes: dict[str, Node]) -> tuple[str, float, float]:
    src = nodes[edge.source]
    dst = nodes[edge.target]
    if edge.direction == "lr":
        x1, y1 = src.x + src.w, src.y + src.h / 2
        x2, y2 = dst.x, dst.y + dst.h / 2
        mid = (x1 + x2) / 2
        path = f"M{x1},{y1} H{mid} V{y2} H{x2}"
        return path, mid, min(y1, y2) - 5 if y1 != y2 else y1 - 7
    if edge.direction == "rl":
        x1, y1 = src.x, src.y + src.h / 2
        x2, y2 = dst.x + dst.w, dst.y + dst.h / 2
        mid = (x1 + x2) / 2
        path = f"M{x1},{y1} H{mid} V{y2} H{x2}"
        return path, mid, min(y1, y2) - 5 if y1 != y2 else y1 - 7
    x1, y1 = src.x + src.w / 2, src.y + src.h
    x2, y2 = dst.x + dst.w / 2, dst.y
    mid = (y1 + y2) / 2
    path = f"M{x1},{y1} V{mid} H{x2} V{y2}"
    return path, (x1 + x2) / 2, mid - 5


def edge_svg(edge: Edge, nodes: dict[str, Node]) -> str:
    color = EDGE_COLORS[edge.kind]
    marker = color[1:]
    path, label_x, label_y = edge_points(edge, nodes)
    dash = ' stroke-dasharray="7 5"' if edge.dashed else ""
    chunks = [
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{edge.width}" '
        f'marker-end="url(#arrow_{marker})"{dash}/>'
    ]
    if edge.label:
        lines = edge.label.split("\n")
        fs = 11
        line_h = 13
        box_w = max(80, max(len(line) for line in lines) * 5.8)
        box_h = len(lines) * line_h + 6
        chunks.append(
            f'<rect x="{label_x - box_w / 2:.1f}" y="{label_y - box_h + 3:.1f}" width="{box_w:.1f}" height="{box_h}" '
            'rx="3" fill="#FFFFFF" fill-opacity="0.94" stroke="none"/>'
        )
        chunks.append(
            f'<text x="{label_x:.1f}" y="{label_y - (len(lines) - 1) * line_h:.1f}" text-anchor="middle" '
            f'font-family="DejaVu Sans,Arial,sans-serif" font-size="{fs}" fill="#333333">'
        )
        for idx, line in enumerate(lines):
            dy = "0" if idx == 0 else str(line_h)
            chunks.append(f'<tspan x="{label_x:.1f}" dy="{dy}">{escape(line)}</tspan>')
        chunks.append("</text>")
    return "".join(chunks)


def write_svg(path: Path) -> None:
    node_map = {node.ident: node for node in NODES}
    markers = []
    for color in sorted(set(EDGE_COLORS.values())):
        marker = color[1:]
        markers.append(
            f'<marker id="arrow_{marker}" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M0,0 L0,6 L8,3 z" fill="{color}"/></marker>'
        )
    chunks = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        *markers,
        "</defs>",
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#FFFFFF"/>',
    ]
    chunks.extend(node_svg(node) for node in NODES if node.background)
    chunks.extend(edge_svg(edge, node_map) for edge in EDGES)
    chunks.extend(node_svg(node) for node in NODES if not node.background)
    chunks.append("</svg>")
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def main() -> None:
    build_spec()
    write_drawio(OUT_DIR / "w4a4_moe_fused_overview.drawio")
    write_svg(OUT_DIR / "w4a4_moe_fused_overview.svg")


if __name__ == "__main__":
    main()
