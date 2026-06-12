"""Generate markdown perf comparison table from sweep csv.

Pivots (m_pe, backend, gran_k) into per-layer tables.
Padded m_pe shown as "(pad N)" suffix when m_pe_padded != m_pe.
Missing cells marked **FAIL**.

Usage:
    python gen_perf_table.py [csv_path]
    python gen_perf_table.py /path/to/6kpro_full.csv > report.md
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

# Backends per granK (cudnn supports granK=32 only — industry MX spec)
BACKENDS_BY_GRANK = {
    32: ["cute_sm120", "cudnn", "dg", "cutlass"],
    128: ["cute_sm120", "dg", "cutlass"],
}

BACKEND_LABEL = {
    "cute_sm120": "cute",
    "cudnn": "cudnn",
    "dg": "dg",
    "cutlass": "cutlass",
}


def load(csv_path):
    rows = defaultdict(
        dict
    )  # (layer, m_pe) -> {(backend, gran_k): (t_us, m_pe_padded)}
    layer_shapes = {}
    m_pe_set = defaultdict(set)
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            layer = r["layer"]
            m_pe = int(r["m_pe"])
            backend = r["backend"]
            gran_k = int(r["gran_k"])
            t_us = float(r["t_us"])
            m_pe_padded = int(r["m_pe_padded"])
            n = int(r["n"])
            k = int(r["k"])
            rows[(layer, m_pe)][(backend, gran_k)] = (t_us, m_pe_padded)
            layer_shapes[layer] = (n, k)
            m_pe_set[layer].add(m_pe)
    return rows, layer_shapes, {layer: sorted(s) for layer, s in m_pe_set.items()}


def fmt_cell(entry, m_pe, cute_t_us=None):
    if entry is None:
        return "**FAIL**"
    t_us, m_pe_padded = entry
    s = "{:.1f}".format(t_us)
    if m_pe_padded != m_pe:
        s += " (pad {})".format(m_pe_padded)
    if cute_t_us is not None and cute_t_us > 0:
        ratio = t_us / cute_t_us
        if abs(ratio - 1.0) >= 0.0005:
            s += " ({:+.1f}%)".format((ratio - 1) * 100)
    return s


def render_layer_grank(layer, n, k, gran_k, m_pes, rows):
    backends = BACKENDS_BY_GRANK[gran_k]
    cols = ["m_pe"] + [BACKEND_LABEL[b] for b in backends]
    title = "### {} (N={}, K={}) -- granK={}".format(layer, n, k, gran_k)
    lines = [
        title,
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for m_pe in m_pes:
        cells = rows.get((layer, m_pe), {})
        cute_entry = cells.get(("cute_sm120", gran_k))
        cute_t_us = cute_entry[0] if cute_entry else None
        row = [str(m_pe)]
        for b in backends:
            entry = cells.get((b, gran_k))
            ref = None if b == "cute_sm120" else cute_t_us
            row.append(fmt_cell(entry, m_pe, ref))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    csv_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent
        / "6kpro_bench_results"
        / "6kpro_full.csv"
    )
    rows, layer_shapes, m_pe_per_layer = load(csv_path)

    print("# Grouped MoE GEMM perf comparison -- `{}`\n".format(csv_path.name))
    print(
        "All t_us = 50-rep median (l2-flush per iter). `pad N` = caller-padded "
        "per-expert M (kernel sees padded rows; TFLOPS uses logical m_pe).\n"
    )
    print("## Backend / API mapping\n")
    print("| Backend | Library API | Notes |")
    print("|---|---|---|")
    print(
        "| **cute** | `flashinfer.grouped_mm.moe_gemm_mxfp8_nt_groupwise` "
        '(backend=`"cute"`, scale_granularity_mnk=(1, 1, granK)) | '
        "PR #3562 cute SM120 ZeroPadding mode (kernel handles all padding internally) |"
    )
    print(
        "| **cudnn** | `flashinfer.grouped_mm.grouped_mm_mxfp8` "
        '(backend=`"cudnn"`) | '
        "cuDNN 9.23 grouped MoE GEMM, granK=32 only (industry MX 1x32 spec) |"
    )
    print(
        "| **dg** | `deep_gemm.m_grouped_fp8_gemm_nt_contiguous` "
        "(recipe=(1, 1, granK)) | "
        "DeepGEMM upstream leavelet/sm120 branch HEAD 76e93aa "
        "(NOT a flashinfer API; caller pads M to 128 per "
        "`get_theoretical_mk_alignment_for_contiguous_layout()`) |"
    )
    print(
        "| **cutlass** | custom `.cu` wrapper around CUTLASS example "
        "`87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu` "
        "(`ScaleGranularityN=1` -- 1D per-token scale matching cute) | "
        "flashinfer `group_gemm_fp8_nt_groupwise` has upstream guard "
        "rejecting num_groups>1 on SM120; bench uses custom `.cu` bypass; "
        "TileM=128 Cooperative, no SwapAB |"
    )
    print()
    for layer in sorted(layer_shapes):
        n, k = layer_shapes[layer]
        for gran_k in (32, 128):
            print()
            print(render_layer_grank(layer, n, k, gran_k, m_pe_per_layer[layer], rows))
            print()


if __name__ == "__main__":
    main()
