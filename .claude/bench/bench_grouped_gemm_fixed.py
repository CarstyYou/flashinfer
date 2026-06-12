"""bench_grouped_gemm_fixed.py — PM PR scope: 112-cell sweep over 4 backends.

Sweeps (layer × m_pe × granK × backend) and writes per-cell median t_us to CSV.
See ../bench_plan.md for the design rationale (fixed config, caller padding contract,
TFLOPS dropped per PM).

Usage:
  python bench_grouped_gemm_fixed.py                              # full 112-cell sweep
  python bench_grouped_gemm_fixed.py --backend cute_sm120
  python bench_grouped_gemm_fixed.py --layer fc1 --gran-k 32
  python bench_grouped_gemm_fixed.py --out /tmp/run42.csv
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _bench_common import (  # noqa: E402
    BACKENDS,
    GRAN_K_SWEEP,
    LAYER_SHAPES,
    M_PE_SWEEP,
    NUM_EXPERT,
    bench_median_us,
    cell_supported,
    make_base_inputs,
    write_row,
)


DEFAULT_OUT = Path(__file__).parent.parent / "6kpro_bench_results" / "pr3562_fixed.csv"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"CSV output path (default: {DEFAULT_OUT})",
    )
    p.add_argument("--backend", choices=list(BACKENDS), default=None)
    p.add_argument("--layer", choices=list(LAYER_SHAPES), default=None)
    p.add_argument("--m-pe", type=int, default=None)
    p.add_argument("--gran-k", type=int, choices=GRAN_K_SWEEP, default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--rep", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    try:
        import deep_gemm

        deep_gemm.set_mk_alignment_for_contiguous_layout(
            deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        )
        dg_align = deep_gemm.get_mk_alignment_for_contiguous_layout()
        print(f"[setup] deep_gemm mk_alignment = {dg_align}")
    except ImportError:
        print("[setup] deep_gemm not available; dg backend cells will FAIL")

    layers = [args.layer] if args.layer else list(LAYER_SHAPES)
    m_pes = [args.m_pe] if args.m_pe else M_PE_SWEEP
    gran_ks = [args.gran_k] if args.gran_k else GRAN_K_SWEEP
    backends = [args.backend] if args.backend else list(BACKENDS)

    total = sum(
        1
        for _layer in layers
        for m_pe in m_pes
        for gran_k in gran_ks
        for backend in backends
        if cell_supported(backend, m_pe, gran_k)
    )
    print(f"[setup] csv = {args.out}")
    print(
        f"[setup] sweep = {len(layers)} layer × {len(m_pes)} m_pe × {len(gran_ks)} granK × {len(backends)} backend → {total} cells"
    )

    done = 0
    for layer in layers:
        n, k = LAYER_SHAPES[layer]
        for m_pe in m_pes:
            a_bf16, b_bf16, m_indptr = make_base_inputs(
                NUM_EXPERT, m_pe, n, k, seed=args.seed
            )
            for gran_k in gran_ks:
                for backend in backends:
                    if not cell_supported(backend, m_pe, gran_k):
                        continue
                    done += 1
                    spec = BACKENDS[backend]
                    tag = f"[{done:3d}/{total}] {layer} m_pe={m_pe:<4d} granK={gran_k:<3d} {backend:<11s}"
                    try:
                        inputs = spec["prep"](a_bf16, b_bf16, m_indptr, m_pe, gran_k)
                        t_us = bench_median_us(
                            lambda inp=inputs: spec["call"](inp),
                            warmup=args.warmup,
                            rep=args.rep,
                        )
                        row = dict(
                            layer=layer,
                            n=n,
                            k=k,
                            num_expert=NUM_EXPERT,
                            m_pe=m_pe,
                            m_pe_padded=inputs["m_pe_padded"],
                            total_rows=NUM_EXPERT * m_pe,
                            backend=backend,
                            gran_k=gran_k,
                            t_us=round(t_us, 3),
                        )
                        write_row(args.out, row)
                        print(
                            f"{tag} → {t_us:8.2f} us  (padded m_pe={inputs['m_pe_padded']})"
                        )
                        del inputs
                    except Exception as exc:
                        print(f"{tag} → FAIL  {type(exc).__name__}: {exc}")
                    torch.cuda.empty_cache()
            del a_bf16, b_bf16, m_indptr
            torch.cuda.empty_cache()

    print(f"[done] wrote {done} cells to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
