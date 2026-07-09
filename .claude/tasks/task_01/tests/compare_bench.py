"""Paired compare of bench_pre_sync.csv vs bench_post_sync.csv -> markdown table."""

import csv
import sys
from pathlib import Path


def load(p):
    with open(p) as f:
        return {tuple(r[:5]): float(r[5]) for r in list(csv.reader(f))[1:]}


def main():
    outdir = Path(sys.argv[1])
    pre = load(outdir / "bench_pre_sync.csv")
    post = load(outdir / "bench_post_sync.csv")
    print("| E | m_pe | N | K | pre (us) | post (us) | delta |")
    print("|---|---|---|---|---|---|---|")
    worst = (None, 0.0)
    for key in pre:
        a, b = pre[key], post.get(key)
        if b is None:
            continue
        pct = (a - b) / a * 100.0
        if pct < worst[1]:
            worst = (key, pct)
        print(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {a:.3f} | {b:.3f} | {pct:+.2f}% |"
        )
    print(f"\nworst cell: {worst[0]} {worst[1]:+.2f}%")


if __name__ == "__main__":
    main()
