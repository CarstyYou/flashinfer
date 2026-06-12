#!/usr/bin/env bash
# Per-cell subprocess to isolate CUDA state — crashes in one cell don't poison others.
# Run from inside container with LD_LIBRARY_PATH set.

set -u
cd "$(dirname "$0")"

OUT="${1:-/home/scratch.xiy_gpu/mega_inference/flashinfer/.claude/6kpro_bench_results/6kpro_full.csv}"
rm -f "$OUT"
echo "[sweep] output: $OUT"

LAYERS=(fc1 fc2)
M_PE=(1 4 8 16 192 256 1024 4096)
GRAN_K=(32 128)
BACKENDS=(cute_sm120 cudnn dg cutlass)

done=0
total=0
for layer in "${LAYERS[@]}"; do
  for mpe in "${M_PE[@]}"; do
    for gk in "${GRAN_K[@]}"; do
      for bk in "${BACKENDS[@]}"; do
        [[ "$bk" == "cudnn" && "$gk" == "128" ]] && continue
        total=$((total + 1))
      done
    done
  done
done
echo "[sweep] $total cells"

for layer in "${LAYERS[@]}"; do
  for mpe in "${M_PE[@]}"; do
    for gk in "${GRAN_K[@]}"; do
      for bk in "${BACKENDS[@]}"; do
        [[ "$bk" == "cudnn" && "$gk" == "128" ]] && continue
        done=$((done + 1))
        out=$(python bench_grouped_gemm_fixed.py --backend "$bk" --layer "$layer" --m-pe "$mpe" --gran-k "$gk" --out "$OUT" 2>&1 | grep -E "→" | head -1)
        echo "[$done/$total] $out"
      done
    done
  done
done

echo "[sweep] done, rows in csv:"
wc -l "$OUT"
