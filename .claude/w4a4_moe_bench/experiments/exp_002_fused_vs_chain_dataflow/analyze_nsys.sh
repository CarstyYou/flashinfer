#!/usr/bin/env bash
set -euo pipefail

host_root=/home/xiy/workspace/flashinfer_exp002_074d93e
host_veloq=/home/xiy/workspace/veloq
container_root=/workspace/source/flashinfer
relative_exp=.claude/w4a4_moe_bench/experiments/exp_002_fused_vs_chain_dataflow

for m in 256 8192; do
  for arm in cutedsl_bf16_fused cutlass_bf16_chain; do
    relative_case="$relative_exp/results/nsys/m${m}/${arm}"
    docker run --rm \
      -v "$host_root:$container_root" \
      -v "$host_veloq:/usr/local/bin/veloq:ro" \
      -w "$container_root" \
      nvcr.io/nvidia/pytorch:26.05-py3 \
      bash -lc "
        set -euo pipefail
        trace='$relative_case/trace.nsys-rep'
        out='$relative_case/veloq'
        mkdir -p \"\$out\"
        veloq --version >\"\$out/version.txt\"
        veloq prep \"\$trace\" >\"\$out/prep.json\"
        veloq summary \"\$trace\" >\"\$out/summary.json\"
        veloq graph-replays \"\$trace\" --device 0 --sort sum:desc --top-nodes 20 --limit 10 >\"\$out/graph_replays.json\"
        veloq stats \"\$trace\" --device 0 --type kernel --group-by demangled,grid_block --limit 100 >\"\$out/stats.json\"
        veloq search \"\$trace\" --device 0 --type kernel --with-nvtx --sort start:asc --limit 100 >\"\$out/kernels.json\"
        veloq gaps \"\$trace\" --device 0 --min-duration 1ns --limit 100 >\"\$out/gaps.json\"
        veloq concurrency \"\$trace\" --device 0 >\"\$out/concurrency.json\"
      "
  done
done
