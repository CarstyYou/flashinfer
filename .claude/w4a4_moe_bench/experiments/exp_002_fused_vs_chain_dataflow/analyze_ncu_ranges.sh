#!/usr/bin/env bash
set -euo pipefail

host_root=/home/xiy/workspace/flashinfer_exp002_074d93e
host_veloq=/home/xiy/workspace/veloq
container_root=/workspace/source/flashinfer
relative_exp=.claude/w4a4_moe_bench/experiments/exp_002_fused_vs_chain_dataflow

for m in 256 8192; do
  for arm in cutedsl_bf16_fused cutlass_bf16_chain; do
    relative_case="$relative_exp/results/ncu/m${m}/${arm}/operator-ledger-v2"
    docker run --rm \
      -v "$host_root:$container_root" \
      -v "$host_veloq:/usr/local/bin/veloq:ro" \
      -w "$container_root" \
      nvcr.io/nvidia/pytorch:26.05-py3 \
      bash -lc "
        set -euo pipefail
        trace='$relative_case/trace.ncu-rep'
        out='$relative_case/veloq'
        mkdir -p \"\$out\"
        veloq --version >\"\$out/version.txt\"
        veloq ncu summary \"\$trace\" >\"\$out/summary.json\"
        veloq ncu ranges \"\$trace\" >\"\$out/ranges.json\"
      "
  done
done
