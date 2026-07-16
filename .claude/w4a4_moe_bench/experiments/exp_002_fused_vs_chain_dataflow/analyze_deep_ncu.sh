#!/usr/bin/env bash
set -euo pipefail

host_root=/home/xiy/workspace/flashinfer_exp002_074d93e
host_veloq=/home/xiy/workspace/veloq
container_root=/workspace/source/flashinfer
relative_exp=.claude/w4a4_moe_bench/experiments/exp_002_fused_vs_chain_dataflow

analyze_one() {
  local m=$1
  local arm=$2
  local skip=$3
  local relative_case="$relative_exp/results/ncu/m${m}/${arm}/deep_launch_${skip}"
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
      veloq info \"\$trace\" >\"\$out/info.json\"
      veloq ncu summary \"\$trace\" >\"\$out/summary.json\"
      veloq ncu launches \"\$trace\" --limit 10 >\"\$out/launches.json\"
      veloq ncu inspect \"\$trace\" --row-id launch:0 >\"\$out/inspect.json\"
      veloq ncu disasm \"\$trace\" --row-id launch:0 >\"\$out/disasm.json\"
    "
}

for m in 256 8192; do
  analyze_one "$m" cutedsl_bf16_fused 1
  for skip in 3 5 6 7 8; do
    analyze_one "$m" cutlass_bf16_chain "$skip"
  done
done

docker run --rm \
  -v "$host_root:$container_root" \
  -w "$container_root" \
  nvcr.io/nvidia/pytorch:26.05-py3 \
  bash -lc "
    set -euo pipefail
    python3 '$relative_exp/build_deep_ncu_evidence.py'
    python3 '$relative_exp/build_static_sass_evidence.py'
    python3 '$relative_exp/build_fused_schedule_models.py'
  "
