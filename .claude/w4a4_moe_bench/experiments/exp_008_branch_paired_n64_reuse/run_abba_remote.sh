#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 primary|secondary|production [M ...]" >&2
  exit 2
fi

pair=$1
shift
if [[ $# -gt 0 ]]; then
  ms=("$@")
else
  ms=(256 8192)
fi
case "$pair" in
  primary) order=(v0 v1 v1 v0) ;;
  secondary) order=(n128 v1 v1 n128) ;;
  production) order=(production v1 v1 production) ;;
  *) echo "invalid pair: $pair" >&2; exit 2 ;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
runner=$script_dir/run_remote.sh
gpu_uuid=GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522

app_clock=$(nvidia-smi \
  --query-gpu=uuid,clocks.applications.graphics \
  --format=csv,noheader,nounits \
  | awk -F, -v uuid="$gpu_uuid" '$1 == uuid {gsub(/ /, "", $2); print $2}')
if [[ "$app_clock" != 2377 ]]; then
  echo "application graphics clock must be 2377 MHz, got ${app_clock:-missing}" >&2
  exit 3
fi

for m in "${ms[@]}"; do
  case "$m" in 256|8192) ;; *) echo "invalid M: $m" >&2; exit 2 ;; esac
  for group in 0 1 2 3 4; do
    existing=0
    for position in 0 1 2 3; do
      arm=${order[$position]}
      case "$arm" in
        production) internal_arm=baseline_4warp ;;
        n128) internal_arm=candidate_8warp_serial_v0 ;;
        v0|v1) internal_arm=candidate_8warp_n64_temporal_replay_v0 ;;
      esac
      output=$script_dir/results/e2e/$pair/$arm/raw/benchmark/m${m}/group_${group}_position_${position}_${internal_arm}.json
      if [[ -f "$output" ]]; then
        grep -q '"status": "complete"' "$output" || {
          echo "incomplete immutable sample: $output" >&2
          exit 4
        }
        existing=$((existing + 1))
      fi
    done
    if [[ "$existing" -eq 4 ]]; then
      echo "skip complete $pair M=$m group=$group"
      continue
    fi
    if [[ "$existing" -ne 0 ]]; then
      echo "partial ABBA group cannot be resumed: $pair M=$m group=$group ($existing/4 positions)" >&2
      exit 4
    fi
    for position in 0 1 2 3; do
      arm=${order[$position]}
      case "$arm" in
        production) internal_arm=baseline_4warp ;;
        n128) internal_arm=candidate_8warp_serial_v0 ;;
        v0|v1) internal_arm=candidate_8warp_n64_temporal_replay_v0 ;;
      esac
      echo "run $pair M=$m group=$group position=$position arm=$arm"
      EXP008_PAIR=$pair EXP008_SAMPLE_LABEL=g${group}_p${position} \
        "$runner" measure "$arm" "$m" canonical \
          --group "$group" --position "$position" \
          --warmup 5 --iters 50 --clock-policy locked
    done
  done
done
