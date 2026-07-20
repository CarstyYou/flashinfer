#!/usr/bin/env bash
set -euo pipefail

mode=${1:-capture}
arm=${2:-all}
case "$mode" in
  dry-run|capture|build) ;;
  *) echo "usage: $0 dry-run|capture|build [baseline|candidate_v2|all]" >&2; exit 2 ;;
esac
case "$arm" in
  baseline|candidate_v2|all) ;;
  *) echo "invalid arm: $arm" >&2; exit 2 ;;
esac

repo=${EXP015_REPO:-/home/xiy/workspace/flashinfer_exp015_748ad}
relative_exp=.claude/w4a4_moe_bench/experiments/exp_015_phase_skeleton_refactor
exp=$repo/$relative_exp
results=$exp/results
capture=$exp/capture_matched_dynamic_ncu.py
builder=$exp/build_matched_dynamic_ncu_evidence.py
container=${EXP015_CONTAINER:-xiyExp0155kp}
gpu_uuid=GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522
expected_clock=2377
lease_id=exp015-phase-skeleton-20260720
lease_dir=/tmp/kdk-direct-ssh-gpu-leases/R6KD-CX8aaS-GPU-16_GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522
pythonpath=$repo:/home/xiy/workspace/w4a4_deps_460:/home/xiy/workspace/w4a4_deps_460/nvidia_cutlass_dsl/dsl_packages

require_abba_complete() {
  local count
  count=$(find "$results/raw/benchmark" -type f \
    -name 'group_*_position_*.json' 2>/dev/null | wc -l | xargs)
  if [[ "$count" != 40 ]]; then
    echo "matched NCU waits for complete M256/M8192 ABBA (40 positions); got $count" >&2
    exit 4
  fi
}

if [[ "$mode" == build ]]; then
  docker exec \
    -e CUDA_VISIBLE_DEVICES= \
    "$container" python3 "$builder" --results "$results"
  exit 0
fi

if [[ "$mode" == capture ]]; then
  require_abba_complete
fi
grep -qx "lease_id=$lease_id" "$lease_dir/metadata"
grep -qx "gpu_uuid=$gpu_uuid" "$lease_dir/metadata"
mkdir -p "$results/runtime"

if [[ "$arm" == all ]]; then
  arms=(baseline candidate_v2)
else
  arms=("$arm")
fi

for selected_arm in "${arms[@]}"; do
  case "$selected_arm" in
    baseline)
      jit=/home/xiy/workspace/exp015_validate_baseline
      ;;
    candidate_v2)
      jit=/home/xiy/workspace/exp015_validate_candidate_v2
      ;;
  esac

  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
    | grep -qx "$gpu_uuid"; then
    echo "foreign compute process detected on leased GPU" >&2
    exit 5
  fi
  gpu_line=$(nvidia-smi --id="$gpu_uuid" \
    --query-gpu=uuid,clocks.applications.graphics \
    --format=csv,noheader,nounits)
  observed_uuid=$(printf '%s' "$gpu_line" | cut -d, -f1 | xargs)
  observed_clock=$(printf '%s' "$gpu_line" | cut -d, -f2 | xargs)
  if [[ "$observed_uuid" != "$gpu_uuid" || "$observed_clock" != "$expected_clock" ]]; then
    echo "GPU/clock identity drift: $gpu_line" >&2
    exit 6
  fi

  command=(
    python3 "$capture"
    --flashinfer-root "$repo"
    --results "$results"
    --jit-root "$jit"
    --arm "$selected_arm"
    --expected-gpu-uuid "$gpu_uuid"
    --expected-app-clock-mhz "$expected_clock"
  )
  if [[ "$mode" == dry-run ]]; then
    command+=(--dry-run)
  fi

  label=matched_ncu_${selected_arm}_m8192
  {
    printf 'docker exec --privileged '
    printf '%q ' \
      -e CUDA_VISIBLE_DEVICES=0 \
      -e PYTHONPATH="$pythonpath" \
      -e FLASHINFER_NVFP4_4OVER6=0 \
      -e FLASHINFER_WORKSPACE_BASE="$jit" \
      -e CUTE_DSL_CACHE_DIR="$jit/cache" \
      -e CUTE_DSL_DUMP_DIR="$jit/dump" \
      -e CUTE_DSL_KEEP=ir,ptx,cubin,sass \
      -e TORCH_CUDA_ARCH_LIST=12.0a \
      -e KDK_LEASE_ID="$lease_id" \
      -e KDK_LEASE_GPU_UUID="$gpu_uuid" \
      -e W4A4_IMAGE_ID=sha256:a4e056e1d34a5cc9387512ffa3abeed778e3dc7966633c5154d771705d8835ac \
      -e W4A4_IMAGE_DIGEST=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba \
      -e W4A4_PYTHON_DEPS_SHA256=32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74 \
      "$container" "${command[@]}"
    printf '\n'
  } > "$results/runtime/$label.command.txt"

  docker exec --privileged \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTHONPATH="$pythonpath" \
    -e FLASHINFER_NVFP4_4OVER6=0 \
    -e FLASHINFER_WORKSPACE_BASE="$jit" \
    -e CUTE_DSL_CACHE_DIR="$jit/cache" \
    -e CUTE_DSL_DUMP_DIR="$jit/dump" \
    -e CUTE_DSL_KEEP=ir,ptx,cubin,sass \
    -e TORCH_CUDA_ARCH_LIST=12.0a \
    -e KDK_LEASE_ID="$lease_id" \
    -e KDK_LEASE_GPU_UUID="$gpu_uuid" \
    -e W4A4_IMAGE_ID=sha256:a4e056e1d34a5cc9387512ffa3abeed778e3dc7966633c5154d771705d8835ac \
    -e W4A4_IMAGE_DIGEST=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba \
    -e W4A4_PYTHON_DEPS_SHA256=32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74 \
    "$container" "${command[@]}" \
    > >(tee "$results/runtime/$label.stdout.log") \
    2> >(tee "$results/runtime/$label.stderr.log" >&2)
done

if [[ "$mode" == capture && "$arm" == all ]]; then
  docker exec \
    -e CUDA_VISIBLE_DEVICES= \
    "$container" python3 "$builder" --results "$results"
fi
