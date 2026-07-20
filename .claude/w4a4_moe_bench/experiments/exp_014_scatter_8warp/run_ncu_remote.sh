#!/usr/bin/env bash
set -euo pipefail

# Candidate-only is the normal path.  An explicit baseline capture remains
# available, but the builder otherwise reuses exp_015 only after exact
# source/cubin/JIT/GPU/launch/work identity validation.

on_host=0
if [[ ${1:-} == --on-host ]]; then
  on_host=1
  shift
fi

mode=${1:-capture}
arm=${2:-candidate_8warp_scatter}
case "$mode" in
  dry-run|capture|build) ;;
  *) echo "usage: $0 [--on-host] dry-run|capture|build [baseline_4warp_scatter|candidate_8warp_scatter]" >&2; exit 2 ;;
esac
case "$arm" in
  baseline_4warp_scatter|candidate_8warp_scatter) ;;
  *) echo "invalid arm: $arm" >&2; exit 2 ;;
esac

host=${EXP014_HOST:-xiy@10.6.142.16}
repo=${EXP014_REPO:-/home/xiy/workspace/flashinfer_exp015_748ad}
relative_exp=.claude/w4a4_moe_bench/experiments/exp_014_scatter_8warp
container=${EXP014_CONTAINER:-xiyExp0145kp}
host_python=${EXP014_HOST_PYTHON:-python3}
gpu_uuid=${EXP014_GPU_UUID:-GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
lease_id=${EXP014_LEASE_ID:-exp014-scatter-8warp-20260720}
lease_dir=${EXP014_LEASE_DIR:-/tmp/kdk-direct-ssh-gpu-leases/R6KD-CX8aaS-GPU-16_GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
baseline_jit=${EXP014_BASELINE_JIT:-/home/xiy/workspace/exp014_scatter_8warp_jit/baseline_4warp_scatter}
candidate_jit=${EXP014_CANDIDATE_JIT:-/home/xiy/workspace/exp014_scatter_8warp_jit/candidate_8warp_scatter}

if [[ $on_host == 0 ]]; then
  remote_script=$repo/$relative_exp/run_ncu_remote.sh
  remote=(
    env
    EXP014_REPO="$repo"
    EXP014_CONTAINER="$container"
    EXP014_HOST_PYTHON="$host_python"
    EXP014_GPU_UUID="$gpu_uuid"
    EXP014_LEASE_ID="$lease_id"
    EXP014_LEASE_DIR="$lease_dir"
    EXP014_BASELINE_JIT="$baseline_jit"
    EXP014_CANDIDATE_JIT="$candidate_jit"
    bash "$remote_script" --on-host "$mode" "$arm"
  )
  printf -v remote_command '%q ' "${remote[@]}"
  exec ssh -T "$host" "$remote_command"
fi

exp=$repo/$relative_exp
results=$exp/results
capture=$exp/capture_dynamic_spill_ncu.py
builder=$exp/build_dynamic_spill_evidence.py
pythonpath=$repo:/home/xiy/workspace/w4a4_deps_460:/home/xiy/workspace/w4a4_deps_460/nvidia_cutlass_dsl/dsl_packages
expected_clock=2377

[[ -f $capture && -f $builder ]]

if [[ $mode == build ]]; then
  exec "$host_python" "$builder" --results "$results"
fi

[[ -f $lease_dir/metadata ]]
grep -qx "lease_id=$lease_id" "$lease_dir/metadata"
grep -qx "gpu_uuid=$gpu_uuid" "$lease_dir/metadata"

case "$arm" in
  baseline_4warp_scatter) jit=$baseline_jit ;;
  candidate_8warp_scatter) jit=$candidate_jit ;;
esac
validation=$results/raw/validation/$arm/validation.json
[[ -f $validation ]] || {
  echo "missing completed validation manifest: $validation" >&2
  exit 4
}
[[ -d $jit ]] || {
  echo "missing registered JIT root: $jit" >&2
  exit 5
}

if [[ $mode == capture ]]; then
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
    | grep -qx "$gpu_uuid"; then
    echo "foreign compute process detected on leased GPU" >&2
    exit 6
  fi
  gpu_line=$(nvidia-smi --id="$gpu_uuid" \
    --query-gpu=uuid,clocks.applications.graphics \
    --format=csv,noheader,nounits)
  observed_uuid=$(printf '%s' "$gpu_line" | cut -d, -f1 | xargs)
  observed_clock=$(printf '%s' "$gpu_line" | cut -d, -f2 | xargs)
  if [[ $observed_uuid != "$gpu_uuid" || $observed_clock != "$expected_clock" ]]; then
    echo "GPU/clock identity drift: $gpu_line" >&2
    exit 7
  fi
fi

command=(
  python3 "$capture"
  --flashinfer-root "$repo"
  --results "$results"
  --jit-root "$jit"
  --arm "$arm"
  --expected-gpu-uuid "$gpu_uuid"
  --expected-app-clock-mhz "$expected_clock"
)
if [[ $mode == dry-run ]]; then
  command+=(--dry-run)
fi

mkdir -p "$results/runtime"
label=dynamic_spill_ncu_${arm}_m8192
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

if [[ $mode == capture ]]; then
  "$host_python" "$builder" --results "$results"
fi
