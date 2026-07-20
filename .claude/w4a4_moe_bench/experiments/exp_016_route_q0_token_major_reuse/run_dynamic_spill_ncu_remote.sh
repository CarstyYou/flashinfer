#!/usr/bin/env bash
set -euo pipefail

# Candidate-only M8192 dynamic-spill capture.  This reuses the already
# correctness-validated Candidate overlay/JIT/cubin and profiles one graph node.

on_host=0
if [[ ${1:-} == --on-host ]]; then
  on_host=1
  shift
fi

mode=${1:-dry-run}
case "$mode" in
  dry-run|capture|build) ;;
  *) echo "usage: $0 [--on-host] dry-run|capture|build" >&2; exit 2 ;;
esac

host=${EXP016_HOST:-xiy@10.6.142.16}
repo=${EXP016_REPO:-/home/xiy/workspace/flashinfer_exp015_748ad}
relative_exp=.claude/w4a4_moe_bench/experiments/exp_016_route_q0_token_major_reuse
container=${EXP016_CONTAINER:-xiyExp0165kp}
host_python=${EXP016_HOST_PYTHON:-python3}
gpu_uuid=${EXP016_GPU_UUID:-GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
lease_id=${EXP016_LEASE_ID:-exp016-route-q0-20260720}
lease_dir=${EXP016_LEASE_DIR:-/tmp/kdk-direct-ssh-gpu-leases/R6KD-CX8aaS-GPU-16_GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
candidate_jit=${EXP016_CANDIDATE_JIT:-/home/xiy/workspace/exp016_route_q0_jit/candidate_token_major_reuse}
validation_relative=${EXP016_VALIDATION_RELATIVE:-raw/validation/candidate_token_major_reuse/m8192/canonical_unequal/case.json}

if [[ $on_host == 0 ]]; then
  remote_script=$repo/$relative_exp/run_dynamic_spill_ncu_remote.sh
  remote=(
    env
    EXP016_REPO="$repo"
    EXP016_CONTAINER="$container"
    EXP016_HOST_PYTHON="$host_python"
    EXP016_GPU_UUID="$gpu_uuid"
    EXP016_LEASE_ID="$lease_id"
    EXP016_LEASE_DIR="$lease_dir"
    EXP016_CANDIDATE_JIT="$candidate_jit"
    EXP016_VALIDATION_RELATIVE="$validation_relative"
    bash "$remote_script" --on-host "$mode"
  )
  printf -v remote_command '%q ' "${remote[@]}"
  exec ssh -T "$host" "$remote_command"
fi

exp=$repo/$relative_exp
results=$exp/results
validation=$results/$validation_relative
capture=$exp/capture_dynamic_spill_ncu.py
builder=$exp/build_dynamic_spill_evidence.py
pythonpath=$repo:/home/xiy/workspace/w4a4_deps_460:/home/xiy/workspace/w4a4_deps_460/nvidia_cutlass_dsl/dsl_packages
expected_clock=2377

[[ -f $capture && -f $builder && -f $validation ]]

if [[ $mode == build ]]; then
  exec "$host_python" "$builder" \
    --results "$results" \
    --validation "$validation"
fi

[[ -d $candidate_jit ]] || {
  echo "missing correctness-validated Candidate JIT root: $candidate_jit" >&2
  exit 4
}

if [[ $mode == capture ]]; then
  [[ -f $lease_dir/metadata ]]
  grep -qx "lease_id=$lease_id" "$lease_dir/metadata"
  grep -qx "gpu_uuid=$gpu_uuid" "$lease_dir/metadata"
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
  if [[ $observed_uuid != "$gpu_uuid" || $observed_clock != "$expected_clock" ]]; then
    echo "GPU/clock identity drift: $gpu_line" >&2
    exit 6
  fi
fi

command=(
  python3 "$capture"
  --flashinfer-root "$repo"
  --results "$results"
  --validation "$validation"
  --jit-root "$candidate_jit"
)
if [[ $mode == dry-run ]]; then
  command+=(--dry-run)
fi

mkdir -p "$results/runtime"
label=dynamic_spill_ncu_candidate_m8192
{
  printf 'docker exec --privileged '
  printf '%q ' \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTHONPATH="$pythonpath" \
    -e FLASHINFER_NVFP4_4OVER6=0 \
    -e FLASHINFER_WORKSPACE_BASE="$candidate_jit" \
    -e CUTE_DSL_CACHE_DIR="$candidate_jit/cache" \
    -e CUTE_DSL_DUMP_DIR="$candidate_jit/dump" \
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
  -e FLASHINFER_WORKSPACE_BASE="$candidate_jit" \
  -e CUTE_DSL_CACHE_DIR="$candidate_jit/cache" \
  -e CUTE_DSL_DUMP_DIR="$candidate_jit/dump" \
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
  "$host_python" "$builder" \
    --results "$results" \
    --validation "$validation"
fi
