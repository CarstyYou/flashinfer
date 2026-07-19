#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 prepare|measure production|compact M [group position]" >&2
  exit 2
fi

action=$1
external_arm=$2
m=$3
shift 3

case "$action" in
  prepare|measure) ;;
  *) echo "unsupported action: $action" >&2; exit 2 ;;
esac
case "$external_arm" in
  production) internal_arm=baseline_4warp ;;
  compact) internal_arm=candidate_4warp_stage4_compact ;;
  *) echo "unsupported arm: $external_arm" >&2; exit 2 ;;
esac
case "$m" in
  256|8192) ;;
  *) echo "exp_009 only measures M256/M8192" >&2; exit 2 ;;
esac

if [[ "$action" == measure ]]; then
  if [[ $# -ne 2 ]]; then
    echo "measure requires group and position" >&2
    exit 2
  fi
  group=$1
  position=$2
  [[ "$group" =~ ^[0-2]$ ]] || { echo "group must be 0..2" >&2; exit 2; }
  [[ "$position" =~ ^[0-3]$ ]] || { echo "position must be 0..3" >&2; exit 2; }
elif [[ $# -ne 0 ]]; then
  echo "prepare takes no trailing arguments" >&2
  exit 2
fi

repo=${EXP009_REPO:-/home/xiy/workspace/flashinfer_exp008_748ad}
repo_real=$(readlink -f "$repo")
parent_git=${EXP009_PARENT_GIT:-/lustre/raplab/client/xiy/workspace/flashinfer/.git}
host_cutlass=$repo/3rdparty/cutlass
host_submodule_root=${EXP009_SUBMODULE_ROOT:-/home/xiy/workspace/flashinfer_exp002_074d93e}
relative_exp=.claude/w4a4_moe_bench/experiments/exp_009_intern_stage4_compact_lightcheck
exp=$repo/$relative_exp
results=$exp/results/canonical
runtime=$exp/results/runtime
deps=${EXP009_DEPS:-/home/xiy/workspace/w4a4_deps_460}
jit_base=${EXP009_JIT_BASE:-/home/xiy/workspace/exp009_stage4_compact_jit}
jit=$jit_base/$external_arm/m$m
container_repo=/workspace/source/flashinfer
container_exp=$container_repo/$relative_exp
container_results=$container_exp/results/canonical
container_jit=/workspace/jit
image=nvcr.io/nvidia/pytorch:26.05-py3
gpu_uuid=${EXP009_GPU_UUID:-GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
lease_id=${EXP009_LEASE_ID:-exp009-stage4-20260719T142457Z-carstydev}
lease_dir=${EXP009_LEASE_DIR:-/tmp/kdk-direct-ssh-gpu-leases/R6KD-CX8aaS-GPU-16_GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
expected_app_clock_mhz=2377
image_digest=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba
deps_sha256=32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74

if [[ "$external_arm" == production ]]; then
  overlay=$container_repo/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py
else
  overlay=$container_exp/results/overlays/intern_stage4_compact/moe_dynamic_kernel.py
fi

mkdir -p "$runtime/logs" "$jit"
grep -qx "lease_id=$lease_id" "$lease_dir/metadata"
grep -qx "gpu_uuid=$gpu_uuid" "$lease_dir/metadata"
observed_uuid=$(nvidia-smi --id="$gpu_uuid" --query-gpu=uuid --format=csv,noheader,nounits | xargs)
if [[ "$observed_uuid" != "$gpu_uuid" ]]; then
  echo "GPU UUID drift: $observed_uuid != $gpu_uuid" >&2
  exit 3
fi
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
    | grep -qx "$gpu_uuid"; then
  echo "foreign compute process detected on $gpu_uuid" >&2
  exit 3
fi
observed_app_clock=$(nvidia-smi --id="$gpu_uuid" \
  --query-gpu=clocks.applications.graphics --format=csv,noheader,nounits | xargs)
if [[ "$observed_app_clock" != "$expected_app_clock_mhz" ]]; then
  echo "application graphics clock drift: $observed_app_clock != $expected_app_clock_mhz" >&2
  exit 3
fi

launcher_start_identity=$(ps -p "$$" -o lstart= | sed 's/^ *//')
sed -i '/^launcher_pid=/d; /^launcher_start_identity=/d' "$lease_dir/metadata"
printf 'launcher_pid=%s\nlauncher_start_identity=%s\n' \
  "$$" "$launcher_start_identity" >> "$lease_dir/metadata"

cleanup_launcher_record() {
  if [[ -f "$lease_dir/metadata" ]] \
    && grep -qx "lease_id=$lease_id" "$lease_dir/metadata" \
    && grep -qx "launcher_pid=$$" "$lease_dir/metadata"; then
    sed -i '/^launcher_pid=/d; /^launcher_start_identity=/d' "$lease_dir/metadata"
    printf 'launcher_pid=pending\nlauncher_start_identity=pending\n' >> "$lease_dir/metadata"
  fi
}
trap cleanup_launcher_record EXIT

uid=$(id -u)
gid=$(id -g)
docker_args=(
  docker run --rm
  --gpus "device=$gpu_uuid"
  --user "$uid:$gid"
  -v "$repo:$container_repo"
  -v "$repo_real:$repo_real:ro"
  -v "$parent_git:$parent_git:ro"
  -v "$host_cutlass:$container_repo/3rdparty/cutlass:ro"
  -v "$host_submodule_root:$host_submodule_root:ro"
  -v "$deps:/workspace/deps:ro"
  -v "$jit:$container_jit"
  -e HOME=/tmp
  -e PYTHONDONTWRITEBYTECODE=1
  -e PYTHONPATH="$container_repo:/workspace/deps:/workspace/deps/nvidia_cutlass_dsl/dsl_packages"
  -e CUDA_VISIBLE_DEVICES=0
  -e KDK_LEASE_ID="$lease_id"
  -e KDK_LEASE_GPU_UUID="$gpu_uuid"
  -e W4A4_IMAGE_DIGEST="$image_digest"
  -e W4A4_PYTHON_DEPS_SHA256="$deps_sha256"
  -e FLASHINFER_NVFP4_4OVER6=0
  -e FLASHINFER_WORKSPACE_BASE="$container_jit"
  -e CUTE_DSL_CACHE_DIR="$container_jit/cache"
  -e CUTE_DSL_DUMP_DIR="$container_jit/dump"
  -e CUTE_DSL_KEEP=ir,ptx,cubin,sass
  -w "$container_repo"
  "$image"
)

worker=(
  python3 "$container_exp/run_exp009_arm.py"
  --flashinfer-root "$container_repo"
  --results "$container_results"
  --arm "$internal_arm"
  --m "$m"
  --fixture canonical
  --overlay "$overlay"
  --jit-root "$container_jit"
  --expected-gpu-uuid "$gpu_uuid"
  --comparison-anchor baseline_4warp
  --comparison-subject candidate_4warp_stage4_compact
)

if [[ "$action" == prepare ]]; then
  worker+=(prepare)
  label=prepare_${external_arm}_m${m}
else
  worker+=(
    measure
    --group "$group"
    --position "$position"
    --warmup 5
    --iters 50
    --clock-policy locked
  )
  label=measure_g${group}_p${position}_${external_arm}_m${m}
fi

{
  printf '%q ' "${docker_args[@]}" "${worker[@]}"
  printf '\n'
} > "$runtime/logs/$label.command.txt"

"${docker_args[@]}" "${worker[@]}" \
  > >(tee "$runtime/logs/$label.stdout.log") \
  2> >(tee "$runtime/logs/$label.stderr.log" >&2)

nvidia-smi --id="$gpu_uuid" \
  --query-gpu=uuid,clocks.current.graphics,clocks.applications.graphics,memory.used,utilization.gpu \
  --format=csv,noheader,nounits > "$runtime/logs/$label.post_gpu.csv"
