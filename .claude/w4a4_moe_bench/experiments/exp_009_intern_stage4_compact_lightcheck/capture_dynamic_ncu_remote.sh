#!/usr/bin/env bash
# Run this on the direct-SSH GPU host; it consumes the existing exp_009 lease.
set -euo pipefail

mode=${1:?usage: capture_dynamic_ncu_remote.sh dry-run|capture}
case "$mode" in
  dry-run|capture) ;;
  *) echo "unsupported mode: $mode" >&2; exit 2 ;;
esac

gpu_uuid=${EXP009_GPU_UUID:-GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
expected_app_clock_mhz=2377
lease_id=${EXP009_LEASE_ID:-exp009-stage4-20260719T142457Z-carstydev}
lease_dir=${EXP009_LEASE_DIR:-/tmp/kdk-direct-ssh-gpu-leases/R6KD-CX8aaS-GPU-16_GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
host_root=${EXP009_REPO:-/home/xiy/workspace/flashinfer_exp008_748ad}
host_root_real=$(readlink -f "$host_root")
host_git=${EXP009_PARENT_GIT:-/lustre/raplab/client/xiy/workspace/flashinfer/.git}
host_cutlass=$host_root/3rdparty/cutlass
host_submodule_root=${EXP009_SUBMODULE_ROOT:-/home/xiy/workspace/flashinfer_exp002_074d93e}
host_deps=${EXP009_DEPS:-/home/xiy/workspace/w4a4_deps_460}
host_jit=${EXP009_JIT_BASE:-/home/xiy/workspace/exp009_stage4_compact_jit}/compact/m256
container_root=/workspace/source/flashinfer
container_jit=/workspace/jit
relative_exp=.claude/w4a4_moe_bench/experiments/exp_009_intern_stage4_compact_lightcheck
image=nvcr.io/nvidia/pytorch:26.05-py3
uid=$(id -u)
gid=$(id -g)

app_args=(
  python3 "$container_root/$relative_exp/capture_dynamic_ncu.py"
  --flashinfer-root "$container_root"
  --jit-root "$container_jit"
  --expected-gpu-uuid "$gpu_uuid"
  --expected-app-clock-mhz "$expected_app_clock_mhz"
)

if [[ "$mode" == dry-run ]]; then
  docker run --rm --runtime=runc --user "$uid:$gid" \
    --entrypoint python3 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$host_root:$container_root:ro" \
    -v "$host_jit:$container_jit:ro" \
    -w "$container_root" \
    "$image" \
    "$container_root/$relative_exp/capture_dynamic_ncu.py" \
    --flashinfer-root "$container_root" \
    --jit-root "$container_jit" \
    --expected-gpu-uuid "$gpu_uuid" \
    --expected-app-clock-mhz "$expected_app_clock_mhz" \
    --dry-run
  exit 0
fi

grep -qx "lease_id=$lease_id" "$lease_dir/metadata"
grep -qx "gpu_uuid=$gpu_uuid" "$lease_dir/metadata"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
  | grep -qx "$gpu_uuid"; then
  echo "leased GPU has a foreign compute process" >&2
  exit 3
fi
gpu_line=$(nvidia-smi --id="$gpu_uuid" \
  --query-gpu=uuid,clocks.applications.graphics \
  --format=csv,noheader,nounits)
observed_uuid=$(printf '%s' "$gpu_line" | cut -d, -f1 | xargs)
observed_app_clock=$(printf '%s' "$gpu_line" | cut -d, -f2 | xargs)
if [[ "$observed_uuid" != "$gpu_uuid" || "$observed_app_clock" != "$expected_app_clock_mhz" ]]; then
  echo "GPU/clock identity drift: $gpu_line" >&2
  exit 3
fi

launcher_start_identity=$(ps -p "$$" -o lstart= | sed 's/^ *//')
sed -i '/^launcher_pid=/d; /^launcher_start_identity=/d' "$lease_dir/metadata"
printf 'launcher_pid=%s\nlauncher_start_identity=%s\n' \
  "$$" "$launcher_start_identity" >> "$lease_dir/metadata"

cleanup() {
  status=$?
  trap - EXIT
  docker run --rm --runtime=runc --entrypoint chown \
    -v "$host_root:$host_root" \
    "$image" \
    -R "$uid:$gid" \
      "$host_root/$relative_exp/results/ncu/candidate_m256" \
      "$host_root/$relative_exp/results/canonical/profile_targets/candidate_4warp_stage4_compact/m256" \
    >/dev/null 2>&1 || true
  if [[ -f "$lease_dir/metadata" ]] \
    && grep -qx "lease_id=$lease_id" "$lease_dir/metadata" \
    && grep -qx "launcher_pid=$$" "$lease_dir/metadata"; then
    sed -i '/^launcher_pid=/d; /^launcher_start_identity=/d' "$lease_dir/metadata"
    printf 'launcher_pid=pending\nlauncher_start_identity=pending\n' >> "$lease_dir/metadata"
  fi
  exit "$status"
}
trap cleanup EXIT

docker_args=(
  docker run --rm
  --gpus "device=$gpu_uuid"
  --cap-add SYS_ADMIN
  -v "$host_root:$container_root"
  -v "$host_root_real:$host_root_real:ro"
  -v "$host_git:$host_git:ro"
  -v "$host_cutlass:$container_root/3rdparty/cutlass:ro"
  -v "$host_submodule_root:$host_submodule_root:ro"
  -v "$host_deps:/workspace/deps:ro"
  -v "$host_jit:$container_jit"
  -e HOME=/tmp
  -e PYTHONDONTWRITEBYTECODE=1
  -e PYTHONPATH="$container_root:/workspace/deps:/workspace/deps/nvidia_cutlass_dsl/dsl_packages"
  -e CUDA_VISIBLE_DEVICES=0
  -e KDK_LEASE_ID="$lease_id"
  -e KDK_LEASE_GPU_UUID="$gpu_uuid"
  -e W4A4_IMAGE_DIGEST=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba
  -e W4A4_PYTHON_DEPS_SHA256=32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74
  -e FLASHINFER_NVFP4_4OVER6=0
  -e FLASHINFER_WORKSPACE_BASE="$container_jit"
  -e CUTE_DSL_CACHE_DIR="$container_jit/cache"
  -e CUTE_DSL_DUMP_DIR="$container_jit/dump"
  -e CUTE_DSL_KEEP=ir,ptx,cubin,sass
  -w "$container_root"
  "$image"
)

"${docker_args[@]}" "${app_args[@]}"
