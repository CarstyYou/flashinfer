#!/usr/bin/env bash
set -euo pipefail

arm=${1:?usage: capture_ncu_remote.sh anchor|candidate}
case "$arm" in
  anchor|candidate) ;;
  *) echo "unsupported arm: $arm" >&2; exit 2 ;;
esac

gpu_uuid=GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522
lease_id=exp007-native-n64-20260719-carstydev
lease_dir=/tmp/kdk-direct-ssh-gpu-leases/R6KD-CX8aaS-GPU-16_GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522
host_root=/home/xiy/workspace/flashinfer_exp007_748ad
host_git=/lustre/raplab/client/xiy/workspace/flashinfer/.git
host_cutlass=/home/xiy/workspace/flashinfer_exp005_748ad/3rdparty/cutlass
host_submodule_root=/home/xiy/workspace/flashinfer_exp002_074d93e
host_deps=/home/xiy/workspace/w4a4_deps_460
host_jit=/home/xiy/workspace/exp007_native_n64_jit/$arm/m8192/canonical
container_root=/workspace/source/flashinfer
relative_exp=.claude/w4a4_moe_bench/experiments/exp_007_native_n64_spill_reduction
runtime_dir=$host_root/$relative_exp/results/runtime
label=ncu_${arm}_m8192_canonical_v0
uid=$(id -u)
gid=$(id -g)

grep -qx "lease_id=$lease_id" "$lease_dir/metadata"
grep -qx "gpu_uuid=$gpu_uuid" "$lease_dir/metadata"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
  | grep -qx "$gpu_uuid"; then
  echo "leased GPU has a foreign compute process" >&2
  exit 3
fi

mkdir -p "$runtime_dir"
launcher_start_identity=$(ps -p "$$" -o lstart= | sed 's/^ *//')
sed -i '/^launcher_pid=/d; /^launcher_start_identity=/d' "$lease_dir/metadata"
printf 'launcher_pid=%s\nlauncher_start_identity=%s\n' \
  "$$" "$launcher_start_identity" >> "$lease_dir/metadata"

cleanup() {
  status=$?
  trap - EXIT
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
    --format=csv,noheader,nounits > "$runtime_dir/post_${label}_gpu.csv" 2>/dev/null || true
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits > "$runtime_dir/post_${label}_processes.csv" 2>/dev/null || true
  docker run --rm --entrypoint chown \
    -v "$host_root:$host_root" \
    nvcr.io/nvidia/pytorch:26.05-py3 \
    -R "$uid:$gid" "$host_root/$relative_exp/results/ncu" \
      "$host_root/$relative_exp/results/canonical/profile_targets" \
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
  -v "$host_git:$host_git:ro"
  -v "$host_cutlass:$container_root/3rdparty/cutlass:ro"
  -v "$host_submodule_root:$host_submodule_root:ro"
  -v "$host_deps:/workspace/deps:ro"
  -v "$host_jit:/workspace/jit"
  -e HOME=/tmp
  -e PYTHONPATH="$container_root:/workspace/deps:/workspace/deps/nvidia_cutlass_dsl/dsl_packages"
  -e CUDA_VISIBLE_DEVICES=0
  -e KDK_LEASE_ID="$lease_id"
  -e KDK_LEASE_GPU_UUID="$gpu_uuid"
  -e W4A4_IMAGE_DIGEST=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba
  -e W4A4_PYTHON_DEPS_SHA256=32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74
  -e FLASHINFER_NVFP4_4OVER6=0
  -e FLASHINFER_WORKSPACE_BASE=/workspace/jit
  -e CUTE_DSL_CACHE_DIR=/workspace/jit/cache
  -e CUTE_DSL_DUMP_DIR=/workspace/jit/dump
  -e CUTE_DSL_KEEP=ir,ptx,cubin,sass
  -w "$container_root"
  nvcr.io/nvidia/pytorch:26.05-py3
)

app_args=(
  python3 "$container_root/$relative_exp/capture_dynamic_spill.py"
  --flashinfer-root "$container_root"
  --jit-root /workspace/jit
  --expected-gpu-uuid "$gpu_uuid"
  --arm "$arm"
)

{
  printf '%q ' "${docker_args[@]}" "${app_args[@]}"
  printf '\n'
} > "$runtime_dir/$label.command.txt"

"${docker_args[@]}" "${app_args[@]}" \
  > >(tee "$runtime_dir/$label.stdout.log") \
  2> >(tee "$runtime_dir/$label.stderr.log" >&2)
