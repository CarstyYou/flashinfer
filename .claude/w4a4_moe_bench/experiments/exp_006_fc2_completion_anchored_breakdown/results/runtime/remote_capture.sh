#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 measurement_no_marker|completion_anchored_probe" >&2
  exit 2
fi

arm=$1
case "$arm" in
  measurement_no_marker|completion_anchored_probe) ;;
  *) echo "invalid arm: $arm" >&2; exit 2 ;;
esac

gpu_uuid=GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522
lease_id=exp006-fc2-abi339-20260718T172656Z-carstydev
lease_dir=/tmp/kdk-direct-ssh-gpu-leases/R6KD-CX8aaS-GPU-16_GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522
host_root=/home/xiy/workspace/flashinfer_exp006_748ad
host_git=/lustre/raplab/client/xiy/workspace/flashinfer/.git
host_deps=/home/xiy/workspace/w4a4_deps_460
host_jit=/home/xiy/workspace/exp006_fc2_abi339_jit_${arm}
container_root=/workspace/source/flashinfer
relative_exp=.claude/w4a4_moe_bench/experiments/exp_006_fc2_completion_anchored_breakdown
container_exp=$container_root/$relative_exp
runtime_dir=$host_root/$relative_exp/results/runtime
output_dir=$container_exp/results/raw/$arm

grep -qx "lease_id=$lease_id" "$lease_dir/metadata"
grep -qx "gpu_uuid=$gpu_uuid" "$lease_dir/metadata"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
  | grep -qx "$gpu_uuid"; then
  echo "leased GPU acquired a foreign compute process before launch" >&2
  exit 3
fi
if [[ -e "$host_jit" ]]; then
  echo "fresh JIT root already exists: $host_jit" >&2
  exit 4
fi
if [[ -e "$host_root/$relative_exp/results/raw/$arm" ]]; then
  echo "immutable capture output already exists for $arm" >&2
  exit 4
fi

mkdir -p "$host_jit" "$runtime_dir"
launcher_start_identity=$(ps -p "$$" -o lstart= | sed 's/^ *//')
sed -i '/^launcher_pid=/d; /^launcher_start_identity=/d' "$lease_dir/metadata"
printf 'launcher_pid=%s\nlauncher_start_identity=%s\n' \
  "$$" "$launcher_start_identity" >> "$lease_dir/metadata"

cleanup_launcher_record() {
  if grep -qx "lease_id=$lease_id" "$lease_dir/metadata" \
    && grep -qx "launcher_pid=$$" "$lease_dir/metadata" \
    && grep -qx "launcher_start_identity=$launcher_start_identity" "$lease_dir/metadata"; then
    sed -i '/^launcher_pid=/d; /^launcher_start_identity=/d' "$lease_dir/metadata"
    printf 'launcher_pid=pending\nlauncher_start_identity=pending\n' >> "$lease_dir/metadata"
  fi
}
trap cleanup_launcher_record EXIT

docker_args=(
  docker run --rm
  --user 1036:1036
  --gpus "device=$gpu_uuid"
  -v "$host_root:$container_root"
  -v "$host_root:/lustre/raplab/client/xiy/workspace/flashinfer_exp006_748ad"
  -v "$host_git:$host_git:ro"
  -v "$host_deps:/workspace/deps:ro"
  -v "$host_jit:/workspace/jit"
  -e HOME=/tmp
  -e PYTHONPATH="$container_root:/workspace/deps:/workspace/deps/nvidia_cutlass_dsl/dsl_packages"
  -e CUDA_VISIBLE_DEVICES="$gpu_uuid"
  -e KDK_LEASE_ID="$lease_id"
  -e KDK_LEASE_GPU_UUID="$gpu_uuid"
  -e W4A4_IMAGE_DIGEST=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba
  -e W4A4_PYTHON_DEPS_SHA256=32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74
  -e FLASHINFER_NVFP4_4OVER6=0
  -e FLASHINFER_WORKSPACE_BASE=/workspace/jit
  -e CUTE_DSL_CACHE_DIR=/workspace/jit/cache
  -e CUTE_DSL_DUMP_DIR=/workspace/jit/cutedsl_dump
  -e CUTE_DSL_KEEP=ir,ptx,cubin,sass
  -e CUTE_DSL_KEEP_IR=1
  -e CUTE_DSL_KEEP_PTX=1
  -e CUTE_DSL_KEEP_CUBIN=1
  -w "$container_root"
  nvcr.io/nvidia/pytorch:26.05-py3
)
capture_args=(
  python3 "$container_exp/capture_completion_timing.py"
  --flashinfer-root "$container_root"
  --arm "$arm"
  --kernel-overlay "$container_exp/results/overlays/$arm/moe_dynamic_kernel.py"
  --dispatch-overlay "$container_exp/results/overlays/$arm/moe_dispatch.py"
  --jit-root /workspace/jit
  --output "$output_dir"
  --expected-gpu-uuid "$gpu_uuid"
  --warmup 2
  --replays 5
)

{
  printf '%q ' "${docker_args[@]}" "${capture_args[@]}"
  printf '\n'
} > "$runtime_dir/${arm}.command.txt"

"${docker_args[@]}" "${capture_args[@]}" \
  > >(tee "$runtime_dir/${arm}.stdout.log") \
  2> >(tee "$runtime_dir/${arm}.stderr.log" >&2)

nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
  --format=csv,noheader,nounits > "$runtime_dir/post_${arm}_gpu.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits > "$runtime_dir/post_${arm}_processes.csv"
