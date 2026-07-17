#!/usr/bin/env bash
set -euo pipefail

gpu_uuid=GPU-2fdb0b79-0ba7-f356-b714-6c461b71ce12
lease_id=exp005-20260717T092602Z-candidateA
host_root=/home/xiy/workspace/flashinfer_exp005_748ad
host_git=/lustre/raplab/client/xiy/workspace/flashinfer/.git
host_submodule_root=/home/xiy/workspace/flashinfer_exp002_074d93e
host_deps=/home/xiy/workspace/w4a4_deps_460
container_root=/workspace/source/flashinfer
relative_exp=.claude/w4a4_moe_bench/experiments/exp_005_8warp_spill_reduction
container_results="$container_root/$relative_exp/results"
lease_dir=/tmp/kdk-direct-ssh-gpu-leases/R6KD-CX8aaS-GPU-16_GPU-2fdb0b79-0ba7-f356-b714-6c461b71ce12
runtime_dir="$host_root/$relative_exp/results/runtime"
run_label=directed_route_topology

grep -qx "lease_id=$lease_id" "$lease_dir/metadata"
grep -qx "gpu_uuid=$gpu_uuid" "$lease_dir/metadata"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
  | grep -qx "$gpu_uuid"; then
  echo "leased GPU acquired a foreign compute process before launch" >&2
  exit 3
fi

mkdir -p "$runtime_dir"
sed -i '/^launcher_pid=/d; /^launcher_start_identity=/d' "$lease_dir/metadata"
printf 'launcher_pid=%s\nlauncher_start_identity=%s\n' \
  "$$" "$(ps -p "$$" -o lstart= | sed 's/^ *//')" >>"$lease_dir/metadata"

docker_args=(
  docker run --rm
  --gpus "device=$gpu_uuid"
  -v "$host_root:$container_root"
  -v "$host_git:$host_git:ro"
  -v "$host_submodule_root:$host_submodule_root:ro"
  -v "$host_deps:/workspace/deps:ro"
  -e PYTHONPATH="$container_root:/workspace/deps:/workspace/deps/nvidia_cutlass_dsl/dsl_packages"
  -e CUDA_VISIBLE_DEVICES=0
  -e KDK_LEASE_ID="$lease_id"
  -e W4A4_IMAGE_DIGEST=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba
  -e W4A4_PYTHON_DEPS_SHA256=32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74
  -e FLASHINFER_NVFP4_4OVER6=0
  -w "$container_root"
  nvcr.io/nvidia/pytorch:26.05-py3
)
app_args=(
  python3 "$container_root/$relative_exp/run_exp005.py"
  --flashinfer-root "$container_root"
  --results "$container_results"
  prepare-directed
  --expected-gpu-uuid "$gpu_uuid"
)

{
  printf '%q ' "${docker_args[@]}" "${app_args[@]}"
  printf '\n'
} >"$runtime_dir/$run_label.command.txt"

"${docker_args[@]}" "${app_args[@]}" \
  > >(tee "$runtime_dir/$run_label.stdout.log") \
  2> >(tee "$runtime_dir/$run_label.stderr.log" >&2)

nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
  --format=csv,noheader,nounits >"$runtime_dir/post_${run_label}_gpu.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits >"$runtime_dir/post_${run_label}_processes.csv"
