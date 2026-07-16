#!/usr/bin/env bash
set -euo pipefail

gpu_uuid=${W4A4_GPU_UUID:?set W4A4_GPU_UUID to the leased full GPU UUID}
lease_id=${KDK_LEASE_ID:?set KDK_LEASE_ID to the active advisory lease}
rerun_id=${W4A4_RERUN_ID:?set W4A4_RERUN_ID to a unique benchmark rerun ID}

host_root=/home/xiy/workspace/flashinfer_exp002_074d93e
host_git_alternate=/lustre/raplab/client/xiy/workspace/flashinfer/.git
host_deps=/home/xiy/workspace/w4a4_deps_460
host_jit=/home/xiy/workspace/exp002_jit_dsl460
container_root=/workspace/source/flashinfer
relative_exp=.claude/w4a4_moe_bench/experiments/exp_002_fused_vs_chain_dataflow
host_results="$host_root/$relative_exp/results"
container_results="$container_root/$relative_exp/results"

if [[ -e "$host_results/evidence.identity.json" ]]; then
  echo "refusing to overwrite an existing canonical rerun" >&2
  exit 3
fi
mkdir -p "$host_jit" "$host_results/manifests"
if find "$host_jit" -mindepth 1 -print -quit | grep -q .; then
  echo "dedicated JIT workspace is not empty: $host_jit" >&2
  exit 3
fi

docker_args=(
  docker run --rm
  --gpus "device=$gpu_uuid"
  -v "$host_root:$container_root"
  -v "$host_git_alternate:$host_git_alternate:ro"
  -v "$host_deps:/workspace/deps:ro"
  -v "$host_jit:/workspace/jit"
  -e PYTHONPATH="$container_root:/workspace/deps:/workspace/deps/nvidia_cutlass_dsl/dsl_packages"
  -e CUDA_VISIBLE_DEVICES=0
  -e KDK_LEASE_ID="$lease_id"
  -e W4A4_RERUN_ID="$rerun_id"
  -e FLASHINFER_WORKSPACE_BASE=/workspace/jit
  -e FLASHINFER_CUTEDSL_IKET_OVERLAY=0
  -e FLASHINFER_NVFP4_4OVER6=0
  -e W4A4_IMAGE_DIGEST=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba
  -e W4A4_PYTHON_DEPS_SHA256=32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74
  -w "$container_root"
  nvcr.io/nvidia/pytorch:26.05-py3
)
app_args=(
  python3 "$container_root/$relative_exp/run_exp002.py"
  --flashinfer-root "$container_root"
  --results "$container_results"
  --expected-gpu-uuid "$gpu_uuid"
  benchmark
  --m-values 256 1024 8192
  --warmup 5
  --iters 50
  --repeats 5
  --l2-flush-bytes 201326592
)

{
  printf 'KDK_LEASE_ID=%q W4A4_RERUN_ID=%q W4A4_GPU_UUID=%q ' \
    "$lease_id" "$rerun_id" "$gpu_uuid"
  printf '%q ' "${docker_args[@]}" "${app_args[@]}"
  printf '\n'
} | sed 's/ $//' >"$host_results/manifests/benchmark_command.txt"

"${docker_args[@]}" "${app_args[@]}" \
  > >(tee "$host_results/manifests/benchmark_stdout.log") \
  2> >(tee "$host_results/manifests/benchmark_stderr.log" >&2)
docker image inspect nvcr.io/nvidia/pytorch:26.05-py3 --format '{{.Id}}' \
  >"$host_results/manifests/docker-image-id.txt"
