#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 M ARM" >&2
  exit 2
fi
m=$1
arm=$2
case "$m" in 256|8192) ;; *) echo "unsupported M: $m" >&2; exit 2 ;; esac
case "$arm" in
  cutedsl_bf16_fused|cutlass_bf16_chain) ;;
  *) echo "unsupported arm: $arm" >&2; exit 2 ;;
esac

gpu_uuid=${W4A4_GPU_UUID:?set W4A4_GPU_UUID to the leased full GPU UUID}
lease_id=${KDK_LEASE_ID:?set KDK_LEASE_ID to the active advisory lease}
rerun_id=${W4A4_RERUN_ID:?set W4A4_RERUN_ID to the benchmark rerun ID}
host_root=/home/xiy/workspace/flashinfer_exp002_074d93e
host_git_alternate=/lustre/raplab/client/xiy/workspace/flashinfer/.git
host_deps=/home/xiy/workspace/w4a4_deps_460
host_jit=/home/xiy/workspace/exp002_jit_dsl460
container_root=/workspace/source/flashinfer
relative_exp=.claude/w4a4_moe_bench/experiments/exp_002_fused_vs_chain_dataflow
host_exp="$host_root/$relative_exp"
host_out="$host_exp/results/nsys/m${m}/${arm}"
container_out="$container_root/$relative_exp/results/nsys/m${m}/${arm}"

if [[ -e "$host_out/trace.nsys-rep" ]]; then
  echo "refusing to overwrite raw report: $host_out/trace.nsys-rep" >&2
  exit 3
fi
mkdir -p "$host_out"

docker_args=(
  docker run --rm
  --gpus "device=$gpu_uuid"
  --cap-add SYS_ADMIN
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
nsys_args=(
  nsys profile
  --trace=cuda,nvtx,osrt
  --sample=none
  --cpuctxsw=none
  --cuda-graph-trace=node:host-only
  --capture-range=cudaProfilerApi
  --capture-range-end=stop
  --force-overwrite=false
  --output="$container_out/trace"
)
app_args=(
  python3 "$container_root/$relative_exp/run_exp002.py"
  --flashinfer-root "$container_root"
  --results "$container_root/$relative_exp/results"
  --expected-gpu-uuid "$gpu_uuid"
  single-replay --m "$m" --arm "$arm"
)

{
  printf '%q ' "${docker_args[@]}" "${nsys_args[@]}" "${app_args[@]}"
  printf '\n'
} | sed 's/ $//' >"$host_out/command.txt"
"${docker_args[@]}" "${nsys_args[@]}" "${app_args[@]}" \
  >"$host_out/stdout.log" 2>"$host_out/stderr.log"
sha256sum "$host_out/trace.nsys-rep" >"$host_out/trace.nsys-rep.sha256"
docker image inspect nvcr.io/nvidia/pytorch:26.05-py3 --format '{{.Id}}' \
  >"$host_out/docker-image-id.txt"
docker run --rm nvcr.io/nvidia/pytorch:26.05-py3 nsys --version \
  >"$host_out/nsys-version.txt"
profile_manifest="$host_exp/results/manifests/profile_m${m}_${arm}.json"
cp "$profile_manifest" "$host_out/profile_manifest.json"
sha256sum "$host_out/profile_manifest.json" >"$host_out/profile_manifest.json.sha256"
