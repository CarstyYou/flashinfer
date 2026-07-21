#!/usr/bin/env bash
set -euo pipefail

# Thin remote template for the canonical M8192 Triton capture. This script only
# captures the .nsys-rep; derived evidence must be extracted through VeloQ.
host_root=${EXP017_HOST_ROOT:?set EXP017_HOST_ROOT to the remote FlashInfer checkout}
gpu_uuid=${W4A4_GPU_UUID:?set W4A4_GPU_UUID to the leased full GPU UUID}
lease_id=${KDK_LEASE_ID:?set KDK_LEASE_ID to the active advisory lease}
rerun_id=${W4A4_RERUN_ID:?set W4A4_RERUN_ID to a unique rerun ID}

relative_exp=.claude/w4a4_moe_bench/experiments/exp_017_opt_vs_triton_phase_share
relative_exp001=.claude/w4a4_moe_bench/experiments/exp_001_backend_case_sweep
container_root=/workspace/source/flashinfer
host_results=$host_root/$relative_exp/results
container_results=$container_root/$relative_exp/results
container_fixtures=$container_root/$relative_exp001/results/fixtures
host_jit=${EXP017_SGLANG_HOST_JIT:-/home/xiy/workspace/exp017_sglang_triton_jit/$rerun_id}
container_jit=/workspace/jit
trace_prefix=$container_results/raw/nsys/sglang_triton_fp8_m8192
host_topology_preflight=$host_results/triton_topology_preflight.json
container_topology_preflight=$container_results/triton_topology_preflight.json

image=lmsysorg/sglang:latest
image_digest=sha256:00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2
image_id=sha256:663867442f321ded36228bafd889fd1db05cbef7a7c8ea6e072df33234dabbfd
sglang_commit=0b3bb0cbe31873994c9f989fddfe2f87ca839fdd
expected_app_clock_mhz=2377

actual_image_id=$(docker image inspect "$image" --format '{{.Id}}')
repo_digests=$(docker image inspect "$image" --format '{{join .RepoDigests " "}}')
if [[ "$actual_image_id" != "$image_id" || "$repo_digests" != *"lmsysorg/sglang@$image_digest"* ]]; then
  echo "SGLang image drift: id=$actual_image_id digests=$repo_digests" >&2
  exit 3
fi
observed_uuid=$(nvidia-smi --id="$gpu_uuid" --query-gpu=uuid --format=csv,noheader,nounits | xargs)
[[ "$observed_uuid" == "$gpu_uuid" ]] || { echo "GPU UUID drift" >&2; exit 3; }
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
    | grep -qx "$gpu_uuid"; then
  echo "foreign compute process detected on leased GPU" >&2
  exit 3
fi
observed_app_clock=$(nvidia-smi --id="$gpu_uuid" \
  --query-gpu=clocks.applications.graphics --format=csv,noheader,nounits | xargs)
[[ "$observed_app_clock" == "$expected_app_clock_mhz" ]] || {
  echo "application clock drift: $observed_app_clock MHz" >&2
  exit 3
}

mkdir -p "$host_results/raw/nsys" "$host_results/runtime" "$host_jit"
if find "$host_jit" -mindepth 1 -print -quit | grep -q .; then
  echo "dedicated SGLang Triton cache is not empty: $host_jit" >&2
  exit 3
fi
if [[ -e "$host_topology_preflight" \
      || -e "$host_results/triton_capture_manifest.json" \
      || -e "$host_results/raw/nsys/sglang_triton_fp8_m8192.nsys-rep" ]]; then
  echo "refusing to overwrite existing exp017 Triton evidence" >&2
  exit 3
fi

docker_args=(
  docker run --rm
  --gpus "device=$gpu_uuid"
  -v "$host_root:$container_root"
  -v "$host_jit:$container_jit"
  -e CUDA_VISIBLE_DEVICES=0
  -e PYTHONDONTWRITEBYTECODE=1
  -e KDK_LEASE_ID="$lease_id"
  -e W4A4_RERUN_ID="$rerun_id"
  -e W4A4_IMAGE_DIGEST="$image_digest"
  -e W4A4_IMAGE_ID="$image_id"
  -e W4A4_SGLANG_COMMIT="$sglang_commit"
  -e W4A4_SGLANG_JIT_DIR="$container_jit"
  -e TRITON_CACHE_DIR="$container_jit"
  -w "$container_root/$relative_exp"
  "$image"
)
preflight_command=(
  python3 "$container_root/$relative_exp/capture_triton_nsys.py"
  --fixture-dir "$container_fixtures"
  --results "$container_results"
  --topology-preflight "$container_topology_preflight"
  --preflight-only
  --expected-gpu-uuid "$gpu_uuid"
  --expected-app-clock-mhz "$expected_app_clock_mhz"
)
capture_command=(
  nsys profile
  --force-overwrite=false
  --trace=cuda,nvtx,osrt
  --sample=none
  --cpuctxsw=none
  --cuda-graph-trace=node:host-only
  --capture-range=cudaProfilerApi
  --capture-range-end=stop
  --output="$trace_prefix"
  python3 "$container_root/$relative_exp/capture_triton_nsys.py"
  --fixture-dir "$container_fixtures"
  --results "$container_results"
  --topology-preflight "$container_topology_preflight"
  --expected-gpu-uuid "$gpu_uuid"
  --expected-app-clock-mhz "$expected_app_clock_mhz"
  --warmup 5
  --replays 5
  --l2-flush-bytes 201326592
)

{
  printf '%q ' "${docker_args[@]}" "${preflight_command[@]}"
  printf '\n'
  printf '%q ' "${docker_args[@]}" "${capture_command[@]}"
  printf '\n'
} >"$host_results/runtime/triton_nsys.command.txt"

"${docker_args[@]}" "${preflight_command[@]}" \
  > >(tee "$host_results/runtime/triton_topology_preflight.stdout.log") \
  2> >(tee "$host_results/runtime/triton_topology_preflight.stderr.log" >&2)

"${docker_args[@]}" "${capture_command[@]}" \
  > >(tee "$host_results/runtime/triton_nsys.stdout.log") \
  2> >(tee "$host_results/runtime/triton_nsys.stderr.log" >&2)

nvidia-smi --id="$gpu_uuid" \
  --query-gpu=uuid,clocks.current.graphics,clocks.applications.graphics,memory.used,utilization.gpu \
  --format=csv,noheader,nounits >"$host_results/runtime/triton_nsys.post_gpu.csv"

printf '%s\n' \
  "Capture complete. Analyze the .nsys-rep with VeloQ info/summary/graph recipes;" \
  "do not export or query raw sqlite/parquet tables."
