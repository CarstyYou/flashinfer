#!/usr/bin/env bash
set -euo pipefail

gpu_uuid=${W4A4_GPU_UUID:?set W4A4_GPU_UUID to the leased full GPU UUID}
lease_id=${KDK_LEASE_ID:?set KDK_LEASE_ID to the active advisory lease}
rerun_id=${W4A4_RERUN_ID:?set W4A4_RERUN_ID to the common fresh rerun ID}

host_root=${W4A4_HOST_ROOT:-/home/xiy/workspace/flashinfer_exp001_corrected_074d93e}
container_root=/workspace/source/flashinfer
relative_exp=.claude/w4a4_moe_bench/experiments/exp_001_backend_case_sweep
host_results="$host_root/$relative_exp/results/sglang_triton"
container_results="$container_root/$relative_exp/results/sglang_triton"
container_fixtures="$container_root/$relative_exp/results/fixtures"
host_jit=${W4A4_SGLANG_HOST_JIT:-/home/xiy/workspace/exp001_sglang_triton_jit}
image=lmsysorg/sglang:latest
image_digest=sha256:00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2
image_id=sha256:663867442f321ded36228bafd889fd1db05cbef7a7c8ea6e072df33234dabbfd
sglang_commit=0b3bb0cbe31873994c9f989fddfe2f87ca839fdd

actual_image_id=$(docker image inspect "$image" --format '{{.Id}}')
repo_digests=$(docker image inspect "$image" --format '{{join .RepoDigests " "}}')
if [[ "$actual_image_id" != "$image_id" || "$repo_digests" != *"lmsysorg/sglang@$image_digest"* ]]; then
  echo "SGLang image drift: id=$actual_image_id digests=$repo_digests" >&2
  exit 3
fi
if [[ -e "$host_results/evidence.identity.json" ]]; then
  echo "refusing to overwrite an existing canonical SGLang rerun" >&2
  exit 3
fi
mkdir -p "$host_results/manifests" "$host_jit"
if find "$host_jit" -mindepth 1 -print -quit | grep -q .; then
  echo "dedicated SGLang Triton cache is not empty: $host_jit" >&2
  exit 3
fi

docker_args=(
  docker run --rm
  --gpus "device=$gpu_uuid"
  -v "$host_root:$container_root"
  -v "$host_jit:/workspace/jit"
  -e CUDA_VISIBLE_DEVICES=0
  -e KDK_LEASE_ID="$lease_id"
  -e W4A4_RERUN_ID="$rerun_id"
  -e W4A4_IMAGE_DIGEST="$image_digest"
  -e W4A4_IMAGE_ID="$image_id"
  -e W4A4_SGLANG_COMMIT="$sglang_commit"
  -e W4A4_SGLANG_JIT_DIR=/workspace/jit
  -e TRITON_CACHE_DIR=/workspace/jit
  -w "$container_root/$relative_exp"
  "$image"
)
app_args=(
  python3 "$container_root/$relative_exp/bench_triton_fp8.py"
  --fixture-dir "$container_fixtures"
  --results "$container_results"
  --expected-gpu-uuid "$gpu_uuid"
  --m-values 256 512 1024 2048 4096 8192
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
printf '%s\n' "$actual_image_id" >"$host_results/manifests/docker-image-id.txt"
printf '%s\n' "$repo_digests" >"$host_results/manifests/docker-repo-digests.txt"
