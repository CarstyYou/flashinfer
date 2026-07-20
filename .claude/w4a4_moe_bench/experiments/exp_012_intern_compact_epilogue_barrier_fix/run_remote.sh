#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 baseline-control|candidate" >&2
  exit 2
fi
action=$1
case "$action" in baseline-control|candidate) ;; *) exit 2 ;; esac

repo=${EXP012_REPO:-/home/xiy/workspace/flashinfer_exp008_748ad}
repo_real=$(readlink -f "$repo")
relative_exp=.claude/w4a4_moe_bench/experiments/exp_012_intern_compact_epilogue_barrier_fix
exp=$repo/$relative_exp
results=$exp/results
runtime=$results/runtime
container_repo=/workspace/source/flashinfer
container_exp=$container_repo/$relative_exp
deps=${EXP012_DEPS:-/home/xiy/workspace/w4a4_deps_460}
image=nvcr.io/nvidia/pytorch:26.05-py3
gpu_uuid=${EXP012_GPU_UUID:-GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
lease_id=${EXP012_LEASE_ID:-exp012-intern-barrier-20260719}
lease_dir=${EXP012_LEASE_DIR:-/tmp/kdk-direct-ssh-gpu-leases/R6KD-CX8aaS-GPU-16_GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
expected_app_clock_mhz=2377
host_git=/lustre/raplab/client/xiy/workspace/flashinfer/.git
host_cutlass=$repo/3rdparty/cutlass
host_submodule_root=/home/xiy/workspace/flashinfer_exp002_074d93e

mkdir -p "$runtime"
grep -qx "lease_id=$lease_id" "$lease_dir/metadata"
grep -qx "gpu_uuid=$gpu_uuid" "$lease_dir/metadata"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | grep -qx "$gpu_uuid"; then
  echo "foreign compute process detected on leased GPU" >&2
  exit 3
fi
observed=$(nvidia-smi --id="$gpu_uuid" --query-gpu=uuid,clocks.applications.graphics --format=csv,noheader,nounits)
[[ "$(printf '%s' "$observed" | cut -d, -f1 | xargs)" == "$gpu_uuid" ]]
[[ "$(printf '%s' "$observed" | cut -d, -f2 | xargs)" == "$expected_app_clock_mhz" ]]

launcher_start_identity=$(ps -p "$$" -o lstart= | sed 's/^ *//')
sed -i '/^launcher_pid=/d; /^launcher_start_identity=/d' "$lease_dir/metadata"
printf 'launcher_pid=%s\nlauncher_start_identity=%s\n' "$$" "$launcher_start_identity" >> "$lease_dir/metadata"
cleanup_launcher_record() {
  if grep -qx "lease_id=$lease_id" "$lease_dir/metadata" && grep -qx "launcher_pid=$$" "$lease_dir/metadata"; then
    sed -i '/^launcher_pid=/d; /^launcher_start_identity=/d' "$lease_dir/metadata"
    printf 'launcher_pid=pending\nlauncher_start_identity=pending\n' >> "$lease_dir/metadata"
  fi
}
trap cleanup_launcher_record EXIT

if [[ "$action" == baseline-control ]]; then
  overlay=$repo/.claude/w4a4_moe_bench/experiments/exp_009_intern_stage4_compact_lightcheck/results/overlays/intern_stage4_compact/moe_dynamic_kernel.py
  expected_sha=42ca8d40e18b5d0f001236b09b85cbc0aa30e6010f0954efd538d8b9a2fb57d2
  jit=/home/xiy/workspace/exp012_intern_barrier_baseline_jit
  output=$results/raw/baseline_control_m8192.json
  ms=(8192)
  replays=3
  [[ ! -e "$jit" ]] || { echo "baseline JIT root already exists: $jit" >&2; exit 4; }
  mkdir -p "$jit"
else
  overlay=$results/overlays/post_quant_barrier/moe_dynamic_kernel.py
  expected_sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate_sha256"])' "$results/overlays/post_quant_barrier/identity.json")
  jit=/home/xiy/workspace/exp012_intern_barrier_jit
  output=$results/raw/candidate_m4096_m8192_m2048.json
  ms=(4096 8192 2048)
  replays=4
  [[ ! -e "$jit" ]] || { echo "candidate JIT root already exists: $jit" >&2; exit 4; }
  mkdir -p "$jit"
fi
[[ -f "$overlay" && -d "$jit" && ! -e "$output" ]]

uid=$(id -u)
gid=$(id -g)
container_jit=/workspace/jit
container_output=$container_exp/${output#"$exp/"}
container_overlay=$container_exp/${overlay#"$exp/"}
if [[ "$action" == baseline-control ]]; then
  container_overlay=$container_repo/${overlay#"$repo/"}
fi
docker_args=(
  docker run --rm --user "$uid:$gid" --gpus "device=$gpu_uuid"
  -v "$repo:$container_repo"
  -v "$repo_real:$repo_real:ro"
  -v "$host_git:$host_git:ro"
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
  -e W4A4_IMAGE_DIGEST=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba
  -e W4A4_PYTHON_DEPS_SHA256=32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74
  -e FLASHINFER_NVFP4_4OVER6=0
  -e FLASHINFER_WORKSPACE_BASE="$container_jit"
  -e CUTE_DSL_CACHE_DIR="$container_jit/cache"
  -e CUTE_DSL_DUMP_DIR="$container_jit/dump"
  -e CUTE_DSL_KEEP=ir,ptx,cubin,sass
  -e TORCH_CUDA_ARCH_LIST=12.0a
  -w "$container_repo"
  "$image"
)
command=(
  python3 "$container_exp/run_correctness.py"
  --flashinfer-root "$container_repo"
  --overlay "$container_overlay"
  --expected-overlay-sha256 "$expected_sha"
  --jit-root "$container_jit"
  --output "$container_output"
  --expected-gpu-uuid "$gpu_uuid"
  --m "${ms[@]}"
  --replays "$replays"
)
{
  printf '%q ' "${docker_args[@]}" "${command[@]}"
  printf '\n'
} > "$runtime/$action.command.txt"

set +e
"${docker_args[@]}" "${command[@]}" \
  > >(tee "$runtime/$action.stdout.log") \
  2> >(tee "$runtime/$action.stderr.log" >&2)
status=$?
set -e
printf '%s\n' "$status" > "$runtime/$action.exit_code.txt"
exit "$status"
