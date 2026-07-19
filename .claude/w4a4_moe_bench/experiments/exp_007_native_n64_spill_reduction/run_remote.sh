#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 prepare|profile anchor|candidate M fixture [worker args...]" >&2
  exit 2
fi

command_name=$1
external_arm=$2
m=$3
fixture=$4
shift 4

case "$command_name" in
  prepare|profile) ;;
  *) echo "invalid command: $command_name" >&2; exit 2 ;;
esac
case "$external_arm" in
  anchor)
    internal_arm=candidate_8warp_serial_v0
    overlay_name=anchor_8warp_n128
    ;;
  candidate)
    internal_arm=candidate_8warp_n64_temporal_replay_v0
    overlay_name=candidate_8warp_native_n64_v0
    ;;
  *) echo "invalid arm: $external_arm" >&2; exit 2 ;;
esac
case "$m" in
  256|8192) ;;
  *) echo "exp_007 only registers M=256 and M=8192, got $m" >&2; exit 2 ;;
esac
case "$fixture" in
  canonical|sparse_empty|exact_128|tail_129|hot_expert|canary_up|canary_gate|canary_up_v1|canary_gate_v1|canary_up_v2|canary_gate_v2) ;;
  *) echo "unsupported fixture: $fixture" >&2; exit 2 ;;
esac
if [[ "$fixture" == canary_* && "$m" != 256 ]]; then
  echo "branch/half/slice canaries are registered only for M=256" >&2
  exit 2
fi

gpu_uuid=GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522
lease_id=exp007-native-n64-20260719-carstydev
lease_dir=/tmp/kdk-direct-ssh-gpu-leases/R6KD-CX8aaS-GPU-16_GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522
host_root=/home/xiy/workspace/flashinfer_exp007_748ad
host_git=/lustre/raplab/client/xiy/workspace/flashinfer/.git
host_cutlass=/home/xiy/workspace/flashinfer_exp005_748ad/3rdparty/cutlass
host_submodule_root=/home/xiy/workspace/flashinfer_exp002_074d93e
host_deps=/home/xiy/workspace/w4a4_deps_460
host_jit=/home/xiy/workspace/exp007_native_n64_jit/${external_arm}/m${m}/${fixture}
container_root=/workspace/source/flashinfer
relative_exp=.claude/w4a4_moe_bench/experiments/exp_007_native_n64_spill_reduction
worker_rel=.claude/w4a4_moe_bench/experiments/exp_005_8warp_spill_reduction/run_exp005_arm.py
worker_fixture=$fixture
container_results=$container_root/$relative_exp/results/canonical
worker_prefix=(python3 "$container_root/$worker_rel")
if [[ "$fixture" == canary_up || "$fixture" == canary_gate ]]; then
  canary_branch=${fixture#canary_}
  worker_rel=$relative_exp/run_branch_half_slice_canary.py
  worker_fixture=canonical
  container_results=$container_root/$relative_exp/results/canary/$fixture
  worker_prefix=(python3 "$container_root/$worker_rel" --canary "$canary_branch")
elif [[ "$fixture" == canary_up_v1 || "$fixture" == canary_gate_v1 ]]; then
  canary_branch=${fixture#canary_}
  canary_branch=${canary_branch%_v1}
  worker_rel=$relative_exp/run_branch_half_slice_canary_v1.py
  worker_fixture=canonical
  container_results=$container_root/$relative_exp/results/canary/$fixture
  worker_prefix=(python3 "$container_root/$worker_rel" --canary "$canary_branch")
elif [[ "$fixture" == canary_up_v2 || "$fixture" == canary_gate_v2 ]]; then
  canary_branch=${fixture#canary_}
  canary_branch=${canary_branch%_v2}
  worker_rel=$relative_exp/run_branch_half_slice_canary_v2.py
  worker_fixture=canonical
  container_results=$container_root/$relative_exp/results/canary/$fixture
  worker_prefix=(python3 "$container_root/$worker_rel" --canary "$canary_branch")
fi
overlay=$container_root/$relative_exp/results/overlays/$overlay_name/moe_dynamic_kernel.py
runtime_dir=$host_root/$relative_exp/results/runtime
label=${command_name}_${external_arm}_m${m}_${fixture}

grep -qx "lease_id=$lease_id" "$lease_dir/metadata"
grep -qx "gpu_uuid=$gpu_uuid" "$lease_dir/metadata"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
  | grep -qx "$gpu_uuid"; then
  echo "leased GPU has a foreign compute process" >&2
  exit 3
fi
if [[ ! -f "$host_root/$relative_exp/results/overlays/$overlay_name/moe_dynamic_kernel.py" ]]; then
  echo "missing immutable overlay: $overlay_name" >&2
  exit 4
fi

mkdir -p "$runtime_dir"
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

docker_args=(
  docker run --rm
  --user "$(id -u):$(id -g)"
  --gpus "device=$gpu_uuid"
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

worker_args=(
  "${worker_prefix[@]}"
  --flashinfer-root "$container_root"
  --results "$container_results"
  --arm "$internal_arm"
  --m "$m"
  --fixture "$worker_fixture"
  --overlay "$overlay"
  --jit-root /workspace/jit
  --expected-gpu-uuid "$gpu_uuid"
  --comparison-anchor candidate_8warp_serial_v0
  --comparison-subject candidate_8warp_n64_temporal_replay_v0
  "$command_name"
  "$@"
)

mkdir -p "$host_jit"
{
  printf '%q ' "${docker_args[@]}" "${worker_args[@]}"
  printf '\n'
} > "$runtime_dir/$label.command.txt"

"${docker_args[@]}" "${worker_args[@]}" \
  > >(tee "$runtime_dir/$label.stdout.log") \
  2> >(tee "$runtime_dir/$label.stderr.log" >&2)

nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
  --format=csv,noheader,nounits > "$runtime_dir/post_${label}_gpu.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits > "$runtime_dir/post_${label}_processes.csv"
