#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 capture-control|capture-probe|standalone|actual-chain" >&2
  exit 2
fi

action=$1
case "$action" in
  capture-control|capture-probe|standalone|actual-chain) ;;
  *) echo "unsupported action: $action" >&2; exit 2 ;;
esac

repo=${EXP010_REPO:-/home/xiy/workspace/flashinfer_exp010_748ad}
repo_real=$(readlink -f "$repo")
relative_exp=.claude/w4a4_moe_bench/experiments/exp_010_scatter_vs_chain_finalize
exp=$repo/$relative_exp
results=$exp/results
runtime=$results/runtime
deps=${EXP010_DEPS:-/home/xiy/workspace/w4a4_deps_460}
jit_base=${EXP010_JIT_BASE:-/home/xiy/workspace/exp010_scatter_finalize_jit}
container_repo=/workspace/source/flashinfer
container_exp=$container_repo/$relative_exp
container_results=$container_exp/results
container_jit=/workspace/jit
image=nvcr.io/nvidia/pytorch:26.05-py3
gpu_uuid=${EXP010_GPU_UUID:-GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
lease_id=${EXP010_LEASE_ID:-exp010-scatter-finalize-20260719}
lease_dir=${EXP010_LEASE_DIR:-/tmp/kdk-direct-ssh-gpu-leases/R6KD-CX8aaS-GPU-16_GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
expected_app_clock_mhz=2377
host_submodule_root=/home/xiy/workspace/flashinfer_exp002_074d93e
host_git=/lustre/raplab/client/xiy/workspace/flashinfer/.git
host_base_real=/lustre/raplab/client/xiy/workspace/flashinfer_exp008_748ad
host_cutlass=$repo/3rdparty/cutlass
host_cccl=/home/xiy/workspace/flashinfer/3rdparty/cccl

mkdir -p "$runtime" "$jit_base" "$results/raw/nsys"
grep -qx "lease_id=$lease_id" "$lease_dir/metadata"
grep -qx "gpu_uuid=$gpu_uuid" "$lease_dir/metadata"
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
  echo "application clock drift: $observed_app_clock" >&2
  exit 3
}

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

jit=$jit_base/$action
mkdir -p "$jit"
uid=$(id -u)
gid=$(id -g)
docker_args=(
  docker run --rm
  --gpus "device=$gpu_uuid"
  --user "$uid:$gid"
  -v "$repo:$container_repo"
  -v "$repo_real:$repo_real:ro"
  -v "$host_git:$host_git:ro"
  -v "$host_base_real:$host_base_real:ro"
  -v "$host_cutlass:$container_repo/3rdparty/cutlass:ro"
  -v "$host_cccl:$container_repo/3rdparty/cccl:ro"
  -v "$host_submodule_root:$host_submodule_root:ro"
  -v "$deps:/workspace/deps:ro"
  -v "$jit:$container_jit"
  -e HOME=/tmp
  -e PYTHONDONTWRITEBYTECODE=1
  -e PYTHONPATH="$container_repo:/workspace/deps:/workspace/deps/nvidia_cutlass_dsl/dsl_packages"
  -e CUDA_VISIBLE_DEVICES="$gpu_uuid"
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

case "$action" in
  capture-control)
    arm=measurement_no_marker
    command=(
      python3 "$container_exp/capture_production_anchor.py"
      --flashinfer-root "$container_repo"
      --arm "$arm"
      --kernel-overlay "$container_repo/.claude/w4a4_moe_bench/experiments/exp_006_fc2_completion_anchored_breakdown/results/overlays/$arm/moe_dynamic_kernel.py"
      --dispatch-overlay "$container_repo/.claude/w4a4_moe_bench/experiments/exp_006_fc2_completion_anchored_breakdown/results/overlays/$arm/moe_dispatch.py"
      --jit-root "$container_jit"
      --output "$container_results/raw/production_$arm"
      --expected-gpu-uuid "$gpu_uuid"
      --warmup 2 --replays 5
    )
    ;;
  capture-probe)
    arm=completion_anchored_probe
    command=(
      python3 "$container_exp/capture_production_anchor.py"
      --flashinfer-root "$container_repo"
      --arm "$arm"
      --kernel-overlay "$container_repo/.claude/w4a4_moe_bench/experiments/exp_006_fc2_completion_anchored_breakdown/results/overlays/$arm/moe_dynamic_kernel.py"
      --dispatch-overlay "$container_repo/.claude/w4a4_moe_bench/experiments/exp_006_fc2_completion_anchored_breakdown/results/overlays/$arm/moe_dispatch.py"
      --jit-root "$container_jit"
      --output "$container_results/raw/production_$arm"
      --expected-gpu-uuid "$gpu_uuid"
      --warmup 2 --replays 5
    )
    ;;
  standalone)
    command=(
      python3 "$container_exp/run_exp010.py"
      --production-timing "$container_results/raw/production_completion_anchored_probe/timing_0.pt"
      --results "$container_results"
      --build-dir "$container_jit/torch_ext"
      --warmup 2 --replays 5
    )
    ;;
  actual-chain)
    command=(
      bash -lc
      "nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --cuda-graph-trace=node:host-only --capture-range=cudaProfilerApi --capture-range-end=stop -o '$container_results/raw/nsys/actual_chain' python3 '$container_exp/capture_actual_chain.py' --flashinfer-root '$container_repo' --output '$container_results/raw/nsys/actual_chain.json' --warmup 5"
    )
    ;;
esac

label=$action
{
  printf '%q ' "${docker_args[@]}" "${command[@]}"
  printf '\n'
} > "$runtime/$label.command.txt"

"${docker_args[@]}" "${command[@]}" \
  > >(tee "$runtime/$label.stdout.log") \
  2> >(tee "$runtime/$label.stderr.log" >&2)

nvidia-smi --id="$gpu_uuid" \
  --query-gpu=uuid,clocks.current.graphics,clocks.applications.graphics,memory.used,utilization.gpu \
  --format=csv,noheader,nounits > "$runtime/$label.post_gpu.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits > "$runtime/$label.post_processes.csv"
