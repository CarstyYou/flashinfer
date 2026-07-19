#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 build | capture <variant> <probe|no_marker> | ncu <variant> | ncu-chain" >&2
  exit 2
fi

action=$1
variant=${2:-}
mode=${3:-}
if [[ "$action" == capture || "$action" == ncu ]]; then
  case "$variant" in
    identity|shared_equal_scale|static_schedule|precomputed_phys_row) ;;
    *) echo "unsupported variant: $variant" >&2; exit 2 ;;
  esac
  if [[ "$action" == capture ]]; then
    case "$mode" in
      probe|no_marker) ;;
      *) echo "unsupported mode: $mode" >&2; exit 2 ;;
    esac
  fi
elif [[ "$action" != build && "$action" != ncu-chain ]]; then
  echo "unsupported action: $action" >&2
  exit 2
fi

repo=${EXP011_REPO:-/home/xiy/workspace/flashinfer_exp010_748ad}
repo_real=$(readlink -f "$repo")
relative_exp=.claude/w4a4_moe_bench/experiments/exp_011_route_q0_pack_vs_chain
exp=$repo/$relative_exp
results=$exp/results
runtime=$results/runtime
deps=${EXP011_DEPS:-/home/xiy/workspace/w4a4_deps_460}
jit_base=${EXP011_JIT_BASE:-/home/xiy/workspace/exp011_route_q0_jit}
container_repo=/workspace/source/flashinfer
container_exp=$container_repo/$relative_exp
container_results=$container_exp/results
image=nvcr.io/nvidia/pytorch:26.05-py3
gpu_uuid=${EXP011_GPU_UUID:-GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
lease_id=${EXP011_LEASE_ID:-exp011-route-q0-pack-20260719}
lease_dir=${EXP011_LEASE_DIR:-/tmp/kdk-direct-ssh-gpu-leases/R6KD-CX8aaS-GPU-16_GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
expected_app_clock_mhz=2377
host_submodule_root=/home/xiy/workspace/flashinfer_exp002_074d93e
host_git=/lustre/raplab/client/xiy/workspace/flashinfer/.git
host_base_real=/lustre/raplab/client/xiy/workspace/flashinfer_exp008_748ad
host_cutlass=$repo/3rdparty/cutlass
host_cccl=/home/xiy/workspace/flashinfer/3rdparty/cccl

mkdir -p "$runtime" "$jit_base" "$results/raw"
grep -qx "lease_id=$lease_id" "$lease_dir/metadata"
grep -qx "gpu_uuid=$gpu_uuid" "$lease_dir/metadata"
observed_uuid=$(nvidia-smi --id="$gpu_uuid" --query-gpu=uuid --format=csv,noheader,nounits | xargs)
[[ "$observed_uuid" == "$gpu_uuid" ]] || { echo "GPU UUID drift" >&2; exit 3; }
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | grep -qx "$gpu_uuid"; then
  echo "foreign compute process detected on leased GPU" >&2
  exit 3
fi
observed_app_clock=$(nvidia-smi --id="$gpu_uuid" --query-gpu=clocks.applications.graphics --format=csv,noheader,nounits | xargs)
[[ "$observed_app_clock" == "$expected_app_clock_mhz" ]] || {
  echo "application clock drift: $observed_app_clock" >&2
  exit 3
}

launcher_start_identity=$(ps -p "$$" -o lstart= | sed 's/^ *//')
sed -i '/^launcher_pid=/d; /^launcher_start_identity=/d' "$lease_dir/metadata"
printf 'launcher_pid=%s\nlauncher_start_identity=%s\n' "$$" "$launcher_start_identity" >> "$lease_dir/metadata"
cleanup_launcher_record() {
  if [[ -f "$lease_dir/metadata" ]] \
    && grep -qx "lease_id=$lease_id" "$lease_dir/metadata" \
    && grep -qx "launcher_pid=$$" "$lease_dir/metadata"; then
    sed -i '/^launcher_pid=/d; /^launcher_start_identity=/d' "$lease_dir/metadata"
    printf 'launcher_pid=pending\nlauncher_start_identity=pending\n' >> "$lease_dir/metadata"
  fi
}
trap cleanup_launcher_record EXIT

if [[ "$action" == build ]]; then
  label=build
  jit=$jit_base/build
elif [[ "$action" == capture ]]; then
  label=${variant}_${mode}
  jit=$jit_base/$label
elif [[ "$action" == ncu ]]; then
  label=ncu_${variant}
  jit=$jit_base/$label
else
  label=ncu_chain_expand
  jit=$jit_base/$label
fi
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
  -v "$jit:/workspace/jit"
  -e HOME=/tmp
  -e PYTHONDONTWRITEBYTECODE=1
  -e PYTHONPATH="$container_repo:/workspace/deps:/workspace/deps/nvidia_cutlass_dsl/dsl_packages"
  -e CUDA_VISIBLE_DEVICES="$gpu_uuid"
  -e KDK_LEASE_ID="$lease_id"
  -e KDK_LEASE_GPU_UUID="$gpu_uuid"
  -e W4A4_IMAGE_DIGEST=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba
  -e W4A4_PYTHON_DEPS_SHA256=32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74
  -e FLASHINFER_NVFP4_4OVER6=0
  -e FLASHINFER_WORKSPACE_BASE=/workspace/jit
  -e CUTE_DSL_CACHE_DIR=/workspace/jit/cache
  -e CUTE_DSL_DUMP_DIR=/workspace/jit/dump
  -e CUTE_DSL_KEEP=ir,ptx,cubin,sass
  -e TORCH_CUDA_ARCH_LIST=12.0a
  -w "$container_repo"
  "$image"
)

if [[ "$action" == build ]]; then
  command=(
    python3 "$container_exp/build_overlays.py"
    --flashinfer-root "$container_repo"
    --output "$container_results/overlays"
  )
elif [[ "$action" == capture ]]; then
  command=(
    python3 "$container_exp/capture_arm.py"
    --flashinfer-root "$container_repo"
    --variant "$variant"
    --mode "$mode"
    --overlay-root "$container_results/overlays"
    --jit-root /workspace/jit
    --output "$container_results/raw/$label"
    --expected-gpu-uuid "$gpu_uuid"
    --warmup 2 --replays 5
  )
elif [[ "$action" == ncu ]]; then
  mkdir -p "$results/raw/ncu/$variant"
  metrics=gpu__time_duration.sum,l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum,l1tex__t_requests_pipe_lsu_mem_global_op_red.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,smsp__sass_thread_inst_executed_op_conversion_pred_on.sum,smsp__sass_thread_inst_executed_op_memory_pred_on.sum,smsp__sass_thread_inst_executed_op_bit_pred_on.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,sm__inst_executed.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,sass__inst_executed_register_spilling_op_read,sass__inst_executed_register_spilling_op_write,launch__registers_per_thread,launch__stack_size
  command=(
    ncu --force-overwrite
    --profile-from-start off
    --target-processes all
    --graph-profiling node
    --replay-mode kernel
    --cache-control all
    --kernel-name regex:MoEDynamicKernel
    --kernel-name-base demangled
    --launch-count 1
    --metrics "$metrics"
    --export "$container_results/raw/ncu/$variant/trace"
    python3 "$container_exp/profile_arm.py"
    --flashinfer-root "$container_repo"
    --variant "$variant"
    --overlay-root "$container_results/overlays"
    --jit-root /workspace/jit
    --output "$container_results/raw/ncu/$variant/target.json"
    --expected-gpu-uuid "$gpu_uuid"
    --warmup 2
  )
else
  mkdir -p "$results/raw/ncu/chain_expand"
  metrics=gpu__time_duration.sum,l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum,l1tex__t_requests_pipe_lsu_mem_global_op_red.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,smsp__sass_thread_inst_executed_op_conversion_pred_on.sum,smsp__sass_thread_inst_executed_op_memory_pred_on.sum,smsp__sass_thread_inst_executed_op_bit_pred_on.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,sm__inst_executed.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,launch__registers_per_thread,launch__stack_size
  command=(
    ncu --force-overwrite
    --profile-from-start off
    --target-processes all
    --graph-profiling node
    --replay-mode kernel
    --cache-control all
    --kernel-name regex:expandInputRowsKernel
    --kernel-name-base demangled
    --launch-count 1
    --metrics "$metrics"
    --export "$container_results/raw/ncu/chain_expand/trace"
    python3 "$container_repo/.claude/w4a4_moe_bench/experiments/exp_010_scatter_vs_chain_finalize/capture_actual_chain.py"
    --flashinfer-root "$container_repo"
    --output "$container_results/raw/ncu/chain_expand/target.json"
    --warmup 5
  )
fi

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
