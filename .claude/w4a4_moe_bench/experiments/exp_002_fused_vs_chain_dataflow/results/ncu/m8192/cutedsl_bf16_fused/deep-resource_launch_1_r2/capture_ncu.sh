#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 M ARM [operator-ledger|operator-ledger-v2|node-ledger|deep|deep-resource] [launch-skip]" >&2
  exit 2
fi

m=$1
arm=$2
mode=${3:-operator-ledger}
launch_skip=${4:-}
capture_tag=${W4A4_CAPTURE_TAG:-}

case "$m" in
  256|8192) ;;
  *) echo "unsupported M: $m" >&2; exit 2 ;;
esac

case "$arm" in
  cutedsl_bf16_fused|cutlass_bf16_chain) ;;
  *) echo "unsupported arm: $arm" >&2; exit 2 ;;
esac

case "$mode" in
  operator-ledger|operator-ledger-v2|node-ledger|deep|deep-resource) ;;
  *) echo "unsupported capture mode: $mode" >&2; exit 2 ;;
esac

host_root=/home/xiy/workspace/flashinfer_exp002_074d93e
host_git_alternate=/lustre/raplab/client/xiy/workspace/flashinfer/.git
host_deps=${W4A4_DEPS_ROOT:-/home/xiy/workspace/w4a4_deps_460}
host_jit=${W4A4_JIT_ROOT:-/home/xiy/workspace/exp002_jit_dsl460}
gpu_uuid=${W4A4_GPU_UUID:?set W4A4_GPU_UUID to the leased full GPU UUID}
lease_id=${KDK_LEASE_ID:?set KDK_LEASE_ID to the active advisory lease}
rerun_id=${W4A4_RERUN_ID:?set W4A4_RERUN_ID to the benchmark rerun ID}
container_root=/workspace/source/flashinfer
relative_exp=.claude/w4a4_moe_bench/experiments/exp_002_fused_vs_chain_dataflow
host_exp="$host_root/$relative_exp"

suffix=$mode
if [[ -n "$launch_skip" ]]; then
  suffix="${mode}_launch_${launch_skip}"
fi
if [[ -n "$capture_tag" ]]; then
  if [[ ! "$capture_tag" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "invalid W4A4_CAPTURE_TAG: $capture_tag" >&2
    exit 2
  fi
  suffix="${suffix}_${capture_tag}"
fi
host_out="$host_exp/results/ncu/m${m}/${arm}/${suffix}"
container_out="$container_root/$relative_exp/results/ncu/m${m}/${arm}/${suffix}"

if [[ -e "$host_out" ]]; then
  echo "refusing to overwrite capture directory: $host_out" >&2
  exit 3
fi
mkdir -p "$host_out"
cp "$host_exp/capture_ncu.sh" "$host_out/capture_ncu.sh"
(cd "$host_out" && sha256sum capture_ncu.sh >capture_ncu.sh.sha256)

app_results="$container_root/$relative_exp/results"
profile_manifest_root="$host_exp/results"
app_entry="$container_root/$relative_exp/run_exp002.py"
if [[ "$mode" == deep-resource ]]; then
  # Keep the canonical results immutable.  The profiler app writes its updated
  # profile manifest into this capture-local prerequisite snapshot.
  followup_results="$host_out/profile_results"
  mkdir -p \
    "$followup_results/manifests" \
    "$followup_results/environment-locks" \
    "$followup_results/artifact-locks" \
    "$followup_results/protocol-locks"
  cp "$host_exp/results/correctness.json" "$followup_results/"
  cp "$host_exp/results/evidence.identity.json" "$followup_results/"
  cp "$host_exp/results/benchmark_summary.csv" "$followup_results/"
  cp "$host_exp/results/manifests/jit_artifacts.json" \
    "$followup_results/manifests/"
  cp "$host_exp/results/environment-locks/"*.json \
    "$followup_results/environment-locks/"
  cp "$host_exp/results/artifact-locks/"*.json \
    "$followup_results/artifact-locks/"
  cp "$host_exp/results/protocol-locks/"*.json \
    "$followup_results/protocol-locks/"
  app_results="$container_out/profile_results"
  profile_manifest_root="$followup_results"
  app_entry="$container_root/$relative_exp/run_exp002_profile_followup.py"
fi

ledger_metrics=gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,sm__inst_executed.sum,sm__inst_executed_pipe_tensor_subpipe_hmma.sum,sm__ops_path_tensor_src_fp4_dst_fp32.sum

# Whole-graph traffic ledger with complete DRAM/L2 authority counters and
# explicit LSU-global, TMA, and local-address-space families. These hierarchy
# families are intentionally kept separate; they are not one additive total.
ledger_v2_metrics=gpu__time_duration.sum,dram__bytes.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_sectors.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum,lts__t_sectors_op_membar.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_atom.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_red.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ldgsts_cache_access.sum,l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum,l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum,l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_red.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,sm__inst_executed.sum,sm__inst_executed_pipe_tensor_subpipe_hmma.sum,sm__ops_path_tensor_src_fp4_dst_fp32.sum

deep_metrics=$ledger_metrics,sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.avg.per_cycle_active,smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_lsu.sum,smsp__inst_executed_pipe_uniform.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct

metrics=$ledger_metrics
if [[ "$mode" == deep ]]; then
  metrics=$deep_metrics
elif [[ "$mode" == operator-ledger-v2 ]]; then
  metrics=$ledger_v2_metrics
fi

ncu_args=(
  ncu
  --target-processes all
  --kernel-name-base demangled
  --force-overwrite
  --export "$container_out/trace"
  --log-file "$container_out/ncu.log"
)
if [[ "$mode" == deep-resource ]]; then
  # This bundle intentionally uses NCU sections rather than a custom metric
  # whitelist. SourceCounters/InstructionStats are required for NCU's SASS
  # classification of register-spill instructions; the remaining sections
  # cover compute/memory utilization, occupancy, scheduler throttles/stalls,
  # and launch resources on the exact target architecture.
  ncu_args+=(
    --section SpeedOfLight
    --section ComputeWorkloadAnalysis
    --section MemoryWorkloadAnalysis
    --section Occupancy
    --section SchedulerStats
    --section WarpStateStats
    --section LaunchStats
    --section InstructionStats
    --section SourceCounters
  )
else
  ncu_args+=(--metrics "$metrics")
fi
if [[ "$mode" == operator-ledger || "$mode" == operator-ledger-v2 ]]; then
  if [[ -n "$launch_skip" ]]; then
    echo "operator-ledger does not accept launch-skip" >&2
    exit 2
  fi
  ncu_args+=(
    --replay-mode app-range
    --cache-control all
  )
else
  ncu_args+=(
    --profile-from-start off
    --graph-profiling node
    --replay-mode kernel
    --cache-control all
  )
fi
if [[ -n "$launch_skip" ]]; then
  ncu_args+=(--launch-skip "$launch_skip" --launch-count 1)
fi

app_args=(
  python3 "$app_entry"
  --flashinfer-root "$container_root"
  --results "$app_results"
  --expected-gpu-uuid "$gpu_uuid"
  single-replay
  --m "$m"
  --arm "$arm"
)

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
  -e W4A4_FOLLOWUP_VALIDATION_MANIFEST="$container_out/followup_validation.json"
  -e FLASHINFER_WORKSPACE_BASE=/workspace/jit
  -e FLASHINFER_CUTEDSL_IKET_OVERLAY=0
  -e FLASHINFER_NVFP4_4OVER6=0
  -e W4A4_IMAGE_DIGEST=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba
  -e W4A4_PYTHON_DEPS_SHA256=32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74
  -w "$container_root"
  nvcr.io/nvidia/pytorch:26.05-py3
)

{
  printf '%q ' "${docker_args[@]}" "${ncu_args[@]}" "${app_args[@]}"
  printf '\n'
} | sed 's/ $//' >"$host_out/command.txt"

"${docker_args[@]}" "${ncu_args[@]}" "${app_args[@]}" \
  >"$host_out/stdout.log" 2>"$host_out/stderr.log"

(cd "$host_out" && sha256sum trace.ncu-rep >trace.ncu-rep.sha256)
docker image inspect nvcr.io/nvidia/pytorch:26.05-py3 --format '{{.Id}}' \
  >"$host_out/docker-image-id.txt"

profile_manifest="$profile_manifest_root/manifests/profile_m${m}_${arm}.json"
cp "$profile_manifest" "$host_out/profile_manifest.json"
(cd "$host_out" && sha256sum profile_manifest.json >profile_manifest.json.sha256)

if [[ "$mode" == operator-ledger-v2 ]]; then
  native_export=(
    ncu
    --import "$container_out/trace.ncu-rep"
    --csv
    --page raw
    --print-units base
  )
  {
    printf '%q ' "${docker_args[@]}" "${native_export[@]}"
    printf '\n'
  } | sed 's/ $//' >"$host_out/native_raw.command.txt"
  "${docker_args[@]}" "${native_export[@]}" \
    >"$host_out/native_raw.csv" 2>"$host_out/native_raw.stderr.log"
  sha256sum "$host_out/native_raw.csv" >"$host_out/native_raw.csv.sha256"
fi
