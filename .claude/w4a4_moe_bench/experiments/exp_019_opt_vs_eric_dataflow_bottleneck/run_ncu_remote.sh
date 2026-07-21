#!/usr/bin/env bash
set -euo pipefail

on_host=0
if [[ ${1:-} == --on-host ]]; then
  on_host=1
  shift
fi
mode=${1:-m8192}
case "$mode" in
  m8192|m1024|ledger-m8192|ledger-m1024|all) ;;
  *) echo "usage: $0 [--on-host] {m8192|m1024|ledger-m8192|ledger-m1024|all}" >&2; exit 2 ;;
esac

host=${EXP019_GPU_HOST:-10.6.142.16}
repo=${EXP019_REPO:-/home/xiy/workspace/flashinfer_exp001_corrected_074d93e}
relative_exp=.claude/w4a4_moe_bench/experiments/exp_019_opt_vs_eric_dataflow_bottleneck

if [[ $on_host == 0 ]]; then
  local_repo=$(git rev-parse --show-toplevel)
  ssh -o BatchMode=yes "$host" "mkdir -p '$repo/$relative_exp'"
  rsync -a --exclude results --exclude __pycache__ \
    "$local_repo/$relative_exp/" "$host:$repo/$relative_exp/"
  rsync -aR \
    "$local_repo/./.claude/w4a4_moe_bench/moe_dynamic_kernel_opt.py" \
    "$local_repo/./.claude/w4a4_moe_bench/moe_dyanmice_kernel_ab_stage4_compact.py" \
    "$local_repo/./.claude/w4a4_moe_bench/experiments/exp_001_backend_case_sweep/fixture.py" \
    "$local_repo/./.claude/w4a4_moe_bench/experiments/exp_001_backend_case_sweep/nvfp4_fixture.py" \
    "$local_repo/./.claude/w4a4_moe_bench/experiments/exp_001_backend_case_sweep/bench_triton_fp8.py" \
    "$local_repo/./.claude/w4a4_moe_bench/experiments/exp_005_8warp_spill_reduction/exp005_common.py" \
    "$local_repo/./.claude/w4a4_moe_bench/experiments/exp_005_8warp_spill_reduction/run_exp005_arm.py" \
    "$local_repo/./.claude/w4a4_moe_bench/experiments/exp_009_intern_stage4_compact_lightcheck/build_adapter.py" \
    "$local_repo/./.claude/w4a4_moe_bench/experiments/exp_018_triton_opt_eric_benchmark/run_arm.py" \
    "$host:$repo/"
  remote=(env EXP019_REPO="$repo" bash "$repo/$relative_exp/run_ncu_remote.sh" --on-host "$mode")
  printf -v remote_command '%q ' "${remote[@]}"
  ssh -T "$host" "$remote_command"
  mkdir -p "$local_repo/$relative_exp/results"
  rsync -a --prune-empty-dirs \
    --include '*/' --include '*.json' --include '*native_raw.csv' --exclude '*' \
    "$host:$repo/$relative_exp/results/" "$local_repo/$relative_exp/results/"
  exit 0
fi

exp=$repo/$relative_exp
results=$exp/results
raw=$results/raw/ncu
runtime=$results/runtime/ncu
veloq=$results/veloq/ncu
container_root=/workspace/source/flashinfer
container_exp=$container_root/$relative_exp
container_results=$container_exp/results
fixture_dir=$container_root/.claude/w4a4_moe_bench/experiments/exp_001_backend_case_sweep/results/fixtures
deps=${EXP019_DEPS:-/home/xiy/workspace/w4a4_deps_460}
ncu_root=${EXP019_NCU_ROOT:-/opt/nvidia/nsight-compute/2025.3.1}
ncu=/opt/ncu/ncu
veloq_bin=${EXP019_VELOQ:-/home/xiy/workspace/flashinfer_exp017_85ae/.claude/w4a4_moe_bench/experiments/exp_017_opt_vs_triton_phase_share/results/runtime/veloq}
image=nvcr.io/nvidia/pytorch:26.05-py3
image_id=sha256:a4e056e1d34a5cc9387512ffa3abeed778e3dc7966633c5154d771705d8835ac
image_digest=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba
deps_sha256=32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74
gpu_uuid=GPU-2fdb0b79-0ba7-f356-b714-6c461b71ce12
expected_clock=2377
lease_root=${EXP019_LEASE_ROOT:-/tmp/kdk-direct-ssh-gpu-leases}
lease_id=${EXP019_RERUN_ID:-exp019-production-ncu-$(date -u +%Y%m%dT%H%M%SZ)-$$}
lease_dir=$lease_root/$(hostname)_${gpu_uuid}
jit_base=${EXP019_JIT_BASE:-/home/xiy/workspace/exp019_ncu_jit/$lease_id}

sections=(
  SpeedOfLight
  ComputeWorkloadAnalysis
  MemoryWorkloadAnalysis
  Occupancy
  SchedulerStats
  WarpStateStats
  LaunchStats
  InstructionStats
  SourceCounters
)
ledger_metrics=gpu__time_duration.sum,dram__bytes.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_sectors.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum,lts__t_sectors_op_membar.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_atom.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_red.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ldgsts_cache_access.sum,l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum,l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum,l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_red.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,sm__inst_executed.sum,sm__inst_executed_pipe_tensor_subpipe_hmma.sum,sm__ops_path_tensor_src_fp4_dst_fp32.sum

gpu_has_process() {
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
    | grep -qx "$gpu_uuid"
}

gpu_snapshot() {
  nvidia-smi --id="$gpu_uuid" \
    --query-gpu=uuid,name,clocks.applications.graphics,memory.used,utilization.gpu \
    --format=csv,noheader,nounits
}

gpu_contract_ok() {
  local row observed_uuid clock
  row=$(gpu_snapshot)
  observed_uuid=$(printf '%s' "$row" | cut -d, -f1 | xargs)
  clock=$(printf '%s' "$row" | cut -d, -f3 | xargs)
  [[ $observed_uuid == "$gpu_uuid" && $clock == "$expected_clock" ]]
  ! gpu_has_process
}

lease_owned=0
acquire_lease() {
  mkdir -p "$lease_root"
  gpu_contract_ok
  mkdir "$lease_dir"
  lease_owned=1
  {
    printf 'lease_id=%s\n' "$lease_id"
    printf 'gpu_uuid=%s\n' "$gpu_uuid"
    printf 'owner=%s@%s\n' "$(id -un)" "$(hostname)"
    printf 'purpose=exp019 exact Opt/Eric production-kernel NCU\n'
  } >"$lease_dir/metadata"
  gpu_contract_ok
}

cleanup() {
  local status=$?
  if [[ $lease_owned == 1 && -f $lease_dir/metadata ]] \
      && grep -qx "lease_id=$lease_id" "$lease_dir/metadata" \
      && ! gpu_has_process; then
    gpu_snapshot >"$runtime/post_gpu.csv" || true
    rm -f "$lease_dir/metadata"
    rmdir "$lease_dir"
  fi
  exit "$status"
}

require_identity() {
  [[ -d $repo && -d $deps && -x $ncu_root/ncu && -x $veloq_bin ]]
  [[ $(docker image inspect "$image" --format '{{.Id}}') == "$image_id" ]]
  docker image inspect "$image" --format '{{json .RepoDigests}}' \
    | grep -q "nvcr.io/nvidia/pytorch@$image_digest"
}

veloq_command() {
  docker run --rm \
    --entrypoint /opt/veloq \
    -v "$repo:$container_root" \
    -v "$veloq_bin:/opt/veloq:ro" \
    -e VELOQ_NCU_REPORT_DIR=/opt/nvidia/nsight-compute/2026.1.1/extras/python \
    -e VELOQ_PYTHON=python3 \
    -w "$container_exp" \
    "$image" "$@"
}

analyze_cell() {
  local arm=$1 m=$2
  local report=$container_results/raw/ncu/$arm/m$m/trace.ncu-rep
  local out=$veloq/$arm/m$m
  veloq_command info "$report" >"$out/info.json"
  veloq_command ncu summary "$report" >"$out/summary.json"
  veloq_command ncu launches "$report" --limit 10 >"$out/launches.json"
  veloq_command ncu inspect "$report" --row-id launch:0 >"$out/inspect.json"
  veloq_command ncu source-metrics "$report" --row-id launch:0 \
    --counter '*pcsamp_warps_issue_stalled_*' --by file \
    >"$out/pc_stalls_file.json"
}

run_cell() {
  local arm=$1 m=$2
  local out=$raw/$arm/m$m
  local host_jit=$jit_base/$arm/m$m
  local container_jit=/workspace/jit
  [[ ! -e $out && ! -e $host_jit ]]
  mkdir -p "$out" "$host_jit" "$veloq/$arm/m$m"

  local selection=()
  local section
  for section in "${sections[@]}"; do
    selection+=(--section "$section")
  done
  local command=(
    "$ncu"
    --target-processes all
    --kernel-name-base demangled
    --force-overwrite
    --profile-from-start off
    --graph-profiling node
    --replay-mode kernel
    --cache-control all
    --clock-control none
    --kernel-name 'regex:.*MoEDynamicKernel.*'
    --launch-count 1
    "${selection[@]}"
    --metrics "$ledger_metrics"
    --export "$container_results/raw/ncu/$arm/m$m/trace"
    python3 "$container_exp/profile_ncu_target.py"
    --arm "$arm"
    --m "$m"
    --flashinfer-root "$container_root"
    --fixture-dir "$fixture_dir"
    --jit-root "$container_jit"
    --output "$container_results/raw/ncu/$arm/m$m/target_manifest.json"
    --expected-gpu-uuid "$gpu_uuid"
    --expected-app-clock-mhz "$expected_clock"
    --rerun-id "$lease_id"
  )
  printf '%q ' "${command[@]}" >"$out/command.txt"
  printf '\n' >>"$out/command.txt"

  docker run --rm \
    --gpus "device=$gpu_uuid" \
    --cap-add SYS_ADMIN \
    -v "$repo:$container_root" \
    -v "$deps:/workspace/deps:ro" \
    -v "$host_jit:$container_jit" \
    -v "$ncu_root:/opt/ncu:ro" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH="$container_root:/workspace/deps:/workspace/deps/nvidia_cutlass_dsl/dsl_packages" \
    -e FLASHINFER_WORKSPACE_BASE="$container_jit" \
    -e CUTE_DSL_DUMP_DIR="$container_jit/dump" \
    -e CUTE_DSL_KEEP=ir,ptx,cubin,sass \
    -e KDK_LEASE_ID="$lease_id" \
    -e W4A4_RERUN_ID="$lease_id" \
    -e W4A4_IMAGE_ID="$image_id" \
    -e W4A4_IMAGE_DIGEST="$image_digest" \
    -e W4A4_PYTHON_DEPS_SHA256="$deps_sha256" \
    -w "$container_root" \
    "$image" "${command[@]}" \
    >"$out/stdout.log" 2>"$out/stderr.log"

  [[ -s $out/trace.ncu-rep && -s $out/target_manifest.json ]]
  "$ncu_root/ncu" --import "$out/trace.ncu-rep" --csv --page raw \
    --print-units base >"$out/native_raw.csv" 2>"$out/native_raw.stderr.log"
  cp "$out/native_raw.csv" "$raw/$arm/m$m/ledger_native_raw.csv"
  sha256sum "$out/trace.ncu-rep" "$out/target_manifest.json" \
    "$out/native_raw.csv" >"$out/sha256sums.txt"
  analyze_cell "$arm" "$m"
}

run_ledger_cell() {
  local arm=$1 m=$2
  local out=$raw/$arm/m$m/ledger
  local host_jit=$jit_base/ledger/$arm/m$m
  local container_jit=/workspace/jit
  [[ ! -e $out && ! -e $host_jit ]]
  mkdir -p "$out" "$host_jit"

  local command=(
    "$ncu"
    --target-processes all
    --kernel-name-base demangled
    --force-overwrite
    --profile-from-start off
    --graph-profiling node
    --replay-mode kernel
    --cache-control all
    --clock-control none
    --kernel-name 'regex:.*MoEDynamicKernel.*'
    --launch-count 1
    --metrics "$ledger_metrics"
    --export "$container_results/raw/ncu/$arm/m$m/ledger/trace"
    python3 "$container_exp/profile_ncu_target.py"
    --arm "$arm" --m "$m"
    --flashinfer-root "$container_root"
    --fixture-dir "$fixture_dir"
    --jit-root "$container_jit"
    --output "$container_results/raw/ncu/$arm/m$m/ledger/target_manifest.json"
    --expected-gpu-uuid "$gpu_uuid"
    --expected-app-clock-mhz "$expected_clock"
    --rerun-id "$lease_id"
  )
  docker run --rm \
    --gpus "device=$gpu_uuid" --cap-add SYS_ADMIN \
    -v "$repo:$container_root" -v "$deps:/workspace/deps:ro" \
    -v "$host_jit:$container_jit" -v "$ncu_root:/opt/ncu:ro" \
    -e CUDA_VISIBLE_DEVICES=0 -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH="$container_root:/workspace/deps:/workspace/deps/nvidia_cutlass_dsl/dsl_packages" \
    -e FLASHINFER_WORKSPACE_BASE="$container_jit" \
    -e CUTE_DSL_CACHE_DIR="$container_jit/cache" \
    -e CUTE_DSL_DUMP_DIR="$container_jit/dump" \
    -e CUTE_DSL_KEEP=ir,ptx,cubin,sass \
    -e KDK_LEASE_ID="$lease_id" -e W4A4_RERUN_ID="$lease_id" \
    -e W4A4_IMAGE_ID="$image_id" -e W4A4_IMAGE_DIGEST="$image_digest" \
    -e W4A4_PYTHON_DEPS_SHA256="$deps_sha256" \
    -w "$container_root" "$image" "${command[@]}" \
    >"$out/stdout.log" 2>"$out/stderr.log"
  [[ -s $out/trace.ncu-rep && -s $out/target_manifest.json ]]
  "$ncu_root/ncu" --import "$out/trace.ncu-rep" --csv --page raw \
    --print-units base >"$raw/$arm/m$m/ledger_native_raw.csv"
}

require_identity
mkdir -p "$runtime" "$raw" "$veloq"
acquire_lease
trap cleanup EXIT INT TERM HUP
gpu_snapshot >"$runtime/pre_gpu.csv"
"$ncu_root/ncu" --version >"$runtime/version.txt"

if [[ $mode == m8192 ]]; then
  cells=(latest_opt_fp4:8192 eric_stage4_fp4:8192)
elif [[ $mode == m1024 ]]; then
  cells=(latest_opt_fp4:1024 eric_stage4_fp4:1024)
elif [[ $mode == all ]]; then
  cells=(latest_opt_fp4:1024 eric_stage4_fp4:1024 eric_stage4_fp4:8192 latest_opt_fp4:8192)
else
  cells=()
fi
for cell in "${cells[@]}"; do
  arm=${cell%%:*}
  m=${cell##*:}
  gpu_contract_ok
  run_cell "$arm" "$m"
  gpu_contract_ok
done

if [[ $mode == ledger-m8192 || $mode == ledger-m1024 ]]; then
  m=${mode#ledger-m}
  for arm in latest_opt_fp4 eric_stage4_fp4; do
    gpu_contract_ok
    run_ledger_cell "$arm" "$m"
    gpu_contract_ok
  done
fi
