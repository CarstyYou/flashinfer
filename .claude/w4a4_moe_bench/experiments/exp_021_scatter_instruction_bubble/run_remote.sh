#!/usr/bin/env bash
set -euo pipefail

on_host=0
if [[ ${1:-} == --on-host ]]; then on_host=1; shift; fi
mode=${1:-all}
case "$mode" in smoke|ncu|analyze|all) ;; *) exit 2 ;; esac

host=${EXP021_HOST:-10.6.142.16}
repo=${EXP021_REPO:-/home/xiy/workspace/flashinfer_exp001_corrected_074d93e}
rel=.claude/w4a4_moe_bench/experiments/exp_021_scatter_instruction_bubble

if [[ $on_host == 0 ]]; then
  local_repo=$(git rev-parse --show-toplevel)
  ssh -T "$host" "mkdir -p '$repo/$rel'"
  rsync -a --exclude results --exclude __pycache__ "$local_repo/$rel/" "$host:$repo/$rel/"
  remote=(env EXP021_REPO="$repo" bash "$repo/$rel/run_remote.sh" --on-host "$mode")
  printf -v command '%q ' "${remote[@]}"
  ssh -T "$host" "$command"
  mkdir -p "$local_repo/$rel/results"
  rsync -a --prune-empty-dirs --include '*/' --include '*.json' --include '*.csv' \
    --include '*.txt' --exclude '*' "$host:$repo/$rel/results/" "$local_repo/$rel/results/"
  exit 0
fi

exp=$repo/$rel
results=$exp/results
raw=$results/raw
capture=$raw/v2
fixture=$repo/.claude/w4a4_moe_bench/experiments/exp_001_backend_case_sweep/results/fixtures/m8192.npz
deps=/home/xiy/workspace/w4a4_deps_460
image=nvcr.io/nvidia/pytorch:26.05-py3
gpu_uuid=${EXP021_GPU_UUID:-GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
ncu_root=/opt/nvidia/nsight-compute/2025.3.1
veloq=/home/xiy/workspace/flashinfer_exp017_85ae/.claude/w4a4_moe_bench/experiments/exp_017_opt_vs_triton_phase_share/results/runtime/veloq
lease_root=/tmp/kdk-direct-ssh-gpu-leases
run_id=${EXP021_RUN_ID:-exp021-$(date -u +%Y%m%dT%H%M%SZ)-$$}
lease_dir=$lease_root/$(hostname)_$gpu_uuid
jit=/home/xiy/workspace/exp021_scatter_jit/$run_id
container_root=/workspace/source/flashinfer
container_exp=$container_root/$rel

gpu_free() {
  ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | grep -qx "$gpu_uuid"
}
cleanup() {
  status=$?
  if [[ -f $lease_dir/metadata ]] && grep -qx "lease_id=$run_id" "$lease_dir/metadata" && gpu_free; then
    rm -f "$lease_dir/metadata"; rmdir "$lease_dir"
  fi
  exit "$status"
}
acquire() {
  [[ -f $fixture && -d $deps && -x $ncu_root/ncu && -x $veloq ]]
  gpu_free
  mkdir -p "$lease_root"
  mkdir "$lease_dir"
  printf 'lease_id=%s\ngpu_uuid=%s\npurpose=exp021 scatter instruction bubble\n' "$run_id" "$gpu_uuid" >"$lease_dir/metadata"
  trap cleanup EXIT INT TERM HUP
}

docker_target() {
  local jit_dir=$1 output=$2
  docker run --rm --gpus "device=$gpu_uuid" \
    -v "$repo:$container_root" -v "$deps:/workspace/deps:ro" -v "$jit_dir:/workspace/jit" \
    -e CUDA_VISIBLE_DEVICES=0 -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH="$container_root:/workspace/deps:/workspace/deps/nvidia_cutlass_dsl/dsl_packages" \
    -e FLASHINFER_WORKSPACE_BASE=/workspace/jit -e CUTE_DSL_DUMP_DIR=/workspace/jit/dump \
    -e CUTE_DSL_KEEP=ir,ptx,cubin,sass -w "$container_exp" "$image" \
    python3 scatter_probe.py --fixture "$container_root/${fixture#$repo/}" --output "$output"
}

analyze() {
  local report=$container_exp/results/raw/v2/trace.ncu-rep
  local out=$results/veloq/v2
  mkdir -p "$out"
  docker run --rm --entrypoint /opt/veloq -v "$repo:$container_root" -v "$veloq:/opt/veloq:ro" \
    -e VELOQ_NCU_REPORT_DIR=/opt/nvidia/nsight-compute/2026.1.1/extras/python \
    -e VELOQ_PYTHON=python3 -w "$container_exp" "$image" info "$report" >"$out/info.json"
  for command in summary launches inspect disasm; do
    case "$command" in
      summary) args=(ncu summary "$report") ;;
      launches) args=(ncu launches "$report" --limit 10) ;;
      inspect) args=(ncu inspect "$report" --row-id launch:0) ;;
      disasm) args=(ncu disasm "$report" --row-id launch:0) ;;
    esac
    docker run --rm --entrypoint /opt/veloq -v "$repo:$container_root" -v "$veloq:/opt/veloq:ro" \
      -e VELOQ_NCU_REPORT_DIR=/opt/nvidia/nsight-compute/2026.1.1/extras/python \
      -e VELOQ_PYTHON=python3 -w "$container_exp" "$image" "${args[@]}" >"$out/$command.json"
  done
  for by in reason sass line; do
    docker run --rm --entrypoint /opt/veloq -v "$repo:$container_root" -v "$veloq:/opt/veloq:ro" \
      -e VELOQ_NCU_REPORT_DIR=/opt/nvidia/nsight-compute/2026.1.1/extras/python \
      -e VELOQ_PYTHON=python3 -w "$container_exp" "$image" \
      ncu warp-stalls "$report" --row-id launch:0 --by "$by" >"$out/warp_stalls_$by.json"
  done
  docker run --rm --entrypoint /opt/veloq -v "$repo:$container_root" -v "$veloq:/opt/veloq:ro" \
    -e VELOQ_NCU_REPORT_DIR=/opt/nvidia/nsight-compute/2026.1.1/extras/python \
    -e VELOQ_PYTHON=python3 -w "$container_exp" "$image" \
    ncu source-metrics "$report" --row-id launch:0 --counter '*pcsamp_warps_issue_stalled_*' --by sass --limit 2000 \
    >"$out/pc_stalls_sass.json"
  docker run --rm --entrypoint /opt/veloq -v "$repo:$container_root" -v "$veloq:/opt/veloq:ro" \
    -e VELOQ_NCU_REPORT_DIR=/opt/nvidia/nsight-compute/2026.1.1/extras/python \
    -e VELOQ_PYTHON=python3 -w "$container_exp" "$image" \
    ncu source-metrics "$report" --row-id launch:0 --counter 'inst_executed,thread_inst_executed' --by sass --limit 2000 \
    >"$out/pc_executed_sass.json"
}

acquire
mkdir -p "$raw" "$capture" "$jit"
if [[ $mode == smoke || $mode == all ]]; then
  docker_target "$jit/smoke" "$container_exp/results/raw/smoke_target.json"
fi
if [[ $mode == ncu || $mode == all ]]; then
  mkdir -p "$jit/ncu"
  sections=(SpeedOfLight MemoryWorkloadAnalysis Occupancy SchedulerStats WarpStateStats LaunchStats InstructionStats SourceCounters)
  section_args=()
  for section in "${sections[@]}"; do section_args+=(--section "$section"); done
  command=(/opt/ncu/ncu --target-processes all --kernel-name-base demangled --profile-from-start off
    --replay-mode kernel --cache-control all --clock-control none
    --kernel-name 'regex:.*scatter_probe_kernel.*' --launch-count 1 "${section_args[@]}"
    --export "$container_exp/results/raw/v2/trace" python3 scatter_probe.py
    --fixture "$container_root/${fixture#$repo/}" --output "$container_exp/results/raw/v2/ncu_target.json")
  printf '%q ' "${command[@]}" >"$capture/command.txt"; printf '\n' >>"$capture/command.txt"
  docker run --rm --gpus "device=$gpu_uuid" --cap-add SYS_ADMIN \
    -v "$repo:$container_root" -v "$deps:/workspace/deps:ro" -v "$jit/ncu:/workspace/jit" \
    -v "$ncu_root:/opt/ncu:ro" -e CUDA_VISIBLE_DEVICES=0 -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH="$container_root:/workspace/deps:/workspace/deps/nvidia_cutlass_dsl/dsl_packages" \
    -e FLASHINFER_WORKSPACE_BASE=/workspace/jit -e CUTE_DSL_DUMP_DIR=/workspace/jit/dump \
    -e CUTE_DSL_KEEP=ir,ptx,cubin,sass -w "$container_exp" "$image" "${command[@]}" \
    >"$capture/ncu.stdout.log" 2>"$capture/ncu.stderr.log"
  "$ncu_root/ncu" --import "$capture/trace.ncu-rep" --csv --page raw --print-units base \
    >"$capture/native_raw.csv" 2>"$capture/native_raw.stderr.log"
  sha256sum "$capture/trace.ncu-rep" "$capture/ncu_target.json" >"$capture/sha256sums.txt"
fi
if [[ $mode == analyze || $mode == all ]]; then analyze; fi
