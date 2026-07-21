#!/usr/bin/env bash
set -euo pipefail

# Background-only exp_017 NCU supplement. It profiles the current exact
# Latest-opt fused launch and the four material SGLang Triton FP8 launches.

on_host=0
if [[ ${1:-} == --on-host ]]; then
  on_host=1
  shift
fi
mode=${1:-all}
case "$mode" in
  all|opt-deep|triton-deep|analyze) ;;
  *) echo "usage: $0 [--on-host] {all|opt-deep|triton-deep|analyze}" >&2; exit 2 ;;
esac

host=${EXP017_HOST:-xiy@10.6.142.16}
repo=${EXP017_REPO:-/home/xiy/workspace/flashinfer_exp017_85ae}
relative_exp=.claude/w4a4_moe_bench/experiments/exp_017_opt_vs_triton_phase_share
# NCU is launch-local diagnostic evidence, not benchmark authority.  Both NCU
# arms use the same idle sibling 5KP because the benchmark-authority GPU is
# occupied; the report/manifest keeps the two evidence identities separate.
gpu_uuid=${EXP017_GPU_UUID:-GPU-c2ac6efb-f30a-c323-6d38-83908adfb14f}
lease_root=${EXP017_LEASE_ROOT:-/tmp/kdk-direct-ssh-gpu-leases}
gpu_host_id=${EXP017_GPU_HOST_ID:-R6KD-CX8aaS-GPU-16}
lease_dir=$lease_root/${gpu_host_id}_${gpu_uuid}
lease_id=${EXP017_LEASE_ID:-exp017-ncu-$(date -u +%Y%m%dT%H%M%SZ)-$$}
expected_clock=2377
memory_threshold_mib=256
wait_seconds=${EXP017_WAIT_SECONDS:-14400}
client_host=${EXP017_CLIENT_HOST:-$(hostname)}

if [[ $on_host == 0 ]]; then
  remote=(
    env
    EXP017_REPO="$repo"
    EXP017_GPU_UUID="$gpu_uuid"
    EXP017_LEASE_ROOT="$lease_root"
    EXP017_GPU_HOST_ID="$gpu_host_id"
    EXP017_LEASE_ID="$lease_id"
    EXP017_WAIT_SECONDS="$wait_seconds"
    EXP017_CLIENT_HOST="$client_host"
    bash "$repo/$relative_exp/run_ncu_remote.sh" --on-host "$mode"
  )
  printf -v command '%q ' "${remote[@]}"
  exec ssh -T "$host" "$command"
fi

exp=$repo/$relative_exp
results=$exp/results
runtime=$results/runtime
raw=$results/raw/ncu
veloq_out=$results/veloq/ncu
container_root=/workspace/source/flashinfer
container_exp=$container_root/$relative_exp
container_results=$container_exp/results
deps=/home/xiy/workspace/w4a4_deps_460
ncu_root=/opt/nvidia/nsight-compute/2025.3.1
ncu=/opt/ncu/ncu
git_common=/lustre/raplab/client/xiy/workspace/flashinfer/.git
cutlass_worktree=/lustre/raplab/client/xiy/workspace/flashinfer/3rdparty/cutlass
opt_image=nvcr.io/nvidia/pytorch:26.05-py3
opt_image_id=sha256:a4e056e1d34a5cc9387512ffa3abeed778e3dc7966633c5154d771705d8835ac
opt_image_digest=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba
sglang_image=lmsysorg/sglang:latest
sglang_image_id=sha256:663867442f321ded36228bafd889fd1db05cbef7a7c8ea6e072df33234dabbfd
sglang_image_digest=sha256:00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2
python_deps_sha256=32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74
fixture_dir=$container_root/.claude/w4a4_moe_bench/experiments/exp_001_backend_case_sweep/results/fixtures
opt_source=$repo/.claude/w4a4_moe_bench/moe_dynamic_kernel_opt.py
expected_opt_sha=ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19

deep_sections=(
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

require_identity() {
  [[ $(hostname) == "$gpu_host_id" ]]
  [[ -d $repo && -d $deps && -x $ncu_root/ncu && -d $git_common && -d $cutlass_worktree ]]
  [[ $(sha256sum "$opt_source" | awk '{print $1}') == "$expected_opt_sha" ]]
  [[ $(docker image inspect "$opt_image" --format '{{.Id}}') == "$opt_image_id" ]]
  [[ $(docker image inspect "$opt_image" --format '{{json .RepoDigests}}') == *"nvcr.io/nvidia/pytorch@$opt_image_digest"* ]]
  [[ $(docker image inspect "$sglang_image" --format '{{.Id}}') == "$sglang_image_id" ]]
  [[ $(docker image inspect "$sglang_image" --format '{{json .RepoDigests}}') == *"lmsysorg/sglang@$sglang_image_digest"* ]]
}

gpu_snapshot() {
  nvidia-smi --id="$gpu_uuid" \
    --query-gpu=index,uuid,name,clocks.applications.graphics,memory.used,utilization.gpu \
    --format=csv,noheader,nounits
}

gpu_has_process() {
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
    | grep -qx "$gpu_uuid"
}

gpu_is_idle() {
  local line observed_uuid clock memory
  line=$(gpu_snapshot)
  observed_uuid=$(printf '%s' "$line" | cut -d, -f2 | xargs)
  clock=$(printf '%s' "$line" | cut -d, -f4 | xargs)
  memory=$(printf '%s' "$line" | cut -d, -f5 | xargs)
  [[ $observed_uuid == "$gpu_uuid" && $clock == "$expected_clock" ]] || return 1
  [[ $memory -le $memory_threshold_mib ]] || return 1
  ! gpu_has_process
}

lease_owned=0
acquire_lease() {
  local deadline=$((SECONDS + wait_seconds))
  while ! gpu_is_idle || [[ -e $lease_dir ]]; do
    if (( SECONDS >= deadline )); then
      echo "timed out waiting for exact exp017 GPU: $(gpu_snapshot)" >&2
      return 1
    fi
    sleep 15
  done
  umask 077
  mkdir "$lease_dir"
  lease_owned=1
  local start_identity
  start_identity=$(awk '{print $22}' /proc/$$/stat)
  {
    printf 'lease_id=%s\n' "$lease_id"
    printf 'owner=%s\n' "$(id -un)"
    printf 'client_host=%s\n' "$client_host"
    printf 'gpu_host=%s\n' "$gpu_host_id"
    printf 'gpu_uuid=%s\n' "$gpu_uuid"
    printf 'gpu_index_at_acquire=%s\n' "$(gpu_snapshot | cut -d, -f1 | xargs)"
    printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'purpose=exp017 current Latest-opt vs Triton FP8 NCU supplement\n'
    printf 'launcher_pid=%s\n' "$$"
    printf 'launcher_start_identity=%s\n' "$start_identity"
  } >"$lease_dir/metadata"
  chmod 600 "$lease_dir/metadata"
  if ! gpu_is_idle; then
    rm -f "$lease_dir/metadata"
    rmdir "$lease_dir"
    lease_owned=0
    echo "GPU changed after lease acquisition" >&2
    return 1
  fi
}

release_lease() {
  local status=$?
  if [[ $lease_owned == 1 && -f $lease_dir/metadata ]] \
      && grep -qx "lease_id=$lease_id" "$lease_dir/metadata"; then
    # All docker launches are foreground. Never signal any unowned process.
    if ! gpu_has_process; then
      gpu_snapshot >"$runtime/ncu.post_gpu.csv" || true
      rm -f "$lease_dir/metadata"
      rmdir "$lease_dir"
      lease_owned=0
    else
      echo "GPU still has a process; retaining owned lease for safe audit" >&2
    fi
  fi
  exit "$status"
}

common_ncu=(
  "$ncu"
  --target-processes all
  --kernel-name-base demangled
  --force-overwrite
  --profile-from-start off
  --graph-profiling node
  --replay-mode kernel
  --cache-control all
  --clock-control none
)

run_opt_deep() {
  local out=$raw/opt_fused_deep
  local jit=/home/xiy/workspace/exp017_ncu_jit/$lease_id/opt
  [[ ! -e $out && ! -e $jit ]]
  mkdir -p "$out" "$jit" "$runtime"
  local command=(
    "${common_ncu[@]}"
    --kernel-name 'regex:.*MoEDynamicKernel.*'
    --launch-count 1
    "${deep_sections[@]}"
    --export "$container_results/raw/ncu/opt_fused_deep/trace"
    python3 "$container_exp/profile_opt_ncu_target.py"
    --flashinfer-root "$container_root"
    --fixture-dir "$fixture_dir"
    --jit-root /workspace/jit
    --output "$container_results/raw/ncu/opt_fused_deep/target_manifest.json"
    --expected-gpu-uuid "$gpu_uuid"
    --expected-app-clock-mhz "$expected_clock"
  )
  {
    printf 'docker run --rm --gpus device=%q --cap-add SYS_ADMIN ' "$gpu_uuid"
    printf '%q ' "${command[@]}"
    printf '\n'
  } >"$out/command.txt"
  docker run --rm \
    --gpus "device=$gpu_uuid" \
    --cap-add SYS_ADMIN \
    -v "$repo:$container_root" \
    -v "$git_common:$git_common:ro" \
    -v "$cutlass_worktree:$cutlass_worktree:ro" \
    -v "$deps:/workspace/deps:ro" \
    -v "$jit:/workspace/jit" \
    -v "$ncu_root:/opt/ncu:ro" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH="$container_root:/workspace/deps:/workspace/deps/nvidia_cutlass_dsl/dsl_packages" \
    -e FLASHINFER_WORKSPACE_BASE=/workspace/jit \
    -e CUTE_DSL_CACHE_DIR=/workspace/jit/cache \
    -e CUTE_DSL_DUMP_DIR=/workspace/jit/dump \
    -e CUTE_DSL_KEEP=ir,ptx,cubin,sass \
    -e TORCH_CUDA_ARCH_LIST=12.0a \
    -e FLASHINFER_NVFP4_4OVER6=0 \
    -e KDK_LEASE_ID="$lease_id" \
    -e W4A4_IMAGE_ID="$opt_image_id" \
    -e W4A4_IMAGE_DIGEST="$opt_image_digest" \
    -e W4A4_PYTHON_DEPS_SHA256="$python_deps_sha256" \
    -w "$container_exp" \
    "$opt_image" "${command[@]}" \
    >"$out/stdout.log" 2>"$out/stderr.log"
  sha256sum "$out/trace.ncu-rep" "$out/target_manifest.json" >"$out/sha256sums.txt"
}

run_triton_deep() {
  local out=$raw/triton_material_deep
  local jit=/home/xiy/workspace/exp017_ncu_jit/$lease_id/triton
  local target_results=$container_results/raw/ncu/triton_material_deep/target_results
  local topology_preflight=$target_results/triton_topology_preflight.json
  [[ ! -e $out && ! -e $jit ]]
  mkdir -p "$out/target_results" "$jit" "$runtime"
  local docker_args=(
    docker run --rm
    --gpus "device=$gpu_uuid"
    --cap-add SYS_ADMIN
    -v "$repo:$container_root"
    -v "$jit:/workspace/jit"
    -v "$ncu_root:/opt/ncu:ro"
    -e CUDA_VISIBLE_DEVICES=0
    -e PYTHONDONTWRITEBYTECODE=1
    -e KDK_LEASE_ID="$lease_id"
    -e W4A4_RERUN_ID="$lease_id-triton"
    -e W4A4_IMAGE_DIGEST="$sglang_image_digest"
    -e W4A4_IMAGE_ID="$sglang_image_id"
    -e W4A4_SGLANG_COMMIT=0b3bb0cbe31873994c9f989fddfe2f87ca839fdd
    -e W4A4_SGLANG_JIT_DIR=/workspace/jit
    -e TRITON_CACHE_DIR=/workspace/jit
    -w "$container_exp"
    "$sglang_image"
  )
  local preflight=(
    python3 "$container_exp/capture_triton_nsys.py"
    --fixture-dir "$fixture_dir"
    --results "$target_results"
    --topology-preflight "$topology_preflight"
    --preflight-only
    --expected-gpu-uuid "$gpu_uuid"
    --expected-app-clock-mhz "$expected_clock"
  )
  local command=(
    "${common_ncu[@]}"
    --kernel-name 'regex:.*(fused_moe_kernel|act_and_mul_kernel|moe_sum_reduce).*'
    --launch-count 4
    "${deep_sections[@]}"
    --export "$container_results/raw/ncu/triton_material_deep/trace"
    python3 "$container_exp/capture_triton_nsys.py"
    --fixture-dir "$fixture_dir"
    --results "$target_results"
    --topology-preflight "$topology_preflight"
    --expected-gpu-uuid "$gpu_uuid"
    --expected-app-clock-mhz "$expected_clock"
    --warmup 5
    --replays 5
  )
  {
    printf '%q ' "${docker_args[@]}" "${preflight[@]}"
    printf '\n'
    printf '%q ' "${docker_args[@]}" "${command[@]}"
    printf '\n'
  } >"$out/command.txt"
  "${docker_args[@]}" "${preflight[@]}" \
    >"$out/preflight.stdout.log" 2>"$out/preflight.stderr.log"
  "${docker_args[@]}" "${command[@]}" \
    >"$out/stdout.log" 2>"$out/stderr.log"
  sha256sum "$out/trace.ncu-rep" \
    "$out/target_results/triton_topology_preflight.json" \
    "$out/target_results/triton_capture_manifest.json" >"$out/sha256sums.txt"
}

veloq_command() {
  local capture_image=$1
  shift
  # VeloQ 0.2.2 needs the 2026.1 IAction API (timed_warp_samples is absent
  # from the 2025.3 reader).  The 2026.1 reader is backward-compatible with
  # these 2025.3 reports; capture_image remains part of the call-site audit.
  : "$capture_image"
  docker run --rm \
    --entrypoint "$container_results/runtime/veloq" \
    -v "$repo:$container_root" \
    -e VELOQ_NCU_REPORT_DIR=/opt/nvidia/nsight-compute/2026.1.1/extras/python \
    -e VELOQ_PYTHON=python3 \
    -w "$container_exp" \
    "$opt_image" "$@"
}

analyze_reports() {
  [[ -x $runtime/veloq ]]
  mkdir -p "$veloq_out/opt_fused" "$veloq_out/triton_material"
  "$runtime/veloq" --version >"$veloq_out/version.txt"
  local opt_report=results/raw/ncu/opt_fused_deep/trace.ncu-rep
  local triton_report=results/raw/ncu/triton_material_deep/trace.ncu-rep
  veloq_command "$opt_image" info "$opt_report" >"$veloq_out/opt_fused/info.json"
  veloq_command "$opt_image" ncu summary "$opt_report" >"$veloq_out/opt_fused/summary.json"
  veloq_command "$opt_image" ncu launches "$opt_report" --limit 10 >"$veloq_out/opt_fused/launches.json"
  veloq_command "$opt_image" ncu inspect "$opt_report" --row-id launch:0 >"$veloq_out/opt_fused/inspect.json"
  veloq_command "$opt_image" ncu disasm "$opt_report" --row-id launch:0 >"$veloq_out/opt_fused/disasm.json"
  veloq_command "$opt_image" ncu source-metrics "$opt_report" --row-id launch:0 --counter '*pcsamp_warps_issue_stalled_*' --by file >"$veloq_out/opt_fused/pc_stalls_file.json"
  veloq_command "$opt_image" ncu source-metrics "$opt_report" --row-id launch:0 --counter '*pcsamp_warps_issue_stalled_*' --by sass --limit 5000 >"$veloq_out/opt_fused/pc_stalls_sass_full.json"

  veloq_command "$sglang_image" info "$triton_report" >"$veloq_out/triton_material/info.json"
  veloq_command "$sglang_image" ncu summary "$triton_report" >"$veloq_out/triton_material/summary.json"
  veloq_command "$sglang_image" ncu launches "$triton_report" --limit 10 >"$veloq_out/triton_material/launches.json"
  local row
  for row in 0 1 2 3; do
    veloq_command "$sglang_image" ncu inspect "$triton_report" --row-id "launch:$row" >"$veloq_out/triton_material/inspect_${row}.json"
    veloq_command "$sglang_image" ncu source-metrics "$triton_report" --row-id "launch:$row" --counter '*pcsamp_warps_issue_stalled_*' --by file >"$veloq_out/triton_material/pc_stalls_file_${row}.json"
  done
}

require_identity
mkdir -p "$runtime"
if [[ $mode == analyze ]]; then
  analyze_reports
  exit 0
fi

acquire_lease
trap release_lease EXIT INT TERM HUP
gpu_snapshot >"$runtime/ncu.pre_gpu.csv"
"$ncu_root/ncu" --version >"$runtime/ncu.capture_version.txt"
if [[ $mode == all || $mode == opt-deep ]]; then
  run_opt_deep
fi
if [[ $mode == all || $mode == triton-deep ]]; then
  run_triton_deep
fi
if [[ $mode == all ]]; then
  analyze_reports
fi
