#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 prepare production|intern|exp008 M" >&2
  echo "       $0 diagnose intern M" >&2
  echo "       $0 measure production_intern|production_exp008 production|intern|exp008 M group position" >&2
  exit 2
}

[[ $# -ge 3 ]] || usage
action=$1
shift

case "$action" in
  prepare)
    [[ $# -eq 2 ]] || usage
    pair=none
    external_arm=$1
    m=$2
    group=
    position=
    ;;
  diagnose)
    [[ $# -eq 2 ]] || usage
    pair=none
    external_arm=$1
    m=$2
    group=
    position=
    [[ "$external_arm" == intern ]] || {
      echo "only Intern correctness failures are diagnosed" >&2
      exit 2
    }
    ;;
  measure)
    [[ $# -eq 5 ]] || usage
    pair=$1
    external_arm=$2
    m=$3
    group=$4
    position=$5
    ;;
  *) usage ;;
esac

case "$m" in
  256|512|1024|2048|4096|8192) ;;
  *) echo "invalid M: $m" >&2; exit 2 ;;
esac
case "$external_arm" in
  production) internal_arm=baseline_4warp ;;
  intern) internal_arm=candidate_4warp_stage4_compact ;;
  exp008) internal_arm=candidate_8warp_n64_temporal_replay_v0 ;;
  *) echo "invalid arm: $external_arm" >&2; exit 2 ;;
esac

if [[ "$action" == measure ]]; then
  [[ "$group" =~ ^[0-2]$ ]] || { echo "group must be 0..2" >&2; exit 2; }
  [[ "$position" =~ ^[0-3]$ ]] || { echo "position must be 0..3" >&2; exit 2; }
  case "$pair" in
    production_intern)
      order=(production intern intern production)
      comparison_subject=candidate_4warp_stage4_compact
      ;;
    production_exp008)
      order=(production exp008 exp008 production)
      comparison_subject=candidate_8warp_n64_temporal_replay_v0
      ;;
    *) echo "invalid pair: $pair" >&2; exit 2 ;;
  esac
  [[ "${order[$position]}" == "$external_arm" ]] || {
    echo "registered order mismatch: $pair position=$position expects ${order[$position]}" >&2
    exit 2
  }
else
  case "$external_arm" in
    intern) comparison_subject=candidate_4warp_stage4_compact ;;
    exp008) comparison_subject=candidate_8warp_n64_temporal_replay_v0 ;;
    production) comparison_subject=candidate_4warp_stage4_compact ;;
  esac
fi

gpu_uuid=${EXP009_GPU_UUID:-GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
expected_app_clock_mhz=2377
lease_id=${EXP009_LEASE_ID:-exp009-full-m-sweep-20260719-carstydev}
lease_dir=${EXP009_LEASE_DIR:-/tmp/kdk-direct-ssh-gpu-leases/R6KD-CX8aaS-GPU-16_GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
host_root=${EXP009_REPO:-/home/xiy/workspace/flashinfer_exp008_748ad}
host_root_real=$(readlink -f "$host_root")
host_git=${EXP009_PARENT_GIT:-/lustre/raplab/client/xiy/workspace/flashinfer/.git}
host_cutlass=$host_root/3rdparty/cutlass
host_submodule_root=${EXP009_SUBMODULE_ROOT:-/home/xiy/workspace/flashinfer_exp002_074d93e}
host_deps=${EXP009_DEPS:-/home/xiy/workspace/w4a4_deps_460}
host_jit_base=${EXP009_FULL_SWEEP_JIT_BASE:-/home/xiy/workspace/exp009_full_m_sweep_jit}
host_jit=$host_jit_base/$external_arm/m$m
container_root=/workspace/source/flashinfer
relative_exp=.claude/w4a4_moe_bench/experiments/exp_009_intern_stage4_compact_lightcheck
relative_exp008=.claude/w4a4_moe_bench/experiments/exp_008_branch_paired_n64_reuse
host_exp=$host_root/$relative_exp
container_exp=$container_root/$relative_exp
host_sweep=$host_exp/results/full_m_sweep
container_sweep=$container_exp/results/full_m_sweep
host_runtime=$host_sweep/runtime
image=nvcr.io/nvidia/pytorch:26.05-py3
extra_mounts=()

case "$external_arm" in
  production)
    host_overlay=$host_root/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py
    container_overlay=$container_root/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py
    expected_overlay_sha256=94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106
    ;;
  intern)
    host_overlay=$host_exp/results/overlays/intern_stage4_compact/moe_dynamic_kernel.py
    container_overlay=$container_exp/results/overlays/intern_stage4_compact/moe_dynamic_kernel.py
    expected_overlay_sha256=42ca8d40e18b5d0f001236b09b85cbc0aa30e6010f0954efd538d8b9a2fb57d2
    ;;
  exp008)
    # Never trust the mutable historical workspace copy.  Bind the hash-locked
    # exp_008 v1 overlay into the exact module path seen by the worker.
    host_overlay=${EXP009_LOCKED_EXP008_OVERLAY:-/home/xiy/workspace/exp009_locked_overlays/branch_paired_n64_v1_f3c246.py}
    container_overlay=$container_root/$relative_exp008/results/overlays/branch_paired_n64_v1/moe_dynamic_kernel.py
    expected_overlay_sha256=f3c246817679d962a3f7160dbe8b9e68262c919e26e306f349200961fc4ac971
    extra_mounts=(-v "$host_overlay:$container_overlay:ro")
    ;;
esac

for required in "$host_overlay" "$host_root" "$host_git" "$host_cutlass" "$host_deps"; do
  [[ -e "$required" ]] || { echo "missing prerequisite: $required" >&2; exit 4; }
done
observed_overlay_sha256=$(sha256sum "$host_overlay" | cut -d' ' -f1)
[[ "$observed_overlay_sha256" == "$expected_overlay_sha256" ]] || {
  echo "overlay hash drift: $observed_overlay_sha256 != $expected_overlay_sha256" >&2
  exit 4
}

canonical_host_results=$host_sweep/canonical/$external_arm
canonical_container_results=$container_sweep/canonical/$external_arm
canonical_preparation=$canonical_host_results/raw/$internal_arm/m${m}/canonical/preparation.json
if [[ "$action" == prepare ]]; then
  host_results=$canonical_host_results
  container_results=$canonical_container_results
  mkdir -p "$host_jit"
elif [[ "$action" == diagnose ]]; then
  host_results=$canonical_host_results
  container_results=$canonical_container_results
  failure=$canonical_host_results/raw/$internal_arm/m${m}/canonical/failure.json
  diagnostic=$host_sweep/diagnostics/intern/m${m}.json
  container_failure=$canonical_container_results/raw/$internal_arm/m${m}/canonical/failure.json
  container_diagnostic=$container_sweep/diagnostics/intern/m${m}.json
  [[ -f "$failure" ]] || {
    echo "missing failed correctness evidence: $failure" >&2
    exit 5
  }
  [[ ! -e "$diagnostic" ]] || {
    echo "immutable diagnostic already exists: $diagnostic" >&2
    exit 4
  }
  [[ -d "$host_jit" ]] || {
    echo "missing failed prepare JIT: $host_jit" >&2
    exit 5
  }
  mkdir -p "$(dirname "$diagnostic")"
else
  [[ -f "$canonical_preparation" ]] || {
    echo "correctness preparation did not pass: $canonical_preparation" >&2
    exit 5
  }
  host_results=$host_sweep/bench/$pair/$external_arm
  container_results=$container_sweep/bench/$pair/$external_arm
  pair_preparation=$host_results/raw/$internal_arm/m${m}/canonical/preparation.json
  mkdir -p "$(dirname "$pair_preparation")"
  if [[ -f "$pair_preparation" ]]; then
    cmp -s "$canonical_preparation" "$pair_preparation" || {
      echo "pair preparation identity drift: $pair_preparation" >&2
      exit 4
    }
  else
    cp --preserve=mode,timestamps "$canonical_preparation" "$pair_preparation"
  fi
fi
mkdir -p "$host_runtime"

grep -qx "lease_id=$lease_id" "$lease_dir/metadata"
grep -qx "gpu_uuid=$gpu_uuid" "$lease_dir/metadata"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
    | grep -qx "$gpu_uuid"; then
  echo "leased GPU has a foreign compute process" >&2
  exit 3
fi
observed=$(nvidia-smi --id="$gpu_uuid" \
  --query-gpu=uuid,clocks.applications.graphics --format=csv,noheader,nounits)
observed_uuid=$(printf '%s' "$observed" | cut -d, -f1 | xargs)
observed_clock=$(printf '%s' "$observed" | cut -d, -f2 | xargs)
[[ "$observed_uuid" == "$gpu_uuid" && "$observed_clock" == "$expected_app_clock_mhz" ]] || {
  echo "GPU/clock identity drift: $observed" >&2
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

uid=$(id -u)
gid=$(id -g)
docker_args=(
  docker run --rm
  --user "$uid:$gid"
  --gpus "device=$gpu_uuid"
  -v "$host_root:$container_root"
  "${extra_mounts[@]}"
  -v "$host_root_real:$host_root_real:ro"
  -v "$host_git:$host_git:ro"
  -v "$host_cutlass:$container_root/3rdparty/cutlass:ro"
  -v "$host_submodule_root:$host_submodule_root:ro"
  -v "$host_deps:/workspace/deps:ro"
  -v "$host_jit:/workspace/jit"
  -e HOME=/tmp
  -e PYTHONDONTWRITEBYTECODE=1
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
  "$image"
)

if [[ "$action" == diagnose ]]; then
  worker=(
    python3 "$container_exp/diagnose_failure.py"
    --flashinfer-root "$container_root"
    --overlay "$container_overlay"
    --jit-root /workspace/jit
    --failure "$container_failure"
    --output "$container_diagnostic"
    --expected-gpu-uuid "$gpu_uuid"
    --m "$m"
  )
  label=diagnose_${external_arm}_m${m}
else
  worker=(
    python3 "$container_exp/run_exp009_arm.py"
    --flashinfer-root "$container_root"
    --results "$container_results"
    --arm "$internal_arm"
    --m "$m"
    --fixture canonical
    --overlay "$container_overlay"
    --jit-root /workspace/jit
    --expected-gpu-uuid "$gpu_uuid"
    --comparison-anchor baseline_4warp
    --comparison-subject "$comparison_subject"
  )
fi

if [[ "$action" == prepare ]]; then
  worker+=(prepare)
  label=prepare_${external_arm}_m${m}
elif [[ "$action" == measure ]]; then
  worker+=(
    measure
    --group "$group"
    --position "$position"
    --warmup 5
    --iters 50
    --clock-policy locked
  )
  label=measure_${pair}_${external_arm}_m${m}_g${group}_p${position}
fi

{
  printf '%q ' "${docker_args[@]}" "${worker[@]}"
  printf '\n'
} > "$host_runtime/$label.command.txt"

"${docker_args[@]}" "${worker[@]}" \
  > >(tee "$host_runtime/$label.stdout.log") \
  2> >(tee "$host_runtime/$label.stderr.log" >&2)

nvidia-smi --id="$gpu_uuid" \
  --query-gpu=uuid,clocks.current.graphics,clocks.applications.graphics,memory.used,utilization.gpu \
  --format=csv,noheader,nounits > "$host_runtime/$label.post_gpu.csv"
