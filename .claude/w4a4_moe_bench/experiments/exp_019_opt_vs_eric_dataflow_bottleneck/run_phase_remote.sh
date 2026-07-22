#!/usr/bin/env bash
set -euo pipefail

on_host=0
if [[ ${1:-} == --on-host ]]; then
  on_host=1
  shift
fi
mode=${1:-m8192}
case "$mode" in
  m8192|m1024) ;;
  *) echo "usage: $0 [--on-host] {m8192|m1024}" >&2; exit 2 ;;
esac

host=${EXP019_GPU_HOST:-10.6.142.16}
repo=${EXP019_REPO:-/home/xiy/workspace/flashinfer_exp001_corrected_074d93e}
relative_exp=.claude/w4a4_moe_bench/experiments/exp_019_opt_vs_eric_dataflow_bottleneck

if [[ $on_host == 0 ]]; then
  local_repo=$(git rev-parse --show-toplevel)
  rsync -a --exclude results --exclude __pycache__ \
    "$local_repo/$relative_exp/" "$host:$repo/$relative_exp/"
  rsync -a --exclude __pycache__ \
    "$local_repo/.claude/w4a4_moe_bench/breakdown_harness/" \
    "$host:$repo/.claude/w4a4_moe_bench/breakdown_harness/"
  for dependency in \
    exp_005_8warp_spill_reduction \
    exp_014_scatter_8warp \
    exp_016_route_q0_token_major_reuse \
    exp_017_opt_vs_triton_phase_share \
    exp_018_triton_opt_eric_benchmark; do
    rsync -a --exclude results --exclude __pycache__ \
      "$local_repo/.claude/w4a4_moe_bench/experiments/$dependency/" \
      "$host:$repo/.claude/w4a4_moe_bench/experiments/$dependency/"
  done
  remote=(env EXP019_REPO="$repo" bash "$repo/$relative_exp/run_phase_remote.sh" --on-host "$mode")
  printf -v remote_command '%q ' "${remote[@]}"
  ssh -T "$host" "$remote_command"
  mkdir -p "$local_repo/$relative_exp/results"
  rsync -a --prune-empty-dirs --include '*/' --include '*.json' --exclude '*' \
    "$host:$repo/$relative_exp/results/raw/phase/" \
    "$local_repo/$relative_exp/results/raw/phase/"
  exit 0
fi

exp=$repo/$relative_exp
results=$exp/results
raw=$results/raw/phase
overlay=$results/phase_overlays
container_root=/workspace/source/flashinfer
container_exp=$container_root/$relative_exp
container_results=$container_exp/results
fixture_dir=$container_root/.claude/w4a4_moe_bench/experiments/exp_001_backend_case_sweep/results/fixtures
deps=${EXP019_DEPS:-/home/xiy/workspace/w4a4_deps_460}
image=nvcr.io/nvidia/pytorch:26.05-py3
image_id=sha256:a4e056e1d34a5cc9387512ffa3abeed778e3dc7966633c5154d771705d8835ac
image_digest=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba
deps_sha256=32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74
gpu_uuid=GPU-2fdb0b79-0ba7-f356-b714-6c461b71ce12
expected_clock=2377
lease_root=${EXP019_LEASE_ROOT:-/tmp/kdk-direct-ssh-gpu-leases}
lease_id=${EXP019_RERUN_ID:-exp019-phase-${mode}-$(date -u +%Y%m%dT%H%M%SZ)-$$}
lease_dir=$lease_root/$(hostname)_${gpu_uuid}
jit_base=${EXP019_JIT_BASE:-/home/xiy/workspace/exp019_phase_jit/$lease_id}
m=${mode#m}

gpu_has_process() {
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
    | grep -qx "$gpu_uuid"
}

gpu_contract_ok() {
  local row observed_uuid clock
  row=$(nvidia-smi --id="$gpu_uuid" \
    --query-gpu=uuid,clocks.applications.graphics --format=csv,noheader,nounits)
  observed_uuid=$(printf '%s' "$row" | cut -d, -f1 | xargs)
  clock=$(printf '%s' "$row" | cut -d, -f2 | xargs)
  [[ $observed_uuid == "$gpu_uuid" && $clock == "$expected_clock" ]]
  ! gpu_has_process
}

lease_owned=0
cleanup() {
  local status=$?
  if [[ $lease_owned == 1 && -f $lease_dir/metadata ]] \
      && grep -qx "lease_id=$lease_id" "$lease_dir/metadata" \
      && ! gpu_has_process; then
    rm -f "$lease_dir/metadata"
    rmdir "$lease_dir"
  fi
  exit "$status"
}

[[ -d $repo && -d $deps ]]
[[ $(docker image inspect "$image" --format '{{.Id}}') == "$image_id" ]]
docker image inspect "$image" --format '{{json .RepoDigests}}' \
  | grep -q "nvcr.io/nvidia/pytorch@$image_digest"
gpu_contract_ok
mkdir -p "$lease_root"
mkdir "$lease_dir"
lease_owned=1
printf 'lease_id=%s\ngpu_uuid=%s\npurpose=exp019 paired phase probe\n' \
  "$lease_id" "$gpu_uuid" >"$lease_dir/metadata"
trap cleanup EXIT INT TERM HUP
gpu_contract_ok

common_docker=(
  docker run --rm
  -v "$repo:$container_root"
  -v "$deps:/workspace/deps:ro"
  -e PYTHONDONTWRITEBYTECODE=1
  -e PYTHONPATH="$container_root:/workspace/deps:/workspace/deps/nvidia_cutlass_dsl/dsl_packages"
  -w "$container_exp"
  "$image"
)

if [[ ! -f $overlay/identity.json ]]; then
  "${common_docker[@]}" python3 "$container_exp/build_phase_overlays.py" \
    --flashinfer-root "$container_root" \
    --output "$container_results/phase_overlays" >/dev/null
else
  "${common_docker[@]}" python3 "$container_exp/build_phase_overlays.py" \
    --flashinfer-root "$container_root" \
    --output "$container_results/phase_overlays" --check-existing >/dev/null
fi

cells=(
  latest_opt_fp4:control_no_marker
  eric_stage4_fp4:control_no_marker
  latest_opt_fp4:probe
  eric_stage4_fp4:probe
)
for cell in "${cells[@]}"; do
  arm=${cell%%:*}
  phase_mode=${cell##*:}
  out=$raw/$mode/block0/${arm}_${phase_mode}.json
  log=$raw/$mode/block0/${arm}_${phase_mode}.log
  host_jit=$jit_base/$arm/$phase_mode
  [[ ! -e $out && ! -e $host_jit ]]
  mkdir -p "$(dirname "$out")" "$host_jit"
  gpu_contract_ok
  docker run --rm \
    --gpus "device=$gpu_uuid" \
    -v "$repo:$container_root" \
    -v "$deps:/workspace/deps:ro" \
    -v "$host_jit:/workspace/jit" \
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
    -e W4A4_IMAGE_ID="$image_id" \
    -e W4A4_IMAGE_DIGEST="$image_digest" \
    -e W4A4_PYTHON_DEPS_SHA256="$deps_sha256" \
    -w "$container_exp" \
    "$image" python3 "$container_exp/capture_phase.py" \
      --flashinfer-root "$container_root" \
      --overlay-root "$container_results/phase_overlays" \
      --arm "$arm" --mode "$phase_mode" --m "$m" --cyclic-block 0 \
      --jit-root /workspace/jit --output "$container_results/raw/phase/$mode/block0/${arm}_${phase_mode}.json" \
      --fixture-dir "$fixture_dir" --expected-gpu-uuid "$gpu_uuid" \
      --expected-app-clock-mhz "$expected_clock" >"$log" 2>&1
  gpu_contract_ok
done
