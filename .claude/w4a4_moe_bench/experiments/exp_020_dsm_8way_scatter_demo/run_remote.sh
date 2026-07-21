#!/usr/bin/env bash
set -euo pipefail

on_host=0
if [[ ${1:-} == --on-host ]]; then
  on_host=1
  shift
fi
mode=${1:-all}
case "$mode" in
  validate|benchmark|all) ;;
  *) echo "usage: $0 [--on-host] {validate|benchmark|all}" >&2; exit 2 ;;
esac

host=${EXP020_GPU_HOST:-10.6.142.16}
repo=${EXP020_REPO:-/home/xiy/workspace/flashinfer_exp001_corrected_074d93e}
relative_exp=.claude/w4a4_moe_bench/experiments/exp_020_dsm_8way_scatter_demo

if [[ $on_host == 0 ]]; then
  local_repo=$(git rev-parse --show-toplevel)
  rsync -a --exclude __pycache__ --exclude 'results/raw/' \
    "$local_repo/$relative_exp/" "$host:$repo/$relative_exp/"
  if [[ $mode == benchmark \
        && -f $local_repo/$relative_exp/results/raw/demo.json ]]; then
    ssh -T "$host" mkdir -p "$repo/$relative_exp/results/raw"
    rsync -a "$local_repo/$relative_exp/results/raw/demo.json" \
      "$host:$repo/$relative_exp/results/raw/demo.json"
  fi
  remote=(env EXP020_REPO="$repo" bash "$repo/$relative_exp/run_remote.sh" --on-host "$mode")
  printf -v remote_command '%q ' "${remote[@]}"
  ssh -T "$host" "$remote_command"
  mkdir -p "$local_repo/$relative_exp/results/raw"
  rsync -a "$host:$repo/$relative_exp/results/manifest.json" \
    "$local_repo/$relative_exp/results/manifest.json"
  rsync -a "$host:$repo/$relative_exp/results/raw/demo.json" \
    "$local_repo/$relative_exp/results/raw/demo.json"
  exit 0
fi

exp=$repo/$relative_exp
results=$exp/results
fixture=$repo/.claude/w4a4_moe_bench/experiments/exp_001_backend_case_sweep/results/fixtures/m8192.npz
deps=${EXP020_DEPS:-/home/xiy/workspace/w4a4_deps_460}
image=nvcr.io/nvidia/pytorch:26.05-py3
image_id=sha256:a4e056e1d34a5cc9387512ffa3abeed778e3dc7966633c5154d771705d8835ac
image_digest=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba
gpu_uuid=${EXP020_GPU_UUID:-GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
expected_clock=${EXP020_CLOCK_MHZ:-2377}
lease_root=${EXP020_LEASE_ROOT:-/tmp/kdk-direct-ssh-gpu-leases}
run_id=${EXP020_RUN_ID:-exp020-dsm-demo-$(date -u +%Y%m%dT%H%M%SZ)-$$}
lease_dir=$lease_root/$(hostname)_$gpu_uuid
jit_host=${EXP020_JIT_ROOT:-/home/xiy/workspace/exp020_dsm_demo_jit/$run_id}

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
      && grep -qx "lease_id=$run_id" "$lease_dir/metadata" \
      && ! gpu_has_process; then
    rm -f "$lease_dir/metadata"
    rmdir "$lease_dir"
  fi
  exit "$status"
}

[[ -f $exp/run_demo.py && -f $exp/dsm_scatter_demo.py && -f $fixture ]]
[[ -d $deps/nvidia_cutlass_dsl/dsl_packages ]]
[[ $(docker image inspect "$image" --format '{{.Id}}') == "$image_id" ]]
docker image inspect "$image" --format '{{json .RepoDigests}}' | grep -q "nvcr.io/nvidia/pytorch@$image_digest"
gpu_contract_ok
mkdir -p "$lease_root"
mkdir "$lease_dir"
lease_owned=1
printf 'lease_id=%s\ngpu_uuid=%s\npurpose=exp020 DSM 8-way scatter demo\n' \
  "$run_id" "$gpu_uuid" >"$lease_dir/metadata"
trap cleanup EXIT INT TERM HUP

mkdir -p "$results/raw" "$jit_host"
gpu_contract_ok
docker run --rm \
  --gpus "device=$gpu_uuid" \
  -v "$repo:/workspace/source/flashinfer" \
  -v "$deps:/workspace/deps:ro" \
  -v "$jit_host:/workspace/jit" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/workspace/source/flashinfer:/workspace/deps:/workspace/deps/nvidia_cutlass_dsl/dsl_packages \
  -e FLASHINFER_WORKSPACE_BASE=/workspace/jit \
  -e CUTE_DSL_CACHE_DIR=/workspace/jit/cache \
  -e CUTE_DSL_DUMP_DIR=/workspace/jit/dump \
  -e CUTE_DSL_KEEP=ir,ptx,cubin,sass \
  -e TORCH_CUDA_ARCH_LIST=12.0a \
  -e EXP020_EXPECTED_GPU_UUID="$gpu_uuid" \
  -e EXP020_EXPECTED_CLOCK_MHZ="$expected_clock" \
  -e EXP020_IMAGE_ID="$image_id" \
  -e EXP020_IMAGE_DIGEST="$image_digest" \
  -w "/workspace/source/flashinfer/$relative_exp" \
  "$image" python3 run_demo.py "$mode" \
    --fixture "/workspace/source/flashinfer/.claude/w4a4_moe_bench/experiments/exp_001_backend_case_sweep/results/fixtures/m8192.npz" \
    --results "/workspace/source/flashinfer/$relative_exp/results" \
    --jit-root /workspace/jit \
    >"$results/raw/run.stdout.log" 2>"$results/raw/run.stderr.log"
gpu_contract_ok
