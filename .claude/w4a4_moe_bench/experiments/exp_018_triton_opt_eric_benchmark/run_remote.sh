#!/usr/bin/env bash
set -euo pipefail

relative_exp=.claude/w4a4_moe_bench/experiments/exp_018_triton_opt_eric_benchmark
remote_host=${EXP018_GPU_HOST:-10.6.142.16}
remote_root=${EXP018_REMOTE_ROOT:-/home/xiy/workspace/flashinfer_exp001_corrected_074d93e}

if [[ ${EXP018_ON_GPU_HOST:-0} != 1 ]]; then
  repo_root=$(git rev-parse --show-toplevel)
  ssh -o BatchMode=yes "$remote_host" "mkdir -p '$remote_root/$relative_exp'"
  rsync -a --exclude results --exclude __pycache__ \
    "$repo_root/$relative_exp/" "$remote_host:$remote_root/$relative_exp/"
  rsync -aR --exclude __pycache__ \
    "$repo_root/./.claude/w4a4_moe_bench/breakdown_harness/" \
    "$repo_root/./.claude/w4a4_moe_bench/moe_dynamic_kernel_opt.py" \
    "$repo_root/./.claude/w4a4_moe_bench/moe_dyanmice_kernel_ab_stage4_compact.py" \
    "$remote_host:$remote_root/"
  ssh -o BatchMode=yes "$remote_host" \
    "EXP018_ON_GPU_HOST=1 EXP018_REMOTE_ROOT='$remote_root' bash '$remote_root/$relative_exp/run_remote.sh'"
  mkdir -p "$repo_root/$relative_exp/results"
  rsync -a --delete "$remote_host:$remote_root/$relative_exp/results/" \
    "$repo_root/$relative_exp/results/"
  exit 0
fi

host_root=$remote_root
host_exp=$host_root/$relative_exp
host_results=$host_exp/results
container_root=/workspace/source/flashinfer
container_exp=$container_root/$relative_exp
container_results=$container_exp/results
container_fixtures=$container_root/.claude/w4a4_moe_bench/experiments/exp_001_backend_case_sweep/results/fixtures
host_deps=/home/xiy/workspace/w4a4_deps_460
gpu_uuid=${EXP018_GPU_UUID:-GPU-2fdb0b79-0ba7-f356-b714-6c461b71ce12}
expected_clock=2377
lease_root=${EXP018_LEASE_ROOT:-/tmp/kdk-direct-ssh-gpu-leases}
host_id=$(hostname)
lease_id=${EXP018_RERUN_ID:-exp018-eric-benchmark-$(date -u +%Y%m%dT%H%M%SZ)-$$}
lease_dir=$lease_root/${host_id}_${gpu_uuid}
jit_base=/home/xiy/workspace/exp018_jit/$lease_id

fp4_image=nvcr.io/nvidia/pytorch:26.05-py3
fp4_image_id=sha256:a4e056e1d34a5cc9387512ffa3abeed778e3dc7966633c5154d771705d8835ac
fp4_image_digest=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba
fp4_deps=32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74
triton_image=lmsysorg/sglang:latest
triton_image_id=sha256:663867442f321ded36228bafd889fd1db05cbef7a7c8ea6e072df33234dabbfd
triton_image_digest=sha256:00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2
sglang_commit=0b3bb0cbe31873994c9f989fddfe2f87ca839fdd

gpu_has_process() {
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
    | grep -qx "$gpu_uuid"
}

gpu_contract_ok() {
  local row observed_uuid app_clock
  row=$(nvidia-smi --id="$gpu_uuid" \
    --query-gpu=uuid,clocks.applications.graphics --format=csv,noheader,nounits)
  IFS=, read -r observed_uuid app_clock <<<"$row"
  observed_uuid=${observed_uuid// /}
  app_clock=${app_clock// /}
  [[ $observed_uuid == "$gpu_uuid" && $app_clock == "$expected_clock" ]]
}

acquire_lease() {
  mkdir -p "$lease_root"
  gpu_contract_ok
  ! gpu_has_process
  mkdir "$lease_dir"
  {
    printf 'lease_id=%s\n' "$lease_id"
    printf 'gpu_uuid=%s\n' "$gpu_uuid"
    printf 'owner=%s@%s\n' "$(id -un)" "$host_id"
    printf 'purpose=exp018 Triton FP8 vs Latest Opt vs Eric Stage4 benchmark\n'
  } >"$lease_dir/metadata"
  gpu_contract_ok
  ! gpu_has_process
}

cleanup() {
  local code=$?
  if ! gpu_has_process; then
    if [[ -d $jit_base ]]; then
      docker run --rm -v "$jit_base:/workspace/jit" "$fp4_image" \
        find /workspace/jit -mindepth 1 -depth -delete >/dev/null
      rmdir "$jit_base"
    fi
    if [[ -f $lease_dir/metadata ]] && grep -qx "lease_id=$lease_id" "$lease_dir/metadata"; then
      rm -f "$lease_dir/metadata"
      rmdir "$lease_dir"
    fi
  else
    echo "selected GPU still has a process; retaining lease and JIT for audit" >&2
  fi
  exit "$code"
}

verify_images() {
  [[ $(docker image inspect "$fp4_image" --format '{{.Id}}') == "$fp4_image_id" ]]
  [[ $(docker image inspect "$triton_image" --format '{{.Id}}') == "$triton_image_id" ]]
  docker image inspect "$fp4_image" --format '{{json .RepoDigests}}' | grep -q "$fp4_image_digest"
  docker image inspect "$triton_image" --format '{{json .RepoDigests}}' | grep -q "$triton_image_digest"
}

artifact_lock() {
  python3 - "$host_results/raw/$1/block_0.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["jit_identity"]["artifact_set_sha256"])
PY
}

run_arm() {
  local arm=$1 block=$2 policy=$3 expected_hash=${4:-}
  local host_jit=$jit_base/$arm
  local container_jit=/workspace/jit
  local common=(
    --gpus "device=$gpu_uuid"
    -v "$host_root:$container_root"
    -v "$host_jit:$container_jit"
    -e CUDA_VISIBLE_DEVICES=0
    -e KDK_LEASE_ID="$lease_id"
    -e W4A4_RERUN_ID="$lease_id"
    -w "$container_root"
  )
  local command=(
    python3 "$container_exp/run_arm.py"
    --arm "$arm"
    --block "$block"
    --jit-policy "$policy"
    --jit-root "$container_jit"
    --expected-gpu-uuid "$gpu_uuid"
    --rerun-id "$lease_id"
    --flashinfer-root "$container_root"
    --fixture-dir "$container_fixtures"
    --results "$container_results"
  )
  if [[ $policy == reuse ]]; then
    command+=(--expected-jit-artifact-set-sha256 "$expected_hash")
  fi
  if [[ $arm == sglang_triton_fp8 ]]; then
    docker run --rm "${common[@]}" \
      -e W4A4_IMAGE_ID="$triton_image_id" \
      -e W4A4_IMAGE_DIGEST="$triton_image_digest" \
      -e W4A4_SGLANG_COMMIT="$sglang_commit" \
      "$triton_image" "${command[@]}"
  else
    docker run --rm "${common[@]}" \
      -v "$host_deps:/workspace/deps:ro" \
      -e PYTHONPATH="$container_root:/workspace/deps:/workspace/deps/nvidia_cutlass_dsl/dsl_packages" \
      -e W4A4_IMAGE_ID="$fp4_image_id" \
      -e W4A4_IMAGE_DIGEST="$fp4_image_digest" \
      -e W4A4_PYTHON_DEPS_SHA256="$fp4_deps" \
      "$fp4_image" "${command[@]}"
  fi
}

[[ -d $host_root/.git && -d $host_deps && -f $host_exp/run_arm.py ]]
[[ ! -e $host_results ]]
verify_images
acquire_lease
trap cleanup EXIT INT TERM HUP
mkdir -p "$host_results" "$jit_base"

for block in 0 1 2; do
  case $block in
    0) order=(latest_opt_fp4 eric_stage4_fp4 sglang_triton_fp8) ;;
    1) order=(eric_stage4_fp4 sglang_triton_fp8 latest_opt_fp4) ;;
    2) order=(sglang_triton_fp8 latest_opt_fp4 eric_stage4_fp4) ;;
  esac
  for arm in "${order[@]}"; do
    gpu_contract_ok
    ! gpu_has_process
    if [[ $block == 0 ]]; then
      mkdir -p "$jit_base/$arm"
      run_arm "$arm" "$block" fresh
    else
      run_arm "$arm" "$block" reuse "$(artifact_lock "$arm")"
    fi
    ! gpu_has_process
    sleep 2
  done
done

docker run --rm -v "$host_root:$container_root" -w "$container_root" \
  "$fp4_image" python3 "$container_exp/build_result.py" --results "$container_results"
