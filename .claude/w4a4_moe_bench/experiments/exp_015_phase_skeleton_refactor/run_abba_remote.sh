#!/usr/bin/env bash
set -euo pipefail

repo=${EXP015_REPO:-/home/xiy/workspace/flashinfer_exp015_748ad}
relative_exp=.claude/w4a4_moe_bench/experiments/exp_015_phase_skeleton_refactor
exp=$repo/$relative_exp
results=$exp/results
worker=$exp/run_exp015_arm.py
container=${EXP015_CONTAINER:-xiyExp0155kp}
gpu_uuid=GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522
lease_id=exp015-phase-skeleton-20260720
lease_dir=/tmp/kdk-direct-ssh-gpu-leases/R6KD-CX8aaS-GPU-16_GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522
expected_clock=2377
pythonpath=$repo:/home/xiy/workspace/w4a4_deps_460:/home/xiy/workspace/w4a4_deps_460/nvidia_cutlass_dsl/dsl_packages
order=(baseline candidate_v2 candidate_v2 baseline)

grep -qx "lease_id=$lease_id" "$lease_dir/metadata"
grep -qx "gpu_uuid=$gpu_uuid" "$lease_dir/metadata"
mkdir -p "$results/runtime"

for m in 256 8192; do
  for group in 0 1 2 3 4; do
    existing=0
    for position in 0 1 2 3; do
      arm=${order[$position]}
      output=$results/raw/benchmark/m${m}/group_${group}_position_${position}_${arm}.json
      [[ ! -e $output ]] || existing=$((existing + 1))
    done
    if [[ $existing -eq 4 ]]; then
      continue
    fi
    if [[ $existing -ne 0 ]]; then
      echo "refusing to splice partial ABBA group: M=$m group=$group ($existing/4)" >&2
      exit 4
    fi

    for position in 0 1 2 3; do
      arm=${order[$position]}
      if [[ $arm == baseline ]]; then
        overlay=$results/overlays/baseline/moe_dynamic_kernel.py
        jit=/home/xiy/workspace/exp015_validate_baseline
        artifact_set=286ee3e50518e70a343758174c6aa95db76038b5bdc0aa529f2208e45cc9ef3b
        cubin=4b835aa8ce91a4dd12b4dc4f43508c205c117aaeb193995fff57dd3ddbeb7725
      else
        overlay=$results/overlays/candidate_v2/moe_dynamic_kernel.py
        jit=/home/xiy/workspace/exp015_validate_candidate_v2
        artifact_set=4358e71894a2602030e32e33b13298a3b739ff4c554aaead82b4ef9c0373d3cc
        cubin=fee96b35d9b2c83e354504774fba2e2bc10e54f0316ade18f8adbdabb2ecbada
      fi

      if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
        | grep -qx "$gpu_uuid"; then
        echo "foreign compute process detected on leased GPU" >&2
        exit 5
      fi
      observed_clock=$(nvidia-smi --id="$gpu_uuid" \
        --query-gpu=clocks.applications.graphics --format=csv,noheader,nounits \
        | xargs)
      [[ $observed_clock == "$expected_clock" ]]

      echo "M=$m group=$group position=$position arm=$arm"
      docker exec \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e PYTHONPATH="$pythonpath" \
        -e FLASHINFER_NVFP4_4OVER6=0 \
        -e FLASHINFER_WORKSPACE_BASE="$jit" \
        -e CUTE_DSL_CACHE_DIR="$jit/cache" \
        -e CUTE_DSL_DUMP_DIR="$jit/dump" \
        -e CUTE_DSL_KEEP=ir,ptx,cubin,sass \
        -e TORCH_CUDA_ARCH_LIST=12.0a \
        -e KDK_LEASE_ID="$lease_id" \
        -e KDK_LEASE_GPU_UUID="$gpu_uuid" \
        -e W4A4_IMAGE_ID=sha256:a4e056e1d34a5cc9387512ffa3abeed778e3dc7966633c5154d771705d8835ac \
        -e W4A4_IMAGE_DIGEST=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba \
        -e W4A4_PYTHON_DEPS_SHA256=32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74 \
        "$container" python3 "$worker" \
        --flashinfer-root "$repo" \
        --results "$results" \
        --arm "$arm" \
        --overlay "$overlay" \
        --jit-root "$jit" \
        --expected-gpu-uuid "$gpu_uuid" \
        measure \
        --m "$m" \
        --group "$group" \
        --position "$position" \
        --expected-app-clock-mhz "$expected_clock" \
        --expected-jit-artifact-set-sha256 "$artifact_set" \
        --expected-cubin-sha256 "$cubin" \
        > "$results/runtime/abba_m${m}_g${group}_p${position}_${arm}.stdout.json"
    done
  done
done
