#!/usr/bin/env bash
set -euo pipefail

repo=/home/xiy/workspace/flashinfer_exp008_748ad
repo_real=$(readlink -f "$repo")
rel=.claude/w4a4_moe_bench/experiments/exp_013_compact_epilogue_stage2
exp=$repo/$rel
worker=$exp/run_quick_benchmark.py
gpu_uuid=GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522
lease_id=${EXP013_LEASE_ID:-exp013-compact-epilogue-20260719}
lease_dir=/tmp/kdk-direct-ssh-gpu-leases/R6KD-CX8aaS-GPU-16_GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522
deps=/home/xiy/workspace/w4a4_deps_460
host_git=/lustre/raplab/client/xiy/workspace/flashinfer/.git
host_cutlass=$repo/3rdparty/cutlass
host_submodule_root=/home/xiy/workspace/flashinfer_exp002_074d93e
container_repo=/workspace/source/flashinfer
image=nvcr.io/nvidia/pytorch:26.05-py3
order=(exp008 exp013_v2 exp013_v2 exp008)

grep -qx "lease_id=$lease_id" "$lease_dir/metadata"
grep -qx "gpu_uuid=$gpu_uuid" "$lease_dir/metadata"

for m in 256 8192; do
  for group in 0 1; do
    for position in 0 1 2 3; do
      arm=${order[$position]}
      if [[ $arm == exp008 ]]; then
        overlay=/home/xiy/workspace/exp013_anchor_exp008_v1.py
        container_overlay=/workspace/exp013_anchor_exp008_v1.py
        overlay_mount=(-v "$overlay:$container_overlay:ro")
        expected_sha=f3c246817679d962a3f7160dbe8b9e68262c919e26e306f349200961fc4ac971
        jit=/home/xiy/workspace/exp008_branch_paired_jit/v1/m${m}/canonical
      else
        overlay=$exp/results/overlays/compact_epi_m64_stage2_v2/moe_dynamic_kernel.py
        container_overlay=$container_repo/${overlay#"$repo/"}
        overlay_mount=()
        expected_sha=e2fb46e49001b7fe17761fcd9af92b8775d41c6b0c5932172e3c57839d4199e5
        jit=/home/xiy/workspace/exp013_v2_correctness_jit
      fi
      output=$exp/results/perf/raw/m${m}/group_${group}_position_${position}_${arm}.json
      if [[ -f $output ]]; then
        grep -q '"status": "complete"' "$output"
        continue
      fi
      if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | grep -qx "$gpu_uuid"; then
        echo "foreign process detected on leased GPU" >&2
        exit 3
      fi
      mkdir -p "$(dirname "$output")" "$exp/results/runtime"
      docker run --rm --user "$(id -u):$(id -g)" --gpus "device=$gpu_uuid" \
        -v "$repo:$container_repo" \
        -v "$repo_real:$repo_real:ro" \
        -v "$host_git:$host_git:ro" \
        -v "$host_cutlass:$container_repo/3rdparty/cutlass:ro" \
        -v "$host_submodule_root:$host_submodule_root:ro" \
        -v "$deps:/workspace/deps:ro" \
        -v "$jit:/workspace/jit" \
        "${overlay_mount[@]}" \
        -e HOME=/tmp \
        -e PYTHONDONTWRITEBYTECODE=1 \
        -e PYTHONPATH="$container_repo:/workspace/deps:/workspace/deps/nvidia_cutlass_dsl/dsl_packages" \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e KDK_LEASE_ID="$lease_id" \
        -e KDK_LEASE_GPU_UUID="$gpu_uuid" \
        -e W4A4_IMAGE_DIGEST=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba \
        -e W4A4_PYTHON_DEPS_SHA256=32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74 \
        -e FLASHINFER_NVFP4_4OVER6=0 \
        -e FLASHINFER_WORKSPACE_BASE=/workspace/jit \
        -e CUTE_DSL_CACHE_DIR=/workspace/jit/cache \
        -e CUTE_DSL_DUMP_DIR=/workspace/jit/dump \
        -e CUTE_DSL_KEEP=ir,ptx,cubin,sass \
        -w "$container_repo" "$image" \
        python3 "$container_repo/$rel/run_quick_benchmark.py" \
          --flashinfer-root "$container_repo" \
          --overlay "$container_overlay" \
          --expected-overlay-sha256 "$expected_sha" \
          --jit-root /workspace/jit \
          --output "$container_repo/${output#"$repo/"}" \
          --expected-gpu-uuid "$gpu_uuid" \
          --external-arm "$arm" \
          --m "$m" --group "$group" --position "$position" \
        > "$exp/results/runtime/perf_m${m}_g${group}_p${position}_${arm}.stdout.log" \
        2> "$exp/results/runtime/perf_m${m}_g${group}_p${position}_${arm}.stderr.log"
    done
  done
done
