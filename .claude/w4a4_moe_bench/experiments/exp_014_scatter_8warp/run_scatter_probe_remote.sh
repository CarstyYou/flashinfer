#!/usr/bin/env bash
set -euo pipefail

# Build and capture the matched baseline/candidate Scatter phase probes.
# Default scope is canonical M8192; EXP014_PROBE_M may select another registered M.

host=${EXP014_HOST:-xiy@10.6.142.16}
repo=${EXP014_REPO:-/home/xiy/workspace/flashinfer_exp015_748ad}
relative_exp=.claude/w4a4_moe_bench/experiments/exp_014_scatter_8warp
container=${EXP014_CONTAINER:-xiyExp0145kp}
gpu_uuid=${EXP014_GPU_UUID:-GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
lease_id=${EXP014_LEASE_ID:-exp014-scatter-8warp-20260720}
lease_dir=${EXP014_LEASE_DIR:-/tmp/kdk-direct-ssh-gpu-leases/R6KD-CX8aaS-GPU-16_GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
m=${EXP014_PROBE_M:-8192}
expected_clock=${EXP014_CLOCK_MHZ:-2377}
jit_base=${EXP014_PROBE_JIT_BASE:-/home/xiy/workspace/exp014_scatter_probe_jit}

if [[ ${1:-} != --on-host ]]; then
  remote_script=$repo/$relative_exp/run_scatter_probe_remote.sh
  remote=(
    env
    EXP014_REPO="$repo"
    EXP014_CONTAINER="$container"
    EXP014_GPU_UUID="$gpu_uuid"
    EXP014_LEASE_ID="$lease_id"
    EXP014_LEASE_DIR="$lease_dir"
    EXP014_PROBE_M="$m"
    EXP014_CLOCK_MHZ="$expected_clock"
    EXP014_PROBE_JIT_BASE="$jit_base"
    bash "$remote_script" --on-host
  )
  printf -v remote_command '%q ' "${remote[@]}"
  exec ssh -T "$host" "$remote_command"
fi

exp=$repo/$relative_exp
results=$exp/results
runtime=$results/runtime/scatter_phase_probe
builder=$exp/build_scatter_phase_probe.py
worker=$exp/run_exp014_scatter_probe.py
overlay_root=$results/scatter_phase_probe_overlays
pythonpath=$repo:/home/xiy/workspace/w4a4_deps_460:/home/xiy/workspace/w4a4_deps_460/nvidia_cutlass_dsl/dsl_packages
arms=(baseline_4warp_scatter candidate_8warp_scatter)

case "$m" in
  256|512|1024|2048|4096|8192) ;;
  *) echo "unregistered exp014 probe M: $m" >&2; exit 2 ;;
esac
[[ -f $builder && -f $worker && -f $lease_dir/metadata ]]
grep -qx "lease_id=$lease_id" "$lease_dir/metadata"
grep -qx "gpu_uuid=$gpu_uuid" "$lease_dir/metadata"
mkdir -p "$runtime"

exec 9>"$lease_dir/exp014-scatter-probe.lock"
flock -n 9 || {
  echo "another exp014 Scatter probe launcher is active" >&2
  exit 3
}

if [[ -f $overlay_root/identity.json ]]; then
  python3 "$builder" --flashinfer-root "$repo" --output "$overlay_root" \
    --check-existing > "$runtime/check_overlays.stdout.json"
else
  [[ ! -e $overlay_root ]] || {
    echo "partial probe overlay root exists: $overlay_root" >&2
    exit 4
  }
  python3 "$builder" --flashinfer-root "$repo" --output "$overlay_root" \
    > "$runtime/build_overlays.stdout.json"
fi

assert_idle_and_clocked() {
  local processes observed_uuid observed_clock
  processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid \
    --format=csv,noheader,nounits || true)
  if grep -q "$gpu_uuid" <<<"$processes"; then
    echo "foreign compute process detected on leased GPU" >&2
    return 5
  fi
  IFS=, read -r observed_uuid observed_clock < <(
    nvidia-smi --id="$gpu_uuid" \
      --query-gpu=uuid,clocks.applications.graphics \
      --format=csv,noheader,nounits
  )
  [[ ${observed_uuid// /} == "$gpu_uuid" ]]
  [[ ${observed_clock// /} == "$expected_clock" ]]
}

for arm in "${arms[@]}"; do
  output=$results/raw/scatter_phase_probe/$arm/m$m
  jit=$jit_base/$arm/m$m
  if [[ -f $output/capture.json ]]; then
    [[ -d $jit ]] || {
      echo "completed capture has no retained JIT root: $arm M$m" >&2
      exit 6
    }
    continue
  fi
  [[ ! -e $output ]] || {
    echo "refusing partial probe capture: $output" >&2
    exit 7
  }
  [[ ! -e $jit ]] || {
    echo "fresh probe capture requires absent JIT root: $jit" >&2
    exit 8
  }
  mkdir -p "$jit"
  assert_idle_and_clocked
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
      --arm "$arm" \
      --overlay-root "$overlay_root" \
      --jit-root "$jit" \
      --output "$output" \
      --expected-gpu-uuid "$gpu_uuid" \
      --expected-app-clock-mhz "$expected_clock" \
      --m "$m" \
      > "$runtime/${arm}_m${m}.stdout.json" \
      2> "$runtime/${arm}_m${m}.stderr.log"
done
