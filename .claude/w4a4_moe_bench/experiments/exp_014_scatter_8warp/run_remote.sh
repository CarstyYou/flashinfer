#!/usr/bin/env bash
set -euo pipefail

# Run the exp_014 paired gate on the leased 5KP host.  The first invocation is
# local and only re-enters this file over direct SSH; --on-host performs the
# actual ownership, correctness, and five-group A-B-B-A work.

host=${EXP014_HOST:-xiy@10.6.142.16}
repo=${EXP014_REPO:-/home/xiy/workspace/flashinfer_exp015_748ad}
relative_exp=.claude/w4a4_moe_bench/experiments/exp_014_scatter_8warp
container=${EXP014_CONTAINER:-xiyExp0145kp}
host_python=${EXP014_HOST_PYTHON:-python3}
gpu_uuid=${EXP014_GPU_UUID:-GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
lease_id=${EXP014_LEASE_ID:-exp014-scatter-8warp-20260720}
lease_dir=${EXP014_LEASE_DIR:-/tmp/kdk-direct-ssh-gpu-leases/R6KD-CX8aaS-GPU-16_GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
baseline_jit=${EXP014_BASELINE_JIT:-/home/xiy/workspace/exp014_scatter_8warp_jit/baseline_4warp_scatter}
candidate_jit=${EXP014_CANDIDATE_JIT:-/home/xiy/workspace/exp014_scatter_8warp_jit/candidate_8warp_scatter}

if [[ ${1:-} != --on-host ]]; then
  remote_script=$repo/$relative_exp/run_remote.sh
  remote=(
    env
    EXP014_REPO="$repo"
    EXP014_CONTAINER="$container"
    EXP014_HOST_PYTHON="$host_python"
    EXP014_GPU_UUID="$gpu_uuid"
    EXP014_LEASE_ID="$lease_id"
    EXP014_LEASE_DIR="$lease_dir"
    EXP014_BASELINE_JIT="$baseline_jit"
    EXP014_CANDIDATE_JIT="$candidate_jit"
    bash "$remote_script" --on-host
  )
  printf -v remote_command '%q ' "${remote[@]}"
  exec ssh -T "$host" "$remote_command"
fi

exp=$repo/$relative_exp
results=$exp/results
runtime=$results/runtime
worker=$exp/run_exp014_arm.py
builder=$exp/build_exp014_evidence.py
ownership=$exp/check_scatter_ownership.py
overlay_builder=$exp/build_overlays.py
pythonpath=$repo:/home/xiy/workspace/w4a4_deps_460:/home/xiy/workspace/w4a4_deps_460/nvidia_cutlass_dsl/dsl_packages
expected_clock=2377
order=(baseline_4warp_scatter candidate_8warp_scatter candidate_8warp_scatter baseline_4warp_scatter)
arms=(baseline_4warp_scatter candidate_8warp_scatter)

[[ -f $worker && -f $builder && -f $ownership && -f $overlay_builder ]]
[[ -f $lease_dir/metadata ]]
grep -qx "lease_id=$lease_id" "$lease_dir/metadata"
grep -qx "gpu_uuid=$gpu_uuid" "$lease_dir/metadata"

# The lease metadata is authoritative; flock only prevents two exp_014 launchers
# owned by the same lease from accidentally overlapping.
exec 9>"$lease_dir/exp014-scatter-8warp.lock"
flock -n 9 || {
  echo "another exp_014 launcher is active under this lease" >&2
  exit 3
}

launcher_start_identity=$(ps -p "$$" -o lstart= | sed 's/^ *//')
sed -i '/^launcher_pid=/d; /^launcher_start_identity=/d' "$lease_dir/metadata"
printf 'launcher_pid=%s\nlauncher_start_identity=%s\n' \
  "$$" "$launcher_start_identity" >> "$lease_dir/metadata"
cleanup_launcher_record() {
  if grep -qx "lease_id=$lease_id" "$lease_dir/metadata" \
    && grep -qx "launcher_pid=$$" "$lease_dir/metadata"; then
    sed -i '/^launcher_pid=/d; /^launcher_start_identity=/d' "$lease_dir/metadata"
    printf 'launcher_pid=pending\nlauncher_start_identity=pending\n' >> "$lease_dir/metadata"
  fi
}
trap cleanup_launcher_record EXIT

mkdir -p "$runtime"
"$host_python" "$overlay_builder" > "$runtime/build_overlays.stdout.json"
"$host_python" "$ownership" > "$results/ownership_gate.json"

assert_idle_and_clocked() {
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
    | grep -qx "$gpu_uuid"; then
    echo "foreign compute process detected on leased GPU" >&2
    return 4
  fi
  local observed_uuid observed_clock
  IFS=, read -r observed_uuid observed_clock < <(
    nvidia-smi --id="$gpu_uuid" \
      --query-gpu=uuid,clocks.applications.graphics \
      --format=csv,noheader,nounits
  )
  [[ ${observed_uuid// /} == "$gpu_uuid" ]]
  [[ ${observed_clock// /} == "$expected_clock" ]]
}

overlay_for() {
  printf '%s/results/overlays/%s/moe_dynamic_kernel.py\n' "$exp" "$1"
}

jit_for() {
  case "$1" in
    baseline_4warp_scatter) printf '%s\n' "$baseline_jit" ;;
    candidate_8warp_scatter) printf '%s\n' "$candidate_jit" ;;
    *) return 2 ;;
  esac
}

run_worker() {
  local arm=$1
  local jit=$2
  shift 2
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
      --overlay "$(overlay_for "$arm")" \
      --jit-root "$jit" \
      --expected-gpu-uuid "$gpu_uuid" \
      "$@"
}

for arm in "${arms[@]}"; do
  validation=$results/raw/validation/$arm/validation.json
  jit=$(jit_for "$arm")
  if [[ -f $validation ]]; then
    [[ -d $jit ]] || {
      echo "validated arm has no registered JIT root: $arm ($jit)" >&2
      exit 5
    }
    continue
  fi
  [[ ! -e $results/raw/validation/$arm ]] || {
    echo "refusing partial validation evidence: $arm" >&2
    exit 6
  }
  [[ ! -e $jit ]] || {
    echo "fresh validation requires an absent JIT root: $jit" >&2
    exit 7
  }
  mkdir -p "$jit"
  assert_idle_and_clocked
  run_worker "$arm" "$jit" validate \
    > "$runtime/validate_${arm}.stdout.json" \
    2> "$runtime/validate_${arm}.stderr.log"
done

# This stage checks both arm identities, all correctness cases, input parity,
# cubin/JIT registration, and ownership before timing is allowed to begin.
"$host_python" "$builder" --results "$results" --stage validation

manifest_field() {
  local arm=$1 field=$2
  "$host_python" - "$results/raw/validation/$arm/validation.json" "$field" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
field = sys.argv[2]
if field == "artifact":
    answer = value["jit_artifact_set_sha256"]
elif field == "cubin":
    cubins = value["cubin_sha256"]
    if len(cubins) != 1:
        raise SystemExit(f"expected exactly one cubin, got {cubins}")
    answer = cubins[0]
else:
    raise SystemExit(f"unknown manifest field: {field}")
if not isinstance(answer, str) or len(answer) != 64:
    raise SystemExit(f"invalid {field}: {answer!r}")
print(answer)
PY
}

for m in 256 512 1024 2048 4096 8192; do
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
      exit 8
    fi

    for position in 0 1 2 3; do
      arm=${order[$position]}
      jit=$(jit_for "$arm")
      artifact_set=$(manifest_field "$arm" artifact)
      cubin=$(manifest_field "$arm" cubin)
      assert_idle_and_clocked
      echo "M=$m group=$group position=$position arm=$arm"
      run_worker "$arm" "$jit" measure \
        --m "$m" \
        --group "$group" \
        --position "$position" \
        --expected-app-clock-mhz "$expected_clock" \
        --expected-jit-artifact-set-sha256 "$artifact_set" \
        --expected-cubin-sha256 "$cubin" \
        > "$runtime/abba_m${m}_g${group}_p${position}_${arm}.stdout.json" \
        2> "$runtime/abba_m${m}_g${group}_p${position}_${arm}.stderr.log"
    done
  done
done

"$host_python" "$builder" --results "$results" --stage performance
