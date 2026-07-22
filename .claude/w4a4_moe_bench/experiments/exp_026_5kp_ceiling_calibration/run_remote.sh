#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-xiy@10.6.142.16}
TARGET_UUID=${TARGET_UUID:-GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522}
EXPECTED_HOST=${EXPECTED_HOST:-R6KD-CX8aaS-GPU-16}
IMAGE=${IMAGE:-nvcr.io/nvidia/pytorch:26.05-py3}
REMOTE_BENCH_ROOT=${REMOTE_BENCH_ROOT:-/home/xiy/workspace/exp026_sm120_mma_benchmarks}
REMOTE_EXP_ROOT=${REMOTE_EXP_ROOT:-/home/xiy/workspace/exp026_5kp_ceiling_calibration}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ ${1:-} != "--on-host" ]]; then
    LOCAL_BENCH_ROOT=${LOCAL_BENCH_ROOT:-/home/scratch.xiy_gpu/mega_inference/sm120_mma_benchmarks}
    test -d "$LOCAL_BENCH_ROOT/.git" -o -f "$LOCAL_BENCH_ROOT/.git"
    test -z "$(cd "$LOCAL_BENCH_ROOT" && git status --porcelain)" || {
        echo "benchmark repository must be clean" >&2
        exit 2
    }
    BENCH_COMMIT=$(cd "$LOCAL_BENCH_ROOT" && git rev-parse HEAD)
    BENCH_REMOTE=$(cd "$LOCAL_BENCH_ROOT" && git config --get remote.origin.url)
    BENCH_REMOTE_MAIN=$(git ls-remote "$BENCH_REMOTE" refs/heads/main | awk '{print $1}')
    test "$BENCH_REMOTE_MAIN" = "$BENCH_COMMIT" || {
        echo "benchmark commit is not canonical remote main" >&2
        exit 2
    }

    ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_BENCH_ROOT' '$REMOTE_EXP_ROOT'"
    rsync -a --exclude .git --exclude build/ \
        "$LOCAL_BENCH_ROOT/" "$REMOTE_HOST:$REMOTE_BENCH_ROOT/"
    rsync -a "$SCRIPT_DIR/run_remote.sh" "$REMOTE_HOST:$REMOTE_EXP_ROOT/run_remote.sh"
    ssh "$REMOTE_HOST" \
        "BENCH_COMMIT='$BENCH_COMMIT' BENCH_REMOTE='$BENCH_REMOTE' \
         TARGET_UUID='$TARGET_UUID' EXPECTED_HOST='$EXPECTED_HOST' IMAGE='$IMAGE' \
         REMOTE_BENCH_ROOT='$REMOTE_BENCH_ROOT' REMOTE_EXP_ROOT='$REMOTE_EXP_ROOT' \
         bash '$REMOTE_EXP_ROOT/run_remote.sh' --on-host"

    rm -rf "$SCRIPT_DIR/results/raw"
    mkdir -p "$SCRIPT_DIR/results/raw"
    rsync -a --delete "$REMOTE_HOST:$REMOTE_EXP_ROOT/results/raw/" \
        "$SCRIPT_DIR/results/raw/"
    {
        echo "benchmark_remote=$BENCH_REMOTE"
        echo "remote_ref=refs/heads/main"
        echo "remote_ref_commit=$BENCH_REMOTE_MAIN"
    } > "$SCRIPT_DIR/results/raw/remote_ref_identity.txt"
    exit 0
fi

test "$(hostname)" = "$EXPECTED_HOST"
test -n "${BENCH_COMMIT:-}"
test -n "${BENCH_REMOTE:-}"

RAW="$REMOTE_EXP_ROOT/results/raw"
LEASE="/tmp/kdk-direct-ssh-gpu-leases/${EXPECTED_HOST}_${TARGET_UUID}"
CONTAINER="exp026-5kp-ceiling-$$"
TELEMETRY_PID=

rm -rf "$RAW"
mkdir -p "$RAW" "$RAW/ipc" "$RAW/saturation" "$RAW/compute_window" \
    "$RAW/recovery" "$RAW/dram" "$RAW/telemetry" "$(dirname "$LEASE")"

cleanup() {
    if [[ -n ${TELEMETRY_PID:-} ]]; then
        kill "$TELEMETRY_PID" 2>/dev/null || true
        wait "$TELEMETRY_PID" 2>/dev/null || true
    fi
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    rmdir "$LEASE" 2>/dev/null || true
}
trap cleanup EXIT

if ! mkdir "$LEASE" 2>/dev/null; then
    echo "GPU lease is busy: $LEASE" >&2
    exit 3
fi

if nvidia-smi -i "$TARGET_UUID" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
    echo "target GPU has a foreign compute process" >&2
    exit 4
fi

MEM_USED=$(nvidia-smi -i "$TARGET_UUID" --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d ' ')
if (( MEM_USED > 256 )); then
    echo "target GPU memory is not idle: ${MEM_USED} MiB" >&2
    exit 5
fi

GPU_QUERY=uuid,pci.bus_id,name,pstate,clocks.current.graphics,clocks.current.memory,clocks.applications.graphics,clocks.applications.memory,power.draw,power.limit,temperature.gpu,memory.used,utilization.gpu
nvidia-smi -i "$TARGET_UUID" --query-gpu="$GPU_QUERY" \
    --format=csv > "$RAW/gpu_pre.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv > "$RAW/processes_pre.csv"
docker image inspect "$IMAGE" > "$RAW/container_image.json"
{
    uname -a
    nvidia-smi --version
    nvidia-smi -i "$TARGET_UUID" --query-gpu=driver_version,vbios_version \
        --format=csv
} > "$RAW/host_software.txt"

{
    echo "benchmark_remote=$BENCH_REMOTE"
    echo "benchmark_commit=$BENCH_COMMIT"
    echo "benchmark_status=clean"
    echo "target_uuid=$TARGET_UUID"
    echo "host=$(hostname)"
    sha256sum "$REMOTE_BENCH_ROOT/CMakeLists.txt" \
        "$REMOTE_BENCH_ROOT/IPC_benchmark.cu" \
        "$REMOTE_BENCH_ROOT/IPC_bench_recipe.cuh" \
        "$REMOTE_BENCH_ROOT/back_to_back_throttle_benchmark.cu" \
        "$REMOTE_BENCH_ROOT/dram_bw_benchmark.cu"
} > "$RAW/source_identity.txt"

docker run -d --rm --name "$CONTAINER" --gpus "device=$TARGET_UUID" \
    --cap-add SYS_ADMIN \
    -v "$REMOTE_BENCH_ROOT:/workspace" \
    -v /opt/nvidia/nsight-compute/2025.3.1:/opt/ncu:ro \
    -w /workspace "$IMAGE" sleep infinity >/dev/null

docker exec "$CONTAINER" bash -lc '
    nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader
    python - <<"PY"
import torch
p = torch.cuda.get_device_properties(0)
print("visible_devices={}".format(torch.cuda.device_count()))
print("device_name={}".format(p.name))
print("sm_count={}".format(p.multi_processor_count))
print("compute_capability={}.{}".format(p.major, p.minor))
PY
' > "$RAW/container_gpu_identity.txt" 2>&1
grep -q "$TARGET_UUID" "$RAW/container_gpu_identity.txt"
grep -q 'visible_devices=1' "$RAW/container_gpu_identity.txt"
grep -q 'sm_count=110' "$RAW/container_gpu_identity.txt"

docker exec "$CONTAINER" bash -lc '
    nvcc --version
    cmake --version | head -1
    cmake -S /workspace -B /workspace/build -GNinja -DCMAKE_BUILD_TYPE=Release
    cmake --build /workspace/build --clean-first --target \
      SM120_benchmark SM120_back_to_back_throttle_benchmark \
      SM120_DRAM_bw_benchmark -j2
    git -C /workspace/build/_deps/cutlass-src rev-parse HEAD
' > "$RAW/build.log" 2>&1

docker exec "$CONTAINER" sha256sum \
    /workspace/build/SM120_benchmark \
    /workspace/build/SM120_back_to_back_throttle_benchmark \
    /workspace/build/SM120_DRAM_bw_benchmark > "$RAW/binary_sha256.txt"

docker exec "$CONTAINER" bash -lc '
    for bin in SM120_benchmark SM120_back_to_back_throttle_benchmark; do
      echo "BIN=$bin"
      cuobjdump --dump-sass "/workspace/build/$bin" |
        grep -Eo "(QMMA|OMMA)[^;[:space:]]*" | sort -u
    done
' > "$RAW/sass_instructions.txt" 2>&1
grep -q 'QMMA.16832.F32.E4M3.E4M3' "$RAW/sass_instructions.txt"
grep -q 'OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X' "$RAW/sass_instructions.txt"

docker exec "$CONTAINER" bash -lc '
    cuobjdump --dump-sass /workspace/build/SM120_back_to_back_throttle_benchmark |
      awk '\''/Function :/{fn=$0; emitted=0}
             fn ~ /ILi64000E/ && /QMMA|OMMA/ && !emitted {
                 print fn; print $0; emitted=1
             }'\''
' > "$RAW/sass_mode_bindings.txt" 2>&1
grep -q 'QMMA.16832.F32.E4M3.E4M3' "$RAW/sass_mode_bindings.txt"
grep -q 'QMMA.SF.16832.F32.E4M3.E4M3.E8' "$RAW/sass_mode_bindings.txt"
grep -q 'OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X' "$RAW/sass_mode_bindings.txt"

{
    grep -n -A15 'if (mode == "fp8-e4m3-vs32")' \
        "$REMOTE_BENCH_ROOT/back_to_back_throttle_benchmark.cu"
    grep -n 'case 64000' "$REMOTE_BENCH_ROOT/back_to_back_throttle_benchmark.cu"
} > "$RAW/source_mode_bindings.txt"

run_with_telemetry() {
    local label=$1
    shift
    mkdir -p "$(dirname "$RAW/telemetry/${label}.csv")" \
        "$(dirname "$RAW/${label}.log")"
    nvidia-smi -i "$TARGET_UUID" \
        --query-gpu=timestamp,uuid,pstate,clocks.current.graphics,clocks.current.memory,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used \
        --format=csv -lms 20 > "$RAW/telemetry/${label}.csv" &
    TELEMETRY_PID=$!
    set +e
    docker exec "$CONTAINER" bash -lc "$*" > "$RAW/${label}.log" 2>&1
    local status=$?
    set -e
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
    TELEMETRY_PID=
    return "$status"
}

for rep in 1 2 3 4 5; do
    docker exec "$CONTAINER" /workspace/build/SM120_benchmark \
        > "$RAW/ipc/run_${rep}.log" 2>&1
done

for mode in fp8-e4m3-noscale nvfp4-e2m1-vs16 fp8-e4m3-vs32; do
    for blocks in 4 8 12 16; do
        run_with_telemetry "saturation/${mode}_b${blocks}" \
            "/workspace/build/SM120_back_to_back_throttle_benchmark 8 16000 0 '$mode' '$blocks'"
    done
done

for mode in fp8-e4m3-noscale nvfp4-e2m1-vs16 fp8-e4m3-vs32; do
    for rep in 1 2 3; do
        run_with_telemetry "compute_window/${mode}_run${rep}" \
            "/workspace/build/SM120_back_to_back_throttle_benchmark 32 64000 0 '$mode' 8"
    done
    run_with_telemetry "recovery/${mode}" \
        "/workspace/build/SM120_back_to_back_throttle_benchmark 16 2000 50000 '$mode' 8"
done

for rep in 1 2 3; do
    run_with_telemetry "dram/run_${rep}" \
        "/workspace/build/SM120_DRAM_bw_benchmark"
done

for kernel in stream_read_kernel stream_write_kernel stream_copy_kernel; do
    docker exec "$CONTAINER" bash -lc \
      "/opt/ncu/ncu --target-processes all --kernel-name 'regex:${kernel}' \
       --launch-skip 1 --launch-count 4 \
       --metrics dram__bytes_op_read.sum,dram__bytes_op_write.sum \
       --csv --log-file '/workspace/${kernel}_ncu.csv' \
       /workspace/build/SM120_DRAM_bw_benchmark >/dev/null" \
       > "$RAW/dram/${kernel}_ncu_driver.log" 2>&1
    docker cp "$CONTAINER:/workspace/${kernel}_ncu.csv" \
        "$RAW/dram/${kernel}_ncu.csv"
done

nvidia-smi -i "$TARGET_UUID" --query-gpu="$GPU_QUERY" \
    --format=csv > "$RAW/gpu_post.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv > "$RAW/processes_post.csv"
if nvidia-smi -i "$TARGET_UUID" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
    echo "foreign process appeared on target GPU during capture" >&2
    exit 6
fi

echo "capture_complete benchmark_commit=$BENCH_COMMIT"
