#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run_veloq_ncu_remote.sh plan|analyze}
case "$mode" in
  plan|analyze) ;;
  *) echo "unsupported mode: $mode" >&2; exit 2 ;;
esac

host_root=/home/xiy/workspace/flashinfer_exp008_748ad
relative_exp=.claude/w4a4_moe_bench/experiments/exp_008_branch_paired_n64_reuse
ncu_root=$host_root/$relative_exp/results/ncu
veloq=/home/xiy/workspace/veloq
image=nvcr.io/nvidia/pytorch:26.05-py3
ncu_report_dir=/opt/nvidia/nsight-compute/2026.1.1/extras/python
uid=$(id -u)
gid=$(id -g)

test -x "$veloq"

if [[ "$mode" == plan ]]; then
  for arm in n128 v0 v1; do
    report=$ncu_root/$arm/m8192/canonical_v0/trace.ncu-rep
    printf '%s\n' \
      "$veloq info $report" \
      "$veloq ncu summary $report" \
      "$veloq ncu launches $report --limit 10" \
      "$veloq ncu inspect $report --row-id launch:0" \
      "$veloq ncu metrics $report --counter 'sass__inst_executed_register_spilling_op_read'" \
      "$veloq ncu metrics $report --counter 'sass__inst_executed_register_spilling_op_write'" \
      "$veloq ncu metrics $report --counter 'sass__inst_executed_register_spilling_mem_local_op_read'" \
      "$veloq ncu metrics $report --counter 'sass__inst_executed_register_spilling_mem_local_op_write'" \
      "$veloq ncu metrics $report --counter 'sm__inst_executed_pipe_tensor_subpipe_hmma.sum'" \
      "$veloq ncu metrics $report --counter 'sm__ops_path_tensor_src_fp4_dst_fp32.sum'" \
      "$veloq ncu metrics $report --counter 'launch__registers_per_thread'" \
      "$veloq ncu metrics $report --counter 'launch__shared_mem_per_block'" \
      "$veloq ncu disasm $report --row-id launch:0"
  done
  exit 0
fi

docker run --rm --runtime=runc --entrypoint chown \
  -v "$host_root:$host_root" \
  "$image" -R "$uid:$gid" "$ncu_root"

veloq_run() {
  docker run --rm --runtime=runc --entrypoint "$veloq" --user "$uid:$gid" \
    -e HOME=/tmp \
    -e VELOQ_NCU_REPORT_DIR="$ncu_report_dir" \
    -e VELOQ_PYTHON=python3 \
    -v "$host_root:$host_root" \
    -v "$veloq:$veloq:ro" \
    "$image" "$@"
}

for arm in n128 v0 v1; do
  capture=$ncu_root/$arm/m8192/canonical_v0
  report=$capture/trace.ncu-rep
  output=$capture/veloq_api
  test -s "$report"
  test -f "$capture/capture_identity.json"
  test -f "$capture/dynamic_ncu.json"
  mkdir -p "$output"

  # The .veloq sidecar is VeloQ's private cache.  This workflow never opens or
  # parses it; every durable response below comes from the supported CLI API.
  veloq_run clean "$report" >/dev/null
  veloq_run info "$report" > "$output/info.json"
  veloq_run ncu summary "$report" > "$output/summary.json"
  veloq_run ncu launches "$report" --limit 10 > "$output/launches.json"
  veloq_run ncu inspect "$report" --row-id launch:0 > "$output/inspect_launch0.json"
  veloq_run ncu metrics "$report" \
    --counter 'sass__inst_executed_register_spilling_op_read' \
    > "$output/metrics_spill_refill_instructions.json"
  veloq_run ncu metrics "$report" \
    --counter 'sass__inst_executed_register_spilling_op_write' \
    > "$output/metrics_spill_store_instructions.json"
  veloq_run ncu metrics "$report" \
    --counter 'sass__inst_executed_register_spilling_mem_local_op_read' \
    > "$output/metrics_spill_refill_bytes.json"
  veloq_run ncu metrics "$report" \
    --counter 'sass__inst_executed_register_spilling_mem_local_op_write' \
    > "$output/metrics_spill_store_bytes.json"
  veloq_run ncu metrics "$report" \
    --counter 'sm__inst_executed_pipe_tensor_subpipe_hmma.sum' \
    > "$output/metrics_tensor_instructions.json"
  veloq_run ncu metrics "$report" \
    --counter 'sm__ops_path_tensor_src_fp4_dst_fp32.sum' \
    > "$output/metrics_fp4_tensor_ops.json"
  veloq_run ncu metrics "$report" \
    --counter 'launch__registers_per_thread' \
    > "$output/metrics_launch_registers.json"
  veloq_run ncu metrics "$report" \
    --counter 'launch__shared_mem_per_block' \
    > "$output/metrics_launch_shared.json"
  veloq_run ncu disasm "$report" --row-id launch:0 \
    > "$output/disasm_launch0.json"
done

docker run --rm --runtime=runc --user "$uid:$gid" \
  --entrypoint python3 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$host_root:/workspace/source/flashinfer" \
  -w /workspace/source/flashinfer/$relative_exp \
  "$image" build_dynamic_ncu_evidence.py

printf 'VeloQ exp_008 NCU API analysis and cross-arm summary passed.\n'
