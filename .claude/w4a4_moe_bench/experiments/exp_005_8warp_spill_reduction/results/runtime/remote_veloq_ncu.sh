#!/usr/bin/env bash
set -euo pipefail

host_root=/home/xiy/workspace/flashinfer_exp005_748ad
relative_exp=.claude/w4a4_moe_bench/experiments/exp_005_8warp_spill_reduction
ncu_root="$host_root/$relative_exp/results/ncu"
veloq=/home/xiy/workspace/veloq
image=nvcr.io/nvidia/pytorch:26.05-py3
ncu_report_dir=/opt/nvidia/nsight-compute/2026.1.1/extras/python
uid=$(id -u)
gid=$(id -g)

# NCU runs as container root. Restore ownership only inside this experiment's
# generated NCU evidence tree before creating VeloQ sidecars.
docker run --rm --entrypoint chown \
  -v "$host_root:$host_root" \
  "$image" -R "$uid:$gid" "$ncu_root"

veloq_run() {
  docker run --rm --entrypoint "$veloq" --user "$uid:$gid" \
    -e HOME=/tmp \
    -e VELOQ_NCU_REPORT_DIR="$ncu_report_dir" \
    -e VELOQ_PYTHON=python3 \
    -v "$host_root:$host_root" \
    -v "$veloq:$veloq:ro" \
    "$image" "$@"
}

for arm in baseline_4warp candidate_8warp_serial_v0; do
  capture="$ncu_root/$arm/m8192/canonical_v1"
  report="$capture/trace.ncu-rep"
  output="$capture/veloq"
  mkdir -p "$output"

  veloq_run clean "$report" >/dev/null
  veloq_run info "$report" >"$output/info.json"
  veloq_run ncu summary "$report" >"$output/summary.json"
  veloq_run ncu launches --limit 10 "$report" >"$output/launches.json"
  veloq_run ncu inspect --row-id launch:0 "$report" \
    >"$output/inspect_launch0.json"
  veloq_run ncu disasm --row-id launch:0 "$report" \
    >"$output/disasm_launch0.json"

  python3 - "$output" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for name in (
    "info.json",
    "summary.json",
    "launches.json",
    "inspect_launch0.json",
    "disasm_launch0.json",
):
    payload = json.loads((root / name).read_text())
    if payload.get("schema") != "v1" or "error" in payload or "data" not in payload:
        raise SystemExit("invalid VeloQ response: {}".format(root / name))
launches = json.loads((root / "launches.json").read_text())
rows = launches["data"]["rows"]
if len(rows) != 1 or rows[0].get("row_id") != "launch:0":
    raise SystemExit("expected exactly launch:0 in {}".format(root))
PY
done

printf 'VeloQ canonical_v1 evidence generated and validated.\n'
