#!/usr/bin/env bash
set -euo pipefail

host_root=/home/xiy/workspace/flashinfer_exp007_748ad
relative_exp=.claude/w4a4_moe_bench/experiments/exp_007_native_n64_spill_reduction
ncu_root=$host_root/$relative_exp/results/ncu
veloq=/home/xiy/workspace/veloq
image=nvcr.io/nvidia/pytorch:26.05-py3
ncu_report_dir=/opt/nvidia/nsight-compute/2026.1.1/extras/python
uid=$(id -u)
gid=$(id -g)

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

for arm in anchor candidate; do
  capture=$ncu_root/$arm/m8192/canonical_v0
  report=$capture/trace.ncu-rep
  output=$capture/veloq
  mkdir -p "$output"
  veloq_run clean "$report" >/dev/null
  veloq_run info "$report" > "$output/info.json"
  veloq_run ncu summary "$report" > "$output/summary.json"
  veloq_run ncu launches --limit 10 "$report" > "$output/launches.json"
  veloq_run ncu inspect --row-id launch:0 "$report" > "$output/inspect_launch0.json"
  veloq_run ncu disasm --row-id launch:0 "$report" > "$output/disasm_launch0.json"
done

python3 - "$ncu_root" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = {
    "anchor": "b2bc3c4c229ebee967a6b0d3c5649bc06e3629d46793a19af845665f93683f17",
    "candidate": "a9557634cf3d1bff59ca93739e75a1acd1187707222255fead78e2e6e8a73af9",
}
summary = {"schema": "exp007.veloq-ncu-identity.v1", "arms": {}}
for arm, cubin_sha in expected.items():
    capture = root / arm / "m8192/canonical_v0"
    output = capture / "veloq"
    payloads = {}
    for name in (
        "info.json",
        "summary.json",
        "launches.json",
        "inspect_launch0.json",
        "disasm_launch0.json",
    ):
        path = output / name
        payload = json.loads(path.read_text())
        if payload.get("schema") != "v1" or "error" in payload or "data" not in payload:
            raise SystemExit(f"invalid VeloQ response: {path}")
        payloads[name] = payload
    launches = payloads["launches.json"]["data"]["rows"]
    if len(launches) != 1 or launches[0].get("row_id") != "launch:0":
        raise SystemExit(f"expected exactly launch:0 for {arm}")
    launch = launches[0]
    if launch.get("grid_size") != [1, 1, 110] or launch.get("block_size") != [288, 1, 1]:
        raise SystemExit(f"launch geometry drift for {arm}: {launch}")
    if "MoEDynamicKernel" not in str(launch.get("kernel_demangled")):
        raise SystemExit(f"kernel identity drift for {arm}")
    observed_cubin = payloads["disasm_launch0.json"]["data"]["auxiliary"].get("cubin_sha")
    if observed_cubin != cubin_sha:
        raise SystemExit(
            f"loaded cubin mismatch for {arm}: {observed_cubin} != {cubin_sha}"
        )
    summary["arms"][arm] = {
        "launch_row_id": "launch:0",
        "kernel": launch["kernel_demangled"],
        "grid": launch["grid_size"],
        "block": launch["block_size"],
        "context_id": launch["context_id"],
        "stream_id": launch["stream_id"],
        "device_id": launch["device_id"],
        "loaded_cubin_sha256": observed_cubin,
        "report_sha256": hashlib.sha256((capture / "trace.ncu-rep").read_bytes()).hexdigest(),
    }
(root / "veloq_identity.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

printf 'VeloQ exp_007 NCU identity validation passed.\n'
