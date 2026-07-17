#!/usr/bin/env bash
set -euo pipefail

host_root=/home/xiy/workspace/flashinfer_exp005_748ad
relative_exp=.claude/w4a4_moe_bench/experiments/exp_005_8warp_spill_reduction
experiment="$host_root/$relative_exp"
results="$experiment/results"
image=nvcr.io/nvidia/pytorch:26.05-py3
uid=$(id -u)
gid=$(id -g)

baseline="$results/ncu/baseline_4warp/m8192/canonical_v1"
candidate="$results/ncu/candidate_8warp_serial_v0/m8192/canonical_v1"

args=(
  "$experiment/build_ncu_evidence.py"
  --results "$results"
  --native-csv "baseline_4warp=$baseline/native_raw.csv"
  --native-csv "candidate_8warp_serial_v0=$candidate/native_raw.csv"
  --veloq-launches "baseline_4warp=$baseline/veloq/launches.json"
  --veloq-launches "candidate_8warp_serial_v0=$candidate/veloq/launches.json"
  --capture-identity "baseline_4warp=$baseline/capture_identity.json"
  --capture-identity "candidate_8warp_serial_v0=$candidate/capture_identity.json"
  --report "baseline_4warp=$baseline/trace.ncu-rep"
  --report "candidate_8warp_serial_v0=$candidate/trace.ncu-rep"
  --preparation "baseline_4warp=$results/raw/baseline_4warp/m8192/canonical/preparation.json"
  --preparation "candidate_8warp_serial_v0=$results/raw/candidate_8warp_serial_v0/m8192/canonical/preparation.json"
  --profile-target "baseline_4warp=$results/profile_targets/baseline_4warp/m8192/target.json"
  --profile-target "candidate_8warp_serial_v0=$results/profile_targets/candidate_8warp_serial_v0/m8192/target.json"
  --static-evidence "$results/static_spill_evidence.json"
  --correctness "$results/correctness/m8192.json"
)

set +e
docker run --rm --entrypoint python3 --user "$uid:$gid" \
  -e HOME=/tmp \
  -v "$host_root:$host_root" \
  "$image" "${args[@]}" \
  >"$results/runtime/ncu_v1_evidence_builder.stdout.log" \
  2>"$results/runtime/ncu_v1_evidence_builder.stderr.log"
builder_status=$?
set -e

# Candidate A is already known to retain a non-zero static spill frame, so a
# fail-closed evidence builder must return 2 while still writing the evidence.
if [[ $builder_status -ne 2 ]]; then
  echo "unexpected NCU evidence builder status: $builder_status" >&2
  sed -n '1,240p' "$results/runtime/ncu_v1_evidence_builder.stderr.log" >&2
  exit 3
fi

test -s "$results/ncu/evidence.json"
test -s "$results/ncu/metrics.csv"
grep -q '"overall_gate_pass": false' "$results/ncu/evidence.json"

printf 'NCU evidence built; expected zero-spill gate failure recorded.\n'
