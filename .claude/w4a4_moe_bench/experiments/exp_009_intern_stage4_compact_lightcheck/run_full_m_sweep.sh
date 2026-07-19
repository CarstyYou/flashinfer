#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
runner=$script_dir/run_full_m_sweep_remote.sh
sweep_root=$script_dir/results/full_m_sweep
status_root=$sweep_root/status
m_values=(256 512 1024 2048 4096 8192)
arms=(production intern exp008)
mkdir -p "$status_root"

internal_arm() {
  case "$1" in
    production) echo baseline_4warp ;;
    intern) echo candidate_4warp_stage4_compact ;;
    exp008) echo candidate_8warp_n64_temporal_replay_v0 ;;
    *) return 2 ;;
  esac
}

prepare_path() {
  local arm=$1 m=$2 internal
  internal=$(internal_arm "$arm")
  printf '%s/canonical/%s/raw/%s/m%s/canonical/preparation.json\n' \
    "$sweep_root" "$arm" "$internal" "$m"
}

failure_path() {
  local arm=$1 m=$2 internal
  internal=$(internal_arm "$arm")
  printf '%s/canonical/%s/raw/%s/m%s/canonical/failure.json\n' \
    "$sweep_root" "$arm" "$internal" "$m"
}

ensure_prepare() {
  local arm=$1 m=$2 preparation failure status rc
  preparation=$(prepare_path "$arm" "$m")
  failure=$(failure_path "$arm" "$m")
  status=$status_root/prepare_${arm}_m${m}.status
  if [[ -f "$preparation" ]]; then
    grep -q '"status": "complete"' "$preparation" || {
      echo "incomplete immutable preparation: $preparation" >&2
      return 4
    }
    printf 'status=passed\narm=%s\nm=%s\npreparation=%s\n' \
      "$arm" "$m" "$preparation" > "$status"
    echo "skip complete preparation arm=$arm M=$m"
    return 0
  fi
  if [[ -f "$failure" ]]; then
    grep -q '"status": "failed"' "$failure" || {
      echo "malformed immutable failure evidence: $failure" >&2
      return 4
    }
    printf 'status=failed\narm=%s\nm=%s\nfailure=%s\n' \
      "$arm" "$m" "$failure" > "$status"
    echo "skip known failed preparation arm=$arm M=$m"
    return 1
  fi
  echo "prepare arm=$arm M=$m"
  set +e
  "$runner" prepare "$arm" "$m"
  rc=$?
  set -e
  if [[ $rc -eq 0 && -f "$preparation" ]]; then
    printf 'status=passed\narm=%s\nm=%s\npreparation=%s\n' \
      "$arm" "$m" "$preparation" > "$status"
    return 0
  fi
  printf 'status=failed\narm=%s\nm=%s\nexit_code=%s\nfailure=%s\n' \
    "$arm" "$m" "$rc" "$failure" > "$status"
  if [[ ! -f "$failure" ]]; then
    echo "prepare failed without structured failure evidence: arm=$arm M=$m rc=$rc" >&2
  else
    echo "correctness failed; retaining evidence and skipping benchmark: arm=$arm M=$m" >&2
  fi
  return 1
}

ensure_intern_diagnosis() {
  local m=$1 diagnostic
  diagnostic=$sweep_root/diagnostics/intern/m${m}.json
  if [[ -f "$diagnostic" ]]; then
    grep -q '"status": "complete"' "$diagnostic" || {
      echo "incomplete immutable Intern diagnostic: $diagnostic" >&2
      return 4
    }
    echo "skip complete Intern diagnostic M=$m"
    return 0
  fi
  echo "diagnose failed Intern correctness M=$m"
  "$runner" diagnose intern "$m"
  [[ -f "$diagnostic" ]] && grep -q '"status": "complete"' "$diagnostic" || {
    echo "Intern failure diagnosis did not complete: M=$m" >&2
    return 4
  }
}

sample_path() {
  local pair=$1 arm=$2 m=$3 position=$4 internal
  internal=$(internal_arm "$arm")
  printf '%s/bench/%s/%s/raw/benchmark/m%s/group_0_position_%s_%s.json\n' \
    "$sweep_root" "$pair" "$arm" "$m" "$position" "$internal"
}

run_sample() {
  local pair=$1 arm=$2 m=$3 position=$4 path
  path=$(sample_path "$pair" "$arm" "$m" "$position")
  if [[ -f "$path" ]]; then
    grep -q '"status": "complete"' "$path" || {
      echo "incomplete immutable sample: $path" >&2
      return 4
    }
    echo "skip complete $pair M=$m position=$position arm=$arm"
    return 0
  fi
  echo "run $pair M=$m position=$position arm=$arm"
  "$runner" measure "$pair" "$arm" "$m" 0 "$position"
}

declare -A correctness
for m in "${m_values[@]}"; do
  for arm in "${arms[@]}"; do
    if ensure_prepare "$arm" "$m"; then
      correctness["$arm:$m"]=pass
    else
      correctness["$arm:$m"]=fail
      if [[ "$arm" == intern ]]; then
        # A known invalid point is a valid sweep outcome only if its numerical,
        # sentinel and workspace failure mode was captured successfully.
        ensure_intern_diagnosis "$m"
      fi
    fi
  done
done

for m in "${m_values[@]}"; do
  production_ok=${correctness["production:$m"]}
  intern_ok=${correctness["intern:$m"]}
  exp008_ok=${correctness["exp008:$m"]}
  [[ "$production_ok" == pass && "$exp008_ok" == pass ]] || {
    echo "Production/exp_008 correctness unexpectedly failed at M=$m" >&2
    exit 4
  }
  if [[ "$intern_ok" == pass ]]; then
    run_sample production_intern production "$m" 0
    sleep "${EXP009_COOLDOWN_SECONDS:-2}"
    run_sample production_intern intern "$m" 1
  else
    run_sample production_exp008 production "$m" 0
  fi
  sleep "${EXP009_COOLDOWN_SECONDS:-2}"
  run_sample production_exp008 exp008 "$m" 1
done

python3 "$script_dir/summarize_full_m_sweep.py" \
  --results "$sweep_root" \
  --summary "$sweep_root/summary.json"
