#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
runner=$script_dir/run_remote.sh

for m in 256 8192; do
  "$runner" prepare production "$m"
  "$runner" prepare compact "$m"
done

order=(production compact compact production)
for m in 256 8192; do
  for group in 0 1 2; do
    for position in 0 1 2 3; do
      "$runner" measure "${order[$position]}" "$m" "$group" "$position"
    done
  done
done
