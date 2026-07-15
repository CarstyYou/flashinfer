#!/usr/bin/env python3
"""Replace only explicitly rerun M rows while preserving both raw inputs."""

import argparse
import csv
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def read_rows(path: Path):
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial", type=Path, default=RESULTS / "triton_arm_initial.csv"
    )
    parser.add_argument(
        "--rerun", type=Path, default=RESULTS / "triton_arm_m8192_rerun.csv"
    )
    parser.add_argument("--output", type=Path, default=RESULTS / "triton_arm_raw.csv")
    args = parser.parse_args()
    initial = read_rows(args.initial)
    rerun = read_rows(args.rerun)
    replacements = {int(row["m"]): row for row in rerun}
    if not replacements:
        raise ValueError("rerun contains no rows")
    if any(row.get("backend") != "triton_fp8" for row in initial + rerun):
        raise ValueError("unexpected backend in Triton evidence")
    initial_cases = {int(row["m"]) for row in initial}
    if not set(replacements).issubset(initial_cases):
        raise ValueError("rerun contains an unexpected M")
    merged = [replacements.get(int(row["m"]), row) for row in initial]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
