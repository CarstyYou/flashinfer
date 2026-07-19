#!/usr/bin/env python3
"""Validate and summarize the correctness-gated quick six-M benchmark."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results" / "full_m_sweep"
M_VALUES = (256, 512, 1024, 2048, 4096, 8192)
GPU_UUID = "GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522"
GPU_CLOCK_MHZ = "2377"
WARMUP = 5
ITERS = 50
L2_FLUSH_BYTES = 192 << 20
ARMS = {
    "production": "baseline_4warp",
    "intern": "candidate_4warp_stage4_compact",
    "exp008": "candidate_8warp_n64_temporal_replay_v0",
}


class EvidenceError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def read_json(path):
    require(path.is_file(), "missing JSON: {}".format(path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise EvidenceError("invalid JSON {}: {}".format(path, error)) from error
    require(isinstance(value, dict), "expected JSON object: {}".format(path))
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_positive(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise EvidenceError("{} is not numeric: {!r}".format(label, value)) from error
    require(
        math.isfinite(number) and number > 0.0, "invalid {}: {}".format(label, number)
    )
    return number


def canonical_dir(results, arm, m):
    return (
        results / "canonical" / arm / "raw" / ARMS[arm] / "m{}".format(m) / "canonical"
    )


def correctness(results, arm, m):
    directory = canonical_dir(results, arm, m)
    preparation = directory / "preparation.json"
    failure = directory / "failure.json"
    require(
        preparation.is_file() != failure.is_file(),
        "{}/M{} requires exactly one preparation or failure".format(arm, m),
    )
    if preparation.is_file():
        value = read_json(preparation)
        require(value.get("status") == "complete", "incomplete preparation")
        require(
            value.get("arm") == ARMS[arm] and value.get("m") == m,
            "preparation identity drift",
        )
        outputs = value.get("outputs")
        require(isinstance(outputs, list) and outputs, "missing correctness replays")
        for output in outputs:
            require(output.get("formal_pass") is True, "correctness replay failed")
            require(output.get("sentinel_nan_remaining") == 0, "sentinel replay failed")
        return {
            "status": "passed",
            "path": str(preparation.relative_to(results)),
            "sha256": file_sha256(preparation),
        }

    value = read_json(failure)
    require(arm == "intern", "only Intern may be invalid")
    require(value.get("status") == "failed", "invalid failure evidence")
    require(
        value.get("arm") == ARMS[arm] and value.get("m") == m, "failure identity drift"
    )
    diagnostic_path = results / "diagnostics" / "intern" / "m{}.json".format(m)
    diagnostic = read_json(diagnostic_path)
    require(diagnostic.get("status") == "complete", "diagnostic incomplete")
    require(
        diagnostic.get("m") == m and diagnostic.get("replay_count") == 3,
        "diagnostic identity drift",
    )
    replays = diagnostic.get("replays")
    require(isinstance(replays, list) and len(replays) == 3, "diagnostic replay drift")
    require(
        all(item.get("gates", {}).get("formal_pass") is False for item in replays),
        "failure not reproduced",
    )
    require(
        all(item.get("gates", {}).get("sentinel_pass") is True for item in replays),
        "non-finite failure",
    )
    require(
        all(item.get("gates", {}).get("workspace_pass") is True for item in replays),
        "workspace failure",
    )
    return {
        "status": "invalid",
        "classification": diagnostic.get("observed_classification"),
        "zero_rows": [int(item["zero_rows"]) for item in replays],
        "failure_path": str(failure.relative_to(results)),
        "failure_sha256": file_sha256(failure),
        "diagnostic_path": str(diagnostic_path.relative_to(results)),
        "diagnostic_sha256": file_sha256(diagnostic_path),
    }


def measurement_path(results, pair, arm, m, position):
    return (
        results
        / "bench"
        / pair
        / arm
        / "raw"
        / "benchmark"
        / "m{}".format(m)
        / "group_0_position_{}_{}.json".format(position, ARMS[arm])
    )


def measurement(results, pair, arm, m, position):
    path = measurement_path(results, pair, arm, m, position)
    value = read_json(path)
    label = "{}/{}/M{}".format(pair, arm, m)
    require(
        value.get("schema") == "exp005.arm-measurement.v1",
        "{} schema drift".format(label),
    )
    require(value.get("status") == "complete", "{} incomplete".format(label))
    require(
        value.get("arm") == ARMS[arm] and value.get("m") == m,
        "{} identity drift".format(label),
    )
    require(
        value.get("group") == 0 and value.get("position") == position,
        "{} position drift".format(label),
    )
    require(
        value.get("warmup") == WARMUP and value.get("iters") == ITERS,
        "{} iteration drift".format(label),
    )
    require(
        value.get("l2_flush_bytes") == L2_FLUSH_BYTES, "{} L2 flush drift".format(label)
    )
    require(
        value.get("timing") == "outer CUDA graph with external CUDA events",
        "{} timing drift".format(label),
    )
    gpu = value.get("runtime", {}).get("gpu", {})
    require(gpu.get("uuid") == GPU_UUID, "{} GPU drift".format(label))
    require(
        str(gpu.get("applications_graphics_clock_mhz")) == GPU_CLOCK_MHZ,
        "{} clock drift".format(label),
    )
    return {
        "latency_us": finite_positive(
            value.get("sample_us"), "{} latency".format(label)
        ),
        "path": str(path.relative_to(results)),
        "sha256": file_sha256(path),
    }


def expected_measurements(results, intern_valid):
    paths = set()
    for m in M_VALUES:
        if intern_valid[m]:
            paths.add(
                measurement_path(
                    results, "production_intern", "production", m, 0
                ).resolve()
            )
            paths.add(
                measurement_path(results, "production_intern", "intern", m, 1).resolve()
            )
        else:
            paths.add(
                measurement_path(
                    results, "production_exp008", "production", m, 0
                ).resolve()
            )
        paths.add(
            measurement_path(results, "production_exp008", "exp008", m, 1).resolve()
        )
    return paths


def collect(results):
    correctness_by_m = {}
    intern_valid = {}
    for m in M_VALUES:
        correctness_by_m[m] = {arm: correctness(results, arm, m) for arm in ARMS}
        intern_valid[m] = correctness_by_m[m]["intern"]["status"] == "passed"

    observed = {
        path.resolve() for path in (results / "bench").glob("**/benchmark/m*/*.json")
    }
    expected = expected_measurements(results, intern_valid)
    require(
        observed == expected,
        "selected benchmark file set drift; missing={}, unexpected={}".format(
            sorted(str(path) for path in expected - observed),
            sorted(str(path) for path in observed - expected),
        ),
    )

    rows = []
    for m in M_VALUES:
        if intern_valid[m]:
            production = measurement(results, "production_intern", "production", m, 0)
            intern = measurement(results, "production_intern", "intern", m, 1)
        else:
            production = measurement(results, "production_exp008", "production", m, 0)
            intern = None
        exp008 = measurement(results, "production_exp008", "exp008", m, 1)
        production_us = production["latency_us"]
        rows.append(
            {
                "m": m,
                "correctness": correctness_by_m[m],
                "production": production,
                "intern": intern,
                "exp008": exp008,
                "intern_speedup_percent": (
                    None
                    if intern is None
                    else (production_us / intern["latency_us"] - 1.0) * 100.0
                ),
                "intern_speedup_vs_exp008_percent": (
                    None
                    if intern is None
                    else (exp008["latency_us"] / intern["latency_us"] - 1.0) * 100.0
                ),
                "exp008_speedup_percent": (production_us / exp008["latency_us"] - 1.0)
                * 100.0,
            }
        )
    return {
        "schema": "exp009.quick-full-m-sweep.v1",
        "gate_pass": True,
        "rows": rows,
        "protocol": {
            "samples_per_arm_m": 1,
            "sample_aggregation": "50 timed CUDA Graph replays",
            "warmup": WARMUP,
            "iters": ITERS,
            "l2_flush_bytes": L2_FLUSH_BYTES,
            "timing": "outer CUDA graph with external CUDA events",
            "invalid_policy": "failed correctness is retained and never benchmarked",
        },
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--summary", type=Path, default=DEFAULT_RESULTS / "summary.json"
    )
    args = parser.parse_args()
    try:
        summary = collect(args.results.resolve())
    except Exception as error:
        summary = {
            "schema": "exp009.quick-full-m-sweep.v1",
            "gate_pass": False,
            "error": "{}: {}".format(type(error).__name__, error),
        }
        write_json(args.summary.resolve(), summary)
        print(json.dumps(summary, sort_keys=True))
        return 2
    write_json(args.summary.resolve(), summary)
    print(
        json.dumps(
            {"gate_pass": True, "row_count": len(summary["rows"])}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
