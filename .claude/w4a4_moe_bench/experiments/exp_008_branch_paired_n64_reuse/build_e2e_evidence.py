#!/usr/bin/env python3
"""Build fail-closed paired-ABBA E2E evidence for exp_008.

External result roots are the arm identity authority.  In particular, v0 and
v1 intentionally share the same exp_005 internal arm name, so this collector
never uses the measurement payload's ``arm`` field to distinguish them.  The
internal arm is still checked as an expected harness contract.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

GPU_UUID = "GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522"
APPLICATIONS_GRAPHICS_CLOCK_MHZ = "2377"
M_VALUES = (256, 8192)
GROUPS = tuple(range(5))
POSITIONS = tuple(range(4))
WARMUP = 5
ITERS = 50
L2_FLUSH_BYTES = 192 << 20
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260719
RATIO_BAND = (0.98, 1.02)

EXTERNAL_ARMS = {
    "production": {
        "internal_arm": "baseline_4warp",
        "overlay_sha256": "94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106",
    },
    "n128": {
        "internal_arm": "candidate_8warp_serial_v0",
        "overlay_sha256": "3cd9e6a26056d9221f59ea6749cd601c25cbef017cf6e7349efe0925180407c1",
    },
    "v0": {
        "internal_arm": "candidate_8warp_n64_temporal_replay_v0",
        "overlay_sha256": "1953cbb7717cda4461a4f199d05f370a4bdb35b4b8ef7556443caf36b0b12ec2",
    },
    "v1": {
        "internal_arm": "candidate_8warp_n64_temporal_replay_v0",
        "overlay_sha256": "f3c246817679d962a3f7160dbe8b9e68262c919e26e306f349200961fc4ac971",
    },
}

# Tuple order is the registered output order and is not inferred from payloads.
PAIR_SPECS = (
    (
        "primary",
        {
            "baseline": "v0",
            "candidate": "v1",
            "external_abba_order": ("v0", "v1", "v1", "v0"),
        },
    ),
    (
        "secondary",
        {
            "baseline": "n128",
            "candidate": "v1",
            "external_abba_order": ("n128", "v1", "v1", "n128"),
        },
    ),
    (
        "production",
        {
            "baseline": "production",
            "candidate": "v1",
            "external_abba_order": ("production", "v1", "v1", "production"),
        },
    ),
)


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


def canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def write_csv(path, rows):
    require(bool(rows), "refusing to write an empty raw CSV")
    fieldnames = list(rows[0])
    require(
        all(set(row) == set(fieldnames) for row in rows),
        "raw CSV rows have inconsistent fields",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def finite_positive(value, label):
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise EvidenceError(
            "{} is not numeric: {!r}".format(label, value)
        ) from error
    require(math.isfinite(converted) and converted > 0.0, "invalid {}: {}".format(label, converted))
    return converted


def exact_int(value, expected, label):
    require(
        isinstance(value, int) and not isinstance(value, bool) and value == expected,
        "{} drift: {!r} != {!r}".format(label, value, expected),
    )


def runtime_gpu(runtime, label):
    require(isinstance(runtime, dict), "{} runtime is not an object".format(label))
    gpu = runtime.get("gpu")
    require(isinstance(gpu, dict), "{} runtime.gpu is not an object".format(label))
    require(
        gpu.get("uuid") == GPU_UUID,
        "{} GPU UUID drift: {!r}".format(label, gpu.get("uuid")),
    )
    require(
        str(gpu.get("applications_graphics_clock_mhz", "")).strip()
        == APPLICATIONS_GRAPHICS_CLOCK_MHZ,
        "{} applications graphics clock drift: {!r}".format(
            label, gpu.get("applications_graphics_clock_mhz")
        ),
    )
    foreign = gpu.get("foreign_processes_before_cuda_context")
    require(foreign == [], "{} recorded foreign GPU processes: {!r}".format(label, foreign))
    return gpu


def runtime_comparison_identity(runtime, label):
    require(isinstance(runtime, dict), "{} runtime is not an object".format(label))
    source = runtime.get("source")
    imports = runtime.get("imports")
    require(isinstance(source, dict), "{} runtime.source is not an object".format(label))
    require(isinstance(imports, dict), "{} runtime.imports is not an object".format(label))
    required_source = (
        "locked_source_commit",
        "checkout_head",
        "cutlass_commit",
        "production_kernel",
        "production_kernel_sha256",
    )
    required_runtime = (
        "image_digest",
        "python_deps_sha256",
        "cuda_runtime",
        "nvcc",
        "ptxas",
        "python",
        "torch",
    )
    required_imports = (
        "flashinfer",
        "cutlass_python",
        "cutlass_python_version",
    )
    for field in required_source:
        require(source.get(field) is not None, "{} missing source.{}".format(label, field))
    for field in required_runtime:
        require(runtime.get(field) is not None, "{} missing runtime.{}".format(label, field))
    for field in required_imports:
        require(imports.get(field) is not None, "{} missing imports.{}".format(label, field))
    return {
        "source": {field: source[field] for field in required_source},
        "runtime": {field: runtime[field] for field in required_runtime},
        "imports": {field: imports[field] for field in required_imports},
    }


def preparation_comparison_identity(canonical, label):
    required = ("case", "fixture", "weights", "reference_sha256")
    for field in required:
        require(canonical.get(field) is not None, "{} missing {}".format(label, field))
    return {
        "case": canonical["case"],
        "fixture": canonical["fixture"],
        "weights": canonical["weights"],
        "reference_sha256": canonical["reference_sha256"],
        "runtime": runtime_comparison_identity(canonical["runtime"], label),
    }


def preparation_relative_path(external_arm, m):
    internal = EXTERNAL_ARMS[external_arm]["internal_arm"]
    return Path("raw") / internal / "m{}".format(m) / "canonical" / "preparation.json"


def validate_preparation(results, pair_name, external_arm, m):
    arm_spec = EXTERNAL_ARMS[external_arm]
    relative = preparation_relative_path(external_arm, m)
    canonical_path = results / "canonical" / external_arm / relative
    pair_path = results / "e2e" / pair_name / external_arm / relative
    canonical_sha = file_sha256(canonical_path) if canonical_path.is_file() else None
    pair_sha = file_sha256(pair_path) if pair_path.is_file() else None
    require(canonical_sha is not None, "missing canonical preparation: {}".format(canonical_path))
    require(pair_sha is not None, "missing pair preparation: {}".format(pair_path))
    require(
        canonical_sha == pair_sha,
        "pair preparation is not the canonical byte-identical copy: {}".format(pair_path),
    )
    canonical = read_json(canonical_path)
    pair_value = read_json(pair_path)
    require(canonical == pair_value, "pair preparation JSON differs from canonical: {}".format(pair_path))
    label = "{}/{}/m{} preparation".format(pair_name, external_arm, m)
    require(canonical.get("schema") == "exp005.arm-preparation.v1", "{} schema drift".format(label))
    require(canonical.get("status") == "complete", "{} is incomplete".format(label))
    require(canonical.get("arm") == arm_spec["internal_arm"], "{} internal arm drift".format(label))
    exact_int(canonical.get("m"), m, "{} M".format(label))
    require(canonical.get("fixture_kind") == "canonical", "{} fixture drift".format(label))
    runtime = canonical.get("runtime")
    runtime_gpu(runtime, label)
    source = runtime.get("source") if isinstance(runtime, dict) else None
    require(isinstance(source, dict), "{} runtime.source is not an object".format(label))
    require(
        source.get("overlay_sha256") == arm_spec["overlay_sha256"],
        "{} overlay SHA drift".format(label),
    )
    jit_hash = canonical.get("jit_artifact_set_sha256")
    require(isinstance(jit_hash, str) and len(jit_hash) == 64, "{} invalid JIT hash".format(label))
    cubins = canonical.get("cubin_sha256")
    require(
        isinstance(cubins, list)
        and bool(cubins)
        and all(isinstance(value, str) and len(value) == 64 for value in cubins),
        "{} invalid cubin identity".format(label),
    )
    return {
        "pair": pair_name,
        "external_arm": external_arm,
        "internal_arm": arm_spec["internal_arm"],
        "m": m,
        "canonical_preparation_path": str(canonical_path.relative_to(results)),
        "pair_preparation_path": str(pair_path.relative_to(results)),
        "canonical_preparation_sha256": canonical_sha,
        "pair_preparation_sha256": pair_sha,
        "overlay_sha256": arm_spec["overlay_sha256"],
        "jit_artifact_set_sha256": jit_hash,
        "cubin_sha256": list(cubins),
        "jit_root": runtime.get("jit_root"),
        "comparison_identity": preparation_comparison_identity(canonical, label),
    }


def expected_internal_order(pair_spec):
    return [
        EXTERNAL_ARMS[external_arm]["internal_arm"]
        for external_arm in pair_spec["external_abba_order"]
    ]


def measurement_path(results, pair_name, external_arm, m, group, position):
    internal = EXTERNAL_ARMS[external_arm]["internal_arm"]
    return (
        results
        / "e2e"
        / pair_name
        / external_arm
        / "raw"
        / "benchmark"
        / "m{}".format(m)
        / "group_{}_position_{}_{}.json".format(group, position, internal)
    )


def validate_measurement(
    results,
    path,
    pair_name,
    pair_spec,
    external_arm,
    m,
    group,
    position,
    preparation,
):
    value = read_json(path)
    label = "{}/m{}/g{}/p{}/{}".format(
        pair_name, m, group, position, external_arm
    )
    arm_spec = EXTERNAL_ARMS[external_arm]
    require(value.get("schema") == "exp005.arm-measurement.v1", "{} schema drift".format(label))
    require(value.get("status") == "complete", "{} is incomplete".format(label))
    require(value.get("arm") == arm_spec["internal_arm"], "{} internal arm drift".format(label))
    exact_int(value.get("m"), m, "{} M".format(label))
    require(value.get("fixture_kind") == "canonical", "{} fixture drift".format(label))
    exact_int(value.get("group"), group, "{} group".format(label))
    exact_int(value.get("position"), position, "{} position".format(label))
    exact_int(value.get("warmup"), WARMUP, "{} warmup".format(label))
    exact_int(value.get("iters"), ITERS, "{} iters".format(label))
    exact_int(
        value.get("l2_flush_bytes"), L2_FLUSH_BYTES, "{} L2 flush".format(label)
    )
    require(
        value.get("declared_clock_policy") == "locked",
        "{} clock policy is not locked".format(label),
    )
    require(
        value.get("timing") == "outer CUDA graph with external CUDA events",
        "{} timing boundary drift".format(label),
    )
    require(
        value.get("order") == expected_internal_order(pair_spec),
        "{} internal ABBA order drift".format(label),
    )
    sample_us = finite_positive(value.get("sample_us"), "{} sample_us".format(label))
    runtime = value.get("runtime")
    gpu = runtime_gpu(runtime, label)
    source = runtime.get("source") if isinstance(runtime, dict) else None
    require(isinstance(source, dict), "{} runtime.source is not an object".format(label))
    require(
        source.get("overlay_sha256") == arm_spec["overlay_sha256"],
        "{} overlay SHA drift".format(label),
    )
    require(
        value.get("jit_artifact_set_sha256")
        == preparation["jit_artifact_set_sha256"],
        "{} JIT artifact hash differs from canonical preparation".format(label),
    )
    require(
        runtime.get("jit_root") == preparation["jit_root"],
        "{} JIT root differs from canonical preparation".format(label),
    )
    require(
        runtime_comparison_identity(runtime, label)
        == preparation["comparison_identity"]["runtime"],
        "{} runtime environment differs from canonical preparation".format(label),
    )
    role = "baseline" if external_arm == pair_spec["baseline"] else "candidate"
    return {
        "pair": pair_name,
        "m": m,
        "group": group,
        "position": position,
        "external_arm": external_arm,
        "role": role,
        "internal_arm": value["arm"],
        "sample_us": sample_us,
        "warmup": value["warmup"],
        "iters": value["iters"],
        "l2_flush_bytes": value["l2_flush_bytes"],
        "declared_clock_policy": value["declared_clock_policy"],
        "gpu_uuid": gpu["uuid"],
        "applications_graphics_clock_mhz": str(
            gpu["applications_graphics_clock_mhz"]
        ),
        "overlay_sha256": source["overlay_sha256"],
        "jit_artifact_set_sha256": value["jit_artifact_set_sha256"],
        "cubin_sha256": json.dumps(preparation["cubin_sha256"], sort_keys=True),
        "canonical_preparation_sha256": preparation[
            "canonical_preparation_sha256"
        ],
        "pair_preparation_sha256": preparation["pair_preparation_sha256"],
        "measurement_sha256": file_sha256(path),
        "measurement_path": str(path.relative_to(results)),
    }


def validate_exact_sample_file_set(results, pair_name, pair_spec, m):
    expected = set()
    for group in GROUPS:
        for position, external_arm in enumerate(pair_spec["external_abba_order"]):
            expected.add(
                measurement_path(
                    results, pair_name, external_arm, m, group, position
                ).resolve()
            )
    observed = set()
    for external_arm in (pair_spec["baseline"], pair_spec["candidate"]):
        directory = (
            results
            / "e2e"
            / pair_name
            / external_arm
            / "raw"
            / "benchmark"
            / "m{}".format(m)
        )
        if directory.is_dir():
            observed.update(path.resolve() for path in directory.glob("*.json"))
    missing = sorted(str(path) for path in expected - observed)
    unexpected = sorted(str(path) for path in observed - expected)
    require(
        not missing and not unexpected,
        "{}/m{} sample file set mismatch; missing={}, unexpected={}".format(
            pair_name, m, missing, unexpected
        ),
    )


def quantile(values, q):
    require(bool(values), "quantile requires non-empty values")
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * q
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def mean(values):
    require(bool(values), "mean requires non-empty values")
    return sum(float(value) for value in values) / len(values)


def arm_statistics(values):
    require(len(values) == 10, "each arm/M requires exactly ten ABBA position samples")
    average = mean(values)
    return {
        "count": len(values),
        "median_us": statistics.median(values),
        "p10_us": quantile(values, 0.10),
        "p90_us": quantile(values, 0.90),
        "mean_us": average,
        "cv": statistics.pstdev(values) / average,
    }


def classify_ratio_ci(ci_low, ci_high):
    if ci_low > RATIO_BAND[1]:
        return "faster"
    if ci_high < RATIO_BAND[0]:
        return "slower"
    if ci_low >= RATIO_BAND[0] and ci_high <= RATIO_BAND[1]:
        return "equivalent"
    return "inconclusive"


def summarize_case(pair_name, pair_spec, m, rows):
    require(len(rows) == 20, "{}/m{} requires exactly 20 samples".format(pair_name, m))
    by_key = {}
    by_arm = {
        pair_spec["baseline"]: [],
        pair_spec["candidate"]: [],
    }
    for row in rows:
        key = (int(row["group"]), int(row["position"]))
        require(key not in by_key, "duplicate sample key {}/m{} {}".format(pair_name, m, key))
        expected_external = pair_spec["external_abba_order"][key[1]]
        require(
            row["external_arm"] == expected_external,
            "external ABBA order drift at {}/m{} {}".format(pair_name, m, key),
        )
        by_key[key] = row
        by_arm[row["external_arm"]].append(float(row["sample_us"]))
    expected_keys = {(group, position) for group in GROUPS for position in POSITIONS}
    require(set(by_key) == expected_keys, "{}/m{} has incomplete ABBA keys".format(pair_name, m))

    groups = []
    for group in GROUPS:
        baseline_samples = [
            float(by_key[(group, position)]["sample_us"])
            for position in POSITIONS
            if pair_spec["external_abba_order"][position] == pair_spec["baseline"]
        ]
        candidate_samples = [
            float(by_key[(group, position)]["sample_us"])
            for position in POSITIONS
            if pair_spec["external_abba_order"][position] == pair_spec["candidate"]
        ]
        require(
            len(baseline_samples) == 2 and len(candidate_samples) == 2,
            "{}/m{}/g{} is not a complete ABBA group".format(pair_name, m, group),
        )
        baseline_us = mean(baseline_samples)
        candidate_us = mean(candidate_samples)
        ratio = baseline_us / candidate_us
        groups.append(
            {
                "group": group,
                "baseline_position_samples_us": baseline_samples,
                "candidate_position_samples_us": candidate_samples,
                "baseline_mean_us": baseline_us,
                "candidate_mean_us": candidate_us,
                "paired_ratio_baseline_over_candidate": ratio,
                "paired_speedup_percent": (ratio - 1.0) * 100.0,
            }
        )

    pair_seed_offset = {
        "primary": 0,
        "secondary": 100000,
        "production": 200000,
    }[pair_name]
    seed = BOOTSTRAP_SEED + pair_seed_offset + m
    rng = random.Random(seed)
    bootstrap_ratios = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
        bootstrap_ratios.append(
            mean([item["baseline_mean_us"] for item in sampled])
            / mean([item["candidate_mean_us"] for item in sampled])
        )
    ci_low = quantile(bootstrap_ratios, 0.025)
    ci_high = quantile(bootstrap_ratios, 0.975)
    baseline_stats = arm_statistics(by_arm[pair_spec["baseline"]])
    candidate_stats = arm_statistics(by_arm[pair_spec["candidate"]])
    aggregate_baseline_us = mean([item["baseline_mean_us"] for item in groups])
    aggregate_candidate_us = mean([item["candidate_mean_us"] for item in groups])
    ratio = aggregate_baseline_us / aggregate_candidate_us
    return {
        "m": m,
        "baseline_external_arm": pair_spec["baseline"],
        "candidate_external_arm": pair_spec["candidate"],
        "groups": groups,
        "arms": {
            pair_spec["baseline"]: baseline_stats,
            pair_spec["candidate"]: candidate_stats,
        },
        "aggregate_ratio_baseline_over_candidate": ratio,
        "aggregate_baseline_us": aggregate_baseline_us,
        "aggregate_candidate_us": aggregate_candidate_us,
        "speedup_percent": (ratio - 1.0) * 100.0,
        "group_bootstrap": {
            "unit": "one complete ABBA group",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": seed,
            "ratio_ci95": [ci_low, ci_high],
            "speedup_percent_ci95": [
                (ci_low - 1.0) * 100.0,
                (ci_high - 1.0) * 100.0,
            ],
        },
        "ratio_band": list(RATIO_BAND),
        "equivalence_band_percent": [-2.0, 2.0],
        "classification": classify_ratio_ci(ci_low, ci_high),
    }


def collect(results):
    raw_rows = []
    preparations = []
    pair_summaries = {}
    for pair_name, pair_spec in PAIR_SPECS:
        pair_cases = {}
        preparation_by_arm_m = {}
        for external_arm in (pair_spec["baseline"], pair_spec["candidate"]):
            for m in M_VALUES:
                identity = validate_preparation(
                    results, pair_name, external_arm, m
                )
                preparation_by_arm_m[(external_arm, m)] = identity
                preparations.append(identity)
        for m in M_VALUES:
            baseline_identity = preparation_by_arm_m[(pair_spec["baseline"], m)][
                "comparison_identity"
            ]
            candidate_identity = preparation_by_arm_m[(pair_spec["candidate"], m)][
                "comparison_identity"
            ]
            require(
                baseline_identity == candidate_identity,
                "{}/m{} cross-arm preparation identity drift".format(pair_name, m),
            )
            validate_exact_sample_file_set(results, pair_name, pair_spec, m)
            case_rows = []
            for group in GROUPS:
                for position, external_arm in enumerate(
                    pair_spec["external_abba_order"]
                ):
                    path = measurement_path(
                        results, pair_name, external_arm, m, group, position
                    )
                    row = validate_measurement(
                        results,
                        path,
                        pair_name,
                        pair_spec,
                        external_arm,
                        m,
                        group,
                        position,
                        preparation_by_arm_m[(external_arm, m)],
                    )
                    case_rows.append(row)
                    raw_rows.append(row)
            pair_cases["m{}".format(m)] = summarize_case(
                pair_name, pair_spec, m, case_rows
            )
        pair_summaries[pair_name] = {
            "baseline_external_arm": pair_spec["baseline"],
            "candidate_external_arm": pair_spec["candidate"],
            "registered_external_abba_order": list(
                pair_spec["external_abba_order"]
            ),
            "cases": pair_cases,
        }
    require(len(raw_rows) == 120, "exp_008 E2E evidence requires exactly 120 samples")
    summary = {
        "schema": "exp008.e2e-paired-abba.v1",
        "gate_pass": True,
        "sample_count": len(raw_rows),
        "identity_authority": (
            "external result root + registered external ABBA order; internal arm is validation-only"
        ),
        "protocol": {
            "m_values": list(M_VALUES),
            "groups": len(GROUPS),
            "positions_per_group": len(POSITIONS),
            "warmup": WARMUP,
            "iters": ITERS,
            "l2_flush_bytes": L2_FLUSH_BYTES,
            "gpu_uuid": GPU_UUID,
            "applications_graphics_clock_mhz": APPLICATIONS_GRAPHICS_CLOCK_MHZ,
            "clock_policy": "locked",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "ratio_band": list(RATIO_BAND),
            "speedup_definition": "(baseline_us / candidate_us - 1) * 100",
        },
        "external_arm_registry": EXTERNAL_ARMS,
        "preparation_identity": preparations,
        "pairs": pair_summaries,
        "evidence_boundary": [
            "Each JSON sample is one independent process position containing 50 timed graph replays; the 50 replays are not treated as independent samples.",
            "Bootstrap resampling unit is one complete ABBA group.",
            "Positive speedup means the registered candidate external arm is faster.",
        ],
    }
    summary["evidence_sha256"] = canonical_sha256(summary)
    return raw_rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument(
        "--raw-csv", type=Path, default=RESULTS / "e2e" / "raw.csv"
    )
    parser.add_argument(
        "--summary", type=Path, default=RESULTS / "e2e" / "summary.json"
    )
    args = parser.parse_args()
    results = args.results.resolve()
    raw_csv = args.raw_csv.resolve()
    summary_path = args.summary.resolve()
    try:
        rows, summary = collect(results)
    except Exception as error:
        failure = {
            "schema": "exp008.e2e-paired-abba.v1",
            "gate_pass": False,
            "error": "{}: {}".format(type(error).__name__, error),
            "results": str(results),
        }
        # Overwrite a potentially stale PASS summary and remove its raw table.
        write_json(summary_path, failure)
        if raw_csv.exists():
            raw_csv.unlink()
        print(json.dumps(failure, sort_keys=True))
        return 2
    write_csv(raw_csv, rows)
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "gate_pass": True,
                "sample_count": len(rows),
                "raw_csv": str(raw_csv),
                "summary": str(summary_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
