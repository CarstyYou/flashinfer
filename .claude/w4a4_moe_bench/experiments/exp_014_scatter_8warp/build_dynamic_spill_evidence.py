#!/usr/bin/env python3
"""Build the fail-closed exp_014 M8192 dynamic spill summary.

The candidate must have a fresh native-NCU capture under exp_014.  The
4-effective-warp baseline may either have its own exp_014 capture or reuse the
already accepted exp_015 ``candidate_v2`` capture.  Reuse is legal only when
the source, cubin, JIT artifact set, GPU, launch shape, fixture, and work
counters all match the frozen identities below.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"
EXP015_ROOT = ROOT.parent / "exp_015_phase_skeleton_refactor"
sys.path.insert(0, str(EXP015_ROOT))

import build_matched_dynamic_ncu_evidence as exp015  # noqa: E402


BASELINE = "baseline_4warp_scatter"
CANDIDATE = "candidate_8warp_scatter"
ARMS = (BASELINE, CANDIDATE)
M = 8192
FIXTURE = "canonical"

EXPECTED_GPU_UUID = exp015.EXPECTED_GPU_UUID
EXPECTED_APPLICATION_CLOCK_MHZ = exp015.EXPECTED_APPLICATION_CLOCK_MHZ
EXPECTED_GRID = exp015.EXPECTED_GRID
EXPECTED_BLOCK = exp015.EXPECTED_BLOCK
EXPECTED_WORK = exp015.EXPECTED_WORK
METRIC_IDS = exp015.METRIC_IDS
EXPECTED_UNITS = exp015.EXPECTED_UNITS
SPILL_METRICS = exp015.SPILL_METRICS
WORK_METRICS = exp015.WORK_METRICS

EXPECTED_SOURCE_SHA256 = {
    BASELINE: "b6e141179794561f2144bdec079b7e109fddfccc2db6ba0f19c22d30ea4b34ca",
    CANDIDATE: "c88cef63492b60c0a77484b50f6400b83a103d168e1535b78972341503810184",
}

# exp_014 baseline is byte-identical to the accepted exp_015 candidate_v2.
# Keep the reuse contract explicit so a later rebuild cannot silently inherit
# old zero-spill evidence.
EXP015_REUSE_ARM = exp015.CANDIDATE
EXP015_REUSE_SOURCE_SHA256 = EXPECTED_SOURCE_SHA256[BASELINE]
EXP015_REUSE_CUBIN_SHA256 = (
    "fee96b35d9b2c83e354504774fba2e2bc10e54f0316ade18f8adbdabb2ecbada"
)
EXP015_REUSE_JIT_ARTIFACT_SET_SHA256 = (
    "4358e71894a2602030e32e33b13298a3b739ff4c554aaead82b4ef9c0373d3cc"
)

EvidenceError = exp015.EvidenceError
require = exp015.require
read_json = exp015.read_json
file_sha256 = exp015.file_sha256
write_json = exp015.write_json
parse_native_raw = exp015.parse_native_raw


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def evidence_path(path: Path, results: Path) -> str:
    try:
        return str(path.resolve().relative_to(results.resolve()))
    except ValueError:
        return str(path.resolve())


def validation_path(results: Path, arm: str) -> Path:
    return results / "raw" / "validation" / arm / "validation.json"


def capture_root(results: Path, arm: str) -> Path:
    return results / "ncu" / arm


def validated_arm_identity(results: Path, arm: str) -> dict[str, Any]:
    path = validation_path(results, arm)
    value = read_json(path)
    runtime = value.get("runtime")
    require(isinstance(runtime, Mapping), f"{arm} validation runtime is missing")
    source = runtime.get("source")
    gpu = runtime.get("gpu")
    require(isinstance(source, Mapping), f"{arm} validation source is missing")
    require(isinstance(gpu, Mapping), f"{arm} validation GPU is missing")
    cubins = value.get("cubin_sha256")
    artifact_set = value.get("jit_artifact_set_sha256")
    checks = {
        "schema": value.get("schema") == "exp014.arm-validation.v1",
        "status": value.get("status") == "complete",
        "gate_pass": value.get("gate_pass") is True,
        "arm": value.get("arm") == arm,
        "source": source.get("overlay_sha256") == EXPECTED_SOURCE_SHA256[arm],
        "one_cubin": isinstance(cubins, list)
        and len(cubins) == 1
        and is_sha256(cubins[0]),
        "artifact_set": is_sha256(artifact_set),
        "gpu": gpu.get("uuid") == EXPECTED_GPU_UUID,
        "clock": int(float(gpu.get("applications_graphics_clock_mhz", -1)))
        == EXPECTED_APPLICATION_CLOCK_MHZ,
    }
    require(all(checks.values()), f"{arm} validation identity drift: {checks}")
    overlay = results / "overlays" / arm / "moe_dynamic_kernel.py"
    require(
        file_sha256(overlay) == EXPECTED_SOURCE_SHA256[arm],
        f"{arm} frozen overlay drift",
    )
    return {
        "path": path,
        "sha256": file_sha256(path),
        "source_sha256": EXPECTED_SOURCE_SHA256[arm],
        "cubin_sha256": cubins[0],
        "jit_artifact_set_sha256": artifact_set,
        "overlay": overlay,
    }


def validate_capture(results: Path, arm: str) -> dict[str, Any]:
    prerequisite = validated_arm_identity(results, arm)
    root = capture_root(results, arm)
    native = root / "native_raw.csv"
    report = root / "trace.ncu-rep"
    target_path = root / "profile_target.json"
    identity_path = root / "capture_identity.json"
    metrics, observed = parse_native_raw(native)
    target = read_json(target_path)
    identity = read_json(identity_path)

    checks = {
        "identity_schema": identity.get("schema")
        == "exp014.dynamic-spill-capture-identity.v1",
        "identity_arm": identity.get("arm") == arm,
        "identity_m": identity.get("m") == M,
        "identity_fixture": identity.get("fixture") == FIXTURE,
        "identity_source": identity.get("source_sha256")
        == prerequisite["source_sha256"],
        "identity_cubin": identity.get("cubin_sha256") == prerequisite["cubin_sha256"],
        "identity_artifacts": identity.get("jit_artifact_set_sha256")
        == prerequisite["jit_artifact_set_sha256"],
        "identity_gpu": identity.get("gpu_uuid") == EXPECTED_GPU_UUID,
        "identity_clock": identity.get("expected_application_graphics_clock_mhz")
        == EXPECTED_APPLICATION_CLOCK_MHZ,
        "identity_grid": identity.get("expected_grid") == EXPECTED_GRID,
        "identity_block": identity.get("expected_block") == EXPECTED_BLOCK,
        "identity_metrics": identity.get("required_metric_ids")
        == list(METRIC_IDS.values()),
        "identity_observed": identity.get("observed_launch") == observed,
        "identity_values": identity.get("metrics") == metrics,
        "report_hash": identity.get("trace_sha256") == file_sha256(report),
        "csv_hash": identity.get("native_raw_sha256") == file_sha256(native),
        "target_hash": identity.get("profile_target_sha256")
        == file_sha256(target_path),
        "validation_hash": identity.get("validation_manifest_sha256")
        == prerequisite["sha256"],
        "observed_grid": observed["grid"] == EXPECTED_GRID,
        "observed_block": observed["block"] == EXPECTED_BLOCK,
    }
    require(all(checks.values()), f"{arm} capture identity drift: {checks}")

    runtime = target.get("runtime")
    require(isinstance(runtime, Mapping), f"{arm} profile runtime is missing")
    target_gpu = runtime.get("gpu")
    target_source = runtime.get("source")
    require(isinstance(target_gpu, Mapping), f"{arm} target GPU is missing")
    require(isinstance(target_source, Mapping), f"{arm} target source is missing")
    target_checks = {
        "schema": target.get("schema") == "exp014.dynamic-spill-profile-target.v1",
        "complete": target.get("status") == "complete",
        "arm": target.get("arm") == arm,
        "m": target.get("m") == M,
        "fixture": target.get("fixture_kind") == FIXTURE,
        "source": target.get("source_sha256") == prerequisite["source_sha256"],
        "cubin": target.get("cubin_sha256") == prerequisite["cubin_sha256"],
        "artifacts": target.get("jit_artifact_set_sha256")
        == prerequisite["jit_artifact_set_sha256"],
        "gpu": target.get("gpu_uuid") == EXPECTED_GPU_UUID,
        "launch": target.get("expected_launch")
        == {
            "grid": EXPECTED_GRID,
            "block": EXPECTED_BLOCK,
            "kernel": "MoEDynamicKernel",
        },
        "runtime_gpu": target_gpu.get("uuid") == EXPECTED_GPU_UUID,
        "runtime_clock": int(
            float(target_gpu.get("applications_graphics_clock_mhz", -1))
        )
        == EXPECTED_APPLICATION_CLOCK_MHZ,
        "runtime_source": target_source.get("overlay_sha256")
        == prerequisite["source_sha256"],
    }
    require(all(target_checks.values()), f"{arm} profile-target drift: {target_checks}")

    return {
        "arm": arm,
        "m": M,
        "fixture": FIXTURE,
        "source_sha256": prerequisite["source_sha256"],
        "cubin_sha256": prerequisite["cubin_sha256"],
        "jit_artifact_set_sha256": prerequisite["jit_artifact_set_sha256"],
        "gpu_uuid": EXPECTED_GPU_UUID,
        "observed_launch": observed,
        "metrics": metrics,
        "provenance": "fresh exp_014 native-NCU graph-node capture",
        "artifacts": {
            "capture_identity": evidence_path(identity_path, results),
            "profile_target": evidence_path(target_path, results),
            "ncu_report": evidence_path(report, results),
            "native_raw": evidence_path(native, results),
            "validation": evidence_path(prerequisite["path"], results),
        },
    }


def reuse_exp015_baseline(exp015_results: Path) -> dict[str, Any]:
    record = exp015.validate_capture(exp015_results, EXP015_REUSE_ARM)
    checks = {
        "source": record.get("source_sha256") == EXP015_REUSE_SOURCE_SHA256,
        "cubin": record.get("cubin_sha256") == EXP015_REUSE_CUBIN_SHA256,
        "artifacts": record.get("jit_artifact_set_sha256")
        == EXP015_REUSE_JIT_ARTIFACT_SET_SHA256,
        "gpu": record.get("gpu_uuid") == EXPECTED_GPU_UUID,
        "m": record.get("m") == M,
        "fixture": record.get("fixture") == FIXTURE,
        "grid": record.get("observed_launch", {}).get("grid") == EXPECTED_GRID,
        "block": record.get("observed_launch", {}).get("block") == EXPECTED_BLOCK,
        "metrics": set(record.get("metrics", {})) == set(METRIC_IDS),
    }
    require(all(checks.values()), f"exp_015 baseline reuse rejected: {checks}")
    return {
        **record,
        "arm": BASELINE,
        "provenance": (
            "reused exact exp_015 candidate_v2 capture; source/cubin/JIT/GPU/"
            "launch/work identity revalidated"
        ),
        "reused_from": {
            "experiment": "exp_015_phase_skeleton_refactor",
            "arm": EXP015_REUSE_ARM,
            "results": str(exp015_results.resolve()),
        },
    }


def build_evidence(results: Path, exp015_results: Path) -> dict[str, Any]:
    candidate = validate_capture(results, CANDIDATE)
    baseline_root = capture_root(results, BASELINE)
    if baseline_root.exists():
        baseline = validate_capture(results, BASELINE)
        baseline_mode = "fresh_exp014_capture"
    else:
        exp014_baseline = validated_arm_identity(results, BASELINE)
        require(
            exp014_baseline["cubin_sha256"] == EXP015_REUSE_CUBIN_SHA256,
            "exp_014 baseline cubin is not the exact fee96b exp_015 reuse identity",
        )
        baseline = reuse_exp015_baseline(exp015_results)
        baseline["exp014_validation"] = {
            "path": evidence_path(exp014_baseline["path"], results),
            "sha256": exp014_baseline["sha256"],
            "source_sha256": exp014_baseline["source_sha256"],
            "cubin_sha256": exp014_baseline["cubin_sha256"],
            "jit_artifact_set_sha256": exp014_baseline["jit_artifact_set_sha256"],
        }
        baseline_mode = "exact_exp015_reuse"

    records = [baseline, candidate]
    by_arm = {record["arm"]: record for record in records}
    zero_spill = {
        arm: all(by_arm[arm]["metrics"][metric] == 0 for metric in SPILL_METRICS)
        for arm in ARMS
    }
    pairwise_work = {
        metric: by_arm[BASELINE]["metrics"][metric]
        == by_arm[CANDIDATE]["metrics"][metric]
        for metric in WORK_METRICS
    }
    ledger_work = {
        f"{arm}:{metric}": by_arm[arm]["metrics"][metric] == expected
        for arm in ARMS
        for metric, expected in EXPECTED_WORK.items()
    }
    gate = (
        all(zero_spill.values())
        and all(pairwise_work.values())
        and all(ledger_work.values())
    )
    return {
        "schema": "exp014.dynamic-spill-evidence.v1",
        "status": "pass" if gate else "reject",
        "scope": {
            "m": M,
            "fixture": FIXTURE,
            "execution": "one uninstrumented CUDA Graph replay node per arm",
            "gpu_uuid": EXPECTED_GPU_UUID,
            "grid": EXPECTED_GRID,
            "block": EXPECTED_BLOCK,
            "metric_ids": METRIC_IDS,
            "baseline_mode": baseline_mode,
        },
        "records": records,
        "checks": {
            "zero_dynamic_spill": zero_spill,
            "pairwise_tensor_work_identity": pairwise_work,
            "accepted_tensor_work_identity": ledger_work,
        },
        "gate_pass": gate,
        "evidence_boundary": (
            "Dynamic NCU counts prove executed spill/refill work for one M8192 "
            "graph node. They do not replace static cubin STACK/LOCAL/LDL/STL "
            "inspection, correctness validation, or the E2E performance gate."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--exp015-results",
        type=Path,
        default=EXP015_ROOT / "results",
        help="exact baseline-reuse evidence root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="default: <results>/dynamic_spill_evidence.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = args.results.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else results / "dynamic_spill_evidence.json"
    )
    value = build_evidence(results, args.exp015_results.resolve())
    write_json(output, value)
    print(json.dumps(value, sort_keys=True))
    return 0 if value["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
