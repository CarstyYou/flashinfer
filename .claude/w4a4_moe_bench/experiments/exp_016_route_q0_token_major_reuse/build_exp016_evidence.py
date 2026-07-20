#!/usr/bin/env python3
"""Build compact, GPU-free evidence for exp_016.

The collector recursively discovers immutable JSON captures below
``results/raw``.  It never imports torch or opens a CUDA context.  Missing
captures produce an explicit ``Unresolved`` report; malformed or conflicting
supplied evidence fails closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"

BASELINE = "baseline_pair_major"
CANDIDATE = "candidate_token_major_reuse"
ARMS = (BASELINE, CANDIDATE)
ABBA = (BASELINE, CANDIDATE, CANDIDATE, BASELINE)

EXPECTED_OVERLAY_SHA256 = {
    BASELINE: "c88cef63492b60c0a77484b50f6400b83a103d168e1535b78972341503810184",
    CANDIDATE: "ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19",
}
M_VALUES = (256, 512, 1024, 2048, 4096, 8192)
GROUPS = (0, 1, 2)
POSITIONS = (0, 1, 2, 3)
VALIDATION_CASES = (
    (256, "canonical", "equal"),
    (256, "canonical", "unequal"),
    (256, "hot_expert", "unequal"),
    (256, "tail_129", "unequal"),
    (8192, "canonical", "unequal"),
)

WARMUP = 5
ITERS = 50
L2_FLUSH_BYTES = 192 << 20
MAX_CV = 0.015
M8192_MIN_IMPROVEMENT_PERCENT = 2.0
SWEEP_MAX_REGRESSION_PERCENT = 1.5

VALIDATION_SCHEMA = "exp016.validation-case.v1"
BENCHMARK_SCHEMA = "exp016.benchmark-position.v1"


class EvidenceError(RuntimeError):
    """A supplied evidence record is invalid or cannot be traced."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def finite(value: Any, label: str, *, positive: bool = False) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} is not finite")
    if positive:
        require(result > 0.0, f"{label} must be positive")
    return result


def exact_int(value: Any, expected: int, label: str) -> None:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value == expected,
        f"{label} drift: {value!r} != {expected}",
    )


def close(actual: Any, expected: float, label: str) -> None:
    observed = finite(actual, label)
    require(
        math.isclose(observed, expected, rel_tol=1.0e-9, abs_tol=1.0e-9),
        f"{label} mismatch: {observed} != {expected}",
    )


def relative(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"invalid JSON {path}: {error}") from error
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def population_cv(values: Sequence[float]) -> float:
    require(bool(values), "cannot compute CV of an empty sample set")
    mean = statistics.fmean(values)
    require(mean > 0.0, "sample mean must be positive")
    return statistics.pstdev(values) / mean


def load_overlay_identity(
    results: Path,
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """Return local source identity plus hard failures and incompleteness."""
    failures: list[str] = []
    missing: list[str] = []
    identity_path = results / "overlays" / "identity.json"
    if not identity_path.is_file():
        return None, failures, [relative(identity_path, results)]
    try:
        identity = read_json(identity_path)
        require(
            identity.get("schema") == "exp016.route-q0-overlay.v1",
            "overlay identity schema drift",
        )
        mechanism = identity.get("mechanism_gate")
        require(
            isinstance(mechanism, Mapping) and mechanism.get("gate_pass") is True,
            "overlay mechanism gate failed",
        )
        source_sha = identity.get("source_sha256")
        source_role = identity.get("source_role")
        if source_role is None and source_sha == EXPECTED_OVERLAY_SHA256[BASELINE]:
            source_role = BASELINE
        require(source_role in ARMS, "overlay source role drift")
        require(
            source_sha == EXPECTED_OVERLAY_SHA256[source_role],
            "overlay source identity drift",
        )
        ledger = identity.get("work_ledger_m8192_topk8_h2048")
        require(isinstance(ledger, Mapping), "overlay work ledger missing")
        ledger_fields = (
            "bf16_block_loads_baseline",
            "bf16_block_loads_candidate",
            "productive_claims_baseline",
            "productive_claims_candidate",
            "row_allocation_atomics_baseline",
            "row_allocation_atomics_candidate",
            "packed_fp4_stores_baseline",
            "packed_fp4_stores_candidate",
            "sfa_stores_baseline",
            "sfa_stores_candidate",
        )
        require(
            all(isinstance(ledger.get(field), int) for field in ledger_fields),
            "overlay work ledger field drift",
        )
        require(
            ledger["bf16_block_loads_baseline"]
            == 8 * ledger["bf16_block_loads_candidate"],
            "BF16 block-load reuse ledger drift",
        )
        require(
            ledger["row_allocation_atomics_baseline"]
            == ledger["row_allocation_atomics_candidate"]
            and ledger["packed_fp4_stores_baseline"]
            == ledger["packed_fp4_stores_candidate"]
            and ledger["sfa_stores_baseline"] == ledger["sfa_stores_candidate"],
            "preserved output-work ledger drift",
        )
        arms = identity.get("arms")
        require(isinstance(arms, Mapping), "overlay arm identity missing")
        normalized_arms = {}
        for arm in ARMS:
            record = arms.get(arm)
            require(isinstance(record, Mapping), f"missing overlay identity for {arm}")
            expected = EXPECTED_OVERLAY_SHA256[arm]
            require(
                record.get("sha256") == expected, f"{arm} registered source hash drift"
            )
            overlay = results / "overlays" / arm / "moe_dynamic_kernel.py"
            require(overlay.is_file(), f"missing overlay source: {overlay}")
            require(file_sha256(overlay) == expected, f"{arm} overlay bytes drift")
            normalized_arms[arm] = {
                "path": relative(overlay, results),
                "sha256": expected,
            }
        return (
            {
                "path": relative(identity_path, results),
                "sha256": file_sha256(identity_path),
                "arms": normalized_arms,
                "mechanism_gate_pass": True,
                "source_role": source_role,
                "work_ledger": {field: ledger[field] for field in ledger_fields},
            },
            failures,
            missing,
        )
    except EvidenceError as error:
        failures.append(str(error))
        return None, failures, missing


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"missing {label}")
    return value


def _required_nonempty(
    record: Mapping[str, Any], fields: Sequence[str], label: str
) -> None:
    for field in fields:
        require(record.get(field) not in (None, ""), f"missing {label}.{field}")


def audit_runtime(
    runtime: Any,
    arm: str,
    label: str,
) -> dict[str, Any]:
    runtime = _required_mapping(runtime, f"{label}.runtime")
    source = _required_mapping(runtime.get("source"), f"{label}.runtime.source")
    gpu = _required_mapping(runtime.get("gpu"), f"{label}.runtime.gpu")
    imports = _required_mapping(runtime.get("imports"), f"{label}.runtime.imports")
    harness = _required_mapping(runtime.get("harness"), f"{label}.runtime.harness")

    _required_nonempty(
        runtime,
        (
            "hostname",
            "python",
            "packages",
            "torch",
            "cuda_runtime",
            "nvcc",
            "ptxas",
            "image_digest",
            "image_id",
            "python_deps_sha256",
        ),
        f"{label}.runtime",
    )
    _required_nonempty(
        source,
        (
            "locked_flashinfer_commit",
            "checkout_head",
            "cutlass_commit",
            "production_kernel_sha256",
            "overlay",
            "overlay_sha256",
            "oracle_source_sha256",
        ),
        f"{label}.source",
    )
    require(
        source["overlay_sha256"] == EXPECTED_OVERLAY_SHA256[arm],
        f"{label} loaded source hash drift",
    )
    _required_nonempty(
        gpu,
        (
            "uuid",
            "name",
            "pci_bus_id",
            "driver",
            "applications_graphics_clock_mhz",
            "max_graphics_clock_mhz",
            "compute_capability",
            "sm_count",
            "foreign_processes_before_cuda_context",
        ),
        f"{label}.gpu",
    )
    exact_int(gpu["sm_count"], 110, f"{label} SM count")
    require(
        gpu["compute_capability"] in ([12, 0], [12, 1]), f"{label} is not SM120/121"
    )
    require(
        gpu["foreign_processes_before_cuda_context"] == [],
        f"{label} had a foreign GPU process",
    )
    _required_nonempty(
        imports,
        ("flashinfer", "target_module", "cutlass_python", "cutlass_python_version"),
        f"{label}.imports",
    )
    require(
        imports["target_module"] == source["overlay"], f"{label} imported source drift"
    )
    require(is_sha256(harness.get("sha256")), f"{label} harness hash invalid")

    # Exclude timestamps, PIDs, lease IDs, JIT roots, current clocks and power.
    return {
        "host": runtime["hostname"],
        "python": runtime["python"],
        "packages": runtime["packages"],
        "torch": runtime["torch"],
        "cuda_runtime": runtime["cuda_runtime"],
        "nvcc": runtime["nvcc"],
        "ptxas": runtime["ptxas"],
        "image_digest": runtime["image_digest"],
        "image_id": runtime["image_id"],
        "python_deps_sha256": runtime["python_deps_sha256"],
        "gpu": {
            field: gpu[field]
            for field in (
                "uuid",
                "name",
                "pci_bus_id",
                "driver",
                "applications_graphics_clock_mhz",
                "max_graphics_clock_mhz",
                "compute_capability",
                "sm_count",
            )
        },
        "source_common": {
            field: source[field]
            for field in (
                "locked_flashinfer_commit",
                "checkout_head",
                "cutlass_commit",
                "production_kernel_sha256",
                "oracle_source_sha256",
            )
        },
        "imports_common": {
            field: imports[field]
            for field in ("flashinfer", "cutlass_python", "cutlass_python_version")
        },
        "harness_sha256": harness["sha256"],
    }


def audit_artifacts(value: Mapping[str, Any], label: str) -> tuple[str, str]:
    artifacts = value.get("jit_artifacts")
    require(
        isinstance(artifacts, list) and artifacts, f"{label} JIT artifact list missing"
    )
    artifact_hash = value.get("jit_artifact_set_sha256")
    require(is_sha256(artifact_hash), f"{label} JIT artifact-set hash invalid")
    require(
        artifact_hash == canonical_sha256(artifacts),
        f"{label} JIT artifact-set hash mismatch",
    )
    cubins = value.get("cubin_sha256")
    require(
        isinstance(cubins, list) and len(cubins) == 1 and is_sha256(cubins[0]),
        f"{label} must identify exactly one cubin",
    )
    artifact_cubins = {
        item.get("sha256")
        for item in artifacts
        if isinstance(item, Mapping) and str(item.get("path", "")).endswith(".cubin")
    }
    require(
        artifact_cubins == {cubins[0]}, f"{label} cubin/artifact inventory mismatch"
    )
    return artifact_hash, cubins[0]


def audit_validation(
    value: Mapping[str, Any], path: Path, results: Path
) -> dict[str, Any]:
    label = relative(path, results)
    require(value.get("schema") == VALIDATION_SCHEMA, f"{label} schema drift")
    require(
        value.get("status") == "complete" and value.get("gate_pass") is True,
        f"{label} validation gate failed",
    )
    arm = value.get("arm")
    require(arm in ARMS, f"{label} arm drift")
    m = value.get("m")
    require(isinstance(m, int) and m > 0, f"{label} M invalid")
    fixture = value.get("fixture")
    scale_kind = value.get("scale_kind")
    require(isinstance(fixture, str) and fixture, f"{label} fixture missing")
    require(scale_kind in ("equal", "unequal"), f"{label} scale kind drift")
    runtime = audit_runtime(value.get("runtime"), arm, label)
    artifact_hash, cubin = audit_artifacts(value, label)
    specialization = _required_mapping(
        value.get("specialization"), f"{label}.specialization"
    )
    require(
        specialization.get("gate_pass") is True, f"{label} specialization gate failed"
    )
    case_identity = _required_mapping(
        value.get("case_identity"), f"{label}.case_identity"
    )
    _required_nonempty(
        case_identity,
        ("fixture", "weights", "oracle_weights", "reference_sha256"),
        f"{label}.case_identity",
    )
    require(
        is_sha256(case_identity["reference_sha256"]), f"{label} reference hash invalid"
    )
    replays = value.get("replays")
    require(
        isinstance(replays, list) and len(replays) == 2, f"{label} replay count drift"
    )
    digests = []
    for replay_index, replay in enumerate(replays):
        replay = _required_mapping(replay, f"{label}.replay[{replay_index}]")
        exact_int(replay.get("replay"), replay_index, f"{label} replay index")
        require(replay.get("gate_pass") is True, f"{label} replay gate failed")
        require(is_sha256(replay.get("output_sha256")), f"{label} output hash invalid")
        payload = _required_mapping(
            replay.get("logical_payload"), f"{label}.logical_payload"
        )
        require(
            payload.get("gate_pass") is True, f"{label} logical payload gate failed"
        )
        digest = {}
        for field in (
            "route_metadata_sha256",
            "packed_fp4_sha256",
            "sfa_sha256",
            "combined_sha256",
        ):
            require(is_sha256(payload.get(field)), f"{label} invalid {field}")
            digest[field] = payload[field]
        exact_int(payload.get("logical_routes"), m * 8, f"{label} logical route count")
        digests.append(digest)
    require(
        value.get("logical_payload_replay_stable") is True,
        f"{label} payload replay stability failed",
    )
    require(digests[0] == digests[1], f"{label} logical payload replay digest drift")
    stability = _required_mapping(
        value.get("output_stability_gate"), f"{label}.output_stability_gate"
    )
    require(
        stability.get("gate_pass") is True, f"{label} output replay stability failed"
    )
    return {
        "key": (arm, m, fixture, scale_kind),
        "arm": arm,
        "m": m,
        "fixture": fixture,
        "scale_kind": scale_kind,
        "runtime": runtime,
        "artifact_set_sha256": artifact_hash,
        "cubin_sha256": cubin,
        "case_identity": dict(case_identity),
        "logical_digest": digests[0],
        "source": label,
        "source_sha256": file_sha256(path),
    }


def audit_benchmark(
    value: Mapping[str, Any], path: Path, results: Path
) -> dict[str, Any]:
    label = relative(path, results)
    require(value.get("schema") == BENCHMARK_SCHEMA, f"{label} schema drift")
    require(value.get("status") == "complete", f"{label} benchmark incomplete")
    arm = value.get("arm")
    require(arm in ARMS, f"{label} arm drift")
    m = value.get("m")
    group = value.get("group")
    position = value.get("position")
    require(isinstance(m, int) and m > 0, f"{label} M invalid")
    require(isinstance(group, int) and group >= 0, f"{label} group invalid")
    require(position in POSITIONS, f"{label} position invalid")
    require(arm == ABBA[position], f"{label} ABBA arm/order drift")
    require(value.get("abba_order") == list(ABBA), f"{label} ABBA declaration drift")
    require(value.get("fixture") == "canonical", f"{label} fixture drift")
    require(value.get("scale_kind") == "unequal", f"{label} scale kind drift")
    protocol = _required_mapping(value.get("protocol"), f"{label}.protocol")
    exact_int(protocol.get("warmup"), WARMUP, f"{label} warmup")
    exact_int(protocol.get("iters"), ITERS, f"{label} iterations")
    exact_int(protocol.get("l2_flush_bytes"), L2_FLUSH_BYTES, f"{label} L2 flush")
    samples = value.get("samples_us")
    require(
        isinstance(samples, list) and len(samples) == ITERS,
        f"{label} sample count drift",
    )
    samples = [finite(sample, f"{label} sample", positive=True) for sample in samples]
    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    cv = population_cv(samples)
    stored = _required_mapping(value.get("statistics_us"), f"{label}.statistics_us")
    exact_int(stored.get("count"), ITERS, f"{label} stored sample count")
    close(stored.get("mean"), mean, f"{label} stored mean")
    close(stored.get("median"), median, f"{label} stored median")
    close(stored.get("cv"), cv, f"{label} stored CV")
    runtime = audit_runtime(value.get("runtime"), arm, label)
    artifact_hash, cubin = audit_artifacts(value, label)
    specialization = _required_mapping(
        value.get("specialization"), f"{label}.specialization"
    )
    require(
        specialization.get("gate_pass") is True, f"{label} specialization gate failed"
    )
    gpu_after = _required_mapping(value.get("gpu_after"), f"{label}.gpu_after")
    require(
        gpu_after.get("uuid") == runtime["gpu"]["uuid"],
        f"{label} GPU changed during capture",
    )
    require(
        str(gpu_after.get("applications_graphics_clock_mhz"))
        == str(runtime["gpu"]["applications_graphics_clock_mhz"]),
        f"{label} application clock changed during capture",
    )
    fixture_identity = _required_mapping(
        value.get("fixture_identity"), f"{label}.fixture_identity"
    )
    weight_identity = _required_mapping(
        value.get("weight_identity"), f"{label}.weight_identity"
    )
    require(is_sha256(value.get("output_sha256")), f"{label} output hash invalid")
    return {
        "key": (arm, m, group, position),
        "arm": arm,
        "m": m,
        "group": group,
        "position": position,
        "samples_us": samples,
        "mean_us": mean,
        "median_us": median,
        "cv": cv,
        "runtime": runtime,
        "artifact_set_sha256": artifact_hash,
        "cubin_sha256": cubin,
        "fixture_identity": dict(fixture_identity),
        "weight_identity": dict(weight_identity),
        "source": label,
        "source_sha256": file_sha256(path),
    }


def discover(results: Path) -> dict[str, Any]:
    raw_root = results / "raw"
    validations: dict[tuple[Any, ...], dict[str, Any]] = {}
    benchmarks: dict[tuple[Any, ...], dict[str, Any]] = {}
    failures: list[str] = []
    inventory = []
    unknown = []
    ignored_failed = []
    if not raw_root.is_dir():
        return {
            "validations": validations,
            "benchmarks": benchmarks,
            "failures": failures,
            "inventory": inventory,
            "unknown": unknown,
            "ignored_failed": ignored_failed,
        }
    for path in sorted(raw_root.rglob("*.json")):
        record = {
            "path": relative(path, results),
            "sha256": file_sha256(path),
        }
        try:
            value = read_json(path)
            schema = value.get("schema")
            record["schema"] = schema
            if schema == VALIDATION_SCHEMA:
                if (
                    value.get("status") != "complete"
                    or value.get("gate_pass") is not True
                ):
                    record["kind"] = "ignored_failed_validation"
                    record["reason"] = (
                        "capture is not eligible evidence: status/gate_pass is not "
                        "complete/true; a later valid capture may supersede it"
                    )
                    ignored_failed.append(record["path"])
                    inventory.append(record)
                    continue
                normalized = audit_validation(value, path, results)
                require(
                    normalized["key"] not in validations,
                    f"duplicate validation key {normalized['key']}",
                )
                validations[normalized["key"]] = normalized
                record["kind"] = "validation"
            elif schema == BENCHMARK_SCHEMA:
                if value.get("status") != "complete":
                    record["kind"] = "ignored_incomplete_benchmark"
                    record["reason"] = (
                        "capture is not eligible evidence: status is not complete"
                    )
                    ignored_failed.append(record["path"])
                    inventory.append(record)
                    continue
                normalized = audit_benchmark(value, path, results)
                require(
                    normalized["key"] not in benchmarks,
                    f"duplicate benchmark key {normalized['key']}",
                )
                benchmarks[normalized["key"]] = normalized
                record["kind"] = "benchmark"
            else:
                record["kind"] = "ignored"
                unknown.append(record["path"])
        except EvidenceError as error:
            record["kind"] = "invalid"
            record["error"] = str(error)
            failures.append(str(error))
        inventory.append(record)
    return {
        "validations": validations,
        "benchmarks": benchmarks,
        "failures": failures,
        "inventory": inventory,
        "unknown": unknown,
        "ignored_failed": ignored_failed,
    }


def build_identity(
    overlay: dict[str, Any] | None,
    validations: Mapping[tuple[Any, ...], Mapping[str, Any]],
    benchmarks: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str], list[str]]:
    failures: list[str] = []
    missing: list[str] = []
    records = list(validations.values()) + list(benchmarks.values())
    if overlay is None:
        missing.append("results/overlays/identity.json and registered overlay bytes")
    runtime_hashes = {canonical_sha256(record["runtime"]) for record in records}
    if len(runtime_hashes) > 1:
        failures.append("GPU/toolchain/source-common identity differs across captures")
    elif not runtime_hashes:
        missing.append("runtime identity from validation/benchmark captures")

    arm_identity = {}
    for arm in ARMS:
        arm_records = [record for record in records if record["arm"] == arm]
        artifact_hashes = {record["artifact_set_sha256"] for record in arm_records}
        cubins = {record["cubin_sha256"] for record in arm_records}
        if len(artifact_hashes) > 1:
            failures.append(f"{arm} JIT artifact-set identity differs across captures")
        if len(cubins) > 1:
            failures.append(f"{arm} cubin identity differs across captures")
        if not arm_records:
            missing.append(f"{arm} capture identity")
        arm_identity[arm] = {
            "source_sha256": EXPECTED_OVERLAY_SHA256[arm],
            "artifact_set_sha256": next(iter(artifact_hashes))
            if len(artifact_hashes) == 1
            else None,
            "cubin_sha256": next(iter(cubins)) if len(cubins) == 1 else None,
            "capture_count": len(arm_records),
        }
    gate = not failures and not missing
    return (
        {
            "status": "pass" if gate else ("fail" if failures else "incomplete"),
            "gate_pass": gate if not missing else None,
            "stable_runtime_identity_sha256": next(iter(runtime_hashes))
            if len(runtime_hashes) == 1
            else None,
            "runtime_identity": records[0]["runtime"]
            if len(runtime_hashes) == 1
            else None,
            "overlay_identity": overlay,
            "arms": arm_identity,
        },
        failures,
        missing,
    )


def build_correctness(
    validations: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str], list[str]]:
    failures: list[str] = []
    missing: list[str] = []
    paired = []
    expected_keys = {
        (arm, m, fixture, scale_kind)
        for arm in ARMS
        for m, fixture, scale_kind in VALIDATION_CASES
    }
    extra = [list(key) for key in sorted(set(validations) - expected_keys)]
    for m, fixture, scale_kind in VALIDATION_CASES:
        key_a = (BASELINE, m, fixture, scale_kind)
        key_b = (CANDIDATE, m, fixture, scale_kind)
        absent = [
            arm
            for arm, key in ((BASELINE, key_a), (CANDIDATE, key_b))
            if key not in validations
        ]
        case_label = f"M{m}/{fixture}/{scale_kind}"
        if absent:
            missing.append(f"{case_label}: missing {', '.join(absent)} validation")
            paired.append(
                {
                    "m": m,
                    "fixture": fixture,
                    "scale_kind": scale_kind,
                    "status": "incomplete",
                }
            )
            continue
        baseline = validations[key_a]
        candidate = validations[key_b]
        case_failures = []
        if baseline["runtime"] != candidate["runtime"]:
            case_failures.append("runtime identity mismatch")
        if baseline["case_identity"] != candidate["case_identity"]:
            case_failures.append("fixture/reference identity mismatch")
        digest_equal = {
            field: baseline["logical_digest"][field]
            == candidate["logical_digest"][field]
            for field in (
                "route_metadata_sha256",
                "packed_fp4_sha256",
                "sfa_sha256",
                "combined_sha256",
            )
        }
        if not all(digest_equal.values()):
            case_failures.append("logical FP4/SFA/metadata digest mismatch")
        if case_failures:
            failures.extend(f"{case_label}: {reason}" for reason in case_failures)
        paired.append(
            {
                "m": m,
                "fixture": fixture,
                "scale_kind": scale_kind,
                "status": "fail" if case_failures else "pass",
                "fixture_identity_equal": baseline["case_identity"]
                == candidate["case_identity"],
                "logical_digest_equal": digest_equal,
                "logical_digest": baseline["logical_digest"]
                if not case_failures
                else None,
                "sources": {
                    BASELINE: baseline["source"],
                    CANDIDATE: candidate["source"],
                },
            }
        )
    gate = not failures and not missing
    return (
        {
            "status": "pass" if gate else ("fail" if failures else "incomplete"),
            "gate_pass": gate if not missing else None,
            "expected_case_count_per_arm": len(VALIDATION_CASES),
            "observed_case_count_per_arm": {
                arm: sum(record["arm"] == arm for record in validations.values())
                for arm in ARMS
            },
            "paired_cases": paired,
            "extra_supported_schema_records": extra,
        },
        failures,
        missing,
    )


def validate_benchmark_pair_identity(
    rows: Sequence[Mapping[str, Any]],
    validations: Mapping[tuple[Any, ...], Mapping[str, Any]],
    m: int,
    group: int,
) -> None:
    fixture_hashes = {canonical_sha256(row["fixture_identity"]) for row in rows}
    weight_hashes = {canonical_sha256(row["weight_identity"]) for row in rows}
    require(len(fixture_hashes) == 1, f"M{m}/group{group} fixture identity mismatch")
    require(len(weight_hashes) == 1, f"M{m}/group{group} weight identity mismatch")
    if m == 8192:
        for arm in ARMS:
            validation = validations.get((arm, 8192, "canonical", "unequal"))
            if validation is None:
                continue
            arm_rows = [row for row in rows if row["arm"] == arm]
            for row in arm_rows:
                require(
                    row["fixture_identity"] == validation["case_identity"]["fixture"],
                    f"M8192/group{group}/{arm} benchmark/validation fixture mismatch",
                )
                require(
                    row["weight_identity"] == validation["case_identity"]["weights"],
                    f"M8192/group{group}/{arm} benchmark/validation weights mismatch",
                )


def build_performance(
    benchmarks: Mapping[tuple[Any, ...], Mapping[str, Any]],
    validations: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str], list[str], list[dict[str, Any]]]:
    failures: list[str] = []
    missing: list[str] = []
    group_rows = []
    csv_rows = []
    supported_keys = {
        (ABBA[position], m, group, position)
        for m in M_VALUES
        for group in GROUPS
        for position in POSITIONS
    }
    extra = [list(key) for key in sorted(set(benchmarks) - supported_keys)]

    for m in M_VALUES:
        for group in GROUPS:
            keys = [(ABBA[position], m, group, position) for position in POSITIONS]
            absent = [
                position
                for position, key in zip(POSITIONS, keys, strict=True)
                if key not in benchmarks
            ]
            if absent:
                missing.append(f"M{m}/group{group}: missing ABBA positions {absent}")
                group_rows.append(
                    {
                        "m": m,
                        "group": group,
                        "status": "incomplete",
                        "missing_positions": absent,
                    }
                )
                continue
            rows = [benchmarks[key] for key in keys]
            try:
                validate_benchmark_pair_identity(rows, validations, m, group)
            except EvidenceError as error:
                failures.append(str(error))
                group_rows.append(
                    {"m": m, "group": group, "status": "fail", "reason": str(error)}
                )
                continue
            baseline_samples = rows[0]["samples_us"] + rows[3]["samples_us"]
            candidate_samples = rows[1]["samples_us"] + rows[2]["samples_us"]
            baseline_median = statistics.median(baseline_samples)
            candidate_median = statistics.median(candidate_samples)
            baseline_cv = population_cv(baseline_samples)
            candidate_cv = population_cv(candidate_samples)
            improvement = (baseline_median / candidate_median - 1.0) * 100.0
            cv_gate = baseline_cv <= MAX_CV and candidate_cv <= MAX_CV
            row = {
                "m": m,
                "group": group,
                "status": "pass" if cv_gate else "unstable",
                "baseline_median_us": baseline_median,
                "candidate_median_us": candidate_median,
                "improvement_percent": improvement,
                "baseline_cv": baseline_cv,
                "candidate_cv": candidate_cv,
                "cv_gate_pass": cv_gate,
                "sources": [capture["source"] for capture in rows],
            }
            group_rows.append(row)
            for capture in rows:
                csv_rows.append(
                    {
                        "m": m,
                        "group": group,
                        "position": capture["position"],
                        "arm": capture["arm"],
                        "count": len(capture["samples_us"]),
                        "position_median_us": capture["median_us"],
                        "position_cv": capture["cv"],
                        "group_baseline_median_us": baseline_median,
                        "group_candidate_median_us": candidate_median,
                        "group_improvement_percent": improvement,
                        "group_baseline_cv": baseline_cv,
                        "group_candidate_cv": candidate_cv,
                        "group_cv_gate_pass": cv_gate,
                        "source": capture["source"],
                        "source_sha256": capture["source_sha256"],
                    }
                )

    cases = []
    for m in M_VALUES:
        rows = [
            row
            for row in group_rows
            if row["m"] == m
            and row["status"] != "incomplete"
            and "improvement_percent" in row
        ]
        incomplete_groups = [
            group
            for group in GROUPS
            if not any(
                row["m"] == m and row["group"] == group and "improvement_percent" in row
                for row in group_rows
            )
        ]
        if incomplete_groups:
            cases.append(
                {"m": m, "status": "incomplete", "missing_groups": incomplete_groups}
            )
            continue
        improvements = [row["improvement_percent"] for row in rows]
        all_cv = all(row["cv_gate_pass"] for row in rows)
        median_improvement = statistics.median(improvements)
        baseline_median = statistics.median(row["baseline_median_us"] for row in rows)
        candidate_median = statistics.median(row["candidate_median_us"] for row in rows)
        if m == 8192:
            performance_gate = (
                all_cv
                and all(value > 0.0 for value in improvements)
                and median_improvement >= M8192_MIN_IMPROVEMENT_PERCENT
            )
        else:
            performance_gate = (
                all_cv and median_improvement >= -SWEEP_MAX_REGRESSION_PERCENT
            )
        cases.append(
            {
                "m": m,
                "status": "pass"
                if performance_gate
                else ("unstable" if not all_cv else "fail"),
                "baseline_median_us": baseline_median,
                "candidate_median_us": candidate_median,
                "median_group_improvement_percent": median_improvement,
                "group_improvement_percent": improvements,
                "all_group_cv_gate_pass": all_cv,
                "gate_pass": performance_gate,
            }
        )

    m8192 = next(case for case in cases if case["m"] == 8192)
    full_complete = all(case["status"] != "incomplete" for case in cases)
    full_gate = full_complete and all(case.get("gate_pass") is True for case in cases)
    if m8192["status"] == "fail":
        failures.append("M8192 stable ABBA result did not meet the positive 2% gate")
    for case in cases:
        if case["m"] != 8192 and case["status"] == "fail":
            failures.append(f"M{case['m']} has a stable regression beyond 1.5%")
    unstable = [case["m"] for case in cases if case["status"] == "unstable"]
    if unstable:
        missing.append(f"ABBA CV exceeds 1.5% and must be recaptured for M={unstable}")
    status = "fail" if failures else ("pass" if full_gate else "incomplete")
    return (
        {
            "status": status,
            "gate_pass": full_gate if full_complete and not unstable else None,
            "protocol": {
                "order": list(ABBA),
                "groups_per_m": len(GROUPS),
                "samples_per_position": ITERS,
                "aggregation": "combine the two A positions and two B positions; median and population CV over 100 replays per arm/group",
                "improvement_percent": "(baseline_median / candidate_median - 1) * 100",
                "max_cv": MAX_CV,
                "m8192_gate": "all 3 deltas positive; median delta >= 2%; all arm/group CV <= 1.5%",
                "sweep_gate": "each M median delta >= -1.5%; all arm/group CV <= 1.5%",
            },
            "groups": group_rows,
            "cases": cases,
            "m8192_quick_gate_pass": m8192.get("gate_pass")
            if m8192["status"] not in ("incomplete", "unstable")
            else None,
            "full_sweep_gate_pass": full_gate
            if full_complete and not unstable
            else None,
            "extra_supported_schema_records": extra,
        },
        failures,
        missing,
        csv_rows,
    )


def decide_scoped_verdict(
    hard_failures: Sequence[str],
    correctness: Mapping[str, Any],
    performance: Mapping[str, Any],
) -> tuple[str, str, str]:
    if (
        hard_failures
        or correctness["status"] == "fail"
        or performance["status"] == "fail"
    ):
        return "Reject", "rejected", "已提供证据存在硬失败，或稳定性能未通过门槛。"
    if (
        correctness["status"] == "pass"
        and performance.get("full_sweep_gate_pass") is True
    ):
        return "Accept", "complete", "正确性、身份与完整三组 ABBA sweep 均通过。"
    if performance.get("m8192_quick_gate_pass") is True:
        return (
            "Unresolved",
            "incomplete",
            "M8192 快速门通过；仍需补齐完整 M256–8192 sweep。",
        )
    return "Unresolved", "incomplete", "证据尚未齐全或测量稳定性不足，不能作性能结论。"


def audit_p3_supporting_evidence(
    value: Mapping[str, Any], identity: Mapping[str, Any], label: str
) -> dict[str, Any]:
    capture_gate = value.get("capture_integrity_gate_pass")
    instrumentation_gate = value.get("instrumentation_gate_pass")
    sass_gate = value.get("sass_spill_gate_pass")
    phase_gate = value.get("phase_improvement_gate_pass")
    for name, gate in (
        ("capture integrity", capture_gate),
        ("instrumentation", instrumentation_gate),
        ("SASS spill", sass_gate),
        ("phase improvement", phase_gate),
    ):
        require(gate in (True, False), f"{label} {name} gate is not boolean")

    environment = _required_mapping(value.get("environment"), f"{label}.environment")
    expected_runtime = _required_mapping(
        identity.get("runtime_identity"), "top-level stable runtime identity"
    )
    for field in (
        "python",
        "packages",
        "torch",
        "cuda_runtime",
        "nvcc",
        "ptxas",
        "image_digest",
        "image_id",
        "python_deps_sha256",
    ):
        require(
            environment.get(field) == expected_runtime.get(field),
            f"{label} environment.{field} differs from uninstrumented evidence",
        )
    require(
        environment.get("gpu") == expected_runtime.get("gpu"),
        f"{label} GPU identity differs from uninstrumented evidence",
    )
    require(
        environment.get("imports") == expected_runtime.get("imports_common"),
        f"{label} import identity differs from uninstrumented evidence",
    )
    harness = _required_mapping(environment.get("harness"), f"{label}.harness")
    require(
        harness.get("run_exp016_arm_sha256") == expected_runtime.get("harness_sha256"),
        f"{label} base harness identity differs from uninstrumented evidence",
    )

    instrumentation = _required_mapping(
        value.get("instrumentation"), f"{label}.instrumentation"
    )
    captures = _required_mapping(value.get("captures"), f"{label}.captures")
    compact_instrumentation = {}
    arm_instrumentation_gates = []
    arm_sass_gates = []
    arm_smem_gates = []
    resource_fields = (
        "registers_per_thread",
        "stack_bytes_per_thread",
        "static_local_bytes_outside_stack",
        "static_shared_bytes_per_cta",
    )
    zero_sass_fields = (
        "spill_refill_annotation_count",
        "spill_refill_annotation_unique_pc_count",
        "ldl_opcode_count",
        "stl_opcode_count",
        "local_sass_opcode_count",
    )
    for arm in ARMS:
        record = _required_mapping(instrumentation.get(arm), f"{label}.{arm}")
        control = _required_mapping(
            record.get("control_resource"), f"{label}.{arm}.control_resource"
        )
        probe = _required_mapping(
            record.get("probe_resource"), f"{label}.{arm}.probe_resource"
        )
        require(
            all(isinstance(control.get(field), int) for field in resource_fields)
            and all(isinstance(probe.get(field), int) for field in resource_fields),
            f"{label}.{arm} resource record is incomplete",
        )
        resource_equal = all(
            control[field] == probe[field] for field in resource_fields
        )
        require(
            record.get("resource_identity_equal") is resource_equal,
            f"{label}.{arm} resource identity gate/count disagreement",
        )
        smem_gate = (
            control["static_shared_bytes_per_cta"]
            == probe["static_shared_bytes_per_cta"]
        )
        arm_smem_gates.append(smem_gate)

        spill_summaries = {}
        spill_gates = []
        for mode in ("control", "probe"):
            spill = _required_mapping(
                record.get(f"{mode}_sass_spill"), f"{label}.{arm}.{mode}_sass_spill"
            )
            counts = _required_mapping(
                spill.get("counts"), f"{label}.{arm}.{mode}.counts"
            )
            require(
                all(isinstance(counts.get(field), int) for field in zero_sass_fields),
                f"{label}.{arm}.{mode} SASS count set is incomplete",
            )
            zero_spill = all(counts[field] == 0 for field in zero_sass_fields)
            require(
                counts.get("annotation_pcs_equal_local_sass_pcs") is True,
                f"{label}.{arm}.{mode} SASS PC cross-check failed",
            )
            require(
                spill.get("gate_pass") is zero_spill,
                f"{label}.{arm}.{mode} SASS spill gate/count disagreement",
            )
            spill_gates.append(zero_spill)
            spill_summaries[mode] = {
                "gate_pass": zero_spill,
                "counts": {field: counts[field] for field in zero_sass_fields},
                "cubin_sha256": spill.get("cubin_sha256"),
                "source_sha256": spill.get("sha256"),
            }
        arm_sass_gate = all(spill_gates)
        require(
            record.get("sass_spill_gate_pass") is arm_sass_gate,
            f"{label}.{arm} SASS spill aggregate disagreement",
        )
        perturbation = finite(
            record.get("probe_e2e_perturbation_percent"),
            f"{label}.{arm} probe E2E perturbation",
        )
        allowed = finite(
            record.get("max_abs_allowed_perturbation_percent"),
            f"{label}.{arm} allowed perturbation",
            positive=True,
        )
        small = abs(perturbation) <= allowed
        require(
            record.get("e2e_perturbation_small") is small,
            f"{label}.{arm} perturbation gate/value disagreement",
        )
        arm_gate = resource_equal and small and arm_sass_gate
        require(
            record.get("gate_pass") is arm_gate,
            f"{label}.{arm} instrumentation aggregate disagreement",
        )
        arm_instrumentation_gates.append(arm_gate)
        arm_sass_gates.append(arm_sass_gate)
        compact_instrumentation[arm] = {
            "gate_pass": arm_gate,
            "resource_identity_equal": resource_equal,
            "smem_identity_gate_pass": smem_gate,
            "control_resource": dict(control),
            "probe_resource": dict(probe),
            "probe_e2e_perturbation_percent": perturbation,
            "e2e_perturbation_small": small,
            "sass_spill_gate_pass": arm_sass_gate,
            "sass_spill": spill_summaries,
        }

        arm_captures = _required_mapping(captures.get(arm), f"{label}.captures.{arm}")
        for mode in ("control_no_marker", "probe"):
            capture = _required_mapping(
                arm_captures.get(mode), f"{label}.captures.{arm}.{mode}"
            )
            require(
                capture.get("capture_gate_pass") is True,
                f"{label}.{arm}.{mode} capture gate failed",
            )
            base_source = _required_mapping(
                capture.get("base_source"), f"{label}.{arm}.{mode}.base_source"
            )
            require(
                base_source.get("kernel_sha256") == EXPECTED_OVERLAY_SHA256[arm],
                f"{label}.{arm}.{mode} base source identity drift",
            )

    computed_instrumentation = all(arm_instrumentation_gates)
    computed_sass = all(arm_sass_gates)
    computed_smem = all(arm_smem_gates)
    require(
        instrumentation_gate is computed_instrumentation,
        f"{label} instrumentation aggregate disagreement",
    )
    require(sass_gate is computed_sass, f"{label} SASS aggregate disagreement")

    phase = _required_mapping(
        value.get("phase_comparison"), f"{label}.phase_comparison"
    )
    baseline_us = finite(
        phase.get("baseline_grid_critical_wall_median_us"),
        f"{label} baseline phase latency",
        positive=True,
    )
    candidate_us = finite(
        phase.get("candidate_grid_critical_wall_median_us"),
        f"{label} candidate phase latency",
        positive=True,
    )
    delta = candidate_us - baseline_us
    reduction = (baseline_us - candidate_us) / baseline_us * 100.0
    close(phase.get("candidate_minus_baseline_us"), delta, f"{label} phase delta")
    close(phase.get("latency_reduction_percent"), reduction, f"{label} phase reduction")
    computed_phase = (
        candidate_us < baseline_us
        and phase.get("candidate_faster") is True
        and phase.get("all_candidate_samples_faster") is True
    )
    require(phase_gate is computed_phase, f"{label} phase gate/value disagreement")
    computed_overall = capture_gate and computed_instrumentation and computed_phase
    require(
        value.get("gate_pass") is computed_overall,
        f"{label} overall gate/component disagreement",
    )
    require(
        phase.get("interpretation_legal_as_diagnostic") is computed_overall
        and phase.get("interpretation_legal_as_production_phase_truth") is False,
        f"{label} diagnostic evidence boundary drift",
    )
    require(
        value.get("additive_sm_estimate_used") is False,
        f"{label} illegally used an additive SM estimate",
    )
    return {
        "gates": {
            "capture_integrity_gate_pass": capture_gate,
            "instrumentation_gate_pass": computed_instrumentation,
            "sass_spill_gate_pass": computed_sass,
            "smem_identity_gate_pass": computed_smem,
            "phase_improvement_gate_pass": computed_phase,
        },
        "phase_comparison": dict(phase),
        "instrumentation": compact_instrumentation,
        "classification": value.get("evidence_classification"),
        "performance_authority": value.get("performance_authority"),
    }


def audit_dynamic_spill_supporting_evidence(
    value: Mapping[str, Any], identity: Mapping[str, Any], label: str
) -> dict[str, Any]:
    candidate = _required_mapping(value.get("candidate"), f"{label}.candidate")
    expected = _required_mapping(
        _required_mapping(identity.get("arms"), "top-level arm identity").get(
            CANDIDATE
        ),
        "top-level Candidate identity",
    )
    require(candidate.get("arm") == CANDIDATE, f"{label} arm drift")
    require(
        candidate.get("source_sha256") == expected.get("source_sha256"),
        f"{label} source identity drift",
    )
    require(
        candidate.get("cubin_sha256") == expected.get("cubin_sha256"),
        f"{label} cubin identity drift",
    )
    require(
        candidate.get("jit_artifact_set_sha256") == expected.get("artifact_set_sha256"),
        f"{label} JIT artifact-set identity drift",
    )
    runtime = _required_mapping(identity.get("runtime_identity"), "runtime identity")
    require(
        candidate.get("gpu_uuid")
        == _required_mapping(runtime.get("gpu"), "GPU identity").get("uuid"),
        f"{label} GPU identity drift",
    )
    require(
        candidate.get("m") == 8192
        and candidate.get("fixture") == "canonical"
        and candidate.get("scale_kind") == "unequal",
        f"{label} fixture identity drift",
    )
    launch = _required_mapping(candidate.get("observed_launch"), f"{label}.launch")
    require(
        launch.get("grid") == [1, 1, 110] and launch.get("block") == [288, 1, 1],
        f"{label} launch topology drift",
    )
    metrics = _required_mapping(candidate.get("metrics"), f"{label}.metrics")
    metric_names = (
        "spill_register_read_instructions",
        "spill_register_write_instructions",
        "spill_local_load_bytes",
        "spill_local_store_bytes",
    )
    require(
        set(metrics) == set(metric_names), f"{label} dynamic metric inventory drift"
    )
    require(
        all(
            isinstance(metrics[name], int) and metrics[name] >= 0
            for name in metric_names
        ),
        f"{label} dynamic metric value drift",
    )
    zero_spill = all(metrics[name] == 0 for name in metric_names)
    checks = _required_mapping(value.get("checks"), f"{label}.checks")
    require(
        checks.get("zero_dynamic_spill") is zero_spill
        and checks.get("dynamic_local_load_store_bytes")
        == metrics["spill_local_load_bytes"] + metrics["spill_local_store_bytes"],
        f"{label} dynamic spill gate/count disagreement",
    )
    require(
        value.get("gate_pass") is zero_spill, f"{label} overall gate/count disagreement"
    )
    return {
        "metrics": dict(metrics),
        "observed_launch": dict(launch),
        "identity": {
            "source_sha256": candidate["source_sha256"],
            "cubin_sha256": candidate["cubin_sha256"],
            "jit_artifact_set_sha256": candidate["jit_artifact_set_sha256"],
            "gpu_uuid": candidate["gpu_uuid"],
        },
        "zero_dynamic_spill": zero_spill,
    }


def load_supporting_gate(
    results: Path,
    identity: Mapping[str, Any],
    *,
    name: str,
    filename: str,
    schema: str,
) -> dict[str, Any]:
    """Load one compact downstream gate without opening its large raw evidence."""
    path = results / filename
    if not path.is_file():
        return {
            "name": name,
            "status": "incomplete",
            "gate_pass": None,
            "source": filename,
            "reason": "evidence file is missing",
        }
    try:
        value = read_json(path)
        require(value.get("schema") == schema, f"{filename} schema drift")
        gate = value.get("gate_pass")
        require(gate in (True, False), f"{filename} gate_pass must be boolean")
        status = value.get("status")
        require(isinstance(status, str) and status, f"{filename} status missing")
        summary: dict[str, Any]
        if schema == "exp016.p3-phase-evidence.v1":
            summary = audit_p3_supporting_evidence(value, identity, filename)
        elif schema == "exp016.dynamic-spill-evidence.v1":
            summary = audit_dynamic_spill_supporting_evidence(value, identity, filename)
        else:
            raise EvidenceError(f"unsupported supporting evidence schema: {schema}")
        normalized_status = status.lower()
        return {
            "name": name,
            "status": "pass"
            if gate
            else (
                "incomplete"
                if normalized_status in ("incomplete", "pending", "unresolved")
                else "fail"
            ),
            "gate_pass": gate
            if gate
            else (
                None
                if normalized_status in ("incomplete", "pending", "unresolved")
                else False
            ),
            "source": filename,
            "source_sha256": file_sha256(path),
            "schema": schema,
            "upstream_status": status,
            "summary": summary,
        }
    except EvidenceError as error:
        return {
            "name": name,
            "status": "fail",
            "gate_pass": False,
            "source": filename,
            "source_sha256": file_sha256(path),
            "reason": str(error),
        }


def decide_overall_verdict(
    scoped_verdict: str,
    supporting_gates: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, str]:
    if scoped_verdict == "Reject":
        return "Reject", "rejected", "Validation/E2E scoped gate 已 Reject。"
    dynamic = supporting_gates["dynamic_spill"]
    if dynamic["gate_pass"] is False:
        return "Reject", "rejected", "Dynamic spill gate 失败。"
    if scoped_verdict == "Accept" and all(
        gate["gate_pass"] is True for gate in supporting_gates.values()
    ):
        return (
            "Accept",
            "complete",
            "Validation/E2E、P3 phase 与 dynamic spill 门禁均通过。",
        )
    return (
        "Unresolved",
        "incomplete",
        "Validation/E2E 可独立判定；整体仍等待 P3 phase 与 dynamic spill 门禁闭合。",
    )


def build(results: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    overlay, overlay_failures, overlay_missing = load_overlay_identity(results)
    discovered = discover(results)
    identity, identity_failures, identity_missing = build_identity(
        overlay, discovered["validations"], discovered["benchmarks"]
    )
    correctness, correctness_failures, correctness_missing = build_correctness(
        discovered["validations"]
    )
    performance, performance_failures, performance_missing, csv_rows = (
        build_performance(discovered["benchmarks"], discovered["validations"])
    )
    failures = (
        overlay_failures
        + discovered["failures"]
        + identity_failures
        + correctness_failures
        + performance_failures
    )
    missing = list(
        dict.fromkeys(
            overlay_missing
            + identity_missing
            + correctness_missing
            + performance_missing
        )
    )
    scoped_verdict, scoped_status, scoped_reason = decide_scoped_verdict(
        failures, correctness, performance
    )
    supporting_gates = {
        "p3_phase": load_supporting_gate(
            results,
            identity,
            name="P3 phase",
            filename="p3_phase_evidence.json",
            schema="exp016.p3-phase-evidence.v1",
        ),
        "dynamic_spill": load_supporting_gate(
            results,
            identity,
            name="Dynamic spill",
            filename="dynamic_spill_evidence.json",
            schema="exp016.dynamic-spill-evidence.v1",
        ),
    }
    for gate in supporting_gates.values():
        if gate["gate_pass"] is not True:
            missing.append(
                f"{gate['name']} gate: {gate.get('reason', gate['status'])} ({gate['source']})"
            )
    verdict, status, reason = decide_overall_verdict(scoped_verdict, supporting_gates)
    evidence = {
        "schema": "exp016.evidence.v1",
        "status": status,
        "verdict": verdict,
        "reason": reason,
        "validation_e2e_scope": {
            "verdict": scoped_verdict,
            "status": scoped_status,
            "reason": scoped_reason,
        },
        "identity": identity,
        "correctness": correctness,
        "performance": performance,
        "hard_failures": failures,
        "missing_evidence": missing,
        "ignored_raw_json": discovered["unknown"],
        "ignored_failed_captures": discovered["ignored_failed"],
        "supporting_gates": supporting_gates,
        "evidence_boundary": (
            "Raw validation and uninstrumented E2E ABBA are the performance authority. "
            "The P3 matched probe is diagnostic; dynamic spill is imported from its "
            "identity-locked native NCU gate."
        ),
    }
    manifest = {
        "schema": "exp016.evidence-manifest.v1",
        "builder": {
            "path": relative(Path(__file__), ROOT),
            "sha256": file_sha256(Path(__file__)),
        },
        "raw_inventory": discovered["inventory"],
        "raw_inventory_sha256": canonical_sha256(discovered["inventory"]),
        "overlay_identity": overlay,
        "verdict": verdict,
        "status": status,
        "validation_e2e_verdict": scoped_verdict,
        "supporting_gates": supporting_gates,
    }
    return evidence, manifest, csv_rows


def render_markdown(evidence: Mapping[str, Any]) -> str:
    p3_support = evidence["supporting_gates"]["p3_phase"]
    p3_subgates = p3_support.get("summary", {}).get("gates", {})
    p3_gate_text = " / ".join(
        "pass" if p3_subgates.get(name) is True else "incomplete"
        for name in (
            "capture_integrity_gate_pass",
            "instrumentation_gate_pass",
            "sass_spill_gate_pass",
            "smem_identity_gate_pass",
            "phase_improvement_gate_pass",
        )
    )
    lines = [
        "# exp_016：Route/Q0 Token-major 输入复用",
        "",
        f"> **整体判定：{evidence['verdict']}** — {evidence['reason']}",
        "",
        f"> **Validation + E2E scoped 判定：{evidence['validation_e2e_scope']['verdict']}** — "
        f"{evidence['validation_e2e_scope']['reason']}",
        "",
        "## 正确性与身份",
        "",
        "| 项目 | 状态 |",
        "|---|---:|",
        f"| Source / cubin / GPU / toolchain identity | {evidence['identity']['status']} |",
        f"| Paired FP4 / SFA / metadata digest | {evidence['correctness']['status']} |",
        f"| P3 phase gate | {evidence['supporting_gates']['p3_phase']['status']} |",
        f"| P3 capture / instrumentation / SASS spill / SMEM / phase | {p3_gate_text} |",
        f"| Dynamic spill gate | {evidence['supporting_gates']['dynamic_spill']['status']} |",
        "",
        "每个 paired case 均按逻辑 route 顺序比较；physical row 顺序差异不会造成误判。",
        "",
        "## 未插桩 E2E ABBA",
        "",
        "| M | Baseline median (us) | Candidate median (us) | Median improvement | Gate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for case in evidence["performance"]["cases"]:
        if case["status"] == "incomplete":
            lines.append(f"| {case['m']} | — | — | — | incomplete |")
        else:
            lines.append(
                f"| {case['m']} | {case['baseline_median_us']:.3f} | "
                f"{case['candidate_median_us']:.3f} | "
                f"{case['median_group_improvement_percent']:+.2f}% | {case['status']} |"
            )
    available_groups = [
        row for row in evidence["performance"]["groups"] if "improvement_percent" in row
    ]
    if available_groups:
        lines.extend(
            [
                "",
                "### 每组 ABBA",
                "",
                "| M | Group | Baseline median (us) | Candidate median (us) | Improvement | CV (A / B) | Gate |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in available_groups:
            lines.append(
                f"| {row['m']} | {row['group']} | {row['baseline_median_us']:.3f} | "
                f"{row['candidate_median_us']:.3f} | {row['improvement_percent']:+.2f}% | "
                f"{row['baseline_cv'] * 100:.2f}% / {row['candidate_cv'] * 100:.2f}% | "
                f"{'pass' if row['cv_gate_pass'] else 'unstable'} |"
            )
    lines.extend(
        [
            "",
            "Improvement = `(baseline median / candidate median - 1) × 100%`。每组把两个 A position "
            "与两个 B position 各自合并为 100 个 replay 后计算 median/CV。",
        ]
    )
    p3_gate = evidence["supporting_gates"]["p3_phase"]
    spill_gate = evidence["supporting_gates"]["dynamic_spill"]
    if p3_gate.get("gate_pass") is True:
        p3 = p3_gate["summary"]
        phase = p3["phase_comparison"]
        instrumentation = p3["instrumentation"]
        ledger = evidence["identity"]["overlay_identity"]["work_ledger"]
        lines.extend(
            [
                "",
                "## Route/Q0 机制证据",
                "",
                "Candidate 是整体 token-major P3 重构：同一 token 的 BF16 input load 与 block absmax "
                "只做一次，再按 8 个 expert 分别 quant/store；M8192 BF16 block load 从 "
                f"{ledger['bf16_block_loads_baseline']:,} 降至 "
                f"{ledger['bf16_block_loads_candidate']:,}，productive producer claim 从 "
                f"{ledger['productive_claims_baseline']:,} 降至 "
                f"{ledger['productive_claims_candidate']:,}，route metadata ownership/shuffle 也随之改变。"
                "row-allocation atomic 与 FP4/SFA store "
                "数量不变，因此本实验接受的是组合机制，不把收益拆给单一子变化。",
                "",
                "| P3 grid critical wall | Baseline | Candidate | 降低 |",
                "|---|---:|---:|---:|",
                f"| M8192 matched probe | {phase['baseline_grid_critical_wall_median_us']:.3f} µs | "
                f"{phase['candidate_grid_critical_wall_median_us']:.3f} µs | "
                f"{phase['latency_reduction_percent']:.2f}% |",
                "",
                "| Arm | REG control→probe | SMEM | STACK | LOCAL | SASS zero-spill | Probe E2E 扰动 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for arm in ARMS:
            record = instrumentation[arm]
            control = record["control_resource"]
            probe = record["probe_resource"]
            lines.append(
                f"| {arm} | {control['registers_per_thread']}→"
                f"{probe['registers_per_thread']} | "
                f"{control['static_shared_bytes_per_cta']}→"
                f"{probe['static_shared_bytes_per_cta']} B | "
                f"{control['stack_bytes_per_thread']}→{probe['stack_bytes_per_thread']} B | "
                f"{control['static_local_bytes_outside_stack']}→"
                f"{probe['static_local_bytes_outside_stack']} B | "
                f"{'pass' if record['sass_spill_gate_pass'] else 'fail'} | "
                f"{record['probe_e2e_perturbation_percent']:+.2f}% |"
            )
        lines.extend(
            [
                "",
                "P3 口径为 `max(all CTA end) - min(all CTA start)`；它只用于定位收益，"
                "不替代未插桩 E2E。",
                "",
                f"P3 provenance: [p3_phase_evidence.json](p3_phase_evidence.json), "
                f"SHA256 `{p3_gate['source_sha256']}`。SASS sidecar 对 control/probe 四个 cubin "
                "均验证为 zero-spill；SMEM gate 验证每个 arm 的 control/probe 静态 SMEM 相同。",
            ]
        )
    if spill_gate.get("gate_pass") is True:
        metrics = spill_gate["summary"]["metrics"]
        lines.extend(
            [
                "",
                "## Spill",
                "",
                "Candidate M8192 的 executed register spill/refill instruction 与 local "
                f"load/store byte counters 全为 0（{len(metrics)} 项动态指标）。",
                "",
                f"Dynamic spill provenance: [dynamic_spill_evidence.json](dynamic_spill_evidence.json), "
                f"SHA256 `{spill_gate['source_sha256']}`。",
            ]
        )
    if evidence["hard_failures"]:
        lines.extend(["", "## 硬失败", ""])
        lines.extend(f"- {item}" for item in evidence["hard_failures"])
    if evidence["missing_evidence"]:
        lines.extend(["", "## 待补证据", ""])
        lines.extend(f"- {item}" for item in evidence["missing_evidence"])
    lines.extend(
        [
            "",
            "> 性能判定以未插桩 E2E 为准；P3 matched probe 与 dynamic spill 作为独立、"
            "identity-locked 的支持证据。",
            "",
        ]
    )
    return "\n".join(lines)


CSV_FIELDS = (
    "m",
    "group",
    "position",
    "arm",
    "count",
    "position_median_us",
    "position_cv",
    "group_baseline_median_us",
    "group_candidate_median_us",
    "group_improvement_percent",
    "group_baseline_cv",
    "group_candidate_cv",
    "group_cv_gate_pass",
    "source",
    "source_sha256",
)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = args.results.resolve()
    results.mkdir(parents=True, exist_ok=True)
    evidence, manifest, csv_rows = build(results)
    evidence_path = results / "evidence.json"
    result_path = results / "result.md"
    csv_path = results / "abba_raw.csv"
    write_json(evidence_path, evidence)
    result_path.write_text(render_markdown(evidence))
    write_csv(csv_path, csv_rows)
    manifest["outputs"] = {
        "evidence.json": file_sha256(evidence_path),
        "result.md": file_sha256(result_path),
        "abba_raw.csv": file_sha256(csv_path),
    }
    write_json(results / "manifest.json", manifest)
    print(
        json.dumps(
            {"status": evidence["status"], "verdict": evidence["verdict"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
