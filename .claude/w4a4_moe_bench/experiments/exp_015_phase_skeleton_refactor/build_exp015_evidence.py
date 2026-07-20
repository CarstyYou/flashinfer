#!/usr/bin/env python3
"""Build GPU-free, fail-closed evidence for exp_015.

Inputs are the immutable artifacts emitted by ``run_exp015_arm.py``.  This
collector opens no CUDA context.  PyTorch is imported lazily only to load the
two CPU output tensors per correctness case.

Required evidence:

* baseline and candidate_v2 ``validation.json`` plus all eight ``case.json``
  files and output tensors;
* exactly 5 complete A-B-B-A groups for M=256 and M=8192.

Optional evidence:

* ``exp015.static_resource_evidence.v1``: one record per arm/distinct cubin;
* ``exp015.matched-dynamic-ncu-evidence.v1``: one matched M8192 canonical
  record per arm.  Because each arm has one cubin, these two records cover the
  complete distinct-cubin inventory.

Missing optional evidence keeps the final verdict ``pending``.  Malformed or
incomplete supplied evidence fails closed.  No-regression uses the complete
ABBA group as the bootstrap unit and the fixed -1.5% speedup boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EXP005 = ROOT.parent / "exp_005_8warp_spill_reduction"
sys.path.insert(0, str(EXP005))

import exp005_common as common  # noqa: E402


BASELINE = "baseline"
CANDIDATE = "candidate_v2"
ARMS = (BASELINE, CANDIDATE)
EXPECTED_SOURCE = {
    BASELINE: "f3c246817679d962a3f7160dbe8b9e68262c919e26e306f349200961fc4ac971",
    CANDIDATE: "b6e141179794561f2144bdec079b7e109fddfccc2db6ba0f19c22d30ea4b34ca",
}
EXPECTED_PRODUCTION = common.EXPECTED_KERNEL_SHA256
EXPECTED_GPU_UUID = "GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522"
EXPECTED_APPLICATION_CLOCK_MHZ = 2377
EXPECTED_IMAGE_ID = (
    "sha256:a4e056e1d34a5cc9387512ffa3abeed778e3dc7966633c5154d771705d8835ac"
)
EXPECTED_ORACLE_SHA256 = (
    "9c12529c20d99009ea9dfd846193337fcaf533f4b688d60b773661fe084fe0d2"
)
EXPECTED_VALIDATION_HARNESS_SHA256 = (
    "ae013ad2d2fb3ddd2c5bc6835284eb0b3d278ef24e13023c79dc76d26683e502"
)
EXPECTED_MEASUREMENT_HARNESS_SHA256 = (
    "70e7a2f8333d642e456812472cc6abca0aacdb49255f060b7ae070fc04b57da5"
)

M_VALUES = (256, 8192)
M256_FIXTURES = (
    "canonical",
    "sparse_empty",
    "exact_128",
    "tail_129",
    "hot_expert",
    "canary_gate_v2",
    "canary_up_v2",
)
CASE_SPECS = tuple((256, fixture) for fixture in M256_FIXTURES) + ((8192, "canonical"),)

GROUPS = tuple(range(5))
POSITIONS = tuple(range(4))
ABBA = (BASELINE, CANDIDATE, CANDIDATE, BASELINE)
WARMUP = 5
ITERS = 50
L2_FLUSH_BYTES = 192 << 20
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260720
NO_REGRESSION_BOUNDARY_PERCENT = -1.5

STATIC_SCHEMA = "exp015.static_resource_evidence.v1"
DYNAMIC_SCHEMA = "exp015.matched-dynamic-ncu-evidence.v1"
EXPECTED_DYNAMIC_WORK = {
    "executed_tensor_instructions": 31_162_368,
    "fp4_tensor_ops": 510_564_237_312,
}
EXPECTED_DYNAMIC_GRID = [1, 1, 110]
EXPECTED_DYNAMIC_BLOCK = [288, 1, 1]
DYNAMIC_SPILL_METRICS = (
    "spill_register_read",
    "spill_register_write",
    "spill_local_read",
    "spill_local_write",
)
DYNAMIC_WORK_METRICS = (
    "executed_tensor_instructions",
    "fp4_tensor_ops",
)


class EvidenceError(RuntimeError):
    """The supplied artifact set is structurally invalid or untraceable."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"invalid JSON {path}: {error}") from error
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    require(path.is_file(), f"missing source artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric: {value!r}",
    )
    converted = float(value)
    require(math.isfinite(converted), f"{label} is not finite: {converted}")
    if positive:
        require(converted > 0.0, f"{label} must be positive: {converted}")
    return converted


def exact_int(value: Any, expected: int, label: str) -> None:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value == expected,
        f"{label} drift: {value!r} != {expected}",
    )


def exact_nonnegative_int(value: Any, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer: {value!r}",
    )
    return value


def close_number(actual: Any, expected: float, label: str) -> None:
    value = finite_number(actual, label)
    require(
        math.isclose(value, expected, rel_tol=1.0e-12, abs_tol=1.0e-9),
        f"{label} mismatch: {value} != {expected}",
    )


def evidence_path(path: Path, results: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(results.resolve()))
    except ValueError:
        return str(resolved)


def overlay_path(results: Path, arm: str) -> Path:
    return results / "overlays" / arm / "moe_dynamic_kernel.py"


def case_path(results: Path, arm: str, m: int, fixture: str) -> Path:
    return results / "raw" / "validation" / arm / f"m{m}" / fixture / "case.json"


def validation_path(results: Path, arm: str) -> Path:
    return results / "raw" / "validation" / arm / "validation.json"


def benchmark_path(results: Path, arm: str, m: int, group: int, position: int) -> Path:
    return (
        results
        / "raw"
        / "benchmark"
        / f"m{m}"
        / f"group_{group}_position_{position}_{arm}.json"
    )


def stable_runtime_identity(
    runtime: Mapping[str, Any],
    imports: Mapping[str, Any],
    label: str,
    *,
    expected_harness_sha256: str,
) -> dict[str, Any]:
    require(isinstance(runtime, Mapping), f"{label} runtime is not an object")
    require(isinstance(imports, Mapping), f"{label} imports is not an object")
    source = runtime.get("source")
    gpu = runtime.get("gpu")
    harness = runtime.get("harness")
    require(isinstance(source, Mapping), f"{label} runtime.source is missing")
    require(isinstance(gpu, Mapping), f"{label} runtime.gpu is missing")
    require(isinstance(harness, Mapping), f"{label} runtime.harness is missing")
    required_runtime = (
        "hostname",
        "python",
        "packages",
        "torch",
        "cuda_runtime",
        "nvcc",
        "ptxas",
        "cuda_visible_devices",
        "image_digest",
        "image_id",
        "python_deps_sha256",
        "lease_id",
    )
    required_source = (
        "locked_flashinfer_commit",
        "checkout_head",
        "cutlass_commit",
        "production_kernel_sha256",
        "oracle_source_sha256",
    )
    required_gpu = (
        "uuid",
        "name",
        "pci_bus_id",
        "driver",
        "applications_graphics_clock_mhz",
        "max_graphics_clock_mhz",
        "compute_capability",
        "sm_count",
        "foreign_processes_before_cuda_context",
    )
    required_imports = (
        "flashinfer",
        "cutlass_python",
        "cutlass_python_version",
    )
    for field in required_runtime:
        require(runtime.get(field) is not None, f"{label} missing runtime.{field}")
    for field in required_source:
        require(source.get(field) is not None, f"{label} missing source.{field}")
    for field in required_gpu:
        require(gpu.get(field) is not None, f"{label} missing gpu.{field}")
    for field in required_imports:
        require(imports.get(field) is not None, f"{label} missing imports.{field}")
    require(gpu["uuid"] == EXPECTED_GPU_UUID, f"{label} GPU UUID drift")
    require(
        int(float(gpu["applications_graphics_clock_mhz"]))
        == EXPECTED_APPLICATION_CLOCK_MHZ,
        f"{label} application clock drift",
    )
    require(
        gpu["foreign_processes_before_cuda_context"] == [],
        f"{label} foreign GPU process",
    )
    require(
        runtime["image_digest"] == common.EXPECTED_IMAGE_DIGEST,
        f"{label} image digest drift",
    )
    require(runtime["image_id"] == EXPECTED_IMAGE_ID, f"{label} image ID drift")
    require(
        runtime["python_deps_sha256"] == common.EXPECTED_PYTHON_DEPS_SHA256,
        f"{label} Python dependency drift",
    )
    require(
        source["production_kernel_sha256"] == EXPECTED_PRODUCTION,
        f"{label} production source drift",
    )
    require(
        source["locked_flashinfer_commit"] == common.EXPECTED_FLASHINFER_COMMIT
        and source["checkout_head"] == common.EXPECTED_FLASHINFER_COMMIT,
        f"{label} FlashInfer commit drift",
    )
    require(
        source["cutlass_commit"] == common.EXPECTED_CUTLASS_COMMIT,
        f"{label} CUTLASS commit drift",
    )
    require(
        source["oracle_source_sha256"] == EXPECTED_ORACLE_SHA256,
        f"{label} FP32 oracle source drift",
    )
    require(
        harness.get("sha256") == expected_harness_sha256,
        f"{label} runtime harness source drift",
    )
    require(runtime["cuda_runtime"] == "13.2", f"{label} CUDA runtime drift")
    require("V13.2.78" in str(runtime["nvcc"]), f"{label} nvcc version drift")
    require(
        imports["cutlass_python_version"] == "4.6.0",
        f"{label} CuteDSL/CUTLASS Python version drift",
    )
    require(
        gpu["compute_capability"] in ([12, 0], [12, 1]),
        f"{label} compute capability drift",
    )
    require(gpu["sm_count"] == 110, f"{label} SM count drift")
    return {
        "runtime": {field: runtime[field] for field in required_runtime},
        "source": {field: source[field] for field in required_source},
        "gpu": {field: gpu[field] for field in required_gpu},
        "imports": {field: imports[field] for field in required_imports},
        "harness_sha256": harness["sha256"],
    }


def environment_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the intentionally phase-specific validation/measurement harness."""
    return {key: value for key, value in identity.items() if key != "harness_sha256"}


def validate_arm_source(
    results: Path,
    arm: str,
    runtime: Mapping[str, Any],
    imports: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    expected_hash = EXPECTED_SOURCE[arm]
    frozen_overlay = overlay_path(results, arm)
    frozen_hash = file_sha256(frozen_overlay)
    require(frozen_hash == expected_hash, f"{label} frozen overlay hash drift")
    source = runtime.get("source")
    require(isinstance(source, Mapping), f"{label} missing runtime.source")
    require(source.get("overlay_sha256") == expected_hash, f"{label} source hash drift")
    require(
        source.get("overlay") == imports.get("target_module"),
        f"{label} loaded target module differs from selected overlay",
    )
    return {
        "arm": arm,
        "expected_sha256": expected_hash,
        "frozen_overlay": evidence_path(frozen_overlay, results),
        "frozen_overlay_sha256": frozen_hash,
        "loaded_overlay": source["overlay"],
        "loaded_target_module": imports["target_module"],
    }


def load_outputs(directory: Path, records: Sequence[Mapping[str, Any]]) -> list[Any]:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise EvidenceError(
            "CPU PyTorch is required to load correctness tensors"
        ) from error
    expected = [directory / f"output_{replay}.pt" for replay in range(2)]
    observed = sorted(directory.glob("output_*.pt"))
    require(observed == expected, f"expected exactly two outputs in {directory}")
    outputs = [
        torch.load(path, map_location="cpu", weights_only=True) for path in expected
    ]
    require(
        all(isinstance(value, torch.Tensor) for value in outputs),
        f"non-tensor output in {directory}",
    )
    require(len(records) == 2, f"case output record count drift in {directory}")
    for replay, (output, record) in enumerate(zip(outputs, records, strict=True)):
        expected_hash = record.get("output_sha256")
        # Reuse the exact logical-tensor hashing contract from the runtime worker.
        contiguous = output.detach().contiguous()
        digest = hashlib.sha256()
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
        require(
            digest.hexdigest() == expected_hash, f"output_{replay} logical hash drift"
        )
    return outputs


def tensor_error(actual: Any, expected: Any) -> dict[str, float]:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise EvidenceError("CPU PyTorch is required for cross-arm errors") from error
    require(
        tuple(actual.shape) == tuple(expected.shape), "cross-arm output shape mismatch"
    )
    require(actual.dtype == expected.dtype, "cross-arm output dtype mismatch")
    require(bool(torch.isfinite(actual).all()), "non-finite actual output")
    require(bool(torch.isfinite(expected).all()), "non-finite expected output")
    actual_f = actual.float()
    expected_f = expected.float()
    error = actual_f - expected_f
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.flatten(), expected_f.flatten(), dim=0
    )
    denominator = torch.linalg.vector_norm(expected_f, dim=1).clamp_min(1.0e-12)
    token_relative = torch.linalg.vector_norm(error, dim=1) / denominator
    values = {
        "cosine_loss": max(0.0, 1.0 - float(cosine.item())),
        "relative_l2": float(
            (
                torch.linalg.vector_norm(error)
                / torch.linalg.vector_norm(expected_f).clamp_min(1.0e-12)
            ).item()
        ),
        "max_abs": float(error.abs().max().item()),
        "token_rel_l2_p99": float(torch.quantile(token_relative, 0.99).item()),
    }
    require(
        all(math.isfinite(value) and value >= 0.0 for value in values.values()),
        f"invalid tensor errors: {values}",
    )
    return values


def metric_projection(value: Mapping[str, Any], label: str) -> dict[str, float]:
    result = {}
    for metric in common.CORRECTNESS_SPECS:
        result[metric] = finite_number(value.get(metric), f"{label}.{metric}")
        require(result[metric] >= 0.0, f"{label}.{metric} is negative")
    return result


def require_metric_match(
    observed: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    observed_projection = metric_projection(observed, f"{label}.observed")
    expected_projection = metric_projection(expected, f"{label}.expected")
    for metric in common.CORRECTNESS_SPECS:
        close_number(
            observed_projection[metric],
            expected_projection[metric],
            f"{label}.{metric}",
        )


def route_projection(case: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    records = case.get("route_task_evidence")
    require(
        isinstance(records, list) and len(records) == 2,
        f"{label} route replay count drift",
    )
    projection = []
    fields = (
        "expected_task_count",
        "observed_task_tail",
        "observed_task_head",
        "terminal_head_overshoot",
        "producer_claim_count",
        "expected_pair_head",
        "observed_pair_head",
        "task_descriptor_multiset_sha256",
    )
    for replay, record in enumerate(records):
        summary = record.get("summary")
        require(isinstance(summary, Mapping), f"{label} route summary missing")
        verification = summary.get("verification")
        require(
            isinstance(verification, Mapping), f"{label} route verification missing"
        )
        require(
            bool(verification.get("gate_pass")),
            f"{label} replay {replay} route gate failed",
        )
        for field in fields:
            require(
                verification.get(field) is not None, f"{label} missing route {field}"
            )
        for field in fields[:-1]:
            exact_nonnegative_int(
                verification[field], f"{label} replay {replay} route {field}"
            )
        require(
            is_sha256(verification["task_descriptor_multiset_sha256"]),
            f"{label} replay {replay} invalid task descriptor hash",
        )
        projection.append({field: verification[field] for field in fields})
    require(
        projection[0] == projection[1], f"{label} route terminal/multiset replay drift"
    )
    return projection


def validate_case_structure(
    results: Path,
    arm: str,
    m: int,
    fixture: str,
    case: Mapping[str, Any],
    validation: Mapping[str, Any],
    label: str,
) -> None:
    require(case.get("schema") == "exp015.validation-case.v1", f"{label} schema drift")
    require(case.get("status") == "complete", f"{label} incomplete")
    require(case.get("gate_pass") is True, f"{label} runtime gate failed")
    require(case.get("arm") == arm, f"{label} arm drift")
    exact_int(case.get("m"), m, f"{label} M")
    require(case.get("fixture_kind") == fixture, f"{label} fixture drift")
    require(
        case.get("runtime_identity_sha256") == canonical_sha256(validation["runtime"]),
        f"{label} runtime identity link drift",
    )
    outputs = case.get("outputs")
    require(
        isinstance(outputs, list) and len(outputs) == 2, f"{label} replay count drift"
    )
    reference = case.get("reference")
    require(isinstance(reference, Mapping), f"{label} reference missing")
    require(reference.get("dtype") == "float32", f"{label} reference is not FP32")
    require(is_sha256(reference.get("sha256")), f"{label} invalid reference hash")
    require(
        is_sha256(reference.get("implementation_sha256")),
        f"{label} invalid oracle hash",
    )
    for replay, output in enumerate(outputs):
        require(output.get("replay") == replay, f"{label} replay index drift")
        require(
            output.get("reference_sha256") == reference["sha256"],
            f"{label} reference link drift",
        )
        require(is_sha256(output.get("output_sha256")), f"{label} invalid output hash")
        broad = output.get("broad_oracle_diagnostics")
        require(isinstance(broad, Mapping), f"{label} broad oracle diagnostics missing")
        require(
            broad.get("formal_pass") is True and broad.get("finite") is True,
            f"{label} oracle gate failed",
        )
        exact_int(output.get("sentinel_nan_remaining"), 0, f"{label} sentinel NaNs")
        metric_projection(
            output.get("reference_error", {}), f"{label}.r{replay}.reference_error"
        )
        if fixture in ("canary_gate_v2", "canary_up_v2"):
            canary = output.get("write_canary")
            require(
                isinstance(canary, Mapping) and canary.get("gate_pass") is True,
                f"{label} write canary failed",
            )
    case_dir = case_path(results, arm, m, fixture).parent
    route_records = case.get("route_task_evidence")
    require(
        isinstance(route_records, list) and len(route_records) == 2,
        f"{label} route evidence count drift",
    )
    for replay, record in enumerate(route_records):
        json_name = record.get("json")
        tensor_name = record.get("pt")
        require(
            json_name == f"workspace_replay_{replay}.json"
            and tensor_name == f"workspace_replay_{replay}.pt",
            f"{label} route raw evidence path drift",
        )
        workspace_json = case_dir / json_name
        workspace_tensor = case_dir / tensor_name
        require(
            read_json(workspace_json) == record.get("summary"),
            f"{label} inline/raw route JSON mismatch",
        )
        require(
            workspace_tensor.is_file() and workspace_tensor.stat().st_size > 0,
            f"{label} raw route tensor evidence missing",
        )
    route_projection(case, label)


def load_arm(
    results: Path,
    arm: str,
    *,
    output_loader: Callable[[Path, Sequence[Mapping[str, Any]]], list[Any]],
) -> dict[str, Any]:
    frozen_harness = results / "harness" / "run_exp015_arm_validation_v1.py"
    require(
        file_sha256(frozen_harness) == EXPECTED_VALIDATION_HARNESS_SHA256,
        "frozen validation harness hash drift",
    )
    manifest_path = validation_path(results, arm)
    manifest = read_json(manifest_path)
    label = f"{arm} validation"
    require(
        manifest.get("schema") == "exp015.arm-validation.v1", f"{label} schema drift"
    )
    require(
        manifest.get("status") == "complete" and manifest.get("gate_pass") is True,
        f"{label} failed",
    )
    require(manifest.get("arm") == arm, f"{label} arm drift")
    require(
        manifest.get("case_order") == [list(spec) for spec in CASE_SPECS],
        f"{label} case order drift",
    )
    summaries = manifest.get("cases")
    require(
        isinstance(summaries, list) and len(summaries) == len(CASE_SPECS),
        f"{label} case count drift",
    )
    imports = manifest.get("imports")
    runtime = manifest.get("runtime")
    identity = stable_runtime_identity(
        runtime,
        imports,
        label,
        expected_harness_sha256=EXPECTED_VALIDATION_HARNESS_SHA256,
    )
    source = validate_arm_source(results, arm, runtime, imports, label)

    expected_paths = {
        case_path(results, arm, m, fixture).resolve() for m, fixture in CASE_SPECS
    }
    observed_paths = {
        path.resolve()
        for path in (results / "raw" / "validation" / arm).glob("m*/*/case.json")
    }
    require(observed_paths == expected_paths, f"{label} exact case file set mismatch")

    cases = {}
    outputs = {}
    for summary, (m, fixture) in zip(summaries, CASE_SPECS, strict=True):
        path = case_path(results, arm, m, fixture)
        relative = str(path.relative_to(results))
        require(
            summary.get("m") == m and summary.get("fixture_kind") == fixture,
            f"{label} case summary order drift",
        )
        require(summary.get("path") == relative, f"{label} case path link drift")
        require(
            summary.get("sha256") == file_sha256(path), f"{label} case hash link drift"
        )
        require(summary.get("gate_pass") is True, f"{label} case summary failed")
        case = read_json(path)
        case_label = f"{arm}/m{m}/{fixture}"
        validate_case_structure(results, arm, m, fixture, case, manifest, case_label)
        loaded = output_loader(path.parent, case["outputs"])
        require(len(loaded) == 2, f"{case_label} loader did not return two outputs")
        cases[(m, fixture)] = {
            "value": case,
            "path": evidence_path(path, results),
            "sha256": file_sha256(path),
        }
        outputs[(m, fixture)] = loaded

    cubins = manifest.get("cubin_sha256")
    require(isinstance(cubins, list) and cubins, f"{label} missing cubin inventory")
    require(
        len(cubins) == len(set(cubins)) and all(is_sha256(value) for value in cubins),
        f"{label} invalid cubin inventory",
    )
    artifact_cubins = {
        item.get("sha256")
        for item in manifest.get("jit_artifacts", [])
        if str(item.get("path", "")).endswith(".cubin")
    }
    require(
        artifact_cubins == set(cubins), f"{label} cubin/artifact inventory mismatch"
    )
    for key, case_record in cases.items():
        retained = (
            case_record["value"]
            .get("artifact_stages", {})
            .get("all_retained_cubin_sha256")
        )
        require(
            isinstance(retained, list) and retained,
            f"{label} {key} missing retained cubins",
        )
        require(
            set(retained).issubset(set(cubins)),
            f"{label} {key} references unknown cubin",
        )
    return {
        "manifest": manifest,
        "manifest_path": evidence_path(manifest_path, results),
        "manifest_sha256": file_sha256(manifest_path),
        "identity": identity,
        "source": source,
        "cases": cases,
        "outputs": outputs,
        "cubins": sorted(cubins),
    }


def build_correctness(
    results: Path,
    arms: Mapping[str, Mapping[str, Any]],
    *,
    error_fn: Callable[[Any, Any], dict[str, float]],
) -> dict[str, Any]:
    baseline = arms[BASELINE]
    candidate = arms[CANDIDATE]
    require(
        baseline["identity"] == candidate["identity"],
        "cross-arm stable runtime identity drift",
    )
    cases = {}
    for m, fixture in CASE_SPECS:
        key = (m, fixture)
        base_record = baseline["cases"][key]
        candidate_record = candidate["cases"][key]
        base_case = base_record["value"]
        candidate_case = candidate_record["value"]
        identity = {}
        for field in ("case", "fixture", "weights", "reference"):
            base_value = base_case.get(field)
            candidate_value = candidate_case.get(field)
            identity[field] = {
                "baseline_sha256": canonical_sha256(base_value),
                "candidate_sha256": canonical_sha256(candidate_value),
                "equal": base_value == candidate_value,
            }
        identity_gate = all(item["equal"] for item in identity.values())

        base_outputs = baseline["outputs"][key]
        candidate_outputs = candidate["outputs"][key]
        base_self = metric_projection(
            error_fn(base_outputs[1], base_outputs[0]), "baseline_self"
        )
        candidate_self = metric_projection(
            error_fn(candidate_outputs[1], candidate_outputs[0]), "candidate_self"
        )
        require_metric_match(
            base_self,
            base_case["output_stability"],
            f"baseline/m{m}/{fixture} stored self drift",
        )
        require_metric_match(
            candidate_self,
            candidate_case["output_stability"],
            f"candidate/m{m}/{fixture} stored self drift",
        )
        comparisons = []
        for candidate_replay, candidate_output in enumerate(candidate_outputs):
            for baseline_replay, baseline_output in enumerate(base_outputs):
                comparisons.append(
                    {
                        "comparison": f"candidate_r{candidate_replay}_vs_baseline_r{baseline_replay}",
                        **metric_projection(
                            error_fn(candidate_output, baseline_output),
                            f"cross/m{m}/{fixture}",
                        ),
                    }
                )
        candidate_worst = {
            metric: max(item[metric] for item in comparisons)
            for metric in common.CORRECTNESS_SPECS
        }
        strict = common.evaluate_cross_arm_correctness(
            base_self, candidate_self, candidate_worst
        )
        baseline_route = route_projection(base_case, f"baseline/m{m}/{fixture}")
        candidate_route = route_projection(candidate_case, f"candidate/m{m}/{fixture}")
        route_parity = baseline_route == candidate_route
        gate = identity_gate and route_parity and bool(strict["gate_pass"])
        name = f"m{m}_{fixture}"
        cases[name] = {
            "m": m,
            "fixture": fixture,
            "source_trace": {
                BASELINE: {
                    "path": base_record["path"],
                    "sha256": base_record["sha256"],
                },
                CANDIDATE: {
                    "path": candidate_record["path"],
                    "sha256": candidate_record["sha256"],
                },
            },
            "cross_arm_identity": identity,
            "identity_gate": identity_gate,
            "baseline_self_drift": base_self,
            "candidate_self_drift": candidate_self,
            "candidate_vs_baseline_four_way": comparisons,
            "candidate_vs_baseline_worst": candidate_worst,
            "strict_cross_arm_gate": strict,
            "route_descriptor_terminal_parity": route_parity,
            "route_projection": baseline_route
            if route_parity
            else {
                BASELINE: baseline_route,
                CANDIDATE: candidate_route,
            },
            "gate_pass": gate,
        }
    gate = len(cases) == 8 and all(case["gate_pass"] for case in cases.values())
    return {
        "status": "pass" if gate else "reject",
        "case_count": len(cases),
        "required_case_count": 8,
        "threshold_policy": (
            "exp005_common.evaluate_cross_arm_correctness: "
            "min(cap, max(floor, 3 * baseline_self_drift))"
        ),
        "stable_runtime_identity": baseline["identity"],
        "cases": cases,
        "gate_pass": gate,
        "evidence_boundary": (
            "route exact-once remains descriptor-multiset plus terminal-head "
            "inference; no per-task consumed bitmap exists"
        ),
    }


def quantile(values: Sequence[float], q: float) -> float:
    require(bool(values), "quantile requires non-empty values")
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def arm_statistics(values: Sequence[float]) -> dict[str, float]:
    require(len(values) == 10, "each arm/M requires ten position means")
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "mean_us": mean,
        "median_us": statistics.median(values),
        "p10_us": quantile(values, 0.10),
        "p90_us": quantile(values, 0.90),
        "cv": statistics.pstdev(values) / mean,
    }


def classify_no_regression(ci_low: float, ci_high: float) -> str:
    boundary = NO_REGRESSION_BOUNDARY_PERCENT
    require(ci_low <= ci_high, "invalid performance CI order")
    if ci_high < boundary:
        return "reject"
    if ci_low >= boundary:
        return "pass"
    return "inconclusive"


def summarize_abba(m: int, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(rows) == 20, f"M={m} requires exactly 20 ABBA positions")
    by_key = {}
    by_arm = {BASELINE: [], CANDIDATE: []}
    for row in rows:
        key = (int(row["group"]), int(row["position"]))
        require(key not in by_key, f"M={m} duplicate ABBA key {key}")
        require(row["arm"] == ABBA[key[1]], f"M={m} ABBA order drift at {key}")
        by_key[key] = row
        by_arm[row["arm"]].append(float(row["sample_us"]))
    expected = {(group, position) for group in GROUPS for position in POSITIONS}
    require(set(by_key) == expected, f"M={m} incomplete ABBA key set")

    groups = []
    for group in GROUPS:
        baseline_samples = [
            float(by_key[(group, position)]["sample_us"]) for position in (0, 3)
        ]
        candidate_samples = [
            float(by_key[(group, position)]["sample_us"]) for position in (1, 2)
        ]
        baseline_us = statistics.fmean(baseline_samples)
        candidate_us = statistics.fmean(candidate_samples)
        ratio = baseline_us / candidate_us
        groups.append(
            {
                "group": group,
                "baseline_position_mean_us": baseline_samples,
                "candidate_position_mean_us": candidate_samples,
                "baseline_mean_us": baseline_us,
                "candidate_mean_us": candidate_us,
                "paired_ratio_baseline_over_candidate": ratio,
                "paired_speedup_percent": (ratio - 1.0) * 100.0,
            }
        )

    seed = BOOTSTRAP_SEED + m
    rng = random.Random(seed)
    bootstrap_speedups = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
        ratio = statistics.fmean(
            item["baseline_mean_us"] for item in sampled
        ) / statistics.fmean(item["candidate_mean_us"] for item in sampled)
        bootstrap_speedups.append((ratio - 1.0) * 100.0)
    ci_low = quantile(bootstrap_speedups, 0.025)
    ci_high = quantile(bootstrap_speedups, 0.975)
    aggregate_baseline = statistics.fmean(item["baseline_mean_us"] for item in groups)
    aggregate_candidate = statistics.fmean(item["candidate_mean_us"] for item in groups)
    aggregate_ratio = aggregate_baseline / aggregate_candidate
    return {
        "m": m,
        "groups": groups,
        "arms": {
            BASELINE: arm_statistics(by_arm[BASELINE]),
            CANDIDATE: arm_statistics(by_arm[CANDIDATE]),
        },
        "aggregate_baseline_us": aggregate_baseline,
        "aggregate_candidate_us": aggregate_candidate,
        "aggregate_ratio_baseline_over_candidate": aggregate_ratio,
        "aggregate_speedup_percent": (aggregate_ratio - 1.0) * 100.0,
        "group_bootstrap": {
            "unit": "one complete ABBA group",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": seed,
            "speedup_percent_ci95": [ci_low, ci_high],
        },
        "no_regression_boundary_percent": NO_REGRESSION_BOUNDARY_PERCENT,
        "verdict": classify_no_regression(ci_low, ci_high),
    }


def validate_measurement(
    results: Path,
    arms: Mapping[str, Mapping[str, Any]],
    *,
    arm: str,
    m: int,
    group: int,
    position: int,
) -> dict[str, Any]:
    path = benchmark_path(results, arm, m, group, position)
    value = read_json(path)
    label = f"benchmark/m{m}/g{group}/p{position}/{arm}"
    require(
        value.get("schema") == "exp015.benchmark-position.v1", f"{label} schema drift"
    )
    require(value.get("status") == "complete", f"{label} incomplete")
    require(
        value.get("arm") == arm and arm == ABBA[position], f"{label} arm/order drift"
    )
    exact_int(value.get("m"), m, f"{label} M")
    exact_int(value.get("group"), group, f"{label} group")
    exact_int(value.get("position"), position, f"{label} position")
    require(value.get("fixture_kind") == "canonical", f"{label} fixture drift")
    require(value.get("abba_order") == list(ABBA), f"{label} ABBA declaration drift")
    protocol = value.get("protocol")
    require(isinstance(protocol, Mapping), f"{label} protocol missing")
    exact_int(protocol.get("warmup"), WARMUP, f"{label} warmup")
    exact_int(protocol.get("iters"), ITERS, f"{label} iters")
    exact_int(protocol.get("l2_flush_bytes"), L2_FLUSH_BYTES, f"{label} L2 flush")
    exact_int(
        protocol.get("expected_app_clock_mhz"),
        EXPECTED_APPLICATION_CLOCK_MHZ,
        f"{label} app clock",
    )
    require(protocol.get("clock_policy") == "locked", f"{label} clock policy drift")
    require(
        protocol.get("timing")
        == "CUDA Graph external CUDA events; one sample per replay",
        f"{label} timing boundary drift",
    )
    require(
        protocol.get("process_scope")
        == "one arm/M/group/position in an independent process",
        f"{label} process scope drift",
    )
    require(
        protocol.get("jit_policy")
        == (
            "reuse one immutable, correctness-validated per-arm JIT root; "
            "artifact-set and cubin hashes are checked before and after"
        ),
        f"{label} JIT policy drift",
    )

    samples = value.get("samples_us")
    require(
        isinstance(samples, list) and len(samples) == ITERS,
        f"{label} raw sample count drift",
    )
    samples = [
        finite_number(sample, f"{label} sample", positive=True) for sample in samples
    ]
    expected_stats = {
        "count": ITERS,
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "p10": quantile(samples, 0.10),
        "p90": quantile(samples, 0.90),
        "min": min(samples),
        "max": max(samples),
    }
    observed_stats = value.get("statistics_us")
    require(isinstance(observed_stats, Mapping), f"{label} statistics missing")
    exact_int(observed_stats.get("count"), ITERS, f"{label} statistics count")
    for field in ("mean", "median", "p10", "p90", "min", "max"):
        close_number(
            observed_stats.get(field),
            expected_stats[field],
            f"{label} statistics {field}",
        )
    close_number(value.get("sample_us"), expected_stats["mean"], f"{label} sample_us")

    runtime = value.get("runtime")
    imports = value.get("imports")
    identity = stable_runtime_identity(
        runtime,
        imports,
        label,
        expected_harness_sha256=EXPECTED_MEASUREMENT_HARNESS_SHA256,
    )
    require(
        environment_identity(identity) == environment_identity(arms[arm]["identity"]),
        f"{label} stable runtime environment drift",
    )
    validate_arm_source(results, arm, runtime, imports, label)
    canonical_case = arms[arm]["cases"][(m, "canonical")]["value"]
    require(
        value.get("fixture") == canonical_case.get("fixture"),
        f"{label} fixture identity drift",
    )
    require(
        value.get("weights") == canonical_case.get("weights"),
        f"{label} weight identity drift",
    )
    cubins = value.get("cubin_sha256")
    require(
        isinstance(cubins, list) and sorted(cubins) == arms[arm]["cubins"],
        f"{label} cubin identity drift",
    )
    artifacts = value.get("jit_artifacts")
    require(isinstance(artifacts, list), f"{label} JIT artifacts missing")
    artifact_cubins = {
        item.get("sha256")
        for item in artifacts
        if str(item.get("path", "")).endswith(".cubin")
    }
    require(artifact_cubins == set(cubins), f"{label} JIT/cubin inventory drift")
    jit_hash = value.get("jit_artifact_set_sha256")
    require(is_sha256(jit_hash), f"{label} invalid JIT artifact-set hash")
    require(
        jit_hash == canonical_sha256(artifacts),
        f"{label} JIT artifact-set hash mismatch",
    )
    require(
        jit_hash == arms[arm]["manifest"].get("jit_artifact_set_sha256"),
        f"{label} JIT artifact-set differs from validated arm",
    )
    compile_identity = value.get("compile_identity")
    require(
        isinstance(compile_identity, Mapping)
        and compile_identity.get("compiled_max_active_clusters") == [110],
        f"{label} compile identity drift",
    )
    require(is_sha256(value.get("output_sha256")), f"{label} invalid output hash")
    gpu_after = value.get("gpu_after")
    require(isinstance(gpu_after, Mapping), f"{label} gpu_after missing")
    require(gpu_after.get("uuid") == EXPECTED_GPU_UUID, f"{label} gpu_after UUID drift")
    require(
        int(float(gpu_after.get("applications_graphics_clock_mhz")))
        == EXPECTED_APPLICATION_CLOCK_MHZ,
        f"{label} gpu_after clock drift",
    )
    return {
        "arm": arm,
        "m": m,
        "group": group,
        "position": position,
        "sample_us": expected_stats["mean"],
        "raw_replays": ITERS,
        "jit_root": runtime.get("jit_root"),
        "cubin_sha256": sorted(cubins),
        "path": evidence_path(path, results),
        "sha256": file_sha256(path),
    }


def build_performance(
    results: Path, arms: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    measurement_harness = ROOT / "run_exp015_arm.py"
    require(
        file_sha256(measurement_harness) == EXPECTED_MEASUREMENT_HARNESS_SHA256,
        "measurement harness source hash drift",
    )
    rows = []
    expected_paths = set()
    for m in M_VALUES:
        for group in GROUPS:
            for position, arm in enumerate(ABBA):
                expected_paths.add(
                    benchmark_path(results, arm, m, group, position).resolve()
                )
    observed_paths = set()
    benchmark_root = results / "raw" / "benchmark"
    if benchmark_root.is_dir():
        observed_paths = {path.resolve() for path in benchmark_root.glob("m*/*.json")}
    require(observed_paths == expected_paths, "benchmark exact file set mismatch")

    for m in M_VALUES:
        for group in GROUPS:
            for position, arm in enumerate(ABBA):
                rows.append(
                    validate_measurement(
                        results,
                        arms,
                        arm=arm,
                        m=m,
                        group=group,
                        position=position,
                    )
                )
    require(len(rows) == 40, "exp015 requires exactly 40 benchmark positions")
    expected_jit_root = {
        arm: arms[arm]["manifest"]["runtime"]["jit_root"] for arm in ARMS
    }
    require(
        expected_jit_root[BASELINE] != expected_jit_root[CANDIDATE],
        "baseline and candidate share a JIT root",
    )
    require(
        all(
            isinstance(row["jit_root"], str)
            and row["jit_root"] == expected_jit_root[row["arm"]]
            for row in rows
        ),
        "benchmark did not reuse the registered per-arm validation JIT root",
    )

    cases = {}
    for m in M_VALUES:
        case_rows = [row for row in rows if row["m"] == m]
        cases[f"m{m}"] = summarize_abba(m, case_rows)
    verdicts = [case["verdict"] for case in cases.values()]
    if "reject" in verdicts:
        status = "reject"
    elif "inconclusive" in verdicts:
        status = "inconclusive"
    else:
        status = "pass"
    return {
        "status": status,
        "position_count": len(rows),
        "protocol": {
            "m_values": list(M_VALUES),
            "groups": len(GROUPS),
            "positions_per_group": len(POSITIONS),
            "registered_order": list(ABBA),
            "warmup": WARMUP,
            "iters": ITERS,
            "l2_flush_bytes": L2_FLUSH_BYTES,
            "speedup_definition": "baseline_us / candidate_us - 1",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_unit": "one complete ABBA group",
            "no_regression_boundary_percent": NO_REGRESSION_BOUNDARY_PERCENT,
        },
        "cases": cases,
        "raw_position_sources": rows,
        "gate_pass": status == "pass",
        "evidence_boundary": (
            "the 50 graph replays inside a position are raw observations, not "
            "independent bootstrap units"
        ),
    }


def distinct_cubin_inventory(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    inventory: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for cubin in arms[arm]["cubins"]:
            record = inventory.setdefault(
                cubin, {"cubin_sha256": cubin, "arms": [], "cases": []}
            )
            record["arms"].append(arm)
            record["cases"].extend(
                {
                    "arm": arm,
                    "m": m,
                    "fixture": fixture,
                }
                for (m, fixture), case in arms[arm]["cases"].items()
                if cubin
                in case["value"]["artifact_stages"]["all_retained_cubin_sha256"]
            )
    return {
        "count": len(inventory),
        "cubins": [inventory[key] for key in sorted(inventory)],
        "by_arm": {arm: arms[arm]["cubins"] for arm in ARMS},
    }


def validate_static_resource(
    path: Path | None,
    expected_by_arm: Mapping[str, Sequence[str]],
    results: Path,
) -> dict[str, Any]:
    if path is None:
        return {
            "status": "pending",
            "reason": "static_resource_evidence.json was not supplied",
            "gate_pass": None,
        }
    value = read_json(path)
    require(value.get("schema") == STATIC_SCHEMA, "static resource schema drift")
    require(set(expected_by_arm) == set(ARMS), "static expected-arm coverage drift")
    arm_records = value.get("arms")
    require(
        isinstance(arm_records, Mapping)
        and set(arm_records) == {"baseline", "candidate"},
        "static resource arm coverage drift",
    )
    require(
        all(len(expected_by_arm[arm]) == 1 for arm in ARMS),
        "static collector must be extended for more than one cubin per arm",
    )
    expected_cubins = {cubin for arm in ARMS for cubin in expected_by_arm[arm]}
    require(
        all(is_sha256(cubin) for cubin in expected_cubins),
        "invalid expected static cubin hash",
    )
    observed_cubins = {
        record.get("cubin", {}).get("sha256") for record in arm_records.values()
    }
    require(
        observed_cubins == expected_cubins,
        "static resource distinct cubin coverage mismatch",
    )

    records = []
    normalized_by_arm = {}
    for arm in ARMS:
        source_label = "baseline" if arm == BASELINE else "candidate"
        entry = arm_records[source_label]
        require(isinstance(entry, Mapping), f"{arm} static record is not an object")
        cubin = entry.get("cubin", {}).get("sha256")
        require(cubin == expected_by_arm[arm][0], f"{arm} static cubin drift")
        require(entry.get("label") == source_label, f"{arm} static label drift")
        resource = entry.get("resource")
        sass = entry.get("sass")
        require(
            isinstance(resource, Mapping), f"{cubin} static resource section missing"
        )
        require(isinstance(sass, Mapping), f"{cubin} static SASS section missing")
        require(
            isinstance(entry.get("kernel_symbol"), str)
            and bool(entry["kernel_symbol"].strip()),
            f"{cubin} kernel symbol is missing",
        )
        counts = sass.get("selected_instruction_counts")
        require(isinstance(counts, Mapping), f"{cubin} selected SASS counts missing")
        registers = exact_nonnegative_int(
            resource.get("registers_per_thread"), f"{cubin} REG"
        )
        stack = exact_nonnegative_int(
            resource.get("stack_bytes_per_thread"), f"{cubin} STACK"
        )
        local = exact_nonnegative_int(
            resource.get("local_bytes_outside_stack"), f"{cubin} LOCAL"
        )
        ldl = exact_nonnegative_int(counts.get("ldl"), f"{cubin} LDL")
        stl = exact_nonnegative_int(counts.get("stl"), f"{cubin} STL")
        omma = exact_nonnegative_int(counts.get("omma"), f"{cubin} OMMA")
        call = exact_nonnegative_int(counts.get("call"), f"{cubin} CALL")
        ret = exact_nonnegative_int(counts.get("ret"), f"{cubin} RET")
        checks = {
            "registers_at_most_160": registers <= 160,
            "stack_zero": stack == 0,
            "local_zero": local == 0,
            "ldl_zero": ldl == 0,
            "stl_zero": stl == 0,
            "omma_exactly_448": omma == 448,
        }
        recorded_gates = entry.get("gates")
        require(isinstance(recorded_gates, Mapping), f"{cubin} recorded gates missing")
        require(
            recorded_gates.get("checks") == checks,
            f"{cubin} recorded/recomputed static checks mismatch",
        )
        require(
            bool(recorded_gates.get("pass")) == all(checks.values()),
            f"{cubin} recorded/recomputed static gate mismatch",
        )
        require(
            entry.get("status") == ("pass" if all(checks.values()) else "fail"),
            f"{cubin} static status mismatch",
        )
        normalized_by_arm[arm] = {
            "call": call,
            "ret": ret,
            "kernel_symbol": entry["kernel_symbol"],
        }
        records.append(
            {
                "arm": arm,
                "cubin_sha256": cubin,
                "kernel_symbol": entry.get("kernel_symbol"),
                "values": {
                    "registers_per_thread": registers,
                    "stack_bytes_per_thread": stack,
                    "static_local_bytes_outside_stack": local,
                    "ldl_instruction_count": ldl,
                    "stl_instruction_count": stl,
                    "omma_static_instruction_count": omma,
                    "call_instruction_count": call,
                    "ret_instruction_count": ret,
                },
                "checks": checks,
                "gate_pass": all(checks.values()),
            }
        )
    call_frame_checks = {
        "candidate_adds_no_call": normalized_by_arm[CANDIDATE]["call"]
        <= normalized_by_arm[BASELINE]["call"],
        "candidate_adds_no_ret": normalized_by_arm[CANDIDATE]["ret"]
        <= normalized_by_arm[BASELINE]["ret"],
    }
    comparison = value.get("comparison")
    require(isinstance(comparison, Mapping), "static comparison section missing")
    require(
        comparison.get("checks") == call_frame_checks,
        "static recorded/recomputed call-frame checks mismatch",
    )
    require(
        bool(comparison.get("pass")) == all(call_frame_checks.values()),
        "static recorded/recomputed call-frame gate mismatch",
    )
    gate = all(record["gate_pass"] for record in records) and all(
        call_frame_checks.values()
    )
    require(
        value.get("status") == ("pass" if gate else "fail"),
        "static top-level status mismatch",
    )
    require(value.get("errors") in (None, []), "static collector reported errors")
    return {
        "status": "pass" if gate else "reject",
        "source": evidence_path(path, results),
        "source_sha256": file_sha256(path),
        "records": records,
        "no_extra_call_frame_checks": call_frame_checks,
        "gate_pass": gate,
    }


def validate_dynamic_ncu(
    path: Path | None,
    arms: Mapping[str, Mapping[str, Any]],
    results: Path,
) -> dict[str, Any]:
    if path is None:
        return {
            "status": "pending",
            "reason": "dynamic_ncu_evidence.json was not supplied",
            "gate_pass": None,
        }
    value = read_json(path)
    require(value.get("schema") == DYNAMIC_SCHEMA, "dynamic NCU schema drift")
    require(set(arms) == set(ARMS), "dynamic NCU expected-arm coverage drift")
    scope = value.get("scope")
    require(isinstance(scope, Mapping), "dynamic NCU scope is missing")
    require(
        scope.get("m") == 8192 and scope.get("fixture") == "canonical",
        "dynamic NCU scope case drift",
    )
    require(scope.get("gpu_uuid") == EXPECTED_GPU_UUID, "dynamic NCU scope GPU drift")
    require(scope.get("grid") == EXPECTED_DYNAMIC_GRID, "dynamic NCU scope grid drift")
    require(
        scope.get("block") == EXPECTED_DYNAMIC_BLOCK,
        "dynamic NCU scope block drift",
    )
    require(
        scope.get("expected_exp008_work_ledger") == EXPECTED_DYNAMIC_WORK,
        "dynamic NCU expected-work ledger drift",
    )
    entries = value.get("records")
    require(isinstance(entries, list), "dynamic NCU records must be a list")
    require(len(entries) == len(ARMS), "dynamic NCU record count drift")
    by_arm = {}
    for entry in entries:
        require(isinstance(entry, Mapping), "dynamic NCU entry is not an object")
        arm = entry.get("arm")
        require(arm in ARMS, f"dynamic NCU arm drift: {arm}")
        require(arm not in by_arm, f"duplicate dynamic NCU record {arm}")
        require(
            entry.get("m") == 8192 and entry.get("fixture") == "canonical",
            f"dynamic NCU case drift: {arm}",
        )
        require(
            entry.get("source_sha256") == EXPECTED_SOURCE[arm],
            f"dynamic NCU source drift: {arm}",
        )
        cubin = entry.get("cubin_sha256")
        require(
            arms[arm]["cubins"] == [cubin],
            f"dynamic NCU cubin/arm mismatch: {arm}",
        )
        require(
            entry.get("jit_artifact_set_sha256")
            == arms[arm]["manifest"].get("jit_artifact_set_sha256"),
            f"dynamic NCU JIT artifact-set mismatch: {arm}",
        )
        require(
            entry.get("gpu_uuid") == EXPECTED_GPU_UUID,
            f"dynamic NCU GPU mismatch: {arm}",
        )
        launch = entry.get("observed_launch")
        require(isinstance(launch, Mapping), f"dynamic NCU launch missing: {arm}")
        require(
            launch.get("grid") == EXPECTED_DYNAMIC_GRID
            and launch.get("block") == EXPECTED_DYNAMIC_BLOCK,
            f"dynamic NCU launch shape drift: {arm}",
        )
        require(
            "MoEDynamicKernel" in str(launch.get("kernel_symbol", "")),
            f"dynamic NCU kernel symbol drift: {arm}",
        )
        metrics = entry.get("metrics")
        require(isinstance(metrics, Mapping), f"dynamic NCU metrics missing for {arm}")
        normalized = {}
        for metric in DYNAMIC_SPILL_METRICS + DYNAMIC_WORK_METRICS:
            normalized[metric] = exact_nonnegative_int(
                metrics.get(metric), f"dynamic NCU {arm} {metric}"
            )
        artifacts = entry.get("artifacts")
        require(isinstance(artifacts, Mapping), f"dynamic NCU artifacts missing: {arm}")
        require(
            set(artifacts)
            == {"capture_identity", "profile_target", "ncu_report", "native_raw"},
            f"dynamic NCU artifact set drift: {arm}",
        )
        artifact_trace = {}
        for artifact, source in artifacts.items():
            require(
                isinstance(source, str) and bool(source),
                f"dynamic NCU artifact path missing: {arm}/{artifact}",
            )
            artifact_path = (results / source).resolve()
            require(
                artifact_path.is_relative_to(results.resolve()),
                f"dynamic NCU artifact escapes results: {arm}/{artifact}",
            )
            artifact_trace[artifact] = {
                "path": evidence_path(artifact_path, results),
                "sha256": file_sha256(artifact_path),
            }
        require(
            entry.get("source_record") == artifacts["native_raw"],
            f"dynamic NCU source record drift: {arm}",
        )
        by_arm[arm] = {
            "arm": arm,
            "m": 8192,
            "fixture": "canonical",
            "source_sha256": entry["source_sha256"],
            "cubin_sha256": cubin,
            "jit_artifact_set_sha256": entry["jit_artifact_set_sha256"],
            "gpu_uuid": entry["gpu_uuid"],
            "observed_launch": dict(launch),
            "metrics": normalized,
            "source_record": entry.get("source_record"),
            "artifacts": artifact_trace,
        }
    require(set(by_arm) == set(ARMS), "dynamic NCU arm coverage mismatch")
    covered_cubins = {record["cubin_sha256"] for record in by_arm.values()}
    expected_cubins = {cubin for arm in ARMS for cubin in arms[arm]["cubins"]}
    require(
        covered_cubins == expected_cubins,
        "dynamic NCU distinct cubin coverage mismatch",
    )

    spill_checks = {
        arm: all(
            by_arm[arm]["metrics"][metric] == 0 for metric in DYNAMIC_SPILL_METRICS
        )
        for arm in ARMS
    }
    work_checks = {
        metric: by_arm[BASELINE]["metrics"][metric]
        == by_arm[CANDIDATE]["metrics"][metric]
        for metric in DYNAMIC_WORK_METRICS
    }
    ledger_checks = {
        f"{arm}:{metric}": by_arm[arm]["metrics"][metric] == expected
        for arm in ARMS
        for metric, expected in EXPECTED_DYNAMIC_WORK.items()
    }
    checks = value.get("checks")
    recomputed_checks = {
        "zero_dynamic_spill": spill_checks,
        "pairwise_tensor_work_identity": work_checks,
        "exp008_tensor_work_identity": ledger_checks,
    }
    require(
        checks == recomputed_checks,
        "dynamic NCU recorded/recomputed checks mismatch",
    )
    gate = (
        all(spill_checks.values())
        and all(work_checks.values())
        and all(ledger_checks.values())
    )
    require(value.get("gate_pass") is gate, "dynamic NCU top-level gate mismatch")
    require(
        value.get("status") == ("pass" if gate else "reject"),
        "dynamic NCU top-level status mismatch",
    )
    return {
        "status": "pass" if gate else "reject",
        "source": evidence_path(path, results),
        "source_sha256": file_sha256(path),
        "scope": {
            "m": 8192,
            "fixture": "canonical",
            "gpu_uuid": EXPECTED_GPU_UUID,
            "grid": EXPECTED_DYNAMIC_GRID,
            "block": EXPECTED_DYNAMIC_BLOCK,
        },
        "records": [by_arm[arm] for arm in ARMS],
        "zero_dynamic_spill_checks": spill_checks,
        "tensor_work_parity_checks": work_checks,
        "exp008_tensor_work_identity_checks": ledger_checks,
        "gate_pass": gate,
    }


def final_verdict(components: Mapping[str, Mapping[str, Any]]) -> str:
    statuses = {name: value["status"] for name, value in components.items()}
    if "reject" in statuses.values():
        return "reject"
    if "pending" in statuses.values():
        return "pending"
    if "inconclusive" in statuses.values():
        return "inconclusive"
    require(set(statuses.values()) == {"pass"}, f"unknown component status: {statuses}")
    return "pass"


def collect(
    results: Path,
    *,
    static_resource_path: Path | None = None,
    dynamic_ncu_path: Path | None = None,
    output_loader: Callable[
        [Path, Sequence[Mapping[str, Any]]], list[Any]
    ] = load_outputs,
    error_fn: Callable[[Any, Any], dict[str, float]] = tensor_error,
) -> dict[str, Any]:
    results = results.resolve()
    arms = {arm: load_arm(results, arm, output_loader=output_loader) for arm in ARMS}
    correctness = build_correctness(results, arms, error_fn=error_fn)
    performance = build_performance(results, arms)
    cubins = distinct_cubin_inventory(arms)
    static = validate_static_resource(static_resource_path, cubins["by_arm"], results)
    dynamic = validate_dynamic_ncu(dynamic_ncu_path, arms, results)
    components = {
        "correctness_and_route": correctness,
        "performance_no_regression": performance,
        "static_resource": static,
        "dynamic_ncu": dynamic,
    }
    verdict = final_verdict(components)
    source_identity = {
        arm: {
            "validation_manifest": arms[arm]["manifest_path"],
            "validation_manifest_sha256": arms[arm]["manifest_sha256"],
            "source": arms[arm]["source"],
            "cubin_sha256": arms[arm]["cubins"],
        }
        for arm in ARMS
    }
    output = {
        "schema": "exp015.evidence.v1",
        "comparison": "exp008 accepted baseline vs phase-skeleton candidate_v2",
        "expected_source_sha256": EXPECTED_SOURCE,
        "source_identity": source_identity,
        "distinct_cubin_inventory": cubins,
        "components": components,
        "final_verdict": verdict,
        "gate_pass": verdict == "pass",
        "evidence_boundary": [
            "correctness thresholds are baseline-derived and use four cross-replay comparisons",
            "performance bootstrap resamples five complete ABBA groups, not 50 inner replays",
            "missing static or dynamic NCU evidence keeps the final verdict pending",
        ],
    }
    output["evidence_sha256"] = canonical_sha256(output)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=RESULTS / "evidence.json")
    parser.add_argument("--static-resource-evidence", type=Path)
    parser.add_argument("--dynamic-ncu-evidence", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = args.results.resolve()
    output_path = args.output.resolve()
    static_path = (
        args.static_resource_evidence.resolve()
        if args.static_resource_evidence is not None
        else None
    )
    dynamic_path = (
        args.dynamic_ncu_evidence.resolve()
        if args.dynamic_ncu_evidence is not None
        else None
    )
    try:
        output = collect(
            results,
            static_resource_path=static_path,
            dynamic_ncu_path=dynamic_path,
        )
    except Exception as error:
        failure = {
            "schema": "exp015.evidence.v1",
            "final_verdict": "reject",
            "gate_pass": False,
            "structural_error": f"{type(error).__name__}: {error}",
            "results": str(results),
        }
        write_json(output_path, failure)
        print(json.dumps(failure, sort_keys=True))
        return 2
    write_json(output_path, output)
    print(
        json.dumps(
            {
                "final_verdict": output["final_verdict"],
                "gate_pass": output["gate_pass"],
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )
    if output["final_verdict"] == "pass":
        return 0
    if output["final_verdict"] == "reject":
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
