#!/usr/bin/env python3
"""Build fail-closed, GPU-free validation and paired evidence for exp_014."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"
BASELINE = "baseline_4warp_scatter"
CANDIDATE = "candidate_8warp_scatter"
ARMS = (BASELINE, CANDIDATE)
ABBA = (BASELINE, CANDIDATE, CANDIDATE, BASELINE)
M_VALUES = (256, 512, 1024, 2048, 4096, 8192)
GROUPS = tuple(range(5))
POSITIONS = (0, 1, 2, 3)
WARMUP = 5
ITERS = 50
L2_FLUSH_BYTES = 192 << 20
EXPECTED_GPU_UUID = "GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522"
EXPECTED_CLOCK_MHZ = 2377
EXPECTED_HARNESS_SHA256 = (
    "6498f3e70c1834825c6ceed21e0d47f7a43bee7bb8ef902bd2fc579e2333d81b"
)
EXPECTED_OVERLAY_SHA256 = {
    BASELINE: "b6e141179794561f2144bdec079b7e109fddfccc2db6ba0f19c22d30ea4b34ca",
    CANDIDATE: "c88cef63492b60c0a77484b50f6400b83a103d168e1535b78972341503810184",
}
EXPECTED_JIT_ROOT = {
    BASELINE: os.environ.get(
        "EXP014_BASELINE_JIT",
        "/home/xiy/workspace/exp014_scatter_8warp_jit/baseline_4warp_scatter",
    ),
    CANDIDATE: os.environ.get(
        "EXP014_CANDIDATE_JIT",
        "/home/xiy/workspace/exp014_scatter_8warp_jit/candidate_8warp_scatter",
    ),
}
EXPECTED_CASES = (
    (256, "canonical"),
    (512, "canonical"),
    (1024, "canonical"),
    (2048, "canonical"),
    (4096, "canonical"),
    (8192, "canonical"),
    (256, "sparse_empty"),
    (256, "exact_128"),
    (256, "tail_129"),
    (256, "hot_expert"),
    (256, "canary_gate_v2"),
    (256, "canary_up_v2"),
)


class EvidenceError(RuntimeError):
    """Input evidence is missing, malformed, or not identity matched."""


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
    require(path.is_file(), f"missing artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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
        require(result > 0.0, f"{label} is not positive")
    return result


def exact_int(value: Any, expected: int, label: str) -> None:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value == expected,
        f"{label} drift: {value!r} != {expected}",
    )


def relative(path: Path, results: Path) -> str:
    """Serialize an artifact path relative to the relocatable results root."""
    try:
        return Path(os.path.relpath(path.resolve(), results.resolve())).as_posix()
    except ValueError as error:
        raise EvidenceError(
            f"cannot express artifact relative to results root: {path}"
        ) from error


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def overlay_path(results: Path, arm: str) -> Path:
    return results / "overlays" / arm / "moe_dynamic_kernel.py"


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


def case_path(results: Path, arm: str, m: int, fixture: str) -> Path:
    return results / "raw" / "validation" / arm / f"m{m}" / fixture / "case.json"


def validate_registered_sources(results: Path) -> dict[str, Any]:
    harness_hash = file_sha256(ROOT / "run_exp014_arm.py")
    require(
        harness_hash == EXPECTED_HARNESS_SHA256,
        f"runtime harness drift: {harness_hash}",
    )
    identity_path = results / "overlays" / "identity.json"
    identity = read_json(identity_path)
    require(
        identity.get("schema") == "exp014.kernel-overlay-identity.v1",
        "overlay identity schema drift",
    )
    require(
        identity.get("path_base") == "w4a4_moe_bench_root",
        "overlay identity path base drift",
    )
    expected_identity_paths = {
        "source": "moe_dynamic_kernel_opt.py",
        "baseline": (
            "experiments/exp_014_scatter_8warp/results/overlays/"
            "baseline_4warp_scatter/moe_dynamic_kernel.py"
        ),
        "candidate": (
            "experiments/exp_014_scatter_8warp/results/overlays/"
            "candidate_8warp_scatter/moe_dynamic_kernel.py"
        ),
        "diff": (
            "experiments/exp_014_scatter_8warp/results/overlays/"
            "candidate_8warp_scatter.diff"
        ),
    }
    for field, expected in expected_identity_paths.items():
        require(
            identity.get(field) == expected,
            f"overlay identity {field} path drift",
        )
    require(
        identity.get("baseline_sha256") == EXPECTED_OVERLAY_SHA256[BASELINE],
        "baseline registered hash drift",
    )
    require(
        identity.get("candidate_sha256") == EXPECTED_OVERLAY_SHA256[CANDIDATE],
        "candidate registered hash drift",
    )
    overlays = {}
    for arm in ARMS:
        path = overlay_path(results, arm)
        observed = file_sha256(path)
        require(
            observed == EXPECTED_OVERLAY_SHA256[arm],
            f"{arm} frozen overlay drift: {observed}",
        )
        overlays[arm] = {
            "path": relative(path, results),
            "sha256": observed,
        }
    require(
        overlays[BASELINE]["sha256"] != overlays[CANDIDATE]["sha256"],
        "baseline and candidate overlays are identical",
    )
    return {
        "harness": {
            "path": relative(ROOT / "run_exp014_arm.py", results),
            "sha256": harness_hash,
        },
        "identity": {
            "path": relative(identity_path, results),
            "sha256": file_sha256(identity_path),
        },
        "overlays": overlays,
    }


def validate_ownership(results: Path) -> dict[str, Any]:
    path = results / "ownership_gate.json"
    value = read_json(path)
    require(
        value.get("schema") == "exp014.scatter-ownership-gate.v1",
        "ownership schema drift",
    )
    require(value.get("status") == "pass", "ownership gate failed")
    require(
        value.get("mapping") == "warp_m=(warp>>1)*32, warp_n=(warp&1)*64",
        "ownership mapping drift",
    )
    exact_int(value.get("vector_width"), 8, "ownership vector width")
    cases = value.get("cases")
    require(isinstance(cases, list) and len(cases) == 48, "ownership case count drift")
    full_tiles = [case for case in cases if case.get("valid_rows") == 128]
    require(len(full_tiles) == 4, "ownership full-tile coverage drift")
    for index, case in enumerate(cases):
        require(isinstance(case, Mapping), f"ownership case {index} is malformed")
        require(case.get("exactly_one_owner") is True, "ownership is not exact-once")
        exact_int(case.get("invalid_writes"), 0, "ownership invalid writes")
        exact_int(
            case.get("observed_elements"),
            int(case.get("expected_elements", -1)),
            "ownership element count",
        )
    require(
        all(case.get("active_warps") == list(range(8)) for case in full_tiles),
        "full Scatter tile does not engage all eight math warps",
    )
    return {
        "status": "pass",
        "case_count": len(cases),
        "full_tile_active_warps": list(range(8)),
        "source": relative(path, results),
        "source_sha256": file_sha256(path),
        "gate_pass": True,
    }


def stable_runtime(
    runtime: Mapping[str, Any], imports: Mapping[str, Any]
) -> dict[str, Any]:
    source = runtime.get("source")
    gpu = runtime.get("gpu")
    require(isinstance(source, Mapping), "runtime.source is missing")
    require(isinstance(gpu, Mapping), "runtime.gpu is missing")
    require(isinstance(imports, Mapping), "imports is missing")
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
        require(runtime.get(field) is not None, f"missing runtime.{field}")
    for field in required_source:
        require(source.get(field) is not None, f"missing runtime.source.{field}")
    for field in required_gpu:
        require(gpu.get(field) is not None, f"missing runtime.gpu.{field}")
    for field in required_imports:
        require(imports.get(field) is not None, f"missing imports.{field}")
    require(gpu["uuid"] == EXPECTED_GPU_UUID, "GPU UUID drift")
    exact_int(gpu["sm_count"], 110, "GPU SM count")
    require(
        int(float(gpu["applications_graphics_clock_mhz"])) == EXPECTED_CLOCK_MHZ,
        "GPU application clock drift",
    )
    require(gpu["foreign_processes_before_cuda_context"] == [], "foreign GPU process")
    return {
        "runtime": {field: runtime[field] for field in required_runtime},
        "source": {field: source[field] for field in required_source},
        "gpu": {field: gpu[field] for field in required_gpu},
        "imports": {field: imports[field] for field in required_imports},
    }


def validate_case(
    case: Mapping[str, Any], case_file: Path, arm: str, m: int, fixture: str
) -> None:
    label = f"{arm}/m{m}/{fixture}"
    require(case.get("schema") == "exp014.validation-case.v1", f"{label} schema drift")
    require(case.get("status") == "complete", f"{label} is incomplete")
    require(case.get("gate_pass") is True, f"{label} gate failed")
    require(case.get("arm") == arm, f"{label} arm drift")
    exact_int(case.get("m"), m, f"{label} M")
    require(case.get("fixture_kind") == fixture, f"{label} fixture drift")
    outputs = case.get("outputs")
    require(isinstance(outputs, list) and len(outputs) == 2, f"{label} replay drift")
    output_files = sorted(case_file.parent.glob("output_*.pt"))
    require(
        output_files
        == [
            case_file.parent / "output_0.pt",
            case_file.parent / "output_1.pt",
        ]
        and all(path.stat().st_size > 0 for path in output_files),
        f"{label} raw output tensor set drift",
    )
    reference = case.get("reference")
    require(isinstance(reference, Mapping), f"{label} reference missing")
    require(reference.get("dtype") == "float32", f"{label} reference is not FP32")
    require(is_sha256(reference.get("sha256")), f"{label} reference hash invalid")
    require(
        is_sha256(reference.get("implementation_sha256")),
        f"{label} oracle implementation hash invalid",
    )
    for replay, output in enumerate(outputs):
        require(output.get("replay") == replay, f"{label} replay order drift")
        require(is_sha256(output.get("output_sha256")), f"{label} output hash invalid")
        require(
            output.get("reference_sha256") == reference["sha256"],
            f"{label} output/reference link drift",
        )
        broad = output.get("broad_oracle_diagnostics")
        require(isinstance(broad, Mapping), f"{label} oracle diagnostics missing")
        require(
            broad.get("formal_pass") is True and broad.get("finite") is True,
            f"{label} FP32 oracle gate failed",
        )
        exact_int(output.get("sentinel_nan_remaining"), 0, f"{label} sentinel NaN")
        if fixture.startswith("canary_"):
            canary = output.get("write_canary")
            require(
                isinstance(canary, Mapping) and canary.get("gate_pass") is True,
                f"{label} write canary failed",
            )
    stability = case.get("output_stability_gate")
    require(
        isinstance(stability, Mapping) and stability.get("gate_pass") is True,
        f"{label} replay stability failed",
    )
    route = case.get("route_task_evidence")
    require(
        isinstance(route, list) and len(route) == 2, f"{label} route evidence drift"
    )
    for replay, record in enumerate(route):
        require(record.get("replay") == replay, f"{label} route replay order drift")
        require(
            record.get("json") == f"workspace_replay_{replay}.json"
            and record.get("pt") == f"workspace_replay_{replay}.pt",
            f"{label} route artifact link drift",
        )
        summary = record.get("summary")
        require(isinstance(summary, Mapping), f"{label} route summary missing")
        verification = summary.get("verification")
        require(
            isinstance(verification, Mapping) and verification.get("gate_pass") is True,
            f"{label} route/task gate failed",
        )
        workspace_json = case_file.parent / record["json"]
        workspace_tensor = case_file.parent / record["pt"]
        require(
            read_json(workspace_json) == summary,
            f"{label} inline/raw route JSON mismatch",
        )
        require(
            workspace_tensor.is_file() and workspace_tensor.stat().st_size > 0,
            f"{label} raw route tensor missing",
        )
    require(case.get("route_replay_equal") is True, f"{label} route replay drift")


def load_validation(results: Path, arm: str) -> dict[str, Any]:
    path = validation_path(results, arm)
    value = read_json(path)
    label = f"{arm} validation"
    require(value.get("schema") == "exp014.arm-validation.v1", f"{label} schema drift")
    require(
        value.get("status") == "complete" and value.get("gate_pass") is True,
        f"{label} failed",
    )
    require(value.get("arm") == arm, f"{label} arm drift")
    require(
        value.get("case_order") == [list(case) for case in EXPECTED_CASES],
        f"{label} case matrix drift",
    )
    runtime = value.get("runtime")
    imports = value.get("imports")
    require(isinstance(runtime, Mapping), f"{label} runtime missing")
    require(isinstance(imports, Mapping), f"{label} imports missing")
    source = runtime.get("source")
    harness = runtime.get("harness")
    require(isinstance(source, Mapping), f"{label} source missing")
    require(isinstance(harness, Mapping), f"{label} harness missing")
    require(
        source.get("overlay_sha256") == EXPECTED_OVERLAY_SHA256[arm],
        f"{label} loaded overlay hash drift",
    )
    require(
        harness.get("sha256") == EXPECTED_HARNESS_SHA256,
        f"{label} loaded harness hash drift",
    )
    require(
        runtime.get("jit_root") == EXPECTED_JIT_ROOT[arm], f"{label} JIT root drift"
    )
    require(
        imports.get("target_module") == source.get("overlay"),
        f"{label} imported module/overlay mismatch",
    )
    summaries = value.get("cases")
    require(
        isinstance(summaries, list) and len(summaries) == len(EXPECTED_CASES),
        f"{label} case summary count drift",
    )
    cases = {}
    expected_paths = set()
    for summary, (m, fixture) in zip(summaries, EXPECTED_CASES, strict=True):
        case_file = case_path(results, arm, m, fixture)
        expected_paths.add(case_file.resolve())
        require(summary.get("m") == m, f"{label} summary M drift")
        require(
            summary.get("fixture_kind") == fixture, f"{label} summary fixture drift"
        )
        require(
            summary.get("path") == relative(case_file, results),
            f"{label} path link drift",
        )
        require(
            summary.get("sha256") == file_sha256(case_file), f"{label} hash link drift"
        )
        require(summary.get("gate_pass") is True, f"{label} summary gate failed")
        case = read_json(case_file)
        validate_case(case, case_file, arm, m, fixture)
        require(
            case.get("runtime_identity_sha256") == canonical_sha256(runtime),
            f"{label} case/runtime identity link drift",
        )
        cases[(m, fixture)] = {
            "value": case,
            "path": relative(case_file, results),
            "sha256": file_sha256(case_file),
        }
    observed_paths = {
        item.resolve()
        for item in (results / "raw" / "validation" / arm).glob("m*/*/case.json")
    }
    require(observed_paths == expected_paths, f"{label} exact case file set mismatch")
    cubins = value.get("cubin_sha256")
    require(
        isinstance(cubins, list) and len(cubins) == 1 and is_sha256(cubins[0]),
        f"{label} cubin inventory invalid",
    )
    artifacts = value.get("jit_artifacts")
    require(isinstance(artifacts, list) and artifacts, f"{label} JIT artifacts missing")
    require(
        value.get("jit_artifact_set_sha256") == canonical_sha256(artifacts),
        f"{label} JIT artifact-set hash mismatch",
    )
    artifact_cubins = {
        item.get("sha256")
        for item in artifacts
        if str(item.get("path", "")).endswith(".cubin")
    }
    require(artifact_cubins == set(cubins), f"{label} cubin/artifact mismatch")
    return {
        "value": value,
        "path": relative(path, results),
        "sha256": file_sha256(path),
        "identity": stable_runtime(runtime, imports),
        "cases": cases,
        "cubins": cubins,
    }


def paired_validation(results: Path) -> dict[str, Any]:
    arms = {arm: load_validation(results, arm) for arm in ARMS}
    require(
        arms[BASELINE]["identity"] == arms[CANDIDATE]["identity"],
        "baseline/candidate stable runtime identity mismatch",
    )
    case_rows = []
    for m, fixture in EXPECTED_CASES:
        baseline = arms[BASELINE]["cases"][(m, fixture)]["value"]
        candidate = arms[CANDIDATE]["cases"][(m, fixture)]["value"]
        for field in ("case", "fixture", "weights", "reference"):
            require(
                baseline.get(field) == candidate.get(field),
                f"paired M={m} {fixture} {field} identity mismatch",
            )
        stable_route_fields = (
            "expected_task_count",
            "observed_task_tail",
            "observed_task_head",
            "terminal_head_overshoot",
            "producer_claim_count",
            "expected_pair_head",
            "observed_pair_head",
            "task_descriptor_multiset_sha256",
        )
        baseline_route = [
            {
                field: record["summary"]["verification"].get(field)
                for field in stable_route_fields
            }
            for record in baseline["route_task_evidence"]
        ]
        candidate_route = [
            {
                field: record["summary"]["verification"].get(field)
                for field in stable_route_fields
            }
            for record in candidate["route_task_evidence"]
        ]
        require(
            all(
                value is not None
                for projection in baseline_route + candidate_route
                for value in projection.values()
            ),
            f"paired M={m} {fixture} route/task projection is incomplete",
        )
        require(
            baseline_route == candidate_route,
            f"paired M={m} {fixture} route/task evidence mismatch",
        )
        case_rows.append(
            {
                "m": m,
                "fixture": fixture,
                "input_and_reference_identity_equal": True,
                "both_fp32_oracle_gates_pass": True,
                "route_task_identity_equal": True,
            }
        )
    return {
        "status": "pass",
        "case_count_per_arm": len(EXPECTED_CASES),
        "paired_cases": case_rows,
        "stable_runtime_identity_sha256": canonical_sha256(arms[BASELINE]["identity"]),
        "arms": {
            arm: {
                "validation": arms[arm]["path"],
                "validation_sha256": arms[arm]["sha256"],
                "cubin_sha256": arms[arm]["cubins"],
                "jit_artifact_set_sha256": arms[arm]["value"][
                    "jit_artifact_set_sha256"
                ],
                "jit_root": arms[arm]["value"]["runtime"]["jit_root"],
            }
            for arm in ARMS
        },
        "gate_pass": True,
        "evidence_boundary": (
            "paired gate requires identical inputs/reference/route evidence and "
            "both arms to pass the independent FP32 oracle; it does not require "
            "bitwise-equal atomic output ordering"
        ),
        "_loaded": arms,
    }


def validate_benchmark(
    results: Path,
    validations: Mapping[str, Any],
    arm: str,
    m: int,
    group: int,
    position: int,
) -> dict[str, Any]:
    path = benchmark_path(results, arm, m, group, position)
    value = read_json(path)
    label = f"M={m}/g{group}/p{position}/{arm}"
    require(
        value.get("schema") == "exp014.benchmark-position.v1", f"{label} schema drift"
    )
    require(value.get("status") == "complete", f"{label} incomplete")
    require(value.get("arm") == arm == ABBA[position], f"{label} arm/order drift")
    exact_int(value.get("m"), m, f"{label} M")
    exact_int(value.get("group"), group, f"{label} group")
    exact_int(value.get("position"), position, f"{label} position")
    require(value.get("fixture_kind") == "canonical", f"{label} fixture drift")
    require(value.get("abba_order") == list(ABBA), f"{label} ABBA declaration drift")
    protocol = value.get("protocol")
    require(isinstance(protocol, Mapping), f"{label} protocol missing")
    exact_int(protocol.get("warmup"), WARMUP, f"{label} warmup")
    exact_int(protocol.get("iters"), ITERS, f"{label} iterations")
    exact_int(protocol.get("l2_flush_bytes"), L2_FLUSH_BYTES, f"{label} L2 flush")
    exact_int(
        protocol.get("expected_app_clock_mhz"), EXPECTED_CLOCK_MHZ, f"{label} clock"
    )
    samples = value.get("samples_us")
    require(
        isinstance(samples, list) and len(samples) == ITERS,
        f"{label} sample count drift",
    )
    samples = [finite(sample, f"{label} sample", positive=True) for sample in samples]
    sample_mean = statistics.fmean(samples)
    require(
        math.isclose(
            finite(value.get("sample_us"), f"{label} sample_us", positive=True),
            sample_mean,
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        ),
        f"{label} stored mean mismatch",
    )
    runtime = value.get("runtime")
    imports = value.get("imports")
    require(isinstance(runtime, Mapping), f"{label} runtime missing")
    require(isinstance(imports, Mapping), f"{label} imports missing")
    source = runtime.get("source")
    harness = runtime.get("harness")
    require(isinstance(source, Mapping), f"{label} source missing")
    require(isinstance(harness, Mapping), f"{label} harness missing")
    require(
        source.get("overlay_sha256") == EXPECTED_OVERLAY_SHA256[arm],
        f"{label} source drift",
    )
    require(harness.get("sha256") == EXPECTED_HARNESS_SHA256, f"{label} harness drift")
    require(
        runtime.get("jit_root") == EXPECTED_JIT_ROOT[arm], f"{label} JIT root drift"
    )
    require(
        stable_runtime(runtime, imports) == validations[arm]["identity"],
        f"{label} validation/measurement environment mismatch",
    )
    canonical = validations[arm]["cases"][(m, "canonical")]["value"]
    require(
        value.get("fixture") == canonical.get("fixture"),
        f"{label} fixture identity mismatch",
    )
    require(
        value.get("weights") == canonical.get("weights"),
        f"{label} weight identity mismatch",
    )
    require(
        value.get("cubin_sha256") == validations[arm]["cubins"], f"{label} cubin drift"
    )
    require(
        value.get("jit_artifact_set_sha256")
        == validations[arm]["value"]["jit_artifact_set_sha256"],
        f"{label} JIT artifact-set drift",
    )
    gpu_after = value.get("gpu_after")
    require(isinstance(gpu_after, Mapping), f"{label} gpu_after missing")
    require(gpu_after.get("uuid") == EXPECTED_GPU_UUID, f"{label} GPU-after drift")
    require(
        int(float(gpu_after.get("applications_graphics_clock_mhz")))
        == EXPECTED_CLOCK_MHZ,
        f"{label} GPU-after clock drift",
    )
    return {
        "arm": arm,
        "m": m,
        "group": group,
        "position": position,
        "mean_us": sample_mean,
        "source": relative(path, results),
        "source_sha256": file_sha256(path),
    }


def build_performance(results: Path, validation: Mapping[str, Any]) -> dict[str, Any]:
    validations = validation["_loaded"]
    expected_paths = {
        benchmark_path(results, ABBA[position], m, group, position).resolve()
        for m in M_VALUES
        for group in GROUPS
        for position in POSITIONS
    }
    root = results / "raw" / "benchmark"
    observed_paths = (
        {path.resolve() for path in root.glob("m*/*.json")} if root.is_dir() else set()
    )
    require(
        observed_paths == expected_paths,
        "paired benchmark exact file set mismatch",
    )
    rows = [
        validate_benchmark(results, validations, ABBA[position], m, group, position)
        for m in M_VALUES
        for group in GROUPS
        for position in POSITIONS
    ]
    cases = {}
    for m in M_VALUES:
        groups = []
        for group in GROUPS:
            by_position = {
                row["position"]: row
                for row in rows
                if row["m"] == m and row["group"] == group
            }
            baseline_us = statistics.fmean(
                by_position[position]["mean_us"] for position in (0, 3)
            )
            candidate_us = statistics.fmean(
                by_position[position]["mean_us"] for position in (1, 2)
            )
            speedup = (baseline_us / candidate_us - 1.0) * 100.0
            groups.append(
                {
                    "group": group,
                    "baseline_us": baseline_us,
                    "candidate_us": candidate_us,
                    "speedup_percent": speedup,
                }
            )
        baseline_us = statistics.fmean(group["baseline_us"] for group in groups)
        candidate_us = statistics.fmean(group["candidate_us"] for group in groups)
        cases[f"m{m}"] = {
            "baseline_us": baseline_us,
            "candidate_us": candidate_us,
            "speedup_percent": (baseline_us / candidate_us - 1.0) * 100.0,
            "group_speedup_percent": [group["speedup_percent"] for group in groups],
            "groups": groups,
        }
    m256 = cases["m256"]
    m8192 = cases["m8192"]
    all_aggregates_within_regression_limit = all(
        case["speedup_percent"] >= -1.5 for case in cases.values()
    )
    if (
        all(value > 0.0 for value in m8192["group_speedup_percent"])
        and m8192["speedup_percent"] > 0.0
        and m256["speedup_percent"] >= -1.5
        and all_aggregates_within_regression_limit
    ):
        decision = "accept"
        reason = "M8192 五组均有收益，且全部 M>=256 聚合值未越过 -1.5% 回退线"
    elif all(value < -1.5 for value in m8192["group_speedup_percent"]):
        decision = "reject"
        reason = "M8192 五组均越过 -1.5% 回退线"
    else:
        decision = "inconclusive"
        reason = "五组 paired ABBA 未给出一致方向"
    return {
        "status": "complete",
        "protocol": {
            "m_values": list(M_VALUES),
            "groups": len(GROUPS),
            "positions_per_group": len(POSITIONS),
            "order": list(ABBA),
            "warmup": WARMUP,
            "iters": ITERS,
            "l2_flush_bytes": L2_FLUSH_BYTES,
            "speedup_definition": "baseline_us / candidate_us - 1",
            "scope": "final paired performance acceptance",
        },
        "cases": cases,
        "decision": decision,
        "reason": reason,
        "raw_positions": rows,
    }


def validate_final_diagnostics(
    results: Path,
    validation: Mapping[str, Any],
    performance: Mapping[str, Any],
) -> dict[str, Any]:
    require(performance.get("decision") == "accept", "performance gate did not accept")

    phase_path = results / "scatter_phase_evidence.json"
    phase = read_json(phase_path)
    require(
        phase.get("schema") == "exp014.scatter-phase-evidence.v1",
        "Scatter phase evidence schema drift",
    )
    require(phase.get("status") == "complete", "Scatter phase evidence incomplete")
    require(phase.get("gate_pass") is True, "Scatter phase gate failed")
    require(phase.get("static_zero_spill") is True, "probe cubin has static spill")
    phase_case = phase.get("case")
    require(isinstance(phase_case, Mapping), "Scatter phase case is missing")
    exact_int(phase_case.get("m"), 8192, "Scatter phase M")
    exact_int(
        phase_case.get("candidate_scatter_warps"),
        8,
        "candidate Scatter warp count",
    )
    body = phase.get("comparison", {}).get("body_ns", {}).get("aggregate", {})
    body_baseline = finite(body.get("baseline"), "phase body baseline", positive=True)
    body_candidate = finite(
        body.get("candidate"), "phase body candidate", positive=True
    )
    require(body_candidate < body_baseline, "Scatter body latency did not decrease")

    dynamic_path = results / "dynamic_spill_evidence.json"
    dynamic = read_json(dynamic_path)
    require(
        dynamic.get("schema") == "exp014.dynamic-spill-evidence.v1",
        "dynamic spill evidence schema drift",
    )
    require(dynamic.get("status") == "pass", "dynamic spill evidence failed")
    require(dynamic.get("gate_pass") is True, "dynamic spill gate failed")

    static_path = results / "static_resource_evidence.json"
    static = read_json(static_path)
    static_arms = static.get("arms")
    require(isinstance(static_arms, Mapping), "static resource arms are missing")
    static_projection: dict[str, Any] = {}
    for arm, static_arm in ((BASELINE, "baseline"), (CANDIDATE, "candidate")):
        record = static_arms.get(static_arm)
        require(
            isinstance(record, Mapping), f"missing static resource arm: {static_arm}"
        )
        expected_cubins = validation["arms"][arm]["cubin_sha256"]
        require(
            isinstance(expected_cubins, list) and len(expected_cubins) == 1,
            f"{arm} validation cubin identity is ambiguous",
        )
        cubin = record.get("cubin")
        resource = record.get("resource")
        sass = record.get("sass")
        require(isinstance(cubin, Mapping), f"{arm} static cubin is missing")
        require(isinstance(resource, Mapping), f"{arm} static resource is missing")
        require(isinstance(sass, Mapping), f"{arm} static SASS is missing")
        counts = sass.get("selected_instruction_counts")
        require(isinstance(counts, Mapping), f"{arm} selected SASS counts are missing")
        require(
            cubin.get("sha256") == expected_cubins[0],
            f"{arm} static cubin identity mismatch",
        )
        zero_spill = (
            resource.get("stack_bytes_per_thread") == 0
            and resource.get("local_bytes_outside_stack") == 0
            and counts.get("ldl") == 0
            and counts.get("stl") == 0
        )
        require(zero_spill, f"{arm} uninstrumented cubin has static spill")
        static_projection[arm] = {
            "cubin_sha256": expected_cubins[0],
            "registers_per_thread": resource.get("registers_per_thread"),
            "stack_bytes_per_thread": 0,
            "local_bytes_outside_stack": 0,
            "ldl_instructions": 0,
            "stl_instructions": 0,
            "static_zero_spill": True,
        }

    return {
        "status": "pass",
        "scatter_phase": {
            "source": relative(phase_path, results),
            "source_sha256": file_sha256(phase_path),
            "m": 8192,
            "candidate_scatter_warps": 8,
            "body_baseline_ns": body_baseline,
            "body_candidate_ns": body_candidate,
            "body_speedup_percent": finite(
                body.get("speedup_pct"), "phase body speedup"
            ),
            "diagnostic_only": True,
            "gate_pass": True,
        },
        "static_resources": {
            "source": relative(static_path, results),
            "source_sha256": file_sha256(static_path),
            "arms": static_projection,
            "zero_spill": True,
            "note": "register count is an observation, not the exp_014 acceptance cap",
        },
        "dynamic_spill": {
            "source": relative(dynamic_path, results),
            "source_sha256": file_sha256(dynamic_path),
            "scope": "one canonical M8192 graph node per exact arm identity",
            "zero_spill": True,
        },
        "decision": "accept",
    }


def source_manifest(
    results: Path,
    registered: Mapping[str, Any],
    ownership: Mapping[str, Any],
    validation: Mapping[str, Any],
    performance: Mapping[str, Any] | None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sources = [registered["harness"], registered["identity"]]
    sources.extend(registered["overlays"].values())
    sources.append(
        {
            "path": ownership["source"],
            "sha256": ownership["source_sha256"],
        }
    )
    for arm in ARMS:
        sources.append(
            {
                "path": validation["arms"][arm]["validation"],
                "sha256": validation["arms"][arm]["validation_sha256"],
            }
        )
        for case in validation["_loaded"][arm]["cases"].values():
            sources.append({"path": case["path"], "sha256": case["sha256"]})
    if performance is not None:
        sources.extend(
            {"path": row["source"], "sha256": row["source_sha256"]}
            for row in performance["raw_positions"]
        )
    if diagnostics is not None:
        sources.extend(
            {
                "path": diagnostics[key]["source"],
                "sha256": diagnostics[key]["source_sha256"],
            }
            for key in ("scatter_phase", "static_resources", "dynamic_spill")
        )
    paths = [source["path"] for source in sources]
    require(len(paths) == len(set(paths)), "duplicate source in evidence manifest")
    return {
        "schema": "exp014.evidence-manifest.v1",
        "path_base": "results_root",
        "source_count": len(sources),
        "sources": sorted(sources, key=lambda item: item["path"]),
        "source_set_sha256": canonical_sha256(
            sorted(sources, key=lambda item: item["path"])
        ),
        "derived_outputs": ["evidence.json", "manifest.json", "paired_summary.md"],
        "results_root": ".",
    }


def render_summary(evidence: Mapping[str, Any]) -> str:
    validation = evidence["validation"]
    lines = [
        "# exp_014 Paired 证据",
        "",
        "- Ownership：通过；full tile 的 W0–W7 均有 Scatter work，且每个合法元素恰好一个 owner。",
        f"- Correctness：两臂各 {validation['case_count_per_arm']} 个 case 通过，并已校验输入、reference 与 route/task 身份一致。",
    ]
    performance = evidence.get("performance")
    if performance is None:
        lines.extend(["- Paired ABBA：尚未采集。", ""])
        return "\n".join(lines)
    lines.extend(
        [
            f"- Paired 判定：`{performance['decision']}`；{performance['reason']}。",
            "",
            "| M | Baseline 4-warp (us) | Candidate 8-warp (us) | Speedup | 五组 speedup |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for m in M_VALUES:
        case = performance["cases"][f"m{m}"]
        groups = " / ".join(f"{value:+.2f}%" for value in case["group_speedup_percent"])
        lines.append(
            f"| {m} | {case['baseline_us']:.3f} | {case['candidate_us']:.3f} | "
            f"{case['speedup_percent']:+.2f}% | {groups} |"
        )
    lines.extend(
        [
            "",
            "> 五组 A-B-B-A 为最终未插桩 fused E2E 性能判定。",
            "",
        ]
    )
    return "\n".join(lines)


def clean_validation(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "_loaded"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--stage", choices=("validation", "performance", "final"), required=True
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = args.results.resolve()
    registered = validate_registered_sources(results)
    ownership = validate_ownership(results)
    validation = paired_validation(results)
    performance = (
        build_performance(results, validation)
        if args.stage in ("performance", "final")
        else None
    )
    diagnostics = (
        validate_final_diagnostics(results, validation, performance)
        if args.stage == "final" and performance is not None
        else None
    )
    evidence = {
        "schema": "exp014.evidence.v1",
        "stage": args.stage,
        "registered_sources": registered,
        "ownership": ownership,
        "validation": clean_validation(validation),
        "performance": performance,
        "diagnostics": diagnostics,
        "decision": diagnostics["decision"] if diagnostics is not None else None,
        "status": "complete" if performance is not None else "validation_pass",
    }
    manifest = source_manifest(
        results, registered, ownership, validation, performance, diagnostics
    )
    write_json(results / "evidence.json", evidence)
    write_json(results / "manifest.json", manifest)
    write_text(results / "paired_summary.md", render_summary(evidence))
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvidenceError as error:
        print(f"exp_014 evidence rejected: {error}", file=os.sys.stderr)
        raise SystemExit(2) from error
