#!/usr/bin/env python3
"""Fresh-process runtime harness for exp_014 fused Scatter 4-to-8 warp A/B.

The harness has two deliberately separate modes:

``validate``
    Run the complete, pre-registered correctness matrix for exactly one arm in
    one process and one fresh JIT root.  Every case uses the same deterministic
    fixture/weight seed, an independent FP32 PyTorch MoE oracle, two CUDA Graph
    replays, and route/task workspace validation.

``measure``
    Produce exactly one canonical ABBA position for one arm/M.  The caller is
    responsible for launching A-B-B-A x 5 as separate processes and providing
    a different empty JIT root to every invocation.  This command fixes the
    protocol at warmup=5, timed=50, and a 192 MiB L2 flush before every replay.

The selected overlay is installed before FlashInfer imports and must match the
registered SHA-256 for the arm.  This file intentionally does not aggregate
cross-arm correctness or performance: those gates require both immutable arm
outputs and belong in a GPU-free evidence builder.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parent
EXP005 = ROOT.parent / "exp_005_8warp_spill_reduction"
sys.path.insert(0, str(EXP005))

import exp005_common as common  # noqa: E402
import run_exp005_arm as worker  # noqa: E402


BASELINE = "baseline_4warp_scatter"
CANDIDATE = "candidate_8warp_scatter"
ARMS = (BASELINE, CANDIDATE)
EXPECTED_OVERLAY_SHA256 = {
    BASELINE: "b6e141179794561f2144bdec079b7e109fddfccc2db6ba0f19c22d30ea4b34ca",
    CANDIDATE: "c88cef63492b60c0a77484b50f6400b83a103d168e1535b78972341503810184",
}
EXPECTED_PRODUCTION_SHA256 = common.EXPECTED_KERNEL_SHA256
EXPECTED_FLASHINFER_COMMIT = common.EXPECTED_FLASHINFER_COMMIT
EXPECTED_CUTLASS_COMMIT = common.EXPECTED_CUTLASS_COMMIT
EXPECTED_IMAGE_ID = (
    "sha256:a4e056e1d34a5cc9387512ffa3abeed778e3dc7966633c5154d771705d8835ac"
)
TARGET_MODULE = common.TARGET_MODULE
TARGET_RELATIVE_PATH = common.TARGET_RELATIVE_PATH

M_VALUES = (256, 512, 1024, 2048, 4096, 8192)
CANONICAL = "canonical"
DIRECTED = ("sparse_empty", "exact_128", "tail_129", "hot_expert")
CANARIES = ("canary_gate_v2", "canary_up_v2")
M256_FIXTURES = (CANONICAL,) + DIRECTED + CANARIES
VALIDATION_CASES = tuple((m, CANONICAL) for m in M_VALUES) + tuple(
    (256, fixture) for fixture in DIRECTED + CANARIES
)

EXPECTED_GRID = (1, 1, 110)
EXPECTED_BLOCK = (288, 1, 1)
NUM_CTA_WARPS = EXPECTED_BLOCK[0] // 32
EXPECTED_MAX_ACTIVE_CLUSTERS = 110
L2_FLUSH_BYTES = 192 << 20
WARMUP = 5
ITERS = 50
ABBA = (BASELINE, CANDIDATE, CANDIDATE, BASELINE)

N64 = 64
INTERMEDIATE = 512
MARKER_CODES = (1, 2, 3, 4, 5, 6, 7, 9)
FIXED_CODE = 2
SCATTER_OUTPUT_SCALE = 0.25
CANARY_BLOCK_RELATIVE_L2_LIMIT = 0.15
# Atomic Scatter has nondeterministic accumulation order.  This is only the
# one-arm fail-fast cap; the final candidate gate is baseline-derived.
SELF_DRIFT_CAP_OVERRIDES = {"cosine_loss": 2.0e-4}


def checked_output(command: Sequence[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip()


def git(repo: Path, *args: str) -> str:
    return checked_output(["git", "-c", f"safe.directory={repo}", *args], cwd=repo)


def atomic_torch_save(value: Any, path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"immutable tensor evidence already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_new_output(path: Path, description: str) -> None:
    if path.exists():
        raise RuntimeError(f"immutable {description} already exists: {path}")


def require_registered_measurement_jit(args: argparse.Namespace) -> str:
    if not args.jit_root.is_dir():
        raise RuntimeError(
            f"registered measurement JIT root is missing: {args.jit_root}"
        )
    artifacts = common.artifact_manifest(args.jit_root)
    if not artifacts:
        raise RuntimeError("registered measurement JIT root has no retained artifacts")
    artifact_set_sha256 = common.canonical_sha256(artifacts)
    if artifact_set_sha256 != args.expected_jit_artifact_set_sha256:
        raise RuntimeError(
            "registered measurement JIT artifact-set drift: "
            f"{artifact_set_sha256} != {args.expected_jit_artifact_set_sha256}"
        )
    cubins = sorted(
        {str(item["sha256"]) for item in artifacts if item["path"].endswith(".cubin")}
    )
    if cubins != [args.expected_cubin_sha256]:
        raise RuntimeError(
            f"registered measurement cubin drift: {cubins} != "
            f"{[args.expected_cubin_sha256]}"
        )
    return artifact_set_sha256


def validate_source(repo: Path, overlay: Path, arm: str) -> dict[str, Any]:
    production = repo / TARGET_RELATIVE_PATH
    cutlass = repo / "3rdparty/cutlass"
    if not production.is_file():
        raise RuntimeError(f"production kernel is missing: {production}")
    if not overlay.is_file():
        raise RuntimeError(f"selected overlay is missing: {overlay}")

    production_hash = common.file_sha256(production)
    overlay_hash = common.file_sha256(overlay)
    if production_hash != EXPECTED_PRODUCTION_SHA256:
        raise RuntimeError(
            f"production kernel hash drift: {production_hash} "
            f"!= {EXPECTED_PRODUCTION_SHA256}"
        )
    if overlay_hash != EXPECTED_OVERLAY_SHA256[arm]:
        raise RuntimeError(
            f"{arm} overlay hash drift: {overlay_hash} "
            f"!= {EXPECTED_OVERLAY_SHA256[arm]}"
        )
    if EXPECTED_OVERLAY_SHA256[BASELINE] == EXPECTED_OVERLAY_SHA256[CANDIDATE]:
        raise RuntimeError("registered baseline and candidate hashes are identical")

    checkout_head = git(repo, "rev-parse", "HEAD")
    git(repo, "merge-base", "--is-ancestor", EXPECTED_FLASHINFER_COMMIT, checkout_head)
    cutlass_head = git(cutlass, "rev-parse", "HEAD")
    if cutlass_head != EXPECTED_CUTLASS_COMMIT:
        raise RuntimeError(
            f"CUTLASS commit drift: {cutlass_head} != {EXPECTED_CUTLASS_COMMIT}"
        )

    return {
        "locked_flashinfer_commit": EXPECTED_FLASHINFER_COMMIT,
        "checkout_head": checkout_head,
        "cutlass_commit": cutlass_head,
        "production_kernel": str(production),
        "production_kernel_sha256": production_hash,
        "overlay": str(overlay),
        "overlay_sha256": overlay_hash,
        "oracle_source": str(worker.FIXTURE_PATH),
        "oracle_source_sha256": common.file_sha256(worker.FIXTURE_PATH),
    }


def configure_source_checkout(repo: Path, jit_root: Path) -> dict[str, str]:
    flashinfer = importlib.import_module("flashinfer")
    imported_root = Path(flashinfer.__file__).resolve().parents[1]
    if imported_root != repo:
        raise RuntimeError(f"imported FlashInfer root {imported_root} != {repo}")

    from flashinfer.jit import env as jit_env

    jit_workspace = Path(jit_env.FLASHINFER_WORKSPACE_DIR).resolve()
    try:
        jit_workspace.relative_to(jit_root)
    except ValueError as error:
        raise RuntimeError(
            f"FlashInfer JIT workspace {jit_workspace} is outside {jit_root}"
        ) from error

    jit_env.FLASHINFER_CSRC_DIR = repo / "csrc"
    jit_env.FLASHINFER_INCLUDE_DIR = repo / "include"
    jit_env.FLASHINFER_AOT_DIR = jit_workspace / "aot_disabled_for_exp014"
    jit_env.CUTLASS_INCLUDE_DIRS = [
        repo / "3rdparty/cutlass/include",
        repo / "3rdparty/cutlass/tools/util/include",
    ]
    jit_env.CCCL_INCLUDE_DIRS = [
        repo / "3rdparty/cccl/cub",
        repo / "3rdparty/cccl/libcudacxx/include",
        repo / "3rdparty/cccl/thrust",
    ]
    jit_env.SPDLOG_INCLUDE_DIR = repo / "3rdparty/spdlog/include"

    target = importlib.import_module(TARGET_MODULE)
    cutlass = importlib.import_module("cutlass")
    return {
        "flashinfer": str(Path(flashinfer.__file__).resolve()),
        "target_module": str(Path(target.__file__).resolve()),
        "cutlass_python": str(Path(cutlass.__file__).resolve()),
        "cutlass_python_version": str(getattr(cutlass, "__version__", "unknown")),
        "flashinfer_jit_workspace": str(jit_workspace),
    }


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in (
        "flashinfer-python",
        "nvidia-cutlass-dsl",
        "cutlass-dsl",
        "torch",
        "tvm-ffi",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "not-installed-as-distribution"
    return result


def gpu_inventory() -> list[dict[str, str]]:
    fields = (
        "index,uuid,name,pci.bus_id,driver_version,clocks.current.graphics,"
        "clocks.applications.graphics,clocks.max.graphics,power.draw"
    )
    output = checked_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    )
    keys = (
        "index",
        "uuid",
        "name",
        "pci_bus_id",
        "driver",
        "graphics_clock_mhz",
        "applications_graphics_clock_mhz",
        "max_graphics_clock_mhz",
        "power_draw_w",
    )
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(keys):
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line}")
        rows.append(dict(zip(keys, values, strict=True)))
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return rows


def selected_gpu(expected_uuid: str) -> dict[str, str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if len(tokens) != 1:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must select exactly one physical GPU; got "
            + repr(visible)
        )
    token = tokens[0]
    inventory = gpu_inventory()
    if token.isdigit():
        matches = [row for row in inventory if row["index"] == token]
    else:
        matches = [row for row in inventory if row["uuid"].startswith(token)]
    if len(matches) != 1:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r} does not uniquely map to nvidia-smi"
        )
    gpu = matches[0]
    if gpu["uuid"] != expected_uuid:
        raise RuntimeError(f"GPU UUID drift: {gpu['uuid']} != {expected_uuid}")
    return gpu


def foreign_processes(gpu_uuid: str) -> list[dict[str, str]]:
    output = checked_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in output.splitlines():
        if not line.strip() or "No running processes" in line:
            continue
        values = [value.strip() for value in line.split(",", 2)]
        if len(values) == 3 and values[0] == gpu_uuid:
            rows.append({"gpu_uuid": values[0], "pid": values[1], "process": values[2]})
    return rows


def require_clock(gpu: Mapping[str, str], expected_mhz: int | None) -> None:
    if expected_mhz is None:
        return
    observed = gpu["applications_graphics_clock_mhz"]
    try:
        observed_mhz = int(float(observed))
    except ValueError as error:
        raise RuntimeError(
            f"application graphics clock is not numeric: {observed}"
        ) from error
    if observed_mhz != expected_mhz:
        raise RuntimeError(
            f"application graphics clock drift: {observed_mhz} != {expected_mhz} MHz"
        )


def runtime_identity(
    args: argparse.Namespace, source: Mapping[str, Any]
) -> dict[str, Any]:
    common.require_clean_compiler_environment()
    required_environment = {
        "W4A4_IMAGE_DIGEST": common.EXPECTED_IMAGE_DIGEST,
        "W4A4_IMAGE_ID": EXPECTED_IMAGE_ID,
        "W4A4_PYTHON_DEPS_SHA256": common.EXPECTED_PYTHON_DEPS_SHA256,
    }
    for key, expected in required_environment.items():
        if os.environ.get(key) != expected:
            raise RuntimeError(f"{key} identity drift")
    if not os.environ.get("KDK_LEASE_ID", "").strip():
        raise RuntimeError("KDK_LEASE_ID is required")
    workspace_base = os.environ.get("FLASHINFER_WORKSPACE_BASE", "").strip()
    if not workspace_base or Path(workspace_base).resolve() != args.jit_root:
        raise RuntimeError("FLASHINFER_WORKSPACE_BASE must equal --jit-root")
    if os.environ.get("CUTE_DSL_KEEP") != "ir,ptx,cubin,sass":
        raise RuntimeError("CUTE_DSL_KEEP must equal ir,ptx,cubin,sass")
    dump_value = os.environ.get("CUTE_DSL_DUMP_DIR", "").strip()
    if not dump_value:
        raise RuntimeError("CUTE_DSL_DUMP_DIR is required")
    dump_dir = Path(dump_value).resolve()
    try:
        dump_dir.relative_to(args.jit_root)
    except ValueError as error:
        raise RuntimeError("CUTE_DSL_DUMP_DIR must be inside --jit-root") from error

    gpu = selected_gpu(args.expected_gpu_uuid)
    require_clock(gpu, getattr(args, "expected_app_clock_mhz", None))
    foreign = foreign_processes(gpu["uuid"])
    if foreign:
        raise RuntimeError(f"foreign compute processes on selected GPU: {foreign}")

    if args.device_index != 0:
        raise RuntimeError("the one-visible-GPU contract requires --device-index=0")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"PyTorch must see exactly one GPU, got {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(args.device_index)
    properties = torch.cuda.get_device_properties(args.device_index)
    capability = list(torch.cuda.get_device_capability(args.device_index))
    if capability not in ([12, 0], [12, 1]):
        raise RuntimeError(f"exp_014 requires SM120/121, got {capability}")
    if int(properties.multi_processor_count) != EXPECTED_GRID[2]:
        raise RuntimeError(
            f"SM count drift: {properties.multi_processor_count} != {EXPECTED_GRID[2]}"
        )
    if properties.name != gpu["name"]:
        raise RuntimeError(
            f"PyTorch/nvidia-smi device-name mismatch: {properties.name} != {gpu['name']}"
        )

    return {
        "timestamp_unix": time.time(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "python": sys.version,
        "packages": package_versions(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nvcc": worker.command_output(["nvcc", "--version"]),
        "ptxas": worker.command_output(["ptxas", "--version"]),
        "gpu": {
            **gpu,
            "compute_capability": capability,
            "sm_count": int(properties.multi_processor_count),
            "foreign_processes_before_cuda_context": foreign,
        },
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "image_digest": os.environ["W4A4_IMAGE_DIGEST"],
        "image_id": os.environ["W4A4_IMAGE_ID"],
        "python_deps_sha256": os.environ["W4A4_PYTHON_DEPS_SHA256"],
        "lease_id": os.environ["KDK_LEASE_ID"],
        "jit_root": str(args.jit_root),
        "source": dict(source),
        "harness": {
            "path": str(Path(__file__).resolve()),
            "sha256": common.file_sha256(Path(__file__).resolve()),
        },
    }


def packed_byte(code: int) -> int:
    if not 0 <= code < 16:
        raise ValueError(code)
    return code | (code << 4)


def canary_weights(fixture_module: Any, weights: Any, branch: str) -> Any:
    if branch not in ("gate", "up"):
        raise ValueError(branch)
    w1 = torch.empty_like(weights.w1_packed)
    w1.fill_(packed_byte(FIXED_CODE))
    branch_base = 0 if branch == "up" else INTERMEDIATE
    for segment, code in enumerate(MARKER_CODES):
        begin = branch_base + segment * N64
        w1[:, begin : begin + N64, :].fill_(packed_byte(code))

    w2 = torch.zeros_like(weights.w2_packed)
    for k in range(INTERMEDIATE):
        byte_index = k // 2
        nibble = FIXED_CODE if k % 2 == 0 else FIXED_CODE << 4
        w2[:, k, byte_index] = nibble

    manifest = dict(weights.manifest)
    manifest.update(
        {
            "fixture_kind": f"branch_half_slice_canary_{branch}",
            "canary_revision": "v2_final_scatter_scale",
            "marker_branch": branch,
            "marker_n64_codes": list(MARKER_CODES),
            "fixed_other_branch_code": FIXED_CODE,
            "fc2_map": "FP4 diagonal activation k -> output k for k in [0,512)",
            "w1_packed_sha256": fixture_module.tensor_sha256(w1),
            "w2_packed_sha256": fixture_module.tensor_sha256(w2),
        }
    )
    return replace(weights, w1_packed=w1, w2_packed=w2, manifest=manifest)


def make_case(
    fixture_module: Any,
    canonical_weights: Any,
    *,
    m: int,
    fixture_kind: str,
    device: torch.device,
    seed: int,
) -> tuple[Any, Any]:
    base = fixture_module.make_routed_fixture(m, device=device, seed=seed)
    if fixture_kind in DIRECTED:
        return worker.make_directed_fixture(
            fixture_module, base, fixture_kind
        ), canonical_weights
    if fixture_kind == CANONICAL:
        return base, canonical_weights
    if fixture_kind not in CANARIES or m != 256:
        raise ValueError(f"unsupported validation case: M={m} {fixture_kind}")

    branch = "gate" if fixture_kind == "canary_gate_v2" else "up"
    x = torch.full_like(base.x, 0.125)
    topk_weights = (base.topk_weights * SCATTER_OUTPUT_SCALE).contiguous()
    manifest = dict(base.manifest)
    manifest.update(
        {
            "fixture_kind": f"branch_half_slice_canary_{branch}",
            "canary_revision": "v2_final_scatter_scale",
            "marker_branch": branch,
            "x_pattern": "constant_bf16_0.125",
            "scatter_output_scale": SCATTER_OUTPUT_SCALE,
            "x_sha256": fixture_module.tensor_sha256(x),
            "topk_weights_sha256": fixture_module.tensor_sha256(topk_weights),
            "topk_weight_sum": float(topk_weights[0].sum().item()),
        }
    )
    fixture = fixture_module.RoutedFixture(m, x, base.topk_ids, topk_weights, manifest)
    return fixture, canary_weights(fixture_module, canonical_weights, branch)


def canary_diagnostics(
    actual: torch.Tensor, reference: torch.Tensor, *, branch: str
) -> dict[str, Any]:
    actual_target = actual[:, :INTERMEDIATE].float()
    reference_target = reference[:, :INTERMEDIATE].float()
    reference_hashes = []
    blocks = []
    for segment, marker in enumerate(MARKER_CODES):
        begin = segment * N64
        end = begin + N64
        actual_block = actual_target[:, begin:end]
        reference_block = reference_target[:, begin:end]
        reference_norm = torch.linalg.vector_norm(reference_block)
        relative_l2 = torch.linalg.vector_norm(
            actual_block - reference_block
        ) / reference_norm.clamp_min(1.0e-12)
        reference_hash = worker.tensor_sha256(reference_block)
        reference_hashes.append(reference_hash)
        blocks.append(
            {
                "segment": segment,
                "output_range": [begin, end],
                "marker_code": marker,
                "reference_sha256": reference_hash,
                "reference_l2": float(reference_norm.item()),
                "actual_l2": float(torch.linalg.vector_norm(actual_block).item()),
                "relative_l2": float(relative_l2.item()),
                "pass": bool(relative_l2.item() <= CANARY_BLOCK_RELATIVE_L2_LIMIT),
            }
        )

    reference_nonzero = int(torch.count_nonzero(reference_target).item())
    actual_nonzero = int(torch.count_nonzero(actual_target).item())
    target_elements = reference_target.numel()
    checks = {
        "every_reference_target_is_nonzero": reference_nonzero == target_elements,
        "every_actual_target_is_nonzero": actual_nonzero == target_elements,
        "eight_reference_blocks_are_distinguishable": len(set(reference_hashes))
        == len(MARKER_CODES),
        "all_blocks_match_reference": all(block["pass"] for block in blocks),
    }
    return {
        "branch": branch,
        "target_range": [0, INTERMEDIATE],
        "target_elements": target_elements,
        "reference_nonzero": reference_nonzero,
        "actual_nonzero": actual_nonzero,
        "block_relative_l2_limit": CANARY_BLOCK_RELATIVE_L2_LIMIT,
        "blocks": blocks,
        "checks": checks,
        "gate_pass": all(checks.values()),
    }


def self_drift_gate(values: Mapping[str, float]) -> dict[str, Any]:
    caps = {
        metric: max(
            float(specification["cap"]),
            SELF_DRIFT_CAP_OVERRIDES.get(metric, 0.0),
        )
        for metric, specification in common.CORRECTNESS_SPECS.items()
    }
    checks = {
        metric: math.isfinite(float(values[metric]))
        and 0.0 <= float(values[metric]) <= caps[metric]
        for metric in common.CORRECTNESS_SPECS
    }
    return {
        "policy": "preliminary per-arm cap only; final threshold is baseline-derived",
        "caps": caps,
        "checks": checks,
        "gate_pass": all(checks.values()),
    }


def artifact_delta(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    old = {(str(item["path"]), str(item["sha256"])) for item in before}
    return [
        dict(item)
        for item in after
        if (str(item["path"]), str(item["sha256"])) not in old
    ]


def case_directory(results: Path, arm: str, m: int, fixture: str) -> Path:
    return results / "raw" / "validation" / arm / f"m{m}" / fixture


def run_validation_case(
    args: argparse.Namespace,
    runtime: Mapping[str, Any],
    fixture_module: Any,
    canonical_weights: Any,
    *,
    m: int,
    fixture_kind: str,
) -> dict[str, Any]:
    raw_dir = case_directory(args.results, args.arm, m, fixture_kind)
    require_new_output(raw_dir, "validation case directory")
    raw_dir.mkdir(parents=True)

    device = torch.device("cuda", args.device_index)
    fixture, weights = make_case(
        fixture_module,
        canonical_weights,
        m=m,
        fixture_kind=fixture_kind,
        device=device,
        seed=args.seed,
    )
    artifacts_before = common.artifact_manifest(args.jit_root)
    reference = fixture_module.reference_moe_nvfp4(fixture, weights)
    artifacts_after_reference = common.artifact_manifest(args.jit_root)

    case_args = argparse.Namespace(**vars(args), m=m)
    arm = worker.build_arm(case_args, fixture, weights)
    arm.eager()
    arm.capture()
    artifacts_after_kernel = common.artifact_manifest(args.jit_root)

    output_records = []
    output_values = []
    route_records = []
    for replay in range(2):
        output, elapsed_ms = arm.replay(sentinel=True)
        output = output.clone()
        workspace_tensors, workspace_summary = worker._workspace_snapshot(
            arm.wrapper, fixture, num_cta_warps=NUM_CTA_WARPS
        )
        atomic_torch_save(output.detach().cpu(), raw_dir / f"output_{replay}.pt")
        atomic_torch_save(workspace_tensors, raw_dir / f"workspace_replay_{replay}.pt")
        common.write_json(
            raw_dir / f"workspace_replay_{replay}.json", workspace_summary
        )

        broad = fixture_module.output_diagnostics(output, reference)
        reference_error = worker.tensor_error(output, reference)
        nan_remaining = int(torch.isnan(output).sum().item())
        record: dict[str, Any] = {
            "replay": replay,
            "event_elapsed_us": elapsed_ms * 1000.0,
            "output_sha256": worker.tensor_sha256(output),
            "reference_sha256": worker.tensor_sha256(reference),
            "reference_error": reference_error,
            "broad_oracle_diagnostics": broad,
            "sentinel_nan_remaining": nan_remaining,
            "sentinel_evidence_boundary": (
                "auxiliary missing-write signal only; exact-once work comes from "
                "route/task evidence and directed write canaries"
            ),
        }
        if fixture_kind in CANARIES:
            record["write_canary"] = canary_diagnostics(
                output,
                reference,
                branch="gate" if fixture_kind == "canary_gate_v2" else "up",
            )
        output_records.append(record)
        output_values.append(output.detach().cpu())
        route_records.append(workspace_summary)

    stability = worker.tensor_error(output_values[1], output_values[0])
    stability_gate = self_drift_gate(stability)
    route_stable_fields = (
        "expected_task_count",
        "task_descriptor_multiset_sha256",
        "observed_task_tail",
        "observed_task_head",
        "expected_pair_head",
        "observed_pair_head",
    )
    route_replay_equal = all(
        route_records[0]["verification"].get(field)
        == route_records[1]["verification"].get(field)
        for field in route_stable_fields
    )
    output_gate = all(
        bool(record["broad_oracle_diagnostics"]["formal_pass"])
        and bool(record["broad_oracle_diagnostics"]["finite"])
        and record["sentinel_nan_remaining"] == 0
        and (fixture_kind not in CANARIES or bool(record["write_canary"]["gate_pass"]))
        for record in output_records
    )
    route_gate = route_replay_equal and all(
        bool(record["verification"]["gate_pass"]) for record in route_records
    )

    compile_identity = worker._compile_identity()
    if compile_identity["compiled_max_active_clusters"] != [
        EXPECTED_MAX_ACTIVE_CLUSTERS
    ]:
        raise RuntimeError("compiled max_active_clusters drift")
    cubins = [
        item for item in artifacts_after_kernel if item["path"].endswith(".cubin")
    ]
    if not cubins:
        raise RuntimeError(f"M={m} {fixture_kind} produced no retained cubin")

    payload = {
        "schema": "exp014.validation-case.v1",
        "status": "complete"
        if output_gate and route_gate and stability_gate["gate_pass"]
        else "failed",
        "arm": args.arm,
        "m": m,
        "fixture_kind": fixture_kind,
        "runtime_identity_sha256": common.canonical_sha256(runtime),
        "case": {
            "experts": common.E,
            "hidden": common.H,
            "intermediate_tp": common.I,
            "topk": common.TOPK,
        },
        "fixture": fixture.manifest,
        "weights": weights.manifest,
        "reference": {
            "dtype": str(reference.dtype).replace("torch.", ""),
            "sha256": worker.tensor_sha256(reference),
            "implementation": str(worker.FIXTURE_PATH),
            "implementation_sha256": common.file_sha256(worker.FIXTURE_PATH),
        },
        "outputs": output_records,
        "output_stability": stability,
        "output_stability_gate": stability_gate,
        "route_task_evidence": [
            {
                "replay": replay,
                "json": f"workspace_replay_{replay}.json",
                "pt": f"workspace_replay_{replay}.pt",
                "summary": summary,
            }
            for replay, summary in enumerate(route_records)
        ],
        "route_replay_equal": route_replay_equal,
        "compile_identity": compile_identity,
        "artifact_stages": {
            "before_case_sha256": common.canonical_sha256(artifacts_before),
            "reference_delta": artifact_delta(
                artifacts_before, artifacts_after_reference
            ),
            "kernel_launch_delta": artifact_delta(
                artifacts_after_reference, artifacts_after_kernel
            ),
            "all_retained_cubin_sha256": sorted(
                {str(item["sha256"]) for item in cubins}
            ),
            "association_boundary": (
                "kernel_launch_delta is a fresh-JIT temporal association; exact "
                "symbol/launch identity still requires profiler/static parsing"
            ),
        },
        "launch_contract": {
            "expected_grid": list(EXPECTED_GRID),
            "expected_block": list(EXPECTED_BLOCK),
            "expected_kernel": "MoEDynamicKernel",
            "observed_by_profiler": False,
        },
        "gates": {
            "oracle_and_write_coverage": output_gate,
            "two_replay_stability": bool(stability_gate["gate_pass"]),
            "route_task": route_gate,
        },
        "gate_pass": output_gate and route_gate and bool(stability_gate["gate_pass"]),
        "cross_arm_boundary": (
            "final baseline-derived thresholds and candidate-vs-baseline errors "
            "cannot be decided by a one-arm runtime process"
        ),
    }
    common.write_json(raw_dir / "case.json", payload)
    if not payload["gate_pass"]:
        raise RuntimeError(f"validation gate failed for M={m} {fixture_kind}")
    return {
        "m": m,
        "fixture_kind": fixture_kind,
        "path": str((raw_dir / "case.json").relative_to(args.results)),
        "sha256": common.file_sha256(raw_dir / "case.json"),
        "reference_sha256": payload["reference"]["sha256"],
        "output_sha256": [record["output_sha256"] for record in output_records],
        "task_descriptor_multiset_sha256": [
            record["verification"]["task_descriptor_multiset_sha256"]
            for record in route_records
        ],
        "gate_pass": True,
    }


def validate(args: argparse.Namespace, runtime: Mapping[str, Any]) -> int:
    validation_root = args.results / "raw" / "validation" / args.arm
    require_new_output(validation_root, "arm validation directory")
    fixture_module = worker.load_fixture_module()
    device = torch.device("cuda", args.device_index)
    canonical_weights = fixture_module.make_canonical_weights(
        device=device, seed=args.seed
    )
    cases = []
    for m, fixture_kind in VALIDATION_CASES:
        cases.append(
            run_validation_case(
                args,
                runtime,
                fixture_module,
                canonical_weights,
                m=m,
                fixture_kind=fixture_kind,
            )
        )
        torch.cuda.empty_cache()

    artifacts = common.artifact_manifest(args.jit_root)
    cubins = [item for item in artifacts if item["path"].endswith(".cubin")]
    if not cubins:
        raise RuntimeError("validation retained no cubin artifacts")
    manifest = {
        "schema": "exp014.arm-validation.v1",
        "status": "complete",
        "arm": args.arm,
        "runtime": runtime,
        "imports": args.imports,
        "case_order": [list(case) for case in VALIDATION_CASES],
        "cases": cases,
        "jit_artifacts": artifacts,
        "jit_artifact_set_sha256": common.canonical_sha256(artifacts),
        "cubin_sha256": sorted({str(item["sha256"]) for item in cubins}),
        "compile_identity": worker._compile_identity(),
        "gate_pass": all(bool(case["gate_pass"]) for case in cases),
    }
    output = validation_root / "validation.json"
    common.write_json(output, manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


def benchmark_output_path(args: argparse.Namespace) -> Path:
    return (
        args.results
        / "raw"
        / "benchmark"
        / f"m{args.m}"
        / f"group_{args.group}_position_{args.position}_{args.arm}.json"
    )


def require_abba_position(arm: str, position: int) -> None:
    expected = ABBA[position]
    if arm != expected:
        raise RuntimeError(
            f"ABBA position {position} requires arm={expected}, got {arm}"
        )


def quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def measure(args: argparse.Namespace, runtime: Mapping[str, Any]) -> int:
    require_abba_position(args.arm, args.position)
    output_path = benchmark_output_path(args)
    require_new_output(output_path, "benchmark position")

    fixture_module = worker.load_fixture_module()
    device = torch.device("cuda", args.device_index)
    fixture = fixture_module.make_routed_fixture(args.m, device=device, seed=args.seed)
    weights = fixture_module.make_canonical_weights(device=device, seed=args.seed)
    arm = worker.build_arm(args, fixture, weights)
    arm.eager()
    arm.capture()
    flush, flush_bytes = worker.make_flusher(device, L2_FLUSH_BYTES)

    for _ in range(args.warmup):
        flush()
        arm.replay()
    samples_us = []
    last_output = None
    for _ in range(args.iters):
        flush()
        last_output, elapsed_ms = arm.replay()
        samples_us.append(elapsed_ms * 1000.0)
    assert last_output is not None
    if not bool(torch.isfinite(last_output).all().item()):
        raise RuntimeError("benchmark output contains non-finite values")

    gpu_after = selected_gpu(args.expected_gpu_uuid)
    require_clock(gpu_after, args.expected_app_clock_mhz)
    artifacts = common.artifact_manifest(args.jit_root)
    artifact_set_sha256 = common.canonical_sha256(artifacts)
    if artifact_set_sha256 != args.registered_jit_artifact_set_sha256:
        raise RuntimeError("measurement mutated the registered JIT artifact set")
    cubins = [item for item in artifacts if item["path"].endswith(".cubin")]
    if not cubins:
        raise RuntimeError("benchmark fresh JIT root retained no cubin")
    cubin_sha256 = sorted({str(item["sha256"]) for item in cubins})
    if cubin_sha256 != [args.expected_cubin_sha256]:
        raise RuntimeError("measurement loaded an unregistered cubin")
    payload = {
        "schema": "exp014.benchmark-position.v1",
        "status": "complete",
        "arm": args.arm,
        "m": args.m,
        "fixture_kind": CANONICAL,
        "group": args.group,
        "position": args.position,
        "abba_order": list(ABBA),
        "protocol": {
            "warmup": args.warmup,
            "iters": args.iters,
            "l2_flush_bytes": flush_bytes,
            "clock_policy": "locked",
            "expected_app_clock_mhz": args.expected_app_clock_mhz,
            "timing": "CUDA Graph external CUDA events; one sample per replay",
            "process_scope": "one arm/M/group/position in an independent process",
            "jit_policy": (
                "reuse one immutable, correctness-validated per-arm JIT root; "
                "artifact-set and cubin hashes are checked before and after"
            ),
        },
        "samples_us": samples_us,
        "statistics_us": {
            "count": len(samples_us),
            "mean": statistics.fmean(samples_us),
            "median": statistics.median(samples_us),
            "p10": quantile(samples_us, 0.10),
            "p90": quantile(samples_us, 0.90),
            "min": min(samples_us),
            "max": max(samples_us),
        },
        "sample_us": statistics.fmean(samples_us),
        "fixture": fixture.manifest,
        "weights": weights.manifest,
        "output_sha256": worker.tensor_sha256(last_output),
        "runtime": runtime,
        "imports": args.imports,
        "gpu_after": gpu_after,
        "jit_artifacts": artifacts,
        "jit_artifact_set_sha256": artifact_set_sha256,
        "cubin_sha256": cubin_sha256,
        "compile_identity": worker._compile_identity(),
        "evidence_boundary": (
            "this is one raw ABBA position; paired ratios, group bootstrap CI, "
            "and the no-regression verdict require the complete 20-position set"
        ),
    }
    common.write_json(output_path, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--device-index", type=int, default=0, choices=(0,))
    parser.add_argument("--seed", type=int, default=2026, choices=(2026,))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--m", type=int, choices=M_VALUES, required=True)
    measure_parser.add_argument("--group", type=int, choices=range(5), required=True)
    measure_parser.add_argument("--position", type=int, choices=range(4), required=True)
    measure_parser.add_argument("--warmup", type=int, choices=(WARMUP,), default=WARMUP)
    measure_parser.add_argument("--iters", type=int, choices=(ITERS,), default=ITERS)
    measure_parser.add_argument("--expected-app-clock-mhz", type=int, required=True)
    measure_parser.add_argument("--expected-jit-artifact-set-sha256", required=True)
    measure_parser.add_argument("--expected-cubin-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.results = args.results.resolve()
    args.overlay = args.overlay.resolve()
    args.jit_root = args.jit_root.resolve()
    if args.command == "validate":
        common.require_empty_directory(args.jit_root)
    else:
        args.registered_jit_artifact_set_sha256 = require_registered_measurement_jit(
            args
        )
    source = validate_source(args.flashinfer_root, args.overlay, args.arm)
    if TARGET_MODULE in sys.modules:
        raise RuntimeError("target module imported before exp_014 overlay installation")
    worker.install_overlay(args.overlay)
    args.imports = configure_source_checkout(args.flashinfer_root, args.jit_root)
    if Path(args.imports["target_module"]) != args.overlay:
        raise RuntimeError("target module did not resolve to selected arm overlay")
    runtime = runtime_identity(args, source)
    if args.command == "validate":
        return validate(args, runtime)
    if args.command == "measure":
        return measure(args, runtime)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
