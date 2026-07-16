#!/usr/bin/env python3
"""Run exp_001's fresh paired CuteDSL/CUTLASS BF16-input benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from fixture import load_fixture
from nvfp4_fixture import (
    E,
    H,
    I,
    TOPK,
    CanonicalWeights,
    RoutedFixture,
    make_canonical_weights,
    output_diagnostics,
    reference_moe_nvfp4,
    tensor_sha256,
)

ARMS = (
    "cutedsl_bf16_fused",
    "cutlass_bf16_chain",
)
PAIRED_ARMS = ARMS
EXPECTED_FLASHINFER_COMMIT = "074d93e4aa54c75bee1b3dfdb39b7f075a3ff2af"
EXPECTED_CUTLASS_COMMIT = "b46b16d003484063bca4ed365e44095c4c6ed633"
EXPECTED_IMAGE_DIGEST = (
    "sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba"
)
EXPECTED_PYTHON_DEPS_SHA256 = (
    "32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74"
)
EXPECTED_PYTHON_DEPS_ROOT = Path("/workspace/deps")
COMPARISON_GROUP_ID = "exp001_cutedsl_vs_cutlass_bf16_fresh"
RERUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = EXPERIMENT_ROOT / "results" / "pair"
DEFAULT_FIXTURES = EXPERIMENT_ROOT / "results" / "fixtures"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def command_output(command: list[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            command, cwd=str(cwd) if cwd else None, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"ERROR: {error}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    """Hash a stable sequence of file-content digests and relative paths."""
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(f"{file_sha256(path)}  ./{relative}\n".encode())
    return digest.hexdigest()


def configure_source_checkout(repo_root: Path) -> None:
    """Force JIT includes to the exact source checkout under test."""
    import flashinfer
    from flashinfer.jit import env as jit_env

    imported_root = Path(flashinfer.__file__).resolve().parents[1]
    if imported_root != repo_root:
        raise RuntimeError(
            f"imported FlashInfer root {imported_root} != requested {repo_root}"
        )
    csrc = repo_root / "csrc"
    jit_env.FLASHINFER_CSRC_DIR = csrc
    jit_env.FLASHINFER_INCLUDE_DIR = repo_root / "include"
    jit_env.FLASHINFER_AOT_DIR = (
        jit_env.FLASHINFER_WORKSPACE_DIR / "aot_disabled_for_exp001"
    )
    jit_env.CUTLASS_INCLUDE_DIRS = [
        repo_root / "3rdparty" / "cutlass" / "include",
        repo_root / "3rdparty" / "cutlass" / "tools" / "util" / "include",
    ]
    jit_env.CCCL_INCLUDE_DIRS = [
        repo_root / "3rdparty" / "cccl" / "cub",
        repo_root / "3rdparty" / "cccl" / "libcudacxx" / "include",
        repo_root / "3rdparty" / "cccl" / "thrust",
    ]
    jit_env.SPDLOG_INCLUDE_DIR = repo_root / "3rdparty" / "spdlog" / "include"


def validate_source(repo_root: Path) -> dict[str, Any]:
    flashinfer_git = ["git", "-c", f"safe.directory={repo_root}"]
    cutlass_root = repo_root / "3rdparty" / "cutlass"
    cutlass_git = ["git", "-c", f"safe.directory={cutlass_root}"]
    flashinfer_commit = command_output(
        [*flashinfer_git, "rev-parse", "HEAD"], cwd=repo_root
    )
    cutlass_commit = command_output(
        [*cutlass_git, "rev-parse", "HEAD"], cwd=cutlass_root
    )
    if flashinfer_commit != EXPECTED_FLASHINFER_COMMIT:
        raise RuntimeError(
            f"FlashInfer commit drift: {flashinfer_commit} != {EXPECTED_FLASHINFER_COMMIT}"
        )
    if cutlass_commit != EXPECTED_CUTLASS_COMMIT:
        raise RuntimeError(
            f"CUTLASS commit drift: {cutlass_commit} != {EXPECTED_CUTLASS_COMMIT}"
        )
    allowed_prefix = (
        ".claude/w4a4_moe_bench/experiments/exp_001_backend_case_sweep/"
    )
    allowed_paths = {
        ".claude/w4a4_moe_bench/cutedsl_460_requirements.lock.txt",
    }
    tracked_changes = command_output(
        [*flashinfer_git, "diff", "--name-only"], cwd=repo_root
    ).splitlines()
    staged_changes = command_output(
        [*flashinfer_git, "diff", "--cached", "--name-only"], cwd=repo_root
    ).splitlines()
    unexpected_tracked = [
        path
        for path in tracked_changes + staged_changes
        if not path.startswith(allowed_prefix) and path not in allowed_paths
    ]
    if unexpected_tracked:
        raise RuntimeError(
            "production source outside the exp_001 overlay is dirty: "
            + ", ".join(unexpected_tracked)
        )
    cutlass_changes = command_output(
        [*cutlass_git, "diff", "--name-only"], cwd=cutlass_root
    )
    cutlass_staged = command_output(
        [*cutlass_git, "diff", "--cached", "--name-only"], cwd=cutlass_root
    )
    cutlass_untracked = command_output(
        [*cutlass_git, "ls-files", "--others", "--exclude-standard"],
        cwd=cutlass_root,
    )
    if cutlass_changes or cutlass_staged or cutlass_untracked:
        raise RuntimeError("CUTLASS submodule is dirty")
    untracked = command_output(
        [*flashinfer_git, "ls-files", "--others", "--exclude-standard"],
        cwd=repo_root,
    ).splitlines()
    unexpected_untracked = [
        path
        for path in untracked
        if not path.startswith(allowed_prefix) and path not in allowed_paths
    ]
    if unexpected_untracked:
        raise RuntimeError(
            "unexpected untracked source outside exp_001 overlay: "
            + ", ".join(unexpected_untracked)
        )
    overlays = []
    for name in (
        "plan.md",
        "fixture.py",
        "nvfp4_fixture.py",
        "make_fixtures.py",
        "run_pair.py",
        "run_pair.sh",
        "build_result.py",
    ):
        path = EXPERIMENT_ROOT / name
        if path.exists():
            overlays.append(
                {"path": str(path.relative_to(repo_root)), "sha256": file_sha256(path)}
            )
    dependency_lock = (
        repo_root / ".claude/w4a4_moe_bench/cutedsl_460_requirements.lock.txt"
    )
    overlays.append(
        {
            "path": str(dependency_lock.relative_to(repo_root)),
            "sha256": file_sha256(dependency_lock),
        }
    )
    return {
        "flashinfer_commit": flashinfer_commit,
        "cutlass_commit": cutlass_commit,
        "source_status": command_output(
            [*flashinfer_git, "status", "--short"], cwd=repo_root
        ),
        "experiment_overlays": overlays,
    }


def query_gpu_uuid() -> str:
    output = command_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"])
    return output.splitlines()[0].strip()


def runtime_manifest(
    *, repo_root: Path, expected_gpu_uuid: str, source: dict[str, Any]
) -> dict[str, Any]:
    import cutlass
    import tvm_ffi

    actual_uuid = query_gpu_uuid()
    if actual_uuid != expected_gpu_uuid:
        raise RuntimeError(f"GPU UUID drift: {actual_uuid} != {expected_gpu_uuid}")
    workspace = os.environ.get("FLASHINFER_WORKSPACE_BASE", "")
    if not workspace:
        raise RuntimeError(
            "FLASHINFER_WORKSPACE_BASE must name a dedicated exp_001 path"
        )
    image_digest = os.environ.get("W4A4_IMAGE_DIGEST", "")
    if image_digest != EXPECTED_IMAGE_DIGEST:
        raise RuntimeError(
            f"container image digest drift: {image_digest} != {EXPECTED_IMAGE_DIGEST}"
        )
    dependency_sha256 = os.environ.get("W4A4_PYTHON_DEPS_SHA256", "")
    if dependency_sha256 != EXPECTED_PYTHON_DEPS_SHA256:
        raise RuntimeError(
            "Python dependency overlay hash drift: "
            f"{dependency_sha256} != {EXPECTED_PYTHON_DEPS_SHA256}"
        )
    actual_dependency_sha256 = tree_sha256(EXPECTED_PYTHON_DEPS_ROOT)
    if actual_dependency_sha256 != dependency_sha256:
        raise RuntimeError(
            "Python dependency overlay content drift: "
            f"{actual_dependency_sha256} != {dependency_sha256}"
        )
    lease_id = os.environ.get("KDK_LEASE_ID", "")
    if not lease_id:
        raise RuntimeError("KDK_LEASE_ID must identify the active GPU lease")
    rerun_id = os.environ.get("W4A4_RERUN_ID", "")
    if not RERUN_ID_PATTERN.fullmatch(rerun_id):
        raise RuntimeError("W4A4_RERUN_ID must be a unique 8-128 character identifier")
    forbidden_overrides = (
        "FLASHINFER_AUTOTUNER_LOAD_FROM_FILE",
        "FLASHINFER_TACTICS_BLOCKLIST",
    )
    enabled_overrides = [key for key in forbidden_overrides if os.environ.get(key)]
    if enabled_overrides:
        raise RuntimeError(f"forbidden tactic overrides are set: {enabled_overrides}")
    module_paths = {
        "cutlass_module": Path(cutlass.__file__).resolve(),
        "tvm_ffi_module": Path(tvm_ffi.__file__).resolve(),
    }
    for name, path in module_paths.items():
        try:
            path.relative_to(EXPECTED_PYTHON_DEPS_ROOT)
        except ValueError as error:
            raise RuntimeError(
                f"{name} did not load from dependency overlay: {path}"
            ) from error
    foreign_processes = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid",
            "--format=csv,noheader",
        ]
    )
    if foreign_processes:
        raise RuntimeError(
            "foreign GPU compute process present before experiment:\n"
            + foreign_processes
        )
    return {
        "timestamp_unix": time.time(),
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_uuid": actual_uuid,
        "resource_lease_id": lease_id,
        "measurement_rerun_id": rerun_id,
        "gpu_name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "image": "nvcr.io/nvidia/pytorch:26.05-py3",
        "image_digest": image_digest,
        "python_dependency_overlay": {
            "sha256": dependency_sha256,
            "apache_tvm_ffi": importlib.metadata.version("apache-tvm-ffi"),
            "nvidia_cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
            "numpy": importlib.metadata.version("numpy"),
            "cuda_python": importlib.metadata.version("cuda-python"),
            "nvidia_cuda_nvdisasm": importlib.metadata.version("nvidia-cuda-nvdisasm"),
            **{name: str(path) for name, path in module_paths.items()},
        },
        "flashinfer_workspace_base": str(Path(workspace).resolve()),
        "enable_pdl": True,
        "use_fused_finalize": True,
        "cutlass_autotuner_contract": "default non-tuning fallback tactic -1 / built-in heuristic",
        "cuda_graph": "outer graph, explicit non-default capture stream, external events",
        "foreign_compute_process_query": foreign_processes,
        "environment": {
            key: os.environ.get(key, "")
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "FLASHINFER_WORKSPACE_BASE",
                "FLASHINFER_LOGLEVEL",
                "FLASHINFER_CUTEDSL_IKET_OVERLAY",
                "FLASHINFER_NVFP4_4OVER6",
                "FLASHINFER_AUTOTUNER_LOAD_FROM_FILE",
                "FLASHINFER_TACTICS_BLOCKLIST",
                "W4A4_IMAGE_DIGEST",
                "W4A4_PYTHON_DEPS_SHA256",
                "PYTHONPATH",
            )
        },
        "source": source,
        "repo_root": str(repo_root),
    }


def jit_manifest() -> list[dict[str, Any]]:
    workspace = Path(os.environ["FLASHINFER_WORKSPACE_BASE"]).resolve()
    if not workspace.exists():
        return []
    artifacts = []
    suffixes = {".so", ".cu", ".cuh", ".cpp", ".ptx", ".cubin", ".json"}
    for path in sorted(workspace.rglob("*")):
        if path.is_file() and path.suffix in suffixes:
            artifacts.append(
                {
                    "path": str(path.relative_to(workspace)),
                    "size": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return artifacts


def validate_jit_identity(results: Path) -> None:
    expected_path = results / "manifests" / "jit_artifacts.json"
    if not expected_path.exists():
        raise RuntimeError("benchmark JIT artifact manifest is missing")
    if json.loads(expected_path.read_text()) != jit_manifest():
        raise RuntimeError("profile JIT artifact identity drift")


def stable_runtime_identity(runtime: dict[str, Any]) -> dict[str, Any]:
    identity = dict(runtime)
    identity.pop("timestamp_unix", None)
    # Ephemeral Docker container hostnames differ across benchmark/profile runs.
    identity.pop("hostname", None)
    # A later profiler capture may reacquire the same physical GPU under a new lease.
    identity.pop("resource_lease_id", None)
    identity.pop("measurement_rerun_id", None)
    return identity


def build_environment_lock(runtime: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": 2,
        "runtime": stable_runtime_identity(runtime),
    }
    return {**payload, "fingerprint_sha256": canonical_sha256(payload)}


def build_artifact_lock(
    artifacts: list[dict[str, Any]], arm_contracts: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    artifact_set_sha256 = canonical_sha256(artifacts)
    per_arm = {
        arm: canonical_sha256(
            {
                "arm": arm,
                "resolved_contract": arm_contracts[arm],
                "dedicated_jit_artifact_set_sha256": artifact_set_sha256,
            }
        )
        for arm in ARMS
    }
    payload = {
        "schema_version": 2,
        "workspace_policy": "dedicated empty workspace before benchmark",
        "jit_artifacts": artifacts,
        "artifact_set_sha256": artifact_set_sha256,
        "per_arm_artifact_fingerprint_sha256": per_arm,
    }
    return {**payload, "fingerprint_sha256": canonical_sha256(payload)}


def build_protocol_lock(args: argparse.Namespace, flush_bytes: int) -> dict[str, Any]:
    payload = {
        "schema_version": 2,
        "arms": list(ARMS),
        "m_values": list(args.m_values),
        "warmup": args.warmup,
        "iterations_per_sample": args.iters,
        "repeats": args.repeats,
        "order_policy": "even A>B; odd B>A; every repeat contains all arms",
        "cache_policy": {
            "l2_flush_bytes": flush_bytes,
            "placement": "before each replay, outside timed interval",
        },
        "timing": "outer CUDA graph replay; external CUDA events inside graph",
        "aggregation": "median of per-repeat means; preserve all raw samples",
    }
    return {**payload, "fingerprint_sha256": canonical_sha256(payload)}


def write_content_addressed_lock(
    results: Path, directory: str, lock: dict[str, Any]
) -> dict[str, Any]:
    digest = str(lock["fingerprint_sha256"])
    path = results / directory / f"{digest}.json"
    if path.exists():
        if json.loads(path.read_text()) != lock:
            raise RuntimeError(f"content-addressed lock collision at {path}")
    else:
        write_json(path, lock)
    return str(path.relative_to(results))


def write_evidence_identity(
    results: Path,
    runtime: dict[str, Any],
    artifacts: list[dict[str, Any]],
    arm_contracts: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    environment = build_environment_lock(runtime)
    artifact = build_artifact_lock(artifacts, arm_contracts)
    environment_path = write_content_addressed_lock(
        results, "environment-locks", environment
    )
    artifact_path = write_content_addressed_lock(results, "artifact-locks", artifact)
    protocol_path = write_content_addressed_lock(results, "protocol-locks", protocol)
    identity = {
        "schema_version": 2,
        "comparison_group_id": COMPARISON_GROUP_ID,
        "rerun_id": runtime["measurement_rerun_id"],
        "environment_lock_digest": environment["fingerprint_sha256"],
        "environment_lock_path": environment_path,
        "artifact_lock_digest": artifact["fingerprint_sha256"],
        "artifact_lock_path": artifact_path,
        "protocol_lock_digest": protocol["fingerprint_sha256"],
        "protocol_lock_path": protocol_path,
        "per_arm_artifact_fingerprint_sha256": artifact[
            "per_arm_artifact_fingerprint_sha256"
        ],
    }
    rerun_path = results / "reruns" / f"{identity['rerun_id']}.json"
    if rerun_path.exists():
        raise RuntimeError(f"rerun_id already exists: {identity['rerun_id']}")
    write_json(rerun_path, identity)
    write_json(results / "evidence.identity.json", identity)
    # Compatibility pointer only; immutable lock payloads live in the digest paths.
    write_json(
        results / "environment.lock.json",
        {
            "schema_version": 2,
            "canonical_identity": "evidence.identity.json",
            **identity,
        },
    )
    return identity


def validate_evidence_identity(
    results: Path, runtime: dict[str, Any]
) -> dict[str, Any]:
    path = results / "evidence.identity.json"
    if not path.exists():
        raise RuntimeError("benchmark evidence identity is missing")
    identity = json.loads(path.read_text())
    if identity.get("rerun_id") != runtime["measurement_rerun_id"]:
        raise RuntimeError("profile rerun_id does not match benchmark rerun")
    environment = json.loads((results / identity["environment_lock_path"]).read_text())
    if environment != build_environment_lock(runtime):
        raise RuntimeError("profile shared environment/toolchain drift")
    artifact = json.loads((results / identity["artifact_lock_path"]).read_text())
    if artifact.get("jit_artifacts") != jit_manifest():
        raise RuntimeError("profile JIT artifact identity drift")
    return identity


def validate_output(output: torch.Tensor, m: int) -> None:
    if tuple(output.shape) != (m, H):
        raise ValueError(f"output shape {tuple(output.shape)} != {(m, H)}")
    if output.dtype != torch.bfloat16:
        raise ValueError(f"output dtype {output.dtype} != torch.bfloat16")
    if not torch.isfinite(output).all().item():
        raise ValueError("output contains non-finite values")
    if not torch.count_nonzero(output).item():
        raise ValueError("output is all zero")


def load_routed_fixture(
    root: Path, m: int, *, device: torch.device
) -> RoutedFixture:
    """Load the exact persisted fixture shared with the SGLang runtime."""
    x, topk_ids, topk_weights, manifest = load_fixture(root, m, device)
    return RoutedFixture(m, x, topk_ids, topk_weights, manifest)


@dataclass
class CapturedArm:
    name: str
    launch: Callable[[], torch.Tensor]
    metadata: dict[str, Any]
    output: torch.Tensor | None = None
    graph: torch.cuda.CUDAGraph | None = None
    start: torch.cuda.Event | None = None
    end: torch.cuda.Event | None = None
    capture_stream: torch.cuda.Stream | None = None

    def eager(self) -> torch.Tensor:
        self.output = self.launch()
        torch.cuda.synchronize()
        assert self.output is not None
        return self.output

    def capture(self) -> None:
        self.capture_stream = torch.cuda.Stream()
        with torch.cuda.stream(self.capture_stream):
            self.launch()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        self.start = torch.cuda.Event(enable_timing=True, external=True)
        self.end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            self.start.record()
            self.output = self.launch()
            self.end.record()
        torch.cuda.synchronize()

    def replay_ms(self) -> float:
        if self.graph is None or self.start is None or self.end is None:
            raise RuntimeError(f"arm {self.name} has not been captured")
        self.graph.replay()
        torch.cuda.synchronize()
        return float(self.start.elapsed_time(self.end))


def build_arm(
    name: str,
    *,
    fixture: RoutedFixture,
    weights: CanonicalWeights,
    max_num_tokens: int,
) -> CapturedArm:
    if name == "cutedsl_bf16_fused":
        from flashinfer.fused_moe.cute_dsl import B12xMoEWrapper

        values = weights.cutedsl()
        wrapper = B12xMoEWrapper(
            num_experts=E,
            top_k=TOPK,
            hidden_size=H,
            intermediate_size=I,
            use_cuda_graph=True,
            max_num_tokens=max_num_tokens,
            output_dtype=torch.bfloat16,
            device=str(fixture.x.device),
            activation="silu",
            quant_mode="w4a4",
            source_format="modelopt",
        )

        def launch_cutedsl() -> torch.Tensor:
            return wrapper.run(
                x=fixture.x,
                w1_weight=values["w1_fp4"],
                w1_weight_sf=values["w1_sf"],
                w2_weight=values["w2_fp4"],
                w2_weight_sf=values["w2_sf"],
                token_selected_experts=fixture.topk_ids,
                token_final_scales=fixture.topk_weights,
                w1_alpha=values["w1_alpha"],
                w2_alpha=values["w2_alpha"],
                fc2_input_scale=values["fc2_input_scale"],
            )

        return CapturedArm(
            name,
            launch_cutedsl,
            {
                "boundary": "BF16 input -> fused W4A4 MoE -> BF16 output",
                "role": "paired target",
                "backend": "flashinfer.fused_moe.cute_dsl.B12xMoEWrapper",
                "expected_launches": "one material fused kernel; NSys is authority",
                "input_dtype": "bfloat16",
            },
        )

    if name != "cutlass_bf16_chain":
        raise ValueError(f"unknown arm {name}")

    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    values = weights.cutlass()
    a1_gs = torch.ones((), device=fixture.x.device, dtype=torch.float32)
    a2_gs = torch.ones((), device=fixture.x.device, dtype=torch.float32)
    output = torch.empty_like(fixture.x)
    quant_scales = [
        a1_gs,
        values["fc1_blockscale_i32"],
        1.0 / (a1_gs * values["fc1_gs"]),
        a2_gs,
        values["fc2_blockscale_i32"],
        1.0 / (a2_gs * values["fc2_gs"]),
    ]
    input_tensor, input_sf = fixture.x, None
    boundary = "BF16 input -> native CUTLASS online quant chain -> BF16 output"
    role = "paired baseline"

    def launch_cutlass() -> torch.Tensor:
        cutlass_fused_moe(
            input_tensor,
            fixture.topk_ids,
            fixture.topk_weights,
            values["fc1_weight"],
            values["fc2_weight"],
            torch.bfloat16,
            quant_scales=quant_scales,
            input_sf=input_sf,
            output=output,
            tune_max_num_tokens=max_num_tokens,
            enable_pdl=True,
            activation_type=ActivationType.Swiglu,
            use_fused_finalize=True,
        )
        # The current custom-op binding returns an auxiliary list on the normal
        # path; the API writes the requested result into this stable buffer.
        return output

    return CapturedArm(
        name,
        launch_cutlass,
        {
            "boundary": boundary,
            "role": role,
            "backend": "flashinfer.fused_moe.cutlass_fused_moe",
            "expected_launches": "multi-kernel chain; NSys is authority",
            "input_dtype": str(input_tensor.dtype).replace("torch.", ""),
            "input_sf": input_sf is not None,
            "enable_pdl": True,
            "use_fused_finalize": True,
        },
    )


def make_l2_flusher(
    *, device: torch.device, num_bytes: int
) -> tuple[Callable[[], None], int]:
    buffer = torch.empty((num_bytes + 3) // 4, device=device, dtype=torch.int32)
    state = 0

    def flush() -> None:
        nonlocal state
        state = (state + 1) & 0x7FFFFFFF
        buffer.fill_(state)
        torch.cuda.synchronize()

    flush()
    return flush, buffer.numel() * buffer.element_size()


def observed_cuda_kernels(arm: CapturedArm) -> list[str]:
    """Record one graph replay's CUDA kernels without entering timed samples."""
    from torch.profiler import ProfilerActivity, profile

    if arm.graph is None:
        raise RuntimeError(f"arm {arm.name} has not been captured")
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        arm.graph.replay()
        torch.cuda.synchronize()
    names: list[str] = []
    for event in trace.events():
        if "cuda" in str(getattr(event, "device_type", "")).lower():
            if event.name not in names:
                names.append(event.name)
    if not names:
        raise RuntimeError(f"profiler observed no CUDA kernels for {arm.name}")
    lowered = [name.lower() for name in names]
    if arm.name == "cutedsl_bf16_fused":
        if not any("moedynamickernel" in name for name in lowered):
            raise RuntimeError(f"CuteDSL fused kernel not observed: {names}")
    else:
        required = ("expandinputrowskernel",)
        missing = [
            needle for needle in required if not any(needle in name for name in lowered)
        ]
        if not any(
            "cutlass::device_kernel" in name or "cutlass13device_kernel" in name
            for name in lowered
        ):
            missing.append("CUTLASS device GEMM kernel")
        if missing:
            raise RuntimeError(f"CUTLASS BF16 online chain kernels missing {missing}: {names}")
    return names


def correctness_case(
    fixture: RoutedFixture,
    weights: CanonicalWeights,
    arms: dict[str, CapturedArm],
) -> dict[str, Any]:
    reference = reference_moe_nvfp4(fixture, weights)
    torch.cuda.synchronize()
    if tuple(reference.shape) != (fixture.m, H) or reference.dtype != torch.float32:
        raise ValueError(
            f"reference shape/dtype {tuple(reference.shape)}/{reference.dtype} is invalid"
        )
    if (
        not torch.isfinite(reference).all().item()
        or not torch.count_nonzero(reference).item()
    ):
        raise ValueError("reference is non-finite or all zero")
    result: dict[str, Any] = {
        "fixture": fixture.manifest,
        "reference_sha256": tensor_sha256(reference),
        "formal_gate": {
            "oracle": "dequantized per-expert PyTorch MoE with input and activation NVFP4 round trips",
            "criterion": "at least 97% satisfy abs<max(0.05,1.5*oracle.std) OR rel<0.5",
        },
        "arms": {},
    }
    outputs: dict[str, torch.Tensor] = {}
    for name, arm in arms.items():
        # JIT/workspace/tactic preparation is eager and outside the graph.  The
        # formal correctness source is the first actual graph replay output.
        arm.eager()
        arm.capture()
        arm.replay_ms()
        assert arm.output is not None
        output = arm.output.clone()
        validate_output(output, fixture.m)
        outputs[name] = output
        result["arms"][name] = {
            **arm.metadata,
            **output_diagnostics(output, reference),
            "observed_cuda_kernels": observed_cuda_kernels(arm),
            "dispatch_pass": True,
        }
    target = outputs["cutedsl_bf16_fused"].float()
    baseline = outputs["cutlass_bf16_chain"].float()
    result["cross_backend_diagnostic"] = {
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                target.flatten(), baseline.flatten(), dim=0
            ).item()
        ),
        "relative_l2": float(
            (
                torch.linalg.vector_norm(target - baseline)
                / torch.linalg.vector_norm(baseline).clamp_min(1e-12)
            ).item()
        ),
    }
    result["paired_gate_pass"] = all(
        bool(result["arms"][name]["formal_pass"]) for name in PAIRED_ARMS
    )
    return result


def benchmark(args: argparse.Namespace, runtime: dict[str, Any]) -> int:
    results = args.results.resolve()
    if (results / "evidence.identity.json").exists():
        raise RuntimeError("canonical results already contain a completed rerun")
    if jit_manifest():
        raise RuntimeError("dedicated JIT workspace is not empty before benchmark")
    device = torch.device("cuda", args.device_index)
    weights = make_canonical_weights(device=device, seed=args.seed)
    flush_l2, flush_bytes = make_l2_flusher(
        device=device, num_bytes=args.l2_flush_bytes
    )
    protocol = build_protocol_lock(args, flush_bytes)
    correctness: dict[str, Any] = {
        "runtime": runtime,
        "experiment_config": {
            "seed": args.seed,
            "max_num_tokens": args.max_num_tokens,
            "m_values": args.m_values,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "l2_flush_bytes": flush_bytes,
        },
        "weights": weights.manifest,
        "measurement_protocol": protocol,
        "cases": {},
    }
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    arm_contracts: dict[str, dict[str, Any]] = {}

    for m in args.m_values:
        print(f"exp001 paired benchmark prepare m={m}", flush=True)
        fixture = load_routed_fixture(args.fixtures, m, device=device)
        arms = {
            name: build_arm(
                name,
                fixture=fixture,
                weights=weights,
                max_num_tokens=args.max_num_tokens,
            )
            for name in ARMS
        }
        current_contracts = {name: arms[name].metadata for name in ARMS}
        if arm_contracts and current_contracts != arm_contracts:
            raise RuntimeError("resolved arm contract drifted across M cases")
        arm_contracts = current_contracts
        case_correctness = correctness_case(fixture, weights, arms)
        correctness["cases"][str(m)] = case_correctness
        write_json(results / "correctness.json", correctness)
        write_json(results / "fixtures" / f"m{m}_manifest.json", fixture.manifest)
        if not case_correctness["paired_gate_pass"]:
            print(f"formal paired correctness failed at m={m}; stopping timing")
            artifacts = jit_manifest()
            write_json(results / "manifests" / "jit_artifacts.json", artifacts)
            identity = write_evidence_identity(
                results, runtime, artifacts, arm_contracts, protocol
            )
            correctness["evidence_identity"] = identity
            write_json(results / "correctness.json", correctness)
            return 2

        for _ in range(args.warmup):
            for name in ARMS:
                flush_l2()
                arms[name].replay_ms()

        samples: dict[str, list[float]] = {name: [] for name in ARMS}
        for repeat in range(args.repeats):
            order = ARMS if repeat % 2 == 0 else tuple(reversed(ARMS))
            for order_index, name in enumerate(order):
                elapsed_ms = 0.0
                for _ in range(args.iters):
                    flush_l2()
                    elapsed_ms += arms[name].replay_ms()
                sample_ms = elapsed_ms / args.iters
                samples[name].append(sample_ms)
                raw_rows.append(
                    {
                        "m": m,
                        "repeat": repeat,
                        "order": ">".join(order),
                        "order_index": order_index,
                        "arm": name,
                        "sample_us": sample_ms * 1000.0,
                        "iters": args.iters,
                        "l2_flush_bytes": flush_bytes,
                        "fixture_sha256": fixture.manifest["fixture_sha256"],
                        "occupancy_sha256": fixture.manifest["occupancy_sha256"],
                    }
                )
                write_csv(results / "benchmark_raw.csv", raw_rows)

        for name in ARMS:
            arm_samples = samples[name]
            median_us = statistics.median(arm_samples) * 1000.0
            spread_percent = (
                (max(arm_samples) - min(arm_samples))
                / statistics.median(arm_samples)
                * 100.0
            )
            summary_rows.append(
                {
                    "m": m,
                    "arm": name,
                    "median_us": median_us,
                    "min_us": min(arm_samples) * 1000.0,
                    "max_us": max(arm_samples) * 1000.0,
                    "spread_percent": spread_percent,
                    "stable_le_5_percent": spread_percent <= 5.0,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "repeats": args.repeats,
                    "timing": "cuda_graph_external_events_inside",
                    "l2_flush_bytes": flush_bytes,
                    "boundary": arms[name].metadata["boundary"],
                    "role": arms[name].metadata["role"],
                    "fixture_sha256": fixture.manifest["fixture_sha256"],
                    "occupancy_sha256": fixture.manifest["occupancy_sha256"],
                }
            )
        write_csv(results / "benchmark_summary.csv", summary_rows)
        del arms, fixture
        torch.cuda.empty_cache()

    correctness["all_paired_gates_pass"] = all(
        value["paired_gate_pass"] for value in correctness["cases"].values()
    )
    artifacts = jit_manifest()
    write_json(results / "manifests" / "jit_artifacts.json", artifacts)
    identity = write_evidence_identity(
        results, runtime, artifacts, arm_contracts, protocol
    )
    correctness["evidence_identity"] = identity
    for row in raw_rows:
        row["comparison_group_id"] = identity["comparison_group_id"]
        row["rerun_id"] = identity["rerun_id"]
        row["environment_lock_digest"] = identity["environment_lock_digest"]
        row["protocol_lock_digest"] = identity["protocol_lock_digest"]
        row["artifact_fingerprint_sha256"] = identity[
            "per_arm_artifact_fingerprint_sha256"
        ][row["arm"]]
    for row in summary_rows:
        row["comparison_group_id"] = identity["comparison_group_id"]
        row["rerun_id"] = identity["rerun_id"]
        row["environment_lock_digest"] = identity["environment_lock_digest"]
        row["protocol_lock_digest"] = identity["protocol_lock_digest"]
        row["artifact_fingerprint_sha256"] = identity[
            "per_arm_artifact_fingerprint_sha256"
        ][row["arm"]]
    write_csv(results / "benchmark_raw.csv", raw_rows)
    write_csv(results / "benchmark_summary.csv", summary_rows)
    write_json(results / "correctness.json", correctness)
    write_json(results / "manifests" / "runtime.json", runtime)
    unstable = any(not row["stable_le_5_percent"] for row in summary_rows)
    return int(unstable)


def validate_profile_prerequisite(
    results: Path,
    fixture: RoutedFixture,
    weights: CanonicalWeights,
    m: int,
    arm: str,
    runtime: dict[str, Any],
    max_num_tokens: int,
) -> str:
    path = results / "correctness.json"
    if not path.exists():
        raise RuntimeError("run correctness-qualified benchmark before profiling")
    correctness = json.loads(path.read_text())
    case = correctness.get("cases", {}).get(str(m))
    if not case or not case.get("paired_gate_pass"):
        raise RuntimeError(f"m={m} has no passing paired correctness gate")
    if not case["arms"].get(arm, {}).get("formal_pass"):
        raise RuntimeError(f"m={m} arm={arm} has no passing per-arm oracle gate")
    if correctness.get("experiment_config", {}).get("max_num_tokens") != max_num_tokens:
        raise RuntimeError("profile max_num_tokens drift")

    fixture_keys = (
        "x_sha256",
        "topk_ids_sha256",
        "topk_weights_sha256",
        "occupancy_sha256",
        "seed",
        "shape",
    )
    for key in fixture_keys:
        if case["fixture"].get(key) != fixture.manifest.get(key):
            raise RuntimeError(f"profile fixture drift at {key}")
    if correctness["weights"] != weights.manifest:
        raise RuntimeError("profile canonical weight/scale manifest drift")

    validate_jit_identity(results)
    identity = validate_evidence_identity(results, runtime)
    if correctness.get("evidence_identity") != identity:
        raise RuntimeError("correctness/evidence identity mismatch")

    previous_runtime = correctness["runtime"]
    runtime_checks = (
        ("gpu_uuid", previous_runtime["gpu_uuid"], runtime["gpu_uuid"]),
        (
            "dependency overlay",
            previous_runtime["python_dependency_overlay"],
            runtime["python_dependency_overlay"],
        ),
        (
            "environment",
            previous_runtime["environment"],
            runtime["environment"],
        ),
        (
            "source identity",
            {
                key: previous_runtime["source"][key]
                for key in (
                    "flashinfer_commit",
                    "cutlass_commit",
                    "experiment_overlays",
                )
            },
            {
                key: runtime["source"][key]
                for key in (
                    "flashinfer_commit",
                    "cutlass_commit",
                    "experiment_overlays",
                )
            },
        ),
        (
            "runtime software",
            {
                key: previous_runtime[key]
                for key in (
                    "python",
                    "torch",
                    "cuda_runtime",
                    "gpu_name",
                    "compute_capability",
                    "image",
                    "image_digest",
                    "repo_root",
                )
            },
            {
                key: runtime[key]
                for key in (
                    "python",
                    "torch",
                    "cuda_runtime",
                    "gpu_name",
                    "compute_capability",
                    "image",
                    "image_digest",
                    "repo_root",
                )
            },
        ),
    )
    for label, expected, actual in runtime_checks:
        if expected != actual:
            raise RuntimeError(f"profile {label} drift")

    summary_rows = []
    with (results / "benchmark_summary.csv").open(newline="") as file:
        summary_rows = list(csv.DictReader(file))
    matches = [row for row in summary_rows if int(row["m"]) == m and row["arm"] == arm]
    if len(matches) != 1 or matches[0]["stable_le_5_percent"].lower() != "true":
        raise RuntimeError(f"m={m} arm={arm} has no unique stable benchmark row")
    row = matches[0]
    expected_fields = {
        "comparison_group_id": identity["comparison_group_id"],
        "rerun_id": identity["rerun_id"],
        "environment_lock_digest": identity["environment_lock_digest"],
        "protocol_lock_digest": identity["protocol_lock_digest"],
        "artifact_fingerprint_sha256": identity["per_arm_artifact_fingerprint_sha256"][
            arm
        ],
    }
    for key, expected in expected_fields.items():
        if row.get(key) != expected:
            raise RuntimeError(f"benchmark row identity drift at {key}")
    return identity


def single_replay(args: argparse.Namespace, runtime: dict[str, Any]) -> int:
    results = args.results.resolve()
    device = torch.device("cuda", args.device_index)
    weights = make_canonical_weights(device=device, seed=args.seed)
    fixture = load_routed_fixture(args.fixtures, args.m, device=device)
    identity = validate_profile_prerequisite(
        results,
        fixture,
        weights,
        args.m,
        args.arm,
        runtime,
        args.max_num_tokens,
    )
    arm = build_arm(
        args.arm,
        fixture=fixture,
        weights=weights,
        max_num_tokens=args.max_num_tokens,
    )
    output = arm.eager()
    validate_output(output, args.m)
    arm.capture()
    validate_jit_identity(results)
    for _ in range(args.warmup):
        arm.replay_ms()

    nvtx_name = f"exp001_m{args.m}_{args.arm}_single_replay"
    manifest_path = results / "manifests" / f"profile_m{args.m}_{args.arm}.json"
    profile_manifest = {
        "runtime": runtime,
        "m": args.m,
        "arm": args.arm,
        "nvtx_range": nvtx_name,
        "fixture": fixture.manifest,
        "weights": weights.manifest,
        "arm_metadata": arm.metadata,
        "graph_profiling": "node",
        "capture_trigger": "cudaProfilerStart/Stop around one NVTX-named graph replay",
        "warmup": args.warmup,
        "status": "ready",
        "comparison_group_id": identity["comparison_group_id"],
        "rerun_id": identity["rerun_id"],
        "environment_lock_digest": identity["environment_lock_digest"],
        "protocol_lock_digest": identity["protocol_lock_digest"],
        "artifact_fingerprint_sha256": identity["per_arm_artifact_fingerprint_sha256"][
            args.arm
        ],
    }
    write_json(manifest_path, profile_manifest)
    print(f"PROFILE_READY nvtx={nvtx_name}", flush=True)
    cudart = torch.cuda.cudart()
    start_status = int(cudart.cudaProfilerStart())
    if start_status != 0:
        raise RuntimeError(f"cudaProfilerStart failed with status {start_status}")
    torch.cuda.nvtx.range_push(nvtx_name)
    try:
        elapsed_ms = arm.replay_ms()
    finally:
        torch.cuda.nvtx.range_pop()
        stop_status = int(cudart.cudaProfilerStop())
        if stop_status != 0:
            raise RuntimeError(f"cudaProfilerStop failed with status {stop_status}")
    profile_manifest.update(
        {
            "status": "complete",
            "event_elapsed_us": elapsed_ms * 1000.0,
            "output_sha256": tensor_sha256(arm.output),
            "jit_artifacts": jit_manifest(),
        }
    )
    write_json(manifest_path, profile_manifest)
    print(
        f"PROFILE_COMPLETE m={args.m} arm={args.arm} event_us={elapsed_ms * 1000.0:.3f}",
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-num-tokens", type=int, default=8192)
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument(
        "--m-values", nargs="+", type=int,
        default=[256, 512, 1024, 2048, 4096, 8192]
    )
    benchmark_parser.add_argument("--warmup", type=int, default=5)
    benchmark_parser.add_argument("--iters", type=int, default=50)
    benchmark_parser.add_argument("--repeats", type=int, default=5)
    benchmark_parser.add_argument("--l2-flush-bytes", type=int, default=192 << 20)

    profile_parser = subparsers.add_parser("single-replay")
    profile_parser.add_argument("--m", type=int, choices=[256, 8192], required=True)
    profile_parser.add_argument("--arm", choices=ARMS, required=True)
    profile_parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.flashinfer_root = args.flashinfer_root.resolve()
    if str(args.flashinfer_root) not in sys.path:
        sys.path.insert(0, str(args.flashinfer_root))
    source = validate_source(args.flashinfer_root)
    configure_source_checkout(args.flashinfer_root)
    runtime = runtime_manifest(
        repo_root=args.flashinfer_root,
        expected_gpu_uuid=args.expected_gpu_uuid,
        source=source,
    )
    torch.cuda.set_device(args.device_index)
    args.results.mkdir(parents=True, exist_ok=True)
    if args.command == "benchmark":
        return benchmark(args, runtime)
    if args.command == "single-replay":
        return single_replay(args, runtime)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
