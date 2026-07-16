#!/usr/bin/env python3
"""Prepare and replay the exp_003 M8192 instrumented fused-MoE target.

The module intentionally does not import FlashInfer at module import time.  A
candidate run first installs an import hook for the exact
``moe_dynamic_kernel`` module, then imports FlashInfer from the requested
source checkout.  This is the only supported marker-overlay mechanism.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = EXPERIMENT_ROOT / "results"
EXP002_FIXTURE = (
    EXPERIMENT_ROOT.parent / "exp_002_fused_vs_chain_dataflow" / "fixture.py"
)
TARGET_RELATIVE_PATH = Path(
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py"
)
TARGET_MODULE = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dynamic_kernel"
EXPECTED_FLASHINFER_COMMIT = "074d93e4aa54c75bee1b3dfdb39b7f075a3ff2af"
EXPECTED_CUTLASS_COMMIT = "b46b16d003484063bca4ed365e44095c4c6ed633"
EXPECTED_KERNEL_SHA256 = (
    "94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106"
)
EXPECTED_IMAGE_DIGEST = (
    "sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba"
)
EXPECTED_PYTHON_DEPS_SHA256 = (
    "32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74"
)
EXPECTED_PYTHON_DEPS_ROOT = Path("/workspace/deps")
M = 8192
E = 256
H = 2048
I = 512
TOPK = 8
TILE_M = 128
TILE_N = 128
SLICE_CHUNK = 1
EXPECTED_GRID = [1, 1, 110]
EXPECTED_BLOCK = [160, 1, 1]
PREDECLARED_CTA_Z = (0, 55, 109, 13, 27, 41, 69, 83, 96)
INSTRUMENTATION_SOURCE_MODEL = {
    "qmma_static_count_per_warp_slice": None,
    "fc2_blocks_per_slice": 16,
    "locked_geometry": {
        "tile_mnk": [128, 128, 128],
        "hidden_k_tiles": 16,
        "fc2_output_blocks": 16,
    },
    # The runner does not own tracker-cubin disassembly.  The analyzer must
    # replace these nulls with same-cubin PC/SASS closure before a causal
    # starvation verdict is legal.
    "pc_sass_verified_ranges": None,
    "instrumented_sass_qmma_count": None,
    "authority": "source model only; same-PID IKET tracker cubin is required",
}
OVERLAY_ENV_KEYS = (
    "EXP003_MARKER_OVERLAY",
    "W4A4_EXP003_MARKER_OVERLAY",
    "FLASHINFER_CUTEDSL_IKET_OVERLAY",
)
FALSE_ENV_VALUES = {"", "0", "false", "off", "no", "none"}
ARTIFACT_SUFFIXES = {
    ".so",
    ".cu",
    ".cuh",
    ".cpp",
    ".ptx",
    ".cubin",
    ".sass",
    ".json",
    ".mlir",
}
IKET_MAX_USER_EVENT_IDS = 30
IKET_RANGE_APIS = {"range_push", "range_start"}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(f"{file_sha256(path)}  ./{relative}\n".encode())
    return digest.hexdigest(), len(files)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def command_output(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            list(command),
            cwd=str(cwd) if cwd else None,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"ERROR: {error}"


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


@dataclass
class OwnedManifest:
    path: Path
    value: dict[str, Any]

    @classmethod
    def claim(cls, path: Path, initial: dict[str, Any]) -> "OwnedManifest":
        write_json_exclusive(path, initial)
        return cls(path, initial)

    def update(self, **values: Any) -> None:
        self.value.update(values)
        write_json_atomic(self.path, self.value)


class ExactModuleOverlayFinder(importlib.abc.MetaPathFinder):
    """Map exactly one module name to one audited overlay file."""

    def __init__(self, fullname: str, overlay: Path):
        self.fullname = fullname
        self.overlay = overlay

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: object | None = None,
    ):
        del path, target
        if fullname != self.fullname:
            return None
        return importlib.util.spec_from_file_location(fullname, self.overlay)


def resolve_overlay_from_env(
    environment: dict[str, str] | None = None,
) -> tuple[str | None, Path | None]:
    environment = os.environ if environment is None else environment
    selected: list[tuple[str, Path]] = []
    for key in OVERLAY_ENV_KEYS:
        raw = environment.get(key, "").strip()
        if raw.lower() in FALSE_ENV_VALUES:
            continue
        if raw.lower() in {"1", "true", "on", "yes"}:
            raise RuntimeError(f"{key} must be an overlay file path, not {raw!r}")
        selected.append((key, Path(raw).expanduser().resolve()))
    if not selected:
        return None, None
    unique_paths = {path for _, path in selected}
    if len(unique_paths) != 1:
        raise RuntimeError(f"conflicting marker overlay environment: {selected}")
    key, path = selected[0]
    if not path.is_file():
        raise RuntimeError(f"marker overlay does not exist: {path}")
    return key, path


def validate_overlay_event_id_budget(overlay: Path) -> dict[str, Any]:
    """Fail before JIT when an overlay exceeds IKET's user-event ID budget.

    NativeDump reserves event IDs 0 and 31, so one instrumented kernel may use
    at most 30 distinct named events.  Repeated sites may share a name, but all
    names must be compile-time string literals so the budget is auditable.
    """
    tree = ast.parse(overlay.read_text(encoding="utf-8"), filename=str(overlay))
    sites: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (
            not isinstance(function, ast.Attribute)
            or function.attr not in IKET_RANGE_APIS
        ):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            raise RuntimeError(
                f"IKET range name must be a string literal at {overlay}:{node.lineno}"
            )
        name = node.args[0].value
        if not isinstance(name, str) or not name:
            raise RuntimeError(
                f"IKET range name must be a nonempty string at {overlay}:{node.lineno}"
            )
        sites.append({"api": function.attr, "name": name, "line": node.lineno})
    names = sorted({site["name"] for site in sites})
    if len(names) > IKET_MAX_USER_EVENT_IDS:
        raise RuntimeError(
            "IKET overlay exceeds NativeDump's user-event ID budget: "
            f"{len(names)} > {IKET_MAX_USER_EVENT_IDS}; names={names}"
        )
    return {
        "unique_named_event_count": len(names),
        "max_user_event_ids": IKET_MAX_USER_EVENT_IDS,
        "unique_named_events": names,
        "range_site_count": len(sites),
        "range_sites": sites,
    }


def install_overlay_before_flashinfer_import(overlay: Path) -> None:
    if any(
        name == "flashinfer" or name.startswith("flashinfer.") for name in sys.modules
    ):
        raise RuntimeError("FlashInfer was imported before marker overlay installation")
    sys.meta_path.insert(0, ExactModuleOverlayFinder(TARGET_MODULE, overlay))


def git_output(repo: Path, *args: str) -> str:
    return command_output(["git", "-c", f"safe.directory={repo}", *args], cwd=repo)


def validate_source_checkout(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    kernel = repo_root / TARGET_RELATIVE_PATH
    cutlass_root = repo_root / "3rdparty" / "cutlass"
    if not kernel.is_file():
        raise RuntimeError(f"production kernel source is missing: {kernel}")
    commit = git_output(repo_root, "rev-parse", "HEAD")
    cutlass_commit = git_output(cutlass_root, "rev-parse", "HEAD")
    if commit != EXPECTED_FLASHINFER_COMMIT:
        raise RuntimeError(
            f"FlashInfer commit drift: {commit} != {EXPECTED_FLASHINFER_COMMIT}"
        )
    if cutlass_commit != EXPECTED_CUTLASS_COMMIT:
        raise RuntimeError(
            f"CUTLASS commit drift: {cutlass_commit} != {EXPECTED_CUTLASS_COMMIT}"
        )
    kernel_hash = file_sha256(kernel)
    if kernel_hash != EXPECTED_KERNEL_SHA256:
        raise RuntimeError(
            f"production kernel hash drift: {kernel_hash} != {EXPECTED_KERNEL_SHA256}"
        )
    unstaged = git_output(repo_root, "diff", "--name-only")
    staged = git_output(repo_root, "diff", "--cached", "--name-only")
    if unstaged or staged:
        raise RuntimeError(
            "exact source checkout has tracked changes:\n"
            f"unstaged:\n{unstaged}\nstaged:\n{staged}"
        )
    identity_files = [
        kernel,
        repo_root / "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py",
        repo_root / "flashinfer/fused_moe/cute_dsl/b12x_moe.py",
        EXP002_FIXTURE,
        Path(__file__).resolve(),
        EXPERIMENT_ROOT / "analyze_exp003.py",
        EXPERIMENT_ROOT / "validate_binary_gate.py",
        EXPERIMENT_ROOT / "validate_pc_sass.py",
        EXPERIMENT_ROOT / "plan.md",
    ]
    return {
        "repo_root": str(repo_root),
        "flashinfer_commit": commit,
        "cutlass_commit": cutlass_commit,
        "production_kernel": {
            "path": str(kernel),
            "sha256": kernel_hash,
        },
        "tracked_status": {"unstaged": unstaged, "staged": staged},
        "identity_files": [
            {"path": str(path), "sha256": file_sha256(path)} for path in identity_files
        ],
    }


def configure_source_checkout(repo_root: Path) -> dict[str, Any]:
    flashinfer = importlib.import_module("flashinfer")
    imported_root = Path(flashinfer.__file__).resolve().parents[1]
    if imported_root != repo_root:
        raise RuntimeError(
            f"imported FlashInfer root {imported_root} != requested {repo_root}"
        )
    from flashinfer.jit import env as jit_env

    csrc = repo_root / "csrc"
    jit_env.FLASHINFER_CSRC_DIR = csrc
    jit_env.FLASHINFER_INCLUDE_DIR = repo_root / "include"
    jit_env.FLASHINFER_AOT_DIR = (
        jit_env.FLASHINFER_WORKSPACE_DIR / "aot_disabled_for_exp003"
    )
    jit_env.CUTLASS_INCLUDE_DIRS = [
        repo_root / "3rdparty/cutlass/include",
        repo_root / "3rdparty/cutlass/tools/util/include",
    ]
    jit_env.CCCL_INCLUDE_DIRS = [
        repo_root / "3rdparty/cccl/cub",
        repo_root / "3rdparty/cccl/libcudacxx/include",
        repo_root / "3rdparty/cccl/thrust",
    ]
    jit_env.SPDLOG_INCLUDE_DIR = repo_root / "3rdparty/spdlog/include"
    target_module = importlib.import_module(TARGET_MODULE)
    cutlass = importlib.import_module("cutlass")
    tvm_ffi = importlib.import_module("tvm_ffi")
    return {
        "flashinfer_module": str(Path(flashinfer.__file__).resolve()),
        "cutlass_module": str(Path(cutlass.__file__).resolve()),
        "tvm_ffi_module": str(Path(tvm_ffi.__file__).resolve()),
        "target_module": TARGET_MODULE,
        "target_module_origin": str(Path(target_module.__file__).resolve()),
    }


def load_fixture_module():
    spec = importlib.util.spec_from_file_location(
        "exp003_exp002_fixture", EXP002_FIXTURE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixture module from {EXP002_FIXTURE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def jit_artifact_manifest(workspace: Path) -> list[dict[str, Any]]:
    if not workspace.exists():
        return []
    artifacts: list[dict[str, Any]] = []
    for path in sorted(workspace.rglob("*")):
        if path.is_file() and path.suffix in ARTIFACT_SUFFIXES:
            artifacts.append(
                {
                    "path": path.relative_to(workspace).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return artifacts


def require_fresh_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    entries = list(workspace.iterdir())
    if entries:
        raise RuntimeError(
            f"preparation requires an empty dedicated JIT workspace: {workspace}"
        )


def query_gpu_identity(expected_uuid: str, device_index: int) -> dict[str, Any]:
    query = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,driver_version",
            "--format=csv,noheader",
        ]
    )
    rows = [row.strip() for row in query.splitlines() if row.strip()]
    matching = [row for row in rows if expected_uuid in row]
    if len(matching) != 1:
        raise RuntimeError(
            f"expected exactly one GPU row for {expected_uuid}; got {matching or rows}"
        )
    torch.cuda.set_device(device_index)
    actual_uuid = (
        command_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"])
        .splitlines()[0]
        .strip()
    )
    if actual_uuid != expected_uuid:
        raise RuntimeError(f"visible GPU UUID drift: {actual_uuid} != {expected_uuid}")
    capability = list(torch.cuda.get_device_capability(device_index))
    if capability not in ([12, 0], [12, 1]):
        raise RuntimeError(f"exp003 requires SM120/SM121, got {capability}")
    return {
        "expected_uuid": expected_uuid,
        "visible_uuid": actual_uuid,
        "device_index": device_index,
        "nvidia_smi_row": matching[0],
        "name": torch.cuda.get_device_name(device_index),
        "compute_capability": capability,
    }


def dependency_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def iket_distribution_identity(run_iket_value: str) -> dict[str, Any]:
    """Resolve IKET identity without executing ``run-iket``.

    IKET 0.7.10 does not expose a version flag.  Prefer the active Python
    distribution metadata, then inspect the executable's adjacent dist-info
    metadata (the same non-executing method used by KDK's safe wrapper).
    """
    version = dependency_version("iket")
    method = "importlib.metadata"
    resolved_text = shutil.which(run_iket_value) or run_iket_value
    run_iket = Path(resolved_text).expanduser().resolve()
    if not run_iket.is_file():
        raise RuntimeError(f"run-iket executable is missing: {run_iket}")
    if version == "NOT_INSTALLED":
        method = "adjacent dist-info METADATA"
        version = "unknown"
        environment_root = run_iket.parent.parent
        patterns = (
            "lib/python*/site-packages/iket-*.dist-info/METADATA",
            "lib64/python*/site-packages/iket-*.dist-info/METADATA",
        )
        for pattern in patterns:
            for metadata_path in sorted(environment_root.glob(pattern)):
                fields: dict[str, str] = {}
                for line in metadata_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines():
                    if line.startswith("Name:"):
                        fields["name"] = line.partition(":")[2].strip()
                    elif line.startswith("Version:"):
                        fields["version"] = line.partition(":")[2].strip()
                    if fields.get("name") and fields.get("version"):
                        break
                if fields.get("name", "").lower() == "iket":
                    version = fields["version"]
                    break
            if version != "unknown":
                break
    if version != "0.7.10":
        raise RuntimeError(f"IKET version drift: {version} != 0.7.10")
    return {
        "distribution_version": version,
        "version_method": method,
        "run_iket": str(run_iket),
        "run_iket_sha256": file_sha256(run_iket),
    }


def runtime_identity(
    *,
    source: dict[str, Any],
    import_identity: dict[str, Any],
    overlay_key: str | None,
    overlay: Path | None,
    overlay_event_id_budget: dict[str, Any] | None,
    expected_gpu_uuid: str,
    device_index: int,
) -> dict[str, Any]:
    workspace_raw = os.environ.get("FLASHINFER_WORKSPACE_BASE", "")
    if not workspace_raw:
        raise RuntimeError("FLASHINFER_WORKSPACE_BASE must name an exp003 JIT root")
    if os.environ.get("CUTE_DSL_COMPILER_OPT") != "iket":
        raise RuntimeError("CUTE_DSL_COMPILER_OPT must equal 'iket'")
    if os.environ.get("CUTE_DSL_KEEP") != "ir,ptx,cubin,sass":
        raise RuntimeError(
            "CUTE_DSL_KEEP must equal 'ir,ptx,cubin,sass' for binary closure"
        )
    dump_dir_raw = os.environ.get("CUTE_DSL_DUMP_DIR", "")
    if not dump_dir_raw:
        raise RuntimeError("CUTE_DSL_DUMP_DIR must name the exp003 artifact root")
    workspace = Path(workspace_raw).resolve()
    dump_dir = Path(dump_dir_raw).resolve()
    try:
        dump_dir.relative_to(workspace)
    except ValueError as error:
        raise RuntimeError(
            "CUTE_DSL_DUMP_DIR must be inside the exp003 JIT root"
        ) from error
    if os.environ.get("W4A4_IMAGE_DIGEST") != EXPECTED_IMAGE_DIGEST:
        raise RuntimeError("container image digest drift or missing W4A4_IMAGE_DIGEST")
    if os.environ.get("W4A4_PYTHON_DEPS_SHA256") != EXPECTED_PYTHON_DEPS_SHA256:
        raise RuntimeError("Python dependency overlay identity drift")
    if not os.environ.get("KDK_LEASE_ID"):
        raise RuntimeError("KDK_LEASE_ID must identify the active GPU lease")
    if EXPECTED_PYTHON_DEPS_ROOT.is_dir():
        actual_deps_hash, deps_files = tree_sha256(EXPECTED_PYTHON_DEPS_ROOT)
        if actual_deps_hash != EXPECTED_PYTHON_DEPS_SHA256:
            raise RuntimeError(f"dependency overlay content drift: {actual_deps_hash}")
    else:
        actual_deps_hash, deps_files = "MISSING", 0
    provider_root_raw = os.environ.get("EXP003_IKET_PROVIDER_ROOT", "")
    if not provider_root_raw:
        raise RuntimeError(
            "EXP003_IKET_PROVIDER_ROOT must name the audited read-only provider"
        )
    run_iket_value = os.environ.get("EXP003_RUN_IKET", "run-iket")
    provider: dict[str, Any] = {
        "root": provider_root_raw,
        **iket_distribution_identity(run_iket_value),
    }
    provider_root = Path(provider_root_raw).resolve()
    if not provider_root.is_dir():
        raise RuntimeError(f"IKET provider root is missing: {provider_root}")
    iket_spec = importlib.util.find_spec("iket")
    if iket_spec is None or iket_spec.origin is None:
        raise RuntimeError("IKET Python module is not importable")
    iket_origin = Path(iket_spec.origin).resolve()
    try:
        iket_origin.relative_to(provider_root)
    except ValueError as error:
        raise RuntimeError(
            f"IKET import resolves outside the audited provider: {iket_origin}"
        ) from error
    try:
        Path(provider["run_iket"]).relative_to(provider_root)
    except ValueError as error:
        raise RuntimeError(
            "run-iket executable is outside the audited provider root"
        ) from error
    provider_hash, provider_files = tree_sha256(provider_root)
    provider.update(
        {
            "tree_sha256": provider_hash,
            "file_count": provider_files,
            "import_origin": str(iket_origin),
        }
    )
    return {
        "timestamp_unix": time.time(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nvcc": command_output(["nvcc", "--version"]),
        "ptxas": command_output(["ptxas", "--version"]),
        "gpu": query_gpu_identity(expected_gpu_uuid, device_index),
        "container": {
            "image": "nvcr.io/nvidia/pytorch:26.05-py3",
            "digest": EXPECTED_IMAGE_DIGEST,
        },
        "dependencies": {
            "expected_tree_sha256": EXPECTED_PYTHON_DEPS_SHA256,
            "actual_tree_sha256": actual_deps_hash,
            "file_count": deps_files,
            "nvidia_cutlass_dsl": dependency_version("nvidia-cutlass-dsl"),
            "apache_tvm_ffi": dependency_version("apache-tvm-ffi"),
        },
        "iket_provider": provider,
        "source": source,
        "imports": import_identity,
        "overlay": (
            None
            if overlay is None
            else {
                "environment_key": overlay_key,
                "path": str(overlay),
                "sha256": file_sha256(overlay),
                "event_id_budget": overlay_event_id_budget,
            }
        ),
        "lease_id": os.environ["KDK_LEASE_ID"],
        "jit_workspace": str(Path(workspace_raw).resolve()),
        "environment": {
            key: os.environ.get(key, "")
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "FLASHINFER_WORKSPACE_BASE",
                "CUTE_DSL_CACHE_DIR",
                "CUTE_DSL_COMPILER_OPT",
                "CUTE_DSL_DUMP_DIR",
                "CUTE_DSL_KEEP",
                "EXP003_MARKER_OVERLAY",
                "W4A4_EXP003_MARKER_OVERLAY",
                "FLASHINFER_CUTEDSL_IKET_OVERLAY",
                "EXP003_IKET_PROVIDER_ROOT",
                "EXP003_RUN_IKET",
                "W4A4_IMAGE_DIGEST",
                "W4A4_PYTHON_DEPS_SHA256",
                "KDK_LEASE_ID",
                "PYTHONPATH",
            )
        },
    }


@dataclass(frozen=True)
class TaskDescriptor:
    task_slot: int
    expert: int
    m_tile: int
    slice_begin: int
    slice_count: int
    valid_rows: int

    def as_dict(self, *, ready: int | None = None) -> dict[str, int]:
        value = {
            "task_slot": self.task_slot,
            "expert": self.expert,
            "m_tile": self.m_tile,
            "slice_begin": self.slice_begin,
            "slice_count": self.slice_count,
            "valid_rows": self.valid_rows,
        }
        if ready is not None:
            value["ready"] = ready
        return value


@dataclass(frozen=True)
class DynamicTaskModel:
    row_counts: list[int]
    expert_tile_base: list[int]
    tasks: list[TaskDescriptor]
    expected_task_head: int
    expected_task_tail: int
    grid_z: int
    ready_value: int


def build_dynamic_task_model(
    row_counts: Sequence[int],
    *,
    tile_m: int = TILE_M,
    gate_tiles: int = I // TILE_N,
    slice_chunk: int = SLICE_CHUNK,
    grid_z: int = EXPECTED_GRID[2],
) -> DynamicTaskModel:
    if tile_m <= 0 or gate_tiles <= 0 or slice_chunk <= 0 or grid_z <= 0:
        raise ValueError("task geometry values must be positive")
    counts = [int(value) for value in row_counts]
    if any(value < 0 for value in counts):
        raise ValueError("row counts must be nonnegative")
    expert_tile_base = [0]
    for count in counts:
        expert_tile_base.append(expert_tile_base[-1] + (count + tile_m - 1) // tile_m)
    slice_groups = (gate_tiles + slice_chunk - 1) // slice_chunk
    tasks: list[TaskDescriptor] = []
    for expert, count in enumerate(counts):
        for local_tile in range((count + tile_m - 1) // tile_m):
            m_tile = expert_tile_base[expert] + local_tile
            valid_rows = min(tile_m, count - local_tile * tile_m)
            for group in range(slice_groups):
                slice_begin = group * slice_chunk
                slice_count = min(slice_chunk, gate_tiles - slice_begin)
                task_slot = m_tile * slice_groups + group
                tasks.append(
                    TaskDescriptor(
                        task_slot=task_slot,
                        expert=expert,
                        m_tile=m_tile,
                        slice_begin=slice_begin,
                        slice_count=slice_count,
                        valid_rows=valid_rows,
                    )
                )
    expected_tail = expert_tile_base[-1] * slice_groups
    if len(tasks) != expected_tail:
        raise AssertionError("host task model is internally inconsistent")
    return DynamicTaskModel(
        row_counts=counts,
        expert_tile_base=expert_tile_base,
        tasks=tasks,
        # Deferred queue consumers use atomicAdd and every CTA makes one
        # terminal claim after the last valid slot.
        expected_task_head=expected_tail + grid_z,
        expected_task_tail=expected_tail,
        grid_z=grid_z,
        # full_tile_publish_enabled is hardcoded to zero in the locked source;
        # task_ready is unused and remains zero.
        ready_value=0,
    )


def _cpu_list(tensor: torch.Tensor) -> list[Any]:
    return tensor.detach().cpu().tolist()


def workspace_snapshot(
    workspace: Any,
    fixture: Any,
    *,
    grid_z: int = EXPECTED_GRID[2],
) -> dict[str, Any]:
    row_counts_tensor = workspace.row_counts.detach().cpu()
    row_counts = [int(value) for value in row_counts_tensor.tolist()]
    model = build_dynamic_task_model(row_counts, grid_z=grid_z)
    tail = int(workspace.task_tail.item())
    head = int(workspace.task_head.item())
    descriptor_tensors = {
        "expert": workspace.task_expert[:tail].detach().cpu(),
        "m_tile": workspace.task_m_tile[:tail].detach().cpu(),
        "slice_begin": workspace.task_slice_begin[:tail].detach().cpu(),
        "slice_count": workspace.task_slice_count[:tail].detach().cpu(),
        "valid_rows": workspace.task_valid_rows[:tail].detach().cpu(),
        "ready": workspace.task_ready[:tail].detach().cpu(),
    }
    actual_tasks = [
        {
            "task_slot": slot,
            **{
                key: int(tensor[slot].item())
                for key, tensor in descriptor_tensors.items()
            },
        }
        for slot in range(tail)
    ]
    expected_tasks = [task.as_dict(ready=model.ready_value) for task in model.tasks]
    expected_counts = torch.bincount(
        fixture.topk_ids.detach().cpu().flatten().long(), minlength=E
    ).to(torch.int32)
    expected_write_rows = expected_counts
    actual_write_rows = workspace.expert_write_rows.detach().cpu()
    actual_tile_base = workspace.expert_tile_base.detach().cpu()
    expected_tile_base = torch.tensor(model.expert_tile_base, dtype=torch.int32)

    physical_rows = model.expert_tile_base[-1] * TILE_M
    token_map = workspace.token_map[:physical_rows].detach().cpu()
    token_weights = workspace.token_weights[:physical_rows].detach().cpu()
    topk_ids = fixture.topk_ids.detach().cpu().long()
    topk_weights = fixture.topk_weights.detach().cpu().float()
    valid_token_ids: list[int] = []
    valid_weights: list[float] = []
    routing_membership = True
    routing_weight_match = True
    duplicate_pairs = False
    seen_pairs: set[tuple[int, int]] = set()
    token_weight_sums = torch.zeros(M, dtype=torch.float64)
    token_occurrences = torch.zeros(M, dtype=torch.int32)
    for expert, count in enumerate(row_counts):
        base_row = model.expert_tile_base[expert] * TILE_M
        for offset in range(count):
            token = int(token_map[base_row + offset].item())
            weight = float(token_weights[base_row + offset].item())
            valid_token_ids.append(token)
            valid_weights.append(weight)
            if token < 0 or token >= M:
                routing_membership = False
                continue
            matches = torch.nonzero(topk_ids[token] == expert, as_tuple=False).flatten()
            if matches.numel() != 1:
                routing_membership = False
            else:
                expected_weight = float(topk_weights[token, int(matches.item())].item())
                if abs(weight - expected_weight) > 1e-6:
                    routing_weight_match = False
            pair = (token, expert)
            if pair in seen_pairs:
                duplicate_pairs = True
            seen_pairs.add(pair)
            token_weight_sums[token] += weight
            token_occurrences[token] += 1

    task_mismatches = [
        {"slot": index, "actual": actual, "expected": expected}
        for index, (actual, expected) in enumerate(
            zip(actual_tasks, expected_tasks, strict=False)
        )
        if actual != expected
    ]
    if len(actual_tasks) != len(expected_tasks):
        task_mismatches.append(
            {
                "length": {
                    "actual": len(actual_tasks),
                    "expected": len(expected_tasks),
                }
            }
        )
    checks = {
        "row_counts_match_fixture": bool(
            torch.equal(row_counts_tensor, expected_counts)
        ),
        "row_counts_sum": int(row_counts_tensor.sum().item()) == M * TOPK,
        "expert_write_rows_match": bool(
            torch.equal(actual_write_rows, expected_write_rows)
        ),
        "expert_tile_base_match": bool(
            torch.equal(actual_tile_base, expected_tile_base)
        ),
        "task_tail_match": tail == model.expected_task_tail,
        "task_head_terminal_claim_model_match": head == model.expected_task_head,
        "task_descriptors_match": not task_mismatches,
        "task_ready_deferred_all_zero": bool(
            torch.all(descriptor_tensors["ready"] == model.ready_value).item()
        ),
        "all_work_published": int(workspace.all_work_published.item()) == 1,
        "valid_token_count": len(valid_token_ids) == M * TOPK,
        "token_id_range": all(0 <= value < M for value in valid_token_ids),
        "token_occurrences_topk": bool(torch.all(token_occurrences == TOPK).item()),
        "routing_membership": routing_membership,
        "routing_pairs_unique": not duplicate_pairs and len(seen_pairs) == M * TOPK,
        "routing_weight_match": routing_weight_match,
        "routing_weight_finite_nonnegative": bool(
            torch.isfinite(torch.tensor(valid_weights)).all().item()
            and all(value >= 0.0 for value in valid_weights)
        ),
        "routing_weight_sum": bool(
            torch.allclose(
                token_weight_sums,
                torch.ones(M, dtype=torch.float64),
                rtol=1e-6,
                atol=1e-6,
            )
        ),
    }
    tensor_hashes = {
        "row_counts": tensor_sha256(row_counts_tensor),
        "expert_write_rows": tensor_sha256(actual_write_rows),
        "expert_tile_base": tensor_sha256(actual_tile_base),
        "task_expert": tensor_sha256(descriptor_tensors["expert"]),
        "task_m_tile": tensor_sha256(descriptor_tensors["m_tile"]),
        "task_slice_begin": tensor_sha256(descriptor_tensors["slice_begin"]),
        "task_slice_count": tensor_sha256(descriptor_tensors["slice_count"]),
        "task_valid_rows": tensor_sha256(descriptor_tensors["valid_rows"]),
        "task_ready": tensor_sha256(descriptor_tensors["ready"]),
        "token_map_physical_rows": tensor_sha256(token_map),
        "token_weights_physical_rows": tensor_sha256(token_weights),
    }
    return {
        "policy": {
            "full_tile_publish_enabled": 0,
            "task_ready": "unused deferred queue; expected all zero",
            "terminal_claim": "one atomicAdd overshoot per CTA",
        },
        "grid_z": grid_z,
        "expected_task_count": model.expected_task_tail,
        "task_head": head,
        "task_tail": tail,
        "expected_task_head": model.expected_task_head,
        "expected_task_tail": model.expected_task_tail,
        "row_counts": row_counts,
        "expert_tile_base": _cpu_list(actual_tile_base),
        "row_counts_sum": int(row_counts_tensor.sum().item()),
        "physical_tiles": model.expert_tile_base[-1],
        "physical_rows": physical_rows,
        "task_table": actual_tasks,
        "task_table_sha256": canonical_sha256(actual_tasks),
        "task_mismatches": task_mismatches[:32],
        "routing": {
            "valid_rows": len(valid_token_ids),
            "unique_token_expert_pairs": len(seen_pairs),
            "token_weight_sum_max_abs_error": float(
                (token_weight_sums - 1.0).abs().max().item()
            ),
        },
        "hashes": tensor_hashes,
        "checks": checks,
        "task_model_pass": all(checks.values()),
    }


def correctness_diagnostics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    fixture_module: Any,
) -> dict[str, Any]:
    diagnostics = dict(fixture_module.output_diagnostics(actual, reference))
    actual_f32 = actual.float()
    reference_f32 = reference.float()
    per_token_relative_l2 = torch.linalg.vector_norm(
        actual_f32 - reference_f32, dim=1
    ) / torch.linalg.vector_norm(reference_f32, dim=1).clamp_min(1e-12)
    per_token_finite = torch.isfinite(actual).all(dim=1)
    per_token_nonzero = torch.count_nonzero(actual, dim=1) > 0
    diagnostics.update(
        {
            "per_token_relative_l2_p99": float(
                torch.quantile(per_token_relative_l2, 0.99).item()
            ),
            "per_token_relative_l2_max": float(per_token_relative_l2.max().item()),
            "finite_token_count": int(per_token_finite.sum().item()),
            "nonzero_token_count": int(per_token_nonzero.sum().item()),
            "all_tokens_finite": bool(per_token_finite.all().item()),
            "all_tokens_nonzero": bool(per_token_nonzero.all().item()),
        }
    )
    thresholds = {
        "formal_pass": bool(diagnostics["formal_pass"]),
        "cosine_ge_0_999": float(diagnostics["cosine"]) >= 0.999,
        "relative_l2_le_0_02": float(diagnostics["relative_l2"]) <= 0.02,
        "max_abs_error_le_0_08": float(diagnostics["max_abs_error"]) <= 0.08,
        "all_tokens_finite": bool(diagnostics["all_tokens_finite"]),
        "all_tokens_nonzero": bool(diagnostics["all_tokens_nonzero"]),
    }
    diagnostics["thresholds"] = thresholds
    diagnostics["gate_pass"] = all(thresholds.values())
    return diagnostics


def marker_correctness_vs_control(
    candidate: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    """Detect marker-induced numerical drift relative to the no-marker control.

    The per-token relative-L2 p99 is retained, but not given an arbitrary
    absolute threshold: the exact no-marker IKET-compiler control measured
    0.060367 because a small tail of reference tokens has low L2 norm.  The
    candidate therefore has to remain within a predeclared small envelope of
    that same-case control while still passing the independent oracle gates.
    """

    checks = {
        "cosine_not_below_control_minus_1e_5": float(candidate["cosine"])
        >= float(control["cosine"]) - 1e-5,
        "relative_l2_not_above_control_plus_0_002": float(candidate["relative_l2"])
        <= float(control["relative_l2"]) + 0.002,
        "max_abs_error_not_above_control_plus_0_005": float(candidate["max_abs_error"])
        <= float(control["max_abs_error"]) + 0.005,
        "per_token_relative_l2_p99_not_above_control_plus_0_005": float(
            candidate["per_token_relative_l2_p99"]
        )
        <= float(control["per_token_relative_l2_p99"]) + 0.005,
    }
    return {
        "control": {
            key: control[key]
            for key in (
                "cosine",
                "relative_l2",
                "max_abs_error",
                "per_token_relative_l2_p99",
            )
        },
        "candidate": {
            key: candidate[key]
            for key in (
                "cosine",
                "relative_l2",
                "max_abs_error",
                "per_token_relative_l2_p99",
            )
        },
        "checks": checks,
        "gate_pass": all(checks.values()),
    }


def lightweight_output_gate(output: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(output).all(dim=1)
    nonzero = torch.count_nonzero(output, dim=1) > 0
    return {
        "shape": list(output.shape),
        "dtype": str(output.dtype).replace("torch.", ""),
        "all_tokens_finite": bool(finite.all().item()),
        "all_tokens_nonzero": bool(nonzero.all().item()),
        "output_sha256": tensor_sha256(output),
        "gate_pass": bool(finite.all().item() and nonzero.all().item()),
    }


@dataclass
class CapturedFusedArm:
    wrapper: Any
    launch: Callable[[], torch.Tensor]
    output: torch.Tensor | None = None
    graph: torch.cuda.CUDAGraph | None = None
    capture_stream: torch.cuda.Stream | None = None
    eager_launch_count: int = 0
    graph_capture_count: int = 0
    graph_replay_count: int = 0

    def eager_setup_once(self) -> torch.Tensor:
        if self.eager_launch_count:
            raise RuntimeError("eager setup may execute exactly once")
        self.output = self.launch()
        self.eager_launch_count += 1
        torch.cuda.synchronize()
        return self.output

    def capture_once(self) -> None:
        if self.eager_launch_count != 1 or self.graph_capture_count:
            raise RuntimeError("graph capture requires exactly one eager setup")
        self.capture_stream = torch.cuda.Stream()
        self.capture_stream.wait_stream(torch.cuda.current_stream())
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            self.output = self.launch()
        self.graph_capture_count += 1
        torch.cuda.current_stream().wait_stream(self.capture_stream)
        torch.cuda.synchronize()

    def replay_once(self) -> torch.Tensor:
        if self.graph is None or self.graph_capture_count != 1:
            raise RuntimeError("graph has not been captured exactly once")
        if self.graph_replay_count:
            raise RuntimeError("target graph may replay exactly once")
        self.graph.replay()
        self.graph_replay_count += 1
        torch.cuda.synchronize()
        assert self.output is not None
        return self.output

    def protocol(self) -> dict[str, Any]:
        return {
            "eager_setup_launches": self.eager_launch_count,
            "graph_captures": self.graph_capture_count,
            "warmup_replays": 0,
            "target_graph_replays": self.graph_replay_count,
        }


def build_fused_arm(
    fixture: Any, weights: Any, *, max_num_tokens: int
) -> CapturedFusedArm:
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

    def launch() -> torch.Tensor:
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

    return CapturedFusedArm(wrapper=wrapper, launch=launch)


def validate_dynamic_wrapper(arm: CapturedFusedArm) -> Any:
    workspace = getattr(arm.wrapper, "_dynamic_workspace", None)
    if workspace is None:
        raise RuntimeError("M8192 did not allocate the dynamic MoE workspace")
    if workspace.__class__.__name__ != "Sm120DynamicMoEWorkspace":
        raise RuntimeError(
            f"unexpected workspace class: {workspace.__class__.__name__}"
        )
    return workspace


def fixed_fixture_and_weights(args: argparse.Namespace, fixture_module: Any):
    if args.m != M or args.max_num_tokens != M:
        raise RuntimeError("exp003 is locked to M=max_num_tokens=8192")
    device = torch.device("cuda", args.device_index)
    fixture = fixture_module.make_routed_fixture(M, device=device, seed=args.seed)
    weights = fixture_module.make_canonical_weights(device=device, seed=args.seed)
    if fixture.manifest.get("seed") != args.seed + M:
        raise RuntimeError("fixture seed drift")
    return fixture, weights


def preparation_manifest_path(args: argparse.Namespace) -> Path:
    directory = (
        args.manifest_dir.resolve()
        if args.manifest_dir
        else (args.results.resolve() / "preparation" / args.command)
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "manifest.json"


def target_manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest_dir is None:
        raise RuntimeError("target requires an explicit fresh --manifest-dir")
    root = args.manifest_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    pid_dir = root / f"pid_{os.getpid()}"
    pid_dir.mkdir()
    return pid_dir / "target_manifest.json"


def artifact_identity(workspace: Path) -> dict[str, Any]:
    artifacts = jit_artifact_manifest(workspace)
    return {
        "workspace": str(workspace),
        "jit_artifacts": artifacts,
        "jit_artifact_set_sha256": canonical_sha256(artifacts),
    }


def load_control_preparation(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"control preparation manifest is missing: {path}")
    value = json.loads(path.read_text())
    if value.get("command") != "prepare-control":
        raise RuntimeError("control prerequisite has the wrong command identity")
    if value.get("status") != "complete" or not value.get("gate_pass"):
        raise RuntimeError("control prerequisite did not pass")
    return value


def prepare(
    args: argparse.Namespace, runtime: dict[str, Any], fixture_module: Any
) -> int:
    manifest_path = preparation_manifest_path(args)
    manifest = OwnedManifest.claim(
        manifest_path,
        {
            "schema_version": 1,
            "status": "running",
            "command": args.command,
            "pid": os.getpid(),
            "runtime": runtime,
        },
    )
    try:
        fixture, weights = fixed_fixture_and_weights(args, fixture_module)
        # The independent oracle is deliberately evaluated before the target
        # arm.  The last fused workload execution is therefore one graph replay.
        reference = fixture_module.reference_moe_nvfp4(fixture, weights)
        torch.cuda.synchronize()
        arm = build_fused_arm(fixture, weights, max_num_tokens=args.max_num_tokens)
        arm.eager_setup_once()
        arm.capture_once()
        output = arm.replay_once()
        workspace = validate_dynamic_wrapper(arm)
        correctness = correctness_diagnostics(output, reference, fixture_module)
        workspace_evidence = workspace_snapshot(workspace, fixture)
        artifacts = artifact_identity(Path(runtime["jit_workspace"]))
        marker_vs_control = None
        marker_gate_pass = True
        if args.command == "prepare-candidate":
            control = load_control_preparation(args.control_manifest.resolve())
            if control.get("fixture") != fixture.manifest:
                raise RuntimeError("candidate fixture drift from control prerequisite")
            if control.get("weights") != weights.manifest:
                raise RuntimeError("candidate weight drift from control prerequisite")
            marker_vs_control = marker_correctness_vs_control(
                correctness, control["correctness"]
            )
            marker_gate_pass = bool(marker_vs_control["gate_pass"])
        passed = bool(
            correctness["gate_pass"]
            and workspace_evidence["task_model_pass"]
            and marker_gate_pass
        )
        manifest.update(
            status="complete" if passed else "failed_gate",
            case={
                "m": M,
                "experts": E,
                "hidden": H,
                "intermediate_tp": I,
                "topk": TOPK,
                "seed": args.seed,
            },
            fixture=fixture.manifest,
            weights=weights.manifest,
            correctness=correctness,
            marker_vs_control_correctness=marker_vs_control,
            workspace=workspace_evidence,
            graph={
                "kernel_name_pattern": "MoEDynamicKernel",
                "expected_grid": EXPECTED_GRID,
                "expected_block": EXPECTED_BLOCK,
                "protocol": arm.protocol(),
            },
            instrumentation_evidence=INSTRUMENTATION_SOURCE_MODEL,
            identity=artifacts,
            gate_pass=passed,
        )
        if not passed:
            return 2
        print(
            f"EXP003_PREPARE_COMPLETE command={args.command} manifest={manifest_path}"
        )
        return 0
    except BaseException as error:
        manifest.update(
            status="error",
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise


def stable_prerequisite_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    """Drop only execution-instance fields from a preparation/target runtime."""
    return {
        key: runtime.get(key)
        for key in (
            "python",
            "torch",
            "cuda_runtime",
            "nvcc",
            "ptxas",
            "gpu",
            "container",
            "dependencies",
            "iket_provider",
            "source",
            "imports",
            "overlay",
            "jit_workspace",
        )
    }


def load_candidate_prerequisite(path: Path, runtime: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"candidate prerequisite manifest is missing: {path}")
    value = json.loads(path.read_text())
    if value.get("status") != "complete" or not value.get("gate_pass"):
        raise RuntimeError("candidate prerequisite did not pass")
    previous_runtime = value.get("runtime", {})
    if not runtime.get("overlay"):
        raise RuntimeError("target marker overlay identity is missing")
    if stable_prerequisite_runtime(previous_runtime) != stable_prerequisite_runtime(
        runtime
    ):
        raise RuntimeError("target runtime/toolchain/source/overlay identity drift")
    fixture = value.get("fixture", {})
    if fixture.get("m") != M or fixture.get("seed") != 2026 + M:
        raise RuntimeError("candidate fixture identity drift")
    return value


def parse_selected_cta(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3 or any(not re.fullmatch(r"\d+", part) for part in parts):
        raise RuntimeError("EXP003_SELECTED_CTA must have form x,y,z")
    selected = [int(part) for part in parts]
    if selected[0:2] != [0, 0] or selected[2] not in PREDECLARED_CTA_Z:
        raise RuntimeError(
            "EXP003_SELECTED_CTA must be one of the predeclared coordinates: "
            + ", ".join(f"0,0,{z}" for z in PREDECLARED_CTA_Z)
        )
    if any(
        value >= bound for value, bound in zip(selected, EXPECTED_GRID, strict=True)
    ):
        raise RuntimeError(
            f"EXP003_SELECTED_CTA {selected} is outside grid {EXPECTED_GRID}"
        )
    return selected


def capture_identity_from_env() -> dict[str, Any]:
    required = {
        "capture_id": os.environ.get("EXP003_CAPTURE_ID", ""),
        "cluster_id": os.environ.get("EXP003_CLUSTER_ID", ""),
        "selected_cta_raw": os.environ.get("EXP003_SELECTED_CTA", ""),
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"target capture identity is missing: {missing}")
    return {
        "capture_id": required["capture_id"],
        "cluster_id": required["cluster_id"],
        "selected_cta": parse_selected_cta(required["selected_cta_raw"]),
        "iket_pass": os.environ.get("EXP003_IKET_PASS", "unknown"),
    }


def target(
    args: argparse.Namespace, runtime: dict[str, Any], fixture_module: Any
) -> int:
    capture = capture_identity_from_env()
    manifest_path = target_manifest_path(args)
    manifest = OwnedManifest.claim(
        manifest_path,
        {
            "schema_version": 1,
            "status": "running",
            "pid": os.getpid(),
            "capture": capture,
            # These values are capture-output properties.  The runner must not
            # guess them; the analyzer joins the same-PID decoded trace.
            "trace_overflow": None,
            "graph": {
                "kernel_name_pattern": "MoEDynamicKernel",
                "expected_grid": EXPECTED_GRID,
                "expected_block": EXPECTED_BLOCK,
                "target_replay_ordinal": 1,
                "context_id": None,
                "graph_launch_key": None,
                "grid_id": None,
                "identity_authority": "same-PID decoded IKET trace",
            },
            "runtime": runtime,
            "instrumentation_evidence": INSTRUMENTATION_SOURCE_MODEL,
        },
    )
    try:
        prerequisite = load_candidate_prerequisite(
            args.candidate_manifest.resolve(), runtime
        )
        workspace_path = Path(runtime["jit_workspace"])
        before = artifact_identity(workspace_path)
        if (
            before["jit_artifact_set_sha256"]
            != prerequisite["identity"]["jit_artifact_set_sha256"]
        ):
            raise RuntimeError("candidate JIT artifact set drift before target")
        fixture, weights = fixed_fixture_and_weights(args, fixture_module)
        if fixture.manifest != prerequisite["fixture"]:
            raise RuntimeError("target fixture drift from candidate prerequisite")
        if weights.manifest != prerequisite["weights"]:
            raise RuntimeError(
                "target canonical weight drift from candidate prerequisite"
            )
        reference_cpu = (
            fixture_module.reference_moe_nvfp4(fixture, weights).detach().cpu()
        )
        torch.cuda.synchronize()
        arm = build_fused_arm(fixture, weights, max_num_tokens=args.max_num_tokens)
        arm.eager_setup_once()
        arm.capture_once()
        nvtx_name = (
            f"exp003_{capture['capture_id']}_{capture['cluster_id']}_"
            f"pid{os.getpid()}_graph_replay"
        )
        torch.cuda.nvtx.range_push(nvtx_name)
        try:
            output = arm.replay_once()
        finally:
            torch.cuda.nvtx.range_pop()
        workspace = validate_dynamic_wrapper(arm)
        # Copy once after the target replay, then keep all correctness checks
        # on CPU so no post-target diagnostic kernels pollute the IKET launch
        # inventory.
        correctness = correctness_diagnostics(
            output.detach().cpu(), reference_cpu, fixture_module
        )
        control_correctness = prerequisite.get("marker_vs_control_correctness", {}).get(
            "control"
        )
        if not isinstance(control_correctness, dict):
            raise RuntimeError("candidate prerequisite lacks control correctness")
        marker_vs_control = marker_correctness_vs_control(
            correctness, control_correctness
        )
        workspace_evidence = workspace_snapshot(workspace, fixture)
        after = artifact_identity(workspace_path)
        artifact_stable = before == after
        passed = bool(
            correctness["gate_pass"]
            and marker_vs_control["gate_pass"]
            and workspace_evidence["task_model_pass"]
            and artifact_stable
            and arm.protocol()
            == {
                "eager_setup_launches": 1,
                "graph_captures": 1,
                "warmup_replays": 0,
                "target_graph_replays": 1,
            }
        )
        manifest.update(
            status="complete" if passed else "failed_gate",
            case={
                "m": M,
                "experts": E,
                "hidden": H,
                "intermediate_tp": I,
                "topk": TOPK,
                "seed": args.seed,
            },
            fixture=fixture.manifest,
            weights=weights.manifest,
            correctness=correctness,
            marker_vs_control_correctness=marker_vs_control,
            workspace=workspace_evidence,
            graph={
                **manifest.value["graph"],
                "nvtx_range": nvtx_name,
                "protocol": arm.protocol(),
            },
            identity={
                "candidate_manifest": str(args.candidate_manifest.resolve()),
                "candidate_manifest_sha256": file_sha256(
                    args.candidate_manifest.resolve()
                ),
                "jit_before": before,
                "jit_after": after,
                "jit_artifact_set_stable": artifact_stable,
            },
            gate_pass=passed,
        )
        if not passed:
            return 2
        print(f"EXP003_TARGET_COMPLETE pid={os.getpid()} manifest={manifest_path}")
        return 0
    except BaseException as error:
        manifest.update(
            status="error",
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--device-index", type=int, default=0, choices=[0])
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--seed", type=int, default=2026, choices=[2026])
    parser.add_argument("--m", type=int, default=M, choices=[M])
    parser.add_argument("--max-num-tokens", type=int, default=M, choices=[M])
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-control")
    candidate_parser = subparsers.add_parser("prepare-candidate")
    candidate_parser.add_argument(
        "--control-manifest",
        type=Path,
        default=DEFAULT_RESULTS / "preparation" / "prepare-control" / "manifest.json",
    )
    target_parser = subparsers.add_parser("target")
    target_parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=DEFAULT_RESULTS / "preparation" / "prepare-candidate" / "manifest.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.results = args.results.resolve()
    overlay_key, overlay = resolve_overlay_from_env()
    candidate_mode = args.command in {"prepare-candidate", "target"}
    if candidate_mode and overlay is None:
        raise RuntimeError(f"{args.command} requires a marker overlay environment path")
    if not candidate_mode and overlay is not None:
        raise RuntimeError("prepare-control must not install a marker overlay")
    overlay_event_id_budget = (
        validate_overlay_event_id_budget(overlay) if overlay is not None else None
    )
    source = validate_source_checkout(args.flashinfer_root)
    workspace_raw = os.environ.get("FLASHINFER_WORKSPACE_BASE", "")
    if not workspace_raw:
        raise RuntimeError("FLASHINFER_WORKSPACE_BASE is required")
    workspace = Path(workspace_raw).resolve()
    if args.command in {"prepare-control", "prepare-candidate"}:
        require_fresh_workspace(workspace)
    if str(args.flashinfer_root) not in sys.path:
        sys.path.insert(0, str(args.flashinfer_root))
    if overlay is not None:
        install_overlay_before_flashinfer_import(overlay)
    import_identity = configure_source_checkout(args.flashinfer_root)
    expected_origin = (
        overlay if overlay is not None else args.flashinfer_root / TARGET_RELATIVE_PATH
    )
    if Path(import_identity["target_module_origin"]) != expected_origin.resolve():
        raise RuntimeError(
            "target module origin drift: "
            f"{import_identity['target_module_origin']} != {expected_origin}"
        )
    fixture_module = load_fixture_module()
    runtime = runtime_identity(
        source=source,
        import_identity=import_identity,
        overlay_key=overlay_key,
        overlay=overlay,
        overlay_event_id_budget=overlay_event_id_budget,
        expected_gpu_uuid=args.expected_gpu_uuid,
        device_index=args.device_index,
    )
    args.results.mkdir(parents=True, exist_ok=True)
    if args.command in {"prepare-control", "prepare-candidate"}:
        return prepare(args, runtime, fixture_module)
    if args.command == "target":
        return target(args, runtime, fixture_module)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
