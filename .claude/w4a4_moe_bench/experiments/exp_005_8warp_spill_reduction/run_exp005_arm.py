#!/usr/bin/env python3
"""One-process/one-arm/M GPU worker for exp_005.

Every invocation installs exactly one immutable kernel overlay before importing
FlashInfer.  ``prepare`` requires a fresh JIT root.  Later ``measure`` and
``profile`` invocations reuse only that same arm/M/fixture root from new Python
processes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from exp005_common import (
    ALL_FIXTURES,
    BASELINE,
    CANDIDATE,
    CANONICAL_FIXTURE,
    DEFAULT_RESULTS,
    E,
    EXPECTED_CUTLASS_COMMIT,
    EXPECTED_FLASHINFER_COMMIT,
    EXPECTED_GRID,
    EXPECTED_IMAGE_DIGEST,
    EXPECTED_KERNEL_SHA256,
    EXPECTED_PYTHON_DEPS_SHA256,
    H,
    I,
    KNOWN_ARMS,
    MAX_ACTIVE_CLUSTERS,
    M_VALUES,
    NUM_SMS,
    TARGET_MODULE,
    TARGET_RELATIVE_PATH,
    TOPK,
    artifact_manifest,
    canonical_sha256,
    case_directory,
    expected_block,
    file_sha256,
    require_arm_m,
    require_clean_compiler_environment,
    require_empty_directory,
    verify_workspace_evidence,
    write_json,
)


ROOT = Path(__file__).resolve().parent
FIXTURE_PATH = ROOT.parent / "exp_002_fused_vs_chain_dataflow" / "fixture.py"


def command_output(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            list(command), cwd=cwd, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"ERROR: {error}"


class ExactModuleOverlayFinder(importlib.abc.MetaPathFinder):
    def __init__(self, overlay: Path):
        self.overlay = overlay

    def find_spec(self, fullname: str, path=None, target=None):
        del path, target
        if fullname != TARGET_MODULE:
            return None
        return importlib.util.spec_from_file_location(fullname, self.overlay)


def install_overlay(overlay: Path) -> None:
    if TARGET_MODULE in sys.modules:
        raise RuntimeError("target module imported before exp_005 overlay installation")
    sys.meta_path.insert(0, ExactModuleOverlayFinder(overlay))


def git(repo: Path, *args: str) -> str:
    return command_output(["git", "-c", f"safe.directory={repo}", *args], cwd=repo)


def validate_source(repo: Path, overlay: Path, arm: str) -> dict[str, Any]:
    production = repo / TARGET_RELATIVE_PATH
    cutlass = repo / "3rdparty/cutlass"
    if not production.is_file() or not overlay.is_file():
        raise RuntimeError("production kernel or selected arm overlay is missing")
    production_hash = file_sha256(production)
    if production_hash != EXPECTED_KERNEL_SHA256:
        raise RuntimeError(f"production kernel hash drift: {production_hash}")
    checkout_head = git(repo, "rev-parse", "HEAD")
    ancestor = git(
        repo, "merge-base", "--is-ancestor", EXPECTED_FLASHINFER_COMMIT, checkout_head
    )
    if ancestor.startswith("ERROR"):
        raise RuntimeError(
            f"locked source commit {EXPECTED_FLASHINFER_COMMIT} is not an ancestor "
            f"of checkout {checkout_head}"
        )
    cutlass_head = git(cutlass, "rev-parse", "HEAD")
    if cutlass_head != EXPECTED_CUTLASS_COMMIT:
        raise RuntimeError(f"CUTLASS commit drift: {cutlass_head}")
    overlay_hash = file_sha256(overlay)
    if arm == BASELINE and overlay_hash != production_hash:
        raise RuntimeError("baseline overlay is not byte-identical to production")
    if arm != BASELINE and overlay_hash == production_hash:
        raise RuntimeError("candidate overlay is byte-identical to production")
    return {
        "locked_source_commit": EXPECTED_FLASHINFER_COMMIT,
        "checkout_head": checkout_head,
        "cutlass_commit": cutlass_head,
        "production_kernel": str(production),
        "production_kernel_sha256": production_hash,
        "overlay": str(overlay),
        "overlay_sha256": overlay_hash,
    }


def configure_source_checkout(repo: Path) -> dict[str, str]:
    flashinfer = importlib.import_module("flashinfer")
    imported_root = Path(flashinfer.__file__).resolve().parents[1]
    if imported_root != repo:
        raise RuntimeError(f"imported FlashInfer root {imported_root} != {repo}")
    from flashinfer.jit import env as jit_env

    jit_env.FLASHINFER_CSRC_DIR = repo / "csrc"
    jit_env.FLASHINFER_INCLUDE_DIR = repo / "include"
    jit_env.FLASHINFER_AOT_DIR = (
        jit_env.FLASHINFER_WORKSPACE_DIR / "aot_disabled_for_exp005"
    )
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
    }


def load_fixture_module():
    spec = importlib.util.spec_from_file_location("exp005_fixture", FIXTURE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixture module: {FIXTURE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    error = actual_f - expected_f
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.flatten(), expected_f.flatten(), dim=0
    )
    denominator = torch.linalg.vector_norm(expected_f, dim=1).clamp_min(1e-12)
    token_relative = torch.linalg.vector_norm(error, dim=1) / denominator
    return {
        "cosine_loss": max(0.0, 1.0 - float(cosine.item())),
        "relative_l2": float(
            (
                torch.linalg.vector_norm(error)
                / torch.linalg.vector_norm(expected_f).clamp_min(1e-12)
            ).item()
        ),
        "max_abs": float(error.abs().max().item()),
        "token_rel_l2_p99": float(torch.quantile(token_relative, 0.99).item()),
    }


def _gpu_query() -> dict[str, str]:
    fields = (
        "uuid,name,pci.bus_id,driver_version,clocks.current.graphics,"
        "clocks.applications.graphics,clocks.max.graphics,power.draw"
    )
    output = command_output(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
    )
    if output.startswith("ERROR"):
        raise RuntimeError(output)
    rows = [row for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            "exp_005 worker requires exactly one visible GPU; got " + repr(rows)
        )
    values = [value.strip() for value in rows[0].split(",")]
    keys = (
        "uuid",
        "name",
        "pci_bus_id",
        "driver",
        "graphics_clock_mhz",
        "applications_graphics_clock_mhz",
        "max_graphics_clock_mhz",
        "power_draw_w",
    )
    if len(values) != len(keys):
        raise RuntimeError(f"unexpected nvidia-smi GPU row: {rows[0]}")
    return dict(zip(keys, values, strict=True))


def _foreign_processes(gpu_uuid: str) -> list[dict[str, str]]:
    output = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    if output.startswith("ERROR"):
        raise RuntimeError(output)
    rows = []
    for line in output.splitlines():
        if not line.strip() or "No running processes" in line:
            continue
        values = [value.strip() for value in line.split(",", 2)]
        if len(values) == 3 and values[0] == gpu_uuid:
            rows.append({"gpu_uuid": values[0], "pid": values[1], "process": values[2]})
    return rows


def runtime_identity(
    args: argparse.Namespace, source: dict[str, Any]
) -> dict[str, Any]:
    require_clean_compiler_environment()
    required = {
        "W4A4_IMAGE_DIGEST": EXPECTED_IMAGE_DIGEST,
        "W4A4_PYTHON_DEPS_SHA256": EXPECTED_PYTHON_DEPS_SHA256,
    }
    for key, expected in required.items():
        if os.environ.get(key) != expected:
            raise RuntimeError(f"{key} identity drift")
    if not os.environ.get("KDK_LEASE_ID"):
        raise RuntimeError("KDK_LEASE_ID is required")
    if Path(os.environ.get("FLASHINFER_WORKSPACE_BASE", "")).resolve() != args.jit_root:
        raise RuntimeError("FLASHINFER_WORKSPACE_BASE must equal --jit-root")
    if os.environ.get("CUTE_DSL_KEEP") != "ir,ptx,cubin,sass":
        raise RuntimeError("CUTE_DSL_KEEP must equal ir,ptx,cubin,sass")
    dump_dir = Path(os.environ.get("CUTE_DSL_DUMP_DIR", "")).resolve()
    try:
        dump_dir.relative_to(args.jit_root)
    except ValueError as error:
        raise RuntimeError("CUTE_DSL_DUMP_DIR must be inside --jit-root") from error

    gpu = _gpu_query()
    if gpu["uuid"] != args.expected_gpu_uuid:
        raise RuntimeError(f"GPU UUID drift: {gpu['uuid']} != {args.expected_gpu_uuid}")
    foreign = _foreign_processes(gpu["uuid"])
    if foreign:
        raise RuntimeError(f"foreign GPU compute processes detected: {foreign}")

    torch.cuda.set_device(args.device_index)
    properties = torch.cuda.get_device_properties(args.device_index)
    capability = list(torch.cuda.get_device_capability(args.device_index))
    if capability not in ([12, 0], [12, 1]):
        raise RuntimeError(f"exp_005 requires SM120/121, got {capability}")
    if int(properties.multi_processor_count) != NUM_SMS:
        raise RuntimeError(
            f"SM count drift: {properties.multi_processor_count} != {NUM_SMS}"
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
        "gpu": {
            **gpu,
            "compute_capability": capability,
            "sm_count": int(properties.multi_processor_count),
            "foreign_processes_before_cuda_context": foreign,
        },
        "image_digest": os.environ["W4A4_IMAGE_DIGEST"],
        "python_deps_sha256": os.environ["W4A4_PYTHON_DEPS_SHA256"],
        "lease_id": os.environ["KDK_LEASE_ID"],
        "jit_root": str(args.jit_root),
        "source": source,
    }


def make_directed_fixture(fixture_module: Any, base: Any, kind: str):
    if kind == CANONICAL_FIXTURE:
        return base
    m = base.m
    token = torch.arange(m, device=base.x.device, dtype=torch.int64)[:, None]
    slot = torch.arange(TOPK, device=base.x.device, dtype=torch.int64)[None, :]
    ids = ((token * 11 + slot) % (E - 1) + 1).to(torch.int32)
    if kind == "sparse_empty":
        ids = slot.expand(m, TOPK).to(torch.int32)
    elif kind == "exact_128":
        ids[:128, 0] = 0
    elif kind == "tail_129":
        ids[:129, 0] = 0
    elif kind == "hot_expert":
        ids[:, 0] = 0
    else:
        raise ValueError(f"unknown directed fixture: {kind}")
    sorted_ids = ids.sort(dim=-1).values
    if bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any().item()):
        raise RuntimeError(
            f"directed fixture {kind} contains duplicate per-token experts"
        )
    slot_weights = torch.arange(1, TOPK + 1, device=base.x.device, dtype=torch.float32)
    weights = (slot_weights / slot_weights.sum()).expand(m, TOPK).contiguous()
    occupancy = torch.bincount(ids.flatten().long(), minlength=E)
    manifest = {
        "fixture_kind": f"directed_{kind}",
        "seed": base.manifest["seed"],
        "m": m,
        "shape": {"experts": E, "hidden": H, "intermediate_tp": I, "topk": TOPK},
        "x_sha256": fixture_module.tensor_sha256(base.x),
        "topk_ids_sha256": fixture_module.tensor_sha256(ids),
        "topk_weights_sha256": fixture_module.tensor_sha256(weights),
        "occupancy_sha256": fixture_module.tensor_sha256(occupancy),
        "occupancy_min": int(occupancy.min().item()),
        "occupancy_max": int(occupancy.max().item()),
        "zero_token_experts": int((occupancy == 0).sum().item()),
        "duplicate_expert_ids": 0,
        "weight_sum_max_abs_error": float(
            (weights.sum(dim=-1) - 1.0).abs().max().item()
        ),
    }
    return fixture_module.RoutedFixture(m, base.x, ids, weights, manifest)


def make_case(args: argparse.Namespace):
    fixture_module = load_fixture_module()
    device = torch.device("cuda", args.device_index)
    base = fixture_module.make_routed_fixture(args.m, device=device, seed=args.seed)
    fixture = make_directed_fixture(fixture_module, base, args.fixture)
    weights = fixture_module.make_canonical_weights(device=device, seed=args.seed)
    return fixture_module, fixture, weights


@dataclass
class CapturedArm:
    launch: Callable[[], torch.Tensor]
    wrapper: Any
    graph: torch.cuda.CUDAGraph | None = None
    output: torch.Tensor | None = None
    start: torch.cuda.Event | None = None
    end: torch.cuda.Event | None = None

    def eager(self) -> torch.Tensor:
        self.output = self.launch()
        torch.cuda.synchronize()
        return self.output

    def capture(self) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.launch()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        self.start = torch.cuda.Event(enable_timing=True, external=True)
        self.end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(self.graph, stream=stream):
            self.start.record()
            self.output = self.launch()
            self.end.record()
        torch.cuda.synchronize()

    def replay(self, *, sentinel: bool = False) -> tuple[torch.Tensor, float]:
        if self.graph is None or self.start is None or self.end is None:
            raise RuntimeError("CUDA graph is not captured")
        if sentinel:
            self.wrapper._moe_output.fill_(float("nan"))
            torch.cuda.synchronize()
        self.graph.replay()
        torch.cuda.synchronize()
        assert self.output is not None
        return self.output, float(self.start.elapsed_time(self.end))


def build_arm(args: argparse.Namespace, fixture: Any, weights: Any) -> CapturedArm:
    from flashinfer.fused_moe.cute_dsl import B12xMoEWrapper

    values = weights.cutedsl()
    wrapper = B12xMoEWrapper(
        num_experts=E,
        top_k=TOPK,
        hidden_size=H,
        intermediate_size=I,
        use_cuda_graph=True,
        max_num_tokens=args.m,
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

    return CapturedArm(launch, wrapper)


def _workspace_snapshot(
    wrapper: Any, fixture: Any, *, num_cta_warps: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    workspace = wrapper._dynamic_workspace
    if workspace is None:
        raise RuntimeError("exp_005 case did not select the dynamic workspace")
    task_tail = int(workspace.task_tail.item())
    tensor_names = (
        "row_counts",
        "expert_write_rows",
        "expert_tile_base",
        "task_ready",
        "task_expert",
        "task_m_tile",
        "task_slice_begin",
        "task_slice_count",
        "task_valid_rows",
        "tile_write_count",
    )
    tensors: dict[str, torch.Tensor] = {}
    for name in tensor_names:
        value = getattr(workspace, name).detach().cpu().clone()
        if name.startswith("task_"):
            value = value[:task_tail]
        tensors[name] = value
    scalar_names = (
        "pair_head",
        "producers_done_count",
        "all_work_published",
        "task_head",
        "task_tail",
        "barrier_count",
        "barrier_epoch",
    )
    scalars = {name: int(getattr(workspace, name).item()) for name in scalar_names}
    expected_rows = torch.bincount(fixture.topk_ids.flatten().long(), minlength=E).cpu()
    plain: dict[str, Any] = {
        **{name: value.tolist() for name, value in tensors.items()},
        **scalars,
        "routed_rows": int(fixture.m * TOPK),
    }
    verification = verify_workspace_evidence(
        plain,
        expected_row_counts=expected_rows.tolist(),
        num_cta_warps=num_cta_warps,
        grid_z=NUM_SMS,
    )
    summary = {
        "schema": "exp005.workspace-route-task-evidence.v1",
        "workspace_type": type(workspace).__name__,
        "workspace_capacity": {
            "routed_rows": int(workspace.routed_rows_capacity),
            "physical_tiles": int(workspace.physical_tiles_capacity),
            "tasks": int(workspace.task_capacity),
        },
        "scalars": scalars,
        "tensor_sha256": {
            name: tensor_sha256(value) for name, value in tensors.items()
        },
        "expected_row_counts_sha256": tensor_sha256(expected_rows),
        "verification": verification,
    }
    return tensors, summary


def _compile_identity() -> dict[str, Any]:
    dispatch = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch"
    )
    entries = list(dispatch._DYNAMIC_KERNEL_CACHE.items())
    if not entries:
        raise RuntimeError("dynamic kernel cache is empty after launch")
    macs = sorted({int(value[1]) for _, value in entries})
    if macs != [MAX_ACTIVE_CLUSTERS]:
        raise RuntimeError(
            f"compiled max_active_clusters drift: {macs} != {[MAX_ACTIVE_CLUSTERS]}"
        )
    return {
        "dynamic_cache_entries": len(entries),
        "compiled_max_active_clusters": macs,
        "compiled_object_types": sorted(
            {type(value[0]).__name__ for _, value in entries}
        ),
    }


def make_flusher(device: torch.device, bytes_: int = 192 << 20):
    buffer = torch.empty((bytes_ + 3) // 4, dtype=torch.int32, device=device)
    state = 0

    def flush() -> None:
        nonlocal state
        state += 1
        buffer.fill_(state)
        torch.cuda.synchronize()

    flush()
    return flush, buffer.numel() * buffer.element_size()


def preparation_path(args: argparse.Namespace) -> Path:
    return (
        case_directory(args.results, args.arm, args.m, args.fixture)
        / "preparation.json"
    )


def prepare(args: argparse.Namespace, runtime: dict[str, Any]) -> int:
    fixture_module, fixture, weights = make_case(args)
    reference = fixture_module.reference_moe_nvfp4(fixture, weights)
    arm = build_arm(args, fixture, weights)
    arm.eager()
    arm.capture()
    raw_dir = case_directory(args.results, args.arm, args.m, args.fixture)
    raw_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    route_evidence = []
    for replay in range(2):
        output, elapsed_ms = arm.replay(sentinel=True)
        output = output.clone()
        tensors, workspace_summary = _workspace_snapshot(
            arm.wrapper,
            fixture,
            num_cta_warps=expected_block(args.arm)[0] // 32,
        )
        torch.save(output.detach().cpu(), raw_dir / f"output_{replay}.pt")
        torch.save(tensors, raw_dir / f"workspace_replay_{replay}.pt")
        write_json(raw_dir / f"workspace_replay_{replay}.json", workspace_summary)
        diagnostics = fixture_module.output_diagnostics(output, reference)
        diagnostics["sentinel_nan_remaining"] = int(torch.isnan(output).sum().item())
        diagnostics["event_elapsed_us"] = elapsed_ms * 1000.0
        if not diagnostics["formal_pass"] or diagnostics["sentinel_nan_remaining"]:
            raise RuntimeError(f"correctness/sentinel gate failed on replay {replay}")
        if not workspace_summary["verification"]["gate_pass"]:
            raise RuntimeError(f"workspace route/task gate failed on replay {replay}")
        outputs.append(diagnostics)
        route_evidence.append(
            {
                "json": f"workspace_replay_{replay}.json",
                "pt": f"workspace_replay_{replay}.pt",
                "verification": workspace_summary["verification"],
            }
        )
    output_0 = torch.load(raw_dir / "output_0.pt", weights_only=True)
    output_1 = torch.load(raw_dir / "output_1.pt", weights_only=True)
    artifacts = artifact_manifest(args.jit_root)
    cubins = [item for item in artifacts if item["path"].endswith(".cubin")]
    if not cubins:
        raise RuntimeError("fresh JIT preparation produced no cubin artifact")
    compile_identity = _compile_identity()
    payload = {
        "schema": "exp005.arm-preparation.v1",
        "status": "complete",
        "arm": args.arm,
        "m": args.m,
        "fixture_kind": args.fixture,
        "runtime": runtime,
        "case": {
            "m": args.m,
            "experts": E,
            "hidden": H,
            "intermediate_tp": I,
            "topk": TOPK,
        },
        "fixture": fixture.manifest,
        "weights": weights.manifest,
        "reference_sha256": tensor_sha256(reference),
        "outputs": outputs,
        "output_stability": tensor_error(output_1, output_0),
        "route_task_evidence": route_evidence,
        "compile_identity": compile_identity,
        "launch_contract": {
            "num_sms": NUM_SMS,
            "max_active_clusters": MAX_ACTIVE_CLUSTERS,
            "expected_grid": list(EXPECTED_GRID),
            "expected_block": list(expected_block(args.arm)),
            "expected_final_replay_kernel": "MoEDynamicKernel",
            "profiler_observed_grid": None,
            "profiler_observed_block": None,
            "profiler_observed_kernel": None,
            "profiler_verification": "pending_external_profiler_artifact",
            "evidence_boundary": (
                "expected geometry is a contract, not a source-derived observed launch; "
                "NCU/NSys artifact must populate observed fields"
            ),
        },
        "jit_artifacts": artifacts,
        "cubin_sha256": sorted(item["sha256"] for item in cubins),
        "jit_artifact_set_sha256": canonical_sha256(artifacts),
    }
    write_json(preparation_path(args), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def require_preparation(
    args: argparse.Namespace, runtime: dict[str, Any]
) -> dict[str, Any]:
    path = preparation_path(args)
    if not path.is_file():
        raise RuntimeError(f"arm/M preparation is missing: {path}")
    value = json.loads(path.read_text())
    if (
        value.get("status") != "complete"
        or value.get("arm") != args.arm
        or value.get("m") != args.m
        or value.get("fixture_kind") != args.fixture
    ):
        raise RuntimeError("arm/M preparation identity drift")
    artifacts = artifact_manifest(args.jit_root)
    if canonical_sha256(artifacts) != value.get("jit_artifact_set_sha256"):
        raise RuntimeError("per-arm/M JIT artifact identity drift")
    stable = ("image_digest", "python_deps_sha256", "source", "jit_root", "imports")
    for field in stable:
        if runtime.get(field) != value["runtime"].get(field):
            raise RuntimeError(f"runtime identity drift at {field}")
    stable_gpu_fields = (
        "uuid",
        "name",
        "pci_bus_id",
        "driver",
        "compute_capability",
        "sm_count",
    )
    for field in stable_gpu_fields:
        if runtime["gpu"].get(field) != value["runtime"]["gpu"].get(field):
            raise RuntimeError(f"runtime GPU identity drift at {field}")
    return value


def measure(args: argparse.Namespace, runtime: dict[str, Any]) -> int:
    prerequisite = require_preparation(args, runtime)
    _, fixture, weights = make_case(args)
    if fixture.manifest != prerequisite.get("fixture"):
        raise RuntimeError("benchmark fixture identity drift")
    if weights.manifest != prerequisite.get("weights"):
        raise RuntimeError("benchmark weight identity drift")
    arm = build_arm(args, fixture, weights)
    arm.eager()
    arm.capture()
    flush, flush_bytes = make_flusher(fixture.x.device)
    for _ in range(args.warmup):
        flush()
        arm.replay()
    total_ms = 0.0
    for _ in range(args.iters):
        flush()
        _, elapsed_ms = arm.replay()
        total_ms += elapsed_ms
    if (
        canonical_sha256(artifact_manifest(args.jit_root))
        != prerequisite["jit_artifact_set_sha256"]
    ):
        raise RuntimeError("JIT artifact identity drift after benchmark replay")
    payload = {
        "schema": "exp005.arm-measurement.v1",
        "status": "complete",
        "arm": args.arm,
        "m": args.m,
        "fixture_kind": args.fixture,
        "group": args.group,
        "position": args.position,
        "order": [
            args.comparison_anchor,
            args.comparison_subject,
            args.comparison_subject,
            args.comparison_anchor,
        ],
        "declared_clock_policy": args.clock_policy,
        "sample_us": total_ms * 1000.0 / args.iters,
        "warmup": args.warmup,
        "iters": args.iters,
        "l2_flush_bytes": flush_bytes,
        "timing": "outer CUDA graph with external CUDA events",
        "runtime": runtime,
        "jit_artifact_set_sha256": prerequisite["jit_artifact_set_sha256"],
    }
    output = (
        args.results
        / "raw"
        / "benchmark"
        / f"m{args.m}"
        / f"group_{args.group}_position_{args.position}_{args.arm}.json"
    )
    if output.exists():
        raise RuntimeError(f"immutable benchmark sample already exists: {output}")
    write_json(output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def profile(args: argparse.Namespace, runtime: dict[str, Any]) -> int:
    prerequisite = require_preparation(args, runtime)
    _, fixture, weights = make_case(args)
    if fixture.manifest != prerequisite.get("fixture"):
        raise RuntimeError("profile fixture identity drift")
    arm = build_arm(args, fixture, weights)
    arm.eager()
    arm.capture()
    for _ in range(args.warmup):
        arm.replay()
    nvtx = f"exp005_{args.arm}_m{args.m}_final_replay"
    torch.cuda.nvtx.range_push(nvtx)
    cudart = torch.cuda.cudart()
    try:
        if int(cudart.cudaProfilerStart()) != 0:
            raise RuntimeError("cudaProfilerStart failed")
        output, elapsed_ms = arm.replay()
        if int(cudart.cudaProfilerStop()) != 0:
            raise RuntimeError("cudaProfilerStop failed")
    finally:
        torch.cuda.nvtx.range_pop()
    payload = {
        "schema": "exp005.profile-target.v1",
        "status": "complete",
        "arm": args.arm,
        "m": args.m,
        "fixture_kind": args.fixture,
        "nvtx_range": nvtx,
        "event_elapsed_us": elapsed_ms * 1000.0,
        "output_sha256": tensor_sha256(output),
        "expected_launch": {
            "grid": list(EXPECTED_GRID),
            "block": list(expected_block(args.arm)),
            "kernel": "MoEDynamicKernel",
        },
        "profiler_observed_launch": {
            "grid": None,
            "block": None,
            "kernel": None,
            "verification": "pending_profiler_artifact_parse",
        },
        "runtime": runtime,
        "jit_artifact_set_sha256": canonical_sha256(artifact_manifest(args.jit_root)),
    }
    output_path = (
        args.results / "profile_targets" / args.arm / f"m{args.m}" / "target.json"
    )
    write_json(output_path, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--arm", choices=KNOWN_ARMS, required=True)
    parser.add_argument("--m", type=int, choices=M_VALUES, required=True)
    parser.add_argument("--fixture", choices=ALL_FIXTURES, default=CANONICAL_FIXTURE)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--device-index", type=int, default=0, choices=[0])
    parser.add_argument("--seed", type=int, default=2026, choices=[2026])
    parser.add_argument("--comparison-anchor", choices=KNOWN_ARMS, default=BASELINE)
    parser.add_argument("--comparison-subject", choices=KNOWN_ARMS, default=CANDIDATE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--group", type=int, choices=range(5), required=True)
    measure_parser.add_argument("--position", type=int, choices=range(4), required=True)
    measure_parser.add_argument("--warmup", type=int, default=5)
    measure_parser.add_argument("--iters", type=int, default=50)
    measure_parser.add_argument(
        "--clock-policy", choices=("locked", "unlocked"), required=True
    )
    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.results = args.results.resolve()
    args.overlay = args.overlay.resolve()
    args.jit_root = args.jit_root.resolve()
    require_arm_m(args.arm, args.m)
    source = validate_source(args.flashinfer_root, args.overlay, args.arm)
    if args.command == "prepare":
        require_empty_directory(args.jit_root)
        if preparation_path(args).exists():
            raise RuntimeError(
                f"immutable preparation already exists: {preparation_path(args)}"
            )
    if str(args.flashinfer_root) not in sys.path:
        sys.path.insert(0, str(args.flashinfer_root))
    install_overlay(args.overlay)
    imports = configure_source_checkout(args.flashinfer_root)
    if Path(imports["target_module"]) != args.overlay:
        raise RuntimeError("target module did not resolve to selected arm overlay")
    runtime = runtime_identity(args, source)
    runtime["imports"] = imports
    args.results.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        return prepare(args, runtime)
    if args.command == "measure":
        return measure(args, runtime)
    if args.command == "profile":
        return profile(args, runtime)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
