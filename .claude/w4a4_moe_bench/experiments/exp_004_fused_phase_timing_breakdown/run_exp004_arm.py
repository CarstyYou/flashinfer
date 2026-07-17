#!/usr/bin/env python3
"""One-process/one-arm GPU worker for exp_004.

The worker is intentionally executable only on an exclusively leased 5KP.  It
installs exact kernel and dispatch overlays before importing FlashInfer, builds
one fresh JIT identity, validates correctness/work, and can retain five raw
phase-capture replays.  No production source is edited.
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
from typing import Any, Callable, Mapping, Sequence

import torch

from exp004_common import (
    ALL_ARMS,
    DEFAULT_RESULTS,
    DISPATCH_MODULE,
    DISPATCH_RELATIVE_PATH,
    E,
    EXPECTED_BLOCK,
    EXPECTED_CUTLASS_COMMIT,
    EXPECTED_DISPATCH_SHA256,
    EXPECTED_FLASHINFER_COMMIT,
    EXPECTED_GRID,
    EXPECTED_IMAGE_DIGEST,
    EXPECTED_KERNEL_SHA256,
    EXPECTED_PYTHON_DEPS_SHA256,
    EXPECTED_WRAPPER_SHA256,
    H,
    I,
    KERNEL_MODULE,
    KERNEL_RELATIVE_PATH,
    M,
    MAX_ACTIVE_CLUSTERS,
    MEASURED_REPLAYS,
    MEASUREMENT_CONTROL,
    NORMAL,
    PROBE,
    SENTINEL,
    TOPK,
    WRAPPER_RELATIVE_PATH,
    artifact_manifest,
    canonical_sha256,
    decode_probe_buffer,
    file_sha256,
    read_json,
    require_clean_compiler_environment,
    require_empty_directory,
    timing_ticks_capacity,
    validate_hardware_identity,
    validate_no_marker_buffer,
    verify_workspace_evidence,
    write_json,
)


ROOT = Path(__file__).resolve().parent
FIXTURE_PATH = ROOT.parent / "exp_002_fused_vs_chain_dataflow" / "fixture.py"
CLOCK_CALIBRATION_SOURCE = ROOT / "clock_calibration.cu"


def command_output(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            list(command), cwd=cwd, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"ERROR: {error}"


class ExactModuleOverlayFinder(importlib.abc.MetaPathFinder):
    def __init__(self, mapping: Mapping[str, Path]):
        self.mapping = dict(mapping)

    def find_spec(self, fullname: str, path=None, target=None):
        del path, target
        overlay = self.mapping.get(fullname)
        if overlay is None:
            return None
        return importlib.util.spec_from_file_location(fullname, overlay)


def install_overlays(kernel: Path, dispatch: Path) -> None:
    imported = [
        name for name in (KERNEL_MODULE, DISPATCH_MODULE) if name in sys.modules
    ]
    if imported:
        raise RuntimeError(
            f"target modules imported before overlay installation: {imported}"
        )
    sys.meta_path.insert(
        0,
        ExactModuleOverlayFinder({KERNEL_MODULE: kernel, DISPATCH_MODULE: dispatch}),
    )


def git(repo: Path, *args: str) -> str:
    return command_output(["git", "-c", f"safe.directory={repo}", *args], cwd=repo)


def validate_source(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.flashinfer_root
    production = {
        "kernel": repo / KERNEL_RELATIVE_PATH,
        "dispatch": repo / DISPATCH_RELATIVE_PATH,
        "wrapper": repo / WRAPPER_RELATIVE_PATH,
    }
    expected = {
        "kernel": EXPECTED_KERNEL_SHA256,
        "dispatch": EXPECTED_DISPATCH_SHA256,
        "wrapper": EXPECTED_WRAPPER_SHA256,
    }
    hashes = {name: file_sha256(path) for name, path in production.items()}
    if hashes != expected:
        raise RuntimeError(f"production source identity drift: {hashes} != {expected}")
    overlays = {
        "kernel": args.kernel_overlay,
        "dispatch": args.dispatch_overlay,
    }
    if not all(path.is_file() for path in overlays.values()):
        raise RuntimeError("selected overlay is missing")
    overlay_hashes = {name: file_sha256(path) for name, path in overlays.items()}
    if args.arm == NORMAL:
        if overlay_hashes != {
            "kernel": expected["kernel"],
            "dispatch": expected["dispatch"],
        }:
            raise RuntimeError(
                "normal_no_marker overlays are not byte-identical production"
            )
    else:
        if overlay_hashes["kernel"] == expected["kernel"]:
            raise RuntimeError(
                "measurement kernel overlay is unexpectedly production-identical"
            )
        if overlay_hashes["dispatch"] == expected["dispatch"]:
            raise RuntimeError(
                "measurement dispatch overlay is unexpectedly production-identical"
            )

    checkout_head = git(repo, "rev-parse", "HEAD")
    ancestor = git(
        repo, "merge-base", "--is-ancestor", EXPECTED_FLASHINFER_COMMIT, checkout_head
    )
    if ancestor.startswith("ERROR"):
        raise RuntimeError(
            f"locked source {EXPECTED_FLASHINFER_COMMIT} is not an ancestor of {checkout_head}"
        )
    cutlass = repo / "3rdparty/cutlass"
    cutlass_head = git(cutlass, "rev-parse", "HEAD")
    if cutlass_head != EXPECTED_CUTLASS_COMMIT:
        raise RuntimeError(f"CUTLASS source drift: {cutlass_head}")
    return {
        "checkout_head": checkout_head,
        "locked_source_commit": EXPECTED_FLASHINFER_COMMIT,
        "cutlass_commit": cutlass_head,
        "production": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in production.items()
        },
        "overlays": {
            name: {"path": str(path), "sha256": overlay_hashes[name]}
            for name, path in overlays.items()
        },
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
        jit_env.FLASHINFER_WORKSPACE_DIR / "aot_disabled_for_exp004"
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
    kernel = importlib.import_module(KERNEL_MODULE)
    dispatch = importlib.import_module(DISPATCH_MODULE)
    cutlass = importlib.import_module("cutlass")
    return {
        "flashinfer": str(Path(flashinfer.__file__).resolve()),
        "kernel_module": str(Path(kernel.__file__).resolve()),
        "dispatch_module": str(Path(dispatch.__file__).resolve()),
        "cutlass_python": str(Path(cutlass.__file__).resolve()),
        "cutlass_python_version": str(getattr(cutlass, "__version__", "unknown")),
    }


def load_fixture_module():
    spec = importlib.util.spec_from_file_location("exp004_fixture", FIXTURE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixture: {FIXTURE_PATH}")
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
    dot = torch.sum(actual_f.double() * expected_f.double())
    norms = torch.linalg.vector_norm(actual_f.double()) * torch.linalg.vector_norm(
        expected_f.double()
    )
    cosine = float((dot / norms.clamp_min(1e-30)).item())
    token_denominator = torch.linalg.vector_norm(expected_f, dim=1).clamp_min(1e-12)
    token_relative = torch.linalg.vector_norm(error, dim=1) / token_denominator
    return {
        "cosine": cosine,
        "cosine_loss": max(0.0, 1.0 - cosine),
        "relative_l2": float(
            (
                torch.linalg.vector_norm(error)
                / torch.linalg.vector_norm(expected_f).clamp_min(1e-12)
            ).item()
        ),
        "max_abs": float(error.abs().max().item()),
        "token_rel_l2_p99": float(torch.quantile(token_relative, 0.99).item()),
    }


def correctness_gate(metrics: Mapping[str, float]) -> dict[str, Any]:
    checks = {
        "cosine": float(metrics["cosine"]) >= 0.999,
        "relative_l2": float(metrics["relative_l2"]) <= 0.02,
        "max_abs": float(metrics["max_abs"]) <= 0.08,
    }
    return {"checks": checks, "gate_pass": all(checks.values())}


def output_contract(output: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    checks = {
        "shape": tuple(output.shape) == tuple(reference.shape),
        "dtype_bfloat16": output.dtype == torch.bfloat16,
        "finite": bool(torch.isfinite(output).all().item()),
        "every_token_nonzero": bool(
            torch.all(torch.linalg.vector_norm(output.float(), dim=1) > 0).item()
        ),
    }
    return {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "checks": checks,
        "gate_pass": all(checks.values()),
    }


def _gpu_query(expected_uuid: str) -> dict[str, str]:
    fields = (
        "uuid,name,pci.bus_id,driver_version,clocks.current.graphics,"
        "clocks.applications.graphics,clocks.max.graphics,power.draw"
    )
    output = command_output(
        [
            "nvidia-smi",
            f"--id={expected_uuid}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = [row for row in output.splitlines() if row.strip()]
    if output.startswith("ERROR") or len(rows) != 1:
        raise RuntimeError(f"exp004 requires exactly one visible GPU: {output}")
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
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != len(keys):
        raise RuntimeError(f"unexpected nvidia-smi row: {rows[0]}")
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
    result = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",", 2)]
        if len(values) == 3 and values[0] == gpu_uuid and values[1] != str(os.getpid()):
            result.append(
                {"gpu_uuid": values[0], "pid": values[1], "process": values[2]}
            )
    return result


def require_no_foreign_process(runtime: Mapping[str, Any]) -> list[dict[str, str]]:
    foreign = _foreign_processes(str(runtime["gpu"]["uuid"]))
    if foreign:
        raise RuntimeError(
            f"foreign compute process appeared during capture: {foreign}"
        )
    return foreign


def runtime_identity(
    args: argparse.Namespace, source: Mapping[str, Any]
) -> dict[str, Any]:
    require_clean_compiler_environment()
    if os.environ.get("W4A4_IMAGE_DIGEST") != EXPECTED_IMAGE_DIGEST:
        raise RuntimeError("container image digest drift")
    if os.environ.get("W4A4_PYTHON_DEPS_SHA256") != EXPECTED_PYTHON_DEPS_SHA256:
        raise RuntimeError("Python dependency identity drift")
    if not os.environ.get("KDK_LEASE_ID"):
        raise RuntimeError("KDK_LEASE_ID is required")
    if os.environ.get("KDK_LEASE_GPU_UUID") != args.expected_gpu_uuid:
        raise RuntimeError("KDK_LEASE_GPU_UUID must equal the selected full GPU UUID")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.expected_gpu_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must isolate the leased full GPU UUID")
    if Path(os.environ.get("FLASHINFER_WORKSPACE_BASE", "")).resolve() != args.jit_root:
        raise RuntimeError("FLASHINFER_WORKSPACE_BASE must equal the fresh JIT root")
    if os.environ.get("CUTE_DSL_KEEP") != "ir,ptx,cubin,sass":
        raise RuntimeError("CUTE_DSL_KEEP must preserve ir,ptx,cubin,sass")

    gpu = _gpu_query(args.expected_gpu_uuid)
    if gpu["uuid"] != args.expected_gpu_uuid:
        raise RuntimeError(f"GPU UUID drift: {gpu['uuid']} != {args.expected_gpu_uuid}")
    foreign = _foreign_processes(gpu["uuid"])
    if foreign:
        raise RuntimeError(f"foreign compute process on leased 5KP: {foreign}")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    capability = list(torch.cuda.get_device_capability(0))
    hardware = {
        **gpu,
        "compute_capability": capability,
        "sm_count": int(properties.multi_processor_count),
    }
    identity_gate = validate_hardware_identity(hardware)
    if not identity_gate["gate_pass"]:
        raise RuntimeError(f"5KP identity gate failed: {identity_gate}")
    return {
        "timestamp_unix": time.time(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nvcc": command_output(["nvcc", "--version"]),
        "ptxas": command_output(["ptxas", "--version"]),
        "gpu": {**hardware, "foreign_processes_before_cuda_context": foreign},
        "hardware_gate": identity_gate,
        "image_digest": os.environ["W4A4_IMAGE_DIGEST"],
        "python_deps_sha256": os.environ["W4A4_PYTHON_DEPS_SHA256"],
        "lease_id": os.environ["KDK_LEASE_ID"],
        "lease_gpu_uuid": os.environ["KDK_LEASE_GPU_UUID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "jit_root": str(args.jit_root),
        "source": dict(source),
    }


def make_case():
    fixture_module = load_fixture_module()
    device = torch.device("cuda", 0)
    fixture = fixture_module.make_routed_fixture(M, device=device, seed=2026)
    weights = fixture_module.make_canonical_weights(device=device, seed=2026)
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

    def replay(
        self, *, output_sentinel: bool, reset_probe: bool
    ) -> tuple[torch.Tensor, float]:
        if self.graph is None or self.start is None or self.end is None:
            raise RuntimeError("CUDA graph is not captured")
        if output_sentinel:
            self.wrapper._moe_output.fill_(float("nan"))
        if reset_probe:
            workspace = self.wrapper._dynamic_workspace
            workspace.exp004_timing_ticks.fill_(SENTINEL)
            workspace.exp004_task_cta_z.fill_(SENTINEL)
        torch.cuda.synchronize()
        self.graph.replay()
        torch.cuda.synchronize()
        assert self.output is not None
        return self.output, float(self.start.elapsed_time(self.end))


def build_arm(fixture: Any, weights: Any) -> CapturedArm:
    from flashinfer.fused_moe.cute_dsl import B12xMoEWrapper

    values = weights.cutedsl()
    wrapper = B12xMoEWrapper(
        num_experts=E,
        top_k=TOPK,
        hidden_size=H,
        intermediate_size=I,
        use_cuda_graph=True,
        max_num_tokens=M,
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


def workspace_snapshot(
    wrapper: Any, fixture: Any
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    workspace = wrapper._dynamic_workspace
    if workspace is None:
        raise RuntimeError("case did not dispatch to dynamic workspace")
    task_tail = int(workspace.task_tail.item())
    names = (
        "row_counts",
        "expert_write_rows",
        "expert_tile_base",
        "task_ready",
        "task_expert",
        "task_m_tile",
        "task_slice_begin",
        "task_slice_count",
        "task_valid_rows",
    )
    tensors: dict[str, torch.Tensor] = {}
    for name in names:
        value = getattr(workspace, name).detach().cpu().clone()
        if name.startswith("task_"):
            value = value[:task_tail]
        tensors[name] = value
    scalars = {
        name: int(getattr(workspace, name).item())
        for name in (
            "pair_head",
            "producers_done_count",
            "all_work_published",
            "task_head",
            "task_tail",
            "barrier_count",
            "barrier_epoch",
        )
    }
    expected_rows = torch.bincount(fixture.topk_ids.flatten().long(), minlength=E).cpu()
    plain = {**{name: tensor.tolist() for name, tensor in tensors.items()}, **scalars}
    verification = verify_workspace_evidence(
        plain, expected_row_counts=expected_rows.tolist()
    )
    return tensors, {
        "schema": "exp004.workspace-evidence.v1",
        "task_capacity": int(workspace.task_capacity),
        "scalars": scalars,
        "tensor_sha256": {
            name: tensor_sha256(value) for name, value in tensors.items()
        },
        "expected_row_counts_sha256": tensor_sha256(expected_rows),
        "verification": verification,
    }


def task_descriptors(
    workspace_tensors: Mapping[str, torch.Tensor],
) -> list[dict[str, int]]:
    return [
        {
            "expert": int(workspace_tensors["task_expert"][slot]),
            "m_tile": int(workspace_tensors["task_m_tile"][slot]),
            "slice": int(workspace_tensors["task_slice_begin"][slot]),
            "valid_rows": int(workspace_tensors["task_valid_rows"][slot]),
        }
        for slot in range(len(workspace_tensors["task_expert"]))
    ]


def timing_snapshot(
    arm: CapturedArm,
    arm_name: str,
    workspace_tensors: Mapping[str, torch.Tensor],
    run_id: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    workspace = arm.wrapper._dynamic_workspace
    if arm_name == NORMAL:
        return {}, {"kind": "not_present", "gate_pass": True}
    ticks = workspace.exp004_timing_ticks.detach().cpu().clone()
    cta = workspace.exp004_task_cta_z.detach().cpu().clone()
    if ticks.numel() != timing_ticks_capacity(int(workspace.task_capacity)):
        raise RuntimeError("timing workspace capacity drift")
    if arm_name == MEASUREMENT_CONTROL:
        gate = validate_no_marker_buffer(
            ticks.tolist(), cta.tolist(), task_capacity=int(workspace.task_capacity)
        )
    elif arm_name == PROBE:
        _, gate = decode_probe_buffer(
            ticks.tolist(),
            cta.tolist(),
            run_id=run_id,
            task_tail=int(workspace.task_tail.item()),
            task_capacity=int(workspace.task_capacity),
            task_descriptors=task_descriptors(workspace_tensors),
            emit_rows=False,
        )
    else:
        raise AssertionError(arm_name)
    return {"timing_ticks": ticks, "task_cta_z": cta}, gate


def compile_identity() -> dict[str, Any]:
    dispatch = importlib.import_module(DISPATCH_MODULE)
    entries = list(dispatch._DYNAMIC_KERNEL_CACHE.items())
    if not entries:
        raise RuntimeError("dynamic kernel cache is empty")
    macs = sorted({int(value[1]) for _, value in entries})
    if macs != [MAX_ACTIVE_CLUSTERS]:
        raise RuntimeError(f"compiled MAC drift: {macs}")
    return {
        "cache_entries": len(entries),
        "max_active_clusters": macs,
        "cache_keys_sha256": canonical_sha256([repr(key) for key, _ in entries]),
        "compiled_types": sorted({type(value[0]).__name__ for _, value in entries}),
    }


def preparation_path(args: argparse.Namespace) -> Path:
    return args.results / "raw" / "preparation" / args.arm / "preparation.json"


def failed_preparation_path(args: argparse.Namespace) -> Path:
    return args.results / "raw" / "preparation" / args.arm / "preparation_failure.json"


def prepare(args: argparse.Namespace, runtime: Mapping[str, Any]) -> int:
    fixture_module, fixture, weights = make_case()
    reference = fixture_module.reference_moe_nvfp4(fixture, weights)
    arm = build_arm(fixture, weights)
    arm.eager()
    arm.capture()
    raw = preparation_path(args).parent
    raw.mkdir(parents=True, exist_ok=True)
    outputs = []
    workspace_evidence = []
    timing_gates = []
    for replay in range(2):
        output, elapsed_ms = arm.replay(
            output_sentinel=True, reset_probe=args.arm != NORMAL
        )
        output_cpu = output.detach().cpu().clone()
        tensors, workspace = workspace_snapshot(arm.wrapper, fixture)
        timing_tensors, timing_gate = timing_snapshot(
            arm, args.arm, tensors, f"prepare_{replay}"
        )
        torch.save(output_cpu, raw / f"output_{replay}.pt")
        torch.save(tensors, raw / f"workspace_{replay}.pt")
        if timing_tensors:
            torch.save(timing_tensors, raw / f"timing_{replay}.pt")
        write_json(raw / f"workspace_{replay}.json", workspace)
        metrics = tensor_error(output_cpu, reference.cpu())
        gate = correctness_gate(metrics)
        tensor_gate = output_contract(output, reference)
        sentinel_nan = int(torch.isnan(output).sum().item())
        outputs.append(
            {
                **metrics,
                "gate": gate,
                "output_contract": tensor_gate,
                "sentinel_nan_remaining": sentinel_nan,
                "event_elapsed_us": elapsed_ms * 1000.0,
                "output_sha256": tensor_sha256(output_cpu),
                "gpu_state_after": _gpu_query(runtime["gpu"]["uuid"]),
            }
        )
        workspace_evidence.append(workspace)
        timing_gates.append(timing_gate)
        failed_gates = []
        if not gate["gate_pass"]:
            failed_gates.append("reference_correctness")
        if not tensor_gate["gate_pass"] or sentinel_nan:
            failed_gates.append("output_contract")
        if not workspace["verification"]["gate_pass"]:
            failed_gates.append("workspace_contract")
        if not timing_gate["gate_pass"]:
            failed_gates.append("timing_event_contract")
        if failed_gates:
            artifacts = artifact_manifest(args.jit_root)
            cubins = [item for item in artifacts if item["path"].endswith(".cubin")]
            timing_failed = "timing_event_contract" in failed_gates
            payload = {
                "schema": "exp004.arm-preparation-failure.v1",
                "status": (
                    "failed_timing_event_gate"
                    if timing_failed
                    else "failed_preparation_gate"
                ),
                "arm": args.arm,
                "failed_replay": replay,
                "failure": {
                    "gate": timing_gate,
                    "failed_gates": failed_gates,
                },
                "runtime": dict(runtime),
                "case": {
                    "m": M,
                    "experts": E,
                    "hidden": H,
                    "intermediate_tp": I,
                    "topk": TOPK,
                },
                "fixture": fixture.manifest,
                "weights": weights.manifest,
                "reference_sha256": tensor_sha256(reference),
                "outputs": outputs,
                "workspace_gates": [
                    item["verification"] for item in workspace_evidence
                ],
                "timing_gates": timing_gates,
                "compile_identity": compile_identity(),
                "expected_launch": {
                    "grid": list(EXPECTED_GRID),
                    "block": list(EXPECTED_BLOCK),
                },
                "jit_artifacts": artifacts,
                "cubin_sha256": sorted(item["sha256"] for item in cubins),
                "jit_artifact_set_sha256": canonical_sha256(artifacts),
                "foreign_processes_after": require_no_foreign_process(runtime),
                "failed_timing_sha256": tensor_sha256(timing_tensors["timing_ticks"]),
                "failed_task_cta_sha256": tensor_sha256(timing_tensors["task_cta_z"]),
            }
            write_json(failed_preparation_path(args), payload)
            raise RuntimeError(f"prepare gates failed: {failed_gates}")
    artifacts = artifact_manifest(args.jit_root)
    cubins = [item for item in artifacts if item["path"].endswith(".cubin")]
    if not cubins:
        raise RuntimeError("fresh JIT produced no cubin")
    foreign_after = require_no_foreign_process(runtime)
    payload = {
        "schema": "exp004.arm-preparation.v1",
        "status": "complete",
        "arm": args.arm,
        "runtime": dict(runtime),
        "case": {"m": M, "experts": E, "hidden": H, "intermediate_tp": I, "topk": TOPK},
        "fixture": fixture.manifest,
        "weights": weights.manifest,
        "reference_sha256": tensor_sha256(reference),
        "outputs": outputs,
        "workspace_gates": [item["verification"] for item in workspace_evidence],
        "timing_gates": timing_gates,
        "compile_identity": compile_identity(),
        "expected_launch": {"grid": list(EXPECTED_GRID), "block": list(EXPECTED_BLOCK)},
        "jit_artifacts": artifacts,
        "cubin_sha256": sorted(item["sha256"] for item in cubins),
        "jit_artifact_set_sha256": canonical_sha256(artifacts),
        "foreign_processes_after": foreign_after,
    }
    write_json(preparation_path(args), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def require_preparation(
    args: argparse.Namespace, runtime: Mapping[str, Any]
) -> dict[str, Any]:
    value = read_json(preparation_path(args))
    if value.get("status") != "complete" or value.get("arm") != args.arm:
        raise RuntimeError("preparation identity drift")
    if (
        canonical_sha256(artifact_manifest(args.jit_root))
        != value["jit_artifact_set_sha256"]
    ):
        raise RuntimeError("JIT artifact set drift")
    for field in (
        "image_digest",
        "python_deps_sha256",
        "source",
        "jit_root",
        "imports",
    ):
        if runtime.get(field) != value["runtime"].get(field):
            raise RuntimeError(f"runtime identity drift at {field}")
    for field in (
        "uuid",
        "name",
        "pci_bus_id",
        "driver",
        "compute_capability",
        "sm_count",
    ):
        if runtime["gpu"].get(field) != value["runtime"]["gpu"].get(field):
            raise RuntimeError(f"GPU identity drift at {field}")
    return value


def capture_phases(args: argparse.Namespace, runtime: Mapping[str, Any]) -> int:
    prerequisite = require_preparation(args, runtime)
    if args.arm == NORMAL:
        raise RuntimeError(
            "normal_no_marker has no phase buffer; capture control/probe"
        )
    fixture_module, fixture, weights = make_case()
    if (
        fixture.manifest != prerequisite["fixture"]
        or weights.manifest != prerequisite["weights"]
    ):
        raise RuntimeError("capture fixture/weight drift")
    reference = fixture_module.reference_moe_nvfp4(fixture, weights)
    arm = build_arm(fixture, weights)
    arm.eager()
    arm.capture()
    for _ in range(args.warmup):
        arm.replay(output_sentinel=False, reset_probe=True)
    root = args.results / "raw" / "phase_capture" / args.arm
    root.mkdir(parents=True, exist_ok=False)
    manifest_runs = []
    for replay in range(MEASURED_REPLAYS):
        output, elapsed_ms = arm.replay(output_sentinel=True, reset_probe=True)
        output_cpu = output.detach().cpu().clone()
        tensors, workspace = workspace_snapshot(arm.wrapper, fixture)
        timing_tensors, timing_gate = timing_snapshot(
            arm, args.arm, tensors, f"run_{replay}"
        )
        metrics = tensor_error(output_cpu, reference.cpu())
        output_gate = correctness_gate(metrics)
        tensor_gate = output_contract(output, reference)
        run = root / f"run_{replay}"
        run.mkdir()
        torch.save(timing_tensors, run / "timing.pt")
        torch.save(tensors, run / "workspace.pt")
        write_json(run / "workspace.json", workspace)
        metadata = {
            "schema": "exp004.phase-capture-run.v1",
            "arm": args.arm,
            "run_id": f"run_{replay}",
            "event_elapsed_us": elapsed_ms * 1000.0,
            "output_sha256": tensor_sha256(output_cpu),
            "correctness": {
                **metrics,
                "gate": output_gate,
                "output_contract": tensor_gate,
            },
            "sentinel_nan_remaining": int(torch.isnan(output).sum().item()),
            "workspace_gate": workspace["verification"],
            "timing_gate": timing_gate,
            "timing_sha256": tensor_sha256(timing_tensors["timing_ticks"]),
            "task_cta_sha256": tensor_sha256(timing_tensors["task_cta_z"]),
            "jit_artifact_set_sha256": prerequisite["jit_artifact_set_sha256"],
            "runtime": dict(runtime),
            "gpu_state_after": _gpu_query(runtime["gpu"]["uuid"]),
            "foreign_processes_after": require_no_foreign_process(runtime),
        }
        if (
            not output_gate["gate_pass"]
            or not tensor_gate["gate_pass"]
            or metadata["sentinel_nan_remaining"]
            or not workspace["verification"]["gate_pass"]
            or not timing_gate["gate_pass"]
        ):
            raise RuntimeError(f"capture run failed: {metadata}")
        write_json(run / "metadata.json", metadata)
        manifest_runs.append(
            {
                "run_id": metadata["run_id"],
                "metadata": str((run / "metadata.json").relative_to(args.results)),
            }
        )
    write_json(
        root / "manifest.json",
        {
            "schema": "exp004.phase-capture-manifest.v1",
            "arm": args.arm,
            "runs": manifest_runs,
            "preparation": str(preparation_path(args).relative_to(args.results)),
            "preparation_sha256": file_sha256(preparation_path(args)),
        },
    )
    return 0


def profile(args: argparse.Namespace, runtime: Mapping[str, Any]) -> int:
    prerequisite = require_preparation(args, runtime)
    _, fixture, weights = make_case()
    arm = build_arm(fixture, weights)
    arm.eager()
    arm.capture()
    for _ in range(args.warmup):
        arm.replay(output_sentinel=False, reset_probe=args.arm != NORMAL)
    nvtx = f"exp004_{args.arm}_m8192_final_replay"
    torch.cuda.nvtx.range_push(nvtx)
    cudart = torch.cuda.cudart()
    try:
        if int(cudart.cudaProfilerStart()) != 0:
            raise RuntimeError("cudaProfilerStart failed")
        output, elapsed_ms = arm.replay(
            output_sentinel=False, reset_probe=args.arm != NORMAL
        )
        if int(cudart.cudaProfilerStop()) != 0:
            raise RuntimeError("cudaProfilerStop failed")
    finally:
        torch.cuda.nvtx.range_pop()
    path = args.results / "profile_targets" / args.arm / "target.json"
    if path.exists():
        raise FileExistsError(f"immutable profile target already exists: {path}")
    foreign_after = require_no_foreign_process(runtime)
    write_json(
        path,
        {
            "schema": "exp004.profile-target.v1",
            "status": "complete",
            "arm": args.arm,
            "nvtx_range": nvtx,
            "event_elapsed_us": elapsed_ms * 1000.0,
            "output_sha256": tensor_sha256(output),
            "expected_launch": {
                "grid": list(EXPECTED_GRID),
                "block": list(EXPECTED_BLOCK),
                "kernel": "MoEDynamicKernel",
            },
            "profiler_observed_launch": {"verification": "pending_profiler_artifact"},
            "runtime": dict(runtime),
            "jit_artifact_set_sha256": prerequisite["jit_artifact_set_sha256"],
            "gpu_state_after": _gpu_query(runtime["gpu"]["uuid"]),
            "foreign_processes_after": foreign_after,
        },
    )
    return 0


def capture_calibration(args: argparse.Namespace, runtime: Mapping[str, Any]) -> int:
    if args.arm != PROBE:
        raise RuntimeError("clock calibration is bound to probe_candidate identity")
    prerequisite = require_preparation(args, runtime)
    root = args.results / "raw" / "calibration"
    if root.exists():
        raise FileExistsError(f"immutable calibration already exists: {root}")
    build = root / "jit"
    build.mkdir(parents=True)

    from torch.utils.cpp_extension import load

    source_sha256 = file_sha256(CLOCK_CALIBRATION_SOURCE)
    module = load(
        name=f"exp004_clock_calibration_{source_sha256[:12]}",
        sources=[str(CLOCK_CALIBRATION_SOURCE)],
        build_directory=str(build),
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    for _ in range(args.warmup):
        module.clock_store_pairs(args.samples)
    torch.cuda.synchronize()

    rows = []
    all_deltas = []
    for replay in range(MEASURED_REPLAYS):
        pairs = module.clock_store_pairs(args.samples)
        torch.cuda.synchronize()
        pairs = pairs.detach().cpu().clone()
        deltas = pairs[:, :, 1] - pairs[:, :, 0]
        if bool(torch.any(deltas <= 0)):
            raise RuntimeError("clock calibration produced non-positive delta")
        torch.save(pairs, root / f"run_{replay}.pt")
        flat = deltas.flatten().double()
        row = {
            "run_id": f"run_{replay}",
            "samples_per_warp": args.samples,
            "warps": 5,
            "delta_tick_p50": float(torch.quantile(flat, 0.50).item()),
            "delta_tick_p95": float(torch.quantile(flat, 0.95).item()),
            "delta_tick_max": int(flat.max().item()),
            "pairs_sha256": tensor_sha256(pairs),
            "gpu_state_after": _gpu_query(runtime["gpu"]["uuid"]),
        }
        rows.append(row)
        all_deltas.append(flat)
    aggregate = torch.cat(all_deltas)
    foreign_after = _foreign_processes(runtime["gpu"]["uuid"])
    if foreign_after:
        raise RuntimeError(
            f"foreign process appeared during calibration: {foreign_after}"
        )
    payload = {
        "schema": "exp004.clock-calibration.v1",
        "status": "complete",
        "method": (
            "five lane-0 warp tracks; each delta contains one GMEM timestamp "
            "store plus the following clock64 read"
        ),
        "source": str(CLOCK_CALIBRATION_SOURCE),
        "source_sha256": source_sha256,
        "runs": rows,
        "aggregate": {
            "samples": int(aggregate.numel()),
            "delta_tick_p50": float(torch.quantile(aggregate, 0.50).item()),
            "delta_tick_p95": float(torch.quantile(aggregate, 0.95).item()),
            "delta_tick_max": int(aggregate.max().item()),
        },
        "runtime": dict(runtime),
        "probe_jit_artifact_set_sha256": prerequisite["jit_artifact_set_sha256"],
        "extension_artifacts": artifact_manifest(build),
        "foreign_processes_after": foreign_after,
    }
    write_json(root / "manifest.json", payload)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--arm", choices=ALL_ARMS, required=True)
    parser.add_argument("--kernel-overlay", type=Path, required=True)
    parser.add_argument("--dispatch-overlay", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    capture = subparsers.add_parser("capture-phases")
    capture.add_argument("--warmup", type=int, default=5)
    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--warmup", type=int, default=5)
    calibration_parser = subparsers.add_parser("capture-calibration")
    calibration_parser.add_argument("--warmup", type=int, default=3)
    calibration_parser.add_argument("--samples", type=int, default=4096)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.results = args.results.resolve()
    args.kernel_overlay = args.kernel_overlay.resolve()
    args.dispatch_overlay = args.dispatch_overlay.resolve()
    args.jit_root = args.jit_root.resolve()
    source = validate_source(args)
    if args.command == "prepare":
        require_empty_directory(args.jit_root)
        if preparation_path(args).exists():
            raise RuntimeError(
                f"immutable preparation exists: {preparation_path(args)}"
            )
    if str(args.flashinfer_root) not in sys.path:
        sys.path.insert(0, str(args.flashinfer_root))
    install_overlays(args.kernel_overlay, args.dispatch_overlay)
    imports = configure_source_checkout(args.flashinfer_root)
    if Path(imports["kernel_module"]) != args.kernel_overlay:
        raise RuntimeError("kernel module did not resolve to selected overlay")
    if Path(imports["dispatch_module"]) != args.dispatch_overlay:
        raise RuntimeError("dispatch module did not resolve to selected overlay")
    runtime = runtime_identity(args, source)
    runtime["imports"] = imports
    args.results.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        return prepare(args, runtime)
    if args.command == "capture-phases":
        return capture_phases(args, runtime)
    if args.command == "profile":
        return profile(args, runtime)
    if args.command == "capture-calibration":
        return capture_calibration(args, runtime)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
