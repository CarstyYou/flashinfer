#!/usr/bin/env python3
"""One-process/one-arm GPU worker for exp_003.

The coordinator invokes this program in a new Python process for every arm.
The arm gets its own immutable source overlay and JIT root; no Python module or
JIT cache can leak between baseline and candidate.
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

from exp003_common import (
    ALL_ARMS,
    DEFAULT_RESULTS,
    E,
    EXPECTED_BLOCK,
    EXPECTED_CUTLASS_COMMIT,
    EXPECTED_FLASHINFER_COMMIT,
    EXPECTED_GRID,
    EXPECTED_IMAGE_DIGEST,
    EXPECTED_KERNEL_SHA256,
    EXPECTED_PYTHON_DEPS_SHA256,
    H,
    I,
    M,
    TARGET_MODULE,
    TARGET_RELATIVE_PATH,
    TOPK,
    artifact_manifest,
    canonical_sha256,
    file_sha256,
    require_clean_compiler_environment,
    require_empty_directory,
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


def source_diff(left: Path, right: Path) -> str:
    """Return a normal unified diff; exit status 1 means files differ."""
    try:
        completed = subprocess.run(
            ["diff", "-u", str(left), str(right)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as error:
        return f"ERROR: {error}"
    if completed.returncode not in (0, 1):
        return f"ERROR: diff exited {completed.returncode}: {completed.stdout.strip()}"
    return completed.stdout


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
        raise RuntimeError("target module imported before exp_003 overlay installation")
    sys.meta_path.insert(0, ExactModuleOverlayFinder(overlay))


def git(repo: Path, *args: str) -> str:
    return command_output(["git", "-c", f"safe.directory={repo}", *args], cwd=repo)


def validate_source(repo: Path, overlay: Path, arm: str) -> dict[str, Any]:
    production = repo / TARGET_RELATIVE_PATH
    cutlass = repo / "3rdparty/cutlass"
    if not production.is_file() or not overlay.is_file():
        raise RuntimeError("production kernel or arm overlay is missing")
    kernel_hash = file_sha256(production)
    if kernel_hash != EXPECTED_KERNEL_SHA256:
        raise RuntimeError(f"production kernel hash drift: {kernel_hash}")
    head = git(repo, "rev-parse", "HEAD")
    ancestor = git(
        repo, "merge-base", "--is-ancestor", EXPECTED_FLASHINFER_COMMIT, head
    )
    if ancestor.startswith("ERROR"):
        raise RuntimeError(
            f"locked source {EXPECTED_FLASHINFER_COMMIT} is not an ancestor of {head}"
        )
    cutlass_head = git(cutlass, "rev-parse", "HEAD")
    if cutlass_head != EXPECTED_CUTLASS_COMMIT:
        raise RuntimeError(f"CUTLASS commit drift: {cutlass_head}")
    overlay_hash = file_sha256(overlay)
    if arm == "baseline" and overlay_hash != kernel_hash:
        raise RuntimeError("baseline overlay is not byte-identical to production")
    if arm != "baseline" and overlay_hash == kernel_hash:
        raise RuntimeError(f"candidate arm {arm} is byte-identical to baseline")
    return {
        "locked_source_commit": EXPECTED_FLASHINFER_COMMIT,
        "checkout_head": head,
        "cutlass_commit": cutlass_head,
        "production_kernel": str(production),
        "production_kernel_sha256": kernel_hash,
        "overlay": str(overlay),
        "overlay_sha256": overlay_hash,
        "overlay_diff": source_diff(production, overlay),
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
        jit_env.FLASHINFER_WORKSPACE_DIR / "aot_disabled_for_exp003_spill"
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
    return {
        "flashinfer": str(Path(flashinfer.__file__).resolve()),
        "target_module": str(Path(target.__file__).resolve()),
    }


def load_fixture_module():
    spec = importlib.util.spec_from_file_location("exp003_spill_fixture", FIXTURE_PATH)
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
        raise RuntimeError(
            "CUTE_DSL_DUMP_DIR must be inside the per-arm JIT root"
        ) from error

    torch.cuda.set_device(args.device_index)
    visible_uuid = command_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]
    ).splitlines()[0]
    if visible_uuid != args.expected_gpu_uuid:
        raise RuntimeError(
            f"GPU UUID drift: {visible_uuid} != {args.expected_gpu_uuid}"
        )
    capability = list(torch.cuda.get_device_capability(args.device_index))
    if capability not in ([12, 0], [12, 1]):
        raise RuntimeError(f"exp_003 requires SM120/121, got {capability}")
    return {
        "timestamp_unix": time.time(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nvcc": command_output(["nvcc", "--version"]),
        "ptxas": command_output(["ptxas", "--version"]),
        "driver": command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "gpu": {
            "uuid": visible_uuid,
            "name": torch.cuda.get_device_name(args.device_index),
            "compute_capability": capability,
        },
        "image_digest": os.environ["W4A4_IMAGE_DIGEST"],
        "python_deps_sha256": os.environ["W4A4_PYTHON_DEPS_SHA256"],
        "lease_id": os.environ["KDK_LEASE_ID"],
        "jit_root": str(args.jit_root),
        "source": source,
    }


@dataclass
class CapturedArm:
    launch: Callable[[], torch.Tensor]
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

    def replay(self) -> tuple[torch.Tensor, float]:
        if self.graph is None or self.start is None or self.end is None:
            raise RuntimeError("graph is not captured")
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

    return CapturedArm(launch)


def make_case(args: argparse.Namespace):
    fixture_module = load_fixture_module()
    device = torch.device("cuda", args.device_index)
    fixture = fixture_module.make_routed_fixture(M, device=device, seed=args.seed)
    weights = fixture_module.make_canonical_weights(device=device, seed=args.seed)
    return fixture_module, fixture, weights


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


def prepare(args: argparse.Namespace, runtime: dict[str, Any]) -> int:
    fixture_module, fixture, weights = make_case(args)
    reference = fixture_module.reference_moe_nvfp4(fixture, weights)
    arm = build_arm(fixture, weights)
    arm.eager()
    arm.capture()
    output_0, _ = arm.replay()
    output_0 = output_0.clone()
    output_1, _ = arm.replay()
    output_1 = output_1.clone()
    diagnostics = [
        fixture_module.output_diagnostics(output, reference)
        for output in (output_0, output_1)
    ]
    if not all(value["formal_pass"] for value in diagnostics):
        raise RuntimeError("quant-aware independent reference gate failed")
    arm_dir = args.results / "arms" / args.arm
    raw_dir = args.results / "raw" / args.arm
    raw_dir.mkdir(parents=True, exist_ok=True)
    torch.save(output_0.detach().cpu(), raw_dir / "output_0.pt")
    torch.save(output_1.detach().cpu(), raw_dir / "output_1.pt")
    artifacts = artifact_manifest(args.jit_root)
    payload = {
        "schema": "exp003.spill-root-cause.arm-preparation.v1",
        "status": "complete",
        "arm": args.arm,
        "runtime": runtime,
        "case": {"m": M, "experts": E, "hidden": H, "intermediate_tp": I, "topk": TOPK},
        "fixture": fixture.manifest,
        "weights": weights.manifest,
        "reference_sha256": tensor_sha256(reference),
        "outputs": [
            {
                "path": str((raw_dir / f"output_{index}.pt").relative_to(args.results)),
                **value,
            }
            for index, value in enumerate(diagnostics)
        ],
        "launch": {"grid": list(EXPECTED_GRID), "block": list(EXPECTED_BLOCK)},
        "jit_artifacts": artifacts,
        "jit_artifact_set_sha256": canonical_sha256(artifacts),
    }
    write_json(arm_dir / "preparation.json", payload)
    return 0


def require_preparation(
    args: argparse.Namespace, runtime: dict[str, Any]
) -> dict[str, Any]:
    path = args.results / "arms" / args.arm / "preparation.json"
    if not path.is_file():
        raise RuntimeError(f"arm preparation is missing: {path}")
    value = json.loads(path.read_text())
    if value.get("status") != "complete" or value.get("arm") != args.arm:
        raise RuntimeError("arm preparation did not complete")
    before = artifact_manifest(args.jit_root)
    if canonical_sha256(before) != value.get("jit_artifact_set_sha256"):
        raise RuntimeError("per-arm JIT artifact identity drift before replay")
    stable_fields = ("gpu", "image_digest", "python_deps_sha256", "source", "jit_root")
    for field in stable_fields:
        if runtime.get(field) != value["runtime"].get(field):
            raise RuntimeError(f"runtime identity drift at {field}")
    return value


def measure(args: argparse.Namespace, runtime: dict[str, Any]) -> int:
    prerequisite = require_preparation(args, runtime)
    _, fixture, weights = make_case(args)
    if fixture.manifest != prerequisite.get("fixture"):
        raise RuntimeError("paired benchmark fixture identity drift")
    if weights.manifest != prerequisite.get("weights"):
        raise RuntimeError("paired benchmark weight/scale identity drift")
    arm = build_arm(fixture, weights)
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
    after = artifact_manifest(args.jit_root)
    before_hash = json.loads(
        (args.results / "arms" / args.arm / "preparation.json").read_text()
    )["jit_artifact_set_sha256"]
    if canonical_sha256(after) != before_hash:
        raise RuntimeError("per-arm JIT artifact identity drift after replay")
    payload = {
        "schema": "exp003.spill-root-cause.arm-measurement.v1",
        "status": "complete",
        "arm": args.arm,
        "repeat": args.repeat,
        "order": args.order,
        "sample_us": total_ms * 1000.0 / args.iters,
        "warmup": args.warmup,
        "iters": args.iters,
        "l2_flush_bytes": flush_bytes,
        "timing": "outer CUDA graph with external CUDA events",
        "runtime": runtime,
        "jit_artifact_set_sha256": before_hash,
    }
    output = (
        args.results / "raw" / "benchmark" / f"repeat_{args.repeat}_{args.arm}.json"
    )
    write_json(output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def profile(args: argparse.Namespace, runtime: dict[str, Any]) -> int:
    prerequisite = require_preparation(args, runtime)
    _, fixture, weights = make_case(args)
    if fixture.manifest != prerequisite.get("fixture"):
        raise RuntimeError("NCU fixture identity drift")
    if weights.manifest != prerequisite.get("weights"):
        raise RuntimeError("NCU weight/scale identity drift")
    arm = build_arm(fixture, weights)
    arm.eager()
    arm.capture()
    for _ in range(args.warmup):
        arm.replay()
    nvtx = f"exp003_spill_{args.arm}_fused_main"
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
        "schema": "exp003.spill-root-cause.profile-target.v1",
        "status": "complete",
        "arm": args.arm,
        "nvtx_range": nvtx,
        "event_elapsed_us": elapsed_ms * 1000.0,
        "output_sha256": tensor_sha256(output),
        "runtime": runtime,
        "jit_artifact_set_sha256": canonical_sha256(artifact_manifest(args.jit_root)),
    }
    write_json(args.results / "arms" / args.arm / "profile_target.json", payload)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--arm", choices=ALL_ARMS, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--device-index", type=int, default=0, choices=[0])
    parser.add_argument("--seed", type=int, default=2026, choices=[2026])
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--repeat", type=int, choices=range(5), required=True)
    measure_parser.add_argument("--order", required=True)
    measure_parser.add_argument("--warmup", type=int, default=5)
    measure_parser.add_argument("--iters", type=int, default=50)
    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.results = args.results.resolve()
    args.overlay = args.overlay.resolve()
    args.jit_root = args.jit_root.resolve()
    source = validate_source(args.flashinfer_root, args.overlay, args.arm)
    # This gate must run before importing even the FlashInfer package: import
    # initialization may populate the workspace and would make a stale cache
    # look like fresh JIT output.
    if args.command == "prepare":
        require_empty_directory(args.jit_root)
    if str(args.flashinfer_root) not in sys.path:
        sys.path.insert(0, str(args.flashinfer_root))
    install_overlay(args.overlay)
    imports = configure_source_checkout(args.flashinfer_root)
    if Path(imports["target_module"]) != args.overlay:
        raise RuntimeError("target module did not resolve to the selected arm overlay")
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
