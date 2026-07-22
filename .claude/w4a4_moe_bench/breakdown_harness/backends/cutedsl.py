"""Reusable CuteDSL W4A4 fused-MoE backend primitives.

This module owns no experiment identity, source pins, result paths, CLI, or
prepare/measure/profile protocol.  Experiments provide those policies and use
these leaves to install an overlay, build/capture the canonical arm, and inspect
the dynamic workspace.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from breakdown_harness.case import E, H, I, TOPK
from breakdown_harness.backends.cutedsl_workspace import verify_workspace_evidence


DEFAULT_TARGET_MODULE = (
    "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dynamic_kernel"
)


class ExactModuleOverlayFinder(importlib.abc.MetaPathFinder):
    """Resolve exactly one Python module to an immutable source overlay."""

    def __init__(self, target_module: str, overlay: Path):
        self.target_module = target_module
        self.overlay = overlay

    def find_spec(self, fullname: str, path=None, target=None):
        del path, target
        if fullname != self.target_module:
            return None
        return importlib.util.spec_from_file_location(fullname, self.overlay)


def install_overlay(
    overlay: Path, *, target_module: str = DEFAULT_TARGET_MODULE
) -> ExactModuleOverlayFinder:
    """Install an exact-module overlay before the target is imported."""
    if target_module in sys.modules:
        raise RuntimeError(f"target module imported before overlay: {target_module}")
    finder = ExactModuleOverlayFinder(target_module, overlay)
    sys.meta_path.insert(0, finder)
    return finder


def configure_source_checkout(
    repo: Path,
    *,
    target_module: str = DEFAULT_TARGET_MODULE,
    aot_dir_name: str = "aot_disabled_for_breakdown",
) -> dict[str, str]:
    """Point FlashInfer JIT includes at ``repo`` and import the target module."""
    flashinfer = importlib.import_module("flashinfer")
    imported_root = Path(flashinfer.__file__).resolve().parents[1]
    if imported_root != repo:
        raise RuntimeError(f"imported FlashInfer root {imported_root} != {repo}")
    from flashinfer.jit import env as jit_env

    jit_env.FLASHINFER_CSRC_DIR = repo / "csrc"
    jit_env.FLASHINFER_INCLUDE_DIR = repo / "include"
    jit_env.FLASHINFER_AOT_DIR = jit_env.FLASHINFER_WORKSPACE_DIR / aot_dir_name
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
    target = importlib.import_module(target_module)
    cutlass = importlib.import_module("cutlass")
    return {
        "flashinfer": str(Path(flashinfer.__file__).resolve()),
        "target_module": str(Path(target.__file__).resolve()),
        "cutlass_python": str(Path(cutlass.__file__).resolve()),
        "cutlass_python_version": str(getattr(cutlass, "__version__", "unknown")),
    }


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


def build_w4a4_arm(*, m: int, fixture: Any, weights: Any) -> CapturedArm:
    """Build the canonical FlashInfer W4A4 dynamic fused-MoE call."""
    from flashinfer.fused_moe.cute_dsl import B12xMoEWrapper

    values = weights.cutedsl()
    wrapper = B12xMoEWrapper(
        num_experts=E,
        top_k=TOPK,
        hidden_size=H,
        intermediate_size=I,
        use_cuda_graph=True,
        max_num_tokens=m,
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


def snapshot_dynamic_workspace(
    wrapper: Any,
    fixture: Any,
    *,
    num_cta_warps: int,
    schema: str = "breakdown-harness.cutedsl-workspace.v1",
    verifier: Callable[..., Mapping[str, Any]] = verify_workspace_evidence,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Snapshot and verify the canonical dynamic scheduler workspace."""
    workspace = wrapper._dynamic_workspace
    if workspace is None:
        raise RuntimeError("dynamic fused-MoE workspace is unavailable")
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
    verification = dict(
        verifier(
            plain,
            expected_row_counts=expected_rows.tolist(),
            num_cta_warps=num_cta_warps,
        )
    )
    summary = {
        "schema": schema,
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


def dynamic_compile_identity(*, expected_max_active_clusters: int) -> dict[str, Any]:
    """Inspect the populated dynamic-kernel cache after a launch."""
    dispatch = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch"
    )
    entries = list(dispatch._DYNAMIC_KERNEL_CACHE.items())
    if not entries:
        raise RuntimeError("dynamic kernel cache is empty after launch")
    clusters = sorted({int(value[1]) for _, value in entries})
    if clusters != [expected_max_active_clusters]:
        raise RuntimeError(
            "compiled max_active_clusters drift: "
            f"{clusters} != {[expected_max_active_clusters]}"
        )
    return {
        "dynamic_cache_entries": len(entries),
        "compiled_max_active_clusters": clusters,
        "compiled_object_types": sorted(
            {type(value[0]).__name__ for _, value in entries}
        ),
    }


def make_l2_flusher(device: torch.device, bytes_: int = 192 << 20):
    buffer = torch.empty((bytes_ + 3) // 4, dtype=torch.int32, device=device)
    state = 0

    def flush() -> None:
        nonlocal state
        state += 1
        buffer.fill_(state)
        torch.cuda.synchronize()

    flush()
    return flush, buffer.numel() * buffer.element_size()
