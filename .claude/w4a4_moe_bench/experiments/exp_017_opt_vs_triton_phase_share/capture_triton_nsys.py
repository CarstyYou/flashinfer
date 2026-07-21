#!/usr/bin/env python3
"""Capture five NVTX-scoped SGLang Triton FP8 CUDA Graph replays with NSys."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
EXP001 = ROOT.parent / "exp_001_backend_case_sweep"
if str(EXP001) not in sys.path:
    sys.path.insert(0, str(EXP001))

M = 8192
REPLAYS = 5
EXPECTED_APP_CLOCK_MHZ = 2377
EXPECTED_TRITON_CONFIG = {
    "BLOCK_SIZE_K": 256,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "GROUP_SIZE_M": 64,
    "num_stages": 2,
    "num_warps": 4,
}
DEFAULT_FIXTURES = EXP001 / "results" / "fixtures"
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_TOPOLOGY_PREFLIGHT = DEFAULT_RESULTS / "triton_topology_preflight.json"

# Repeated kernel names are intentionally distinguished by ordinal position.
# This is the expected CUDA Graph kernel-node topology for the locked M8192 arm.
EXPECTED_TOPOLOGY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("route_align", ("moe_align_block_size_kernel",)),
    ("route_count_sort", ("count_and_sort_expert_tokens_kernel",)),
    ("q0_fill", ("fillfunctor<float>",)),
    ("q0_absmax", ("per_tensor_absmax_kernel",)),
    ("q0_quant", ("per_tensor_quant_fp8_kernel",)),
    ("fc1", ("fused_moe_kernel",)),
    ("swiglu", ("act_and_mul_kernel",)),
    ("q1_fill", ("fillfunctor<float>",)),
    ("q1_absmax", ("per_tensor_absmax_kernel",)),
    ("q1_quant", ("per_tensor_quant_fp8_kernel",)),
    ("fc2", ("fused_moe_kernel",)),
    ("topk_reduce", ("moe_sum_reduce",)),
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_graph_topology(kernel_names: Sequence[str]) -> dict[str, Any]:
    """Fail closed unless the observed graph has the canonical 12 kernels."""
    if len(kernel_names) != len(EXPECTED_TOPOLOGY):
        raise RuntimeError(
            f"graph topology drift: observed {len(kernel_names)} CUDA kernels, "
            f"expected {len(EXPECTED_TOPOLOGY)}"
        )
    forbidden = [
        name
        for name in kernel_names
        if "cutlass" in name.lower() or "deepgemm" in name.lower()
    ]
    if forbidden:
        raise RuntimeError(f"alternate backend kernels observed: {forbidden}")

    nodes: list[dict[str, Any]] = []
    for ordinal, (name, (role, tokens)) in enumerate(
        zip(kernel_names, EXPECTED_TOPOLOGY, strict=True), start=1
    ):
        lowered = name.lower()
        if not all(token in lowered for token in tokens):
            raise RuntimeError(
                f"graph topology drift at node {ordinal} ({role}): {name!r} "
                f"does not contain {tokens}"
            )
        nodes.append({"ordinal": ordinal, "role": role, "kernel_name": name})

    payload = {
        "schema": "exp017.sglang-triton-graph-topology.v1",
        "node_count": len(nodes),
        "nodes": nodes,
        "identity_rule": (
            "graph execution ordinal plus kernel name; repeated fused_moe/fill/"
            "absmax/quant names are not grouped by name alone"
        ),
    }
    return {**payload, "fingerprint_sha256": canonical_sha256(payload)}


def validate_resolved_config(contract: dict[str, Any]) -> None:
    config = contract.get("config", {})
    if config.get("up") != EXPECTED_TRITON_CONFIG:
        raise RuntimeError(
            f"M8192 Triton config drift: {config.get('up')} != {EXPECTED_TRITON_CONFIG}"
        )
    if config.get("down") is not None or config.get("down_max_block_m") is not None:
        raise RuntimeError(f"unexpected independent FC2 config: {config}")
    if contract.get("backend") != "sglang_legacy_triton_fp8_chain":
        raise RuntimeError(f"backend dispatch drift: {contract.get('backend')}")


def capture_replays(
    *,
    replays: int,
    flush_l2: Callable[[], None],
    synchronize: Callable[[], None],
    graph_replay: Callable[[], None],
    nvtx_push: Callable[[str], None],
    nvtx_pop: Callable[[], None],
    profiler_start: Callable[[], int],
    profiler_stop: Callable[[], int],
    validate_output_at: Callable[[int], None],
) -> list[str]:
    """Execute the capture protocol; kept dependency-free for CPU mock tests."""
    if replays != REPLAYS:
        raise ValueError(f"canonical capture requires exactly {REPLAYS} replays")
    labels = [f"exp017_sglang_triton_fp8_m{M}_replay_{i:02d}" for i in range(1, 6)]
    profiler_started = False
    body_error: BaseException | None = None
    try:
        for index, label in enumerate(labels):
            # The flush is outside the NVTX replay range. The first flush also
            # precedes cudaProfilerStart; later flushes remain outside all five
            # replay ranges and are excluded by NVTX-scoped analysis.
            flush_l2()
            synchronize()
            if index == 0:
                status = int(profiler_start())
                if status != 0:
                    raise RuntimeError(f"cudaProfilerStart failed: {status}")
                profiler_started = True
            nvtx_push(label)
            try:
                graph_replay()
                synchronize()
            finally:
                nvtx_pop()
            if index in (0, replays - 1):
                validate_output_at(index)
                synchronize()
    except BaseException as error:
        body_error = error
        raise
    finally:
        if profiler_started:
            # In particular, the fifth replay and its correctness gate are
            # complete before cudaProfilerStop closes the NSys capture.
            synchronize()
            status = int(profiler_stop())
            if status != 0 and body_error is None:
                raise RuntimeError(f"cudaProfilerStop failed: {status}")
    return labels


def observe_graph_kernels(torch: Any, graph: Any) -> list[str]:
    """Observe one already-warmed graph replay outside the NSys capture."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        graph.replay()
        torch.cuda.synchronize()
    names = [
        event.name
        for event in trace.events()
        if "cuda" in str(getattr(event, "device_type", "")).lower()
    ]
    if not names:
        raise RuntimeError("PyTorch profiler observed no graph CUDA kernels")
    return names


def capture_graph(torch: Any, launch: Callable[[], Any]) -> tuple[Any, Any]:
    """Create a graph without adding timing-event nodes to its topology."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = launch()
    torch.cuda.synchronize()
    return graph, output


def checked_replay_correctness(
    *,
    output: Any,
    reference: Any,
    validate_output: Callable[[Any, int], None],
    diagnostics: Callable[[Any, Any], dict[str, Any]],
) -> dict[str, Any]:
    validate_output(output, M)
    result = diagnostics(output, reference)
    if not result.get("formal_pass", False):
        raise RuntimeError(f"replay correctness gate failed: {result}")
    return result


def query_application_clock(command_output: Callable[[list[str]], str]) -> int:
    raw = command_output(
        [
            "nvidia-smi",
            "--query-gpu=clocks.applications.graphics",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        return int(raw.splitlines()[0].strip())
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"could not parse application clock: {raw!r}") from error


def build_source_lock() -> dict[str, Any]:
    return {
        "capture_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "exp001_benchmark": {
            "path": str((EXP001 / "bench_triton_fp8.py").resolve()),
            "sha256": file_sha256(EXP001 / "bench_triton_fp8.py"),
        },
        "exp001_fixture": {
            "path": str((EXP001 / "fixture.py").resolve()),
            "sha256": file_sha256(EXP001 / "fixture.py"),
        },
    }


def stable_runtime_identity(
    runtime: dict[str, Any], application_clock_mhz: int
) -> dict[str, Any]:
    # Separate docker invocations have different container hostnames. Every
    # material runtime property remains locked, while timestamp/hostname are
    # deliberately excluded from the cross-process preflight gate.
    identity = {
        key: value
        for key, value in runtime.items()
        if key not in {"timestamp_unix", "hostname"}
    }
    identity["application_clock_mhz"] = application_clock_mhz
    return {**identity, "fingerprint_sha256": canonical_sha256(identity)}


def make_topology_preflight(
    *,
    source_lock: dict[str, Any],
    runtime_identity: dict[str, Any],
    fixture: dict[str, Any],
    weights: dict[str, Any],
    launch_contract: dict[str, Any],
    graph_topology: dict[str, Any],
    artifact_fingerprint_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema": "exp017.sglang-triton-topology-preflight.v1",
        "status": "complete",
        "collection": "torch.profiler outside NSys; no concurrent CUPTI subscriber",
        "source_lock": source_lock,
        "runtime_identity": runtime_identity,
        "fixture": fixture,
        "weights": weights,
        "launch_contract": launch_contract,
        "graph_topology": graph_topology,
        "artifact_fingerprint_sha256": artifact_fingerprint_sha256,
    }
    return {**payload, "fingerprint_sha256": canonical_sha256(payload)}


def validate_topology_preflight(
    preflight: dict[str, Any],
    *,
    source_lock: dict[str, Any],
    runtime_identity: dict[str, Any],
    fixture: dict[str, Any],
    weights: dict[str, Any],
    launch_contract: dict[str, Any],
    artifact_fingerprint_sha256: str,
) -> dict[str, Any]:
    if preflight.get("schema") != "exp017.sglang-triton-topology-preflight.v1":
        raise RuntimeError("topology preflight schema drift")
    if preflight.get("status") != "complete":
        raise RuntimeError("topology preflight is incomplete")
    fingerprint = preflight.get("fingerprint_sha256")
    payload = {
        key: value for key, value in preflight.items() if key != "fingerprint_sha256"
    }
    if fingerprint != canonical_sha256(payload):
        raise RuntimeError("topology preflight fingerprint mismatch")

    expected = {
        "source_lock": source_lock,
        "runtime_identity": runtime_identity,
        "fixture": fixture,
        "weights": weights,
        "launch_contract": launch_contract,
        "artifact_fingerprint_sha256": artifact_fingerprint_sha256,
    }
    for field, value in expected.items():
        if preflight.get(field) != value:
            raise RuntimeError(f"topology preflight {field} drift")

    stored_topology = preflight.get("graph_topology")
    if not isinstance(stored_topology, dict):
        raise RuntimeError("topology preflight has no graph topology")
    names = [node.get("kernel_name", "") for node in stored_topology.get("nodes", [])]
    rebuilt_topology = validate_graph_topology(names)
    if stored_topology != rebuilt_topology:
        raise RuntimeError("topology preflight graph fingerprint drift")
    return stored_topology


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument(
        "--expected-app-clock-mhz", type=int, default=EXPECTED_APP_CLOCK_MHZ
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--replays", type=int, default=REPLAYS)
    parser.add_argument("--l2-flush-bytes", type=int, default=192 << 20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--topology-preflight", type=Path, default=DEFAULT_TOPOLOGY_PREFLIGHT
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="observe graph topology outside NSys, write its lock, then exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.replays != REPLAYS:
        raise RuntimeError(f"canonical capture requires exactly {REPLAYS} replays")
    manifest_path = args.results / "triton_capture_manifest.json"
    topology_preflight_path = args.topology_preflight.resolve()
    if args.preflight_only and topology_preflight_path.exists():
        raise RuntimeError(
            f"refusing to overwrite topology preflight: {topology_preflight_path}"
        )
    if not args.preflight_only and manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite completed evidence: {manifest_path}")
    if not args.preflight_only and not topology_preflight_path.is_file():
        raise RuntimeError(
            "formal NSys capture requires an out-of-process topology preflight: "
            f"{topology_preflight_path}"
        )

    import torch
    from bench_triton_fp8 import (
        artifact_manifest,
        build_launch,
        command_output,
        fp8_oracle,
        make_fp8_weights,
        make_l2_flusher,
        output_diagnostics,
        runtime_manifest,
    )
    from fixture import load_fixture, validate_output

    runtime = runtime_manifest(args.expected_gpu_uuid)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    application_clock_mhz = query_application_clock(command_output)
    if application_clock_mhz != args.expected_app_clock_mhz:
        raise RuntimeError(
            f"application clock drift: {application_clock_mhz} != "
            f"{args.expected_app_clock_mhz} MHz"
        )

    weights = make_fp8_weights(device=device, seed=args.seed)
    x, topk_ids, topk_weights, fixture = load_fixture(args.fixture_dir, M, device)
    launch, contract = build_launch(x, topk_ids, topk_weights, weights)
    validate_resolved_config(contract)

    eager_output = launch()
    torch.cuda.synchronize()
    validate_output(eager_output, M)
    graph, graph_output = capture_graph(torch, launch)
    artifacts_before = artifact_manifest(weights)
    source_lock = build_source_lock()
    runtime_identity = stable_runtime_identity(runtime, application_clock_mhz)

    if args.preflight_only:
        # This is the only path that imports torch.profiler. The standalone
        # process exits before NSys starts, avoiding concurrent CUPTI owners.
        kernel_names = observe_graph_kernels(torch, graph)
        topology = validate_graph_topology(kernel_names)
        preflight = make_topology_preflight(
            source_lock=source_lock,
            runtime_identity=runtime_identity,
            fixture=fixture,
            weights=weights.manifest,
            launch_contract=contract,
            graph_topology=topology,
            artifact_fingerprint_sha256=artifacts_before["fingerprint_sha256"],
        )
        write_json(topology_preflight_path, preflight)
        print(
            json.dumps(
                {
                    "status": "topology_preflight_complete",
                    "node_count": topology["node_count"],
                    "fingerprint_sha256": preflight["fingerprint_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0

    # Formal NSys capture must not instantiate torch.profiler/CUPTI. It reads
    # the standalone topology lock and revalidates every material identity.
    preflight = json.loads(topology_preflight_path.read_text())
    topology = validate_topology_preflight(
        preflight,
        source_lock=source_lock,
        runtime_identity=runtime_identity,
        fixture=fixture,
        weights=weights.manifest,
        launch_contract=contract,
        artifact_fingerprint_sha256=artifacts_before["fingerprint_sha256"],
    )

    reference = fp8_oracle(x, topk_ids, topk_weights, weights)
    eager_correctness = checked_replay_correctness(
        output=eager_output,
        reference=reference,
        validate_output=validate_output,
        diagnostics=output_diagnostics,
    )
    flush_l2, actual_flush_bytes = make_l2_flusher(device, args.l2_flush_bytes)
    for _ in range(args.warmup):
        flush_l2()
        graph.replay()
        torch.cuda.synchronize()

    protocol = {
        "schema": "exp017.sglang-triton-nsys-protocol.v1",
        "m": M,
        "warmup_graph_replays": args.warmup,
        "captured_graph_replays": args.replays,
        "l2_flush_bytes": actual_flush_bytes,
        "order_per_replay": "L2 flush -> CUDA sync -> unique NVTX push -> graph replay -> CUDA sync -> NVTX pop",
        "correctness_replays": [1, 5],
        "capture_range": "cudaProfilerStart/Stop",
        "nsys_graph_mode": "node:host-only",
        "analysis_interface": "VeloQ only; no direct sqlite/parquet queries",
    }
    protocol["fingerprint_sha256"] = canonical_sha256(protocol)
    manifest: dict[str, Any] = {
        "schema": "exp017.sglang-triton-nsys-capture.v1",
        "status": "ready",
        "source_lock": source_lock,
        "runtime_identity": runtime_identity,
        "fixture": fixture,
        "weights": weights.manifest,
        "launch_contract": contract,
        "expected_graph_topology": topology,
        "topology_preflight": {
            "path": str(topology_preflight_path),
            "fingerprint_sha256": preflight["fingerprint_sha256"],
            "collection": "standalone torch.profiler process outside NSys",
        },
        "post_capture_topology_gate": {
            "status": "pending_veloq",
            "requirement": (
                "each NVTX replay must contain the same ordered 12 CUDA graph "
                "kernel nodes"
            ),
        },
        "protocol": protocol,
        "artifact_fingerprint_sha256": artifacts_before["fingerprint_sha256"],
        "eager_correctness": eager_correctness,
    }
    write_json(args.results / "triton_artifact.lock.json", artifacts_before)
    write_json(manifest_path, manifest)

    replay_correctness: dict[str, Any] = {}

    def validate_replay(index: int) -> None:
        replay_correctness[str(index + 1)] = checked_replay_correctness(
            output=graph_output,
            reference=reference,
            validate_output=validate_output,
            diagnostics=output_diagnostics,
        )

    cudart = torch.cuda.cudart()
    labels = capture_replays(
        replays=args.replays,
        flush_l2=flush_l2,
        synchronize=torch.cuda.synchronize,
        graph_replay=graph.replay,
        nvtx_push=torch.cuda.nvtx.range_push,
        nvtx_pop=torch.cuda.nvtx.range_pop,
        profiler_start=cudart.cudaProfilerStart,
        profiler_stop=cudart.cudaProfilerStop,
        validate_output_at=validate_replay,
    )

    artifacts_after = artifact_manifest(weights)
    if artifacts_after["fingerprint_sha256"] != artifacts_before["fingerprint_sha256"]:
        raise RuntimeError("SGLang source/JIT artifact drift during capture")
    manifest.update(
        {
            "status": "capture_complete_topology_pending_veloq",
            "nvtx_ranges": labels,
            "replay_correctness": replay_correctness,
            "artifact_stable_during_capture": True,
        }
    )
    manifest["fingerprint_sha256"] = canonical_sha256(manifest)
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "capture_complete_topology_pending_veloq",
                "nvtx_ranges": labels,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
