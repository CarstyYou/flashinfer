#!/usr/bin/env python3
"""Profile one exact exp_018 Opt/Eric production CUDA-graph replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent
BENCH_ROOT = ROOT.parents[1]
REPO = ROOT.parents[3]
EXP018 = ROOT.parent / "exp_018_triton_opt_eric_benchmark"
EXP018_RUNNER = EXP018 / "run_arm.py"

ARMS = ("latest_opt_fp4", "eric_stage4_fp4")
M_VALUES = (1024, 8192)
WARMUP = 5
EXPECTED_GPU_UUID = "GPU-2fdb0b79-0ba7-f356-b714-6c461b71ce12"
EXPECTED_CLOCK_MHZ = 2377
EXPECTED_IMAGE = {
    "W4A4_IMAGE_ID": (
        "sha256:a4e056e1d34a5cc9387512ffa3abeed778e3dc7966633c5154d771705d8835ac"
    ),
    "W4A4_IMAGE_DIGEST": (
        "sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba"
    ),
    "W4A4_PYTHON_DEPS_SHA256": (
        "32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74"
    ),
}
SOURCE_RELATIVE = {
    "latest_opt_fp4": Path(".claude/w4a4_moe_bench/moe_dynamic_kernel_opt.py"),
    "eric_stage4_fp4": Path(
        ".claude/w4a4_moe_bench/moe_dyanmice_kernel_ab_stage4_compact.py"
    ),
}
EXPECTED_SOURCE_SHA256 = {
    "latest_opt_fp4": (
        "ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19"
    ),
    "eric_stage4_fp4": (
        "3a5000a990bb978b434f1c7dac621de25112d9f3cec4a5fdfab5f2970b0dc3b8"
    ),
}
EXPECTED_ADAPTER_SHA256 = (
    "98adfba7f4e0d00af24383a556e9c93088355539b50dc82480091225e0448120"
)
EXPECTED_CUBIN_SHA256 = {
    "latest_opt_fp4": (
        "e9b322e4c978c490adbe0a9bf0f9a183288c0ecddb1fd72e5a904c487be541f3"
    ),
    "eric_stage4_fp4": (
        "4c728f4ee6115f342f0b32e578ca4901abd7f35ac233035fb8eaf54fce3900b0"
    ),
}
EXPECTED_JIT_ARTIFACT_SET_SHA256 = {
    "latest_opt_fp4": (
        "4aa7f95202db57421e1099e68f923ab585fb3ac914bdcd117aa3aab03d78e7c9"
    ),
    "eric_stage4_fp4": (
        "fe6b597c157315a4dc2b071f5b3815dda5b3289dbd59c4c6aa3025512f6de3ed"
    ),
}
EXPECTED_SYMBOL = (
    "kernel_cutlass_kernel_flashinferfused_moecute_dslblackwell_sm12x"
    "moe_dynamic_kernelMoEDynamicKernel_object_at__tensorptrbf16gmemalign16"
    "o204820481_tensorptri32gmemo1_tensorptrf32gmemo1_tens_0"
)
EXPECTED_GRID = [1, 1, 110]
EXPECTED_BLOCK = {
    "latest_opt_fp4": [288, 1, 1],
    "eric_stage4_fp4": [160, 1, 1],
}
DISPATCH_RELATIVE = Path(
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py"
)
WRAPPER_RELATIVE = Path("flashinfer/fused_moe/cute_dsl/b12x_moe.py")
EXPECTED_DISPATCH_SHA256 = (
    "cba2d0966631a47a576747e8322b57116122f2c8e5e868f8efb3f5ea692391a4"
)
EXPECTED_WRAPPER_SHA256 = (
    "bcac806795c035decd0773f4f801d477e7ebf14c1d67c3e49eee42ee0579c0a4"
)
EXPECTED_FIXTURE_SHA256 = {
    1024: "0fa7e8a7d8d1d32172971f987d6f55b534aabf8d12a84a910d010cec25ba04a5",
    8192: "c113ecd5ddeff77154ddbd23fc3dc3c83f8ee822e880179ca5c16b1145372438",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise RuntimeError(f"immutable NCU target manifest exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def static_identity(flashinfer_root: Path, arm: str) -> dict[str, Any]:
    paths = {
        "source": flashinfer_root / SOURCE_RELATIVE[arm],
        "dispatch": flashinfer_root / DISPATCH_RELATIVE,
        "wrapper": flashinfer_root / WRAPPER_RELATIVE,
        "exp018_runner": EXP018_RUNNER,
    }
    expected = {
        "source": EXPECTED_SOURCE_SHA256[arm],
        "dispatch": EXPECTED_DISPATCH_SHA256,
        "wrapper": EXPECTED_WRAPPER_SHA256,
        "exp018_runner": (
            "1f2fc64e8db7adf6c95c56c4467e05eb47f4d82f54d4b3c2fdebf6b35026adc6"
        ),
    }
    observed = {name: file_sha256(path) for name, path in paths.items()}
    if observed != expected:
        raise RuntimeError(
            f"production source identity drift: {observed} != {expected}"
        )
    return {
        name: {"path": str(paths[name]), "sha256": observed[name]} for name in paths
    }


def load_exp018_runner() -> Any:
    name = "exp019_exact_exp018_runner"
    spec = importlib.util.spec_from_file_location(name, EXP018_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exp_018 runner: {EXP018_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def gate_output(
    benchmark: Any,
    args: argparse.Namespace,
    captured: Any,
    routed: Any,
    output: Any,
    reference: Any,
) -> dict[str, Any]:
    basic = benchmark.sanity(output, args.m, benchmark.fp4_worker.tensor_sha256)
    route = benchmark.workspace_gate(args, captured, routed)
    diagnostics = benchmark.nvfp4.output_diagnostics(output, reference)
    route_pass = bool(
        route.get("gate_pass", route.get("verification", {}).get("gate_pass"))
    )
    checks = {
        "sanity": bool(basic["pass"]),
        "full_oracle": bool(diagnostics["formal_pass"]),
        "route_task": route_pass,
    }
    return {
        "sanity": basic,
        "oracle": diagnostics,
        "workspace": route,
        "checks": checks,
        "gate_pass": all(checks.values()),
    }


def require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise RuntimeError(f"{label} drift: {observed!r} != {expected!r}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.fixture_dir = args.fixture_dir.resolve()
    args.jit_root = args.jit_root.resolve()
    args.output = args.output.resolve()
    args.block = 0
    args.jit_policy = "fresh"
    args.expected_jit_artifact_set_sha256 = None
    args.device_index = 0
    args.results = args.output.parent

    identity = static_identity(args.flashinfer_root, args.arm)
    for key, expected in EXPECTED_IMAGE.items():
        require_equal(key, os.environ.get(key), expected)

    benchmark = load_exp018_runner()
    fixtures = benchmark.fixture_identity(args.fixture_dir)
    require_equal(
        "selected fixture",
        fixtures["npz_sha256"][str(args.m)],
        EXPECTED_FIXTURE_SHA256[args.m],
    )
    benchmark.configure_jit(args)
    environment = benchmark.require_environment(args)
    telemetry_before = benchmark.telemetry(args.expected_gpu_uuid, gate_foreign=True)
    if telemetry_before["foreign_compute_processes"]:
        raise RuntimeError(
            f"foreign compute process present: "
            f"{telemetry_before['foreign_compute_processes']}"
        )
    runtime = benchmark.runtime(args, environment)
    context = benchmark.prepare(args)
    source = context["source_identity"]
    require_equal(
        "selected source", source["source_sha256"], EXPECTED_SOURCE_SHA256[args.arm]
    )
    expected_overlay = (
        EXPECTED_ADAPTER_SHA256
        if args.arm == "eric_stage4_fp4"
        else EXPECTED_SOURCE_SHA256[args.arm]
    )
    require_equal("selected overlay", source["overlay_sha256"], expected_overlay)

    import torch

    x, ids, routing, fixture = benchmark.persisted.load_fixture(
        args.fixture_dir, args.m, torch.device("cuda")
    )
    routed = benchmark.nvfp4.RoutedFixture(args.m, x, ids, routing, fixture)
    weights = context["weights"]
    reference = benchmark.nvfp4.reference_moe_nvfp4(routed, weights)
    weight_identity = benchmark.normalized_weight_identity(args.arm, weights)
    captured = benchmark.fp4_worker.build_w4a4_arm(
        m=args.m, fixture=routed, weights=weights
    )

    eager = gate_output(benchmark, args, captured, routed, captured.eager(), reference)
    if not eager["gate_pass"]:
        raise RuntimeError(f"{args.arm} M={args.m} eager correctness failed: {eager}")
    captured.capture()
    for _ in range(args.warmup):
        captured.replay()
    pre_output, _ = captured.replay(sentinel=True)
    pre_profile = gate_output(benchmark, args, captured, routed, pre_output, reference)
    if not pre_profile["gate_pass"]:
        raise RuntimeError(
            f"{args.arm} M={args.m} pre-profile correctness failed: {pre_profile}"
        )

    captured.wrapper._moe_output.fill_(float("nan"))
    torch.cuda.synchronize()
    nvtx_range = f"exp019_{args.arm}_m{args.m}_production_ncu"
    cudart = torch.cuda.cudart()
    started = False
    range_pushed = False
    body_error: BaseException | None = None
    output = None
    elapsed_ms = 0.0
    try:
        status = int(cudart.cudaProfilerStart())
        if status != 0:
            raise RuntimeError(f"cudaProfilerStart failed: {status}")
        started = True
        torch.cuda.nvtx.range_push(nvtx_range)
        range_pushed = True
        output, elapsed_ms = captured.replay()
    except BaseException as error:
        body_error = error
        raise
    finally:
        if range_pushed:
            torch.cuda.nvtx.range_pop()
        if started:
            torch.cuda.synchronize()
            status = int(cudart.cudaProfilerStop())
            if status != 0 and body_error is None:
                raise RuntimeError(f"cudaProfilerStop failed: {status}")

    assert output is not None
    post_profile = gate_output(benchmark, args, captured, routed, output, reference)
    if not post_profile["gate_pass"]:
        raise RuntimeError(
            f"{args.arm} M={args.m} post-profile correctness failed: {post_profile}"
        )

    jit = benchmark.jit_identity(args)
    require_equal(
        "production cubin", jit["cubin_sha256"], [EXPECTED_CUBIN_SHA256[args.arm]]
    )
    require_equal("production symbol", jit["symbols"], [EXPECTED_SYMBOL])
    # exp_018's artifact-set hash covers the complete six-M process cache and
    # therefore cannot equal this one-M profiler cache.  The loaded code
    # identity is closed by the exact production cubin + symbol above; retain
    # both artifact-set hashes as provenance instead of using a false gate.
    jit["exp018_artifact_set_sha256_reference"] = EXPECTED_JIT_ARTIFACT_SET_SHA256[
        args.arm
    ]
    telemetry_after = benchmark.telemetry(args.expected_gpu_uuid, gate_foreign=False)
    telemetry_gate = benchmark.telemetry_verdict(telemetry_before, telemetry_after)
    if not telemetry_gate["pass"]:
        raise RuntimeError(f"telemetry gate failed: {telemetry_gate}")

    payload = {
        "schema": "exp019.production-ncu-target.v1",
        "status": "complete",
        "arm": args.arm,
        "m": args.m,
        "case": {"E": 256, "H": 2048, "I_tp": 512, "topk": 8},
        "source_identity": {"locked_files": identity, "runtime": source},
        "harness_sources": benchmark.harness_source_manifest(args.arm),
        "fixture_identity": fixtures,
        "fixture_manifest": fixture,
        "weight_identity": weight_identity,
        "correctness": {
            "eager": eager,
            "pre_profile_graph": pre_profile,
            "post_profile_graph": post_profile,
            "gate_pass": True,
        },
        "profile": {
            "nvtx_range": nvtx_range,
            "warmup_graph_replays": args.warmup,
            "profiled_graph_replays": 1,
            "event_elapsed_us": elapsed_ms * 1000.0,
            "boundary": "one complete production CUDA Graph replay",
        },
        "launch_contract": {
            "grid": EXPECTED_GRID,
            "block": EXPECTED_BLOCK[args.arm],
            "symbol": EXPECTED_SYMBOL,
        },
        "jit_identity": jit,
        "compile_identity": benchmark.fp4_worker.dynamic_compile_identity(
            expected_max_active_clusters=110
        ),
        "runtime": runtime,
        "telemetry_before": telemetry_before,
        "telemetry_after": telemetry_after,
        "telemetry_gate": telemetry_gate,
    }
    write_json(args.output, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--m", type=int, choices=M_VALUES, required=True)
    parser.add_argument("--flashinfer-root", type=Path, default=REPO)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=EXP018.parent / "exp_001_backend_case_sweep/results/fixtures",
    )
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-gpu-uuid",
        choices=(EXPECTED_GPU_UUID,),
        default=EXPECTED_GPU_UUID,
    )
    parser.add_argument(
        "--expected-app-clock-mhz",
        type=int,
        choices=(EXPECTED_CLOCK_MHZ,),
        default=EXPECTED_CLOCK_MHZ,
    )
    parser.add_argument("--rerun-id", default=os.environ.get("W4A4_RERUN_ID", ""))
    parser.add_argument("--warmup", type=int, choices=(WARMUP,), default=WARMUP)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = run(parse_args(argv))
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "status": payload["status"],
                "arm": payload["arm"],
                "m": payload["m"],
                "cubin_sha256": payload["jit_identity"]["cubin_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
