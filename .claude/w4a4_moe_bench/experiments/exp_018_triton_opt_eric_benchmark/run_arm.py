#!/usr/bin/env python3
"""Run one identity-locked exp_018 arm/block without profiler instrumentation."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence
import xml.etree.ElementTree as ET

import torch


ROOT = Path(__file__).resolve().parent
BENCH_ROOT = ROOT.parents[1]
REPO = ROOT.parents[3]
EXP001 = ROOT.parent / "exp_001_backend_case_sweep"
EXP005 = ROOT.parent / "exp_005_8warp_spill_reduction"
EXP009 = ROOT.parent / "exp_009_intern_stage4_compact_lightcheck"
for dependency in (EXP001, EXP005, EXP009):
    sys.path.insert(0, str(dependency))

import fixture as persisted  # noqa: E402
import nvfp4_fixture as nvfp4  # noqa: E402
import bench_triton_fp8 as triton  # noqa: E402
import exp005_common as common  # noqa: E402
import run_exp005_arm as fp4_worker  # noqa: E402
import build_adapter as eric_adapter  # noqa: E402


ARMS = ("latest_opt_fp4", "eric_stage4_fp4", "sglang_triton_fp8")
M_VALUES = (256, 512, 1024, 2048, 4096, 8192)
BLOCK_ORDERS = (
    ("latest_opt_fp4", "eric_stage4_fp4", "sglang_triton_fp8"),
    ("eric_stage4_fp4", "sglang_triton_fp8", "latest_opt_fp4"),
    ("sglang_triton_fp8", "latest_opt_fp4", "eric_stage4_fp4"),
)
WARMUP, ITERS, FLUSH_BYTES, CLOCK_MHZ, SEED = 5, 50, 192 << 20, 2377, 2026
TARGET_MODULE = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dynamic_kernel"
FIXTURE_DIR = EXP001 / "results/fixtures"
SOURCES = {
    "latest_opt_fp4": BENCH_ROOT / "moe_dynamic_kernel_opt.py",
    "eric_stage4_fp4": BENCH_ROOT / "moe_dyanmice_kernel_ab_stage4_compact.py",
}
SOURCE_SHA = {
    "latest_opt_fp4": "ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19",
    "eric_stage4_fp4": "3a5000a990bb978b434f1c7dac621de25112d9f3cec4a5fdfab5f2970b0dc3b8",
}
ERIC_ADAPTER_SHA = "98adfba7f4e0d00af24383a556e9c93088355539b50dc82480091225e0448120"
BLOCK_THREADS = {"latest_opt_fp4": 288, "eric_stage4_fp4": 160}
FIXTURE_MANIFEST_SHA = (
    "683ec75341e4d8317dfdc5c4b04229f9695f9aa286d575c4f6e1fdef55d90801"
)
FIXTURE_SHA = {
    256: "86b505097acd06bed5a50c3528c78525e6087c07ed69f86606607599ffa21686",
    512: "e6ddb487121a0d681a06bcb453f38623b3d5d8477f2232bbbf78cd2ea4ef23a3",
    1024: "0fa7e8a7d8d1d32172971f987d6f55b534aabf8d12a84a910d010cec25ba04a5",
    2048: "5375fd8b3e5e15f8c956998bfc3e2f3ee59948a2aeaf5ba1294ec6a74092bde3",
    4096: "a1ac93cb8dfb2e81a000476efc36b75588f79f1954b406396b77c172464ce2cc",
    8192: "c113ecd5ddeff77154ddbd23fc3dc3c83f8ee822e880179ca5c16b1145372438",
}
FP4_ENV = {
    "W4A4_IMAGE_ID": "sha256:a4e056e1d34a5cc9387512ffa3abeed778e3dc7966633c5154d771705d8835ac",
    "W4A4_IMAGE_DIGEST": "sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba",
    "W4A4_PYTHON_DEPS_SHA256": "32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74",
}
TRITON_ENV = {
    "W4A4_IMAGE_ID": "sha256:663867442f321ded36228bafd889fd1db05cbef7a7c8ea6e072df33234dabbfd",
    "W4A4_IMAGE_DIGEST": "sha256:00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2",
    "W4A4_SGLANG_COMMIT": "0b3bb0cbe31873994c9f989fddfe2f87ca839fdd",
}


def command(
    command: Sequence[str], *, optional: bool = False, cwd: Path | None = None
) -> str:
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        if optional:
            return f"unavailable: {error}"
        raise


def fixture_identity(root: Path) -> dict[str, Any]:
    manifest = root / "manifest.json"
    if triton.file_sha256(manifest) != FIXTURE_MANIFEST_SHA:
        raise RuntimeError("fixture manifest drift")
    for m, expected in FIXTURE_SHA.items():
        if triton.file_sha256(root / f"m{m}.npz") != expected:
            raise RuntimeError(f"M={m} fixture drift")
    return {
        "manifest_path": str(manifest),
        "manifest_sha256": FIXTURE_MANIFEST_SHA,
        "seed": SEED,
        "npz_sha256": {str(m): value for m, value in FIXTURE_SHA.items()},
    }


def configure_jit(args: argparse.Namespace) -> str | None:
    args.jit_root.mkdir(parents=True, exist_ok=True)
    before = common.artifact_manifest(args.jit_root)
    before_hash = common.canonical_sha256(before)
    if args.jit_policy == "fresh":
        if args.block != 0 or any(args.jit_root.iterdir()):
            raise RuntimeError("only block0 may use an empty fresh JIT root")
    elif (
        args.block == 0
        or not before
        or before_hash != args.expected_jit_artifact_set_sha256
    ):
        raise RuntimeError("reuse JIT root does not match the block0 artifact lock")
    values = (
        {
            "FLASHINFER_WORKSPACE_BASE": args.jit_root,
            "CUTE_DSL_DUMP_DIR": args.jit_root / "dump",
        }
        if args.arm != "sglang_triton_fp8"
        else {"TRITON_CACHE_DIR": args.jit_root, "W4A4_SGLANG_JIT_DIR": args.jit_root}
    )
    for key, path in values.items():
        current = os.environ.get(key)
        if current and Path(current).resolve() != path.resolve():
            raise RuntimeError(f"{key} conflicts with --jit-root")
        os.environ[key] = str(path)
    if args.arm != "sglang_triton_fp8":
        os.environ["CUTE_DSL_KEEP"] = "ir,ptx,cubin,sass"
    return before_hash if before else None


def require_environment(args: argparse.Namespace) -> dict[str, str]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}", args.rerun_id):
        raise RuntimeError("invalid or missing rerun ID")
    if not os.environ.get("KDK_LEASE_ID", "").strip():
        raise RuntimeError("KDK_LEASE_ID is required")
    expected = TRITON_ENV if args.arm == "sglang_triton_fp8" else FP4_ENV
    for key, value in expected.items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"{key} identity drift")
    forbidden = (
        "CUTE_DSL_COMPILER_OPT",
        "FLASHINFER_CUTEDSL_IKET_OVERLAY",
        "EXP003_RUN_IKET",
        "EXP003_IKET_PROVIDER_ROOT",
        "EXP003_MARKER_OVERLAY",
        "W4A4_EXP003_MARKER_OVERLAY",
        "FLASHINFER_AUTOTUNER_LOAD_FROM_FILE",
        "FLASHINFER_TACTICS_BLOCKLIST",
    )
    if enabled := [key for key in forbidden if os.environ.get(key, "").strip()]:
        raise RuntimeError(
            f"instrumentation/compiler overrides are forbidden: {enabled}"
        )
    return dict(expected)


def foreign_processes(uuid: str) -> list[dict[str, str]]:
    rows = []
    output = command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    for line in output.splitlines():
        values = [part.strip() for part in line.split(",", 2)]
        if len(values) == 3 and values[0] == uuid:
            rows.append({"gpu_uuid": values[0], "pid": values[1], "process": values[2]})
    return rows


def telemetry(uuid: str, *, gate_foreign: bool) -> dict[str, Any]:
    fields = (
        "uuid,name,pci.bus_id,driver_version,clocks.current.graphics,clocks.current.memory,"
        "clocks.applications.graphics,temperature.gpu,power.draw,pstate"
    )
    row = command(
        [
            "nvidia-smi",
            "-i",
            uuid,
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    if len(row) != 1:
        raise RuntimeError("selected GPU telemetry is ambiguous")
    keys = (
        "uuid",
        "name",
        "pci_bus_id",
        "driver",
        "graphics_clock_mhz",
        "memory_clock_mhz",
        "applications_graphics_clock_mhz",
        "temperature_c",
        "power_w",
        "pstate",
    )
    values = [part.strip() for part in row[0].split(",")]
    result: dict[str, Any] = dict(zip(keys, values, strict=True))
    if (
        result["uuid"] != uuid
        or int(float(result["applications_graphics_clock_mhz"])) != CLOCK_MHZ
    ):
        raise RuntimeError("GPU UUID/application clock drift")
    xml = ET.fromstring(command(["nvidia-smi", "-q", "-x", "-i", uuid]))
    counters = {}
    prefix = "clocks_event_reasons_counters_"
    for child in xml.iter():
        if child.tag.startswith(prefix) and (
            match := re.search(r"([0-9]+)\s*us", child.text or "")
        ):
            counters[child.tag.removeprefix(prefix)] = int(match.group(1))
    if not {"sw_therm_slowdown", "hw_therm_slowdown"}.issubset(counters):
        raise RuntimeError("thermal slowdown counters unavailable")
    result["throttle_counters_us"] = counters
    result["foreign_compute_processes"] = (
        foreign_processes(uuid) if gate_foreign else "not_queried_after_context"
    )
    return result


def telemetry_verdict(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    names = sorted(
        name
        for name in before["throttle_counters_us"]
        if "therm" in name or name == "hw_power_brake"
    )
    checks = {
        "uuid_stable": before["uuid"] == after["uuid"],
        "application_clock_stable": before["applications_graphics_clock_mhz"]
        == after["applications_graphics_clock_mhz"],
        "no_foreign_process_before": not before["foreign_compute_processes"],
        "slowdown_not_increased": all(
            after["throttle_counters_us"][name] <= before["throttle_counters_us"][name]
            for name in names
        ),
    }
    return {
        "checks": checks,
        "checked_throttle_counters": names,
        "pass": all(checks.values()),
    }


def runtime(args: argparse.Namespace, environment: Mapping[str, str]) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one GPU must be visible")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    if list(torch.cuda.get_device_capability(0)) not in ([12, 0], [12, 1]):
        raise RuntimeError("exp_018 requires SM120/121")
    freeze = command([sys.executable, "-m", "pip", "freeze"])
    return {
        "timestamp_unix": time.time(),
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nvcc": command(["nvcc", "--version"], optional=True),
        "gpu_uuid": args.expected_gpu_uuid,
        "gpu_name": properties.name,
        "sm_count": properties.multi_processor_count,
        "lease_id": os.environ["KDK_LEASE_ID"],
        "image_environment": dict(environment),
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
        "jit_root": str(args.jit_root),
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm == "sglang_triton_fp8":
        init = triton.initialize_sglang()
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts_impl,
        )

        callable_source = Path(
            inspect.getsourcefile(fused_experts_impl) or ""
        ).resolve()
        weights = triton.make_fp8_weights(device=torch.device("cuda"), seed=SEED)
        return {
            "weights": weights,
            "source_identity": {
                "kind": "sglang_legacy_triton_fp8_chain",
                "sglang": init,
                "sglang_commit": TRITON_ENV["W4A4_SGLANG_COMMIT"],
                "callable_source": str(callable_source),
                "callable_source_sha256": triton.file_sha256(callable_source),
                "adapter_source_sha256": triton.file_sha256(
                    EXP001 / "bench_triton_fp8.py"
                ),
            },
        }
    source = SOURCES[args.arm]
    if triton.file_sha256(source) != SOURCE_SHA[args.arm]:
        raise RuntimeError(f"{args.arm} source drift")
    overlay, adapter = source, None
    if args.arm == "eric_stage4_fp4":
        adapter = eric_adapter.build_adapter(
            source=source,
            output_dir=args.jit_root / "eric_adapter",
            expected_original_sha256=SOURCE_SHA[args.arm],
        )
        overlay = args.jit_root / "eric_adapter" / eric_adapter.ADAPTER_NAME
        if triton.file_sha256(overlay) != ERIC_ADAPTER_SHA:
            raise RuntimeError("Eric compatibility adapter drift")
    fp4_worker.install_overlay(overlay)
    imports = fp4_worker.configure_source_checkout(args.flashinfer_root)
    if Path(imports["target_module"]).resolve() != overlay.resolve():
        raise RuntimeError("selected FP4 overlay was not imported")
    weights = nvfp4.make_canonical_weights(device=torch.device("cuda"), seed=SEED)
    return {
        "weights": weights,
        "source_identity": {
            "kind": "cutedsl_dynamic_fused_moe",
            "source": str(source),
            "source_sha256": SOURCE_SHA[args.arm],
            "overlay": str(overlay),
            "overlay_sha256": triton.file_sha256(overlay),
            "adapter": adapter,
            "flashinfer_commit": command(
                ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
                cwd=args.flashinfer_root,
            ),
            "cutlass_commit": command(
                ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
                cwd=args.flashinfer_root / "3rdparty/cutlass",
            ),
            "target_module": TARGET_MODULE,
            "expected_block_threads": BLOCK_THREADS[args.arm],
            "wrapper_kwargs": {
                "E": 256,
                "topk": 8,
                "H": 2048,
                "I_tp": 512,
                "activation": "silu",
                "quant_mode": "w4a4",
                "use_cuda_graph": True,
            },
        },
    }


def normalized_weight_identity(arm: str, weights: Any) -> dict[str, Any]:
    manifest = weights.manifest
    if arm == "sglang_triton_fp8":
        packed = {"w1": manifest["w1_sha256"], "w2": manifest["w2_sha256"]}
        scales = {"w1": manifest["w1_scale_sha256"], "w2": manifest["w2_scale_sha256"]}
    else:
        packed = {
            "w1": manifest["w1_packed_sha256"],
            "w2": manifest["w2_packed_sha256"],
        }
        scales = {
            "w1": manifest["w1_cutedsl_scale_sha256"],
            "w2": manifest["w2_cutedsl_scale_sha256"],
            "w1_global": manifest["w1_global_scale_sha256"],
            "w2_global": manifest["w2_global_scale_sha256"],
        }
    return {
        "seed": SEED,
        "packed_weights_sha256": packed,
        "scales_sha256": scales,
        "manifest": manifest,
    }


def sanity(
    output: torch.Tensor, m: int, hash_tensor: Callable[[torch.Tensor], str]
) -> dict[str, Any]:
    value = {
        "shape": list(output.shape),
        "dtype": str(output.dtype).replace("torch.", ""),
        "finite": bool(torch.isfinite(output).all()),
        "nonzero": bool(torch.count_nonzero(output)),
        "sentinel_nan_remaining": int(torch.isnan(output).sum()),
        "output_sha256": hash_tensor(output),
    }
    value["pass"] = (
        value["shape"] == [m, 2048]
        and value["dtype"] == "bfloat16"
        and value["finite"]
        and value["nonzero"]
        and not value["sentinel_nan_remaining"]
    )
    return value


def workspace_gate(
    args: argparse.Namespace, captured: Any, routed: Any
) -> dict[str, Any]:
    warps = BLOCK_THREADS[args.arm] // 32
    _, summary = fp4_worker._workspace_snapshot(
        captured.wrapper, routed, num_cta_warps=warps
    )
    if args.arm == "latest_opt_fp4":
        verification = summary["verification"]
        expected = (math.ceil(routed.m / warps) + 110) * warps
        observed = int(captured.wrapper._dynamic_workspace.pair_head.item())
        verification["checks"]["pair_head_terminal_state"] = observed == expected
        verification.update(
            {
                "gate_pass": all(verification["checks"].values()),
                "producer_counter_unit": "token",
                "expected_pair_head": expected,
                "observed_pair_head": observed,
            }
        )
    return summary


def measure(
    flush: Callable[[], None], replay_ms: Callable[[], float]
) -> tuple[float, dict[str, float]]:
    for _ in range(WARMUP):
        flush()
        replay_ms()
    samples = []
    for _ in range(ITERS):
        flush()
        samples.append(replay_ms() * 1000.0)
    return statistics.fmean(samples), {
        "min_us": min(samples),
        "max_us": max(samples),
        "median_us": statistics.median(samples),
    }


def run_cell(
    args: argparse.Namespace, context: Mapping[str, Any], m: int
) -> dict[str, Any]:
    x, ids, routing, fixture = persisted.load_fixture(
        args.fixture_dir, m, torch.device("cuda")
    )
    full_oracle = args.block == 0
    if args.arm == "sglang_triton_fp8":
        launch, launch_identity = triton.build_launch(
            x, ids, routing, context["weights"]
        )
        launch()
        torch.cuda.synchronize()
        reference = (
            triton.fp8_oracle(x, ids, routing, context["weights"])
            if full_oracle
            else None
        )
        captured = triton.CapturedCall(launch)
        captured.capture()

        def replay(sentinel: bool = False):
            if sentinel:
                captured.output.fill_(float("nan"))
                torch.cuda.synchronize()
            elapsed = captured.replay_ms()
            return captured.output.clone(), elapsed

        timed_replay = captured.replay_ms
        hash_tensor, oracle = triton.tensor_sha256, triton.output_diagnostics
        flush, flush_bytes = triton.make_l2_flusher(x.device, FLUSH_BYTES)
        workspace = lambda: {"not_applicable": True, "gate_pass": True}
    else:
        routed = nvfp4.RoutedFixture(m, x, ids, routing, fixture)
        reference = (
            nvfp4.reference_moe_nvfp4(routed, context["weights"])
            if full_oracle
            else None
        )
        captured = fp4_worker.build_arm(
            argparse.Namespace(m=m, device_index=0), routed, context["weights"]
        )
        captured.eager()
        captured.capture()

        def replay(sentinel: bool = False):
            output, elapsed = captured.replay(sentinel=sentinel)
            return output.clone(), elapsed

        timed_replay = lambda: captured.replay()[1]
        hash_tensor, oracle = fp4_worker.tensor_sha256, nvfp4.output_diagnostics
        flush, flush_bytes = fp4_worker.make_flusher(x.device, FLUSH_BYTES)
        workspace = lambda: workspace_gate(args, captured, routed)
        launch_identity = {
            "grid": [1, 1, 110],
            "block": [BLOCK_THREADS[args.arm], 1, 1],
        }

    def qualify() -> dict[str, Any]:
        output, elapsed = replay(True)
        basic, route = sanity(output, m, hash_tensor), workspace()
        diagnostics = oracle(output, reference) if reference is not None else None
        passed = (
            basic["pass"]
            and bool(
                route.get("gate_pass", route.get("verification", {}).get("gate_pass"))
            )
            and (diagnostics is None or diagnostics["formal_pass"])
        )
        return {
            "event_elapsed_us": elapsed * 1000.0,
            "sanity": basic,
            "oracle": diagnostics,
            "workspace": route,
            "pass": passed,
        }

    pre = [qualify(), qualify()]
    if not all(item["pass"] for item in pre):
        raise RuntimeError("pre-timing qualification failed")
    sample_us, timed = measure(flush, timed_replay)
    post = qualify()
    if not post["pass"]:
        raise RuntimeError("post-timing qualification failed")
    launch_identity.update({"timed_statistics": timed, "l2_flush_bytes": flush_bytes})
    return {
        "sample_us": sample_us,
        "fixture_manifest": fixture,
        "launch_identity": launch_identity,
        "correctness": {
            "mode": "full_oracle" if full_oracle else "sentinel_sanity",
            "pre_replays": pre,
            "post_timing": post,
            "qualification_pass": True,
        },
    }


def jit_identity(args: argparse.Namespace) -> dict[str, Any]:
    artifacts = common.artifact_manifest(args.jit_root)
    if not artifacts:
        raise RuntimeError("JIT artifact set is empty")
    digest = common.canonical_sha256(artifacts)
    if args.jit_policy == "reuse" and digest != args.expected_jit_artifact_set_sha256:
        raise RuntimeError("reuse run mutated the block0 JIT artifact set")
    symbols = set()
    for item in artifacts:
        path = args.jit_root / item["path"]
        if path.suffix in {".sass", ".ptx"}:
            symbols.update(
                re.findall(
                    r"(?:Function\s*:\s*|\.entry\s+)([^\s({]+)",
                    path.read_text(errors="ignore"),
                )
            )
        elif path.suffix == ".json" and path.stat().st_size < 10 << 20:
            text = path.read_text(errors="ignore")
            symbols.update(re.findall(r'"(?:name|kernel_name)"\s*:\s*"([^"]+)"', text))
    target_symbols = sorted(name for name in symbols if "moe" in name.lower())
    if not target_symbols:
        raise RuntimeError("JIT metadata contains no MoE kernel symbol")
    cubins = sorted(
        {item["sha256"] for item in artifacts if item["path"].endswith(".cubin")}
    )
    if args.arm != "sglang_triton_fp8" and not cubins:
        raise RuntimeError("CuteDSL JIT artifact set contains no cubin")
    return {
        "artifact_count": len(artifacts),
        "artifact_set_sha256": digest,
        "cubin_sha256": cubins,
        "symbols": target_symbols,
        "symbol_gate": True,
    }


def protocol() -> dict[str, Any]:
    return {
        "m_values": list(M_VALUES),
        "case": {"E": 256, "H": 2048, "I_tp": 512, "topk": 8},
        "boundary": "BF16 input -> complete MoE -> BF16 output",
        "warmup": WARMUP,
        "iters": ITERS,
        "l2_flush_bytes": FLUSH_BYTES,
        "blocks": 3,
        "block_orders": [list(order) for order in BLOCK_ORDERS],
        "timing": "single CUDA Graph replay with external CUDA events",
        "qualification": "block0 full oracle; block1/2 two-pre plus one-post sentinel sanity",
        "cooldown_between_arm_processes_seconds": 2,
        "profilers": "forbidden",
        "application_graphics_clock_mhz": CLOCK_MHZ,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--block", type=int, choices=range(3), required=True)
    parser.add_argument("--jit-policy", choices=("fresh", "reuse"), required=True)
    parser.add_argument("--expected-jit-artifact-set-sha256")
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--rerun-id", default=os.environ.get("W4A4_RERUN_ID", ""))
    parser.add_argument("--flashinfer-root", type=Path, default=REPO)
    parser.add_argument("--fixture-dir", type=Path, default=FIXTURE_DIR)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("jit_root", "flashinfer_root", "fixture_dir", "results"):
        setattr(args, name, getattr(args, name).resolve())
    output = args.results / "raw" / args.arm / f"block_{args.block}.json"
    csv_path = output.with_suffix(".csv")
    if output.exists() or csv_path.exists():
        raise RuntimeError(f"immutable output exists: {output}")
    fixtures = fixture_identity(args.fixture_dir)
    configure_jit(args)
    environment = require_environment(args)
    before = telemetry(args.expected_gpu_uuid, gate_foreign=True)
    if before["foreign_compute_processes"]:
        raise RuntimeError(
            f"foreign compute process present: {before['foreign_compute_processes']}"
        )
    runtime_identity = runtime(args, environment)
    context = prepare(args)
    cells = []
    for m in M_VALUES:
        cell = {
            "m": m,
            "status": "Invalid",
            "reason": "not executed",
            "sample_us": None,
            "correctness": {
                "mode": "full_oracle" if args.block == 0 else "sentinel_sanity",
                "qualification_pass": False,
            },
            "fixture_sha256": FIXTURE_SHA[m],
            "fixture_manifest_sha256": FIXTURE_MANIFEST_SHA,
            "launch_identity": None,
        }
        try:
            value = run_cell(args, context, m)
            cell.update(value)
            cell.update({"status": "Pass", "reason": ""})
        except Exception as error:
            cell["reason"] = f"{type(error).__name__}: {error}"
        cells.append(cell)
        torch.cuda.empty_cache()
    jit = jit_identity(args)
    after = telemetry(args.expected_gpu_uuid, gate_foreign=False)
    gate = telemetry_verdict(before, after)
    if not gate["pass"]:
        raise RuntimeError(f"telemetry gate failed: {gate}")
    contract = protocol()
    source = dict(context["source_identity"])
    source["runner_sha256"] = triton.file_sha256(Path(__file__))
    payload = {
        "schema": "exp018.arm-block.v1",
        "arm": args.arm,
        "block": args.block,
        "rerun_id": args.rerun_id,
        "protocol": contract,
        "protocol_sha256": triton.canonical_sha256(contract),
        "runtime": runtime_identity,
        "source_identity": source,
        "fixture_identity": fixtures,
        "weight_identity": normalized_weight_identity(args.arm, context["weights"]),
        "telemetry_before": before,
        "telemetry_after": after,
        "telemetry_gate": gate,
        "jit_identity": jit,
        "cells": cells,
        "block_status": "complete"
        if all(cell["status"] == "Pass" for cell in cells)
        else "complete_with_invalid_cells",
    }
    common.write_json(output, payload)
    common.write_csv(
        csv_path,
        [
            {
                "schema": payload["schema"],
                "rerun_id": args.rerun_id,
                "arm": args.arm,
                "block": args.block,
                "m": cell["m"],
                "status": cell["status"],
                "reason": cell["reason"],
                "sample_us": cell["sample_us"],
                "qualification_pass": cell["correctness"]["qualification_pass"],
                "fixture_sha256": cell["fixture_sha256"],
                "protocol_sha256": payload["protocol_sha256"],
                "jit_artifact_set_sha256": jit["artifact_set_sha256"],
                "telemetry_gate_pass": gate["pass"],
            }
            for cell in cells
        ],
    )
    print(
        json.dumps(
            {"output": str(output), "block_status": payload["block_status"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
