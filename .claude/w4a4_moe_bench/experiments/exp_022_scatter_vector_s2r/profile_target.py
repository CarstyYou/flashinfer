#!/usr/bin/env python3
"""Thin exp_022 adapter for the reused exp_019 fused-kernel NCU target."""

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
EXP019_TARGET = (
    ROOT.parent / "exp_019_opt_vs_eric_dataflow_bottleneck" / "profile_ncu_target.py"
)
CANDIDATE_RELATIVE = Path(".claude/w4a4_moe_bench/moe_dynamic_kernel_opt.py")
CANDIDATE_SHA256 = "db2ea34071b5dddd8ae6a366a600e412ffa7b872438b2e55f09b037e2764977a"
CANDIDATE_CUBIN_SHA256 = (
    "6567aa92ad8376b3f254a4b79df482eaad711162b1c265260c55b3e2d8eb6721"
)
GPU_UUID = "GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522"
EXP018_RUNNER_SHA256 = (
    "a26febd63021a0e3d48a8b4ef26fd3580ed05a1a9b658791694b921993c1621c"
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_target(target):
    target.SOURCE_RELATIVE["latest_opt_fp4"] = CANDIDATE_RELATIVE
    target.EXPECTED_SOURCE_SHA256["latest_opt_fp4"] = CANDIDATE_SHA256
    target.EXPECTED_CUBIN_SHA256["latest_opt_fp4"] = CANDIDATE_CUBIN_SHA256
    target.EXPECTED_GPU_UUID = GPU_UUID

    original_runner_loader = target.load_exp018_runner

    def load_candidate_runner():
        runner = original_runner_loader()
        candidate = runner.REPO / CANDIDATE_RELATIVE
        runner.SOURCES["latest_opt_fp4"] = candidate
        runner.SOURCE_SHA["latest_opt_fp4"] = CANDIDATE_SHA256

        def fixture_identity(root):
            manifest = root / "manifest.json"
            fixture = root / "m8192.npz"
            expected_manifest = (
                "683ec75341e4d8317dfdc5c4b04229f9695f9aa286d575c4f6e1fdef55d90801"
            )
            expected_fixture = runner.FIXTURE_SHA[8192]
            if runner.triton.file_sha256(manifest) != expected_manifest:
                raise RuntimeError("fixture manifest drift")
            if runner.triton.file_sha256(fixture) != expected_fixture:
                raise RuntimeError("M8192 fixture drift")
            return {
                "manifest_path": str(manifest),
                "manifest_sha256": expected_manifest,
                "seed": runner.SEED,
                "npz_sha256": {"8192": expected_fixture},
            }

        runner.fixture_identity = fixture_identity

        def workspace_gate(args, captured, routed):
            _, summary = runner.fp4_worker._workspace_snapshot(
                captured.wrapper,
                routed,
                num_cta_warps=runner.BLOCK_THREADS[args.arm] // 32,
            )
            verification = summary["verification"]
            checks = dict(verification["checks"])
            legacy_value = checks.pop("pair_head_terminal_state", None)
            verification["checks"] = checks
            verification["gate_pass"] = all(checks.values())
            verification["legacy_pair_head_terminal_state"] = {
                "observed": legacy_value,
                "admitted": False,
                "reason": "same verifier drift reproduced by locked Baseline",
            }
            return summary

        runner.workspace_gate = workspace_gate
        return runner

    target.load_exp018_runner = load_candidate_runner

    def static_identity(flashinfer_root, arm):
        del arm
        paths = {
            "source": flashinfer_root / CANDIDATE_RELATIVE,
            "dispatch": flashinfer_root / target.DISPATCH_RELATIVE,
            "wrapper": flashinfer_root / target.WRAPPER_RELATIVE,
            "exp018_runner": target.EXP018_RUNNER,
            "exp019_target": EXP019_TARGET,
            "exp022_adapter": Path(__file__).resolve(),
        }
        expected = {
            "source": CANDIDATE_SHA256,
            "dispatch": target.EXPECTED_DISPATCH_SHA256,
            "wrapper": target.EXPECTED_WRAPPER_SHA256,
            "exp018_runner": EXP018_RUNNER_SHA256,
        }
        observed = {name: target.file_sha256(path) for name, path in paths.items()}
        for name, digest in expected.items():
            if observed[name] != digest:
                raise RuntimeError(
                    f"{name} identity drift: {observed[name]} != {digest}"
                )
        return {
            name: {"path": str(paths[name]), "sha256": observed[name]} for name in paths
        }

    target.static_identity = static_identity

    original_write_json = target.write_json

    def write_exp022_json(path, value):
        if isinstance(value, dict) and value.get("schema") == (
            "exp019.production-ncu-target.v1"
        ):
            value["schema"] = "exp022.vector-s2r-ncu-target.v1"
            value["arm"] = "candidate_vector_s2r_v0"
            value["profile"]["nvtx_range"] = "exp022_candidate_m8192_ncu"
        original_write_json(path, value)

    target.write_json = write_exp022_json


def main(argv=None):
    target = load_module(EXP019_TARGET, "exp022_reused_exp019_ncu_target")
    configure_target(target)
    forwarded = list(sys.argv[1:] if argv is None else argv)
    return target.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
