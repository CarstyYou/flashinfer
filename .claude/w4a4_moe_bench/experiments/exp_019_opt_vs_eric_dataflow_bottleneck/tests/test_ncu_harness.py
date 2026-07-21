"""CPU-only identity and CLI smoke tests for the exp_019 NCU harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = ROOT / "profile_ncu_target.py"
SHELL_PATH = ROOT / "run_ncu_remote.sh"
SPEC = importlib.util.spec_from_file_location("exp019_ncu_target_test", TARGET_PATH)
assert SPEC is not None and SPEC.loader is not None
target = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = target
SPEC.loader.exec_module(target)


def test_identity_constants_match_exp018_block0_manifests() -> None:
    for arm in target.ARMS:
        path = target.EXP018 / "results/raw" / arm / "block_0.json"
        payload = json.loads(path.read_text())
        assert payload["arm"] == arm
        assert payload["block_status"] == "complete"
        assert (
            payload["source_identity"]["source_sha256"]
            == (target.EXPECTED_SOURCE_SHA256[arm])
        )
        assert payload["jit_identity"]["cubin_sha256"] == [
            target.EXPECTED_CUBIN_SHA256[arm]
        ]
        assert (
            payload["jit_identity"]["artifact_set_sha256"]
            == (target.EXPECTED_JIT_ARTIFACT_SET_SHA256[arm])
        )
        assert payload["jit_identity"]["symbols"] == [target.EXPECTED_SYMBOL]
        cells = {cell["m"]: cell for cell in payload["cells"]}
        for m in target.M_VALUES:
            assert cells[m]["status"] == "Pass"
            assert cells[m]["correctness"]["qualification_pass"] is True
            assert cells[m]["fixture_sha256"] == target.EXPECTED_FIXTURE_SHA256[m]
            assert cells[m]["launch_identity"]["grid"] == target.EXPECTED_GRID
            assert cells[m]["launch_identity"]["block"] == (target.EXPECTED_BLOCK[arm])


def test_live_source_dispatch_wrapper_identity_is_locked() -> None:
    observed = target.static_identity(target.REPO, "latest_opt_fp4")
    assert (
        observed["source"]["sha256"] == target.EXPECTED_SOURCE_SHA256["latest_opt_fp4"]
    )
    assert observed["dispatch"]["sha256"] == target.EXPECTED_DISPATCH_SHA256
    assert observed["wrapper"]["sha256"] == target.EXPECTED_WRAPPER_SHA256


def test_cli_and_remote_protocol_smoke() -> None:
    for arm in target.ARMS:
        for m in target.M_VALUES:
            args = target.parse_args(
                [
                    "--arm",
                    arm,
                    "--m",
                    str(m),
                    "--jit-root",
                    "/tmp/jit",
                    "--output",
                    "/tmp/manifest.json",
                ]
            )
            assert (args.arm, args.m) == (arm, m)

    subprocess.run(["bash", "-n", str(SHELL_PATH)], check=True)
    shell = SHELL_PATH.read_text()
    for section in (
        "SpeedOfLight",
        "ComputeWorkloadAnalysis",
        "MemoryWorkloadAnalysis",
        "Occupancy",
        "SchedulerStats",
        "WarpStateStats",
        "LaunchStats",
        "InstructionStats",
        "SourceCounters",
    ):
        assert section in shell
    for option in (
        "--profile-from-start off",
        "--graph-profiling node",
        "--replay-mode kernel",
        "--cache-control all",
        "--clock-control none",
        "--launch-count 1",
    ):
        assert option in shell
    assert "--kernel-name 'regex:.*MoEDynamicKernel.*'" in shell
    assert "latest_opt_fp4:8192 eric_stage4_fp4:8192" in shell
    assert "latest_opt_fp4:1024 eric_stage4_fp4:1024" in shell
    assert "$jit_base/$arm/m$m" in shell
