#!/usr/bin/env python3
"""Thin exp_009 registration wrapper around the exp_005 GPU worker.

This module does not copy the benchmark implementation.  It adds one distinct
non-production arm with the intern kernel's actual five-warp/160-thread launch
contract, then delegates to ``run_exp005_arm.main``.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Sequence


EXP_ROOT = Path(__file__).resolve().parent
EXP005_ROOT = EXP_ROOT.parent / "exp_005_8warp_spill_reduction"
EXP005_COMMON = EXP005_ROOT / "exp005_common.py"
EXP005_WORKER = EXP005_ROOT / "run_exp005_arm.py"

ARM_NAME = "candidate_4warp_stage4_compact"
EXPECTED_BLOCK = (160, 1, 1)
FULL_M_VALUES = (256, 512, 1024, 2048, 4096, 8192)
EXPECTED_PYTHON_DEPS_SHA256 = (
    "32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74"
)


class WorkerAdapterError(RuntimeError):
    """The historical worker cannot be extended without changing its meaning."""


def _failure_classification(
    diagnostics: Sequence[dict[str, Any]], workspace: Sequence[dict[str, Any]]
) -> str:
    if diagnostics:
        last = diagnostics[-1]
        if int(last.get("sentinel_nan_remaining", 0)):
            return "sentinel_or_incomplete_output"
        if last.get("formal_pass") is False:
            return "numerical_accuracy_failure"
    for replay in workspace:
        verification = replay.get("verification", {})
        if isinstance(verification, dict) and verification.get("gate_pass") is False:
            return "workspace_route_task_failure"
    return "runtime_or_harness_failure"


def install_prepare_failure_evidence(worker: ModuleType) -> None:
    """Keep actionable correctness diagnostics when the reused worker fails.

    The historical exp_005 worker deliberately fails before writing a complete
    preparation record.  For the mentor sweep, retain the already-computed
    numerical diagnostics and workspace gate beside its immutable raw outputs.
    The successful path and benchmark implementation remain the exp_005 code.
    """

    if getattr(worker, "_exp009_failure_evidence_installed", False):
        return
    original_prepare = worker.prepare
    original_make_case = worker.make_case

    def prepare_with_failure_evidence(args, runtime):
        captured_diagnostics: list[dict[str, Any]] = []
        patched_modules: list[tuple[ModuleType, Any]] = []

        def make_case_with_diagnostics(case_args):
            fixture_module, fixture, weights = original_make_case(case_args)
            original_diagnostics = fixture_module.output_diagnostics

            def output_diagnostics(actual, expected):
                value = original_diagnostics(actual, expected)
                # The caller appends sentinel_nan_remaining to this same dict
                # before it evaluates the gate, so retain the object itself.
                captured_diagnostics.append(value)
                return value

            fixture_module.output_diagnostics = output_diagnostics
            patched_modules.append((fixture_module, original_diagnostics))
            return fixture_module, fixture, weights

        worker.make_case = make_case_with_diagnostics
        try:
            return original_prepare(args, runtime)
        except Exception as error:
            raw_dir = worker.case_directory(
                args.results, args.arm, args.m, args.fixture
            )
            workspace = []
            for path in sorted(raw_dir.glob("workspace_replay_*.json")):
                try:
                    value = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                if isinstance(value, dict):
                    workspace.append(
                        {
                            "path": path.name,
                            "verification": value.get("verification"),
                        }
                    )
            payload = {
                "schema": "exp009.arm-preparation-failure.v1",
                "status": "failed",
                "arm": args.arm,
                "m": args.m,
                "fixture_kind": args.fixture,
                "error_type": type(error).__name__,
                "error": str(error),
                "classification": _failure_classification(
                    captured_diagnostics, workspace
                ),
                "outputs": captured_diagnostics,
                "workspace_replays": workspace,
                "runtime": runtime,
            }
            worker.write_json(raw_dir / "failure.json", payload)
            raise
        finally:
            worker.make_case = original_make_case
            for fixture_module, original_diagnostics in patched_modules:
                fixture_module.output_diagnostics = original_diagnostics

    worker.prepare = prepare_with_failure_evidence
    worker._exp009_failure_evidence_installed = True


def register_arm(common: ModuleType, worker: ModuleType) -> None:
    """Register the exp_009 arm in the imported exp_005 module globals."""

    common_arms = tuple(getattr(common, "KNOWN_ARMS", ()))
    worker_arms = tuple(getattr(worker, "KNOWN_ARMS", ()))
    if common_arms != worker_arms:
        raise WorkerAdapterError(
            f"exp_005 KNOWN_ARMS identity drift: {common_arms!r} != {worker_arms!r}"
        )
    blocks = dict(getattr(common, "EXPECTED_BLOCKS", {}))
    common_m_values = tuple(getattr(common, "M_VALUES", ()))
    worker_m_values = tuple(getattr(worker, "M_VALUES", ()))
    if common_m_values != worker_m_values:
        raise WorkerAdapterError(
            f"exp_005 M_VALUES identity drift: {common_m_values!r} != "
            f"{worker_m_values!r}"
        )
    unexpected_m = tuple(
        value for value in common_m_values if value not in FULL_M_VALUES
    )
    if unexpected_m:
        raise WorkerAdapterError(f"unexpected historical M values: {unexpected_m!r}")
    common.M_VALUES = FULL_M_VALUES
    worker.M_VALUES = FULL_M_VALUES
    if ARM_NAME in common_arms:
        if blocks.get(ARM_NAME) != EXPECTED_BLOCK:
            raise WorkerAdapterError(
                f"existing {ARM_NAME} block drift: {blocks.get(ARM_NAME)!r}"
            )
        common.EXPECTED_PYTHON_DEPS_SHA256 = EXPECTED_PYTHON_DEPS_SHA256
        worker.CANDIDATE = ARM_NAME
        worker.EXPECTED_PYTHON_DEPS_SHA256 = EXPECTED_PYTHON_DEPS_SHA256
        for m in FULL_M_VALUES:
            worker.require_arm_m(ARM_NAME, m)
        return

    common.KNOWN_ARMS = common_arms + (ARM_NAME,)
    blocks[ARM_NAME] = EXPECTED_BLOCK
    common.EXPECTED_BLOCKS = blocks
    common.EXPECTED_PYTHON_DEPS_SHA256 = EXPECTED_PYTHON_DEPS_SHA256

    # run_exp005_arm imports KNOWN_ARMS by value.  Its expected_block and
    # require_arm_m callables still resolve the patched exp005_common globals.
    worker.KNOWN_ARMS = common.KNOWN_ARMS
    worker.CANDIDATE = ARM_NAME
    worker.EXPECTED_PYTHON_DEPS_SHA256 = EXPECTED_PYTHON_DEPS_SHA256

    if tuple(worker.KNOWN_ARMS) != tuple(common.KNOWN_ARMS):
        raise WorkerAdapterError("worker/common arm registration did not converge")
    if tuple(worker.expected_block(ARM_NAME)) != EXPECTED_BLOCK:
        raise WorkerAdapterError("worker did not resolve the 160-thread launch block")
    for m in FULL_M_VALUES:
        worker.require_arm_m(ARM_NAME, m)


def load_worker() -> ModuleType:
    expected_common = EXP005_COMMON.resolve()
    expected_worker = EXP005_WORKER.resolve()
    if not expected_common.is_file() or not expected_worker.is_file():
        raise WorkerAdapterError("exp_005 worker sources are missing")
    exp005_path = str(EXP005_ROOT.resolve())
    if exp005_path not in sys.path:
        sys.path.insert(0, exp005_path)

    common = importlib.import_module("exp005_common")
    worker = importlib.import_module("run_exp005_arm")
    if Path(common.__file__).resolve() != expected_common:
        raise WorkerAdapterError(f"unexpected exp005_common import: {common.__file__}")
    if Path(worker.__file__).resolve() != expected_worker:
        raise WorkerAdapterError(f"unexpected run_exp005_arm import: {worker.__file__}")
    register_arm(common, worker)
    install_prepare_failure_evidence(worker)
    return worker


def main(argv: Sequence[str] | None = None) -> int:
    return int(load_worker().main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
