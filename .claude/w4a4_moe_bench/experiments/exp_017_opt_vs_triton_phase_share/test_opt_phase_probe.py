#!/usr/bin/env python3
"""CPU-only contracts for the exp_017 latest-opt full-phase probe."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent
FLASHINFER = ROOT.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_opt_phase_overlays as builder
from capture_opt_phase import partition_visible_processes
from capture_opt_phase import (
    EXPECTED_FIXTURE_SHA256,
    EXPECTED_OCCUPANCY_SHA256,
    load_exp001_fixture_module,
)
import exp017_opt_phase_common as common


def test_live_source_identity_and_overlay_contract(tmp_path: Path) -> None:
    assert common.file_sha256(FLASHINFER / common.OPT_RELATIVE_PATH) == (
        common.EXPECTED_OPT_SHA256
    )
    output = tmp_path / "overlays"
    identity = builder.build(FLASHINFER, output)
    checked = builder.verify_existing(FLASHINFER, output)
    assert identity == checked
    assert identity["event_abi"] == common.EVENT_ABI
    assert all(identity["cross_mode"].values())

    _, control_kernel, control_dispatch = builder.overlay_paths(output, common.CONTROL)
    _, probe_kernel, probe_dispatch = builder.overlay_paths(output, common.PROBE)
    assert control_kernel.read_bytes() == probe_kernel.read_bytes()
    assert common.normalize_dispatch_flag(
        control_dispatch.read_text(encoding="utf-8")
    ) == common.normalize_dispatch_flag(probe_dispatch.read_text(encoding="utf-8"))

    base = (FLASHINFER / common.OPT_RELATIVE_PATH).read_text(encoding="utf-8")
    probe = probe_kernel.read_text(encoding="utf-8")
    assert common.barrier_fingerprint(probe) == common.barrier_fingerprint(base)
    assert probe.count("_exp017_read_globaltimer()") == 18
    assert probe.count("exp017_phase_events") >= 10
    for phase in common.PHASE_NAMES:
        assert phase in json.dumps(common.EVENT_ABI)
    assert "fc1_gate_up_swiglu_to_sC(" in probe
    assert "quantize_q1_sC_to_sA_sSFA(" in probe
    assert "fc2_to_sC(" in probe
    assert "scatter_sC_to_gmem(" in probe


def test_build_fails_closed_on_opt_source_drift(tmp_path: Path) -> None:
    fake = tmp_path / "repo"
    for relative in (
        common.OPT_RELATIVE_PATH,
        common.DISPATCH_RELATIVE_PATH,
        common.WRAPPER_RELATIVE_PATH,
    ):
        destination = fake / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FLASHINFER / relative, destination)
    opt = fake / common.OPT_RELATIVE_PATH
    opt.write_text(opt.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="kernel source drift"):
        builder.build(fake, tmp_path / "bad-overlays")


def _probe_events() -> list[int]:
    rows = []
    for entry, reader_final, tma_final in ((100, 220, 230), (110, 240, 235)):
        row = [0] * common.EVENTS_PER_CTA
        row[common.ENTRY_SLOT] = entry
        row[common.CURSOR_SLOT] = reader_final - 1
        row[common.READER_FINAL_SLOT] = reader_final
        row[common.TMA_FINAL_SLOT] = tma_final
        for index in range(len(common.PHASE_NAMES)):
            row[common.PHASE_SLOT_BASE + index] = 5
        rows.extend(row)
    return rows


def test_event_parser_closes_named_residual_and_launch_skew() -> None:
    result = common.validate_events(_probe_events(), mode=common.PROBE, grid_ctas=2)
    assert result["gate_pass"]
    assert result["closure_error_ns"] == 0
    assert result["share_sum_percent"] == pytest.approx(100.0)
    rows = {row["name"]: row for row in result["phase_rows"]}
    assert rows["clear_init"]["sum_cta_ns"] == 10
    assert rows["cta_residual"]["sum_cta_ns"] == 160
    assert rows["launch_skew_early_finish"]["sum_cta_ns"] == 20


def test_control_is_zero_and_probe_rejects_nonclosing_intervals() -> None:
    control = [0] * (2 * common.EVENTS_PER_CTA)
    assert common.validate_events(control, mode=common.CONTROL, grid_ctas=2)[
        "gate_pass"
    ]
    control[3] = 1
    with pytest.raises(common.PhaseProbeError, match="control wrote events"):
        common.validate_events(control, mode=common.CONTROL, grid_ctas=2)

    probe = _probe_events()
    probe[common.PHASE_SLOT_BASE] = 1_000
    with pytest.raises(common.PhaseProbeError, match="exceeds span"):
        common.validate_events(probe, mode=common.PROBE, grid_ctas=2)


def test_partition_visible_processes_exempts_only_capture_pid() -> None:
    capture, foreign = partition_visible_processes(
        [
            {"pid": "1", "process": "python3"},
            {"pid": "42", "process": "other"},
        ],
        own_pid=1,
    )
    assert capture == [{"pid": "1", "process": "python3"}]
    assert foreign == [{"pid": "42", "process": "other"}]


def test_persisted_m8192_fixture_identity_is_locked() -> None:
    path = ROOT.parent / "exp_001_backend_case_sweep" / "results" / "fixtures"
    assert common.file_sha256(path / "m8192.npz") == EXPECTED_FIXTURE_SHA256
    with np.load(path / "m8192.npz") as fixture:
        ids = fixture["topk_ids"]
    occupancy = np.bincount(ids.reshape(-1), minlength=256).astype(np.int64)
    assert hashlib.sha256(occupancy.tobytes()).hexdigest() == (
        EXPECTED_OCCUPANCY_SHA256
    )
    assert load_exp001_fixture_module().__name__ == "exp017_exp001_fixture"


def test_replay_summary_preserves_side_specific_phase_rows() -> None:
    timing = common.validate_events(_probe_events(), mode=common.PROBE, grid_ctas=2)
    runs = [
        {
            "mode": common.PROBE,
            "event_elapsed_us": 100.0 + index,
            "phase_timing": timing,
        }
        for index in range(5)
    ]
    summary = common.summarize_replays(runs)
    assert summary["event_elapsed_us"]["median"] == 102.0
    assert len(summary["phase_rows"]) == len(common.PHASE_NAMES) + 2
