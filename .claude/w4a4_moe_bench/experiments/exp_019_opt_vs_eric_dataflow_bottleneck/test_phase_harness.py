#!/usr/bin/env python3
"""CPU-only source-anchor, ABI, occurrence, and closure tests for exp_019."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parent
FLASHINFER = ROOT.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_phase_overlays as builder
import phase_common as common


def test_builds_matched_opt_and_eric_overlays(tmp_path: Path) -> None:
    output = tmp_path / "overlays"
    identity = builder.build(FLASHINFER, output)
    assert builder.verify_existing(FLASHINFER, output) == identity
    assert identity["event_abi"] == common.EVENT_ABI
    assert identity["occurrence_abi"] == common.OCCURRENCE_ABI
    assert all(identity["cross_mode"].values())

    for arm, timers in (("latest_opt_fp4", 18), ("eric_stage4_fp4", 17)):
        _, control_kernel, control_dispatch = builder.overlay_paths(
            output, arm, common.CONTROL
        )
        _, probe_kernel, probe_dispatch = builder.overlay_paths(
            output, arm, common.PROBE
        )
        assert control_kernel.read_bytes() == probe_kernel.read_bytes()
        assert common.normalize_dispatch_flag(
            control_dispatch.read_text(encoding="utf-8")
        ) == common.normalize_dispatch_flag(probe_dispatch.read_text(encoding="utf-8"))
        source = probe_kernel.read_text(encoding="utf-8")
        base = (FLASHINFER / common.SOURCE_RELATIVE_PATH[arm]).read_text(
            encoding="utf-8"
        )
        assert common.barrier_fingerprint(source) == common.barrier_fingerprint(base)
        assert source.count("_exp017_read_globaltimer()") == timers
        assert source.count("_count = _ld_global_u64(") == len(common.PHASE_NAMES)
        assert "_EXP019_PHASE_STORAGE_PER_CTA = 24" in probe_dispatch.read_text(
            encoding="utf-8"
        )

    eric = builder.overlay_paths(output, "eric_stage4_fp4", common.PROBE)[1].read_text(
        encoding="utf-8"
    )
    activation = eric.index("# Activation + quant into sA")
    phase_b = eric.index("# PHASE B: Sweep ALL FC2 output tiles", activation)
    assert activation < eric.index("exp019_fc1_count", activation) < phase_b
    assert activation < eric.index("exp019_q1_count", activation) < phase_b
    output_loop = eric.index("for output_tile_idx in range", phase_b)
    assert output_loop < eric.index("exp019_fc2_count", output_loop)
    assert output_loop < eric.index("exp019_scatter_count", output_loop)
    assert (
        identity["arms"]["eric_stage4_fp4"]["modes"][common.PROBE]["base"][
            "adapter_sha256"
        ]
        == common.EXPECTED_ERIC_ADAPTER_SHA256
    )


def test_build_fails_closed_on_eric_source_drift(tmp_path: Path) -> None:
    fake = tmp_path / "repo"
    for relative in (
        *common.SOURCE_RELATIVE_PATH.values(),
        common.DISPATCH_RELATIVE_PATH,
        common.WRAPPER_RELATIVE_PATH,
    ):
        destination = fake / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FLASHINFER / relative, destination)
    eric = fake / common.SOURCE_RELATIVE_PATH["eric_stage4_fp4"]
    eric.write_text(eric.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="eric_stage4_fp4 source drift"):
        builder.build(fake, tmp_path / "bad")


def _probe_storage(grid_ctas: int = 2) -> list[int]:
    timing = []
    for entry, reader_final, tma_final in ((100, 220, 230), (110, 240, 235)):
        row = [0] * common.EVENTS_PER_CTA
        row[common.exp017.ENTRY_SLOT] = entry
        row[common.exp017.CURSOR_SLOT] = reader_final - 1
        row[common.exp017.READER_FINAL_SLOT] = reader_final
        row[common.exp017.TMA_FINAL_SLOT] = tma_final
        for index in range(len(common.PHASE_NAMES)):
            row[common.exp017.PHASE_SLOT_BASE + index] = 5
        timing.extend(row)
    occurrences = []
    for _ in range(grid_ctas):
        occurrences.extend([1] * len(common.PHASE_NAMES))
    return timing + occurrences


def test_reused_timing_abi_closes_and_occurrences_are_independent() -> None:
    result = common.validate_phase_storage(
        _probe_storage(), mode=common.PROBE, grid_ctas=2
    )
    timing = result["timing"]
    assert timing["closure_error_ns"] == 0
    assert timing["share_sum_percent"] == pytest.approx(100.0)
    assert result["occurrence_totals"] == {phase: 2 for phase in common.PHASE_NAMES}
    semantic = common.aggregate_semantic_rows(timing["phase_rows"])
    assert sum(row["share_percent"] for row in semantic) == pytest.approx(100.0)

    control = [0] * (2 * (common.EVENTS_PER_CTA + common.OCCURRENCES_PER_CTA))
    assert common.validate_phase_storage(control, mode=common.CONTROL, grid_ctas=2)[
        "storage_all_zero"
    ]
    control[-1] = 1
    with pytest.raises(common.PhaseHarnessError, match="control wrote occurrences"):
        common.validate_phase_storage(control, mode=common.CONTROL, grid_ctas=2)


def test_arm_specific_interval_occurrence_contract() -> None:
    opt = common.expected_occurrences(
        "latest_opt_fp4", task_count=7, slice_count=9, grid_ctas=2
    )
    eric = common.expected_occurrences(
        "eric_stage4_fp4", task_count=7, slice_count=9, grid_ctas=2
    )
    assert opt["claim_cache_control"] == eric["claim_cache_control"] == 9
    assert eric["fc1_gate_up_swiglu"] == 2 * opt["fc1_gate_up_swiglu"]
    assert eric["q1"] == 2 * opt["q1"]
    assert eric["fc2_epilogue_r2s"] == 2 * opt["fc2_epilogue_r2s"]
    assert eric["scatter"] == 2 * opt["scatter"]
    assert common.occurrence_gate(eric, eric)["gate_pass"]
    assert not common.occurrence_gate({**eric, "scatter": eric["scatter"] - 1}, eric)[
        "gate_pass"
    ]
    assert common.cyclic_process_order(0) != common.cyclic_process_order(1)
