from __future__ import annotations

from analyze_phase_timing import (
    _intersection_ticks,
    _merge_intervals,
    analyze,
    summarize_run,
)
from exp004_common import (
    MEASUREMENT_CONTROL,
    PROBE,
    SENTINEL,
    timing_ticks_capacity,
    write_json,
)
from build_result import render
from test_exp004_common import synthetic_buffer


def test_interval_union_and_intersection():
    assert _merge_intervals([(5, 10), (1, 4), (3, 8), (12, 13)]) == [
        (1, 10),
        (12, 13),
    ]
    assert _intersection_ticks([(0, 10), (20, 30)], [(5, 25)]) == 10


def test_direct_summary_preserves_additive_denominator():
    ticks, _ = synthetic_buffer(task_tail=2, task_capacity=3)
    summary = summarize_run(ticks, task_tail=2, task_capacity=3)
    assert summary["complete_warp_tasks"] == 8
    assert (
        sum(summary["phase_totals"].values()) + summary["residual_total"]
        == summary["denominator_ticks"]
    )
    assert (
        abs(sum(summary["phase_share_pct"].values()) + summary["residual_pct"] - 100)
        < 1e-9
    )
    assert len(summary["overlap"]) == 5 * 6


def test_small_end_to_end_reducer(tmp_path, monkeypatch):
    results = tmp_path / "results"
    write_json(
        results / "raw" / "binary_identity.json",
        {
            "arms": {
                PROBE: {
                    "identity": {
                        "cubin_sha256": "cubin",
                        "sass_sha256": "sass",
                        "kernel_overlay_sha256": "source",
                    }
                }
            },
            "gpu_identity": {PROBE: {"uuid": "GPU-test"}},
            "gates": {"formal_gate_pass": True},
        },
    )
    write_json(results / "raw" / "correctness.json", {"gate_pass": True})
    write_json(
        results / "raw" / "calibration" / "manifest.json",
        {"aggregate": {"delta_tick_p95": 0.01}},
    )
    probe_ticks, probe_cta = synthetic_buffer(task_tail=2, task_capacity=3)
    control_ticks = [SENTINEL] * timing_ticks_capacity(3)
    control_cta = [SENTINEL] * 3
    descriptors = [
        {"expert": 1, "m_tile": 10, "slice": 0, "slice_count": 1, "valid_rows": 128},
        {"expert": 2, "m_tile": 11, "slice": 1, "slice_count": 1, "valid_rows": 17},
    ]
    workspace_json = {
        "task_capacity": 3,
        "scalars": {"task_tail": 2},
        "verification": {
            "gate_pass": True,
            "task_descriptor_order_sha256": "descriptor",
        },
    }
    for arm in (MEASUREMENT_CONTROL, PROBE):
        root = results / "raw" / "phase_capture" / arm
        manifest_runs = []
        for replay in range(5):
            run_id = f"run_{replay}"
            run = root / run_id
            run.mkdir(parents=True)
            (run / "timing.pt").touch()
            (run / "workspace.pt").touch()
            write_json(run / "workspace.json", workspace_json)
            write_json(
                run / "metadata.json",
                {
                    "event_elapsed_us": 100.0 + (0.2 if arm == PROBE else replay % 2),
                    "timing_gate": {"gate_pass": True},
                    "workspace_gate": {"gate_pass": True},
                    "correctness": {
                        "gate": {"gate_pass": True},
                        "output_contract": {"gate_pass": True},
                    },
                    "runtime": {
                        "gpu": {"uuid": "GPU-test"},
                        "source": {"overlays": {"kernel": {"sha256": "source"}}},
                    },
                    "gpu_state_after": {
                        "graphics_clock_mhz": "2000",
                        "applications_graphics_clock_mhz": "2000",
                        "max_graphics_clock_mhz": "2500",
                        "power_draw_w": "100",
                    },
                },
            )
            manifest_runs.append(
                {"run_id": run_id, "metadata": f"{run_id}/metadata.json"}
            )
        write_json(root / "manifest.json", {"arm": arm, "runs": manifest_runs})

    def fake_timing(path):
        if MEASUREMENT_CONTROL in path.parts:
            return control_ticks, control_cta
        return probe_ticks, probe_cta

    monkeypatch.setattr("analyze_phase_timing._plain_timing", fake_timing)
    monkeypatch.setattr("analyze_phase_timing.EXPECTED_TASK_TAIL", 2)
    monkeypatch.setattr(
        "analyze_phase_timing._descriptors", lambda path: (descriptors, {})
    )

    assert analyze(results) == 0
    gates = __import__("json").loads(
        (results / "derived" / "analysis_gates.json").read_text()
    )
    assert gates["formal_gate_pass"]
    assert (results / "raw" / "phase_events.csv").read_text().count("\n") == 1531
    report, formal = render(results)
    assert formal
    assert "FC1 Gate" in report
    assert "Warp 4 Down TMA overlap" in report
