from __future__ import annotations

from build_result import decide, render_report, select_static_candidate


def inputs():
    candidate = "activation_in_place_up"
    static = {
        "baseline_reproduction_pass": True,
        "deltas": {
            candidate: {
                "complete_tail_removal_no_replacement": True,
                "stack_bytes_delta": -56,
                "stored_words_delta": -14,
            }
        },
        "arms": {
            "baseline": {
                "resource": {"stack_bytes_per_thread": 488},
                "local": {"stored_words": 122},
            },
            candidate: {
                "resource": {"stack_bytes_per_thread": 432},
                "local": {"stored_words": 108},
            },
        },
    }
    base_metrics = {
        "local_load_sectors": 4_950_272,
        "local_store_sectors": 4_950_272,
        "executed_local_load_instructions": 1_000_000,
        "executed_local_store_instructions": 500_000,
        "achieved_occupancy_pct": 10.42,
        "eligible_warps_per_cycle": 0.4,
        "issue_active_pct": 20.0,
        "tc_subpipe_active_pct": 25.0,
        "stall_wait_pct": 20.0,
        "stall_long_scoreboard_pct": 30.0,
        "stall_short_scoreboard_pct": 3.0,
        "stall_barrier_pct": 0.0,
    }
    candidate_metrics = dict(base_metrics)
    candidate_metrics["local_load_sectors"] -= 568_064
    candidate_metrics["local_store_sectors"] -= 568_064
    ncu = {
        "baseline_reproduction_pass": True,
        "deltas": {
            candidate: {
                "dynamic_14_word_closure_pass": True,
                "work_identity_pass": True,
                "local_load_sector_reduction": 568_064,
                "local_store_sector_reduction": 568_064,
                "executed_local_load_instruction_reduction": 10,
                "executed_local_store_instruction_reduction": 10,
            }
        },
        "arms": {"baseline": base_metrics, candidate: candidate_metrics},
    }
    correctness = {"candidate": candidate, "gate_pass": True}
    benchmark = {
        "candidate": candidate,
        "material_improvement": False,
        "stable_le_5_percent": True,
        "candidate_faster_pair_count": 5,
        "speedup_fraction": 0.001,
        "speedup_percent": 0.1,
        "materiality_T_fraction": 0.002,
        "arms": {
            "baseline": {"median_us": 100.0, "spread_percent": 0.05},
            "candidate": {"median_us": 99.9, "spread_percent": 0.05},
        },
    }
    return static, ncu, correctness, benchmark


def test_non_material_tail_is_closed_not_p0() -> None:
    static, ncu, correctness, benchmark = inputs()
    decision = decide(
        static=static,
        ncu=ncu,
        correctness=correctness,
        benchmark=benchmark,
        def_use=None,
    )
    assert decision["h14_attribution"] == "accepted"
    assert decision["h14_criticality"] == "non_material"
    assert decision["formal_experiment_closed"] is True
    report = render_report(
        static=static,
        ncu=ncu,
        correctness=correctness,
        benchmark=benchmark,
        decision=decision,
        def_use=None,
    )
    assert "H108 criticality" in report
    assert "Local sectors" in report


def test_failed_dynamic_closure_blocks_criticality() -> None:
    static, ncu, correctness, benchmark = inputs()
    ncu["deltas"]["activation_in_place_up"]["dynamic_14_word_closure_pass"] = False
    decision = decide(
        static=static,
        ncu=ncu,
        correctness=correctness,
        benchmark=benchmark,
        def_use=None,
    )
    assert decision["h14_attribution"] == "inconclusive"
    assert decision["h14_criticality"].startswith("not_tested")


def test_primary_then_fallback_static_selection() -> None:
    static, _, _, _ = inputs()
    static["deltas"]["activation_in_place_up"][
        "selected_opcode_projection_equal_except_local"
    ] = True
    assert select_static_candidate(static) == "activation_in_place_up"
    static["deltas"]["activation_in_place_up"][
        "complete_tail_removal_no_replacement"
    ] = False
    static["deltas"]["activation_in_place_gate"] = {
        "complete_tail_removal_no_replacement": True,
        "selected_opcode_projection_equal_except_local": True,
    }
    assert select_static_candidate(static) == "activation_in_place_gate"


def test_no_qualified_static_arm_returns_none() -> None:
    static, _, _, _ = inputs()
    assert select_static_candidate(static) is None
