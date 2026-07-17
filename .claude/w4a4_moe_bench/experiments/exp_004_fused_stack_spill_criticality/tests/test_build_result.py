from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from build_result import build_manifest, main, make_decision, render_report
from exp004_common import read_json, validate_manifest_schema


ROOT = Path(__file__).resolve().parents[1]


def evidence():
    return read_json(ROOT / "results" / "spill_localization_evidence.json")


def copy_manifest_inputs(destination: Path) -> Path:
    source = ROOT / "results"
    relative_paths = (
        "spill_localization_evidence.json",
        "static_spill_evidence.json",
        "ncu/spill_evidence.json",
        "arms/baseline/preparation.json",
        "arms/up_first_attribution/preparation.json",
        "overlays/identity.json",
        "resource_cleanup.json",
        "attribution_evidence.json",
    )
    for relative in relative_paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    return destination


def test_spill_is_p0_without_latency_claim() -> None:
    decision = make_decision(evidence())
    assert decision["severity"] == "P0_hard_failure_by_project_policy"
    assert decision["latency_causality"] == "not_tested_and_not_required_for_P0"
    assert decision["formal_experiment_closed"] is False


def test_partial_source_localization_blocks_optimization() -> None:
    decision = make_decision(evidence())
    assert decision["physical_localization_status"] == "mechanism_localized"
    assert decision["source_value_localization_status"] == (
        "partially_semantic_localized"
    )
    assert decision["optimization_recommendation_allowed"] is False


def test_report_names_mixed_tail_and_exact_problem_point() -> None:
    data = evidence()
    report = render_report(data, make_decision(data))
    assert "两个物理问题点已定位" in report
    assert "第一段 FC1 收尾阶段" in report
    assert "在各自 producer 后被逐步保存" in report
    assert "activation 入口被保存" in report
    assert "5` 个 second-pass accumulator" in report
    assert "8` 个 index/address scalar" in report
    assert "1` 个 control scalar" in report
    assert "优化建议：无" in report


def test_report_rejects_superseded_criticality_story() -> None:
    data = evidence()
    report = render_report(data, make_decision(data))
    assert "H14 criticality" not in report
    assert "H108 criticality" not in report
    assert "优先调查 FC1 pass order" not in report
    assert "缩短双 FP32 accumulator overlap lifetime" not in report


def test_compact_evidence_closes_every_physical_tail_chain() -> None:
    data = evidence()
    tail = data["tail_14_word_bundle"]
    assert tail["stack_word_count"] == 14
    assert tail["value_class_counts"] == {
        "index_address_scalar": 8,
        "long_lived_control_scalar": 1,
        "second_pass_accumulator": 5,
    }
    assert all(row["physical_chain_closed"] for row in tail["chains"])
    assert data["main_108_word_bundle"]["reload_first_use_is_scale_fmul_count"] == 108


def test_decision_keeps_main_and_tail_problem_points_separate() -> None:
    decision = make_decision(evidence())
    assert set(decision["physical_problem_points"]) == {
        "main_108_word_bundle",
        "tail_14_word_bundle",
    }
    assert set(decision["physical_mechanisms"]) == {
        "main_108_word_bundle",
        "tail_14_word_bundle",
    }


def test_manifest_clean_rebuild_is_deterministic_and_has_no_active_legacy_gates(
    tmp_path: Path,
) -> None:
    results = copy_manifest_inputs(tmp_path / "results")
    data = read_json(results / "spill_localization_evidence.json")
    decision = make_decision(data)
    first = build_manifest(results, data, decision)
    second = build_manifest(results, data, decision)
    assert first == second
    assert validate_manifest_schema(first) == []
    assert "candidate_gates" not in first
    assert "candidate_gates" not in first["static_spill"]
    assert "attribution" not in first
    assert set(first["arms"]) == {"baseline", "up_first_attribution"}


def test_manifest_rejects_preparation_ncu_cubin_drift(tmp_path: Path) -> None:
    results = copy_manifest_inputs(tmp_path / "results")
    path = results / "ncu" / "spill_evidence.json"
    ncu = read_json(path)
    ncu["inputs"]["baseline"]["cubin_sha256"] = "0" * 64
    path.write_text(json.dumps(ncu))
    data = read_json(results / "spill_localization_evidence.json")
    with pytest.raises(ValueError, match="preparation/NCU cubin identity drift"):
        build_manifest(results, data, make_decision(data))


def test_successful_partial_report_generation_exits_zero(tmp_path: Path) -> None:
    results = copy_manifest_inputs(tmp_path / "results")
    assert main(["--results", str(results)]) == 0
    manifest = read_json(results / "validation.manifest.json")
    assert manifest["status"] == "localization_partial"
    assert manifest["decision"]["formal_experiment_closed"] is False
    assert (
        (results / "result.md")
        .read_text()
        .startswith("# exp_004：Fused Spill 问题点定位")
    )
