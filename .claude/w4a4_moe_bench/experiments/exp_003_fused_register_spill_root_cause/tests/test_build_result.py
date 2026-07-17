from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest

from build_result import build_manifest, main, make_decision, render_report
from exp003_common import file_sha256, read_json, validate_manifest_schema


ROOT = Path(__file__).resolve().parents[1]


def evidence():
    return read_json(ROOT / "results" / "spill_root_cause_evidence.json")


def copy_manifest_inputs(destination: Path) -> Path:
    source = ROOT / "results"
    relative_paths = (
        "spill_root_cause_evidence.json",
        "static_spill_evidence.json",
        "ncu/spill_evidence.json",
        "correctness/up_first_attribution.json",
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
    assert decision["formal_experiment_closed"] is True
    assert decision["tc_cadence_hypothesis"].startswith("unverified_")


def test_tail_source_identity_residual_does_not_block_scoped_mechanism_closure() -> None:
    decision = make_decision(evidence())
    assert decision["physical_mechanism_status"] == "all_bundles_closed"
    assert decision["main_108_status"] == "physical_formation_mechanism_closed"
    assert decision["tail_14_status"] == (
        "physical_formation_mechanism_closed_source_identity_partial"
    )
    assert decision["source_attribution_status"] == (
        "high_confidence_program_order_inference_not_compiler_certified"
    )
    assert decision["tail_14_disposition"] == "deferred_reprofile_after_main_change"
    assert decision["production_optimization_recommendation_allowed"] is False
    assert decision["followup_experiment_allowed"] is True


def test_report_names_mixed_tail_and_exact_problem_point() -> None:
    data = evidence()
    report = render_report(data, make_decision(data))
    assert "两个物理问题点已定位" in report
    assert "第一段 FC1 收尾阶段" in report
    assert "在各自 producer 后被逐步保存" in report
    assert "activation 入口被保存" in report
    assert "5` 个 second-pass accumulator register values" in report
    assert "8` 个 index/address scalar" in report
    assert "1` 个 control scalar" in report
    assert "Main 108 的物理形成机制已闭合" in report
    assert "Tail 14 的物理形成机制已闭合" in report
    assert "高置信推断" in report
    assert "完整源码级 root cause 不宣称已严格证明" in report
    assert "性能因果尚未建立" in report
    assert "假设（不是结论）" in report


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
    assert set(decision["formation_causes"]) == {
        "main_108_word_bundle",
        "tail_14_word_bundle",
    }
    assert set(decision["source_interpretation"]) == {
        "main_108_word_bundle",
        "tail_14_word_bundle",
    }


@pytest.mark.parametrize(
    ("mutator", "match"),
    (
        (
            lambda data: data["root_cause"].__setitem__(
                "main_bundle_status", "mechanism_localized"
            ),
            "root-cause closure gate",
        ),
        (
            lambda data: data["main_108_word_bundle"].__setitem__(
                "reload_first_use_is_scale_fmul_count", 107
            ),
            "root-cause closure gate",
        ),
        (
            lambda data: data["cross_layer_semantics"]["baseline_ptx"].__setitem__(
                "internal_accumulator_def_use_closed", False
            ),
            "root-cause closure gate",
        ),
        (
            lambda data: data["main_108_word_bundle"].__setitem__(
                "physical_chain_closed", False
            ),
            "root-cause closure gate",
        ),
        (
            lambda data: data["tail_14_word_bundle"].__setitem__(
                "value_class_counts", {"wrong": 14}
            ),
            "root-cause closure gate",
        ),
        (
            lambda data: data["up_first_observation"].__setitem__(
                "main_108_word_bundle_preserved", False
            ),
            "root-cause closure gate",
        ),
        (
            lambda data: data["up_first_observation"].__setitem__(
                "tail_14_word_bundle_eliminated", False
            ),
            "root-cause closure gate",
        ),
        (
            lambda data: data["up_first_observation"].__setitem__(
                "tensor_work_identity_pass", False
            ),
            "root-cause closure gate",
        ),
        (
            lambda data: data["up_first_observation"].__setitem__(
                "local_sector_reduction_per_direction", 568_063
            ),
            "root-cause closure gate",
        ),
        (
            lambda data: data["up_first_observation"].__setitem__(
                "stack_bytes_per_thread", [488, 433]
            ),
            "root-cause closure gate",
        ),
        (
            lambda data: data["up_first_observation"].__setitem__(
                "stack_byte_reduction", 55
            ),
            "root-cause closure gate",
        ),
        (
            lambda data: data["up_first_observation"].__setitem__(
                "executed_local_instruction_reduction_per_direction", 142_015
            ),
            "root-cause closure gate",
        ),
    ),
)
def test_root_cause_closure_is_fail_closed(mutator, match: str) -> None:
    data = copy.deepcopy(evidence())
    mutator(data)
    with pytest.raises(ValueError, match=match):
        make_decision(data)


def test_manifest_clean_rebuild_is_deterministic_and_has_no_active_legacy_gates(
    tmp_path: Path,
) -> None:
    results = copy_manifest_inputs(tmp_path / "results")
    data = read_json(results / "spill_root_cause_evidence.json")
    decision = make_decision(data)
    first = build_manifest(results, data, decision)
    second = build_manifest(results, data, decision)
    assert first == second
    assert validate_manifest_schema(first) == []
    assert "candidate_gates" not in first
    assert "candidate_gates" not in first["static_spill"]
    assert "attribution" not in first
    assert first["retained_attribution_evidence"]["gate_pass"] is False
    assert first["correctness"]["strict_cross_arm_gate_pass"] is False
    assert set(first["arms"]) == {"baseline", "up_first_attribution"}
    assert first["capture_provenance"]["capture_family"] == "renumbered_exp004"


def test_manifest_accepts_complete_native_exp003_capture_family(tmp_path: Path) -> None:
    results = copy_manifest_inputs(tmp_path / "results")
    schema_updates = {
        "arms/baseline/preparation.json": (
            "exp003.spill-root-cause.arm-preparation.v1"
        ),
        "arms/up_first_attribution/preparation.json": (
            "exp003.spill-root-cause.arm-preparation.v1"
        ),
        "overlays/identity.json": "exp003.spill-root-cause.overlays.v1",
        "static_spill_evidence.json": (
            "exp003.spill-root-cause.static-spill-evidence.v1"
        ),
        "ncu/spill_evidence.json": (
            "exp003.spill-root-cause.ncu-spill-evidence.v1"
        ),
        "correctness/up_first_attribution.json": (
            "exp003.spill-root-cause.correctness.v1"
        ),
        "attribution_evidence.json": (
            "exp003.spill-root-cause.attribution-evidence.v1"
        ),
    }
    for relative, schema in schema_updates.items():
        path = results / relative
        record = read_json(path)
        record["schema"] = schema
        path.write_text(json.dumps(record))
    evidence_path = results / "spill_root_cause_evidence.json"
    data = read_json(evidence_path)
    data["provenance"]["input_files"]["static_evidence"]["sha256"] = file_sha256(
        results / "static_spill_evidence.json"
    )
    data["provenance"]["input_files"]["ncu_evidence"]["sha256"] = file_sha256(
        results / "ncu" / "spill_evidence.json"
    )
    evidence_path.write_text(json.dumps(data))
    manifest = build_manifest(results, data, make_decision(data))
    assert validate_manifest_schema(manifest) == []
    assert manifest["capture_provenance"]["capture_family"] == "native_exp003"
    assert manifest["capture_provenance"]["legacy_experiment_id"] is None
    assert manifest["capture_provenance"]["legacy_dangling_references"] == []


def test_manifest_rejects_mixed_capture_schema_family(tmp_path: Path) -> None:
    results = copy_manifest_inputs(tmp_path / "results")
    path = results / "correctness" / "up_first_attribution.json"
    record = read_json(path)
    record["schema"] = "exp003.spill-root-cause.correctness.v1"
    path.write_text(json.dumps(record))
    data = read_json(results / "spill_root_cause_evidence.json")
    with pytest.raises(ValueError, match="mixed or unsupported capture schema family"):
        build_manifest(results, data, make_decision(data))


def test_manifest_rejects_preparation_ncu_cubin_drift(tmp_path: Path) -> None:
    results = copy_manifest_inputs(tmp_path / "results")
    path = results / "ncu" / "spill_evidence.json"
    ncu = read_json(path)
    ncu["inputs"]["baseline"]["cubin_sha256"] = "0" * 64
    path.write_text(json.dumps(ncu))
    evidence_path = results / "spill_root_cause_evidence.json"
    data = read_json(evidence_path)
    data["provenance"]["input_files"]["ncu_evidence"]["sha256"] = file_sha256(path)
    evidence_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="preparation/NCU cubin identity drift"):
        build_manifest(results, data, make_decision(data))


def test_manifest_rejects_root_cause_upstream_hash_drift(tmp_path: Path) -> None:
    results = copy_manifest_inputs(tmp_path / "results")
    path = results / "static_spill_evidence.json"
    path.write_text(path.read_text() + " ")
    data = read_json(results / "spill_root_cause_evidence.json")
    with pytest.raises(ValueError, match="upstream identity drift at static_evidence"):
        build_manifest(results, data, make_decision(data))


def test_manifest_rejects_up_first_ncu_identity_drift(tmp_path: Path) -> None:
    results = copy_manifest_inputs(tmp_path / "results")
    evidence_path = results / "spill_root_cause_evidence.json"
    data = read_json(evidence_path)
    data["identity"]["up_first_ncu_sha256"] = "0" * 64
    evidence_path.write_text(json.dumps(data))
    with pytest.raises(
        ValueError, match="root-cause evidence identity drift at up_first_ncu_sha256"
    ):
        build_manifest(results, data, make_decision(data))


def test_manifest_rejects_resource_cleanup_failure(tmp_path: Path) -> None:
    results = copy_manifest_inputs(tmp_path / "results")
    cleanup_path = results / "resource_cleanup.json"
    cleanup = read_json(cleanup_path)
    cleanup["owned_container_absent"] = False
    cleanup_path.write_text(json.dumps(cleanup))
    data = read_json(results / "spill_root_cause_evidence.json")
    with pytest.raises(ValueError, match="resource cleanup gate failed"):
        build_manifest(results, data, make_decision(data))


def test_successful_closed_report_generation_exits_zero(tmp_path: Path) -> None:
    results = copy_manifest_inputs(tmp_path / "results")
    assert main(["--results", str(results)]) == 0
    manifest = read_json(results / "validation.manifest.json")
    assert manifest["status"] == "formation_mechanism_closed"
    assert manifest["decision"]["formal_experiment_closed"] is True
    assert (
        (results / "result.md")
        .read_text()
        .startswith("# exp_003：Fused Register Spill 成因分析")
    )
