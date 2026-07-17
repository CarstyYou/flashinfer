#!/usr/bin/env python3
"""Render exp_003 register-spill root-cause evidence without causal overreach."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from exp003_common import (
    DEFAULT_RESULTS,
    file_sha256,
    read_json,
    validate_manifest_schema,
    write_json,
)


def make_decision(evidence: Mapping[str, Any]) -> dict[str, Any]:
    cause = evidence["root_cause"]
    main = evidence["main_108_word_bundle"]
    tail = evidence["tail_14_word_bundle"]
    up_first = evidence["up_first_observation"]
    cross = evidence["cross_layer_semantics"]
    expected_bundles = {"main_108_word_bundle", "tail_14_word_bundle"}
    if set(cause["physical_locations"]) != expected_bundles:
        raise ValueError("exp_003 must identify both physical spill bundles")
    if set(cause["physical_mechanisms"]) != expected_bundles:
        raise ValueError("exp_003 must explain both physical spill mechanisms")
    if set(cause["formation_causes"]) != expected_bundles:
        raise ValueError("exp_003 must explain why both spill bundles form")
    if set(cause["source_interpretation"]) != expected_bundles:
        raise ValueError("exp_003 must state the source-attribution boundary")
    for field in ("physical_mechanisms", "formation_causes", "source_interpretation"):
        if not all(
            isinstance(cause[field][bundle], str) and cause[field][bundle].strip()
            for bundle in expected_bundles
        ):
            raise ValueError(f"exp_003 {field} must contain non-empty explanations")
    if cause["production_optimization_recommendation_allowed"] is not False:
        raise ValueError("exp_003 must not emit a production optimization")
    main_gate = (
        main["stack_word_count"] == 108
        and main["omma_output_vector_count"] == 27
        and main["stl64_instruction_count"] == 54
        and main["ldl_reload_count"] == 108
        and main["reload_first_use_is_scale_fmul_count"] == 108
        and main["physical_chain_closed"] is True
        and len(main["store_stack_slots"]) == 108
        and len(set(main["store_stack_slots"])) == 108
        and main["store_stack_slots"] == main["reload_stack_slots"]
        and main["value_class"] == "first_pass_accumulator"
        and cause["registers_per_thread"] == 255
        and cause["main_bundle_status"] == "physical_formation_mechanism_closed"
    )
    source_lines = cross["source"]["semantic_lines"]
    cross_layer_gate = (
        {
            "gate_acc_alloc",
            "up_acc_alloc",
            "gate_gemm_phase",
            "up_gemm_phase",
            "activation_call",
        }
        <= set(source_lines)
        and cross["baseline_mlir"]["internal_accumulator_def_use_closed"] is True
        and cross["baseline_ptx"]["internal_accumulator_def_use_closed"] is True
        and cross["compiler_certified_source_value_to_stack_slot_map"] is False
        and cross["compiler_certified_virtual_to_physical_register_map"] is False
        and cause["source_attribution_status"]
        == "high_confidence_program_order_inference_not_compiler_certified"
    )
    tail_gate = (
        tail["stack_word_count"] == 14
        and tail["stack_bytes_per_thread"] == 56
        and tail["value_class_counts"]
        == {
            "index_address_scalar": 8,
            "long_lived_control_scalar": 1,
            "second_pass_accumulator": 5,
        }
        and len(tail["chains"]) == 14
        and all(row["physical_chain_closed"] for row in tail["chains"])
        and cause["tail_physical_status"] == "physical_formation_mechanism_closed"
        and cause["tail_source_value_status"] == "partially_semantic_localized"
        and cause["tail_disposition"] == "deferred_reprofile_after_main_change"
    )
    up_first_gate = (
        up_first["main_108_word_bundle_preserved"] is True
        and up_first["tail_14_word_bundle_eliminated"] is True
        and up_first["stack_bytes_per_thread"] == [488, 432]
        and up_first["stack_byte_reduction"] == 56
        and up_first["local_sector_reduction_per_direction"] == 568_064
        and up_first["executed_local_instruction_reduction_per_direction"] == 142_016
        and up_first["tensor_work_identity_pass"] is True
    )
    if not main_gate or not cross_layer_gate or not tail_gate or not up_first_gate:
        raise ValueError("exp_003 root-cause closure gate failed")
    return {
        "experiment_goal": "spill_root_cause_analysis",
        "severity": "P0_hard_failure_by_project_policy",
        "physical_problem_points": dict(cause["physical_locations"]),
        "physical_mechanisms": dict(cause["physical_mechanisms"]),
        "formation_causes": dict(cause["formation_causes"]),
        "source_interpretation": dict(cause["source_interpretation"]),
        "source_attribution_status": cause["source_attribution_status"],
        "physical_mechanism_status": "all_bundles_closed",
        "main_108_status": "physical_formation_mechanism_closed",
        "tail_14_status": "physical_formation_mechanism_closed_source_identity_partial",
        "tail_14_disposition": cause["tail_disposition"],
        "overall_root_cause_status": cause["overall_status"],
        "latency_causality": "not_tested_and_not_required_for_P0",
        "tc_cadence_hypothesis": "unverified_spill_may_be_primary_contributor",
        "production_optimization_recommendation_allowed": False,
        "followup_experiment_allowed": True,
        "followup_scope": "main_108_live_range_mechanism",
        "formal_experiment_closed": True,
        "closure_kind": (
            "physical_formation_mechanisms_closed_with_source_attribution_limits"
        ),
    }


def correctness_summary(
    results: Path,
    baseline: Mapping[str, Any],
    up_first: Mapping[str, Any],
) -> dict[str, Any]:
    path = results / "correctness" / "up_first_attribution.json"
    record = read_json(path)
    if record.get("candidate") != "up_first_attribution":
        raise ValueError("unexpected correctness evidence candidate")
    if record.get("fixture_identity_pass") is not True:
        raise ValueError("diagnostic attribution fixture identity failed")
    if record.get("quant_aware_oracle_pass") is not True:
        raise ValueError("diagnostic attribution oracle failed")
    strict = record.get("strict_candidate_gate", {})
    if record.get("gate_pass") is not False or strict.get("gate_pass") is not False:
        raise ValueError("expected retained strict cross-arm correctness failure")
    if strict.get("baseline_self_drift_within_hard_caps") is not False:
        raise ValueError("expected retained baseline self-drift hard-cap failure")
    if baseline["fixture"] != up_first["fixture"]:
        raise ValueError("diagnostic attribution fixture drift")
    return {
        "status": "diagnostic_only_strict_cross_arm_invalid",
        "path": "correctness/up_first_attribution.json",
        "sha256": file_sha256(path),
        "fixture_identity_pass": True,
        "quant_aware_oracle_pass": True,
        "strict_cross_arm_gate_pass": False,
        "baseline_self_drift_within_hard_caps": False,
        "scope": (
            "structural source/program-order attribution only; not formal equivalence, "
            "not latency evidence"
        ),
    }


def build_manifest(
    results: Path, evidence: Mapping[str, Any], decision: Mapping[str, Any]
) -> dict[str, Any]:
    evidence_path = results / "spill_root_cause_evidence.json"
    cause = evidence["root_cause"]
    if read_json(evidence_path) != evidence:
        raise ValueError(
            "in-memory root-cause evidence differs from the evidence file"
        )
    if evidence.get("schema") != "exp003.spill-root-cause.root-cause-evidence.v1":
        raise ValueError("unexpected root-cause evidence schema")

    preparations = {
        arm: read_json(results / "arms" / arm / "preparation.json")
        for arm in ("baseline", "up_first_attribution")
    }
    baseline = preparations["baseline"]
    up_first = preparations["up_first_attribution"]
    for arm, preparation in preparations.items():
        if preparation.get("schema") not in {
            "exp004.arm-preparation.v1",
            "exp003.spill-root-cause.arm-preparation.v1",
        }:
            raise ValueError(f"{arm}: unexpected preparation schema")
        if preparation.get("status") != "complete" or preparation.get("arm") != arm:
            raise ValueError(f"{arm}: incomplete or mismatched preparation")
    for field in ("case", "fixture", "weights", "reference_sha256", "launch"):
        if baseline[field] != up_first[field]:
            raise ValueError(f"comparison identity drift at {field}")

    stable_runtime_fields = (
        "cuda_runtime",
        "driver",
        "gpu",
        "image_digest",
        "nvcc",
        "ptxas",
        "python",
        "python_deps_sha256",
    )
    environment = {field: baseline["runtime"][field] for field in stable_runtime_fields}
    for field in stable_runtime_fields:
        if baseline["runtime"][field] != up_first["runtime"][field]:
            raise ValueError(f"runtime identity drift at {field}")

    overlay_identity_path = results / "overlays" / "identity.json"
    overlay_identity = read_json(overlay_identity_path)
    if overlay_identity.get("schema") not in {
        "exp004.overlays.v1",
        "exp003.spill-root-cause.overlays.v1",
    }:
        raise ValueError("unexpected overlay identity schema")

    def artifact(preparation: Mapping[str, Any], suffix: str) -> Mapping[str, Any]:
        matches = [
            item
            for item in preparation["jit_artifacts"]
            if item["path"].endswith(suffix) and "hardware_info" not in item["path"]
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one JIT artifact ending in {suffix!r}")
        return matches[0]

    ncu = read_json(results / "ncu" / "spill_evidence.json")
    static = read_json(results / "static_spill_evidence.json")
    upstream_hashes = {
        "static_evidence": file_sha256(results / "static_spill_evidence.json"),
        "ncu_evidence": file_sha256(results / "ncu" / "spill_evidence.json"),
    }
    for label, actual in upstream_hashes.items():
        recorded = evidence["provenance"]["input_files"][label]["sha256"]
        if recorded != actual:
            raise ValueError(f"root-cause upstream identity drift at {label}")
    arm_records: dict[str, Any] = {}
    for arm, preparation in preparations.items():
        identity = overlay_identity["arms"][arm]
        runtime_source = preparation["runtime"]["source"]
        if identity["overlay_sha256"] != runtime_source["overlay_sha256"]:
            raise ValueError(f"{arm}: overlay SHA drift")
        if identity["production_sha256"] != runtime_source["production_kernel_sha256"]:
            raise ValueError(f"{arm}: production source SHA drift")
        cubin = artifact(preparation, ".cubin")
        ptx = artifact(preparation, ".ptx")
        mlir = artifact(preparation, "_clean.mlir")
        ncu_input = ncu["inputs"][arm]
        if cubin["sha256"] != ncu_input["cubin_sha256"]:
            raise ValueError(f"{arm}: preparation/NCU cubin identity drift")
        arm_records[arm] = {
            "preparation": f"arms/{arm}/preparation.json",
            "preparation_sha256": file_sha256(
                results / "arms" / arm / "preparation.json"
            ),
            "jit_artifact_set_sha256": preparation["jit_artifact_set_sha256"],
            "overlay_sha256": identity["overlay_sha256"],
            "overlay_diff_sha256": identity["diff_sha256"],
            "cubin_sha256": cubin["sha256"],
            "ptx_sha256": ptx["sha256"],
            "mlir_sha256": mlir["sha256"],
            "ncu_trace_sha256": ncu_input["trace_rep_sha256"],
            "launch": preparation["launch"],
        }

    expected_identity = evidence["identity"]
    identity_checks = {
        "baseline_cubin_sha256": arm_records["baseline"]["cubin_sha256"],
        "up_first_cubin_sha256": arm_records["up_first_attribution"]["cubin_sha256"],
        "baseline_ncu_sha256": arm_records["baseline"]["ncu_trace_sha256"],
        "up_first_ncu_sha256": arm_records["up_first_attribution"][
            "ncu_trace_sha256"
        ],
    }
    for key, actual in identity_checks.items():
        if expected_identity[key] != actual:
            raise ValueError(f"root-cause evidence identity drift at {key}")

    source = baseline["runtime"]["source"]
    if (
        evidence["cross_layer_semantics"]["source"]["sha256"]
        != source["production_kernel_sha256"]
    ):
        raise ValueError("root-cause source SHA does not match preparation")
    if (
        evidence["cross_layer_semantics"]["baseline_mlir"]["sha256"]
        != arm_records["baseline"]["mlir_sha256"]
    ):
        raise ValueError("baseline MLIR identity drift")
    if (
        evidence["cross_layer_semantics"]["baseline_ptx"]["sha256"]
        != arm_records["baseline"]["ptx_sha256"]
    ):
        raise ValueError("baseline PTX identity drift")
    if (
        evidence["cross_layer_semantics"]["up_first_ptx"]["sha256"]
        != arm_records["up_first_attribution"]["ptx_sha256"]
    ):
        raise ValueError("up-first PTX identity drift")

    cleanup_path = results / "resource_cleanup.json"
    cleanup = read_json(cleanup_path)
    cleanup_checks = (
        "lease_absent_after_release",
        "owned_compute_process_absent",
        "owned_container_absent",
    )
    if not all(cleanup.get(field) is True for field in cleanup_checks):
        raise ValueError("resource cleanup gate failed")
    attribution_path = results / "attribution_evidence.json"
    attribution = read_json(attribution_path)
    if (
        attribution.get("formal_verdict")
        != "source_and_program_order_inference_only"
        or attribution.get("gate_pass") is not False
    ):
        raise ValueError("retained attribution boundary drift")
    correctness_record = read_json(
        results / "correctness" / "up_first_attribution.json"
    )
    consumed_capture_schemas = {
        *(preparation["schema"] for preparation in preparations.values()),
        overlay_identity["schema"],
        static["schema"],
        ncu["schema"],
        correctness_record["schema"],
        attribution["schema"],
    }
    schema_families = {
        "renumbered_exp004": {
            "exp004.arm-preparation.v1",
            "exp004.overlays.v1",
            "exp004.static-spill-evidence.v1",
            "exp004.ncu-spill-evidence.v1",
            "exp004.correctness.v1",
            "exp004.attribution-evidence.v1",
        },
        "native_exp003": {
            "exp003.spill-root-cause.arm-preparation.v1",
            "exp003.spill-root-cause.overlays.v1",
            "exp003.spill-root-cause.static-spill-evidence.v1",
            "exp003.spill-root-cause.ncu-spill-evidence.v1",
            "exp003.spill-root-cause.correctness.v1",
            "exp003.spill-root-cause.attribution-evidence.v1",
        },
    }
    matching_families = [
        name
        for name, expected_schemas in schema_families.items()
        if consumed_capture_schemas == expected_schemas
    ]
    if len(matching_families) != 1:
        raise ValueError(
            "mixed or unsupported capture schema family: "
            + ", ".join(sorted(consumed_capture_schemas))
        )
    capture_family = matching_families[0]
    retained_legacy_capture = capture_family == "renumbered_exp004"
    manifest = {
        "schema": "exp003.spill-root-cause.validation-manifest.v1",
        "status": "formation_mechanism_closed",
        "capture_provenance": {
            "capture_family": capture_family,
            "legacy_experiment_id": "exp004" if retained_legacy_capture else None,
            "consumed_capture_schemas": sorted(consumed_capture_schemas),
            "policy": (
                "retained capture schemas, paths, lease IDs, commands, and hashes are "
                "historical identities and are not rewritten during renumbering"
                if retained_legacy_capture
                else "all consumed capture artifacts use the native exp003 schema family"
            ),
            "legacy_dangling_references": (
                [
                    {
                        "artifact": "static_spill_evidence.json:h108_boundary",
                        "historical_target": "spill_localization_evidence.json",
                        "canonical_replacement": "spill_root_cause_evidence.json",
                        "reason": "historical derived artifact retained byte-for-byte",
                    }
                ]
                if retained_legacy_capture
                else []
            ),
        },
        "case": baseline["case"],
        "environment": environment,
        "source": {
            "checkout_head": source["checkout_head"],
            "locked_source_commit": source["locked_source_commit"],
            "cutlass_commit": source["cutlass_commit"],
            "production_kernel_sha256": source["production_kernel_sha256"],
            "overlay_identity": "overlays/identity.json",
            "overlay_identity_sha256": file_sha256(overlay_identity_path),
        },
        "fixture": baseline["fixture"],
        "arms": arm_records,
        "correctness": correctness_summary(results, baseline, up_first),
        "benchmark": {
            "status": "not_part_of_current_question",
            "reason": (
                "exp_003 closes the spill formation-mechanism goal; project policy marks "
                "register spill P0 without a latency proof"
            ),
        },
        "static_spill": {
            "path": "static_spill_evidence.json",
            "sha256": upstream_hashes["static_evidence"],
        },
        "ncu": {
            "path": "ncu/spill_evidence.json",
            "sha256": upstream_hashes["ncu_evidence"],
        },
        "root_cause_evidence": {
            "path": "spill_root_cause_evidence.json",
            "sha256": file_sha256(evidence_path),
            "physical_locations": cause["physical_locations"],
            "physical_mechanisms": cause["physical_mechanisms"],
            "formation_causes": cause["formation_causes"],
            "source_interpretation": cause["source_interpretation"],
            "source_attribution_status": cause["source_attribution_status"],
            "overall_status": cause["overall_status"],
        },
        "decision": dict(decision),
        "resource_cleanup": {
            "path": "resource_cleanup.json",
            "sha256": file_sha256(cleanup_path),
            "gate_pass": True,
        },
        "retained_attribution_evidence": {
            "status": "diagnostic_only",
            "formal_verdict": attribution["formal_verdict"],
            "gate_pass": attribution["gate_pass"],
            "path": "attribution_evidence.json",
            "sha256": file_sha256(attribution_path),
        },
    }
    errors = validate_manifest_schema(manifest)
    if errors:
        raise ValueError("invalid root-cause manifest: " + "; ".join(errors))
    return manifest


def pc(value: int) -> str:
    return f"`0x{value:x}`"


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def render_report(evidence: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    identity = evidence["identity"]
    main = evidence["main_108_word_bundle"]
    tail = evidence["tail_14_word_bundle"]
    up_first = evidence["up_first_observation"]
    cross = evidence["cross_layer_semantics"]
    rows = [
        "# exp_003：Fused Register Spill 成因分析",
        "",
        "## 结论",
        "",
        "- **Register spill 是本项目的 P0 hard failure。** 该判定来自工程约束，不需要先证明 latency 影响。",
        "- **两个物理问题点已定位：** 第一段 FC1 收尾阶段，`108 words/lane` 主块中的已完成 accumulator vectors 在各自 producer 后被逐步保存，并跨完整第二段 FC1 保活到 activation；`14 words/lane` 尾块在 activation 入口被保存，物理寄存器由 activation 临时值复用，随后再恢复。",
        "- 主块是 first-pass accumulator；尾块不是同质 accumulator，而是 `5` 个 second-pass accumulator register values、`8` 个 index/address scalar 和 `1` 个 control scalar。",
        "- **Main 108 的物理形成机制已闭合：** first-pass FP32 accumulator 跨完整 second-pass FC1 保活并与后续 working set 重叠；在观测到的 `255 registers/thread` allocation 下，compiler 将 108 words/lane 暂存到 local memory，并在 activation 前后恢复。",
        "- **Baseline 的 Gate/Up 归属是高置信推断：** source/IR program order 表明 first pass 是 Gate、second pass 是 Up，因此 Main 对应 `gate_acc` 跨 Up FC1 保活；但没有 compiler-certified `SSA → physical register/stack slot` 映射。",
        "- **Tail 14 的物理形成机制已闭合：** activation 入口处，每个参与计算的 lane 仍有 5 个 second-pass accumulator register values 与 9 个 index/address/control scalar；activation temporaries 复用其物理寄存器，allocator 因而执行 save/reuse/restore。Baseline source order 将 second pass 解释为 Up，同样不是 compiler-certified slot attribution。",
        "- **实验按 scope 正式收口：** physical spill mechanism closed；source/program-order formation model supported；9 个 scalar 的唯一 source SSA 延期。完整源码级 root cause 不宣称已严格证明。",
        "- **性能因果尚未建立：** 本实验不证明 spill 影响 latency 或 TC cadence，也不提出 production optimization。",
        "",
        "## 1. 证据身份",
        "",
        "| 项目 | 身份 |",
        "|---|---|",
        "| Kernel / launch | `MoEDynamicKernel`, grid `1×1×110`, block `160×1×1` |",
        f"| Baseline cubin | `{identity['baseline_cubin_sha256']}` |",
        f"| Baseline SASS | `{identity['baseline_sass_sha256']}` |",
        f"| Baseline NCU | `{identity['baseline_ncu_sha256']}` |",
        f"| Up-first cubin | `{identity['up_first_cubin_sha256']}` |",
        f"| Up-first SASS | `{identity['up_first_sass_sha256']}` |",
        "",
        "## 2. Spill 发生在哪里",
        "",
        "### 2.1 Main 108-word bundle",
        "",
        f"`{main['omma_output_vector_count']}×OMMA output vectors → "
        f"{main['stl64_instruction_count']}×STL.64 → "
        f"{main['stack_word_count']} stack words → "
        f"{main['ldl_reload_count']}×LDL → "
        f"{main['reload_first_use_is_scale_fmul_count']}/"
        f"{main['stack_word_count']} reloads first used by scale FMUL`",
        "",
        "代表物理链：",
        "",
        "```text",
        "OMMA @0x7810 produces R12..R15",
        "  → STL.64 @0x7860/@0x7880 saves stack[0x170..0x17c]",
        "  → LDL @0xba90..0xbac0 restores R203..R200",
        "  → gate scale/sigmoid @0xbce0..0xbd80",
        "  → up scale @0xbd90",
        "  → SwiGLU product @0xbda0",
        "```",
        "",
        f"Store PC 范围 `{main['store_pc_range'][0]}..{main['store_pc_range'][1]}`；"
        f"reload PC 范围 `{main['reload_pc_range'][0]}..{main['reload_pc_range'][1]}`。"
        "这把主 bundle 定位为第一段 FC1 收尾阶段随 producer 逐步保存、跨完整第二段 FC1 保活、在 activation 前后按需恢复的 first-pass accumulator。",
        "",
        "### 2.2 Tail 14-word bundle",
        "",
        "| Slot / reg | 原值 producer / 类别 | STL | activation 临时复用 | LDL 恢复 | 原值首个 consumer | 定位状态 |",
        "|---|---|---:|---|---:|---|---|",
    ]
    for item in tail["chains"]:
        reuse = "→".join(f"0x{entry['pc']:x}" for entry in item["temporary_reuse"])
        status = (
            "physical + accumulator semantic closed"
            if item["value_class"] == "second_pass_accumulator"
            else "physical closed；unique source SSA unresolved"
        )
        rows.append(
            f"| `{item['stack_slot_hex']}/{item['register']}` | "
            f"{item['producer_detail']} @ {pc(item['producer_pc'])} / "
            f"`{item['value_class']}` | {pc(item['store_pc'])} | `{reuse}` | "
            f"{pc(item['reload_pc'])} | {item['first_original_consumer_detail']} @ "
            f"{pc(item['first_original_consumer_pc'])} | {status} |"
        )
    rows.extend(
        [
            "",
            f"`up_first` 保留全部 108-word 主 bundle，但消除完整 14-word Tail 的 save/restore；stack 从 `{up_first['stack_bytes_per_thread'][0]} B/thread` 降到 `{up_first['stack_bytes_per_thread'][1]} B/thread`，NCU local sectors 每方向精确减少 `{up_first['local_sector_reduction_per_direction']:,}`。这支持 mixed activation-entry live set 机制，但不能把原因收窄成某一个源码构造。该 arm 仅作结构与 program-order attribution；strict cross-arm correctness gate 未通过，因此不用于 formal equivalence 或性能结论。",
            "",
            "## 3. 收口边界",
            "",
            "| 范围 | 状态 | 处理 |",
            "|---|---|---|",
            f"| Main 108 | **physical formation mechanism closed**：first-pass accumulator 跨 complete second pass 保活，SASS 108-word roundtrip 闭合；source order `gate_acc` {cross['source']['semantic_lines']['gate_acc_alloc']}，Gate FC1 {cross['source']['semantic_lines']['gate_gemm_phase']} → Up FC1 {cross['source']['semantic_lines']['up_gemm_phase']} → activation {cross['source']['semantic_lines']['activation_call']} | Gate/Up attribution = high-confidence program-order inference；允许进入 Main live-range 的下一轮受控实验 |",
            "| Tail 14 | **physical formation mechanism closed / source identity partial**：activation temporary reuse 与 14 条 producer→store→reuse→reload→consumer 全闭合；包含 5 个 second-pass accumulator register values 和 9 个 scalar | Up attribution = program-order inference；9 个 scalar 身份延期，Main 改动后重新 profile |",
            "| Latency / TC cadence | 未测试因果 | 不作为本实验结论 |",
            "",
            "收口状态：**两个 spill pressure peak 的物理形成机制均闭合；源码归属为高置信 program-order inference；Tail source identity residual 显式延期，exp_003 按该 scope 完成。**",
            "",
            "## 4. 后续待验证假设",
            "",
            "**假设（不是结论）：register spill 是 TC cadence 偏低的主要贡献者。**",
            "",
            "下一实验需要 correctness-equivalent 的 reduced/no-spill arm，并锁住 Tensor work、launch topology 与 task schedule；随后同时比较 stack/local traffic、latency、TC subpipe active、Issue Active 和 warp stalls。没有 matched counterfactual 前，不得把相关性写成因果。",
            "",
            "已删除的旧 cadence 实验不保留任何性能结论：IKET marker 将 stack 从 `488` 改成 `432 B/thread`，测量本身改变了 spill。以后 cadence 插桩必须先通过完整 resource/SASS identity gate。",
            "",
        ]
    )
    report = "\n".join(rows)
    banned = (
        "优先调查 FC1 pass order",
        "缩短双 FP32 accumulator overlap lifetime",
        "H14 criticality",
        "H108 criticality",
    )
    if any(term in report for term in banned):
        raise AssertionError(
            "report contains a superseded or unsupported recommendation"
        )
    if decision["production_optimization_recommendation_allowed"] is not False:
        raise AssertionError("report decision illegally permits optimization")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)
    results = args.results.resolve()
    evidence = read_json(results / "spill_root_cause_evidence.json")
    decision = make_decision(evidence)
    manifest = build_manifest(results, evidence, decision)
    report = render_report(evidence, decision)
    write_text_atomic(results / "result.md", report)
    write_json(results / "validation.manifest.json", manifest)
    # Generation succeeds only after the scoped root-cause closure gates pass.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
