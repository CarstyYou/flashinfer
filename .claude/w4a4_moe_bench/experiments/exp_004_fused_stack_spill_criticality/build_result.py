#!/usr/bin/env python3
"""Render exp_004 spill problem-point evidence without causal overreach."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from exp004_common import (
    DEFAULT_RESULTS,
    file_sha256,
    read_json,
    validate_manifest_schema,
    write_json,
)


def make_decision(evidence: Mapping[str, Any]) -> dict[str, Any]:
    point = evidence["problem_point"]
    expected_bundles = {"main_108_word_bundle", "tail_14_word_bundle"}
    if set(point["physical_locations"]) != expected_bundles:
        raise ValueError("exp_004 must localize both physical spill bundles")
    if set(point["physical_mechanisms"]) != expected_bundles:
        raise ValueError("exp_004 must explain both physical spill mechanisms")
    if point["optimization_recommendation_allowed"] is not False:
        raise ValueError(
            "exp_004 must not emit an optimization before source localization closes"
        )
    return {
        "experiment_goal": "spill_problem_point_localization",
        "severity": "P0_hard_failure_by_project_policy",
        "physical_problem_points": dict(point["physical_locations"]),
        "physical_mechanisms": dict(point["physical_mechanisms"]),
        "physical_localization_status": "mechanism_localized",
        "source_value_localization_status": point["tail_source_value_status"],
        "overall_localization_status": point["overall_status"],
        "latency_causality": "not_tested_and_not_required_for_P0",
        "optimization_recommendation_allowed": False,
        "formal_experiment_closed": False,
        "closure_kind": "source_value_and_virtual_to_physical_bridge_unresolved",
    }


def build_manifest(
    results: Path, evidence: Mapping[str, Any], decision: Mapping[str, Any]
) -> dict[str, Any]:
    evidence_path = results / "spill_localization_evidence.json"
    if read_json(evidence_path) != evidence:
        raise ValueError(
            "in-memory localization evidence differs from the evidence file"
        )

    preparations = {
        arm: read_json(results / "arms" / arm / "preparation.json")
        for arm in ("baseline", "up_first_attribution")
    }
    baseline = preparations["baseline"]
    up_first = preparations["up_first_attribution"]
    for arm, preparation in preparations.items():
        if preparation.get("schema") != "exp004.arm-preparation.v1":
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
    if overlay_identity.get("schema") != "exp004.overlays.v1":
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
    }
    for key, actual in identity_checks.items():
        if expected_identity[key] != actual:
            raise ValueError(f"localization evidence identity drift at {key}")

    source = baseline["runtime"]["source"]
    if (
        evidence["cross_layer_semantics"]["source"]["sha256"]
        != source["production_kernel_sha256"]
    ):
        raise ValueError("localization source SHA does not match preparation")
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
    superseded_path = results / "attribution_evidence.json"
    manifest = {
        "schema": "exp004.validation-manifest.v2",
        "status": "localization_partial",
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
        "correctness": {
            "identity_equal": True,
            "baseline_formal_outputs_pass": all(
                item["formal_pass"] for item in baseline["outputs"]
            ),
            "up_first_formal_outputs_pass": all(
                item["formal_pass"] for item in up_first["outputs"]
            ),
            "scope": "artifact counterfactual only; no latency comparison",
        },
        "benchmark": {
            "status": "not_part_of_current_question",
            "reason": (
                "exp_004 localizes the spill problem point; project policy marks "
                "register spill P0 without a latency proof"
            ),
        },
        "static_spill": {
            "path": "static_spill_evidence.json",
            "sha256": file_sha256(results / "static_spill_evidence.json"),
        },
        "ncu": {
            "path": "ncu/spill_evidence.json",
            "sha256": file_sha256(results / "ncu" / "spill_evidence.json"),
        },
        "problem_point_localization": {
            "path": "spill_localization_evidence.json",
            "sha256": file_sha256(evidence_path),
            "physical_locations": evidence["problem_point"]["physical_locations"],
            "physical_mechanisms": evidence["problem_point"]["physical_mechanisms"],
            "overall_status": evidence["problem_point"]["overall_status"],
        },
        "decision": dict(decision),
        "resource_cleanup": {
            "path": "resource_cleanup.json",
            "sha256": file_sha256(cleanup_path),
        },
        "superseded_attribution_run": {
            "status": "superseded_by_problem_point_localization",
            "path": "attribution_evidence.json",
            "sha256": file_sha256(superseded_path),
        },
    }
    errors = validate_manifest_schema(manifest)
    if errors:
        raise ValueError("invalid localization manifest: " + "; ".join(errors))
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
    cross = evidence["cross_layer_semantics"]
    rows = [
        "# exp_004：Fused Spill 问题点定位",
        "",
        "## 结论",
        "",
        "- **Register spill 是本项目的 P0 hard failure。** 该判定来自工程约束，不需要先证明 latency 影响。",
        "- **两个物理问题点已定位：** 第一段 FC1 收尾阶段，`108 words/lane` 主块中的已完成 accumulator vectors 在各自 producer 后被逐步保存，并跨完整第二段 FC1 保活到 activation；`14 words/lane` 尾块在 activation 入口被保存，物理寄存器由 activation 临时值复用，随后再恢复。",
        "- 主块是 first-pass accumulator；尾块不是同质 accumulator，而是 `5` 个 second-pass accumulator、`8` 个 index/address scalar 和 `1` 个 control scalar。",
        "- **源码问题点尚未完全闭合：** 缺少 MLIR/PTX virtual value 到 SASS physical register/stack slot 的 compiler-certified 映射，9 个 scalar 也没有唯一 source SSA。",
        "- **优化建议：无。** 在源码值与 allocator live interval 闭合前，任何 pass-order、lifetime 或调度改法都属于无证据推测。",
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
            "`up_first` 保留全部 108-word 主 bundle，但消除全部 14 个 scalar save/restore；stack 从 `488 B/thread` 降到 `432 B/thread`，NCU local sectors 每方向精确减少 `568,064`。这支持 mixed activation-entry live set 机制，但不能把原因收窄成某一个源码构造。",
            "",
            "## 3. 跨层闭合与边界",
            "",
            "| 层级 | 已闭合 | 未闭合 |",
            "|---|---|---|",
            f"| Python source | `gate_acc` {cross['source']['semantic_lines']['gate_acc_alloc']}；`up_acc` {cross['source']['semantic_lines']['up_acc_alloc']}；Gate/Up GEMM 与 activation phase | source variable 到具体 physical spill slot |",
            f"| MLIR | `%rmem_118/%rmem_119` 的 alloc、GEMM、activation def-use（关键行 {cross['baseline_mlir']['semantic_lines']['gate_rmem_alloc']}/{cross['baseline_mlir']['semantic_lines']['up_rmem_alloc']}/{cross['baseline_mlir']['semantic_lines']['gate_activation_load']}） | MLIR SSA 到 ptxas physical allocation |",
            f"| PTX | `%r2878/%r4955` 的 MMA def 与 activation use（关键行 {cross['baseline_ptx']['semantic_lines']['first_pass_mma_def']}/{cross['baseline_ptx']['semantic_lines']['second_pass_mma_def']}/{cross['baseline_ptx']['semantic_lines']['first_pass_activation_use']}） | PTX 无 spill local op 与 source location；virtual register 到 SASS register/slot |",
            "| SASS | 108-word 全量 roundtrip；14-word 每项 producer→store→temporary reuse→reload→consumer | 9 个 scalar 的唯一源码 SSA |",
            "",
            "因此当前状态是：**SASS physical mechanism-localized；source-value partially localized；实验尚未完全收口。**",
            "",
            "## 4. 下一步取证",
            "",
            "只补一类证据：同一编译身份下的 backend register-allocation/liveness dump，必须闭合：",
            "",
            "```text",
            "backend SSA/PTX virtual register",
            "  → physical register",
            "  → stack slot + spill/reload PC",
            "  → producer/consumer live interval",
            "  → source/MLIR location",
            "```",
            "",
            "单独增加 lineinfo 只能补 `PC→source line`，不足以闭合 value/register allocation。闭合前不进入优化设计。",
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
    if decision["optimization_recommendation_allowed"] is not False:
        raise AssertionError("report decision illegally permits optimization")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)
    results = args.results.resolve()
    evidence = read_json(results / "spill_localization_evidence.json")
    decision = make_decision(evidence)
    manifest = build_manifest(results, evidence, decision)
    report = render_report(evidence, decision)
    write_text_atomic(results / "result.md", report)
    write_json(results / "validation.manifest.json", manifest)
    # Generation succeeded. The unresolved source bridge is represented by the
    # manifest state and must not be conflated with a process failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
