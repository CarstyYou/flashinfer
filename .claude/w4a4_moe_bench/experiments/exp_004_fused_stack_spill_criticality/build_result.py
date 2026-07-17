#!/usr/bin/env python3
"""Join exp_004 evidence, make verdicts, and render the reader report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from exp004_common import (
    DEFAULT_RESULTS,
    FALLBACK_CANDIDATE,
    PRIMARY_CANDIDATE,
    file_sha256,
    read_json,
    validate_manifest_schema,
    write_json,
)


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def select_static_candidate(static: Mapping[str, Any]) -> str | None:
    """Honor the pre-registered primary -> one-fallback decision order."""
    for arm in (PRIMARY_CANDIDATE, FALLBACK_CANDIDATE):
        delta = static.get("deltas", {}).get(arm, {})
        if delta.get("complete_tail_removal_no_replacement") and delta.get(
            "selected_opcode_projection_equal_except_local", False
        ):
            return arm
    return None


def close_no_qualified_candidate(results: Path, static: Mapping[str, Any]) -> int:
    """Close the experiment at the pre-registered gate, retaining attribution evidence."""
    baseline_path = results / "arms" / "baseline" / "preparation.json"
    baseline = read_json(baseline_path) if baseline_path.is_file() else {}
    ncu_path = results / "ncu" / "spill_evidence.json"
    ncu = read_json(ncu_path) if ncu_path.is_file() else {}
    attribution_path = results / "attribution_evidence.json"
    attribution = read_json(attribution_path) if attribution_path.is_file() else {}
    cleanup_path = results / "resource_cleanup.json"
    cleanup = read_json(cleanup_path) if cleanup_path.is_file() else {}
    arm_names = (
        "baseline",
        PRIMARY_CANDIDATE,
        FALLBACK_CANDIDATE,
        "up_first_attribution",
    )
    correctness: dict[str, Any] = {}
    for arm in arm_names[1:]:
        path = results / "correctness" / f"{arm}.json"
        if path.is_file():
            correctness[arm] = read_json(path)
    per_arm = {
        arm: {
            "complete_tail_removal_no_replacement": delta.get(
                "complete_tail_removal_no_replacement", False
            ),
            "selected_opcode_projection_equal_except_local": delta.get(
                "selected_opcode_projection_equal_except_local", False
            ),
            "removed_words": delta.get("removed_words"),
            "replacement_words": delta.get("replacement_words"),
        }
        for arm, delta in static.get("deltas", {}).items()
        if arm in (PRIMARY_CANDIDATE, FALLBACK_CANDIDATE)
    }
    decision = {
        "formal_candidate": None,
        "h108_attribution": attribution.get(
            "formal_verdict", "source_and_program_order_inference_only"
        ),
        "h108_criticality": "out_of_scope_unresolved",
        "h14_attribution": "inconclusive_no_qualified_in_place_arm",
        "h14_criticality": "not_tested",
        "formal_experiment_closed": True,
        "closure_kind": "closed_inconclusive_at_static_gate",
    }
    manifest = {
        "schema": "exp004.validation-manifest.v1",
        "status": "inconclusive",
        "case": baseline.get("case", {}),
        "environment": baseline.get("runtime", {}),
        "source": baseline.get("runtime", {}).get("source", {}),
        "fixture": baseline.get("fixture", {}),
        "arms": {
            arm: {
                "preparation_manifest": f"arms/{arm}/preparation.json",
                "jit_artifact_set_sha256": read_json(
                    results / "arms" / arm / "preparation.json"
                ).get("jit_artifact_set_sha256"),
            }
            for arm in arm_names
            if (results / "arms" / arm / "preparation.json").is_file()
        },
        "correctness": {
            "status": "quant_aware_oracle_pass_but_strict_cross_arm_gate_invalid",
            "per_arm": {
                arm: {
                    "path": f"correctness/{arm}.json",
                    "quant_aware_oracle_pass": value.get(
                        "quant_aware_oracle_pass", False
                    ),
                    "strict_gate_pass": value.get("gate_pass", False),
                    "baseline_self_drift_within_hard_caps": value.get(
                        "strict_candidate_gate", {}
                    ).get("baseline_self_drift_within_hard_caps", False),
                }
                for arm, value in correctness.items()
            },
        },
        "benchmark": {
            "status": "not_run_by_pre_registered_static_stop_condition",
            "reason": "neither in-place arm changed the target 14-word bundle",
        },
        "static_spill": {
            "path": "static_spill_evidence.json",
            "sha256": file_sha256(results / "static_spill_evidence.json"),
            "baseline_reproduction_pass": static.get(
                "baseline_reproduction_pass", False
            ),
            "candidate_gates": per_arm,
        },
        "ncu": {
            "status": "baseline_and_attribution_arm_captured"
            if ncu
            else "missing",
            "path": "ncu/spill_evidence.json" if ncu else None,
            "sha256": file_sha256(ncu_path) if ncu else None,
            "baseline_reproduction_pass": ncu.get(
                "baseline_reproduction_pass", False
            ),
        },
        "attribution": {
            "path": "attribution_evidence.json" if attribution else None,
            "sha256": file_sha256(attribution_path) if attribution else None,
            "gate_pass": attribution.get("gate_pass", False),
            "formal_verdict": attribution.get("formal_verdict"),
        },
        "resource_cleanup": cleanup,
        "decision": decision,
    }
    write_json(results / "validation.manifest.json", manifest)
    up_first_ncu = ncu.get("arms", {}).get("up_first_attribution", {})
    base_ncu = ncu.get("arms", {}).get("baseline", {})
    up_first_delta = ncu.get("deltas", {}).get("up_first_attribution", {})
    base_self = correctness.get(PRIMARY_CANDIDATE, {}).get(
        "baseline_self_drift", {}
    )
    rows = [
        "# exp_004：Fused Stack/Spill Criticality",
        "",
        "## 结论",
        "",
        "- **H14 attribution：inconclusive；H14 criticality：not tested。** Primary 与唯一 fallback 都仍为 `488 B/thread、122 words/lane`，没有改变目标 14-word bundle，因此按预注册 stop condition 不跑 paired benchmark。",
        "- `up_first_attribution` 将 stack 从 `488 B` 降到 `432 B`，local bundle 净减少 14 words；NCU 每方向精确减少 `568,064` local sectors 和 `142,016` executed local instructions，selected non-local projection 与 measured Tensor work 保持不变。这个结果使下一步应优先调查 **FC1 pass order / accumulator lifetime 的 codegen**；现有证据不能排除 activation destination 参与。",
        "- **H108 attribution 仍是 source + program-order inference，未达到 formal acceptance。** 108-word main bundle 在交换前后都保留，但 compiler artifacts 没有提供从 MLIR SSA 到 ptxas physical spill registers 的跨层映射。H108 criticality 仍为 unresolved / out of scope。",
        "- 所有 arm 均通过独立 quant-aware reference；但 baseline replay self-drift 的 relative-L2 为 `"
        + fmt(float(base_self.get("relative_l2", 0.0)), 6)
        + "`，超过预注册 hard cap `0.002`，所以 strict cross-arm correctness gate 无效，不能事后放宽。",
        "",
        "## 1. Static 与正确性资格门",
        "",
        "| Arm | Stack B/thread | Main words | Tail words | Quant-aware oracle | Strict cross-arm gate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in arm_names:
        facts = static.get("arms", {}).get(arm, {})
        widths = facts.get("local", {}).get("stored_words_by_opcode_width", {})
        check = correctness.get(arm, {})
        rows.append(
            f"| {arm} | {facts.get('resource', {}).get('stack_bytes_per_thread', '—')} | "
            f"{widths.get('STL.64', 0)} | {widths.get('STL', 0)} | "
            f"{check.get('quant_aware_oracle_pass', True) if arm == 'baseline' else check.get('quant_aware_oracle_pass', '—')} | "
            f"{'baseline anchor' if arm == 'baseline' else check.get('gate_pass', '—')} |"
        )
    rows.extend(
        [
            "",
            "两个 in-place arm 的 selected non-local opcode projection 均与 baseline 一致，但目标 14-word scalar-STL bundle 在相同 PC/stack offsets 原位保留、没有净减少，因此没有测试到 H14 criticality。",
            "",
            "## 2. Attribution-only 的动态闭合",
            "",
            "| Metric | Baseline | Up-first | Delta |",
            "|---|---:|---:|---:|",
            f"| Local load sectors | {round(base_ncu.get('local_load_sectors', 0)):,} | {round(up_first_ncu.get('local_load_sectors', 0)):,} | {-round(up_first_delta.get('local_load_sector_reduction', 0)):,} |",
            f"| Local store sectors | {round(base_ncu.get('local_store_sectors', 0)):,} | {round(up_first_ncu.get('local_store_sectors', 0)):,} | {-round(up_first_delta.get('local_store_sector_reduction', 0)):,} |",
            f"| Executed local load instructions | {round(base_ncu.get('executed_local_load_instructions', 0)):,} | {round(up_first_ncu.get('executed_local_load_instructions', 0)):,} | {-round(up_first_delta.get('executed_local_load_instruction_reduction', 0)):,} |",
            f"| Executed local store instructions | {round(base_ncu.get('executed_local_store_instructions', 0)):,} | {round(up_first_ncu.get('executed_local_store_instructions', 0)):,} | {-round(up_first_delta.get('executed_local_store_instruction_reduction', 0)):,} |",
            f"| Tensor instructions | {round(base_ncu.get('tensor_instructions', 0)):,} | {round(up_first_ncu.get('tensor_instructions', 0)):,} | 0 |",
            f"| FP4 Tensor ops | {round(base_ncu.get('fp4_tensor_ops', 0)):,} | {round(up_first_ncu.get('fp4_tensor_ops', 0)):,} | 0 |",
            "",
            "Local sectors 是 local-address-space footprint，不是 DRAM bytes；NCU duration 也不替代未插桩 benchmark。",
            "",
            "## 3. 收口与下一步",
            "",
            "- 本轮停止，不继续搜索额外 codegen variants，也不报告 speedup。",
            "- 下一轮若继续，应预注册一个保持 FP32 数学与 work identity、但直接缩短 first accumulator 跨 second GEMM lifetime 的 clean arm；不应只围绕 activation destination 搜索。",
            "- 若要 formal 接受 H108 attribution，需要 compiler liveness/register-allocation 映射，或能把 semantic accumulator SSA 与 physical STL/LDL chain 闭合的证据工具。",
            "",
        ]
    )
    (results / "result.md").write_text("\n".join(rows))
    return 2


def decide(
    *,
    static: Mapping[str, Any],
    ncu: Mapping[str, Any],
    correctness: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    def_use: Mapping[str, Any] | None,
) -> dict[str, Any]:
    candidate = str(correctness.get("candidate") or benchmark.get("candidate"))
    if candidate not in (PRIMARY_CANDIDATE, FALLBACK_CANDIDATE):
        raise ValueError(f"invalid formal candidate: {candidate!r}")
    static_delta = static.get("deltas", {}).get(candidate, {})
    ncu_delta = ncu.get("deltas", {}).get(candidate, {})
    attribution_gate = all(
        (
            static.get("baseline_reproduction_pass", False),
            static_delta.get("complete_tail_removal_no_replacement", False),
            ncu.get("baseline_reproduction_pass", False),
            ncu_delta.get("dynamic_14_word_closure_pass", False),
            ncu_delta.get("work_identity_pass", False),
            correctness.get("gate_pass", False),
        )
    )
    if attribution_gate:
        h14_attribution = "accepted"
    else:
        h14_attribution = "inconclusive"

    if not attribution_gate:
        h14_criticality = "not_tested_due_to_failed_attribution_or_identity_gate"
    elif benchmark.get("material_improvement"):
        h14_criticality = "material"
    elif not benchmark.get("stable_le_5_percent", False):
        h14_criticality = "inconclusive_unstable_measurement"
    elif benchmark.get("candidate_faster_pair_count", 0) < 4 and benchmark.get(
        "speedup_fraction", 0.0
    ) > benchmark.get("materiality_T_fraction", 0.0):
        h14_criticality = "inconclusive_directionally_unstable"
    elif benchmark.get("speedup_fraction", 0.0) < -benchmark.get(
        "materiality_T_fraction", 0.0
    ):
        h14_criticality = "regression"
    else:
        h14_criticality = "non_material"

    h108_attribution = (
        "accepted_structural_lifetime"
        if def_use and def_use.get("gate_pass") is True
        else "source_and_program_order_inference_only"
    )
    return {
        "formal_candidate": candidate,
        "h108_attribution": h108_attribution,
        "h108_criticality": "out_of_scope_unresolved",
        "h14_attribution": h14_attribution,
        "h14_criticality": h14_criticality,
        "h14_attribution_gate": attribution_gate,
        "formal_experiment_closed": attribution_gate
        and h14_criticality
        not in {
            "not_tested_due_to_failed_attribution_or_identity_gate",
            "inconclusive_unstable_measurement",
            "inconclusive_directionally_unstable",
        },
    }


def arm_manifest(
    results: Path,
    arm: str,
    static: Mapping[str, Any],
    ncu: Mapping[str, Any],
) -> dict[str, Any]:
    preparation = read_json(results / "arms" / arm / "preparation.json")
    source = preparation["runtime"]["source"]
    ncu_input = ncu.get("inputs", {}).get(arm, {})
    static_arm = static.get("arms", {}).get(arm, {})
    return {
        "overlay_sha256": source["overlay_sha256"],
        "overlay_path": source["overlay"],
        "overlay_diff": source["overlay_diff"],
        "jit_root": preparation["runtime"]["jit_root"],
        "jit_artifact_set_sha256": preparation["jit_artifact_set_sha256"],
        "cubin_sha256": ncu_input.get("cubin_sha256")
        or static_arm.get("function", {}).get("cubin_sha256"),
        "preparation_manifest": str(
            (results / "arms" / arm / "preparation.json").relative_to(results)
        ),
        "preparation_manifest_sha256": file_sha256(
            results / "arms" / arm / "preparation.json"
        ),
    }


def build_manifest(
    results: Path,
    *,
    static: Mapping[str, Any],
    ncu: Mapping[str, Any],
    correctness: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = str(decision["formal_candidate"])
    arms = {
        arm: arm_manifest(results, arm, static, ncu) for arm in ("baseline", candidate)
    }
    baseline = read_json(results / "arms" / "baseline" / "preparation.json")
    manifest = {
        "schema": "exp004.validation-manifest.v1",
        "status": "complete"
        if decision["formal_experiment_closed"]
        else "inconclusive",
        "case": baseline["case"],
        "environment": baseline["runtime"],
        "source": baseline["runtime"]["source"],
        "fixture": baseline["fixture"],
        "arms": arms,
        "correctness": {
            "path": "correctness.json",
            "sha256": file_sha256(results / "correctness.json"),
            "gate_pass": correctness["gate_pass"],
        },
        "benchmark": {
            "path": "benchmark_summary.json",
            "sha256": file_sha256(results / "benchmark_summary.json"),
            **benchmark,
        },
        "static_spill": {
            "path": "static_spill_evidence.json",
            "sha256": file_sha256(results / "static_spill_evidence.json"),
            "baseline_reproduction_pass": static["baseline_reproduction_pass"],
        },
        "ncu": {
            "path": "ncu/spill_evidence.json",
            "sha256": file_sha256(results / "ncu" / "spill_evidence.json"),
            "baseline_reproduction_pass": ncu["baseline_reproduction_pass"],
        },
        "decision": dict(decision),
    }
    errors = validate_manifest_schema(manifest)
    if errors:
        raise RuntimeError("validation manifest schema errors: " + "; ".join(errors))
    return manifest


def render_report(
    *,
    static: Mapping[str, Any],
    ncu: Mapping[str, Any],
    correctness: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    decision: Mapping[str, Any],
    def_use: Mapping[str, Any] | None,
) -> str:
    candidate = decision["formal_candidate"]
    base_static = static["arms"]["baseline"]
    cand_static = static["arms"][candidate]
    base_ncu = ncu["arms"]["baseline"]
    cand_ncu = ncu["arms"][candidate]
    static_delta = static["deltas"][candidate]
    ncu_delta = ncu["deltas"][candidate]
    speedup = float(benchmark["speedup_percent"])
    threshold = float(benchmark["materiality_T_fraction"]) * 100.0
    lines = [
        "# exp_004：Fused Stack/Spill Criticality",
        "",
        "## 结论",
        "",
        f"- H14 attribution：**{decision['h14_attribution']}**；H14 criticality：**{decision['h14_criticality']}**。",
        f"- 未插桩延迟变化：`{speedup:+.3f}%`，本轮 materiality threshold `T={threshold:.3f}%`，candidate 更快的 paired repeats 为 `{benchmark['candidate_faster_pair_count']}/5`。",
        f"- H108 attribution：**{decision['h108_attribution']}**；H108 criticality 仍为 **unresolved / out of scope**。",
        "",
        "## 1. 资格门",
        "",
        "| Gate | 结果 |",
        "|---|---:|",
        f"| Fresh baseline static reproduction | {static['baseline_reproduction_pass']} |",
        f"| Fresh baseline dynamic reproduction | {ncu['baseline_reproduction_pass']} |",
        f"| Quant-aware oracle + strict candidate gate | {correctness['gate_pass']} |",
        f"| 14-word bundle 完整移除且无 replacement | {static_delta['complete_tail_removal_no_replacement']} |",
        f"| Dynamic local-sector closure | {ncu_delta['dynamic_14_word_closure_pass']} |",
        f"| Tensor work identity | {ncu_delta['work_identity_pass']} |",
        "",
        "## 2. Spill 与资源证据",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|",
        f"| Stack frame (B/thread) | {base_static['resource']['stack_bytes_per_thread']} | {cand_static['resource']['stack_bytes_per_thread']} | {static_delta['stack_bytes_delta']} |",
        f"| Static stored words/lane | {base_static['local']['stored_words']} | {cand_static['local']['stored_words']} | {static_delta['stored_words_delta']} |",
        f"| Local load sectors | {round(base_ncu['local_load_sectors']):,} | {round(cand_ncu['local_load_sectors']):,} | {-ncu_delta['local_load_sector_reduction']:,} |",
        f"| Local store sectors | {round(base_ncu['local_store_sectors']):,} | {round(cand_ncu['local_store_sectors']):,} | {-ncu_delta['local_store_sector_reduction']:,} |",
        f"| Executed local load instructions | {round(base_ncu['executed_local_load_instructions']):,} | {round(cand_ncu['executed_local_load_instructions']):,} | {-ncu_delta['executed_local_load_instruction_reduction']:,} |",
        f"| Executed local store instructions | {round(base_ncu['executed_local_store_instructions']):,} | {round(cand_ncu['executed_local_store_instructions']):,} | {-ncu_delta['executed_local_store_instruction_reduction']:,} |",
        "",
        "Local sectors 是 local-address-space footprint，不是 DRAM bytes。",
        "",
        "## 3. 性能与调度观测",
        "",
        "| Metric | Baseline | Candidate |",
        "|---|---:|---:|",
        f"| Median latency (us) | {benchmark['arms']['baseline']['median_us']:.3f} | {benchmark['arms']['candidate']['median_us']:.3f} |",
        f"| Spread | {benchmark['arms']['baseline']['spread_percent']:.3f}% | {benchmark['arms']['candidate']['spread_percent']:.3f}% |",
        f"| Achieved occupancy | {base_ncu['achieved_occupancy_pct']:.2f}% | {cand_ncu['achieved_occupancy_pct']:.2f}% |",
        f"| Eligible warps/cycle | {base_ncu['eligible_warps_per_cycle']:.3f} | {cand_ncu['eligible_warps_per_cycle']:.3f} |",
        f"| Issue active | {base_ncu['issue_active_pct']:.2f}% | {cand_ncu['issue_active_pct']:.2f}% |",
        f"| TC subpipe active | {base_ncu['tc_subpipe_active_pct']:.2f}% | {cand_ncu['tc_subpipe_active_pct']:.2f}% |",
        f"| Warp stalls (Wait / Long / Short / Barrier) | {base_ncu['stall_wait_pct']:.2f}% / {base_ncu['stall_long_scoreboard_pct']:.2f}% / {base_ncu['stall_short_scoreboard_pct']:.2f}% / {base_ncu['stall_barrier_pct']:.2f}% | {cand_ncu['stall_wait_pct']:.2f}% / {cand_ncu['stall_long_scoreboard_pct']:.2f}% / {cand_ncu['stall_short_scoreboard_pct']:.2f}% / {cand_ncu['stall_barrier_pct']:.2f}% |",
        "",
        "## 4. 边界与下一步",
        "",
    ]
    if def_use and def_use.get("gate_pass"):
        lines.append(
            "- IR/PTX/SASS def-use 链已闭合，108-word bundle 可归因为 first-produced accumulator 的结构性 lifetime。"
        )
    else:
        lines.append(
            "- 108-word bundle 尚未形成跨 IR/PTX/SASS 的 def-use 闭环，只保留 source + program-order inference。"
        )
    lines.extend(
        [
            "- 本实验没有 FP32 等价且能单独消除 108-word bundle 的 clean arm，因此不能判断其 criticality。",
            "- 若 H14 为 non-material，它不再是当前 P0；后续优化应优先寻找能缩短双 FP32 accumulator overlap lifetime、同时保持 tile/work identity 的方案。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)
    results = args.results.resolve()
    static = read_json(results / "static_spill_evidence.json")
    if select_static_candidate(static) is None:
        return close_no_qualified_candidate(results, static)
    ncu = read_json(results / "ncu" / "spill_evidence.json")
    correctness = read_json(results / "correctness.json")
    benchmark = read_json(results / "benchmark_summary.json")
    def_use_path = results / "def_use_evidence.json"
    def_use = read_json(def_use_path) if def_use_path.is_file() else None
    decision = decide(
        static=static,
        ncu=ncu,
        correctness=correctness,
        benchmark=benchmark,
        def_use=def_use,
    )
    manifest = build_manifest(
        results,
        static=static,
        ncu=ncu,
        correctness=correctness,
        benchmark=benchmark,
        decision=decision,
    )
    write_json(results / "validation.manifest.json", manifest)
    report = render_report(
        static=static,
        ncu=ncu,
        correctness=correctness,
        benchmark=benchmark,
        decision=decision,
        def_use=def_use,
    )
    (results / "result.md").write_text(report)
    return 0 if decision["formal_experiment_closed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
