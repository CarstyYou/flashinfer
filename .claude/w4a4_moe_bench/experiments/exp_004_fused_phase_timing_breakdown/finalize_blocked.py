#!/usr/bin/env python3
"""Fail-closed finalizer for an exp_004 prepare/probe failure.

This path intentionally does not consume phase timings.  It closes the
experiment only when the production/control identities are valid, the primary
clock probe has an independently evidenced measurement-gate failure, and the
preflighted IKET fallback is unavailable.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from exp004_common import (
    DEFAULT_RESULTS,
    MEASUREMENT_CONTROL,
    NORMAL,
    PROBE,
    canonical_sha256,
    file_sha256,
    read_json,
    write_json,
)


BLOCKED_PREFLIGHT_SCHEMA = "exp004.blocked-preflight.v1"
BLOCKED_GATE_SCHEMA = "exp004.blocked-gate.v1"
BLOCKED_VERDICT = "measurement_perturbation_prevented_formal_timing"
BLOCKED_STATUS = "blocked_by_measurement_gate"

RESOURCE_FIELDS = (
    "registers_per_thread",
    "stack_bytes_per_thread",
    "static_shared_bytes_per_cta",
    "static_local_bytes_outside_stack",
)
LOCAL_SASS_FIELDS = (
    "stl_instruction_count",
    "ldl_instruction_count",
    "spill_annotation_count",
)
SEMANTIC_FIELDS = ("omma", "utmaldg", "ldsm", "bar", "atomg", "redg", "ldg")
EXPECTED_RESOURCE = {
    "registers_per_thread": 255,
    "stack_bytes_per_thread": 488,
    "static_shared_bytes_per_cta": 1024,
    "static_local_bytes_outside_stack": 0,
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if result != value:
        raise ValueError(f"{label} must be an exact integer")
    return result


def _boolean(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _git_commit(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 40 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a lowercase 40-hex git commit")
    return text


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _selected_integers(
    value: Any, fields: Sequence[str], *, label: str
) -> dict[str, int]:
    source = _mapping(value, label=label)
    return {
        field: _integer(source.get(field), label=f"{label}.{field}") for field in fields
    }


def _validate_arm(value: Any, *, arm: str) -> dict[str, Any]:
    source = _mapping(value, label=f"arms.{arm}")
    return {
        "cubin_sha256": _sha256(
            source.get("cubin_sha256"), label=f"arms.{arm}.cubin_sha256"
        ),
        "sass_sha256": _sha256(
            source.get("sass_sha256"), label=f"arms.{arm}.sass_sha256"
        ),
        "resource": _selected_integers(
            source.get("resource"), RESOURCE_FIELDS, label=f"arms.{arm}.resource"
        ),
        "local_sass": _selected_integers(
            source.get("local_sass"),
            LOCAL_SASS_FIELDS,
            label=f"arms.{arm}.local_sass",
        ),
        "semantic_projection": _selected_integers(
            source.get("semantic_projection"),
            SEMANTIC_FIELDS,
            label=f"arms.{arm}.semantic_projection",
        ),
        "artifact_hashes": {
            str(name): _sha256(digest, label=f"arms.{arm}.artifact_hashes.{name}")
            for name, digest in _mapping(
                source.get("artifact_hashes"),
                label=f"arms.{arm}.artifact_hashes",
            ).items()
        },
    }


def _validate_evidence_identity(value: Any) -> dict[str, Any]:
    identity = _mapping(value, label="evidence_identity")
    source = _mapping(identity.get("source"), label="evidence_identity.source")
    gpu = _mapping(identity.get("gpu"), label="evidence_identity.gpu")
    toolchain = _mapping(identity.get("toolchain"), label="evidence_identity.toolchain")
    capability = gpu.get("compute_capability")
    if not isinstance(capability, Sequence) or isinstance(capability, (str, bytes)):
        raise ValueError("evidence_identity.gpu.compute_capability must be a pair")
    capability_values = [
        _integer(item, label="evidence_identity.gpu.compute_capability")
        for item in capability
    ]
    if len(capability_values) != 2:
        raise ValueError("evidence_identity.gpu.compute_capability must be a pair")
    return {
        "source": {
            "flashinfer_commit": _git_commit(
                source.get("flashinfer_commit"),
                label="evidence_identity.source.flashinfer_commit",
            ),
            "cutlass_commit": _git_commit(
                source.get("cutlass_commit"),
                label="evidence_identity.source.cutlass_commit",
            ),
            **{
                name: _sha256(
                    source.get(name), label=f"evidence_identity.source.{name}"
                )
                for name in (
                    "kernel_sha256",
                    "dispatch_sha256",
                    "wrapper_sha256",
                )
            },
        },
        "gpu": {
            "uuid": _text(gpu.get("uuid"), label="evidence_identity.gpu.uuid"),
            "name": _text(gpu.get("name"), label="evidence_identity.gpu.name"),
            "pci_bus_id": _text(
                gpu.get("pci_bus_id"), label="evidence_identity.gpu.pci_bus_id"
            ),
            "compute_capability": capability_values,
            "sm_count": _integer(
                gpu.get("sm_count"), label="evidence_identity.gpu.sm_count"
            ),
            "driver": _text(gpu.get("driver"), label="evidence_identity.gpu.driver"),
        },
        "toolchain": {
            name: _text(
                toolchain.get(name), label=f"evidence_identity.toolchain.{name}"
            )
            for name in (
                "nvcc",
                "ptxas",
                "python",
                "torch",
                "cuda_runtime",
                "image_digest",
                "python_deps_sha256",
                "cutlass_dsl_module",
                "cutlass_dsl_version",
            )
        },
    }


def _validate_runtime_observations(value: Any) -> dict[str, Any]:
    source = _mapping(value, label="runtime_observations")
    if set(source) != {NORMAL, MEASUREMENT_CONTROL, PROBE}:
        raise ValueError("runtime_observations must contain exactly the three arms")
    result = {}
    for arm in (NORMAL, MEASUREMENT_CONTROL, PROBE):
        observation = _mapping(source[arm], label=f"runtime_observations.{arm}")
        result[arm] = {
            name: _text(
                observation.get(name), label=f"runtime_observations.{arm}.{name}"
            )
            for name in (
                "applications_graphics_clock_mhz",
                "graphics_clock_mhz",
                "max_graphics_clock_mhz",
                "power_draw_w",
                "lease_id",
            )
        }
        result[arm]["jit_artifact_set_sha256"] = _sha256(
            observation.get("jit_artifact_set_sha256"),
            label=f"runtime_observations.{arm}.jit_artifact_set_sha256",
        )
    return result


def _relative_to_results(path: Path, results: Path, *, label: str) -> str:
    try:
        return path.resolve().relative_to(results.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} must be inside the results directory") from error


def build_blocked_gate(
    *,
    results: Path,
    blocked_preflight_path: Path,
    iket_preflight_path: Path,
) -> dict[str, Any]:
    results = results.resolve()
    blocked_preflight_path = blocked_preflight_path.resolve()
    iket_preflight_path = iket_preflight_path.resolve()
    blocked = read_json(blocked_preflight_path)
    iket = read_json(iket_preflight_path)
    if blocked.get("schema") != BLOCKED_PREFLIGHT_SCHEMA:
        raise ValueError(
            f"unsupported blocked preflight schema: {blocked.get('schema')!r}"
        )
    if iket.get("schema") != "exp004.iket-fallback-preflight.v1":
        raise ValueError(f"unsupported IKET preflight schema: {iket.get('schema')!r}")

    identity = _mapping(blocked.get("identity_gates"), label="identity_gates")
    identity_gates = {
        name: _boolean(identity.get(name), label=f"identity_gates.{name}")
        for name in ("source", "hardware", "toolchain")
    }
    if not all(identity_gates.values()):
        raise ValueError(
            "cannot label the failure measurement perturbation when source, "
            "hardware, or toolchain identity is unresolved"
        )
    evidence_identity = _validate_evidence_identity(blocked.get("evidence_identity"))
    runtime_observations = _validate_runtime_observations(
        blocked.get("runtime_observations")
    )

    source_arms = _mapping(blocked.get("arms"), label="arms")
    if set(source_arms) != {NORMAL, MEASUREMENT_CONTROL, PROBE}:
        raise ValueError(
            "blocked preflight must contain exactly the three exp_004 arms"
        )
    arms = {arm: _validate_arm(source_arms[arm], arm=arm) for arm in source_arms}

    normal = arms[NORMAL]
    measurement = arms[MEASUREMENT_CONTROL]
    probe = arms[PROBE]
    normal_anchor = normal["resource"] == EXPECTED_RESOURCE
    control_static_identity = (
        normal_anchor
        and measurement["resource"] == normal["resource"]
        and measurement["local_sass"] == normal["local_sass"]
        and measurement["semantic_projection"] == normal["semantic_projection"]
    )
    if not control_static_identity:
        raise ValueError(
            "normal/measurement control identity is not closed; this is not a "
            "probe-only measurement-perturbation verdict"
        )
    probe_resource_spill_identity = (
        probe["resource"] == normal["resource"]
        and probe["local_sass"] == normal["local_sass"]
    )
    probe_semantic_projection = (
        probe["semantic_projection"] == normal["semantic_projection"]
    )

    event = _mapping(blocked.get("event_contract"), label="event_contract")
    expected_ticks = _integer(
        event.get("expected_tick_writes"),
        label="event_contract.expected_tick_writes",
    )
    observed_ticks = _integer(
        event.get("observed_tick_writes"),
        label="event_contract.observed_tick_writes",
    )
    expected_cta = _integer(
        event.get("expected_task_cta_writes"),
        label="event_contract.expected_task_cta_writes",
    )
    observed_cta = _integer(
        event.get("observed_task_cta_writes"),
        label="event_contract.observed_task_cta_writes",
    )
    if min(expected_ticks, expected_cta) <= 0 or min(observed_ticks, observed_cta) < 0:
        raise ValueError("event counts must be non-negative with non-zero expectations")
    probe_event_contract = (
        observed_ticks == expected_ticks and observed_cta == expected_cta
    )

    lowering = _mapping(blocked.get("probe_lowering"), label="probe_lowering")
    ptx_clock64_count = _integer(
        lowering.get("ptx_clock64_count"),
        label="probe_lowering.ptx_clock64_count",
    )
    store_counts_raw = lowering.get("ptx_global_store_opcode_counts")
    if store_counts_raw is not None:
        store_counts = _mapping(
            store_counts_raw,
            label="probe_lowering.ptx_global_store_opcode_counts",
        )
        normal_store_counts = _mapping(
            store_counts.get("normal_no_marker"),
            label="probe lowering normal store counts",
        )
        probe_store_counts = _mapping(
            store_counts.get("probe_candidate"),
            label="probe lowering probe store counts",
        )
        added_store_counts = _mapping(
            store_counts.get("probe_minus_normal"),
            label="probe lowering differential store counts",
        )
        normalized_store_counts = {
            arm: {
                opcode: _integer(
                    values.get(opcode),
                    label=f"probe lowering {arm} st.global.{opcode}",
                )
                for opcode in ("b64", "b32", "u64")
            }
            for arm, values in (
                ("normal_no_marker", normal_store_counts),
                ("probe_candidate", probe_store_counts),
                ("probe_minus_normal", added_store_counts),
            )
        }
        instrumentation_present = (
            ptx_clock64_count > 0
            and normalized_store_counts["probe_minus_normal"]["b64"] > 0
            and normalized_store_counts["probe_minus_normal"]["b32"] > 0
        )
        lowering_summary: dict[str, Any] = {
            "ptx_clock64_count": ptx_clock64_count,
            "ptx_global_store_opcode_counts": normalized_store_counts,
            "instrumentation_present": instrumentation_present,
        }
    else:
        legacy_store_count = _integer(
            lowering.get("ptx_probe_store_count"),
            label="probe_lowering.ptx_probe_store_count",
        )
        instrumentation_present = ptx_clock64_count > 0 and legacy_store_count > 0
        lowering_summary = {
            "ptx_clock64_count": ptx_clock64_count,
            "legacy_ptx_u64_store_count": legacy_store_count,
            "instrumentation_present": instrumentation_present,
        }

    preparation = _mapping(
        blocked.get("probe_preparation_gates"), label="probe_preparation_gates"
    )
    probe_preparation_gates = {
        name: _boolean(preparation.get(name), label=f"probe_preparation_gates.{name}")
        for name in (
            "reference_correctness",
            "output_contract",
            "workspace_contract",
        )
    }
    reference_metrics = _mapping(
        preparation.get("reference_metrics"),
        label="probe_preparation_gates.reference_metrics",
    )
    probe_reference_metrics = {
        name: _number(
            reference_metrics.get(name),
            label=f"probe_preparation_gates.reference_metrics.{name}",
        )
        for name in ("cosine", "relative_l2", "max_abs", "token_rel_l2_p99")
    }

    provider = _mapping(iket.get("provider"), label="IKET provider")
    iket_fallback_available = _boolean(
        provider.get("ready"), label="IKET provider.ready"
    )
    if iket_fallback_available:
        raise ValueError(
            "IKET fallback is available; execute its registered path before "
            "closing exp_004 as measurement-blocked"
        )

    primary_measurement_blocked = (
        not probe_resource_spill_identity
        or not probe_preparation_gates["reference_correctness"]
        or not probe_event_contract
    )
    if not primary_measurement_blocked:
        raise ValueError(
            "no primary measurement gate failed; blocked closure is illegal"
        )

    gates = {
        "source_identity": identity_gates["source"],
        "hardware_identity": identity_gates["hardware"],
        "toolchain_identity": identity_gates["toolchain"],
        "normal_measurement_control_static_identity": control_static_identity,
        "probe_resource_spill_identity": probe_resource_spill_identity,
        "probe_semantic_projection": probe_semantic_projection,
        "probe_event_contract": probe_event_contract,
        "probe_instrumentation_present_in_ptx": instrumentation_present,
        "probe_reference_correctness": probe_preparation_gates["reference_correctness"],
        "probe_output_contract": probe_preparation_gates["output_contract"],
        "probe_workspace_contract": probe_preparation_gates["workspace_contract"],
        "iket_fallback_available": iket_fallback_available,
    }
    stop_reasons = []
    if not probe_resource_spill_identity:
        stop_reasons.append("probe_resource_spill_identity_drift")
    if not probe_preparation_gates["reference_correctness"]:
        stop_reasons.append("probe_reference_correctness_failed")
    if not probe_event_contract:
        stop_reasons.append("probe_event_contract_incomplete")

    payload: dict[str, Any] = {
        "schema": BLOCKED_GATE_SCHEMA,
        "verdict": BLOCKED_VERDICT,
        "status": BLOCKED_STATUS,
        "formal_gate_pass": False,
        "diagnostic_share_allowed": False,
        "closure_gate_pass": True,
        "gates": gates,
        "stop_reasons": stop_reasons,
        "evidence_identity": evidence_identity,
        "runtime_observations": runtime_observations,
        "arms": arms,
        "event_contract": {
            "expected_tick_writes": expected_ticks,
            "observed_tick_writes": observed_ticks,
            "expected_task_cta_writes": expected_cta,
            "observed_task_cta_writes": observed_cta,
        },
        "probe_lowering": {
            **lowering_summary,
            "runtime_zero_write_cause": "unresolved",
        },
        "probe_preparation_gates": {
            **probe_preparation_gates,
            "reference_metrics": probe_reference_metrics,
        },
        "fallback": {
            "provider": provider.get("requested"),
            "required_version": provider.get("required_audited_version"),
            "observed_version": provider.get("observed_version"),
            "available": False,
            "preflight_status": iket.get("status"),
        },
        "evidence": {
            "blocked_preflight": _relative_to_results(
                blocked_preflight_path, results, label="blocked preflight"
            ),
            "blocked_preflight_sha256": file_sha256(blocked_preflight_path),
            "iket_preflight": _relative_to_results(
                iket_preflight_path, results, label="IKET preflight"
            ),
            "iket_preflight_sha256": file_sha256(iket_preflight_path),
        },
        "interpretation": {
            "phase_share_published": False,
            "zero_write_cause": "unresolved; no causal attribution is made",
            "post_stop_capture_required": False,
        },
    }
    payload["evidence_sha256"] = canonical_sha256(payload)
    return payload


def render_blocked_result(gate: Mapping[str, Any]) -> str:
    arms = _mapping(gate["arms"], label="blocked gate arms")
    event = _mapping(gate["event_contract"], label="blocked gate event contract")
    lowering = _mapping(gate["probe_lowering"], label="blocked gate lowering")
    preparation = _mapping(
        gate["probe_preparation_gates"], label="blocked gate probe preparation"
    )
    metrics = _mapping(
        preparation["reference_metrics"], label="blocked gate reference metrics"
    )
    fallback = _mapping(gate["fallback"], label="blocked gate fallback")
    lines = [
        "# exp_004：Fused Phase Timing Breakdown",
        "",
        "## 结论",
        "",
        "实验按预注册门槛停止：clock probe 改变了 production 的 stack/spill 身份；本次 probe replay 的 reference correctness 未通过，运行时事件槽位也没有写入。结论为 `measurement perturbation prevented formal timing`；不发布任何 phase 占比。",
        "",
        "## 失败门槛",
        "",
        "| Gate | 结果 | 证据 |",
        "|---|---:|---|",
        "| Production / measurement control 静态身份 | PASS | REG、STACK、SMEM、local SASS 与 semantic projection 一致 |",
        "| Probe resource / spill 身份 | FAIL | 见下表；probe 不再代表 production spill 结构 |",
        (
            "| Probe reference correctness | FAIL | "
            f"cosine `{metrics['cosine']:.6f}`；relative-L2 "
            f"`{metrics['relative_l2']:.6f}`；max-abs `{metrics['max_abs']:.6f}` |"
        ),
        (
            "| Probe event contract | FAIL | "
            f"ticks `{event['observed_tick_writes']}/{event['expected_tick_writes']}`；"
            f"CTA map `{event['observed_task_cta_writes']}/{event['expected_task_cta_writes']}` |"
        ),
        (
            "| IKET fallback | UNAVAILABLE | "
            f"provider `{fallback.get('provider')}`；要求版本 "
            f"`{fallback.get('required_version')}`；observed "
            f"`{fallback.get('observed_version')}` |"
        ),
        "",
        "## 静态证据",
        "",
        "| Arm | REG/thread | STACK B/thread | Static SMEM B/CTA | STL | LDL | Spill annotations |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in (NORMAL, MEASUREMENT_CONTROL, PROBE):
        value = _mapping(arms[arm], label=f"blocked gate arm {arm}")
        resource = _mapping(value["resource"], label=f"{arm} resource")
        local = _mapping(value["local_sass"], label=f"{arm} local SASS")
        lines.append(
            f"| `{arm}` | {resource['registers_per_thread']} | "
            f"{resource['stack_bytes_per_thread']} | "
            f"{resource['static_shared_bytes_per_cta']} | "
            f"{local['stl_instruction_count']} | "
            f"{local['ldl_instruction_count']} | "
            f"{local['spill_annotation_count']} |"
        )
    store_counts = lowering.get("ptx_global_store_opcode_counts")
    if isinstance(store_counts, Mapping):
        added = _mapping(
            store_counts.get("probe_minus_normal"),
            label="render differential store counts",
        )
        lowering_line = (
            f"- Probe PTX 相对 no-marker 新增 `{added['b64']}` 个 `st.global.b64` "
            f"与 `{added['b32']}` 个 `st.global.b32`，并检出 "
            f"`{lowering['ptx_clock64_count']}` 个 clock64 occurrence；"
            "`st.global.u64` 不作为 probe-store 证据。"
        )
    else:
        lowering_line = (
            f"- Legacy PTX 记录检出 `{lowering['ptx_clock64_count']}` 个 clock64 occurrence；"
            "旧 `st.global.u64` 计数不能识别 probe store。"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            lowering_line,
            "- 运行时 0-write 的具体原因尚未定位；不归因于 cache、pointer、CUDA Graph 或其他机制。",
            "- Immediate-stop 后不继续 cross-arm correctness、phase capture、calibration 或 NCU；没有合法的 phase timing 数据可解释。",
            "",
        ]
    )
    return "\n".join(lines)


def build_blocked_manifest(
    *, results: Path, gate_path: Path, result_path: Path
) -> dict[str, Any]:
    results = results.resolve()
    gate_path = gate_path.resolve()
    result_path = result_path.resolve()
    gate = read_json(gate_path)
    if (
        gate.get("schema") != BLOCKED_GATE_SCHEMA
        or gate.get("status") != BLOCKED_STATUS
        or not gate.get("closure_gate_pass")
    ):
        raise ValueError("blocked gate is not a closed exp_004 verdict")
    evidence = _mapping(gate["evidence"], label="blocked gate evidence")
    overlay_identity = results / "overlays" / "identity.json"
    if not overlay_identity.is_file():
        raise ValueError("missing compact overlay identity")
    manifest: dict[str, Any] = {
        "schema": "exp004.run-manifest.v1",
        "status": BLOCKED_STATUS,
        "verdict": BLOCKED_VERDICT,
        "overlay_identity": {
            "path": _relative_to_results(
                overlay_identity, results, label="overlay identity"
            ),
            "sha256": file_sha256(overlay_identity),
        },
        "evidence_identity": gate["evidence_identity"],
        "runtime_observations": gate["runtime_observations"],
        "arms": gate["arms"],
        "correctness": {
            "status": "probe_preparation_failed_before_cross_arm_correctness",
            "probe_preparation_gates": gate["probe_preparation_gates"],
        },
        "phase_captures": {},
        "profiles": {},
        "blocked_gate": {
            "path": _relative_to_results(gate_path, results, label="blocked gate"),
            "sha256": file_sha256(gate_path),
            "closure_gate_pass": True,
        },
        "raw_preflight": {
            "path": evidence["blocked_preflight"],
            "sha256": evidence["blocked_preflight_sha256"],
        },
        "iket_preflight": {
            "path": evidence["iket_preflight"],
            "sha256": evidence["iket_preflight_sha256"],
        },
        "result": {
            "path": _relative_to_results(result_path, results, label="result"),
            "sha256": file_sha256(result_path),
            "phase_share_published": False,
        },
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    return manifest


def finalize(
    *, results: Path, blocked_preflight_path: Path, iket_preflight_path: Path
) -> dict[str, Any]:
    results = results.resolve()
    stale_phase_artifacts = [
        path
        for path in (
            results / "derived" / "analysis_gates.json",
            results / "derived" / "mma_phase_share.csv",
            results / "derived" / "w4_overlap.csv",
            results / "raw" / "phase_capture",
        )
        if path.exists()
    ]
    if stale_phase_artifacts:
        raise RuntimeError(
            "refusing to mix blocked closure with phase artifacts: "
            + ", ".join(str(path) for path in stale_phase_artifacts)
        )
    gate = build_blocked_gate(
        results=results,
        blocked_preflight_path=blocked_preflight_path,
        iket_preflight_path=iket_preflight_path,
    )
    gate_path = results / "derived" / "blocked_gate.json"
    write_json(gate_path, gate)
    result_path = results / "result.md"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_name(f".{result_path.name}.tmp")
    temporary.write_text(render_blocked_result(gate))
    temporary.replace(result_path)
    manifest = build_blocked_manifest(
        results=results, gate_path=gate_path, result_path=result_path
    )
    write_json(results / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--blocked-preflight", type=Path)
    parser.add_argument("--iket-preflight", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = args.results.resolve()
    blocked = (
        args.blocked_preflight.resolve()
        if args.blocked_preflight
        else results / "raw" / "blocked_preflight.json"
    )
    iket = (
        args.iket_preflight.resolve()
        if args.iket_preflight
        else results / "iket_fallback_preflight.json"
    )
    finalize(
        results=results,
        blocked_preflight_path=blocked,
        iket_preflight_path=iket,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
