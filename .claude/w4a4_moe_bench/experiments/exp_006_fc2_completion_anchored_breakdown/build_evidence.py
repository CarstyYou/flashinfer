#!/usr/bin/env python3
"""Build the fail-closed exp_006 evidence package from sealed artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from exp006_common import EVENT_ABI


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"
ARMS = ("measurement_no_marker", "completion_anchored_probe")
GRID_Z = 110
EXPECTED_CUBINS = {
    "measurement_no_marker": (
        "b1e068ec8eeb0988a84ad7b62e1958f5623e0911d233a45f080364491d7ee47f"
    ),
    "completion_anchored_probe": (
        "eed241ee29120468f2d72b690874784da502649964a46b3e337e3987d53fcc25"
    ),
}
RESOURCE_RE = re.compile(
    r"REG\s*:\s*(\d+)\s+STACK\s*:\s*(\d+)\s+"
    r"SHARED\s*:\s*(\d+)\s+LOCAL\s*:\s*(\d+)",
    re.I,
)
SASS_RE = re.compile(
    r"/\*\s*([0-9a-fA-F]+)\s*\*/\s+"
    r"(?:(@[!A-Za-z0-9.]+)\s+)?([A-Z][A-Za-z0-9_.]*)\s*(.*?)\s*;"
)

# Exact-PC proof is intentionally bound to EXPECTED_CUBINS.  Each record is one
# of the four compile-time-unrolled FC2 output-tile bodies; each body executes
# four loop iterations for the locked 16-output-tile case.  A/C/D/E/F are the
# timestamp-read PCs.  Store PCs are sealed separately because the compiler is
# allowed to delay a timestamp store while preserving the CS2R boundary.
BOUNDARY_PCS = (
    {
        "A": 0x130C0, "A_store": 0x131A0,
        "C": 0x13B50, "C_store": 0x15850,
        "pre_barrier": 0x15860,
        "D": 0x15870, "D_store": 0x15960,
        "E": 0x16000, "E_store": 0x16070,
        "post_barrier": 0x16080,
        "F": 0x16090, "F_store": 0x16100,
    },
    {
        "A": 0x16110, "A_store": 0x16180,
        "C": 0x16A90, "C_store": 0x17060,
        "pre_barrier": 0x17B30,
        "D": 0x17B40, "D_store": 0x17BC0,
        "E": 0x18290, "E_store": 0x18300,
        "post_barrier": 0x18310,
        "F": 0x18320, "F_store": 0x18390,
    },
    {
        "A": 0x183A0, "A_store": 0x18420,
        "C": 0x18D30, "C_store": 0x19300,
        "pre_barrier": 0x19DD0,
        "D": 0x19DE0, "D_store": 0x19E60,
        "E": 0x1A530, "E_store": 0x1A5A0,
        "post_barrier": 0x1A5B0,
        "F": 0x1A5C0, "F_store": 0x1A630,
    },
    {
        "A": 0x1A640, "A_store": 0x1A6C0,
        "C": 0x1AFD0, "C_store": 0x1C060,
        "pre_barrier": 0x1C070,
        "D": 0x1C080, "D_store": 0x1C100,
        "E": 0x1C7D0, "E_store": 0x1C840,
        "post_barrier": 0x1C850,
        "F": 0x1C860, "F_store": 0x1C8F0,
    },
)

READER_PHASE_SPECS = (
    {
        "phase": "FC2_GEMM",
        "semantic_owner": "FC2 GEMM producer",
        "boundary": "max(A0..A3)-to-min(D0..D3)",
        "start_data_state": (
            "all W0-W3 have reached the per-tile pre-consumer-wait edge; "
            "the FC2 output tile has not been produced"
        ),
        "end_data_state": (
            "epilogue scale/cast and accumulator R-to-S are complete with CTA "
            "visibility, and all W0-W3 have passed the pre-scatter barrier"
        ),
        "downstream_ready_condition": (
            "the materialized FC2 tile is safe for W0-W3 to consume in scatter"
        ),
        "components": (
            "FC2_issue_path",
            "FC2_completion_materialize_pre_sync",
        ),
        "aggregation_formula": (
            "sum_tasks_tiles[(max(C0..C3)-max(A0..A3)) + "
            "(min(D0..D3)-max(C0..C3))]"
        ),
        "included_work": (
            "pipeline wait/load",
            "accumulator clear",
            "warp-level OMMA",
            "epilogue scale/cast",
            "accumulator R-to-S",
            "CTA visibility fence",
            "pre-scatter sync",
        ),
        "excluded_work": (
            "FC2 setup before A",
            "atomic scatter loop",
            "post-scatter sync",
            "final handoff after F",
        ),
        "required_sass_checks": (
            "a_store_immediately_before_consumer_trywait",
            "issue_has_64_omma",
            "c_after_last_omma_issue",
            "no_omma_between_c_and_d",
            "materialization_before_d",
            "cta_visibility_before_d",
            "pre_barrier",
            "d_immediately_after_pre_barrier",
        ),
    },
    {
        "phase": "FC2_atomic_scatter",
        "semantic_owner": "FC2 scatter/output-reduction consumer-producer",
        "boundary": "min(D0..D3)-to-max(F0..F3)",
        "start_data_state": (
            "the FC2 tile is materialized R-to-S with CTA visibility and the "
            "pre-scatter barrier has released W0-W3"
        ),
        "end_data_state": (
            "all W0-W3 have completed the scatter loop and passed the "
            "post-scatter barrier"
        ),
        "downstream_ready_condition": (
            "the CTA may advance beyond this tile's scatter stage; final "
            "operator handoff after F remains outside this interval"
        ),
        "components": (
            "FC2_atomic_scatter_body",
            "FC2_post_scatter_sync",
        ),
        "aggregation_formula": (
            "sum_tasks_tiles[(max(E0..E3)-min(D0..D3)) + "
            "(max(F0..F3)-max(E0..E3))]"
        ),
        "included_work": (
            "atomic scatter loop",
            "route-weight math and addressing",
            "post-scatter sync",
        ),
        "excluded_work": (
            "FC2 setup before A",
            "OMMA",
            "epilogue scale/cast",
            "accumulator R-to-S",
            "pre-scatter sync",
            "final handoff after F",
        ),
        "required_sass_checks": (
            "scatter_has_global_reduction",
            "no_omma_after_d_before_e",
            "no_accumulator_smem_store_after_d_before_e",
            "no_tma_after_d_before_e",
            "e_is_timer",
            "no_scatter_after_e_before_post_barrier",
            "post_barrier",
            "f_immediately_after_post_barrier",
        ),
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def artifact_hash(capture: dict[str, Any], suffix: str) -> str:
    matches = [
        item["sha256"]
        for item in capture["jit_artifacts"]
        if item["path"].endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {suffix} artifact, found {len(matches)}")
    return str(matches[0])


def capture_gate(capture: dict[str, Any], arm: str) -> dict[str, Any]:
    if capture.get("arm") != arm or capture.get("schema") != "exp006.completion-capture.v1":
        raise ValueError(f"{arm}: capture identity mismatch")
    runs = capture.get("runs", [])
    eager = capture.get("eager", {})
    eager_gate_names = (
        "correctness_gate",
        "descriptor_gate",
        "event_gate",
        "runtime_case_gate",
        "workspace_gate",
    )
    run_gate_names = eager_gate_names
    checks = {
        "five_replays": len(runs) == 5,
        "eager_no_failed_gates": not eager.get("failed_gates"),
        "eager_all_contract_gates": all(
            eager.get(name, {}).get("gate_pass") for name in eager_gate_names
        ),
        "all_run_gates": all(not run.get("failed_gates") for run in runs),
        "all_run_contract_gates": all(
            all(run.get(name, {}).get("gate_pass") for name in run_gate_names)
            for run in runs
        ),
        "event_abi_exact": capture.get("event_abi") == EVENT_ABI,
        "overlay_gate": capture.get("overlay_gate", {}).get("gate_pass"),
        "jit_identity_gate": capture.get("jit_identity_gate", {}).get("gate_pass"),
        "no_foreign_process_after": not capture.get("foreign_processes_after"),
    }
    return {"checks": checks, "gate_pass": all(checks.values())}


def parse_static(root: Path, arm: str, capture: dict[str, Any]) -> dict[str, Any]:
    arm_root = root / "raw" / "static" / arm
    cubin = arm_root / "kernel.cubin"
    ptx = arm_root / "kernel.ptx"
    sass = arm_root / "kernel.sass"
    resource = arm_root / "resource.txt"
    elf = arm_root / "elf.txt"
    provenance_path = arm_root / "provenance.json"
    for path in (cubin, ptx, sass, resource, elf, provenance_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{arm}: missing static artifact {path}")

    cubin_hash = sha256(cubin)
    ptx_hash = sha256(ptx)
    expected_cubin = artifact_hash(capture, ".cubin")
    expected_ptx = artifact_hash(capture, ".ptx")
    resource_match = RESOURCE_RE.search(resource.read_text())
    if resource_match is None:
        raise ValueError(f"{arm}: resource usage is incomplete")
    registers, stack, shared_static, local_static = map(int, resource_match.groups())

    instructions = [
        {
            "pc": int(match.group(1), 16),
            "predicate": match.group(2) or "",
            "opcode": match.group(3),
            "operands": match.group(4).strip(),
        }
        for match in SASS_RE.finditer(sass.read_text())
    ]
    counts = Counter(row["opcode"] for row in instructions)
    local_load = sum(value for opcode, value in counts.items() if opcode.startswith("LDL"))
    local_store = sum(value for opcode, value in counts.items() if opcode.startswith("STL"))
    omma = sum(value for opcode, value in counts.items() if opcode.startswith("OMMA"))
    cs2r = sum(value for opcode, value in counts.items() if opcode.startswith("CS2R"))
    spill_annotations = elf.read_text().count("SpillRefill")
    provenance = read_json(provenance_path)
    derived_hashes = {
        "cubin": cubin_hash,
        "ptx": ptx_hash,
        "sass": sha256(sass),
        "resource": sha256(resource),
        "elf": sha256(elf),
    }
    tool_checks = {
        name: bool(tool.get("resolved_path") and tool.get("sha256") and tool.get("version"))
        for name, tool in provenance.get("tools", {}).items()
    }
    checks = {
        "cubin_matches_capture": cubin_hash == expected_cubin == EXPECTED_CUBINS[arm],
        "ptx_matches_capture": ptx_hash == expected_ptx,
        "sass_has_kernel": "MoEDynamicKernel" in sass.read_text(),
        "spill_annotation_closure": spill_annotations == local_load + local_store,
        "provenance_schema": provenance.get("schema")
        == "exp006.static-provenance.v1",
        "provenance_arm": provenance.get("arm") == arm,
        "provenance_capture": provenance.get("capture_json", {}).get("sha256")
        == sha256(root / "raw" / arm / "capture.json"),
        "provenance_container": provenance.get("container_image_digest")
        == capture["runtime"]["image_digest"],
        "provenance_provider_cubin": provenance.get("provider", {})
        .get("cubin", {})
        .get("sha256")
        == expected_cubin,
        "provenance_provider_ptx": provenance.get("provider", {})
        .get("ptx", {})
        .get("sha256")
        == expected_ptx,
        "provenance_derived_artifacts": provenance.get("artifacts")
        == derived_hashes,
        "provenance_tools": set(tool_checks) == {"nvdisasm", "cuobjdump"}
        and all(tool_checks.values()),
        "provenance_commands": set(provenance.get("commands", {}))
        == {"sass", "resource", "elf"},
    }
    return {
        "artifact_hashes": {**derived_hashes, "provenance": sha256(provenance_path)},
        "provenance": {
            "tools": provenance["tools"],
            "provider": provenance["provider"],
            "commands": provenance["commands"],
        },
        "resource": {
            "registers_per_thread": registers,
            "actual_compiler_stack_bytes": stack,
            "static_shared_bytes": shared_static,
            "static_local_bytes_outside_stack": local_static,
            "static_ldl": local_load,
            "static_stl": local_store,
            "compiler_spill_refill_annotations": spill_annotations,
            "static_omma": omma,
            "timer_reads": cs2r,
            "instruction_count": len(instructions),
        },
        "checks": checks,
        "gate_pass": all(checks.values()),
        "_instructions": instructions,
    }


def _store_offset_bytes(row: dict[str, Any]) -> int | None:
    match = re.search(r"desc\[UR22\]\[[^\]]*?(?:\+0x([0-9a-fA-F]+))?\]", row["operands"])
    if match is None:
        return None
    return int(match.group(1), 16) if match.group(1) else 0


def _has_affine_event_address(rows: list[dict[str, Any]]) -> bool:
    operands = [row["operands"] for row in rows]
    return (
        any(
            row["opcode"] == "UIMAD"
            and "UR11, 0x153, UR7" in row["operands"]
            for row in rows
        )
        and any(
            row["opcode"] == "UIMAD"
            and "UR5, 0x14" in row["operands"]
            for row in rows
        )
        and any(
            row["opcode"].startswith("IADD3") and ", 0x9," in row["operands"]
            for row in rows
        )
        and bool(operands)
    )


def boundary_proof(static: dict[str, Any]) -> dict[str, Any]:
    instructions = static["_instructions"]
    by_pc = {row["pc"]: row for row in instructions}
    ordered_pcs = [row["pc"] for row in instructions]
    index = {pc: position for position, pc in enumerate(ordered_pcs)}
    warp_identity_checks = {
        "tid_read": by_pc.get(0x10, {}).get("opcode") == "S2R"
        and "SR_TID.X" in by_pc.get(0x10, {}).get("operands", ""),
        "warp_index_shift": by_pc.get(0x90, {}).get("opcode") == "SHF.R.U32.HI"
        and "0x5" in by_pc.get(0x90, {}).get("operands", ""),
        "warp_index_to_ur7": by_pc.get(0xC0, {}).get("opcode") == "R2UR"
        and by_pc.get(0xC0, {}).get("operands", "").startswith("UR7,"),
        "compute_warp_range_w0_w3": by_pc.get(0x3F00, {}).get("opcode")
        == "UISETP.GT.AND"
        and "UR7, 0x3" in by_pc.get(0x3F00, {}).get("operands", ""),
        "lane0_predicate_definition": by_pc.get(0x130B0, {}).get("opcode")
        == "LOP3.LUT"
        and by_pc.get(0x130B0, {}).get("operands", "").startswith("P1,"),
        "lane0_predicate_stable_through_fc2": not any(
            row["operands"].startswith("P1,")
            for row in instructions[index[0x130B0] + 1 : index[0x1C8F0] + 1]
        ),
    }
    records = []
    marker_names = ("A", "C", "D", "E", "F")
    for body, sealed in enumerate(BOUNDARY_PCS):
        pcs = tuple(sealed.values())
        if any(pc not in by_pc for pc in pcs):
            raise ValueError(f"boundary PC absent from exact probe SASS: {pcs}")
        a_pc, c_pc, d_pc, e_pc, f_pc = (sealed[name] for name in marker_names)
        pre_bar_pc = sealed["pre_barrier"]
        post_bar_pc = sealed["post_barrier"]
        c_index = index[c_pc]
        previous = c_index - 1
        while previous >= 0 and by_pc[ordered_pcs[previous]]["opcode"] == "NOP":
            previous -= 1
        issue = instructions[index[a_pc] + 1 : index[c_pc]]
        before_d = instructions[index[c_pc] + 1 : index[d_pc]]
        scatter = instructions[index[d_pc] + 1 : index[e_pc]]
        after_e = instructions[index[e_pc] + 1 : index[post_bar_pc]]
        expected_offsets = {
            "A": body * 0xA0,
            "C": body * 0xA0 + 0x20,
            "D": body * 0xA0 + 0x40,
            "E": body * 0xA0 + 0x60,
            "F": body * 0xA0 + 0x80,
        }
        marker_checks = {}
        for name in marker_names:
            timer = by_pc[sealed[name]]
            store = by_pc[sealed[f"{name}_store"]]
            address_rows = instructions[index[sealed[name]] : index[sealed[f"{name}_store"]] + 1]
            timer_register = timer["operands"].split(",", 1)[0]
            marker_checks[name] = {
                "timer_is_lane0_globaltimer": timer["opcode"] == "CS2R"
                and timer["predicate"] == "@!P1"
                and "SR_GLOBALTIMERLO" in timer["operands"],
                "store_is_same_lane_and_value": store["opcode"] == "STG.E.64"
                and store["predicate"] == "@!P1"
                and timer_register in store["operands"],
                "timer_value_not_overwritten_before_store": not any(
                    row["operands"].split(",", 1)[0].strip() == timer_register
                    for row in address_rows[1:-1]
                ),
                "store_offset_matches_20_slot_abi": _store_offset_bytes(store)
                == expected_offsets[name],
                # C deliberately keeps its timestamp live until materialization
                # finishes and reuses the already-proven A affine base.  The
                # other four markers rebuild the address next to their store.
                "warp_indexed_affine_address": _has_affine_event_address(
                    address_rows
                    if name != "C"
                    else instructions[
                        index[sealed["A"]] : index[sealed["A_store"]] + 1
                    ]
                ),
            }
        issue_omma_count = sum(
            row["opcode"].startswith("OMMA") for row in issue
        )
        materialize_sts_count = sum(
            row["opcode"].startswith("STS") for row in before_d
        )
        scatter_reduction_count = sum(
            row["opcode"].startswith(("REDG", "ATOMG")) for row in scatter
        )
        checks = {
            "all_marker_store_contracts": all(
                all(item.values()) for item in marker_checks.values()
            ),
            "a_store_immediately_before_consumer_trywait": (
                index[sealed["A_store"]] + 1 < len(ordered_pcs)
                and by_pc[ordered_pcs[index[sealed["A_store"]] + 1]][
                    "opcode"
                ].startswith("SYNCS.PHASECHK")
                and by_pc[ordered_pcs[index[sealed["A_store"]] + 1]][
                    "opcode"
                ].endswith("TRYWAIT")
            ),
            "issue_has_64_omma": issue_omma_count == 64,
            "c_after_last_omma_issue": previous >= 0
            and by_pc[ordered_pcs[previous]]["opcode"].startswith("OMMA"),
            "no_omma_between_c_and_d": not any(
                row["opcode"].startswith("OMMA") for row in before_d
            ),
            "pre_barrier": by_pc[pre_bar_pc]["opcode"].startswith("BAR.SYNC"),
            "d_immediately_after_pre_barrier": index[d_pc] == index[pre_bar_pc] + 1
            and by_pc[d_pc]["opcode"] == "CS2R",
            "materialization_before_d": materialize_sts_count == 64
            and any(row["opcode"].startswith("F2FP") for row in before_d),
            "cta_visibility_before_d": any(
                row["opcode"].startswith("MEMBAR") for row in before_d
            )
            and any(row["opcode"].startswith("FENCE") for row in before_d),
            "scatter_has_global_reduction": any(
                row["opcode"].startswith(("REDG", "ATOMG")) for row in scatter
            )
            and scatter_reduction_count == 1,
            "no_omma_after_d_before_e": not any(
                row["opcode"].startswith("OMMA") for row in scatter
            ),
            "no_accumulator_smem_store_after_d_before_e": not any(
                row["opcode"].startswith("STS") for row in scatter
            ),
            "no_tma_after_d_before_e": not any(
                row["opcode"].startswith(("UTMA", "TMA")) for row in scatter
            ),
            "e_is_timer": by_pc[e_pc]["opcode"] == "CS2R",
            "no_scatter_after_e_before_post_barrier": not any(
                row["opcode"].startswith(("REDG", "ATOMG")) for row in after_e
            ),
            "post_barrier": by_pc[post_bar_pc]["opcode"].startswith("BAR.SYNC"),
            "f_immediately_after_post_barrier": index[f_pc] == index[post_bar_pc] + 1
            and by_pc[f_pc]["opcode"] == "CS2R",
            "ordered": (
                index[a_pc] < index[sealed["A_store"]] < index[c_pc]
                < index[sealed["C_store"]] < index[pre_bar_pc] < index[d_pc]
                < index[sealed["D_store"]] < index[e_pc]
                < index[sealed["E_store"]] < index[post_bar_pc] < index[f_pc]
                < index[sealed["F_store"]]
            ),
        }
        records.append(
            {
                "unrolled_body": body,
                "pcs_hex": {
                    "A": hex(a_pc),
                    "C": hex(c_pc),
                    "pre_barrier": hex(pre_bar_pc),
                    "D": hex(d_pc),
                    "E": hex(e_pc),
                    "post_barrier": hex(post_bar_pc),
                    "F": hex(f_pc),
                    "stores": {
                        name: hex(sealed[f"{name}_store"])
                        for name in marker_names
                    },
                },
                "event_address": {
                    "formula_slots": "339*task + 20*tile + warp_idx + {9,13,17,21,25}",
                    "byte_offsets_within_four_body_unroll": expected_offsets,
                    "marker_checks": marker_checks,
                },
                "checks": checks,
                "static_counts": {
                    "issue_omma": issue_omma_count,
                    "completion_materialize_sts": materialize_sts_count,
                    "scatter_reduction_loop_instruction": scatter_reduction_count,
                },
                "gate_pass": all(checks.values()),
            }
        )
    return {
        "schema": "exp006.sass-boundary-proof.v2",
        "scope": "four compile-time-unrolled FC2 tile bodies",
        "warp_slot_identity": {
            "formula_slots": "339*task + 20*tile + warp_idx + {9,13,17,21,25}",
            "checks": warp_identity_checks,
            "gate_pass": all(warp_identity_checks.values()),
        },
        "records": records,
        "gate_pass": all(warp_identity_checks.values())
        and all(record["gate_pass"] for record in records),
    }


def ncu_row(path: Path) -> dict[str, str]:
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    rows = list(csv.reader(lines[start:]))
    if len(rows) != 3:
        raise ValueError(f"expected exactly one NCU kernel row: {path}")
    header, _units, values = rows[:3]
    return dict(zip(header, values, strict=True))


def relative_delta(subject: float, anchor: float) -> float:
    return subject / anchor - 1.0


def timing_gate(
    timing: dict[str, Any], probe_capture: dict[str, Any], results: Path
) -> dict[str, Any]:
    inputs = timing.get("inputs", [])
    aggregate = timing.get("aggregate", {})
    timing_paths = [
        results / "raw" / "completion_anchored_probe" / f"timing_{index}.pt"
        for index in range(5)
    ]
    expected_hashes = [sha256(path) for path in timing_paths]
    observed_hashes = [item.get("sha256") for item in inputs]
    expected_names = [f"timing_{index}.pt" for index in range(5)]
    observed_names = [Path(str(item.get("path", ""))).name for item in inputs]
    expected_phases = {
        "FC2_issue_path",
        "FC2_completion_materialize_pre_sync",
        "FC2_atomic_scatter_body",
        "FC2_post_scatter_sync",
    }
    cross_tile_edge_count = 0
    cross_tile_edges_valid = True
    for replay in timing.get("replays", []):
        for task in replay.get("task_records", []):
            boundaries = task.get("tile_boundaries_ns", [])
            for previous, current in zip(boundaries, boundaries[1:], strict=False):
                for warp in range(4):
                    cross_tile_edge_count += 1
                    if int(current[f"A{warp}"]) < int(previous[f"F{warp}"]):
                        cross_tile_edges_valid = False
    checks = {
        "schema": timing.get("schema") == "exp006.completion-timing.v2",
        "event_abi_exact": timing.get("event_abi") == EVENT_ABI,
        "five_bound_inputs": len(inputs) == 5
        and observed_names == expected_names
        and observed_hashes == expected_hashes,
        "five_replays": aggregate.get("replays") == 5,
        "all_replay_validation": len(timing.get("replays", [])) == 5
        and all(
            replay.get("validation", {}).get("pass")
            and replay.get("descriptor_order_sha256")
            == probe_capture.get("descriptor_order_sha256")
            for replay in timing.get("replays", [])
        ),
        "descriptor_identity": aggregate.get("descriptor_order_sha256")
        == probe_capture.get("descriptor_order_sha256"),
        "phase_set": set(aggregate.get("phase_replay_statistics", {}))
        == expected_phases,
        "fc2_closure": aggregate.get("fc2_envelope_closure", {}).get("pass")
        and aggregate.get("fc2_envelope_closure", {}).get("delta_ns") == 0,
        "whole_kernel_closure": aggregate.get("whole_kernel_closure", {}).get("pass")
        and aggregate.get("whole_kernel_closure", {}).get("delta_ns") == 0,
        "same_warp_cross_tile_f_to_a": cross_tile_edge_count > 0
        and cross_tile_edges_valid,
    }
    return {
        "checks": checks,
        "same_warp_cross_tile_edge_count": cross_tile_edge_count,
        "gate_pass": all(checks.values()),
    }


def build_phase_ownership_audit(
    reader_phase_rows: list[dict[str, Any]],
    marker_interval_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    proof: dict[str, Any],
    source_identity: dict[str, Any],
    static_artifact_hashes: dict[str, str],
    timing_authority: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed when a timing marker changes functional phase ownership."""

    rows_by_phase = {row["phase"]: row for row in reader_phase_rows}
    markers_by_phase = {row["phase"]: row for row in marker_interval_rows}
    proof_records = proof.get("records", [])
    expected_components = [
        component
        for spec in READER_PHASE_SPECS
        for component in spec["components"]
    ]
    closure = aggregate["fc2_envelope_closure"]
    required_static_hashes = ("cubin", "ptx", "sass", "resource", "elf", "provenance")
    production_kernel = source_identity.get("production", {}).get("kernel", {})
    probe_overlay_kernel = source_identity.get("overlays", {}).get("kernel", {})
    source_sass_checks = {
        "production_kernel_source_bound": bool(
            production_kernel.get("path") and production_kernel.get("sha256")
        ),
        "probe_overlay_source_bound": bool(
            probe_overlay_kernel.get("path") and probe_overlay_kernel.get("sha256")
        ),
        "exact_static_artifacts_bound": all(
            static_artifact_hashes.get(name) for name in required_static_hashes
        ),
        "sass_boundary_proof_pass": proof.get("gate_pass") is True,
        "all_four_unrolled_bodies_covered": len(proof_records) == 4
        and all(record.get("gate_pass") for record in proof_records),
    }

    phase_records = []
    for spec in READER_PHASE_SPECS:
        row = rows_by_phase.get(spec["phase"], {})
        component_duration_ns = sum(
            int(aggregate["phase_totals_ns"].get(component, -1))
            for component in spec["components"]
        )
        required_sass_checks = {
            name: len(proof_records) == 4
            and all(record.get("checks", {}).get(name) is True for record in proof_records)
            for name in spec["required_sass_checks"]
        }
        checks = {
            "semantic_owner_bound": bool(spec["semantic_owner"]),
            "start_data_state_bound": bool(spec["start_data_state"]),
            "end_data_state_bound": bool(spec["end_data_state"]),
            "downstream_ready_condition_bound": bool(
                spec["downstream_ready_condition"]
            ),
            "reader_boundary_matches_spec": row.get("boundary") == spec["boundary"],
            "reader_components_match_spec": row.get("components")
            == list(spec["components"]),
            "fine_marker_aggregation_closes": row.get("aggregate_duration_ns")
            == component_duration_ns,
            "share_denominator_recomputes": abs(
                float(row.get("sm_equivalent_share_pct", float("nan")))
                - component_duration_ns
                / float(timing_authority.get("sm_equivalent_denominator_ns", 0))
                * 100.0
            )
            < 1e-12
            if timing_authority.get("sm_equivalent_denominator_ns")
            else False,
            "component_gates_pass": row.get("component_gates_pass") is True
            and all(
                markers_by_phase.get(component, {}).get("marker_cost_gate_pass")
                and markers_by_phase.get(component, {}).get(
                    "replay_share_cv_gate_pass"
                )
                for component in spec["components"]
            ),
            "required_source_sass_coverage": all(source_sass_checks.values())
            and all(required_sass_checks.values()),
        }
        phase_records.append(
            {
                "phase": spec["phase"],
                "semantic_owner": spec["semantic_owner"],
                "start_boundary": spec["boundary"].split("-to-")[0],
                "start_data_state": spec["start_data_state"],
                "end_boundary": spec["boundary"].split("-to-")[1],
                "end_data_state": spec["end_data_state"],
                "downstream_ready_condition": spec["downstream_ready_condition"],
                "included_work": list(spec["included_work"]),
                "excluded_work": list(spec["excluded_work"]),
                "fine_marker_aggregation": {
                    "components": list(spec["components"]),
                    "formula": spec["aggregation_formula"],
                    "component_duration_ns": component_duration_ns,
                    "reader_duration_ns": row.get("aggregate_duration_ns"),
                },
                "source_sass_coverage": {
                    "production_kernel": production_kernel,
                    "probe_overlay_kernel": probe_overlay_kernel,
                    "artifact_hashes": {
                        name: static_artifact_hashes.get(name)
                        for name in required_static_hashes
                    },
                    "proof_schema": proof.get("schema"),
                    "unrolled_body_count": len(proof_records),
                    "required_sass_checks": required_sass_checks,
                },
                "checks": checks,
                "verdict": "PASS" if all(checks.values()) else "REJECT",
            }
        )

    reader_duration_sum_ns = sum(
        int(row.get("aggregate_duration_ns", -1)) for row in reader_phase_rows
    )
    global_checks = {
        "reader_phase_set_exact": set(rows_by_phase)
        == {spec["phase"] for spec in READER_PHASE_SPECS},
        "fine_marker_set_exact": set(markers_by_phase) == set(expected_components),
        "no_component_double_count": len(expected_components)
        == len(set(expected_components)),
        "reader_additive_sum_closes": reader_duration_sum_ns
        == closure["additive_phase_sum_ns"],
        "residual_closes_envelope": reader_duration_sum_ns
        + closure["intertile_residual_ns"]
        == closure["envelope_sum_ns"],
        "source_sass_coverage_pass": all(source_sass_checks.values()),
        "timing_authority_bound": timing_authority.get("gate_pass") is True,
        "all_phase_audits_pass": all(
            record["verdict"] == "PASS" for record in phase_records
        ),
    }
    verdict = "PASS" if all(global_checks.values()) else "REJECT"
    return {
        "schema": "exp006.phase-ownership-boundary-data-audit.v1",
        "verdict": verdict,
        "phases": phase_records,
        "closure": {
            "reader_additive_sum_ns": reader_duration_sum_ns,
            "intertile_residual_ns": closure["intertile_residual_ns"],
            "envelope_sum_ns": closure["envelope_sum_ns"],
            "residual_scope": (
                "inter-tile control gap outside both reader phases but inside "
                "the A-to-F tile-sweep envelope"
            ),
        },
        "timing_authority": timing_authority,
        "checks": global_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = args.results.resolve()
    output = (args.output or results / "evidence.json").resolve()
    if output.exists():
        raise FileExistsError(f"immutable evidence output exists: {output}")

    captures = {
        arm: read_json(results / "raw" / arm / "capture.json") for arm in ARMS
    }
    capture_gates = {arm: capture_gate(captures[arm], arm) for arm in ARMS}
    if not all(gate["gate_pass"] for gate in capture_gates.values()):
        raise ValueError(f"capture gate failed: {capture_gates}")
    if captures[ARMS[0]]["descriptor_order_sha256"] != captures[ARMS[1]][
        "descriptor_order_sha256"
    ]:
        raise ValueError("cross-arm descriptor order mismatch")
    if captures[ARMS[0]]["reference_sha256"] != captures[ARMS[1]]["reference_sha256"]:
        raise ValueError("cross-arm reference mismatch")

    static = {
        arm: parse_static(results, arm, captures[arm]) for arm in ARMS
    }
    if not all(item["gate_pass"] for item in static.values()):
        raise ValueError("static identity gate failed")
    proof = boundary_proof(static["completion_anchored_probe"])
    if not proof["gate_pass"]:
        raise ValueError("SASS boundary proof failed")

    ncu = {
        arm: ncu_row(results / "raw" / "ncu" / arm / "native_raw.csv")
        for arm in ARMS
    }
    ncu_capture_hashes = {}
    ncu_capture_gates = {}
    for arm in ARMS:
        profile_capture = read_json(
            results / "raw" / "ncu" / arm / "capture" / "capture.json"
        )
        ncu_capture_gates[arm] = capture_gate(profile_capture, arm)
        if not ncu_capture_gates[arm]["gate_pass"]:
            raise ValueError(f"{arm}: NCU capture contract gate failed")
        if (
            profile_capture.get("descriptor_order_sha256")
            != captures[arm].get("descriptor_order_sha256")
            or profile_capture.get("reference_sha256")
            != captures[arm].get("reference_sha256")
            or profile_capture.get("fixture") != captures[arm].get("fixture")
            or profile_capture.get("runtime", {}).get("gpu", {}).get("uuid")
            != captures[arm].get("runtime", {}).get("gpu", {}).get("uuid")
            or profile_capture.get("runtime", {}).get("source")
            != captures[arm].get("runtime", {}).get("source")
        ):
            raise ValueError(f"{arm}: NCU/timing capture identity mismatch")
        ncu_capture_hashes[arm] = artifact_hash(profile_capture, ".cubin")
        if (
            ncu_capture_hashes[arm]
            != artifact_hash(captures[arm], ".cubin")
            or ncu_capture_hashes[arm] != EXPECTED_CUBINS[arm]
        ):
            raise ValueError(f"{arm}: NCU cubin differs from timing cubin")

    control_latency = float(captures[ARMS[0]]["latency_us"]["median"])
    probe_latency = float(captures[ARMS[1]]["latency_us"]["median"])
    overhead = relative_delta(probe_latency, control_latency)
    occupancy = {
        arm: float(ncu[arm]["sm__warps_active.avg.pct_of_peak_sustained_active"])
        for arm in ARMS
    }
    ncu_launch = {
        arm: {
            "registers_per_thread": int(ncu[arm]["launch__registers_per_thread"]),
            "shared_mem_per_block_bytes": int(ncu[arm]["launch__shared_mem_per_block"]),
            "configured_stack_bytes": int(ncu[arm]["launch__stack_size"]),
            "grid_z": int(ncu[arm]["launch__grid_dim_z"]),
            "block_threads": int(ncu[arm]["launch__block_size"]),
        }
        for arm in ARMS
    }
    resources = {arm: static[arm]["resource"] for arm in ARMS}
    stack_drift = relative_delta(
        resources[ARMS[1]]["actual_compiler_stack_bytes"],
        resources[ARMS[0]]["actual_compiler_stack_bytes"],
    )
    local_instruction_drift = relative_delta(
        resources[ARMS[1]]["static_ldl"] + resources[ARMS[1]]["static_stl"],
        resources[ARMS[0]]["static_ldl"] + resources[ARMS[0]]["static_stl"],
    )
    ldl_drift = relative_delta(
        resources[ARMS[1]]["static_ldl"], resources[ARMS[0]]["static_ldl"]
    )
    stl_drift = relative_delta(
        resources[ARMS[1]]["static_stl"], resources[ARMS[0]]["static_stl"]
    )
    drift_checks = {
        "whole_launch_overhead_le_5pct": overhead <= 0.05,
        "achieved_occupancy_unchanged": occupancy[ARMS[0]] == occupancy[ARMS[1]],
        "registers_unchanged": resources[ARMS[0]]["registers_per_thread"]
        == resources[ARMS[1]]["registers_per_thread"],
        "ncu_launch_registers_unchanged": ncu_launch[ARMS[0]]["registers_per_thread"]
        == ncu_launch[ARMS[1]]["registers_per_thread"],
        "ncu_launch_smem_unchanged": ncu_launch[ARMS[0]]["shared_mem_per_block_bytes"]
        == ncu_launch[ARMS[1]]["shared_mem_per_block_bytes"],
        "ncu_launch_shape_unchanged": (
            ncu_launch[ARMS[0]]["grid_z"], ncu_launch[ARMS[0]]["block_threads"]
        )
        == (ncu_launch[ARMS[1]]["grid_z"], ncu_launch[ARMS[1]]["block_threads"]),
        "stack_abs_drift_le_25pct": abs(stack_drift) <= 0.25,
        "ldl_abs_drift_le_25pct": abs(ldl_drift) <= 0.25,
        "stl_abs_drift_le_25pct": abs(stl_drift) <= 0.25,
        "useful_omma_unchanged": resources[ARMS[0]]["static_omma"]
        == resources[ARMS[1]]["static_omma"],
    }

    timing = read_json(results / "completion_timing.json")
    sealed_timing_gate = timing_gate(
        timing, captures["completion_anchored_probe"], results
    )
    if not sealed_timing_gate["gate_pass"]:
        raise ValueError(f"completion timing identity gate failed: {sealed_timing_gate}")
    aggregate = timing["aggregate"]
    replay_count = int(aggregate["replays"])
    marker_interval_rows = []
    for phase, stats in aggregate["phase_replay_statistics"].items():
        aggregate_duration_ns = aggregate["phase_totals_ns"][phase]
        aggregate_additive_ns = aggregate["fc2_envelope_closure"][
            "additive_phase_sum_ns"
        ]
        aggregate_denominator_ns = aggregate["sm_equivalent_denominator_ns"]
        marker_interval_rows.append(
            {
                "phase": phase,
                "aggregate_duration_ns": aggregate_duration_ns,
                "wall_equivalent_us": stats["duration_ns"]["mean"] / GRID_Z / 1000.0,
                "fc2_additive_share_pct": aggregate_duration_ns
                / aggregate_additive_ns
                * 100.0,
                "sm_equivalent_share_pct": aggregate_duration_ns
                / aggregate_denominator_ns
                * 100.0,
                "duration_replay_cv": stats["duration_ns"]["cv"],
                "reporting_class": stats["reporting_class"],
                "marker_cost_gate_pass": stats["marker_cost_gate_pass_all_replays"],
                "replay_share_cv_gate_pass": stats["replay_share_cv_gate_pass"],
            }
        )
    marker_interval_rows.sort(
        key=lambda row: (
            "FC2_issue_path",
            "FC2_completion_materialize_pre_sync",
            "FC2_atomic_scatter_body",
            "FC2_post_scatter_sync",
        ).index(row["phase"])
    )
    marker_phase_gate = all(
        row["marker_cost_gate_pass"] and row["replay_share_cv_gate_pass"]
        for row in marker_interval_rows
    ) and aggregate["fc2_envelope_closure"]["pass"]
    reader_phase_rows = []
    aggregate_additive_ns = aggregate["fc2_envelope_closure"][
        "additive_phase_sum_ns"
    ]
    aggregate_denominator_ns = aggregate["sm_equivalent_denominator_ns"]
    marker_rows_by_phase = {
        row["phase"]: row for row in marker_interval_rows
    }
    for spec in READER_PHASE_SPECS:
        components = spec["components"]
        duration_ns = sum(aggregate["phase_totals_ns"][name] for name in components)
        reader_phase_rows.append(
            {
                "phase": spec["phase"],
                "boundary": spec["boundary"],
                "components": list(components),
                "includes": ", ".join(spec["included_work"]),
                "aggregate_duration_ns": duration_ns,
                "wall_equivalent_us": duration_ns
                / (replay_count * GRID_Z)
                / 1000.0,
                "fc2_additive_share_pct": duration_ns
                / aggregate_additive_ns
                * 100.0,
                "sm_equivalent_share_pct": duration_ns
                / aggregate_denominator_ns
                * 100.0,
                "reporting_class": "diagnostic_estimate",
                "component_gates_pass": all(
                    marker_rows_by_phase[name]["marker_cost_gate_pass"]
                    and marker_rows_by_phase[name]["replay_share_cv_gate_pass"]
                    for name in components
                ),
            }
        )

    per_replay_global_wall_ns = [
        int(replay["global_wall_ns"]) for replay in timing["replays"]
    ]
    timing_authority_checks = {
        "provider_is_globaltimer": all(
            replay.get("timestamp_unit") == "globaltimer_ns"
            for replay in timing["replays"]
        ),
        "five_replays_bound": len(per_replay_global_wall_ns) == replay_count == 5,
        "sm_equivalent_denominator_recomputes": sum(
            wall_ns * GRID_Z for wall_ns in per_replay_global_wall_ns
        )
        == aggregate["sm_equivalent_denominator_ns"],
        "same_warp_cross_tile_f_to_a": sealed_timing_gate["checks"][
            "same_warp_cross_tile_f_to_a"
        ],
    }
    timing_authority = {
        "provider": "%globaltimer explicit timestamp probe",
        "timestamp_unit": "globaltimer_ns",
        "rollup": "SM-equivalent additive time",
        "formula": (
            "sum_replays[grid_z * (max(all CTA final) - min(all CTA entry))]"
        ),
        "grid_z": GRID_Z,
        "replay_count": replay_count,
        "per_replay_global_wall_ns": per_replay_global_wall_ns,
        "sm_equivalent_denominator_ns": aggregate["sm_equivalent_denominator_ns"],
        "not_timing_authority": (
            "control/probe CUDA-event medians and uninstrumented benchmark latency"
        ),
        "checks": timing_authority_checks,
        "gate_pass": all(timing_authority_checks.values()),
    }
    phase_ownership_audit = build_phase_ownership_audit(
        reader_phase_rows,
        marker_interval_rows,
        aggregate,
        proof,
        captures[ARMS[1]]["runtime"]["source"],
        static["completion_anchored_probe"]["artifact_hashes"],
        timing_authority,
    )
    phase_gate = marker_phase_gate and phase_ownership_audit["verdict"] == "PASS"

    closure = aggregate["fc2_envelope_closure"]
    additive_wall_us = sum(row["wall_equivalent_us"] for row in reader_phase_rows)
    residual_wall_us = closure["intertile_residual_ns"] / (replay_count * GRID_Z) / 1000.0
    envelope_wall_us = closure["envelope_sum_ns"] / (replay_count * GRID_Z) / 1000.0
    denominator_ns = float(aggregate["sm_equivalent_denominator_ns"])
    additive_sm_share = closure["additive_phase_sum_ns"] / denominator_ns * 100.0
    residual_sm_share = closure["intertile_residual_ns"] / denominator_ns * 100.0
    envelope_sm_share = closure["envelope_sum_ns"] / denominator_ns * 100.0

    ncu_artifacts = {}
    for arm in ARMS:
        arm_root = results / "raw" / "ncu" / arm
        paths = {
            "report": arm_root / "trace.ncu-rep",
            "native_csv": arm_root / "native_raw.csv",
            "capture_json": arm_root / "capture" / "capture.json",
        }
        if any(not path.is_file() or path.stat().st_size == 0 for path in paths.values()):
            raise ValueError(f"{arm}: incomplete NCU artifact set")
        ncu_artifacts[arm] = {
            name: {"path": str(path), "sha256": sha256(path), "size": path.stat().st_size}
            for name, path in paths.items()
        }

    for arm in ARMS:
        static[arm].pop("_instructions", None)
    payload = {
        "schema": "exp006.evidence.v4",
        "classification": "diagnostic_estimate",
        "case": captures[ARMS[0]]["fixture"],
        "event_abi": captures[ARMS[1]]["event_abi"],
        "identity": {
            "production_source": captures[ARMS[0]]["runtime"]["source"]["production"],
            "container_image_digest": captures[ARMS[0]]["runtime"]["image_digest"],
            "gpu_identity": {
                key: captures[ARMS[0]]["runtime"]["gpu"][key]
                for key in (
                    "name",
                    "uuid",
                    "pci_bus_id",
                    "compute_capability",
                    "sm_count",
                    "driver",
                )
            },
            "gpu_capture_snapshots": {
                arm: captures[arm]["runtime"]["gpu"] for arm in ARMS
            },
            "nvcc": captures[ARMS[0]]["runtime"]["nvcc"],
            "ptxas": captures[ARMS[0]]["runtime"]["ptxas"],
            "descriptor_order_sha256": captures[ARMS[0]]["descriptor_order_sha256"],
            "reference_sha256": captures[ARMS[0]]["reference_sha256"],
            "capture_json_sha256": {
                arm: sha256(results / "raw" / arm / "capture.json") for arm in ARMS
            },
            "completion_timing_sha256": sha256(results / "completion_timing.json"),
        },
        "capture_gates": capture_gates,
        "timing_gate": sealed_timing_gate,
        "latency": {
            "control_median_us": control_latency,
            "probe_median_us": probe_latency,
            "probe_overhead_pct": overhead * 100.0,
        },
        "resource_drift": {
            "arms": resources,
            "achieved_occupancy_pct": occupancy,
            "ncu_launch": ncu_launch,
            "stack_relative_drift_pct": stack_drift * 100.0,
            "local_sass_relative_drift_pct": local_instruction_drift * 100.0,
            "ldl_relative_drift_pct": ldl_drift * 100.0,
            "stl_relative_drift_pct": stl_drift * 100.0,
            "checks": drift_checks,
            "gate_pass": all(drift_checks.values()),
            "ncu_cubin_sha256": ncu_capture_hashes,
            "ncu_capture_gates": ncu_capture_gates,
            "ncu_artifacts": ncu_artifacts,
        },
        "static": static,
        "sass_boundary_proof": proof,
        "phase_timing": {
            "phases": reader_phase_rows,
            "marker_intervals": marker_interval_rows,
            "phase_ownership_boundary_data_audit": phase_ownership_audit,
            "timing_authority": timing_authority,
            "fc2_additive_phase_wall_equivalent_us": additive_wall_us,
            "fc2_intertile_residual_wall_equivalent_us": residual_wall_us,
            "fc2_envelope_wall_equivalent_us": envelope_wall_us,
            "fc2_additive_phase_sm_equivalent_share_pct": additive_sm_share,
            "fc2_intertile_residual_sm_equivalent_share_pct": residual_sm_share,
            "fc2_envelope_sm_equivalent_share_pct": envelope_sm_share,
            "closure": closure,
            "regressions": aggregate["regressions"],
            "gate_pass": phase_gate,
        },
    }
    payload["gate_pass"] = (
        all(gate["gate_pass"] for gate in capture_gates.values())
        and payload["resource_drift"]["gate_pass"]
        and proof["gate_pass"]
        and sealed_timing_gate["gate_pass"]
        and phase_gate
    )
    if not payload["gate_pass"]:
        raise ValueError("final evidence gate failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "gate_pass": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
