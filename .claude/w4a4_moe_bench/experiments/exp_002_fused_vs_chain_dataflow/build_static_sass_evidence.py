#!/usr/bin/env python3
"""Summarize static local-memory instructions from the captured fused binary."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"
DISASM_RELATIVE_PATH = Path(
    "ncu/m8192/cutedsl_bf16_fused/deep_launch_1/veloq/disasm.json"
)
TARGET_MANIFEST_RELATIVE_PATH = Path(
    "ncu/m8192/cutedsl_bf16_fused/deep_launch_1/target_manifest.json"
)
STACK_OFFSET_RE = re.compile(r"\[R1(?:\+0x([0-9a-fA-F]+))?\]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stack_offset(operands: str) -> int | None:
    match = STACK_OFFSET_RE.search(operands)
    if match is None:
        return None
    return int(match.group(1) or "0", 16)


def analyze_disassembly(
    disassembly: dict[str, Any],
    *,
    disassembly_sha256: str,
    target_manifest: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        row
        for row in disassembly["data"]["rows"]
        if "MoEDynamicKernel" in str(row.get("function_name", ""))
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one MoEDynamicKernel disassembly row, got {len(rows)}"
        )

    row = rows[0]
    instructions = row["instructions"]
    opcode_counts = Counter(str(inst["opcode"]) for inst in instructions)
    local_stores = [
        inst for inst in instructions if str(inst["opcode"]).startswith("STL")
    ]
    local_loads = [
        inst for inst in instructions if str(inst["opcode"]).startswith("LDL")
    ]
    stl64 = [inst for inst in local_stores if inst["opcode"] == "STL.64"]
    stl32 = [inst for inst in local_stores if inst["opcode"] == "STL"]
    unsupported_local_stores = [
        inst for inst in local_stores if inst["opcode"] not in {"STL", "STL.64"}
    ]
    if unsupported_local_stores:
        raise ValueError(f"unsupported local-store opcodes: {unsupported_local_stores}")

    stl64_offsets: list[int] = []
    for inst in stl64:
        offset = stack_offset(str(inst.get("operands", "")))
        if offset is None:
            raise ValueError(f"STL.64 has no R1-relative stack operand: {inst}")
        stl64_offsets.append(offset)

    stl32_offsets: list[int] = []
    for inst in stl32:
        offset = stack_offset(str(inst.get("operands", "")))
        if offset is None:
            raise ValueError(f"STL has no R1-relative stack operand: {inst}")
        stl32_offsets.append(offset)

    scalar_offsets_covered_by_stl64 = {
        offset for base in stl64_offsets for offset in (base, base + 4)
    }
    stored_scalar_offsets = scalar_offsets_covered_by_stl64 | set(stl32_offsets)
    loads_by_offset: dict[int, list[int]] = {}
    for inst in local_loads:
        offset = stack_offset(str(inst.get("operands", "")))
        if offset is not None:
            loads_by_offset.setdefault(offset, []).append(int(inst["address"]))
    store_address_by_offset: dict[int, int] = {}
    for inst in stl64:
        base = stack_offset(str(inst["operands"]))
        assert base is not None
        store_address_by_offset[base] = int(inst["address"])
        store_address_by_offset[base + 4] = int(inst["address"])
    for inst in stl32:
        offset = stack_offset(str(inst["operands"]))
        assert offset is not None
        store_address_by_offset[offset] = int(inst["address"])
    missing_later_reloads = sorted(
        offset
        for offset, store_address in store_address_by_offset.items()
        if not any(
            load_address > store_address
            for load_address in loads_by_offset.get(offset, [])
        )
    )
    tensor_instructions = [
        inst for inst in instructions if "MMA" in str(inst["opcode"])
    ]
    first_local_load_address = min(
        (int(inst["address"]) for inst in local_loads), default=-1
    )
    last_tensor_before_local_load = max(
        (
            int(inst["address"])
            for inst in tensor_instructions
            if int(inst["address"]) < first_local_load_address
        ),
        default=-1,
    )
    last_stl64_address = max((int(inst["address"]) for inst in stl64), default=-1)
    auxiliary = disassembly["data"].get("auxiliary", {})

    return {
        "schema": "exp002.static-local-sass.v1",
        "identity": {
            "comparison_group_id": target_manifest["comparison_group_id"],
            "rerun_id": target_manifest["rerun_id"],
            "environment_lock_digest": target_manifest["environment_lock_digest"],
            "protocol_lock_digest": target_manifest["protocol_lock_digest"],
            "artifact_fingerprint_sha256": target_manifest[
                "artifact_fingerprint_sha256"
            ],
            "ncu_report_sha256": target_manifest["report_sha256"],
            "disassembly_json_sha256": disassembly_sha256,
        },
        "function": {
            "name": row["function_name"],
            "start": row["start"],
            "length": row["length"],
            "instruction_count": len(instructions),
            "source_lineinfo_present": bool(
                auxiliary.get("source_lineinfo_present", False)
            ),
        },
        "static_instruction_facts": {
            "tensor_opcodes": {
                opcode: count
                for opcode, count in sorted(opcode_counts.items())
                if "MMA" in opcode
            },
            "local_load_opcode_counts": {
                opcode: count
                for opcode, count in sorted(opcode_counts.items())
                if opcode.startswith("LDL")
            },
            "local_store_opcode_counts": {
                opcode: count
                for opcode, count in sorted(opcode_counts.items())
                if opcode.startswith("STL")
            },
            "stl64": {
                "count": len(stl64),
                "bytes_per_lane_if_each_static_instruction_executes_once": (
                    len(stl64) * 8
                ),
                "address_first": (
                    hex(min(int(inst["address"]) for inst in stl64)) if stl64 else None
                ),
                "address_last": hex(last_stl64_address) if stl64 else None,
                "stack_offsets": [hex(offset) for offset in sorted(stl64_offsets)],
                "covered_32bit_stack_slots": len(scalar_offsets_covered_by_stl64),
                "all_slots_have_a_later_static_ldl": not any(
                    offset in scalar_offsets_covered_by_stl64
                    for offset in missing_later_reloads
                ),
                "missing_later_ldl_offsets": [
                    hex(offset)
                    for offset in missing_later_reloads
                    if offset in scalar_offsets_covered_by_stl64
                ],
            },
            "stack_roundtrip_model": {
                "stl64_covered_32bit_words": len(scalar_offsets_covered_by_stl64),
                "stl_covered_32bit_words": len(stl32_offsets),
                "total_stored_32bit_words_per_lane": len(stored_scalar_offsets),
                "unique_static_ldl_stack_offsets": len(loads_by_offset),
                "all_stored_slots_have_a_later_static_ldl": not missing_later_reloads,
                "missing_later_ldl_offsets": [
                    hex(offset) for offset in missing_later_reloads
                ],
                "program_order_markers": {
                    "stl64_first": (
                        hex(min(int(inst["address"]) for inst in stl64))
                        if stl64
                        else None
                    ),
                    "stl64_last": (
                        hex(max(int(inst["address"]) for inst in stl64))
                        if stl64
                        else None
                    ),
                    "last_tensor_opcode_before_first_ldl": (
                        hex(last_tensor_before_local_load)
                        if last_tensor_before_local_load >= 0
                        else None
                    ),
                    "first_ldl": (
                        hex(first_local_load_address)
                        if first_local_load_address >= 0
                        else None
                    ),
                    "stl_first": (
                        hex(min(int(inst["address"]) for inst in stl32))
                        if stl32
                        else None
                    ),
                    "stl_last": (
                        hex(max(int(inst["address"]) for inst in stl32))
                        if stl32
                        else None
                    ),
                },
            },
        },
        "classification": {
            "opcode_and_stack_offset_counts": "fact_from_captured_static_sass",
            "source_object_or_phase_attribution": (
                "inference_from_source_program_order; no source lineinfo"
            ),
            "dynamic_frequency_or_latency": "not_measured_by_static_sass",
        },
        "limitations": [
            "Static SASS does not establish dynamic execution frequency or critical-path latency.",
            "The captured binary has no source line information, so a specific source-variable attribution is an inference.",
            "Use the launch-level NCU local-memory counters for dynamic traffic; do not multiply these static counts into operator traffic without a validated execution model.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    results = args.results.resolve()
    disassembly_path = results / DISASM_RELATIVE_PATH
    target_manifest_path = results / TARGET_MANIFEST_RELATIVE_PATH
    payload = analyze_disassembly(
        json.loads(disassembly_path.read_text()),
        disassembly_sha256=sha256_file(disassembly_path),
        target_manifest=json.loads(target_manifest_path.read_text()),
    )
    output = results / "ncu" / "static_local_sass.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
