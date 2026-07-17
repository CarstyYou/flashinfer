#!/usr/bin/env python3
"""Build per-arm static SASS/resource evidence for exp_003."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from exp003_common import (
    ALL_ARMS,
    BASELINE_SPILL_WORDS,
    BASELINE_STACK_BYTES,
    DEFAULT_RESULTS,
    MAIN_BUNDLE_WORDS,
    TAIL_BUNDLE_WORDS,
    canonical_sha256,
    file_sha256,
    read_json,
    write_json,
)


STACK_OFFSET_RE = re.compile(r"\[R1(?:\+0x([0-9a-fA-F]+))?\]")
WIDTH_BY_OPCODE = {"STL": 1, "STL.64": 2, "STL.128": 4}
RESOURCE_PATTERNS = {
    "registers_per_thread": re.compile(r"(?:Used\s+)?(\d+)\s+registers", re.I),
    "stack_bytes_per_thread": re.compile(r"(\d+)\s+bytes stack frame", re.I),
    "static_spill_store_bytes": re.compile(r"(\d+)\s+bytes spill stores", re.I),
    "static_spill_load_bytes": re.compile(r"(\d+)\s+bytes spill loads", re.I),
    "shared_bytes_per_cta": re.compile(r"(\d+)\s+bytes smem", re.I),
}
NVDISASM_RESOURCE_RE = re.compile(
    r"REG\s*:\s*(\d+)\s+STACK\s*:\s*(\d+)\s+"
    r"SHARED\s*:\s*(\d+)\s+LOCAL\s*:\s*(\d+)",
    re.I,
)
SASS_INSTRUCTION_RE = re.compile(
    r"/\*\s*([0-9a-fA-F]+)\s*\*/\s+"
    r"(?:@[!A-Za-z0-9.]+\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;"
)


def stack_offset(operands: str) -> int | None:
    match = STACK_OFFSET_RE.search(operands)
    return None if match is None else int(match.group(1) or "0", 16)


def parse_resource_text(text: str) -> dict[str, int | None]:
    parsed = {
        name: (int(match.group(1)) if (match := pattern.search(text)) else None)
        for name, pattern in RESOURCE_PATTERNS.items()
    }
    # Prefer the exact REG/STACK/SHARED/LOCAL tuple emitted from the binary
    # (cuobjdump on the locked SM120 toolchain) when ptxas prose also exists.
    nvdisasm = NVDISASM_RESOURCE_RE.search(text)
    if nvdisasm:
        registers, stack, shared, local = map(int, nvdisasm.groups())
        parsed.update(
            {
                "registers_per_thread": registers,
                "stack_bytes_per_thread": stack,
                "shared_bytes_per_cta": shared,
                "nvdisasm_local_bytes": local,
                "authority": "binary_REG_STACK_SHARED_LOCAL_tuple",
            }
        )
    else:
        parsed.update(
            {
                "nvdisasm_local_bytes": None,
                "authority": "ptxas_prose_if_present",
            }
        )
    return parsed


def selected_opcode_projection(counts: Mapping[str, int]) -> dict[str, int]:
    groups = {
        "tensor": lambda opcode: "MMA" in opcode,
        "tma": lambda opcode: "TMA" in opcode or opcode.startswith("UTMA"),
        "ldsm": lambda opcode: opcode.startswith("LDSM"),
        "barrier": lambda opcode: "BAR" in opcode or "MBAR" in opcode,
        "atomic": lambda opcode: "ATOM" in opcode or "RED" in opcode,
        "global": lambda opcode: opcode.startswith(("LDG", "STG")),
        "local": lambda opcode: opcode.startswith(("LDL", "STL")),
    }
    return {
        name: sum(count for opcode, count in counts.items() if predicate(opcode))
        for name, predicate in groups.items()
    }


def disassembly_from_sass(
    text: str, *, cubin_sha256: str | None = None
) -> dict[str, Any]:
    """Convert target-function-only nvdisasm text to the VeloQ row shape.

    Callers must use ``nvdisasm -fun`` (or otherwise isolate exactly one
    MoEDynamicKernel). Multiple function headers are rejected.
    """
    # nvdisasm repeats the same symbol in a banner, a ``.section`` directive,
    # and a label.  Stop before punctuation so those three spellings collapse
    # to one function identity instead of looking like multiple functions.
    function_headers = re.findall(r"\.text\.([A-Za-z0-9_$.]+)", text)
    unique_headers = set(function_headers)
    if len(unique_headers) > 1:
        raise ValueError(
            "raw SASS contains multiple functions; rerun nvdisasm for only MoEDynamicKernel"
        )
    instructions = [
        {
            "address": int(match.group(1), 16),
            "opcode": match.group(2),
            "operands": match.group(3).strip(),
        }
        for match in SASS_INSTRUCTION_RE.finditer(text)
    ]
    if not instructions:
        raise ValueError("no nvdisasm instructions parsed from raw SASS")
    function_name = next(iter(unique_headers), "MoEDynamicKernel<nvdisasm-target>")
    if "MoEDynamicKernel" not in function_name:
        function_name = f"MoEDynamicKernel<{function_name}>"
    return {
        "data": {
            "rows": [
                {
                    "function_name": function_name,
                    "start": min(item["address"] for item in instructions),
                    "length": max(item["address"] for item in instructions) + 16,
                    "instructions": instructions,
                }
            ],
            "auxiliary": {
                "cubin_sha": cubin_sha256,
                "source_lineinfo_present": False,
            },
        }
    }


def analyze_disassembly(
    disassembly: Mapping[str, Any], *, arm: str, resource_text: str = ""
) -> dict[str, Any]:
    rows = [
        row
        for row in disassembly["data"]["rows"]
        if "MoEDynamicKernel" in str(row.get("function_name", ""))
    ]
    if len(rows) != 1:
        raise ValueError(f"{arm}: expected one MoEDynamicKernel, got {len(rows)}")
    row = rows[0]
    instructions = row["instructions"]
    counts = Counter(str(inst["opcode"]) for inst in instructions)
    stores = [inst for inst in instructions if str(inst["opcode"]).startswith("STL")]
    loads = [inst for inst in instructions if str(inst["opcode"]).startswith("LDL")]
    unsupported = sorted(
        {str(inst["opcode"]) for inst in stores} - set(WIDTH_BY_OPCODE)
    )
    if unsupported:
        raise ValueError(f"{arm}: unsupported local-store widths: {unsupported}")

    store_rows: list[dict[str, Any]] = []
    scalar_slots: set[int] = set()
    for inst in stores:
        opcode = str(inst["opcode"])
        base = stack_offset(str(inst.get("operands", "")))
        if base is None:
            raise ValueError(f"{arm}: local store has no R1 stack operand: {inst}")
        words = WIDTH_BY_OPCODE[opcode]
        slots = [base + index * 4 for index in range(words)]
        scalar_slots.update(slots)
        store_rows.append(
            {
                "pc": int(inst["address"]),
                "pc_hex": hex(int(inst["address"])),
                "opcode": opcode,
                "stack_offset": base,
                "stack_offset_hex": hex(base),
                "words": words,
                "slots": slots,
            }
        )
    load_offsets: dict[int, list[int]] = {}
    for inst in loads:
        offset = stack_offset(str(inst.get("operands", "")))
        if offset is not None:
            load_offsets.setdefault(offset, []).append(int(inst["address"]))
    later_reload_missing = [
        slot
        for store in store_rows
        for slot in store["slots"]
        if not any(pc > store["pc"] for pc in load_offsets.get(slot, ()))
    ]
    width_words = Counter()
    for store in store_rows:
        width_words[store["opcode"]] += int(store["words"])
    auxiliary = disassembly["data"].get("auxiliary", {})
    return {
        "arm": arm,
        "function": {
            "name": row["function_name"],
            "start": row["start"],
            "length": row["length"],
            "instruction_count": len(instructions),
            "cubin_sha256": auxiliary.get("cubin_sha"),
            "source_lineinfo_present": bool(auxiliary.get("source_lineinfo_present")),
        },
        "resource": parse_resource_text(resource_text),
        "opcode_counts": dict(sorted(counts.items())),
        "selected_opcode_projection": selected_opcode_projection(counts),
        "local": {
            "load_instruction_count": len(loads),
            "store_instruction_count": len(stores),
            "store_opcode_counts": {
                opcode: counts[opcode]
                for opcode in sorted(WIDTH_BY_OPCODE)
                if counts[opcode]
            },
            "stored_words": len(scalar_slots),
            "stored_words_by_opcode_width": dict(width_words),
            "stack_slots": sorted(scalar_slots),
            "stores": store_rows,
            "all_stored_slots_have_later_ldl": not later_reload_missing,
            "missing_later_ldl_slots": sorted(set(later_reload_missing)),
        },
    }


def compare_arms(facts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if "baseline" not in facts:
        raise ValueError("baseline static evidence is required")
    baseline = facts["baseline"]
    baseline_local = baseline["local"]
    baseline_resource = baseline["resource"]
    baseline_gate = {
        "stack_bytes_488": baseline_resource.get("stack_bytes_per_thread")
        == BASELINE_STACK_BYTES,
        "stored_words_122": baseline_local["stored_words"] == BASELINE_SPILL_WORDS,
        "main_108_words": baseline_local["stored_words_by_opcode_width"].get("STL.64")
        == MAIN_BUNDLE_WORDS,
        "tail_14_words": baseline_local["stored_words_by_opcode_width"].get("STL")
        == TAIL_BUNDLE_WORDS,
        "all_slots_roundtrip": baseline_local["all_stored_slots_have_later_ldl"],
    }
    deltas: dict[str, Any] = {}
    for arm, fact in facts.items():
        if arm == "baseline":
            continue
        local = fact["local"]
        resource = fact["resource"]
        removed_words = baseline_local["stored_words"] - local["stored_words"]
        expected_remaining = BASELINE_SPILL_WORDS - TAIL_BUNDLE_WORDS
        replacement_words = max(0, local["stored_words"] - expected_remaining)
        deltas[arm] = {
            "stored_words_delta": local["stored_words"]
            - baseline_local["stored_words"],
            "removed_words": removed_words,
            "replacement_words": replacement_words,
            "stack_bytes_delta": (
                None
                if resource.get("stack_bytes_per_thread") is None
                or baseline_resource.get("stack_bytes_per_thread") is None
                else resource["stack_bytes_per_thread"]
                - baseline_resource["stack_bytes_per_thread"]
            ),
            "selected_opcode_projection_equal_except_local": all(
                value == fact["selected_opcode_projection"].get(key)
                for key, value in baseline["selected_opcode_projection"].items()
                if key != "local"
            ),
            "complete_tail_removal_no_replacement": (
                removed_words == TAIL_BUNDLE_WORDS and replacement_words == 0
            ),
        }
    return {
        "schema": "exp003.spill-root-cause.static-spill-evidence.v1",
        "baseline_reproduction": baseline_gate,
        "baseline_reproduction_pass": all(baseline_gate.values()),
        "arms": facts,
        "deltas": deltas,
        "h108_boundary": (
            "static store/reload position and width do not identify a semantic value or "
            "source root cause; consume spill_root_cause_evidence.json for the verified "
            "producer/reuse/reload/consumer chains"
        ),
    }


def parse_arm_file(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        arm, separator, raw = value.partition("=")
        if not separator or arm not in ALL_ARMS:
            raise ValueError(f"expected ARM=PATH, got {value!r}")
        parsed[arm] = Path(raw).resolve()
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--disasm", action="append", default=[], help="ARM=veloq/disasm.json"
    )
    parser.add_argument(
        "--sass",
        action="append",
        default=[],
        help="ARM=target-function-only nvdisasm text",
    )
    parser.add_argument(
        "--resource-text", action="append", default=[], help="ARM=ptxas_or_resource.txt"
    )
    parser.add_argument(
        "--cubin",
        action="append",
        default=[],
        help="ARM=exact JIT cubin used to produce the SASS",
    )
    args = parser.parse_args(argv)
    disassembly_paths = parse_arm_file(args.disasm)
    sass_paths = parse_arm_file(args.sass)
    overlap = set(disassembly_paths) & set(sass_paths)
    if overlap:
        raise RuntimeError(
            f"provide either --disasm or --sass per arm, not both: {sorted(overlap)}"
        )
    source_paths = {**disassembly_paths, **sass_paths}
    resource_paths = parse_arm_file(args.resource_text)
    cubin_paths = parse_arm_file(args.cubin)
    if "baseline" not in source_paths:
        raise RuntimeError("baseline disassembly is mandatory")
    facts: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for arm, path in source_paths.items():
        resource_path = resource_paths.get(arm)
        resource_text = resource_path.read_text() if resource_path else ""
        cubin_path = cubin_paths.get(arm)
        cubin_sha256 = file_sha256(cubin_path) if cubin_path else None
        if arm in sass_paths and cubin_path is None:
            raise RuntimeError(f"{arm}: raw SASS requires its exact --cubin identity")
        preparation_path = args.results.resolve() / "arms" / arm / "preparation.json"
        if cubin_sha256 and preparation_path.is_file():
            preparation = read_json(preparation_path)
            if cubin_sha256 not in {
                item.get("sha256") for item in preparation.get("jit_artifacts", [])
            }:
                raise RuntimeError(
                    f"{arm}: SASS cubin is absent from the fresh-JIT artifact lock"
                )
        disassembly = (
            read_json(path)
            if arm in disassembly_paths
            else disassembly_from_sass(
                path.read_text(), cubin_sha256=cubin_sha256
            )
        )
        facts[arm] = analyze_disassembly(
            disassembly, arm=arm, resource_text=resource_text
        )
        identities[arm] = {
            "disassembly_kind": "veloq_json"
            if arm in disassembly_paths
            else "nvdisasm_target_text",
            "disassembly": str(path),
            "disassembly_sha256": file_sha256(path),
            "resource_text": str(resource_path) if resource_path else None,
            "resource_text_sha256": file_sha256(resource_path)
            if resource_path
            else None,
            "cubin": str(cubin_path) if cubin_path else None,
            "cubin_sha256": cubin_sha256,
            "fresh_jit_artifact_identity_gate": cubin_sha256 is not None,
        }
    payload = compare_arms(facts)
    payload["inputs"] = identities
    payload["evidence_sha256"] = canonical_sha256(payload)
    output = args.results.resolve() / "static_spill_evidence.json"
    write_json(output, payload)
    return 0 if payload["baseline_reproduction_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
