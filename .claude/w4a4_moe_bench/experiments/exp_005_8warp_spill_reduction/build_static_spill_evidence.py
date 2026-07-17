#!/usr/bin/env python3
"""Build reproducible binary resource/spill evidence for exp_005.

The collector is deliberately GPU-free.  It reads the already-built fresh-JIT
cubins, validates each cubin against its arm preparation manifest, and invokes
only ``cuobjdump``/``nvdisasm``.  ``--remote`` executes those read-only commands
through SSH when the retained artifacts live on the 5KP frontend.

This script closes physical stack save/reload facts.  It does not turn static
SASS presence into dynamic traffic or latency, and it does not claim an exact
source value mapping when line information/allocation maps are absent.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Mapping, Sequence

from exp005_common import (
    ALL_ARMS,
    BASELINE,
    CANDIDATE,
    DEFAULT_RESULTS,
    canonical_sha256,
    file_sha256,
    write_json,
)


SASS_INSTRUCTION_RE = re.compile(
    r"/\*\s*([0-9a-fA-F]+)\s*\*/\s+"
    r"(?:@[!A-Za-z0-9.]+\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;"
)
STACK_OFFSET_RE = re.compile(r"\[R1(?:\+0x([0-9a-fA-F]+))?\]")
REGISTER_RE = re.compile(r"(?<![A-Z0-9])R([0-9]+)(?![0-9])")
RESOURCE_RE = re.compile(
    r"REG\s*:\s*(\d+)\s+STACK\s*:\s*(\d+)\s+"
    r"SHARED\s*:\s*(\d+)\s+LOCAL\s*:\s*(\d+)",
    re.I,
)
FRAME_RE = re.compile(r"frame size:\s*0x([0-9a-fA-F]+)")
MIN_STACK_RE = re.compile(r"min stack size:\s*0x([0-9a-fA-F]+)")
SPILL_ANNOTATION_RE = re.compile(r"SpillRefill\s*:\s*Offset\s*:\s*0x([0-9a-fA-F]+)")
SECTION_LINE_RE = re.compile(
    r"^\s*[0-9a-fA-F]+\s+[0-9a-fA-F]+\s+[0-9a-fA-F]+\s+"
    r"\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\.\S+)\s*$",
    re.M,
)
FUNCTION_RE = re.compile(r"Function\s+(\S*MoEDynamicKernel\S*):")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_assignment(values: Sequence[str], *, label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        arm, separator, path = value.partition("=")
        if not separator or arm not in ALL_ARMS or not path:
            raise ValueError(f"{label}: expected ARM=PATH, got {value!r}")
        if arm in parsed:
            raise ValueError(f"{label}: duplicate arm {arm}")
        parsed[arm] = path
    return parsed


class CommandReader:
    """Read local or remote artifacts and run read-only binary tools."""

    def __init__(self, remote: str | None):
        self.remote = remote

    def run(self, argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        if self.remote:
            command = " ".join(shlex.quote(value) for value in argv)
            full = ["ssh", "-o", "BatchMode=yes", self.remote, command]
        else:
            full = list(argv)
        completed = subprocess.run(full, capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError(
                f"command failed ({completed.returncode}): {full!r}\n"
                + completed.stderr.decode(errors="replace")
            )
        return completed

    def read_bytes(self, path: str) -> bytes:
        if self.remote:
            return self.run(["cat", "--", path]).stdout
        return Path(path).read_bytes()

    def file_sha256(self, path: str) -> str:
        if self.remote:
            output = self.run(["sha256sum", "--", path]).stdout.decode()
            return output.split()[0]
        return file_sha256(Path(path))


def parse_sass(text: str) -> list[dict[str, Any]]:
    instructions = [
        {
            "pc": int(match.group(1), 16),
            "opcode": match.group(2),
            "operands": match.group(3).strip(),
        }
        for match in SASS_INSTRUCTION_RE.finditer(text)
    ]
    if not instructions:
        raise ValueError("no SASS instructions parsed")
    return instructions


def stack_offset(operands: str) -> int | None:
    match = STACK_OFFSET_RE.search(operands)
    return None if match is None else int(match.group(1) or "0", 16)


def local_width_bytes(opcode: str) -> int:
    if ".128" in opcode:
        return 16
    if ".64" in opcode:
        return 8
    return 4


def operand_parts(instruction: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(instruction["operands"]).split(","))


def register_span(base: int, width: int) -> tuple[int, ...]:
    return tuple(range(base, base + width))


def destination_registers(instruction: Mapping[str, Any]) -> frozenset[int]:
    """Return conservative, unambiguous GPR destinations for local tracing."""

    opcode = str(instruction["opcode"])
    if opcode.startswith(
        (
            "ST",
            "BRA",
            "BAR",
            "SYNCS",
            "MEMBAR",
            "FENCE",
            "EXIT",
            "NOP",
        )
    ):
        return frozenset()
    parts = operand_parts(instruction)
    if not parts:
        return frozenset()
    match = re.fullmatch(r"\s*R([0-9]+)(?:\.[A-Za-z0-9_]+)?\s*", parts[0])
    if match is None:
        return frozenset()
    base = int(match.group(1))
    if "MMA" in opcode or ".128" in opcode:
        width = 4
    elif ".64" in opcode:
        width = 2
    elif opcode.startswith("LDSM") and opcode.endswith(".4"):
        width = 4
    else:
        width = 1
    return frozenset(register_span(base, width))


def register_is_source(instruction: Mapping[str, Any], register: int) -> bool:
    opcode = str(instruction["opcode"])
    parts = operand_parts(instruction)
    source_parts = parts if opcode.startswith("ST") else parts[1:]
    token = re.compile(rf"(?<![A-Z0-9])R{register}(?![0-9])")
    return any(token.search(part) for part in source_parts)


def preceding_definition(
    instructions: Sequence[Mapping[str, Any]], before: int, register: int
) -> Mapping[str, Any] | None:
    for index in range(before - 1, -1, -1):
        instruction = instructions[index]
        if register in destination_registers(instruction):
            return instruction
    return None


def first_consumer(
    instructions: Sequence[Mapping[str, Any]], after: int, register: int
) -> Mapping[str, Any] | None:
    for index in range(after + 1, len(instructions)):
        instruction = instructions[index]
        if register_is_source(instruction, register):
            return instruction
        if register in destination_registers(instruction):
            return None
    return None


def instruction_identity(instruction: Mapping[str, Any] | None) -> Any:
    if instruction is None:
        return None
    return {
        "pc": int(instruction["pc"]),
        "pc_hex": hex(int(instruction["pc"])),
        "opcode": instruction["opcode"],
        "operands": instruction["operands"],
    }


def source_contract(path: Path) -> dict[str, Any]:
    text = path.read_text()
    lines = text.splitlines()
    markers = {
        "gate_acc_allocation": "gate_acc = cute.make_rmem_tensor",
        "gate_gemm_start": "# Gate GEMM (inlined",
        "pass_gate_barrier": "self.pass_gate_barrier.arrive_unaligned()",
        "up_gemm_start": "# Up GEMM (inlined",
        "activation_start": "# Activation + quant into sA",
        "activation_reads_gate": "gate_slice = tRS_rGate",
        "activation_reads_up": "up_slice = tRS_rUp",
    }
    located: dict[str, Any] = {}
    for name, needle in markers.items():
        matches = [index + 1 for index, line in enumerate(lines) if needle in line]
        if len(matches) != 1:
            raise ValueError(f"{path}: expected one {name!r} marker, got {matches}")
        located[name] = {"line": matches[0], "text": lines[matches[0] - 1].strip()}
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "markers": located,
        "source_fact": (
            "gate_acc is produced by Gate GEMM, remains live across the serial Up "
            "GEMM, and is read with up_acc by the activation loop"
        ),
    }


def preparation_identity(
    *, reader: CommandReader, path: str, arm: str, cubin_sha256: str
) -> dict[str, Any]:
    raw = reader.read_bytes(path)
    preparation = json.loads(raw)
    if preparation.get("arm") != arm:
        raise ValueError(f"{arm}: preparation arm mismatch")
    matches = [
        item
        for item in preparation.get("jit_artifacts", [])
        if item.get("sha256") == cubin_sha256
        and str(item.get("path", "")).endswith(".cubin")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{arm}: cubin hash is not uniquely locked by preparation.json"
        )
    return {
        "path": path,
        "sha256": sha256_bytes(raw),
        "schema": preparation.get("schema"),
        "arm": preparation.get("arm"),
        "case": preparation.get("case"),
        "locked_jit_artifact": matches[0],
        "compile_tools": {
            "nvcc": preparation.get("runtime", {}).get("nvcc"),
            "ptxas": preparation.get("runtime", {}).get("ptxas"),
        },
        "source": preparation.get("runtime", {}).get("source"),
    }


def analyze_arm(
    *,
    arm: str,
    cubin_path: str,
    preparation_path: str,
    source_path: Path,
    reader: CommandReader,
    cuobjdump: str,
    nvdisasm: str,
) -> dict[str, Any]:
    cubin_sha256 = reader.file_sha256(cubin_path)
    preparation = preparation_identity(
        reader=reader,
        path=preparation_path,
        arm=arm,
        cubin_sha256=cubin_sha256,
    )
    source = source_contract(source_path)
    locked_source_sha = preparation["source"].get("overlay_sha256")
    if source["sha256"] != locked_source_sha:
        raise ValueError(f"{arm}: local source does not match fresh-JIT source lock")

    resource_run = reader.run([cuobjdump, "--dump-resource-usage", cubin_path])
    elf_run = reader.run([cuobjdump, "--dump-elf", cubin_path])
    sass_run = reader.run([nvdisasm, "-c", cubin_path])
    resource_text = resource_run.stdout.decode(errors="replace")
    elf_text = elf_run.stdout.decode(errors="replace")
    sass_text = sass_run.stdout.decode(errors="replace")
    resource_match = RESOURCE_RE.search(resource_text)
    if resource_match is None:
        raise ValueError(f"{arm}: cannot parse REG/STACK/SHARED/LOCAL")
    registers, stack, shared, local = map(int, resource_match.groups())
    frame_match = FRAME_RE.search(elf_text)
    min_stack_match = MIN_STACK_RE.search(elf_text)
    if frame_match is None or min_stack_match is None:
        raise ValueError(f"{arm}: missing ELF frame/min-stack attributes")
    frame = int(frame_match.group(1), 16)
    minimum_stack = int(min_stack_match.group(1), 16)
    if frame != stack or minimum_stack != stack:
        raise ValueError(f"{arm}: ELF and resource stack sizes disagree")

    instructions = parse_sass(sass_text)
    pc_to_index = {int(item["pc"]): index for index, item in enumerate(instructions)}
    if len(pc_to_index) != len(instructions):
        raise ValueError(f"{arm}: duplicate SASS PCs")
    local_instructions = [
        item for item in instructions if str(item["opcode"]).startswith(("STL", "LDL"))
    ]
    annotations = {int(value, 16) for value in SPILL_ANNOTATION_RE.findall(elf_text)}
    local_pcs = {int(item["pc"]) for item in local_instructions}
    if annotations != local_pcs:
        raise ValueError(f"{arm}: compiler SpillRefill annotations != local SASS PCs")

    stores: list[dict[str, Any]] = []
    stored_slots: set[int] = set()
    producer_opcodes: Counter[str] = Counter()
    for item in local_instructions:
        opcode = str(item["opcode"])
        if not opcode.startswith("STL"):
            continue
        offset = stack_offset(str(item["operands"]))
        if offset is None:
            raise ValueError(f"{arm}: local store has no R1-relative offset: {item}")
        registers_in_operands = [
            int(value) for value in REGISTER_RE.findall(str(item["operands"]))
        ]
        if len(registers_in_operands) < 2 or registers_in_operands[0] != 1:
            raise ValueError(f"{arm}: cannot parse local-store source: {item}")
        source_base = registers_in_operands[-1]
        width = local_width_bytes(opcode)
        slots = list(range(offset, offset + width, 4))
        stored_slots.update(slots)
        index = pc_to_index[int(item["pc"])]
        traces = []
        for lane, slot in enumerate(slots):
            physical_register = source_base + lane
            producer = preceding_definition(instructions, index, physical_register)
            if producer is not None:
                producer_opcodes[str(producer["opcode"])] += 1
            traces.append(
                {
                    "stack_offset": slot,
                    "stack_offset_hex": hex(slot),
                    "physical_register": f"R{physical_register}",
                    "preceding_definition": instruction_identity(producer),
                }
            )
        stores.append(
            {
                "pc": int(item["pc"]),
                "pc_hex": hex(int(item["pc"])),
                "opcode": opcode,
                "operands": item["operands"],
                "width_bytes_per_lane": width,
                "stack_offset": offset,
                "stack_offset_hex": hex(offset),
                "slots": slots,
                "physical_value_trace": traces,
            }
        )

    loads: list[dict[str, Any]] = []
    loaded_slots: set[int] = set()
    consumer_opcodes: Counter[str] = Counter()
    for item in local_instructions:
        opcode = str(item["opcode"])
        if not opcode.startswith("LDL"):
            continue
        offset = stack_offset(str(item["operands"]))
        if offset is None:
            raise ValueError(f"{arm}: local load has no R1-relative offset: {item}")
        parts = operand_parts(item)
        destination_match = re.fullmatch(
            r"\s*R([0-9]+)(?:\.[A-Za-z0-9_]+)?\s*", parts[0]
        )
        if destination_match is None:
            raise ValueError(f"{arm}: cannot parse local-load destination: {item}")
        destination_base = int(destination_match.group(1))
        width = local_width_bytes(opcode)
        slots = list(range(offset, offset + width, 4))
        loaded_slots.update(slots)
        index = pc_to_index[int(item["pc"])]
        traces = []
        for lane, slot in enumerate(slots):
            physical_register = destination_base + lane
            consumer = first_consumer(instructions, index, physical_register)
            if consumer is not None:
                consumer_opcodes[str(consumer["opcode"])] += 1
            traces.append(
                {
                    "stack_offset": slot,
                    "stack_offset_hex": hex(slot),
                    "physical_register": f"R{physical_register}",
                    "first_consumer_before_redefinition": instruction_identity(
                        consumer
                    ),
                }
            )
        loads.append(
            {
                "pc": int(item["pc"]),
                "pc_hex": hex(int(item["pc"])),
                "opcode": opcode,
                "operands": item["operands"],
                "width_bytes_per_lane": width,
                "stack_offset": offset,
                "stack_offset_hex": hex(offset),
                "slots": slots,
                "physical_value_trace": traces,
            }
        )

    expected_slots = set(range(0, stack, 4))
    sections = sorted(set(SECTION_LINE_RE.findall(elf_text)))
    source_line_sections = [
        value for value in sections if value in {".debug_line", ".nv_debug_line_sass"}
    ]
    pass_gate_barriers = [
        item
        for item in instructions
        if str(item["opcode"]).startswith("BAR.ARV")
        and re.search(r"(?:^|\s)0x2(?:,|\s|$)", str(item["operands"]))
    ]
    if len(pass_gate_barriers) != 1:
        raise ValueError(f"{arm}: expected one BAR.ARV barrier-id 2")
    pass_gate_pc = int(pass_gate_barriers[0]["pc"])

    all_store_sources_from_mma = bool(stores) and all(
        trace["preceding_definition"] is not None
        and "MMA" in trace["preceding_definition"]["opcode"]
        for store in stores
        for trace in store["physical_value_trace"]
    )
    static_local_counts = Counter(str(item["opcode"]) for item in local_instructions)
    store_width = sum(item["width_bytes_per_lane"] for item in stores)
    load_width = sum(item["width_bytes_per_lane"] for item in loads)
    compiler_annotation_gate = annotations == local_pcs
    exact_frame_roundtrip = (
        stored_slots == expected_slots
        and loaded_slots == expected_slots
        and store_width == stack
        and load_width == stack
    )
    zero_spill = stack == 0 and not annotations and not local_instructions

    return {
        "arm": arm,
        "identity": {
            "cubin": cubin_path,
            "cubin_sha256": cubin_sha256,
            "preparation": preparation,
            "source": source,
        },
        "resource": {
            "registers_per_thread": registers,
            "stack_bytes_per_thread": stack,
            "shared_bytes_per_cta": shared,
            "static_local_bytes_outside_stack": local,
            "elf_frame_bytes_per_thread": frame,
            "elf_minimum_stack_bytes_per_thread": minimum_stack,
            "local_zero_interpretation": (
                "LOCAL=0 is static local allocation outside the stack; it does not "
                "negate compiler spill/refill in the nonzero stack frame"
            ),
        },
        "compiler_spill_refill": {
            "annotation_count": len(annotations),
            "annotation_pcs": [hex(value) for value in sorted(annotations)],
            "annotation_exactly_matches_local_sass": compiler_annotation_gate,
            "source_lineinfo_present": bool(source_line_sections),
            "source_line_sections": source_line_sections,
            "elf_sections": sections,
            "static_local_opcode_counts": dict(sorted(static_local_counts.items())),
            "static_local_instruction_count": len(local_instructions),
            "static_store_instruction_count": len(stores),
            "static_load_instruction_count": len(loads),
            "stored_32bit_words_per_lane": len(stored_slots),
            "loaded_32bit_words_per_lane": len(loaded_slots),
            "static_store_width_bytes_per_lane": store_width,
            "static_load_width_bytes_per_lane": load_width,
            "stack_slots": [hex(value) for value in sorted(expected_slots)],
            "exact_full_stack_save_reload": exact_frame_roundtrip,
            "stores": stores,
            "loads": loads,
        },
        "physical_trace_summary": {
            "pass_gate_barrier_id_2_pc": pass_gate_pc,
            "pass_gate_barrier_id_2_pc_hex": hex(pass_gate_pc),
            "store_pc_window": (
                [
                    hex(min(item["pc"] for item in stores)),
                    hex(max(item["pc"] for item in stores)),
                ]
                if stores
                else None
            ),
            "load_pc_window": (
                [
                    hex(min(item["pc"] for item in loads)),
                    hex(max(item["pc"] for item in loads)),
                ]
                if loads
                else None
            ),
            "stores_before_barrier": sum(item["pc"] < pass_gate_pc for item in stores),
            "stores_after_barrier": sum(item["pc"] > pass_gate_pc for item in stores),
            "store_source_preceding_definition_opcodes": dict(
                sorted(producer_opcodes.items())
            ),
            "all_store_source_words_have_preceding_mma_definition": (
                all_store_sources_from_mma
            ),
            "reload_first_consumer_opcodes": dict(sorted(consumer_opcodes.items())),
            "scope": (
                "physical register/program-order trace only; no source variable identity"
            ),
        },
        "gates": {
            "fresh_jit_identity": True,
            "compiler_annotation_closure": compiler_annotation_gate,
            "exact_full_stack_save_reload": exact_frame_roundtrip,
            "zero_spill_static_gate": zero_spill,
        },
        "tool_output_sha256": {
            "resource_stdout": sha256_bytes(resource_run.stdout),
            "elf_stdout": sha256_bytes(elf_run.stdout),
            "sass_stdout": sha256_bytes(sass_run.stdout),
        },
    }


def compare_arms(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    baseline = arms[BASELINE]
    candidate = arms[CANDIDATE]
    base_resource = baseline["resource"]
    cand_resource = candidate["resource"]
    base_spill = baseline["compiler_spill_refill"]
    cand_spill = candidate["compiler_spill_refill"]

    stack_base = int(base_resource["stack_bytes_per_thread"])
    stack_candidate = int(cand_resource["stack_bytes_per_thread"])
    words_base = int(base_spill["stored_32bit_words_per_lane"])
    words_candidate = int(cand_spill["stored_32bit_words_per_lane"])
    lineinfo_absent = not any(
        arm["compiler_spill_refill"]["source_lineinfo_present"] for arm in arms.values()
    )
    all_candidate_store_sources_from_mma = candidate["physical_trace_summary"][
        "all_store_source_words_have_preceding_mma_definition"
    ]
    return {
        "resource_delta_candidate_minus_baseline": {
            "registers_per_thread": int(cand_resource["registers_per_thread"])
            - int(base_resource["registers_per_thread"]),
            "stack_bytes_per_thread": stack_candidate - stack_base,
            "shared_bytes_per_cta": int(cand_resource["shared_bytes_per_cta"])
            - int(base_resource["shared_bytes_per_cta"]),
            "static_local_bytes_outside_stack": int(
                cand_resource["static_local_bytes_outside_stack"]
            )
            - int(base_resource["static_local_bytes_outside_stack"]),
        },
        "spill_delta_candidate_minus_baseline": {
            "stored_32bit_words_per_lane": words_candidate - words_base,
            "static_store_instruction_count": int(
                cand_spill["static_store_instruction_count"]
            )
            - int(base_spill["static_store_instruction_count"]),
            "static_load_instruction_count": int(
                cand_spill["static_load_instruction_count"]
            )
            - int(base_spill["static_load_instruction_count"]),
            "compiler_annotation_count": int(cand_spill["annotation_count"])
            - int(base_spill["annotation_count"]),
            "stack_reduction_pct": (
                None
                if stack_base == 0
                else 100.0 * (stack_base - stack_candidate) / stack_base
            ),
        },
        "candidate_zero_spill_static_gate": candidate["gates"][
            "zero_spill_static_gate"
        ],
        "candidate_gate_live_range_mapping": {
            "status": (
                "supported_physical_program_order_inference_not_source_line_proven"
                if lineinfo_absent and all_candidate_store_sources_from_mma
                else "unresolved"
            ),
            "source_fact": candidate["identity"]["source"]["source_fact"],
            "binary_facts": {
                "all_local_sass_is_compiler_annotated_spill_refill": cand_spill[
                    "annotation_exactly_matches_local_sass"
                ],
                "all_saved_words_have_preceding_mma_definition": (
                    all_candidate_store_sources_from_mma
                ),
                "pass_gate_barrier_id_2_pc_hex": candidate["physical_trace_summary"][
                    "pass_gate_barrier_id_2_pc_hex"
                ],
                "source_lineinfo_present": cand_spill["source_lineinfo_present"],
            },
            "allowed_conclusion": (
                "Candidate retains a 224-byte/thread compiler spill roundtrip for "
                "56 32-bit words/lane.  The source live range and physical SASS "
                "pattern support investigating gate_acc first."
            ),
            "not_proven": (
                "The cubin contains no source line information or SSA-to-physical "
                "allocation map, so these 56 physical words cannot all be formally "
                "named gate_acc from this artifact alone."
            ),
        },
        "evidence_boundary": [
            "Static SASS and ELF annotations prove instruction presence and stack layout, not dynamic execution count or latency.",
            "The two arms change an inseparable 4-to-8-warp/layout bundle; no latency change may be attributed only to spill.",
            "LOCAL=0 means no static local allocation outside stack; it does not mean no spill when STACK and compiler SpillRefill are nonzero.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", help="SSH host holding retained artifacts")
    parser.add_argument("--arm", action="append", required=True, help="ARM=CUBIN")
    parser.add_argument(
        "--preparation", action="append", required=True, help="ARM=preparation.json"
    )
    parser.add_argument(
        "--source", action="append", required=True, help="ARM=local overlay source"
    )
    parser.add_argument(
        "--cuobjdump", default="cuobjdump", help="local/remote cuobjdump path"
    )
    parser.add_argument(
        "--nvdisasm", default="nvdisasm", help="local/remote nvdisasm path"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_RESULTS / "static_spill_evidence.json"
    )
    args = parser.parse_args(argv)

    cubins = parse_assignment(args.arm, label="--arm")
    preparations = parse_assignment(args.preparation, label="--preparation")
    sources = parse_assignment(args.source, label="--source")
    for label, values in (
        ("--arm", cubins),
        ("--preparation", preparations),
        ("--source", sources),
    ):
        if set(values) != set(ALL_ARMS):
            raise ValueError(f"{label}: both exp_005 arms are mandatory")

    reader = CommandReader(args.remote)
    nvdisasm_version = reader.run([args.nvdisasm, "--version"]).stdout.decode()
    cuobjdump_version = reader.run([args.cuobjdump, "--version"]).stdout.decode()
    arms = {
        arm: analyze_arm(
            arm=arm,
            cubin_path=cubins[arm],
            preparation_path=preparations[arm],
            source_path=Path(sources[arm]).resolve(),
            reader=reader,
            cuobjdump=args.cuobjdump,
            nvdisasm=args.nvdisasm,
        )
        for arm in ALL_ARMS
    }
    payload = {
        "schema": "exp005.static-resource-spill-evidence.v1",
        "status": (
            "static_candidate_zero_spill_gate_passed"
            if arms[CANDIDATE]["gates"]["zero_spill_static_gate"]
            else "static_candidate_zero_spill_gate_failed"
        ),
        "scope": {
            "case": "M256 fresh-JIT binary identity; applicability to other M values requires matching cubin hashes",
            "analysis_mode": "paired inseparable 4-to-8-warp ownership/layout bundle",
            "gpu_used_by_collector": False,
        },
        "collector": {
            "remote": args.remote,
            "nvdisasm": args.nvdisasm,
            "nvdisasm_version": nvdisasm_version.strip(),
            "cuobjdump": args.cuobjdump,
            "cuobjdump_version": cuobjdump_version.strip(),
            "commands_are_binary_read_only": True,
        },
        "arms": arms,
        "comparison": compare_arms(arms),
    }
    payload["evidence_sha256"] = canonical_sha256(payload)
    write_json(args.output.resolve(), payload)
    return 0 if payload["comparison"]["candidate_zero_spill_static_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
