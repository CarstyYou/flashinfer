#!/usr/bin/env python3
"""Build exact spill root-cause evidence from retained compiler artifacts.

This script does not estimate spill latency and does not propose a production
optimization. It verifies the physical SASS save/reuse/restore chains, closes
the dominant bundle's source mechanism, and records the deferred tail mapping.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"
SASS_RE = re.compile(
    r"/\*\s*([0-9a-fA-F]+)\s*\*/\s+"
    r"(?:@[!A-Za-z0-9.]+\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;"
)
STACK_RE = re.compile(r"\[R1(?:\+0x([0-9a-fA-F]+))?\]")


@dataclass(frozen=True)
class Instruction:
    pc: int
    opcode: str
    operands: str

    def to_json(self) -> dict[str, Any]:
        return {
            "pc": self.pc,
            "pc_hex": hex(self.pc),
            "opcode": self.opcode,
            "operands": self.operands,
        }


@dataclass(frozen=True)
class TailChain:
    stack_slot: int
    register: str
    value_class: str
    producer_pc: int
    producer_detail: str
    store_pc: int
    temporary_reuse_pcs: tuple[int, ...]
    reload_pc: int
    first_original_consumer_pc: int
    first_original_consumer_detail: str


TAIL_CHAINS = (
    TailChain(
        0x1E4,
        "R151",
        "second_pass_accumulator",
        0xB9E0,
        "OMMA R148..R151",
        0xBB40,
        (0xBDC0,),
        0xF050,
        0x11470,
        "FMUL scale",
    ),
    TailChain(
        0x1E0,
        "R72",
        "second_pass_accumulator",
        0xBA60,
        "OMMA R72..R75",
        0xBB60,
        (0xBE20,),
        0xF080,
        0x10540,
        "FMUL scale",
    ),
    TailChain(
        0x1DC,
        "R73",
        "second_pass_accumulator",
        0xBA60,
        "OMMA R72..R75",
        0xBB80,
        (0xBE80,),
        0xF090,
        0xFF80,
        "FMUL scale",
    ),
    TailChain(
        0x1D8,
        "R74",
        "second_pass_accumulator",
        0xBA60,
        "OMMA R72..R75",
        0xBB90,
        (0xBEE0,),
        0xF0B0,
        0xFB30,
        "FMUL scale",
    ),
    TailChain(
        0x1D4,
        "R75",
        "second_pass_accumulator",
        0xBA60,
        "OMMA R72..R75",
        0xBBA0,
        (0xBF40,),
        0xF170,
        0x10D70,
        "FMUL scale",
    ),
    TailChain(
        0x1D0,
        "R251",
        "index_address_scalar",
        0x37C0,
        "SHF index",
        0xBBB0,
        (0xBFA0,),
        0xF260,
        0x15140,
        "IMAD address",
    ),
    TailChain(
        0x1CC,
        "R250",
        "index_address_scalar",
        0x37D0,
        "SHF index",
        0xBBC0,
        (0xC000,),
        0xF2A0,
        0x15150,
        "IMAD address",
    ),
    TailChain(
        0x1C8,
        "R249",
        "index_address_scalar",
        0x37E0,
        "SHF index",
        0xBBD0,
        (0xC060,),
        0xF2E0,
        0x15160,
        "IMAD address",
    ),
    TailChain(
        0x1C4,
        "R248",
        "index_address_scalar",
        0x37F0,
        "SHF index",
        0xBBE0,
        (0xC0C0,),
        0xF320,
        0x15170,
        "IMAD address",
    ),
    TailChain(
        0x1C0,
        "R247",
        "index_address_scalar",
        0x3800,
        "SHF index",
        0xBBF0,
        (0xC140,),
        0xF360,
        0x15180,
        "IMAD address",
    ),
    TailChain(
        0x1BC,
        "R246",
        "index_address_scalar",
        0x3810,
        "SHF index",
        0xBC00,
        (0xC1A0,),
        0xF390,
        0x15190,
        "IMAD address",
    ),
    TailChain(
        0x1B8,
        "R245",
        "index_address_scalar",
        0x3820,
        "SHF index",
        0xBC10,
        (0xC240,),
        0xF3B0,
        0x151A0,
        "IMAD address",
    ),
    TailChain(
        0x1B4,
        "R2",
        "long_lived_control_scalar",
        0xB650,
        "LDG.E scalar",
        0xBC20,
        (0xBCA0, 0xC2A0),
        0xF3C0,
        0x120F0,
        "FSETP control",
    ),
    TailChain(
        0x1B0,
        "R0",
        "index_address_scalar",
        0x37B0,
        "IMAD index",
        0xBC30,
        (0xBCB0, 0xC300),
        0xF3D0,
        0x151F0,
        "IMAD address",
    ),
)


TAIL_PRODUCER_OPCODE_PREFIXES = {
    0xB9E0: "OMMA.",
    0xBA60: "OMMA.",
    0x37B0: "IMAD",
    0x37C0: "SHF.",
    0x37D0: "SHF.",
    0x37E0: "SHF.",
    0x37F0: "SHF.",
    0x3800: "SHF.",
    0x3810: "SHF.",
    0x3820: "SHF.",
    0xB650: "LDG.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_sass(path: Path) -> list[Instruction]:
    instructions = [
        Instruction(int(match.group(1), 16), match.group(2), match.group(3).strip())
        for match in SASS_RE.finditer(path.read_text())
    ]
    if not instructions:
        raise ValueError(f"no SASS instructions parsed from {path}")
    return instructions


def stack_offset(instruction: Instruction) -> int | None:
    match = STACK_RE.search(instruction.operands)
    return None if match is None else int(match.group(1) or "0", 16)


def assert_contains(instruction: Instruction, *fragments: str) -> None:
    missing = [
        fragment for fragment in fragments if fragment not in instruction.operands
    ]
    if missing:
        raise AssertionError(
            f"{hex(instruction.pc)} {instruction.opcode} is missing {missing}: "
            f"{instruction.operands}"
        )


def operand_parts(instruction: Instruction) -> tuple[str, ...]:
    return tuple(part.strip() for part in instruction.operands.split(","))


def register_number(register: str) -> int:
    match = re.fullmatch(r"R([0-9]+)", register)
    if match is None:
        raise AssertionError(f"not a physical general register: {register}")
    return int(match.group(1))


def register_token_present(text: str, register: str) -> bool:
    token = re.compile(rf"(?<![A-Z0-9]){re.escape(register)}(?![0-9])")
    return token.search(text) is not None


def written_registers(instruction: Instruction) -> frozenset[str]:
    """Return GPRs unambiguously defined by the relevant SASS instruction."""

    if instruction.opcode.startswith(("ST", "BRA", "EXIT", "NOP")):
        return frozenset()
    parts = operand_parts(instruction)
    if not parts:
        return frozenset()
    match = re.fullmatch(r"R([0-9]+)(?:\.[A-Za-z0-9_]+)?", parts[0])
    if match is None:
        return frozenset()
    base = int(match.group(1))
    width = 4 if instruction.opcode.startswith("OMMA.") else 1
    return frozenset(f"R{base + index}" for index in range(width))


def register_is_source(instruction: Instruction, register: str) -> bool:
    """Check a GPR occurrence in source operands, excluding the first destination."""

    parts = operand_parts(instruction)
    return any(register_token_present(part, register) for part in parts[1:])


def last_register_definition(
    instructions: list[Instruction], before_index: int, register: str
) -> Instruction | None:
    for instruction in reversed(instructions[:before_index]):
        if register in written_registers(instruction):
            return instruction
    return None


def first_register_reference(
    instructions: list[Instruction], after_index: int, register: str
) -> Instruction | None:
    for instruction in instructions[after_index + 1 :]:
        if register_token_present(instruction.operands, register):
            return instruction
    return None


def verify_tail_chain(
    instructions: list[Instruction], chain: TailChain
) -> dict[str, Any]:
    by_pc = {instruction.pc: instruction for instruction in instructions}
    index_by_pc = {
        instruction.pc: index for index, instruction in enumerate(instructions)
    }
    required = (
        chain.producer_pc,
        chain.store_pc,
        *chain.temporary_reuse_pcs,
        chain.reload_pc,
        chain.first_original_consumer_pc,
    )
    missing = [hex(pc) for pc in required if pc not in by_pc]
    if missing:
        raise AssertionError(f"tail chain has missing PCs: {missing}")

    producer = by_pc[chain.producer_pc]
    store = by_pc[chain.store_pc]
    reuse = [by_pc[pc] for pc in chain.temporary_reuse_pcs]
    reload = by_pc[chain.reload_pc]
    consumer = by_pc[chain.first_original_consumer_pc]
    expected_producer_prefix = TAIL_PRODUCER_OPCODE_PREFIXES[chain.producer_pc]
    if not producer.opcode.startswith(expected_producer_prefix):
        raise AssertionError(
            f"tail producer opcode drift at {hex(chain.producer_pc)}: {producer}"
        )
    if chain.register not in written_registers(producer):
        raise AssertionError(
            f"tail producer does not define {chain.register}: {producer}"
        )
    last_def = last_register_definition(
        instructions, index_by_pc[chain.store_pc], chain.register
    )
    if last_def is None or last_def.pc != chain.producer_pc:
        raise AssertionError(
            f"tail producer is not the last {chain.register} definition before store: "
            f"{last_def}"
        )
    if store.opcode != "STL" or stack_offset(store) != chain.stack_slot:
        raise AssertionError(f"bad tail store at {hex(chain.store_pc)}: {store}")
    if not register_is_source(store, chain.register):
        raise AssertionError(f"tail store does not save {chain.register}: {store}")
    for reuse_instruction in reuse:
        if chain.register not in written_registers(reuse_instruction):
            raise AssertionError(
                f"tail reuse PC does not overwrite {chain.register}: "
                f"{reuse_instruction}"
            )
    if not reload.opcode.startswith("LDL") or stack_offset(reload) != chain.stack_slot:
        raise AssertionError(f"bad tail reload at {hex(chain.reload_pc)}: {reload}")
    if chain.register not in written_registers(reload):
        raise AssertionError(f"tail reload changed physical register: {reload}")
    first_reference = first_register_reference(
        instructions, index_by_pc[chain.reload_pc], chain.register
    )
    if (
        first_reference is None
        or first_reference.pc != chain.first_original_consumer_pc
    ):
        raise AssertionError(
            f"tail first post-reload reference to {chain.register} is not the declared "
            f"consumer: {first_reference}"
        )
    if not register_is_source(consumer, chain.register):
        raise AssertionError(
            f"tail first post-reload consumer does not read {chain.register}: {consumer}"
        )
    if chain.first_original_consumer_detail.startswith("FMUL"):
        expected_consumer_prefix = "FMUL"
    elif chain.first_original_consumer_detail.startswith("IMAD"):
        expected_consumer_prefix = "IMAD"
    else:
        expected_consumer_prefix = "FSETP."
    if not consumer.opcode.startswith(expected_consumer_prefix):
        raise AssertionError(
            f"tail consumer opcode drift at {hex(chain.first_original_consumer_pc)}: "
            f"{consumer}"
        )
    if not (
        chain.producer_pc
        < chain.store_pc
        < min(chain.temporary_reuse_pcs)
        <= max(chain.temporary_reuse_pcs)
        < chain.reload_pc
        < chain.first_original_consumer_pc
    ):
        raise AssertionError(f"invalid tail chain order for {chain.register}")
    return {
        **asdict(chain),
        "stack_slot_hex": hex(chain.stack_slot),
        "producer": producer.to_json(),
        "store": store.to_json(),
        "temporary_reuse": [item.to_json() for item in reuse],
        "reload": reload.to_json(),
        "first_original_consumer": consumer.to_json(),
        "producer_is_last_definition_before_store": True,
        "reuse_pcs_overwrite_physical_register": True,
        "consumer_is_first_post_reload_reference": True,
        "consumer_reads_reloaded_register": True,
        "physical_chain_closed": True,
        "unique_source_ssa_closed": False
        if chain.value_class != "second_pass_accumulator"
        else None,
    }


def verify_tail_chains(instructions: list[Instruction]) -> list[dict[str, Any]]:
    rows = [verify_tail_chain(instructions, chain) for chain in TAIL_CHAINS]
    if len(rows) != 14:
        raise AssertionError("tail chain must contain exactly 14 stack words")
    return rows


def main_store_source_register(store: Instruction) -> str:
    parts = operand_parts(store)
    if len(parts) != 2 or re.fullmatch(r"R[0-9]+", parts[1]) is None:
        raise AssertionError(f"cannot identify STL.64 source pair: {store}")
    return parts[1]


def verify_main_representative_chain(
    instructions: list[Instruction], producer_store_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_pc = {instruction.pc: instruction for instruction in instructions}
    expected = {
        0x7810: (
            "OMMA.",
            ("R12", "R68", "R84", "R128", "R24", "R52", "URZ"),
        ),
        0x7860: ("STL.64", ("[R1+0x170]", "R12")),
        0x7880: ("STL.64", ("[R1+0x178]", "R14")),
        0xBA90: ("LDL", ("R203", "[R1+0x170]")),
        0xBAA0: ("LDL", ("R202", "[R1+0x174]")),
        0xBAB0: ("LDL", ("R201", "[R1+0x178]")),
        0xBAC0: ("LDL", ("R200", "[R1+0x17c]")),
        0xBCE0: ("FMUL", ("R8", "R203", "UR12")),
        0xBD00: ("FMUL", ("R9", "R202", "UR12")),
        0xBDB0: ("FMUL", ("R8", "R201", "UR12")),
        0xBDD0: ("FMUL", ("R9", "R200", "UR12")),
        0xBD80: ("FMUL", ("R10", "R8", "R10")),
        0xBD90: ("FMUL", ("R8", "R60", "UR12")),
        0xBDA0: ("FMUL", ("R10", "R8", "R10")),
    }
    missing = [hex(pc) for pc in expected if pc not in by_pc]
    if missing:
        raise AssertionError(f"main representative chain has missing PCs: {missing}")
    for pc, (opcode_prefix, operands) in expected.items():
        instruction = by_pc[pc]
        if not instruction.opcode.startswith(opcode_prefix):
            raise AssertionError(
                f"main representative opcode drift at {hex(pc)}: {instruction}"
            )
        parts = operand_parts(instruction)
        if parts != operands:
            raise AssertionError(
                f"main representative operands drift at {hex(pc)}: "
                f"expected {operands}, got {parts}"
            )
    representative_row = next(
        (row for row in producer_store_rows if row["producer"]["pc"] == 0x7810),
        None,
    )
    if representative_row is None:
        raise AssertionError("representative OMMA 0x7810 has no producer/store closure")
    if [item["pc"] for item in representative_row["stores"]] != [0x7860, 0x7880]:
        raise AssertionError(
            "representative OMMA does not close through both STL.64 pairs"
        )
    return [by_pc[pc].to_json() for pc in expected]


def verify_main_bundle(
    instructions: list[Instruction], *, verify_representative: bool = True
) -> dict[str, Any]:
    stores_with_index = [
        (index, instruction)
        for index, instruction in enumerate(instructions)
        if instruction.opcode == "STL.64"
    ]
    stores = [instruction for _, instruction in stores_with_index]
    if len(stores) != 54:
        raise AssertionError(f"expected 54 STL.64, got {len(stores)}")
    slots: list[int] = []
    for store in stores:
        offset = stack_offset(store)
        if offset is None:
            raise AssertionError(f"STL.64 has no stack offset: {store}")
        slots.extend((offset, offset + 4))
    expected_slots = list(range(0, 0x1B0, 4))
    if sorted(slots) != expected_slots:
        raise AssertionError("108-word main bundle does not cover stack[0x000..0x1ac]")

    producer_groups: dict[int, dict[str, Any]] = {}
    for store_index, store in stores_with_index:
        source_base = main_store_source_register(store)
        source_number = register_number(source_base)
        source_pair = (source_base, f"R{source_number + 1}")
        producers = [
            last_register_definition(instructions, store_index, register)
            for register in source_pair
        ]
        if any(producer is None for producer in producers):
            raise AssertionError(f"STL.64 source has no producer: {store}")
        producer = producers[0]
        if producer is None or producers[1] is None:
            raise AssertionError(f"STL.64 source has no producer: {store}")
        if producers[1].pc != producer.pc or not producer.opcode.startswith("OMMA."):
            raise AssertionError(
                f"STL.64 source pair does not share one nearest OMMA producer: "
                f"{store}, {producers}"
            )
        produced = written_registers(producer)
        if not set(source_pair).issubset(produced):
            raise AssertionError(
                f"STL.64 source pair is outside producer output vector: "
                f"{store}, {producer}"
            )
        group = producer_groups.setdefault(
            producer.pc,
            {
                "producer": producer.to_json(),
                "output_registers": sorted(
                    produced, key=lambda register: register_number(register)
                ),
                "stores": [],
                "stored_registers": [],
            },
        )
        group["stores"].append(store.to_json())
        group["stored_registers"].extend(source_pair)
    if len(producer_groups) != 27:
        raise AssertionError(
            f"expected 27 unique OMMA output vectors, got {len(producer_groups)}"
        )
    producer_store_rows = sorted(
        producer_groups.values(), key=lambda row: row["producer"]["pc"]
    )
    for row in producer_store_rows:
        if len(row["stores"]) != 2:
            raise AssertionError(
                f"OMMA vector must close through two STL.64 stores: {row}"
            )
        if set(row["stored_registers"]) != set(row["output_registers"]):
            raise AssertionError(
                f"OMMA output vector is not fully covered by STL.64 stores: {row}"
            )

    reloads_by_slot: dict[int, list[tuple[int, Instruction]]] = {}
    for index, instruction in enumerate(instructions):
        if instruction.opcode.startswith("LDL"):
            offset = stack_offset(instruction)
            if offset is not None:
                reloads_by_slot.setdefault(offset, []).append((index, instruction))
    reload_rows: list[dict[str, Any]] = []
    for slot in expected_slots:
        matches = reloads_by_slot.get(slot, [])
        if len(matches) != 1:
            raise AssertionError(f"stack slot {hex(slot)} has {len(matches)} reloads")
        index, reload = matches[0]
        register = reload.operands.split(",", 1)[0].strip()
        if register not in written_registers(reload):
            raise AssertionError(f"main reload does not define {register}: {reload}")
        use = first_register_reference(instructions, index, register)
        if (
            use is None
            or use.opcode != "FMUL"
            or "UR12" not in use.operands
            or not register_is_source(use, register)
        ):
            raise AssertionError(
                f"main reload {hex(reload.pc)} {register} first use is not scale FMUL: {use}"
            )
        reload_rows.append(
            {
                "stack_slot": slot,
                "stack_slot_hex": hex(slot),
                "reload": reload.to_json(),
                "first_use": use.to_json(),
            }
        )
    representative_chain = (
        verify_main_representative_chain(instructions, producer_store_rows)
        if verify_representative
        else []
    )
    return {
        "value_class": "first_pass_accumulator",
        "mechanism": (
            "during the first FC1 pass tail, 27 completed accumulator vectors are "
            "progressively spilled after their respective nearest OMMA producers and "
            "kept live across the complete second FC1 pass"
        ),
        "omma_output_vector_count": len(producer_store_rows),
        "stl64_instruction_count": len(stores),
        "store_pc_range": [
            hex(min(item.pc for item in stores)),
            hex(max(item.pc for item in stores)),
        ],
        "stack_word_count": len(expected_slots),
        "stack_slot_range": [hex(expected_slots[0]), hex(expected_slots[-1])],
        "store_stack_slots": expected_slots,
        "ldl_reload_count": len(reload_rows),
        "reload_stack_slots": [item["stack_slot"] for item in reload_rows],
        "reload_pc_range": [
            hex(min(item["reload"]["pc"] for item in reload_rows)),
            hex(max(item["reload"]["pc"] for item in reload_rows)),
        ],
        "reload_first_use_is_scale_fmul_count": sum(
            item["first_use"]["opcode"] == "FMUL"
            and "UR12" in item["first_use"]["operands"]
            for item in reload_rows
        ),
        "physical_chain_closed": True,
        "producer_store_chain_count": len(producer_store_rows),
        "producer_store_chains": producer_store_rows,
        "representative_chain": representative_chain,
        "per_slot_reload_first_use": reload_rows,
    }


def line_number(text: str, needle: str) -> int:
    for number, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return number
    raise AssertionError(f"artifact is missing expected text: {needle}")


def verify_cross_layer_semantics(
    source: Path,
    baseline_mlir: Path,
    baseline_ptx: Path,
    up_first_ptx: Path,
) -> dict[str, Any]:
    source_text = source.read_text()
    mlir_text = baseline_mlir.read_text()
    baseline_ptx_text = baseline_ptx.read_text()
    up_first_ptx_text = up_first_ptx.read_text()
    source_lines = {
        "gate_acc_alloc": line_number(source_text, "gate_acc = cute.make_rmem_tensor"),
        "up_acc_alloc": line_number(source_text, "up_acc = ("),
        "gate_gemm_phase": line_number(source_text, "# Gate GEMM"),
        "up_gemm_phase": line_number(source_text, "# Up GEMM"),
        "activation_call": line_number(source_text, "gated_activation_f32("),
    }
    mlir_lines = {
        "gate_rmem_alloc": line_number(mlir_text, "%rmem_118 = cute.memref.alloca()"),
        "up_rmem_alloc": line_number(mlir_text, "%rmem_119 = cute.memref.alloca()"),
        "gate_gemm": line_number(mlir_text, "cute.gemm(%370, %slice_516"),
        "up_gemm": line_number(mlir_text, "cute.gemm(%370, %slice_511"),
        "gate_activation_load": line_number(
            mlir_text, "%369 = cute.memref.load(%slice_513"
        ),
        "up_activation_load": line_number(
            mlir_text, "%371 = cute.memref.load(%slice_514"
        ),
    }
    baseline_ptx_lines = {
        "first_pass_mma_def": line_number(
            baseline_ptx_text, "{%r2878, %r2879, %r2880, %r2881}"
        ),
        "second_pass_mma_def": line_number(
            baseline_ptx_text, "{%r4955, %r4956, %r4957, %r4958}"
        ),
        "first_pass_activation_use": line_number(
            baseline_ptx_text, "%r5524, %r38, %r2878"
        ),
        "second_pass_activation_use": line_number(
            baseline_ptx_text, "%r5525, %r38, %r4955"
        ),
    }
    up_first_ptx_lines = {
        "second_pass_activation_operand_first": line_number(
            up_first_ptx_text, "%r5524, %r38, %r4955"
        ),
        "first_pass_activation_operand_second": line_number(
            up_first_ptx_text, "%r5525, %r38, %r2878"
        ),
    }
    return {
        "source": {
            "path": str(source.resolve()),
            "sha256": sha256(source),
            "semantic_lines": source_lines,
        },
        "baseline_mlir": {
            "path": str(baseline_mlir.resolve()),
            "sha256": sha256(baseline_mlir),
            "semantic_lines": mlir_lines,
            "internal_accumulator_def_use_closed": True,
            "source_locations_present": False,
        },
        "baseline_ptx": {
            "path": str(baseline_ptx.resolve()),
            "sha256": sha256(baseline_ptx),
            "semantic_lines": baseline_ptx_lines,
            "internal_accumulator_def_use_closed": True,
            "source_locations_present": bool(
                re.search(r"^\s*\.loc\b", baseline_ptx_text, re.MULTILINE)
            ),
            "local_load_count": baseline_ptx_text.count("ld.local"),
            "local_store_count": baseline_ptx_text.count("st.local"),
        },
        "up_first_ptx": {
            "path": str(up_first_ptx.resolve()),
            "sha256": sha256(up_first_ptx),
            "semantic_lines": up_first_ptx_lines,
            "activation_operands_swapped": True,
        },
        "compiler_certified_virtual_to_physical_register_map": False,
        "compiler_certified_source_value_to_stack_slot_map": False,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"identity input is not a file: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def assert_identity_equal(label: str, *values: Any) -> None:
    if not values or any(value != values[0] for value in values[1:]):
        raise AssertionError(f"identity drift for {label}: {values}")


def verify_evidence_identity(
    static: dict[str, Any],
    ncu: dict[str, Any],
    input_files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if static.get("schema") not in {
        "exp004.static-spill-evidence.v1",
        "exp003.spill-root-cause.static-spill-evidence.v1",
    }:
        raise AssertionError(
            f"unexpected static evidence schema: {static.get('schema')}"
        )
    if ncu.get("schema") not in {
        "exp004.ncu-spill-evidence.v1",
        "exp003.spill-root-cause.ncu-spill-evidence.v1",
    }:
        raise AssertionError(f"unexpected NCU evidence schema: {ncu.get('schema')}")
    if not static.get("baseline_reproduction_pass"):
        raise AssertionError("static baseline reproduction identity gate failed")
    if not ncu.get("baseline_reproduction_pass"):
        raise AssertionError("NCU baseline reproduction identity gate failed")

    arms = ("baseline", "up_first_attribution")
    for arm in arms:
        static_input = static["inputs"][arm]
        static_arm = static["arms"][arm]
        ncu_input = ncu["inputs"][arm]
        launch = ncu["launches"][arm]
        if not static_input.get("fresh_jit_artifact_identity_gate"):
            raise AssertionError(f"static fresh-JIT identity gate failed for {arm}")
        if not ncu_input.get("profile_jit_artifact_identity_gate"):
            raise AssertionError(f"profile/JIT identity gate failed for {arm}")
        if not ncu_input.get("benchmark_cubin_identity_gate"):
            raise AssertionError(f"benchmark/cubin identity gate failed for {arm}")
        assert_identity_equal(
            f"{arm} cubin",
            static_input["cubin_sha256"],
            static_arm["function"]["cubin_sha256"],
            ncu_input["cubin_sha256"],
        )
        assert_identity_equal(
            f"{arm} kernel",
            static_arm["function"]["name"],
            launch["kernel"],
        )
        if launch.get("launch_count") != 1:
            raise AssertionError(
                f"expected one selected NCU launch for {arm}: {launch}"
            )

    assert_identity_equal(
        "kernel across arms",
        static["arms"]["baseline"]["function"]["name"],
        static["arms"]["up_first_attribution"]["function"]["name"],
    )
    assert_identity_equal(
        "grid across arms",
        ncu["launches"]["baseline"]["grid"],
        ncu["launches"]["up_first_attribution"]["grid"],
    )
    assert_identity_equal(
        "block across arms",
        ncu["launches"]["baseline"]["block"],
        ncu["launches"]["up_first_attribution"]["block"],
    )
    if not ncu["deltas"]["up_first_attribution"].get("work_identity_pass"):
        raise AssertionError("NCU work-identity gate failed")
    if not static["deltas"]["up_first_attribution"].get(
        "complete_tail_removal_no_replacement"
    ):
        raise AssertionError("static up-first tail-removal identity gate failed")
    assert_identity_equal(
        "baseline SASS",
        input_files["baseline_sass"]["sha256"],
        static["inputs"]["baseline"]["disassembly_sha256"],
    )
    assert_identity_equal(
        "up-first SASS",
        input_files["up_first_sass"]["sha256"],
        static["inputs"]["up_first_attribution"]["disassembly_sha256"],
    )
    if input_files["baseline_ptx"]["sha256"] == input_files["up_first_ptx"]["sha256"]:
        raise AssertionError("baseline and up-first PTX identities unexpectedly match")
    return {
        "kernel": static["arms"]["baseline"]["function"]["name"],
        "grid": ncu["launches"]["baseline"]["grid"],
        "block": ncu["launches"]["baseline"]["block"],
        "baseline_cubin_sha256": static["inputs"]["baseline"]["cubin_sha256"],
        "baseline_sass_sha256": input_files["baseline_sass"]["sha256"],
        "baseline_ncu_sha256": ncu["inputs"]["baseline"]["trace_rep_sha256"],
        "up_first_cubin_sha256": static["inputs"]["up_first_attribution"][
            "cubin_sha256"
        ],
        "up_first_sass_sha256": input_files["up_first_sass"]["sha256"],
        "up_first_ncu_sha256": ncu["inputs"]["up_first_attribution"][
            "trace_rep_sha256"
        ],
        "identity_gates_pass": True,
    }


def exact_integer(value: Any, label: str) -> int:
    numeric = float(value)
    if not numeric.is_integer():
        raise AssertionError(f"{label} must be an integer count, got {value}")
    return int(numeric)


def assert_declared_delta(declared: dict[str, Any], key: str, calculated: int) -> None:
    if exact_integer(declared[key], key) != calculated:
        raise AssertionError(
            f"NCU declared/calculated delta drift for {key}: "
            f"{declared[key]} != {calculated}"
        )


def verify_stack_reduction(
    static: dict[str, Any], tail_word_count: int
) -> tuple[int, int, int]:
    baseline_stack = exact_integer(
        static["arms"]["baseline"]["resource"]["stack_bytes_per_thread"],
        "baseline stack bytes/thread",
    )
    up_first_stack = exact_integer(
        static["arms"]["up_first_attribution"]["resource"]["stack_bytes_per_thread"],
        "up-first stack bytes/thread",
    )
    stack_reduction = baseline_stack - up_first_stack
    if stack_reduction != tail_word_count * 4:
        raise AssertionError(
            f"stack reduction does not close the tail bundle: {stack_reduction}"
        )
    if (
        exact_integer(
            static["deltas"]["up_first_attribution"]["stack_bytes_delta"],
            "static stack delta",
        )
        != -stack_reduction
    ):
        raise AssertionError("static declared/calculated stack delta drift")
    return baseline_stack, up_first_stack, stack_reduction


def verify_dynamic_reduction(ncu: dict[str, Any]) -> dict[str, int | bool]:
    dynamic_delta = ncu["deltas"]["up_first_attribution"]
    if not dynamic_delta["dynamic_14_word_closure_pass"]:
        raise AssertionError("NCU dynamic 14-word closure failed")
    baseline_ncu = ncu["arms"]["baseline"]
    up_first_ncu = ncu["arms"]["up_first_attribution"]
    load_sector_reduction = exact_integer(
        baseline_ncu["local_load_sectors"], "baseline local-load sectors"
    ) - exact_integer(up_first_ncu["local_load_sectors"], "up-first local-load sectors")
    store_sector_reduction = exact_integer(
        baseline_ncu["local_store_sectors"], "baseline local-store sectors"
    ) - exact_integer(
        up_first_ncu["local_store_sectors"], "up-first local-store sectors"
    )
    load_instruction_reduction = exact_integer(
        baseline_ncu["executed_local_load_instructions"],
        "baseline executed local-load instructions",
    ) - exact_integer(
        up_first_ncu["executed_local_load_instructions"],
        "up-first executed local-load instructions",
    )
    store_instruction_reduction = exact_integer(
        baseline_ncu["executed_local_store_instructions"],
        "baseline executed local-store instructions",
    ) - exact_integer(
        up_first_ncu["executed_local_store_instructions"],
        "up-first executed local-store instructions",
    )
    assert_declared_delta(
        dynamic_delta, "local_load_sector_reduction", load_sector_reduction
    )
    assert_declared_delta(
        dynamic_delta, "local_store_sector_reduction", store_sector_reduction
    )
    assert_declared_delta(
        dynamic_delta,
        "executed_local_load_instruction_reduction",
        load_instruction_reduction,
    )
    assert_declared_delta(
        dynamic_delta,
        "executed_local_store_instruction_reduction",
        store_instruction_reduction,
    )
    if load_sector_reduction != store_sector_reduction:
        raise AssertionError("local-load/store sector reductions differ")
    if load_instruction_reduction != store_instruction_reduction:
        raise AssertionError("executed local-load/store instruction reductions differ")
    if (
        exact_integer(
            dynamic_delta["expected_14_word_sector_reduction"],
            "expected tail-sector reduction",
        )
        != load_sector_reduction
    ):
        raise AssertionError("NCU expected/calculated tail-sector reduction drift")
    return {
        "local_sector_reduction_per_direction": load_sector_reduction,
        "executed_local_instruction_reduction_per_direction": (
            load_instruction_reduction
        ),
        "tensor_work_identity_pass": bool(dynamic_delta["work_identity_pass"]),
    }


def build_evidence(args: argparse.Namespace) -> dict[str, Any]:
    static = load_json(args.static_evidence)
    ncu = load_json(args.ncu_evidence)
    path_arguments = {
        "baseline_sass": args.baseline_sass,
        "up_first_sass": args.up_first_sass,
        "baseline_mlir": args.baseline_mlir,
        "baseline_ptx": args.baseline_ptx,
        "up_first_ptx": args.up_first_ptx,
        "source": args.source,
        "static_evidence": args.static_evidence,
        "ncu_evidence": args.ncu_evidence,
    }
    input_files = {name: file_identity(path) for name, path in path_arguments.items()}
    identity = verify_evidence_identity(static, ncu, input_files)

    baseline = parse_sass(args.baseline_sass)
    up_first = parse_sass(args.up_first_sass)
    tail = verify_tail_chains(baseline)
    main = verify_main_bundle(baseline)
    up_first_main = verify_main_bundle(up_first, verify_representative=False)
    tail_slots = {row["stack_slot"] for row in tail}
    up_first_tail_stores = [
        item
        for item in up_first
        if item.opcode == "STL" and stack_offset(item) in tail_slots
    ]
    up_first_tail_reloads = [
        item
        for item in up_first
        if item.opcode.startswith("LDL") and stack_offset(item) in tail_slots
    ]
    if up_first_tail_stores or up_first_tail_reloads:
        raise AssertionError(
            "up-first must eliminate every tail STL and corresponding LDL"
        )
    assert_identity_equal(
        "main store slot set across arms",
        main["store_stack_slots"],
        up_first_main["store_stack_slots"],
    )
    assert_identity_equal(
        "main reload slot set across arms",
        main["reload_stack_slots"],
        up_first_main["reload_stack_slots"],
    )

    baseline_stack, up_first_stack, stack_reduction = verify_stack_reduction(
        static, len(tail)
    )
    dynamic_reduction = verify_dynamic_reduction(ncu)

    cross_layer = verify_cross_layer_semantics(
        args.source,
        args.baseline_mlir,
        args.baseline_ptx,
        args.up_first_ptx,
    )
    classes: dict[str, int] = {}
    for row in tail:
        classes[row["value_class"]] = classes.get(row["value_class"], 0) + 1
    return {
        "schema": "exp003.spill-root-cause.root-cause-evidence.v1",
        "question": (
            "why fused-kernel register spill forms, where it occurs, and which "
            "live values create each pressure peak"
        ),
        "non_goals": [
            "prove spill latency impact",
            "report spill speedup",
            "claim that spill causes latency or TC-cadence loss",
            "recommend a production optimization before a controlled follow-up experiment",
        ],
        "identity": identity,
        "provenance": {
            "invocation": {
                "python_executable": str(Path(sys.executable).resolve()),
                "builder": file_identity(Path(__file__)),
                "arguments": {
                    f"--{name.replace('_', '-')}": value["path"]
                    for name, value in input_files.items()
                },
            },
            "input_files": input_files,
            "upstream_capture_inputs": {
                arm: {
                    "static": static["inputs"][arm],
                    "ncu": ncu["inputs"][arm],
                }
                for arm in ("baseline", "up_first_attribution")
            },
        },
        "main_108_word_bundle": main,
        "tail_14_word_bundle": {
            "stack_word_count": len(tail),
            "stack_bytes_per_thread": len(tail) * 4,
            "value_class_counts": classes,
            "homogeneous_accumulator_bundle": False,
            "physical_mechanism": (
                "activation-entry register live-range interference: original live values are "
                "saved, their physical registers are reused by activation temporaries, and "
                "the original values are restored before their later consumers"
            ),
            "source_value_status": "partially_semantic_localized",
            "disposition": "deferred_reprofile_after_main_change",
            "chains": tail,
        },
        "up_first_observation": {
            "main_108_word_bundle_preserved": (
                main["store_stack_slots"] == up_first_main["store_stack_slots"]
                and main["reload_stack_slots"] == up_first_main["reload_stack_slots"]
            ),
            "tail_14_word_bundle_eliminated": not (
                up_first_tail_stores or up_first_tail_reloads
            ),
            "stack_bytes_per_thread": [baseline_stack, up_first_stack],
            "stack_byte_reduction": stack_reduction,
            "local_sector_reduction_per_direction": dynamic_reduction[
                "local_sector_reduction_per_direction"
            ],
            "executed_local_instruction_reduction_per_direction": dynamic_reduction[
                "executed_local_instruction_reduction_per_direction"
            ],
            "tensor_work_identity_pass": dynamic_reduction["tensor_work_identity_pass"],
            "interpretation": (
                "corroborates the mixed activation-entry live-set mechanism; it does not "
                "identify one unique source construct as the cause"
            ),
        },
        "cross_layer_semantics": cross_layer,
        "root_cause": {
            "physical_locations": {
                "main_108_word_bundle": (
                    "during the first FC1 pass tail, 27 completed accumulator vectors are "
                    "progressively spilled after their respective nearest OMMA producers; "
                    "they are reloaded before/during activation after remaining live across "
                    "the complete second FC1 pass"
                ),
                "tail_14_word_bundle": (
                    "activation entry: baseline SASS stores 0xbb40..0xbc30; "
                    "restores 0xf050..0xf3d0 before later consumers"
                ),
            },
            "physical_mechanisms": {
                "main_108_word_bundle": (
                    "each completed first-pass accumulator vector is spilled after its own "
                    "nearest OMMA producer during first-pass tail processing, remains live "
                    "across the complete second FC1 pass, and is reloaded for activation"
                ),
                "tail_14_word_bundle": (
                    "activation temporaries reuse physical registers holding five second-pass "
                    "accumulator register values and nine index/address/control scalars; the allocator saves "
                    "and restores those original values"
                ),
            },
            "formation_causes": {
                "main_108_word_bundle": (
                    "the first-pass FP32 accumulator remains live across the complete "
                    "second FC1 pass until activation. Its live range overlaps the "
                    "second-pass accumulator and other live working state; under the observed "
                    "255-register allocation, completed first-pass vectors are saved to "
                    "local memory and reloaded for activation"
                ),
                "tail_14_word_bundle": (
                    "at activation entry, five live second-pass accumulator register values plus "
                    "nine long-lived "
                    "index/address/control scalars overlap activation temporaries. The "
                    "allocator saves those values, reuses their physical registers for "
                    "activation, and restores them before later consumers"
                ),
            },
            "source_interpretation": {
                "main_108_word_bundle": (
                    "production source/IR program order identifies the first pass as Gate "
                    "and the second pass as Up, so the first-pass bundle is attributed to "
                    "gate_acc; no compiler-certified SSA-to-physical-slot map is available"
                ),
                "tail_14_word_bundle": (
                    "production source/IR program order identifies the second-pass "
                    "accumulator as Up; the nine scalar values do not have unique source SSA identities"
                ),
            },
            "source_attribution_status": (
                "high_confidence_program_order_inference_not_compiler_certified"
            ),
            "registers_per_thread": static["arms"]["baseline"]["resource"][
                "registers_per_thread"
            ],
            "main_bundle_status": "physical_formation_mechanism_closed",
            "tail_physical_status": "physical_formation_mechanism_closed",
            "tail_source_value_status": "partially_semantic_localized",
            "tail_disposition": "deferred_reprofile_after_main_change",
            "overall_status": (
                "formation_mechanism_closed_source_attribution_inferred"
            ),
            "p0_by_project_policy": True,
            "latency_causality_required_for_p0": False,
            "latency_causality_tested": False,
            "production_optimization_recommendation_allowed": False,
            "followup_experiment_allowed": True,
            "followup_scope": "main_108_live_range_mechanism",
        },
        "deferred_residual": [
            "compiler-certified MLIR/PTX virtual value to SASS physical register and stack-slot mapping",
            "unique source SSA/expression for the nine index/address/control scalars",
            "a single source construct that causes the complete mixed 14-word tail",
        ],
        "followup_hypothesis": {
            "claim": "register spill is a primary contributor to reduced TC cadence",
            "status": "unverified",
            "required_counterfactual": (
                "a correctness-equivalent reduced/no-spill arm with tensor work, launch "
                "topology, and task schedule controlled"
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-sass", type=Path, required=True)
    parser.add_argument("--up-first-sass", type=Path, required=True)
    parser.add_argument("--baseline-mlir", type=Path, required=True)
    parser.add_argument("--baseline-ptx", type=Path, required=True)
    parser.add_argument("--up-first-ptx", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--static-evidence",
        type=Path,
        default=DEFAULT_RESULTS / "static_spill_evidence.json",
    )
    parser.add_argument(
        "--ncu-evidence",
        type=Path,
        default=DEFAULT_RESULTS / "ncu" / "spill_evidence.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS / "spill_root_cause_evidence.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    evidence = build_evidence(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
