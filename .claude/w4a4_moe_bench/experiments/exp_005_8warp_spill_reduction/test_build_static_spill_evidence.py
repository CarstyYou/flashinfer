from __future__ import annotations

from build_static_spill_evidence import (
    destination_registers,
    first_consumer,
    parse_sass,
    preceding_definition,
    stack_offset,
)


def test_parse_and_trace_physical_spill_roundtrip() -> None:
    instructions = parse_sass(
        """
/*0100*/ OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X R60, R2, R4, R60, R8, R10, URZ ;
/*0110*/ STL.64 [R1+0x8], R60 ;
/*0120*/ MOV.64 R60, RZ ;
/*0130*/ LDL.LU R20, [R1+0x8] ;
/*0140*/ FMUL R30, R20, R4 ;
"""
    )
    assert stack_offset(instructions[1]["operands"]) == 8
    assert destination_registers(instructions[0]) == frozenset({60, 61, 62, 63})
    producer = preceding_definition(instructions, 1, 61)
    assert producer is not None and producer["opcode"].startswith("OMMA")
    consumer = first_consumer(instructions, 3, 20)
    assert consumer is not None and consumer["opcode"] == "FMUL"


def test_predicated_sass_is_parsed() -> None:
    instructions = parse_sass("/*0200*/ @!P1 LDL.LU R7, [R1] ;")
    assert instructions == [{"pc": 0x200, "opcode": "LDL.LU", "operands": "R7, [R1]"}]
