from __future__ import annotations

from build_static_spill_evidence import (
    analyze_disassembly,
    compare_arms,
    disassembly_from_sass,
)


def disassembly(*, include_tail: bool):
    instructions = []
    pc = 0x100
    for index in range(54):
        instructions.append(
            {"address": pc, "opcode": "STL.64", "operands": f"[R1+0x{index * 8:x}],R2"}
        )
        pc += 0x10
    if include_tail:
        for index in range(14):
            instructions.append(
                {
                    "address": pc,
                    "opcode": "STL",
                    "operands": f"[R1+0x{432 + index * 4:x}],R3",
                }
            )
            pc += 0x10
    for offset in range(0, 488 if include_tail else 432, 4):
        instructions.append(
            {"address": pc, "opcode": "LDL.LU", "operands": f"R4,[R1+0x{offset:x}]"}
        )
        pc += 0x10
    instructions.extend(
        [
            {"address": pc, "opcode": "OMMA.16816.F32", "operands": "R0,R1,R2,R3"},
            {"address": pc + 16, "opcode": "LDSM.16.M88.4", "operands": "R0,[R1]"},
        ]
    )
    return {
        "data": {
            "rows": [
                {
                    "function_name": "MoEDynamicKernel<test>",
                    "start": 0,
                    "length": pc + 32,
                    "instructions": instructions,
                }
            ],
            "auxiliary": {"cubin_sha": "abc", "source_lineinfo_present": False},
        }
    }


def test_static_baseline_and_exact_tail_removal() -> None:
    baseline = analyze_disassembly(
        disassembly(include_tail=True),
        arm="baseline",
        resource_text="REG:255 STACK:488 SHARED:1024 LOCAL:0",
    )
    candidate = analyze_disassembly(
        disassembly(include_tail=False),
        arm="activation_in_place_up",
        resource_text="REG:255 STACK:432 SHARED:1024 LOCAL:0",
    )
    result = compare_arms({"baseline": baseline, "activation_in_place_up": candidate})
    assert result["baseline_reproduction_pass"] is True
    assert (
        result["deltas"]["activation_in_place_up"][
            "complete_tail_removal_no_replacement"
        ]
        is True
    )
    assert result["deltas"]["activation_in_place_up"]["stack_bytes_delta"] == -56
    assert (
        baseline["resource"]["authority"]
        == "binary_REG_STACK_SHARED_LOCAL_tuple"
    )


def test_parse_target_nvdisasm_text() -> None:
    value = disassembly_from_sass(
        """
.section .text.MoEDynamicKernel_test
/*7860*/ STL.64 [R1+0x8], R2 ;
/*7870*/ LDL.LU R4, [R1+0x8] ;
"""
    )
    row = value["data"]["rows"][0]
    assert row["instructions"][0]["opcode"] == "STL.64"
    assert row["instructions"][0]["address"] == 0x7860
