from __future__ import annotations

from build_binary_identity import (
    _selected_projection,
    opcode_projection,
    parse_instructions,
)


def test_probe_opcode_projection_accepts_insertions_and_projected_branch():
    baseline = parse_instructions(
        """
        /*0000*/ BRA 0x20;
        /*0010*/ IADD3 R2, R3, R4, R5;
        /*0020*/ EXIT;
        """
    )
    candidate = parse_instructions(
        """
        /*0000*/ BRA 0x30;
        /*0010*/ CS2R R8, SR_CLOCKLO;
        /*0020*/ IADD3 R2, R3, R4, R5;
        /*0030*/ EXIT;
        """
    )
    result = opcode_projection(baseline, candidate)
    assert result["insertion_only_opcode_projection"]
    assert result["branch_target_projection_pass"]
    assert result["inserted_opcode_counts"] == {"CS2R": 1}


def test_semantic_projection_uses_omma_opcode():
    counts = {
        "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X": 7,
        "UTMALDG.3D": 2,
        "LDSM.16.M88.4": 3,
        "BAR.ARV": 1,
        "ATOMG.E.ADD.S32.STRONG.GPU": 4,
        "REDG.E.ADD.BF16": 5,
        "LDG.E": 6,
        "STG.E": 8,
    }
    assert _selected_projection(counts) == {
        "omma": 7,
        "utmaldg": 2,
        "ldsm": 3,
        "bar": 1,
        "atomg": 4,
        "redg": 5,
        "ldg": 6,
        "stg": 8,
    }
