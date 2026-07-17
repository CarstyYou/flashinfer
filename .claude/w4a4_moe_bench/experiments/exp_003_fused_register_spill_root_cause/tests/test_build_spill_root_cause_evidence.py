from __future__ import annotations

import pytest

from build_spill_root_cause_evidence import (
    Instruction,
    TAIL_CHAINS,
    verify_dynamic_reduction,
    verify_evidence_identity,
    verify_main_bundle,
    verify_stack_reduction,
    verify_tail_chain,
)


def tail_instructions() -> list[Instruction]:
    chain = TAIL_CHAINS[0]
    return [
        Instruction(
            chain.producer_pc,
            "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X",
            "R148, R68, R162, R148, R10, R23, URZ",
        ),
        Instruction(chain.store_pc, "STL", "[R1+0x1e4], R151"),
        Instruction(chain.temporary_reuse_pcs[0], "FMUL", "R151, R9, R11"),
        Instruction(chain.reload_pc, "LDL.LU", "R151, [R1+0x1e4]"),
        Instruction(
            chain.first_original_consumer_pc,
            "FMUL",
            "R151, R151, UR12",
        ),
    ]


def mutate_instruction(
    instructions: list[Instruction], pc: int, *, opcode: str, operands: str
) -> list[Instruction]:
    return [
        Instruction(item.pc, opcode, operands) if item.pc == pc else item
        for item in instructions
    ]


def test_tail_chain_closes_definition_reuse_reload_and_source_consumer() -> None:
    row = verify_tail_chain(tail_instructions(), TAIL_CHAINS[0])
    assert row["producer_is_last_definition_before_store"] is True
    assert row["reuse_pcs_overwrite_physical_register"] is True
    assert row["consumer_is_first_post_reload_reference"] is True
    assert row["consumer_reads_reloaded_register"] is True


@pytest.mark.parametrize(
    ("pc", "opcode", "operands", "message"),
    [
        (
            TAIL_CHAINS[0].producer_pc,
            "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X",
            "R147, R68, R162, R147, R10, R23, URZ",
            "does not define",
        ),
        (
            TAIL_CHAINS[0].producer_pc,
            "MOV",
            "R151, R9",
            "producer opcode drift",
        ),
        (
            TAIL_CHAINS[0].temporary_reuse_pcs[0],
            "FMUL",
            "R99, R9, R11",
            "does not overwrite",
        ),
        (
            TAIL_CHAINS[0].first_original_consumer_pc,
            "FMUL",
            "R151, R9, UR12",
            "does not read",
        ),
    ],
)
def test_tail_chain_mutations_fail_closed(
    pc: int, opcode: str, operands: str, message: str
) -> None:
    mutated = mutate_instruction(
        tail_instructions(), pc, opcode=opcode, operands=operands
    )
    with pytest.raises(AssertionError, match=message):
        verify_tail_chain(mutated, TAIL_CHAINS[0])


def test_tail_chain_rejects_an_earlier_post_reload_reference() -> None:
    instructions = tail_instructions()
    instructions.insert(-1, Instruction(0xF060, "FMUL", "R9, R9, R151"))
    with pytest.raises(AssertionError, match="first post-reload reference"):
        verify_tail_chain(instructions, TAIL_CHAINS[0])


def synthetic_main_bundle() -> list[Instruction]:
    instructions: list[Instruction] = []
    pc = 0x100
    for vector in range(27):
        register = vector * 4
        offset = vector * 16
        instructions.extend(
            [
                Instruction(
                    pc,
                    "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X",
                    f"R{register}, R200, R201, R{register}, R202, R203, URZ",
                ),
                Instruction(
                    pc + 0x10,
                    "STL.64",
                    f"[R1+0x{offset:x}], R{register}",
                ),
                Instruction(
                    pc + 0x20,
                    "STL.64",
                    f"[R1+0x{offset + 8:x}], R{register + 2}",
                ),
            ]
        )
        pc += 0x30
    for slot_index, offset in enumerate(range(0, 0x1B0, 4)):
        register = 128 + slot_index
        instructions.extend(
            [
                Instruction(pc, "LDL.LU", f"R{register}, [R1+0x{offset:x}]"),
                Instruction(pc + 0x10, "FMUL", f"R250, R{register}, UR12"),
            ]
        )
        pc += 0x20
    return instructions


def test_main_bundle_derives_27_nearest_omma_producer_store_chains() -> None:
    result = verify_main_bundle(synthetic_main_bundle(), verify_representative=False)
    assert result["omma_output_vector_count"] == 27
    assert result["producer_store_chain_count"] == 27
    assert result["stl64_instruction_count"] == 54
    assert result["stack_word_count"] == 108
    assert result["reload_first_use_is_scale_fmul_count"] == 108
    assert "first FC1 pass tail" in result["mechanism"]
    assert "after first FC1 pass" not in result["mechanism"]


def test_main_bundle_rejects_non_omma_nearest_source_definition() -> None:
    instructions = synthetic_main_bundle()
    instructions.insert(1, Instruction(0x108, "MOV", "R1, R9"))
    with pytest.raises(AssertionError, match="share one nearest OMMA producer"):
        verify_main_bundle(instructions, verify_representative=False)


def identity_payloads():
    static = {
        "schema": "exp004.static-spill-evidence.v1",
        "baseline_reproduction_pass": True,
        "arms": {
            arm: {
                "function": {"name": "kernel", "cubin_sha256": cubin},
            }
            for arm, cubin in (
                ("baseline", "cubin-b"),
                ("up_first_attribution", "cubin-u"),
            )
        },
        "inputs": {
            arm: {
                "cubin_sha256": cubin,
                "disassembly_sha256": sass,
                "fresh_jit_artifact_identity_gate": True,
            }
            for arm, cubin, sass in (
                ("baseline", "cubin-b", "sass-b"),
                ("up_first_attribution", "cubin-u", "sass-u"),
            )
        },
        "deltas": {
            "up_first_attribution": {"complete_tail_removal_no_replacement": True}
        },
    }
    ncu = {
        "schema": "exp004.ncu-spill-evidence.v1",
        "baseline_reproduction_pass": True,
        "inputs": {
            arm: {
                "cubin_sha256": cubin,
                "trace_rep_sha256": trace,
                "profile_jit_artifact_identity_gate": True,
                "benchmark_cubin_identity_gate": True,
            }
            for arm, cubin, trace in (
                ("baseline", "cubin-b", "ncu-b"),
                ("up_first_attribution", "cubin-u", "ncu-u"),
            )
        },
        "launches": {
            arm: {
                "kernel": "kernel",
                "grid": [1, 1, 7],
                "block": [160, 1, 1],
                "launch_count": 1,
            }
            for arm in ("baseline", "up_first_attribution")
        },
        "deltas": {"up_first_attribution": {"work_identity_pass": True}},
    }
    input_files = {
        "baseline_sass": {"sha256": "sass-b"},
        "up_first_sass": {"sha256": "sass-u"},
        "baseline_ptx": {"sha256": "ptx-b"},
        "up_first_ptx": {"sha256": "ptx-u"},
    }
    return static, ncu, input_files


def test_identity_gate_records_evidence_derived_launch() -> None:
    static, ncu, inputs = identity_payloads()
    result = verify_evidence_identity(static, ncu, inputs)
    assert result["grid"] == [1, 1, 7]
    assert result["block"] == [160, 1, 1]
    assert result["identity_gates_pass"] is True


def test_hash_mutation_fails_closed() -> None:
    static, ncu, inputs = identity_payloads()
    inputs["baseline_sass"]["sha256"] = "mutated"
    with pytest.raises(AssertionError, match="identity drift for baseline SASS"):
        verify_evidence_identity(static, ncu, inputs)


@pytest.mark.parametrize("drift", ["kernel", "grid", "cubin", "ptx"])
def test_cross_artifact_identity_mutations_fail_closed(drift: str) -> None:
    static, ncu, inputs = identity_payloads()
    if drift == "kernel":
        ncu["launches"]["up_first_attribution"]["kernel"] = "other_kernel"
    elif drift == "grid":
        ncu["launches"]["up_first_attribution"]["grid"] = [2, 1, 7]
    elif drift == "cubin":
        ncu["inputs"]["up_first_attribution"]["cubin_sha256"] = "other-cubin"
    else:
        inputs["up_first_ptx"]["sha256"] = inputs["baseline_ptx"]["sha256"]
    with pytest.raises(AssertionError, match="identity|PTX"):
        verify_evidence_identity(static, ncu, inputs)


def test_stack_and_ncu_deltas_are_read_from_evidence() -> None:
    static = {
        "arms": {
            "baseline": {"resource": {"stack_bytes_per_thread": 100}},
            "up_first_attribution": {"resource": {"stack_bytes_per_thread": 88}},
        },
        "deltas": {"up_first_attribution": {"stack_bytes_delta": -12}},
    }
    assert verify_stack_reduction(static, 3) == (100, 88, 12)

    ncu = {
        "arms": {
            "baseline": {
                "local_load_sectors": 1000,
                "local_store_sectors": 1000,
                "executed_local_load_instructions": 100,
                "executed_local_store_instructions": 90,
            },
            "up_first_attribution": {
                "local_load_sectors": 930,
                "local_store_sectors": 930,
                "executed_local_load_instructions": 83,
                "executed_local_store_instructions": 73,
            },
        },
        "deltas": {
            "up_first_attribution": {
                "dynamic_14_word_closure_pass": True,
                "local_load_sector_reduction": 70,
                "local_store_sector_reduction": 70,
                "executed_local_load_instruction_reduction": 17,
                "executed_local_store_instruction_reduction": 17,
                "expected_14_word_sector_reduction": 70,
                "work_identity_pass": True,
            }
        },
    }
    assert verify_dynamic_reduction(ncu) == {
        "local_sector_reduction_per_direction": 70,
        "executed_local_instruction_reduction_per_direction": 17,
        "tensor_work_identity_pass": True,
    }


def test_declared_dynamic_delta_mutation_fails_closed() -> None:
    _, ncu, _ = identity_payloads()
    ncu["arms"] = {
        "baseline": {
            "local_load_sectors": 10,
            "local_store_sectors": 10,
            "executed_local_load_instructions": 10,
            "executed_local_store_instructions": 10,
        },
        "up_first_attribution": {
            "local_load_sectors": 8,
            "local_store_sectors": 8,
            "executed_local_load_instructions": 8,
            "executed_local_store_instructions": 8,
        },
    }
    ncu["deltas"]["up_first_attribution"].update(
        {
            "dynamic_14_word_closure_pass": True,
            "local_load_sector_reduction": 3,
            "local_store_sector_reduction": 2,
            "executed_local_load_instruction_reduction": 2,
            "executed_local_store_instruction_reduction": 2,
            "expected_14_word_sector_reduction": 2,
        }
    )
    with pytest.raises(AssertionError, match="declared/calculated delta drift"):
        verify_dynamic_reduction(ncu)
