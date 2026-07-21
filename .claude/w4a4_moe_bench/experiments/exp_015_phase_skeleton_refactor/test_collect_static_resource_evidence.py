#!/usr/bin/env python3
"""Fixture tests for collect_static_resource_evidence.py (GPU/tool free)."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_static_resource_evidence as collector  # noqa: E402


SYMBOL = (
    "kernel_cutlass_kernel_flashinferfused_moe_"
    "MoEDynamicKernel_full_specialized_symbol_0"
)


def resource_fixture(symbol=SYMBOL, registers=160, stack=0, shared=83968, local=0):
    return """
Fatbin elf code:
================
arch = sm_120a

Resource usage:
 Common:
  GLOBAL:0
 Function {symbol}:
  REG:{registers} STACK:{stack} SHARED:{shared} LOCAL:{local} CONSTANT[0]:2192 TEXTURE:0 SURFACE:0 SAMPLER:0
""".format(
        symbol=symbol,
        registers=registers,
        stack=stack,
        shared=shared,
        local=local,
    )


def sass_fixture(symbol=SYMBOL, omma=3, extra_opcodes=(), include_exit=True):
    lines = [
        ".target sm_120a",
        '.section .text.{},"ax",@progbits'.format(symbol),
        ".global {}".format(symbol),
        ".type {},@function".format(symbol),
        "{}:".format(symbol),
    ]
    pc = 0
    for _unused_index in range(omma):
        lines.append(
            "/*{pc:04x}*/ OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X "
            "R60, R2, R4, R60, R8, R10, URZ ;".format(pc=pc)
        )
        pc += 0x10
    for opcode in extra_opcodes:
        lines.append(
            "/*{pc:04x}*/ @!P1 {opcode} R7, [R1] ;".format(pc=pc, opcode=opcode)
        )
        pc += 0x10
    if include_exit:
        lines.append("/*{pc:04x}*/ EXIT ;".format(pc=pc))
    return "\n".join(lines) + "\n"


class ResourceParserTests(unittest.TestCase):
    def test_complete_resource_record(self):
        parsed = collector.parse_resource_usage(resource_fixture())
        self.assertEqual(parsed["kernel_symbol"], SYMBOL)
        self.assertEqual(parsed["registers_per_thread"], 160)
        self.assertEqual(parsed["stack_bytes_per_thread"], 0)
        self.assertEqual(parsed["shared_bytes_per_cta"], 83968)
        self.assertEqual(parsed["local_bytes_outside_stack"], 0)

    def test_missing_resource_field_fails_closed(self):
        malformed = resource_fixture().replace(" LOCAL:0", "")
        with self.assertRaisesRegex(collector.EvidenceError, "exactly one complete"):
            collector.parse_resource_usage(malformed)

    def test_multiple_resource_records_are_ambiguous(self):
        ambiguous = resource_fixture() + resource_fixture(symbol=SYMBOL + "_other")
        with self.assertRaisesRegex(collector.EvidenceError, "found 2"):
            collector.parse_resource_usage(ambiguous)


class SassParserTests(unittest.TestCase):
    def test_selected_instruction_families(self):
        parsed = collector.parse_sass(
            sass_fixture(
                omma=2,
                extra_opcodes=(
                    "LDL.LU",
                    "STL.64",
                    "CALL.REL.NOINC",
                    "RET.REL.NODEC",
                ),
            )
        )
        counts = parsed["selected_instruction_counts"]
        self.assertEqual(parsed["kernel_symbol"], SYMBOL)
        self.assertEqual(counts["omma"], 2)
        self.assertEqual(counts["ldl"], 1)
        self.assertEqual(counts["stl"], 1)
        self.assertEqual(counts["call"], 1)
        self.assertEqual(counts["ret"], 1)
        self.assertEqual(counts["exit"], 1)

    def test_missing_exit_rejects_truncated_disassembly(self):
        with self.assertRaisesRegex(collector.EvidenceError, "no EXIT"):
            collector.parse_sass(sass_fixture(include_exit=False))

    def test_duplicate_pc_rejects_multiple_code_functions(self):
        text = sass_fixture() + "/*0000*/ NOP ;\n"
        with self.assertRaisesRegex(collector.EvidenceError, "duplicate SASS PCs"):
            collector.parse_sass(text)

    def test_multiple_global_functions_are_ambiguous(self):
        other = SYMBOL + "_other"
        text = sass_fixture() + (".global {0}\n.type {0},@function\n".format(other))
        with self.assertRaisesRegex(collector.EvidenceError, "found 2"):
            collector.parse_sass(text)


class EvidenceGateTests(unittest.TestCase):
    def analysis(self, **resource_overrides):
        return collector.analyze_static_outputs(
            resource_fixture(**resource_overrides), sass_fixture(omma=448)
        )

    def test_symbol_mismatch_fails_closed(self):
        with self.assertRaisesRegex(collector.EvidenceError, "symbol mismatch"):
            collector.analyze_static_outputs(
                resource_fixture(), sass_fixture(symbol=SYMBOL + "_different")
            )

    def test_expected_static_contract_passes(self):
        gates = collector.evaluate_arm_gates(self.analysis())
        self.assertTrue(gates["pass"])
        self.assertEqual(gates["failed"], [])

    def test_each_hard_limit_is_enforced(self):
        cases = (
            ({"registers": 161}, "registers_at_most_160"),
            ({"stack": 8}, "stack_zero"),
            ({"local": 16}, "local_zero"),
        )
        for overrides, failed_gate in cases:
            with self.subTest(failed_gate=failed_gate):
                gates = collector.evaluate_arm_gates(self.analysis(**overrides))
                self.assertFalse(gates["pass"])
                self.assertIn(failed_gate, gates["failed"])

        ldl = collector.analyze_static_outputs(
            resource_fixture(), sass_fixture(omma=448, extra_opcodes=("LDL.LU",))
        )
        stl = collector.analyze_static_outputs(
            resource_fixture(), sass_fixture(omma=448, extra_opcodes=("STL.64",))
        )
        wrong_omma = collector.analyze_static_outputs(
            resource_fixture(), sass_fixture(omma=447)
        )
        self.assertIn("ldl_zero", collector.evaluate_arm_gates(ldl)["failed"])
        self.assertIn("stl_zero", collector.evaluate_arm_gates(stl)["failed"])
        self.assertIn(
            "omma_exactly_448",
            collector.evaluate_arm_gates(wrong_omma)["failed"],
        )

    def test_call_ret_relative_inline_gate(self):
        baseline = self.analysis()
        candidate_ok = self.analysis()
        comparison = collector.evaluate_comparison(baseline, candidate_ok)
        self.assertTrue(comparison["pass"])
        self.assertTrue(comparison["both_arms_call_ret_zero"])

        candidate_call = collector.analyze_static_outputs(
            resource_fixture(),
            sass_fixture(omma=448, extra_opcodes=("CALL.REL.NOINC",)),
        )
        comparison = collector.evaluate_comparison(baseline, candidate_call)
        self.assertFalse(comparison["pass"])
        self.assertIn("candidate_adds_no_call", comparison["failed"])


if __name__ == "__main__":
    unittest.main()
