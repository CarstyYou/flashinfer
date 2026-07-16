from __future__ import annotations

import unittest

from build_static_sass_evidence import analyze_disassembly


class StaticSassEvidenceTest(unittest.TestCase):
    def test_counts_current_binary_facts_without_dynamic_claim(self) -> None:
        disassembly = {
            "data": {
                "rows": [
                    {
                        "function_name": "MoEDynamicKernel",
                        "start": 0,
                        "length": 64,
                        "instructions": [
                            {"address": 0, "opcode": "OMMA.X", "operands": "R0"},
                            {
                                "address": 16,
                                "opcode": "STL.64",
                                "operands": "[R1+0x8],R2",
                            },
                            {
                                "address": 32,
                                "opcode": "LDL.LU",
                                "operands": "R4[R1+0x8]",
                            },
                            {
                                "address": 48,
                                "opcode": "LDL.LU",
                                "operands": "R5[R1+0xc]",
                            },
                        ],
                    }
                ],
                "auxiliary": {"source_lineinfo_present": False},
            }
        }
        target_manifest = {
            "comparison_group_id": "group",
            "rerun_id": "rerun",
            "environment_lock_digest": "env",
            "protocol_lock_digest": "protocol",
            "artifact_fingerprint_sha256": "artifact",
            "report_sha256": "report",
        }
        evidence = analyze_disassembly(
            disassembly,
            disassembly_sha256="disasm",
            target_manifest=target_manifest,
        )
        facts = evidence["static_instruction_facts"]
        self.assertEqual(facts["tensor_opcodes"], {"OMMA.X": 1})
        self.assertEqual(facts["local_store_opcode_counts"], {"STL.64": 1})
        self.assertEqual(facts["stl64"]["covered_32bit_stack_slots"], 2)
        self.assertTrue(facts["stl64"]["all_slots_have_a_later_static_ldl"])
        self.assertEqual(
            facts["stack_roundtrip_model"]["total_stored_32bit_words_per_lane"],
            2,
        )
        self.assertTrue(
            facts["stack_roundtrip_model"]["all_stored_slots_have_a_later_static_ldl"]
        )
        self.assertEqual(
            evidence["classification"]["dynamic_frequency_or_latency"],
            "not_measured_by_static_sass",
        )


if __name__ == "__main__":
    unittest.main()
