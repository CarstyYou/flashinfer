"""CPU-only tests for exp_003's static binary gate."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "validate_binary_gate.py"


def load_binary_gate():
    spec = importlib.util.spec_from_file_location(
        "exp003_binary_gate_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = load_binary_gate()


def resource_text(*, stack: int) -> str:
    return f"""
Resource usage:
 Common:
  GLOBAL:0
 Function test_kernel:
  REG:255 STACK:{stack} SHARED:1024 LOCAL:0 CONSTANT[0]:2192
"""


SASS = """
// A comment saying OMMA.SF and LDG.E must not count.
        /*0010*/                   OMMA.SF.16864.F32 R1, R2, R3 ;
        /*0020*/              @!P1 UTMALDG.2D [UR1], [UR2], desc[URZ] ;
        /*0030*/                   LDSM.16.M88.4 R1, [R2] ;
        /*0040*/                   BAR.SYNC.DEFER_BLOCKING 0x0 ;
        /*0050*/                   DEPBAR.LE SB0, 0x0 ;
        /*0060*/                   ATOMG.E.ADD.S32 PT, R1, [R2], R3 ;
        /*0070*/                   REDG.E.ADD.BF16x8 [R2], R3 ;
        /*0080*/                   LDG.E R1, [R2] ;
        /*0090*/                   STG.E [R2], R1 ;
        /*00a0*/                   UTMALDG.2D [UR1], [UR2], desc[URZ] ; // LDG
        /*00b0*/                   STS [R2], R1 ;
        /*00c0*/  .word 0xdeadbeef ; // OMMA
"""


class OpcodeParserTest(unittest.TestCase):
    def test_counts_exact_instruction_opcode_tokens_only(self):
        self.assertEqual(
            gate.semantic_opcode_counts(SASS),
            {
                "OMMA": 1,
                "UTMALDG": 2,
                "LDSM": 1,
                "BAR": 1,
                "ATOMG": 1,
                "REDG": 1,
                "LDG": 1,
                "STG": 1,
            },
        )


class ResourceParserTest(unittest.TestCase):
    def test_parses_required_resource_tuple(self):
        self.assertEqual(
            gate.parse_resource_text(resource_text(stack=488), label="control"),
            {
                "kernel": "test_kernel",
                "REG": 255,
                "STACK": 488,
                "SHARED": 1024,
                "LOCAL": 0,
            },
        )

    def test_rejects_ambiguous_or_incomplete_records(self):
        with self.assertRaises(gate.BinaryGateError):
            gate.parse_resource_text("Function k:\n REG:1 STACK:2\n", label="bad")
        with self.assertRaises(gate.BinaryGateError):
            gate.parse_resource_text(
                resource_text(stack=1) + resource_text(stack=2), label="bad"
            )


class BinaryGateTest(unittest.TestCase):
    def write_fixture(self, root: Path, *, candidate_stack: int = 432):
        paths = {
            "control_resource": root / "control.resource.txt",
            "candidate_resource": root / "candidate.resource.txt",
            "control_sass": root / "control.sass",
            "candidate_sass": root / "candidate.sass",
            "control_cfg": root / "control.cfg.dot",
            "candidate_cfg": root / "candidate.cfg.dot",
            "sha256_manifest": root / "sha256.txt",
        }
        paths["control_resource"].write_text(resource_text(stack=488))
        paths["candidate_resource"].write_text(resource_text(stack=candidate_stack))
        paths["control_sass"].write_text(SASS)
        paths["candidate_sass"].write_text(SASS + "\n// marker-only text\n")
        paths["control_cfg"].write_text("digraph control {}\n")
        paths["candidate_cfg"].write_text("digraph candidate {}\n")

        manifest_rows = []
        for key in (
            "control_resource",
            "candidate_resource",
            "control_sass",
            "candidate_sass",
            "control_cfg",
            "candidate_cfg",
        ):
            path = paths[key]
            manifest_rows.append(f"{gate.sha256_file(path)}  /out/{path.name}")
        manifest_rows.extend(
            [
                f"{'1' * 64}  /control/dump/kernel.cubin",
                f"{'2' * 64}  /candidate/dump/kernel.cubin",
            ]
        )
        paths["sha256_manifest"].write_text("\n".join(manifest_rows) + "\n")
        return paths

    def test_stack_drift_disables_formal_dominance_but_semantics_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory))
            payload = gate.build_binary_gate(**paths)

        self.assertTrue(payload["comparisons"]["semantic_opcode_counts"]["pass"])
        self.assertTrue(payload["binary_semantic_omma_gate"]["pass"])
        self.assertFalse(payload["comparisons"]["resource_identity"]["pass"])
        self.assertEqual(
            payload["comparisons"]["resource_identity"]["fields"]["STACK"],
            {"control": 488, "candidate": 432, "delta": -56, "equal": False},
        )
        self.assertFalse(payload["formal_dominance"]["eligible"])
        self.assertIn(
            "resource identity failed: STACK",
            payload["formal_dominance"]["reasons"],
        )
        self.assertFalse(payload["scope"]["full_sass_identity_assessed"])
        self.assertFalse(payload["scope"]["full_cfg_identity_assessed"])

    def test_complete_equal_fixture_is_eligible_and_json_is_canonical(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory), candidate_stack=488)
            payload = gate.build_binary_gate(**paths)
            first = gate.canonical_json(payload)
            second = gate.canonical_json(payload)

        self.assertTrue(payload["formal_dominance"]["eligible"])
        self.assertEqual(first, second)
        self.assertEqual(json.loads(first), payload)
        self.assertTrue(first.endswith("\n"))

    def test_missing_optional_identity_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory), candidate_stack=488)
            payload = gate.build_binary_gate(
                control_resource=paths["control_resource"],
                candidate_resource=paths["candidate_resource"],
                control_sass=paths["control_sass"],
                candidate_sass=paths["candidate_sass"],
            )

        self.assertTrue(payload["comparisons"]["semantic_opcode_counts"]["pass"])
        self.assertTrue(payload["comparisons"]["resource_identity"]["pass"])
        self.assertFalse(payload["formal_dominance"]["eligible"])
        self.assertIn(
            "control/candidate CFG evidence pair is missing",
            payload["formal_dominance"]["reasons"],
        )
        self.assertIn(
            "control/candidate cubin hash evidence is missing or ambiguous",
            payload["formal_dominance"]["reasons"],
        )

    def test_manifest_digest_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory), candidate_stack=488)
            paths["candidate_sass"].write_text(SASS + "\n// changed after hashing\n")
            payload = gate.build_binary_gate(**paths)

        self.assertFalse(payload["comparisons"]["sha256_manifest_validation"]["pass"])
        self.assertFalse(payload["formal_dominance"]["eligible"])


if __name__ == "__main__":
    unittest.main()
