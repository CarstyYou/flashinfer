"""CPU-only tests for exp_003's tracker-cubin PC/SASS gate."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "validate_pc_sass.py"


def load_gate():
    spec = importlib.util.spec_from_file_location("exp003_pc_sass_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = load_gate()


def instrumentation(name: str, offset: int, *, pos: str = "rangeStart"):
    return {
        "name": name,
        "rangePos": pos,
        "offset": offset,
        "offsetHex": f"0x{offset:05x}",
    }


def config_document(*, break_pair_at: str | None = None):
    records = []
    starts = (
        ("wait", 0x10),
        ("wait", 0x40),
        ("fc2_pre_scatter_barrier", 0x70),
        ("fc2_post_scatter_barrier", 0xA0),
        ("gate_pass_wait", 0xD0),
        ("final_pass_wait", 0x100),
    )
    for name, offset in starts:
        records.append(instrumentation(name, offset))
        if name == break_pair_at:
            records.append(instrumentation("unrelated", offset + 0x20))
        else:
            records.append(instrumentation("pop_range", offset + 0x20, pos="rangeEnd"))
    return {
        "meta": {"version": 0},
        "configs": [
            {
                "kernel": "test_kernel",
                "maxTsCntPerWarp": 1024,
                "instrumentations": records,
            }
        ],
    }


def sass_text(*, second_wait: str | None = None, first_barrier: str | None = None):
    second_wait = second_wait or "SYNCS.PHASECHK.TRANS64.TRYWAIT"
    first_barrier = first_barrier or "BAR.SYNC.DEFER_BLOCKING"
    opcodes = {
        0x10: "CS2R.32",
        0x20: "SYNCS.PHASECHK.TRANS64.TRYWAIT",
        0x30: "CS2R.32",
        0x40: "PMTRIG",
        0x50: second_wait,
        0x60: "CS2R.32",
        0x70: "CS2R.32",
        0x80: first_barrier,
        0x90: "CS2R.32",
        0xA0: "PMTRIG",
        0xB0: "BAR.SYNC",
        0xC0: "CS2R.32",
        0xD0: "CS2R.32",
        0xE0: "BAR.SYNC.DEFER_BLOCKING",
        0xF0: "CS2R.32",
        0x100: "PMTRIG",
        0x110: "BAR.SYNC.DEFER_BLOCKING",
        0x120: "CS2R.32",
    }
    rows = [
        "//--------------------- .text.test_kernel --------------------------",
        '\t.section\t.text.test_kernel,"ax",@progbits',
        ".text.test_kernel:",
    ]
    for pc, opcode in opcodes.items():
        operand = " P0, [UR1], RZ" if opcode.startswith("SYNCS") else " R0"
        rows.append(f"        /*{pc:04x}*/                   {opcode}{operand} ;")
    rows.extend(
        [
            "//--------------------- .nv.shared.test_kernel -------------------",
            '\t.section\t.nv.shared.test_kernel,"aw",@nobits',
        ]
    )
    return "\n".join(rows) + "\n"


class Fixture:
    def __init__(
        self,
        root: Path,
        *,
        document=None,
        sass: str | None = None,
    ):
        self.instrument_config = root / "instrument.config.json"
        self.sass = root / "tracker.sass"
        self.cubin = root / "tracker.cubin"
        self.instrument_config.write_text(
            json.dumps(document if document is not None else config_document())
        )
        self.sass.write_text(sass if sass is not None else sass_text())
        self.cubin.write_bytes(b"synthetic tracker cubin")
        self.expected_sass_sha256 = gate.sha256_file(self.sass)
        self.expected_cubin_sha256 = gate.sha256_file(self.cubin)

    def kwargs(self):
        return {
            "instrument_config": self.instrument_config,
            "sass": self.sass,
            "cubin": self.cubin,
            "expected_sass_sha256": self.expected_sass_sha256,
            "expected_cubin_sha256": self.expected_cubin_sha256,
        }


class PcSassGateTest(unittest.TestCase):
    def test_all_sites_prove_all_analyzer_wait_leaves(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            payload = gate.build_pc_sass_gate(**fixture.kwargs())

        self.assertTrue(payload["overall_pass"])
        self.assertEqual(payload["status"], "pass")
        self.assertEqual(len(payload["sites"]), 6)
        self.assertEqual(
            payload["verified_range_names"],
            sorted(
                [
                    "fc1_gate_wait",
                    "fc1_up_wait",
                    "fc2_wait",
                    "fc2_pre_scatter_barrier",
                    "fc2_post_scatter_barrier",
                    "gate_pass_wait",
                    "final_pass_wait",
                ]
            ),
        )
        first = payload["sites"][0]
        self.assertEqual(first["start_offset_hex"], "0x00010")
        self.assertEqual(first["end_offset_hex"], "0x00030")
        self.assertEqual(
            [row["opcode"] for row in first["opcodes"]],
            ["CS2R.32", "SYNCS.PHASECHK.TRANS64.TRYWAIT"],
        )
        self.assertFalse(first["proof"]["marker_only_instructions_qualify"])
        self.assertEqual(
            [row["opcode"] for row in first["proof"]["marker_only_instructions"]],
            ["CS2R.32"],
        )

    def test_one_bad_generic_wait_site_invalidates_all_generic_wait_names(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), sass=sass_text(second_wait="PMTRIG"))
            payload = gate.build_pc_sass_gate(**fixture.kwargs())

        self.assertFalse(payload["overall_pass"])
        self.assertFalse(
            payload["static_range_status"]["wait"]["all_sites_semantic_pass"]
        )
        self.assertNotIn("wait", payload["verified_static_range_names"])
        for name in gate.GENERIC_WAIT_LOGICAL_NAMES:
            self.assertNotIn(name, payload["verified_range_names"])
        bad_site = [
            site
            for site in payload["sites"]
            if site["static_range_name"] == "wait" and site["start_offset"] == 0x40
        ][0]
        self.assertFalse(bad_site["proof"]["pass"])
        self.assertEqual(
            [row["opcode"] for row in bad_site["proof"]["marker_only_instructions"]],
            ["PMTRIG", "PMTRIG"],
        )

    def test_non_whitelisted_barrier_invalidates_only_that_static_name(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(
                Path(directory), sass=sass_text(first_barrier="DEPBAR.LE")
            )
            payload = gate.build_pc_sass_gate(**fixture.kwargs())

        self.assertFalse(payload["overall_pass"])
        self.assertNotIn("fc2_pre_scatter_barrier", payload["verified_range_names"])
        self.assertIn("fc2_post_scatter_barrier", payload["verified_range_names"])
        self.assertIn("fc1_gate_wait", payload["verified_range_names"])

    def test_hash_mismatch_fails_identity_and_verifies_no_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            kwargs = fixture.kwargs()
            kwargs["expected_sass_sha256"] = "0" * 64
            payload = gate.build_pc_sass_gate(**kwargs)

        self.assertFalse(payload["artifact_identity"]["pass"])
        self.assertEqual(payload["verified_range_names"], [])
        self.assertFalse(payload["overall_pass"])
        self.assertTrue(all(site["proof"]["pass"] for site in payload["sites"]))

    def test_pairing_must_be_immediate_pop_range_end(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(
                Path(directory), document=config_document(break_pair_at="wait")
            )
            with self.assertRaisesRegex(
                gate.PcSassGateError, "immediately followed by pop_range/rangeEnd"
            ):
                gate.build_pc_sass_gate(**fixture.kwargs())

    def test_missing_required_static_name_fails_closed(self):
        document = config_document()
        document["configs"][0]["instrumentations"] = [
            row
            for row in document["configs"][0]["instrumentations"]
            if row["name"] != "final_pass_wait"
        ]
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), document=document)
            with self.assertRaisesRegex(
                gate.PcSassGateError, "lacks required static range"
            ):
                gate.build_pc_sass_gate(**fixture.kwargs())

    def test_missing_instruction_inside_range_fails_closed(self):
        broken = sass_text().replace(
            "        /*0050*/                   SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR1], RZ ;\n",
            "",
        )
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), sass=broken)
            with self.assertRaisesRegex(gate.PcSassGateError, "not contiguous"):
                gate.build_pc_sass_gate(**fixture.kwargs())

    def test_canonical_json_and_main_failure_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = Fixture(root, sass=sass_text(first_barrier="DEPBAR.LE"))
            payload = gate.build_pc_sass_gate(**fixture.kwargs())
            rendered = gate.canonical_json(payload)
            output = root / "gate.json"
            return_code = gate.main(
                [
                    "--instrument-config",
                    str(fixture.instrument_config),
                    "--sass",
                    str(fixture.sass),
                    "--cubin",
                    str(fixture.cubin),
                    "--expected-sass-sha256",
                    fixture.expected_sass_sha256,
                    "--expected-cubin-sha256",
                    fixture.expected_cubin_sha256,
                    "--output",
                    str(output),
                ]
            )
            output_payload = json.loads(output.read_text())

        self.assertEqual(return_code, 1)
        self.assertTrue(rendered.endswith("\n"))
        self.assertEqual(json.loads(rendered), payload)
        self.assertEqual(output_payload, payload)


if __name__ == "__main__":
    unittest.main()
