#!/usr/bin/env python3
"""CPU-only tests for exact-cubin P3 SASS spill auditing."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import audit_p3_sass_spill as spill_audit
from exp016_p3_probe_common import file_sha256, write_json


KERNEL_SYMBOL = "synthetic::MoEDynamicKernel"


def write_capture(root: Path, cubin_sha256: str) -> Path:
    capture = root / "capture/capture.json"
    write_json(
        capture,
        {
            "arm": "baseline_pair_major",
            "mode": "probe",
            "cubin_sha256": [cubin_sha256],
            "static_resource_usage": {
                "records": [
                    {
                        "cubin_sha256": cubin_sha256,
                        "kernel_symbol": KERNEL_SYMBOL,
                    }
                ]
            },
        },
    )
    return capture


def write_fake_cuobjdump(root: Path, *, with_spill: bool) -> Path:
    tool = root / "fake-cuobjdump"
    local_sass = ""
    annotations = ""
    if with_spill:
        local_sass = (
            "        /*0010*/                   LDL.LU R2, [R1] ;\n"
            "        /*0020*/                   STL.64 [R1+0x8], R2 ;\n"
        )
        annotations = (
            "        SpillRefill : Offset : 0x0010\n"
            "        SpillRefill : Offset : 0x0020\n"
        )
    tool.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  --version) printf 'fake cuobjdump 1.0\\n' ;;\n"
        "  --dump-sass)\n"
        f"    printf '%s\\n' 'Function : {KERNEL_SYMBOL}' ;\n"
        "    printf '%s\\n' '        /*0000*/                   MOV R1, R2 ;' ;\n"
        + "".join(
            f"    printf '%s\\n' '{line}' ;\n" for line in local_sass.splitlines()
        )
        + "    ;;\n"
        "  --dump-elf)\n"
        f"    printf '%s\\n' 'Function: {KERNEL_SYMBOL}(0x0)' ;\n"
        + "".join(
            f"    printf '%s\\n' '{line}' ;\n" for line in annotations.splitlines()
        )
        + "    ;;\n"
        "  *) exit 3 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    return tool


class P3SassSpillAuditTest(unittest.TestCase):
    def test_zero_spill_writes_binary_locked_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cubin = root / "jit/dump/kernel.cubin"
            cubin.parent.mkdir(parents=True)
            cubin.write_bytes(b"synthetic cubin")
            capture = write_capture(root, file_sha256(cubin))
            tool = write_fake_cuobjdump(root, with_spill=False)
            evidence = spill_audit.audit(
                capture_path=capture,
                jit_root=root / "jit",
                cuobjdump=str(tool),
            )
            self.assertTrue(evidence["gate_pass"])
            self.assertTrue(evidence["sass_spill_gate_pass"])
            self.assertEqual(evidence["counts"]["spill_refill_annotation_count"], 0)
            self.assertEqual(evidence["counts"]["local_sass_opcode_count"], 0)
            self.assertTrue((capture.parent / spill_audit.RAW_SASS_NAME).is_file())
            self.assertTrue((capture.parent / spill_audit.RAW_ELF_NAME).is_file())

    def test_nonzero_annotation_and_local_sass_fail_spill_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cubin = root / "jit/dump/kernel.cubin"
            cubin.parent.mkdir(parents=True)
            cubin.write_bytes(b"synthetic cubin")
            capture = write_capture(root, file_sha256(cubin))
            tool = write_fake_cuobjdump(root, with_spill=True)
            evidence = spill_audit.audit(
                capture_path=capture,
                jit_root=root / "jit",
                cuobjdump=str(tool),
            )
            self.assertFalse(evidence["gate_pass"])
            self.assertFalse(evidence["sass_spill_gate_pass"])
            self.assertEqual(evidence["counts"]["spill_refill_annotation_count"], 2)
            self.assertEqual(evidence["counts"]["ldl_opcode_count"], 1)
            self.assertEqual(evidence["counts"]["stl_opcode_count"], 1)

    def test_missing_or_duplicate_exact_cubin_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jit = root / "jit"
            jit.mkdir()
            capture = write_capture(root, "0" * 64)
            tool = write_fake_cuobjdump(root, with_spill=False)
            with self.assertRaises(spill_audit.SpillAuditError):
                spill_audit.audit(
                    capture_path=capture,
                    jit_root=jit,
                    cuobjdump=str(tool),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jit = root / "jit"
            first = jit / "a/kernel.cubin"
            second = jit / "b/kernel.cubin"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"same cubin")
            second.write_bytes(b"same cubin")
            capture = write_capture(root, file_sha256(first))
            tool = write_fake_cuobjdump(root, with_spill=False)
            with self.assertRaises(spill_audit.SpillAuditError):
                spill_audit.audit(
                    capture_path=capture,
                    jit_root=jit,
                    cuobjdump=str(tool),
                )


if __name__ == "__main__":
    unittest.main()
