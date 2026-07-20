#!/usr/bin/env python3
"""CPU-only contract tests for exp_016 dynamic-spill capture/evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import build_dynamic_spill_evidence as builder
import capture_dynamic_spill_ncu as capture


class DynamicSpillNcuTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_native_csv(
        self,
        path: Path,
        *,
        missing: str | None = None,
        na: str | None = None,
        spill: int = 0,
    ) -> None:
        identity = (
            "ID",
            "Kernel Name",
            "Context",
            "Stream",
            "Block Size",
            "Grid Size",
            "Device",
        )
        names = [name for name in builder.METRIC_IDS if name != missing]
        header = [*identity, *(builder.METRIC_IDS[name] for name in names)]
        units = ["" for _ in identity] + [
            builder.EXPECTED_UNITS[name] for name in names
        ]
        values = [
            "0",
            "generated_MoEDynamicKernel_symbol",
            "1",
            "7",
            "(288, 1, 1)",
            "(1, 1, 110)",
            "0",
            *("n/a" if name == na else str(spill) for name in names),
        ]
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["==PROF== Connected"])
            writer.writerow(header)
            writer.writerow(units)
            writer.writerow(values)

    def write_validation(
        self,
        path: Path,
        *,
        cubin_sha256: str,
        artifact_set_sha256: str,
        jit_root: Path,
    ) -> None:
        payload = {
            "schema": "exp016.validation-case.v1",
            "status": "complete",
            "gate_pass": True,
            "arm": builder.CANDIDATE,
            "m": builder.M,
            "fixture": builder.FIXTURE,
            "scale_kind": builder.SCALE_KIND,
            "specialization": {"gate_pass": True},
            "replays": [{"gate_pass": True}, {"gate_pass": True}],
            "jit_artifact_set_sha256": artifact_set_sha256,
            "cubin_sha256": [cubin_sha256],
            "runtime": {
                "jit_root": str(jit_root.resolve()),
                "nvcc": "Cuda compilation tools, release 13.2",
                "ptxas": "Cuda compilation tools, release 13.2",
                "source": {"overlay_sha256": builder.EXPECTED_SOURCE_SHA256},
                "gpu": {
                    "uuid": "GPU-test-exp016",
                    "applications_graphics_clock_mhz": "2377",
                },
            },
        }
        path.write_text(json.dumps(payload))

    def test_parser_requires_real_dynamic_metrics(self) -> None:
        valid = self.root / "valid.csv"
        self.write_native_csv(valid)
        metrics, observed = builder.parse_native_raw(valid)
        self.assertEqual(metrics["spill_local_load_bytes"], 0)
        self.assertEqual(observed["grid"], builder.EXPECTED_GRID)
        self.assertEqual(observed["block"], builder.EXPECTED_BLOCK)

        missing = self.root / "missing.csv"
        self.write_native_csv(missing, missing="spill_local_store_bytes")
        with self.assertRaisesRegex(builder.EvidenceError, "metric absent"):
            builder.parse_native_raw(missing)

        na = self.root / "na.csv"
        self.write_native_csv(na, na="spill_register_read_instructions")
        with self.assertRaisesRegex(builder.EvidenceError, "missing/N-A"):
            builder.parse_native_raw(na)

    def test_validation_and_jit_are_exactly_reused(self) -> None:
        results = builder.DEFAULT_RESULTS
        jit = self.root / "jit"
        jit.mkdir()
        cubin = jit / "candidate.cubin"
        cubin.write_bytes(b"exp016-candidate-cubin")
        cubin_sha = capture.sha256_file(cubin)
        artifact_set = capture.common.canonical_sha256(
            capture.common.artifact_manifest(jit)
        )
        validation = self.root / "case.json"
        self.write_validation(
            validation,
            cubin_sha256=cubin_sha,
            artifact_set_sha256=artifact_set,
            jit_root=jit,
        )
        identity = capture.validate_prerequisite(
            results=results, validation=validation, jit_root=jit
        )
        self.assertEqual(identity["cubin_sha256"], cubin_sha)
        self.assertEqual(identity["jit_artifact_set_sha256"], artifact_set)

        wrong = self.root / "wrong-jit"
        wrong.mkdir()
        with self.assertRaisesRegex(RuntimeError, "JIT root drift"):
            capture.validate_prerequisite(
                results=results, validation=validation, jit_root=wrong
            )

    def test_command_profiles_one_demangled_graph_node(self) -> None:
        args = argparse.Namespace(
            flashinfer_root=Path("/workspace/flashinfer"),
            results=Path("/workspace/results"),
            validation=Path("/workspace/case.json"),
            jit_root=Path("/workspace/jit"),
            ncu="ncu",
        )
        prerequisite = {
            "overlay": Path("/workspace/candidate/moe_dynamic_kernel.py"),
            "source_sha256": builder.EXPECTED_SOURCE_SHA256,
            "cubin_sha256": "c" * 64,
            "jit_artifact_set_sha256": "d" * 64,
            "gpu_uuid": "GPU-test-exp016",
            "application_graphics_clock_mhz": 2377,
            "sha256": "e" * 64,
        }
        command = capture.build_ncu_command(
            args,
            prerequisite,
            report_base=Path("/tmp/trace"),
            target_output=Path("/tmp/profile_target.json"),
        )
        self.assertEqual(command[command.index("--launch-count") + 1], "1")
        self.assertEqual(command[command.index("--graph-profiling") + 1], "node")
        self.assertEqual(command[command.index("--profile-from-start") + 1], "off")
        self.assertEqual(command[command.index("--kernel-name-base") + 1], "demangled")
        self.assertEqual(
            tuple(command[command.index("--metrics") + 1].split(",")),
            capture.EXPLICIT_METRIC_IDS,
        )

    def test_gate_rejects_any_executed_spill(self) -> None:
        base_record = {"metrics": {name: 0 for name in builder.DYNAMIC_SPILL_METRICS}}
        with mock.patch.object(builder, "validate_capture", return_value=base_record):
            passed = builder.build_evidence(self.root, self.root / "case.json")
        self.assertTrue(passed["gate_pass"])
        self.assertEqual(passed["checks"]["dynamic_local_load_store_bytes"], 0)

        spilled_record = {
            "metrics": {
                **base_record["metrics"],
                "spill_local_store_bytes": 128,
            }
        }
        with mock.patch.object(
            builder, "validate_capture", return_value=spilled_record
        ):
            rejected = builder.build_evidence(self.root, self.root / "case.json")
        self.assertFalse(rejected["gate_pass"])
        self.assertEqual(rejected["status"], "reject")
        self.assertEqual(rejected["checks"]["dynamic_local_load_store_bytes"], 128)


if __name__ == "__main__":
    unittest.main()
