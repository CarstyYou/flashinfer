#!/usr/bin/env python3
"""CPU-only contract tests for the exp_014 dynamic-spill capture path."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import build_dynamic_spill_evidence as builder
import capture_dynamic_spill_ncu as capture


def synthetic_record(arm: str, *, spill: int = 0) -> dict[str, object]:
    return {
        "arm": arm,
        "m": builder.M,
        "fixture": builder.FIXTURE,
        "source_sha256": builder.EXPECTED_SOURCE_SHA256[arm],
        "cubin_sha256": (
            builder.EXP015_REUSE_CUBIN_SHA256 if arm == builder.BASELINE else "c" * 64
        ),
        "jit_artifact_set_sha256": (
            builder.EXP015_REUSE_JIT_ARTIFACT_SET_SHA256
            if arm == builder.BASELINE
            else "d" * 64
        ),
        "gpu_uuid": builder.EXPECTED_GPU_UUID,
        "observed_launch": {
            "grid": builder.EXPECTED_GRID,
            "block": builder.EXPECTED_BLOCK,
        },
        "metrics": {
            **builder.EXPECTED_WORK,
            **{metric: spill for metric in builder.SPILL_METRICS},
        },
    }


class DynamicSpillNcuTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_native_csv(
        self, path: Path, *, missing: str | None = None, na: str | None = None
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
        values_by_name = {
            **builder.EXPECTED_WORK,
            **{metric: 0 for metric in builder.SPILL_METRICS},
        }
        values = [
            "0",
            "generated_MoEDynamicKernel_symbol",
            "1",
            "7",
            "(288, 1, 1)",
            "(1, 1, 110)",
            "0",
            *("n/a" if name == na else str(values_by_name[name]) for name in names),
        ]
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["==PROF== Connected"])
            writer.writerow(header)
            writer.writerow(units)
            writer.writerow(values)

    def test_metric_parser_fails_closed(self) -> None:
        valid = self.root / "valid.csv"
        self.write_native_csv(valid)
        metrics, observed = builder.parse_native_raw(valid)
        self.assertEqual(metrics["spill_local_write"], 0)
        self.assertEqual(observed["grid"], builder.EXPECTED_GRID)
        self.assertEqual(observed["block"], builder.EXPECTED_BLOCK)

        missing = self.root / "missing.csv"
        self.write_native_csv(missing, missing="spill_local_write")
        with self.assertRaisesRegex(builder.EvidenceError, "metric absent"):
            builder.parse_native_raw(missing)

        na = self.root / "na.csv"
        self.write_native_csv(na, na="spill_register_read")
        with self.assertRaisesRegex(builder.EvidenceError, "missing/N-A"):
            builder.parse_native_raw(na)

    def test_command_profiles_one_demangled_graph_node(self) -> None:
        args = argparse.Namespace(
            flashinfer_root=Path("/workspace/flashinfer"),
            results=Path("/workspace/results"),
            jit_root=Path("/workspace/jit"),
            arm=builder.CANDIDATE,
            expected_gpu_uuid=builder.EXPECTED_GPU_UUID,
            expected_app_clock_mhz=builder.EXPECTED_APPLICATION_CLOCK_MHZ,
            ncu="ncu",
        )
        prerequisite = {
            "overlay": Path("/workspace/candidate/moe_dynamic_kernel.py"),
            "source_sha256": builder.EXPECTED_SOURCE_SHA256[builder.CANDIDATE],
            "cubin_sha256": "c" * 64,
            "jit_artifact_set_sha256": "d" * 64,
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

    def test_exp015_reuse_requires_exact_fee96_identity(self) -> None:
        record = synthetic_record(builder.BASELINE)
        exp015_record = {
            **record,
            "arm": builder.EXP015_REUSE_ARM,
            "source_sha256": builder.EXP015_REUSE_SOURCE_SHA256,
        }
        with mock.patch.object(
            builder.exp015, "validate_capture", return_value=exp015_record
        ):
            reused = builder.reuse_exp015_baseline(self.root)
        self.assertEqual(reused["arm"], builder.BASELINE)
        self.assertEqual(reused["cubin_sha256"], builder.EXP015_REUSE_CUBIN_SHA256)

        wrong = {**exp015_record, "cubin_sha256": "0" * 64}
        with (
            mock.patch.object(builder.exp015, "validate_capture", return_value=wrong),
            self.assertRaisesRegex(builder.EvidenceError, "reuse rejected"),
        ):
            builder.reuse_exp015_baseline(self.root)

    def test_summary_rejects_nonzero_candidate_spill(self) -> None:
        baseline = synthetic_record(builder.BASELINE)
        candidate = synthetic_record(builder.CANDIDATE, spill=1)
        exp014_baseline = {
            "path": self.root / "baseline-validation.json",
            "sha256": "e" * 64,
            "source_sha256": builder.EXPECTED_SOURCE_SHA256[builder.BASELINE],
            "cubin_sha256": builder.EXP015_REUSE_CUBIN_SHA256,
            "jit_artifact_set_sha256": "f" * 64,
        }
        with (
            mock.patch.object(builder, "validate_capture", return_value=candidate),
            mock.patch.object(builder, "reuse_exp015_baseline", return_value=baseline),
            mock.patch.object(
                builder, "validated_arm_identity", return_value=exp014_baseline
            ),
        ):
            value = builder.build_evidence(self.root / "results", self.root / "exp015")
        self.assertFalse(value["gate_pass"])
        self.assertEqual(value["status"], "reject")
        self.assertFalse(value["checks"]["zero_dynamic_spill"][builder.CANDIDATE])


if __name__ == "__main__":
    unittest.main()
