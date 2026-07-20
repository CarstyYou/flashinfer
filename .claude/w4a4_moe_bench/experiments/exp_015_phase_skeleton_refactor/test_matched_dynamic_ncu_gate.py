#!/usr/bin/env python3
"""CPU-only contract tests for the exp_015 matched dynamic NCU gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import build_matched_dynamic_ncu_evidence as builder
import capture_matched_dynamic_ncu as capture


class MatchedDynamicNcuGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.results = Path(self.temporary.name) / "results"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def source_for_arm(self, arm: str) -> Path:
        if arm == builder.BASELINE:
            return (
                builder.ROOT.parent
                / "exp_008_branch_paired_n64_reuse/results/overlays/branch_paired_n64_v1/moe_dynamic_kernel.py"
            )
        return builder.ROOT.parents[1] / "moe_dynamic_kernel_opt.py"

    def write_native_csv(
        self,
        path: Path,
        *,
        spill: int = 0,
        tensor: int = builder.EXPECTED_WORK["executed_tensor_instructions"],
        fp4: int = builder.EXPECTED_WORK["fp4_tensor_ops"],
        missing_metric: str | None = None,
        na_metric: str | None = None,
        block: str = "(288, 1, 1)",
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
        metric_names = [name for name in builder.METRIC_IDS if name != missing_metric]
        header = [*identity, *(builder.METRIC_IDS[name] for name in metric_names)]
        units = ["" for _ in identity] + [
            builder.EXPECTED_UNITS[name] for name in metric_names
        ]
        metric_values = {
            "executed_tensor_instructions": tensor,
            "fp4_tensor_ops": fp4,
            "spill_register_read": spill,
            "spill_register_write": spill,
            "spill_local_read": spill,
            "spill_local_write": spill,
        }
        values = [
            "0",
            "generated_MoEDynamicKernel_symbol",
            "1",
            "7",
            block,
            "(1, 1, 110)",
            "0",
            *(
                "n/a" if name == na_metric else str(metric_values[name])
                for name in metric_names
            ),
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["==PROF== Connected"])
            writer.writerow(header)
            writer.writerow(units)
            writer.writerow(values)

    def write_capture(
        self,
        arm: str,
        *,
        spill: int = 0,
        tensor: int = builder.EXPECTED_WORK["executed_tensor_instructions"],
        block: str = "(288, 1, 1)",
    ) -> None:
        overlay = self.results / "overlays" / arm / "moe_dynamic_kernel.py"
        overlay.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.source_for_arm(arm), overlay)

        root = builder.capture_root(self.results, arm)
        root.mkdir(parents=True)
        native = root / "native_raw.csv"
        report = root / "trace.ncu-rep"
        target_path = root / "profile_target.json"
        self.write_native_csv(native, spill=spill, tensor=tensor, block=block)
        report.write_bytes(b"synthetic NCU report")
        metrics, observed = builder.parse_native_raw(native)
        target = {
            "schema": "exp015.matched-ncu-profile-target.v1",
            "status": "complete",
            "arm": arm,
            "m": builder.M,
            "fixture_kind": builder.FIXTURE,
            "source_sha256": builder.EXPECTED_SOURCE_SHA256[arm],
            "cubin_sha256": builder.EXPECTED_CUBIN_SHA256[arm],
            "jit_artifact_set_sha256": builder.EXPECTED_JIT_ARTIFACT_SET_SHA256[arm],
            "gpu_uuid": builder.EXPECTED_GPU_UUID,
            "expected_launch": {
                "grid": builder.EXPECTED_GRID,
                "block": builder.EXPECTED_BLOCK,
                "kernel": "MoEDynamicKernel",
            },
            "runtime": {
                "gpu": {
                    "uuid": builder.EXPECTED_GPU_UUID,
                    "applications_graphics_clock_mhz": str(
                        builder.EXPECTED_APPLICATION_CLOCK_MHZ
                    ),
                },
                "source": {"overlay_sha256": builder.EXPECTED_SOURCE_SHA256[arm]},
            },
        }
        target_path.write_text(json.dumps(target))
        validation_path = self.results / "raw" / "validation" / arm / "validation.json"
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.write_text(
            json.dumps(
                {
                    "schema": "exp015.arm-validation.v1",
                    "status": "complete",
                    "gate_pass": True,
                    "arm": arm,
                    "cubin_sha256": [builder.EXPECTED_CUBIN_SHA256[arm]],
                    "jit_artifact_set_sha256": (
                        builder.EXPECTED_JIT_ARTIFACT_SET_SHA256[arm]
                    ),
                    "runtime": {
                        "source": {
                            "overlay_sha256": builder.EXPECTED_SOURCE_SHA256[arm]
                        }
                    },
                }
            )
        )
        identity = {
            "schema": "exp015.matched-ncu-capture-identity.v1",
            "arm": arm,
            "m": builder.M,
            "fixture": builder.FIXTURE,
            "source_sha256": builder.EXPECTED_SOURCE_SHA256[arm],
            "cubin_sha256": builder.EXPECTED_CUBIN_SHA256[arm],
            "jit_artifact_set_sha256": builder.EXPECTED_JIT_ARTIFACT_SET_SHA256[arm],
            "gpu_uuid": builder.EXPECTED_GPU_UUID,
            "expected_application_graphics_clock_mhz": (
                builder.EXPECTED_APPLICATION_CLOCK_MHZ
            ),
            "expected_grid": builder.EXPECTED_GRID,
            "expected_block": builder.EXPECTED_BLOCK,
            "kernel_symbol": observed["kernel_symbol"],
            "observed_launch": observed,
            "metrics": metrics,
            "required_metric_ids": list(builder.METRIC_IDS.values()),
            "trace_sha256": builder.file_sha256(report),
            "native_raw_sha256": builder.file_sha256(native),
            "profile_target_sha256": builder.file_sha256(target_path),
            "validation_manifest_sha256": builder.file_sha256(validation_path),
        }
        (root / "capture_identity.json").write_text(json.dumps(identity))

    def test_metric_selection_uses_proven_section_and_work_metric_union(self) -> None:
        arguments = capture.metric_selection_args()
        self.assertEqual(arguments.count("--section"), len(capture.SECTION_IDS))
        self.assertIn("InstructionStats", capture.SECTION_IDS)
        self.assertIn("SourceCounters", capture.SECTION_IDS)
        metrics_index = arguments.index("--metrics")
        self.assertEqual(
            tuple(arguments[metrics_index + 1].split(",")),
            capture.EXPLICIT_METRIC_IDS,
        )
        self.assertEqual(
            set(builder.METRIC_IDS),
            set(builder.SPILL_METRICS) | set(builder.WORK_METRICS),
        )

    def test_ncu_command_profiles_one_demangled_graph_node(self) -> None:
        args = argparse.Namespace(
            flashinfer_root=Path("/workspace/flashinfer"),
            results=Path("/workspace/results"),
            jit_root=Path("/workspace/jit"),
            arm=builder.BASELINE,
            expected_gpu_uuid=builder.EXPECTED_GPU_UUID,
            expected_app_clock_mhz=capture.EXPECTED_APPLICATION_CLOCK_MHZ,
            ncu="ncu",
        )
        prerequisite = {
            "overlay": Path(
                "/workspace/results/overlays/baseline/moe_dynamic_kernel.py"
            ),
            "source_sha256": builder.EXPECTED_SOURCE_SHA256[builder.BASELINE],
            "cubin_sha256": builder.EXPECTED_CUBIN_SHA256[builder.BASELINE],
            "jit_artifact_set_sha256": builder.EXPECTED_JIT_ARTIFACT_SET_SHA256[
                builder.BASELINE
            ],
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

    def test_parse_native_raw_requires_all_numeric_integral_metrics(self) -> None:
        valid = self.results / "valid.csv"
        self.write_native_csv(valid)
        metrics, launch = builder.parse_native_raw(valid)
        self.assertEqual(
            metrics,
            {**builder.EXPECTED_WORK, **{name: 0 for name in builder.SPILL_METRICS}},
        )
        self.assertEqual(launch["grid"], builder.EXPECTED_GRID)
        self.assertEqual(launch["block"], builder.EXPECTED_BLOCK)

        missing = self.results / "missing.csv"
        self.write_native_csv(missing, missing_metric="spill_local_write")
        with self.assertRaisesRegex(builder.EvidenceError, "metric absent"):
            builder.parse_native_raw(missing)

        na = self.results / "na.csv"
        self.write_native_csv(na, na_metric="fp4_tensor_ops")
        with self.assertRaisesRegex(builder.EvidenceError, "missing/N-A"):
            builder.parse_native_raw(na)

    def test_parse_native_raw_preserves_launch_identity_for_gate(self) -> None:
        path = self.results / "wrong_block.csv"
        self.write_native_csv(path, block="(160, 1, 1)")
        _, launch = builder.parse_native_raw(path)
        self.assertNotEqual(launch["block"], builder.EXPECTED_BLOCK)

    def test_matched_gate_passes_zero_spill_and_exp008_work_identity(self) -> None:
        for arm in builder.ARMS:
            self.write_capture(arm)
        value = builder.build_evidence(self.results)
        self.assertTrue(value["gate_pass"])
        self.assertEqual(value["status"], "pass")
        self.assertTrue(all(value["checks"]["zero_dynamic_spill"].values()))

    def test_nonzero_spill_rejects_without_erasing_evidence(self) -> None:
        self.write_capture(builder.BASELINE)
        self.write_capture(builder.CANDIDATE, spill=1)
        value = builder.build_evidence(self.results)
        self.assertFalse(value["gate_pass"])
        self.assertEqual(value["status"], "reject")
        self.assertFalse(value["checks"]["zero_dynamic_spill"][builder.CANDIDATE])

    def test_equal_but_wrong_tensor_work_rejects_against_ledger(self) -> None:
        wrong = builder.EXPECTED_WORK["executed_tensor_instructions"] - 1
        for arm in builder.ARMS:
            self.write_capture(arm, tensor=wrong)
        value = builder.build_evidence(self.results)
        self.assertTrue(all(value["checks"]["pairwise_tensor_work_identity"].values()))
        self.assertFalse(value["gate_pass"])
        self.assertFalse(
            value["checks"]["exp008_tensor_work_identity"][
                "baseline:executed_tensor_instructions"
            ]
        )

    def test_identity_hash_drift_fails_closed(self) -> None:
        for arm in builder.ARMS:
            self.write_capture(arm)
        identity_path = (
            builder.capture_root(self.results, builder.CANDIDATE)
            / "capture_identity.json"
        )
        identity = json.loads(identity_path.read_text())
        identity["cubin_sha256"] = "0" * 64
        identity_path.write_text(json.dumps(identity))
        with self.assertRaisesRegex(builder.EvidenceError, "cubin hash drift"):
            builder.build_evidence(self.results)

    def test_observed_block_drift_fails_closed(self) -> None:
        self.write_capture(builder.BASELINE, block="(160, 1, 1)")
        self.write_capture(builder.CANDIDATE)
        with self.assertRaisesRegex(builder.EvidenceError, "block drift"):
            builder.build_evidence(self.results)


if __name__ == "__main__":
    unittest.main()
