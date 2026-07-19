#!/usr/bin/env python3
"""CPU-only machine checks for the exp_008 marker infrastructure."""

from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest


import build_phase_marker_overlays as builder
from build_phase_marker_evidence import (
    PerturbationContractError,
    SPILL_METRICS,
    _cross_version_identity_gate,
    evaluate_version,
)
from capture_phase_timing import _artifact_gate
from exp008_marker_common import (
    COMPUTE_WARPS,
    CONTROL,
    CTA_CALIBRATION,
    CTA_ENTRY,
    CTA_LOOP_EXIT,
    CTA_LOOP_START,
    CTA_TICKS,
    CTA_W8_FINAL,
    EVENT_ABI,
    PROBE,
    SENTINEL,
    TASK_CLAIM_START,
    TASK_MAIN,
    TASK_PAIR,
    TASK_TICKS,
    additive_rollup,
    barrier_fingerprint,
    canonical_sha256,
    read_json,
    sha256_file,
    validate_control_events,
    validate_probe_events,
)


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]


class MarkerBuilderTests(unittest.TestCase):
    def test_control_probe_share_kernel_and_do_not_add_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "marker_overlays"
            summary = builder.build_all(REPO, output)
            self.assertTrue(summary["gate_pass"])
            for version in ("v0", "v1"):
                control = output / version / CONTROL
                probe = output / version / PROBE
                self.assertEqual(
                    sha256_file(control / "moe_dynamic_kernel.py"),
                    sha256_file(probe / "moe_dynamic_kernel.py"),
                )
                control_dispatch = (control / "moe_dispatch.py").read_text()
                probe_dispatch = (probe / "moe_dispatch.py").read_text()
                self.assertEqual(
                    builder._normalized_dispatch(control_dispatch),
                    builder._normalized_dispatch(probe_dispatch),
                )
                ast.parse((probe / "moe_dynamic_kernel.py").read_text())
                ast.parse(probe_dispatch)
                identity = read_json(probe / "identity.json")
                self.assertEqual(
                    identity["base"]["barrier_fingerprint"],
                    identity["overlay"]["barrier_fingerprint"],
                )
                self.assertEqual(identity["contracts"]["new_source_barriers"], 0)

    def test_direct_kernel_transform_preserves_barrier_fingerprint(self) -> None:
        for version, path in builder.BASE_KERNEL.items():
            source = path.read_text()
            instrumented = builder._instrument_kernel(source)
            self.assertEqual(
                barrier_fingerprint(source),
                barrier_fingerprint(instrumented),
                version,
            )
            self.assertEqual(instrumented.count("mov.u64 $0, %globaltimer;"), 1)
            self.assertIn("ld.volatile.shared.s32", instrumented)
            self.assertIn("ld.volatile.shared.u64", instrumented)
            final_marker = instrumented.split(
                "                    self.pass_final_barrier.arrive_unaligned()\n",
                1,
            )[1].split("                    slice_idx += Int32(1)\n", 1)[0]
            self.assertNotIn("ctrl_base_addr + Int32(28)", final_marker)
            self.assertIn("route_phys_rows_addr + Int32(8)", final_marker)
            self.assertIn("task_slot_probe", final_marker)


class MarkerBufferTests(unittest.TestCase):
    def test_disabled_buffers_remain_exact_sentinel(self) -> None:
        gate = validate_control_events(
            [[SENTINEL] * TASK_TICKS for _ in range(3)],
            [SENTINEL] * 3,
            [[SENTINEL] * CTA_TICKS for _ in range(2)],
        )
        self.assertTrue(gate["gate_pass"])
        task = [[SENTINEL] * TASK_TICKS]
        task[0][0] = 0
        self.assertFalse(
            validate_control_events(task, [SENTINEL], [[SENTINEL] * CTA_TICKS])[
                "gate_pass"
            ]
        )

    @staticmethod
    def _cta(entry: int, loop: int, exit_: int, final: int) -> list[int]:
        row = [SENTINEL] * CTA_TICKS
        for warp in range(COMPUTE_WARPS):
            row[CTA_ENTRY + warp] = entry + warp
            row[CTA_LOOP_START + warp] = loop + warp
            row[CTA_LOOP_EXIT + warp] = exit_ + warp
        row[CTA_W8_FINAL] = final
        row[CTA_CALIBRATION] = loop + 20
        row[CTA_CALIBRATION + 1] = loop + 30
        return row

    @staticmethod
    def _task(
        claim: int, cache: int, fc1_act: int, q1: int, done: int
    ) -> list[int]:
        row = [SENTINEL] * TASK_TICKS
        main = (cache, fc1_act, q1, done)
        for event, base in enumerate(main):
            for warp in range(COMPUTE_WARPS):
                row[TASK_MAIN + event * COMPUTE_WARPS + warp] = base + warp
        pair = (
            cache + 10,
            cache + 30,
            cache + 45,
            cache + 50,
            cache + 70,
            fc1_act - 10,
        )
        for event, base in enumerate(pair):
            for warp in range(COMPUTE_WARPS):
                row[TASK_PAIR + event * COMPUTE_WARPS + warp] = base + warp
        row[TASK_CLAIM_START] = claim
        return row

    def test_probe_schema_and_additive_closure(self) -> None:
        ctas = [self._cta(0, 100, 500, 510), self._cta(10, 120, 550, 560)]
        tasks = [
            self._task(150, 200, 300, 350, 450),
            self._task(160, 220, 330, 390, 500),
            [SENTINEL] * TASK_TICKS,
        ]
        owners = [0, 1, SENTINEL]
        gate = validate_probe_events(tasks, owners, ctas, task_tail=2)
        self.assertTrue(gate["gate_pass"])
        rollup = additive_rollup(tasks, owners, ctas, task_tail=2)
        self.assertTrue(rollup["closure"]["gate_pass"])
        self.assertEqual(rollup["closure"]["delta_ns"], 0)

    def test_unused_slot_must_remain_sentinel(self) -> None:
        ctas = [self._cta(0, 100, 500, 510)]
        tasks = [self._task(150, 200, 300, 350, 450), [SENTINEL] * TASK_TICKS]
        owners = [0, SENTINEL]
        tasks[1][0] = 7
        with self.assertRaisesRegex(ValueError, "unused task slot"):
            validate_probe_events(tasks, owners, ctas, task_tail=1)


class MarkerArtifactTests(unittest.TestCase):
    class Worker:
        def __init__(self, artifacts: list[dict]):
            self.artifacts = artifacts

        def artifact_manifest(self, _jit_root: Path) -> list[dict]:
            return self.artifacts

    def test_retained_cubin_is_static_sass_authority(self) -> None:
        artifacts = [
            {"path": "module.so", "size": 1, "sha256": "so"},
            {"path": "module.cubin", "size": 1, "sha256": "cubin"},
            {"path": "module.ptx", "size": 1, "sha256": "ptx"},
        ]
        _, gate = _artifact_gate(self.Worker(artifacts), Path("/fresh-jit"))
        self.assertTrue(gate["gate_pass"])
        self.assertEqual(gate["suffix_counts"][".sass"], 0)
        self.assertTrue(gate["static_sass_extraction_required_post_capture"])

    def test_missing_ptx_still_fails_closed(self) -> None:
        artifacts = [
            {"path": "module.so", "size": 1, "sha256": "so"},
            {"path": "module.cubin", "size": 1, "sha256": "cubin"},
        ]
        _, gate = _artifact_gate(self.Worker(artifacts), Path("/fresh-jit"))
        self.assertFalse(gate["gate_pass"])
        self.assertIn("fresh JIT retained no .ptx artifact", gate["errors"])


class MarkerPerturbationTests(unittest.TestCase):
    @staticmethod
    def _capture(*, arm: str, latency_us: float, version: str = "v0") -> dict:
        run = {"gate_pass": True}
        if arm == PROBE:
            run.update(
                {
                    "task_tail": 100,
                    "event_gate": {"grid_z": 100},
                    "marker_calibration": {"median": 1.0},
                    "additive_rollup": {
                        "schema": "exp008.additive-phase-rollup.v1",
                        "phases": {
                            "front_end_route_q0": {"duration_ns": 10_000},
                            "claim_cache_transition": {"duration_ns": 5_000},
                            "fc1_interleaved_activation_envelope": {
                                "duration_ns": 20_000
                            },
                            "q1": {"duration_ns": 10_000},
                            "combined_fc2_scatter": {"duration_ns": 20_000},
                            "residual": {"duration_ns": 5_000},
                        },
                        "closure": {"gate_pass": True},
                    },
                }
            )
        dispatch_sha = f"dispatch-{arm}"
        artifacts = [
            {
                "path": f"{version}/{arm}/kernel.cubin",
                "size": 1,
                "sha256": f"cubin-{arm}",
            }
        ]
        manifest = {
            "schema": "exp008.phase-marker-overlay.v1",
            "version": version,
            "arm": arm,
            "probe_enabled": arm == PROBE,
            "event_abi": EVENT_ABI,
            "base": {
                "kernel_sha256": "v0-base",
                "dispatch_sha256": "production-dispatch",
                "wrapper_sha256": "production-wrapper",
            },
            "overlay": {
                "kernel_sha256": "same-instrumented-kernel",
                "dispatch_sha256": dispatch_sha,
                "normalized_dispatch_sha256": "same-normalized-dispatch",
            },
            "contracts": {
                "dispatch_difference": "_EXP008_PHASE_PROBE_ENABLED only",
                "jit_cache_key_contains_probe_flag": True,
                "new_source_barriers": 0,
            },
        }
        return {
            "schema": "exp008.phase-marker-capture.v1",
            "version": version,
            "arm": arm,
            "event_abi": EVENT_ABI,
            "source": {
                "version": version,
                "arm": arm,
                "kernel_sha256": "same-instrumented-kernel",
                "dispatch_sha256": dispatch_sha,
                "overlay_identity": manifest,
            },
            "overlay_gate": {"gate_pass": True, "manifest": manifest},
            "jit_identity_gate": {
                "gate_pass": True,
                "artifact_set_sha256": canonical_sha256(artifacts),
            },
            "jit_artifacts": artifacts,
            "fixture": {"fixture": "canonical", "seed": 2026},
            "weights": {"weights": "canonical", "seed": 2026},
            "reference_sha256": "same-reference",
            "runtime": {
                "hostname": f"container-{version}-{arm}",
                "python": "3.12",
                "torch": "locked",
                "cuda_runtime": "locked",
                "nvcc": "locked",
                "ptxas": "locked",
                "image_digest": "locked",
                "python_deps_sha256": "locked",
                "jit_root": f"/fresh-jit/{version}/{arm}",
                "gpu": {
                    "uuid": "GPU-locked",
                    "name": "5KP",
                    "driver": "locked",
                    "applications_graphics_clock_mhz": "2377",
                    "compute_capability": [12, 0],
                    "sm_count": 110,
                },
            },
            "runs": [run, dict(run), dict(run)],
            "latency_us": {"median": latency_us},
        }

    @staticmethod
    def _resource(*, arm: str, dynamic_spill: float = 0.0) -> dict:
        dynamic = {metric: 0.0 for metric in SPILL_METRICS}
        dynamic[SPILL_METRICS[0]] = dynamic_spill
        artifacts = [
            {
                "path": f"v0/{arm}/kernel.cubin",
                "size": 1,
                "sha256": f"cubin-{arm}",
            }
        ]
        return {
            "schema": "exp008.marker-resource-evidence.v1",
            "version": "v0",
            "arm": arm,
            "gate_pass": True,
            "identity": {
                "kernel_source_sha256": "same-instrumented-kernel",
                "dispatch_source_sha256": f"dispatch-{arm}",
                "jit_artifact_set_sha256": canonical_sha256(artifacts),
                "cubin_sha256": f"cubin-{arm}",
                "kernel_symbol": "MoEDynamicKernel",
                "static_kernel_symbol": "MoEDynamicKernel",
                "ncu_kernel_symbol": "MoEDynamicKernel",
                "gpu_uuid": "GPU-locked",
            },
            "resource": {
                "registers_per_thread": 128,
                "smem_bytes": 196_608,
                "stack_bytes_per_thread": 0,
                "static_spill_load_bytes": 0,
                "static_spill_store_bytes": 0,
                "compiler_spillrefill_sass": 0,
                "achieved_occupancy_pct": 12.5,
                "dynamic_spill_metrics": dynamic,
            },
        }

    def test_low_perturbation_zero_spill_is_quantitative_probe_only(self) -> None:
        result = evaluate_version(
            "v0",
            self._capture(arm=CONTROL, latency_us=100.0),
            self._capture(arm=PROBE, latency_us=103.0),
            self._resource(arm=CONTROL),
            self._resource(arm=PROBE),
        )
        self.assertTrue(result["quantitative_phase_delta_eligible"])
        self.assertEqual(
            result["classification"],
            "quantitative_probe_level_diagnostic",
        )
        self.assertFalse(result["may_replace_uninstrumented_gate_b_latency"])

    def test_dynamic_spill_transition_forbids_quantitative_attribution(self) -> None:
        result = evaluate_version(
            "v0",
            self._capture(arm=CONTROL, latency_us=100.0),
            self._capture(arm=PROBE, latency_us=103.0),
            self._resource(arm=CONTROL),
            self._resource(arm=PROBE, dynamic_spill=1.0),
        )
        self.assertFalse(result["quantitative_phase_delta_eligible"])
        self.assertTrue(result["resource"]["spill_transition_or_count_change"])
        self.assertEqual(
            result["classification"],
            "qualitative_only_spill_or_local_resource_changed",
        )

    def test_large_negative_latency_shift_stops_attribution(self) -> None:
        result = evaluate_version(
            "v0",
            self._capture(arm=CONTROL, latency_us=100.0),
            self._capture(arm=PROBE, latency_us=80.0),
            self._resource(arm=CONTROL),
            self._resource(arm=PROBE),
        )
        self.assertEqual(result["classification"], "stop_attribution")
        self.assertFalse(result["quantitative_phase_delta_eligible"])

    def test_fixture_drift_fails_closed(self) -> None:
        probe = self._capture(arm=PROBE, latency_us=103.0)
        probe["fixture"]["seed"] = 7
        with self.assertRaisesRegex(PerturbationContractError, "identity drift"):
            evaluate_version(
                "v0",
                self._capture(arm=CONTROL, latency_us=100.0),
                probe,
                self._resource(arm=CONTROL),
                self._resource(arm=PROBE),
            )

    def test_resource_cubin_must_belong_to_capture_jit_set(self) -> None:
        probe_resource = self._resource(arm=PROBE)
        probe_resource["identity"]["cubin_sha256"] = "foreign-cubin"
        with self.assertRaisesRegex(PerturbationContractError, "resource/capture"):
            evaluate_version(
                "v0",
                self._capture(arm=CONTROL, latency_us=100.0),
                self._capture(arm=PROBE, latency_us=103.0),
                self._resource(arm=CONTROL),
                probe_resource,
            )

    def test_cross_version_identity_requires_same_gpu_and_four_jit_roots(self) -> None:
        captures = {
            version: {
                arm: self._capture(
                    arm=arm,
                    latency_us=100.0,
                    version=version,
                )
                for arm in (CONTROL, PROBE)
            }
            for version in ("v0", "v1")
        }
        self.assertTrue(_cross_version_identity_gate(captures)["gate_pass"])
        captures["v1"][PROBE]["runtime"]["gpu"]["uuid"] = "GPU-drift"
        self.assertFalse(_cross_version_identity_gate(captures)["gate_pass"])


if __name__ == "__main__":
    unittest.main()
