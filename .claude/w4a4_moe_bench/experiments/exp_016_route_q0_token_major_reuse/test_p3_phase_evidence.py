#!/usr/bin/env python3
"""CPU-only fail-closed tests for the exp_016 P3 evidence reducer."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import unittest

import build_p3_phase_evidence as evidence_builder
from exp016_p3_probe_common import (
    ARMS,
    BASELINE,
    CANDIDATE,
    CONTROL,
    EVENT_ABI,
    EXPECTED_BASE_KERNEL_SHA256,
    EXPECTED_DISPATCH_SHA256,
    EXPECTED_WRAPPER_SHA256,
    MODES,
    PROBE,
    file_sha256,
    write_json,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def artifact_set_sha256(artifacts: object) -> str:
    payload = json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_zero_spill_sidecar(capture_path: Path) -> Path:
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    sass_path = capture_path.parent / "p3_spill.sass.txt"
    elf_path = capture_path.parent / "p3_spill.elf.txt"
    sass_path.write_text("synthetic zero-spill SASS\n", encoding="utf-8")
    elf_path.write_text("synthetic zero-spill ELF\n", encoding="utf-8")
    cubin_sha = capture["cubin_sha256"][0]
    kernel_symbol = capture["static_resource_usage"]["records"][0]["kernel_symbol"]
    sidecar = {
        "schema": evidence_builder.SASS_SPILL_SCHEMA,
        "arm": capture["arm"],
        "mode": capture["mode"],
        "capture": {
            "path": capture_path.name,
            "sha256": file_sha256(capture_path),
        },
        "jit": {
            "root": f"/synthetic/jit/{capture['arm']}/{capture['mode']}",
            "cubin_inventory_count": 1,
            "matched_cubin_relative_path": "dump/kernel.cubin",
            "matched_cubin_sha256": cubin_sha,
        },
        "kernel_symbol": kernel_symbol,
        "tool": {
            "path": "/synthetic/cuobjdump",
            "sha256": digest("cuobjdump"),
            "version": "synthetic cuobjdump",
            "commands": [
                ["--dump-sass", "/synthetic/kernel.cubin"],
                ["--dump-elf", "/synthetic/kernel.cubin"],
            ],
        },
        "raw_sass": {
            "path": sass_path.name,
            "sha256": file_sha256(sass_path),
            "size": sass_path.stat().st_size,
        },
        "raw_elf": {
            "path": elf_path.name,
            "sha256": file_sha256(elf_path),
            "size": elf_path.stat().st_size,
        },
        "counts": {
            "sass_instruction_count": 100,
            "spill_refill_annotation_count": 0,
            "spill_refill_annotation_unique_pc_count": 0,
            "ldl_opcode_count": 0,
            "stl_opcode_count": 0,
            "local_sass_opcode_count": 0,
            "local_sass_opcode_histogram": {},
            "annotation_pcs_equal_local_sass_pcs": True,
        },
        "integrity_checks": {
            "capture_cubin_sha256_matches_unique_jit_cubin": True,
            "capture_kernel_symbol_in_sass": True,
            "capture_kernel_symbol_in_elf": True,
            "sass_instruction_parse_nonempty": True,
            "annotation_pcs_equal_local_sass_pcs": True,
        },
        "evidence_integrity_gate_pass": True,
        "sass_spill_gate_pass": True,
        "gate_pass": True,
    }
    sidecar_path = evidence_builder.sass_spill_path(capture_path)
    write_json(sidecar_path, sidecar)
    return sidecar_path


def refresh_sidecar_capture_sha(capture_path: Path) -> None:
    sidecar_path = evidence_builder.sass_spill_path(capture_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["capture"]["sha256"] = file_sha256(capture_path)
    write_json(sidecar_path, sidecar)


def source_identity(arm: str, mode: str) -> dict[str, object]:
    return {
        "arm": arm,
        "mode": mode,
        "kernel_overlay": f"/synthetic/{arm}/moe_dynamic_kernel.py",
        "kernel_sha256": digest(f"kernel-{arm}"),
        "dispatch_overlay": f"/synthetic/{arm}/{mode}/moe_dispatch.py",
        "dispatch_sha256": digest(f"dispatch-{mode}"),
        "base_kernel_sha256": EXPECTED_BASE_KERNEL_SHA256[arm],
        "event_abi": EVENT_ABI,
    }


def overlay_identities() -> tuple[dict[str, object], dict[tuple[str, str], object]]:
    arm_identities: dict[tuple[str, str], object] = {}
    arms: dict[str, object] = {}
    cross_mode: dict[str, object] = {}
    for arm in ARMS:
        arms[arm] = {}
        for mode in MODES:
            source = source_identity(arm, mode)
            identity = {
                "schema": "exp016.p3-phase-probe-overlay.v1",
                "arm": arm,
                "mode": mode,
                "probe_enabled": mode == PROBE,
                "event_abi": EVENT_ABI,
                "classification": (
                    "diagnostic" if mode == PROBE else "marker-disabled ABI control"
                ),
                "base": {
                    "kernel_path": f"/synthetic/base/{arm}/moe_dynamic_kernel.py",
                    "kernel_sha256": EXPECTED_BASE_KERNEL_SHA256[arm],
                    "dispatch_path": "/synthetic/base/moe_dispatch.py",
                    "dispatch_sha256": EXPECTED_DISPATCH_SHA256,
                    "wrapper_path": "/synthetic/base/b12x_moe.py",
                    "wrapper_sha256": EXPECTED_WRAPPER_SHA256,
                    "barrier_fingerprint": {"synthetic_sync": 1},
                },
                "boundary": {
                    "start_anchor_count": 1,
                    "end_anchor_count": 1,
                    "timer_read_call_sites": 2,
                    "new_barriers": 0,
                },
                "overlay": {
                    "kernel_sha256": source["kernel_sha256"],
                    "dispatch_sha256": source["dispatch_sha256"],
                    "barrier_fingerprint": {"synthetic_sync": 1},
                },
            }
            arm_identities[(arm, mode)] = identity
            arms[arm][mode] = identity
        cross_mode[arm] = {
            "kernel_source_shared": True,
            "dispatch_diff_is_only_probe_flag": True,
            "event_abi_shared": True,
        }
    root = {
        "schema": "exp016.p3-phase-probe-overlays.v1",
        "event_abi": EVENT_ABI,
        "arms": arms,
        "cross_mode": cross_mode,
    }
    return root, arm_identities


def gate() -> dict[str, bool]:
    return {"gate_pass": True}


def timing(mode: str, wall_us: float | None) -> dict[str, object]:
    if mode == CONTROL:
        return {
            "gate_pass": True,
            "grid_critical_wall_us": None,
            "all_sentinel": True,
        }
    return {
        "gate_pass": True,
        "grid_critical_wall_us": wall_us,
        "all_sentinel": False,
        "additive_estimate_reported": False,
    }


def make_capture(
    arm: str,
    mode: str,
    *,
    root_identity: dict[str, object],
    arm_identity: object,
    e2e_samples: list[float],
    wall_samples: list[float] | None,
    registers: int = 128,
    stack: int = 64,
    local: int = 0,
) -> dict[str, object]:
    source = source_identity(arm, mode)
    runtime = {
        "source": copy.deepcopy(source),
        "hostname": "synthetic-5kp",
        "python": "3.12.0",
        "packages": {"torch": "2.synthetic"},
        "torch": "2.synthetic",
        "cuda_runtime": "13.0",
        "nvcc": "13.0",
        "ptxas": "13.0",
        "image_digest": digest("image-digest"),
        "image_id": digest("image-id"),
        "python_deps_sha256": digest("python-deps"),
        "gpu": {
            "uuid": "GPU-synthetic",
            "name": "NVIDIA Graphics Device",
            "pci_bus_id": "0000:01:00.0",
            "driver": "synthetic",
            "applications_graphics_clock_mhz": 2100,
            "max_graphics_clock_mhz": 2100,
            "compute_capability": [12, 0],
            "sm_count": 110,
            "foreign_processes_before_cuda_context": [],
        },
        "imports": {
            "target_module": source["kernel_overlay"],
            "cutlass_python": "/synthetic/cutlass/__init__.py",
            "cutlass_python_version": "4.6.0",
            "flashinfer": "/synthetic/flashinfer/__init__.py",
        },
        "harness": {
            "sha256": digest("capture-harness"),
            "run_exp016_arm_sha256": digest("run-harness"),
        },
    }
    runs = []
    for replay, e2e_us in enumerate(e2e_samples):
        wall_us = None if wall_samples is None else wall_samples[replay]
        runs.append(
            {
                "replay": replay,
                "event_elapsed_us": e2e_us,
                "output_sha256": digest(f"output-{arm}-{mode}-{replay}"),
                "ticks_sha256": digest(f"ticks-{arm}-{mode}-{replay}"),
                "correctness_gate": gate(),
                "route_task_gate": gate(),
                "p3_timing": timing(mode, wall_us),
                "gate_pass": True,
            }
        )
    if wall_samples is None:
        p3_summary: dict[str, object] = {"grid_critical_wall_us": None}
    else:
        p3_summary = {
            "grid_critical_wall_us": {
                "median": statistics.median(wall_samples),
                "min": min(wall_samples),
                "max": max(wall_samples),
                "samples": len(wall_samples),
            },
            "additive_estimate_reported": False,
        }
    cubin_sha = digest(f"cubin-{arm}-{mode}")
    artifacts = [
        {
            "path": f"jit/{arm}/{mode}/kernel.cubin",
            "sha256": cubin_sha,
            "size": 12345,
        }
    ]
    return {
        "schema": evidence_builder.CAPTURE_SCHEMA,
        "classification": (
            "diagnostic matched probe"
            if mode == PROBE
            else "marker-disabled ABI control"
        ),
        "arm": arm,
        "mode": mode,
        "m": 8192,
        "fixture": "canonical",
        "scale_kind": "unequal",
        "event_abi": EVENT_ABI,
        "source": source,
        "overlay_gate": {
            "gate_pass": True,
            "errors": [],
            "arm": arm,
            "mode": mode,
            "kernel": source["kernel_overlay"],
            "dispatch": source["dispatch_overlay"],
            "root_identity": root_identity,
            "arm_identity": arm_identity,
        },
        "runtime": runtime,
        "fixture_identity": {"fixture": "synthetic-canonical", "seed": 2026},
        "weight_identity": {"weights": "synthetic-unequal", "seed": 2026},
        "reference_sha256": digest("reference"),
        "eager": {
            "correctness_gate": gate(),
            "route_task_gate": gate(),
            "specialization_gate": gate(),
            "p3_timing": timing(
                mode,
                None if wall_samples is None else wall_samples[0],
            ),
            "gate_pass": True,
        },
        "runs": runs,
        "p3_summary": p3_summary,
        "probe_e2e_us": {
            "median": statistics.median(e2e_samples),
            "min": min(e2e_samples),
            "max": max(e2e_samples),
            "samples": len(e2e_samples),
        },
        "jit_artifacts": artifacts,
        "jit_artifact_set_sha256": artifact_set_sha256(artifacts),
        "cubin_sha256": [cubin_sha],
        "static_resource_usage": {
            "schema": "exp016.p3-phase-resource-usage.v1",
            "gate_pass": True,
            "raw_sha256": digest(f"resource artifact {arm} {mode}\n"),
            "raw_path": f"/remote/.{mode}.in-progress/resource_usage.txt",
            "records": [
                {
                    "cubin_sha256": cubin_sha,
                    "kernel_symbol": "synthetic::MoEDynamicKernel",
                    "registers_per_thread": registers,
                    "stack_bytes_per_thread": stack,
                    "static_shared_bytes_per_cta": 8192,
                    "static_local_bytes_outside_stack": local,
                }
            ],
        },
    }


def write_complete_capture_set(results: Path) -> dict[tuple[str, str], Path]:
    root_identity, identities = overlay_identities()
    paths = {}
    for arm in ARMS:
        for mode in MODES:
            if arm == BASELINE:
                walls = [290.0, 300.0, 310.0, 300.0, 300.0]
                control_e2e = [1000.0] * 5
                probe_e2e = [1005.0] * 5
            else:
                walls = [190.0, 200.0, 210.0, 200.0, 200.0]
                control_e2e = [900.0] * 5
                probe_e2e = [904.0] * 5
            capture = make_capture(
                arm,
                mode,
                root_identity=root_identity,
                arm_identity=identities[(arm, mode)],
                e2e_samples=control_e2e if mode == CONTROL else probe_e2e,
                wall_samples=None if mode == CONTROL else walls,
            )
            path = evidence_builder.capture_path(results, arm, mode)
            write_json(path, capture)
            (path.parent / "resource_usage.txt").write_text(
                f"resource artifact {arm} {mode}\n", encoding="utf-8"
            )
            write_zero_spill_sidecar(path)
            paths[(arm, mode)] = path
    return paths


class P3PhaseEvidenceTest(unittest.TestCase):
    def test_missing_captures_are_unresolved(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = evidence_builder.build_evidence(Path(directory))
        self.assertEqual(evidence["status"], "Unresolved")
        self.assertFalse(evidence["gate_pass"])
        self.assertEqual(evidence["observed_capture_count"], 0)
        self.assertEqual(len(evidence["missing_captures"]), 4)

    def test_valid_matched_capture_set_reduces_only_grid_wall(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            write_complete_capture_set(results)
            evidence = evidence_builder.build_evidence(results)
        comparison = evidence["phase_comparison"]
        self.assertEqual(evidence["status"], "Complete")
        self.assertTrue(evidence["gate_pass"])
        self.assertEqual(evidence["evidence_classification"], "diagnostic")
        self.assertEqual(comparison["baseline_grid_critical_wall_median_us"], 300.0)
        self.assertEqual(comparison["candidate_grid_critical_wall_median_us"], 200.0)
        self.assertEqual(comparison["candidate_minus_baseline_us"], -100.0)
        self.assertAlmostEqual(comparison["latency_reduction_percent"], 100.0 / 3.0)
        self.assertTrue(comparison["candidate_faster"])
        self.assertTrue(comparison["interpretation_legal_as_diagnostic"])
        self.assertFalse(comparison["interpretation_legal_as_production_phase_truth"])
        self.assertFalse(evidence["additive_sm_estimate_used"])

    def test_resource_mismatch_downgrades_to_unresolved(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            paths = write_complete_capture_set(results)
            path = paths[(CANDIDATE, PROBE)]
            capture = json.loads(path.read_text(encoding="utf-8"))
            capture["static_resource_usage"]["records"][0][
                "static_shared_bytes_per_cta"
            ] += 128
            write_json(path, capture)
            refresh_sidecar_capture_sha(path)
            evidence = evidence_builder.build_evidence(results)
        self.assertEqual(evidence["status"], "Complete")
        self.assertFalse(evidence["gate_pass"])
        self.assertFalse(evidence["instrumentation_gate_pass"])
        self.assertEqual(evidence["evidence_classification"], "diagnostic/unresolved")
        self.assertFalse(
            evidence["phase_comparison"][
                "interpretation_legal_as_production_phase_truth"
            ]
        )

    def test_candidate_phase_regression_fails_phase_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            paths = write_complete_capture_set(results)
            path = paths[(CANDIDATE, PROBE)]
            capture = json.loads(path.read_text(encoding="utf-8"))
            walls = [390.0, 400.0, 410.0, 400.0, 400.0]
            for run, wall in zip(capture["runs"], walls, strict=True):
                run["p3_timing"]["grid_critical_wall_us"] = wall
            capture["p3_summary"]["grid_critical_wall_us"] = {
                "median": 400.0,
                "min": 390.0,
                "max": 410.0,
                "samples": 5,
            }
            write_json(path, capture)
            refresh_sidecar_capture_sha(path)
            evidence = evidence_builder.build_evidence(results)
        self.assertTrue(evidence["instrumentation_gate_pass"])
        self.assertFalse(evidence["phase_improvement_gate_pass"])
        self.assertFalse(evidence["gate_pass"])
        self.assertFalse(evidence["phase_comparison"]["candidate_faster"])
        self.assertEqual(evidence["evidence_classification"], "diagnostic/unresolved")

    def test_material_probe_e2e_perturbation_is_unresolved(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            paths = write_complete_capture_set(results)
            path = paths[(BASELINE, PROBE)]
            capture = json.loads(path.read_text(encoding="utf-8"))
            for run in capture["runs"]:
                run["event_elapsed_us"] = 1100.0
            capture["probe_e2e_us"] = {
                "median": 1100.0,
                "min": 1100.0,
                "max": 1100.0,
                "samples": 5,
            }
            write_json(path, capture)
            refresh_sidecar_capture_sha(path)
            evidence = evidence_builder.build_evidence(results)
        self.assertFalse(evidence["instrumentation_gate_pass"])
        self.assertFalse(
            evidence["instrumentation"][BASELINE]["e2e_perturbation_small"]
        )

    def test_missing_sass_spill_sidecar_is_unresolved(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            paths = write_complete_capture_set(results)
            evidence_builder.sass_spill_path(paths[(BASELINE, PROBE)]).unlink()
            evidence = evidence_builder.build_evidence(results)
        self.assertEqual(evidence["status"], "Unresolved")
        self.assertFalse(evidence["gate_pass"])
        self.assertFalse(evidence["sass_spill_gate_pass"])
        self.assertEqual(len(evidence["missing_sass_spill_sidecars"]), 1)

    def test_sass_raw_hash_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            paths = write_complete_capture_set(results)
            path = paths[(CANDIDATE, CONTROL)]
            (path.parent / "p3_spill.sass.txt").write_text(
                "tampered SASS\n", encoding="utf-8"
            )
            with self.assertRaises(evidence_builder.EvidenceError):
                evidence_builder.build_evidence(results)

    def test_nonzero_sass_spill_is_valid_evidence_but_unresolved(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            paths = write_complete_capture_set(results)
            sidecar_path = evidence_builder.sass_spill_path(paths[(CANDIDATE, PROBE)])
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["counts"].update(
                {
                    "spill_refill_annotation_count": 1,
                    "spill_refill_annotation_unique_pc_count": 1,
                    "ldl_opcode_count": 1,
                    "stl_opcode_count": 0,
                    "local_sass_opcode_count": 1,
                    "local_sass_opcode_histogram": {"LDL": 1},
                }
            )
            sidecar["sass_spill_gate_pass"] = False
            sidecar["gate_pass"] = False
            write_json(sidecar_path, sidecar)
            evidence = evidence_builder.build_evidence(results)
        self.assertEqual(evidence["status"], "Complete")
        self.assertFalse(evidence["sass_spill_gate_pass"])
        self.assertFalse(evidence["instrumentation_gate_pass"])
        self.assertFalse(evidence["gate_pass"])
        self.assertEqual(evidence["evidence_classification"], "diagnostic/unresolved")

    def test_failed_capture_gate_or_runtime_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            paths = write_complete_capture_set(results)
            path = paths[(BASELINE, CONTROL)]
            capture = json.loads(path.read_text(encoding="utf-8"))
            capture["runs"][0]["route_task_gate"]["gate_pass"] = False
            write_json(path, capture)
            with self.assertRaises(evidence_builder.EvidenceError):
                evidence_builder.build_evidence(results)

    def test_byte_exact_base_or_captured_resource_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            paths = write_complete_capture_set(results)
            path = paths[(BASELINE, PROBE)]
            capture = json.loads(path.read_text(encoding="utf-8"))
            capture["overlay_gate"]["arm_identity"]["base"]["wrapper_sha256"] = digest(
                "wrong-wrapper"
            )
            capture["overlay_gate"]["root_identity"]["arms"][BASELINE][PROBE]["base"][
                "wrapper_sha256"
            ] = digest("wrong-wrapper")
            write_json(path, capture)
            with self.assertRaises(evidence_builder.EvidenceError):
                evidence_builder.build_evidence(results)

        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            paths = write_complete_capture_set(results)
            path = paths[(CANDIDATE, CONTROL)]
            (path.parent / "resource_usage.txt").write_text(
                "tampered resource artifact\n", encoding="utf-8"
            )
            with self.assertRaises(evidence_builder.EvidenceError):
                evidence_builder.build_evidence(results)

        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            paths = write_complete_capture_set(results)
            path = paths[(CANDIDATE, PROBE)]
            capture = json.loads(path.read_text(encoding="utf-8"))
            capture["runtime"]["nvcc"] = "drifted"
            write_json(path, capture)
            with self.assertRaises(evidence_builder.EvidenceError):
                evidence_builder.build_evidence(results)


if __name__ == "__main__":
    unittest.main()
