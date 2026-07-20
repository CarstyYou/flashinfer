#!/usr/bin/env python3
"""CPU-only tests for the exp_015 evidence collector."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import build_exp015_evidence as evidence


ZERO_ERROR = {
    "cosine_loss": 0.0,
    "relative_l2": 0.0,
    "max_abs": 0.0,
    "token_rel_l2_p99": 0.0,
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def route_record(task_count: int = 4) -> dict[str, object]:
    verification = {
        "gate_pass": True,
        "expected_task_count": task_count,
        "observed_task_tail": task_count,
        "observed_task_head": task_count + 110,
        "terminal_head_overshoot": 110,
        "producer_claim_count": 18,
        "expected_pair_head": 2500,
        "observed_pair_head": 2500,
        "task_descriptor_multiset_sha256": "d" * 64,
    }
    return {"summary": {"verification": verification}}


def correctness_arm() -> dict[str, object]:
    cases = {}
    outputs = {}
    for m, fixture in evidence.CASE_SPECS:
        case = {
            "case": {
                "experts": 256,
                "hidden": 2048,
                "intermediate_tp": 512,
                "topk": 8,
            },
            "fixture": {"m": m, "fixture": fixture},
            "weights": {"seed": 2026},
            "reference": {"dtype": "float32", "sha256": "a" * 64},
            "output_stability": dict(ZERO_ERROR),
            "route_task_evidence": [route_record(), route_record()],
        }
        cases[(m, fixture)] = {
            "value": case,
            "path": f"raw/m{m}/{fixture}/case.json",
            "sha256": "b" * 64,
        }
        outputs[(m, fixture)] = [f"m{m}-{fixture}-r0", f"m{m}-{fixture}-r1"]
    return {"identity": {"runtime": "same"}, "cases": cases, "outputs": outputs}


def synthetic_runtime(
    arm: str,
    overlay: Path,
    jit_root: str,
    *,
    harness_sha256: str = evidence.EXPECTED_VALIDATION_HARNESS_SHA256,
) -> tuple[dict, dict]:
    runtime = {
        "hostname": "R6KD-CX8aaS-GPU-16",
        "python": "3.12",
        "packages": {"nvidia-cutlass-dsl": "4.6.0"},
        "torch": "2.12",
        "cuda_runtime": "13.2",
        "nvcc": "Cuda compilation tools, release 13.2, V13.2.78",
        "ptxas": "ptxas release 13.2",
        "cuda_visible_devices": "2",
        "image_digest": evidence.common.EXPECTED_IMAGE_DIGEST,
        "image_id": evidence.EXPECTED_IMAGE_ID,
        "python_deps_sha256": evidence.common.EXPECTED_PYTHON_DEPS_SHA256,
        "lease_id": "exp015-test-lease",
        "jit_root": jit_root,
        "source": {
            "locked_flashinfer_commit": evidence.common.EXPECTED_FLASHINFER_COMMIT,
            "checkout_head": evidence.common.EXPECTED_FLASHINFER_COMMIT,
            "cutlass_commit": evidence.common.EXPECTED_CUTLASS_COMMIT,
            "production_kernel_sha256": evidence.EXPECTED_PRODUCTION,
            "oracle_source_sha256": evidence.EXPECTED_ORACLE_SHA256,
            "overlay": str(overlay),
            "overlay_sha256": evidence.EXPECTED_SOURCE[arm],
        },
        "gpu": {
            "uuid": evidence.EXPECTED_GPU_UUID,
            "name": "NVIDIA Graphics Device",
            "pci_bus_id": "00000000:18:00.0",
            "driver": "999.0",
            "applications_graphics_clock_mhz": "2377",
            "max_graphics_clock_mhz": "2377",
            "compute_capability": [12, 0],
            "sm_count": 110,
            "foreign_processes_before_cuda_context": [],
        },
        "harness": {
            "path": "/workspace/run_exp015_arm.py",
            "sha256": harness_sha256,
        },
    }
    imports = {
        "flashinfer": "/workspace/flashinfer/__init__.py",
        "target_module": str(overlay),
        "cutlass_python": "/site-packages/cutlass/__init__.py",
        "cutlass_python_version": "4.6.0",
        "flashinfer_jit_workspace": jit_root,
    }
    return runtime, imports


def write_validation_arm(results: Path, arm: str) -> None:
    frozen_harness = results / "harness" / "run_exp015_arm_validation_v1.py"
    if not frozen_harness.exists():
        frozen_harness.parent.mkdir(parents=True, exist_ok=True)
        frozen_harness.write_text("# synthetic frozen validation harness\n")
    baseline_source = (
        evidence.ROOT.parent
        / "exp_008_branch_paired_n64_reuse/results/overlays/branch_paired_n64_v1/moe_dynamic_kernel.py"
    )
    candidate_source = evidence.ROOT.parents[1] / "moe_dynamic_kernel_opt.py"
    source = baseline_source if arm == evidence.BASELINE else candidate_source
    frozen = evidence.overlay_path(results, arm)
    frozen.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, frozen)
    runtime, imports = synthetic_runtime(
        arm,
        frozen,
        f"/jit/validation/{arm}",
        harness_sha256=evidence.file_sha256(frozen_harness),
    )
    cubin = "a" * 64 if arm == evidence.BASELINE else "b" * 64
    summaries = []
    for m, fixture in evidence.CASE_SPECS:
        path = evidence.case_path(results, arm, m, fixture)
        path.parent.mkdir(parents=True, exist_ok=True)
        route_records = []
        for replay in range(2):
            route = route_record()
            route.update(
                {
                    "replay": replay,
                    "json": f"workspace_replay_{replay}.json",
                    "pt": f"workspace_replay_{replay}.pt",
                }
            )
            write_json(path.parent / route["json"], route["summary"])
            (path.parent / route["pt"]).write_bytes(b"synthetic-route-tensor")
            route_records.append(route)
            (path.parent / f"output_{replay}.pt").write_bytes(b"synthetic-output")
        output_records = []
        for replay in range(2):
            output = {
                "replay": replay,
                "event_elapsed_us": 1.0,
                "output_sha256": ("1" if replay == 0 else "2") * 64,
                "reference_sha256": "3" * 64,
                "reference_error": dict(ZERO_ERROR),
                "broad_oracle_diagnostics": {
                    "formal_pass": True,
                    "finite": True,
                },
                "sentinel_nan_remaining": 0,
            }
            if fixture in ("canary_gate_v2", "canary_up_v2"):
                output["write_canary"] = {"gate_pass": True}
            output_records.append(output)
        case = {
            "schema": "exp015.validation-case.v1",
            "status": "complete",
            "arm": arm,
            "m": m,
            "fixture_kind": fixture,
            "runtime_identity_sha256": evidence.canonical_sha256(runtime),
            "case": {
                "experts": 256,
                "hidden": 2048,
                "intermediate_tp": 512,
                "topk": 8,
            },
            "fixture": {"m": m, "fixture": fixture},
            "weights": {"seed": 2026},
            "reference": {
                "dtype": "float32",
                "sha256": "3" * 64,
                "implementation": "/workspace/fixture.py",
                "implementation_sha256": evidence.EXPECTED_ORACLE_SHA256,
            },
            "outputs": output_records,
            "output_stability": dict(ZERO_ERROR),
            "route_task_evidence": route_records,
            "artifact_stages": {"all_retained_cubin_sha256": [cubin]},
            "gate_pass": True,
        }
        write_json(path, case)
        summaries.append(
            {
                "m": m,
                "fixture_kind": fixture,
                "path": str(path.relative_to(results)),
                "sha256": evidence.file_sha256(path),
                "gate_pass": True,
            }
        )
    manifest = {
        "schema": "exp015.arm-validation.v1",
        "status": "complete",
        "arm": arm,
        "runtime": runtime,
        "imports": imports,
        "case_order": [list(spec) for spec in evidence.CASE_SPECS],
        "cases": summaries,
        "jit_artifacts": [{"path": "dump/kernel.cubin", "size": 123, "sha256": cubin}],
        "cubin_sha256": [cubin],
        "gate_pass": True,
    }
    write_json(evidence.validation_path(results, arm), manifest)


class CorrectnessPolicyTest(unittest.TestCase):
    def test_eight_cases_pass_baseline_derived_policy(self) -> None:
        arms = {
            evidence.BASELINE: correctness_arm(),
            evidence.CANDIDATE: correctness_arm(),
        }

        result = evidence.build_correctness(
            Path("/unused"), arms, error_fn=lambda actual, expected: dict(ZERO_ERROR)
        )

        self.assertEqual(result["case_count"], 8)
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["gate_pass"])

    def test_route_parity_failure_rejects_case(self) -> None:
        arms = {
            evidence.BASELINE: correctness_arm(),
            evidence.CANDIDATE: correctness_arm(),
        }
        candidate_case = arms[evidence.CANDIDATE]["cases"][(256, "canonical")]["value"]
        candidate_case["route_task_evidence"] = [
            route_record(task_count=5),
            route_record(task_count=5),
        ]

        result = evidence.build_correctness(
            Path("/unused"), arms, error_fn=lambda actual, expected: dict(ZERO_ERROR)
        )

        self.assertEqual(result["status"], "reject")
        self.assertFalse(
            result["cases"]["m256_canonical"]["route_descriptor_terminal_parity"]
        )


class ValidationArtifactIntegrationTest(unittest.TestCase):
    def test_exact_harness_directory_and_hash_chain_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            for arm in evidence.ARMS:
                write_validation_arm(results, arm)

            frozen_hash = evidence.file_sha256(
                results / "harness/run_exp015_arm_validation_v1.py"
            )
            with mock.patch.object(
                evidence, "EXPECTED_VALIDATION_HARNESS_SHA256", frozen_hash
            ):
                loaded = {
                    arm: evidence.load_arm(
                        results,
                        arm,
                        output_loader=lambda directory, records: [
                            f"{directory}-r0",
                            f"{directory}-r1",
                        ],
                    )
                    for arm in evidence.ARMS
                }

            self.assertEqual(len(loaded[evidence.BASELINE]["cases"]), 8)
            self.assertEqual(
                loaded[evidence.BASELINE]["identity"],
                loaded[evidence.CANDIDATE]["identity"],
            )
            inventory = evidence.distinct_cubin_inventory(loaded)
            self.assertEqual(inventory["count"], 2)
            self.assertEqual(
                {len(record["cases"]) for record in inventory["cubins"]}, {8}
            )


class NoRegressionTest(unittest.TestCase):
    def rows(self, baseline_us: float, candidate_us: float) -> list[dict]:
        rows = []
        for group in evidence.GROUPS:
            for position, arm in enumerate(evidence.ABBA):
                rows.append(
                    {
                        "group": group,
                        "position": position,
                        "arm": arm,
                        "sample_us": baseline_us
                        if arm == evidence.BASELINE
                        else candidate_us,
                    }
                )
        return rows

    def test_group_speedup_and_bootstrap_use_registered_formula(self) -> None:
        summary = evidence.summarize_abba(256, self.rows(100.0, 80.0))
        self.assertAlmostEqual(summary["aggregate_speedup_percent"], 25.0)
        self.assertEqual(summary["verdict"], "pass")
        self.assertEqual(summary["group_bootstrap"]["samples"], 10_000)
        self.assertEqual(summary["group_bootstrap"]["unit"], "one complete ABBA group")

    def test_no_regression_boundary_classification(self) -> None:
        self.assertEqual(evidence.classify_no_regression(-3.0, -2.0), "reject")
        self.assertEqual(evidence.classify_no_regression(-2.0, -1.0), "inconclusive")
        self.assertEqual(evidence.classify_no_regression(-1.5, 0.0), "pass")

    def test_abba_order_drift_fails_closed(self) -> None:
        rows = self.rows(100.0, 100.0)
        rows[0]["arm"] = evidence.CANDIDATE
        with self.assertRaisesRegex(evidence.EvidenceError, "ABBA order drift"):
            evidence.summarize_abba(256, rows)


class OptionalEvidenceTest(unittest.TestCase):
    def static_payload(
        self,
        cubins_by_arm: dict[str, list[str]],
        *,
        registers: int = 160,
    ) -> dict:
        checks = {
            "registers_at_most_160": registers <= 160,
            "stack_zero": True,
            "local_zero": True,
            "ldl_zero": True,
            "stl_zero": True,
            "omma_exactly_448": True,
        }
        arm_pass = all(checks.values())
        arms = {}
        for arm in evidence.ARMS:
            label = "baseline" if arm == evidence.BASELINE else "candidate"
            arms[label] = {
                "label": label,
                "cubin": {"sha256": cubins_by_arm[arm][0]},
                "kernel_symbol": "MoEDynamicKernel",
                "resource": {
                    "registers_per_thread": registers,
                    "stack_bytes_per_thread": 0,
                    "local_bytes_outside_stack": 0,
                },
                "sass": {
                    "selected_instruction_counts": {
                        "ldl": 0,
                        "stl": 0,
                        "omma": 448,
                        "call": 0,
                        "ret": 0,
                    }
                },
                "gates": {"checks": checks, "pass": arm_pass},
                "status": "pass" if arm_pass else "fail",
            }
        comparison_checks = {
            "candidate_adds_no_call": True,
            "candidate_adds_no_ret": True,
        }
        return {
            "schema": evidence.STATIC_SCHEMA,
            "arms": arms,
            "comparison": {"checks": comparison_checks, "pass": True},
            "errors": [],
            "status": "pass" if arm_pass else "fail",
        }

    def dynamic_payload(
        self,
        results: Path,
        baseline_cubin: str,
        candidate_cubin: str,
        *,
        spill_local_read: int = 0,
    ) -> dict:
        records = []
        for arm, cubin in (
            (evidence.BASELINE, baseline_cubin),
            (evidence.CANDIDATE, candidate_cubin),
        ):
            artifacts = {
                "capture_identity": f"ncu/{arm}/capture_identity.json",
                "profile_target": f"ncu/{arm}/profile_target.json",
                "ncu_report": f"ncu/{arm}/trace.ncu-rep",
                "native_raw": f"ncu/{arm}/native_raw.csv",
            }
            for source in artifacts.values():
                artifact = results / source
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(f"synthetic-{arm}-{artifact.name}".encode())
            records.append(
                {
                    "arm": arm,
                    "m": 8192,
                    "fixture": "canonical",
                    "source_sha256": evidence.EXPECTED_SOURCE[arm],
                    "cubin_sha256": cubin,
                    "jit_artifact_set_sha256": (
                        "1" if arm == evidence.BASELINE else "2"
                    )
                    * 64,
                    "gpu_uuid": evidence.EXPECTED_GPU_UUID,
                    "observed_launch": {
                        "grid": evidence.EXPECTED_DYNAMIC_GRID,
                        "block": evidence.EXPECTED_DYNAMIC_BLOCK,
                        "kernel_symbol": "MoEDynamicKernel",
                    },
                    "metrics": {
                        "spill_register_read": 0,
                        "spill_register_write": 0,
                        "spill_local_read": spill_local_read,
                        "spill_local_write": 0,
                        **evidence.EXPECTED_DYNAMIC_WORK,
                    },
                    "source_record": artifacts["native_raw"],
                    "artifacts": artifacts,
                }
            )
        zero_spill = {
            arm: all(
                record["metrics"][metric] == 0
                for metric in evidence.DYNAMIC_SPILL_METRICS
            )
            for arm, record in zip(evidence.ARMS, records, strict=True)
        }
        pairwise = {metric: True for metric in evidence.DYNAMIC_WORK_METRICS}
        ledger = {
            f"{arm}:{metric}": True
            for arm in evidence.ARMS
            for metric in evidence.DYNAMIC_WORK_METRICS
        }
        gate = all(zero_spill.values())
        return {
            "schema": evidence.DYNAMIC_SCHEMA,
            "scope": {
                "m": 8192,
                "fixture": "canonical",
                "gpu_uuid": evidence.EXPECTED_GPU_UUID,
                "grid": evidence.EXPECTED_DYNAMIC_GRID,
                "block": evidence.EXPECTED_DYNAMIC_BLOCK,
                "expected_exp008_work_ledger": evidence.EXPECTED_DYNAMIC_WORK,
            },
            "records": records,
            "checks": {
                "zero_dynamic_spill": zero_spill,
                "pairwise_tensor_work_identity": pairwise,
                "exp008_tensor_work_identity": ledger,
            },
            "status": "pass" if gate else "reject",
            "gate_pass": gate,
        }

    def test_static_resource_pass_and_resource_reject(self) -> None:
        cubins = {
            evidence.BASELINE: ["a" * 64],
            evidence.CANDIDATE: ["b" * 64],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "static.json"
            write_json(path, self.static_payload(cubins))
            passed = evidence.validate_static_resource(path, cubins, root)
            self.assertEqual(passed["status"], "pass")

            write_json(path, self.static_payload(cubins, registers=161))
            rejected = evidence.validate_static_resource(path, cubins, root)
            self.assertEqual(rejected["status"], "reject")

    def test_static_missing_cubin_fails_closed(self) -> None:
        cubins = {
            evidence.BASELINE: ["a" * 64],
            evidence.CANDIDATE: ["b" * 64],
        }
        wrong = {
            evidence.BASELINE: ["a" * 64],
            evidence.CANDIDATE: ["a" * 64],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "static.json"
            write_json(path, self.static_payload(wrong))
            with self.assertRaisesRegex(
                evidence.EvidenceError, "distinct cubin coverage"
            ):
                evidence.validate_static_resource(path, cubins, root)

    def test_dynamic_missing_is_pending_and_complete_zero_spill_passes(self) -> None:
        baseline_cubin = "a" * 64
        candidate_cubin = "b" * 64
        arms = {
            evidence.BASELINE: {
                "cubins": [baseline_cubin],
                "manifest": {"jit_artifact_set_sha256": "1" * 64},
            },
            evidence.CANDIDATE: {
                "cubins": [candidate_cubin],
                "manifest": {"jit_artifact_set_sha256": "2" * 64},
            },
        }
        self.assertEqual(
            evidence.validate_dynamic_ncu(None, arms, Path("/unused"))["status"],
            "pending",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "dynamic.json"
            write_json(
                path,
                self.dynamic_payload(root, baseline_cubin, candidate_cubin),
            )
            result = evidence.validate_dynamic_ncu(path, arms, root)
            self.assertEqual(result["status"], "pass")

            payload = self.dynamic_payload(
                root,
                baseline_cubin,
                candidate_cubin,
                spill_local_read=1,
            )
            write_json(path, payload)
            result = evidence.validate_dynamic_ncu(path, arms, root)
            self.assertEqual(result["status"], "reject")


class BenchmarkArtifactIntegrationTest(unittest.TestCase):
    def prepare_results(self, results: Path) -> dict[str, dict]:
        baseline_source = (
            evidence.ROOT.parent
            / "exp_008_branch_paired_n64_reuse/results/overlays/branch_paired_n64_v1/moe_dynamic_kernel.py"
        )
        candidate_source = evidence.ROOT.parents[1] / "moe_dynamic_kernel_opt.py"
        source_paths = {
            evidence.BASELINE: baseline_source,
            evidence.CANDIDATE: candidate_source,
        }
        arms = {}
        for arm in evidence.ARMS:
            frozen = evidence.overlay_path(results, arm)
            frozen.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_paths[arm], frozen)
            runtime, imports = synthetic_runtime(arm, frozen, f"/jit/validation/{arm}")
            cubin = "a" * 64 if arm == evidence.BASELINE else "b" * 64
            artifacts = [{"path": "dump/kernel.cubin", "size": 123, "sha256": cubin}]
            arms[arm] = {
                "identity": evidence.stable_runtime_identity(
                    runtime,
                    imports,
                    f"test-{arm}",
                    expected_harness_sha256=evidence.EXPECTED_VALIDATION_HARNESS_SHA256,
                ),
                "manifest": {
                    "runtime": runtime,
                    "jit_artifact_set_sha256": evidence.canonical_sha256(artifacts),
                },
                "cases": {
                    (m, "canonical"): {
                        "value": {
                            "fixture": {"m": m, "kind": "canonical"},
                            "weights": {"seed": 2026},
                        }
                    }
                    for m in evidence.M_VALUES
                },
                "cubins": [cubin],
            }
        for m in evidence.M_VALUES:
            for group in evidence.GROUPS:
                for position, arm in enumerate(evidence.ABBA):
                    frozen = evidence.overlay_path(results, arm)
                    runtime, imports = synthetic_runtime(
                        arm,
                        frozen,
                        f"/jit/validation/{arm}",
                        harness_sha256=evidence.EXPECTED_MEASUREMENT_HARNESS_SHA256,
                    )
                    samples = [100.0] * evidence.ITERS
                    payload = {
                        "schema": "exp015.benchmark-position.v1",
                        "status": "complete",
                        "arm": arm,
                        "m": m,
                        "fixture_kind": "canonical",
                        "group": group,
                        "position": position,
                        "abba_order": list(evidence.ABBA),
                        "protocol": {
                            "warmup": evidence.WARMUP,
                            "iters": evidence.ITERS,
                            "l2_flush_bytes": evidence.L2_FLUSH_BYTES,
                            "clock_policy": "locked",
                            "expected_app_clock_mhz": evidence.EXPECTED_APPLICATION_CLOCK_MHZ,
                            "timing": "CUDA Graph external CUDA events; one sample per replay",
                            "process_scope": "one arm/M/group/position in an independent process",
                            "jit_policy": (
                                "reuse one immutable, correctness-validated per-arm JIT root; "
                                "artifact-set and cubin hashes are checked before and after"
                            ),
                        },
                        "samples_us": samples,
                        "statistics_us": {
                            "count": evidence.ITERS,
                            "mean": 100.0,
                            "median": 100.0,
                            "p10": 100.0,
                            "p90": 100.0,
                            "min": 100.0,
                            "max": 100.0,
                        },
                        "sample_us": 100.0,
                        "fixture": arms[arm]["cases"][(m, "canonical")]["value"][
                            "fixture"
                        ],
                        "weights": arms[arm]["cases"][(m, "canonical")]["value"][
                            "weights"
                        ],
                        "output_sha256": "f" * 64,
                        "runtime": runtime,
                        "imports": imports,
                        "gpu_after": runtime["gpu"],
                        "jit_artifacts": [
                            {
                                "path": "dump/kernel.cubin",
                                "size": 123,
                                "sha256": arms[arm]["cubins"][0],
                            }
                        ],
                        "cubin_sha256": arms[arm]["cubins"],
                        "compile_identity": {"compiled_max_active_clusters": [110]},
                    }
                    payload["jit_artifact_set_sha256"] = evidence.canonical_sha256(
                        payload["jit_artifacts"]
                    )
                    write_json(
                        evidence.benchmark_path(results, arm, m, group, position),
                        payload,
                    )
        return arms

    def test_complete_two_shape_abba_artifact_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            arms = self.prepare_results(results)
            output = evidence.build_performance(results, arms)
            self.assertEqual(output["position_count"], 40)
            self.assertEqual(output["status"], "pass")
            self.assertEqual(output["cases"]["m256"]["verdict"], "pass")


class FinalVerdictTest(unittest.TestCase):
    def test_missing_dynamic_forces_pending(self) -> None:
        components = {
            "correctness": {"status": "pass"},
            "performance": {"status": "pass"},
            "static": {"status": "pass"},
            "dynamic": {"status": "pending"},
        }
        self.assertEqual(evidence.final_verdict(components), "pending")

    def test_reject_has_priority_over_pending(self) -> None:
        components = {
            "correctness": {"status": "reject"},
            "static": {"status": "pending"},
        }
        self.assertEqual(evidence.final_verdict(components), "reject")


if __name__ == "__main__":
    unittest.main()
