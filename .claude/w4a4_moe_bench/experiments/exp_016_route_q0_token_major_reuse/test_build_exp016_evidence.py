"""CPU-only tests for the compact exp_016 evidence builder."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_exp016_evidence as target  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fixture_identity(m: int, fixture: str, scale_kind: str) -> dict[str, object]:
    return {"m": m, "fixture": fixture, "seed": 2026, "scale_kind": scale_kind}


def weight_identity(scale_kind: str) -> dict[str, object]:
    return {"seed": 2026, "input_scale_kind": scale_kind, "shape": [256]}


def install_overlay_identity(
    results: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, str]:
    contents = {
        target.BASELINE: b"# synthetic baseline\n",
        target.CANDIDATE: b"# synthetic candidate\n",
    }
    hashes = {}
    arms = {}
    for arm, content in contents.items():
        path = results / "overlays" / arm / "moe_dynamic_kernel.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        hashes[arm] = target.file_sha256(path)
        arms[arm] = {"path": str(path), "sha256": hashes[arm]}
    monkeypatch.setattr(target, "EXPECTED_OVERLAY_SHA256", hashes)
    write_json(
        results / "overlays" / "identity.json",
        {
            "schema": "exp016.route-q0-overlay.v1",
            "source_sha256": hashes[target.BASELINE],
            "source_role": target.BASELINE,
            "arms": arms,
            "mechanism_gate": {"gate_pass": True},
            "work_ledger_m8192_topk8_h2048": {
                "bf16_block_loads_baseline": 8_388_608,
                "bf16_block_loads_candidate": 1_048_576,
                "productive_claims_baseline": 3_641,
                "productive_claims_candidate": 911,
                "row_allocation_atomics_baseline": 65_536,
                "row_allocation_atomics_candidate": 65_536,
                "packed_fp4_stores_baseline": 8_388_608,
                "packed_fp4_stores_candidate": 8_388_608,
                "sfa_stores_baseline": 8_388_608,
                "sfa_stores_candidate": 8_388_608,
            },
        },
    )
    return hashes


def runtime(arm: str, hashes: dict[str, str]) -> dict[str, object]:
    overlay = f"/remote/{arm}/moe_dynamic_kernel.py"
    return {
        "hostname": "synthetic-5kp",
        "python": "3.12",
        "packages": {"nvidia-cutlass-dsl": "4.6.0"},
        "torch": "2.12",
        "cuda_runtime": "13.2",
        "nvcc": "Cuda compilation tools, release 13.2, V13.2.78",
        "ptxas": "ptxas release 13.2, V13.2.78",
        "cuda_visible_devices": "0",
        "image_digest": "sha256:image",
        "image_id": "sha256:image-id",
        "python_deps_sha256": "d" * 64,
        "lease_id": "synthetic-lease",
        "jit_root": f"/jit/{arm}",
        "source": {
            "locked_flashinfer_commit": "1" * 40,
            "checkout_head": "1" * 40,
            "cutlass_commit": "2" * 40,
            "production_kernel_sha256": "3" * 64,
            "overlay": overlay,
            "overlay_sha256": hashes[arm],
            "oracle_source_sha256": "4" * 64,
        },
        "gpu": {
            "uuid": "GPU-synthetic",
            "name": "NVIDIA Graphics Device",
            "pci_bus_id": "0000:01:00.0",
            "driver": "580.0",
            "graphics_clock_mhz": "2377",
            "applications_graphics_clock_mhz": "2377",
            "max_graphics_clock_mhz": "3090",
            "power_draw_w": "100",
            "compute_capability": [12, 0],
            "sm_count": 110,
            "foreign_processes_before_cuda_context": [],
        },
        "imports": {
            "flashinfer": "/remote/flashinfer/__init__.py",
            "target_module": overlay,
            "cutlass_python": "/remote/cutlass/__init__.py",
            "cutlass_python_version": "4.6.0",
            "flashinfer_jit_workspace": f"/jit/{arm}/workspace",
        },
        "harness": {"path": "/remote/run_exp016_arm.py", "sha256": "5" * 64},
    }


def artifacts(arm: str) -> tuple[list[dict[str, object]], str, str]:
    cubin = ("a" if arm == target.BASELINE else "b") * 64
    records = [{"path": "dump/kernel.cubin", "size": 123, "sha256": cubin}]
    return records, target.canonical_sha256(records), cubin


def logical_digest(label: str) -> dict[str, object]:
    values = {}
    for index, field in enumerate(
        ("route_metadata_sha256", "packed_fp4_sha256", "sfa_sha256"), start=1
    ):
        values[field] = target.canonical_sha256([label, index])
    values["combined_sha256"] = target.canonical_sha256(values)
    return values


def validation_payload(
    arm: str,
    hashes: dict[str, str],
    m: int,
    fixture: str,
    scale_kind: str,
    *,
    digest_label: str | None = None,
) -> dict[str, object]:
    jit, jit_hash, cubin = artifacts(arm)
    digest = logical_digest(digest_label or f"{m}/{fixture}/{scale_kind}")
    replay_payload = {
        "schema": "exp016.canonical-route-q0-payload.v1",
        "logical_routes": m * 8,
        "gate_pass": True,
        **digest,
    }
    case = {
        "fixture": fixture_identity(m, fixture, scale_kind),
        "weights": weight_identity(scale_kind),
        "oracle_weights": weight_identity(scale_kind),
        "reference_sha256": target.canonical_sha256([m, fixture, scale_kind]),
    }
    return {
        "schema": target.VALIDATION_SCHEMA,
        "status": "complete",
        "gate_pass": True,
        "arm": arm,
        "m": m,
        "fixture": fixture,
        "scale_kind": scale_kind,
        "runtime": runtime(arm, hashes),
        "specialization": {"gate_pass": True},
        "case_identity": case,
        "replays": [
            {
                "replay": replay,
                "gate_pass": True,
                "output_sha256": str(replay + 6) * 64,
                "logical_payload": dict(replay_payload),
            }
            for replay in range(2)
        ],
        "logical_payload_replay_stable": True,
        "output_stability_gate": {"gate_pass": True},
        "jit_artifacts": jit,
        "jit_artifact_set_sha256": jit_hash,
        "cubin_sha256": [cubin],
    }


def write_validations(results: Path, hashes: dict[str, str]) -> None:
    for arm in target.ARMS:
        for m, fixture, scale_kind in target.VALIDATION_CASES:
            write_json(
                results
                / "raw"
                / "validation"
                / arm
                / f"m{m}_{fixture}_{scale_kind}.json",
                validation_payload(arm, hashes, m, fixture, scale_kind),
            )


def benchmark_payload(
    arm: str,
    hashes: dict[str, str],
    m: int,
    group: int,
    position: int,
    sample_us: float,
    *,
    samples: list[float] | None = None,
) -> dict[str, object]:
    jit, jit_hash, cubin = artifacts(arm)
    samples = samples or [sample_us] * target.ITERS
    mean = sum(samples) / len(samples)
    median = __import__("statistics").median(samples)
    cv = __import__("statistics").pstdev(samples) / mean
    rt = runtime(arm, hashes)
    return {
        "schema": target.BENCHMARK_SCHEMA,
        "status": "complete",
        "arm": arm,
        "m": m,
        "fixture": "canonical",
        "scale_kind": "unequal",
        "group": group,
        "position": position,
        "abba_order": list(target.ABBA),
        "protocol": {
            "warmup": target.WARMUP,
            "iters": target.ITERS,
            "l2_flush_bytes": target.L2_FLUSH_BYTES,
        },
        "samples_us": samples,
        "statistics_us": {
            "count": len(samples),
            "mean": mean,
            "median": median,
            "cv": cv,
        },
        "fixture_identity": fixture_identity(m, "canonical", "unequal"),
        "weight_identity": weight_identity("unequal"),
        "output_sha256": "8" * 64,
        "runtime": rt,
        "specialization": {"gate_pass": True},
        "gpu_after": {
            "uuid": rt["gpu"]["uuid"],
            "applications_graphics_clock_mhz": rt["gpu"][
                "applications_graphics_clock_mhz"
            ],
        },
        "jit_artifacts": jit,
        "jit_artifact_set_sha256": jit_hash,
        "cubin_sha256": [cubin],
    }


def write_benchmarks(
    results: Path,
    hashes: dict[str, str],
    m_values: tuple[int, ...],
    *,
    candidate_us: float = 90.0,
) -> None:
    for m in m_values:
        for group in target.GROUPS:
            for position, arm in enumerate(target.ABBA):
                sample = 100.0 if arm == target.BASELINE else candidate_us
                write_json(
                    results
                    / "raw"
                    / "benchmark"
                    / f"m{m}"
                    / f"g{group}_p{position}_{arm}.json",
                    benchmark_payload(arm, hashes, m, group, position, sample),
                )


def write_supporting_gates(results: Path, hashes: dict[str, str]) -> None:
    baseline_runtime = runtime(target.BASELINE, hashes)
    gpu = baseline_runtime["gpu"]
    environment = {
        field: baseline_runtime[field]
        for field in (
            "python",
            "packages",
            "torch",
            "cuda_runtime",
            "nvcc",
            "ptxas",
            "image_digest",
            "image_id",
            "python_deps_sha256",
        )
    }
    environment["gpu"] = {
        field: gpu[field]
        for field in (
            "uuid",
            "name",
            "pci_bus_id",
            "driver",
            "applications_graphics_clock_mhz",
            "max_graphics_clock_mhz",
            "compute_capability",
            "sm_count",
        )
    }
    environment["imports"] = {
        field: baseline_runtime["imports"][field]
        for field in ("flashinfer", "cutlass_python", "cutlass_python_version")
    }
    environment["harness"] = {
        "run_exp016_arm_sha256": baseline_runtime["harness"]["sha256"],
        "sha256": "9" * 64,
    }

    zero_counts = {
        "spill_refill_annotation_count": 0,
        "spill_refill_annotation_unique_pc_count": 0,
        "ldl_opcode_count": 0,
        "stl_opcode_count": 0,
        "local_sass_opcode_count": 0,
        "annotation_pcs_equal_local_sass_pcs": True,
    }
    resource = {
        "registers_per_thread": 128,
        "stack_bytes_per_thread": 0,
        "static_local_bytes_outside_stack": 0,
        "static_shared_bytes_per_cta": 1024,
    }
    instrumentation = {}
    captures = {}
    for arm in target.ARMS:
        instrumentation[arm] = {
            "control_resource": dict(resource),
            "probe_resource": dict(resource),
            "resource_identity_equal": True,
            "control_sass_spill": {
                "counts": dict(zero_counts),
                "gate_pass": True,
                "cubin_sha256": "c" * 64,
                "sha256": "d" * 64,
            },
            "probe_sass_spill": {
                "counts": dict(zero_counts),
                "gate_pass": True,
                "cubin_sha256": "e" * 64,
                "sha256": "f" * 64,
            },
            "sass_spill_gate_pass": True,
            "probe_e2e_perturbation_percent": 0.1,
            "max_abs_allowed_perturbation_percent": 2.0,
            "e2e_perturbation_small": True,
            "gate_pass": True,
        }
        captures[arm] = {
            mode: {
                "capture_gate_pass": True,
                "base_source": {"kernel_sha256": hashes[arm]},
            }
            for mode in ("control_no_marker", "probe")
        }
    write_json(
        results / "p3_phase_evidence.json",
        {
            "schema": "exp016.p3-phase-evidence.v1",
            "status": "Complete",
            "gate_pass": True,
            "capture_integrity_gate_pass": True,
            "instrumentation_gate_pass": True,
            "sass_spill_gate_pass": True,
            "phase_improvement_gate_pass": True,
            "evidence_classification": "diagnostic",
            "performance_authority": "uninstrumented exp016 E2E only",
            "additive_sm_estimate_used": False,
            "environment": environment,
            "phase_comparison": {
                "baseline_grid_critical_wall_median_us": 200.0,
                "candidate_grid_critical_wall_median_us": 100.0,
                "candidate_minus_baseline_us": -100.0,
                "latency_reduction_percent": 50.0,
                "candidate_faster": True,
                "all_candidate_samples_faster": True,
                "interpretation_legal_as_diagnostic": True,
                "interpretation_legal_as_production_phase_truth": False,
            },
            "instrumentation": instrumentation,
            "captures": captures,
        },
    )

    _, artifact_hash, cubin = artifacts(target.CANDIDATE)
    metrics = {
        "spill_register_read_instructions": 0,
        "spill_register_write_instructions": 0,
        "spill_local_load_bytes": 0,
        "spill_local_store_bytes": 0,
    }
    write_json(
        results / "dynamic_spill_evidence.json",
        {
            "schema": "exp016.dynamic-spill-evidence.v1",
            "status": "pass",
            "gate_pass": True,
            "candidate": {
                "arm": target.CANDIDATE,
                "m": 8192,
                "fixture": "canonical",
                "scale_kind": "unequal",
                "source_sha256": hashes[target.CANDIDATE],
                "cubin_sha256": cubin,
                "jit_artifact_set_sha256": artifact_hash,
                "gpu_uuid": gpu["uuid"],
                "metrics": metrics,
                "observed_launch": {
                    "grid": [1, 1, 110],
                    "block": [288, 1, 1],
                },
            },
            "checks": {
                "zero_dynamic_spill": True,
                "dynamic_local_load_store_bytes": 0,
            },
        },
    )


def test_empty_results_are_explicitly_unresolved(tmp_path: Path) -> None:
    evidence, _, rows = target.build(tmp_path)

    assert evidence["verdict"] == "Unresolved"
    assert evidence["status"] == "incomplete"
    assert evidence["hard_failures"] == []
    assert evidence["missing_evidence"]
    assert rows == []


def test_failed_validation_capture_is_ignored_when_v2_is_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashes = install_overlay_identity(tmp_path, monkeypatch)
    write_validations(tmp_path, hashes)
    failed = validation_payload(target.BASELINE, hashes, 256, "canonical", "unequal")
    failed["status"] = "failed"
    failed["gate_pass"] = False
    failed["replays"] = []
    write_json(tmp_path / "raw/validation/baseline_pair_major/old_failed.json", failed)

    evidence, manifest, _ = target.build(tmp_path)

    assert evidence["correctness"]["status"] == "pass"
    assert evidence["hard_failures"] == []
    assert evidence["ignored_failed_captures"] == [
        "raw/validation/baseline_pair_major/old_failed.json"
    ]
    ignored = [
        row for row in manifest["raw_inventory"] if row["kind"].startswith("ignored")
    ]
    assert ignored[0]["reason"]


def test_m8192_quick_gate_is_scoped_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashes = install_overlay_identity(tmp_path, monkeypatch)
    write_validations(tmp_path, hashes)
    write_benchmarks(tmp_path, hashes, (8192,))

    evidence, _, rows = target.build(tmp_path)

    assert evidence["verdict"] == "Unresolved"
    assert evidence["performance"]["m8192_quick_gate_pass"] is True
    assert evidence["performance"]["full_sweep_gate_pass"] is None
    assert len(rows) == 12
    group = next(
        row
        for row in evidence["performance"]["groups"]
        if row["m"] == 8192 and row["group"] == 0
    )
    assert group["baseline_median_us"] == 100.0
    assert group["candidate_median_us"] == 90.0
    assert group["improvement_percent"] == pytest.approx(11.1111111111)


def test_complete_sweep_accepts_and_digest_mismatch_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashes = install_overlay_identity(tmp_path, monkeypatch)
    write_validations(tmp_path, hashes)
    write_benchmarks(tmp_path, hashes, target.M_VALUES, candidate_us=97.0)

    evidence, _, rows = target.build(tmp_path)
    assert evidence["verdict"] == "Unresolved"
    assert evidence["validation_e2e_scope"]["verdict"] == "Accept"
    assert evidence["correctness"]["status"] == "pass"
    assert evidence["performance"]["status"] == "pass"
    assert len(rows) == len(target.M_VALUES) * len(target.GROUPS) * 4

    write_supporting_gates(tmp_path, hashes)
    accepted, _, _ = target.build(tmp_path)
    assert accepted["verdict"] == "Accept"

    path = (
        tmp_path
        / "raw/validation/candidate_token_major_reuse/m8192_canonical_unequal.json"
    )
    value = json.loads(path.read_text())
    value["replays"][0]["logical_payload"]["packed_fp4_sha256"] = "f" * 64
    value["replays"][1]["logical_payload"]["packed_fp4_sha256"] = "f" * 64
    write_json(path, value)

    rejected, _, _ = target.build(tmp_path)
    assert rejected["verdict"] == "Reject"
    assert any("logical FP4/SFA/metadata" in item for item in rejected["hard_failures"])


def test_high_cv_is_unresolved_not_a_false_reject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashes = install_overlay_identity(tmp_path, monkeypatch)
    write_validations(tmp_path, hashes)
    write_benchmarks(tmp_path, hashes, (8192,))
    path = tmp_path / "raw/benchmark/m8192/g0_p1_candidate_token_major_reuse.json"
    noisy = [80.0, 100.0] * 25
    value = benchmark_payload(target.CANDIDATE, hashes, 8192, 0, 1, 90.0, samples=noisy)
    write_json(path, value)

    evidence, _, _ = target.build(tmp_path)

    assert evidence["verdict"] == "Unresolved"
    assert evidence["hard_failures"] == []
    assert evidence["performance"]["m8192_quick_gate_pass"] is None
    assert any("CV exceeds" in item for item in evidence["missing_evidence"])


@pytest.mark.parametrize("drift", ("sass", "smem"))
def test_p3_sass_and_smem_drift_cannot_accept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    hashes = install_overlay_identity(tmp_path, monkeypatch)
    write_validations(tmp_path, hashes)
    write_benchmarks(tmp_path, hashes, target.M_VALUES, candidate_us=97.0)
    write_supporting_gates(tmp_path, hashes)
    path = tmp_path / "p3_phase_evidence.json"
    value = json.loads(path.read_text())
    candidate = value["instrumentation"][target.CANDIDATE]
    if drift == "sass":
        candidate["probe_sass_spill"]["counts"]["ldl_opcode_count"] = 1
    else:
        candidate["probe_resource"]["static_shared_bytes_per_cta"] = 2048
    write_json(path, value)

    evidence, _, _ = target.build(tmp_path)

    assert evidence["validation_e2e_scope"]["verdict"] == "Accept"
    assert evidence["verdict"] == "Unresolved"
    assert evidence["supporting_gates"]["p3_phase"]["gate_pass"] is False
    assert evidence["supporting_gates"]["p3_phase"]["reason"]


def test_stable_m8192_regression_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashes = install_overlay_identity(tmp_path, monkeypatch)
    write_validations(tmp_path, hashes)
    write_benchmarks(tmp_path, hashes, (8192,), candidate_us=101.0)

    evidence, _, _ = target.build(tmp_path)

    assert evidence["verdict"] == "Reject"
    assert any("positive 2% gate" in item for item in evidence["hard_failures"])
