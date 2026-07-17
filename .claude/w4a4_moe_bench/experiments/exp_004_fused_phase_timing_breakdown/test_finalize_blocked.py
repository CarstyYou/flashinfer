from __future__ import annotations

import copy

import pytest

from exp004_common import MEASUREMENT_CONTROL, NORMAL, PROBE, read_json, write_json
from finalize_blocked import BLOCKED_STATUS, finalize
from run_exp004 import refresh_manifest


def _arm(
    digit: str,
    *,
    stack: int = 488,
    stl: int = 68,
    ldl: int = 122,
    annotations: int = 190,
):
    return {
        "cubin_sha256": digit * 64,
        "sass_sha256": chr(ord(digit) + 1) * 64,
        "resource": {
            "registers_per_thread": 255,
            "stack_bytes_per_thread": stack,
            "static_shared_bytes_per_cta": 1024,
            "static_local_bytes_outside_stack": 0,
        },
        "local_sass": {
            "stl_instruction_count": stl,
            "ldl_instruction_count": ldl,
            "spill_annotation_count": annotations,
        },
        "semantic_projection": {
            "omma": 896,
            "utmaldg": 40,
            "ldsm": 200,
            "bar": 34,
            "atomg": 9,
            "redg": 4,
            "ldg": 53,
        },
        "artifact_hashes": {
            "cubin": digit * 64,
            "ptx": chr(ord(digit) + 2) * 64,
            "mlir": chr(ord(digit) + 3) * 64,
        },
    }


def _blocked_preflight():
    return {
        "schema": "exp004.blocked-preflight.v1",
        "identity_gates": {"source": True, "hardware": True, "toolchain": True},
        "evidence_identity": {
            "source": {
                "flashinfer_commit": "d" * 40,
                "cutlass_commit": "e" * 40,
                "kernel_sha256": "a" * 64,
                "dispatch_sha256": "b" * 64,
                "wrapper_sha256": "c" * 64,
            },
            "gpu": {
                "uuid": "GPU-test",
                "name": "NVIDIA Graphics Device",
                "pci_bus_id": "00000000:76:00.0",
                "compute_capability": [12, 0],
                "sm_count": 110,
                "driver": "580.95.05",
            },
            "toolchain": {
                "nvcc": "CUDA 13.2.78",
                "ptxas": "CUDA 13.2.78",
                "python": "3.12",
                "torch": "2.12",
                "cuda_runtime": "13.0",
                "image_digest": "sha256:image",
                "python_deps_sha256": "deps",
                "cutlass_dsl_module": "/workspace/deps/cutlass/__init__.py",
                "cutlass_dsl_version": "4.6.0",
            },
        },
        "runtime_observations": {
            arm: {
                "applications_graphics_clock_mhz": "2377",
                "graphics_clock_mhz": "2370",
                "max_graphics_clock_mhz": "3090",
                "power_draw_w": "60.0",
                "lease_id": "lease-test",
                "jit_artifact_set_sha256": digit * 64,
            }
            for arm, digit in (
                (NORMAL, "4"),
                (MEASUREMENT_CONTROL, "5"),
                (PROBE, "6"),
            )
        },
        "arms": {
            NORMAL: _arm("1"),
            MEASUREMENT_CONTROL: _arm("2"),
            PROBE: _arm("3", stack=456, stl=64, ldl=132, annotations=196),
        },
        "event_contract": {
            "expected_tick_writes": 776_016,
            "observed_tick_writes": 0,
            "expected_task_cta_writes": 2_536,
            "observed_task_cta_writes": 0,
        },
        "probe_lowering": {
            "ptx_clock64_count": 28,
            "ptx_probe_store_count": 37,
        },
        "probe_preparation_gates": {
            "reference_correctness": False,
            "output_contract": True,
            "workspace_contract": True,
            "reference_metrics": {
                "cosine": 0.997062,
                "relative_l2": 0.076544,
                "max_abs": 0.869024,
                "token_rel_l2_p99": 0.305275,
            },
        },
    }


def _iket_preflight(*, ready: bool = False):
    return {
        "schema": "exp004.iket-fallback-preflight.v1",
        "status": (
            "provider_ready_but_not_admitted"
            if ready
            else "blocked_provider_unavailable_or_version_drift"
        ),
        "provider": {
            "requested": "run-iket",
            "observed_version": "0.7.10" if ready else None,
            "required_audited_version": "0.7.10",
            "ready": ready,
        },
    }


def _write_inputs(results, *, blocked=None, iket=None):
    blocked_path = results / "raw" / "blocked_preflight.json"
    iket_path = results / "iket_fallback_preflight.json"
    overlay_path = results / "overlays" / "identity.json"
    write_json(blocked_path, blocked or _blocked_preflight())
    write_json(iket_path, iket or _iket_preflight())
    write_json(overlay_path, {"schema": "exp004.overlays.v1"})
    return blocked_path, iket_path


def test_finalize_blocked_writes_compact_closed_artifacts(tmp_path):
    results = tmp_path / "results"
    blocked, iket = _write_inputs(results)
    manifest = finalize(
        results=results,
        blocked_preflight_path=blocked,
        iket_preflight_path=iket,
    )

    gate = read_json(results / "derived" / "blocked_gate.json")
    report = (results / "result.md").read_text()
    assert manifest["status"] == BLOCKED_STATUS
    assert gate["closure_gate_pass"]
    assert not gate["formal_gate_pass"]
    assert not gate["diagnostic_share_allowed"]
    assert gate["stop_reasons"] == [
        "probe_resource_spill_identity_drift",
        "probe_reference_correctness_failed",
        "probe_event_contract_incomplete",
    ]
    assert gate["probe_lowering"]["runtime_zero_write_cause"] == "unresolved"
    assert "measurement perturbation prevented formal timing" in report
    assert "MMA consumer phase share" not in report
    assert "具体原因尚未定位" in report
    assert "Probe reference correctness" in report
    assert not manifest["result"]["phase_share_published"]
    assert manifest["phase_captures"] == {}
    assert manifest["profiles"] == {}
    assert manifest["overlay_identity"]["path"] == "overlays/identity.json"
    assert len(manifest["overlay_identity"]["sha256"]) == 64
    assert manifest["evidence_identity"]["gpu"]["driver"] == "580.95.05"


def test_refresh_manifest_preserves_blocked_closure(tmp_path):
    results = tmp_path / "results"
    blocked, iket = _write_inputs(results)
    finalize(
        results=results,
        blocked_preflight_path=blocked,
        iket_preflight_path=iket,
    )
    before = read_json(results / "manifest.json")
    refresh_manifest(results)
    after = read_json(results / "manifest.json")
    assert after == before
    assert after["status"] == BLOCKED_STATUS


def test_finalize_refuses_available_iket_fallback(tmp_path):
    results = tmp_path / "results"
    blocked, iket = _write_inputs(results, iket=_iket_preflight(ready=True))
    with pytest.raises(ValueError, match="IKET fallback is available"):
        finalize(
            results=results,
            blocked_preflight_path=blocked,
            iket_preflight_path=iket,
        )
    assert not (results / "derived" / "blocked_gate.json").exists()


def test_finalize_refuses_when_primary_probe_passed(tmp_path):
    results = tmp_path / "results"
    value = copy.deepcopy(_blocked_preflight())
    value["arms"][PROBE] = _arm("3")
    value["event_contract"]["observed_tick_writes"] = 776_016
    value["event_contract"]["observed_task_cta_writes"] = 2_536
    value["probe_preparation_gates"]["reference_correctness"] = True
    blocked, iket = _write_inputs(results, blocked=value)
    with pytest.raises(ValueError, match="no primary measurement gate failed"):
        finalize(
            results=results,
            blocked_preflight_path=blocked,
            iket_preflight_path=iket,
        )


def test_finalize_refuses_control_identity_drift(tmp_path):
    results = tmp_path / "results"
    value = copy.deepcopy(_blocked_preflight())
    value["arms"][MEASUREMENT_CONTROL]["resource"]["stack_bytes_per_thread"] = 480
    blocked, iket = _write_inputs(results, blocked=value)
    with pytest.raises(ValueError, match="normal/measurement control identity"):
        finalize(
            results=results,
            blocked_preflight_path=blocked,
            iket_preflight_path=iket,
        )


def test_finalize_refuses_stale_phase_artifacts(tmp_path):
    results = tmp_path / "results"
    blocked, iket = _write_inputs(results)
    stale = results / "derived" / "mma_phase_share.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("phase,share_pct\nfc1_gate,1\n")
    with pytest.raises(RuntimeError, match="refusing to mix blocked closure"):
        finalize(
            results=results,
            blocked_preflight_path=blocked,
            iket_preflight_path=iket,
        )
