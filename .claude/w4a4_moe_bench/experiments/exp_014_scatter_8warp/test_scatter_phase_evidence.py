#!/usr/bin/env python3
"""CPU-only tests for compact exp_014 Scatter phase evidence."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import build_scatter_phase_evidence as builder
import build_scatter_phase_probe as probe_builder
from exp014_scatter_probe_common import (
    ARMS,
    BASELINE,
    CANDIDATE,
    EVENT_ABI,
    canonical_sha256,
    read_json,
    write_json,
)


def digest(number: int) -> str:
    return f"{number:064x}"


def summary(body: float, including_sync: float) -> dict:
    def stats(value: float) -> dict:
        return {
            "mean": value,
            "median": value,
            "p10": value - 1.0,
            "p90": value + 1.0,
            "min": value - 2.0,
            "max": value + 2.0,
        }

    return {
        "samples": builder.TILE_SAMPLES_PER_REPLAY,
        "body_ns": stats(body),
        "including_sync_ns": stats(including_sync),
    }


def passing_gate() -> dict:
    return {"gate_pass": True}


class ScatterPhaseEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.results = Path(self.temporary.name) / "results"
        probe_builder.build(
            Path(__file__).resolve().parents[4],
            self.results / "scatter_phase_probe_overlays",
        )
        self.root_identity = read_json(
            self.results / "scatter_phase_probe_overlays/identity.json"
        )
        self.cubins = {}
        for index, arm in enumerate(ARMS):
            body_base = 100.0 if arm == BASELINE else 80.0
            sync_base = 120.0 if arm == BASELINE else 100.0
            e2e_base = 200.0 if arm == BASELINE else 190.0
            cubin = digest(100 + index)
            self.cubins[arm] = cubin
            self.write_capture(
                arm,
                body_base=body_base,
                sync_base=sync_base,
                e2e_base=e2e_base,
                cubin=cubin,
            )
        self.write_ownership_gate()
        self.write_static_resource_evidence()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def capture_file(self, arm: str) -> Path:
        return builder.capture_path(self.results, arm)

    def write_capture(
        self,
        arm: str,
        *,
        body_base: float,
        sync_base: float,
        e2e_base: float,
        cubin: str,
    ) -> None:
        arm_identity = self.root_identity["arms"][arm]
        source = {
            "arm": arm,
            "kernel_overlay": f"/probe/{arm}/moe_dynamic_kernel.py",
            "kernel_sha256": arm_identity["overlay"]["kernel_sha256"],
            "dispatch_overlay": f"/probe/{arm}/moe_dispatch.py",
            "dispatch_sha256": arm_identity["overlay"]["dispatch_sha256"],
            "overlay_identity": arm_identity,
            "event_abi": EVENT_ABI,
        }
        runtime = {
            "gpu": {
                "uuid": "GPU-test",
                "name": "RTX PRO 6000 Blackwell Server Edition",
                "compute_capability": [12, 0],
                "sm_count": 110,
                "driver": "test-driver",
                "applications_graphics_clock_mhz": "2377",
            },
            "cuda_runtime": "13.0",
            "torch": "test-torch",
            "nvcc": "test-nvcc",
            "ptxas": "test-ptxas",
            "image_digest": "sha256:test-image",
            "image_id": "sha256:test-id",
            "python_deps_sha256": digest(200),
            "packages": {"cutlass-dsl": "test"},
            "source": source,
        }
        runs = []
        for replay in range(builder.REPLAYS):
            runs.append(
                {
                    "replay": replay,
                    "event_elapsed_us": e2e_base + replay,
                    "output_sha256": digest(300 + replay),
                    "ticks_sha256": digest(400 + replay),
                    "correctness_gate": passing_gate(),
                    "route_task_gate": passing_gate(),
                    "buffer_gate": passing_gate(),
                    "interval_summary": summary(body_base + replay, sync_base + replay),
                    "gate_pass": True,
                }
            )
        elapsed = [run["event_elapsed_us"] for run in runs]
        artifacts = [{"path": "kernel.cubin", "size": 1, "sha256": cubin}]
        capture = {
            "schema": "exp014.scatter-phase-probe-capture.v1",
            "classification": "diagnostic-only",
            "arm": arm,
            "m": builder.M,
            "fixture_kind": "canonical",
            "event_abi": EVENT_ABI,
            "source": source,
            "overlay_gate": {
                "gate_pass": True,
                "errors": [],
                "root_identity": self.root_identity,
                "arm_identity": arm_identity,
            },
            "runtime": runtime,
            "fixture": {"seed": 2026, "case": "canonical"},
            "weights": {"seed": 2026, "case": "canonical"},
            "reference_sha256": digest(500),
            "eager": {
                "correctness_gate": passing_gate(),
                "route_task_gate": passing_gate(),
                "buffer_gate": passing_gate(),
                "interval_summary": summary(body_base, sync_base),
            },
            "runs": runs,
            "probe_e2e_us": {
                "samples": builder.REPLAYS,
                "median": elapsed[2],
                "min": elapsed[0],
                "max": elapsed[-1],
            },
            "jit_artifacts": artifacts,
            "jit_artifact_set_sha256": canonical_sha256(artifacts),
            "cubin_sha256": [cubin],
            "compile_identity": {"compiled_max_active_clusters": [110]},
        }
        write_json(self.capture_file(arm), capture)

    def write_ownership_gate(self) -> None:
        cases = []
        for valid_rows in (1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128):
            active_warps = list(range(8)) if valid_rows == 128 else [0]
            for output_tile_idx in (0, 1, 7, 63):
                cases.append(
                    {
                        "valid_rows": valid_rows,
                        "output_tile_idx": output_tile_idx,
                        "active_warps": active_warps,
                        "exactly_one_owner": True,
                        "invalid_writes": 0,
                        "observed_elements": valid_rows * 128,
                        "expected_elements": valid_rows * 128,
                    }
                )
        write_json(
            self.results / "ownership_gate.json",
            {
                "schema": "exp014.scatter-ownership-gate.v1",
                "status": "pass",
                "mapping": "warp_m=(warp>>1)*32, warp_n=(warp&1)*64",
                "vector_width": 8,
                "cases": cases,
            },
        )

    def write_static_resource_evidence(self) -> None:
        def arm(label: str, cubin: str, registers: int) -> dict:
            failed = [] if registers <= 160 else ["registers_at_most_160"]
            return {
                "label": label,
                "status": "pass" if not failed else "fail",
                "cubin": {"path": f"/remote/{label}/kernel.cubin", "sha256": cubin},
                "resource": {
                    "registers_per_thread": registers,
                    "stack_bytes_per_thread": 0,
                    "local_bytes_outside_stack": 0,
                },
                "sass": {"selected_instruction_counts": {"ldl": 0, "stl": 0}},
                "gates": {"failed": failed, "pass": not failed},
            }

        write_json(
            self.results / builder.STATIC_EVIDENCE_NAME,
            {
                "schema": "exp015.static_resource_evidence.v1",
                "status": "fail",
                "errors": [
                    "candidate: required static gates failed: registers_at_most_160"
                ],
                "arms": {
                    "baseline": arm("baseline", self.cubins[BASELINE], 160),
                    "candidate": arm("candidate", self.cubins[CANDIDATE], 165),
                },
            },
        )

    def test_builds_compact_replay_and_aggregate_evidence(self) -> None:
        evidence = builder.build(self.results)
        self.assertTrue(evidence["gate_pass"])
        self.assertEqual(evidence["case"]["m"], 8192)
        body = evidence["comparison"]["body_ns"]
        self.assertEqual(len(body["per_replay_index"]), builder.REPLAYS)
        self.assertEqual(
            evidence["arms"][BASELINE]["body_ns"]["aggregate"]["value_ns"],
            102.0,
        )
        self.assertEqual(
            evidence["arms"][CANDIDATE]["body_ns"]["aggregate"]["value_ns"],
            82.0,
        )
        self.assertAlmostEqual(body["aggregate"]["speedup_x"], 102.0 / 82.0)
        self.assertAlmostEqual(
            body["aggregate"]["latency_reduction_pct"], 20.0 / 102.0 * 100.0
        )
        self.assertEqual(evidence["case"]["candidate_scatter_warps"], 8)
        self.assertEqual(
            evidence["scatter_ownership"]["full_tile_active_warps"],
            list(range(8)),
        )
        self.assertTrue(evidence["static_zero_spill"])
        self.assertEqual(
            evidence["arms"][CANDIDATE]["static_resources"]["registers_per_thread"],
            165,
        )
        self.assertFalse(
            Path(evidence["arms"][BASELINE]["capture"]["path"]).is_absolute()
        )
        self.assertFalse(
            Path(evidence["identity"]["probe_overlay_root"]["path"]).is_absolute()
        )
        self.assertFalse(
            Path(evidence["static_resource_evidence"]["source"]["path"]).is_absolute()
        )
        self.assertIn("scope_limits", evidence)

    def test_main_writes_requested_output(self) -> None:
        output = self.results / "scatter_phase_evidence.json"
        self.assertEqual(
            builder.main(["--results", str(self.results), "--output", str(output)]),
            0,
        )
        self.assertTrue(output.is_file())
        self.assertTrue(read_json(output)["gate_pass"])

    def test_rejects_failed_correctness(self) -> None:
        path = self.capture_file(CANDIDATE)
        capture = read_json(path)
        capture["runs"][2]["correctness_gate"]["gate_pass"] = False
        write_json(path, capture)
        with self.assertRaises(builder.EvidenceError):
            builder.build(self.results)

    def test_rejects_abi_drift(self) -> None:
        path = self.capture_file(CANDIDATE)
        capture = read_json(path)
        capture["event_abi"] = copy.deepcopy(EVENT_ABI)
        capture["event_abi"]["compute_warps"] = 4
        write_json(path, capture)
        with self.assertRaises(builder.EvidenceError):
            builder.build(self.results)

    def test_rejects_replay_count_drift(self) -> None:
        path = self.capture_file(CANDIDATE)
        capture = read_json(path)
        capture["runs"].pop()
        write_json(path, capture)
        with self.assertRaises(builder.EvidenceError):
            builder.build(self.results)

    def test_rejects_static_cubin_identity_drift(self) -> None:
        path = self.results / builder.STATIC_EVIDENCE_NAME
        evidence = read_json(path)
        evidence["arms"]["candidate"]["cubin"]["sha256"] = digest(999)
        write_json(path, evidence)
        with self.assertRaises(builder.EvidenceError):
            builder.build(self.results)

    def test_rejects_nonzero_static_spill(self) -> None:
        path = self.results / builder.STATIC_EVIDENCE_NAME
        evidence = read_json(path)
        evidence["arms"]["candidate"]["resource"]["stack_bytes_per_thread"] = 8
        write_json(path, evidence)
        with self.assertRaises(builder.EvidenceError):
            builder.build(self.results)

    def test_rejects_four_warp_candidate_ownership(self) -> None:
        path = self.results / "ownership_gate.json"
        ownership = read_json(path)
        full_tile = next(
            case for case in ownership["cases"] if case["valid_rows"] == 128
        )
        full_tile["active_warps"] = list(range(4))
        write_json(path, ownership)
        with self.assertRaises(builder.EvidenceError):
            builder.build(self.results)


if __name__ == "__main__":
    unittest.main()
