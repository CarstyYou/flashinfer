#!/usr/bin/env python3
"""CPU-only portability checks for exp_014 overlay and evidence identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import build_exp014_evidence as evidence
import build_overlays as overlays


class OverlayIdentityPortabilityTest(unittest.TestCase):
    def build_at(
        self, memory_root: Path, source_bytes: bytes
    ) -> tuple[dict[str, object], dict[str, bytes]]:
        source = memory_root / "moe_dynamic_kernel_opt.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(source_bytes)
        overlay_root = (
            memory_root / "experiments/exp_014_scatter_8warp/results/overlays"
        )
        identity = overlays.build_overlays(
            source=source,
            overlay_root=overlay_root,
            memory_root=memory_root,
        )
        artifacts = {
            "baseline": (
                overlay_root / "baseline_4warp_scatter/moe_dynamic_kernel.py"
            ).read_bytes(),
            "candidate": (
                overlay_root / "candidate_8warp_scatter/moe_dynamic_kernel.py"
            ).read_bytes(),
            "diff": (overlay_root / "candidate_8warp_scatter.diff").read_bytes(),
            "identity": (overlay_root / "identity.json").read_bytes(),
        }
        return identity, artifacts

    def test_frozen_artifacts_are_stable_for_baseline_and_promoted_inputs(
        self,
    ) -> None:
        frozen_root = overlays.OVERLAY_ROOT
        expected = {
            "baseline": (
                frozen_root / "baseline_4warp_scatter/moe_dynamic_kernel.py"
            ).read_bytes(),
            "candidate": (
                frozen_root / "candidate_8warp_scatter/moe_dynamic_kernel.py"
            ).read_bytes(),
            "diff": (frozen_root / "candidate_8warp_scatter.diff").read_bytes(),
            "identity": (frozen_root / "identity.json").read_bytes(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left_root = root / "checkout-a/.claude/w4a4_moe_bench"
            right_root = root / "checkout-b/.claude/w4a4_moe_bench"
            left, left_artifacts = self.build_at(left_root, expected["baseline"])
            right, right_artifacts = self.build_at(right_root, expected["candidate"])

        self.assertEqual(left, right)
        self.assertEqual(left_artifacts, expected)
        self.assertEqual(right_artifacts, expected)
        self.assertEqual(
            hashlib.sha256(left_artifacts["identity"]).hexdigest(),
            "615d8cf9c6596c29c2bbff3b84957237a40f0dcccdfd124a43b962a07ac452b9",
        )
        self.assertEqual(
            hashlib.sha256(left_artifacts["baseline"]).hexdigest(),
            overlays.EXPECTED_BASELINE_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(left_artifacts["candidate"]).hexdigest(),
            overlays.EXPECTED_CANDIDATE_SHA256,
        )
        self.assertEqual(left["path_base"], "w4a4_moe_bench_root")
        self.assertEqual(left["source"], "moe_dynamic_kernel_opt.py")
        self.assertEqual(
            left["baseline"],
            "experiments/exp_014_scatter_8warp/results/overlays/"
            "baseline_4warp_scatter/moe_dynamic_kernel.py",
        )
        self.assertEqual(
            left["candidate"],
            "experiments/exp_014_scatter_8warp/results/overlays/"
            "candidate_8warp_scatter/moe_dynamic_kernel.py",
        )
        serialized = json.dumps(left, sort_keys=True)
        self.assertNotIn(str(left_root), serialized)
        self.assertNotIn(str(right_root), serialized)

    def test_manifest_is_stable_across_results_roots(self) -> None:
        registered = {
            "harness": {"path": "../run_exp014_arm.py", "sha256": "1" * 64},
            "identity": {"path": "overlays/identity.json", "sha256": "2" * 64},
            "overlays": {
                evidence.BASELINE: {
                    "path": "overlays/baseline_4warp_scatter/moe_dynamic_kernel.py",
                    "sha256": "3" * 64,
                },
                evidence.CANDIDATE: {
                    "path": "overlays/candidate_8warp_scatter/moe_dynamic_kernel.py",
                    "sha256": "4" * 64,
                },
            },
        }
        ownership = {
            "source": "ownership_gate.json",
            "source_sha256": "5" * 64,
        }
        validation = {
            "arms": {
                evidence.BASELINE: {
                    "validation": "raw/validation/baseline/validation.json",
                    "validation_sha256": "6" * 64,
                },
                evidence.CANDIDATE: {
                    "validation": "raw/validation/candidate/validation.json",
                    "validation_sha256": "7" * 64,
                },
            },
            "_loaded": {
                evidence.BASELINE: {"cases": {}},
                evidence.CANDIDATE: {"cases": {}},
            },
        }
        left = evidence.source_manifest(
            Path("/checkout-a/results"),
            registered,
            ownership,
            validation,
            None,
        )
        right = evidence.source_manifest(
            Path("/checkout-b/results"),
            registered,
            ownership,
            validation,
            None,
        )

        self.assertEqual(left, right)
        self.assertEqual(left["path_base"], "results_root")
        self.assertEqual(left["results_root"], ".")

    def test_parent_artifact_path_is_results_relative(self) -> None:
        self.assertEqual(
            evidence.relative(
                Path("/checkout/exp/run_exp014_arm.py"),
                Path("/checkout/exp/results"),
            ),
            "../run_exp014_arm.py",
        )

    def test_checked_in_registered_sources_use_relocatable_identity(self) -> None:
        registered = evidence.validate_registered_sources(evidence.DEFAULT_RESULTS)
        self.assertEqual(registered["harness"]["path"], "../run_exp014_arm.py")
        self.assertEqual(
            registered["identity"]["sha256"],
            overlays.sha256((overlays.OVERLAY_ROOT / "identity.json").read_bytes()),
        )


if __name__ == "__main__":
    unittest.main()
