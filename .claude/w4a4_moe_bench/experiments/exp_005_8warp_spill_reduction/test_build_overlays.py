#!/usr/bin/env python3
"""Static tests for the exp_005 overlay generator."""

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent / "build_overlays.py"
SPEC = importlib.util.spec_from_file_location("exp005_build_overlays", str(SCRIPT_PATH))
build_overlays = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_overlays)


class BuildOverlaysTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_path = (
            build_overlays.repository_root() / build_overlays.PRODUCTION_SOURCE_REL
        )
        cls.production = cls.source_path.read_bytes()

    def test_build_has_exactly_two_changes_and_valid_identity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "overlays"
            identity = build_overlays.build_overlays(self.source_path, output_dir)

            baseline = (
                output_dir / build_overlays.BASELINE_NAME / "moe_dynamic_kernel.py"
            ).read_bytes()
            candidate = (
                output_dir / build_overlays.CANDIDATE_NAME / "moe_dynamic_kernel.py"
            ).read_bytes()

            self.assertEqual(baseline, self.production)
            expected_candidate = self.production
            for before, after, _ in build_overlays.TRANSFORMS:
                self.assertEqual(expected_candidate.count(before), 1)
                expected_candidate = expected_candidate.replace(before, after, 1)
            self.assertEqual(candidate, expected_candidate)
            self.assertNotEqual(candidate, self.production)
            ast.parse(candidate.decode("utf-8"))

            diff = (
                output_dir / "{}.diff".format(build_overlays.CANDIDATE_NAME)
            ).read_text(encoding="utf-8")
            removed = [line for line in diff.splitlines() if line.startswith("-")]
            added = [line for line in diff.splitlines() if line.startswith("+")]
            # Account for the two ---/+++ file header lines.
            self.assertEqual(len(removed), 3)
            self.assertEqual(len(added), 3)
            self.assertIn("-        self.num_mma_warps = 4", removed)
            self.assertIn("+        self.num_mma_warps = 8", added)
            self.assertIn("-        atom_layout = cute.make_layout((2, 2, 1))", removed)
            self.assertIn("+        atom_layout = cute.make_layout((4, 2, 1))", added)

            persisted_identity = json.loads(
                (output_dir / "identity.json").read_text(encoding="utf-8")
            )
            self.assertEqual(identity, persisted_identity)
            self.assertTrue(
                identity["arms"][build_overlays.BASELINE_NAME][
                    "byte_identical_to_production"
                ]
            )
            self.assertEqual(
                identity["arms"][build_overlays.BASELINE_NAME]["sha256"],
                build_overlays.PRODUCTION_SHA256,
            )
            self.assertEqual(
                [
                    record["match_count"]
                    for record in identity["arms"][build_overlays.CANDIDATE_NAME][
                        "transforms"
                    ]
                ],
                [1, 1],
            )

    def test_exact_transform_rejects_missing_or_duplicate_match(self):
        for malformed in (b"", self.production + self.production):
            with self.subTest(size=len(malformed)), self.assertRaises(RuntimeError):
                build_overlays._apply_exact_transforms(malformed)

    def test_source_hash_lock_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_directory = Path(temporary_directory)
            drifted_source = temporary_directory / "moe_dynamic_kernel.py"
            drifted_source.write_bytes(self.production + b"\n")
            with self.assertRaisesRegex(RuntimeError, "production source hash drift"):
                build_overlays.build_overlays(
                    drifted_source, temporary_directory / "overlays"
                )


if __name__ == "__main__":
    unittest.main()
