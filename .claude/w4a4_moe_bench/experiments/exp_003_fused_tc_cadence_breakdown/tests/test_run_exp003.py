"""CPU-only contract tests for the exp_003 runner helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "run_exp003.py"


def load_runner():
    # The frontend test host intentionally has no Python-3.12 Torch wheel.
    # Pure helper tests do not use Torch, so a module placeholder keeps this
    # test genuinely CPU-only while the container smoke covers real imports.
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            sys.modules["torch"] = types.ModuleType("torch")
    spec = importlib.util.spec_from_file_location("exp003_runner_under_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


class DynamicTaskModelTest(unittest.TestCase):
    def test_deferred_queue_exact_model(self):
        model = runner.build_dynamic_task_model(
            [129, 0, 1], tile_m=128, gate_tiles=4, slice_chunk=1, grid_z=2
        )
        self.assertEqual(model.expert_tile_base, [0, 2, 2, 3])
        self.assertEqual(model.expected_task_tail, 12)
        self.assertEqual(model.expected_task_head, 14)
        self.assertEqual(model.ready_value, 0)
        self.assertEqual(len(model.tasks), 12)
        self.assertEqual(
            model.tasks[0].as_dict(ready=model.ready_value),
            {
                "task_slot": 0,
                "expert": 0,
                "m_tile": 0,
                "slice_begin": 0,
                "slice_count": 1,
                "valid_rows": 128,
                "ready": 0,
            },
        )
        self.assertEqual(model.tasks[4].valid_rows, 1)
        self.assertEqual(model.tasks[-1].expert, 2)
        self.assertEqual(model.tasks[-1].task_slot, 11)

    def test_rejects_invalid_geometry_and_counts(self):
        with self.assertRaises(ValueError):
            runner.build_dynamic_task_model([-1])
        with self.assertRaises(ValueError):
            runner.build_dynamic_task_model([1], grid_z=0)


class OverlayContractTest(unittest.TestCase):
    def test_resolves_one_path_and_rejects_boolean_enable(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = Path(directory) / "overlay.py"
            overlay.write_text("VALUE = 1\n")
            key, path = runner.resolve_overlay_from_env(
                {"EXP003_MARKER_OVERLAY": str(overlay)}
            )
            self.assertEqual(key, "EXP003_MARKER_OVERLAY")
            self.assertEqual(path, overlay.resolve())
        with self.assertRaises(RuntimeError):
            runner.resolve_overlay_from_env({"EXP003_MARKER_OVERLAY": "1"})

    def test_rejects_conflicting_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.py"
            second = Path(directory) / "second.py"
            first.write_text("")
            second.write_text("")
            with self.assertRaises(RuntimeError):
                runner.resolve_overlay_from_env(
                    {
                        "EXP003_MARKER_OVERLAY": str(first),
                        "W4A4_EXP003_MARKER_OVERLAY": str(second),
                    }
                )

    def test_enforces_native_dump_user_event_id_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = Path(directory) / "overlay.py"
            overlay.write_text(
                "\n".join(
                    f'cute.experimental.iket.range_push("event_{index}")'
                    for index in range(30)
                )
                + "\n"
            )
            evidence = runner.validate_overlay_event_id_budget(overlay)
            self.assertEqual(evidence["unique_named_event_count"], 30)
            self.assertEqual(evidence["max_user_event_ids"], 30)

            overlay.write_text(
                "\n".join(
                    f'cute.experimental.iket.range_push("event_{index}")'
                    for index in range(31)
                )
                + "\n"
            )
            with self.assertRaisesRegex(RuntimeError, "31 > 30"):
                runner.validate_overlay_event_id_budget(overlay)


class ManifestSafetyTest(unittest.TestCase):
    def test_claim_is_exclusive_and_owned_update_is_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            owned = runner.OwnedManifest.claim(path, {"status": "running"})
            with self.assertRaises(FileExistsError):
                runner.OwnedManifest.claim(path, {"status": "other"})
            owned.update(status="complete", value=7)
            self.assertEqual(
                json.loads(path.read_text()), {"status": "complete", "value": 7}
            )

    def test_selected_cta_parser(self):
        self.assertEqual(runner.parse_selected_cta("0,0,109"), [0, 0, 109])
        with self.assertRaises(RuntimeError):
            runner.parse_selected_cta("0,55")
        with self.assertRaises(RuntimeError):
            runner.parse_selected_cta("0,0,110")
        with self.assertRaises(RuntimeError):
            runner.parse_selected_cta("0,0,54")

    def test_marker_correctness_is_relative_to_control(self):
        control = {
            "cosine": 0.99991,
            "relative_l2": 0.013,
            "max_abs_error": 0.062,
            "per_token_relative_l2_p99": 0.0604,
        }
        candidate = {
            "cosine": 0.999905,
            "relative_l2": 0.014,
            "max_abs_error": 0.064,
            "per_token_relative_l2_p99": 0.062,
        }
        self.assertTrue(
            runner.marker_correctness_vs_control(candidate, control)["gate_pass"]
        )
        candidate["per_token_relative_l2_p99"] = 0.08
        self.assertFalse(
            runner.marker_correctness_vs_control(candidate, control)["gate_pass"]
        )


class IketIdentityTest(unittest.TestCase):
    def test_reads_adjacent_distribution_metadata_without_cli_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "bin" / "run-iket"
            metadata = (
                root / "lib/python3.12/site-packages/iket-0.7.10.dist-info/METADATA"
            )
            executable.parent.mkdir(parents=True)
            metadata.parent.mkdir(parents=True)
            executable.write_text("this is deliberately not executable\n")
            metadata.write_text("Name: iket\nVersion: 0.7.10\n")
            original = runner.dependency_version
            runner.dependency_version = lambda name: "NOT_INSTALLED"
            try:
                identity = runner.iket_distribution_identity(str(executable))
            finally:
                runner.dependency_version = original
            self.assertEqual(identity["distribution_version"], "0.7.10")
            self.assertEqual(identity["version_method"], "adjacent dist-info METADATA")
            self.assertEqual(identity["run_iket"], str(executable.resolve()))


if __name__ == "__main__":
    unittest.main()
