"""CPU-only ownership gates for the two-layer breakdown harness."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
BENCH_ROOT = ROOT.parent


class BreakdownHarnessContractTest(unittest.TestCase):
    def test_shared_vertical_slice_exists(self):
        for relative in (
            "README.md",
            "case.py",
            "artifacts.py",
            "backends/cutedsl.py",
            "backends/cutedsl_workspace.py",
            "backends/triton_fp8.py",
            "fragments/eric_stage4_adapter.py",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_canonical_case_exposes_stable_case_id(self):
        source = (ROOT / "case.py").read_text()
        functions = {
            node.name
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("case_id", functions)
        self.assertIn('"case_id": case_id(m)', source)

    def test_shared_python_does_not_depend_on_an_experiment(self):
        for path in ROOT.rglob("*.py"):
            if path.name.startswith("test_"):
                continue
            source = path.read_text()
            self.assertNotIn("experiments/exp_", source, str(path.relative_to(ROOT)))
            self.assertNotIn(
                "sys.modules[__name__]", source, str(path.relative_to(ROOT))
            )

    def test_experiments_keep_custom_entrypoints(self):
        entrypoints = (
            "experiments/exp_001_backend_case_sweep/bench_triton_fp8.py",
            "experiments/exp_005_8warp_spill_reduction/run_exp005_arm.py",
            "experiments/exp_009_intern_stage4_compact_lightcheck/build_adapter.py",
            "experiments/exp_018_triton_opt_eric_benchmark/run_arm.py",
        )
        for relative in entrypoints:
            path = BENCH_ROOT / relative
            functions = {
                node.name
                for node in ast.parse(path.read_text()).body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            self.assertIn("main", functions, relative)

    def test_migrated_entrypoints_import_only_shared_harness_code(self):
        entrypoints = (
            "experiments/exp_005_8warp_spill_reduction/run_exp005_arm.py",
            "experiments/exp_018_triton_opt_eric_benchmark/run_arm.py",
        )
        for relative in entrypoints:
            source = (BENCH_ROOT / relative).read_text()
            tree = ast.parse(source)
            imported_modules = {
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            self.assertFalse(
                any(
                    module.startswith("experiments.exp_") for module in imported_modules
                ),
                relative,
            )
            self.assertNotIn("spec_from_file_location", source, relative)

    def test_active_remote_routes_sync_the_shared_harness(self):
        scripts = (
            "experiments/exp_018_triton_opt_eric_benchmark/run_remote.sh",
            "experiments/exp_019_opt_vs_eric_dataflow_bottleneck/run_ncu_remote.sh",
            "experiments/exp_019_opt_vs_eric_dataflow_bottleneck/run_phase_remote.sh",
        )
        for relative in scripts:
            source = (BENCH_ROOT / relative).read_text()
            self.assertIn("w4a4_moe_bench/breakdown_harness", source, relative)

    def test_phase_route_syncs_its_custom_runner_dependency(self):
        relative = (
            "experiments/exp_019_opt_vs_eric_dataflow_bottleneck/run_phase_remote.sh"
        )
        source = (BENCH_ROOT / relative).read_text()
        self.assertIn("exp_018_triton_opt_eric_benchmark", source)

    def test_refactor_has_no_temporary_identity_placeholders(self):
        placeholder = "PENDING" + "_HARNESS_REFACTOR"
        paths = (
            ROOT,
            BENCH_ROOT / "experiments/exp_022_scatter_vector_s2r/profile_target.py",
        )
        for path in paths:
            files = path.rglob("*.py") if path.is_dir() else (path,)
            for source_path in files:
                self.assertNotIn(
                    placeholder,
                    source_path.read_text(),
                    str(source_path),
                )

    def test_active_evidence_binds_shared_source_identity(self):
        scripts = (
            "experiments/exp_001_backend_case_sweep/bench_triton_fp8.py",
            "experiments/exp_005_8warp_spill_reduction/run_exp005_arm.py",
            "experiments/exp_018_triton_opt_eric_benchmark/run_arm.py",
        )
        for relative in scripts:
            source = (BENCH_ROOT / relative).read_text()
            self.assertIn("source_manifest", source, relative)
            self.assertIn("harness_sources", source, relative)


if __name__ == "__main__":
    unittest.main()
