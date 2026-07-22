"""CPU-only ownership checks for the exp001 Triton arm split."""

import ast
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent
BENCH_ROOT = EXPERIMENT_ROOT.parents[1]
SCRIPT = EXPERIMENT_ROOT / "bench_triton_fp8.py"
BACKEND = BENCH_ROOT / "breakdown_harness" / "backends" / "triton_fp8.py"


def parsed(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def test_experiment_owns_identity_defaults_and_cli():
    source = SCRIPT.read_text()
    function_names = {
        node.name for node in parsed(SCRIPT).body if isinstance(node, ast.FunctionDef)
    }
    assert 'COMPARISON_GROUP_ID = "exp001_' in source
    assert 'DEFAULT_RESULTS = EXPERIMENT_ROOT / "results" / "sglang_triton"' in source
    assert 'DEFAULT_FIXTURES = EXPERIMENT_ROOT / "results" / "fixtures"' in source
    assert "EXPECTED_IMAGE_DIGEST" in source
    assert {"parse_args", "main"} <= function_names


def test_shared_backend_has_no_experiment_policy_or_paths():
    source = BACKEND.read_text()
    tree = ast.parse(source)
    names = {
        node.targets[0].id
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    assert "experiments/" not in source
    assert "COMPARISON_GROUP_ID" not in names
    assert "DEFAULT_RESULTS" not in names
    assert "DEFAULT_FIXTURES" not in names
    assert "EXPECTED_IMAGE_DIGEST" not in names


def test_exp001_reexports_shared_runtime_api_for_import_compatibility():
    imported = set()
    for node in ast.walk(parsed(SCRIPT)):
        if isinstance(node, ast.ImportFrom) and node.module == (
            "breakdown_harness.backends.triton_fp8"
        ):
            imported.update(alias.name for alias in node.names)
    assert {
        "Fp8Weights",
        "make_fp8_weights",
        "build_launch",
        "CapturedCall",
        "make_l2_flusher",
    } <= imported
    case_imports = set()
    for node in ast.walk(parsed(SCRIPT)):
        if isinstance(node, ast.ImportFrom) and node.module == "breakdown_harness.case":
            case_imports.update(alias.name for alias in node.names)
    assert {
        "E",
        "H",
        "I",
        "TOPK",
        "load_fixture",
        "validate_output",
    } <= case_imports
