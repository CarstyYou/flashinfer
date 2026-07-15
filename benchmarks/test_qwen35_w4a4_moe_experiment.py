import csv
from pathlib import Path

import pytest

from benchmarks import run_qwen35_w4a4_moe as experiment


def test_qwen35_testlist_contract():
    cases = experiment.load_cases(experiment.DEFAULT_TESTLIST)

    assert tuple(case.num_tokens for case in cases) == experiment.EXPECTED_M_VALUES
    assert all("--use_cuda_events" in case.argv for case in cases)
    assert all("--no_cuda_graph" not in case.argv for case in cases)


def test_checkout_environment_prepends_repo(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/installed/packages")

    env = experiment._checkout_environment()

    assert env["PYTHONPATH"].split(":") == [
        str(experiment.REPO_ROOT),
        "/installed/packages",
    ]


def _result_row(case):
    return {
        "case_tag": case.case_tag,
        "num_tokens": str(case.num_tokens),
        "backend": "b12x",
        "resolved_backend": "dynamic",
        "routing_method": "renormalize",
        "use_cupti": "False",
        "no_cuda_graph": "False",
        "activation_type": "Swiglu",
        "fp4_mode": "nvfp4",
        "input_dtype": "torch.bfloat16",
        "weight_dtype": "torch.bfloat16",
        "random_seed": "42",
        "refcheck": "False",
        "allow_output_mismatch": "False",
        "generate_repro_command": "True",
        "median_time": "0.5",
        "std_time": "0.01",
        "tflops": "100.0",
        "tb_per_sec": "1.0",
    }


def test_validate_rows_accepts_complete_dynamic_sweep():
    cases = experiment.load_cases(experiment.DEFAULT_TESTLIST)
    summary = experiment.validate_rows([_result_row(case) for case in cases], cases)

    assert len(summary) == len(experiment.EXPECTED_M_VALUES)
    assert summary[0]["median_us"] == 500.0
    assert summary[0]["correctness"] == "not_run"


def test_validate_rows_rejects_missing_case():
    cases = experiment.load_cases(experiment.DEFAULT_TESTLIST)

    with pytest.raises(experiment.ExperimentError, match="result case mismatch"):
        experiment.validate_rows([_result_row(case) for case in cases[:-1]], cases)


def test_validate_rows_rejects_wrong_dispatch():
    cases = experiment.load_cases(experiment.DEFAULT_TESTLIST)
    rows = [_result_row(case) for case in cases]
    rows[0]["resolved_backend"] = "static"

    with pytest.raises(experiment.ExperimentError, match="unexpected dispatch"):
        experiment.validate_rows(rows, cases)


def test_contract_rejects_changed_activation(tmp_path: Path):
    changed = experiment.DEFAULT_TESTLIST.read_text().replace(
        "--activation-type Swiglu", "--activation-type Relu2", 1
    )
    testlist = tmp_path / "changed.txt"
    testlist.write_text(changed)

    with pytest.raises(experiment.ExperimentError, match="--activation-type"):
        experiment.load_cases(testlist)


def test_summary_round_trip(tmp_path: Path):
    cases = experiment.load_cases(experiment.DEFAULT_TESTLIST)
    summary = experiment.validate_rows([_result_row(case) for case in cases], cases)

    experiment.write_summary(tmp_path, summary)

    with (tmp_path / "summary.csv").open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 6
    assert rows[-1]["num_tokens"] == "8192"
    assert "Correctness was not run" in (tmp_path / "summary.md").read_text()
