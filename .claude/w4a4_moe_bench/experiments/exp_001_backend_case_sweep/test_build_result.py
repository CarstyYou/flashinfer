import csv
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

import build_result  # noqa: E402


def test_speedup_is_percentage_over_single_cutedsl_denominator():
    assert build_result.speedup_percent(baseline_us=200, cutedsl_us=100) == 100
    assert build_result.speedup_percent(
        baseline_us=80, cutedsl_us=100
    ) == pytest.approx(-20)


def test_unique_summary_requires_all_six_cases_and_exact_arms():
    rows = [
        {"m": str(m), "arm": arm}
        for m in build_result.M_VALUES
        for arm in build_result.PAIR_ARMS
    ]
    resolved = build_result.unique_summary(rows, build_result.PAIR_ARMS)
    assert len(resolved) == 12
    with pytest.raises(build_result.EvidenceError, match="summary case/arm mismatch"):
        build_result.unique_summary(rows[:-1], build_result.PAIR_ARMS)


def test_raw_pairing_rejects_missing_repeat():
    rows = []
    for m in build_result.M_VALUES:
        for repeat in range(5):
            order = ">".join(
                build_result.PAIR_ARMS
                if repeat % 2 == 0
                else reversed(build_result.PAIR_ARMS)
            )
            for arm in build_result.PAIR_ARMS:
                rows.append(
                    {
                        "m": str(m),
                        "repeat": str(repeat),
                        "arm": arm,
                        "sample_us": "100.0",
                        "rerun_id": "fresh-rerun",
                        "order": order,
                    }
                )
    build_result.validate_raw(rows, build_result.PAIR_ARMS, "fresh-rerun")
    with pytest.raises(build_result.EvidenceError, match="raw repeat mismatch"):
        build_result.validate_raw(rows[:-1], build_result.PAIR_ARMS, "fresh-rerun")


def test_checked_in_outputs_are_reproducible(tmp_path):
    if not (RESULTS / "pair" / "evidence.identity.json").exists():
        pytest.skip("fresh GPU rerun has not been materialized")
    rows, context = build_result.build_rows(RESULTS / "pair", RESULTS / "sglang_triton")
    build_result.write_csv(tmp_path / "formal.csv", rows)
    (tmp_path / "result.md").write_text(build_result.render_result(rows, context))
    (tmp_path / "manifest.md").write_text(build_result.render_manifest(context))
    for name in ("formal.csv", "result.md", "manifest.md"):
        assert (tmp_path / name).read_bytes() == (RESULTS / name).read_bytes()


def test_canonical_result_has_production_opt_and_sglang_columns():
    if not (RESULTS / "formal.csv").exists():
        pytest.skip("fresh GPU rerun has not been materialized")
    with (RESULTS / "formal.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 6
    assert "production_cutedsl_fp4_us" in rows[0]
    assert "latest_opt_cutedsl_fp4_us" in rows[0]
    assert "sglang_triton_fp8_us" in rows[0]
    assert not any("vllm" in field.lower() for field in rows[0])
    result = (RESULTS / "result.md").read_text()
    assert "Latest opt CuteDSL FP4" in result
    assert "Production CuteDSL FP4" in result
    assert rows[0]["production_cutedsl_fp4_us"] == "541.1180758476257"
    assert "ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19" in result


def test_old_evidence_is_only_under_superseded_archive():
    archive = RESULTS / "superseded_vllm_prequant"
    assert archive.is_dir()
    assert (archive / "triton_arm_raw.csv").is_file()
    assert (archive / "cutlass_arm_raw.csv").is_file()
    assert not (RESULTS / "triton_arm_raw.csv").exists()
    assert not (RESULTS / "cutlass_arm_raw.csv").exists()
    assert (RESULTS / "production_pair" / "benchmark_raw.csv").is_file()
