import csv
from pathlib import Path
import re

import pytest

import build_result

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOCAL_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def _base_row(backend, m, latency):
    return {
        "backend": backend,
        "m": m,
        "hidden": 2048,
        "intermediate_tp": 512,
        "experts": 256,
        "topk": 8,
        "timing": "cuda_graph_events_inside",
        "flush_l2": 1,
        "l2_flush_bytes": 201326592,
        "warmup": 5,
        "iters": 50,
        "repeats": 5,
        "median_us": latency,
        "samples_ms": ";".join([str(latency / 1000.0)] * 5),
        "error": "",
        "fixture_sha256": f"fixture-{m}",
        "occupancy_sha256": f"occupancy-{m}",
        "gpu_uuid": "GPU-test",
        "functional_sanity": "shape_dtype_finite_nonzero",
    }


def _write(path, rows):
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_evidence(tmp_path):
    cutlass_arm = tmp_path / "cutlass.csv"
    triton = tmp_path / "triton.csv"
    cutlass_rows = []
    for m in (1, *build_result.M_VALUES):
        cutlass_rows.append(_base_row("flashinfer_cutedsl", m, 100))
        cutlass_rows.append(_base_row("cutlass", m, 110))
    _write(cutlass_arm, cutlass_rows)
    _write(
        triton,
        [_base_row("triton_fp8", m, 200) for m in build_result.M_VALUES],
    )
    return cutlass_arm, triton


def test_speedup_is_percentage_over_cutedsl():
    assert build_result.speedup_percent(baseline_us=200, cutedsl_us=100) == 100
    assert build_result.speedup_percent(
        baseline_us=80, cutedsl_us=100
    ) == pytest.approx(-20)


def test_builds_one_canonical_three_backend_result(tmp_path):
    cutlass_arm, triton = _make_evidence(tmp_path)
    cute = build_result.read_backend(
        cutlass_arm, "flashinfer_cutedsl", mixed_backend_file=True
    )
    cutlass = build_result.read_backend(cutlass_arm, "cutlass", mixed_backend_file=True)
    triton_rows = build_result.read_backend(triton, "triton_fp8")
    rows = build_result.compare(cute, cutlass, triton_rows)
    assert all(row["cutedsl_source_arm"] == "cutlass_arm" for row in rows)
    assert rows[0]["cutedsl_us"] == 100
    assert rows[0]["speedup_vs_cutlass_percent"] == pytest.approx(10)
    assert rows[0]["speedup_vs_triton_percent"] == pytest.approx(100)
    build_result.write_outputs(tmp_path, rows)

    result = (tmp_path / "result.md").read_text()
    assert "CUTLASS vs CuteDSL vs Triton" in result
    assert "100.00%" in result
    assert "CuteDSL paired" not in result
    assert "[`plan.md`](../plan.md)" in result
    with (tmp_path / "formal.csv").open(newline="") as csv_file:
        formal = list(csv.DictReader(csv_file))
    assert len(formal) == 6
    assert set(formal[0]) >= {
        "cutlass_us",
        "cutedsl_source_arm",
        "cutedsl_us",
        "triton_fp8_us",
    }
    assert "cutedsl_triton_pair_us" not in formal[0]


def test_missing_prefill_case_is_rejected(tmp_path):
    _, triton = _make_evidence(tmp_path)
    rows = list(csv.DictReader(triton.open()))[:-1]
    _write(triton, rows)
    with pytest.raises(build_result.EvidenceError, match="cases="):
        build_result.read_backend(triton, "triton_fp8")


def test_missing_triton_identity_is_rejected(tmp_path):
    cutlass_arm, triton = _make_evidence(tmp_path)
    cute = build_result.read_backend(
        cutlass_arm, "flashinfer_cutedsl", mixed_backend_file=True
    )
    cutlass = build_result.read_backend(cutlass_arm, "cutlass", mixed_backend_file=True)
    triton_rows = build_result.read_backend(triton, "triton_fp8")
    triton_rows[build_result.M_VALUES[0]]["fixture_sha256"] = ""
    with pytest.raises(build_result.EvidenceError, match="fixture_sha256 missing"):
        build_result.compare(cute, cutlass, triton_rows)


def test_checked_in_outputs_are_reproducible(tmp_path):
    cute = build_result.read_backend(
        RESULTS / "cutlass_arm_raw.csv",
        "flashinfer_cutedsl",
        mixed_backend_file=True,
    )
    cutlass = build_result.read_backend(
        RESULTS / "cutlass_arm_raw.csv", "cutlass", mixed_backend_file=True
    )
    triton = build_result.read_backend(RESULTS / "triton_arm_raw.csv", "triton_fp8")
    for rows in (cute, cutlass, triton):
        build_result.validate_contract(rows)
    build_result.write_outputs(tmp_path, build_result.compare(cute, cutlass, triton))
    assert (tmp_path / "formal.csv").read_bytes() == (
        RESULTS / "formal.csv"
    ).read_bytes()
    assert (tmp_path / "result.md").read_bytes() == (RESULTS / "result.md").read_bytes()
    assert "PENDING" not in (RESULTS / "manifest.md").read_text()


def test_experiment_root_contains_only_plan_and_scripts():
    unexpected = sorted(
        path.name
        for path in ROOT.iterdir()
        if path.is_file() and path.name != "plan.md" and path.suffix != ".py"
    )
    assert unexpected == []
    assert (RESULTS / "result.md").is_file()
    assert (RESULTS / "manifest.md").is_file()
    assert (RESULTS / "formal.csv").is_file()


def test_local_markdown_links_resolve():
    for document in (ROOT / "plan.md", RESULTS / "result.md", RESULTS / "manifest.md"):
        for target in LOCAL_LINK.findall(document.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            path = target.split("#", maxsplit=1)[0]
            assert (document.parent / path).exists(), f"{document}: {target}"
