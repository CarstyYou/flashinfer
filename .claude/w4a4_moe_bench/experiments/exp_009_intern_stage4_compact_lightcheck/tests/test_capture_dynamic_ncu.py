import csv

import pytest

from capture_dynamic_ncu import (
    ARM,
    EXPECTED_BLOCK,
    EXPECTED_GRID,
    REQUIRED_METRICS,
    correctness_checks,
    parse_cubin_symbol,
    parse_native_raw,
)


SYMBOL = "kernel_cutlass_kernel_MoEDynamicKernel_object_at__tensorptr_0"


def _write_native_csv(path, *, rows=1, missing=None, value_override=None):
    metric_ids = [value for value in REQUIRED_METRICS.values() if value != missing]
    header = [
        "ID",
        "Context",
        "Stream",
        "Device",
        "Kernel Name",
        "Block Size",
        "Grid Size",
        *metric_ids,
    ]
    units = ["", "", "", "", "", "", "", *["inst"] * len(metric_ids)]
    values = [
        "1",
        "2",
        "7",
        "0",
        SYMBOL,
        "(160, 1, 1)",
        "(1, 1, 110)",
        *[str((value_override or {}).get(metric_id, 0)) for metric_id in metric_ids],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["==PROF== Disconnected from process 1"])
        writer.writerow(header)
        writer.writerow(units)
        for _ in range(rows):
            writer.writerow(values)


def test_parse_single_node_with_numeric_spill_and_launch_metrics(tmp_path):
    path = tmp_path / "native_raw.csv"
    spill_metric = REQUIRED_METRICS["spill_store_instructions"]
    _write_native_csv(path, value_override={spill_metric: 17})

    metrics, observed = parse_native_raw(path, expected_kernel_symbol=SYMBOL)

    assert metrics["spill_store_instructions"]["value"] == 17
    assert observed["grid"] == EXPECTED_GRID
    assert observed["block"] == EXPECTED_BLOCK
    assert observed["kernel_symbol"] == SYMBOL


def test_parser_fails_closed_on_missing_or_na_metric(tmp_path):
    missing_path = tmp_path / "missing.csv"
    _write_native_csv(missing_path, missing=REQUIRED_METRICS["spill_refill_bytes"])
    with pytest.raises(ValueError, match="required NCU metric absent"):
        parse_native_raw(missing_path, expected_kernel_symbol=SYMBOL)

    na_path = tmp_path / "na.csv"
    metric = REQUIRED_METRICS["spill_refill_bytes"]
    _write_native_csv(na_path, value_override={metric: "n/a"})
    with pytest.raises(ValueError, match="missing/N-A required metric"):
        parse_native_raw(na_path, expected_kernel_symbol=SYMBOL)


def test_parser_requires_exactly_one_graph_node_and_exact_symbol(tmp_path):
    path = tmp_path / "two.csv"
    _write_native_csv(path, rows=2)
    with pytest.raises(ValueError, match="one profiled graph node"):
        parse_native_raw(path, expected_kernel_symbol=SYMBOL)

    one = tmp_path / "one.csv"
    _write_native_csv(one)
    with pytest.raises(ValueError, match="kernel symbol drift"):
        parse_native_raw(one, expected_kernel_symbol=SYMBOL + "_other")


def test_cubin_symbol_parser_requires_one_moe_entry():
    text = f"Function {SYMBOL}:\n REG: 128 STACK: 0 SHARED: 0 LOCAL: 0 CONSTANT[0]: 0\n"
    assert parse_cubin_symbol(text) == SYMBOL
    with pytest.raises(ValueError, match="one cubin kernel resource record"):
        parse_cubin_symbol(text + text)
    with pytest.raises(ValueError, match="unexpected cubin kernel entry"):
        parse_cubin_symbol(
            "Function helper_kernel:\n REG: 1 STACK: 0 SHARED: 0 LOCAL: 0\n"
        )


def _preparation():
    return {
        "status": "complete",
        "arm": ARM,
        "m": 256,
        "fixture_kind": "canonical",
        "case": {
            "m": 256,
            "experts": 256,
            "hidden": 2048,
            "intermediate_tp": 512,
            "topk": 8,
        },
        "outputs": [
            {
                "formal_pass": True,
                "finite": True,
                "nonzero": True,
                "sentinel_nan_remaining": 0,
            },
            {
                "formal_pass": True,
                "finite": True,
                "nonzero": True,
                "sentinel_nan_remaining": 0,
            },
        ],
        "route_task_evidence": [
            {"verification": {"gate_pass": True}},
            {"verification": {"gate_pass": True}},
        ],
        "launch_contract": {
            "expected_grid": EXPECTED_GRID,
            "expected_block": EXPECTED_BLOCK,
            "expected_final_replay_kernel": "MoEDynamicKernel",
        },
    }


def test_preparation_is_the_correctness_gate():
    preparation = _preparation()
    assert all(correctness_checks(preparation).values())

    preparation["outputs"][1]["formal_pass"] = False
    checks = correctness_checks(preparation)
    assert not checks["two_correct_replays"]
    assert all(value for key, value in checks.items() if key != "two_correct_replays")
