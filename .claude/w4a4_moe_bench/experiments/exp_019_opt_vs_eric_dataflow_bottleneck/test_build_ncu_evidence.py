"""CPU-only test for the compact exp_019 paired NCU evidence builder."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "exp019_build_ncu_evidence_test", ROOT / "build_ncu_evidence.py"
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def metric_unit(metric: str) -> str | None:
    if "pct_of_peak" in metric:
        return "%"
    if "registers_per_thread" in metric:
        return "register/thread"
    if "shared_mem_per_block" in metric:
        return "byte/block"
    if "spilling" in metric:
        return "inst"
    if "occupancy_limit" in metric:
        return "block"
    return None


def write_native(path: Path, arm: str, values: dict[str, tuple[str, float]]) -> None:
    header = ["ID", "Kernel Name", "Block Size", "Grid Size", *values]
    units = ["", "", "", "", *(unit for unit, _ in values.values())]
    row = [
        "0",
        builder.EXPECTED_SYMBOL,
        str(tuple(builder.EXPECTED_BLOCK[arm])),
        str(tuple(builder.EXPECTED_GRID)),
        *(str(value) for _, value in values.values()),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows((header, units, row))


def write_cell(results: Path, arm: str, m: int, factor: float) -> None:
    raw = results / "raw/ncu" / arm / f"m{m}"
    veloq = results / "veloq/ncu" / arm / f"m{m}"
    checks = {"sanity": True, "full_oracle": True, "route_task": True}
    correctness_row = {
        "gate_pass": True,
        "checks": checks,
        "oracle": {"reference_sha256": "shared-reference"},
    }
    manifest = {
        "schema": "exp019.production-ncu-target.v1",
        "status": "complete",
        "arm": arm,
        "m": m,
        "launch_contract": {
            "symbol": builder.EXPECTED_SYMBOL,
            "grid": builder.EXPECTED_GRID,
            "block": builder.EXPECTED_BLOCK[arm],
        },
        "source_identity": {
            "locked_files": {"source": {"sha256": builder.EXPECTED_SOURCE_SHA256[arm]}},
            "runtime": {
                "source_sha256": builder.EXPECTED_SOURCE_SHA256[arm],
                "overlay_sha256": builder.EXPECTED_OVERLAY_SHA256[arm],
            },
        },
        "jit_identity": {
            "cubin_sha256": [builder.EXPECTED_CUBIN_SHA256[arm]],
            "symbols": [builder.EXPECTED_SYMBOL],
        },
        "correctness": {
            "gate_pass": True,
            "eager": correctness_row,
            "pre_profile_graph": correctness_row,
            "post_profile_graph": correctness_row,
        },
        "fixture_manifest": {
            "fixture_sha256": builder.EXPECTED_FIXTURE_SHA256[m],
            "x_sha256": "same-x",
            "topk_ids_sha256": "same-ids",
            "topk_weights_sha256": "same-routing",
        },
        "weight_identity": {
            "packed_weights_sha256": {"w1": "same-w1", "w2": "same-w2"},
            "scales_sha256": {"w1": "same-s1", "w2": "same-s2"},
        },
        "profile": {"profiled_graph_replays": 1},
        "runtime": {"gpu_uuid": "same-gpu", "nvcc": "same-nvcc"},
        "telemetry_gate": {"pass": True},
    }
    write_json(raw / "target_manifest.json", manifest)

    metrics = [
        {
            "name": metric,
            "unit": metric_unit(metric),
            "value": 10.0 * factor,
        }
        for metric in builder.SELECTED_METRICS.values()
    ]
    write_json(
        veloq / "inspect.json",
        {
            "command": "ncu.inspect",
            "data": {
                "rows": [
                    {
                        "type": "launch",
                        "row_id": "launch:0",
                        "kernel_demangled": builder.EXPECTED_SYMBOL,
                        "grid_size": builder.EXPECTED_GRID,
                        "block_size": builder.EXPECTED_BLOCK[arm],
                        "has_disasm": True,
                        "metrics": metrics,
                    }
                ]
            },
        },
    )
    write_json(
        veloq / "pc_stalls_file.json",
        {
            "command": "ncu.source-metrics",
            "data": {
                "rows": [],
                "auxiliary": {
                    "unattributed_sass_counter_totals": {
                        f"{builder.PC_STALL_PREFIX}wait": 3.0 * factor,
                        f"{builder.PC_STALL_PREFIX}wait_not_issued": factor,
                        f"{builder.PC_STALL_PREFIX}long_scoreboard": factor,
                        f"{builder.PC_STALL_PREFIX}long_scoreboard_not_issued": factor,
                    },
                    "out_of_cubin_counter_totals": {},
                },
            },
        },
    )
    write_native(
        raw / "native_raw.csv", arm, {"gpu__time_duration.sum": ("ns", 100.0 * factor)}
    )
    additive = {
        metric: (
            "ns"
            if metric == "gpu__time_duration.sum"
            else "byte"
            if "bytes" in metric
            else "inst",
            100.0 * factor,
        )
        for metric in builder.ADDITIVE_METRICS.values()
    }
    write_native(raw / "ledger_native_raw.csv", arm, additive)


def test_builds_compact_legal_paired_tables(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_cell(results, builder.OPT, 8192, 1.0)
    write_cell(results, builder.ERIC, 8192, 1.5)

    evidence = builder.build_evidence(results)
    case = evidence["cases"]["m8192"]
    assert case["status"] == "complete"
    paired = case["paired_direct"]
    issue = next(
        row for row in paired["selected_metrics"] if row["name"] == "issue_active_pct"
    )
    assert issue["delta_kind"] == "percentage_points"
    assert issue["eric_minus_opt_pp"] == 5.0
    dram = next(
        row for row in paired["additive_metrics"] if row["name"] == "dram_total_bytes"
    )
    assert dram["metric"] == "dram__bytes.sum"
    assert dram["eric_vs_opt_percent"] == 50.0
    assert paired["additive_unavailable"] == []
    assert len(json.dumps(evidence, separators=(",", ":")).encode()) < 200_000


def test_missing_targeted_ledger_is_explicit(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_cell(results, builder.OPT, 8192, 1.0)
    write_cell(results, builder.ERIC, 8192, 1.0)
    (results / "raw/ncu" / builder.ERIC / "m8192/ledger_native_raw.csv").unlink()

    paired = builder.build_evidence(results)["cases"]["m8192"]["paired_direct"]
    assert "dram_total_bytes" in paired["additive_unavailable"]
    assert "profiler_duration_ns" not in paired["additive_unavailable"]
