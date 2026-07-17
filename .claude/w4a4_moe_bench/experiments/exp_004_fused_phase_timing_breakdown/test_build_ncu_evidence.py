from __future__ import annotations

import csv

from build_ncu_evidence import METRICS, parse_native_csv


def test_parse_one_identity_locked_ncu_row(tmp_path):
    identity = {
        "ID": ("", "0"),
        "Kernel Name": ("", "MoEDynamicKernel"),
        "Context": ("", "1"),
        "Stream": ("", "2"),
        "Device": ("", "0"),
        "launch__block_dim_x": ("", "160"),
        "launch__block_dim_y": ("", "1"),
        "launch__block_dim_z": ("", "1"),
        "launch__grid_dim_x": ("", "1"),
        "launch__grid_dim_y": ("", "1"),
        "launch__grid_dim_z": ("", "110"),
    }
    metric_units = {
        "gpu__time_duration.sum": "ns",
        "launch__registers_per_thread": "register/thread",
        "launch__registers_per_thread_allocated": "register/thread",
        "launch__shared_mem_per_block": "byte/block",
        "launch__shared_mem_per_block_dynamic": "byte/block",
    }
    columns = dict(identity)
    for metric in METRICS.values():
        columns[metric] = (metric_units.get(metric, "inst"), "7")
    # Metrics with byte semantics use a byte unit; dimensionless launch values
    # are emitted with an empty unit by NCU.
    for metric in METRICS.values():
        if "bytes" in metric or "mem_local" in metric:
            columns[metric] = ("byte", "7")
    columns["launch__stack_size"] = ("", "1024")
    columns["launch__waves_per_multiprocessor"] = ("", "1")
    path = tmp_path / "native.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerow([value[0] for value in columns.values()])
        writer.writerow([value[1] for value in columns.values()])
    metrics, units, launch = parse_native_csv(path)
    assert launch["block"] == [160, 1, 1]
    assert launch["grid"] == [1, 1, 110]
    assert metrics["duration_ns"] == 7
    assert units["duration_ns"] == "ns"
