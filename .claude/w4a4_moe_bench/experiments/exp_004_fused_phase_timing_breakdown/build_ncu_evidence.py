#!/usr/bin/env python3
"""Validate exp_004 NCU launch/resource/dynamic-work captures."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Sequence

from exp004_common import (
    ALL_ARMS,
    DEFAULT_RESULTS,
    EXPECTED_BLOCK,
    EXPECTED_GRID,
    MEASUREMENT_CONTROL,
    NORMAL,
    canonical_sha256,
    file_sha256,
    read_json,
    write_json,
)


METRICS = {
    "duration_ns": "gpu__time_duration.sum",
    "local_load_bytes": "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
    "local_store_bytes": "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
    "spill_refill_instructions": "sass__inst_executed_register_spilling_op_read",
    "spill_store_instructions": "sass__inst_executed_register_spilling_op_write",
    "spill_refill_bytes": "sass__inst_executed_register_spilling_mem_local_op_read",
    "spill_store_bytes": "sass__inst_executed_register_spilling_mem_local_op_write",
    "executed_warp_instructions": "sm__inst_executed.sum",
    "tensor_instructions": "sm__inst_executed_pipe_tensor_subpipe_hmma.sum",
    "fp4_tensor_ops": "sm__ops_path_tensor_src_fp4_dst_fp32.sum",
    "registers_per_thread": "launch__registers_per_thread",
    "allocated_registers_per_thread": "launch__registers_per_thread_allocated",
    "shared_mem_per_block_bytes": "launch__shared_mem_per_block",
    "dynamic_shared_mem_per_block_bytes": "launch__shared_mem_per_block_dynamic",
    "configured_stack_limit_bytes": "launch__stack_size",
    "waves_per_sm": "launch__waves_per_multiprocessor",
}

UNIT_SCALE = {
    "": 1.0,
    "byte": 1.0,
    "Kbyte": 1e3,
    "Mbyte": 1e6,
    "Gbyte": 1e9,
    "inst": 1.0,
    "register/thread": 1.0,
    "byte/block": 1.0,
    "block": 1.0,
    "ns": 1.0,
    "us": 1e3,
    "ms": 1e6,
    "s": 1e9,
}


def _number(raw: str, *, field: str) -> float:
    try:
        value = float(raw.replace(",", ""))
    except ValueError as error:
        raise ValueError(f"non-numeric NCU value {field}={raw!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite NCU value {field}={raw!r}")
    return value


def _integer(raw: str, *, field: str) -> int:
    value = _number(raw, field=field)
    if not value.is_integer():
        raise ValueError(f"non-integral launch identity {field}={raw!r}")
    return int(value)


def parse_native_csv(
    path: Path,
) -> tuple[dict[str, float], dict[str, str], dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    headers = [index for index, row in enumerate(rows) if row and row[0] == "ID"]
    if len(headers) != 1:
        raise ValueError(f"expected one NCU header: {path}")
    index = headers[0]
    header, units = rows[index : index + 2]
    values = [row for row in rows[index + 2 :] if any(row)]
    if len(values) != 1 or not (len(header) == len(units) == len(values[0])):
        raise ValueError(f"expected exactly one complete NCU launch row: {path}")
    by_name = dict(zip(header, values[0], strict=True))
    by_unit = dict(zip(header, units, strict=True))
    parsed: dict[str, float] = {}
    parsed_units: dict[str, str] = {}
    for name, metric in METRICS.items():
        raw = by_name.get(metric)
        if raw is None or raw in ("", "n/a"):
            raise ValueError(f"missing NCU metric {metric}: {path}")
        unit = by_unit[metric]
        if unit not in UNIT_SCALE:
            raise ValueError(f"unknown NCU unit {metric}={unit!r}")
        parsed[name] = _number(raw, field=metric) * UNIT_SCALE[unit]
        parsed_units[name] = unit

    required = (
        "ID",
        "Kernel Name",
        "Context",
        "Stream",
        "Device",
        "launch__block_dim_x",
        "launch__block_dim_y",
        "launch__block_dim_z",
        "launch__grid_dim_x",
        "launch__grid_dim_y",
        "launch__grid_dim_z",
    )
    missing = [name for name in required if name not in by_name]
    if missing:
        raise ValueError(f"missing NCU launch columns {missing}: {path}")
    launch = {
        "row_id": _integer(by_name["ID"], field="ID"),
        "kernel": by_name["Kernel Name"],
        "context": _integer(by_name["Context"], field="Context"),
        "stream": _integer(by_name["Stream"], field="Stream"),
        "device": _integer(by_name["Device"], field="Device"),
        "block": [
            _integer(by_name[f"launch__block_dim_{axis}"], field=f"block_{axis}")
            for axis in "xyz"
        ],
        "grid": [
            _integer(by_name[f"launch__grid_dim_{axis}"], field=f"grid_{axis}")
            for axis in "xyz"
        ],
    }
    return parsed, parsed_units, launch


def build_evidence(results: Path) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in ALL_ARMS:
        root = results / "raw" / "ncu" / arm
        native = root / "native_raw.csv"
        capture = read_json(root / "capture_identity.json")
        metrics, units, launch = parse_native_csv(native)
        launch_gate = {
            "kernel": "MoEDynamicKernel" in str(launch["kernel"]),
            "grid": launch["grid"] == list(EXPECTED_GRID),
            "block": launch["block"] == list(EXPECTED_BLOCK),
            "capture_arm": capture.get("arm") == arm,
            "native_hash": capture.get("native_raw_sha256") == file_sha256(native),
        }
        resource_gate = {
            "registers_255": metrics["registers_per_thread"] == 255,
            "allocated_registers_256": metrics["allocated_registers_per_thread"] == 256,
            "shared_total_92160": metrics["shared_mem_per_block_bytes"] == 92160,
            "shared_dynamic_91136": metrics["dynamic_shared_mem_per_block_bytes"]
            == 91136,
        }
        arms[arm] = {
            "capture_identity": capture,
            "native_raw": str(native),
            "native_raw_sha256": file_sha256(native),
            "launch": launch,
            "metrics": metrics,
            "raw_units": units,
            "launch_gate": launch_gate,
            "resource_gate": resource_gate,
            "gate_pass": all(launch_gate.values()) and all(resource_gate.values()),
        }

    normal = arms[NORMAL]["metrics"]
    cross_arm_exact_fields = (
        "local_load_bytes",
        "local_store_bytes",
        "spill_refill_instructions",
        "spill_store_instructions",
        "spill_refill_bytes",
        "spill_store_bytes",
        "tensor_instructions",
        "fp4_tensor_ops",
        "registers_per_thread",
        "allocated_registers_per_thread",
        "shared_mem_per_block_bytes",
        "dynamic_shared_mem_per_block_bytes",
        "waves_per_sm",
    )
    equality = {
        field: all(arms[arm]["metrics"][field] == normal[field] for arm in ALL_ARMS)
        for field in cross_arm_exact_fields
    }
    # The probe intentionally executes extra clock/address/store instructions,
    # so total executed warp instructions and duration are recorded but are not
    # required to equal no-marker controls.
    control_equivalence = {
        "executed_warp_instructions": arms[MEASUREMENT_CONTROL]["metrics"][
            "executed_warp_instructions"
        ]
        == normal["executed_warp_instructions"],
        "duration_recorded": all(
            arms[arm]["metrics"]["duration_ns"] > 0 for arm in ALL_ARMS
        ),
    }
    gates = {
        "all_arm_launch_resource": all(arms[arm]["gate_pass"] for arm in ALL_ARMS),
        "dynamic_work_spill_identity": all(equality.values()),
        "normal_measurement_control_equivalence": all(control_equivalence.values()),
    }
    payload = {
        "schema": "exp004.ncu-evidence.v1",
        "arms": arms,
        "cross_arm_exact_metrics": equality,
        "control_equivalence": control_equivalence,
        "gates": gates,
        "gate_pass": all(gates.values()),
        "interpretation": {
            "configured_stack_limit_bytes": (
                "runtime configured stack limit; static 488-byte frame is checked "
                "from cubin ELF/resource evidence"
            ),
            "probe_total_instruction_count": (
                "expected to differ because declared probe code executes"
            ),
        },
    }
    payload["evidence_sha256"] = canonical_sha256(payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = args.results.resolve()
    payload = build_evidence(results)
    write_json(results / "raw" / "ncu_evidence.json", payload)
    return 0 if payload["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
