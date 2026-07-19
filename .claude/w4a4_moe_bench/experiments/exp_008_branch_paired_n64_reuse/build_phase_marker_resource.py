#!/usr/bin/env python3
"""Build one marker arm's matched static/dynamic resource evidence.

Static resource and SpillRefill facts are derived from the exact cubin retained
by the standalone timing capture.  Dynamic spill/occupancy facts come from one
NCU profile whose independently compiled cubin must hash-identically match that
timing cubin.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


from build_phase_marker_evidence import SPILL_METRICS
from build_static_spill_evidence import parse_binary, run_checked
from exp008_marker_common import (
    MARKER_ARMS,
    VERSIONS,
    canonical_sha256,
    read_json,
    sha256_file,
    write_json,
)


NCU_METRICS = {
    **{metric: metric for metric in SPILL_METRICS},
    "registers_per_thread": "launch__registers_per_thread",
    "smem_bytes": "launch__shared_mem_per_block",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
}


def _parse_number(value: str, *, metric: str) -> float:
    normalized = value.strip().lower()
    if normalized in ("", "n/a", "na", "not available"):
        raise ValueError(f"missing/N-A required metric {metric}")
    parsed = float(value.replace(",", ""))
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite required metric {metric}: {value!r}")
    return parsed


def _parse_dim(value: str, *, name: str) -> list[int]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, tuple) or len(parsed) != 3:
        raise ValueError(f"invalid NCU {name}: {value!r}")
    result = [int(item) for item in parsed]
    if any(item <= 0 for item in result):
        raise ValueError(f"invalid NCU {name}: {result}")
    return result


def parse_native_raw(
    path: Path, *, expected_kernel_symbol: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    header_rows = [index for index, row in enumerate(rows) if row and row[0] == "ID"]
    if len(header_rows) != 1:
        raise ValueError(f"expected exactly one NCU raw header, got {header_rows}")
    index = header_rows[0]
    if len(rows) < index + 3:
        raise ValueError("NCU raw export lacks unit/value rows")
    header, units = rows[index : index + 2]
    value_rows = [row for row in rows[index + 2 :] if any(row)]
    if len(value_rows) != 1:
        raise ValueError(f"expected exactly one profiled launch, got {len(value_rows)}")
    values = value_rows[0]
    if not (len(header) == len(units) == len(values)):
        raise ValueError("NCU raw CSV width mismatch")
    by_name = dict(zip(header, values, strict=True))
    by_unit = dict(zip(header, units, strict=True))
    metrics: dict[str, dict[str, Any]] = {}
    for label, metric_id in NCU_METRICS.items():
        if metric_id not in by_name:
            raise ValueError(f"required NCU metric absent: {metric_id}")
        metrics[label] = {
            "metric_id": metric_id,
            "value": _parse_number(by_name[metric_id], metric=metric_id),
            "unit": by_unit[metric_id],
        }
    kernel = by_name.get("Kernel Name", "")
    if kernel != expected_kernel_symbol:
        raise ValueError(
            f"profiled kernel symbol drift: {kernel!r} != {expected_kernel_symbol!r}"
        )
    identity = {
        "row_id": int(by_name["ID"]),
        "kernel_symbol": kernel,
        "context_id": int(by_name["Context"]),
        "stream_id": int(by_name["Stream"]),
        "device_id": int(by_name["Device"]),
        "block": _parse_dim(by_name["Block Size"], name="Block Size"),
        "grid": _parse_dim(by_name["Grid Size"], name="Grid Size"),
    }
    return metrics, identity


def _unique_capture_artifact(
    capture: Mapping[str, Any], *, suffix: str
) -> Mapping[str, Any]:
    matches = [
        item
        for item in capture.get("jit_artifacts", [])
        if str(item.get("path", "")).endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one timing {suffix} artifact, got {matches}")
    return matches[0]


def _artifact_path(root: Path, artifact: Mapping[str, Any]) -> Path:
    path = (root / str(artifact["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"artifact escapes JIT root: {path}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(artifact["size"]):
        raise ValueError(f"artifact size drift: {path}")
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError(f"artifact hash drift: {path}")
    return path


def _opcode_bytes(opcode: str) -> int:
    widths = re.findall(r"(?:^|\.)(8|16|32|64|128)(?:\.|$)", opcode)
    if not widths:
        return 4
    return max(int(width) for width in widths) // 8


def _static_spill_site_footprint(histogram: Mapping[str, int]) -> tuple[int, int]:
    loads = sum(
        int(count) * _opcode_bytes(opcode)
        for opcode, count in histogram.items()
        if opcode.startswith("LDL")
    )
    stores = sum(
        int(count) * _opcode_bytes(opcode)
        for opcode, count in histogram.items()
        if opcode.startswith("STL")
    )
    return loads, stores


def build(
    *,
    timing_capture_path: Path,
    timing_jit_root: Path,
    ncu_dir: Path,
    output: Path,
    cuobjdump: str,
    nvdisasm: str,
) -> dict[str, Any]:
    timing_capture_path = timing_capture_path.resolve()
    timing_jit_root = timing_jit_root.resolve()
    ncu_dir = ncu_dir.resolve()
    output = output.resolve()
    capture = read_json(timing_capture_path)
    version = str(capture.get("version"))
    arm = str(capture.get("arm"))
    if version not in VERSIONS or arm not in MARKER_ARMS:
        raise ValueError("timing capture version/arm drift")
    if capture.get("schema") != "exp008.phase-marker-capture.v1":
        raise ValueError("timing capture schema drift")
    if not capture.get("jit_identity_gate", {}).get("gate_pass", False):
        raise ValueError("timing capture JIT gate failed")
    if canonical_sha256(capture.get("jit_artifacts", [])) != capture.get(
        "jit_identity_gate", {}
    ).get("artifact_set_sha256"):
        raise ValueError("timing capture artifact manifest drift")

    cubin_artifact = _unique_capture_artifact(capture, suffix=".cubin")
    cubin_path = _artifact_path(timing_jit_root, cubin_artifact)
    binary = parse_binary(cubin_path=cubin_path, cuobjdump=cuobjdump, nvdisasm=nvdisasm)
    kernel_symbol = str(binary["kernel_symbol"])
    static_resource = binary["resource"]
    spill = binary["compiler_spill_refill"]
    static_load_bytes, static_store_bytes = _static_spill_site_footprint(
        spill["local_sass_opcode_histogram"]
    )

    identity_path = ncu_dir / "capture_identity.json"
    target_path = ncu_dir / "target.json"
    native_path = ncu_dir / "native_raw.csv"
    ncu_identity = read_json(identity_path)
    target = read_json(target_path)
    if ncu_identity.get("schema") != "exp008.phase-marker-ncu-identity.v1":
        raise ValueError("marker NCU identity schema drift")
    metrics, launch = parse_native_raw(
        native_path, expected_kernel_symbol=kernel_symbol
    )
    source = capture["source"]
    gpu_uuid = capture["runtime"]["gpu"]["uuid"]
    checks = {
        "version": ncu_identity.get("version") == target.get("version") == version,
        "arm": ncu_identity.get("arm") == target.get("arm") == arm,
        "timing_capture_sha256": ncu_identity.get("timing_capture_sha256")
        == sha256_file(timing_capture_path),
        "timing_artifact_set": ncu_identity.get("timing_jit_artifact_set_sha256")
        == capture["jit_identity_gate"]["artifact_set_sha256"],
        "timing_cubin": ncu_identity.get("timing_cubin_sha256")
        == cubin_artifact["sha256"]
        == sha256_file(cubin_path),
        "profile_cubin_matches_timing": ncu_identity.get("profile_cubin_sha256")
        == cubin_artifact["sha256"],
        "kernel_source": ncu_identity.get("kernel_source_sha256")
        == source.get("kernel_sha256")
        == target.get("source", {}).get("kernel_sha256"),
        "dispatch_source": ncu_identity.get("dispatch_source_sha256")
        == source.get("dispatch_sha256")
        == target.get("source", {}).get("dispatch_sha256"),
        "gpu_uuid": ncu_identity.get("expected_gpu_uuid")
        == target.get("runtime", {}).get("gpu", {}).get("uuid")
        == gpu_uuid,
        "target_hash": ncu_identity.get("target_sha256") == sha256_file(target_path),
        "native_hash": ncu_identity.get("native_raw_sha256")
        == sha256_file(native_path),
        "target_gate": target.get("gate_pass") is True,
        "target_binary_identity": all(
            target.get("binary_identity_checks", {}).get(suffix) is True
            for suffix in (".cubin", ".ptx")
        ),
        "kernel_symbol": ncu_identity.get("expected_kernel_symbol")
        == kernel_symbol
        == launch["kernel_symbol"],
        "launch_geometry": launch["grid"] == [1, 1, 110]
        and launch["block"] == [288, 1, 1],
        "registers_match_static": metrics["registers_per_thread"]["value"]
        == float(static_resource["registers_per_thread"]),
        "shared_memory_numeric": metrics["smem_bytes"]["value"]
        >= float(static_resource["static_shared_bytes_per_cta"]),
        "compiler_annotation_closure": spill["annotation_exactly_matches_local_sass"]
        is True,
    }
    gate_pass = all(checks.values())
    resource = {
        "registers_per_thread": metrics["registers_per_thread"]["value"],
        "smem_bytes": metrics["smem_bytes"]["value"],
        "stack_bytes_per_thread": static_resource["stack_bytes_per_thread"],
        "static_spill_load_bytes": static_load_bytes,
        "static_spill_store_bytes": static_store_bytes,
        "compiler_spillrefill_sass": spill["annotation_count"],
        "achieved_occupancy_pct": metrics["achieved_occupancy_pct"]["value"],
        "dynamic_spill_metrics": {
            metric: metrics[metric]["value"] for metric in SPILL_METRICS
        },
    }
    payload = {
        "schema": "exp008.marker-resource-evidence.v1",
        "version": version,
        "arm": arm,
        "identity": {
            "kernel_source_sha256": source["kernel_sha256"],
            "dispatch_source_sha256": source["dispatch_sha256"],
            "jit_artifact_set_sha256": capture["jit_identity_gate"][
                "artifact_set_sha256"
            ],
            "cubin_sha256": cubin_artifact["sha256"],
            "kernel_symbol": kernel_symbol,
            "static_kernel_symbol": kernel_symbol,
            "ncu_kernel_symbol": launch["kernel_symbol"],
            "gpu_uuid": gpu_uuid,
        },
        "resource": resource,
        "static": {
            "timing_cubin_path": str(cubin_path),
            "resource": static_resource,
            "compiler_spill_refill": spill,
            "spill_site_footprint_definition": (
                "sum of encoded LDL/STL operand widths over static compiler "
                "SpillRefill SASS sites; not a dynamic byte count"
            ),
        },
        "ncu": {
            "metrics": metrics,
            "launch": launch,
            "capture_identity": ncu_identity,
        },
        "checks": checks,
        "gate_pass": gate_pass,
    }
    write_json(output, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-capture", type=Path, required=True)
    parser.add_argument("--timing-jit-root", type=Path, required=True)
    parser.add_argument("--ncu-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuobjdump", default="/usr/local/cuda/bin/cuobjdump")
    parser.add_argument("--nvdisasm", default="/usr/local/cuda/bin/nvdisasm")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_checked([args.cuobjdump, "--version"])
    run_checked([args.nvdisasm, "--version"])
    result = build(
        timing_capture_path=args.timing_capture,
        timing_jit_root=args.timing_jit_root,
        ncu_dir=args.ncu_dir,
        output=args.output,
        cuobjdump=args.cuobjdump,
        nvdisasm=args.nvdisasm,
    )
    print(
        json.dumps(
            {
                "version": result["version"],
                "arm": result["arm"],
                "gate_pass": result["gate_pass"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
