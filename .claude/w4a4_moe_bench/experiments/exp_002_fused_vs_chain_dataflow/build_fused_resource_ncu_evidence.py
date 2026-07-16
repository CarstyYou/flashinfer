#!/usr/bin/env python3
"""Validate and compact the M8192 fused deep-resource NCU follow-up."""

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CASE = RESULTS / "ncu/m8192/cutedsl_bf16_fused/deep-resource_launch_1_r2"
CANONICAL = RESULTS / "ncu/m8192/cutedsl_bf16_fused/deep_launch_1"
EXPECTED_KERNEL = "MoEDynamicKernel"
EXPECTED_GRID = [1, 1, 110]
EXPECTED_BLOCK = [160, 1, 1]


METRICS = {
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "issue_active_pct": "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
    "issue_active_active_cycle_pct": (
        "smsp__issue_active.avg.pct_of_peak_sustained_active"
    ),
    "compute_memory_throughput_pct": (
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
    ),
    "tensor_pipe_active_pct": (
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
    ),
    "tensor_pipe_active_active_cycle_pct": (
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
    ),
    "tensor_hmma_qmma_omma_pipe_active_pct": (
        "sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed"
    ),
    "tensor_hmma_qmma_omma_pipe_active_active_cycle_pct": (
        "sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_active"
    ),
    "tensor_imma_pipe_active_pct": (
        "sm__pipe_tensor_subpipe_imma_cycles_active.avg.pct_of_peak_sustained_elapsed"
    ),
    "alu_pipe_active_pct": (
        "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed"
    ),
    "alu_pipe_active_active_cycle_pct": (
        "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active"
    ),
    "fma_pipe_active_pct": (
        "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed"
    ),
    "fma_pipe_active_active_cycle_pct": (
        "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"
    ),
    "aluheavy_pipe_active_pct": (
        "sm__pipe_aluheavy_cycles_active.avg.pct_of_peak_sustained_elapsed"
    ),
    "fmaheavy_pipe_active_pct": (
        "sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_elapsed"
    ),
    "xu_executed_pipe_utilization_pct": (
        "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed"
    ),
    "xu_executed_pipe_utilization_active_cycle_pct": (
        "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active"
    ),
    "dram_throughput_pct": ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
    "l2_throughput_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex_throughput_active_pct": (
        "l1tex__throughput.avg.pct_of_peak_sustained_active"
    ),
    "l1tex_lsu_data_pipe_utilization_pct": (
        "l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed"
    ),
    "lsu_executed_pipe_utilization_pct": (
        "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed"
    ),
    "lsu_executed_pipe_utilization_active_cycle_pct": (
        "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active"
    ),
    "tma_pipe_active_pct": (
        "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed"
    ),
    "tma_pipe_active_active_cycle_pct": (
        "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_active"
    ),
    "theoretical_occupancy_pct": "sm__maximum_warps_per_active_cycle_pct",
    "achieved_occupancy_pct": ("sm__warps_active.avg.pct_of_peak_sustained_active"),
    "active_warps_per_sm": "sm__warps_active.avg.per_cycle_active",
    "active_warps_per_smsp": "smsp__warps_active.avg.per_cycle_active",
    "eligible_warps_per_cycle": "smsp__warps_eligible.avg.per_cycle_active",
    "warp_latency_per_issued_instruction": (
        "smsp__average_warp_latency_per_inst_issued.ratio"
    ),
    "dynamic_spill_refill_instructions": (
        "sass__inst_executed_register_spilling_op_read"
    ),
    "dynamic_spill_store_instructions": (
        "sass__inst_executed_register_spilling_op_write"
    ),
    "configured_stack_size": "launch__stack_size",
}

STALL_REASONS = (
    "barrier",
    "branch_resolving",
    "dispatch_stall",
    "drain",
    "lg_throttle",
    "long_scoreboard",
    "math_pipe_throttle",
    "membar",
    "mio_throttle",
    "misc",
    "no_instruction",
    "not_selected",
    "selected",
    "short_scoreboard",
    "sleeping",
    "tex_throttle",
    "wait",
)


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_sidecar(path, sidecar):
    recorded = sidecar.read_text().split()[0]
    actual = sha256(path)
    if recorded != actual:
        raise ValueError("SHA-256 mismatch: {}".format(path))
    return actual


def metric_value(by_name, name):
    item = by_name.get(name)
    if item is None:
        raise ValueError("missing metric {}".format(name))
    value = item.get("value")
    if not isinstance(value, (int, float)):
        raise ValueError("non-numeric metric {}".format(name))
    return {"metric": name, "value": value, "unit": item.get("unit")}


def main():
    report_sha = validate_sidecar(CASE / "trace.ncu-rep", CASE / "trace.ncu-rep.sha256")
    validate_sidecar(
        CASE / "profile_manifest.json", CASE / "profile_manifest.json.sha256"
    )
    validate_sidecar(CASE / "capture_ncu.sh", CASE / "capture_ncu.sh.sha256")

    inspect = read_json(CASE / "veloq/inspect.json")
    rows = inspect.get("data", {}).get("rows", [])
    if len(rows) != 1:
        raise ValueError("expected one NCU launch")
    launch = rows[0]
    if EXPECTED_KERNEL not in launch.get("kernel_demangled", ""):
        raise ValueError("unexpected fused kernel")
    if launch.get("grid_size") != EXPECTED_GRID:
        raise ValueError("grid drift")
    if launch.get("block_size") != EXPECTED_BLOCK:
        raise ValueError("block drift")

    canonical_launch = read_json(CANONICAL / "veloq/launches.json")["data"]["rows"][0]
    for field in ("kernel_demangled", "grid_size", "block_size"):
        if launch[field] != canonical_launch[field]:
            raise ValueError("canonical dispatch drift at {}".format(field))

    by_name = {item["name"]: item for item in launch.get("metrics", [])}
    metrics = {key: metric_value(by_name, name) for key, name in METRICS.items()}
    stalls = {}
    for reason in STALL_REASONS:
        name = "smsp__average_warps_issue_stalled_{}_per_issue_active.ratio".format(
            reason
        )
        stalls[reason] = metric_value(by_name, name)
        # NCU exports this ratio with the raw unit label ``inst``.  Its metric
        # contract is warp cycles per issued instruction, so keep the raw label
        # for provenance while exposing the semantic unit to evidence readers.
        stalls[reason]["raw_ncu_unit"] = stalls[reason]["unit"]
        stalls[reason]["unit"] = "warp cycle / issued instruction"

    warp_latency = metrics["warp_latency_per_issued_instruction"]["value"]
    if warp_latency <= 0:
        raise ValueError("non-positive warp latency")
    stall_pct = {}
    for reason, item in stalls.items():
        stall_pct[reason] = {
            "authority": "stall ratio / total warp latency",
            "numerator_metric": item["metric"],
            "denominator_metric": METRICS["warp_latency_per_issued_instruction"],
            "value": 100.0 * item["value"] / warp_latency,
            "unit": "% of active-warp cycles",
        }
    pct_sum = sum(item["value"] for item in stall_pct.values())
    if abs(pct_sum - 100.0) > 0.1:
        raise ValueError("stall percentages do not close: {}".format(pct_sum))

    cubins = list((CASE / "trace.ncu-rep.veloq/disasm").glob("*.cubin"))
    canonical_cubins = list((CANONICAL / "trace.ncu-rep.veloq/disasm").glob("*.cubin"))
    if len(cubins) != 1 or len(canonical_cubins) != 1:
        raise ValueError("expected one cubin in each profile")
    cubin_sha = sha256(cubins[0])
    canonical_cubin_sha = sha256(canonical_cubins[0])
    if cubin_sha != canonical_cubin_sha:
        raise ValueError("canonical cubin drift")

    resource_text = (CASE / "binary/resource_usage.txt").read_text()
    match = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", resource_text)
    if not match:
        raise ValueError("cannot parse cuobjdump resource usage")
    registers, stack_frame, static_shared, static_local = map(int, match.groups())

    elf_text = (CASE / "binary/elf.txt").read_text()
    frame_match = re.search(r"frame size: 0x([0-9a-fA-F]+)", elf_text)
    stack_match = re.search(r"min stack size: 0x([0-9a-fA-F]+)", elf_text)
    if not frame_match or not stack_match:
        raise ValueError("missing ELF stack attributes")
    elf_frame = int(frame_match.group(1), 16)
    elf_min_stack = int(stack_match.group(1), 16)
    if stack_frame != elf_frame or stack_frame != elf_min_stack:
        raise ValueError("stack frame evidence disagrees")

    annotations = {
        int(value, 16)
        for value in re.findall(
            r"SpillRefill\s*:\s*Offset\s*:\s*0x([0-9a-fA-F]+)", elf_text
        )
    }
    sass_text = (CASE / "binary/sass.txt").read_text()
    opcodes = {
        int(offset, 16): opcode
        for offset, opcode in re.findall(
            r"/\*([0-9a-fA-F]+)\*/\s+([A-Z][A-Z0-9.]*)", sass_text
        )
    }
    local_opcodes = {
        offset: opcode
        for offset, opcode in opcodes.items()
        if opcode.startswith("LDL") or opcode.startswith("STL")
    }
    if annotations != set(local_opcodes):
        raise ValueError("SpillRefill annotations do not match local SASS")
    static_counts = Counter(local_opcodes.values())
    refill_width = sum(
        8 if ".64" in opcode else 4
        for opcode in local_opcodes.values()
        if opcode.startswith("LDL")
    )
    store_width = sum(
        8 if ".64" in opcode else 4
        for opcode in local_opcodes.values()
        if opcode.startswith("STL")
    )
    if refill_width != stack_frame or store_width != stack_frame:
        raise ValueError("static spill width does not cover the stack frame")

    followup = read_json(CASE / "followup_validation.json")
    profile = read_json(CASE / "profile_manifest.json")
    canonical_profile_paths = (
        CANONICAL / "profile_manifest.json",
        RESULTS
        / "ncu/m8192/cutedsl_bf16_fused/operator-ledger-v2/profile_manifest.json",
    )
    canonical_output_hashes = sorted(
        {read_json(path)["output_sha256"] for path in canonical_profile_paths}
    )
    output_hash_stable = len(canonical_output_hashes) == 1

    evidence = {
        "schema": "exp002.fused-resource-ncu-evidence.v1",
        "status": "vetted_with_profile_overlay_drift_disclosed",
        "scope": {"m": 8192, "arm": "cutedsl_bf16_fused"},
        "report_sha256": report_sha,
        "cubin_sha256": cubin_sha,
        "canonical_cubin_sha256": canonical_cubin_sha,
        "dispatch": {
            "kernel_demangled": launch["kernel_demangled"],
            "grid": launch["grid_size"],
            "block": launch["block_size"],
        },
        "metrics": metrics,
        "stall_cycles_per_issued_instruction": stalls,
        "stall_pct_of_active_warp_cycles": stall_pct,
        "binary_resources": {
            "registers_per_thread": registers,
            "actual_stack_frame_bytes_per_thread": stack_frame,
            "minimum_stack_bytes_per_thread": elf_min_stack,
            "static_shared_bytes": static_shared,
            "static_local_bytes_outside_stack": static_local,
            "spillrefill_annotation_count": len(annotations),
            "static_spillrefill_opcodes": dict(sorted(static_counts.items())),
            "static_refill_width_bytes_per_lane": refill_width,
            "static_store_width_bytes_per_lane": store_width,
        },
        "profile_overlay_drift": followup,
        "output_hash_audit": {
            "followup_output_sha256": profile["output_sha256"],
            "canonical_profile_output_sha256_values": canonical_output_hashes,
            "canonical_exact_output_hash_is_stable": output_hash_stable,
            "interpretation": (
                "Exact output SHA is not an equivalence gate because two canonical "
                "profiles already differ; fixture/weights/JIT prerequisites, output "
                "shape-dtype-finite validation, exact cubin, and dispatch are the "
                "accepted diagnostic-profile gates."
            ),
        },
        "non_additivity": [
            "Tensor total contains its tensor subpipes.",
            "ALU/FMA aggregates and ALUHeavy/FMAHeavy drill-downs are not additive.",
            "Compute-memory, DRAM, L2, L1TEX, LSU, and TMA utilization are observation "
            "points, not additive traffic.",
            "The selected stall reason is a non-stall baseline.",
        ],
    }

    derived = CASE / "derived"
    derived.mkdir(exist_ok=True)
    (derived / "fused_resource_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )

    rows = []
    for key, item in sorted(metrics.items()):
        rows.append(("metric", key, item["metric"], item["value"], item["unit"] or ""))
    for key, item in sorted(stalls.items()):
        rows.append(("stall", key, item["metric"], item["value"], item["unit"] or ""))
    for key, item in sorted(stall_pct.items()):
        rows.append(("stall_pct", key, item["authority"], item["value"], item["unit"]))
    rows.extend(
        [
            (
                "binary",
                "actual_stack_frame_bytes_per_thread",
                "cuobjdump/ELF",
                stack_frame,
                "B/thread",
            ),
            (
                "binary",
                "static_spill_refill_instructions",
                "ELF SpillRefill + SASS",
                sum(static_counts.values()),
                "static inst",
            ),
            (
                "binary",
                "static_spill_store_instructions",
                "ELF SpillRefill + SASS",
                sum(v for k, v in static_counts.items() if k.startswith("STL")),
                "static inst",
            ),
            (
                "binary",
                "static_spill_refill_load_instructions",
                "ELF SpillRefill + SASS",
                sum(v for k, v in static_counts.items() if k.startswith("LDL")),
                "static inst",
            ),
        ]
    )
    with (derived / "fused_resource_evidence.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("category", "field", "authority", "value", "unit"))
        writer.writerows(rows)

    print(derived / "fused_resource_evidence.json")


if __name__ == "__main__":
    main()
