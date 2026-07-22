#!/usr/bin/env python3
"""Build the CUTLASS Chain per-op performance ceiling card from exp_002 evidence."""

import csv
import hashlib
import json
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
EXPERIMENTS = EXP_DIR.parent
EXP002 = EXPERIMENTS / "exp_002_fused_vs_chain_dataflow"
EXP026 = EXPERIMENTS / "exp_026_5kp_ceiling_calibration"
RESULTS = EXP_DIR / "results"

BENCHMARK = EXP002 / "results/benchmark_summary.csv"
DEEP = EXP002 / "results/ncu/deep_launch_metrics.json"
TRAFFIC = EXP002 / "results/ncu/operator_traffic_v2.json"
EXP002_RUNTIME = EXP002 / "results/manifests/runtime.json"
CALIBRATION_PROFILE = EXP026 / "results/profile.json"
NSYS_GRAPHS = {
    m: EXP002 / "results/nsys/m{}/cutlass_bf16_chain/veloq/graph_replays.json".format(m)
    for m in (256, 8192)
}
NSYS_KERNELS = {
    m: EXP002 / "results/nsys/m{}/cutlass_bf16_chain/veloq/kernels.json".format(m)
    for m in (256, 8192)
}

ARM = "cutlass_bf16_chain"
CASES = (256, 8192)
BENCHMARK_CASES = (256, 1024, 8192)
HIDDEN = 2048
INTERMEDIATE = 512
TOPK = 8

# NVIDIA RTX Blackwell GPU Architecture v1.1, Table 4. The dense FP4
# figures imply 4096 FLOP/SM/cycle; the second number is sparse and is not used.
OFFICIAL_ARCH_URL = (
    "https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/"
    "quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf"
)
FP4_FLOP_PER_SM_CYCLE = 4096

# Device attributes extracted read-only from the accepted M8192 FC1 NCU report:
# CC 12.0, 110 SM, SM clock 2,377 MHz, memory clock 14,001 MHz, 384-bit bus.
DEVICE_SM_COUNT = 110
DEVICE_SM_CLOCK_HZ = 2_377_000_000
DEVICE_MEMORY_CLOCK_HZ = 14_001_000_000
DEVICE_MEMORY_BUS_BITS = 384
FP4_PEAK_TFLOPS = DEVICE_SM_COUNT * DEVICE_SM_CLOCK_HZ * FP4_FLOP_PER_SM_CYCLE / 1.0e12
DRAM_THEORETICAL_GBS = 2 * DEVICE_MEMORY_CLOCK_HZ * (DEVICE_MEMORY_BUS_BITS / 8) / 1.0e9

NVFP4_CALIBRATION_ID = "nvfp4-e2m1-vs16"

PHASE_LABELS = {
    "expand_quant": "Route/Q0/Pack",
    "fc1": "FC1",
    "activation_requant": "SwiGLU/Q1",
    "fc2": "FC2",
    "finalize": "Finalize",
}

DISPLAY_PHASES = (
    "prefix",
    "expand_quant",
    "gemm_metadata",
    "fc1",
    "activation_requant",
    "fc2",
    "finalize",
)


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def interval_union_ns(nodes):
    intervals = sorted((int(node["start_ns"]), int(node["end_ns"])) for node in nodes)
    total = 0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def load_timeline_case(m, coverage, targets):
    graph = load_json(NSYS_GRAPHS[m])
    require(graph["command"] == "nsys.graph-replays", "unexpected NSys graph command")
    require(graph["data"]["count"] == 1, "expected one accepted CUTLASS graph replay")
    replay = graph["data"]["rows"][0]
    require(replay["capture_mode"] == "graph_nodes", "CUTLASS capture mode drift")
    require(
        replay["event_count"] == replay["kernel_count"] == 9, "CUTLASS node-count drift"
    )
    nodes = sorted(
        replay["top_nodes"], key=lambda item: (item["start_ns"], item["end_ns"])
    )
    require(
        all(node["kind"] == "kernel" and node["count"] == 1 for node in nodes),
        "non-kernel graph node",
    )
    require(
        [node["name"] for node in nodes[:3]]
        == [
            "blockExpertPrefixSumKernel",
            "globalExpertPrefixSumKernel"
            if m == 256
            else "globalExpertPrefixSumLargeKernel",
            "mergeExpertPrefixSumKernel",
        ],
        "prefix topology drift",
    )
    require(
        nodes[4]["name"] == "computeStridesTmaWarpSpecializedKernel",
        "metadata topology drift",
    )
    require(
        interval_union_ns(nodes) == int(coverage["all_kernel_active_union_ns"]),
        "active-union drift",
    )
    require(
        int(replay["busy_ns"]) == int(coverage["all_kernel_active_union_ns"]),
        "busy-time drift",
    )

    material_nodes = []
    for phase in PHASE_LABELS:
        target = targets[(m, phase)]
        node = nodes[int(target["nsys_original_launch_skip"])]
        require(
            int(node["wall_ns"]) == int(target["nsys_duration_ns"]),
            phase + " NSys duration drift",
        )
        material_nodes.append(node)
    require(
        interval_union_ns(material_nodes) == int(coverage["selected_active_union_ns"]),
        "material active-union drift",
    )

    prefix_nodes = nodes[:3]
    prefix_union_ns = interval_union_ns(prefix_nodes)
    prefix_internal_overlap_ns = (
        sum(int(node["wall_ns"]) for node in prefix_nodes) - prefix_union_ns
    )
    metadata_ns = int(nodes[4]["wall_ns"])
    material_sum_ns = sum(
        int(targets[(m, phase)]["nsys_duration_ns"]) for phase in PHASE_LABELS
    )
    category_sum_ns = prefix_union_ns + metadata_ns + material_sum_ns
    active_union_ns = int(coverage["all_kernel_active_union_ns"])
    cross_category_overlap_ns = category_sum_ns - active_union_ns
    require(cross_category_overlap_ns >= 0, "negative cross-category overlap")
    require(
        prefix_internal_overlap_ns + cross_category_overlap_ns
        == int(replay["sum_gpu_ns"]) - active_union_ns,
        "PDL overlap accounting drift",
    )
    return {
        "active_union_ns": active_union_ns,
        "prefix_union_ns": prefix_union_ns,
        "prefix_internal_overlap_ns": prefix_internal_overlap_ns,
        "gemm_metadata_ns": metadata_ns,
        "cross_category_overlap_ns": cross_category_overlap_ns,
        "category_sum_ns": category_sum_ns,
        "category_share_sum_percent": 100.0 * category_sum_ns / active_union_ns,
        "kernel_wall_ns": int(replay["wall_ns"]),
        "kernel_idle_ns": int(replay["idle_inside_replay_ns"]),
    }


def useful_ops(m, phase):
    rows = m * TOPK
    factor = 4 if phase == "fc1" else 2
    return rows * factor * HIDDEN * INTERMEDIATE


def logical_payload(m, phase):
    rows = m * TOPK
    if phase == "expand_quant":
        values = rows * HIDDEN
        return {
            "work_name": "quantized values",
            "work_count": values,
            "payload_floor_bytes": values * (2 + 0.5 + 1 / 16),
            "payload_scope": "routed BF16 read + NVFP4 value/scale write; metadata excluded",
        }
    if phase == "activation_requant":
        outputs = rows * INTERMEDIATE
        return {
            "work_name": "SwiGLU output values",
            "work_count": outputs,
            "payload_floor_bytes": outputs * (4 + 0.5 + 1 / 16),
            "payload_scope": "Gate+Up BF16 read + NVFP4 value/scale write",
        }
    if phase == "finalize":
        outputs = m * HIDDEN
        return {
            "work_name": "output values",
            "work_count": outputs,
            "payload_floor_bytes": rows * HIDDEN * 2 + outputs * 2 + rows * 4,
            "payload_scope": "routed BF16 read + BF16 output + route weight; metadata excluded",
        }
    raise ValueError("payload requested for non-payload phase")


def build():
    calibration = load_json(CALIBRATION_PROFILE)
    require(calibration["decision"] == "accept", "exp_026 calibration is not accepted")
    require(calibration["gpu"]["internal_sku"] == "RTX 5KP", "calibration SKU drift")
    require(
        int(calibration["gpu"]["sm_count"]) == DEVICE_SM_COUNT,
        "calibration SM-count drift",
    )
    tensor_records = {row["id"]: row for row in calibration["tensor_core_records"]}
    require(NVFP4_CALIBRATION_ID in tensor_records, "NVFP4 calibration record missing")
    nvfp4_calibration = tensor_records[NVFP4_CALIBRATION_ID]
    require(
        nvfp4_calibration["audit_status"] == "vetted", "NVFP4 calibration is not vetted"
    )
    require(
        nvfp4_calibration["instruction"] == "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X",
        "NVFP4 instruction-contract drift",
    )
    calibrated_nvfp4_tflops = float(
        nvfp4_calibration["sustained_window"]["sustained_window_tflops"]
    )
    require(calibrated_nvfp4_tflops > 0, "invalid calibrated NVFP4 roof")

    dram_record = calibration["dram_record"]
    require(dram_record["audit_status"] == "vetted", "DRAM calibration is not vetted")
    dram_roofs = {
        key: float(dram_record["physical_gbs"][key])
        for key in ("read", "write", "copy")
    }
    require(
        all(value > 0 for value in dram_roofs.values()), "invalid physical DRAM roof"
    )

    exp002_runtime = load_json(EXP002_RUNTIME)
    exp002_gpu_uuid = exp002_runtime["gpu_uuid"]
    calibration_gpu_uuid = calibration["gpu"]["uuid"]
    require(
        exp002_gpu_uuid != calibration_gpu_uuid, "expected cross-UUID ceiling transfer"
    )
    require(
        exp002_runtime["compute_capability"] == [12, 0], "exp_002 architecture drift"
    )

    benchmark_rows = load_csv(BENCHMARK)
    benchmark = {
        int(row["m"]): row
        for row in benchmark_rows
        if row["arm"] == ARM and int(row["m"]) in BENCHMARK_CASES
    }
    require(set(benchmark) == set(BENCHMARK_CASES), "missing CUTLASS benchmark rows")
    require(len({row["rerun_id"] for row in benchmark.values()}) == 1, "rerun drift")
    require(
        len({row["environment_lock_digest"] for row in benchmark.values()}) == 1,
        "environment drift",
    )
    require(
        len({row["protocol_lock_digest"] for row in benchmark.values()}) == 1,
        "protocol drift",
    )

    deep = load_json(DEEP)
    targets = {
        (int(row["m"]), row["phase"]): row
        for row in deep["targets"]
        if row["arm"] == ARM and int(row["m"]) in CASES
    }
    coverage = {
        int(row["m"]): row
        for row in deep["coverage"]
        if row["arm"] == ARM and int(row["m"]) in CASES
    }
    expected_phases = set(PHASE_LABELS)
    for m in CASES:
        require(
            {phase for case, phase in targets if case == m} == expected_phases,
            "M{} material phase set mismatch".format(m),
        )
        require(m in coverage, "M{} coverage missing".format(m))

    traffic = load_json(TRAFFIC)
    traffic_cases = {
        int(row["m"]): row
        for row in traffic["cases"]
        if row["arm"] == ARM and int(row["m"]) in CASES
    }
    require(set(traffic_cases) == set(CASES), "operator traffic cases missing")

    cases = []
    for m in CASES:
        timeline = load_timeline_case(m, coverage[m], targets)
        union_ns = timeline["active_union_ns"]

        phase_rows = []
        executed_sum = 0
        physical_rows = []
        for phase in PHASE_LABELS:
            target = targets[(m, phase)]
            duration_ns = int(target["nsys_duration_ns"])
            metrics = target["metrics"]
            row = {
                "phase": phase,
                "label": PHASE_LABELS[phase],
                "time_us": duration_ns / 1000.0,
                "share_percent": 100.0 * duration_ns / union_ns,
                "timing_class": "paired cross-capture / canonical NSys",
                "sota_efficiency_percent": None,
                "sota_reason": "no independent contract-equivalent per-op implementation",
            }

            ncu_duration_ns = float(metrics["ncu_duration_ns"])
            dram_read_bytes = float(metrics["dram_read_bytes"])
            dram_write_bytes = float(metrics["dram_write_bytes"])
            dram_bytes = float(metrics["dram_total_bytes"])
            require(
                abs((dram_read_bytes + dram_write_bytes) - dram_bytes) < 1.0,
                phase + " DRAM byte accounting drift",
            )
            dram_gbs = dram_bytes / ncu_duration_ns
            dram_floor_ns = max(
                dram_read_bytes / dram_roofs["read"],
                dram_write_bytes / dram_roofs["write"],
                dram_bytes / dram_roofs["copy"],
            )
            dram_resource_efficiency_percent = 100.0 * dram_floor_ns / ncu_duration_ns
            dram_resource_status = (
                "diagnostic" if dram_resource_efficiency_percent <= 100.0 else "invalid"
            )
            row["ncu_diagnostic"] = {
                "dram_read_bytes": dram_read_bytes,
                "dram_write_bytes": dram_write_bytes,
                "dram_bytes": dram_bytes,
                "ncu_duration_us": ncu_duration_ns / 1000.0,
                "dram_gbs": dram_gbs,
                "theoretical_dram_efficiency_percent": 100.0
                * dram_gbs
                / DRAM_THEORETICAL_GBS,
                "physical_dram_floor_ns": dram_floor_ns,
                "physical_dram_resource_efficiency_percent": dram_resource_efficiency_percent,
                "physical_dram_resource_status": dram_resource_status,
                "physical_dram_roof_transfer": "same-SKU cross-UUID transfer",
                "physical_dram_formula": (
                    "max(read_bytes/read_roof, write_bytes/write_roof, "
                    "total_bytes/copy_roof) / NCU duration"
                ),
                "scope": "NCU kernel replay/cache-control all; not canonical NSys throughput",
            }

            if phase in ("fc1", "fc2"):
                useful = useful_ops(m, phase)
                executed = int(metrics["fp4_to_fp32_tensor_ops"])
                executed_sum += executed
                factor = 4 if phase == "fc1" else 2
                rows_physical = executed / (factor * HIDDEN * INTERMEDIATE)
                require(rows_physical.is_integer(), "non-integral physical routed rows")
                physical_rows.append(int(rows_physical))
                useful_tflops = useful / duration_ns / 1000.0
                executed_tflops = executed / duration_ns / 1000.0
                row.update(
                    {
                        "model": "dense NVFP4 tensor-core compute",
                        "useful_flops": useful,
                        "executed_flops": executed,
                        "physical_routed_rows": int(rows_physical),
                        "padding_efficiency_percent": 100.0 * useful / executed,
                        "useful_tflops": useful_tflops,
                        "executed_tflops": executed_tflops,
                        "calibrated_useful_ceiling_efficiency_percent": (
                            100.0 * useful_tflops / calibrated_nvfp4_tflops
                        ),
                        "calibrated_executed_ceiling_efficiency_percent": (
                            100.0 * executed_tflops / calibrated_nvfp4_tflops
                        ),
                        "nominal_useful_mfu_percent": 100.0
                        * useful_tflops
                        / FP4_PEAK_TFLOPS,
                        "nominal_executed_mfu_percent": 100.0
                        * executed_tflops
                        / FP4_PEAK_TFLOPS,
                        "hardware_ceiling_status": "vetted measured roof / same-SKU cross-UUID transfer",
                        "tensor_pipe_active_percent": float(
                            metrics["tensor_active_pct"]
                        ),
                    }
                )
            else:
                payload = logical_payload(m, phase)
                work_per_second_giga = (
                    payload["work_count"] / (duration_ns * 1.0e-9) / 1.0e9
                )
                logical_gbs = (
                    payload["payload_floor_bytes"] / (duration_ns * 1.0e-9) / 1.0e9
                )
                row.update(
                    {
                        "model": {
                            "expand_quant": "route/quant/transform",
                            "activation_requant": "elementwise/quant/transform",
                            "finalize": "gather/reduction/store",
                        }[phase],
                        "logical_work": payload,
                        "achieved_gwork_per_s": work_per_second_giga,
                        "logical_payload_gbs": logical_gbs,
                        "dram_resource_efficiency_percent": dram_resource_efficiency_percent,
                        "dram_resource_efficiency_status": dram_resource_status,
                        "hardware_ceiling_status": "partial DRAM resource view only",
                        "hardware_ceiling_reason": (
                            "physical DRAM bytes and calibrated directional roofs do not model "
                            "the complete mixed operator"
                        ),
                    }
                )
            phase_rows.append(row)

        require(len(set(physical_rows)) == 1, "FC1/FC2 physical row mismatch")
        require(
            executed_sum == int(traffic_cases[m]["metrics"]["fp4_to_fp32_tensor_ops"]),
            "FC1+FC2 work does not close operator counter",
        )
        phase_rows.extend(
            [
                {
                    "phase": "prefix",
                    "label": "Prefix",
                    "time_us": timeline["prefix_union_ns"] / 1000.0,
                    "share_percent": 100.0 * timeline["prefix_union_ns"] / union_ns,
                    "model": "routing prefix/latency",
                    "hardware_ceiling_status": "unavailable",
                    "sota_efficiency_percent": None,
                },
                {
                    "phase": "gemm_metadata",
                    "label": "GEMM metadata",
                    "time_us": timeline["gemm_metadata_ns"] / 1000.0,
                    "share_percent": 100.0 * timeline["gemm_metadata_ns"] / union_ns,
                    "model": "metadata/latency",
                    "hardware_ceiling_status": "unavailable",
                    "sota_efficiency_percent": None,
                },
            ]
        )
        require(
            abs(
                sum(row["share_percent"] for row in phase_rows)
                - timeline["category_share_sum_percent"]
            )
            < 1.0e-9,
            "non-exclusive share accounting drift",
        )
        cases.append(
            {
                "m": m,
                "logical_routed_rows": m * TOPK,
                "physical_routed_rows": physical_rows[0],
                "overall_benchmark_us": float(benchmark[m]["median_us"]),
                "timeline_active_union_us": union_ns / 1000.0,
                "timeline_vs_benchmark_percent": 100.0
                * (union_ns / 1000.0)
                / float(benchmark[m]["median_us"]),
                "timeline_accounting": timeline,
                "phases": phase_rows,
            }
        )

    return {
        "schema": "operator-performance-ceiling.v1",
        "subject": {
            "arm": ARM,
            "boundary": "BF16 input -> native CUTLASS online NVFP4 chain -> BF16 output",
            "shape": {
                "experts": 256,
                "hidden": HIDDEN,
                "intermediate_tp": INTERMEDIATE,
                "topk": TOPK,
            },
        },
        "verdict": {
            "status": "accept",
            "reason": (
                "FC1/FC2 consume a vetted same-SKU measured NVFP4 roof; mixed non-GEMM "
                "complete roofs and all independent SOTA anchors remain unavailable"
            ),
        },
        "hardware_authority": {
            "calibrated_profile": {
                "path": str(CALIBRATION_PROFILE),
                "sha256": sha256(CALIBRATION_PROFILE),
                "profile_id": calibration["profile_id"],
                "record_id": NVFP4_CALIBRATION_ID,
                "instruction": nvfp4_calibration["instruction"],
                "audit_status": nvfp4_calibration["audit_status"],
                "calibration_gpu_uuid": calibration_gpu_uuid,
                "consumer_gpu_uuid": exp002_gpu_uuid,
                "transfer_scope": "same RTX 5KP SKU and SM count; different GPU UUID",
                "calibrated_nvfp4_sustained_window_tflops": calibrated_nvfp4_tflops,
                "physical_dram_gbs": dram_roofs,
            },
            "official_architecture": {
                "url": OFFICIAL_ARCH_URL,
                "table": "Table 4 dense FP4 value; sparse second value excluded",
                "fp4_flop_per_sm_cycle": FP4_FLOP_PER_SM_CYCLE,
            },
            "device_attributes": {
                "source_report": str(
                    EXP002
                    / "results/ncu/m8192/cutlass_bf16_chain/deep_launch_5/trace.ncu-rep"
                ),
                "extraction": "veloq ncu metrics --counter device__attribute_*",
                "compute_capability": "12.0",
                "sm_count": DEVICE_SM_COUNT,
                "sm_clock_hz": DEVICE_SM_CLOCK_HZ,
                "memory_clock_hz": DEVICE_MEMORY_CLOCK_HZ,
                "memory_bus_bits": DEVICE_MEMORY_BUS_BITS,
            },
            "nominal_dense_nvfp4_peak_tflops": FP4_PEAK_TFLOPS,
            "authority_status": "secondary nominal reference; calibrated profile is primary",
            "theoretical_dram_gbs": DRAM_THEORETICAL_GBS,
        },
        "inputs": {
            "benchmark": {"path": str(BENCHMARK), "sha256": sha256(BENCHMARK)},
            "deep_launch": {"path": str(DEEP), "sha256": sha256(DEEP)},
            "operator_traffic": {"path": str(TRAFFIC), "sha256": sha256(TRAFFIC)},
            "exp002_runtime": {
                "path": str(EXP002_RUNTIME),
                "sha256": sha256(EXP002_RUNTIME),
            },
            "calibration_profile": {
                "path": str(CALIBRATION_PROFILE),
                "sha256": sha256(CALIBRATION_PROFILE),
            },
            "nsys_graph_replays": {
                str(m): {"path": str(NSYS_GRAPHS[m]), "sha256": sha256(NSYS_GRAPHS[m])}
                for m in CASES
            },
            "nsys_kernels": {
                str(m): {
                    "path": str(NSYS_KERNELS[m]),
                    "sha256": sha256(NSYS_KERNELS[m]),
                }
                for m in CASES
            },
        },
        "cases": cases,
        "overall_only_cases": [
            {
                "m": 1024,
                "overall_benchmark_us": float(benchmark[1024]["median_us"]),
                "per_op_timeline": "unavailable",
                "reason": "accepted exp_002 evidence has no per-op NSys/NCU row for M1024",
            }
        ],
    }


def render(model):
    cases = {row["m"]: row for row in model["cases"]}
    by_phase = {m: {row["phase"]: row for row in cases[m]["phases"]} for m in CASES}
    fc1 = by_phase[8192]["fc1"]
    fc2 = by_phase[8192]["fc2"]
    route = by_phase[8192]["expand_quant"]
    activation = by_phase[8192]["activation_requant"]
    finalize = by_phase[8192]["finalize"]
    authority = model["hardware_authority"]["calibrated_profile"]

    def dram_diagnostic(row):
        diag = row["ncu_diagnostic"]
        value = diag["physical_dram_resource_efficiency_percent"]
        if diag["physical_dram_resource_status"] == "invalid":
            return "unavailable（投影为 {:.2f}% > 100%，scope 不闭合）".format(value)
        return "{:.2f}%（diagnostic，不是完整 op ceiling）".format(value)

    lines = [
        "# exp_024：CUTLASS Chain 逐算子性能上界",
        "",
        "## 结论",
        "",
        (
            "M8192 下，相对 5KP 实测 NVFP4 Tensor Core ceiling，FC1 的 Useful / Executed "
            "efficiency 为 **{:.2f}% / {:.2f}%**，FC2 为 **{:.2f}% / {:.2f}%**；"
            "两者 padding efficiency 均为 **{:.2f}%**。"
        ).format(
            fc1["calibrated_useful_ceiling_efficiency_percent"],
            fc1["calibrated_executed_ceiling_efficiency_percent"],
            fc2["calibrated_useful_ceiling_efficiency_percent"],
            fc2["calibrated_executed_ceiling_efficiency_percent"],
            fc1["padding_efficiency_percent"],
        ),
        "",
        (
            "非 GEMM op 不能套 MFU。基于 NCU physical bytes 与 exp_026 directional DRAM roof 的"
            "投影只作为资源诊断：Route/Q0/Pack 为 **{}**，SwiGLU/Q1 为 **{}**，Finalize 为 **{}**。"
        ).format(
            dram_diagnostic(route),
            dram_diagnostic(activation),
            dram_diagnostic(finalize),
        ),
        "",
        (
            "硬件 ceiling verdict 为 **accept**；operator SOTA distance 仍为 unavailable。"
            "计算分母来自同一 RTX 5KP SKU、不同 GPU UUID 的实测迁移，不能表述为同卡同窗测量。"
        ),
        "",
        "## 1. Scope 与证据边界",
        "",
        "```text",
        "BF16 input → Prefix → Route/Q0/Pack → GEMM metadata → FC1 → SwiGLU/Q1 → FC2 → Finalize → BF16 output",
        "```",
        "",
        (
            "逐 op 时间来自 exp_002 canonical NSys；work counter 来自同 rerun 的独立 NCU replay。"
            "计算 ceiling 来自 exp_026 vetted `{}` record。consumer GPU `{}` 与 calibration GPU `{}` "
            "UUID 不同；二者同为 RTX 5KP、110 SM，因此这里只接受 same-SKU transfer。"
        ).format(
            authority["record_id"],
            authority["consumer_gpu_uuid"],
            authority["calibration_gpu_uuid"],
        ),
        "",
        "| M | E2E benchmark | NSys active union | NSys / benchmark |",
        "|---:|---:|---:|---:|",
    ]
    for m in BENCHMARK_CASES:
        if m in cases:
            case = cases[m]
            lines.append(
                "| {} | {:.3f} μs | {:.3f} μs | {:.2f}% |".format(
                    m,
                    case["overall_benchmark_us"],
                    case["timeline_active_union_us"],
                    case["timeline_vs_benchmark_percent"],
                )
            )
        else:
            row = model["overall_only_cases"][0]
            lines.append(
                "| 1024 | {:.3f} μs | unavailable | unavailable |".format(
                    row["overall_benchmark_us"]
                )
            )

    lines.extend(
        [
            "",
            "## 2. 各 op 时间与占比",
            "",
            "| Op | M256 time / share | M8192 time / share |",
            "|---|---:|---:|",
        ]
    )
    for phase in DISPLAY_PHASES:
        a = by_phase[256][phase]
        b = by_phase[8192][phase]
        lines.append(
            "| {} | {:.3f} μs / {:.2f}% | {:.3f} μs / {:.2f}% |".format(
                a["label"],
                a["time_us"],
                a["share_percent"],
                b["time_us"],
                b["share_percent"],
            )
        )
    lines.extend(
        [
            "",
            (
                "占比分母是全部 kernel interval 的 active union。相邻 launch 存在 PDL overlap，"
                "所以各行 duration/share 不是互斥分区；M256 / M8192 的 share 合计为 "
                "**{:.4f}% / {:.4f}%**，cross-category overlap 为 **{:.3f} / {:.3f} μs**，不重新归一化。"
            ).format(
                cases[256]["timeline_accounting"]["category_share_sum_percent"],
                cases[8192]["timeline_accounting"]["category_share_sum_percent"],
                cases[256]["timeline_accounting"]["cross_category_overlap_ns"] / 1000.0,
                cases[8192]["timeline_accounting"]["cross_category_overlap_ns"]
                / 1000.0,
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## 3. M8192 逐算子 ceiling 百分比",
            "",
            "| Op | Hardware ceiling efficiency | Padding | SOTA distance |",
            "|---|---|---:|---|",
        ]
    )
    rows = by_phase[8192]
    for phase in PHASE_LABELS:
        row = rows[phase]
        if phase in ("fc1", "fc2"):
            hardware = "{:.2f}% Useful / {:.2f}% Executed（calibrated TC）".format(
                row["calibrated_useful_ceiling_efficiency_percent"],
                row["calibrated_executed_ceiling_efficiency_percent"],
            )
            padding = "{:.2f}%".format(row["padding_efficiency_percent"])
        else:
            hardware = "complete op ceiling unavailable; DRAM {}".format(
                dram_diagnostic(row)
            )
            padding = "—"
        lines.append(
            "| {} | {} | {} | unavailable |".format(row["label"], hardware, padding)
        )

    lines.extend(
        [
            "",
            (
                "DRAM resource efficiency = `max(R/BWread, W/BWwrite, (R+W)/BWcopy) / NCU duration`。"
                "它只说明 physical DRAM 资源投影；大于 100% 直接标 invalid，不截断，也不称完整 op ceiling。"
            ),
            "",
            "## 4. GEMM ceiling 达成率",
            "",
            "| M | GEMM | Calibrated Useful | Calibrated Executed | Padding | Nominal Useful / Executed（次级） | TC active（诊断） |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for m in CASES:
        for phase in ("fc1", "fc2"):
            row = by_phase[m][phase]
            lines.append(
                "| {} | {} | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}% / {:.2f}% | {:.2f}% |".format(
                    m,
                    row["label"],
                    row["calibrated_useful_ceiling_efficiency_percent"],
                    row["calibrated_executed_ceiling_efficiency_percent"],
                    row["padding_efficiency_percent"],
                    row["nominal_useful_mfu_percent"],
                    row["nominal_executed_mfu_percent"],
                    row["tensor_pipe_active_percent"],
                )
            )

    lines.extend(
        [
            "",
            (
                "主分母是 exp_026 对 exact `{}` 指令的 full-card ~100 ms calibrated window；"
                "由于 exp_002 没有同 launch cycle counter，本报告没有使用 per-cycle normalization。"
                "Nominal 百分比只作架构次级参照；`TC active` 分母不同，也只并列诊断。"
            ).format(authority["instruction"]),
            "",
            (
                "M256 的 {:.2f}% padding efficiency 使 Nominal Useful MFU 为 {:.2f}%–{:.2f}%，"
                "而 Nominal Executed MFU 为 {:.2f}%–{:.2f}%；padding 是 Useful MFU 与 Executed MFU "
                "差距的主要来源，但这不等同于证明它是相对 SOTA latency gap 的首要原因。"
            ).format(
                by_phase[256]["fc1"]["padding_efficiency_percent"],
                min(
                    by_phase[256]["fc1"]["nominal_useful_mfu_percent"],
                    by_phase[256]["fc2"]["nominal_useful_mfu_percent"],
                ),
                max(
                    by_phase[256]["fc1"]["nominal_useful_mfu_percent"],
                    by_phase[256]["fc2"]["nominal_useful_mfu_percent"],
                ),
                min(
                    by_phase[256]["fc1"]["nominal_executed_mfu_percent"],
                    by_phase[256]["fc2"]["nominal_executed_mfu_percent"],
                ),
                max(
                    by_phase[256]["fc1"]["nominal_executed_mfu_percent"],
                    by_phase[256]["fc2"]["nominal_executed_mfu_percent"],
                ),
            ),
            "",
            "## 5. 未闭合项",
            "",
            "1. 用 contract-equivalent、独立实现测 FC1/FC2 grouped-GEMM SOTA；在此之前不报告 SOTA gap。",
            "2. Route/Q0/Pack、SwiGLU/Q1、Finalize 仍需同 layout/责任边界的 standalone calibration；DRAM resource diagnostic 不能替代它。",
            "",
            "原始 throughput、公式输入、digest 与逐 op ceiling status 见 [model.json](model.json)。",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    model = build()
    with (RESULTS / "model.json").open("w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=2, sort_keys=True)
        handle.write("\n")
    (RESULTS / "result.md").write_text(render(model), encoding="utf-8")


if __name__ == "__main__":
    main()
