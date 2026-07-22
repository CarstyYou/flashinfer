#!/usr/bin/env python3
"""Build the SGLang Triton FP8 per-op performance ceiling report.

The builder is intentionally CPU-only.  It reconstructs the accepted M8192
CUDA-graph timing from exp_017, derives routed padding from the pinned fixture,
and keeps NCU evidence launch-local and diagnostic.
"""

import ast
import csv
import hashlib
import io
import json
import statistics
import struct
import zipfile
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
EXPERIMENTS = EXP_DIR.parent
RESULTS = EXP_DIR / "results"

EXP001 = EXPERIMENTS / "exp_001_backend_case_sweep"
EXP017 = EXPERIMENTS / "exp_017_opt_vs_triton_phase_share"
EXP018 = EXPERIMENTS / "exp_018_triton_opt_eric_benchmark"
EXP026 = EXPERIMENTS / "exp_026_5kp_ceiling_calibration"

EVIDENCE = EXP017 / "results/evidence.json"
GRAPH = EXP017 / "results/veloq/triton_graph_replays.json"
CAPTURE = EXP017 / "results/triton_capture_manifest.json"
ARTIFACT = EXP017 / "results/triton_artifact.lock.json"
NCU = EXP017 / "results/ncu_evidence.json"
BENCHMARK = EXP001 / "results/sglang_triton/benchmark_summary.csv"
SANITY_BENCHMARK = EXP018 / "results/benchmark_summary.csv"
SANITY_RAW = EXP018 / "results/benchmark_raw.csv"
FIXTURE = EXP001 / "results/fixtures/m8192.npz"
CALIBRATION_PROFILE = EXP026 / "results/profile.json"
DEVICE_REFERENCE_REPORT = (
    EXPERIMENTS
    / "exp_002_fused_vs_chain_dataflow/results/ncu/m8192/cutlass_bf16_chain/deep_launch_5/trace.ncu-rep"
)

ARM = "sglang_triton_fp8"
M = 8192
EXPERTS = 256
HIDDEN = 2048
INTERMEDIATE = 512
TOPK = 8

# NVIDIA RTX Blackwell GPU Architecture v1.1, Table 4.  The dense FP8
# figures imply 2048 FLOP/SM/cycle; the sparse second number is not used.
OFFICIAL_ARCH_URL = (
    "https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/"
    "quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf"
)
FP8_FLOP_PER_SM_CYCLE = 2048
DEVICE_SM_COUNT = 110
DEVICE_SM_CLOCK_HZ = 2_377_000_000
FP8_PEAK_TFLOPS = DEVICE_SM_COUNT * DEVICE_SM_CLOCK_HZ * FP8_FLOP_PER_SM_CYCLE / 1.0e12

# exp_017's accepted sibling-GPU NCU replay does not expose
# sm__cycles_elapsed.sum.  l1tex__cycles_elapsed.sum is therefore used only
# as an explicitly named cycle proxy.  Its sum/avg ratio is 110 for both
# launches, and each sum agrees with gpc__cycles_elapsed.avg * 110 to within
# 0.008%.  These values must never be relabeled as SM-cycle counters.
L1TEX_CYCLE_PROXY = {
    "fc1": 266_274_162,
    "fc2": 177_121_036,
}
L1TEX_CYCLE_PROXY_INSTANCE_COUNT = 110
L1TEX_GPC_CROSSCHECK_MAX_DELTA_PERCENT = 0.008

GROUPS = {
    "route": ("route_align", "route_count_sort"),
    "q0": ("q0_fill", "q0_absmax", "q0_quant"),
    "fc1": ("fc1",),
    "swiglu": ("swiglu",),
    "q1": ("q1_fill", "q1_absmax", "q1_quant"),
    "fc2": ("fc2",),
    "topk_reduce": ("topk_reduce",),
}

CEILING_MIN_SHARE_PERCENT = 3.0
CEILING_PHASES = (
    "fc1",
    "swiglu",
    "fc2",
    "topk_reduce",
)

LABELS = {
    "route": "Routing / scheduler",
    "q0": "Q0（input quant）",
    "fc1": "FC1",
    "swiglu": "SwiGLU",
    "q1": "Q1（intermediate quant）",
    "fc2": "FC2",
    "topk_reduce": "TopK reduce / finalize",
    "graph_bubble": "Graph bubble",
}

ROLE_TOKENS = {
    "route_align": "moe_align_block_size_kernel",
    "route_count_sort": "count_and_sort_expert_tokens_kernel",
    "q0_fill": "vectorized_elementwise_kernel",
    "q0_absmax": "per_tensor_absmax_kernel",
    "q0_quant": "per_tensor_quant_fp8_kernel",
    "fc1": "fused_moe_kernel",
    "swiglu": "act_and_mul_kernel",
    "q1_fill": "vectorized_elementwise_kernel",
    "q1_absmax": "per_tensor_absmax_kernel",
    "q1_quant": "per_tensor_quant_fp8_kernel",
    "fc2": "fused_moe_kernel",
    "topk_reduce": "moe_sum_reduce_warp_per_token_vec_kernel",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_canonical_digest(payload, key, label):
    expected = payload[key]
    body = dict(payload)
    del body[key]
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    require(hashlib.sha256(encoded).hexdigest() == expected, label + " digest drift")


def read_topk_ids(path):
    """Read the int32 topk_ids member without requiring NumPy on the host."""
    with zipfile.ZipFile(path) as archive:
        payload = archive.read("topk_ids.npy")
    handle = io.BytesIO(payload)
    require(handle.read(6) == b"\x93NUMPY", "invalid NPY magic")
    major, _minor = struct.unpack("BB", handle.read(2))
    require(major in (1, 2, 3), "unsupported NPY version")
    length_size = 2 if major == 1 else 4
    header_len = struct.unpack("<H" if major == 1 else "<I", handle.read(length_size))[
        0
    ]
    header = ast.literal_eval(handle.read(header_len).decode("latin1").strip())
    require(header["descr"] == "<i4", "topk_ids must be little-endian int32")
    require(header["fortran_order"] is False, "Fortran-order fixture is unsupported")
    require(tuple(header["shape"]) == (M, TOPK), "topk_ids shape drift")
    data = handle.read()
    require(len(data) == M * TOPK * 4, "topk_ids byte count drift")
    values = [item[0] for item in struct.iter_unpack("<i", data)]

    array_digest = hashlib.sha256()
    array_digest.update(b"int32")
    array_digest.update(str((M, TOPK)).encode())
    array_digest.update(data)
    return values, array_digest.hexdigest()


def fixture_padding(path, expected):
    require(sha256(path) == expected["fixture_sha256"], "fixture file digest drift")
    values, topk_digest = read_topk_ids(path)
    require(topk_digest == expected["topk_ids_sha256"], "topk_ids digest drift")
    counts = [0] * EXPERTS
    for expert in values:
        require(0 <= expert < EXPERTS, "expert id out of range")
        counts[expert] += 1
    occupancy_bytes = struct.pack("<{}q".format(EXPERTS), *counts)
    occupancy_digest = hashlib.sha256(occupancy_bytes).hexdigest()
    require(occupancy_digest == expected["occupancy_sha256"], "occupancy digest drift")
    require(min(counts) == expected["occupancy_min"], "occupancy min drift")
    require(max(counts) == expected["occupancy_max"], "occupancy max drift")
    require(sum(counts) == M * TOPK, "logical routed-row count drift")
    return counts


def median(values):
    return float(statistics.median(values))


def metric_value(launch, name):
    return float(launch["metrics"][name]["value"])


def build_timeline(graph, capture, evidence):
    require(graph["schema"] == "v1", "graph schema drift")
    require(graph["source"]["kind"] == "nsys", "graph source drift")
    require(graph["command"] == "nsys.graph-replays", "graph command drift")
    graph_rows = graph["data"]["rows"]
    require(graph["data"]["count"] == 5, "expected five graph replays")
    require(len(graph_rows) == 5, "graph replay row count drift")
    require(evidence["gates"]["five_replays_each"], "five-replay gate failed")
    require(
        evidence["gates"]["triton_ordered_12_node_topology"], "topology gate failed"
    )

    expected_nodes = capture["expected_graph_topology"]["nodes"]
    evidence_node_ids = evidence["triton"]["topology"]["node_ids"]
    require(
        len(expected_nodes) == len(evidence_node_ids) == 12, "expected 12 graph nodes"
    )
    for index, node in enumerate(expected_nodes, start=1):
        require(int(node["ordinal"]) == index, "topology ordinal drift")

    first_ordered = sorted(
        graph_rows[0]["top_nodes"], key=lambda item: (item["start_ns"], item["end_ns"])
    )
    node_ids = [int(node["graph_node_id"]) for node in first_ordered]
    require(node_ids == evidence_node_ids, "raw/evidence node-id order drift")
    id_to_role = {
        node_id: node["role"]
        for node_id, node in zip(node_ids, expected_nodes)  # noqa: B905 -- Python 3.6
    }

    per_group_ns = {name: [] for name in GROUPS}
    per_group_share = {name: [] for name in GROUPS}
    bubbles_ns = []
    bubble_share = []
    walls_ns = []
    replay_rows = []
    for replay in graph_rows:
        require(replay["capture_mode"] == "graph_nodes", "capture mode drift")
        require(
            int(replay["wall_ns"]) == int(replay["end_ns"]) - int(replay["start_ns"]),
            "replay wall drift",
        )
        require(
            replay["event_count"] == replay["kernel_count"] == 12, "node count drift"
        )
        require(
            replay["memcpy_count"] == replay["memset_count"] == 0,
            "unexpected graph copy",
        )
        require(
            replay["stream_count"] == 1 and replay["graph_id"] == 2,
            "graph identity drift",
        )
        require(replay["decomposition_available"], "graph decomposition unavailable")
        nodes = {int(node["graph_node_id"]): node for node in replay["top_nodes"]}
        require(set(nodes) == set(id_to_role), "graph node identity drift")
        ordered_nodes = sorted(
            replay["top_nodes"], key=lambda item: (item["start_ns"], item["end_ns"])
        )
        ordered_ids = [int(node["graph_node_id"]) for node in ordered_nodes]
        require(ordered_ids == node_ids, "graph execution order drift")
        require(
            int(ordered_nodes[0]["start_ns"]) == int(replay["start_ns"]),
            "graph start drift",
        )
        require(
            int(ordered_nodes[-1]["end_ns"]) == int(replay["end_ns"]), "graph end drift"
        )
        gap_ns = 0
        for index, node in enumerate(ordered_nodes):
            role = id_to_role[int(node["graph_node_id"])]
            require(
                node["kind"] == "kernel" and node["count"] == 1, "non-kernel graph node"
            )
            require(node["stream_count"] == 1, "multi-stream node drift")
            require(ROLE_TOKENS[role] in node["name"], role + " kernel-name drift")
            require(
                int(node["wall_ns"]) == int(node["end_ns"]) - int(node["start_ns"]),
                role + " wall drift",
            )
            require(int(node["sum_ns"]) == int(node["wall_ns"]), role + " sum drift")
            if index:
                gap = int(node["start_ns"]) - int(ordered_nodes[index - 1]["end_ns"])
                require(gap >= 0, "overlapping serial graph nodes")
                gap_ns += gap
        role_ns = {
            id_to_role[node_id]: int(node["wall_ns"]) for node_id, node in nodes.items()
        }
        node_sum_ns = sum(role_ns.values())
        require(node_sum_ns == int(replay["sum_gpu_ns"]), "graph node sum drift")
        require(node_sum_ns == int(replay["busy_ns"]), "graph busy time drift")
        bubble_ns = int(replay["wall_ns"]) - node_sum_ns
        require(bubble_ns == int(replay["idle_inside_replay_ns"]), "graph bubble drift")
        require(bubble_ns == gap_ns, "graph inter-node gap drift")
        for group, roles in GROUPS.items():
            group_ns = sum(role_ns[role] for role in roles)
            per_group_ns[group].append(group_ns)
            per_group_share[group].append(100.0 * group_ns / int(replay["wall_ns"]))
        bubbles_ns.append(bubble_ns)
        bubble_share.append(100.0 * bubble_ns / int(replay["wall_ns"]))
        walls_ns.append(int(replay["wall_ns"]))
        replay_rows.append(
            {
                "correlation_id": replay["correlation_id"],
                "wall_us": replay["wall_ns"] / 1000.0,
                "bubble_us": bubble_ns / 1000.0,
            }
        )

    medians_ns = {group: median(values) for group, values in per_group_ns.items()}
    medians_ns["graph_bubble"] = median(bubbles_ns)
    median_shares = {group: median(values) for group, values in per_group_share.items()}
    median_shares["graph_bubble"] = median(bubble_share)
    component_sum_ns = sum(medians_ns.values())
    wall_median_ns = median(walls_ns)
    require(
        abs(component_sum_ns - wall_median_ns) / wall_median_ns < 0.001,
        "component medians do not close graph wall",
    )

    # Cross-check independently generated exp_017 summaries, but do not use
    # those summaries as the timing input.
    evidence_groups = evidence["triton"]["groups"]
    evidence_ops = evidence["triton"]["ops"]
    checks = {
        "route": evidence_groups["Routing / scheduler"]["time_us"]["median"],
        "q0": evidence_groups["Q0 / input pack"]["time_us"]["median"],
        "fc1": evidence_ops["fc1"]["time_us"]["median"],
        "swiglu": evidence_ops["swiglu"]["time_us"]["median"],
        "q1": evidence_groups["Q1"]["time_us"]["median"],
        "fc2": evidence_ops["fc2"]["time_us"]["median"],
        "topk_reduce": evidence_ops["topk_reduce"]["time_us"]["median"],
        "graph_bubble": evidence_ops["graph_node_bubble"]["time_us"]["median"],
    }
    for group, expected_us in checks.items():
        require(
            abs(medians_ns[group] / 1000.0 - expected_us) < 0.001,
            group + " summary drift",
        )

    phases = []
    for group in tuple(GROUPS) + ("graph_bubble",):
        phases.append(
            {
                "phase": group,
                "label": LABELS[group],
                "time_us": medians_ns[group] / 1000.0,
                "share_percent": median_shares[group],
            }
        )
    share_sum = sum(row["share_percent"] for row in phases)
    require(abs(share_sum - 100.0) < 0.1, "median share closure drift")
    return {
        "replays": replay_rows,
        "graph_wall_median_us": wall_median_ns / 1000.0,
        "component_median_sum_us": component_sum_ns / 1000.0,
        "median_nonadditivity_us": (component_sum_ns - wall_median_ns) / 1000.0,
        "median_share_sum_percent": share_sum,
        "median_share_nonadditivity_pp": share_sum - 100.0,
        "phases": phases,
    }


def useful_flops(phase, rows):
    return rows * (4 if phase == "fc1" else 2) * HIDDEN * INTERMEDIATE


def logical_non_gemm(phase, duration_us):
    routed_rows = M * TOPK
    seconds = duration_us * 1.0e-6
    if phase == "route":
        count = routed_rows
        return {
            "work_name": "assignments",
            "work_count": count,
            "achieved_gwork_per_s": count / seconds / 1.0e9,
            "payload_floor_bytes": None,
            "payload_scope": "not modeled",
        }
    if phase == "q0":
        values = M * HIDDEN
        # absmax and quant each read BF16 input; quant writes E4M3.
        payload = values * (2 + 2 + 1)
        scope = "absmax BF16 read + quant BF16 read/E4M3 write; scalar fill excluded"
        work_name = "input values"
    elif phase == "swiglu":
        values = routed_rows * INTERMEDIATE
        payload = values * (2 + 2 + 2)
        scope = "Gate/Up BF16 reads + SwiGLU BF16 write"
        work_name = "output values"
    elif phase == "q1":
        values = routed_rows * INTERMEDIATE
        payload = values * (2 + 2 + 1)
        scope = "absmax BF16 read + quant BF16 read/E4M3 write; scalar fill excluded"
        work_name = "output values"
    elif phase == "topk_reduce":
        values = M * HIDDEN
        payload = routed_rows * HIDDEN * 2 + values * 2
        scope = "already weighted top-k BF16 contribution reads + BF16 output"
        work_name = "output values"
    else:
        raise ValueError("unsupported non-GEMM phase")
    return {
        "work_name": work_name,
        "work_count": values,
        "achieved_gwork_per_s": values / seconds / 1.0e9,
        "payload_floor_bytes": payload,
        "logical_payload_gbs": payload / seconds / 1.0e9,
        "payload_scope": scope,
    }


def build():
    evidence = load_json(EVIDENCE)
    graph = load_json(GRAPH)
    capture = load_json(CAPTURE)
    artifact = load_json(ARTIFACT)
    ncu = load_json(NCU)
    calibration = load_json(CALIBRATION_PROFILE)

    require_canonical_digest(evidence, "evidence_sha256", "compact evidence")
    require_canonical_digest(capture, "fingerprint_sha256", "capture manifest")
    require_canonical_digest(artifact, "fingerprint_sha256", "artifact lock")
    require_canonical_digest(ncu, "fingerprint_sha256", "NCU evidence")

    require(
        evidence["case"]
        == {
            "experts": EXPERTS,
            "hidden": HIDDEN,
            "intermediate_tp": INTERMEDIATE,
            "m": M,
            "topk": TOPK,
        },
        "case contract drift",
    )
    require(
        evidence["gates"]["same_fixture_and_routing"], "fixture/routing gate failed"
    )
    require(evidence["gates"]["correctness"], "correctness gate failed")
    require(
        capture["artifact_stable_during_capture"], "JIT artifact changed during capture"
    )
    require(capture["eager_correctness"]["formal_pass"], "Triton correctness failed")
    require(
        capture["artifact_fingerprint_sha256"] == artifact["fingerprint_sha256"],
        "artifact fingerprint drift",
    )
    require(
        capture["launch_contract"]["callable_source_sha256"]
        == evidence["triton"]["launch_contract"]["callable_source_sha256"],
        "callable source drift",
    )
    require(
        capture["runtime_identity"]["sglang_commit"]
        == "0b3bb0cbe31873994c9f989fddfe2f87ca839fdd",
        "SGLang commit drift",
    )
    require(
        capture["runtime_identity"]["triton_version"] == "3.6.0", "Triton version drift"
    )
    require(
        capture["schema"] == "exp017.sglang-triton-nsys-capture.v1",
        "capture schema drift",
    )
    require(capture["protocol"]["m"] == M, "capture M drift")
    require(
        capture["protocol"]["captured_graph_replays"] == 5, "capture replay-count drift"
    )
    require(
        capture["protocol"]["nsys_graph_mode"] == "node:host-only",
        "NSys graph mode drift",
    )
    require(
        all(row["formal_pass"] for row in capture["replay_correctness"].values()),
        "replay correctness failed",
    )
    require(
        ncu["evidence_identity"]["duration_authority"] is False,
        "NCU duration must not be canonical",
    )
    require(evidence["gates"]["ncu_launch_local_only"], "NCU launch-local gate failed")
    require(
        evidence["gates"]["ncu_correctness_and_launch_identity"],
        "NCU launch identity gate failed",
    )
    require(
        evidence["gates"]["ncu_same_sibling_gpu_and_clock"],
        "NCU sibling/clock gate failed",
    )

    require(
        calibration["schema"] == "calibrated-gpu-ceiling-profile.v1",
        "calibration schema drift",
    )
    require(calibration["decision"] == "accept", "calibration profile is not accepted")
    require(calibration["gpu"]["internal_sku"] == "RTX 5KP", "calibration SKU drift")
    require(
        calibration["gpu"]["sm_count"] == DEVICE_SM_COUNT, "calibration SM-count drift"
    )
    calibration_matches = [
        row
        for row in calibration["tensor_core_records"]
        if row["id"] == "fp8-e4m3-noscale"
    ]
    require(
        len(calibration_matches) == 1, "expected one FP8 no-scale calibration record"
    )
    fp8_calibration = calibration_matches[0]
    require(
        fp8_calibration["audit_status"] == "vetted", "FP8 calibration is not vetted"
    )
    require(
        fp8_calibration["consumer_scope"] == "exp_025 SGLang Triton FP8 FC1/FC2",
        "FP8 calibration consumer scope drift",
    )
    require(
        fp8_calibration["instruction"] == "QMMA.16832.F32.E4M3.E4M3",
        "FP8 calibration instruction drift",
    )
    calibrated_work_per_cycle_sm = float(
        fp8_calibration["per_sm_instruction_ceiling"]["median_work_per_cycle_sm"]
    )
    calibrated_full_card_tflops = float(
        fp8_calibration["sustained_window"]["sustained_window_tflops"]
    )
    require(calibrated_work_per_cycle_sm > 0.0, "invalid calibrated FP8 per-cycle roof")
    require(calibrated_full_card_tflops > 0.0, "invalid calibrated FP8 full-card roof")
    require(
        float(fp8_calibration["per_sm_instruction_ceiling"]["cv_percent"]) <= 1.0,
        "unstable calibrated FP8 per-cycle roof",
    )
    require(
        capture["runtime_identity"]["gpu_uuid"] == calibration["gpu"]["uuid"],
        "target/calibration GPU UUID drift",
    )
    require(
        int(capture["runtime_identity"]["application_clock_mhz"])
        == int(calibration["gpu"]["application_graphics_clock_mhz"]),
        "target/calibration application-clock drift",
    )

    timeline = build_timeline(graph, capture, evidence)

    block_m = int(capture["launch_contract"]["config"]["up"]["BLOCK_SIZE_M"])
    require(block_m == 64, "BLOCK_SIZE_M drift")
    counts = fixture_padding(FIXTURE, capture["fixture"])
    physical_rows = sum((count + block_m - 1) // block_m * block_m for count in counts)
    logical_rows = M * TOPK
    require(physical_rows == 73536, "physical routed rows drift")

    phase_by_name = {row["phase"]: row for row in timeline["phases"]}
    ncu_launches = ncu["triton_material_launches"]
    operations = []
    for phase in GROUPS:
        timing = phase_by_name[phase]
        row = {
            "phase": phase,
            "label": LABELS[phase],
            "time_us": timing["time_us"],
            "share_percent": timing["share_percent"],
            "sota_status": "unavailable",
            "sota_reason": "no independent contract-equivalent per-op anchor",
        }
        if phase in ("fc1", "fc2"):
            useful = useful_flops(phase, logical_rows)
            executed = useful_flops(phase, physical_rows)
            useful_tflops = useful / (timing["time_us"] * 1.0e-6) / 1.0e12
            executed_tflops = executed / (timing["time_us"] * 1.0e-6) / 1.0e12
            tc_active = metric_value(ncu_launches[phase], "tc_subpipe_active_pct")
            cycle_proxy = L1TEX_CYCLE_PROXY[phase]
            calibrated_capacity_flops = calibrated_work_per_cycle_sm * cycle_proxy
            row.update(
                {
                    "model": "dense FP8 tensor-core grouped GEMM",
                    "useful_flops": useful,
                    "executed_flops": executed,
                    "executed_work_authority": "dispatch/source-derived physical work; not an NCU dynamic counter",
                    "logical_routed_rows": logical_rows,
                    "physical_routed_rows": physical_rows,
                    "padding_efficiency_percent": 100.0 * useful / executed,
                    "useful_tflops": useful_tflops,
                    "executed_tflops": executed_tflops,
                    "nominal_useful_mfu_percent": 100.0
                    * useful_tflops
                    / FP8_PEAK_TFLOPS,
                    "nominal_executed_mfu_percent": 100.0
                    * executed_tflops
                    / FP8_PEAK_TFLOPS,
                    "calibrated_useful_ceiling_efficiency_percent": 100.0
                    * useful_tflops
                    / calibrated_full_card_tflops,
                    "calibrated_executed_ceiling_efficiency_percent": 100.0
                    * executed_tflops
                    / calibrated_full_card_tflops,
                    "calibrated_full_card_tflops": calibrated_full_card_tflops,
                    "calibrated_work_per_cycle_sm": calibrated_work_per_cycle_sm,
                    "cycle_proxy": {
                        "metric": "l1tex__cycles_elapsed.sum",
                        "value": cycle_proxy,
                        "instance_count_from_sum_over_avg": L1TEX_CYCLE_PROXY_INSTANCE_COUNT,
                        "gpc_avg_times_instance_count_max_delta_percent": L1TEX_GPC_CROSSCHECK_MAX_DELTA_PERCENT,
                        "scope": "accepted sibling-GPU NCU launch; proxy for aggregate SM elapsed cycles, not sm__cycles_elapsed.sum",
                        "diagnostic_useful_efficiency_percent": 100.0
                        * useful
                        / calibrated_capacity_flops,
                        "diagnostic_executed_efficiency_percent": 100.0
                        * executed
                        / calibrated_capacity_flops,
                    },
                    "tensor_pipe_active_percent": tc_active,
                    "hardware_ceiling_status": "diagnostic same-UUID source-contract-compatible full-card estimate; target SASS binding unavailable",
                }
            )
        else:
            work = logical_non_gemm(phase, timing["time_us"])
            row.update(
                {
                    "model": {
                        "route": "routing/scheduler",
                        "q0": "quant/pack",
                        "swiglu": "elementwise ALU/SFU",
                        "q1": "quant/pack",
                        "topk_reduce": "gather/reduction/store",
                    }[phase],
                    "logical_work": work,
                    "hardware_ceiling_status": "unavailable for complete operator",
                }
            )
            if phase in ncu_launches:
                row["ncu_diagnostic"] = {
                    "dram_throughput_percent": metric_value(
                        ncu_launches[phase], "dram_throughput_pct"
                    ),
                    "scope": "sibling-GPU NCU replay; normalized launch-local diagnostic only",
                }
        operations.append(row)

    benchmark_rows = [
        row
        for row in load_csv(BENCHMARK)
        if row["arm"] == ARM and int(row["m"]) in (256, 1024, 8192)
    ]
    require(len(benchmark_rows) == 3, "paired benchmark coverage drift")
    require(
        len({row["rerun_id"] for row in benchmark_rows}) == 1, "benchmark rerun drift"
    )
    require(
        all(row["stable_le_5_percent"] == "True" for row in benchmark_rows),
        "unstable paired benchmark",
    )
    for field in (
        "environment_lock_digest",
        "protocol_lock_digest",
        "artifact_fingerprint_sha256",
    ):
        require(len({row[field] for row in benchmark_rows}) == 1, field + " drift")
    benchmarks = {int(row["m"]): float(row["median_us"]) for row in benchmark_rows}
    require(
        benchmark_rows[-1]["fixture_sha256"] == capture["fixture"]["fixture_sha256"],
        "M8192 benchmark fixture drift",
    )

    sanity_rows = {
        int(row["m"]): float(row["median_us"])
        for row in load_csv(SANITY_BENCHMARK)
        if row["arm"] == ARM
        and row["status"] == "Pass"
        and int(row["m"]) in (256, 1024, 8192)
    }
    require(set(sanity_rows) == {256, 1024, 8192}, "sanity benchmark coverage drift")
    sanity_raw = [
        row
        for row in load_csv(SANITY_RAW)
        if row["arm"] == ARM and int(row["m"]) in (256, 1024, 8192)
    ]
    require(len(sanity_raw) == 9, "sanity raw sample-count drift")
    require(all(row["status"] == "pass" for row in sanity_raw), "sanity raw failure")
    require(len({row["rerun_id"] for row in sanity_raw}) == 1, "sanity rerun drift")
    require(
        len({row["protocol_sha256"] for row in sanity_raw}) == 1,
        "sanity protocol drift",
    )
    benchmark_by_m = {int(row["m"]): row for row in benchmark_rows}
    for m in (256, 1024, 8192):
        samples = [float(row["sample_us"]) for row in sanity_raw if int(row["m"]) == m]
        require(
            len(samples) == 3 and median(samples) == sanity_rows[m],
            "sanity median drift",
        )
        fixture_digests = {
            row["fixture_sha256"] for row in sanity_raw if int(row["m"]) == m
        }
        require(
            fixture_digests == {benchmark_by_m[m]["fixture_sha256"]},
            "sanity fixture drift",
        )

    ncu_rows = []
    for phase in ("fc1", "fc2", "swiglu", "topk_reduce"):
        launch = ncu_launches[phase]
        stalls = launch["pc_sample_stalls"]["reason_share_percent"]
        ncu_rows.append(
            {
                "phase": phase,
                "label": LABELS[phase],
                "tc_active_percent": metric_value(launch, "tc_subpipe_active_pct"),
                "dram_throughput_percent": metric_value(launch, "dram_throughput_pct"),
                "issue_active_percent": metric_value(launch, "issue_active_pct"),
                "achieved_occupancy_percent": metric_value(
                    launch, "achieved_occupancy_pct"
                ),
                "stall_share_percent": {
                    key: float(stalls[key])
                    for key in (
                        "wait",
                        "long_scoreboard",
                        "short_scoreboard",
                        "barrier",
                    )
                },
            }
        )

    operation_by_phase = {row["phase"]: row for row in operations}
    selected_ceiling_phases = tuple(
        phase
        for phase in GROUPS
        if operation_by_phase[phase]["share_percent"] >= CEILING_MIN_SHARE_PERCENT
    )
    require(
        selected_ceiling_phases == CEILING_PHASES,
        "reader ceiling phase selection drift",
    )

    return {
        "schema": "operator-performance-ceiling.v1",
        "subject": {
            "arm": ARM,
            "boundary": "BF16 input -> SGLang Triton W8A8 FP8 MoE chain -> BF16 output",
            "case": {
                "m": M,
                "experts": EXPERTS,
                "hidden": HIDDEN,
                "intermediate_tp": INTERMEDIATE,
                "topk": TOPK,
            },
        },
        "verdict": {
            "status": "revise",
            "accounting": "vetted",
            "validation": "limited to one per-op M regime",
            "reason": "GEMM percentage uses a vetted same-UUID calibrated roof, but target instruction equivalence is source-contract-derived because target SASS binding is unavailable; non-GEMM roofs and all independent SOTA anchors remain unavailable",
        },
        "reader_reporting_policy": {
            "primary_case_m": M,
            "minimum_share_percent": CEILING_MIN_SHARE_PERCENT,
            "ceiling_phases": list(CEILING_PHASES),
            "accounting_only_phases": [
                phase for phase in GROUPS if phase not in CEILING_PHASES
            ]
            + ["graph_bubble"],
            "rule": (
                "all phases remain in time accounting; phases below the primary-case "
                "share threshold are omitted from reader-facing resource analysis"
            ),
        },
        "evidence_status": {
            "timeline_identity": "vetted",
            "fixture_work_accounting": "vetted",
            "ncu_launch_local": "accepted diagnostic",
            "calibrated_fp8_instruction_ceiling": "vetted",
            "target_fp8_sass_binding": "unavailable; target instruction match is source-contract-derived",
            "l1tex_cycle_proxy": "diagnostic cross-check only; audited sibling-GPU proxy and explicitly not an SM-cycle counter",
            "raw_ncu_cubin_revalidation": "unavailable; raw report is not present in this worktree",
            "cross_regime_validation": "limited",
        },
        "hardware_authority": {
            "calibrated_tensor_core": {
                "combined_status": "diagnostic same-UUID source-contract-compatible full-card estimate; target SASS binding unavailable",
                "profile_id": calibration["profile_id"],
                "profile_path": str(CALIBRATION_PROFILE),
                "profile_sha256": sha256(CALIBRATION_PROFILE),
                "target_uuid": calibration["gpu"]["uuid"],
                "record_id": fp8_calibration["id"],
                "instruction": fp8_calibration["instruction"],
                "audit_status": fp8_calibration["audit_status"],
                "work_per_cycle_sm": calibrated_work_per_cycle_sm,
                "full_card_sustained_window_tflops": calibrated_full_card_tflops,
                "target_uuid_match": True,
                "formula": "efficiency = target modeled throughput / exact-mode full-card sustained-window roof",
                "cycle_proxy_diagnostic": {
                    "status": "diagnostic; not used by the primary percentage",
                    "metric": "l1tex__cycles_elapsed.sum",
                    "role": "audited proxy; not sm__cycles_elapsed.sum",
                    "sum_over_avg_instance_count": L1TEX_CYCLE_PROXY_INSTANCE_COUNT,
                    "gpc_avg_times_110_max_delta_percent": L1TEX_GPC_CROSSCHECK_MAX_DELTA_PERCENT,
                    "launch_values": L1TEX_CYCLE_PROXY,
                },
            },
            "official_architecture_url": OFFICIAL_ARCH_URL,
            "fp8_flop_per_sm_cycle": FP8_FLOP_PER_SM_CYCLE,
            "device_sm_count": {
                "value": DEVICE_SM_COUNT,
                "source": str(DEVICE_REFERENCE_REPORT),
                "source_sha256": sha256(DEVICE_REFERENCE_REPORT),
                "applicability": "accepted same-class 5KP NCU device attribute; not the exact NSys target UUID",
                "status": "diagnostic",
            },
            "device_application_clock_hz": DEVICE_SM_CLOCK_HZ,
            "nominal_dense_fp8_peak_tflops": FP8_PEAK_TFLOPS,
            "authority_status": "same-UUID full-card roof vetted; target instruction equivalence source-contract-derived and diagnostic; architecture-derived nominal roof remains diagnostic",
        },
        "identity": {
            "fixture_sha256": capture["fixture"]["fixture_sha256"],
            "occupancy_sha256": capture["fixture"]["occupancy_sha256"],
            "artifact_fingerprint_sha256": artifact["fingerprint_sha256"],
            "callable_source_sha256": capture["launch_contract"][
                "callable_source_sha256"
            ],
            "sglang_commit": capture["runtime_identity"]["sglang_commit"],
            "triton_version": capture["runtime_identity"]["triton_version"],
            "correctness": capture["eager_correctness"],
        },
        "padding": {
            "block_m": block_m,
            "logical_routed_rows": logical_rows,
            "physical_routed_rows": physical_rows,
            "padding_efficiency_percent": 100.0 * logical_rows / physical_rows,
            "derivation": "sum_e ceil(occupancy[e] / BLOCK_SIZE_M) * BLOCK_SIZE_M",
        },
        "timeline": timeline,
        "operations": operations,
        "ncu_diagnostic": {
            "scope": "normalized launch-local metrics from sibling GPU; NCU duration and additive graph traffic are forbidden",
            "report_sha256": ncu["source_files"]["triton_report_sha256"],
            "rows": ncu_rows,
        },
        "benchmarks": [
            {
                "m": m,
                "paired_exp001_us": benchmarks[m],
                "independent_exp018_us": sanity_rows[m],
                "difference_percent": 100.0 * (sanity_rows[m] / benchmarks[m] - 1.0),
                "per_op_timeline": "available" if m == M else "unavailable",
            }
            for m in (256, 1024, 8192)
        ],
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in {
                "evidence": EVIDENCE,
                "graph": GRAPH,
                "capture": CAPTURE,
                "artifact": ARTIFACT,
                "ncu": NCU,
                "benchmark": BENCHMARK,
                "sanity_benchmark": SANITY_BENCHMARK,
                "sanity_benchmark_raw": SANITY_RAW,
                "fixture": FIXTURE,
                "device_reference_report": DEVICE_REFERENCE_REPORT,
                "calibration_profile": CALIBRATION_PROFILE,
            }.items()
        },
    }


def render(model):
    operations = {row["phase"]: row for row in model["operations"]}
    ncu_by_phase = {row["phase"]: row for row in model["ncu_diagnostic"]["rows"]}
    timeline = model["timeline"]
    padding = model["padding"]

    def resource_cell(phase):
        row = operations[phase]
        ncu_row = ncu_by_phase[phase]
        if phase in ("fc1", "fc2"):
            return (
                "TC: Useful **{:.2f}%** / Executed **{:.2f}%**<br>"
                "DRAM throughput: **{:.2f}%**<br>"
                "有效计算占比: **{:.2f}%**"
            ).format(
                row["calibrated_useful_ceiling_efficiency_percent"],
                row["calibrated_executed_ceiling_efficiency_percent"],
                ncu_row["dram_throughput_percent"],
                row["padding_efficiency_percent"],
            )
        return "DRAM throughput: **{:.2f}%**".format(ncu_row["dram_throughput_percent"])

    def optimization_meaning(phase):
        row = ncu_by_phase[phase]
        if phase == "fc1":
            return (
                "主耗时；现有 TC/DRAM diagnostic 均未逼近 ceiling。<br>"
                "下一步：结合 Achieved occupancy {:.2f}% 调查调度与访存等待。"
            ).format(row["achieved_occupancy_percent"])
        if phase == "swiglu":
            return (
                "DRAM throughput 接近上限。<br>"
                "下一步：减少 traffic、改善 locality 或融合。"
            )
        if phase == "fc2":
            return (
                "DRAM diagnostic 比 TC 更接近上限，但仍未封顶。<br>"
                "下一步：结合 Achieved occupancy {:.2f}% 区分 memory wait 与 compute schedule。"
            ).format(row["achieved_occupancy_percent"])
        return (
            "DRAM throughput 接近上限。<br>"
            "下一步：减少读流量、优化 reduction/locality 或与前级融合。"
        )

    lines = [
        "# exp_025：SGLang Triton FP8 逐算子性能上界",
        "",
        "## 结论",
        "",
        (
            "M8192 的 graph wall 为 **{:.3f} μs**。FC1 / FC2 分别占 **{:.2f}% / {:.2f}%**，"
            "TopK reduce 占 **{:.2f}%**；三者合计 **{:.2f}%**，是当前 chain 的主耗时。"
        ).format(
            timeline["graph_wall_median_us"],
            operations["fc1"]["share_percent"],
            operations["fc2"]["share_percent"],
            operations["topk_reduce"]["share_percent"],
            operations["fc1"]["share_percent"]
            + operations["fc2"]["share_percent"]
            + operations["topk_reduce"]["share_percent"],
        ),
        "",
        (
            "相对同一 5KP GPU UUID 上实测、source-contract-compatible 的 full-card calibrated TC roof，"
            "FC1 的 Useful / Executed efficiency 为 "
            "**{:.2f}% / {:.2f}%**，FC2 为 **{:.2f}% / {:.2f}%**；有效计算占比为 **{:.2f}%**。"
            "由于 target SASS binding 缺失，这些值保持 diagnostic estimate。"
        ).format(
            operations["fc1"]["calibrated_useful_ceiling_efficiency_percent"],
            operations["fc1"]["calibrated_executed_ceiling_efficiency_percent"],
            operations["fc2"]["calibrated_useful_ceiling_efficiency_percent"],
            operations["fc2"]["calibrated_executed_ceiling_efficiency_percent"],
            padding["padding_efficiency_percent"],
        ),
        "",
        (
            "第 3 节覆盖 FC1、SwiGLU、FC2、TopK reduce 四个主要 op；"
            "Routing、Q0、Q1 只保留在时间 accounting。SwiGLU / TopK reduce 的 NCU DRAM "
            "throughput 为 **{:.2f}% / {:.2f}%**，只作为 sibling-GPU launch-local diagnostic。"
            "所有 op 的独立同契约 SOTA anchor 仍 unavailable。"
        ).format(
            ncu_by_phase["swiglu"]["dram_throughput_percent"],
            ncu_by_phase["topk_reduce"]["dram_throughput_percent"],
        ),
        "",
        "当前 verdict：**accounting=vetted，GEMM percentage=diagnostic（same UUID measured roof；target SASS binding unavailable），coverage=partial**。M256/M1024 只有 E2E benchmark，不能插值出逐 op 占比。",
        "",
        "## 1. Scope 与整体测量",
        "",
        "```text",
        "BF16 input → Routing → Q0 → FC1 → SwiGLU → Q1 → FC2 → TopK reduce → BF16 output",
        "```",
        "",
        "| M | exp_001 E2E | exp_018 独立 sanity | 差异 | per-op timeline |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in model["benchmarks"]:
        lines.append(
            "| {} | {:.3f} μs | {:.3f} μs | {:+.2f}% | {} |".format(
                row["m"],
                row["paired_exp001_us"],
                row["independent_exp018_us"],
                row["difference_percent"],
                row["per_op_timeline"],
            )
        )

    lines.extend(
        [
            "",
            "M8192 NSys graph wall / exp_001 E2E = **{:.2f}%**；工具与采样协议不同，因此只作一致性检查。".format(
                100.0
                * timeline["graph_wall_median_us"]
                / model["benchmarks"][2]["paired_exp001_us"]
            ),
            "",
            "## 2. M8192 各 op 时间与占比",
            "",
            "| Op | Median time | Graph share |",
            "|---|---:|---:|",
        ]
    )
    for row in timeline["phases"]:
        lines.append(
            "| {} | {:.3f} μs | {:.2f}% |".format(
                row["label"], row["time_us"], row["share_percent"]
            )
        )
    lines.extend(
        [
            "",
            "时间取五次 replay 的 median-of-sums，占比取每次 `op time / replay wall` 后的 median。component median sum 与 graph-wall median 相差 **{:+.3f} μs**，各项 median share 之和偏离 100% **{:+.4f} pp**；两者都是 median non-additivity。".format(
                timeline["median_nonadditivity_us"],
                timeline["median_share_nonadditivity_pp"],
            ),
            "",
            "## 3. M8192 主要 op 的资源达成率",
            "",
            "| Op | Graph share | 已观测资源达成率 | 对优化的含义 |",
            "|---|---:|---|---|",
        ]
    )
    for phase in CEILING_PHASES:
        row = operations[phase]
        lines.append(
            "| {} | {:.2f}% | {} | {} |".format(
                row["label"],
                row["share_percent"],
                resource_cell(phase),
                optimization_meaning(phase),
            )
        )

    lines.extend(
        [
            "",
            (
                "TC diagnostic 的分母来自 exp_026 中同一 GPU UUID 的 "
                "`QMMA.16832.F32.E4M3.E4M3` full-card measured roof。目标 Triton 的 "
                "FP8 instruction 目前由 source/dispatch contract 推导，缺少可追溯 target SASS binding，"
                "因此不能称 exact MFU。NCU DRAM throughput 来自 sibling GPU，只允许 launch-local "
                "diagnostic，不能与 NSys 时间混算或称 complete-op efficiency。"
            ),
            "",
            "## 4. 优化优先级与最小下一步",
            "",
            "1. **FC1**：占比最高，优先调查低 occupancy、访存等待与 grouped-GEMM 调度。",
            "2. **TopK reduce + SwiGLU**：DRAM throughput 已接近上限，优先减少 traffic、改善 locality 或融合。",
            "3. **FC2**：继续区分 memory wait 与 TC schedule；现有 diagnostic 不能单独决定优化顺序。",
            "4. 只有结论需要升级时，再补 target SASS binding、同责任 standalone 或独立 SOTA anchor。",
            "",
            "Raw NCU metrics、TFLOP/s、cycle proxy、公式、input digest 与 identity status 见 [model.json](model.json)。",
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
