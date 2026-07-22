#!/usr/bin/env python3
"""Build the RTX 5KP calibrated-ceiling profile and percentage-first report."""

import csv
import hashlib
import io
import json
import re
import statistics
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
RESULTS = EXP_DIR / "results"
RAW = RESULTS / "raw"

SM_COUNT = 110
APP_CLOCK_HZ = 2_377_000_000
MEM_CLOCK_HZ = 14_001_000_000
MEM_BUS_BITS = 384
BUFFER_BYTES = 512 * 1024 * 1024

MODES = {
    "fp8-e4m3-noscale": {
        "label": "FP8 E4M3×E4M3 no-scale",
        "ipc_tag": "E4M3_E4M3_FP32_QMMA16832",
        "sass": "QMMA.16832.F32.E4M3.E4M3",
        "source_binding": "run_no_scale_test",
        "instruction_k": 32,
        "nominal_work_per_cycle_sm": 2048.0,
        "scope": "exp_025 SGLang Triton FP8 FC1/FC2",
        "status": "vetted",
    },
    "nvfp4-e2m1-vs16": {
        "label": "NVFP4 E2M1×E2M1 VS16",
        "ipc_tag": "NVFP4_FP32_OMMA16864_VS16",
        "sass": "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X",
        "source_binding": "run_blockscaled_test",
        "instruction_k": 64,
        "nominal_work_per_cycle_sm": 4096.0,
        "scope": "exp_024 CUTLASS NVFP4 FC1/FC2",
        "status": "vetted",
    },
    "fp8-e4m3-vs32": {
        "label": "FP8 E4M3×E4M3 VS32",
        "ipc_tag": "E4M3_E4M3_FP32_QMMA16832_VS32",
        "sass": "QMMA.SF.16832.F32.E4M3.E4M3.E8",
        "source_binding": "run_blockscaled_test",
        "instruction_k": 32,
        "nominal_work_per_cycle_sm": 2048.0,
        "scope": "block-scaled FP8 diagnostic only",
        "status": "diagnostic",
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(values):
    return float(statistics.median(values))


def cv_percent(values):
    mean = sum(values) / len(values)
    return (
        0.0
        if len(values) < 2 or mean == 0
        else 100.0 * statistics.pstdev(values) / mean
    )


def parse_number(value):
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", value)
    if not match:
        raise ValueError("no numeric value in {!r}".format(value))
    return float(match.group(0))


def parse_source_identity():
    rows = {}
    hashes = {}
    for line in (RAW / "source_identity.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line and not re.match(r"^[0-9a-f]{64}\s", line):
            key, value = line.split("=", 1)
            rows[key] = value
        elif re.match(r"^[0-9a-f]{64}\s", line):
            digest, path = line.split(None, 1)
            hashes[path] = digest
    require(rows.get("benchmark_status") == "clean", "benchmark source was not clean")
    require(
        re.fullmatch(r"[0-9a-f]{40}", rows.get("benchmark_commit", "")),
        "missing benchmark commit",
    )
    rows["source_sha256"] = hashes
    return rows


def parse_remote_ref_identity(source):
    rows = {}
    for line in (
        (RAW / "remote_ref_identity.txt").read_text(encoding="utf-8").splitlines()
    ):
        key, value = line.split("=", 1)
        rows[key] = value
    require(
        rows.get("benchmark_remote") == source["benchmark_remote"],
        "benchmark remote drift",
    )
    require(rows.get("remote_ref") == "refs/heads/main", "canonical remote ref drift")
    require(
        rows.get("remote_ref_commit") == source["benchmark_commit"],
        "benchmark commit is not on canonical remote main",
    )
    return rows


def parse_ipc():
    values = {mode: [] for mode in MODES}
    pattern = re.compile(
        r"IPC of\s+([^:]+?)\s*:\s*[0-9.]+T/clk/SM, Ops/clk/SM\s+([0-9.]+)"
    )
    for path in sorted((RAW / "ipc").glob("run_*.log")):
        found = {}
        for tag, value in pattern.findall(path.read_text(encoding="utf-8")):
            found[tag.strip()] = float(value)
        for mode, contract in MODES.items():
            require(
                contract["ipc_tag"] in found,
                "{} missing in {}".format(contract["ipc_tag"], path),
            )
            values[mode].append(found[contract["ipc_tag"]])
    require(
        all(len(rows) == 5 for rows in values.values()), "expected five IPC repetitions"
    )
    return {
        mode: {
            "samples_work_per_cycle_sm": rows,
            "median_work_per_cycle_sm": median(rows),
            "min_work_per_cycle_sm": min(rows),
            "max_work_per_cycle_sm": max(rows),
            "cv_percent": cv_percent(rows),
        }
        for mode, rows in values.items()
    }


def parse_throttle(path):
    text = path.read_text(encoding="utf-8")
    config = re.search(
        r"Device: .*?, (\d+) SMs, blocks_per_sm=(\d+), grid=(\d+) blocks.*?\n"
        r"Config: num_kernels=(\d+) testtimes=(\d+) gap_us=(\d+) mode=([^\s]+)",
        text,
    )
    require(config, "missing throttle config in {}".format(path))
    resource = re.search(
        r"Kernel resources: registers/thread=(\d+) static_smem=(\d+)B "
        r"max_active_one_warp_blocks/SM=(\d+)",
        text,
    )
    require(resource, "missing resource line in {}".format(path))
    rows = []
    for line in text.splitlines():
        if re.match(r"^\d+,", line):
            index, duration, rate = line.split(",")
            rows.append(
                {
                    "kernel_idx": int(index),
                    "duration_us": float(duration),
                    "tflops": float(rate),
                }
            )
    require(rows, "missing throttle rows in {}".format(path))
    return {
        "sm_count": int(config.group(1)),
        "blocks_per_sm": int(config.group(2)),
        "grid_blocks": int(config.group(3)),
        "num_kernels": int(config.group(4)),
        "testtimes": int(config.group(5)),
        "gap_us": int(config.group(6)),
        "mode": config.group(7),
        "registers_per_thread": int(resource.group(1)),
        "static_smem_bytes": int(resource.group(2)),
        "max_active_one_warp_blocks_per_sm": int(resource.group(3)),
        "rows": rows,
    }


def parse_compute_windows():
    saturation = {}
    windows = {}
    recovery = {}
    for mode in MODES:
        by_blocks = {}
        for blocks in (4, 8, 12, 16):
            parsed = parse_throttle(
                RAW / "saturation" / "{}_b{}.log".format(mode, blocks)
            )
            require(
                parsed["mode"] == mode and parsed["blocks_per_sm"] == blocks,
                "saturation identity drift",
            )
            require(
                parsed["sm_count"] == SM_COUNT and len(parsed["rows"]) == 8,
                "saturation launch drift",
            )
            by_blocks[blocks] = median([row["tflops"] for row in parsed["rows"][1:]])
        best = max(by_blocks.values())
        saturation[mode] = {
            "median_tflops_by_blocks_per_sm": {
                str(key): value for key, value in by_blocks.items()
            },
            "selected_blocks_per_sm": 8,
            "selected_vs_best_percent": 100.0 * by_blocks[8] / best,
            "best_observed_blocks_per_sm": max(by_blocks, key=by_blocks.get),
        }

        runs = []
        for rep in (1, 2, 3):
            parsed = parse_throttle(
                RAW / "compute_window" / "{}_run{}.log".format(mode, rep)
            )
            require(parsed["mode"] == mode, "compute-window mode drift")
            require(
                parsed["sm_count"] == SM_COUNT and parsed["blocks_per_sm"] == 8,
                "compute-window grid drift",
            )
            require(
                parsed["num_kernels"] == 32 and parsed["testtimes"] == 64000,
                "compute-window protocol drift",
            )
            runs.append(
                {
                    "rested_first_tflops": parsed["rows"][0]["tflops"],
                    "last8_median_tflops": median(
                        [row["tflops"] for row in parsed["rows"][-8:]]
                    ),
                    "registers_per_thread": parsed["registers_per_thread"],
                    "max_active_one_warp_blocks_per_sm": parsed[
                        "max_active_one_warp_blocks_per_sm"
                    ],
                }
            )
        sustained = [row["last8_median_tflops"] for row in runs]
        windows[mode] = {
            "runs": runs,
            "sustained_window_tflops": median(sustained),
            "sustained_window_cv_percent": cv_percent(sustained),
            "rested_first_tflops": median([row["rested_first_tflops"] for row in runs]),
            "window_definition": "median(last 8 kernels) of each 32×64000 back-to-back run; median of 3 runs",
        }

        parsed = parse_throttle(RAW / "recovery" / "{}.log".format(mode))
        recovery[mode] = {
            "gap_us": parsed["gap_us"],
            "median_tflops": median([row["tflops"] for row in parsed["rows"]]),
            "cv_percent": cv_percent([row["tflops"] for row in parsed["rows"]]),
            "role": "directional recovery diagnostic; not a denominator",
        }
    return saturation, windows, recovery


def parse_dram_log(path):
    text = path.read_text(encoding="utf-8")
    patterns = {
        "read": r"Streaming Read\s*:\s*([0-9.]+) GB/s",
        "write": r"Streaming Write\s*:\s*([0-9.]+) GB/s",
        "copy": r"Copy \(R\+W\)\s*:\s*([0-9.]+) GB/s",
        "d2d": r"cudaMemcpy D2D\s*:\s*([0-9.]+) GB/s",
        "nominal": r"peak:\s*([0-9.]+) GB/s",
    }
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        require(match, "{} missing from {}".format(key, path))
        result[key] = float(match.group(1))
    return result


def parse_ncu_csv(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(
        index for index, line in enumerate(lines) if line.startswith('"ID"')
    )
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    rows = {}
    for row in reader:
        require(row["Metric Unit"] == "byte", "NCU DRAM metric unit drift")
        require(row["Metric Value"] != "n/a", "NCU DRAM metric unavailable")
        rows.setdefault(row["Metric Name"], []).append(float(row["Metric Value"]))
    for metric in ("dram__bytes_op_read.sum", "dram__bytes_op_write.sum"):
        require(
            len(rows.get(metric, [])) == 4,
            "expected four NCU launches for {}".format(metric),
        )
    return rows


def parse_dram():
    samples = [
        parse_dram_log(RAW / "dram" / "run_{}.log".format(rep)) for rep in (1, 2, 3)
    ]
    payload = {
        key: [sample[key] for sample in samples]
        for key in ("read", "write", "copy", "d2d", "nominal")
    }
    ncu = {
        "read": parse_ncu_csv(RAW / "dram" / "stream_read_kernel_ncu.csv"),
        "write": parse_ncu_csv(RAW / "dram" / "stream_write_kernel_ncu.csv"),
        "copy": parse_ncu_csv(RAW / "dram" / "stream_copy_kernel_ncu.csv"),
    }
    read_ratio = median(ncu["read"]["dram__bytes_op_read.sum"]) / BUFFER_BYTES
    write_ratio = median(ncu["write"]["dram__bytes_op_write.sum"]) / BUFFER_BYTES
    copy_ratio = median(
        [
            read + write
            for read, write in zip(  # noqa: B905 -- Python 3.6
                ncu["copy"]["dram__bytes_op_read.sum"],
                ncu["copy"]["dram__bytes_op_write.sum"],
            )
        ]
    ) / (2 * BUFFER_BYTES)
    require(0.98 <= read_ratio <= 1.02, "stream-read physical byte scope mismatch")
    require(0.85 <= write_ratio <= 1.02, "stream-write physical byte scope mismatch")
    require(0.90 <= copy_ratio <= 1.02, "stream-copy physical byte scope mismatch")

    physical = {
        "read": median(payload["read"]) * read_ratio,
        "write": median(payload["write"]) * write_ratio,
        "copy": median(payload["copy"]) * copy_ratio,
    }
    nominal = median(payload["nominal"])
    return {
        "payload_gbs_samples": payload,
        "physical_byte_ratio": {
            "read": read_ratio,
            "write": write_ratio,
            "copy": copy_ratio,
        },
        "physical_gbs": physical,
        "nominal_gbs": nominal,
        "physical_vs_nominal_percent": {
            key: 100.0 * value / nominal for key, value in physical.items()
        },
        "run_cv_percent": {
            key: cv_percent(payload[key]) for key in ("read", "write", "copy", "d2d")
        },
        "ncu_launches": ncu,
        "d2d_payload_gbs": median(payload["d2d"]),
        "d2d_role": "runtime D2D copy diagnostic only; not a kernel roof",
        "scope": "physical DRAM bytes; 512 MiB, 4-pass timed sequence",
    }


def parse_telemetry(paths):
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, skipinitialspace=True):
                rows.append(
                    {
                        "clock_mhz": parse_number(row["clocks.current.graphics [MHz]"]),
                        "power_w": parse_number(row["power.draw [W]"]),
                        "temperature_c": parse_number(row["temperature.gpu"]),
                        "util_percent": parse_number(row["utilization.gpu [%]"]),
                        "memory_mib": parse_number(row["memory.used [MiB]"]),
                        "pstate": row["pstate"].strip(),
                    }
                )
    active = [row for row in rows if row["util_percent"] > 0]
    selected = active or rows
    return {
        "samples": len(rows),
        "active_samples": len(active),
        "active_clock_mhz_median": median([row["clock_mhz"] for row in selected]),
        "active_clock_mhz_min": min(row["clock_mhz"] for row in selected),
        "active_clock_mhz_max": max(row["clock_mhz"] for row in selected),
        "active_power_w_max": max(row["power_w"] for row in selected),
        "temperature_c_max": max(row["temperature_c"] for row in selected),
        "memory_mib_max": max(row["memory_mib"] for row in rows),
        "pstates": sorted({row["pstate"] for row in selected}),
    }


def build():
    source = parse_source_identity()
    source["remote_ref_identity"] = parse_remote_ref_identity(source)
    ipc = parse_ipc()
    saturation, windows, recovery = parse_compute_windows()
    dram = parse_dram()

    build_log = (RAW / "build.log").read_text(encoding="utf-8")
    require(
        "ninja: no work to do" not in build_log, "formal binaries were not clean-built"
    )
    require("Cleaning..." in build_log, "clean-first build evidence missing")
    sass = (RAW / "sass_instructions.txt").read_text(encoding="utf-8")
    sass_bindings = (RAW / "sass_mode_bindings.txt").read_text(encoding="utf-8")
    source_bindings = (RAW / "source_mode_bindings.txt").read_text(encoding="utf-8")
    tensor_records = []
    for mode, contract in MODES.items():
        require(contract["sass"] in sass, "{} SASS missing".format(mode))
        require(
            contract["sass"] in sass_bindings,
            "{} 64000-loop SASS binding missing".format(mode),
        )
        require(mode in source_bindings, "{} source selector missing".format(mode))
        require(
            contract["source_binding"] in source_bindings,
            "{} source template missing".format(mode),
        )
        require(ipc[mode]["cv_percent"] <= 1.0, "{} IPC CV exceeds 1%".format(mode))
        require(
            saturation[mode]["selected_vs_best_percent"] >= 99.0,
            "{} 8-block grid is not saturated".format(mode),
        )
        require(
            windows[mode]["sustained_window_cv_percent"] <= 1.0,
            "{} window CV exceeds 1%".format(mode),
        )
        nominal_tflops = (
            contract["nominal_work_per_cycle_sm"] * SM_COUNT * APP_CLOCK_HZ / 1.0e12
        )
        telemetry = parse_telemetry(
            sorted(
                (RAW / "telemetry" / "compute_window").glob("{}_run*.csv".format(mode))
            )
        )
        tensor_records.append(
            {
                "id": mode,
                "label": contract["label"],
                "instruction": contract["sass"],
                "source_dispatch": contract["source_binding"],
                "consumer_scope": contract["scope"],
                "audit_status": contract["status"],
                "architecture_nominal": {
                    "status": "diagnostic",
                    "work_per_cycle_sm": contract["nominal_work_per_cycle_sm"],
                    "derivation": "4 tensor subpartitions × (2×16×8×K FLOPs/instruction) / 16 cycles",
                    "instruction_k": contract["instruction_k"],
                    "evidence_locator": "sm120_mma_benchmarks/IPC_bench_recipe.cuh: TC_benchmark_wrapper",
                },
                "per_sm_instruction_ceiling": ipc[mode],
                "per_cycle_vs_arch_nominal_percent": 100.0
                * ipc[mode]["median_work_per_cycle_sm"]
                / contract["nominal_work_per_cycle_sm"],
                "saturation": saturation[mode],
                "sustained_window": windows[mode],
                "sustained_window_vs_app_clock_nominal_percent": 100.0
                * windows[mode]["sustained_window_tflops"]
                / nominal_tflops,
                "nominal_app_clock_tflops": nominal_tflops,
                "telemetry": telemetry,
                "recovery": recovery[mode],
            }
        )

    require(
        all(value <= 3.0 for value in dram["run_cv_percent"].values()),
        "DRAM CV exceeds 3%",
    )
    pre = (RAW / "gpu_pre.csv").read_text(encoding="utf-8")
    post = (RAW / "gpu_post.csv").read_text(encoding="utf-8")
    require(
        "GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522" in pre
        and "GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522" in post,
        "GPU identity drift",
    )
    require(
        "visible_devices=1"
        in (RAW / "container_gpu_identity.txt").read_text(encoding="utf-8"),
        "container GPU isolation failed",
    )
    require(
        "sm_count=110"
        in (RAW / "container_gpu_identity.txt").read_text(encoding="utf-8"),
        "SM count drift",
    )

    raw_hashes = {
        str(path.relative_to(EXP_DIR)): sha256(path)
        for path in sorted(RAW.rglob("*"))
        if path.is_file()
    }
    profile = {
        "schema": "calibrated-gpu-ceiling-profile.v1",
        "profile_id": "rtx5kp-sm120-20260722",
        "policy": "development / measured compatible roof first",
        "gpu": {
            "internal_sku": "RTX 5KP",
            "architecture": "SM120",
            "uuid": "GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522",
            "host": "R6KD-CX8aaS-GPU-16",
            "sm_count": SM_COUNT,
            "application_graphics_clock_mhz": APP_CLOCK_HZ / 1e6,
            "application_memory_clock_mhz": MEM_CLOCK_HZ / 1e6,
            "memory_bus_bits": MEM_BUS_BITS,
            "power_limit_w": 350.0,
        },
        "benchmark": source,
        "protocol": {
            "tensor_per_cycle": "5 independent process invocations; median Ops/clk/SM",
            "tensor_saturation": "4/8/12/16 one-warp blocks per installed SM; exclude rested kernel 0, then select smallest >=99% of best",
            "tensor_window": "3 runs of 32 kernels × 64000 iterations; last-8 median",
            "dram": "3 process invocations; each median of 10 trials × 4 passes; NCU validates four consecutive launches",
        },
        "tensor_core_records": tensor_records,
        "dram_record": dict(dram, audit_status="vetted"),
        "raw_evidence_sha256": raw_hashes,
        "audit": {
            "vetted": [
                "exact FP8 no-scale and NVFP4 VS16 SASS identity",
                "clean source-to-binary build and exact 64000-loop mode/SASS binding",
                "benchmark commit reachable as canonical remote main",
                "five-run per-cycle stability",
                "8 blocks/SM within 99% of density sweep maximum",
                "three-run ~100 ms compute-window stability",
                "DRAM three-run stability and physical read/write counter scope",
                "target UUID, 110 SMs, single visible container GPU, clean benchmark commit",
            ],
            "diagnostic": [
                "FP8 VS32 record is not consumed by exp_025",
                "architecture nominal ratios use a documented 16-cycle issue assumption",
                "2377 MHz application-clock nominal ratios do not close per-kernel effective clock; boost is plausible but not attributable from sampled telemetry",
                "gap/recovery and runtime D2D records are not denominators",
            ],
            "unavailable": [
                "operator SOTA ceilings",
                "atomic/reduction/latency service ceilings",
                "infinite-duration thermal steady-state compute roof",
            ],
        },
        "decision": "accept",
    }
    return profile


def render(profile):
    tensor = {row["id"]: row for row in profile["tensor_core_records"]}
    dram = profile["dram_record"]
    lines = [
        "# RTX 5KP Calibrated Hardware Ceiling",
        "",
        "## 结论",
        "",
        "本 profile 已通过：**accept**。exp_024 的 NVFP4 与 exp_025 的 FP8 no-scale "
        "现在都有 compatible measured ceiling；报告消费端应优先展示下面的 ceiling "
        "达成率，而不是原始 TFLOP/s / GB/s。这里不建立 operator SOTA。",
        "",
        "## Tensor Core ceiling 质量",
        "",
        "| 模式 | Per-cycle / architecture nominal（diagnostic） | 8 blocks/SM saturation | ~100 ms roof / app-clock nominal（diagnostic） | 重复稳定性 | measured record 状态 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for mode in ("nvfp4-e2m1-vs16", "fp8-e4m3-noscale", "fp8-e4m3-vs32"):
        row = tensor[mode]
        lines.append(
            "| {} | {:.2f}% | {:.2f}% | {:.2f}% | CV {:.3f}% | {} |".format(
                row["label"],
                row["per_cycle_vs_arch_nominal_percent"],
                row["saturation"]["selected_vs_best_percent"],
                row["sustained_window_vs_app_clock_nominal_percent"],
                row["sustained_window"]["sustained_window_cv_percent"],
                "✓ vetted" if row["audit_status"] == "vetted" else "⚠ diagnostic",
            )
        )
    lines.extend(
        [
            "",
            "`~100 ms / nominal` 超过 100% 表示 2377 MHz application-clock nominal "
            "与该测量窗口的有效频率/计数口径尚未闭合；结果与 boost 一致，但现有 telemetry "
            "不能按 kernel 完成归因，因此此列仅作 diagnostic，也不截断为 100%。消费端可用 per-cycle "
            "normalization 时优先用 per-cycle record；否则使用同 window 的 full-card record。",
            "",
            "`Per-cycle / architecture nominal` 同样是带显式 16-cycle issue 假设的校准诊断；"
            "表中的 vetted 指 exact-mode measured record 与证据身份，而不是把 nominal 比值升级为官方 SKU spec。",
            "",
            "## DRAM physical ceiling 质量",
            "",
            "| 方向 | Physical ceiling / memory-clock nominal | 三次重复稳定性 | 状态 |",
            "|---|---:|---:|---|",
            "| Read | {:.2f}% | CV {:.3f}% | ✓ vetted |".format(
                dram["physical_vs_nominal_percent"]["read"],
                dram["run_cv_percent"]["read"],
            ),
            "| Write | {:.2f}% | CV {:.3f}% | ✓ vetted |".format(
                dram["physical_vs_nominal_percent"]["write"],
                dram["run_cv_percent"]["write"],
            ),
            "| Copy (R+W) | {:.2f}% | CV {:.3f}% | ✓ vetted |".format(
                dram["physical_vs_nominal_percent"]["copy"],
                dram["run_cv_percent"]["copy"],
            ),
            "",
            "DRAM 百分比使用 NCU physical read/write bytes 校准；D2D 仅作 runtime copy "
            "diagnostic。Reduction/atomic 不能据此宣称完整 op ceiling。",
            "",
            "完整 raw rates、公式输入、telemetry、binary/source digest 与适用范围见 "
            "[profile.json](profile.json)。",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    profile = build()
    with (RESULTS / "profile.json").open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2, sort_keys=True)
        handle.write("\n")
    (RESULTS / "result.md").write_text(render(profile), encoding="utf-8")


if __name__ == "__main__":
    main()
