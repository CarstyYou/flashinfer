#!/usr/bin/env python3
"""Run and audit the Qwen3.5-35B TP W4A4 B12x MoE experiment.

This is a thin experiment controller around ``flashinfer_benchmark.py``.  It
keeps FlashInfer's benchmark implementation as the single source of truth and
adds the evidence that a standalone testlist otherwise loses: environment and
Git provenance, expected-case completeness, resolved dispatch validation, and
a compact summary.

The experiment is performance-only.  It does not turn an exception-free run
into a correctness result; ``run.meta.json`` records correctness as not run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DRIVER = Path(__file__).with_name("flashinfer_benchmark.py")
DEFAULT_TESTLIST = Path(__file__).parent / "samples" / "qwen35_w4a4_moe_testlist.txt"

EXPECTED_SHAPE = {
    "hidden_size": 2048,
    "intermediate_size": 512,
    "num_experts": 256,
    "top_k": 8,
}
EXPECTED_M_VALUES = (256, 512, 1024, 2048, 4096, 8192)
EXPECTED_ROUTING = "renormalize"
EXPECTED_DISPATCH = "dynamic"
EXPECTED_FIXED_OPTIONS = {
    "--routine": "b12x_fused_moe",
    "--hidden_size": "2048",
    "--intermediate_size": "512",
    "--num_experts": "256",
    "--top_k": "8",
    "--routing_method": EXPECTED_ROUTING,
    "--input_dtype": "bfloat16",
    "--weight_dtype": "bfloat16",
    "--activation-type": "Swiglu",
    "--fp4_mode": "nvfp4",
    "--random_seed": "42",
    "--num_iters": "50",
    "--dry_run_iters": "5",
}
EXPECTED_FLAGS = {"--use_cuda_events", "-vv", "--generate_repro_command"}
EXPECTED_RESULT_FIELDS = {
    "activation_type": "Swiglu",
    "fp4_mode": "nvfp4",
    "input_dtype": "torch.bfloat16",
    "weight_dtype": "torch.bfloat16",
    "random_seed": "42",
    "refcheck": "False",
    "allow_output_mismatch": "False",
    "generate_repro_command": "True",
}


class ExperimentError(RuntimeError):
    """Raised when the experiment contract or its evidence is incomplete."""


@dataclass(frozen=True)
class ExperimentCase:
    case_tag: str
    num_tokens: int
    argv: tuple[str, ...]


def _option_value(argv: Sequence[str], option: str) -> str:
    positions = [index for index, token in enumerate(argv) if token == option]
    if not positions:
        raise ExperimentError(f"missing required option {option}")
    if len(positions) != 1:
        raise ExperimentError(f"option {option} must appear exactly once")
    index = positions[0]
    if index + 1 >= len(argv):
        raise ExperimentError(f"missing value for option {option}")
    return argv[index + 1]


def load_cases(testlist: Path) -> list[ExperimentCase]:
    """Parse and validate the fixed Qwen experiment contract without a GPU."""
    cases: list[ExperimentCase] = []
    for line_number, raw_line in enumerate(testlist.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        argv = tuple(shlex.split(line))
        try:
            case_tag = _option_value(argv, "--case_tag")
            num_tokens = int(_option_value(argv, "--num_tokens"))
            shape = {
                name: int(_option_value(argv, f"--{name}")) for name in EXPECTED_SHAPE
            }
            actual_fixed_options = {
                option: _option_value(argv, option) for option in EXPECTED_FIXED_OPTIONS
            }
        except (ExperimentError, ValueError) as exc:
            raise ExperimentError(f"{testlist}:{line_number}: {exc}") from exc

        if shape != EXPECTED_SHAPE:
            raise ExperimentError(
                f"{testlist}:{line_number}: shape {shape} != {EXPECTED_SHAPE}"
            )
        for option, expected in EXPECTED_FIXED_OPTIONS.items():
            actual = actual_fixed_options[option]
            if actual != expected:
                raise ExperimentError(
                    f"{testlist}:{line_number}: {option}={actual!r}, "
                    f"expected {expected!r}"
                )
        actual_flags = {token for token in argv if token in EXPECTED_FLAGS}
        if actual_flags != EXPECTED_FLAGS:
            raise ExperimentError(
                f"{testlist}:{line_number}: flags {sorted(actual_flags)} != "
                f"{sorted(EXPECTED_FLAGS)}"
            )
        expected_argc = 2 * (len(EXPECTED_FIXED_OPTIONS) + 2) + len(EXPECTED_FLAGS)
        if len(argv) != expected_argc:
            raise ExperimentError(
                f"{testlist}:{line_number}: unexpected arguments in fixed contract"
            )
        expected_tag = f"qwen35_w4a4_moe_m{num_tokens}"
        if case_tag != expected_tag:
            raise ExperimentError(
                f"{testlist}:{line_number}: case_tag={case_tag!r}, "
                f"expected {expected_tag!r}"
            )
        cases.append(
            ExperimentCase(case_tag=case_tag, num_tokens=num_tokens, argv=argv)
        )

    tags = [case.case_tag for case in cases]
    if len(tags) != len(set(tags)):
        raise ExperimentError("case_tag values must be unique")
    m_values = tuple(case.num_tokens for case in cases)
    if m_values != EXPECTED_M_VALUES:
        raise ExperimentError(
            f"M sweep {m_values} does not match expected {EXPECTED_M_VALUES}"
        )
    return cases


def _capture(command: Sequence[str], cwd: Path = REPO_ROOT) -> dict[str, object]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        return {"status": "unavailable", "error": str(exc)}
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def collect_provenance() -> dict[str, object]:
    git_head = _capture(["git", "rev-parse", "HEAD"])
    git_branch = _capture(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    git_status = _capture(["git", "status", "--short"])
    nvidia_smi = _capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.device_id,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )

    probe_code = r"""
import importlib.metadata
import json
import torch
import flashinfer

device = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device)
payload = {
    "flashinfer_version": getattr(flashinfer, "__version__", "unknown"),
    "torch_version": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_device_index": device,
    "cuda_device_name": torch.cuda.get_device_name(device),
    "compute_capability": list(torch.cuda.get_device_capability(device)),
    "total_memory_bytes": props.total_memory,
    "l2_bytes": getattr(props, "L2_cache_size", None),
}
for distribution in ("nvidia-cutlass-dsl", "cuda-python", "cupti-python"):
    try:
        payload[distribution] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        payload[distribution] = None
print(json.dumps(payload, sort_keys=True))
"""
    probe = _capture([sys.executable, "-c", probe_code])
    if probe.get("status") == "ok":
        try:
            probe = json.loads(str(probe["stdout"]).splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            probe = {"status": "error", "error": f"invalid probe output: {exc}"}

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git": {
            "head": git_head,
            "branch": git_branch,
            "status": git_status,
        },
        "runtime": probe,
        "nvidia_smi": nvidia_smi,
    }


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _run_and_tee(command: Sequence[str], log_path: Path) -> int:
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return process.wait()


def validate_rows(
    rows: Iterable[Mapping[str, str]], cases: Sequence[ExperimentCase]
) -> list[dict[str, object]]:
    """Reject missing, duplicate, malformed, or wrong-dispatch evidence rows."""
    by_tag: dict[str, Mapping[str, str]] = {}
    for row in rows:
        tag = row.get("case_tag", "")
        if tag in by_tag:
            raise ExperimentError(f"duplicate result row for {tag!r}")
        by_tag[tag] = row

    expected_tags = {case.case_tag for case in cases}
    actual_tags = set(by_tag)
    if actual_tags != expected_tags:
        raise ExperimentError(
            f"result case mismatch: missing={sorted(expected_tags - actual_tags)}, "
            f"unexpected={sorted(actual_tags - expected_tags)}"
        )

    summary: list[dict[str, object]] = []
    for case in cases:
        row = by_tag[case.case_tag]
        try:
            median_ms = float(row["median_time"])
            std_ms = float(row["std_time"])
            row_m = int(row["num_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExperimentError(
                f"invalid numeric result for {case.case_tag}"
            ) from exc
        if not math.isfinite(median_ms) or median_ms <= 0:
            raise ExperimentError(f"invalid latency for {case.case_tag}: {median_ms}")
        if not math.isfinite(std_ms) or std_ms < 0:
            raise ExperimentError(f"invalid std for {case.case_tag}: {std_ms}")
        if row_m != case.num_tokens:
            raise ExperimentError(
                f"M mismatch for {case.case_tag}: result={row_m}, plan={case.num_tokens}"
            )
        if row.get("backend") != "b12x":
            raise ExperimentError(
                f"unexpected backend for {case.case_tag}: {row.get('backend')!r}"
            )
        if row.get("resolved_backend") != EXPECTED_DISPATCH:
            raise ExperimentError(
                f"unexpected dispatch for {case.case_tag}: "
                f"{row.get('resolved_backend')!r}"
            )
        if row.get("routing_method") != EXPECTED_ROUTING:
            raise ExperimentError(
                f"unexpected routing for {case.case_tag}: {row.get('routing_method')!r}"
            )
        if row.get("use_cupti") != "False" or row.get("no_cuda_graph") != "False":
            raise ExperimentError(
                f"unexpected timing flags for {case.case_tag}: "
                f"use_cupti={row.get('use_cupti')!r}, "
                f"no_cuda_graph={row.get('no_cuda_graph')!r}"
            )
        for field, expected in EXPECTED_RESULT_FIELDS.items():
            actual = row.get(field)
            if actual != expected:
                raise ExperimentError(
                    f"unexpected {field} for {case.case_tag}: "
                    f"{actual!r} != {expected!r}"
                )
        summary.append(
            {
                "case_tag": case.case_tag,
                "num_tokens": case.num_tokens,
                "routed_pairs": case.num_tokens * EXPECTED_SHAPE["top_k"],
                "resolved_backend": row["resolved_backend"],
                "median_us": median_ms * 1000.0,
                "std_us": std_ms * 1000.0,
                "tflops": float(row["tflops"]),
                "tb_per_sec": float(row["tb_per_sec"]),
                "correctness": "not_run",
            }
        )
    return summary


def write_summary(output_dir: Path, summary: Sequence[Mapping[str, object]]) -> None:
    csv_path = output_dir / "summary.csv"
    fieldnames = list(summary[0])
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# Qwen3.5 W4A4 B12x MoE experiment",
        "",
        "Performance-only evidence. Correctness was not run by this controller.",
        "",
        "| M | Routed pairs | Dispatch | Median (us) | Std (us) | TFLOPS |",
        "|---:|---:|:---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['num_tokens']} | {row['routed_pairs']} | "
            f"{row['resolved_backend']} | {row['median_us']:.3f} | "
            f"{row['std_us']:.3f} | {row['tflops']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Timing contract: preselected-topk BF16-to-BF16 fused MoE, CUDA Graph "
            "event timing, FlashInfer cold-L2 policy; router/top-k and weight "
            "preparation are outside the timed region.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testlist", type=Path, default=DEFAULT_TESTLIST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory; defaults to /tmp/flashinfer-qwen35-w4a4-<UTC>.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the experiment contract without importing CUDA packages.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    testlist = args.testlist.resolve()
    cases = load_cases(testlist)
    plan = {
        "schema_version": 1,
        "operator_scope": "preselected_topk_bf16_to_bf16_fused_moe",
        "shape": EXPECTED_SHAPE,
        "m_values": [case.num_tokens for case in cases],
        "routing_method": EXPECTED_ROUTING,
        "random_seed": 42,
        "measurement_iterations": 50,
        "warmup_iterations": 5,
        "timing_requested": "cuda_graph_events_cold_l2",
        "expected_dispatch": EXPECTED_DISPATCH,
        "correctness": {
            "status": "not_run",
            "reason": "This controller is the performance evidence layer only.",
        },
        "testlist": str(testlist),
    }
    if args.validate_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path(f"/tmp/flashinfer-qwen35-w4a4-{timestamp}")
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ExperimentError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(testlist, output_dir / "cases.txt")

    meta = {
        **plan,
        "run_id": output_dir.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "provenance": collect_provenance(),
    }
    meta_path = output_dir / "run.meta.json"
    _write_json(meta_path, meta)

    raw_csv = output_dir / "raw.csv"
    log_path = output_dir / "run.log"
    command = [
        sys.executable,
        str(BENCHMARK_DRIVER),
        "--testlist",
        str(testlist),
        "--output_path",
        str(raw_csv),
    ]
    meta["command"] = command
    _write_json(meta_path, meta)

    try:
        returncode = _run_and_tee(command, log_path)
        if returncode != 0:
            raise ExperimentError(f"benchmark driver exited with {returncode}")
        with raw_csv.open(newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        summary = validate_rows(rows, cases)
        write_summary(output_dir, summary)
    except Exception as exc:
        meta["status"] = "failed"
        meta["error"] = str(exc)
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(meta_path, meta)
        raise

    meta["status"] = "complete"
    meta["result_rows"] = len(summary)
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(meta_path, meta)
    print(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
