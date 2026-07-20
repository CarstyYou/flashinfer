#!/usr/bin/env python3
"""Collect fail-closed static cubin/resource evidence for exp_015.

Both cubins are disassembled afresh in the same invocation.  The caller is
responsible for supplying cubins produced by the locked fresh-JIT preparation;
this collector deliberately does not infer build freshness from timestamps.
"""

import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid


SCHEMA = "exp015.static_resource_evidence.v1"
REGISTER_CAP = 160
EXPECTED_OMMA = 448

RESOURCE_RE = re.compile(
    r"^\s*Function\s+(\S+):\s*\r?\n\s*"
    r"REG\s*:\s*(\d+)\s+STACK\s*:\s*(\d+)\s+"
    r"SHARED\s*:\s*(\d+)\s+LOCAL\s*:\s*(\d+)",
    re.IGNORECASE | re.MULTILINE,
)
SASS_INSTRUCTION_RE = re.compile(
    r"/\*\s*([0-9a-fA-F]+)\s*\*/\s+"
    r"(?:@[!A-Za-z0-9.]+\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;"
)
SASS_GLOBAL_RE = re.compile(r"^\s*\.global\s+([^\s,;]+)\s*$", re.MULTILINE)
SASS_FUNCTION_TYPE_RE = re.compile(
    r"^\s*\.type\s+([^\s,;]+)\s*,\s*@function\b", re.MULTILINE
)


class EvidenceError(ValueError):
    pass


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def opcode_family_count(histogram, family):
    return sum(
        count
        for opcode, count in histogram.items()
        if opcode == family or opcode.startswith(family + ".")
    )


def parse_resource_usage(text):
    matches = RESOURCE_RE.findall(text)
    if len(matches) != 1:
        symbols = [match[0] for match in matches]
        raise EvidenceError(
            "expected exactly one complete kernel resource record; found {} ({})"
            .format(len(matches), ", ".join(symbols) or "none")
        )
    symbol, registers, stack, shared, local = matches[0]
    if not symbol:
        raise EvidenceError("kernel resource symbol is empty")
    return {
        "kernel_symbol": symbol,
        "registers_per_thread": int(registers),
        "stack_bytes_per_thread": int(stack),
        "shared_bytes_per_cta": int(shared),
        "local_bytes_outside_stack": int(local),
    }


def parse_sass(text):
    global_symbols = set(SASS_GLOBAL_RE.findall(text))
    function_symbols = set(SASS_FUNCTION_TYPE_RE.findall(text))
    entry_symbols = sorted(global_symbols & function_symbols)
    if len(entry_symbols) != 1:
        raise EvidenceError(
            "expected exactly one unambiguous global @function symbol; found {} ({})"
            .format(len(entry_symbols), ", ".join(entry_symbols) or "none")
        )

    instructions = []
    for match in SASS_INSTRUCTION_RE.finditer(text):
        instructions.append(
            {
                "pc": int(match.group(1), 16),
                "opcode": match.group(2),
                "operands": match.group(3).strip(),
            }
        )
    if not instructions:
        raise EvidenceError("no SASS instructions parsed")

    pcs = [row["pc"] for row in instructions]
    if len(set(pcs)) != len(pcs):
        raise EvidenceError(
            "duplicate SASS PCs indicate multiple code functions or ambiguous parsing"
        )
    if pcs != sorted(pcs):
        raise EvidenceError("SASS PCs are not monotonically increasing")

    histogram = Counter(row["opcode"] for row in instructions)
    exit_count = opcode_family_count(histogram, "EXIT")
    if exit_count == 0:
        raise EvidenceError("no EXIT instruction parsed; disassembly may be incomplete")

    selected = {
        "ldl": opcode_family_count(histogram, "LDL"),
        "stl": opcode_family_count(histogram, "STL"),
        "omma": opcode_family_count(histogram, "OMMA"),
        "call": opcode_family_count(histogram, "CALL"),
        "ret": opcode_family_count(histogram, "RET"),
        "exit": exit_count,
    }
    return {
        "kernel_symbol": entry_symbols[0],
        "instruction_count": len(instructions),
        "pc_first_hex": hex(pcs[0]),
        "pc_last_hex": hex(pcs[-1]),
        "selected_instruction_counts": selected,
        "opcode_histogram": dict(sorted(histogram.items())),
    }


def analyze_static_outputs(resource_text, sass_text):
    resource = parse_resource_usage(resource_text)
    sass = parse_sass(sass_text)
    if resource["kernel_symbol"] != sass["kernel_symbol"]:
        raise EvidenceError(
            "kernel symbol mismatch between cuobjdump and nvdisasm: {!r} != {!r}"
            .format(resource["kernel_symbol"], sass["kernel_symbol"])
        )
    return {
        "kernel_symbol": resource["kernel_symbol"],
        "resource": {
            key: value for key, value in resource.items() if key != "kernel_symbol"
        },
        "sass": {
            key: value for key, value in sass.items() if key != "kernel_symbol"
        },
    }


def evaluate_arm_gates(analysis):
    resource = analysis["resource"]
    counts = analysis["sass"]["selected_instruction_counts"]
    gates = {
        "registers_at_most_160": resource["registers_per_thread"]
        <= REGISTER_CAP,
        "stack_zero": resource["stack_bytes_per_thread"] == 0,
        "local_zero": resource["local_bytes_outside_stack"] == 0,
        "ldl_zero": counts["ldl"] == 0,
        "stl_zero": counts["stl"] == 0,
        "omma_exactly_448": counts["omma"] == EXPECTED_OMMA,
    }
    return {
        "pass": all(gates.values()),
        "checks": gates,
        "failed": sorted(name for name, passed in gates.items() if not passed),
    }


def evaluate_comparison(baseline, candidate):
    baseline_counts = baseline["sass"]["selected_instruction_counts"]
    candidate_counts = candidate["sass"]["selected_instruction_counts"]
    gates = {
        "candidate_adds_no_call": candidate_counts["call"]
        <= baseline_counts["call"],
        "candidate_adds_no_ret": candidate_counts["ret"] <= baseline_counts["ret"],
    }
    both_zero = all(
        counts[name] == 0
        for counts in (baseline_counts, candidate_counts)
        for name in ("call", "ret")
    )
    return {
        "pass": all(gates.values()),
        "checks": gates,
        "failed": sorted(name for name, passed in gates.items() if not passed),
        "call_delta_candidate_minus_baseline": candidate_counts["call"]
        - baseline_counts["call"],
        "ret_delta_candidate_minus_baseline": candidate_counts["ret"]
        - baseline_counts["ret"],
        "both_arms_call_ret_zero": both_zero,
        "kernel_symbol_equal": baseline["kernel_symbol"]
        == candidate["kernel_symbol"],
        "resource_delta_candidate_minus_baseline": {
            key: candidate["resource"][key] - baseline["resource"][key]
            for key in (
                "registers_per_thread",
                "stack_bytes_per_thread",
                "shared_bytes_per_cta",
                "local_bytes_outside_stack",
            )
        },
    }


def decode_tool_output(value, label):
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceError(
            "{} is not valid UTF-8: {}".format(label, exc)
        ) from exc


def command_record(argv, completed):
    return {
        "argv": list(argv),
        "returncode": completed.returncode,
        "stdout_bytes": len(completed.stdout),
        "stdout_sha256": sha256_bytes(completed.stdout),
        "stderr_bytes": len(completed.stderr),
        "stderr_sha256": sha256_bytes(completed.stderr),
    }


def run_checked(argv, timeout_seconds):
    try:
        completed = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EvidenceError(
            "command could not run {!r}: {}".format(argv, exc)
        ) from exc
    if completed.returncode != 0:
        stderr = decode_tool_output(completed.stderr, "command stderr")
        raise EvidenceError(
            "command failed with exit {}: {!r}: {}".format(
                completed.returncode, argv, stderr[-2000:].strip()
            )
        )
    return completed


def resolve_tool(requested):
    if not requested or "\x00" in requested:
        raise EvidenceError("invalid empty/NUL tool name")
    resolved = shutil.which(requested)
    if resolved is None:
        raise EvidenceError("tool not found on PATH: {!r}".format(requested))
    return str(Path(resolved).resolve())


def collect_tool_identity(requested, timeout_seconds):
    resolved = resolve_tool(requested)
    argv = [resolved, "--version"]
    completed = run_checked(argv, timeout_seconds)
    combined = completed.stdout
    if completed.stderr:
        combined += b"\n" + completed.stderr
    version = decode_tool_output(combined, "tool version").strip()
    if not version:
        raise EvidenceError("tool returned an empty --version: {}".format(resolved))
    return {
        "requested": requested,
        "resolved": resolved,
        "version": version,
        "version_run": command_record(argv, completed),
    }


def cubin_identity(path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise EvidenceError("cubin is not a regular file: {}".format(resolved))
    size = resolved.stat().st_size
    if size <= 0:
        raise EvidenceError("cubin is empty: {}".format(resolved))
    return {
        "path": str(resolved),
        "size_bytes": size,
        "sha256": sha256_file(resolved),
    }


def collect_arm(label, cubin_path, tools, timeout_seconds):
    record = {"label": label, "status": "fail"}
    try:
        before = cubin_identity(cubin_path)
        record["cubin"] = before
        resource_argv = [
            tools["cuobjdump"]["resolved"],
            "--dump-resource-usage",
            before["path"],
        ]
        sass_argv = [tools["nvdisasm"]["resolved"], "-c", before["path"]]
        resource_run = run_checked(resource_argv, timeout_seconds)
        sass_run = run_checked(sass_argv, timeout_seconds)
        after = cubin_identity(Path(before["path"]))
        if before != after:
            raise EvidenceError("cubin changed while evidence was being collected")

        resource_text = decode_tool_output(
            resource_run.stdout, "{} cuobjdump stdout".format(label)
        )
        sass_text = decode_tool_output(
            sass_run.stdout, "{} nvdisasm stdout".format(label)
        )
        analysis = analyze_static_outputs(resource_text, sass_text)
        gates = evaluate_arm_gates(analysis)
        record.update(analysis)
        record["gates"] = gates
        record["tool_runs"] = {
            "cuobjdump_resource": command_record(resource_argv, resource_run),
            "nvdisasm_sass": command_record(sass_argv, sass_run),
        }
        record["status"] = "pass" if gates["pass"] else "fail"
        if not gates["pass"]:
            record["error"] = "required static gates failed: {}".format(
                ", ".join(gates["failed"])
            )
        return record
    except (EvidenceError, OSError) as exc:
        record["error"] = str(exc)
        return record


def write_json_atomic(path, value):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        ".{}.tmp.{}.{}".format(path.name, os.getpid(), uuid.uuid4().hex)
    )
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def build_report(args):
    report = {
        "schema": SCHEMA,
        "analysis_id": str(uuid.uuid4()),
        "generated_at_utc": utc_now(),
        "status": "fail",
        "scope": {
            "static_binary_evidence_only": True,
            "both_arms_disassembled_in_this_invocation": True,
            "artifact_freshness_contract": (
                "caller supplies cubins from the locked fresh-JIT preparation; "
                "this CLI verifies file stability and does not infer JIT freshness "
                "from timestamps"
            ),
        },
        "thresholds": {
            "registers_per_thread_max": REGISTER_CAP,
            "stack_bytes_per_thread": 0,
            "local_bytes_outside_stack": 0,
            "ldl_static_instruction_count": 0,
            "stl_static_instruction_count": 0,
            "omma_static_instruction_count": EXPECTED_OMMA,
            "candidate_call_ret_must_not_exceed_baseline": True,
        },
        "tools": {},
        "arms": {},
        "comparison": None,
        "warnings": [],
        "errors": [],
    }

    try:
        report["tools"] = {
            "cuobjdump": collect_tool_identity(
                args.cuobjdump, args.tool_timeout_seconds
            ),
            "nvdisasm": collect_tool_identity(
                args.nvdisasm, args.tool_timeout_seconds
            ),
        }
    except EvidenceError as exc:
        report["errors"].append("tool identity: {}".format(exc))
        return report

    for label, path in (
        ("baseline", args.baseline_cubin),
        ("candidate", args.candidate_cubin),
    ):
        arm = collect_arm(
            label,
            path,
            report["tools"],
            args.tool_timeout_seconds,
        )
        report["arms"][label] = arm
        if arm["status"] != "pass":
            report["errors"].append(
                "{}: {}".format(label, arm.get("error", "unknown failure"))
            )

    if all(
        report["arms"].get(label, {}).get("kernel_symbol")
        for label in ("baseline", "candidate")
    ):
        comparison = evaluate_comparison(
            report["arms"]["baseline"], report["arms"]["candidate"]
        )
        report["comparison"] = comparison
        if not comparison["pass"]:
            report["errors"].append(
                "comparison gates failed: {}".format(
                    ", ".join(comparison["failed"])
                )
            )
        if not comparison["both_arms_call_ret_zero"]:
            report["warnings"].append(
                "CALL/RET are not both zero; relative no-new-frame gate is authoritative"
            )

    report["status"] = "pass" if not report["errors"] else "fail"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cubin", type=Path, required=True)
    parser.add_argument("--candidate-cubin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cuobjdump",
        default="cuobjdump",
        help="cuobjdump executable name/path (default: PATH lookup)",
    )
    parser.add_argument(
        "--nvdisasm",
        default="nvdisasm",
        help="nvdisasm executable name/path (default: PATH lookup)",
    )
    parser.add_argument("--tool-timeout-seconds", type=int, default=120)
    args = parser.parse_args(argv)

    if args.tool_timeout_seconds <= 0:
        parser.error("--tool-timeout-seconds must be positive")

    output_abs = os.path.realpath(os.path.expanduser(str(args.output)))
    input_abs = {
        os.path.realpath(os.path.expanduser(str(args.baseline_cubin))),
        os.path.realpath(os.path.expanduser(str(args.candidate_cubin))),
    }
    if output_abs in input_abs:
        parser.error("--output must not overwrite an input cubin")

    report = build_report(args)
    try:
        write_json_atomic(args.output, report)
    except OSError as exc:
        print("FAIL: could not write {}: {}".format(args.output, exc), file=sys.stderr)
        return 2

    if report["status"] != "pass":
        print(
            "FAIL: static resource evidence rejected; JSON written to {}".format(
                args.output
            )
        )
        return 1
    print("PASS: static resource evidence written to {}".format(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
