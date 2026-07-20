#!/usr/bin/env python3
"""Build a cubin-locked, GPU-free SASS spill sidecar for one P3 capture.

``LDL``/``STL`` instructions come from ``cuobjdump --dump-sass``.  Compiler
``SpillRefill : Offset`` annotations live in ``cuobjdump --dump-elf``, so both
raw outputs are retained beside the capture.  These are static binary facts;
they do not establish dynamic execution frequency or latency.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from exp016_p3_probe_common import file_sha256, read_json, write_json


SCHEMA = "exp016.p3-sass-spill-evidence.v1"
SIDECAR_NAME = "sass_spill_evidence.json"
RAW_SASS_NAME = "p3_spill.sass.txt"
RAW_ELF_NAME = "p3_spill.elf.txt"

SPILL_ANNOTATION_RE = re.compile(r"SpillRefill\s*:\s*Offset\s*:\s*0x([0-9a-fA-F]+)")
SASS_INSTRUCTION_RE = re.compile(
    r"/\*\s*([0-9a-fA-F]+)\s*\*/\s+"
    r"(?:@[!A-Za-z0-9.]+\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;"
)


class SpillAuditError(RuntimeError):
    """The cubin or disassembly does not satisfy the spill evidence contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SpillAuditError(message)


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"missing {label}")
    return value


def capture_identity(capture: Mapping[str, Any], path: Path) -> dict[str, Any]:
    cubins = capture.get("cubin_sha256")
    require(
        isinstance(cubins, list) and len(cubins) == 1 and is_sha256(cubins[0]),
        "capture must identify exactly one cubin SHA256",
    )
    resources = mapping(capture.get("static_resource_usage"), "resource usage")
    records = resources.get("records")
    require(
        isinstance(records, list) and len(records) == 1,
        "capture must identify exactly one resource record",
    )
    record = mapping(records[0], "resource record")
    require(record.get("cubin_sha256") == cubins[0], "resource/capture cubin drift")
    symbol = record.get("kernel_symbol")
    require(
        isinstance(symbol, str) and "MoEDynamicKernel" in symbol,
        "capture dynamic-kernel symbol missing",
    )
    arm = capture.get("arm")
    mode = capture.get("mode")
    require(isinstance(arm, str) and arm, "capture arm missing")
    require(isinstance(mode, str) and mode, "capture mode missing")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "arm": arm,
        "mode": mode,
        "cubin_sha256": cubins[0],
        "kernel_symbol": symbol,
    }


def find_exact_cubin(jit_root: Path, expected_sha256: str) -> tuple[Path, int]:
    require(jit_root.is_dir(), f"fresh JIT root missing: {jit_root}")
    cubins = sorted(path for path in jit_root.rglob("*.cubin") if path.is_file())
    require(cubins, f"fresh JIT root contains no cubin: {jit_root}")
    matches = [path for path in cubins if file_sha256(path) == expected_sha256]
    require(
        len(matches) == 1,
        f"expected one cubin matching {expected_sha256}, found {len(matches)}",
    )
    return matches[0], len(cubins)


def resolve_tool(value: str | None) -> Path:
    candidate = value or shutil.which("cuobjdump")
    if candidate is None and Path("/usr/local/cuda/bin/cuobjdump").is_file():
        candidate = "/usr/local/cuda/bin/cuobjdump"
    require(candidate is not None, "cuobjdump not found")
    path = Path(candidate).resolve()
    require(path.is_file(), f"cuobjdump is not a file: {path}")
    return path


def run_tool(tool: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(tool), *arguments], capture_output=True, text=True, check=False
    )
    require(
        completed.returncode == 0,
        f"cuobjdump failed ({completed.returncode}): {completed.stderr.strip()}",
    )
    return completed


def parse_spill(sass_text: str, elf_text: str) -> dict[str, Any]:
    instructions = [
        {
            "pc": int(match.group(1), 16),
            "opcode": match.group(2),
        }
        for match in SASS_INSTRUCTION_RE.finditer(sass_text)
    ]
    require(instructions, "no SASS instructions parsed")
    local = [
        instruction
        for instruction in instructions
        if str(instruction["opcode"]).startswith(("LDL", "STL"))
    ]
    ldl = [item for item in local if str(item["opcode"]).startswith("LDL")]
    stl = [item for item in local if str(item["opcode"]).startswith("STL")]
    annotation_offsets = [
        int(value, 16) for value in SPILL_ANNOTATION_RE.findall(elf_text)
    ]
    annotation_pcs = set(annotation_offsets)
    local_pcs = {int(item["pc"]) for item in local}
    histogram = Counter(str(item["opcode"]) for item in local)
    return {
        "sass_instruction_count": len(instructions),
        "spill_refill_annotation_count": len(annotation_offsets),
        "spill_refill_annotation_unique_pc_count": len(annotation_pcs),
        "ldl_opcode_count": len(ldl),
        "stl_opcode_count": len(stl),
        "local_sass_opcode_count": len(local),
        "local_sass_opcode_histogram": dict(sorted(histogram.items())),
        "annotation_pcs_equal_local_sass_pcs": annotation_pcs == local_pcs,
    }


def audit(
    *,
    capture_path: Path,
    jit_root: Path,
    cuobjdump: str | None,
    output: Path | None = None,
) -> dict[str, Any]:
    capture_path = capture_path.resolve()
    require(capture_path.is_file(), f"capture missing: {capture_path}")
    capture = read_json(capture_path)
    identity = capture_identity(capture, capture_path)
    jit_root = jit_root.resolve()
    cubin, cubin_inventory_count = find_exact_cubin(
        jit_root, str(identity["cubin_sha256"])
    )
    require(file_sha256(cubin) == identity["cubin_sha256"], "cubin SHA drift")

    tool = resolve_tool(cuobjdump)
    version_run = run_tool(tool, "--version")
    sass_run = run_tool(tool, "--dump-sass", str(cubin))
    elf_run = run_tool(tool, "--dump-elf", str(cubin))
    require(sass_run.stdout.strip() != "", "cuobjdump produced empty SASS")
    require(elf_run.stdout.strip() != "", "cuobjdump produced empty ELF metadata")
    require(
        str(identity["kernel_symbol"]) in sass_run.stdout,
        "capture kernel symbol absent from SASS",
    )
    require(
        str(identity["kernel_symbol"]) in elf_run.stdout,
        "capture kernel symbol absent from ELF metadata",
    )
    counts = parse_spill(sass_run.stdout, elf_run.stdout)
    integrity_checks = {
        "capture_cubin_sha256_matches_unique_jit_cubin": True,
        "capture_kernel_symbol_in_sass": True,
        "capture_kernel_symbol_in_elf": True,
        "sass_instruction_parse_nonempty": counts["sass_instruction_count"] > 0,
        "annotation_pcs_equal_local_sass_pcs": counts[
            "annotation_pcs_equal_local_sass_pcs"
        ],
    }
    integrity_gate = all(integrity_checks.values())
    spill_gate = (
        counts["spill_refill_annotation_count"] == 0
        and counts["ldl_opcode_count"] == 0
        and counts["stl_opcode_count"] == 0
        and counts["local_sass_opcode_count"] == 0
    )

    output = (output or capture_path.parent / SIDECAR_NAME).resolve()
    require(output.parent == capture_path.parent, "sidecar must stay beside capture")
    sass_output = output.parent / RAW_SASS_NAME
    elf_output = output.parent / RAW_ELF_NAME
    sass_output.write_text(sass_run.stdout, encoding="utf-8")
    elf_output.write_text(elf_run.stdout, encoding="utf-8")
    payload = {
        "schema": SCHEMA,
        "arm": identity["arm"],
        "mode": identity["mode"],
        "capture": {
            "path": capture_path.name,
            "sha256": identity["sha256"],
        },
        "jit": {
            "root": str(jit_root),
            "cubin_inventory_count": cubin_inventory_count,
            "matched_cubin_relative_path": str(cubin.relative_to(jit_root)),
            "matched_cubin_sha256": identity["cubin_sha256"],
        },
        "kernel_symbol": identity["kernel_symbol"],
        "tool": {
            "path": str(tool),
            "sha256": file_sha256(tool),
            "version": version_run.stdout.strip() or version_run.stderr.strip(),
            "commands": [
                ["--dump-sass", str(cubin)],
                ["--dump-elf", str(cubin)],
            ],
        },
        "raw_sass": {
            "path": sass_output.name,
            "sha256": file_sha256(sass_output),
            "size": sass_output.stat().st_size,
        },
        "raw_elf": {
            "path": elf_output.name,
            "sha256": file_sha256(elf_output),
            "size": elf_output.stat().st_size,
        },
        "counts": counts,
        "integrity_checks": integrity_checks,
        "evidence_integrity_gate_pass": integrity_gate,
        "sass_spill_gate_pass": spill_gate,
        "gate_pass": integrity_gate and spill_gate,
        "evidence_boundary": (
            "Static cubin SASS/ELF facts only; zero instructions do not replace "
            "dynamic spill counters when those are required."
        ),
    }
    write_json(output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--cuobjdump")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = audit(
        capture_path=args.capture,
        jit_root=args.jit_root,
        cuobjdump=args.cuobjdump,
        output=args.output,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
