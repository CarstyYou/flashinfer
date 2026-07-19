#!/usr/bin/env python3
"""Build binary-locked static spill/resource evidence for exp_007.

The collector is GPU-free.  It validates every fresh-JIT cubin against its
``preparation.json`` and local immutable overlay, then runs ``cuobjdump`` and
``nvdisasm`` in a read-only CUDA container on the remote frontend.

Compiler ``SpillRefill`` annotations and local SASS instructions are static
binary facts.  They are deliberately not described as dynamic execution or
latency evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Sequence


EXPERIMENT = Path(__file__).resolve().parent
RESULTS = EXPERIMENT / "results"

RESOURCE_RE = re.compile(
    r"Function\s+(\S+):\s*\n\s*"
    r"REG\s*:\s*(\d+)\s+STACK\s*:\s*(\d+)\s+"
    r"SHARED\s*:\s*(\d+)\s+LOCAL\s*:\s*(\d+)",
    re.I,
)
FRAME_RE = re.compile(r"frame size:\s*0x([0-9a-fA-F]+)")
MIN_STACK_RE = re.compile(r"min stack size:\s*0x([0-9a-fA-F]+)")
ELF_FUNCTION_RE = re.compile(r"function:\s*(\S+?)\(0x[0-9a-fA-F]+\)")
SPILL_ANNOTATION_RE = re.compile(r"SpillRefill\s*:\s*Offset\s*:\s*0x([0-9a-fA-F]+)")
SASS_INSTRUCTION_RE = re.compile(
    r"/\*\s*([0-9a-fA-F]+)\s*\*/\s+"
    r"(?:@[!A-Za-z0-9.]+\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;"
)
SASS_GLOBAL_RE = re.compile(r"^\s*\.global\s+(\S+)\s*$", re.M)


@dataclass(frozen=True)
class CaseSpec:
    label: str
    logical_arm: str
    harness_arm: str
    m: int
    overlay: Path


CASES = (
    CaseSpec(
        "anchor_m256",
        "anchor",
        "candidate_8warp_serial_v0",
        256,
        RESULTS / "overlays/anchor_8warp_n128/moe_dynamic_kernel.py",
    ),
    CaseSpec(
        "anchor_m8192",
        "anchor",
        "candidate_8warp_serial_v0",
        8192,
        RESULTS / "overlays/anchor_8warp_n128/moe_dynamic_kernel.py",
    ),
    CaseSpec(
        "candidate_m256",
        "candidate",
        "candidate_8warp_n64_temporal_replay_v0",
        256,
        RESULTS / "overlays/candidate_8warp_native_n64_v0/moe_dynamic_kernel.py",
    ),
    CaseSpec(
        "candidate_m8192",
        "candidate",
        "candidate_8warp_n64_temporal_replay_v0",
        8192,
        RESULTS / "overlays/candidate_8warp_native_n64_v0/moe_dynamic_kernel.py",
    ),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return sha256_bytes(raw)


def run_checked(argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(argv, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {argv!r}\n"
            + completed.stderr.decode(errors="replace")
        )
    return completed


class RemoteReader:
    def __init__(self, *, host: str, image: str, jit_root: str):
        self.host = host
        self.image = image
        self.jit_root = jit_root.rstrip("/")

    def _ssh(self, argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        command = " ".join(shlex.quote(value) for value in argv)
        return run_checked(["ssh", "-o", "BatchMode=yes", self.host, command])

    def read_bytes(self, path: str) -> bytes:
        return self._ssh(["cat", "--", path]).stdout

    def sha256(self, path: str) -> str:
        return self._ssh(["sha256sum", "--", path]).stdout.decode().split()[0]

    def find_one_cubin(self, directory: str) -> str:
        output = self._ssh(
            [
                "find",
                directory,
                "-maxdepth",
                "1",
                "-type",
                "f",
                "-name",
                "*.cubin",
                "-print",
            ]
        ).stdout.decode()
        matches = [line for line in output.splitlines() if line]
        if len(matches) != 1:
            raise ValueError(f"expected one cubin in {directory}, got {matches}")
        return matches[0]

    def cuda_tool(self, tool: str, *args: str) -> subprocess.CompletedProcess[bytes]:
        binary = f"/usr/local/cuda/bin/{tool}"
        docker_argv = [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            binary,
            "-v",
            f"{self.jit_root}:{self.jit_root}:ro",
            self.image,
            *args,
        ]
        return self._ssh(docker_argv)


def preparation_identity(
    *, reader: RemoteReader, path: str, spec: CaseSpec, cubin_sha256: str
) -> dict[str, Any]:
    raw = reader.read_bytes(path)
    value = json.loads(raw)
    if value.get("arm") != spec.harness_arm:
        raise ValueError(
            f"{spec.label}: preparation arm {value.get('arm')!r} != {spec.harness_arm!r}"
        )
    if value.get("case", {}).get("m") != spec.m:
        raise ValueError(f"{spec.label}: preparation M mismatch")
    matches = [
        item
        for item in value.get("jit_artifacts", [])
        if item.get("sha256") == cubin_sha256
        and str(item.get("path", "")).endswith(".cubin")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{spec.label}: cubin hash is not uniquely locked by preparation.json"
        )

    source = value.get("runtime", {}).get("source", {})
    overlay_raw = spec.overlay.read_bytes()
    overlay_sha256 = sha256_bytes(overlay_raw)
    if source.get("overlay_sha256") != overlay_sha256:
        raise ValueError(f"{spec.label}: local overlay != fresh-JIT source lock")

    return {
        "path": path,
        "sha256": sha256_bytes(raw),
        "schema": value.get("schema"),
        "harness_arm": value.get("arm"),
        "case": value.get("case"),
        "locked_jit_artifact": matches[0],
        "source": source,
        "local_overlay": {
            "path": str(spec.overlay.resolve()),
            "sha256": overlay_sha256,
        },
        "compile_tools": {
            "nvcc": value.get("runtime", {}).get("nvcc"),
            "ptxas": value.get("runtime", {}).get("ptxas"),
        },
        "image_digest": value.get("runtime", {}).get("image_digest"),
        "gpu": value.get("runtime", {}).get("gpu"),
    }


def analyze_case(
    *,
    reader: RemoteReader,
    remote_worktree: str,
    remote_jit_root: str,
    spec: CaseSpec,
) -> dict[str, Any]:
    cubin_dir = f"{remote_jit_root}/{spec.logical_arm}/m{spec.m}/canonical/dump"
    cubin_path = reader.find_one_cubin(cubin_dir)
    cubin_sha256 = reader.sha256(cubin_path)
    preparation_path = (
        f"{remote_worktree}/.claude/w4a4_moe_bench/experiments/"
        "exp_007_native_n64_spill_reduction/results/canonical/raw/"
        f"{spec.harness_arm}/m{spec.m}/canonical/preparation.json"
    )
    preparation = preparation_identity(
        reader=reader,
        path=preparation_path,
        spec=spec,
        cubin_sha256=cubin_sha256,
    )

    resource_run = reader.cuda_tool("cuobjdump", "--dump-resource-usage", cubin_path)
    elf_run = reader.cuda_tool("cuobjdump", "--dump-elf", cubin_path)
    sass_run = reader.cuda_tool("nvdisasm", "-c", cubin_path)
    resource_text = resource_run.stdout.decode(errors="replace")
    elf_text = elf_run.stdout.decode(errors="replace")
    sass_text = sass_run.stdout.decode(errors="replace")

    resource_matches = RESOURCE_RE.findall(resource_text)
    if len(resource_matches) != 1:
        raise ValueError(f"{spec.label}: expected one kernel resource record")
    kernel_symbol, registers, stack, shared, local = resource_matches[0]
    registers, stack, shared, local = map(int, (registers, stack, shared, local))

    frame_values = [int(value, 16) for value in FRAME_RE.findall(elf_text)]
    min_stack_values = [int(value, 16) for value in MIN_STACK_RE.findall(elf_text)]
    if frame_values != [stack] or min_stack_values != [stack]:
        raise ValueError(
            f"{spec.label}: resource STACK and ELF frame/min-stack disagree: "
            f"{stack}, {frame_values}, {min_stack_values}"
        )

    elf_symbols = set(ELF_FUNCTION_RE.findall(elf_text))
    sass_symbols = set(SASS_GLOBAL_RE.findall(sass_text))
    if kernel_symbol not in elf_symbols or kernel_symbol not in sass_symbols:
        raise ValueError(f"{spec.label}: kernel symbol does not agree across tools")

    instructions = [
        {
            "pc": int(match.group(1), 16),
            "opcode": match.group(2),
            "operands": match.group(3).strip(),
        }
        for match in SASS_INSTRUCTION_RE.finditer(sass_text)
    ]
    if not instructions:
        raise ValueError(f"{spec.label}: no SASS instructions parsed")
    pcs = [item["pc"] for item in instructions]
    if len(pcs) != len(set(pcs)):
        raise ValueError(f"{spec.label}: duplicate SASS PCs")

    local_instructions = [
        item for item in instructions if str(item["opcode"]).startswith(("LDL", "STL"))
    ]
    local_pcs = {int(item["pc"]) for item in local_instructions}
    annotation_pcs = {int(value, 16) for value in SPILL_ANNOTATION_RE.findall(elf_text)}
    annotation_closure = annotation_pcs == local_pcs
    if not annotation_closure:
        raise ValueError(
            f"{spec.label}: compiler SpillRefill annotations != local SASS PCs"
        )

    local_histogram = Counter(str(item["opcode"]) for item in local_instructions)
    omma_histogram = Counter(
        str(item["opcode"]) for item in instructions if "OMMA" in str(item["opcode"])
    )
    zero_spill = stack == 0 and not annotation_pcs and not local_instructions

    return {
        "label": spec.label,
        "logical_arm": spec.logical_arm,
        "m": spec.m,
        "fixture": "canonical",
        "identity": {
            "cubin_path": cubin_path,
            "cubin_sha256": cubin_sha256,
            "kernel_symbol": kernel_symbol,
            "kernel_symbol_matches_resource_elf_sass": True,
            "preparation": preparation,
        },
        "resource": {
            "registers_per_thread": registers,
            "stack_bytes_per_thread": stack,
            "shared_bytes_per_cta": shared,
            "static_local_bytes_outside_stack": local,
            "elf_frame_bytes_per_thread": frame_values[0],
            "elf_minimum_stack_bytes_per_thread": min_stack_values[0],
        },
        "compiler_spill_refill": {
            "annotation_count": len(annotation_pcs),
            "annotation_pcs_hex": [hex(value) for value in sorted(annotation_pcs)],
            "annotation_exactly_matches_local_sass": annotation_closure,
            "local_sass_instruction_count": len(local_instructions),
            "local_sass_opcode_histogram": dict(sorted(local_histogram.items())),
            "static_fact_only": True,
            "dynamic_execution_claimed": False,
        },
        "tensor_core_work": {
            "omma_static_instruction_count": sum(omma_histogram.values()),
            "omma_static_opcode_histogram": dict(sorted(omma_histogram.items())),
        },
        "gates": {
            "fresh_jit_cubin_locked_by_preparation": True,
            "source_locked_by_preparation": True,
            "kernel_symbol_locked_across_resource_elf_sass": True,
            "compiler_annotation_closure": True,
            "zero_spill_static_gate": zero_spill,
        },
        "tool_output_sha256": {
            "resource_stdout": sha256_bytes(resource_run.stdout),
            "elf_stdout": sha256_bytes(elf_run.stdout),
            "sass_stdout": sha256_bytes(sass_run.stdout),
        },
    }


def build_cross_case_checks(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for arm in ("anchor", "candidate"):
        arm_cases = [value for value in cases.values() if value["logical_arm"] == arm]
        hashes = {value["identity"]["cubin_sha256"] for value in arm_cases}
        symbols = {value["identity"]["kernel_symbol"] for value in arm_cases}
        checks[f"{arm}_m256_m8192_same_cubin_sha256"] = len(hashes) == 1
        checks[f"{arm}_m256_m8192_same_kernel_symbol"] = len(symbols) == 1
    anchor_omma = cases["anchor_m256"]["tensor_core_work"]
    candidate_omma = cases["candidate_m256"]["tensor_core_work"]
    checks["anchor_candidate_same_static_omma_count"] = (
        anchor_omma["omma_static_instruction_count"]
        == candidate_omma["omma_static_instruction_count"]
    )
    checks["anchor_candidate_same_static_omma_histogram"] = (
        anchor_omma["omma_static_opcode_histogram"]
        == candidate_omma["omma_static_opcode_histogram"]
    )
    return checks


def write_csv(path: Path, cases: dict[str, dict[str, Any]]) -> None:
    fields = (
        "label",
        "logical_arm",
        "m",
        "fixture",
        "source_sha256",
        "cubin_sha256",
        "kernel_symbol",
        "registers_per_thread",
        "stack_bytes_per_thread",
        "shared_bytes_per_cta",
        "static_local_bytes_outside_stack",
        "elf_frame_bytes_per_thread",
        "elf_minimum_stack_bytes_per_thread",
        "compiler_spill_refill_annotation_count",
        "local_sass_instruction_count",
        "local_sass_opcode_histogram",
        "omma_static_instruction_count",
        "omma_static_opcode_histogram",
        "zero_spill_static_gate",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label in (spec.label for spec in CASES):
            case = cases[label]
            identity = case["identity"]
            resource = case["resource"]
            spill = case["compiler_spill_refill"]
            omma = case["tensor_core_work"]
            writer.writerow(
                {
                    "label": label,
                    "logical_arm": case["logical_arm"],
                    "m": case["m"],
                    "fixture": case["fixture"],
                    "source_sha256": identity["preparation"]["local_overlay"]["sha256"],
                    "cubin_sha256": identity["cubin_sha256"],
                    "kernel_symbol": identity["kernel_symbol"],
                    **resource,
                    "compiler_spill_refill_annotation_count": spill["annotation_count"],
                    "local_sass_instruction_count": spill[
                        "local_sass_instruction_count"
                    ],
                    "local_sass_opcode_histogram": json.dumps(
                        spill["local_sass_opcode_histogram"], sort_keys=True
                    ),
                    "omma_static_instruction_count": omma[
                        "omma_static_instruction_count"
                    ],
                    "omma_static_opcode_histogram": json.dumps(
                        omma["omma_static_opcode_histogram"], sort_keys=True
                    ),
                    "zero_spill_static_gate": case["gates"]["zero_spill_static_gate"],
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="xiy@10.6.142.16")
    parser.add_argument(
        "--remote-worktree", default="/home/xiy/workspace/flashinfer_exp007_748ad"
    )
    parser.add_argument(
        "--remote-jit-root", default="/home/xiy/workspace/exp007_native_n64_jit"
    )
    parser.add_argument("--image", default="nvcr.io/nvidia/pytorch:26.05-py3")
    parser.add_argument(
        "--output", type=Path, default=RESULTS / "static_spill_evidence.json"
    )
    parser.add_argument(
        "--csv", type=Path, default=RESULTS / "static_spill_summary.csv"
    )
    args = parser.parse_args(argv)

    reader = RemoteReader(
        host=args.remote, image=args.image, jit_root=args.remote_jit_root
    )
    cuobjdump_version = reader.cuda_tool("cuobjdump", "--version")
    nvdisasm_version = reader.cuda_tool("nvdisasm", "--version")
    cases = {
        spec.label: analyze_case(
            reader=reader,
            remote_worktree=args.remote_worktree.rstrip("/"),
            remote_jit_root=args.remote_jit_root.rstrip("/"),
            spec=spec,
        )
        for spec in CASES
    }
    payload = {
        "schema": "exp007.static-resource-spill-evidence.v1",
        "scope": {
            "cases": [spec.label for spec in CASES],
            "collector_uses_gpu": False,
            "evidence_boundary": (
                "Static resource records, SASS instructions, and compiler SpillRefill "
                "annotations prove binary presence only; they do not prove dynamic "
                "execution frequency or latency."
            ),
        },
        "collector": {
            "remote": args.remote,
            "image": args.image,
            "commands_are_binary_read_only": True,
            "cuobjdump_version": (cuobjdump_version.stdout + cuobjdump_version.stderr)
            .decode(errors="replace")
            .strip(),
            "nvdisasm_version": (nvdisasm_version.stdout + nvdisasm_version.stderr)
            .decode(errors="replace")
            .strip(),
        },
        "cases": cases,
        "cross_case_checks": build_cross_case_checks(cases),
    }
    payload["evidence_sha256"] = canonical_sha256(payload)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    write_csv(args.csv.resolve(), cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
