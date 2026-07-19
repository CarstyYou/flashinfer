#!/usr/bin/env python3
"""Build binary-locked static resource/spill evidence for exp_008.

Run this collector inside a CUDA 13.2 container with the experiment worktree
and the fresh exp008 JIT root mounted read-only/read-write respectively.  It
does not open a CUDA context: cuobjdump/nvdisasm only read existing artifacts.
Static SASS presence is not dynamic execution or latency evidence.
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
import subprocess
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

RESOURCE_RE = re.compile(
    r"Function\s+(\S+):\s*\n\s*"
    r"REG\s*:\s*(\d+)\s+STACK\s*:\s*(\d+)\s+"
    r"SHARED\s*:\s*(\d+)\s+LOCAL\s*:\s*(\d+)",
    re.I,
)
FRAME_RE = re.compile(r"frame size:\s*0x([0-9a-fA-F]+)")
MIN_STACK_RE = re.compile(r"min stack size:\s*0x([0-9a-fA-F]+)")
ELF_FUNCTION_RE = re.compile(r"function:\s*(\S+?)\(0x[0-9a-fA-F]+\)")
SPILL_ANNOTATION_RE = re.compile(
    r"SpillRefill\s*:\s*Offset\s*:\s*0x([0-9a-fA-F]+)"
)
SASS_INSTRUCTION_RE = re.compile(
    r"/\*\s*([0-9a-fA-F]+)\s*\*/\s+"
    r"(?:@[!A-Za-z0-9.]+\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;"
)
SASS_GLOBAL_RE = re.compile(r"^\s*\.global\s+(\S+)\s*$", re.M)
SMEM_STRUCT_RE = re.compile(r"smem\.struct_fields\s*=\s*\[(.*?)\]", re.S)
SMEM_FIELD_RE = re.compile(r'"([^":]+):(\d+):(\d+)"')
KERNEL_SMEM_QUERY_RE = re.compile(
    r"cute\.kernel_smem_size\s+@kernels::@([^\s:]+)\s*:\s*i64"
)


@dataclass(frozen=True)
class CaseSpec:
    label: str
    arm: str
    harness_arm: str
    m: int
    overlay: Path


EXPECTED_SOURCE = {
    "n128": "3cd9e6a26056d9221f59ea6749cd601c25cbef017cf6e7349efe0925180407c1",
    "v0": "1953cbb7717cda4461a4f199d05f370a4bdb35b4b8ef7556443caf36b0b12ec2",
    "v1": "f3c246817679d962a3f7160dbe8b9e68262c919e26e306f349200961fc4ac971",
}
EXPECTED_CUBIN = {
    "n128": "b2bc3c4c229ebee967a6b0d3c5649bc06e3629d46793a19af845665f93683f17",
    "v0": "a9557634cf3d1bff59ca93739e75a1acd1187707222255fead78e2e6e8a73af9",
    "v1": "4b835aa8ce91a4dd12b4dc4f43508c205c117aaeb193995fff57dd3ddbeb7725",
}
HARNESS_ARM = {
    "n128": "candidate_8warp_serial_v0",
    "v0": "candidate_8warp_n64_temporal_replay_v0",
    "v1": "candidate_8warp_n64_temporal_replay_v0",
}
OVERLAY = {
    "n128": RESULTS / "overlays/anchor_8warp_n128/moe_dynamic_kernel.py",
    "v0": RESULTS / "overlays/temporal_n64_v0/moe_dynamic_kernel.py",
    "v1": RESULTS / "overlays/branch_paired_n64_v1/moe_dynamic_kernel.py",
}
CASES = tuple(
    CaseSpec(
        label=f"{arm}_m{m}",
        arm=arm,
        harness_arm=HARNESS_ARM[arm],
        m=m,
        overlay=OVERLAY[arm],
    )
    for arm in ("n128", "v0", "v1")
    for m in (256, 8192)
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    )


def evidence_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def run_checked(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {argv!r}\n{completed.stderr}"
        )
    return completed


def unique_artifact(
    preparation: dict[str, Any], *, suffix: str, exclude: str | None = None
) -> dict[str, Any]:
    matches = [
        item
        for item in preparation.get("jit_artifacts", [])
        if str(item.get("path", "")).endswith(suffix)
        and (exclude is None or exclude not in str(item.get("path", "")))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {suffix} artifact excluding {exclude!r}, got {matches}"
        )
    return matches[0]


def parse_dynamic_smem(mlir_text: str, kernel_symbol: str) -> dict[str, Any]:
    struct_matches = SMEM_STRUCT_RE.findall(mlir_text)
    if len(struct_matches) != 1:
        raise ValueError(f"expected one SMEM struct, got {len(struct_matches)}")
    fields = [
        {"name": name, "size_bytes": int(size), "offset_bytes": int(offset)}
        for name, size, offset in SMEM_FIELD_RE.findall(struct_matches[0])
    ]
    if not fields:
        raise ValueError("SMEM struct has no parsed fields")
    fields_by_offset = sorted(fields, key=lambda value: value["offset_bytes"])
    no_overlap = all(
        current["offset_bytes"]
        >= previous["offset_bytes"] + previous["size_bytes"]
        for previous, current in zip(
            fields_by_offset, fields_by_offset[1:], strict=False
        )
    )
    extent = max(
        field["offset_bytes"] + field["size_bytes"] for field in fields_by_offset
    )
    queried_symbols = KERNEL_SMEM_QUERY_RE.findall(mlir_text)
    return {
        "fields": fields_by_offset,
        "field_count": len(fields_by_offset),
        "struct_extent_bytes": extent,
        "first_field_starts_at_zero": fields_by_offset[0]["offset_bytes"] == 0,
        "fields_do_not_overlap": no_overlap,
        "extent_is_1024_byte_aligned": extent % 1024 == 0,
        "kernel_smem_size_query_symbols": queried_symbols,
        "kernel_smem_size_query_matches_kernel": queried_symbols == [kernel_symbol],
        "launch_uses_dynamic_smem_query": "dynamicSmemBytes = %" in mlir_text,
    }


def parse_binary(
    *, cubin_path: Path, cuobjdump: str, nvdisasm: str
) -> dict[str, Any]:
    resource_run = run_checked([cuobjdump, "--dump-resource-usage", str(cubin_path)])
    elf_run = run_checked([cuobjdump, "--dump-elf", str(cubin_path)])
    sass_run = run_checked([nvdisasm, "-c", str(cubin_path)])
    resource_text = resource_run.stdout
    elf_text = elf_run.stdout
    sass_text = sass_run.stdout

    resource_matches = RESOURCE_RE.findall(resource_text)
    if len(resource_matches) != 1:
        raise ValueError(f"expected one kernel resource record in {cubin_path}")
    kernel_symbol, registers, stack, static_shared, local = resource_matches[0]
    registers, stack, static_shared, local = map(
        int, (registers, stack, static_shared, local)
    )

    frame_values = [int(value, 16) for value in FRAME_RE.findall(elf_text)]
    min_stack_values = [int(value, 16) for value in MIN_STACK_RE.findall(elf_text)]
    if frame_values != [stack] or min_stack_values != [stack]:
        raise ValueError(
            "resource STACK and ELF frame/min-stack disagree: "
            f"{stack}, {frame_values}, {min_stack_values}"
        )
    elf_symbols = set(ELF_FUNCTION_RE.findall(elf_text))
    sass_symbols = set(SASS_GLOBAL_RE.findall(sass_text))
    if kernel_symbol not in elf_symbols or kernel_symbol not in sass_symbols:
        raise ValueError("kernel symbol disagrees across resource/ELF/SASS")

    instructions = [
        {
            "pc": int(match.group(1), 16),
            "opcode": match.group(2),
            "operands": match.group(3).strip(),
        }
        for match in SASS_INSTRUCTION_RE.finditer(sass_text)
    ]
    if not instructions:
        raise ValueError(f"no SASS instructions parsed from {cubin_path}")
    pcs = [item["pc"] for item in instructions]
    if len(pcs) != len(set(pcs)):
        raise ValueError(f"duplicate SASS PCs in {cubin_path}")

    local_instructions = [
        item
        for item in instructions
        if str(item["opcode"]).startswith(("LDL", "STL"))
    ]
    local_pcs = {int(item["pc"]) for item in local_instructions}
    annotation_pcs = {
        int(value, 16) for value in SPILL_ANNOTATION_RE.findall(elf_text)
    }
    if annotation_pcs != local_pcs:
        raise ValueError("compiler SpillRefill annotations != local SASS PCs")

    local_histogram = Counter(str(item["opcode"]) for item in local_instructions)
    omma_histogram = Counter(
        str(item["opcode"]) for item in instructions if "OMMA" in str(item["opcode"])
    )
    return {
        "kernel_symbol": kernel_symbol,
        "resource": {
            "registers_per_thread": registers,
            "stack_bytes_per_thread": stack,
            "static_shared_bytes_per_cta": static_shared,
            "static_local_bytes_outside_stack": local,
            "elf_frame_bytes_per_thread": frame_values[0],
            "elf_minimum_stack_bytes_per_thread": min_stack_values[0],
        },
        "compiler_spill_refill": {
            "annotation_count": len(annotation_pcs),
            "annotation_pcs_hex": [hex(value) for value in sorted(annotation_pcs)],
            "local_sass_instruction_count": len(local_instructions),
            "local_sass_opcode_histogram": dict(sorted(local_histogram.items())),
            "annotation_exactly_matches_local_sass": True,
            "static_fact_only": True,
            "dynamic_execution_claimed": False,
        },
        "tensor_core_work": {
            "omma_static_instruction_count": sum(omma_histogram.values()),
            "omma_static_opcode_histogram": dict(sorted(omma_histogram.items())),
            "static_fact_only": True,
        },
        "instruction_count": len(instructions),
        "symbol_checks": {
            "resource_symbol_in_elf": True,
            "resource_symbol_in_sass": True,
        },
        "tool_output_sha256": {
            "resource_stdout": sha256_bytes(resource_run.stdout.encode()),
            "elf_stdout": sha256_bytes(elf_run.stdout.encode()),
            "sass_stdout": sha256_bytes(sass_run.stdout.encode()),
        },
    }


def analyze_case(
    *,
    spec: CaseSpec,
    jit_root: Path,
    cuobjdump: str,
    nvdisasm: str,
    binary_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    preparation_path = (
        RESULTS
        / "canonical"
        / spec.arm
        / "raw"
        / spec.harness_arm
        / f"m{spec.m}"
        / "canonical/preparation.json"
    )
    preparation = json.loads(preparation_path.read_text())
    cubin_artifact = unique_artifact(preparation, suffix=".cubin")
    mlir_artifact = unique_artifact(
        preparation, suffix="_clean.mlir", exclude="hardware_info"
    )
    jit_case_root = jit_root / spec.arm / f"m{spec.m}" / "canonical"
    cubin_path = jit_case_root / str(cubin_artifact["path"])
    mlir_path = jit_case_root / str(mlir_artifact["path"])
    cubin_sha256 = sha256_file(cubin_path)
    mlir_sha256 = sha256_file(mlir_path)
    overlay_sha256 = sha256_file(spec.overlay)

    identity_checks = {
        "preparation_status_complete": preparation.get("status") == "complete",
        "harness_arm": preparation.get("arm") == spec.harness_arm,
        "m": preparation.get("m") == spec.m
        and preparation.get("case", {}).get("m") == spec.m,
        "canonical_fixture": preparation.get("fixture_kind") == "canonical",
        "source_sha256": preparation.get("runtime", {})
        .get("source", {})
        .get("overlay_sha256")
        == EXPECTED_SOURCE[spec.arm]
        == overlay_sha256,
        "cubin_sha256": preparation.get("cubin_sha256")
        == [EXPECTED_CUBIN[spec.arm]]
        and cubin_artifact.get("sha256") == EXPECTED_CUBIN[spec.arm]
        and cubin_sha256 == EXPECTED_CUBIN[spec.arm],
        "cubin_size": cubin_path.stat().st_size == cubin_artifact.get("size"),
        "mlir_sha256": mlir_sha256 == mlir_artifact.get("sha256"),
        "mlir_size": mlir_path.stat().st_size == mlir_artifact.get("size"),
    }

    binary_reused = cubin_sha256 in binary_cache
    if not binary_reused:
        binary_cache[cubin_sha256] = parse_binary(
            cubin_path=cubin_path, cuobjdump=cuobjdump, nvdisasm=nvdisasm
        )
    binary = binary_cache[cubin_sha256]
    mlir_text = mlir_path.read_text()
    dynamic_smem = parse_dynamic_smem(mlir_text, binary["kernel_symbol"])
    resource = dict(binary["resource"])
    total_shared = (
        resource["static_shared_bytes_per_cta"]
        + dynamic_smem["struct_extent_bytes"]
    )
    resource.update(
        {
            "dynamic_shared_struct_extent_bytes": dynamic_smem[
                "struct_extent_bytes"
            ],
            "total_shared_bytes_per_cta": total_shared,
            "total_shared_formula": "cuobjdump SHARED + MLIR dynamic struct extent",
        }
    )
    spill = binary["compiler_spill_refill"]
    zero_spill = (
        resource["stack_bytes_per_thread"] == 0
        and resource["static_local_bytes_outside_stack"] == 0
        and spill["annotation_count"] == 0
        and spill["local_sass_instruction_count"] == 0
    )
    gates = {
        "identity_gate": all(identity_checks.values()),
        "kernel_symbol_locked_across_resource_elf_sass": all(
            binary["symbol_checks"].values()
        ),
        "compiler_annotation_closure": spill[
            "annotation_exactly_matches_local_sass"
        ],
        "dynamic_smem_layout_valid": dynamic_smem["first_field_starts_at_zero"]
        and dynamic_smem["fields_do_not_overlap"]
        and dynamic_smem["extent_is_1024_byte_aligned"]
        and dynamic_smem["kernel_smem_size_query_matches_kernel"]
        and dynamic_smem["launch_uses_dynamic_smem_query"],
        "shared_memory_within_sm120_limit": total_shared <= 101376,
        "zero_spill_static": zero_spill,
    }
    evidence_gate = all(
        value for name, value in gates.items() if name != "zero_spill_static"
    )
    return {
        "label": spec.label,
        "arm": spec.arm,
        "m": spec.m,
        "fixture": "canonical",
        "identity": {
            "preparation_path": evidence_path(preparation_path),
            "preparation_sha256": sha256_file(preparation_path),
            "overlay_path": evidence_path(spec.overlay),
            "source_sha256": overlay_sha256,
            "cubin_path": str(cubin_path.relative_to(jit_root)),
            "cubin_sha256": cubin_sha256,
            "cubin_size": cubin_path.stat().st_size,
            "mlir_path": str(mlir_path.relative_to(jit_root)),
            "mlir_sha256": mlir_sha256,
            "kernel_symbol": binary["kernel_symbol"],
            "identity_checks": identity_checks,
        },
        "resource": resource,
        "dynamic_smem": dynamic_smem,
        "compiler_spill_refill": spill,
        "tensor_core_work": binary["tensor_core_work"],
        "sass_instruction_count": binary["instruction_count"],
        "binary_analysis_reused_by_cubin_sha256": binary_reused,
        "tool_output_sha256": binary["tool_output_sha256"],
        "gates": gates,
        "evidence_gate": evidence_gate,
    }


def cross_case_checks(cases: dict[str, dict[str, Any]]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for arm in ("n128", "v0", "v1"):
        pair = [cases[f"{arm}_m256"], cases[f"{arm}_m8192"]]
        checks[f"{arm}_m256_m8192_same_cubin_sha256"] = (
            len({value["identity"]["cubin_sha256"] for value in pair}) == 1
        )
        checks[f"{arm}_m256_m8192_same_kernel_symbol"] = (
            len({value["identity"]["kernel_symbol"] for value in pair}) == 1
        )
        checks[f"{arm}_m256_m8192_same_mlir_sha256"] = (
            len({value["identity"]["mlir_sha256"] for value in pair}) == 1
        )
        checks[f"{arm}_m256_m8192_same_resource"] = (
            pair[0]["resource"] == pair[1]["resource"]
        )
        checks[f"{arm}_m256_m8192_same_spill_sass"] = (
            pair[0]["compiler_spill_refill"]
            == pair[1]["compiler_spill_refill"]
        )
        checks[f"{arm}_m256_m8192_same_omma_histogram"] = (
            pair[0]["tensor_core_work"] == pair[1]["tensor_core_work"]
        )
    representatives = [cases[f"{arm}_m256"] for arm in ("n128", "v0", "v1")]
    checks["all_arms_static_omma_count_is_448"] = all(
        value["tensor_core_work"]["omma_static_instruction_count"] == 448
        for value in representatives
    )
    checks["all_arms_same_static_omma_histogram"] = (
        len(
            {
                json.dumps(
                    value["tensor_core_work"]["omma_static_opcode_histogram"],
                    sort_keys=True,
                )
                for value in representatives
            }
        )
        == 1
    )
    return checks


def write_csv(path: Path, cases: dict[str, dict[str, Any]]) -> None:
    fields = (
        "label",
        "arm",
        "m",
        "source_sha256",
        "cubin_sha256",
        "kernel_symbol",
        "registers_per_thread",
        "stack_bytes_per_thread",
        "static_shared_bytes_per_cta",
        "dynamic_shared_struct_extent_bytes",
        "total_shared_bytes_per_cta",
        "static_local_bytes_outside_stack",
        "elf_frame_bytes_per_thread",
        "elf_minimum_stack_bytes_per_thread",
        "compiler_spill_refill_annotation_count",
        "local_sass_instruction_count",
        "local_sass_opcode_histogram",
        "omma_static_instruction_count",
        "omma_static_opcode_histogram",
        "zero_spill_static",
        "evidence_gate",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for spec in CASES:
            case = cases[spec.label]
            identity = case["identity"]
            resource = case["resource"]
            spill = case["compiler_spill_refill"]
            omma = case["tensor_core_work"]
            writer.writerow(
                {
                    "label": spec.label,
                    "arm": spec.arm,
                    "m": spec.m,
                    "source_sha256": identity["source_sha256"],
                    "cubin_sha256": identity["cubin_sha256"],
                    "kernel_symbol": identity["kernel_symbol"],
                    **{
                        key: resource[key]
                        for key in (
                            "registers_per_thread",
                            "stack_bytes_per_thread",
                            "static_shared_bytes_per_cta",
                            "dynamic_shared_struct_extent_bytes",
                            "total_shared_bytes_per_cta",
                            "static_local_bytes_outside_stack",
                            "elf_frame_bytes_per_thread",
                            "elf_minimum_stack_bytes_per_thread",
                        )
                    },
                    "compiler_spill_refill_annotation_count": spill[
                        "annotation_count"
                    ],
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
                    "zero_spill_static": case["gates"]["zero_spill_static"],
                    "evidence_gate": case["evidence_gate"],
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--cuobjdump", default="/usr/local/cuda/bin/cuobjdump")
    parser.add_argument("--nvdisasm", default="/usr/local/cuda/bin/nvdisasm")
    parser.add_argument(
        "--output", type=Path, default=RESULTS / "static_spill_evidence.json"
    )
    parser.add_argument(
        "--csv", type=Path, default=RESULTS / "static_spill_summary.csv"
    )
    args = parser.parse_args(argv)

    cuobjdump_version = run_checked([args.cuobjdump, "--version"])
    nvdisasm_version = run_checked([args.nvdisasm, "--version"])
    binary_cache: dict[str, dict[str, Any]] = {}
    cases = {
        spec.label: analyze_case(
            spec=spec,
            jit_root=args.jit_root.resolve(),
            cuobjdump=args.cuobjdump,
            nvdisasm=args.nvdisasm,
            binary_cache=binary_cache,
        )
        for spec in CASES
    }
    cross_checks = cross_case_checks(cases)
    case_gate = all(value["evidence_gate"] for value in cases.values())
    cross_gate = all(cross_checks.values())
    gate = case_gate and cross_gate
    payload = {
        "schema": "exp008.static-resource-spill-evidence.v1",
        "scope": {
            "arms": ["n128", "v0", "v1"],
            "m_values": [256, 8192],
            "case_count": len(cases),
            "distinct_cubin_count": len(binary_cache),
            "collector_uses_gpu": False,
            "evidence_boundary": (
                "Resource records, SASS instructions, and compiler SpillRefill "
                "annotations are static binary facts only; they do not prove "
                "dynamic execution frequency or latency."
            ),
        },
        "collector": {
            "commands_are_binary_read_only": True,
            "jit_root": str(args.jit_root.resolve()),
            "cuobjdump_version": (
                cuobjdump_version.stdout + cuobjdump_version.stderr
            ).strip(),
            "nvdisasm_version": (
                nvdisasm_version.stdout + nvdisasm_version.stderr
            ).strip(),
        },
        "expected_source_sha256": EXPECTED_SOURCE,
        "expected_cubin_sha256": EXPECTED_CUBIN,
        "cases": cases,
        "cross_case_checks": cross_checks,
        "case_evidence_gate": case_gate,
        "cross_case_gate": cross_gate,
        "gate_pass": gate,
    }
    payload["evidence_sha256"] = canonical_sha256(payload)
    args.output.resolve().write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(args.csv.resolve(), cases)
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "distinct_cubin_count": len(binary_cache),
                "gate_pass": gate,
                "output": evidence_path(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
