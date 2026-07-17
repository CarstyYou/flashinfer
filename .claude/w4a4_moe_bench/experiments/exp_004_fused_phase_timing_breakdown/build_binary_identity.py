#!/usr/bin/env python3
"""Build the exp_004 three-arm static binary/resource identity gate.

This collector is GPU-free after JIT preparation.  It reads exact retained
cubins/SASS, invokes only cuobjdump/nvdisasm, and combines those facts with the
separately captured NCU dynamic-work gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import difflib
import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from exp004_common import (
    ALL_ARMS,
    DEFAULT_RESULTS,
    EXPECTED_KERNEL_SHA256,
    MEASUREMENT_CONTROL,
    NORMAL,
    PROBE,
    canonical_sha256,
    file_sha256,
    read_json,
    write_json,
)
from run_exp004 import preparation_path


EXPECTED_BASELINE_CUBIN_SHA256 = (
    "9313fcbc0dd686f0684705e869fdd227608ac83ca43c1dc99d203f8e7143ca79"
)
EXPECTED_BASELINE_SASS_SHA256 = (
    "34b4c38161642a27ca6b4ec41ffad0bd70f6ff99fd8118997a4b2416c5e3abba"
)
RESOURCE_RE = re.compile(
    r"REG\s*:\s*(\d+)\s+STACK\s*:\s*(\d+)\s+"
    r"SHARED\s*:\s*(\d+)\s+LOCAL\s*:\s*(\d+)",
    re.I,
)
FRAME_RE = re.compile(r"frame size:\s*0x([0-9a-fA-F]+)")
MIN_STACK_RE = re.compile(r"min stack size:\s*0x([0-9a-fA-F]+)")
SPILL_ANNOTATION_RE = re.compile(r"SpillRefill\s*:\s*Offset\s*:\s*0x([0-9a-fA-F]+)")
STACK_OFFSET_RE = re.compile(r"\[R1(?:\+0x([0-9a-fA-F]+))?\]")
SASS_RE = re.compile(
    r"/\*\s*([0-9a-fA-F]+)\s*\*/\s+"
    r"(?:(@[!A-Za-z0-9.]+)\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;"
)
BRANCH_TARGET_RE = re.compile(r"(?:^|[^A-Za-z0-9_])0x([0-9a-fA-F]+)(?:$|[^A-Za-z0-9_])")


def _run(command: Sequence[str]) -> bytes:
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {list(command)!r}\n"
            + completed.stderr.decode(errors="replace")
        )
    return completed.stdout


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _assignments(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        arm, separator, raw = value.partition("=")
        if not separator or arm not in ALL_ARMS or not raw or arm in result:
            raise ValueError(f"expected unique ARM=PATH, got {value!r}")
        result[arm] = Path(raw).resolve()
    return result


def _locked_artifacts(preparation: Mapping[str, Any], suffix: str) -> list[Path]:
    root = Path(preparation["runtime"]["jit_root"])
    paths = []
    for artifact in preparation["jit_artifacts"]:
        if str(artifact["path"]).endswith(suffix):
            path = root / artifact["path"]
            if not path.is_file() or file_sha256(path) != artifact["sha256"]:
                raise ValueError(f"fresh-JIT artifact drift: {path}")
            paths.append(path)
    return paths


def _select_artifact(
    preparation: Mapping[str, Any],
    *,
    suffix: str,
    explicit: Path | None,
    label: str,
) -> Path:
    candidates = _locked_artifacts(preparation, suffix)
    if explicit is not None:
        explicit = explicit.resolve()
        if explicit not in {path.resolve() for path in candidates}:
            raise ValueError(f"{label}: explicit artifact is absent from JIT lock")
        return explicit
    if len(candidates) != 1:
        raise ValueError(
            f"{label}: expected one {suffix} artifact, got {len(candidates)}; "
            "pass an explicit ARM=PATH"
        )
    return candidates[0]


def _select_target_cubin(
    preparation: Mapping[str, Any],
    *,
    explicit: Path | None,
    nvdisasm: str,
    label: str,
) -> Path:
    if explicit is not None:
        return _select_artifact(
            preparation,
            suffix=".cubin",
            explicit=explicit,
            label=label,
        )
    candidates = _locked_artifacts(preparation, ".cubin")
    matches = [
        path
        for path in candidates
        if b"MoEDynamicKernel" in _run([nvdisasm, "-c", str(path)])
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{label}: expected one cubin containing MoEDynamicKernel, got "
            f"{len(matches)} from {len(candidates)} retained cubins"
        )
    return matches[0]


def parse_instructions(text: str) -> list[dict[str, Any]]:
    rows = [
        {
            "pc": int(match.group(1), 16),
            "predicate": match.group(2) or "",
            "opcode": match.group(3),
            "operands": match.group(4).strip(),
        }
        for match in SASS_RE.finditer(text)
    ]
    if not rows:
        raise ValueError("no SASS instructions parsed")
    if len({row["pc"] for row in rows}) != len(rows):
        raise ValueError("SASS contains duplicate PCs or multiple code functions")
    return rows


def _width(opcode: str) -> int:
    if ".128" in opcode:
        return 16
    if ".64" in opcode:
        return 8
    return 4


def _stack_offset(operands: str) -> int | None:
    match = STACK_OFFSET_RE.search(operands)
    return None if match is None else int(match.group(1) or "0", 16)


def _selected_projection(counts: Mapping[str, int]) -> dict[str, int]:
    return {
        "omma": sum(
            value for opcode, value in counts.items() if opcode.startswith("OMMA")
        ),
        "utmaldg": sum(
            value for opcode, value in counts.items() if opcode.startswith("UTMALDG")
        ),
        "ldsm": sum(
            value for opcode, value in counts.items() if opcode.startswith("LDSM")
        ),
        "bar": sum(
            value for opcode, value in counts.items() if opcode.startswith("BAR.")
        ),
        "atomg": sum(
            value for opcode, value in counts.items() if opcode.startswith("ATOMG")
        ),
        "redg": sum(
            value for opcode, value in counts.items() if opcode.startswith("REDG")
        ),
        "ldg": sum(
            value for opcode, value in counts.items() if opcode.startswith("LDG")
        ),
        "stg": sum(
            value for opcode, value in counts.items() if opcode.startswith("STG")
        ),
    }


def _target_index(
    instructions: Sequence[Mapping[str, Any]], operands: str
) -> int | None:
    match = BRANCH_TARGET_RE.search(operands)
    if match is None:
        return None
    target = int(match.group(1), 16)
    by_pc = {int(row["pc"]): index for index, row in enumerate(instructions)}
    return by_pc.get(target)


def _branch_like(opcode: str) -> bool:
    return opcode.startswith(("BRA", "BSSY", "BSYNC", "BREAK", "CALL"))


def opcode_projection(
    baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    a = [str(row["opcode"]) for row in baseline]
    b = [str(row["opcode"]) for row in candidate]
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    operations = matcher.get_opcodes()
    insertion_only = all(tag in ("equal", "insert") for tag, *_ in operations)
    mapping: dict[int, int] = {}
    inserted: Counter[str] = Counter()
    for tag, a0, a1, b0, b1 in operations:
        if tag == "equal":
            mapping.update({a0 + offset: b0 + offset for offset in range(a1 - a0)})
        elif tag == "insert":
            inserted.update(b[b0:b1])

    branch_checks = []
    for base_index, candidate_index in mapping.items():
        base_row = baseline[base_index]
        if not _branch_like(str(base_row["opcode"])):
            continue
        candidate_row = candidate[candidate_index]
        base_target = _target_index(baseline, str(base_row["operands"]))
        candidate_target = _target_index(candidate, str(candidate_row["operands"]))
        if base_target is None or candidate_target is None:
            branch_checks.append(
                {
                    "baseline_index": base_index,
                    "status": "unresolved_target",
                    "pass": False,
                }
            )
            continue
        branch_checks.append(
            {
                "baseline_index": base_index,
                "baseline_target": base_target,
                "candidate_index": candidate_index,
                "candidate_target": candidate_target,
                "projected_target": mapping.get(base_target),
                "pass": mapping.get(base_target) == candidate_target,
            }
        )
    return {
        "baseline_instruction_count": len(a),
        "candidate_instruction_count": len(b),
        "insertion_only_opcode_projection": insertion_only and len(mapping) == len(a),
        "mapped_baseline_instructions": len(mapping),
        "inserted_opcode_counts": dict(sorted(inserted.items())),
        "branch_target_checks": branch_checks,
        "branch_target_projection_pass": bool(branch_checks)
        and all(item["pass"] for item in branch_checks),
    }


def analyze_arm(
    *,
    arm: str,
    preparation: Mapping[str, Any],
    cubin: Path,
    cuobjdump: str,
    nvdisasm: str,
    raw_root: Path,
) -> dict[str, Any]:
    source = preparation["runtime"]["source"]["overlays"]
    for name in ("kernel", "dispatch"):
        path = Path(source[name]["path"])
        if not path.is_file() or file_sha256(path) != source[name]["sha256"]:
            raise ValueError(f"{arm}: {name} overlay drift")

    resource_stdout = _run([cuobjdump, "--dump-resource-usage", str(cubin)])
    elf_stdout = _run([cuobjdump, "--dump-elf", str(cubin)])
    disassembly_stdout = _run([nvdisasm, "-c", str(cubin)])
    raw_root.mkdir(parents=True, exist_ok=False)
    (raw_root / "resource.txt").write_bytes(resource_stdout)
    (raw_root / "elf.txt").write_bytes(elf_stdout)
    (raw_root / "nvdisasm.sass").write_bytes(disassembly_stdout)

    resource_text = resource_stdout.decode(errors="replace")
    elf_text = elf_stdout.decode(errors="replace")
    sass_text = disassembly_stdout.decode(errors="replace")
    resource_match = RESOURCE_RE.search(resource_text)
    frame_match = FRAME_RE.search(elf_text)
    minimum_match = MIN_STACK_RE.search(elf_text)
    if resource_match is None or frame_match is None or minimum_match is None:
        raise ValueError(f"{arm}: incomplete cubin resource/ELF identity")
    registers, stack, shared, local = map(int, resource_match.groups())
    frame = int(frame_match.group(1), 16)
    minimum = int(minimum_match.group(1), 16)
    if frame != stack or minimum != stack:
        raise ValueError(f"{arm}: stack resource/ELF mismatch")

    instructions = parse_instructions(sass_text)
    counts = Counter(str(row["opcode"]) for row in instructions)
    local_rows = [
        row for row in instructions if str(row["opcode"]).startswith(("STL", "LDL"))
    ]
    annotation_pcs = {int(value, 16) for value in SPILL_ANNOTATION_RE.findall(elf_text)}
    local_pcs = {int(row["pc"]) for row in local_rows}
    stored_slots: set[int] = set()
    loaded_slots: set[int] = set()
    for row in local_rows:
        offset = _stack_offset(str(row["operands"]))
        if offset is None:
            raise ValueError(f"{arm}: non-R1 local instruction")
        slots = set(range(offset, offset + _width(str(row["opcode"])), 4))
        if str(row["opcode"]).startswith("STL"):
            stored_slots.update(slots)
        else:
            loaded_slots.update(slots)
    expected_slots = set(range(0, stack, 4))
    spill_gate = {
        "stack_488": stack == 488,
        "stored_words_122": len(stored_slots) == 122,
        "loaded_words_122": len(loaded_slots) == 122,
        "full_stack_roundtrip": stored_slots == loaded_slots == expected_slots,
        "main_54_stl64_108_words": counts["STL.64"] == 54,
        "tail_14_stl_words": counts["STL"] == 14,
        "compiler_annotation_closure": annotation_pcs == local_pcs,
    }
    resource_gate = {
        "registers_255": registers == 255,
        "stack_488": stack == 488,
        "static_shared_1024": shared == 1024,
        "static_local_outside_stack_0": local == 0,
    }
    return {
        "identity": {
            "cubin": str(cubin),
            "cubin_sha256": file_sha256(cubin),
            "sass": str(raw_root / "nvdisasm.sass"),
            "sass_sha256": _sha256_bytes(disassembly_stdout),
            "nvdisasm_sha256": _sha256_bytes(disassembly_stdout),
            "kernel_overlay_sha256": source["kernel"]["sha256"],
            "dispatch_overlay_sha256": source["dispatch"]["sha256"],
            "jit_artifact_set_sha256": preparation["jit_artifact_set_sha256"],
        },
        "resource": {
            "registers_per_thread": registers,
            "stack_bytes_per_thread": stack,
            "static_shared_bytes_per_cta": shared,
            "static_local_bytes_outside_stack": local,
            "elf_frame_bytes_per_thread": frame,
            "elf_minimum_stack_bytes_per_thread": minimum,
        },
        "spill": {
            "stored_words_per_lane": len(stored_slots),
            "loaded_words_per_lane": len(loaded_slots),
            "stack_slots": sorted(stored_slots),
            "local_opcode_counts": {
                opcode: count
                for opcode, count in sorted(counts.items())
                if opcode.startswith(("STL", "LDL"))
            },
            "compiler_annotation_count": len(annotation_pcs),
        },
        "selected_opcode_projection": _selected_projection(counts),
        "opcode_sequence": [str(row["opcode"]) for row in instructions],
        "instructions": instructions,
        "resource_gate": resource_gate,
        "spill_gate": spill_gate,
        "gate_pass": all(resource_gate.values()) and all(spill_gate.values()),
        "raw_outputs": {
            "resource_sha256": _sha256_bytes(resource_stdout),
            "elf_sha256": _sha256_bytes(elf_stdout),
            "nvdisasm_sha256": _sha256_bytes(disassembly_stdout),
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    results = args.results.resolve()
    explicit_cubins = _assignments(args.cubin)
    preparations = {arm: read_json(preparation_path(results, arm)) for arm in ALL_ARMS}
    tool_identity = {
        "cuobjdump": args.cuobjdump,
        "cuobjdump_version": _run([args.cuobjdump, "--version"]).decode().strip(),
        "nvdisasm": args.nvdisasm,
        "nvdisasm_version": _run([args.nvdisasm, "--version"]).decode().strip(),
    }
    arms: dict[str, Any] = {}
    for arm in ALL_ARMS:
        cubin = _select_target_cubin(
            preparations[arm],
            explicit=explicit_cubins.get(arm),
            nvdisasm=args.nvdisasm,
            label=f"{arm} cubin",
        )
        arms[arm] = analyze_arm(
            arm=arm,
            preparation=preparations[arm],
            cubin=cubin,
            cuobjdump=args.cuobjdump,
            nvdisasm=args.nvdisasm,
            raw_root=results / "raw" / "binary" / arm,
        )

    normal = arms[NORMAL]
    measurement = arms[MEASUREMENT_CONTROL]
    probe = arms[PROBE]
    historical = {
        "production_source": preparations[NORMAL]["runtime"]["source"]["production"][
            "kernel"
        ]["sha256"]
        == EXPECTED_KERNEL_SHA256,
        "baseline_cubin": normal["identity"]["cubin_sha256"]
        == EXPECTED_BASELINE_CUBIN_SHA256,
        "baseline_sass": normal["identity"]["sass_sha256"]
        == EXPECTED_BASELINE_SASS_SHA256,
        "baseline_projection": normal["selected_opcode_projection"]
        == {
            "omma": 896,
            "utmaldg": 40,
            "ldsm": 200,
            "bar": 34,
            "atomg": 9,
            "redg": 4,
            "ldg": 53,
            "stg": 75,
        },
    }
    control_projection = opcode_projection(
        normal["instructions"], measurement["instructions"]
    )
    probe_projection = opcode_projection(
        measurement["instructions"], probe["instructions"]
    )
    semantic_fields = ("omma", "utmaldg", "ldsm", "bar", "atomg", "redg", "ldg")
    semantic_counts = {
        field: all(
            arms[arm]["selected_opcode_projection"][field]
            == normal["selected_opcode_projection"][field]
            for arm in ALL_ARMS
        )
        for field in semantic_fields
    }
    control_equivalence = {
        "exact_opcode_sequence": measurement["opcode_sequence"]
        == normal["opcode_sequence"],
        "exact_selected_projection": measurement["selected_opcode_projection"]
        == normal["selected_opcode_projection"],
        "branch_projection": control_projection["branch_target_projection_pass"],
    }
    probe_semantic = {
        "insertion_only_opcode_projection": probe_projection[
            "insertion_only_opcode_projection"
        ],
        "branch_target_projection": probe_projection["branch_target_projection_pass"],
        "semantic_work_counts": all(semantic_counts.values()),
        "probe_store_delta_positive": probe["selected_opcode_projection"]["stg"]
        > measurement["selected_opcode_projection"]["stg"],
    }
    ncu = read_json(results / "raw" / "ncu_evidence.json")
    toolchain_fields = (
        "nvcc",
        "ptxas",
        "python",
        "torch",
        "cuda_runtime",
        "image_digest",
        "python_deps_sha256",
    )
    toolchain_equal = all(
        len({str(preparations[arm]["runtime"][field]) for arm in ALL_ARMS}) == 1
        for field in toolchain_fields
    )
    gpu_fields = (
        "uuid",
        "name",
        "pci_bus_id",
        "driver",
        "compute_capability",
        "sm_count",
    )
    gpu_equal = all(
        len(
            {
                canonical_sha256(preparations[arm]["runtime"]["gpu"][field])
                for arm in ALL_ARMS
            }
        )
        == 1
        for field in gpu_fields
    )
    gates = {
        "historical_production_anchor": all(historical.values()),
        "cross_arm_toolchain_identity": toolchain_equal,
        "cross_arm_gpu_identity": gpu_equal,
        "all_static_resource_spill": all(arms[arm]["gate_pass"] for arm in ALL_ARMS),
        "normal_measurement_equivalence": all(control_equivalence.values()),
        "probe_semantic_projection": all(probe_semantic.values()),
        "dynamic_ncu_identity": bool(ncu["gate_pass"]),
    }

    # Keep the large instruction lists in raw per-arm JSON, not the reader-facing
    # top-level identity.  Their hashes and projections remain traceable.
    compact_arms = {}
    for arm, value in arms.items():
        raw = results / "raw" / "binary" / arm / "parsed.json"
        write_json(raw, value)
        compact_arms[arm] = {
            key: item
            for key, item in value.items()
            if key not in ("instructions", "opcode_sequence")
        }
        compact_arms[arm]["parsed_evidence"] = str(raw.relative_to(results))
        compact_arms[arm]["parsed_evidence_sha256"] = file_sha256(raw)
    payload = {
        "schema": "exp004.binary-identity.v1",
        "collector": tool_identity,
        "production_source_identity": preparations[NORMAL]["runtime"]["source"][
            "production"
        ],
        "toolchain_identity": {
            arm: {
                "nvcc": preparations[arm]["runtime"]["nvcc"],
                "ptxas": preparations[arm]["runtime"]["ptxas"],
                "python": preparations[arm]["runtime"]["python"],
                "torch": preparations[arm]["runtime"]["torch"],
                "cuda_runtime": preparations[arm]["runtime"]["cuda_runtime"],
                "image_digest": preparations[arm]["runtime"]["image_digest"],
                "python_deps_sha256": preparations[arm]["runtime"][
                    "python_deps_sha256"
                ],
            }
            for arm in ALL_ARMS
        },
        "gpu_identity": {
            arm: {
                field: preparations[arm]["runtime"]["gpu"][field]
                for field in gpu_fields
            }
            for arm in ALL_ARMS
        },
        "arms": compact_arms,
        "historical_anchor": historical,
        "normal_measurement_projection": control_projection,
        "normal_measurement_equivalence": control_equivalence,
        "measurement_probe_projection": probe_projection,
        "probe_semantic_gate": probe_semantic,
        "semantic_count_identity": semantic_counts,
        "ncu_evidence": "raw/ncu_evidence.json",
        "ncu_evidence_sha256": file_sha256(results / "raw" / "ncu_evidence.json"),
        "gates": {**gates, "formal_gate_pass": all(gates.values())},
    }
    payload["evidence_sha256"] = canonical_sha256(payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--cubin", action="append", default=[], help="ARM=PATH")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--nvdisasm", default="nvdisasm")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build(args)
    output = args.results.resolve() / "raw" / "binary_identity.json"
    write_json(output, payload)
    return 0 if payload["gates"]["formal_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
