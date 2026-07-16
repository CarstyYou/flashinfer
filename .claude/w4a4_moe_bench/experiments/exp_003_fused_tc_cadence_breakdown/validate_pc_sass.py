#!/usr/bin/env python3
"""Validate exp_003 wait/barrier ranges against the tracker-cubin SASS.

The instrument config is the sole authority for range boundaries.  Every
targeted ``rangeStart`` must be immediately followed by its ``pop_range`` /
``rangeEnd`` record, and every static site must contain a whitelisted semantic
instruction in the half-open SASS window ``[start, end)``.  IKET marker
instructions such as CS2R and PMTRIG are reported but never satisfy the proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "exp003.pc_sass_gate.v1"
INSTRUCTION_BYTES = 16

GENERIC_WAIT_RANGE = "wait"
BARRIER_RANGES = (
    "fc2_pre_scatter_barrier",
    "fc2_post_scatter_barrier",
    "gate_pass_wait",
    "final_pass_wait",
)
STATIC_RANGE_NAMES = (GENERIC_WAIT_RANGE, *BARRIER_RANGES)
GENERIC_WAIT_LOGICAL_NAMES = (
    "fc1_gate_wait",
    "fc1_up_wait",
    "fc2_wait",
)
LOGICAL_RANGE_NAMES: Mapping[str, tuple[str, ...]] = {
    GENERIC_WAIT_RANGE: GENERIC_WAIT_LOGICAL_NAMES,
    **{name: (name,) for name in BARRIER_RANGES},
}

# Keep these lists explicit.  A nearby synchronization-looking opcode is not
# accepted until this experiment has established its semantics.
WAIT_OPCODE_WHITELIST = frozenset({"SYNCS.PHASECHK.TRANS64.TRYWAIT"})
BARRIER_OPCODE_WHITELIST = frozenset(
    {
        "BAR.SYNC",
        "BAR.SYNC.DEFER_BLOCKING",
    }
)
MARKER_ONLY_OPCODE_BASES = frozenset({"CS2R", "PMTRIG"})

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_PC_INSTRUCTION_RE = re.compile(
    r"^\s*/\*(?P<pc>[0-9A-Fa-f]+)\*/\s*(?P<body>.*?)\s*;(?:\s*//.*)?$"
)
_OPCODE_RE = re.compile(
    r"^(?:@!?[A-Z][A-Z0-9]*\s+)?"
    r"(?P<opcode>[A-Z][A-Z0-9]*(?:\.[A-Z0-9_]+)*)\b"
)
_SECTION_RE = re.compile(r"^\s*\.section\s+")


class PcSassGateError(RuntimeError):
    """Raised when the evidence cannot be interpreted without guessing."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise PcSassGateError(f"{label} is not a file: {path}")
    return path


def _expected_sha256(value: str, label: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise PcSassGateError(f"{label} must be exactly 64 hexadecimal digits")
    return value.lower()


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PcSassGateError(f"{label} must be an integer")
    return value


def _artifact(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    actual = sha256_file(path)
    result: dict[str, Any] = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": actual,
    }
    if expected_sha256 is not None:
        result["cli_expected_sha256"] = expected_sha256
        result["digest_match"] = actual == expected_sha256
    return result


def _load_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(errors="strict"))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise PcSassGateError(f"cannot parse {path}: {error}") from error
    if not isinstance(document, dict):
        raise PcSassGateError(f"{path}: top level must be an object")
    return document


def select_instrumentation_config(
    document: Mapping[str, Any], kernel: str | None
) -> dict[str, Any]:
    configs = document.get("configs")
    if not isinstance(configs, list):
        raise PcSassGateError("instrument config 'configs' must be an array")
    if kernel is None:
        matches = [value for value in configs if isinstance(value, dict)]
        if len(matches) != 1:
            raise PcSassGateError(
                "--kernel is required unless instrument config has exactly one object"
            )
    else:
        matches = [
            value
            for value in configs
            if isinstance(value, dict) and value.get("kernel") == kernel
        ]
        if len(matches) != 1:
            raise PcSassGateError(
                f"expected one exact instrumentation config for kernel={kernel!r}, "
                f"found {len(matches)}"
            )
    selected = matches[0]
    selected_kernel = selected.get("kernel")
    if not isinstance(selected_kernel, str) or not selected_kernel:
        raise PcSassGateError("selected instrumentation config lacks a kernel name")
    return selected


def _record_offset(record: Mapping[str, Any], label: str) -> int:
    offset = _integer(record.get("offset"), f"{label}.offset")
    if offset < 0:
        raise PcSassGateError(f"{label}.offset must be non-negative")
    offset_hex = record.get("offsetHex")
    if offset_hex is not None:
        if not isinstance(offset_hex, str):
            raise PcSassGateError(f"{label}.offsetHex must be a string")
        try:
            parsed = int(offset_hex, 16)
        except ValueError as error:
            raise PcSassGateError(
                f"{label}.offsetHex is not hexadecimal: {offset_hex!r}"
            ) from error
        if parsed != offset:
            raise PcSassGateError(
                f"{label}: offset/offsetHex disagree ({offset} != {offset_hex})"
            )
    return offset


def collect_static_sites(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect all target sites and prove immediate range-end pairing."""

    raw = config.get("instrumentations")
    if not isinstance(raw, list):
        raise PcSassGateError("selected config 'instrumentations' must be an array")
    sites: list[dict[str, Any]] = []
    counts = {name: 0 for name in STATIC_RANGE_NAMES}
    seen_starts: set[int] = set()

    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            raise PcSassGateError(f"instrumentations[{index}] must be an object")
        name = value.get("name")
        if name not in counts:
            continue
        counts[name] += 1
        if value.get("rangePos") != "rangeStart":
            raise PcSassGateError(
                f"instrumentations[{index}] target {name!r} is not rangeStart"
            )
        if index + 1 >= len(raw) or not isinstance(raw[index + 1], dict):
            raise PcSassGateError(
                f"instrumentations[{index}] target {name!r} lacks adjacent rangeEnd"
            )
        end_record = raw[index + 1]
        if (
            end_record.get("name") != "pop_range"
            or end_record.get("rangePos") != "rangeEnd"
        ):
            raise PcSassGateError(
                f"instrumentations[{index}] target {name!r} is not immediately "
                "followed by pop_range/rangeEnd"
            )
        start = _record_offset(value, f"instrumentations[{index}]")
        end = _record_offset(end_record, f"instrumentations[{index + 1}]")
        if end <= start:
            raise PcSassGateError(
                f"instrumentations[{index}] target {name!r} has non-positive window"
            )
        if start in seen_starts:
            raise PcSassGateError(f"duplicate target rangeStart offset 0x{start:x}")
        seen_starts.add(start)
        sites.append(
            {
                "config_index": index,
                "static_range_name": name,
                "start_offset": start,
                "end_offset": end,
            }
        )

    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise PcSassGateError(
            "instrument config lacks required static range(s): " + ", ".join(missing)
        )
    return sites


def parse_kernel_sass(
    text: str, kernel: str
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Parse exact-PC instruction rows from one exact ``.text`` section."""

    lines = text.splitlines()
    section_label = f".text.{kernel}:"
    starts = [index for index, line in enumerate(lines) if line.strip() == section_label]
    if len(starts) != 1:
        raise PcSassGateError(
            f"SASS must contain one exact {section_label!r} label, found {len(starts)}"
        )
    start_index = starts[0] + 1
    end_index = len(lines)
    for index in range(start_index, len(lines)):
        if _SECTION_RE.match(lines[index]):
            end_index = index
            break

    rows: list[dict[str, Any]] = []
    by_pc: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(
        lines[start_index:end_index], start=start_index + 1
    ):
        instruction_match = _PC_INSTRUCTION_RE.match(line)
        if not instruction_match:
            continue
        opcode_match = _OPCODE_RE.match(instruction_match.group("body"))
        if not opcode_match:
            raise PcSassGateError(
                f"SASS line {line_number}: PC-bearing row has no parseable opcode"
            )
        pc = int(instruction_match.group("pc"), 16)
        if pc in by_pc:
            raise PcSassGateError(
                f"SASS kernel section contains duplicate instruction PC 0x{pc:x}"
            )
        opcode = opcode_match.group("opcode")
        row = {
            "offset": pc,
            "offset_hex": f"0x{pc:05x}",
            "opcode": opcode,
        }
        rows.append(row)
        by_pc[pc] = row

    if not rows:
        raise PcSassGateError(f"SASS section {section_label!r} has no instructions")
    if rows != sorted(rows, key=lambda row: row["offset"]):
        raise PcSassGateError("SASS instruction PCs are not strictly increasing")
    return rows, by_pc


def _whitelist_for(static_range_name: str) -> tuple[str, frozenset[str]]:
    if static_range_name == GENERIC_WAIT_RANGE:
        return "wait", WAIT_OPCODE_WHITELIST
    if static_range_name in BARRIER_RANGES:
        return "barrier", BARRIER_OPCODE_WHITELIST
    raise PcSassGateError(f"unsupported static range name: {static_range_name}")


def prove_site(
    site: Mapping[str, Any], by_pc: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    start = int(site["start_offset"])
    end = int(site["end_offset"])
    expected_pcs = list(range(start, end, INSTRUCTION_BYTES))
    missing_pcs = [pc for pc in expected_pcs if pc not in by_pc]
    if missing_pcs:
        rendered = ", ".join(f"0x{pc:x}" for pc in missing_pcs[:8])
        suffix = " ..." if len(missing_pcs) > 8 else ""
        raise PcSassGateError(
            f"{site['static_range_name']} site at 0x{start:x}: SASS window is "
            f"not contiguous; missing {rendered}{suffix}"
        )
    if end not in by_pc:
        raise PcSassGateError(
            f"{site['static_range_name']} site at 0x{start:x}: rangeEnd PC "
            f"0x{end:x} is absent from SASS"
        )
    opcodes = [dict(by_pc[pc]) for pc in expected_pcs]
    proof_class, whitelist = _whitelist_for(str(site["static_range_name"]))
    matched = [row for row in opcodes if row["opcode"] in whitelist]
    marker_rows = [
        row
        for row in opcodes
        if str(row["opcode"]).split(".", 1)[0] in MARKER_ONLY_OPCODE_BASES
    ]
    semantic_pass = bool(matched)
    return {
        **site,
        "start_offset_hex": f"0x{start:05x}",
        "end_offset_hex": f"0x{end:05x}",
        "window_semantics": "start inclusive, adjacent pop_range/rangeEnd exclusive",
        "opcodes": opcodes,
        "proof": {
            "class": proof_class,
            "strict_opcode_whitelist": sorted(whitelist),
            "matched_semantic_instructions": matched,
            "marker_only_instructions": marker_rows,
            "marker_only_instructions_qualify": False,
            "pass": semantic_pass,
        },
    }


def build_pc_sass_gate(
    *,
    instrument_config: Path,
    sass: Path,
    cubin: Path,
    expected_sass_sha256: str,
    expected_cubin_sha256: str,
    kernel: str | None = None,
) -> dict[str, Any]:
    """Return deterministic PC/SASS evidence for all exp_003 wait leaves."""

    instrument_config = _file(instrument_config, "instrument config")
    sass = _file(sass, "SASS")
    cubin = _file(cubin, "tracker cubin")
    expected_sass_sha256 = _expected_sha256(
        expected_sass_sha256, "expected SASS SHA-256"
    )
    expected_cubin_sha256 = _expected_sha256(
        expected_cubin_sha256, "expected cubin SHA-256"
    )

    config = select_instrumentation_config(_load_document(instrument_config), kernel)
    selected_kernel = str(config["kernel"])
    static_sites = collect_static_sites(config)
    try:
        sass_text = sass.read_text(errors="strict")
    except UnicodeError as error:
        raise PcSassGateError(f"cannot decode SASS {sass}: {error}") from error
    _, by_pc = parse_kernel_sass(sass_text, selected_kernel)
    sites = [prove_site(site, by_pc) for site in static_sites]

    config_artifact = _artifact(instrument_config)
    sass_artifact = _artifact(sass, expected_sha256=expected_sass_sha256)
    cubin_artifact = _artifact(cubin, expected_sha256=expected_cubin_sha256)
    identity_pass = bool(
        sass_artifact["digest_match"] and cubin_artifact["digest_match"]
    )

    static_status: dict[str, Any] = {}
    verified_static_names: list[str] = []
    for name in STATIC_RANGE_NAMES:
        name_sites = [site for site in sites if site["static_range_name"] == name]
        semantic_pass = bool(name_sites) and all(
            site["proof"]["pass"] for site in name_sites
        )
        verified = identity_pass and semantic_pass
        static_status[name] = {
            "site_count": len(name_sites),
            "all_sites_semantic_pass": semantic_pass,
            "artifact_identity_pass": identity_pass,
            "verified": verified,
        }
        if verified:
            verified_static_names.append(name)

    verified_range_names = sorted(
        logical_name
        for static_name in verified_static_names
        for logical_name in LOGICAL_RANGE_NAMES[static_name]
    )
    expected_logical_names = sorted(
        logical_name
        for static_name in STATIC_RANGE_NAMES
        for logical_name in LOGICAL_RANGE_NAMES[static_name]
    )
    overall_pass = verified_range_names == expected_logical_names

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if overall_pass else "fail",
        "fail_closed": True,
        "kernel": selected_kernel,
        "artifacts": {
            "instrument_config": config_artifact,
            "sass": sass_artifact,
            "tracker_cubin": cubin_artifact,
        },
        "artifact_identity": {
            "authority": "CLI-provided SHA-256 values",
            "sass_digest_match": sass_artifact["digest_match"],
            "tracker_cubin_digest_match": cubin_artifact["digest_match"],
            "pass": identity_pass,
            "scope": (
                "digest chain of custody only; this validator does not rerun nvdisasm"
            ),
        },
        "policy": {
            "instruction_window": "[rangeStart, adjacent pop_range/rangeEnd)",
            "instruction_bytes": INSTRUCTION_BYTES,
            "wait_opcode_whitelist": sorted(WAIT_OPCODE_WHITELIST),
            "barrier_opcode_whitelist": sorted(BARRIER_OPCODE_WHITELIST),
            "marker_only_opcode_bases": sorted(MARKER_ONLY_OPCODE_BASES),
            "generic_wait_mapping": {
                GENERIC_WAIT_RANGE: list(GENERIC_WAIT_LOGICAL_NAMES)
            },
        },
        "sites": sites,
        "static_range_status": static_status,
        "verified_static_range_names": sorted(verified_static_names),
        "verified_range_names": verified_range_names,
        "expected_range_names": expected_logical_names,
        "overall_pass": overall_pass,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument-config", type=Path, required=True)
    parser.add_argument("--sass", type=Path, required=True)
    parser.add_argument("--cubin", type=Path, required=True)
    parser.add_argument("--expected-sass-sha256", required=True)
    parser.add_argument("--expected-cubin-sha256", required=True)
    parser.add_argument("--kernel")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_pc_sass_gate(
            instrument_config=args.instrument_config,
            sass=args.sass,
            cubin=args.cubin,
            expected_sass_sha256=args.expected_sass_sha256,
            expected_cubin_sha256=args.expected_cubin_sha256,
            kernel=args.kernel,
        )
    except (OSError, PcSassGateError) as error:
        raise SystemExit(f"PC/SASS gate failed closed: {error}") from error

    rendered = canonical_json(payload)
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
