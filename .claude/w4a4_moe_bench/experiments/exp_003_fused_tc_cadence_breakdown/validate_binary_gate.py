#!/usr/bin/env python3
"""Build the exp_003 static control/candidate binary-gate evidence.

This gate deliberately compares only an explicit semantic-opcode projection and
the nvdisasm resource tuple.  It does not normalize, or claim identity of, the
complete SASS or CFG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "exp003.static_binary_gate.v1"
RESOURCE_FIELDS = ("REG", "STACK", "SHARED", "LOCAL")
SEMANTIC_OPCODES = (
    "OMMA",
    "UTMALDG",
    "LDSM",
    "BAR",
    "ATOMG",
    "REDG",
    "LDG",
    "STG",
)

_RESOURCE_FIELD_RE = re.compile(r"\b(REG|STACK|SHARED|LOCAL):(\d+)\b")
_FUNCTION_RE = re.compile(r"^\s*Function\s+(.+):\s*$")
_PC_INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9A-Fa-f]+\*/\s*(?P<body>.*?)\s*;(?:\s*//.*)?$"
)
_OPCODE_RE = re.compile(
    r"^(?:@!?[A-Z][A-Z0-9]*\s+)?(?P<opcode>[A-Z][A-Z0-9]*(?:\.[A-Z0-9_]+)*)\b"
)
_SHA256_LINE_RE = re.compile(r"^(?P<sha>[0-9a-fA-F]{64})\s+[*]?(?P<path>.+?)\s*$")


class BinaryGateError(RuntimeError):
    """Raised when required static evidence is malformed or ambiguous."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.is_file():
        raise BinaryGateError(f"artifact is not a file: {path}")
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_resource_text(text: str, *, label: str) -> dict[str, Any]:
    """Parse the one kernel resource record emitted by nvdisasm -res-usage."""

    records: list[dict[str, Any]] = []
    current_function: str | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        function_match = _FUNCTION_RE.match(line)
        if function_match:
            current_function = function_match.group(1)
            continue
        fields = {name: int(value) for name, value in _RESOURCE_FIELD_RE.findall(line)}
        if not fields:
            continue
        missing = [name for name in RESOURCE_FIELDS if name not in fields]
        if missing:
            raise BinaryGateError(
                f"{label}:{line_number}: incomplete resource tuple; missing {missing}"
            )
        records.append(
            {
                "kernel": current_function,
                **{name: fields[name] for name in RESOURCE_FIELDS},
            }
        )

    if len(records) != 1:
        raise BinaryGateError(
            f"{label}: expected exactly one kernel resource tuple, found {len(records)}"
        )
    if records[0]["kernel"] is None:
        raise BinaryGateError(f"{label}: resource tuple has no preceding Function record")
    return records[0]


def parse_resource_file(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise BinaryGateError(f"resource input is not a file: {path}")
    return parse_resource_text(path.read_text(errors="strict"), label=str(path))


def semantic_opcode_counts(text: str) -> dict[str, int]:
    """Count exact opcode tokens on nvdisasm PC instruction lines.

    Address comments are required, predicates are skipped, and opcode modifiers
    are removed only after tokenization.  Consequently ``UTMALDG`` is not an
    ``LDG`` and text in comments/directives is never counted.
    """

    counts = {opcode: 0 for opcode in SEMANTIC_OPCODES}
    for line in text.splitlines():
        instruction_match = _PC_INSTRUCTION_RE.match(line)
        if not instruction_match:
            continue
        opcode_match = _OPCODE_RE.match(instruction_match.group("body"))
        if not opcode_match:
            continue
        opcode = opcode_match.group("opcode").split(".", 1)[0]
        if opcode in counts:
            counts[opcode] += 1
    return counts


def semantic_opcode_counts_file(path: Path) -> dict[str, int]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise BinaryGateError(f"SASS input is not a file: {path}")
    return semantic_opcode_counts(path.read_text(errors="strict"))


def parse_sha256_manifest(path: Path) -> list[dict[str, str]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise BinaryGateError(f"SHA-256 manifest is not a file: {path}")
    entries: list[dict[str, str]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SHA256_LINE_RE.match(line)
        if not match:
            raise BinaryGateError(f"{path}:{line_number}: malformed SHA-256 line")
        entries.append(
            {
                "sha256": match.group("sha").lower(),
                "path": match.group("path"),
            }
        )
    if not entries:
        raise BinaryGateError(f"{path}: SHA-256 manifest is empty")
    return entries


def _role_from_manifest_path(raw_path: str) -> str | None:
    components = [component.lower() for component in Path(raw_path).parts]
    basename = Path(raw_path).name.lower()
    control = "control" in components or basename.startswith("control")
    candidate = "candidate" in components or basename.startswith("candidate")
    if control == candidate:
        return None
    return "control" if control else "candidate"


def _one_cubin_hash(
    entries: Sequence[Mapping[str, str]], role: str
) -> tuple[str | None, str | None]:
    matches = [
        entry
        for entry in entries
        if entry["path"].lower().endswith(".cubin")
        and _role_from_manifest_path(entry["path"]) == role
    ]
    if len(matches) != 1:
        return None, f"expected one {role} cubin hash, found {len(matches)}"
    return matches[0]["sha256"], None


def _manifest_artifact_check(
    entries: Sequence[Mapping[str, str]], artifact: Mapping[str, Any] | None
) -> dict[str, Any]:
    if artifact is None:
        return {"present": False, "matched": False, "reason": "artifact not supplied"}
    matches = [entry for entry in entries if Path(entry["path"]).name == artifact["file"]]
    if len(matches) != 1:
        return {
            "present": True,
            "matched": False,
            "reason": f"expected one manifest entry, found {len(matches)}",
        }
    matched = matches[0]["sha256"] == artifact["sha256"]
    return {
        "present": True,
        "matched": matched,
        "manifest_sha256": matches[0]["sha256"],
        "reason": None if matched else "manifest digest does not match artifact",
    }


def _field_comparison(
    control: Mapping[str, int], candidate: Mapping[str, int], fields: Iterable[str]
) -> dict[str, dict[str, int | bool]]:
    return {
        field: {
            "control": control[field],
            "candidate": candidate[field],
            "delta": candidate[field] - control[field],
            "equal": candidate[field] == control[field],
        }
        for field in fields
    }


def build_binary_gate(
    *,
    control_resource: Path,
    candidate_resource: Path,
    control_sass: Path,
    candidate_sass: Path,
    control_cfg: Path | None = None,
    candidate_cfg: Path | None = None,
    sha256_manifest: Path | None = None,
) -> dict[str, Any]:
    """Return deterministic, JSON-serializable static gate evidence."""

    control_resource_data = parse_resource_file(control_resource)
    candidate_resource_data = parse_resource_file(candidate_resource)
    control_counts = semantic_opcode_counts_file(control_sass)
    candidate_counts = semantic_opcode_counts_file(candidate_sass)

    artifacts = {
        "control": {
            "resource": _artifact(control_resource),
            "sass": _artifact(control_sass),
            "cfg": _artifact(control_cfg),
        },
        "candidate": {
            "resource": _artifact(candidate_resource),
            "sass": _artifact(candidate_sass),
            "cfg": _artifact(candidate_cfg),
        },
        "sha256_manifest": _artifact(sha256_manifest),
    }

    resource_fields = _field_comparison(
        control_resource_data, candidate_resource_data, RESOURCE_FIELDS
    )
    resource_identity_pass = all(row["equal"] for row in resource_fields.values())
    opcode_fields = _field_comparison(
        control_counts, candidate_counts, SEMANTIC_OPCODES
    )
    semantic_counts_pass = all(row["equal"] for row in opcode_fields.values())

    manifest_entries: list[dict[str, str]] = []
    manifest_checks: dict[str, Any] = {
        "present": False,
        "pass": False,
        "artifacts": {},
    }
    control_cubin_sha: str | None = None
    candidate_cubin_sha: str | None = None
    cubin_errors: list[str] = []
    if sha256_manifest is not None:
        manifest_entries = parse_sha256_manifest(sha256_manifest)
        control_cubin_sha, control_error = _one_cubin_hash(
            manifest_entries, "control"
        )
        candidate_cubin_sha, candidate_error = _one_cubin_hash(
            manifest_entries, "candidate"
        )
        cubin_errors.extend(
            error for error in (control_error, candidate_error) if error is not None
        )
        checks: dict[str, Any] = {}
        for role in ("control", "candidate"):
            for kind in ("resource", "sass", "cfg"):
                checks[f"{role}_{kind}"] = _manifest_artifact_check(
                    manifest_entries, artifacts[role][kind]
                )
        manifest_checks = {
            "present": True,
            "pass": all(check["matched"] for check in checks.values()),
            "artifacts": checks,
        }

    cfg_pair_present = control_cfg is not None and candidate_cfg is not None
    cubin_hash_evidence_pass = (
        control_cubin_sha is not None
        and candidate_cubin_sha is not None
        and not cubin_errors
    )
    kernel_name_match = (
        control_resource_data["kernel"] == candidate_resource_data["kernel"]
    )

    reasons: list[str] = []
    if not semantic_counts_pass:
        reasons.append("selected semantic opcode counts differ")
    if not resource_identity_pass:
        changed = [name for name, row in resource_fields.items() if not row["equal"]]
        reasons.append("resource identity failed: " + ", ".join(changed))
    if not kernel_name_match:
        reasons.append("resource records name different kernels")
    if not cfg_pair_present:
        reasons.append("control/candidate CFG evidence pair is missing")
    if not manifest_checks["pass"]:
        reasons.append("SHA-256 manifest coverage or validation failed")
    if not cubin_hash_evidence_pass:
        reasons.append("control/candidate cubin hash evidence is missing or ambiguous")

    formal_dominance_eligible = not reasons
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "semantic_comparison": "selected exact opcode-token counts only",
            "resource_comparison": list(RESOURCE_FIELDS),
            "full_sass_identity_assessed": False,
            "full_cfg_identity_assessed": False,
        },
        "artifacts": artifacts,
        "control": {
            "resource": control_resource_data,
            "semantic_opcode_counts": control_counts,
            "cubin_sha256": control_cubin_sha,
        },
        "candidate": {
            "resource": candidate_resource_data,
            "semantic_opcode_counts": candidate_counts,
            "cubin_sha256": candidate_cubin_sha,
        },
        "comparisons": {
            "kernel_name_match": kernel_name_match,
            "resource_identity": {
                "pass": resource_identity_pass,
                "fields": resource_fields,
            },
            "semantic_opcode_counts": {
                "pass": semantic_counts_pass,
                "fields": opcode_fields,
            },
            "cubin_hash_evidence": {
                "pass": cubin_hash_evidence_pass,
                "errors": cubin_errors,
            },
            "sha256_manifest_validation": manifest_checks,
            "cfg_pair_present": cfg_pair_present,
        },
        # This compact projection can be copied into a target manifest without
        # conflating static SASS counts with dynamically executed QMMA ranges.
        "binary_semantic_omma_gate": {
            "pass": semantic_counts_pass,
            "control_static_semantic_omma_count": control_counts["OMMA"],
            "candidate_static_semantic_omma_count": candidate_counts["OMMA"],
            "reason": None
            if semantic_counts_pass
            else "control/candidate exact OMMA opcode-token counts differ",
        },
        "formal_dominance": {
            "eligible": formal_dominance_eligible,
            "fail_closed": True,
            "reasons": reasons,
        },
    }


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-resource", type=Path, required=True)
    parser.add_argument("--candidate-resource", type=Path, required=True)
    parser.add_argument("--control-sass", type=Path, required=True)
    parser.add_argument("--candidate-sass", type=Path, required=True)
    parser.add_argument("--control-cfg", type=Path)
    parser.add_argument("--candidate-cfg", type=Path)
    parser.add_argument("--sha256-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_binary_gate(
            control_resource=args.control_resource,
            candidate_resource=args.candidate_resource,
            control_sass=args.control_sass,
            candidate_sass=args.candidate_sass,
            control_cfg=args.control_cfg,
            candidate_cfg=args.candidate_cfg,
            sha256_manifest=args.sha256_manifest,
        )
    except (BinaryGateError, OSError, UnicodeError) as error:
        raise SystemExit(f"static binary gate failed closed: {error}") from error

    rendered = canonical_json(payload)
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
